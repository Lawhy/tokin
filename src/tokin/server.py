from __future__ import annotations

import json
from http import HTTPStatus
from typing import cast

import numpy as np
from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from openai.types.chat import ChatCompletion
from pydantic import ValidationError
from pydantic_core import InitErrorDetails
from safetensors.numpy import save

from .chat_templates import ChatTemplateError
from .gateway import Gateway, GatewayError
from .harness_apis import OpenAIChatRequest
from .inference_engines import InferenceEngineError
from .rollout import Generation, Session


class UnknownSessionError(GatewayError):
    status_code = HTTPStatus.NOT_FOUND


class Server:
    """`gateway` over HTTP: `POST /sessions` opens a session, which serves OpenAI's chat completions under its own base URL.

    `layers` and `top_k` are the model's, and shape the expert routing a collected session carries.
    `app` is the ASGI application to serve.
    """

    def __init__(self, gateway: Gateway, *, layers: int | None = None, top_k: int | None = None) -> None:
        self.gateway = gateway
        self.layers, self.top_k = layers, top_k
        self.sessions: dict[str, Session] = {}
        self.app = FastAPI()
        self.app.add_api_route("/sessions", self.create_session, methods=["POST"], status_code=HTTPStatus.CREATED)
        self.app.add_api_route("/sessions/{sid}/v1/chat/completions", self.chat_completions, methods=["POST"])
        self.app.add_api_route("/sessions/{sid}", self.get_session, methods=["GET"], response_class=Response)
        self.app.add_api_route(
            "/sessions/{sid}", self.delete_session, methods=["DELETE"], status_code=HTTPStatus.NO_CONTENT
        )
        for kind in (RequestValidationError, GatewayError, ChatTemplateError, InferenceEngineError):
            self.app.add_exception_handler(kind, self.refuse)

    def session(self, sid: str) -> Session:
        if (session := self.sessions.get(sid)) is None:
            raise UnknownSessionError(f"no session {sid!r}; open one with POST /sessions")
        return session

    def encode(self, session: Session) -> bytes:
        """`session` as one safetensors payload for a trainer, every array aligned with its rollout's `token_ids`.

        Rollout `r` gives `token_ids.{r}` (int32) and `loss_mask.{r}` (uint8), and, when its generations
        carry them, `logprobs.{r}` (float32) and `routed_experts.{r}` (int32, `(n - 1, layers, top_k)`).
        `metadata` is UTF-8 JSON as uint8, holding the session id and each rollout's messages and finish
        reasons; a tensor rather than the header's metadata, which `safetensors.numpy.load` drops.
        """
        tensors: dict[str, np.ndarray] = {}
        rollouts = []
        for r, rollout in enumerate(session.rollouts):
            generations = [s for s in rollout.segments if isinstance(s, Generation)]
            tensors[f"token_ids.{r}"] = np.array(rollout.token_ids, dtype=np.int32)
            tensors[f"loss_mask.{r}"] = np.array(rollout.loss_mask, dtype=np.uint8)
            if any(g.logprobs is not None for g in generations):
                tensors[f"logprobs.{r}"] = np.array(rollout.logprobs, dtype=np.float32)
            if any(g.routed_experts is not None for g in generations):
                if self.layers is None or self.top_k is None:
                    raise ValueError("the session carries expert routing, so the server needs layers and top_k")
                tensors[f"routed_experts.{r}"] = rollout.routed_experts(self.layers, self.top_k)
            rollouts.append({"messages": rollout.messages, "finish_reasons": [g.finish_reason for g in generations]})
        meta = json.dumps({"session_id": session.id, "rollouts": rollouts}).encode()
        tensors["metadata"] = np.frombuffer(meta, dtype=np.uint8)
        return save(tensors)

    async def create_session(self) -> dict[str, str]:
        session = Session()
        self.sessions[session.id] = session
        return {"id": session.id}

    async def chat_completions(self, sid: str, request: OpenAIChatRequest) -> ChatCompletion:
        return request.response(await self.gateway.chat(self.session(sid), request.chat_request()))

    async def get_session(self, sid: str) -> Response:
        """The session as `encode` writes it, for a trainer to collect."""
        return Response(self.encode(self.session(sid)), media_type="application/octet-stream")

    async def delete_session(self, sid: str) -> None:
        """Forget the session; a turn still running finishes, but on a session no longer listed."""
        del self.sessions[self.session(sid).id]

    async def refuse(self, _: Request, exc: Exception) -> JSONResponse:
        if isinstance(exc, RequestValidationError):
            # FastAPI locates body errors under `body`, and the body is the harness's whole request.
            errors = [{**e, "loc": e["loc"][1:]} for e in exc.errors()]
            exc = ValidationError.from_exception_data("OpenAIChatRequest", cast(list[InitErrorDetails], errors))
        match exc:
            case GatewayError():
                status = exc.status_code
            case InferenceEngineError():
                status = HTTPStatus.BAD_GATEWAY
            case _:
                status = HTTPStatus.BAD_REQUEST
        # The SDKs retry a 409, but a stale turn sent again only meets the same refusal.
        headers = {"x-should-retry": "false"} if status == HTTPStatus.CONFLICT else None
        return JSONResponse(OpenAIChatRequest.error(exc, status), status, headers=headers)
