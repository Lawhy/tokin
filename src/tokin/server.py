from __future__ import annotations

import asyncio
import json
from functools import cached_property
from http import HTTPStatus
from typing import cast

import numpy as np
from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from openai.types.chat import ChatCompletion
from pydantic import ValidationError
from pydantic_core import InitErrorDetails
from safetensors.numpy import save
from transformers import AutoConfig, PreTrainedConfig

from .chat_templates import ChatTemplateError
from .gateway import Gateway, GatewayError
from .harness_apis import OpenAIChatRequest
from .inference_engines import InferenceEngineError
from .rollout import Generation, Session


class UnknownSessionError(GatewayError):
    status_code = HTTPStatus.NOT_FOUND


class Server:
    """The gateway over HTTP: `POST /sessions` opens a session, which serves OpenAI's chat completions under its own base URL.

    `app` is the ASGI application to serve.
    """

    def __init__(self, gateway: Gateway) -> None:
        self.gateway = gateway
        self.sessions: dict[str, Session] = {}
        self.app = FastAPI()
        self.app.add_api_route("/sessions", self.create_session, methods=["POST"], status_code=HTTPStatus.CREATED)
        self.app.add_api_route(
            "/sessions/{sid}/v1/chat/completions",
            self.chat_completions,
            methods=["POST"],
            response_model=ChatCompletion,
        )
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

    @cached_property
    def model_config(self) -> PreTrainedConfig:
        """The served model's config, beside the tokenizer, whose path must be the model's for its ids to hold."""
        path = self.gateway.template.tokenizer.name_or_path
        try:
            config = AutoConfig.from_pretrained(path)
        except (OSError, ValueError) as e:
            raise RuntimeError(f"no model config at {path!r}: {e!r}") from e
        # A multimodal config keeps the language model's under `text_config`, as sglang's `hf_text_config` reads.
        return cast(PreTrainedConfig, getattr(config, "text_config", None) or config)

    def serialize(self, session: Session) -> bytes:
        """The session as one safetensors payload, the format a trainer collects.

        Each rollout `r` gives arrays aligned with its ids:

            `token_ids.{r}`       `int32`     the ids the engine saw, prompts and generations in order
            `loss_mask.{r}`       `uint8`     1 at each id the model sampled, 0 at each id tokin rendered
            `logprobs.{r}`        `float32`   each sampled id's logprob, 0.0 at rendered ids
            `routed_experts.{r}`  `int32`     (n - 1, layers, top_k), each id's experts as it was fed in;
                                          the last id never was, so it has no row

        `logprobs` and `routed_experts` are there when the generations carry them; the routing takes its
        shape from `model_config`. The first prompt ends at the first 1 in `loss_mask`, where a trainer
        that splits prompt from response cuts.

        `metadata` is UTF-8 JSON as uint8, a tensor because `safetensors.numpy.load` drops header metadata:

            `{"session_id": ..., "rollouts": [{"messages": [...], "finish_reasons": [...]}, ...]}`
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
                config = self.model_config
                try:
                    shape = config.num_hidden_layers, config.num_experts_per_tok
                except AttributeError as e:
                    raise RuntimeError(f"the served model has no expert routing: {e}") from e
                tensors[f"routed_experts.{r}"] = rollout.routed_experts(*shape)
            rollouts.append({"messages": rollout.messages, "finish_reasons": [g.finish_reason for g in generations]})
        meta = json.dumps({"session_id": session.id, "rollouts": rollouts}).encode()
        tensors["metadata"] = np.frombuffer(meta, dtype=np.uint8)
        return save(tensors)

    async def create_session(self) -> dict[str, str]:
        session = Session()
        self.sessions[session.id] = session
        return {"id": session.id}

    async def chat_completions(self, sid: str, request: OpenAIChatRequest, http: Request) -> ChatCompletion | Response:
        """The turn's completion, as one response or, when asked, a stream sent once the turn is done.

        Should the harness hang up first, the turn is cancelled and a 499 nobody reads goes back.
        """

        async def hung_up() -> None:
            # Past the body, ASGI sends only `http.disconnect`; anything else is passed over, as vLLM's does.
            while (await http.receive())["type"] != "http.disconnect":
                pass

        turn = asyncio.ensure_future(self.gateway.chat(self.session(sid), request.chat_request()))
        gone = asyncio.ensure_future(hung_up())
        await asyncio.wait({turn, gone}, return_when=asyncio.FIRST_COMPLETED)
        gone.cancel()
        if not turn.done():
            turn.cancel()
            return Response(status_code=499)
        if request.stream:
            return StreamingResponse(iter(request.events(turn.result())), media_type="text/event-stream")
        return request.response(turn.result())

    async def get_session(self, sid: str) -> Response:
        """The session as `serialize` writes it, for a trainer to collect."""
        return Response(self.serialize(self.session(sid)), media_type="application/octet-stream")

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
