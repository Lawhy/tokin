from __future__ import annotations

from http import HTTPStatus
from typing import cast

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from openai.types.chat import ChatCompletion
from pydantic import ValidationError
from pydantic_core import InitErrorDetails

from .chat_templates import ChatTemplateError
from .gateway import Gateway, GatewayError
from .harness_apis import OpenAIChatRequest
from .inference_engines import InferenceEngineError
from .rollout import Session


class UnknownSessionError(GatewayError):
    status_code = HTTPStatus.NOT_FOUND


def create_app(gateway: Gateway) -> FastAPI:
    """`gateway` over HTTP: `POST /sessions` opens a session, which serves OpenAI's chat completions under its own base URL."""
    app = FastAPI()
    sessions: dict[str, Session] = {}
    app.state.sessions = sessions

    @app.post("/sessions", status_code=HTTPStatus.CREATED)
    async def create_session() -> dict[str, str]:
        session = Session()
        sessions[session.id] = session
        return {"id": session.id}

    @app.post("/sessions/{sid}/v1/chat/completions")
    async def chat_completions(sid: str, request: OpenAIChatRequest) -> ChatCompletion:
        if (session := sessions.get(sid)) is None:
            raise UnknownSessionError(f"no session {sid!r}; open one with POST /sessions")
        return request.response(await gateway.chat(session, request.chat_request()))

    async def refuse(_: Request, exc: Exception) -> JSONResponse:
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

    for kind in (RequestValidationError, GatewayError, ChatTemplateError, InferenceEngineError):
        app.add_exception_handler(kind, refuse)
    return app
