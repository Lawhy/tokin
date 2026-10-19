from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import cast

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, Field, ValidationError
from pydantic_core import InitErrorDetails

from .chat_templates import ChatTemplateError
from .gateway import Gateway, GatewayError
from .harness_apis import OpenAIChatRequest
from .inference_engines import InferenceEngineError
from .rollout import GenerationParams
from .session import Session

logger = logging.getLogger(__name__)


class UnknownSessionError(GatewayError):
    status_code = HTTPStatus.NOT_FOUND


class SessionRequest(BaseModel, extra="forbid"):
    """The body of `POST /sessions`: the params the trainer fixes for every turn of the session."""

    params: GenerationParams = Field(default_factory=lambda: GenerationParams())


class Server:
    """The gateway over HTTP: `POST /sessions` opens a session, serving OpenAI's chat completions at its own base URL.

    `app` is the ASGI application. Startup waits up to `ready_timeout` seconds for the engine to generate.
    """

    def __init__(self, gateway: Gateway, *, ready_timeout: float = 1800) -> None:
        self.gateway = gateway
        self.ready_timeout = ready_timeout
        self.sessions: dict[str, Session] = {}
        self.app = FastAPI(lifespan=self.lifespan)
        self.app.add_api_route("/health", self.health, methods=["GET"])
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

    @asynccontextmanager
    async def lifespan(self, _: FastAPI) -> AsyncIterator[None]:
        try:
            async with asyncio.timeout(self.ready_timeout):
                await self.gateway.engine.ready()
        except TimeoutError as e:
            raise InferenceEngineError(f"the engine could not generate within {self.ready_timeout:.0f} s") from e
        self.gateway.context_length = await self.gateway.engine.context_length()
        yield

    def session(self, sid: str) -> Session:
        if (session := self.sessions.get(sid)) is None:
            raise UnknownSessionError(f"no session {sid!r}; open one with POST /sessions")
        return session

    async def health(self) -> dict[str, str]:
        """Answered only once startup found the engine able to generate."""
        return {"status": "ok"}

    async def create_session(self, request: SessionRequest | None = None) -> dict[str, str]:
        params = request.params if request else GenerationParams()
        self.gateway.refuse_reserved(params)
        session = Session(params=params)
        self.sessions[session.id] = session
        return {"id": session.id}

    async def chat_completions(self, sid: str, request: OpenAIChatRequest, http: Request) -> ChatCompletion | Response:
        """The turn's completion, as one response or, when asked, a stream sent once the turn is done.

        A harness hanging up first cancels the turn; the 499 sent back is never read.
        """

        async def hung_up() -> None:
            # Past the body, ASGI sends only `http.disconnect`; anything else is passed over.
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
        """The session as `Session.pack` writes it, for a trainer to collect."""
        return Response(self.session(sid).pack(), media_type="application/octet-stream")

    async def delete_session(self, sid: str) -> None:
        """Forget the session; a turn still running finishes, but on a session no longer listed."""
        del self.sessions[self.session(sid).id]

    async def refuse(self, request: Request, exc: Exception) -> JSONResponse:
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
        # The access log records only the status; the reason reaches the harness alone unless it is logged here.
        logger.warning("%s %s answered %d: %s", request.method, request.url.path, status, exc)
        # The SDKs retry a 409, but a stale turn sent again only meets the same refusal.
        headers = {"x-should-retry": "false"} if status == HTTPStatus.CONFLICT else None
        return JSONResponse(OpenAIChatRequest.error(exc, status), status, headers=headers)
