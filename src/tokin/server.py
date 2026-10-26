from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Any, cast

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, Field, ValidationError
from pydantic_core import InitErrorDetails

from .chat_templates import ChatTemplateError
from .gateway import ChatResponse, Gateway, GatewayError
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

    `app` is the ASGI application. Startup waits up to `ready_timeout` seconds for the engine to generate; a stream
    still generating sends a keep-alive comment every `keepalive` seconds.
    """

    def __init__(self, gateway: Gateway, *, ready_timeout: float = 1800, keepalive: float = 30) -> None:
        self.gateway = gateway
        self.ready_timeout = ready_timeout
        self.keepalive = keepalive
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
        yield

    def _session(self, sid: str) -> Session:
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

    async def get_session(self, sid: str) -> Response:
        """The session as `Session.pack` writes it, for a trainer to collect."""
        return Response(self._session(sid).pack(), media_type="application/octet-stream")

    async def delete_session(self, sid: str) -> None:
        """Forget the session; a turn still running finishes, but on a session no longer listed."""
        del self.sessions[self._session(sid).id]

    def _answer(self, http: Request, exc: Exception) -> tuple[HTTPStatus, dict[str, Any]]:
        """The status and OpenAI's error body for `exc`, logged with its reason, which the access log does not show."""
        match exc:
            case GatewayError():
                status = exc.status_code
            case InferenceEngineError():
                status = HTTPStatus.BAD_GATEWAY
            case _:
                status = HTTPStatus.BAD_REQUEST
        logger.warning("%s %s answered %d: %s", http.method, http.url.path, status, exc)
        return status, OpenAIChatRequest.error(exc, status)

    async def refuse(self, http: Request, exc: Exception) -> JSONResponse:
        if isinstance(exc, RequestValidationError):
            # FastAPI locates body errors under `body`, and the body is the harness's whole request.
            errors = [{**e, "loc": e["loc"][1:]} for e in exc.errors()]
            exc = ValidationError.from_exception_data("OpenAIChatRequest", cast(list[InitErrorDetails], errors))
        status, body = self._answer(http, exc)
        return JSONResponse(body, status)

    async def _stream(
        self, turn: asyncio.Future[ChatResponse], request: OpenAIChatRequest, http: Request
    ) -> AsyncIterator[str]:
        """The events of a turn still generating: keep-alive comments until it ends, then its reply or its error.

        Its headers are sent already, so an error reaches the harness as OpenAI's error event, not a status. A harness
        hanging up cancels this stream, and the turn with it.
        """
        try:
            while not turn.done():
                await asyncio.wait({turn}, timeout=self.keepalive)
                if not turn.done():
                    yield ": keep-alive\n\n"
            try:
                reply = turn.result()
            except (GatewayError, ChatTemplateError, InferenceEngineError) as exc:
                yield f"data: {json.dumps(self._answer(http, exc)[1])}\n\n"
                return
            for event in request.events(reply):
                yield event
        finally:
            turn.cancel()

    async def chat_completions(self, sid: str, request: OpenAIChatRequest, http: Request) -> ChatCompletion | Response:
        """The turn's completion, as one response or, when asked, a stream.

        A harness hanging up first cancels the turn; the 499 sent back is never read. A stream opens once the turn
        starts generating and sends keep-alive comments until it is done: clients time out waiting for a response's
        headers or on a silent stream (OpenAI's SDKs at 600 s, opencode at 300 s), and a generation can run longer.
        """

        async def hung_up() -> None:
            # Past the body, ASGI sends only `http.disconnect`; anything else is passed over.
            while (await http.receive())["type"] != "http.disconnect":
                pass

        generating = asyncio.Event()
        turn = asyncio.create_task(self.gateway.chat(self._session(sid), request.chat_request(), generating))
        # A stream opens once the turn generates, a response once the turn ends, unless the harness hangs up first.
        opened = asyncio.create_task(generating.wait()) if request.stream else turn
        gone = asyncio.create_task(hung_up())
        await asyncio.wait({turn, opened, gone}, return_when=asyncio.FIRST_COMPLETED)
        gone.cancel()
        opened.cancel()
        if not turn.done():
            if gone.done():
                turn.cancel()
                return Response(status_code=499)
            return StreamingResponse(self._stream(turn, request, http), media_type="text/event-stream")
        # The turn ended before any reply was sent, so a refusal raises here and keeps its status.
        reply = turn.result()
        if request.stream:
            return StreamingResponse(request.events(reply), media_type="text/event-stream")
        return request.response(reply)
