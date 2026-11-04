from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import IO, Any

import click
import httpx2
import uvicorn
from transformers import AutoTokenizer

from .chat_templates import get_chat_template
from .gateway import Gateway
from .inference_engines import SGLangEngine
from .server import Server
from .session import Session

# The base URL `tokin open` prints, which `tokin collect` reads the server and session back from.
_SESSION_URL = re.compile(r"(?P<server>https?://[^/]+(?:/.*?)?)/sessions/(?P<sid>[^/]+)/v1/?")


def build(
    model_path: str, chat_template: str | None, chat_template_kwargs: str, engine_url: str, ready: float
) -> Server:
    """The server the options describe; the template is the family listing `model_path` unless one is named."""
    kwargs = parse_json(chat_template_kwargs, "--chat-template-kwargs")
    template = get_chat_template(chat_template or model_path)(AutoTokenizer.from_pretrained(model_path), **kwargs)
    return Server(Gateway(template, SGLangEngine(engine_url)), ready_timeout=ready)


def parse_json(text: str, option: str) -> Any:
    try:
        return json.loads(text)
    except ValueError as e:
        raise click.BadParameter(f"not JSON: {e}", param_hint=option) from e


@click.group(context_settings={"show_default": True})
def main() -> None:
    """Make any agent harness token-native: serve an inference engine as chat completions, token in, token out."""


@main.command(context_settings={"auto_envvar_prefix": "TOKIN"})
@click.option("--model-path", required=True, help="Checkpoint path or Hub id; the tokenizer is read from it.")
@click.option("--chat-template", help="A family name or `module:Class`; by default the family that lists --model-path.")
@click.option(
    "--chat-template-kwargs", default="{}", help="JSON the template reads, e.g. '{\"enable_thinking\": false}'."
)
@click.option("--engine-url", default="http://localhost:30000", help="The sglang server tokin feeds.")
@click.option("--host", default="127.0.0.1")
@click.option("--port", type=click.IntRange(1, 65535), default=8000)
@click.option(
    "--engine-ready-timeout", type=click.FloatRange(min=0), default=1800.0, help="Seconds startup waits for the engine."
)
@click.option(
    "--timeout-keep-alive",
    type=click.IntRange(min=1),
    default=75,
    help="Seconds an idle connection stays open, past clients' 5 s and load balancers' 60 s.",
)
@click.option("--log-level", type=click.Choice(["debug", "info", "warning", "error"]), default="info")
@click.option("--access-log/--no-access-log", default=False, help="Log every request, which floods at rollout scale.")
def serve(
    model_path: str,
    chat_template: str | None,
    chat_template_kwargs: str,
    engine_url: str,
    host: str,
    port: int,
    engine_ready_timeout: float,
    timeout_keep_alive: int,
    log_level: str,
    access_log: bool,
) -> None:
    """Serve an inference engine to agent harnesses as chat completions, token in, token out."""
    logging.basicConfig(level=log_level.upper(), format="%(levelname)s:     %(name)s: %(message)s")
    server = build(model_path, chat_template, chat_template_kwargs, engine_url, engine_ready_timeout)
    uvicorn.run(
        server.app,
        host=host,
        port=port,
        log_level=log_level,
        access_log=access_log,
        timeout_keep_alive=timeout_keep_alive,
    )


@main.command("open")
@click.option("--url", envvar="TOKIN_URL", default="http://127.0.0.1:8000", help="The tokin server.")
@click.option(
    "--params", default="{}", help="JSON `GenerationParams` every turn runs with, e.g. '{\"temperature\": 1}'."
)
def open_session(url: str, params: str) -> None:
    """Open a session and print the base URL a harness takes for it."""

    async def run() -> str:
        async with httpx2.AsyncClient(base_url=url) as client:
            return await Session.open(client, parse_json(params, "--params"))

    click.echo(f"{url.rstrip('/')}/sessions/{asyncio.run(run())}/v1")


@main.command()
@click.argument("session_url")
@click.option("-o", "--output", type=click.File("w"), default="-", help="Where the JSON lines go.")
def collect(session_url: str, output: IO[str]) -> None:
    """Collect the session at `SESSION_URL` and delete it there, writing one JSON line per rollout.

    A line holds the rollout's `token_ids` and `loss_mask`, and its `logprobs` when the session returns them.
    """
    if (match := _SESSION_URL.fullmatch(session_url)) is None:
        raise click.BadParameter(
            "not a session's base URL, …/sessions/<id>/v1, as `tokin open` prints", param_hint="SESSION_URL"
        )

    async def run() -> Session:
        async with httpx2.AsyncClient(base_url=match["server"]) as client:
            return await Session.collect(client, match["sid"])

    session = asyncio.run(run())
    for rollout in session.rollouts:
        line: dict[str, Any] = {"token_ids": rollout.token_ids, "loss_mask": rollout.loss_mask}
        if session.params.get("return_logprobs"):
            line["logprobs"] = rollout.logprobs
        output.write(json.dumps(line) + "\n")
