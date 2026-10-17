from __future__ import annotations

import json
import logging

import click
import uvicorn
from transformers import AutoTokenizer

from .chat_templates import get_chat_template
from .gateway import Gateway
from .inference_engines import SGLangEngine
from .server import Server


def build(
    model_path: str, chat_template: str | None, chat_template_kwargs: str, engine_url: str, ready: float
) -> Server:
    """The server the options describe; the template is the family listing `model_path` unless one is named."""
    try:
        kwargs = json.loads(chat_template_kwargs)
    except ValueError as e:
        raise click.BadParameter(f"not JSON: {e}", param_hint="--chat-template-kwargs") from e
    template = get_chat_template(chat_template or model_path)(AutoTokenizer.from_pretrained(model_path), **kwargs)
    return Server(Gateway(template, SGLangEngine(engine_url)), ready_timeout=ready)


@click.command(context_settings={"auto_envvar_prefix": "TOKIN", "show_default": True})
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
def main(
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
