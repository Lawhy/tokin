import importlib

from .glm45 import GLM45ToolParser
from .hermes import HermesToolParser
from .qwen3_coder import Qwen3CoderToolParser
from .tool_parser import ToolParser

TOOL_PARSERS: dict[str, type[ToolParser]] = {
    p.name: p for p in (HermesToolParser, Qwen3CoderToolParser, GLM45ToolParser)
}


def get_tool_parser(name: str) -> type[ToolParser]:
    """The grammar called `name`, or a `module:Class` to import."""
    if name in TOOL_PARSERS:
        return TOOL_PARSERS[name]
    if ":" in name:
        module, _, attr = name.partition(":")
        parser = getattr(importlib.import_module(module), attr)
        if isinstance(parser, type) and issubclass(parser, ToolParser):
            return parser
        raise TypeError(f"{name} is not a ToolParser subclass")
    raise ValueError(f"unknown tool parser {name!r}; known: {list(TOOL_PARSERS)}")


__all__ = ["TOOL_PARSERS", "ToolParser", "get_tool_parser"]
