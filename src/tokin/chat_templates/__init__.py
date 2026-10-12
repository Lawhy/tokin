import importlib

from .chat_template import ChatTemplate, ChatTemplateError
from .glm45 import GLM45ChatTemplate
from .qwen25 import Qwen25ChatTemplate
from .qwen35 import Qwen35ChatTemplate

CHAT_TEMPLATES: dict[str, type[ChatTemplate]] = {
    t.name: t for t in (Qwen25ChatTemplate, Qwen35ChatTemplate, GLM45ChatTemplate)
}


def get_chat_template(name: str) -> type[ChatTemplate]:
    """The family called `name`, the one a checkpoint id in `models` belongs to, or a `module:Class` to import."""
    if name in CHAT_TEMPLATES:
        return CHAT_TEMPLATES[name]
    for template in CHAT_TEMPLATES.values():
        if name in template.models:
            return template
    if ":" in name:
        module, _, attr = name.partition(":")
        template = getattr(importlib.import_module(module), attr)
        if isinstance(template, type) and issubclass(template, ChatTemplate):
            return template
        raise TypeError(f"{name} is not a ChatTemplate subclass")
    raise ValueError(f"unknown chat template {name!r}; known: {list(CHAT_TEMPLATES)}")


__all__ = ["CHAT_TEMPLATES", "ChatTemplate", "ChatTemplateError", "get_chat_template"]
