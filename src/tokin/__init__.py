from importlib.metadata import version

from .chat_templates import ChatTemplate, ChatTemplateError, get_chat_template
from .rollout import FinishReason, Generation, GenerationParams, Prompt, Rollout, Session

__version__ = version("tokin")
__all__ = [
    "ChatTemplate",
    "ChatTemplateError",
    "FinishReason",
    "Generation",
    "GenerationParams",
    "Prompt",
    "Rollout",
    "Session",
    "__version__",
    "get_chat_template",
]
