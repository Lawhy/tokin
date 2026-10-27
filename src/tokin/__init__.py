from importlib.metadata import version

from .chat_templates import ChatTemplateError, get_chat_template
from .gateway import ChatRequest, ChatResponse, Gateway, GatewayError
from .inference_engines import InferenceEngineError, SGLangEngine
from .rollout import FinishReason, Generation, GenerationParams, Prompt, Rollout, Turn
from .session import Session

__version__ = version("tokin")
__all__ = [
    "ChatRequest",
    "ChatResponse",
    "ChatTemplateError",
    "FinishReason",
    "Gateway",
    "GatewayError",
    "Generation",
    "GenerationParams",
    "InferenceEngineError",
    "Prompt",
    "Rollout",
    "SGLangEngine",
    "Session",
    "Turn",
    "__version__",
    "get_chat_template",
]
