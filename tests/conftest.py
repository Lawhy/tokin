import jinja2
import pytest

from tokin.chat_templates import ChatTemplate
from tokin.gateway import Gateway
from tokin.inference_engines import InferenceEngine
from tokin.rollout import FinishReason, Generation
from tokin.tool_parsers.hermes import HermesToolParser

# One private-use character per control token, so each is a single id.
START, END = "", ""
CHATML = (
    "{%- for m in messages %}{{ start ~ m.role ~ '\\n' }}"
    "{%- if m.role == 'system' and tools %}tools={{ tools | tojson }}{% endif %}"
    "{{ m.content }}"
    "{%- for c in m.tool_calls or [] %}<call>{{ c.function.arguments | tojson }}</call>{% endfor %}"
    "{%- if m.tool_call_id %}<id>{{ m.tool_call_id }}</id>{% endif %}"
    "{{ end ~ '\\n' }}{%- endfor %}"
    "{%- if add_generation_prompt %}{{ start ~ 'assistant\\n' }}"
    "{%- if enable_thinking is defined and not enable_thinking %}<think></think>{% endif %}{% endif %}"
)


class FakeTokenizer:
    def __init__(self, template: str = CHATML) -> None:
        self.chat_template = template
        self._env = jinja2.Environment(extensions=["jinja2.ext.loopcontrols"])

    def apply_chat_template(self, messages, *, tools=None, add_generation_prompt, tokenize, **kwargs):
        def raise_exception(message):
            raise jinja2.TemplateError(message)

        return self._env.from_string(self.chat_template).render(
            messages=messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            raise_exception=raise_exception,
            start=START,
            end=END,
            **kwargs,
        )

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(map(chr, ids))


class ChatML(ChatTemplate):
    name = "chatml"
    turn_end = f"{END}\n"
    reasoning_start = "<think>"
    reasoning_end = "</think>"
    tool_parser = HermesToolParser()
    stop = (END,)
    kwargs = ("enable_thinking",)


@pytest.fixture
def chatml():
    return ChatML(FakeTokenizer())


class Engine(InferenceEngine):
    def __init__(self, *replies, context_length=1000, routing_shape=(2, 1)):
        self.replies, self.calls, self.length, self.shape = list(replies), [], context_length, routing_shape

    async def context_length(self):
        return self.length

    async def routing_shape(self):
        return self.shape

    async def generate(self, token_ids, params):
        self.calls.append((token_ids, params))
        return self.replies.pop(0)


def said(text, finish=FinishReason.STOP):
    return Generation([ord(c) for c in text] + [ord(END)] * (finish is FinishReason.STOP), finish)


@pytest.fixture
def gateway(chatml):
    def make(*replies, context_length=1000):
        return Gateway(chatml, Engine(*replies, context_length=context_length))

    return make
