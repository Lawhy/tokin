from __future__ import annotations

import copy
import itertools
import json
from collections.abc import Iterable
from typing import Any, ClassVar, cast

import jinja2
from transformers import PreTrainedTokenizerBase

from ..messages import (
    AssistantMessage,
    Message,
    PromptMessage,
    SystemMessage,
    ToolCall,
    ToolSchema,
    UserMessage,
)
from ..rollout import FinishReason, Generation
from ..tool_parsers import ToolParser


class ChatTemplateError(RuntimeError):
    """The messages are not ones this template can render as given, which is the request's fault."""


class ChatTemplate:
    """One model family's chat format, bound to a tokenizer.

    A subclass states the family's facts as class attributes and inherits the
    behaviour; a family without a jinja template overrides `apply` as well.

    Attributes:
        name (str): Registry key, such as `"qwen25"`.
        turn_end (str): What the template writes after assistant content, such as
            `"<|im_end|>\n"`. Empty when a turn ends by the next one starting.
        reasoning_start, reasoning_end (str): The tags around reasoning, such as `"<think>"` and
            `"</think>"`. Empty when the family has no reasoning block.
        tool_parser (ToolParser | None): The grammar the family writes tool calls in; `None` when
            it has none.
        stop (tuple[str, ...]): Every token the model may stop on; the server gets them as stop ids.
        kwargs (tuple[str, ...]): Keyword arguments the template reads, such as `enable_thinking`;
            HF passes them as `**kwargs`, the wire as `chat_template_kwargs`, and any other name
            is a mistake the template would swallow.
        models (tuple[str, ...]): Checkpoints the family was verified against; the tokenizer
            tests run over them.
    """

    name: ClassVar[str] = ""
    turn_end: ClassVar[str] = ""
    reasoning_start: ClassVar[str] = ""
    reasoning_end: ClassVar[str] = ""
    tool_parser: ClassVar[ToolParser | None] = None
    stop: ClassVar[tuple[str, ...]] = ()
    kwargs: ClassVar[tuple[str, ...]] = ()
    models: ClassVar[tuple[str, ...]] = ()

    def __init__(self, tokenizer: PreTrainedTokenizerBase, **kwargs: Any) -> None:
        self.tokenizer = tokenizer
        if unknown := set(kwargs) - set(self.kwargs):
            raise ValueError(f"{self.name or type(self).__name__} takes no kwarg named {sorted(unknown)}")
        self._kwargs = kwargs
        self.stop_ids = frozenset(self._token_id(s) for s in self.stop)
        # The generation prompt follows from the template and its kwargs alone, so whether it leaves reasoning open does too.
        probe = self.apply([UserMessage(role="user", content="tokin-probe")])
        self.opens_reasoning = bool(self.reasoning_end) and probe.rfind(self.reasoning_start) > probe.rfind(
            self.reasoning_end
        )

    def encode(self, text: str) -> list[int]:
        """Never adds special tokens: the template already wrote the ones it wants."""
        return cast(list[int], self.tokenizer.encode(text, add_special_tokens=False))

    def decode(self, ids: list[int]) -> str:
        """The text of `ids` as sampled, special tokens included, so nothing the model wrote drops out."""
        return cast(str, self.tokenizer.decode(ids, skip_special_tokens=False))

    def _token_id(self, token: str) -> int:
        """Input `token` has to be exactly one token here, or the family and the tokenizer do not match."""
        ids = self.encode(token)
        if len(ids) != 1:
            raise ValueError(f"{token!r} is {len(ids)} tokens for this tokenizer, not one")
        return ids[0]

    def order_tool_results(self, messages: list[dict[str, Any]]) -> None:
        """Put each run of tool results in its calls' order, in place, and refuse any other arrangement.

        OpenAI's rule: a result follows the assistant turn whose call it answers, one result per
        call, nothing in between. Templates without call ids pair the two by position, so any
        other arrangement mispairs them without a word. A turn with no results yet passes; the
        stub renders one.
        """
        i = 0
        while i < len(messages):
            message = messages[i]
            i += 1
            if message["role"] == "tool":
                raise ChatTemplateError(f"tool result {message['tool_call_id']!r} does not follow the calls it answers")
            if calls := message.get("tool_calls"):
                order = [call["id"] for call in calls]
                j = i
                while j < len(messages) and messages[j]["role"] == "tool":
                    j += 1
                results = messages[i:j]
                if results and sorted(r["tool_call_id"] for r in results) != sorted(order):
                    raise ChatTemplateError(
                        f"tool results {[r['tool_call_id'] for r in results]} do not answer {order} one each"
                    )
                messages[i:j] = sorted(results, key=lambda r: order.index(r["tool_call_id"]))
                i = j

    def conform(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Copy `messages` into the shape HF templates read: text parts joined, `arguments` as dicts, tool results in their calls' order."""
        copies = cast(list[dict[str, Any]], copy.deepcopy(messages))
        # Only a string is content every template reads; Qwen3's renders a list of parts as nothing.
        for m in (m for m in copies if isinstance(m.get("content"), list)):
            if any(part["type"] != "text" for part in m["content"]):
                raise ChatTemplateError("only text content is supported")
            m["content"] = "".join(part["text"] for part in m["content"])
        for call in (c for m in copies for c in m.get("tool_calls") or []):
            if isinstance(call["function"]["arguments"], str):
                try:
                    call["function"]["arguments"] = json.loads(call["function"]["arguments"])
                except ValueError as e:
                    raise ChatTemplateError(f"tool call {call['id']!r} has arguments that are not JSON: {e}") from e
        self.order_tool_results(copies)
        return copies

    def apply(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        *,
        add_generation_prompt: bool = True,
    ) -> str:
        """Apply the tokenizer's built-in chat template with this family's argument shape and kwargs baked in."""
        conformed = self.conform(messages)
        try:
            text = self.tokenizer.apply_chat_template(
                conformed,
                tools=cast(Any, tools) or None,
                add_generation_prompt=add_generation_prompt,
                tokenize=False,
                **self._kwargs,
            )
        except jinja2.TemplateError as e:
            raise ChatTemplateError(f"template refuses {[m['role'] for m in messages]}: {e}") from e
        return cast(str, text)

    def _stub(self, messages: list[PromptMessage], tool_calls: Iterable[ToolCall] | None) -> list[Message]:
        """A stub history to put before `messages` so the template renders them; tool results get the assistant whose calls they answer."""
        stub: list[Message] = [
            SystemMessage(role="system", content="tokin-stub-system"),
            UserMessage(role="user", content="tokin-stub-user"),
        ]
        if messages[0]["role"] == "tool":
            if not tool_calls:
                raise ChatTemplateError("tool results need the tool_calls they answer")
            # A non-empty reasoning keeps the think block in place whether or not this turn is the last one.
            stub.append(
                AssistantMessage(role="assistant", content="", reasoning_content=" ", tool_calls=list(tool_calls))
            )
        return stub

    def _apply_after_stub(
        self,
        messages: list[PromptMessage],
        tools: list[ToolSchema] | None = None,
        *,
        tool_calls: Iterable[ToolCall] | None = None,
        add_generation_prompt: bool,
    ) -> str:
        """Apply the chat template to `messages` alone by rendering them after a stub history and subtracting it."""
        stub = self._stub(messages, tool_calls)
        before = self.apply(stub, tools, add_generation_prompt=False)
        after = self.apply([*stub, *messages], tools, add_generation_prompt=add_generation_prompt)
        if not after.startswith(before):
            raise ChatTemplateError(f"template rewrites earlier turns when appending {[m['role'] for m in messages]}")
        return after[len(before) :]

    def apply_increment(
        self,
        messages: list[PromptMessage],
        tools: list[ToolSchema] | None = None,
        *,
        tool_calls: Iterable[ToolCall] | None = None,
    ) -> str:
        """Apply the chat template to `messages` as the next increment of a multi-turn rollout."""
        results = list(itertools.takewhile(lambda m: m["role"] == "tool", messages))
        rest = messages[len(results) :]
        if any(m["role"] == "tool" for m in rest):
            raise ChatTemplateError("tool results come before any other message")
        text = self.turn_end
        if results:
            text += self._apply_after_stub(results, tools, tool_calls=tool_calls, add_generation_prompt=not rest)
        if rest:
            text += self._apply_after_stub(rest, tools, add_generation_prompt=True)
        # The model stopped on the token that opens the increment, so the prompt already has it.
        for s in self.stop:
            if text.startswith(s):
                return text[len(s) :]
        raise ChatTemplateError(f"the increment opens with {text[:16]!r}, which the model never stops on")

    def parse(self, generation: Generation, tools: list[ToolSchema] | None = None) -> AssistantMessage:
        """Read back the assistant message `generation` wrote.

        Args:
            generation (Generation): What the engine sampled; a stop token it ended on is not text.
            tools (list[ToolSchema] | None): The schemas the calls may name; XML grammars type
                their argument values by them.
        """
        stopped = generation.finish_reason is FinishReason.STOP
        text = self.decode(generation.token_ids[:-1] if stopped else generation.token_ids).lstrip()
        message: AssistantMessage = {"role": "assistant", "content": None}
        if self.reasoning_end and (self.opens_reasoning or text.startswith(self.reasoning_start)):
            head, _, text = text.partition(self.reasoning_end)
            if reasoning := head.removeprefix(self.reasoning_start).strip():
                message["reasoning_content"] = reasoning
        if self.tool_parser:
            text, calls = self.tool_parser.parse(text, tools)
            if calls:
                message["tool_calls"] = calls
        if content := text.strip():
            message["content"] = content
        return message
