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
    FunctionToolCall,
    Message,
    PromptMessage,
    SystemMessage,
    ToolCall,
    ToolSchema,
    UserMessage,
)
from ..rollout import FinishReason, Generation
from ..tool_parsers import get_tool_parser


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
        tool_call_parser (str): The grammar the family writes tool calls in, by its `get_tool_parser`
            name, which is sglang's `--tool-call-parser` where it has one.
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
    tool_call_parser: ClassVar[str]
    stop: ClassVar[tuple[str, ...]] = ()
    kwargs: ClassVar[tuple[str, ...]] = ()
    models: ClassVar[tuple[str, ...]] = ()

    def __init__(self, tokenizer: PreTrainedTokenizerBase, **kwargs: Any) -> None:
        self.tokenizer = tokenizer
        if unknown := set(kwargs) - set(self.kwargs):
            raise ValueError(f"{self.name or type(self).__name__} takes no kwarg named {sorted(unknown)}")
        self._kwargs = kwargs
        self.tool_parser = get_tool_parser(self.tool_call_parser)
        self.stop_ids = frozenset(self._token_id(s) for s in self.stop)
        # Whether the generation prompt leaves reasoning open follows from the template and its kwargs alone.
        probe = self.apply([UserMessage(role="user", content="tokin-probe")])
        self.opens_reasoning = bool(self.reasoning_end) and probe.rfind(self.reasoning_start) > probe.rfind(
            self.reasoning_end
        )

    def encode(self, text: str) -> list[int]:
        """Never adds special tokens: the template already wrote the ones it wants."""
        return cast(list[int], self.tokenizer.encode(text, add_special_tokens=False))

    def decode(self, ids: list[int]) -> str:
        """The text of `ids`, special tokens included: nothing the model wrote drops out."""
        return cast(str, self.tokenizer.decode(ids, skip_special_tokens=False))

    def _token_id(self, token: str) -> int:
        """The id of `token`, which must be one token: otherwise the family and the tokenizer disagree."""
        ids = self.encode(token)
        if len(ids) != 1:
            raise ValueError(f"{token!r} is {len(ids)} tokens for this tokenizer, not one")
        return ids[0]

    def order_tool_results(self, messages: list[dict[str, Any]]) -> None:
        """Put each run of tool results in its calls' order, in place; refuse any other arrangement.

        OpenAI's rule: results follow the turn whose calls they answer, one per call, nothing between. Templates
        without call ids pair by position: another arrangement mispairs silently. A turn with no results yet passes.
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
        """Copy `messages` to the shape HF templates read: text joined, `arguments` as dicts, results in call order.

        A call the tool parser could not read renders as its block, the text the model wrote, after the reply's
        content; its result stays where it was.
        """
        copies = cast(list[dict[str, Any]], copy.deepcopy(messages))
        # Only a string is content every template reads; Qwen3's renders a list of parts as nothing.
        for m in (m for m in copies if isinstance(m.get("content"), list)):
            if any(part["type"] != "text" for part in m["content"]):
                raise ChatTemplateError("only text content is supported")
            m["content"] = "".join(part["text"] for part in m["content"])
        # Results are put in order while every call is still listed, so an unread one keeps its result.
        self.order_tool_results(copies)
        for m in (m for m in copies if m.get("tool_calls")):
            calls = []
            for call in m.pop("tool_calls"):
                arguments = call["function"]["arguments"]
                if isinstance(arguments, str) and self.tool_parser.call_pattern.fullmatch(arguments):
                    m["content"] = (m.get("content") or "") + arguments
                    continue
                if isinstance(arguments, str):
                    try:
                        call["function"]["arguments"] = json.loads(arguments)
                    except ValueError as e:
                        raise ChatTemplateError(f"tool call {call['id']!r} has arguments that are not JSON: {e}") from e
                calls.append(call)
            if calls:
                m["tool_calls"] = calls
        return copies

    def apply(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        *,
        add_generation_prompt: bool = True,
    ) -> str:
        """The tokenizer's chat template over `messages`, conformed, with the family's kwargs."""
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
        """A stub history to put before `messages`, so the template renders them.

        Tool results get an assistant turn carrying the calls they answer.
        """
        stub: list[Message] = [
            SystemMessage(role="system", content="tokin-stub-system"),
            UserMessage(role="user", content="tokin-stub-user"),
        ]
        if messages[0]["role"] == "tool":
            if not tool_calls:
                raise ChatTemplateError("tool results need the tool_calls they answer")
            # The stub's render is subtracted, so its calls carry no arguments, which may not be JSON.
            calls = [
                FunctionToolCall(
                    id=c["id"], type="function", function={"name": c["function"]["name"], "arguments": "{}"}
                )
                for c in cast(Iterable[FunctionToolCall], tool_calls)
            ]
            # A non-empty reasoning keeps the think block in place whether or not this turn is the last one.
            stub.append(AssistantMessage(role="assistant", content="", reasoning_content=" ", tool_calls=calls))
        return stub

    def _apply_after_stub(
        self,
        messages: list[PromptMessage],
        tools: list[ToolSchema] | None = None,
        *,
        tool_calls: Iterable[ToolCall] | None = None,
        add_generation_prompt: bool,
    ) -> str:
        """The render of `messages` alone: rendered after a stub history, the stub's own render subtracted."""
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
        """The render of `messages` as the next increment of a rollout, its opening stop token dropped.

        `tool_calls` are those the rollout's last reply made. Messages render run by run: tool results after a stub of
        the reply whose calls they answer, any other run after a bare stub, since a later user message makes a template
        such as Qwen3's drop the reasoning of the replies before it.
        """
        runs = [list(run) for _, run in itertools.groupby(messages, key=lambda m: m["role"] == "tool")]
        text, answered = self.turn_end, tool_calls
        for i, run in enumerate(runs):
            last = i == len(runs) - 1
            if run[0]["role"] == "tool":
                text += self._apply_after_stub(run, tools, tool_calls=answered, add_generation_prompt=last)
            else:
                text += self._apply_after_stub(run, tools, add_generation_prompt=last)
                answered = cast(AssistantMessage, run[-1]).get("tool_calls") if run[-1]["role"] == "assistant" else None
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
            head, closed, text = text.partition(self.reasoning_end)
            # Without `</think>`, a tool call ends the reasoning, as sglang's reasoning parser takes it.
            if not closed and (call := self.tool_parser.call_pattern.search(head)):
                head, text = head[: call.start()], head[call.start() :]
            if reasoning := head.removeprefix(self.reasoning_start).strip():
                message["reasoning_content"] = reasoning
        text, calls = self.tool_parser.parse(text, tools)
        if calls:
            message["tool_calls"] = calls
        if content := text.strip():
            message["content"] = content
        return message
