"""Optional real-model adapters (OpenAI, Anthropic).

Install extras to use them::

    pip install -e ".[openai]"      # OPENAI_API_KEY
    pip install -e ".[anthropic]"   # ANTHROPIC_API_KEY

Both translate AgentEval's provider-agnostic :class:`Message`/:class:`ToolSpec` types
into the provider's native tool-calling format.
"""

from __future__ import annotations

import json
import time
from typing import Any

from agenteval.agents.llm import LLMClient, LLMRequest, LLMResponse
from agenteval.types import Message, ToolCall


def _retry(fn: Any, attempts: int = 4, base_delay: float = 1.5) -> Any:
    """Call ``fn`` with exponential backoff on transient errors."""
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # provider SDKs raise many error types
            transient = any(s in type(e).__name__ for s in ("RateLimit", "Timeout", "Connection", "Overloaded"))
            if not transient or i == attempts - 1:
                raise
            time.sleep(base_delay * (2**i))
    raise RuntimeError("unreachable")


class OpenAIClient(LLMClient):
    """OpenAI Chat Completions adapter with native function calling."""

    def __init__(self, model: str = "gpt-4o", api_key: str | None = None, max_tokens: int = 1024) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError("OpenAI adapter requires `pip install agenteval[openai]`") from e
        self._client = OpenAI(api_key=api_key)
        self.model = model
        self.name = f"openai:{model}"
        self.max_tokens = max_tokens

    @staticmethod
    def to_provider_messages(messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "tool" and m.tool_result is not None:
                out.append({"role": "tool", "tool_call_id": m.tool_result.call_id, "content": m.tool_result.to_text()})
            elif m.role == "assistant" and m.tool_calls:
                out.append({
                    "role": "assistant",
                    "content": m.content or None,
                    "tool_calls": [
                        {"id": tc.id, "type": "function",
                         "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                        for tc in m.tool_calls
                    ],
                })
            else:
                out.append({"role": m.role, "content": m.content})
        return out

    def complete(self, request: LLMRequest) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self.to_provider_messages(request.messages),
            "temperature": request.temperature,
            "max_tokens": self.max_tokens,
        }
        if request.mode == "act" and request.tools:
            kwargs["tools"] = [t.to_openai() for t in request.tools]
        if request.mode == "judge":
            kwargs["response_format"] = {"type": "json_object"}
        resp = _retry(lambda: self._client.chat.completions.create(**kwargs))
        msg = resp.choices[0].message
        calls = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"__raw__": tc.function.arguments}
            calls.append(ToolCall(name=tc.function.name, arguments=args, id=tc.id))
        usage = resp.usage
        return LLMResponse(
            content=msg.content or "",
            tool_calls=calls,
            prompt_tokens=getattr(usage, "prompt_tokens", 0),
            completion_tokens=getattr(usage, "completion_tokens", 0),
        )


class AnthropicClient(LLMClient):
    """Anthropic Messages API adapter with native tool use."""

    def __init__(self, model: str = "claude-sonnet-4-5", api_key: str | None = None, max_tokens: int = 1024) -> None:
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError("Anthropic adapter requires `pip install agenteval[anthropic]`") from e
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.name = f"anthropic:{model}"
        self.max_tokens = max_tokens

    @staticmethod
    def to_provider_messages(messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
        """Return ``(system_prompt, messages)``, merging adjacent same-role turns."""
        system_parts: list[str] = []
        out: list[dict[str, Any]] = []

        def push(role: str, blocks: list[dict[str, Any]]) -> None:
            if out and out[-1]["role"] == role:
                out[-1]["content"].extend(blocks)
            else:
                out.append({"role": role, "content": blocks})

        for m in messages:
            if m.role == "system":
                system_parts.append(m.content)
            elif m.role == "tool" and m.tool_result is not None:
                push("user", [{"type": "tool_result", "tool_use_id": m.tool_result.call_id,
                               "content": m.tool_result.to_text(), "is_error": not m.tool_result.ok}])
            elif m.role == "assistant":
                blocks: list[dict[str, Any]] = [{"type": "text", "text": m.content}] if m.content else []
                blocks += [{"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
                           for tc in m.tool_calls]
                push("assistant", blocks or [{"type": "text", "text": "(no content)"}])
            else:
                push("user", [{"type": "text", "text": m.content}])
        return "\n\n".join(system_parts), out

    def complete(self, request: LLMRequest) -> LLMResponse:
        system, msgs = self.to_provider_messages(request.messages)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "system": system,
            "messages": msgs,
            "temperature": request.temperature,
            "max_tokens": self.max_tokens,
        }
        if request.mode == "act" and request.tools:
            kwargs["tools"] = [t.to_anthropic() for t in request.tools]
        resp = _retry(lambda: self._client.messages.create(**kwargs))
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        calls = [ToolCall(name=b.name, arguments=dict(b.input), id=b.id)
                 for b in resp.content if getattr(b, "type", "") == "tool_use"]
        return LLMResponse(
            content=text,
            tool_calls=calls,
            prompt_tokens=resp.usage.input_tokens,
            completion_tokens=resp.usage.output_tokens,
        )
