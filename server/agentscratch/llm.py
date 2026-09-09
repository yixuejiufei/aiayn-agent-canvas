"""Local Transformer LLM brain adapter.

M2 connects to an OpenAI-compatible local model server such as llama.cpp's
``llama-server``, Ollama, or vLLM. The adapter does not pretend a model exists:
connection failures, invalid JSON, and schema violations are surfaced to the run
timeline so they can be used as teaching material.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from pydantic import ValidationError

from .prompts import PromptBuilder
from .schemas import AgentAction, MiniAgentSpec, ToolCall


@dataclass(slots=True)
class BrainResult:
    action: AgentAction
    raw_output: str = ""
    model: str | None = None
    duration_ms: int | None = None
    attempts: int = 1
    usage: dict[str, Any] = field(default_factory=dict)
    request_messages: list[dict[str, Any]] = field(default_factory=list)
    request_tools: list[dict[str, Any]] = field(default_factory=list)
    protocol: str = "json_action"


class LLMParseError(ValueError):
    def __init__(self, message: str, raw_output: str = "", model: str | None = None) -> None:
        super().__init__(message)
        self.raw_output = raw_output
        # A workflow-level validation error can happen after a real model
        # response was received.  Preserve that server-reported identity so
        # metrics never split one run between requested and actual models.
        self.model = model


def _extract_json_object(text: str) -> dict[str, Any]:
    """Parse one JSON object from fenced output or surrounding prose.

    ``JSONDecoder.raw_decode`` is used instead of a regular expression so
    nested objects in ``args`` and braces inside strings are handled safely.
    """
    decoder = json.JSONDecoder()
    saw_object_start = False
    last_error: json.JSONDecodeError | None = None

    for index, character in enumerate(text):
        if character != "{":
            continue
        saw_object_start = True
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if not isinstance(value, dict):
            raise LLMParseError("model output JSON was not an object", text)
        return value

    if not saw_object_start:
        raise LLMParseError("model output did not contain a JSON object", text)
    detail = f": {last_error}" if last_error else ""
    raise LLMParseError(f"model output was not valid JSON{detail}", text)


def parse_action(text: str) -> AgentAction:
    try:
        return AgentAction.model_validate(_extract_json_object(text))
    except ValidationError as exc:
        raise LLMParseError(f"model output did not match AgentAction schema: {exc}", text) from exc


class TinyLLMBrain:
    """OpenAI-compatible local model client for llama.cpp, Ollama, vLLM, etc."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.base_url = (
            base_url or os.getenv("AGENTSCRATCH_LLM_BASE_URL", "http://127.0.0.1:8080/v1")
        ).rstrip("/")
        self.model = model or os.getenv("AGENTSCRATCH_LLM_MODEL", "qwen-tiny")
        self.api_key = api_key or os.getenv("AGENTSCRATCH_LLM_API_KEY", "local")
        self.timeout = timeout if timeout is not None else float(os.getenv("AGENTSCRATCH_LLM_TIMEOUT", "60"))
        self.prompt_builder = PromptBuilder()

    def _messages(self, context: list[dict[str, Any]], spec: MiniAgentSpec) -> list[dict[str, str]]:
        return self.prompt_builder.build(context, [tool.model_dump() for tool in spec.tools])

    @staticmethod
    def _native_tools(spec: MiniAgentSpec) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            name: {"type": parameter.type, "description": parameter.description or ""}
                            for name, parameter in tool.parameters.items()
                        },
                        "required": [name for name, parameter in tool.parameters.items() if parameter.required],
                    },
                },
            }
            for tool in spec.tools
            if tool.name != "final_answer"
        ]

    @staticmethod
    def _json_action_tools(spec: MiniAgentSpec) -> list[dict[str, Any]]:
        """The tool manifest encoded into the JSON-action prompt adapter."""
        return [tool.model_dump() for tool in spec.tools if tool.name != "final_answer"]

    def _completion(self, spec: MiniAgentSpec, context: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
        if spec.brain.tool_protocol == "native_tools":
            messages = self.prompt_builder.build_native(context)
            tools = self._native_tools(spec)
            payload: dict[str, Any] = {
                "model": spec.brain.model or self.model,
                "messages": messages,
                "temperature": spec.brain.temperature,
                "top_p": spec.brain.top_p,
                "top_k": spec.brain.top_k,
                "max_tokens": spec.brain.max_tokens,
                "repeat_penalty": spec.brain.repeat_penalty,
                "tools": tools,
                "tool_choice": "auto",
            }
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
                response.raise_for_status()
                return response.json(), {"messages": messages, "tools": tools, "protocol": "native_tools"}

        action_schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "thought": {"type": "string"},
                "action": {"type": "string", "enum": ["call_tool", "call_tools", "final_answer", "ask_user", "route"]},
                "tool": {"type": ["string", "null"]},
                "target": {"type": ["string", "null"]},
                "call_id": {"type": ["string", "null"]},
                "args": {"type": "object"},
                "tool_calls": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "tool": {"type": "string"},
                            "args": {"type": "object"},
                            "call_id": {"type": ["string", "null"]},
                        },
                        "required": ["tool", "args", "call_id"],
                        "additionalProperties": False,
                    },
                },
                "answer": {"type": ["string", "null"]},
                "done": {"type": "boolean"},
            },
            "required": ["thought", "action", "tool", "target", "call_id", "args", "tool_calls", "answer", "done"],
            "additionalProperties": False,
        }
        payload: dict[str, Any] = {
            "model": spec.brain.model or self.model,
            "messages": self._messages(context, spec),
            "temperature": spec.brain.temperature,
            "top_p": spec.brain.top_p,
            "top_k": spec.brain.top_k,
            "max_tokens": spec.brain.max_tokens,
            "repeat_penalty": spec.brain.repeat_penalty,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "agent_action", "strict": True, "schema": action_schema},
            },
        }
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
            # Some small local OpenAI-compatible servers expose chat
            # completions but not the newer JSON Schema response_format.
            # PromptBuilder already asks for JSON, so preserve the teaching
            # protocol and fall back to plain chat rather than failing a run.
            if getattr(response, "status_code", None) == 400:
                fallback_payload = {key: value for key, value in payload.items() if key != "response_format"}
                response = client.post(f"{self.base_url}/chat/completions", json=fallback_payload, headers=headers)
            response.raise_for_status()
            return response.json(), {
                "messages": payload["messages"],
                "tools": self._json_action_tools(spec),
                "protocol": "json_action",
            }

    @staticmethod
    def _http_error_detail(error: httpx.HTTPError) -> str:
        """Preserve a bounded local-server explanation for replay and repair."""
        if not isinstance(error, httpx.HTTPStatusError):
            return str(error)
        try:
            body = error.response.text.strip()
        except (AttributeError, httpx.HTTPError):
            body = ""
        if not body:
            return str(error)
        # llama.cpp uses this body for valuable errors such as context-window
        # overflow. Keep it bounded so an upstream HTML/proxy error cannot
        # become another large Context message on the next repair turn.
        return f"HTTP {error.response.status_code}: {body[:800]}"

    @staticmethod
    def _native_action(message: dict[str, Any]) -> AgentAction:
        calls = message.get("tool_calls") or []
        if calls:
            parsed_calls: list[ToolCall] = []
            for call in calls:
                function = call.get("function") or {}
                arguments = function.get("arguments", "{}")
                try:
                    args = json.loads(arguments) if isinstance(arguments, str) else arguments
                except json.JSONDecodeError as exc:
                    raise LLMParseError(f"native tool call arguments were not valid JSON: {exc}", str(arguments)) from exc
                if not isinstance(args, dict):
                    raise LLMParseError("native tool call arguments were not an object", str(arguments))
                parsed_calls.append(ToolCall(tool=function.get("name", ""), call_id=call.get("id"), args=args))
            if len(parsed_calls) == 1:
                call = parsed_calls[0]
                return AgentAction(
                    thought="The model requested a native tool call.", action="call_tool",
                    tool=call.tool, call_id=call.call_id, args=call.args,
                )
            return AgentAction(
                thought=f"The model requested {len(parsed_calls)} native tool calls.",
                action="call_tools", tool_calls=parsed_calls,
            )
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise LLMParseError("native response contained neither content nor tool_calls", json.dumps(message))
        return AgentAction(
            thought="The model returned a final native response.",
            action="final_answer",
            answer=content,
            done=True,
        )

    def decide(self, spec: MiniAgentSpec, context: list[dict[str, Any]]) -> BrainResult:
        started = time.perf_counter()
        max_attempts = max(1, spec.brain.max_attempts)
        last_error: Exception | None = None
        last_raw_output = ""
        usage: dict[str, Any] = {}
        request_snapshot: dict[str, Any] = {"messages": [], "tools": [], "protocol": spec.brain.tool_protocol}

        for attempt in range(1, max_attempts + 1):
            try:
                body, request_snapshot = self._completion(spec, context)
            except httpx.HTTPError as exc:
                if isinstance(exc, httpx.HTTPStatusError):
                    last_error = RuntimeError(
                        f"local LLM request was rejected at {self.base_url}: {self._http_error_detail(exc)}"
                    )
                else:
                    last_error = RuntimeError(f"local LLM service unavailable at {self.base_url}: {self._http_error_detail(exc)}")
                if attempt == max_attempts:
                    raise last_error from exc
                continue
            except (TypeError, ValueError) as exc:
                last_error = RuntimeError(f"local LLM returned an invalid response: {exc}")
                if attempt == max_attempts:
                    raise last_error from exc
                continue

            choices = body.get("choices") or []
            message = choices[0].get("message", {}) if choices else {}
            if not choices or (not message.get("content") and not message.get("tool_calls")):
                last_error = LLMParseError("local LLM returned an empty completion")
                if attempt == max_attempts:
                    raise last_error
                continue

            raw_output = message.get("content") or json.dumps(message.get("tool_calls"), ensure_ascii=False)
            last_raw_output = raw_output
            usage = body.get("usage", {}) or {}
            try:
                action = self._native_action(message) if spec.brain.tool_protocol == "native_tools" else parse_action(raw_output)
            except LLMParseError as exc:
                last_error = exc
                if attempt == max_attempts:
                    raise
                continue

            return BrainResult(
                action=action,
                raw_output=raw_output,
                model=body.get("model", self.model),
                duration_ms=int((time.perf_counter() - started) * 1000),
                attempts=attempt,
                usage=usage,
                request_messages=request_snapshot["messages"],
                request_tools=request_snapshot["tools"],
                protocol=request_snapshot["protocol"],
            )

        if last_error is None:
            last_error = LLMParseError("local LLM failed without an error", last_raw_output)
        raise last_error
