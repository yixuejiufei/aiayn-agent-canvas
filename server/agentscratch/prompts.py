"""Prompt construction for the local LLM brain.

Keeping prompt construction separate from the HTTP client makes the exact
context shown to a model easy to inspect and test.  The runtime still stores
the original context messages; this builder only adapts them to the
OpenAI-compatible chat format.
"""

from __future__ import annotations

import json
from typing import Any


def _action_instructions(allow_route: bool) -> str:
    actions = "call_tool|call_tools|final_answer|ask_user" + ("|route" if allow_route else "")
    routing_rule = (
        "WORKFLOW_NEXT_NODES lists the only allowed LLM targets; use action route and set target to one of their exact ids. "
        if allow_route
        else "There are no downstream LLM targets in this workflow: route is prohibited. After an Observation, either call a needed tool or return final_answer. "
    )
    return (
        "You are the decision brain of a teaching agent. "
        "Choose exactly one next action based on the context and available tools. "
        "Return only compact JSON matching this schema: "
        f'{{"thought":"string","action":"{actions}",'
        '"tool":"string|null","target":"string|null","call_id":"string|null","args":{},"tool_calls":[],"answer":"string|null","done":false}. '
        "Example for arithmetic: "
        '{"thought":"The user asks for arithmetic.","action":"call_tool",'
        '"tool":"calculator","target":null,"call_id":null,"args":{"expression":"12*7"},"tool_calls":[],"answer":null,"done":false}. '
        "When independent read-only tools can be used together, action may be call_tools and tool_calls is a non-empty list of {tool,args,call_id}; "
        + routing_rule
        + "Example for final answer: "
        + '{"thought":"I already know the answer.","action":"final_answer",'
        + '"tool":null,"target":null,"call_id":null,"args":{},"tool_calls":[],"answer":"84","done":true}. '
        + "After a successful TOOL_RESULT / Observation, first decide whether it satisfies the current subgoal. "
        + "Do not repeat the same tool with identical arguments merely to rediscover the same result; reuse the Observation and continue to the next needed action or final_answer. "
        + "A repeated call is allowed when the task explicitly needs independent verification, a different subgoal, or a state/time-sensitive result; state that purpose in thought. "
        + "For ask_user, put the actual question in args.question. For a file picker after search_files, use args.response_schema exactly as {\"type\":\"file_selection\",\"source_observation\":\"the exact search_files tool_call_id\",\"candidate_file_ids\":[\"one to three exact observed ids\"],\"allow_custom_input\":true}. Use this schema only for a file choice and only with ids from that exact completed Observation. Without response_schema, ask_user is a plain free-text question and no file choices will be shown. "
        + "For read_file, put the selected opaque identifier in args.file_id. The target field is reserved for workflow routing and is never a tool argument. "
        + "Do not explain outside JSON."
    )


class PromptBuilder:
    """Convert AgentScratch context messages into chat-completion messages."""

    def build(
        self,
        context: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, str]]:
        """Adapt canonical tool-call history for JSON-only local models.

        The runtime persists OpenAI-compatible assistant ``tool_calls`` and
        matching ``tool_call_id`` observations for every protocol.  Small local
        servers that only support text completions still need a textual view of
        that history, so this adapter serializes the canonical records without
        changing their meaning or losing the call linkage.
        """
        allow_route = any(
            message.get("role") == "system" and "WORKFLOW_NEXT_NODES=" in str(message.get("content", ""))
            for message in context
        )
        messages: list[dict[str, str]] = [{"role": "system", "content": _action_instructions(allow_route)}]
        tools_payload = [tool for tool in (tools or []) if tool.get("name") != "final_answer"]

        for message in context:
            role = message.get("role")
            if role in {"system", "user"}:
                messages.append({"role": role, "content": str(message.get("content", ""))})
            elif role == "assistant":
                tool_calls = message.get("tool_calls")
                if tool_calls:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": "TOOL_CALLS=" + json.dumps(tool_calls, ensure_ascii=False),
                        }
                    )
                else:
                    messages.append({"role": "assistant", "content": str(message.get("content", ""))})
            elif role == "tool":
                messages.append(
                    {
                        "role": "system",
                        "content": "TOOL_RESULT="
                        + json.dumps(
                            {
                                "tool_call_id": message.get("tool_call_id"),
                                "name": message.get("name"),
                                "result": message.get("content"),
                            },
                            ensure_ascii=False,
                            default=str,
                        ),
                    }
                )

        if tools_payload:
            messages.append(
                {
                    "role": "system",
                    "content": "AVAILABLE_TOOLS_JSON="
                    + json.dumps(tools_payload, ensure_ascii=False),
                }
            )
        return messages

    def build_native(self, context: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep API roles and tool-call linkage intact for native tool calling."""
        messages: list[dict[str, Any]] = []
        for message in context:
            role = message.get("role")
            if role == "tools":
                continue
            if role in {"system", "user"}:
                messages.append({"role": role, "content": str(message.get("content", ""))})
            elif role == "assistant":
                payload: dict[str, Any] = {"role": "assistant", "content": message.get("content")}
                if message.get("tool_calls"):
                    payload["tool_calls"] = message["tool_calls"]
                messages.append(payload)
            elif role == "tool":
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": message.get("tool_call_id"),
                        "content": str(message.get("content", "")),
                    }
                )
        return messages
