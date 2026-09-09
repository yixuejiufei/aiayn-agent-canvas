"""Shared domain schemas for the AgentScratch teaching runtime."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolParameterSpec(BaseModel):
    type: Literal["string", "number", "boolean", "object", "array"]
    description: str | None = None
    required: bool = True


class ToolSpec(BaseModel):
    name: str = Field(min_length=1, pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$")
    description: str = Field(min_length=1)
    parameters: dict[str, ToolParameterSpec] = Field(default_factory=dict)
    # These are runtime policy, not model instructions: the interpreter always
    # enforces them even when a model omits or misunderstands their meaning.
    risk: Literal["read", "write", "sensitive"] = "read"
    permission: Literal["granted", "blocked"] = "granted"
    requires_confirmation: bool = False
    parallel_safe: bool = True


class BrainSpec(BaseModel):
    type: Literal["tiny_llm", "test", "clarification"] = "tiny_llm"
    model: str | None = None
    device: Literal["cpu", "gpu"] = "cpu"
    tool_protocol: Literal["json_action", "native_tools"] = "json_action"
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=40, ge=0, le=200)
    max_tokens: int = Field(default=512, ge=16, le=4096)
    repeat_penalty: float = Field(default=1.1, ge=0.0, le=2.0)
    max_attempts: int = Field(default=2, ge=1, le=5, description="Local LLM completion attempts per decision.")


class AgentPolicy(BaseModel):
    max_steps: int = Field(default=10, ge=1, le=60)
    # A decision failure means the model response could not become a valid
    # action: transport / protocol / JSON / schema / capability validation.
    # It is deliberately independent from a tool that was validly requested
    # but failed while executing.
    max_decision_failures: int = Field(default=2, ge=0, le=60)
    max_tool_failures: int = Field(default=2, ge=0, le=10)


class MiniAgentSpec(BaseModel):
    version: Literal["0.1"] = "0.1"
    name: str = Field(min_length=1)
    description: str | None = None
    brain: BrainSpec = Field(default_factory=BrainSpec)
    tools: list[ToolSpec] = Field(default_factory=list)
    policy: AgentPolicy = Field(default_factory=AgentPolicy)
    workflow: dict[str, Any] | None = None

    def tool(self, name: str) -> ToolSpec | None:
        return next((item for item in self.tools if item.name == name), None)


class ToolCall(BaseModel):
    tool: str = Field(min_length=1)
    args: dict[str, Any] = Field(default_factory=dict)
    call_id: str | None = None


class AgentAction(BaseModel):
    thought: str = Field(min_length=1)
    action: Literal["call_tool", "call_tools", "final_answer", "ask_user", "route"]
    tool: str | None = None
    target: str | None = None
    call_id: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    tool_calls: list[ToolCall] = Field(default_factory=list, max_length=16)
    answer: str | None = None
    done: bool = False


class RunStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    WAITING_USER = "waiting_user"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentEvent(BaseModel):
    event_id: str
    run_id: str
    step: int
    type: str
    timestamp: float
    node: str
    data: dict[str, Any] = Field(default_factory=dict)


class AgentRunState(BaseModel):
    run_id: str
    agent_id: str
    input: str
    status: RunStatus
    step: int = 0
    # Kept as the backwards-compatible total used by older replays.
    failures: int = 0
    decision_failures: int = 0
    tool_failures: int = 0
    policy_blocks: int = 0
    memory: dict[str, Any] = Field(default_factory=dict)
    context: list[dict[str, Any]] = Field(default_factory=list)
    output: str | None = None
    error: str | None = None
    waiting_question: str | None = None
    # Optional, sanitized choices supplied with an ask_user action.  Their
    # opaque ids are only meaningful for the current paused run.
    waiting_choices: list[dict[str, Any]] = Field(default_factory=list)
    # The validated interaction contract for the current ask_user pause.  It
    # is absent for ordinary free-text clarification, where Harness must not
    # infer a hidden domain-specific response type from natural language.
    waiting_response_schema: dict[str, Any] | None = None
    events: list[AgentEvent] = Field(default_factory=list)


class CreateAgentRequest(BaseModel):
    name: str
    description: str | None = None
    tools: list[str] = Field(default_factory=lambda: ["calculator", "final_answer"])
    brain_type: Literal["tiny_llm", "test", "clarification"] = "tiny_llm"
    brain_model: str | None = None


class CreateRunRequest(BaseModel):
    input: str = Field(min_length=1)


class ResumeRunRequest(BaseModel):
    input: str | None = Field(default=None, min_length=1)
    selection_id: str | None = Field(default=None, min_length=1)


class CreateTeachingCaseRequest(BaseModel):
    case_id: str = Field(min_length=2, max_length=48, pattern=r"^[a-z0-9][a-z0-9-]*$")
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    input: str = Field(min_length=1)
    brain_type: Literal["tiny_llm", "test", "clarification"] = "test"
    brain_model: str | None = None
    tools: list[str] = Field(min_length=1)


class WorkflowNode(BaseModel):
    id: str = Field(min_length=1, max_length=96)
    type: Literal[
        "input",
        "workspace",
        "user_input",
        "system_prompt",
        "tool_definition",
        "react_loop",
        "agent",
        "tool",
        "condition",
        "ask_user",
        "final",
    ]
    label: str = Field(min_length=1, max_length=120)
    x: float = Field(default=0.0, ge=-10000, le=10000)
    y: float = Field(default=0.0, ge=-10000, le=10000)
    parent_id: str | None = Field(default=None, alias="parentId", max_length=96)
    config: dict[str, str] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}


class WorkflowEdge(BaseModel):
    id: str = Field(min_length=1, max_length=160)
    from_: str = Field(alias="from", min_length=1, max_length=96)
    to: str = Field(min_length=1, max_length=96)
    label: str | None = Field(default=None, max_length=32)

    model_config = {"populate_by_name": True}


class WorkflowSpec(BaseModel):
    version: Literal["0.1"] = "0.1"
    name: str = Field(min_length=1, max_length=120)
    nodes: list[WorkflowNode] = Field(min_length=1, max_length=64)
    edges: list[WorkflowEdge] = Field(max_length=128)
    max_steps: int = Field(default=10, ge=1, le=60)


class CreateWorkflowRunRequest(BaseModel):
    workflow: WorkflowSpec
    input: str | None = Field(default=None, min_length=1)

