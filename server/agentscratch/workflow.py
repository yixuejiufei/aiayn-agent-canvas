"""Compile module-oriented AgentScratch canvases into the observable ReAct loop."""

from __future__ import annotations

import json
from typing import Any

from .brain import Brain
from .llm import BrainResult, LLMParseError
from .schemas import AgentAction, AgentRunState, MiniAgentSpec, WorkflowEdge, WorkflowNode, WorkflowSpec


class WorkflowValidationError(ValueError):
    pass


def parse_module_ids(node: WorkflowNode, key: str = "tool_ids") -> list[str]:
    """Read an ID list stored in a string-only editor config field."""
    raw = node.config.get(key, "").strip()
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise WorkflowValidationError(f"module {node.label} has invalid {key} JSON") from exc
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise WorkflowValidationError(f"module {node.label} {key} must be a JSON string array")
    return value


def validate_workflow(workflow: WorkflowSpec) -> None:
    node_ids = [node.id for node in workflow.nodes]
    if len(set(node_ids)) != len(node_ids):
        raise WorkflowValidationError("workflow node ids must be unique")
    nodes = {node.id: node for node in workflow.nodes}
    contexts = [node for node in workflow.nodes if node.type == "input"]
    agents = [node for node in workflow.nodes if node.type == "agent"]
    loops = [node for node in workflow.nodes if node.type == "react_loop"]
    workspaces = [node for node in workflow.nodes if node.type == "workspace"]
    if len(contexts) != 1:
        raise WorkflowValidationError("workflow must contain exactly one context module")
    if not agents:
        raise WorkflowValidationError("workflow must contain at least one agent module")
    if len(loops) > 1:
        raise WorkflowValidationError("teaching workflows currently support one ReAct Loop module per run")
    if len(workspaces) > 1:
        raise WorkflowValidationError("teaching workflows currently support at most one Teaching Workspace module per run")

    outgoing: dict[str, list[WorkflowEdge]] = {node_id: [] for node_id in nodes}
    for edge in workflow.edges:
        if edge.from_ not in nodes or edge.to not in nodes:
            raise WorkflowValidationError(f"edge references an unknown module: {edge.id}")
        if any(existing.to == edge.to for existing in outgoing[edge.from_]):
            raise WorkflowValidationError(f"duplicate connection: {edge.from_} -> {edge.to}")
        outgoing[edge.from_].append(edge)

    tool_nodes = {node.id: node for node in workflow.nodes if node.type == "tool"}
    for node in workflow.nodes:
        if node.parent_id:
            parent = nodes.get(node.parent_id)
            if parent is None:
                raise WorkflowValidationError(f"module {node.label} references an unknown parent")
            valid_parent = (
                node.type in {"user_input", "system_prompt", "tool_definition"} and parent.type == "input"
            ) or (node.type in {"input", "agent"} and parent.type == "react_loop")
            if not valid_parent:
                raise WorkflowValidationError(
                    "Context contains User Input / System Prompt / Tool Definitions; ReAct Loop contains Context / LLM modules"
                )
        if node.type in {"system_prompt", "tool_definition"} and not node.parent_id:
            raise WorkflowValidationError(f"module {node.label} must be contained in a Context module")
        if node.type == "react_loop":
            for key, default, lower, upper in (
                ("max_steps", str(workflow.max_steps), 1, 60),
                ("max_decision_failures", "2", 0, 60),
                ("max_tool_failures", "2", 0, 10),
            ):
                try:
                    limit = int(node.config.get(key, default))
                except ValueError as exc:
                    raise WorkflowValidationError(f"ReAct Loop module {node.label} has invalid {key}") from exc
                if not lower <= limit <= upper:
                    raise WorkflowValidationError(f"ReAct Loop module {node.label} {key} must be between {lower} and {upper}")
        if node.type == "tool":
            if not node.config.get("tool"):
                raise WorkflowValidationError(f"tool module {node.label} must select a tool")
            _parse_args(node)
        if node.type == "tool_definition":
            unknown = set(parse_module_ids(node)) - set(tool_nodes)
            if unknown:
                raise WorkflowValidationError(f"Tool Definitions module {node.label} selects an unknown tool module")
        if node.type == "agent":
            unknown = set(parse_module_ids(node)) - set(tool_nodes)
            if unknown:
                raise WorkflowValidationError(f"LLM module {node.label} selects an unknown tool module")
        edges = outgoing[node.id]
        if node.type not in {"agent", "condition"} and len(edges) > 1:
            raise WorkflowValidationError(f"module {node.label} has more than one outgoing connection")
        if node.type == "condition":
            labels = {edge.label for edge in edges}
            if len(edges) != 2 or labels != {"true", "false"}:
                raise WorkflowValidationError(
                    f"condition module {node.label} needs exactly one true and one false connection"
                )

    # A context module needs one entry point.  Tool modules are deliberately
    # excluded: they are capabilities, not procedural graph steps.
    context_is_contained = contexts[0].parent_id and nodes[contexts[0].parent_id].type == "react_loop"
    if not context_is_contained and not any(nodes[edge.to].type in {"agent", "react_loop"} for edge in outgoing[contexts[0].id]):
        raise WorkflowValidationError("Context module must connect to an LLM or ReAct Loop module")


def _parse_args(node: WorkflowNode) -> dict[str, Any]:
    """Legacy fixed-argument support for old authored workflow examples."""
    raw = node.config.get("args", "").strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise WorkflowValidationError(f"tool module {node.label} has invalid args JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise WorkflowValidationError(f"tool module {node.label} args must be a JSON object")
    return value


class WorkflowBrain:
    """A module-aware brain that keeps ReAct decisions with the model.

    New canvases register Tool modules independently and associate them with an
    Agent through ``tool_ids``.  Old Agent → Tool edges still work so existing
    saved teaching examples remain executable.
    """

    def __init__(
        self,
        workflow: WorkflowSpec,
        delegates: dict[str, Brain],
        agent_specs: dict[str, MiniAgentSpec],
        run: AgentRunState | None = None,
    ):
        validate_workflow(workflow)
        self.workflow = workflow
        self.delegates = delegates
        self.agent_specs = agent_specs
        self.run = run
        self.nodes = {node.id: node for node in workflow.nodes}
        self.outgoing: dict[str, list[WorkflowEdge]] = {node.id: [] for node in workflow.nodes}
        for edge in workflow.edges:
            self.outgoing[edge.from_].append(edge)
        self.context_node_id = next(node.id for node in workflow.nodes if node.type == "input")
        self.current_node_id = self.context_node_id
        if run is not None:
            self._restore_cursor(run)

    def _restore_cursor(self, run: AgentRunState) -> None:
        for event in reversed(run.events):
            if event.type not in {"tool.end", "tool.error", "agent.waiting_user", "agent.route", "agent.finish"}:
                continue
            action = self._action_event_for_step(run, event.step)
            target_id = action.target if action else event.data.get("target")
            if isinstance(target_id, str) and target_id in self.nodes:
                self.current_node_id = target_id
            return

    @staticmethod
    def _action_event_for_step(run: AgentRunState, step: int) -> AgentAction | None:
        for event in reversed(run.events):
            if event.step == step and event.type == "llm.end":
                action = event.data.get("action")
                return AgentAction.model_validate(action) if isinstance(action, dict) else None
        return None

    def _successors(self, node_id: str) -> list[WorkflowNode]:
        return [self.nodes[edge.to] for edge in self.outgoing[node_id]]

    def _next_after(self, node: WorkflowNode) -> WorkflowNode:
        if node.type == "input" and node.parent_id and self.nodes[node.parent_id].type == "react_loop":
            return self.nodes[node.parent_id]
        if node.type == "react_loop":
            contained_agent = next(
                (item for item in self.nodes.values() if item.type == "agent" and item.parent_id == node.id),
                None,
            )
            if contained_agent is not None:
                return contained_agent
        successors = self._successors(node.id)
        if not successors:
            raise WorkflowValidationError(f"module {node.label} has no outgoing connection")
        return successors[0]

    def _is_autonomous_agent(self, node: WorkflowNode) -> bool:
        if "tool_ids" in node.config:
            return True
        return not any(target.type in {"tool", "ask_user", "final"} for target in self._successors(node.id))

    @staticmethod
    def _last_tool_result(context: list[dict[str, Any]]) -> Any:
        for message in reversed(context):
            if message.get("role") == "tool":
                return message.get("content")
        return None

    @staticmethod
    def _last_user_input(context: list[dict[str, Any]]) -> Any:
        for message in reversed(context):
            if message.get("role") == "user":
                return message.get("content")
        return None

    def _environment_context(self, node: WorkflowNode) -> dict[str, Any]:
        """Build a compact factual index of the current execution environment.

        The canonical conversation still carries the full Tool Observations.
        This is a short projection for small models: it says what resources and
        results already exist, without selecting the next action or asserting a
        task-completion condition on the model's behalf.
        """
        if self.run is None:
            return {"completed_observations": [], "execution_budget": {}}

        observations = self.run.memory.get("observations", {})
        completed: list[dict[str, Any]] = []
        if isinstance(observations, dict):
            for call_id, observation in observations.items():
                if not isinstance(observation, dict):
                    continue
                item: dict[str, Any] = {
                    "call_id": call_id,
                    "tool": observation.get("tool"),
                    "status": observation.get("status"),
                    "args": observation.get("args", {}),
                    "result_available_in_context": observation.get("status") == "completed",
                }
                result = observation.get("result")
                # Keep this index compact.  File contents and other full tool
                # payloads remain only in their canonical TOOL_RESULT message.
                if isinstance(result, dict):
                    metadata = {
                        key: result[key]
                        for key in ("file_id", "name", "directory", "characters", "count")
                        if key in result
                    }
                    if metadata:
                        item["result_metadata"] = metadata
                completed.append(item)

        policy = self.agent_specs[node.id].policy
        remaining = max(0, policy.max_steps - self.run.step)
        workspace = self.run.memory.get("workspace", {})
        return {
            "workspace": workspace if isinstance(workspace, dict) else {},
            "completed_observations": completed,
            "known_file_ids": list(self.run.memory.get("discovered_file_ids", [])),
            "tool_constraints": {
                "read_file": "file_id must originate from a successful search_files Observation",
                "identical_search_files": "an identical completed search_files call is rejected by Harness",
            },
            "execution_budget": {
                "step": self.run.step,
                "max_steps": policy.max_steps,
                "remaining_steps": remaining,
                "decision_failures": self.run.decision_failures,
                "max_decision_failures": policy.max_decision_failures,
                "tool_failures": self.run.tool_failures,
                "max_tool_failures": policy.max_tool_failures,
                "policy_blocks": self.run.policy_blocks,
            },
        }

    def _find_target(self, action: AgentAction, source_id: str) -> WorkflowNode | None:
        successors = self._successors(source_id)
        if action.target:
            exact = next((node for node in successors if node.id == action.target), None)
            if exact:
                return exact
            by_label = next((node for node in successors if node.label == action.target), None)
            if by_label:
                return by_label
        if action.action == "route":
            return next((node for node in successors if node.type == "agent"), None)
        if action.action in {"call_tool", "call_tools"}:
            return next((node for node in successors if node.type == "tool" and node.config.get("tool") == action.tool), None)
        if action.action == "ask_user":
            return next((node for node in successors if node.type == "ask_user"), None)
        if action.action == "final_answer":
            return next((node for node in successors if node.type == "final"), None)
        return None

    def _agent_context(self, node: WorkflowNode, context: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # Tool capability is supplied separately via the Agent spec when the
        # delegate builds its request.  The persisted context itself stays a
        # canonical conversation rather than carrying a synthetic ``tools``
        # role that would be mistaken for an Observation by model adapters.
        scoped_context = list(context)
        instruction = node.config.get("prompt", "").strip()
        if instruction:
            scoped_context.append({"role": "system", "content": f"AGENT_MODULE_INSTRUCTION={instruction}"})
        next_agents = [
            {"id": target.id, "label": target.label}
            for target in self._successors(node.id)
            if target.type == "agent"
        ]
        if next_agents:
            scoped_context.append(
                {
                    "role": "system",
                    "content": "WORKFLOW_NEXT_NODES=" + json.dumps(
                        {"current_agent": node.id, "next_agents": next_agents}, ensure_ascii=False
                    ),
                }
            )
        if self.run is not None:
            tool_calls = sum(1 for message in context if message.get("role") == "tool")
            remaining = max(0, self.agent_specs[node.id].policy.max_steps - self.run.step)
            scoped_context.append(
                {
                    "role": "user",
                    "content": (
                        "<agent_status>\n"
                        f"step: {self.run.step}/{self.agent_specs[node.id].policy.max_steps}\n"
                        f"remaining_steps: {remaining}\n"
                        f"tool_observations: {tool_calls}\n"
                        f"decision_failures: {self.run.decision_failures}/{self.agent_specs[node.id].policy.max_decision_failures}\n"
                        f"tool_failures: {self.run.tool_failures}/{self.agent_specs[node.id].policy.max_tool_failures}\n"
                        f"policy_blocks: {self.run.policy_blocks}\n"
                        "</agent_status>"
                    ),
                }
            )
            scoped_context.append(
                {
                    "role": "system",
                    "content": "ENVIRONMENT_CONTEXT_JSON=" + json.dumps(
                        self._environment_context(node), ensure_ascii=False, default=str
                    ),
                }
            )
        return scoped_context

    def _final_action(self, node: WorkflowNode, context: list[dict[str, Any]]) -> BrainResult:
        template = node.config.get("answer", "{{last_tool_result}}")
        last_result = self._last_tool_result(context)
        answer = template.replace("{{last_tool_result}}", "" if last_result is None else str(last_result))
        return BrainResult(
            action=AgentAction(
                thought="The workflow reached its output module.",
                action="final_answer",
                target=node.id,
                answer=answer,
                done=True,
            ),
            raw_output="workflow-output-module",
            model="workflow-graph",
            duration_ms=0,
        )

    def _condition_value(self, node: WorkflowNode, context: list[dict[str, Any]]) -> Any:
        source = node.config.get("source", "last_tool_result")
        if source == "input":
            return next((message.get("content") for message in context if message.get("role") == "user"), None)
        if source == "last_user_input":
            return self._last_user_input(context)
        return self._last_tool_result(context)

    def _condition_action(self, node: WorkflowNode, context: list[dict[str, Any]]) -> BrainResult:
        value = self._condition_value(node, context)
        operator, expected = node.config.get("operator", "equals"), node.config.get("value", "")
        if operator == "is_truthy":
            passed = bool(value)
        elif operator == "is_empty":
            passed = not bool(value)
        elif operator == "contains":
            passed = expected in str(value)
        elif operator in {"greater_than", "less_than"}:
            try:
                passed = float(value) > float(expected) if operator == "greater_than" else float(value) < float(expected)
            except (TypeError, ValueError):
                passed = False
        else:
            passed = str(value) == expected
        branch = "true" if passed else "false"
        edge = next(edge for edge in self.outgoing[node.id] if edge.label == branch)
        target = self.nodes[edge.to]
        self.current_node_id = target.id
        result = self._auto_action(target, context)
        return BrainResult(
            action=result.action.model_copy(update={"thought": f"Condition {node.label} evaluated {branch}. {result.action.thought}"}),
            raw_output=f"workflow-condition:{branch}; {result.raw_output}",
            model=result.model,
            duration_ms=result.duration_ms,
            attempts=result.attempts,
            usage=result.usage,
            request_messages=result.request_messages,
            request_tools=result.request_tools,
            protocol=result.protocol,
        )

    def _auto_action(self, node: WorkflowNode, context: list[dict[str, Any]]) -> BrainResult:
        if node.type == "tool":  # legacy fixed edge workflow
            return BrainResult(
                action=AgentAction(
                    thought=f"The legacy graph routes to {node.config['tool']}.",
                    action="call_tool",
                    target=node.id,
                    tool=node.config["tool"],
                    args=_parse_args(node),
                ),
                raw_output="legacy-graph-tool",
                model="workflow-graph",
                duration_ms=0,
            )
        if node.type == "ask_user":
            fixed_question = node.config.get("question", "").strip() or "请补充完成当前任务所需的信息。"
            return BrainResult(
                action=AgentAction(
                    thought="The graph requests clarification.", action="ask_user", target=node.id,
                    args={"question": fixed_question},
                ), raw_output="workflow-ask-user", model="workflow-graph", duration_ms=0,
            )
        if node.type == "final":
            return self._final_action(node, context)
        if node.type == "condition":
            return self._condition_action(node, context)
        if node.type == "agent":
            return self._decide_from_agent(node, context)
        next_node = self._next_after(node)
        self.current_node_id = next_node.id
        return self._auto_action(next_node, context)

    def _decide_from_agent(self, node: WorkflowNode, context: list[dict[str, Any]]) -> BrainResult:
        spec = self.agent_specs[node.id]
        default_route = node.config.get("default_route", "").strip()
        # ``demo_answer`` used to bypass the configured brain entirely.  It
        # made a case appear to use a selected local model while always
        # returning the same canned string, and persisted in older canvases.
        # A direct-answer workflow must still let its LLM make final_answer.
        if default_route:
            result = BrainResult(
                action=AgentAction(thought=f"The demo route selects {default_route}.", action="route", target=default_route),
                raw_output="workflow-default-route", model="workflow-graph", duration_ms=0,
            )
        else:
            agent_context = self._agent_context(node, context)
            result = self.delegates[node.id].decide(spec, agent_context)
            if not result.request_messages:
                result = BrainResult(
                    action=result.action,
                    raw_output=result.raw_output,
                    model=result.model,
                    duration_ms=result.duration_ms,
                    attempts=result.attempts,
                    usage=result.usage,
                    request_messages=agent_context,
                    protocol=spec.brain.tool_protocol,
                )

        action = result.action
        # Some small JSON-action models understand the selected file identity
        # but place it in ``target`` (a workflow-routing field) instead of the
        # read_file schema.  Repair only that exact, already user-confirmed
        # value; this is protocol normalization, not Harness choosing a file
        # or the next action for the model.
        selection = self.run.memory.get("user_file_selection") if self.run is not None else None
        if (
            action.action == "call_tool"
            and action.tool == "read_file"
            and not action.args.get("file_id")
            and isinstance(selection, dict)
            and selection.get("status") == "resolved"
            and action.target == selection.get("file_id")
            and isinstance(selection.get("file_id"), str)
        ):
            action = action.model_copy(update={"args": {**action.args, "file_id": selection["file_id"]}})
        autonomous = self._is_autonomous_agent(node)
        if action.action == "route":
            target = self._find_target(action, node.id)
            if target is None or target.type != "agent":
                raise LLMParseError(
                    f"LLM module {node.label} routed to an unconnected LLM module",
                    result.raw_output,
                    result.model,
                )
            else:
                self.current_node_id = target.id
                action = action.model_copy(update={"target": target.id})
        elif action.action in {"call_tool", "call_tools"} and autonomous:
            calls = action.tool_calls if action.action == "call_tools" else [action]
            if not calls or any(not getattr(call, "tool", None) or spec.tool(call.tool) is None for call in calls):
                raise LLMParseError(
                    f"LLM module {node.label} selected a tool outside its capability list",
                    result.raw_output,
                    result.model,
                )
            self.current_node_id = node.id
            action = action.model_copy(update={"target": node.id})
        elif action.action in {"final_answer", "ask_user"} and autonomous:
            self.current_node_id = node.id
            action = action.model_copy(update={"target": node.id})
        else:  # legacy graph routing
            target = self._find_target(action, node.id)
            if target is None:
                raise LLMParseError(
                    f"LLM module {node.label} produced {action.action} without a matching outgoing module",
                    result.raw_output,
                    result.model,
                )
            self.current_node_id = target.id
            action = action.model_copy(update={"target": target.id})
            if action.action == "call_tool":
                configured_args = _parse_args(target)
                action = action.model_copy(update={"tool": target.config.get("tool"), "args": configured_args or action.args})

        return BrainResult(
            action=action, raw_output=result.raw_output, model=result.model, duration_ms=result.duration_ms,
            attempts=result.attempts, usage=result.usage,
            request_messages=result.request_messages, request_tools=result.request_tools, protocol=result.protocol,
        )

    def decide(self, spec: MiniAgentSpec, context: list[dict[str, Any]]) -> BrainResult:
        current = self.nodes[self.current_node_id]
        if current.type == "agent":
            return self._decide_from_agent(current, context)
        next_node = self._next_after(current)
        self.current_node_id = next_node.id
        return self._auto_action(next_node, context)
