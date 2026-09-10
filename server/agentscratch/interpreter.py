"""Agent interpreter: the observable loop shared by every AgentScratch run."""

from __future__ import annotations

import copy
import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from . import tools as tool_functions
from .brain import Brain, TestBrain
from .events import EventSink
from .llm import BrainResult, LLMParseError
from .schemas import AgentAction, AgentRunState, MiniAgentSpec, RunStatus, ToolCall


class ActionValidationError(ValueError):
    pass


class PolicyBlockedError(ActionValidationError):
    """A syntactically valid call that Harness policy intentionally denies."""

    def __init__(self, call: ToolCall, message: str):
        super().__init__(message)
        self.call = call


class ContextBuilder:
    """Build the canonical conversation history stored with a run.

    Tool declarations belong to a model *request*, not to the conversation.
    Keeping them out of this history makes the ReAct trace match the standard
    ``system → user → assistant(tool_calls) → tool`` protocol and lets a
    harness choose the visible capability set for each individual request.
    """

    @staticmethod
    def initial(user_input: str) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": "You are a teaching agent. Choose one next action from the supplied tools."},
            {"role": "user", "content": user_input},
        ]


def build_context(user_input: str) -> list[dict[str, Any]]:
    """Compatibility entry point for creating a run's canonical history."""
    return ContextBuilder.initial(user_input)


def action_tool_calls(action: AgentAction) -> list[ToolCall]:
    """Normalize the legacy single-call action and the batched call action."""
    if action.action == "call_tool":
        return [ToolCall(tool=action.tool or "", args=action.args, call_id=action.call_id)]
    if action.action == "call_tools":
        return action.tool_calls
    return []


def _parameter_has_expected_type(value: Any, parameter_type: str, *, required: bool) -> bool:
    if value is None:
        return not required
    if parameter_type == "string":
        return isinstance(value, str) and (bool(value.strip()) or not required)
    if parameter_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if parameter_type == "boolean":
        return isinstance(value, bool)
    if parameter_type == "object":
        return isinstance(value, dict)
    if parameter_type == "array":
        return isinstance(value, list)
    return False


def validate_tool_call(call: ToolCall, spec: MiniAgentSpec, memory: dict[str, Any] | None = None) -> None:
    tool = spec.tool(call.tool)
    if tool is None:
        raise ActionValidationError(f"tool is not allowed: {call.tool}")
    if tool.permission == "blocked":
        raise PolicyBlockedError(call, f"tool permission is blocked: {call.tool}")
    unknown = set(call.args) - set(tool.parameters)
    if unknown:
        raise ActionValidationError(f"unknown parameter(s) for {tool.name}: {', '.join(sorted(unknown))}")
    for name, parameter in tool.parameters.items():
        if parameter.required and name not in call.args:
            raise ActionValidationError(f"missing parameter: {tool.name}.{name}")
        if name in call.args and not _parameter_has_expected_type(call.args[name], parameter.type, required=parameter.required):
            raise ActionValidationError(f"invalid parameter: {tool.name}.{name} must be a {'non-empty ' if parameter.required and parameter.type == 'string' else ''}{parameter.type}")
    if memory is None:
        return
    observations = memory.get("observations", {})
    completed_observations = (
        [observation for observation in observations.values() if isinstance(observation, dict) and observation.get("status") == "completed"]
        if isinstance(observations, dict)
        else []
    )
    if call.tool == "read_file":
        discovered = memory.get("discovered_file_ids", [])
        if not isinstance(discovered, list) or call.args.get("file_id") not in discovered:
            raise ActionValidationError("read_file.file_id must be returned by a successful search_files Observation")
        # Search results are discoverable resources, not an unconditional
        # authorization to process any result. A file-choice interaction
        # narrows the capability to the file the user actually resolved.
        authorization = memory.get("file_read_authorization")
        if isinstance(authorization, dict):
            allowed_ids = authorization.get("allowed_file_ids", [])
            allowed = {item for item in allowed_ids if isinstance(item, str) and item}
            requested_id = call.args.get("file_id")
            if requested_id not in allowed:
                selection_status = str(authorization.get("status") or "awaiting_user_selection")
                if selection_status == "resolved":
                    raise ActionValidationError(
                        "read_file.file_id is not authorized by the current user selection; use the exact confirmed file_id"
                    )
                raise ActionValidationError(
                    "read_file.file_id is not authorized because the user has not uniquely selected a file; ask_user or final_answer remains available"
                )
        # This teaching workspace is a fixed read-only snapshot for one run.
        # Re-reading exactly the same file_id cannot produce new information,
        # so reject it before a second full file payload bloats Context.  This
        # validates the tool's idempotent contract; it does not determine what
        # the LLM should do with the first Observation.
        if any(
            observation.get("tool") == "read_file" and observation.get("args") == call.args
            for observation in completed_observations
        ):
            raise ActionValidationError(
                "read_file already completed for this file_id; its Observation is available in the current Context, so do not repeat the identical read"
            )
    if call.tool == "search_files":
        duplicate_search = any(
            observation.get("tool") == "search_files"
            and observation.get("args") == call.args
            for observation in completed_observations
        )
        if duplicate_search:
            # A search followed by a completed read already makes the source
            # text available in Context.  Harness may reject the redundant
            # search, but must not decide what the LLM should do with that
            # Observation: it might summarize, extract values, calculate, call
            # another tool, ask the user, or return a final answer.
            has_completed_read = any(
                observation.get("tool") == "read_file"
                for observation in completed_observations
            )
            if has_completed_read:
                raise ActionValidationError(
                    "read_file already returned content; do not repeat the identical search_files call, and choose the next valid action from the available Observation"
                )
            raise ActionValidationError("search_files was already attempted with identical arguments; use its Observation, change arguments, or ask_user")


def validate_action(action: AgentAction, spec: MiniAgentSpec, memory: dict[str, Any] | None = None) -> None:
    if action.action == "route":
        if not spec.workflow:
            raise ActionValidationError("route is only supported by a workflow")
        if not action.target:
            raise ActionValidationError("route requires a target node id")
        return
    if action.action == "final_answer":
        if not action.answer:
            raise ActionValidationError("final_answer requires a non-empty answer")
        return
    if action.action == "ask_user":
        question = action.args.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ActionValidationError("ask_user requires a non-empty args.question")
        if "response_schema" in action.args:
            if memory is None:
                raise ActionValidationError("ask_user response_schema requires run state")
            _validated_file_selection_schema(action.args["response_schema"], memory)
        # A user response is a fact in the current Context.  Harness may
        # prevent an identical question loop, but must leave the next action
        # (another, genuinely different clarification / tool / answer) to the
        # model instead of choosing it on the model's behalf.
        if memory:
            answered_question = memory.get("last_answered_user_question")
            answered_response = memory.get("last_user_response")
            normalized_question = " ".join(question.casefold().split())
            normalized_answered = " ".join(answered_question.casefold().split()) if isinstance(answered_question, str) else ""
            if normalized_question and normalized_question == normalized_answered and isinstance(answered_response, str) and answered_response.strip():
                raise ActionValidationError(
                    "ask_user repeats a question the user already answered; use the latest user input and current Observation to choose a new action"
                )
        return
    calls = action_tool_calls(action)
    if not calls:
        raise ActionValidationError("tool action requires at least one tool call")
    for call in calls:
        validate_tool_call(call, spec, memory)


def _file_catalog_from_observation(memory: dict[str, Any], observation_id: str) -> list[dict[str, Any]]:
    """Return only file identities actually returned by one search Observation.

    Candidate labels are presentation data, while ``file_id`` is the opaque,
    capability-scoped value that authorizes ``read_file``.  This projection is
    deliberately derived from a completed Observation rather than trusting an
    LLM-produced candidate list.
    """
    observations = memory.get("observations", {})
    if not isinstance(observations, dict):
        return []
    observation = observations.get(observation_id)
    if not isinstance(observation, dict) or observation.get("tool") != "search_files":
        return []
    if observation.get("status") != "completed" or not isinstance(observation.get("result"), dict):
        return []
    matches = observation["result"].get("matches", [])
    if not isinstance(matches, list):
        return []
    catalog: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in matches:
        if not isinstance(item, dict):
            continue
        file_id, name = item.get("file_id"), item.get("name")
        if not isinstance(file_id, str) or not file_id or not isinstance(name, str) or not name or file_id in seen:
            continue
        seen.add(file_id)
        candidate = {"file_id": file_id, "name": name}
        if isinstance(item.get("directory"), str):
            candidate["directory"] = item["directory"]
        catalog.append(candidate)
    return catalog


def _validated_file_selection_schema(raw_schema: Any, memory: dict[str, Any]) -> dict[str, Any]:
    """Validate a model-declared file-selection UI contract against state."""
    if not isinstance(raw_schema, dict) or raw_schema.get("type") != "file_selection":
        raise ActionValidationError("ask_user response_schema.type must be file_selection")
    source = raw_schema.get("source_observation")
    candidate_ids = raw_schema.get("candidate_file_ids")
    allow_custom = raw_schema.get("allow_custom_input", True)
    if not isinstance(source, str) or not source:
        raise ActionValidationError("file_selection response_schema requires source_observation")
    if not isinstance(candidate_ids, list) or not 1 <= len(candidate_ids) <= 3 or not all(isinstance(item, str) and item for item in candidate_ids):
        raise ActionValidationError("file_selection response_schema requires 1-3 candidate_file_ids")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ActionValidationError("file_selection response_schema candidate_file_ids must be unique")
    if not isinstance(allow_custom, bool):
        raise ActionValidationError("file_selection response_schema allow_custom_input must be boolean")
    catalog = _file_catalog_from_observation(memory, source)
    catalog_ids = {item["file_id"] for item in catalog}
    if not catalog:
        raise ActionValidationError("file_selection source_observation must reference a completed search_files Observation")
    if any(file_id not in catalog_ids for file_id in candidate_ids):
        raise ActionValidationError("file_selection candidate_file_ids must be returned by source_observation")
    return {
        "type": "file_selection",
        "source_observation": source,
        "candidate_file_ids": candidate_ids,
        "allow_custom_input": allow_custom,
    }


def _sanitized_ask_user_interaction(action: AgentAction, memory: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
    """Build UI choices only from an explicit, verified response schema.

    A plain natural-language question intentionally remains plain text.  The
    Harness does not infer that it is a file picker just because a search
    happened earlier in the run.
    """
    raw_schema = action.args.get("response_schema")
    if raw_schema is None:
        return [], [], None
    schema = _validated_file_selection_schema(raw_schema, memory)
    catalog = _file_catalog_from_observation(memory, schema["source_observation"])
    by_id = {item["file_id"]: item for item in catalog}
    choices = [by_id[file_id] for file_id in schema["candidate_file_ids"]]
    return catalog, choices, schema


def execute_tool(name: str, args: dict[str, Any], memory: dict[str, Any]) -> Any:
    if name == "calculator":
        return tool_functions.calculator(args["expression"])
    if name == "mock_search":
        return tool_functions.mock_search(args["query"])
    if name == "search_files":
        return tool_functions.search_files(args.get("directory", "."), args.get("file_name"), memory=memory)
    if name == "read_file":
        return tool_functions.read_file(args["file_id"], memory=memory)
    if name == "read_document":
        return tool_functions.read_document(args["file_id"], memory=memory)
    if name == "extract_numbers":
        return tool_functions.extract_numbers(args["text"])
    if name == "memory_store":
        return tool_functions.memory_store(args["key"], args["value"], memory=memory)
    if name == "memory_lookup":
        return tool_functions.memory_lookup(args["key"], memory=memory)
    if name == "final_answer":
        return tool_functions.final_answer(args["answer"])
    raise ValueError(f"unsupported tool: {name}")


class AgentInterpreter:
    def __init__(self, brain: Brain | None = None):
        self.brain = brain or TestBrain()

    @staticmethod
    def _has_started(run: AgentRunState) -> bool:
        return any(event.type == "run.start" for event in run.events)

    def _ensure_started(self, run: AgentRunState) -> None:
        if self._has_started(run):
            run.status = RunStatus.RUNNING if run.status == RunStatus.CREATED else run.status
            return
        sink = EventSink(run_id=run.run_id)
        run.status = RunStatus.RUNNING
        run.events.append(sink.emit("run.start", "agent", {"input": run.input}))

    @staticmethod
    def _fail(run: AgentRunState, message: str, sink: EventSink) -> AgentRunState:
        run.status = RunStatus.FAILED
        run.error = message
        run.events.append(sink.emit("agent.error", "agent", {"error": message}))
        return run

    @staticmethod
    def _record_failure(run: AgentRunState, category: str) -> None:
        """Keep separately enforceable failure budgets and legacy total aligned."""
        if category == "decision":
            run.decision_failures += 1
        elif category == "tool_execution":
            run.tool_failures += 1
        else:
            raise ValueError(f"unknown failure category: {category}")
        run.failures = run.decision_failures + run.tool_failures

    @staticmethod
    def _decision_repair_message(error: Exception, run: AgentRunState) -> dict[str, str]:
        """Give a factual, task-agnostic recovery state to the next model turn.

        This remains Harness feedback rather than a synthetic Observation: the
        failed action was never accepted.  It names what is already available
        without repeating tool payloads or choosing the task's next subgoal on
        the model's behalf.
        """
        request = next(
            (
                str(message.get("content", "")).strip()
                for message in reversed(run.context)
                if message.get("role") == "user" and str(message.get("content", "")).strip()
            ),
            run.input,
        )
        observations = run.memory.get("observations", {})
        facts: list[str] = []
        if isinstance(observations, dict):
            for observation in observations.values():
                if not isinstance(observation, dict):
                    continue
                tool = str(observation.get("tool") or "unknown_tool")
                args = observation.get("args") if isinstance(observation.get("args"), dict) else {}
                arguments = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
                status = str(observation.get("status") or "unknown")
                if status == "completed":
                    facts.append(f"completed {tool}({arguments}); its Observation is already in Context")
                else:
                    facts.append(f"{status} {tool}({arguments}); no successful Observation was produced")
        observation_facts = "\n".join(f"- {fact}" for fact in facts[-8:]) or "- no completed tool Observation yet"
        return {
            "role": "system",
            "content": (
                "HARNESS_RECOVERY_STATE\n"
                f"UNRESOLVED_USER_REQUEST: {request}\n"
                "AVAILABLE_OBSERVATIONS:\n"
                f"{observation_facts}\n"
                f"LAST_REJECTED_ACTION: {error}\n"
                "Choose exactly one valid next action. You may use a new valid tool call, "
                "return final_answer using available evidence, or ask_user for missing input. "
                "Do not repeat a completed call with identical arguments unless the tool contract says it can yield new information."
            ),
        }

    @staticmethod
    def _decision_error_reason(error: Exception) -> str:
        """Classify only what Harness can observe, never hidden model thought."""
        message = str(error).lower()
        if "service unavailable" in message or "connection" in message or "timed out" in message:
            return "connection"
        if "request was rejected" in message:
            return "request_rejected"
        if "routed to an unconnected" in message:
            return "route"
        if "outside its capability" in message or "tool is not allowed" in message:
            return "capability"
        if "without a matching outgoing" in message:
            return "workflow"
        if isinstance(error, ActionValidationError):
            return "action_validation"
        if isinstance(error, LLMParseError) or "model output" in message or "invalid response" in message:
            return "protocol"
        return "other"

    @staticmethod
    def _with_call_ids(action: AgentAction, run: AgentRunState, step: int) -> AgentAction:
        calls = action_tool_calls(action)
        if not calls:
            return action
        # Model-generated IDs are advisory.  They may be absent or reused by a
        # weak local model on later turns; reusing one would overwrite a prior
        # Observation and break assistant/tool message linkage.  Harness owns
        # the correlation ID and guarantees one unique ID per accepted call.
        numbered = [
            call.model_copy(update={"call_id": f"call-{run.run_id}-{step}-{index + 1}"})
            for index, call in enumerate(calls)
        ]
        if action.action == "call_tool":
            call = numbered[0]
            return action.model_copy(update={"tool": call.tool, "args": call.args, "call_id": call.call_id})
        return action.model_copy(update={"tool_calls": numbered})

    @staticmethod
    def _assistant_message(action: AgentAction, raw_output: str, protocol: str) -> dict[str, Any]:
        calls = action_tool_calls(action)
        if calls:
            # The persisted runtime history always uses the standard assistant
            # tool_calls shape.  ``json_action`` is only an adapter used when
            # a local model server cannot produce native calls itself; it must
            # not create a second, incompatible ReAct history format.
            return {"role": "assistant", "content": None, "tool_calls": [
                {"id": call.call_id, "type": "function", "function": {"name": call.tool, "arguments": json.dumps(call.args, ensure_ascii=False)}}
                for call in calls
            ]}
        return {"role": "assistant", "content": raw_output or action.model_dump_json()}

    @staticmethod
    def _guarded_calls(calls: list[ToolCall], spec: MiniAgentSpec) -> list[dict[str, Any]]:
        return [{"tool": call.tool, "args": call.args, "call_id": call.call_id, "risk": tool.risk} for call in calls if (tool := spec.tool(call.tool)) and (tool.requires_confirmation or tool.risk == "sensitive")]

    def _wait_for_confirmation(self, run: AgentRunState, spec: MiniAgentSpec, sink: EventSink, calls: list[ToolCall]) -> AgentRunState:
        guarded = self._guarded_calls(calls, spec)
        run.memory["pending_tool_calls"] = [call.model_dump() for call in calls]
        run.waiting_choices = []
        run.waiting_response_schema = None
        run.waiting_question = "以下工具调用需要你的确认：" + "；".join(f"{item['tool']}（{item['risk']}）" for item in guarded) + "。输入“确认”执行，输入“拒绝”取消。"
        run.status = RunStatus.WAITING_USER
        run.events.append(sink.emit("agent.confirmation_required", "safety", {"calls": guarded}))
        return run

    @staticmethod
    def _record_observation(
        run: AgentRunState,
        call: ToolCall,
        *,
        result: Any = None,
        error: str | None = None,
    ) -> None:
        """Persist a structured Observation alongside its chat message.

        The model receives the corresponding ``tool`` message in its next
        context.  The keyed record is instead for the scheduler, UI replay and
        recovery; it must never replace the model-visible Observation.
        """
        observations = run.memory.setdefault("observations", {})
        if not isinstance(observations, dict):
            raise RuntimeError("run observations state is invalid")
        observations[call.call_id or "unknown"] = {
            "tool": call.tool,
            "args": copy.deepcopy(call.args),
            "status": "error" if error is not None else "completed",
            "result": result if error is None else None,
            "error": error,
        }

    @staticmethod
    def _append_context(run: AgentRunState, sink: EventSink, message: dict[str, Any], *, source: str) -> None:
        """Append one canonical message and make that growth replayable.

        ``context.build`` remains the authoritative full snapshot sent at the
        start of a decision.  This smaller event marks the exact instant a
        ReAct result is assembled back into the growing conversation, so the
        canvas can show ``Observation → assemble → Context`` without guessing
        from a later snapshot.
        """
        run.context.append(message)
        run.events.append(
            sink.emit(
                "context.append",
                "context",
                {"message": copy.deepcopy(message), "source": source, "message_count": len(run.context)},
            )
        )

    def _execute_calls(self, run: AgentRunState, spec: MiniAgentSpec, sink: EventSink, calls: list[ToolCall]) -> AgentRunState:
        tools = [spec.tool(call.tool) for call in calls]
        parallel = len(calls) > 1 and all(tool is not None and tool.parallel_safe for tool in tools)
        batch_data = {"calls": [{"tool": call.tool, "call_id": call.call_id, "args": call.args} for call in calls], "parallel": parallel}
        if len(calls) > 1:
            run.events.append(sink.emit("tool.batch.start", "tools", batch_data))
            if parallel:
                run.events.append(sink.emit("tool.parallel.start", "tools", batch_data))
        for call in calls:
            run.events.append(sink.emit("tool.start", call.tool, {"args": call.args, "call_id": call.call_id, "parallel": parallel}))

        def invoke(call: ToolCall) -> tuple[Any, int, Exception | None]:
            started = time.perf_counter()
            try:
                return execute_tool(call.tool, call.args, run.memory), int((time.perf_counter() - started) * 1000), None
            except Exception as exc:
                return None, int((time.perf_counter() - started) * 1000), exc

        if parallel:
            with ThreadPoolExecutor(max_workers=len(calls), thread_name_prefix="agentscratch-tool") as executor:
                results = list(executor.map(invoke, calls))
        else:
            results = [invoke(call) for call in calls]

        for call, (result, duration_ms, error) in zip(calls, results, strict=True):
            if error is not None:
                self._record_failure(run, "tool_execution")
                run.events.append(sink.emit("tool.error", call.tool, {"error": str(error), "call_id": call.call_id, "duration_ms": duration_ms, "category": "tool_execution", "tool_failures": run.tool_failures}))
                self._record_observation(run, call, error=str(error))
                self._append_context(
                    run,
                    sink,
                    {"role": "tool", "name": call.tool, "tool_call_id": call.call_id, "content": f"ERROR: {error}"},
                    source="tool_error",
                )
            else:
                run.events.append(sink.emit("tool.end", call.tool, {"result": result, "duration_ms": duration_ms, "call_id": call.call_id, "parallel": parallel}))
                self._record_observation(run, call, result=result)
                self._append_context(
                    run,
                    sink,
                    {"role": "tool", "name": call.tool, "tool_call_id": call.call_id, "content": result},
                    source="observation",
                )
        if len(calls) > 1:
            run.events.append(sink.emit("tool.parallel.end" if parallel else "tool.batch.end", "tools", {**batch_data, "failures": run.failures, "tool_failures": run.tool_failures}))
        if run.tool_failures >= spec.policy.max_tool_failures:
            return self._fail(run, "maximum tool failures reached", sink)
        run.status = RunStatus.RUNNING
        return run

    def resolve_confirmation(self, run: AgentRunState, spec: MiniAgentSpec, response: str) -> AgentRunState:
        raw_calls = run.memory.get("pending_tool_calls")
        if not isinstance(raw_calls, list):
            raise ActionValidationError("run has no pending tool confirmation")
        calls = [ToolCall.model_validate(item) for item in raw_calls]
        sink = EventSink(run_id=run.run_id, step=run.step)
        accepted = response.strip().lower() in {"确认", "同意", "批准", "允许", "yes", "y", "approve", "allow"}
        run.events.append(sink.emit("user.confirmation", "user", {"response": response, "approved": accepted}))
        run.waiting_question = None
        run.memory.pop("pending_tool_calls", None)
        if not accepted:
            run.policy_blocks += len(calls)
            for call in calls:
                self._record_observation(run, call, error="user rejected this tool call")
                self._append_context(
                    run,
                    sink,
                    {"role": "tool", "name": call.tool, "tool_call_id": call.call_id, "content": "ERROR: user rejected this tool call"},
                    source="policy_rejection",
                )
            run.events.append(sink.emit("tool.rejected", "safety", {"calls": [call.model_dump() for call in calls], "category": "policy_block", "policy_blocks": run.policy_blocks}))
            run.status = RunStatus.RUNNING
            return run
        run.events.append(sink.emit("tool.confirmed", "safety", {"calls": [call.model_dump() for call in calls]}))
        return self._execute_calls(run, spec, sink, calls)

    def step(self, run: AgentRunState, spec: MiniAgentSpec) -> AgentRunState:
        if run.status in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.PAUSED, RunStatus.WAITING_USER}:
            return run
        self._ensure_started(run)
        if run.step >= spec.policy.max_steps:
            return self._fail(run, "maximum steps reached", EventSink(run_id=run.run_id, step=run.step))
        sink = EventSink(run_id=run.run_id, step=run.step)
        sink.next_step()
        run.step = sink.step
        run.events.append(sink.emit("context.build", "context", {"messages": copy.deepcopy(run.context)}))
        run.events.append(sink.emit("llm.start", "brain", {"provider": spec.brain.type, "model": spec.brain.model}))
        try:
            result = self.brain.decide(spec, run.context)
            if not isinstance(result, BrainResult):
                result = BrainResult(action=result)
            result.action = self._with_call_ids(result.action, run, sink.step)
            run.events.append(sink.emit("context.request", "context", {"messages": copy.deepcopy(result.request_messages or run.context), "tools": copy.deepcopy(result.request_tools), "protocol": result.protocol if result.request_messages else spec.brain.tool_protocol}))
            run.events.append(sink.emit("llm.end", "brain", {"action": result.action.model_dump(), "raw_output": result.raw_output, "model": result.model, "duration_ms": result.duration_ms, "attempts": result.attempts, "usage": result.usage}))
            action = result.action
            validate_action(action, spec, run.memory)
            run.events.append(sink.emit("action.validate", "validator", {"valid": True, "action": action.model_dump()}))
        except PolicyBlockedError as exc:
            run.policy_blocks += 1
            self._append_context(run, sink, self._decision_repair_message(exc, run), source="policy_repair")
            run.events.append(sink.emit("tool.rejected", "safety", {
                "calls": [exc.call.model_dump()],
                "category": "policy_block",
                "source": "tool_permission",
                "policy_blocks": run.policy_blocks,
            }))
            return run
        except Exception as exc:
            self._record_failure(run, "decision")
            self._append_context(run, sink, self._decision_repair_message(exc, run), source="decision_repair")
            run.events.append(sink.emit("llm.error", "brain", {"error": str(exc), "raw_output": getattr(exc, "raw_output", ""), "model": getattr(exc, "model", None), "category": "decision", "reason_kind": self._decision_error_reason(exc), "decision_failures": run.decision_failures}))
            if run.decision_failures >= spec.policy.max_decision_failures:
                return self._fail(
                    run,
                    f"maximum decision failures reached ({run.decision_failures}/{spec.policy.max_decision_failures}): {exc}",
                    sink,
                )
            return run
        self._append_context(
            run,
            sink,
            self._assistant_message(action, result.raw_output, spec.brain.tool_protocol),
            source="assistant_action",
        )
        if action.action == "final_answer":
            run.output, run.status = action.answer, RunStatus.COMPLETED
            run.events.append(sink.emit("agent.finish", "agent", {"output": run.output}))
            return run
        if action.action == "ask_user":
            catalog, choices, response_schema = _sanitized_ask_user_interaction(action, run.memory)
            run.waiting_question, run.waiting_choices, run.waiting_response_schema, run.status = action.args["question"], choices, response_schema, RunStatus.WAITING_USER
            if response_schema:
                # Keep the whole discovered set briefly: a learner may enter a
                # legitimate custom file name which was not among the 1-3
                # choices.  The resume handler resolves only exact or unique
                # matches and never guesses an ambiguous selection.
                run.memory["waiting_file_catalog"] = catalog
                run.memory["file_read_authorization"] = {
                    "status": "awaiting_user_selection",
                    "allowed_file_ids": [],
                }
            else:
                run.memory.pop("waiting_file_catalog", None)
            run.memory["last_asked_user_question"] = run.waiting_question
            run.events.append(sink.emit("agent.waiting_user", "agent", {"question": action.args.get("question"), "choices": choices, "response_schema": response_schema}))
            return run
        if action.action == "route":
            run.events.append(sink.emit("agent.route", "router", {"target": action.target}))
            run.status = RunStatus.RUNNING
            return run
        calls = action_tool_calls(action)
        if self._guarded_calls(calls, spec):
            return self._wait_for_confirmation(run, spec, sink, calls)
        result_run = self._execute_calls(run, spec, sink, calls)
        return self._fail(run, "maximum steps reached", sink) if run.step >= spec.policy.max_steps and result_run.status == RunStatus.RUNNING else result_run

    def run(self, run: AgentRunState, spec: MiniAgentSpec) -> AgentRunState:
        while run.status not in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.PAUSED, RunStatus.WAITING_USER}:
            self.step(run, spec)
        return run
