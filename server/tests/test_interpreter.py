import pytest

from agentscratch.brain import TestBrain
from agentscratch.interpreter import ActionValidationError, AgentInterpreter, build_context, execute_tool, validate_action
from agentscratch.schemas import AgentAction, AgentPolicy, AgentRunState, BrainSpec, MiniAgentSpec, RunStatus, ToolCall, ToolParameterSpec, ToolSpec
from agentscratch.tools import teaching_workspace_index, teaching_workspace_tree


def make_spec() -> MiniAgentSpec:
    return MiniAgentSpec(
        name="Test Agent",
        tools=[
            ToolSpec(
                name="calculator",
                description="calc",
                parameters={"expression": ToolParameterSpec(type="string")},
            )
        ],
    )


def test_rejects_missing_parameter() -> None:
    action = AgentAction(thought="call", action="call_tool", tool="calculator", args={})
    with pytest.raises(ActionValidationError):
        validate_action(action, make_spec())


def test_rejects_null_or_wrongly_typed_tool_parameters() -> None:
    null_action = AgentAction(thought="call", action="call_tool", tool="calculator", args={"expression": None})
    number_action = AgentAction(thought="call", action="call_tool", tool="calculator", args={"expression": 12})
    with pytest.raises(ActionValidationError, match="calculator.expression"):
        validate_action(null_action, make_spec())
    with pytest.raises(ActionValidationError, match="calculator.expression"):
        validate_action(number_action, make_spec())


def test_rejects_unknown_tool() -> None:
    action = AgentAction(thought="call", action="call_tool", tool="network", args={})
    with pytest.raises(ActionValidationError):
        validate_action(action, make_spec())


def test_rejects_empty_ask_user_question() -> None:
    action = AgentAction(thought="ask", action="ask_user", args={})
    with pytest.raises(ActionValidationError):
        validate_action(action, make_spec())


def test_rejects_identical_ask_user_after_the_user_already_replied() -> None:
    action = AgentAction(thought="ask again", action="ask_user", args={"question": "请告诉我文件名"})
    with pytest.raises(ActionValidationError, match="already answered"):
        validate_action(
            action,
            make_spec(),
            {
                "last_answered_user_question": "请告诉我文件名",
                "last_user_response": "华东销售复盘.md",
            },
        )


def test_duplicate_search_after_successful_read_leaves_next_action_to_model() -> None:
    spec = MiniAgentSpec(
        name="File Agent",
        tools=[
            ToolSpec(name="search_files", description="search", parameters={"file_name": ToolParameterSpec(type="string")}),
            ToolSpec(name="read_file", description="read", parameters={"file_id": ToolParameterSpec(type="string")}),
        ],
    )
    action = AgentAction(thought="search again", action="call_tool", tool="search_files", args={"file_name": "notes.txt"})
    memory = {
        "observations": {
            "call-search": {"tool": "search_files", "args": {"file_name": "notes.txt"}, "status": "completed"},
            "call-read": {"tool": "read_file", "args": {"file_id": "workspace-file"}, "status": "completed"},
        }
    }
    with pytest.raises(ActionValidationError, match="choose the next valid action"):
        validate_action(action, spec, memory)


def test_duplicate_static_file_read_is_rejected_without_repeating_content() -> None:
    spec = MiniAgentSpec(
        name="File Agent",
        tools=[ToolSpec(name="read_file", description="read", parameters={"file_id": ToolParameterSpec(type="string")})],
    )
    action = AgentAction(thought="read again", action="call_tool", tool="read_file", args={"file_id": "workspace-file"})
    memory = {
        "discovered_file_ids": ["workspace-file"],
        "observations": {
            "call-read": {"tool": "read_file", "args": {"file_id": "workspace-file"}, "status": "completed"},
        },
    }
    with pytest.raises(ActionValidationError, match="do not repeat the identical read"):
        validate_action(action, spec, memory)


def test_file_selection_authorization_narrows_discovered_read_capability() -> None:
    spec = MiniAgentSpec(
        name="File Agent",
        tools=[ToolSpec(name="read_file", description="read", parameters={"file_id": ToolParameterSpec(type="string")})],
    )
    read_a = AgentAction(thought="read chosen file", action="call_tool", tool="read_file", args={"file_id": "file-a"})
    read_b = AgentAction(thought="read another file", action="call_tool", tool="read_file", args={"file_id": "file-b"})
    base_memory = {"discovered_file_ids": ["file-a", "file-b"], "observations": {}}

    with pytest.raises(ActionValidationError, match="has not uniquely selected"):
        validate_action(read_a, spec, {**base_memory, "file_read_authorization": {"status": "unmatched", "allowed_file_ids": []}})

    resolved_memory = {**base_memory, "file_read_authorization": {"status": "resolved", "allowed_file_ids": ["file-a"]}}
    validate_action(read_a, spec, resolved_memory)
    with pytest.raises(ActionValidationError, match="current user selection"):
        validate_action(read_b, spec, resolved_memory)


def test_recovery_state_reports_facts_without_deciding_the_next_subgoal() -> None:
    run = AgentRunState(
        run_id="run-recovery",
        agent_id="agent-recovery",
        input="读取文件后完成用户任务",
        status=RunStatus.RUNNING,
        context=build_context("读取文件后完成用户任务"),
        memory={
            "observations": {
                "call-read": {
                    "tool": "read_file",
                    "args": {"file_id": "workspace-file"},
                    "status": "completed",
                    "result": {"content": "不应复制整份文件内容"},
                }
            }
        },
    )
    message = AgentInterpreter._decision_repair_message(
        ActionValidationError("read_file already completed for this file_id"), run
    )
    content = message["content"]
    assert "HARNESS_RECOVERY_STATE" in content
    assert "UNRESOLVED_USER_REQUEST: 读取文件后完成用户任务" in content
    assert 'completed read_file({"file_id":"workspace-file"})' in content
    assert "Observation is already in Context" in content
    assert "不应复制整份文件内容" not in content
    assert "必须总结" not in content


def test_harness_replaces_reused_model_call_ids_with_unique_ids() -> None:
    run = AgentRunState(run_id="run-correlation", agent_id="agent", input="test", status=RunStatus.CREATED, context=[])
    action = AgentAction(thought="call", action="call_tool", tool="calculator", call_id="model-reused", args={"expression": "1 + 1"})
    first = AgentInterpreter._with_call_ids(action, run, step=1)
    second = AgentInterpreter._with_call_ids(action, run, step=2)
    assert first.call_id == "call-run-correlation-1-1"
    assert second.call_id == "call-run-correlation-2-1"
    assert first.call_id != second.call_id


def test_interpreter_runs_tool_then_finishes() -> None:
    spec = make_spec()
    run = AgentRunState(
        run_id="run-test",
        agent_id="agent-test",
        input="7",
        status=RunStatus.CREATED,
        context=build_context("7"),
    )
    result = AgentInterpreter(TestBrain()).run(run, spec)
    assert result.status == RunStatus.COMPLETED
    assert result.output == "7"
    assert any(event.type == "tool.end" for event in result.events)


def test_native_tool_protocol_keeps_tool_call_linkage_in_context() -> None:
    spec = make_spec().model_copy(update={"brain": BrainSpec(tool_protocol="native_tools")})
    run = AgentRunState(
        run_id="run-native",
        agent_id="agent-native",
        input="计算 7 * 6",
        status=RunStatus.CREATED,
        context=build_context("计算 7 * 6"),
    )
    result = AgentInterpreter(TestBrain()).run(run, spec)
    assistant = next(message for message in result.context if message.get("role") == "assistant")
    tool = next(message for message in result.context if message.get("role") == "tool")
    assert assistant["tool_calls"][0]["id"] == tool["tool_call_id"]
    assert tool["content"] == 42
    observation = result.memory["observations"][tool["tool_call_id"]]
    assert observation == {
        "tool": "calculator",
        "args": {"expression": "7 * 6"},
        "status": "completed",
        "result": 42,
        "error": None,
    }
    request_events = [event for event in result.events if event.type == "context.request"]
    assert request_events[0].data["protocol"] == "native_tools"


def test_json_action_protocol_keeps_tool_call_linkage_in_context() -> None:
    spec = make_spec()
    run = AgentRunState(
        run_id="run-json-action",
        agent_id="agent-json-action",
        input="计算 7 * 6",
        status=RunStatus.CREATED,
        context=build_context("计算 7 * 6"),
    )
    result = AgentInterpreter(TestBrain()).run(run, spec)
    assistant = next(message for message in result.context if message.get("role") == "assistant")
    tool = next(message for message in result.context if message.get("role") == "tool")
    assert assistant["content"] is None
    assert assistant["tool_calls"][0]["id"] == tool["tool_call_id"]
    assert tool["content"] == 42


class MultiToolBrain:
    def decide(self, spec, context):
        if any(message.get("role") == "tool" for message in context):
            return AgentAction(thought="observed both", action="final_answer", answer="done", done=True)
        return AgentAction(
            thought="independent reads can run together",
            action="call_tools",
            tool_calls=[
                ToolCall(tool="calculator", args={"expression": "2 + 2"}),
                ToolCall(tool="mock_search", args={"query": "agent"}),
            ],
        )


def test_interpreter_executes_independent_tool_batch_in_parallel() -> None:
    spec = MiniAgentSpec(
        name="Batch Agent",
        tools=[
            ToolSpec(name="calculator", description="calc", parameters={"expression": ToolParameterSpec(type="string")}),
            ToolSpec(name="mock_search", description="search", parameters={"query": ToolParameterSpec(type="string")}),
        ],
    )
    run = AgentRunState(run_id="run-batch", agent_id="agent-batch", input="two reads", status=RunStatus.CREATED, context=build_context("two reads"))
    result = AgentInterpreter(MultiToolBrain()).run(run, spec)
    assert result.status == RunStatus.COMPLETED
    assert len([message for message in result.context if message.get("role") == "tool"]) == 2
    assert len(result.memory["observations"]) == 2
    assert any(event.type == "tool.parallel.start" for event in result.events)
    assert all(event.data.get("parallel") is True for event in result.events if event.type == "tool.start")


def test_test_brain_exposes_parallel_teaching_trigger() -> None:
    spec = MiniAgentSpec(
        name="Parallel Demo",
        tools=[
            ToolSpec(name="calculator", description="calc", parameters={"expression": ToolParameterSpec(type="string")}),
            ToolSpec(name="mock_search", description="search", parameters={"query": ToolParameterSpec(type="string")}),
        ],
    )
    run = AgentRunState(run_id="run-parallel-demo", agent_id="agent-parallel-demo", input="并行演示", status=RunStatus.CREATED, context=build_context("并行演示"))
    result = AgentInterpreter(TestBrain()).run(run, spec)
    assert any(event.type == "tool.parallel.start" for event in result.events)


def test_document_tools_are_run_scoped_and_extract_structured_values() -> None:
    memory = {"resources": {"file-a": {"name": "A", "content": "value=12"}}}
    document = execute_tool("read_document", {"file_id": "file-a"}, memory)
    assert document["content"] == "value=12"
    assert execute_tool("extract_numbers", {"text": document["content"]}, memory) == {"values": [12], "count": 1}
    with pytest.raises(PermissionError):
        execute_tool("read_document", {"file_id": "C:/not-allowed.txt"}, memory)


def test_workspace_file_tools_require_search_before_reading(tmp_path) -> None:
    reports = tmp_path / "季度报告"
    reports.mkdir()
    report = reports / "华东销售复盘.md"
    report.write_text("销售额同比增长 18%。", encoding="utf-8")
    root_report = tmp_path / "根目录文件.txt"
    root_report.write_text("根目录可直接搜索", encoding="utf-8")
    memory = {"workspace": {"root_path": str(tmp_path)}}
    with pytest.raises(PermissionError):
        execute_tool("read_file", {"file_id": "workspace-unknown"}, memory)
    found = execute_tool("search_files", {"directory": "季度报告", "file_name": "华东销售复盘.md"}, memory)
    assert found["count"] == 1
    assert found["matches"][0]["name"] == "华东销售复盘.md"
    document = execute_tool("read_file", {"file_id": found["matches"][0]["file_id"]}, memory)
    assert document["content"] == "销售额同比增长 18%。"
    root_found = execute_tool("search_files", {"file_name": "根目录文件.txt"}, memory)
    assert root_found["count"] == 1
    assert root_found["directory"] == "."
    with pytest.raises(PermissionError):
        execute_tool("search_files", {"directory": "..", "file_name": "anything"}, memory)


def test_teaching_workspace_tree_returns_metadata_not_content(tmp_path) -> None:
    reports = tmp_path / "季度报告"
    reports.mkdir()
    (reports / "华东销售复盘.md").write_text("这段内容不应出现在目录树中", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"not-a-real-image")

    tree = teaching_workspace_tree(str(tmp_path))
    report_directory = next(entry for entry in tree["entries"] if entry["name"] == "季度报告")
    report = report_directory["children"][0]
    image = next(entry for entry in tree["entries"] if entry["name"] == "image.png")

    assert report == {
        "name": "华东销售复盘.md",
        "kind": "file",
        "relative_path": "季度报告/华东销售复盘.md",
        "size_bytes": len("这段内容不应出现在目录树中".encode("utf-8")),
        "readable": True,
    }
    assert image["readable"] is False
    index = teaching_workspace_index(str(tmp_path))
    assert {"path": "季度报告/华东销售复盘.md", "kind": "file", "readable": True} in index["entries"]
    assert all("这段内容" not in str(item) for item in index["entries"])


def test_two_document_teaching_trajectory_uses_parallel_reads_then_dependencies() -> None:
    spec = MiniAgentSpec(
        name="Two document demo",
        tools=[
            ToolSpec(name="read_document", description="read", parameters={"file_id": ToolParameterSpec(type="string")}),
            ToolSpec(name="extract_numbers", description="extract", parameters={"text": ToolParameterSpec(type="string")}),
            ToolSpec(name="calculator", description="calc", parameters={"expression": ToolParameterSpec(type="string")}),
        ],
    )
    run = AgentRunState(
        run_id="run-documents",
        agent_id="agent-documents",
        input="读取教学文件 A 与 B，将其中数值相加",
        status=RunStatus.CREATED,
        context=build_context("读取教学文件 A 与 B，将其中数值相加"),
        memory={"resources": {"file-a": {"name": "A", "content": "value=12"}, "file-b": {"name": "B", "content": "value=30"}}},
    )
    result = AgentInterpreter(TestBrain()).run(run, spec)
    assert result.status == RunStatus.COMPLETED
    assert result.output == "两个文件中的数值相加结果是 42。"
    assert [message["name"] for message in result.context if message.get("role") == "tool"] == [
        "read_document", "read_document", "extract_numbers", "extract_numbers", "calculator",
    ]
    parallel_batches = [event for event in result.events if event.type == "tool.parallel.start"]
    assert len(parallel_batches) == 2
    assert set(result.memory["observations"]) == {
        "call-run-documents-1-1",
        "call-run-documents-1-2",
        "call-run-documents-2-1",
        "call-run-documents-2-2",
        "call-run-documents-3-1",
    }


class ConfirmingBrain:
    def decide(self, spec, context):
        if any(message.get("role") == "tool" for message in context):
            return AgentAction(thought="write was observed", action="final_answer", answer="saved", done=True)
        return AgentAction(thought="save user data", action="call_tool", tool="memory_store", args={"key": "topic", "value": "agents"})


def test_high_risk_tool_waits_for_confirmation_then_resumes() -> None:
    spec = MiniAgentSpec(
        name="Safe Agent",
        tools=[ToolSpec(name="memory_store", description="write", parameters={"key": ToolParameterSpec(type="string"), "value": ToolParameterSpec(type="string")}, risk="write", requires_confirmation=True, parallel_safe=False)],
    )
    run = AgentRunState(run_id="run-confirm", agent_id="agent-confirm", input="save" ,status=RunStatus.CREATED, context=build_context("save"))
    interpreter = AgentInterpreter(ConfirmingBrain())
    waiting = interpreter.step(run, spec)
    assert waiting.status == RunStatus.WAITING_USER
    assert "pending_tool_calls" in waiting.memory
    approved = interpreter.resolve_confirmation(waiting, spec, "确认")
    completed = interpreter.run(approved, spec)
    assert completed.status == RunStatus.COMPLETED
    assert completed.memory["topic"] == "agents"
    assert any(event.type == "agent.confirmation_required" for event in completed.events)
    assert any(event.type == "tool.confirmed" for event in completed.events)


class InvalidToolBrain:
    def decide(self, spec, context):
        return AgentAction(thought="try an unavailable capability", action="call_tool", tool="not_a_tool", args={})


class BrokenCalculatorBrain:
    def decide(self, spec, context):
        return AgentAction(thought="execute a valid but broken expression", action="call_tool", tool="calculator", args={"expression": "2 + ("})


class InvalidReadFileBrain:
    def decide(self, spec, context):
        return AgentAction(thought="invent a file id", action="call_tool", tool="read_file", args={"file_id": "file_id_1"})


def test_decision_failures_have_an_independent_budget_and_repair_context() -> None:
    spec = make_spec().model_copy(update={"policy": AgentPolicy(max_steps=5, max_decision_failures=2, max_tool_failures=0)})
    run = AgentRunState(run_id="run-decision-budget", agent_id="agent", input="test", status=RunStatus.CREATED, context=build_context("test"))
    interpreter = AgentInterpreter(InvalidToolBrain())

    first = interpreter.step(run, spec)
    assert first.status == RunStatus.RUNNING
    assert first.decision_failures == 1
    assert first.tool_failures == 0
    assert first.failures == 1
    assert any(message.get("content", "").startswith("HARNESS_RECOVERY_STATE") for message in first.context)

    failed = interpreter.step(first, spec)
    assert failed.status == RunStatus.FAILED
    assert failed.error == "maximum decision failures reached (2/2): tool is not allowed: not_a_tool"
    assert failed.decision_failures == 2
    assert failed.tool_failures == 0
    error_event = next(event for event in failed.events if event.type == "llm.error")
    assert error_event.data["category"] == "decision"
    assert error_event.data["reason_kind"] == "capability"


def test_invented_file_id_is_a_decision_error_before_tool_execution() -> None:
    spec = MiniAgentSpec(
        name="File reader",
        tools=[ToolSpec(name="read_file", description="read", parameters={"file_id": ToolParameterSpec(type="string")})],
        policy=AgentPolicy(max_steps=3, max_decision_failures=1, max_tool_failures=3),
    )
    run = AgentRunState(run_id="run-file-id", agent_id="agent", input="read", status=RunStatus.CREATED, context=build_context("read"))
    failed = AgentInterpreter(InvalidReadFileBrain()).step(run, spec)
    assert failed.status == RunStatus.FAILED
    assert failed.decision_failures == 1
    assert failed.tool_failures == 0
    assert not any(event.type == "tool.start" for event in failed.events)


def test_tool_execution_failures_do_not_consume_decision_budget() -> None:
    spec = make_spec().model_copy(update={"policy": AgentPolicy(max_steps=5, max_decision_failures=0, max_tool_failures=0)})
    run = AgentRunState(run_id="run-tool-budget", agent_id="agent", input="test", status=RunStatus.CREATED, context=build_context("test"))

    failed = AgentInterpreter(BrokenCalculatorBrain()).step(run, spec)
    assert failed.status == RunStatus.FAILED
    assert failed.error == "maximum tool failures reached"
    assert failed.decision_failures == 0
    assert failed.tool_failures == 1
    assert failed.failures == 1
    assert next(event for event in failed.events if event.type == "tool.error").data["category"] == "tool_execution"


def test_user_rejecting_confirmed_tool_is_a_policy_block_not_a_failure() -> None:
    spec = MiniAgentSpec(
        name="Safe Agent",
        tools=[ToolSpec(name="memory_store", description="write", parameters={"key": ToolParameterSpec(type="string"), "value": ToolParameterSpec(type="string")}, risk="write", requires_confirmation=True, parallel_safe=False)],
    )
    run = AgentRunState(run_id="run-reject", agent_id="agent", input="save", status=RunStatus.CREATED, context=build_context("save"))
    interpreter = AgentInterpreter(ConfirmingBrain())
    waiting = interpreter.step(run, spec)
    rejected = interpreter.resolve_confirmation(waiting, spec, "拒绝")
    assert rejected.status == RunStatus.RUNNING
    assert rejected.policy_blocks == 1
    assert rejected.decision_failures == 0
    assert rejected.tool_failures == 0
    assert rejected.failures == 0


def test_blocked_tool_permission_is_a_policy_block_not_a_decision_failure() -> None:
    spec = MiniAgentSpec(
        name="Blocked tool Agent",
        tools=[ToolSpec(name="calculator", description="calc", parameters={"expression": ToolParameterSpec(type="string")}, permission="blocked")],
    )
    run = AgentRunState(run_id="run-permission-block", agent_id="agent", input="test", status=RunStatus.CREATED, context=build_context("test"))
    blocked = AgentInterpreter(BrokenCalculatorBrain()).step(run, spec)
    assert blocked.status == RunStatus.RUNNING
    assert blocked.policy_blocks == 1
    assert blocked.decision_failures == 0
    assert blocked.tool_failures == 0
    rejected = next(event for event in blocked.events if event.type == "tool.rejected")
    assert rejected.data["source"] == "tool_permission"
