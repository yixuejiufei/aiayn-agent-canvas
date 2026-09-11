from pathlib import Path
import json
from uuid import uuid4

from fastapi.testclient import TestClient

from aiayn_agent_canvas.main import app
from agentscratch.schemas import AgentRunState, MiniAgentSpec, RunStatus
from agentscratch.storage import SQLiteStore


def _create_test_run(client: TestClient) -> str:
    agent = client.post(
        "/api/v1/agents",
        json={
            "name": "Step Agent",
            "tools": ["calculator", "final_answer"],
            "brain_type": "test",
        },
    ).json()
    run = client.post(
        f"/api/v1/agents/{agent['agent_id']}/runs",
        json={"input": "12 * 7"},
    )
    assert run.status_code == 201
    return run.json()["run_id"]


def test_step_and_replay_contract() -> None:
    client = TestClient(app)
    run_id = _create_test_run(client)

    first = client.post(f"/api/v1/runs/{run_id}/steps")
    assert first.status_code == 200
    assert first.json()["status"] == RunStatus.RUNNING.value
    assert first.json()["step"] == 1

    second = client.post(f"/api/v1/runs/{run_id}/steps")
    assert second.status_code == 200
    assert second.json()["status"] == RunStatus.COMPLETED.value
    assert second.json()["output"] == "84"

    events = client.get(f"/api/v1/runs/{run_id}/events").json()
    assert events
    assert events[-1]["type"] == "agent.finish"
    context_events = [event for event in events if event["type"] == "context.build"]
    # Tool declarations are request metadata, not conversation history.  The
    # first model request has system + user; after one tool round it gains the
    # assistant tool_call and its linked tool observation.
    assert [len(event["data"]["messages"]) for event in context_events] == [2, 4]
    assert all(message["role"] != "tools" for event in context_events for message in event["data"]["messages"])
    assert client.get(f"/api/v1/runs/{run_id}/events?after=2").json() == events[2:]

    replay = client.get(f"/api/v1/runs/{run_id}/replay").json()
    assert replay["event_count"] == len(events)
    assert replay["step_count"] == 2
    assert replay["run"]["status"] == RunStatus.COMPLETED.value


def test_sse_stream_returns_persisted_events() -> None:
    client = TestClient(app)
    run_id = _create_test_run(client)
    client.post(f"/api/v1/runs/{run_id}/execute")

    response = client.get(f"/api/v1/runs/{run_id}/events/stream")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: run.start" in response.text
    assert "event: agent.finish" in response.text
    assert "event: stream.end" in response.text


def test_pause_and_resume_run() -> None:
    client = TestClient(app)
    run_id = _create_test_run(client)

    paused = client.post(f"/api/v1/runs/{run_id}/pause")
    assert paused.status_code == 200
    assert paused.json()["status"] == RunStatus.PAUSED.value
    assert paused.json()["events"][-1]["type"] == "run.paused"

    blocked_step = client.post(f"/api/v1/runs/{run_id}/steps")
    assert blocked_step.status_code == 409

    resumed = client.post(f"/api/v1/runs/{run_id}/resume")
    assert resumed.status_code == 200
    assert resumed.json()["status"] == RunStatus.COMPLETED.value
    assert resumed.json()["output"] == "84"
    assert any(event["type"] == "run.resumed" for event in resumed.json()["events"])


def test_ask_user_resume_requires_and_accepts_input() -> None:
    client = TestClient(app)
    agent = client.post(
        "/api/v1/agents",
        json={
            "name": "Clarification Agent",
            "brain_type": "clarification",
            "tools": ["final_answer"],
        },
    ).json()
    run = client.post(
        f"/api/v1/agents/{agent['agent_id']}/runs",
        json={"input": "帮我设计一个学习计划"},
    ).json()

    waiting = client.post(f"/api/v1/runs/{run['run_id']}/steps").json()
    assert waiting["status"] == RunStatus.WAITING_USER.value
    assert waiting["waiting_question"]

    missing = client.post(f"/api/v1/runs/{run['run_id']}/resume", json={})
    assert missing.status_code == 400

    completed = client.post(
        f"/api/v1/runs/{run['run_id']}/resume",
        json={"input": "学习 Python"},
    )
    assert completed.status_code == 200
    assert completed.json()["status"] == RunStatus.COMPLETED.value
    assert "学习 Python" in completed.json()["output"]
    event_types = [event["type"] for event in completed.json()["events"]]
    assert "user.input" in event_types
    assert "run.resumed" in event_types


def test_teaching_case_templates_and_case_run() -> None:
    client = TestClient(app)
    cases = client.get("/api/v1/cases")
    assert cases.status_code == 200
    assert len(cases.json()) >= 5
    ask_user_case = next(item for item in cases.json() if item["case_id"] == "ask-user")

    created = client.post(f"/api/v1/cases/{ask_user_case['case_id']}/runs")
    assert created.status_code == 201
    assert created.json()["run"]["input"] == ask_user_case["input"]
    assert created.json()["agent"]["brain"]["type"] == "clarification"


def test_custom_teaching_case_is_persisted_and_can_override_input() -> None:
    client = TestClient(app)
    case_id = f"custom-{uuid4().hex[:8]}"
    payload = {
        "case_id": case_id,
        "title": "自定义计算案例",
        "description": "练习一次本地计算。",
        "objective": "认识工具调用。",
        "input": "计算 2 + 2",
        "brain_type": "test",
        "brain_model": None,
        "tools": ["calculator", "final_answer"],
    }
    created = client.post("/api/v1/cases", json=payload)
    assert created.status_code == 201
    assert client.get(f"/api/v1/cases/{case_id}").json() == payload
    duplicate = client.post("/api/v1/cases", json=payload)
    assert duplicate.status_code == 409

    run = client.post(
        f"/api/v1/cases/{case_id}/runs",
        json={"input": "计算 3 + 3"},
    )
    assert run.status_code == 201
    assert run.json()["run"]["input"] == "计算 3 + 3"


def test_visual_workflow_compiles_and_resumes_from_persisted_graph_cursor() -> None:
    client = TestClient(app)
    workflow = {
        "version": "0.1",
        "name": "可执行计算工作流",
        "nodes": [
            {"id": "input-1", "type": "input", "label": "输入", "x": 0, "y": 0, "config": {"text": "计算 12 * 7"}},
            {"id": "agent-1", "type": "agent", "label": "Brain", "x": 200, "y": 0, "config": {"brain_type": "test", "model": "qwen-tiny"}},
            {"id": "tool-1", "type": "tool", "label": "Calculator", "x": 400, "y": 0, "config": {"tool": "calculator", "args": "{\"expression\":\"12 * 7\"}"}},
            {"id": "final-1", "type": "final", "label": "输出", "x": 600, "y": 0, "config": {"answer": "{{last_tool_result}}"}},
        ],
        "edges": [
            {"id": "e1", "from": "input-1", "to": "agent-1"},
            {"id": "e2", "from": "agent-1", "to": "tool-1"},
            {"id": "e3", "from": "tool-1", "to": "final-1"},
        ],
    }
    created = client.post("/api/v1/workflows/runs", json={"workflow": workflow})
    assert created.status_code == 201, created.text
    run_id = created.json()["run"]["run_id"]

    first = client.post(f"/api/v1/runs/{run_id}/steps")
    assert first.status_code == 200
    assert first.json()["status"] == RunStatus.RUNNING.value
    assert any(event["type"] == "tool.end" for event in first.json()["events"])

    second = client.post(f"/api/v1/runs/{run_id}/steps")
    assert second.status_code == 200
    assert second.json()["status"] == RunStatus.COMPLETED.value
    assert second.json()["output"] == "84"
    assert second.json()["events"][-1]["type"] == "agent.finish"


def test_visual_workflow_supports_ask_user_resume() -> None:
    client = TestClient(app)
    workflow = {
        "version": "0.1",
        "name": "可执行追问工作流",
        "nodes": [
            {"id": "input-1", "type": "input", "label": "输入", "x": 0, "y": 0, "config": {"text": "帮我设计计划"}},
            {"id": "agent-1", "type": "agent", "label": "Brain", "x": 200, "y": 0, "config": {"brain_type": "clarification"}},
            {"id": "ask-1", "type": "ask_user", "label": "追问", "x": 400, "y": 0, "config": {"question": "请补充主题"}},
            {"id": "final-1", "type": "final", "label": "输出", "x": 600, "y": 0, "config": {"answer": "已完成：{{last_tool_result}}"}},
        ],
        "edges": [
            {"id": "e1", "from": "input-1", "to": "agent-1"},
            {"id": "e2", "from": "agent-1", "to": "ask-1"},
            {"id": "e3", "from": "ask-1", "to": "final-1"},
        ],
    }
    created = client.post("/api/v1/workflows/runs", json={"workflow": workflow})
    assert created.status_code == 201, created.text
    run_id = created.json()["run"]["run_id"]
    waiting = client.post(f"/api/v1/runs/{run_id}/steps").json()
    assert waiting["status"] == RunStatus.WAITING_USER.value

    resumed = client.post(f"/api/v1/runs/{run_id}/resume", json={"input": "Python"})
    assert resumed.status_code == 200
    assert resumed.json()["status"] == RunStatus.COMPLETED.value
    assert resumed.json()["output"] == "已完成："


def test_visual_workflow_condition_branch_and_loop_edge_are_executable() -> None:
    client = TestClient(app)
    workflow = {
        "version": "0.1",
        "name": "带条件和回环的计算工作流",
        "max_steps": 6,
        "nodes": [
            {"id": "input", "type": "input", "label": "输入", "x": 0, "y": 0, "config": {"text": "计算 12 * 7"}},
            {"id": "agent", "type": "agent", "label": "计算 Agent", "x": 180, "y": 0, "config": {"brain_type": "test"}},
            {"id": "tool", "type": "tool", "label": "计算器", "x": 360, "y": 0, "config": {"tool": "calculator", "args": "{\"expression\":\"12 * 7\"}"}},
            {"id": "condition", "type": "condition", "label": "结果大于 50", "x": 540, "y": 0, "config": {"source": "last_tool_result", "operator": "greater_than", "value": "50"}},
            {"id": "success", "type": "final", "label": "成功", "x": 720, "y": 0, "config": {"answer": "通过：{{last_tool_result}}"}},
        ],
        "edges": [
            {"id": "e1", "from": "input", "to": "agent"},
            {"id": "e2", "from": "agent", "to": "tool"},
            {"id": "e3", "from": "tool", "to": "condition"},
            {"id": "e4", "from": "condition", "to": "success", "label": "true"},
            {"id": "e5", "from": "condition", "to": "agent", "label": "false"},
        ],
    }
    created = client.post("/api/v1/workflows/runs", json={"workflow": workflow})
    assert created.status_code == 201, created.text
    completed = client.post(f"/api/v1/runs/{created.json()['run']['run_id']}/execute")
    assert completed.status_code == 200
    assert completed.json()["status"] == RunStatus.COMPLETED.value
    assert completed.json()["output"] == "通过：84"
    action_events = [event for event in completed.json()["events"] if event["type"] == "llm.end"]
    assert any("evaluated true" in event["data"]["action"]["thought"] for event in action_events)


def test_visual_workflow_routes_between_agents() -> None:
    client = TestClient(app)
    workflow = {
        "version": "0.1",
        "name": "多 Agent 路由工作流",
        "nodes": [
            {"id": "input", "type": "input", "label": "输入", "x": 0, "y": 0, "config": {"text": "计算 12 * 7"}},
            {"id": "router", "type": "agent", "label": "分发 Agent", "x": 180, "y": 0, "config": {"brain_type": "test", "default_route": "worker"}},
            {"id": "worker", "type": "agent", "label": "计算 Agent", "x": 360, "y": 0, "config": {"brain_type": "test"}},
            {"id": "tool", "type": "tool", "label": "计算器", "x": 540, "y": 0, "config": {"tool": "calculator", "args": "{\"expression\":\"12 * 7\"}"}},
            {"id": "final", "type": "final", "label": "输出", "x": 720, "y": 0, "config": {"answer": "{{last_tool_result}}"}},
        ],
        "edges": [
            {"id": "e1", "from": "input", "to": "router"},
            {"id": "e2", "from": "router", "to": "worker"},
            {"id": "e3", "from": "worker", "to": "tool"},
            {"id": "e4", "from": "tool", "to": "final"},
        ],
    }
    created = client.post("/api/v1/workflows/runs", json={"workflow": workflow})
    assert created.status_code == 201, created.text
    completed = client.post(f"/api/v1/runs/{created.json()['run']['run_id']}/execute")
    assert completed.status_code == 200
    assert completed.json()["status"] == RunStatus.COMPLETED.value
    assert completed.json()["output"] == "84"
    routes = [event for event in completed.json()["events"] if event["type"] == "agent.route"]
    assert routes and routes[0]["data"]["target"] == "worker"


def test_module_workflow_injects_tool_capability_and_agent_chooses_it() -> None:
    client = TestClient(app)
    workflow = {
        "version": "0.1",
        "name": "模块化 ReAct 工具调用",
        "max_steps": 10,
        "nodes": [
            {"id": "context", "type": "input", "label": "上下文", "x": 0, "y": 0, "config": {"text": "计算 12 * 7"}},
            {"id": "prompt", "type": "system_prompt", "label": "系统提示词", "x": 0, "y": 0, "parentId": "context", "config": {"content": "先思考，再按需调用工具。"}},
            {"id": "definitions", "type": "tool_definition", "label": "工具定义", "x": 0, "y": 0, "parentId": "context", "config": {"tool_ids": "[\"calculator-module\"]"}},
            {"id": "calculator-module", "type": "tool", "label": "计算器工具", "x": 260, "y": 0, "config": {"tool": "calculator"}},
            {"id": "loop", "type": "react_loop", "label": "ReAct 循环", "x": 480, "y": 0, "config": {"max_steps": "10"}},
            {"id": "agent", "type": "agent", "label": "求解 Agent", "x": 0, "y": 0, "parentId": "loop", "config": {"brain_type": "test", "tool_ids": "[\"calculator-module\"]"}},
        ],
        "edges": [
            {"id": "entry", "from": "context", "to": "loop"},
            {"id": "into-agent", "from": "loop", "to": "agent"},
        ],
    }
    created = client.post("/api/v1/workflows/runs", json={"workflow": workflow})
    assert created.status_code == 201, created.text
    assert created.json()["run"]["context"] == [
        {"role": "system", "content": "先思考，再按需调用工具。"},
        {"role": "user", "content": "计算 12 * 7"},
    ]
    completed = client.post(f"/api/v1/runs/{created.json()['run']['run_id']}/execute")
    assert completed.status_code == 200
    assert completed.json()["status"] == RunStatus.COMPLETED.value
    assert completed.json()["output"] == "84"
    actions = [event["data"]["action"] for event in completed.json()["events"] if event["type"] == "llm.end"]
    assert actions[0]["action"] == "call_tool"
    assert actions[0]["tool"] == "calculator"
    assert actions[0]["target"] == "agent"
    requests = [event["data"] for event in completed.json()["events"] if event["type"] == "context.request"]
    assert any(message["content"].startswith("<agent_status>") for message in requests[0]["messages"])
    final_request = requests[-1]
    environment_message = next(
        message["content"]
        for message in final_request["messages"]
        if str(message.get("content", "")).startswith("ENVIRONMENT_CONTEXT_JSON=")
    )
    environment = json.loads(environment_message.removeprefix("ENVIRONMENT_CONTEXT_JSON="))
    assert environment["completed_observations"][0]["tool"] == "calculator"
    assert environment["completed_observations"][0]["result_available_in_context"] is True
    assert environment["execution_budget"]["max_steps"] == 10


def test_context_can_contain_an_explicit_user_input_module() -> None:
    client = TestClient(app)
    workflow = {
        "version": "0.1",
        "name": "显式用户输入模块",
        "nodes": [
            {"id": "context", "type": "input", "label": "上下文", "x": 0, "y": 0, "config": {}},
            {"id": "user-message", "type": "user_input", "label": "用户输入", "x": 0, "y": 0, "parentId": "context", "config": {"text": "计算 12 * 7"}},
            {"id": "definitions", "type": "tool_definition", "label": "工具定义", "x": 0, "y": 0, "parentId": "context", "config": {"tool_ids": "[\"calculator\"]"}},
            {"id": "calculator", "type": "tool", "label": "计算器", "x": 0, "y": 0, "config": {"tool": "calculator"}},
            {"id": "loop", "type": "react_loop", "label": "循环", "x": 0, "y": 0, "config": {"max_steps": "4"}},
            {"id": "agent", "type": "agent", "label": "Agent", "x": 0, "y": 0, "parentId": "loop", "config": {"brain_type": "test", "tool_ids": "[\"calculator\"]"}},
        ],
        "edges": [{"id": "entry", "from": "context", "to": "loop"}, {"id": "into-agent", "from": "loop", "to": "agent"}],
    }
    created = client.post("/api/v1/workflows/runs", json={"workflow": workflow})
    assert created.status_code == 201, created.text
    assert created.json()["run"]["input"] == "计算 12 * 7"
    completed = client.post(f"/api/v1/runs/{created.json()['run']['run_id']}/execute")
    assert completed.json()["output"] == "84"


def test_workflow_direct_answer_ignores_legacy_demo_answer_and_does_not_call_a_tool() -> None:
    client = TestClient(app)
    workflow = {
        "version": "0.1",
        "name": "一问一答",
        "nodes": [
            {"id": "user", "type": "user_input", "label": "用户输入", "x": 0, "y": 0, "config": {"text": "什么是 Agent？"}},
            {"id": "context", "type": "input", "label": "上下文", "x": 0, "y": 0, "parentId": "loop", "config": {}},
            {"id": "loop", "type": "react_loop", "label": "循环", "x": 0, "y": 0, "config": {"max_steps": "3"}},
            {"id": "agent", "type": "agent", "label": "Agent", "x": 0, "y": 0, "parentId": "loop", "config": {"brain_type": "test", "tool_ids": "[]", "demo_answer": "Agent 能观察、决策并行动。"}},
            {"id": "final", "type": "final", "label": "输出", "x": 0, "y": 0, "config": {"answer": "{{last_tool_result}}"}},
        ],
        "edges": [{"id": "start", "from": "user", "to": "loop"}, {"id": "output", "from": "loop", "to": "final"}],
    }
    created = client.post("/api/v1/workflows/runs", json={"workflow": workflow})
    assert created.status_code == 201, created.text
    completed = client.post(f"/api/v1/runs/{created.json()['run']['run_id']}/execute").json()
    assert completed["output"] == "什么是 Agent？"
    assert not any(event["type"] == "tool.start" for event in completed["events"])


def test_sqlite_store_reloads_agents_and_runs() -> None:
    database_path = Path(__file__).resolve().parent / f".runtime-{uuid4().hex}.db"
    database = str(database_path)
    try:
        first_store = SQLiteStore(database)
        spec = MiniAgentSpec(name="Persistent Agent")
        first_store.save_agent("agent-persist", spec)
        first_store.save_run(
            AgentRunState(
                run_id="run-persist",
                agent_id="agent-persist",
                input="hello",
                status=RunStatus.CREATED,
            )
        )
        first_store.close()

        second_store = SQLiteStore(database)
        assert second_store.get_agent("agent-persist") == spec
        reloaded = second_store.get_run("run-persist")
        assert reloaded is not None
        assert reloaded.status == RunStatus.CREATED
        assert reloaded.input == "hello"
        second_store.close()
    finally:
        database_path.unlink(missing_ok=True)
