import time
from pathlib import Path
from threading import Event

import pytest
import aiayn_agent_canvas.main as main_module
from fastapi.testclient import TestClient

from aiayn_agent_canvas.main import app


def test_health() -> None:
    client = TestClient(app)
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_teaching_workspace_tree_endpoint(tmp_path) -> None:
    (tmp_path / "notes.txt").write_text("metadata only", encoding="utf-8")
    client = TestClient(app)
    response = client.get("/api/v1/teaching-workspace/tree", params={"root_path": str(tmp_path)})
    assert response.status_code == 200
    tree = response.json()
    assert tree["root_path"] == str(tmp_path.resolve())
    assert tree["entries"][0]["name"] == "notes.txt"
    assert tree["entries"][0]["readable"] is True


def test_browser_directory_import_creates_a_safe_teaching_workspace(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AIAYN_TEACHING_WORKSPACE_IMPORT_DIR", str(tmp_path / "imports"))
    client = TestClient(app)
    response = client.post(
        "/api/v1/teaching-workspace/import-directory",
        data={"source_name": "lesson-files"},
        files=[
            ("files", ("lesson-files/overview.txt", b"teaching workspace content", "text/plain")),
            ("files", ("lesson-files/notes/detail.md", b"nested content", "text/markdown")),
        ],
    )
    assert response.status_code == 200
    workspace = response.json()
    root = tmp_path / "imports"
    assert workspace["selected"] is True
    assert workspace["root_path"].startswith(str(root.resolve()))
    assert (Path(workspace["root_path"]) / "overview.txt").read_text(encoding="utf-8") == "teaching workspace content"
    assert (Path(workspace["root_path"]) / "notes" / "detail.md").read_text(encoding="utf-8") == "nested content"


def test_workflow_run_accepts_an_imported_teaching_workspace(tmp_path) -> None:
    (tmp_path / "report.txt").write_text("workspace report", encoding="utf-8")
    workflow = {
        "version": "0.1",
        "name": "workspace workflow",
        "max_steps": 4,
        "nodes": [
            {"id": "workspace", "type": "workspace", "label": "教学工作区", "x": 0, "y": 0, "config": {"root_path": str(tmp_path)}},
            {"id": "context", "type": "input", "label": "上下文", "x": 0, "y": 0, "config": {"user_input_module_id": "user", "context_window_size": "8192"}},
            {"id": "user", "type": "user_input", "label": "用户输入", "x": 0, "y": 0, "parentId": "context", "config": {"text": "概括工作区中的报告"}},
            {"id": "definitions", "type": "tool_definition", "label": "工具定义", "x": 0, "y": 0, "parentId": "context", "config": {"tool_ids": "[\"search\", \"read\"]"}},
            {"id": "search", "type": "tool", "label": "搜索文件", "x": 0, "y": 0, "config": {"tool": "search_files"}},
            {"id": "read", "type": "tool", "label": "读取文件", "x": 0, "y": 0, "config": {"tool": "read_file"}},
            {"id": "loop", "type": "react_loop", "label": "ReAct 循环", "x": 0, "y": 0, "config": {"max_steps": "4"}},
            {"id": "agent", "type": "agent", "label": "LLM", "x": 0, "y": 0, "parentId": "loop", "config": {"brain_type": "test", "tool_ids": "[\"search\", \"read\"]"}},
        ],
        "edges": [
            {"id": "entry", "from": "context", "to": "loop"},
            {"id": "agent-entry", "from": "loop", "to": "agent"},
        ],
    }
    response = TestClient(app).post("/api/v1/workflows/runs", json={"workflow": workflow})
    assert response.status_code == 201, response.text
    context = response.json()["run"]["context"]
    assert any(message["content"].startswith("TEACHING_WORKSPACE_INDEX=") for message in context)


def test_agent_loop_with_calculator() -> None:
    client = TestClient(app)
    agent = client.post(
        "/api/v1/agents",
        json={"name": "Calculator Agent", "tools": ["calculator", "final_answer"], "brain_type": "test"},
    ).json()
    run = client.post(
        f"/api/v1/agents/{agent['agent_id']}/runs",
        json={"input": "12 * 7"},
    ).json()
    result = client.post(f"/api/v1/runs/{run['run_id']}/execute").json()
    assert result["status"] == "completed"
    assert result["step"] >= 1
    types = [event["type"] for event in result["events"]]
    assert "run.start" in types
    assert "llm.start" in types
    assert "tool.start" in types
    assert "agent.finish" in types
    assert any(event["data"].get("raw_output") for event in result["events"] if event["type"] == "llm.end")


def test_two_document_case_replays_parallel_and_dependent_tool_rounds() -> None:
    client = TestClient(app)
    created = client.post("/api/v1/cases/two-files-sum/runs")
    assert created.status_code == 201, created.text

    completed = client.post(f"/api/v1/runs/{created.json()['run']['run_id']}/execute")
    assert completed.status_code == 200
    payload = completed.json()
    assert payload["status"] == "completed"
    assert payload["output"] == "两个文件中的数值相加结果是 42。"
    assert len(payload["memory"]["observations"]) == 5
    parallel_starts = [event for event in payload["events"] if event["type"] == "tool.parallel.start"]
    assert len(parallel_starts) == 2


def test_file_summary_case_searches_then_reads_before_answering() -> None:
    client = TestClient(app)
    created = client.post("/api/v1/cases/file-summary/runs")
    assert created.status_code == 201, created.text

    completed = client.post(f"/api/v1/runs/{created.json()['run']['run_id']}/execute")
    assert completed.status_code == 200
    payload = completed.json()
    assert payload["status"] == "completed"
    assert "销售额同比增长 18%" in payload["output"]
    actions = [event["data"]["action"] for event in payload["events"] if event["type"] == "llm.end"]
    assert [action["action"] for action in actions] == ["call_tool", "call_tool", "final_answer"]
    assert [action.get("tool") for action in actions[:2]] == ["search_files", "read_file"]
    observations = [event for event in payload["events"] if event["type"] == "tool.end"]
    assert [event["node"] for event in observations] == ["search_files", "read_file"]


def test_file_choice_case_browses_then_asks_user_before_reading() -> None:
    client = TestClient(app)
    created = client.post("/api/v1/cases/file-choice/runs")
    assert created.status_code == 201, created.text
    run_id = created.json()["run"]["run_id"]

    waiting = client.post(f"/api/v1/runs/{run_id}/execute")
    assert waiting.status_code == 200
    waiting_payload = waiting.json()
    assert waiting_payload["status"] == "waiting_user"
    assert "1." in waiting_payload["waiting_question"]
    assert len(waiting_payload["waiting_choices"]) == 3
    assert all(choice["file_id"] and choice["name"] for choice in waiting_payload["waiting_choices"])
    assert waiting_payload["waiting_response_schema"] == {
        "type": "file_selection",
        "source_observation": next(event["data"]["call_id"] for event in waiting_payload["events"] if event["type"] == "tool.end"),
        "candidate_file_ids": [choice["file_id"] for choice in waiting_payload["waiting_choices"]],
        "allow_custom_input": True,
        "selection_mode": "single",
        "min_selections": 1,
        "max_selections": 1,
    }
    assert [event["node"] for event in waiting_payload["events"] if event["type"] == "tool.end"] == ["search_files"]

    completed = client.post(
        f"/api/v1/runs/{run_id}/resume",
        json={"selection_id": waiting_payload["waiting_choices"][0]["file_id"]},
    )
    assert completed.status_code == 200
    payload = completed.json()
    assert payload["status"] == "completed"
    assert "内容摘要" in payload["output"]
    assert [event["node"] for event in payload["events"] if event["type"] == "tool.end"] == ["search_files", "read_file"]
    assert any(
        event["type"] == "user.input"
        and event["data"]["selection"]["status"] == "resolved"
        and event["data"]["selection"]["source"] == "choice_button"
        for event in payload["events"]
    )

