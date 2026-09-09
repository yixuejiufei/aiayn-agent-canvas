from agentscratch.schemas import AgentEvent, AgentRunState, RunStatus
from agentscratch.storage import SQLiteStore


def _event(event_id: str, run_id: str, step: int, type_: str, data: dict) -> AgentEvent:
    return AgentEvent(
        event_id=event_id,
        run_id=run_id,
        step=step,
        type=type_,
        timestamp=1000.0 + len(event_id),
        node="test",
        data=data,
    )


def test_model_tool_metrics_aggregate_success_failure_and_clear() -> None:
    store = SQLiteStore(":memory:")
    run = AgentRunState(
        run_id="run-metrics",
        agent_id="agent-metrics",
        input="test",
        status=RunStatus.COMPLETED,
        events=[
            _event("start", "run-metrics", 1, "llm.start", {"provider": "tiny_llm", "model": "model-a"}),
            _event("end", "run-metrics", 1, "llm.end", {"model": "model-a", "action": {"action": "call_tools", "tool_calls": [{"tool": "a"}, {"tool": "b"}]}}),
            _event("validate", "run-metrics", 1, "action.validate", {"valid": True, "action": {"action": "call_tools", "tool_calls": [{"tool": "a"}, {"tool": "b"}]}}),
            _event("tool-a", "run-metrics", 1, "tool.start", {}),
            _event("tool-a-ok", "run-metrics", 1, "tool.end", {}),
            _event("tool-b", "run-metrics", 1, "tool.start", {}),
            _event("tool-b-fail", "run-metrics", 1, "tool.error", {}),
            _event("finish", "run-metrics", 1, "agent.finish", {}),
        ],
    )
    store.save_run(run)

    metric = store.model_tool_stats()["models"][0]
    assert metric["model"] == "model-a"
    assert metric["tool_calls_requested"] == 2
    assert metric["tool_calls_validated"] == 2
    assert metric["tool_calls_started"] == 2
    assert metric["tool_calls_succeeded"] == 1
    assert metric["tool_calls_failed"] == 1
    assert metric["tool_execution_success_rate"] == 0.5
    assert metric["proposal_validation_rate"] == 1.0
    assert metric["plan_to_tool_success_rate"] == 0.5
    assert metric["decision_errors"] == 0
    assert metric["decision_error_breakdown"] == {}
    assert metric["completed_runs"] == 1

    store.clear_model_tool_stats()
    assert store.model_tool_stats()["models"] == []


def test_model_metrics_attribute_errors_to_the_server_reported_model() -> None:
    store = SQLiteStore(":memory:")
    run = AgentRunState(
        run_id="run-model-alias", agent_id="agent", input="test", status=RunStatus.FAILED,
        events=[
            _event("requested-1", "run-model-alias", 1, "llm.start", {"provider": "tiny_llm", "model": "selected-2b"}),
            _event("actual-1", "run-model-alias", 1, "llm.end", {"model": "fallback-0.5b", "action": {"action": "call_tool"}}),
            _event("start-tool", "run-model-alias", 1, "tool.start", {}),
            _event("end-tool", "run-model-alias", 1, "tool.end", {}),
            _event("requested-2", "run-model-alias", 2, "llm.start", {"provider": "tiny_llm", "model": "selected-2b"}),
            _event("decision-error", "run-model-alias", 2, "llm.error", {"error": "invalid route"}),
        ],
    )
    store.save_run(run)

    metrics = store.model_tool_stats()["models"]
    assert len(metrics) == 1
    assert metrics[0]["model"] == "fallback-0.5b"
    assert metrics[0]["decisions"] == 1
    assert metrics[0]["decision_errors"] == 1
    assert metrics[0]["decision_error_breakdown"] == {"other": 1}
    assert metrics[0]["tool_calls_succeeded"] == 1
