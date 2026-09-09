from agentscratch.interpreter import AgentInterpreter, build_context
from agentscratch.llm import TinyLLMBrain
from agentscratch.schemas import AgentRunState, BrainSpec, MiniAgentSpec, RunStatus


def test_interpreter_surfaces_local_llm_unavailable():
    spec = MiniAgentSpec(
        name="Tiny Agent",
        brain=BrainSpec(type="tiny_llm", model="qwen-tiny", max_attempts=1),
    )
    run = AgentRunState(
        run_id="run-llm-unavailable",
        agent_id="agent-test",
        input="hello",
        status=RunStatus.CREATED,
        context=build_context("hello"),
    )
    brain = TinyLLMBrain(base_url="http://127.0.0.1:9/v1", model="qwen-tiny", timeout=0.1)
    result = AgentInterpreter(brain).run(run, spec)

    assert result.status == RunStatus.FAILED
    assert result.error is not None
    assert "local LLM service unavailable" in result.error
    assert any(event.type == "llm.error" for event in result.events)

