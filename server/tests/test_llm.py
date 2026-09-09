import pytest
from fastapi.testclient import TestClient

from agentscratch.llm import TinyLLMBrain, parse_action
from agentscratch.prompts import PromptBuilder
from agentscratch.schemas import BrainSpec, MiniAgentSpec, ToolParameterSpec, ToolSpec


def test_parse_plain_json():
    raw = '{"thought":"need math","action":"call_tool","tool":"calculator","args":{"expression":"12*7"}}'
    action = parse_action(raw)
    assert action.tool == "calculator"


def test_parse_fenced_json():
    raw = '```json\n{"thought":"done","action":"final_answer","answer":"84","done":true}\n```'
    action = parse_action(raw)
    assert action.answer == "84"


def test_parse_fenced_json_with_nested_args():
    raw = (
        'Here is the action:\n```json\n'
        '{"thought":"need math","action":"call_tool","tool":"calculator",'
        '"args":{"expression":"12*7"},"answer":null,"done":false}\n```'
    )
    action = parse_action(raw)
    assert action.args == {"expression": "12*7"}


def test_parse_rejects_non_json():
    with pytest.raises(ValueError):
        parse_action("no json here")


def test_prompt_builder_preserves_context_and_serializes_tools():
    messages = PromptBuilder().build(
        [
            {"role": "system", "content": "runtime context"},
            {"role": "user", "content": "12 * 7"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "calculator", "arguments": "{\"expression\":\"12 * 7\"}"}}]},
            {"role": "tool", "name": "calculator", "tool_call_id": "call-1", "content": 84},
        ],
        tools=[{"name": "calculator"}, {"name": "final_answer"}],
    )
    assert messages[0]["role"] == "system"
    assert "runtime context" in messages[1]["content"]
    assert messages[2] == {"role": "user", "content": "12 * 7"}
    assert messages[3]["role"] == "assistant"
    assert "TOOL_CALLS" in messages[3]["content"]
    assert "TOOL_RESULT" in messages[4]["content"]
    assert '"tool_call_id": "call-1"' in messages[4]["content"]
    assert '"name": "calculator"' in messages[5]["content"]
    assert "final_answer" not in messages[5]["content"]


def test_prompt_builder_prohibits_route_without_downstream_llm():
    prompt = PromptBuilder().build([{"role": "user", "content": "12 * 7"}])[0]["content"]
    assert '"action":"call_tool|call_tools|final_answer|ask_user"' in prompt
    assert "route is prohibited" in prompt


def test_prompt_builder_allows_route_only_with_explicit_downstream_llm():
    prompt = PromptBuilder().build([
        {"role": "system", "content": "WORKFLOW_NEXT_NODES={\"next_agents\":[{\"id\":\"llm-b\"}]}"},
        {"role": "user", "content": "delegate"},
    ])[0]["content"]
    assert '"action":"call_tool|call_tools|final_answer|ask_user|route"' in prompt
    assert "route is prohibited" not in prompt


def _completion(content: str):
    return {
        "model": "qwen-test",
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def test_tiny_llm_brain_uses_openai_compatible_endpoint(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return _completion('{"thought":"calculate","action":"call_tool","tool":"calculator","args":{"expression":"12*7"}}')

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, json=None, headers=None):
            captured["url"] = url
            captured["json"] = json
            return FakeResponse()

    monkeypatch.setattr("agentscratch.llm.httpx.Client", FakeClient)
    brain = TinyLLMBrain(base_url="http://127.0.0.1:8080/v1", model="qwen-test")
    result = brain.decide(
        MiniAgentSpec(name="agent"),
        [{"role": "user", "content": "计算 12*7"}],
    )

    assert captured["url"] == "http://127.0.0.1:8080/v1/chat/completions"
    assert captured["json"]["model"] == "qwen-test"
    assert captured["json"]["response_format"]["type"] == "json_schema"
    assert result.action.tool == "calculator"
    assert result.model == "qwen-test"
    assert result.usage["prompt_tokens"] == 10
    assert result.duration_ms is not None
    assert result.request_tools == []


def test_tiny_llm_native_tools_preserves_api_tool_protocol(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "model": "qwen-test",
                "choices": [{"message": {"content": None, "tool_calls": [{
                    "id": "call-weather", "type": "function",
                    "function": {"name": "calculator", "arguments": "{\"expression\":\"12*7\"}"},
                }]}}],
            }

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, json=None, headers=None):
            captured["json"] = json
            return FakeResponse()

    monkeypatch.setattr("agentscratch.llm.httpx.Client", FakeClient)
    spec = MiniAgentSpec(
        name="native",
        brain=BrainSpec(tool_protocol="native_tools"),
        tools=[],
    )
    # Include a real ToolSpec so the request proves the top-level tools shape.
    from agentscratch.schemas import ToolParameterSpec, ToolSpec
    spec.tools = [ToolSpec(name="calculator", description="calculate", parameters={"expression": ToolParameterSpec(type="string")})]
    result = TinyLLMBrain(base_url="http://127.0.0.1:8080/v1", model="qwen-test").decide(
        spec,
        [
            {"role": "system", "content": "rule"},
            {"role": "user", "content": "12 * 7"},
        ],
    )

    assert "response_format" not in captured["json"]
    assert captured["json"]["tools"][0]["function"]["name"] == "calculator"
    assert result.action.call_id == "call-weather"
    assert result.action.args == {"expression": "12*7"}
    assert result.request_tools == captured["json"]["tools"]


def test_native_tools_parses_multiple_tool_calls(monkeypatch: pytest.MonkeyPatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": None, "tool_calls": [
                {"id": "call-calc", "type": "function", "function": {"name": "calculator", "arguments": '{"expression":"2+2"}'}},
                {"id": "call-search", "type": "function", "function": {"name": "mock_search", "arguments": '{"query":"agent"}'}},
            ]}}]}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr("agentscratch.llm.httpx.Client", FakeClient)
    spec = MiniAgentSpec(name="native", brain=BrainSpec(tool_protocol="native_tools"), tools=[
        ToolSpec(name="calculator", description="calc", parameters={"expression": ToolParameterSpec(type="string")}),
        ToolSpec(name="mock_search", description="search", parameters={"query": ToolParameterSpec(type="string")}),
    ])
    result = TinyLLMBrain().decide(spec, [{"role": "user", "content": "both"}])
    assert result.action.action == "call_tools"
    assert [call.call_id for call in result.action.tool_calls] == ["call-calc", "call-search"]


def test_agent_model_overrides_client_default(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return _completion('{"thought":"done","action":"final_answer","answer":"ok","done":true}')

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, json=None, headers=None):
            captured["model"] = json["model"]
            return FakeResponse()

    monkeypatch.setattr("agentscratch.llm.httpx.Client", FakeClient)
    brain = TinyLLMBrain(base_url="http://127.0.0.1:8080/v1", model="env-model")
    spec = MiniAgentSpec(name="agent", brain=BrainSpec(model="agent-model"))
    brain.decide(spec, [{"role": "user", "content": "hello"}])
    assert captured["model"] == "agent-model"


def test_tiny_llm_brain_retries_invalid_json(monkeypatch: pytest.MonkeyPatch):
    calls = []

    class FakeResponse:
        def __init__(self, body):
            self._body = body

        def raise_for_status(self):
            return None

        def json(self):
            return self._body

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, *args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                return FakeResponse(_completion("not json"))
            return FakeResponse(_completion('{"thought":"done","action":"final_answer","answer":"84","done":true}'))

    monkeypatch.setattr("agentscratch.llm.httpx.Client", FakeClient)
    brain = TinyLLMBrain(base_url="http://127.0.0.1:8080/v1", model="qwen-test")
    spec = MiniAgentSpec(name="agent", brain=BrainSpec(max_attempts=2))
    result = brain.decide(spec, [{"role": "user", "content": "hello"}])

    assert len(calls) == 2
    assert result.attempts == 2
    assert result.action.answer == "84"


def test_tiny_llm_brain_reports_last_raw_output(monkeypatch: pytest.MonkeyPatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return _completion("still not json")

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr("agentscratch.llm.httpx.Client", FakeClient)
    brain = TinyLLMBrain(base_url="http://127.0.0.1:8080/v1", model="qwen-test")
    spec = MiniAgentSpec(name="agent", brain=BrainSpec(max_attempts=2))
    with pytest.raises(ValueError) as exc_info:
        brain.decide(spec, [{"role": "user", "content": "hello"}])
    assert exc_info.value.raw_output == "still not json"


def test_agent_api_accepts_brain_model():
    from agentscratch.main import app
    client = TestClient(app)
    response = client.post(
        "/api/v1/agents",
        json={"name": "Model Agent", "brain_type": "tiny_llm", "brain_model": "qwen2.5-0.5b"},
    )
    assert response.status_code == 201
    assert response.json()["brain"]["model"] == "qwen2.5-0.5b"



