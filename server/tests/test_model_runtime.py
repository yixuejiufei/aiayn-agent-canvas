from pathlib import Path

import httpx

from agentscratch.model_runtime import LocalModelRuntime


def test_local_model_eligibility_excludes_partial_and_projection_files() -> None:
    runtime = LocalModelRuntime()
    size = 200 * 1024 * 1024
    assert runtime._is_eligible_gguf(Path("Qwen3.5-2B-IQ4_XS.gguf"), size)
    assert not runtime._is_eligible_gguf(Path("Qwen3.5-2B-Q4_0.incomplete.gguf"), size)
    assert not runtime._is_eligible_gguf(Path("downloading-Qwen3.5.gguf.part"), size)
    assert not runtime._is_eligible_gguf(Path("mmproj-F16.gguf"), size)


def test_connection_uses_existing_service_until_a_managed_model_is_active() -> None:
    runtime = LocalModelRuntime()
    runtime.model_root = Path("Z:/does-not-exist")
    assert runtime.discover() == []
    assert runtime.connection()[0].endswith("/v1")


def test_connection_recovers_surviving_managed_server(monkeypatch) -> None:
    runtime = LocalModelRuntime()
    profile = {"id": "teaching-model", "label": "Teaching Model", "path": "E:/AIModels/teaching.gguf", "size_bytes": 1, "is_instruction_tuned": True, "active": False}
    monkeypatch.setattr(runtime, "discover", lambda: [profile])

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"data": [{"id": "teaching-model"}]}

    monkeypatch.setattr(httpx, "get", lambda *_args, **_kwargs: Response())
    assert runtime.connection() == ("http://127.0.0.1:8081/v1", "teaching-model")
    assert runtime.active_status()["device"] == "reconnected"
