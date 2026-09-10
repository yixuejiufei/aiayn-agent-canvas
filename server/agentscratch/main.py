"""FastAPI application for local AgentScratch runs."""

from __future__ import annotations

import json
import re
import httpx
import sqlite3
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .brain import ClarificationBrain, TestBrain
from .cases import get_case as get_builtin_case, list_cases as list_builtin_cases
from .events import EventSink
from .llm import TinyLLMBrain
from .model_runtime import local_model_runtime
from .interpreter import AgentInterpreter, build_context
from .workflow import WorkflowBrain, WorkflowValidationError, parse_module_ids, validate_workflow
from .schemas import (
    AgentEvent,
    AgentPolicy,
    AgentRunState,
    BrainSpec,
    CreateAgentRequest,
    CreateRunRequest,
    CreateTeachingCaseRequest,
    CreateWorkflowRunRequest,
    MiniAgentSpec,
    ResumeRunRequest,
    RunStatus,
    ToolParameterSpec,
    ToolSpec,
    WorkflowSpec,
)
from .tools import (
    available_tool_names,
    choose_teaching_workspace_directory,
    teaching_resources,
    teaching_workspace_index,
    teaching_workspace_tree,
)
from .storage import SQLiteStore


app = FastAPI(title="AgentScratch API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

store = SQLiteStore()


def build_brain(spec: MiniAgentSpec, run: AgentRunState | None = None, context_window_tokens: int | None = None):
    if spec.workflow:
        workflow = WorkflowSpec.model_validate(spec.workflow)
        base_spec = spec.model_copy(update={"workflow": None})
        tool_nodes = {node.id: node.config.get("tool", "") for node in workflow.nodes if node.type == "tool"}
        context = next(node for node in workflow.nodes if node.type == "input")
        try:
            workflow_context_window = int(context.config.get("context_window_size", "8192"))
        except ValueError as exc:
            raise WorkflowValidationError("Context module has invalid context_window_size") from exc
        if not 256 <= workflow_context_window <= 131072:
            raise WorkflowValidationError("Context module context_window_size must be between 256 and 131072")
        definition_modules = [
            node for node in workflow.nodes if node.type == "tool_definition" and node.parent_id == context.id
        ]
        context_tool_ids = {
            tool_id for module in definition_modules for tool_id in parse_module_ids(module)
        }
        if not definition_modules:
            context_tool_ids = set(tool_nodes)
        delegates = {}
        agent_specs = {}
        for node in workflow.nodes:
            if node.type != "agent":
                continue
            brain_type = node.config.get("brain_type", base_spec.brain.type)
            if brain_type not in {"test", "tiny_llm", "clarification"}:
                raise WorkflowValidationError(f"unsupported agent brain_type: {brain_type}")
            node_spec = base_spec.model_copy(
                update={
                    "brain": BrainSpec(
                        type=brain_type,
                        model=node.config.get("model") or base_spec.brain.model,
                        device=node.config.get("device", base_spec.brain.device),
                        tool_protocol=node.config.get("tool_protocol", base_spec.brain.tool_protocol),
                        temperature=node.config.get("temperature", base_spec.brain.temperature),
                        top_p=node.config.get("top_p", base_spec.brain.top_p),
                        top_k=node.config.get("top_k", base_spec.brain.top_k),
                        max_tokens=node.config.get("max_tokens", base_spec.brain.max_tokens),
                        repeat_penalty=node.config.get("repeat_penalty", base_spec.brain.repeat_penalty),
                        max_attempts=node.config.get("max_attempts", base_spec.brain.max_attempts),
                    )
                }
            )
            selected_ids = set(parse_module_ids(node)) if "tool_ids" in node.config else set()
            if not selected_ids:
                selected_ids = {
                    target.id
                    for target in workflow.nodes
                    if target.type == "tool" and any(edge.from_ == node.id and edge.to == target.id for edge in workflow.edges)
                }
            if not selected_ids:
                selected_ids = set(context_tool_ids)
            selected_names = {tool_nodes[tool_id] for tool_id in selected_ids & context_tool_ids if tool_nodes.get(tool_id)}
            node_spec = node_spec.model_copy(
                update={
                    "tools": [
                        tool for tool in base_spec.tools if tool.name in selected_names or tool.name == "final_answer"
                    ]
                }
            )
            agent_specs[node.id] = node_spec
            delegates[node.id] = build_brain(node_spec, context_window_tokens=workflow_context_window)
        return WorkflowBrain(workflow, delegates, agent_specs, run=run)
    if spec.brain.type == "test":
        return TestBrain()
    if spec.brain.type == "clarification":
        return ClarificationBrain()
    # A selected GGUF is a concrete runtime choice, not merely a label sent in
    # the OpenAI request. Restore it when the API has restarted so llama.cpp
    # cannot silently answer using the fallback external server/model.
    local_model_runtime.ensure_selected(spec.brain.model, spec.brain.device, context_size=context_window_tokens)
    base_url, active_model = local_model_runtime.connection()
    return TinyLLMBrain(base_url=base_url, model=active_model)


def default_tool(name: str) -> ToolSpec:
    definitions: dict[str, ToolSpec] = {
        "calculator": ToolSpec(
            name="calculator",
            description="计算数学表达式",
            parameters={"expression": ToolParameterSpec(type="string")},
            risk="read", parallel_safe=True,
        ),
        "mock_search": ToolSpec(
            name="mock_search",
            description="在本地教学语料中搜索",
            parameters={"query": ToolParameterSpec(type="string")},
            risk="read", parallel_safe=True,
        ),
        "search_files": ToolSpec(
            name="search_files",
            description="在当前运行已授权的教学文件目录中按文件名搜索，返回候选文件的 file_id；directory 可省略，省略或 '.' 表示授权根目录；file_name 省略时列出最多 20 个可读取文本文件的元数据；不能访问本机任意路径",
            parameters={
                "directory": ToolParameterSpec(type="string", description="可选的相对目录，例如 季度报告；根目录使用 '.'", required=False),
                "file_name": ToolParameterSpec(type="string", description="可选的完整或部分文件名，例如 华东销售复盘.md；省略时浏览该目录", required=False),
            },
            risk="read", parallel_safe=True,
        ),
        "read_file": ToolSpec(
            name="read_file",
            description="读取 search_files 已返回的一个 file_id 的文本内容；必须先搜索，不能传入本机路径",
            parameters={"file_id": ToolParameterSpec(type="string", description="search_files Observation 中返回的 file_id")},
            risk="read", parallel_safe=False,
        ),
        "read_document": ToolSpec(
            name="read_document",
            description="读取当前运行已授权教学资源中的文本；只能使用 file_id，不能使用本机路径",
            parameters={"file_id": ToolParameterSpec(type="string", description="已授权资源 ID，例如 file-a")},
            risk="read", parallel_safe=True,
        ),
        "extract_numbers": ToolSpec(
            name="extract_numbers",
            description="从已获得的文本中提取数值，返回结构化 values 数组",
            parameters={"text": ToolParameterSpec(type="string", description="来自用户或工具 Observation 的文本")},
            risk="read", parallel_safe=True,
        ),
        "memory_store": ToolSpec(
            name="memory_store",
            description="保存信息到 Agent 记忆",
            parameters={
                "key": ToolParameterSpec(type="string"),
                "value": ToolParameterSpec(type="string"),
            },
            risk="write", requires_confirmation=True, parallel_safe=False,
        ),
        "memory_lookup": ToolSpec(
            name="memory_lookup",
            description="读取 Agent 记忆",
            parameters={"key": ToolParameterSpec(type="string")},
            risk="read", parallel_safe=True,
        ),
        "final_answer": ToolSpec(
            name="final_answer",
            description="输出最终答案",
            parameters={"answer": ToolParameterSpec(type="string")},
        ),
    }
    if name not in definitions:
        raise ValueError(f"unknown tool: {name}")
    return definitions[name]


@app.get("/api/v1/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "AgentScratch API", "version": "0.1.0"}


@app.post("/api/v1/teaching-workspace/select-directory")
def select_teaching_workspace_directory() -> dict[str, Any]:
    """Open a native folder picker only after the learner presses the UI button."""
    try:
        selected = choose_teaching_workspace_directory()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if selected is None:
        return {"selected": False, "root_path": "", "tree": None}
    return {
        "selected": True,
        "root_path": str(selected),
        "tree": teaching_workspace_tree(str(selected)),
    }


@app.get("/api/v1/teaching-workspace/tree")
def get_teaching_workspace_tree(root_path: str) -> dict[str, Any]:
    try:
        return teaching_workspace_tree(root_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/v1/llm/status")
def llm_status() -> dict[str, Any]:
    base_url, active_model = local_model_runtime.connection()
    brain = TinyLLMBrain(base_url=base_url, model=active_model, timeout=2.0)
    runtime_status = local_model_runtime.active_status()
    try:
        with httpx.Client(timeout=2.0) as client:
            response = client.get(f"{brain.base_url}/models")
            response.raise_for_status()
            body = response.json()
        models = [item.get("id") for item in body.get("data", []) if item.get("id")]
        return {
            "base_url": brain.base_url,
            "model": active_model,
            "available": True,
            "models": models,
            "managed": runtime_status["managed"],
            "context_size": runtime_status["context_size"],
            "message": "local model service available",
        }
    except Exception as exc:
        return {
            "base_url": brain.base_url,
            "model": active_model,
            "available": False,
            "models": [],
            "managed": runtime_status["managed"],
            "context_size": runtime_status["context_size"],
            "message": f"local model service unavailable: {exc}",
        }


@app.get("/api/v1/local-models")
def list_local_models() -> dict[str, Any]:
    """List complete GGUF files from the user-configured model directory."""
    return {
        "directory": str(local_model_runtime.model_root),
        "models": local_model_runtime.discover(),
        "devices": local_model_runtime.devices(),
        "active": local_model_runtime.active_status(),
    }


@app.post("/api/v1/local-models/{model_id}/activate")
def activate_local_model(
    model_id: str,
    device: str = Body("cpu", embed=True),
    context_size: int | None = Body(None, embed=True),
) -> dict[str, Any]:
    """Start one managed llama.cpp server for a selected local GGUF profile."""
    try:
        profile = local_model_runtime.activate(model_id, device, context_size=context_size)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"profile": profile, "active": local_model_runtime.active_status()}


@app.get("/api/v1/tools")
def list_tools() -> list[str]:
    return available_tool_names()


@app.get("/api/v1/metrics/model-tool-calls")
def model_tool_call_metrics() -> dict[str, Any]:
    # The workbench comparison is for real local LLMs.  Deterministic test
    # brains and workflow helper pseudo-models would otherwise make a 100%
    # row that obscures the model the learner is actually evaluating.
    metrics = store.model_tool_stats()
    metrics["models"] = [
        item
        for item in metrics["models"]
        if item["provider"] == "tiny_llm"
        and item["model"] not in {"None", "workflow-graph", "workflow-teaching-case"}
    ]
    return metrics


@app.delete("/api/v1/metrics/model-tool-calls")
def clear_model_tool_call_metrics() -> dict[str, Any]:
    return {"reset_at": store.clear_model_tool_stats(), "models": []}


@app.get("/api/v1/cases")
def teaching_cases() -> list[dict[str, Any]]:
    return list_builtin_cases() + store.list_cases()


def find_teaching_case(case_id: str) -> dict[str, Any] | None:
    return get_builtin_case(case_id) or store.get_case(case_id)


@app.get("/api/v1/cases/{case_id}")
def teaching_case(case_id: str) -> dict[str, Any]:
    case = find_teaching_case(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="teaching case not found")
    return case


@app.post("/api/v1/cases", status_code=201)
def create_teaching_case(request: CreateTeachingCaseRequest) -> dict[str, Any]:
    if find_teaching_case(request.case_id) is not None:
        raise HTTPException(status_code=409, detail="teaching case already exists")
    for name in request.tools:
        if name not in available_tool_names():
            raise HTTPException(status_code=400, detail=f"unknown tool: {name}")
    case = request.model_dump()
    try:
        store.save_case(case)
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="teaching case already exists") from exc
    return case


@app.post("/api/v1/agents", status_code=201)
def create_agent(request: CreateAgentRequest) -> dict[str, Any]:
    for name in request.tools:
        if name not in available_tool_names():
            raise HTTPException(status_code=400, detail=f"unknown tool: {name}")
    spec = MiniAgentSpec(
        name=request.name,
        description=request.description,
        brain=BrainSpec(type=request.brain_type, model=request.brain_model),
        tools=[default_tool(name) for name in request.tools],
    )
    if not any(tool.name == "final_answer" for tool in spec.tools):
        spec.tools.append(default_tool("final_answer"))
    agent_id = f"agent-{uuid.uuid4().hex[:12]}"
    spec_with_id = spec.model_copy(update={"name": spec.name})
    store.save_agent(agent_id, spec_with_id)
    # Return id through a wrapper-like field while keeping schema stable for M0.
    return {"agent_id": agent_id, **spec_with_id.model_dump()}


@app.get("/api/v1/agents")
def list_agents() -> list[dict[str, Any]]:
    return [
        {"agent_id": agent_id, **spec.model_dump()}
        for agent_id, spec in store.list_agents()
    ]


@app.post("/api/v1/agents/{agent_id}/runs", status_code=201)
def create_run(agent_id: str, request: CreateRunRequest) -> dict[str, Any]:
    spec = store.get_agent(agent_id)
    if spec is None:
        raise HTTPException(status_code=404, detail="agent not found")
    run_id = f"run-{uuid.uuid4().hex[:12]}"
    run = AgentRunState(
        run_id=run_id,
        agent_id=agent_id,
        input=request.input,
        status=RunStatus.CREATED,
        context=build_context(request.input),
        memory={"resources": teaching_resources()},
    )
    store.save_run(run)
    return run.model_dump()


@app.post("/api/v1/runs/{run_id}/execute")
def execute_run(run_id: str) -> dict[str, Any]:
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    spec = store.get_agent(run.agent_id)
    if spec is None:
        raise HTTPException(status_code=409, detail="agent definition not found")
    if run.status in {
        RunStatus.COMPLETED,
        RunStatus.FAILED,
        RunStatus.PAUSED,
        RunStatus.WAITING_USER,
    }:
        return run.model_dump()
    brain = build_brain(spec, run=run)
    result = AgentInterpreter(brain).run(run, spec)
    store.save_run(result)
    return result.model_dump()


@app.get("/api/v1/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return run.model_dump()


def _get_stored_run(run_id: str) -> AgentRunState:
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return run


@app.post("/api/v1/runs/{run_id}/steps")
def step_run(run_id: str) -> dict[str, Any]:
    run = _get_stored_run(run_id)
    if run.status == RunStatus.PAUSED:
        raise HTTPException(status_code=409, detail="run is paused; resume it first")
    if run.status == RunStatus.WAITING_USER:
        raise HTTPException(status_code=409, detail="run is waiting for user input")
    if run.status in {RunStatus.COMPLETED, RunStatus.FAILED}:
        return run.model_dump()
    spec = store.get_agent(run.agent_id)
    if spec is None:
        raise HTTPException(status_code=409, detail="agent definition not found")
    brain = build_brain(spec, run=run)
    result = AgentInterpreter(brain).step(run, spec)
    store.save_run(result)
    return result.model_dump()


@app.post("/api/v1/runs/{run_id}/pause")
def pause_run(run_id: str) -> dict[str, Any]:
    run = _get_stored_run(run_id)
    if run.status in {
        RunStatus.COMPLETED,
        RunStatus.FAILED,
        RunStatus.WAITING_USER,
    }:
        raise HTTPException(status_code=409, detail=f"cannot pause a {run.status.value} run")
    if run.status != RunStatus.PAUSED:
        run.status = RunStatus.PAUSED
        sink = EventSink(run_id=run.run_id, step=run.step)
        run.events.append(sink.emit("run.paused", "agent", {"step": run.step}))
        store.save_run(run)
    return run.model_dump()


def _waiting_file_catalog(run: AgentRunState) -> list[dict[str, str]]:
    """Read the short-lived, Observation-grounded catalog for an ask_user run."""
    schema = run.waiting_response_schema
    if not isinstance(schema, dict) or schema.get("type") != "file_selection":
        return []
    raw_catalog = run.memory.get("waiting_file_catalog", [])
    if not isinstance(raw_catalog, list):
        return []
    catalog: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw_catalog:
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


def _resolve_file_selection(
    run: AgentRunState,
    user_input: str,
    selection_id: str | None,
    selection_ids: list[str] | None,
) -> dict[str, Any] | None:
    """Resolve one or more explicit choices without guessing file identity."""
    catalog = _waiting_file_catalog(run)
    if not catalog:
        return None
    schema = run.waiting_response_schema if isinstance(run.waiting_response_schema, dict) else {}
    mode = schema.get("selection_mode", "single")
    min_count = int(schema.get("min_selections", 1))
    max_count = int(schema.get("max_selections", 1 if mode == "single" else len(catalog)))
    explicit_ids = list(selection_ids or ([] if not selection_id else [selection_id]))
    if selection_id and selection_ids:
        raise HTTPException(status_code=400, detail="use selection_id or selection_ids, not both")
    selected: list[dict[str, str]] = []
    source = ""
    if explicit_ids:
        if len(set(explicit_ids)) != len(explicit_ids):
            raise HTTPException(status_code=400, detail="selected files must be unique")
        selected = [next((item for item in catalog if item["file_id"] == file_id), None) for file_id in explicit_ids]
        if any(item is None for item in selected):
            raise HTTPException(status_code=400, detail="selected file is not in the current search Observation")
        source = "choice_buttons" if len(selected) > 1 else "choice_button"
    else:
        if not schema.get("allow_custom_input", True):
            raise HTTPException(status_code=400, detail="this file selection requires choosing one of the offered options")
        tokens = [item.strip() for item in re.split(r"[,，、;；\n]+", user_input) if item.strip()]
        if not tokens:
            return {"status": "unmatched", "raw_input": user_input}
        for token in tokens:
            normalized = token.casefold()
            exact = next((item for item in catalog if normalized in {item["file_id"].casefold(), item["name"].casefold(), f"{item.get('directory', '').rstrip('/')}/{item['name']}".casefold()}), None)
            partial_matches = [] if exact else [item for item in catalog if normalized in item["name"].casefold()]
            if exact:
                selected.append(exact)
            elif len(partial_matches) == 1:
                selected.append(partial_matches[0])
            elif len(partial_matches) > 1:
                return {"status": "ambiguous", "raw_input": user_input, "candidates": partial_matches[:5]}
            else:
                return {"status": "unmatched", "raw_input": user_input}
        if len({item["file_id"] for item in selected}) != len(selected):
            return {"status": "ambiguous", "raw_input": user_input, "candidates": selected}
        source = "custom_names" if len(selected) > 1 else "exact_name"
    if (mode == "single" and len(selected) != 1) or not min_count <= len(selected) <= max_count:
        raise HTTPException(status_code=400, detail=f"this {mode} file selection requires {min_count}-{max_count} files")
    selected = [item for item in selected if item is not None]
    return {
        "status": "resolved",
        "source": source,
        "raw_input": user_input or ", ".join(item["name"] for item in selected),
        "file_id": selected[0]["file_id"], "file_ids": [item["file_id"] for item in selected],
        "name": selected[0]["name"], "names": [item["name"] for item in selected],
        "directory": selected[0].get("directory", ""), "selection_mode": mode,
    }


@app.post("/api/v1/runs/{run_id}/resume")
def resume_run(
    run_id: str,
    request: ResumeRunRequest | None = None,
) -> dict[str, Any]:
    run = _get_stored_run(run_id)
    if run.status not in {RunStatus.PAUSED, RunStatus.WAITING_USER}:
        raise HTTPException(
            status_code=409,
            detail=f"run is not resumable from {run.status.value}",
        )
    previous_status = run.status
    if run.status == RunStatus.WAITING_USER:
        if request is None or (not request.input and not request.selection_id and not request.selection_ids):
            raise HTTPException(status_code=400, detail="input, selection_id, or selection_ids is required to answer the waiting request")
        if "pending_tool_calls" not in run.memory:
            answered_question = run.waiting_question or run.memory.get("last_asked_user_question")
            user_input = (request.input or "").strip()
            selection = _resolve_file_selection(run, user_input, request.selection_id, request.selection_ids)
            if selection and selection["status"] == "resolved" and not user_input:
                user_input = ", ".join(selection.get("names", [selection["name"]]))
            run.context.append({"role": "user", "content": user_input})
            if isinstance(answered_question, str) and answered_question.strip():
                # This is a factual state marker, not a next-action command:
                # it gives small models an explicit link between their
                # question and the newly appended user response.
                run.memory["last_answered_user_question"] = answered_question
                run.memory["last_user_response"] = user_input
                run.context.append({
                    "role": "system",
                    "content": (
                        "HARNESS_USER_RESPONSE_STATE\n"
                        f"ANSWERED_QUESTION: {answered_question}\n"
                        f"LATEST_USER_RESPONSE: {user_input}\n"
                        "The user has already answered this question. Reuse the response and current Observation; do not repeat the identical ask_user question."
                    ),
                })
            if selection:
                run.memory["user_file_selection"] = selection
                run.memory["file_read_authorization"] = {
                    "status": selection["status"],
                    "allowed_file_ids": selection.get("file_ids", []) if selection["status"] == "resolved" else [],
                }
                if selection["status"] == "resolved":
                    authorized_ids = ", ".join(selection.get("file_ids", []))
                    authorized_names = ", ".join(selection.get("names", []))
                    selection_state = (
                        "USER_SELECTION_STATE\nstatus: resolved\n"
                        f"source: {selection['source']}\nselection_mode: {selection.get('selection_mode', 'single')}\n"
                        f"selected_names: {authorized_names}\nauthorized_file_ids: {authorized_ids}\n"
                        "These file_ids are verified from a successful search_files Observation. If read_file is needed, use only these exact values in args.file_id; target is not a tool argument."
                    )
                elif selection["status"] == "ambiguous":
                    candidates = ", ".join(item["name"] for item in selection["candidates"])
                    selection_state = (
                        "USER_SELECTION_STATE\nstatus: ambiguous\n"
                        f"raw_input: {selection['raw_input']}\npossible_matches: {candidates}\n"
                        "No file has been selected. Do not call read_file until the user resolves this ambiguity."
                    )
                else:
                    selection_state = (
                        "USER_SELECTION_STATE\nstatus: unmatched\n"
                        f"raw_input: {selection['raw_input']}\n"
                        "No searched file matches this response. Do not call read_file using an invented file_id; ask a new, specific clarification if needed."
                    )
                run.context.append({"role": "system", "content": selection_state})
            run.events.append(EventSink(run_id=run.run_id, step=run.step).emit("user.input", "user", {"input": user_input, "selection": selection}))
            run.waiting_question, run.waiting_choices, run.waiting_response_schema = None, [], None
            run.memory.pop("waiting_file_catalog", None)
    run.status = RunStatus.RUNNING
    run.events.append(
        EventSink(run_id=run.run_id, step=run.step).emit(
            "run.resumed",
            "agent",
            {"from_status": previous_status.value},
        )
    )
    # Publish the transition before the next model request so an existing SSE
    # subscriber can observe user.input/run.resumed immediately.
    store.save_run(run)
    spec = store.get_agent(run.agent_id)
    if spec is None:
        raise HTTPException(status_code=409, detail="agent definition not found")
    interpreter = AgentInterpreter(build_brain(spec, run=run))
    if "pending_tool_calls" in run.memory:
        result = interpreter.resolve_confirmation(run, spec, request.input if request and request.input else "")
        result = interpreter.run(result, spec) if result.status == RunStatus.RUNNING else result
    else:
        result = interpreter.run(run, spec)
    store.save_run(result)
    return result.model_dump()


@app.post("/api/v1/cases/{case_id}/runs", status_code=201)
def create_case_run(
    case_id: str,
    request: CreateRunRequest | None = None,
) -> dict[str, Any]:
    case = find_teaching_case(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="teaching case not found")
    for name in case["tools"]:
        if name not in available_tool_names():
            raise HTTPException(status_code=500, detail=f"case has unknown tool: {name}")
    spec = MiniAgentSpec(
        name=case["title"],
        description=case["description"],
        brain=BrainSpec(type=case["brain_type"], model=case["brain_model"]),
        tools=[default_tool(name) for name in case["tools"]],
    )
    agent_id = f"agent-{uuid.uuid4().hex[:12]}"
    store.save_agent(agent_id, spec)
    input_text = request.input if request and request.input else case["input"]
    run_id = f"run-{uuid.uuid4().hex[:12]}"
    run = AgentRunState(
        run_id=run_id,
        agent_id=agent_id,
        input=input_text,
        status=RunStatus.CREATED,
        context=build_context(input_text),
        memory={"resources": teaching_resources()},
    )
    store.save_run(run)
    return {
        "case_id": case_id,
        "agent": {"agent_id": agent_id, **spec.model_dump()},
        "run": run.model_dump(),
    }


@app.post("/api/v1/workflows/runs", status_code=201)
def create_workflow_run(request: CreateWorkflowRunRequest) -> dict[str, Any]:
    """Validate an authored graph, persist it with an Agent, and create a Run."""
    workflow = request.workflow
    try:
        validate_workflow(workflow)
    except WorkflowValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    agent_nodes = [node for node in workflow.nodes if node.type == "agent"]
    first_agent = agent_nodes[0]
    brain_type = first_agent.config.get("brain_type", "test")
    for node in agent_nodes:
        configured_type = node.config.get("brain_type", brain_type)
        if configured_type not in {"test", "tiny_llm", "clarification"}:
            raise HTTPException(status_code=400, detail=f"unsupported agent brain_type: {configured_type}")
        protocol = node.config.get("tool_protocol", "json_action")
        if protocol not in {"json_action", "native_tools"}:
            raise HTTPException(status_code=400, detail=f"unsupported tool protocol: {protocol}")
        device = node.config.get("device", "cpu")
        if device not in {"cpu", "gpu"}:
            raise HTTPException(status_code=400, detail=f"unsupported local execution device: {device}")
    tool_nodes = [node for node in workflow.nodes if node.type == "tool"]
    tool_names = [node.config.get("tool", "") for node in tool_nodes]
    tool_names.append("final_answer")
    for name in tool_names:
        if name not in available_tool_names():
            raise HTTPException(status_code=400, detail=f"unknown workflow tool: {name}")
    loop_module = next((node for node in workflow.nodes if node.type == "react_loop"), None)
    context_module = next(node for node in workflow.nodes if node.type == "input")
    try:
        context_window_size = int(context_module.config.get("context_window_size", "8192"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Context module has invalid context window size") from exc
    if not 256 <= context_window_size <= 131072:
        raise HTTPException(status_code=400, detail="Context module window size must be between 256 and 131072")
    effective_max_steps = int(loop_module.config.get("max_steps", workflow.max_steps)) if loop_module else workflow.max_steps
    effective_max_decision_failures = int(loop_module.config.get("max_decision_failures", 2)) if loop_module else 2
    effective_max_tool_failures = int(loop_module.config.get("max_tool_failures", 2)) if loop_module else 2
    workspace_modules = [node for node in workflow.nodes if node.type == "workspace"]
    if len(workspace_modules) > 1:
        raise HTTPException(status_code=400, detail="workflow can contain at most one teaching workspace module")
    workspace_root = ""
    if workspace_modules:
        configured_root = workspace_modules[0].config.get("root_path", "").strip()
        if configured_root:
            root = Path(configured_root).expanduser()
            try:
                workspace_root = str(root.resolve(strict=True))
            except OSError as exc:
                raise HTTPException(status_code=400, detail=f"teaching workspace directory does not exist: {configured_root}") from exc
            if not Path(workspace_root).is_dir():
                raise HTTPException(status_code=400, detail=f"teaching workspace is not a directory: {configured_root}")

    tool_specs: list[ToolSpec] = []
    for node in tool_nodes:
        base = default_tool(node.config["tool"])
        risk = node.config.get("risk", base.risk)
        if risk not in {"read", "write", "sensitive"}:
            raise HTTPException(status_code=400, detail=f"unsupported tool risk: {risk}")
        permission = node.config.get("permission", base.permission)
        if permission not in {"granted", "blocked"}:
            raise HTTPException(status_code=400, detail=f"unsupported tool permission: {permission}")
        tool_specs.append(base.model_copy(update={
            "risk": risk,
            "permission": permission,
            "requires_confirmation": node.config.get("requires_confirmation", str(base.requires_confirmation).lower()) == "true",
            "parallel_safe": node.config.get("parallel_safe", str(base.parallel_safe).lower()) == "true",
        }))
    tool_specs.append(default_tool("final_answer"))
    spec = MiniAgentSpec(
        name=workflow.name,
        description="Compiled from an AgentScratch visual workflow.",
        brain=BrainSpec(
            type=brain_type,
            model=first_agent.config.get("model") or None,
            device=first_agent.config.get("device", "cpu"),
            tool_protocol=first_agent.config.get("tool_protocol", "json_action"),
            temperature=first_agent.config.get("temperature", 0.2),
            top_p=first_agent.config.get("top_p", 0.9),
            top_k=first_agent.config.get("top_k", 40),
            max_tokens=first_agent.config.get("max_tokens", 512),
            repeat_penalty=first_agent.config.get("repeat_penalty", 1.1),
            max_attempts=first_agent.config.get("max_attempts", 2),
        ),
        tools=list({tool.name: tool for tool in tool_specs}.values()),
        policy=AgentPolicy(
            max_steps=effective_max_steps,
            max_decision_failures=effective_max_decision_failures,
            max_tool_failures=effective_max_tool_failures,
        ),
        workflow=workflow.model_dump(by_alias=True),
    )
    input_node = next(node for node in workflow.nodes if node.type == "input")
    configured_input_id = input_node.config.get("user_input_module_id", "").strip()
    user_input_module = next(
        (node for node in workflow.nodes if node.type == "user_input" and node.id == configured_input_id),
        None,
    )
    user_input_module = user_input_module or next(
        (node for node in workflow.nodes if node.type == "user_input" and node.parent_id == input_node.id),
        None,
    )
    user_input_module = user_input_module or next(
        (node for node in workflow.nodes if node.type == "user_input" and not node.parent_id),
        None,
    )
    # ``input.config.text`` remains a compatibility fallback for older saved
    # canvases.  New canvases model the user message as its own Context child.
    input_text = request.input or (user_input_module.config.get("text", "").strip() if user_input_module else input_node.config.get("text", "").strip())
    if not input_text:
        raise HTTPException(status_code=400, detail="workflow input node has no input text")

    agent_id = f"agent-{uuid.uuid4().hex[:12]}"
    store.save_agent(agent_id, spec)
    run_id = f"run-{uuid.uuid4().hex[:12]}"
    run = AgentRunState(
        run_id=run_id,
        agent_id=agent_id,
        input=input_text,
        status=RunStatus.CREATED,
        # A visual workflow owns its author-visible Context.  Do not mix in
        # the compatibility endpoint's English default prompt here: its
        # system-prompt modules are the single source of authored policy.
        context=[{"role": "user", "content": input_text}],
        memory={
            "resources": teaching_resources(),
            **({"workspace": {"root_path": workspace_root}} if workspace_root else {}),
        },
    )
    system_messages = [
        {"role": "system", "content": module.config.get("content", "").strip()}
        for module in workflow.nodes
        if module.type == "system_prompt" and module.parent_id == input_node.id and module.config.get("content", "").strip()
    ]
    if workspace_root:
        workspace_index = teaching_workspace_index(workspace_root)
        system_messages.append({
            "role": "system",
            "content": (
                "TEACHING_WORKSPACE_INDEX=" + json.dumps(workspace_index, ensure_ascii=False) + "\n"
                "This is metadata only, not file content. For search_files, omit directory or use '.' for the root; "
                "use a listed relative directory when needed. A file may only be read using the exact file_id returned by a successful search_files Observation."
            ),
        })
    run.context = system_messages + run.context
    store.save_run(run)
    return {
        "workflow": workflow.model_dump(by_alias=True),
        "agent": {"agent_id": agent_id, **spec.model_dump()},
        "run": run.model_dump(),
    }


@app.get("/api/v1/runs/{run_id}/events")
def get_events(run_id: str, after: int = 0) -> list[dict[str, Any]]:
    _get_stored_run(run_id)
    return [event.model_dump() for event in store.list_events(run_id, after=after)]


@app.get("/api/v1/runs/{run_id}/replay")
def get_replay(run_id: str) -> dict[str, Any]:
    run = _get_stored_run(run_id)
    events = store.list_events(run_id)
    return {
        "run": run.model_dump(mode="json", exclude={"events"}),
        "events": [event.model_dump(mode="json") for event in events],
        "event_count": len(events),
        "step_count": run.step,
    }


def _sse_event(event: AgentEvent) -> str:
    payload = json.dumps(event.model_dump(mode="json"), ensure_ascii=False)
    return f"event: {event.type}\nid: {event.event_id}\ndata: {payload}\n\n"


@app.get("/api/v1/runs/{run_id}/events/stream")
def stream_events(run_id: str, after: int = 0, wait_seconds: int = 30) -> StreamingResponse:
    _get_stored_run(run_id)
    cursor = max(0, after)
    deadline = time.monotonic() + min(max(1, wait_seconds), 30)

    def generate() -> Iterator[str]:
        nonlocal cursor
        while True:
            events = store.list_events(run_id, after=cursor)
            if events:
                for event in events:
                    yield _sse_event(event)
                cursor += len(events)

            run = store.get_run(run_id)
            if run is None:
                break
            if run.status in {
                RunStatus.COMPLETED,
                RunStatus.FAILED,
                RunStatus.PAUSED,
                RunStatus.WAITING_USER,
            }:
                break
            if time.monotonic() >= deadline:
                break
            time.sleep(0.2)
        yield "event: stream.end\ndata: {}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

