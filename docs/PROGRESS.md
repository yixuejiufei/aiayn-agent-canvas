# AgentScratch M0 / M1 / M2 progress

## Backend

- FastAPI application initialized.
- Pydantic schemas for Agent, Run, Action, Event, Tool, and Brain.
- Agent interpreter with context building, action validation, tool execution, event emission, and max-step protection.
- Five local sandbox tools: calculator, mock_search, memory_store, memory_lookup, final_answer.
- API endpoints for health, tools, agents, runs, execution, and run lookup.

## M2 local LLM

- Added `TinyLLMBrain` using an OpenAI-compatible local inference endpoint.
- Supports llama.cpp, Ollama, vLLM, and LM Studio.
- Environment configuration:
  - `AGENTSCRATCH_LLM_BASE_URL`
  - `AGENTSCRATCH_LLM_MODEL`
  - `AGENTSCRATCH_LLM_API_KEY`
  - `AGENTSCRATCH_LLM_TIMEOUT`
- Agent creation supports `brain_model`.
- JSON Action parsing tolerates fenced JSON and surrounding prose.
- Invalid JSON and schema failures can be retried.
- Event timeline records:
  - raw model output
  - parsed action
  - model name
  - duration
  - token usage
  - retry count
  - connection and parsing errors
- `GET /api/v1/llm/status` reports local model availability and model list.
- Local model service is now installed and verified with Qwen2.5-0.5B-Instruct Q4_K_M. A real end-to-end run completed successfully and returned the correct arithmetic result.
- SQLite persistence now stores Agent definitions, Run state, and deduplicated Event timelines.
- Run controls now support one-cycle stepping, persisted event queries, replay payloads, and SSE streaming.
- Pause/resume transitions and ask_user continuation are persisted before the next model cycle.
- Built-in and user-created teaching case templates can be listed, inspected, and used to create runs.
- `context.build` events now persist immutable per-cycle context snapshots instead of later mutated state.

## Frontend

- React + TypeScript + Vite scaffold created.
- Health-check and local LLM status UI wired to the backend.
- Runtime workbench UI wired to Agent creation, run/step controls, replay, and SSE Event Timeline.
- UI wired to case selection, custom case template creation, pause/resume, and ask_user input continuation.
- ReAct execution canvas now renders the real event sequence and supports dragging nodes to rearrange the layout.
- Node inspector now shows what the selected node does, its current content, and related events.
- Agent orchestration editor now supports a node palette, drag/drop placement, node movement, visual connections, property editing, deletion, auto-layout, local draft persistence, and workflow JSON export.
- Authored workflows can now be submitted to `POST /api/v1/workflows/runs`; the graph is compiled into a resumable Workflow Brain and uses the existing Run/Event/SSE lifecycle.
- Workflow runs support input, Agent/Brain, Tool, Condition, Ask user, and Final answer nodes, including persisted cursor recovery across step and resume requests.
- Condition nodes evaluate input, latest user input, or latest tool result with equality, containment, numeric, truthy, and empty checks; each requires explicit `true` and `false` edges.
- Each Agent node now has its own Brain configuration. A model can emit a constrained `route` action only to an authored outgoing Agent node; the editor also exposes a deterministic demo route for teaching flows.
- Backward edges are valid workflow loops. The editor exposes a graph-level maximum-step budget (default 10, configurable from 1 to 60), and the existing interpreter reports a clear failure when the budget is exhausted.
- The editor now uses module-oriented ReAct semantics: a Context module visibly contains System Prompt and Tool Definition modules; independent Tool modules are registered capabilities rather than mandatory flow steps; Agent modules select the capabilities injected into their model context.
- The default teaching loop budget is 10 steps and can be configured up to 60. Runtime visualization groups each cycle as `Context → Think → Act → Observation` and highlights the Observation-to-next-Context return edge.
- Agent modules now support `JSON Action` compatibility mode and `Native Tool Calling` mode. Native mode sends top-level API `tools`, preserves assistant `tool_calls`, and returns tool results with their matching `tool_call_id`.
- Each cycle persists a `context.request` snapshot containing the actual messages, top-level tool definitions, and selected protocol. The workflow appends a compact Agent status message with step budget, tool observations, and failures at the end of the request context.
- A single Agent action can now carry multiple tool calls. Independent tools marked `parallel_safe` execute in parallel and their individual starts, results, and Observation messages remain visible in the event timeline.
- Tool modules now expose an explicit execution permission, `read` / `write` / `sensitive` risk levels, a parallel-safety switch, and a mandatory-user-confirmation switch. The interpreter—not the model—enforces permission and the checkpoint, persists it, and can continue after an explicit approval or rejection.
- ReAct Loop modules now configure three separate harness budgets: maximum cycles, model decision failures (transport/protocol/schema/capability validation), and tool execution failures. A user rejection is recorded as a policy block and consumes neither failure budget; a failed decision adds a concise harness recovery message to the next model context.
- The local-model comparison now uses an event-derived funnel: model tool plan → harness validation → execution start → successful Observation. It reports proposal-validation, plan-to-success and execution-success rates, while decision errors, tool failures and policy blocks remain independently visible. The statistics window can still be reset without deleting replay history.
- `npm install` completed.
- `npm run build` passes.

## Tests

- Backend: 51 tests pass.
- Frontend TypeScript build passes.
- Step, replay, SSE, and SQLite reload contracts verified.
- Pause/resume, ask_user continuation, and custom case persistence contracts verified.
- API run with no local model service fails with an explicit `llm.error` event instead of silently using rules.
- Real local Qwen2.5-0.5B run verified: `calculator(12 * 7)` followed by final answer `84`.
- Browser smoke test verified the execution canvas and node inspector on a two-cycle calculator run.
- Frontend build verified the orchestration editor bundle.
- Live API smoke test verified `input → agent → calculator → final` returns `84` and carries the Agent node prompt into context.
- Automated workflow contracts verify condition branching with a loop edge and routing from one Agent node to another.
- A module-workflow contract verifies that an Agent receives a selected Tool module as context, chooses `call_tool` itself, and resumes the same ReAct loop after the tool result.
- Automated contracts cover native multiple `tool_calls`, parallel execution of independent calls, and persisted high-risk-tool confirmation followed by a resumed run.
- The deterministic test Brain also accepts `并行演示` when calculator and mock_search are available, so learners can reliably trigger the parallel visualization without depending on a local model's tool-choice behavior.
- User Input is now a first-class, draggable Context child. New canvases no longer store it inside Context itself (old documents still have a compatibility fallback); subsequent Agent messages and Observations accumulate in the same runtime Context.
- Runtime Context cards now show actual message count, per-cycle growth, and a growing height. Their outgoing edge explicitly states that assembled Context is entering LLM reasoning.
- The editor has two one-click teaching cases: direct answer (no Tool event) and calculator call (Tool → Observation). Workflow execution locks the same authored canvas into a live event-driven presentation mode until the learner exits it.
- Live API smoke test verified a `memory_store` write pauses in `waiting_user`, emits `agent.confirmation_required`, and resumes to completion after `确认`.
- Canvas live/demo mode now shows a terminal run-state badge plus event-timeline elapsed time. Its replay cursor accumulates (or rewinds) decisions, planned/validated/started tool calls, Observations, decision/tool errors, and policy blocks from the exact visible events.
- The visual node historically stored as `agent` is now presented as an LLM module: it configures inference engine, model, prompt, protocol and available tools, while the surrounding ReAct loop is the Agent execution mechanism. Model metrics use the server-reported identity and reconcile decision errors to it; a selected GGUF is restored before a workflow run rather than silently falling back to an external model.
- Decision failures now carry a Harness-observable subtype (`connection`, `protocol`, `action_validation`, `route`, `capability`, `workflow`, or `other`). The model comparison table and replay panel show these separately; historic event logs use a compatible text-based classification when no subtype was persisted.
- ReAct safety budgets for decision failures and tool execution failures are always visible and configurable beside the top-level run controls. During live execution or replay, their counters display as `current / configured limit`, and a failed run renders its concrete terminal error beside the timeline controls.

## Next

- Add persisted Agent/Run listing UI and richer drag-and-drop teaching case authoring.
- Add end-to-end browser tests for the local teaching flow.
- Add production-oriented concurrency and lifecycle handling for long-running runs.

## P2-A：小模型工具调用训练准备

- 数据按完整轨迹切分为训练 / 验证 / 保留评测，避免同一任务的相邻回合泄漏到评测集。
- 新增离线评测器，分别统计 JSON 解析、行动类型、工具、参数与整条决策准确率，并按任务族群输出结果。
- 当前尚未执行 GPU 微调；下一步先以保留集测量不同 Instruct / Tool-use 模型的基线，再将 Harness 已验证的真实运行轨迹扩入训练集。

