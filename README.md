# aiayn-agent-canvas

面向初学者的「Agent 界的 Scratch」：用可视化积木与真实本地小模型，帮助用户理解 LLM Agent 的运行原理。

## 项目文档

- [项目立项记录](docs/PROJECT_CHARTER.md)
- [第一阶段计划](docs/PHASE1_PLAN.md)
- [M2 本地 LLM 接入说明](docs/M2_LOCAL_LLM.md)
- [当前进度](docs/PROGRESS.md)

## 技术路线

- 前端：React + TypeScript + Vite
- 后端：FastAPI + Python
- 模型：本地 Qwen 小模型 + llama.cpp / GGUF / Ollama
- 核心：Agent DSL + Agent Interpreter + Tool Runtime + Event Stream

当前编排器采用模块化 ReAct 编排：Context 模块可包含独立的 User Input、系统提示词与工具定义模块，Tool 模块独立注册为能力，Agent 模块勾选其可用工具后由模型自主选择是否调用。Agent 的工具结果会作为 Observation 写回下一轮 Context；图级最大步数默认 10、最高 60，用于保护教学演示中的循环。循环还单独限制模型决策错误与工具执行错误，用户拒绝敏感操作只作为策略拦截记录。条件分支与多 Agent 连线仍用于表达作者设定的编排边界。

Agent 模块可选择两种工具调用协议：`JSON Action` 是为本地小模型准备的兼容教学 DSL；`Native Tool Calling` 使用 OpenAI-compatible 的顶层 `tools`、模型 `tool_calls` 与 `tool_call_id` 结果关联。每次运行都会记录实际请求快照，方便比较两种协议。模型可在一轮提出多个工具调用；解释器按工具模块的执行权限、风险等级、确认要求和并行安全属性执行，并将每条 Observation 独立写回上下文。

## 快速开始

### 后端

```bash
python -m pip install -e ..\..\aiayn-agent-core\server
cd server
python -m uvicorn aiayn_agent_canvas.main:app --reload --port 8000
```

### 前端

```bash
cd apps/web
npm install
npm run dev
```

### 测试

```bash
python -m pip install -e ..\..\aiayn-agent-core\server
cd server
python -m pytest -q
```

## 当前状态

M0 / M1 已完成基础工程闭环；M2 已完成本地真实 LLM 接入协议、Prompt Builder、JSON Action 解析、重试和事件可观测性。本地 Qwen2.5-0.5B-Instruct Q4_K_M 模型已接入并完成真实端到端运行。运行控制层现已支持 SQLite 持久化、SSE 事件流、单步执行、回放、暂停/继续、用户输入续跑和教学案例模板；前端已增加 ReAct 执行图、节点详情，以及可拖入节点、移动节点、连线、编辑属性、自动保存和导出 JSON 的 Agent 编排编辑器。编辑器现可运行条件分支、多 Agent 路由与受最大步数保护的循环连线。下一步是补充浏览器端到端测试与 Agent/Run 管理。
