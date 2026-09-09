# M2：本地真实 LLM 接入说明

AgentScratch 的教学主线要求使用真实 Transformer LLM，而不是规则 Mock。M2 通过 OpenAI-compatible API 接入本地模型服务，当前优先支持：

- llama.cpp `llama-server`
- Ollama
- vLLM
- LM Studio
- 其他 OpenAI-compatible 本地推理服务

## 配置

后端默认读取以下环境变量：

```bash
AGENTSCRATCH_LLM_BASE_URL=http://127.0.0.1:8080/v1
AGENTSCRATCH_LLM_MODEL=qwen-tiny
AGENTSCRATCH_LLM_API_KEY=local
AGENTSCRATCH_LLM_TIMEOUT=60
```

也可以在创建 Agent 时指定模型：

```json
{
  "name": "Tiny LLM Agent",
  "brain_type": "tiny_llm",
  "brain_model": "qwen2.5-0.5b",
  "tools": ["calculator", "final_answer"]
}
```

## llama.cpp 启动示例

仓库已经包含一个可用于验收的 Qwen2.5-0.5B-Instruct Q4_K_M GGUF 和 Windows CPU 版
`llama-server`。在 `agent-scratch` 目录执行：

```powershell
.\bin\llama-cpu\llama-server.exe `
  -m .\models\qwen2.5-0.5b-instruct-q4_k_m.gguf `
  --alias qwen-tiny `
  --host 127.0.0.1 `
  --port 8080 `
  --ctx-size 2048
```

Linux/macOS 只需将可执行文件和路径替换为本机安装位置：

```bash
llama-server \
  -m models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --alias qwen-tiny \
  --host 127.0.0.1 \
  --port 8080 \
  --ctx-size 2048
```

启动后检查：

```bash
curl http://127.0.0.1:8080/v1/models
```

AgentScratch 后端状态接口：

```bash
curl http://127.0.0.1:8000/api/v1/llm/status
```

## Ollama 启动示例

```bash
ollama pull qwen2.5:0.5b
ollama serve
```

配置：

```bash
AGENTSCRATCH_LLM_BASE_URL=http://127.0.0.1:11434/v1
AGENTSCRATCH_LLM_MODEL=qwen2.5:0.5b
```

## 决策协议

本地模型每次只输出一个 JSON Action：

```json
{
  "thought": "用户需要计算 12 * 7，应调用 calculator。",
  "action": "call_tool",
  "tool": "calculator",
  "args": {
    "expression": "12 * 7"
  },
  "answer": null,
  "done": false
}
```

系统会记录：

- 原始模型输出 `raw_output`
- 解析后的 Action
- 模型名
- 调用耗时
- token usage
- 重试次数
- JSON 解析错误
- 本地服务不可用错误

这些错误不会被隐藏，而是进入事件时间线，作为教学素材展示。

## 当前边界

M2 已完成：

- OpenAI-compatible 本地模型接入（llama.cpp、Ollama、vLLM、LM Studio）
- 可复用 Prompt Builder 和 JSON Action 解析
- fenced JSON、周围 prose、嵌套 `args` 对象的解析
- JSON/schema 错误和模型连接错误重试
- 原始输出、模型名、耗时、token usage、尝试次数事件记录
- Agent 级 `brain_model` 覆盖环境默认模型
- 仓库内 Qwen2.5-0.5B GGUF 的真实端到端验收

如果本地模型服务未启动，状态接口会明确返回不可用：

```text
GET /api/v1/llm/status
```

会返回：

```json
{
  "available": false
}
```

创建 `tiny_llm` Agent 并运行时，事件流会显示明确的本地模型服务不可用错误。

M2 之后的工作属于运行时/产品层：WebSocket 或 SSE 事件流、单步控制、回放，以及
Agent/Run 的 SQLite 持久化。
