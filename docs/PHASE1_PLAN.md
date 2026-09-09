# AgentScratch 第一阶段计划：核心教学闭环 MVP

> 阶段目标：验证「真实本地 LLM + 可视化积木 + 单步执行 + 事件回放」能否帮助初学者真正理解 Agent 的工作原理。  
> 阶段原则：先做小而完整的教学闭环，不做通用 Agent 平台。

## 一、阶段范围

### 必须包含

1. 一个最小可用的积木式 Agent 编辑器。
2. 一个真实 Agent Loop 解释器。
3. 一个本地 Transformer LLM Brain。
4. 一个受约束的 Action JSON 协议。
5. 一组本地沙盒工具。
6. 事件流驱动的执行可视化。
7. 单步执行、慢速执行、完整回放。
8. 上下文面板，展示模型实际看到的输入。
9. 工具调用面板，展示参数、结果、耗时和错误。
10. 至少 5 个教学课程案例。

### 暂不包含

1. 多 Agent 协作。
2. 复杂审批流程。
3. 教师后台。
4. 学生账号系统。
5. 云端服务。
6. 多模型路由。
7. 正式微调模型。
8. 完整商业化授权。
9. 移动端。
10. 高级 Scratch 扩展生态。

## 二、核心技术目标

### 目标 1：真实 Agent Loop

学生能清楚看到以下循环：

```text
用户输入
   ↓
构造上下文
   ↓
LLM 生成下一步动作
   ↓
解析动作
   ↓
调用工具
   ↓
工具结果写回上下文
   ↓
再次调用 LLM
   ↓
循环直到输出最终答案
```

### 目标 2：真实模型决策

模型必须自主完成：

1. 判断当前任务状态。
2. 选择工具。
3. 填写参数。
4. 判断是否结束。

不允许用规则系统替模型决策。

### 目标 3：强格式约束

模型输出必须是合法 Action JSON，例如：

```json
{
  "thought": "用户需要计算 12 * 7，应该调用 calculator 工具。",
  "action": "call_tool",
  "tool": "calculator",
  "args": {
    "expression": "12 * 7"
  },
  "done": false
}
```

约束格式，不约束语义选择。

### 目标 4：完整可观测性

每个执行步骤都记录事件：

```text
run.start
context.build
llm.start
llm.end
action.validate
tool.start
tool.end
loop.next
agent.finish
agent.error
```

前端根据事件流驱动节点高亮、上下文变化、工具调用和回放。

## 三、推荐技术栈

### 前端

```text
React
TypeScript
Vite
Tailwind CSS
Blockly 或 React Flow
Zustand / Redux Toolkit
WebSocket / SSE
```

第一版建议：

- 如果追求 Scratch 体验：优先 Blockly。
- 如果追求流程图体验：优先 React Flow。
- MVP 可以先用 React Flow 快速验证教学语义，后续再补 Blockly 积木模式。

### 后端

```text
Python 3.11+
FastAPI
Pydantic v2
Uvicorn
SQLite
asyncio
```

### 本地模型

```text
llama.cpp / GGUF
Qwen2.5-0.5B 或 Qwen3-0.6B 起步
可选 Qwen 1.5B / 1.7B
```

模型接入层必须抽象，后续可替换：

```text
TinyLLMBrain
RemoteOpenAICompatibleBrain
FutureYiNengFactoryBrain
TestBrain
```

### 测试

```text
前端：Vitest + Playwright
后端：pytest + httpx
协议：JSON Schema contract tests
事件流：确定性回放测试
```

## 四、系统架构

```text
┌─────────────────────────────────────────┐
│ Web Frontend                            │
│ - Agent Canvas                          │
│ - Context Panel                         │
│ - Tool Call Panel                       │
│ - Event Timeline                        │
│ - Step Controls                         │
└────────────────┬────────────────────────┘
                 │ REST + WebSocket / SSE
┌────────────────▼────────────────────────┐
│ FastAPI Server                          │
│ - Project API                           │
│ - Run API                               │
│ - Event Stream API                      │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│ Agent Interpreter                       │
│ - State Machine                         │
│ - Agent Loop                            │
│ - Context Builder                       │
│ - Action Validator                      │
│ - Safety Guardrails                     │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│ TinyLLMBrain                            │
│ - Prompt Builder                        │
│ - llama.cpp / GGUF Runtime              │
│ - JSON / Grammar Constraint             │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│ Tool Runtime                            │
│ - calculator                            │
│ - mock_search                           │
│ - memory_store                          │
│ - memory_lookup                         │
│ - final_answer                          │
└─────────────────────────────────────────┘
```

## 五、核心数据协议

### Mini Agent Spec

前端画布导出的第一版 DSL：

```json
{
  "version": "0.1",
  "name": "简单工具调用 Agent",
  "brain": {
    "type": "tiny_llm",
    "model": "qwen-0.5b"
  },
  "tools": [
    {
      "name": "calculator",
      "description": "计算数学表达式",
      "parameters": {
        "expression": {
          "type": "string"
        }
      }
    }
  ],
  "policy": {
    "max_steps": 8,
    "temperature": 0.2
  }
}
```

### Agent Action

```json
{
  "thought": "string",
  "action": "call_tool | final_answer | ask_user",
  "tool": "string | null",
  "args": {
    "expression": "12 * 7"
  },
  "answer": "string | null",
  "done": false
}
```

### Agent Event

```json
{
  "event_id": "uuid",
  "run_id": "uuid",
  "step": 1,
  "type": "llm.end",
  "timestamp": 1780000000.123,
  "node": "llm_brain",
  "data": {}
}
```

## 六、产品界面规划

### 1. Agent Canvas

显示主 Agent 流程：

```text
Start
  ↓
LLM Brain
  ↓
Tool Runtime
  ↓
Final Answer
```

支持：

- 查看节点
- 配置工具
- 配置策略
- 运行状态高亮
- 不要求第一版自由拖拽所有复杂结构

### 2. Context Panel

展示模型实际看到的上下文：

```text
System Prompt
User Input
Tools Schema
Previous Thoughts
Tool Calls
Tool Results
Memory
```

重点：必须是模型真实输入，不做伪展示。

### 3. Action Panel

展示模型每一步输出：

```text
thought
action
tool
args
done
raw output
validation result
```

### 4. Tool Panel

展示每次工具调用：

```text
工具名
输入参数
返回结果
耗时
错误信息
重试状态
```

### 5. Event Timeline

展示完整事件：

```text
run.start
context.build
llm.start
llm.end
action.validate
tool.start
tool.end
agent.finish
```

支持：

- 单步执行
- 暂停
- 继续
- 慢速播放
- 回放
- 导出事件记录

### 6. Failure Panel

展示真实失败：

```text
JSON 解析失败
工具选择错误
参数错误
重复调用
超过最大步数
```

失败本身是教学素材，不应被隐藏。

## 七、本地工具设计

第一版只保留 5 个工具，全部本地沙盒执行。

### 1. calculator

```json
{
  "name": "calculator",
  "description": "计算数学表达式",
  "parameters": {
    "expression": {
      "type": "string"
    }
  }
}
```

### 2. mock_search

```json
{
  "name": "mock_search",
  "description": "在本地模拟知识库中搜索信息",
  "parameters": {
    "query": {
      "type": "string"
    }
  }
}
```

### 3. memory_store

```json
{
  "name": "memory_store",
  "description": "保存信息到 Agent 记忆",
  "parameters": {
    "key": {
      "type": "string"
    },
    "value": {
      "type": "string"
    }
  }
}
```

### 4. memory_lookup

```json
{
  "name": "memory_lookup",
  "description": "从 Agent 记忆读取信息",
  "parameters": {
    "key": {
      "type": "string"
    }
  }
}
```

### 5. final_answer

```json
{
  "name": "final_answer",
  "description": "输出最终答案",
  "parameters": {
    "answer": {
      "type": "string"
    }
  }
}
```

## 八、课程案例

第一版至少准备 5 个课程案例：

### 案例 1：第一次工具调用

任务：

```text
帮我计算 12 * 7
```

教学点：

- 用户输入
- 模型选择 calculator
- 参数填充
- 工具返回结果
- 模型生成最终答案

### 案例 2：记忆写入与读取

任务：

```text
记住我的名字是 Alice，然后告诉我我叫什么。
```

教学点：

- memory_store
- memory_lookup
- 多轮工具调用
- 上下文更新

### 案例 3：搜索后回答

任务：

```text
查一下 Agent 的四个核心组成部分。
```

教学点：

- mock_search
- 检索结果写入上下文
- 模型基于工具结果回答

### 案例 4：模型失败与恢复

任务：

```text
诱导模型输出错误工具或错误参数。
```

教学点：

- Action 校验失败
- 错误反馈
- 重试机制
- 最大步数保护

### 案例 5：完整 Agent 循环

任务：

```text
计算一个结果，保存到记忆，再基于记忆回答。
```

教学点：

- 多工具组合
- Agent Loop
- Context 增长
- 事件回放

## 九、开发里程碑

### M0：项目初始化

产出：

```text
agent-scratch/
├── apps/web/
├── server/
├── docs/
├── packages/
└── README.md
```

任务：

1. 初始化前端 Vite + React + TypeScript。
2. 初始化后端 FastAPI。
3. 建立 shared types。
4. 建立 CI 脚本。
5. 建立基础测试。
6. 建立项目 README。

验收标准：

```text
npm run dev 可启动前端
uvicorn 可启动后端
健康检查接口返回正常
前后端可以联通
```

### M1：协议与解释器

任务：

1. 定义 Mini Agent Spec。
2. 定义 Agent Action。
3. 定义 Agent Event。
4. 实现 Pydantic Schema。
5. 实现 Agent Loop 状态机。
6. 实现 Context Builder。
7. 实现 Action Validator。
8. 实现最大步数保护。
9. 实现事件持久化。

验收标准：

```text
使用 TestBrain 可以跑通完整 Agent Loop
事件顺序稳定
非法 action 被拦截
超过 max_steps 会失败退出
```

### M2：本地真实 LLM 接入

任务：

1. 接入 llama.cpp / GGUF。
2. 实现模型配置。
3. 实现 Prompt Builder。
4. 实现 JSON 输出解析。
5. 实现重试。
6. 记录原始模型输出。
7. 记录 token / 耗时。
8. 抽象 Brain 接口。

验收标准：

```text
本地 Qwen 小模型真实参与决策
模型输出可解析为 Action
每一步能看到原始输出
模型不存在时给出明确错误
```

### M3：工具运行时

任务：

1. 实现 calculator。
2. 实现 mock_search。
3. 实现 memory_store。
4. 实现 memory_lookup。
5. 实现 final_answer。
6. 实现工具参数校验。
7. 实现工具执行超时。
8. 记录工具事件。

验收标准：

```text
所有工具本地可运行
参数错误可被捕获
工具结果可写回上下文
每个调用都有事件记录
```

### M4：前端可视化

任务：

1. 实现 Agent Canvas。
2. 实现 Context Panel。
3. 实现 Action Panel。
4. 实现 Tool Panel。
5. 实现 Event Timeline。
6. 实现运行按钮。
7. 实现单步执行。
8. 实现慢速播放。
9. 实现回放。
10. 实现错误展示。

验收标准：

```text
用户可以完整运行一个 Agent
可以逐步看到上下文变化
可以逐步看到模型输出
可以逐步看到工具调用
可以回放完整过程
```

### M5：教学案例与打磨

任务：

1. 完成 5 个教学案例。
2. 每个案例配讲解文案。
3. 每个案例配预期事件流。
4. 优化 UI 文案。
5. 修复模型不稳定问题。
6. 增加 Prompt 优化。
7. 完成端到端测试。

验收标准：

```text
5 个案例稳定可演示
初学者可以在 5 分钟内理解 Agent Loop
至少 3 个非开发用户能完成第一个案例
```

## 十、接口设计草案

### 创建 Agent

```http
POST /api/v1/agents
```

Request：

```json
{
  "name": "简单工具调用 Agent",
  "spec": {}
}
```

### 获取 Agent

```http
GET /api/v1/agents/{agent_id}
```

### 运行 Agent

```http
POST /api/v1/agents/{agent_id}/runs
```

Request：

```json
{
  "input": "帮我计算 12 * 7"
}
```

### 获取运行事件

```http
GET /api/v1/runs/{run_id}/events
```

### 获取回放

```http
GET /api/v1/runs/{run_id}/replay
```

### 单步执行

```http
POST /api/v1/runs/{run_id}/steps
```

### 暂停 / 继续

```http
POST /api/v1/runs/{run_id}/pause
POST /api/v1/runs/{run_id}/resume
```

## 十一、开发任务拆解

### A. 后端任务

1. FastAPI 项目初始化。
2. 数据库模型。
3. Agent CRUD。
4. Run CRUD。
5. Event 存储。
6. Agent Interpreter。
7. Context Builder。
8. LLM Brain 接口。
9. llama.cpp 接入。
10. Action Validator。
11. Tool Registry。
12. 工具沙盒执行。
13. 运行控制器。
14. WebSocket / SSE。
15. 回放 API。
16. 单元测试。
17. 集成测试。

### B. 前端任务

1. Vite + React 初始化。
2. 基础布局。
3. Agent Canvas。
4. 工具配置面板。
5. 运行控制栏。
6. Context Panel。
7. Action Panel。
8. Tool Panel。
9. Event Timeline。
10. 回放控制。
11. 状态管理。
12. API client。
13. WebSocket / SSE 接入。
14. 错误展示。
15. E2E 测试。

### C. 教学内容任务

1. 教学大纲。
2. 5 个课程案例。
3. 每个案例的教学文案。
4. 每步解释文案。
5. 失败案例讲解。
6. 教学视频脚本。
7. 初学者可用性测试。

## 十二、验收指标

### 功能指标

```text
1. 支持创建并运行至少一个 Agent。
2. 支持本地真实 LLM。
3. 支持至少 5 个工具。
4. 支持事件流。
5. 支持单步执行。
6. 支持回放。
7. 支持上下文查看。
8. 支持 5 个课程案例。
```

### 质量指标

```text
1. 后端核心模块单元测试覆盖率 >= 80%。
2. 5 个教学案例稳定成功率 >= 80%。
3. 事件流顺序确定性测试全部通过。
4. 非法 Action 100% 被拦截。
5. 工具调用无网络依赖。
6. 8GB 内存机器可运行 0.5B / 0.6B 模型模式。
```

### 教学效果指标

```text
1. 初学者完成第一个案例时间 <= 5 分钟。
2. 80% 测试用户能说出 Agent Loop 的四个关键步骤。
3. 80% 测试用户能指出工具调用发生的位置。
4. 70% 测试用户能解释上下文为什么变化。
```

## 十三、主要风险与应对

### 风险 1：小模型输出不稳定

现象：

```text
JSON 格式错误
工具选择错误
参数错误
重复调用
提前结束
```

应对：

1. 使用 JSON Schema / Grammar。
2. 优化 system prompt。
3. 增加修复重试。
4. 收集失败样本。
5. 第二阶段 LoRA 微调。
6. 保留失败案例作为教学内容。

### 风险 2：本地模型性能不足

应对：

1. 默认使用 0.5B / 0.6B。
2. 课堂服务器集中推理。
3. 支持事件慢速播放。
4. 支持异步运行。
5. 记录耗时，不阻塞 UI。

### 风险 3：前端交互复杂度过高

应对：

1. 第一版不追求完整自由拖拽。
2. 先固定核心 Agent Loop 模板。
3. 只允许配置工具和策略。
4. 第二阶段再扩展 Blockly。

### 风险 4：教学效果不明显

应对：

1. 每个案例都绑定明确教学目标。
2. 面向非开发者做可用性测试。
3. 强制展示真实上下文和真实模型输出。
4. 增加单步解释和回放。

### 风险 5：范围蔓延

应对：

1. 严格限制第一阶段只做单 Agent。
2. 不做多模型路由。
3. 不做教师后台。
4. 不做云端服务。
5. 不做完整自由编程。

## 十四、阶段完成定义

第一阶段完成的定义：

```text
1. 用户可以打开 Web 应用。
2. 可以选择或创建一个简单 Agent。
3. 可以输入任务。
4. 系统调用本地真实 Qwen 小模型。
5. 模型真实输出 Action。
6. 系统执行本地工具。
7. 前端逐步展示上下文、模型输出、工具调用和事件。
8. 支持单步执行和回放。
9. 至少 5 个教学案例稳定可用。
10. 至少 3 个非开发测试用户能理解 Agent Loop。
```

## 十五、建议时间盒

如果由编程 Agent 辅助开发，建议按 4 周时间盒执行：

| 周期 | 目标 | 产出 |
|---|---|---|
| 第 1 周 | M0 + M1 | 项目骨架 + 协议 + 解释器可跑 |
| 第 2 周 | M2 + M3 | 真实 LLM 接入 + 工具运行时 |
| 第 3 周 | M4 | 可视化执行和回放 |
| 第 4 周 | M5 | 课程案例 + 可用性测试 + 打磨 |

每周保留 20% 时间处理模型不稳定和交互打磨。
