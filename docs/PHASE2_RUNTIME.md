# Phase 2：运行控制与教学案例

本阶段在 M2 的真实本地 LLM 接入之上，补齐可暂停、可续跑、可回放的运行控制层。

## Run 状态

Run 的主要状态为：

- created：已创建，尚未执行
- running：正在执行或等待下一次 step
- paused：用户主动暂停，必须 resume 后继续
- waiting_user：Agent 通过 ask_user 等待补充信息
- completed：运行完成
- failed：运行失败

暂停在 Agent 执行周期之间生效。一个周期包含一次模型决策和随后的一次工具调用；已经开始的模型请求不会被强制中断。

## 控制接口

- POST /api/v1/runs/{run_id}/steps：执行一个模型决策/工具调用周期
- POST /api/v1/runs/{run_id}/pause：暂停 created 或 running 的 Run
- POST /api/v1/runs/{run_id}/resume：继续 paused 的 Run
- POST /api/v1/runs/{run_id}/resume，body 为 {"input":"..."}：回答 ask_user 并继续
- GET /api/v1/runs/{run_id}/events：读取持久化事件，可用 after 游标
- GET /api/v1/runs/{run_id}/events/stream：通过 SSE 订阅事件
- GET /api/v1/runs/{run_id}/replay：读取完整回放包

## ask_user

AgentAction 的 ask_user 必须提供非空的 args.question。运行进入 waiting_user 后，服务会：

1. 保存 waiting_question；
2. 发出 agent.waiting_user 事件；
3. 等待 resume body 中的 input；
4. 将 input 作为新的 user context 写入事件和上下文；
5. 清除 waiting_question 并继续运行。

仓库内的 ask-user 教学案例使用 clarification brain，可在没有本地模型服务时稳定演示这个流程。

## 教学案例模板

GET /api/v1/cases 返回内置案例模板以及已保存的自定义案例，包括 calculator、local-search、memory、ask-user 和 real-llm。

GET /api/v1/cases/{case_id} 返回单个模板详情。

POST /api/v1/cases 使用 case_id、title、description、objective、input、brain_type 和 tools 创建持久化自定义模板。

POST /api/v1/cases/{case_id}/runs 会根据模板创建独立 Agent 和 Run，也可以用 body.input 覆盖案例默认输入。
