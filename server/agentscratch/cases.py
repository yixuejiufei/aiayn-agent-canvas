"""Built-in teaching case templates for the first local classroom flow."""

from __future__ import annotations

from typing import Any


TEACHING_CASES: tuple[dict[str, Any], ...] = (
    {
        "case_id": "calculator",
        "title": "计算器：一次工具调用",
        "description": "观察模型选择 calculator，并把结果带回上下文。",
        "objective": "理解 decision → tool → observation → final answer。",
        "input": "计算 12 * 7",
        "brain_type": "test",
        "brain_model": None,
        "tools": ["calculator", "final_answer"],
    },
    {
        "case_id": "local-search",
        "title": "本地搜索：工具结果如何进入上下文",
        "description": "在本地教学语料中搜索 Agent Loop 的解释。",
        "objective": "理解工具结果会成为下一轮模型决策的上下文。",
        "input": "请搜索 agent loop 是什么",
        "brain_type": "test",
        "brain_model": None,
        "tools": ["mock_search", "final_answer"],
    },
    {
        "case_id": "two-files-sum",
        "title": "两文件求和：并行读取与依赖链",
        "description": "确定性教学轨迹：并行读取 file-a/file-b，再提取数值、计算并输出。",
        "objective": "区分并行无依赖调用与必须等待 Observation 的后续调用。",
        "input": "读取教学文件 A 与 B，将其中数值相加",
        "brain_type": "test",
        "brain_model": None,
        "tools": ["read_document", "extract_numbers", "calculator", "final_answer"],
    },
    {
        "case_id": "file-summary",
        "title": "文件总结：搜索后读取",
        "description": "先在受限目录中搜索指定文件，再使用返回的 file_id 读取并概括内容。",
        "objective": "理解两个有前后依赖的工具调用，以及 Observation 如何决定下一轮行动。",
        "input": "请在“季度报告”目录中找到“华东销售复盘.md”，阅读后概括其中三项主要结论。",
        "brain_type": "test",
        "brain_model": None,
        "tools": ["search_files", "read_file", "final_answer"],
    },
    {
        "case_id": "file-choice",
        "title": "文件总结：搜索后询问选择",
        "description": "先浏览授权工作目录的文件元数据，再根据 Observation 给出候选项并等待用户选择。",
        "objective": "理解 ask_user 可以发生在工具 Observation 之后，用户回复会追加到下一轮 Context。",
        "input": "请先查询工作目录，再让我选择要总结的文件。我暂不指定具体文件名；请根据搜索结果给出最可能的 1-3 个选项。",
        "brain_type": "test",
        "brain_model": None,
        "tools": ["search_files", "read_file", "final_answer"],
    },
    {
        "case_id": "memory",
        "title": "记忆：保存一次事实",
        "description": "把用户提供的事实写入 Agent 的短期记忆。",
        "objective": "理解工具可以改变运行时 memory 状态。",
        "input": "记住我喜欢用 Python 学习 Agent",
        "brain_type": "test",
        "brain_model": None,
        "tools": ["memory_store", "final_answer"],
    },
    {
        "case_id": "ask-user",
        "title": "追问：ask_user 用户输入续跑",
        "description": "先暂停等待补充信息，再从 resume 继续执行。",
        "objective": "理解 Agent 如何把不完整请求交回用户澄清。",
        "input": "帮我设计一个学习计划",
        "brain_type": "clarification",
        "brain_model": None,
        "tools": ["final_answer"],
    },
    {
        "case_id": "real-llm",
        "title": "真实模型：本地 Qwen Action",
        "description": "使用本地 Qwen GGUF 完成一次真实工具调用。",
        "objective": "观察 Transformer LLM 产生的原始 JSON Action。",
        "input": "计算 25 * 4",
        "brain_type": "tiny_llm",
        "brain_model": "qwen-tiny",
        "tools": ["calculator", "final_answer"],
    },
)


def list_cases() -> list[dict[str, Any]]:
    return [dict(case) for case in TEACHING_CASES]


def get_case(case_id: str) -> dict[str, Any] | None:
    return next((dict(case) for case in TEACHING_CASES if case["case_id"] == case_id), None)
