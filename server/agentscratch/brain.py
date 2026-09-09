"""Brain interface and deterministic test implementation."""

from __future__ import annotations

import re
from typing import Protocol

from .llm import BrainResult
from .schemas import AgentAction, MiniAgentSpec, ToolCall


class Brain(Protocol):
    def decide(self, spec: MiniAgentSpec, context: list[dict]) -> BrainResult:
        """Return the next action based only on the current visible context."""


class TestBrain:
    def decide(self, spec: MiniAgentSpec, context: list[dict]) -> BrainResult:
        user_input = next((m["content"] for m in context if m.get("role") == "user"), "")
        tool_messages = [m for m in context if m.get("role") == "tool"]
        last_tool_result = next(
            (m["content"] for m in reversed(tool_messages)), None
        )
        allowed_tools = {tool.name for tool in spec.tools}

        # This is a deliberately labelled deterministic teaching trajectory,
        # not a replacement for model planning.  It exercises the same real
        # interpreter, permission checks, parallel scheduler and Observations
        # that a tiny_llm run uses.
        document_tools = {"read_document", "extract_numbers", "calculator"}
        if "文件" in user_input and "相加" in user_input and document_tools <= allowed_tools:
            read_results = [
                message["content"]
                for message in tool_messages
                if message.get("name") == "read_document" and isinstance(message.get("content"), dict)
            ]
            extracted_results = [
                message["content"]
                for message in tool_messages
                if message.get("name") == "extract_numbers" and isinstance(message.get("content"), dict)
            ]
            calculated = next(
                (message["content"] for message in reversed(tool_messages) if message.get("name") == "calculator"),
                None,
            )
            if calculated is not None:
                return BrainResult(
                    action=AgentAction(
                        thought="The dependent calculation returned an Observation.",
                        action="final_answer",
                        answer=f"两个文件中的数值相加结果是 {calculated}。",
                        done=True,
                    ),
                    raw_output="test-brain-document-sum-final",
                    model="test-brain",
                    duration_ms=0,
                )
            if len(extracted_results) == 2:
                values = [result.get("values", []) for result in extracted_results]
                if all(items for items in values):
                    expression = f"{values[0][0]} + {values[1][0]}"
                    return BrainResult(
                        action=AgentAction(
                            thought="Both extracted Observations are ready, so calculate their sum.",
                            action="call_tool",
                            tool="calculator",
                            args={"expression": expression},
                        ),
                        raw_output="test-brain-document-sum-calculate",
                        model="test-brain",
                        duration_ms=0,
                    )
            if len(read_results) == 2:
                return BrainResult(
                    action=AgentAction(
                        thought="Both file Observations are ready; extract values from their returned text.",
                        action="call_tools",
                        tool_calls=[
                            ToolCall(tool="extract_numbers", call_id="extract-a", args={"text": read_results[0]["content"]}),
                            ToolCall(tool="extract_numbers", call_id="extract-b", args={"text": read_results[1]["content"]}),
                        ],
                    ),
                    raw_output="test-brain-document-sum-extract",
                    model="test-brain",
                    duration_ms=0,
                )
            return BrainResult(
                action=AgentAction(
                    thought="Read the two independently authorized teaching files in parallel.",
                    action="call_tools",
                    tool_calls=[
                        ToolCall(tool="read_document", call_id="read-a", args={"file_id": "file-a"}),
                        ToolCall(tool="read_document", call_id="read-b", args={"file_id": "file-b"}),
                    ],
                ),
                raw_output="test-brain-document-sum-read",
                model="test-brain",
                duration_ms=0,
            )

        file_summary_tools = {"search_files", "read_file"}
        if "先查询工作目录，再让我选择要总结的文件" in user_input and file_summary_tools <= allowed_tools:
            # Case 4 deliberately demonstrates a genuine Observation-led
            # clarification: browse metadata first, then formulate the
            # question from returned candidates instead of guessing a file.
            search_message = next(
                (message for message in tool_messages if message.get("name") == "search_files" and isinstance(message.get("content"), dict)),
                None,
            )
            search_result = search_message["content"] if search_message else None
            file_result = next(
                (message["content"] for message in reversed(tool_messages) if message.get("name") == "read_file" and isinstance(message.get("content"), dict)),
                None,
            )
            learner_messages = [
                str(message.get("content", ""))
                for message in context
                if message.get("role") == "user" and not str(message.get("content", "")).startswith("<agent_status>")
            ]
            if search_result is None:
                return BrainResult(
                    action=AgentAction(
                        thought="The user did not name a file, so browse the authorized workspace metadata before asking.",
                        action="call_tool",
                        tool="search_files",
                        args={"directory": "."},
                    ),
                    raw_output="test-brain-file-choice-browse",
                    model="test-brain",
                    duration_ms=0,
                )

            matches = [item for item in search_result.get("matches", []) if isinstance(item, dict) and isinstance(item.get("name"), str)]
            requested_teaching_files = "教学文件 A.txt 和教学文件 B.txt" in user_input
            candidates = (
                [item for item in matches if item.get("name") in {"教学文件 A.txt", "教学文件 B.txt"}]
                if requested_teaching_files
                else sorted(
                    matches,
                    key=lambda item: (
                        "复盘" not in str(item.get("name", "")),
                        str(item.get("directory", "")) != "季度报告",
                        str(item.get("name", "")),
                    ),
                )[:3]
            )
            if not candidates:
                return BrainResult(
                    action=AgentAction(
                        thought="The directory browse returned no readable files, so ask the learner to provide another directory or file name.",
                        action="ask_user",
                        args={"question": "我已查询当前工作目录，但没有找到可读取的文本文件。请提供要总结的相对目录或完整文件名。"},
                    ),
                    raw_output="test-brain-file-choice-empty",
                    model="test-brain",
                    duration_ms=0,
                )

            selection_state = next(
                (
                    str(message.get("content", ""))
                    for message in reversed(context)
                    if message.get("role") == "system" and str(message.get("content", "")).startswith("USER_SELECTION_STATE")
                ),
                "",
            )
            if "status: ambiguous" in selection_state or "status: unmatched" in selection_state:
                recovery_question = (
                    "“销售复盘”可能对应多个文件，请提供完整文件名或从候选中选择。"
                    if "status: ambiguous" in selection_state
                    else "未找到与该名称匹配的文件，请提供完整文件名或从候选中选择。"
                )
                return BrainResult(
                    action=AgentAction(
                        thought="The submitted file name is not uniquely resolvable, so request a more specific selection without reading any file.",
                        action="ask_user",
                        args={
                            "question": recovery_question,
                            "response_schema": {
                                "type": "file_selection",
                                "source_observation": search_message.get("tool_call_id"),
                                "candidate_file_ids": [item["file_id"] for item in candidates],
                                "allow_custom_input": True,
                            },
                        },
                    ),
                    raw_output="test-brain-file-choice-recover",
                    model="test-brain",
                    duration_ms=0,
                )

            selected = None
            if len(learner_messages) > 1:
                selection = learner_messages[-1].strip()
                if selection.isdigit() and 1 <= int(selection) <= len(candidates):
                    selected = candidates[int(selection) - 1]
                else:
                    selected = next(
                        (item for item in matches if str(item.get("name", "")).casefold() in selection.casefold()),
                        None,
                    )
            if selected is None:
                options = "\n".join(
                    f"{index}. {item['name']}（{item.get('directory') or '根目录'}）"
                    for index, item in enumerate(candidates, start=1)
                )
                return BrainResult(
                    action=AgentAction(
                        thought="Use the file metadata Observation to offer the most likely choices and request the learner's selection.",
                        action="ask_user",
                        args={
                            "question": (
                                "我已查询工作目录，找到以下最可能需要总结的文件：\n"
                                f"{options}\n"
                                "请输入序号（1-3），或直接输入完整文件名作为自定义选择。"
                            ),
                            "response_schema": {
                                "type": "file_selection",
                                "source_observation": search_message.get("tool_call_id"),
                                "candidate_file_ids": [item["file_id"] for item in candidates],
                                "allow_custom_input": True,
                            },
                        },
                    ),
                    raw_output="test-brain-file-choice-ask",
                    model="test-brain",
                    duration_ms=0,
                )

            if file_result is not None:
                content = str(file_result.get("content", file_result))
                summary_lines = [
                    line.strip() for line in content.splitlines()
                    if line.strip() and not line.lstrip().startswith("#")
                ]
                excerpt = "\n".join(summary_lines[:3]) or content[:360]
                return BrainResult(
                    action=AgentAction(
                        thought="The learner selected a discovered file and its Observation is now available, so provide a concise content summary.",
                        action="final_answer",
                        answer=f"《{selected['name']}》的内容摘要：\n{excerpt}",
                        done=True,
                    ),
                    raw_output="test-brain-file-choice-final",
                    model="test-brain",
                    duration_ms=0,
                )
            return BrainResult(
                action=AgentAction(
                    thought="The learner selected a file from the search Observation; read its exact file_id next.",
                    action="call_tool",
                    tool="read_file",
                    args={"file_id": selected["file_id"]},
                ),
                raw_output="test-brain-file-choice-read",
                model="test-brain",
                duration_ms=0,
            )
        if "没有提供文件名" in user_input and file_summary_tools <= allowed_tools:
            return BrainResult(
                action=AgentAction(
                    thought="The task lacks a file identifier, so ask the user instead of guessing.",
                    action="ask_user",
                    args={"question": "请告诉我要总结的文件名，以及它所在的相对目录；根目录可写“.”。"},
                ),
                raw_output="test-brain-file-summary-ask-user",
                model="test-brain",
                duration_ms=0,
            )
        if "找到“" in user_input and "阅读后概括" in user_input and file_summary_tools <= allowed_tools:
            file_match = re.search(r"找到“([^”]+)”", user_input)
            directory_match = re.search(r"在“([^”]+)”目录", user_input)
            expected_name = file_match.group(1) if file_match else ""
            expected_directory = directory_match.group(1) if directory_match else "."
            search_result = next(
                (message["content"] for message in tool_messages if message.get("name") == "search_files" and isinstance(message.get("content"), dict)),
                None,
            )
            file_result = next(
                (message["content"] for message in reversed(tool_messages) if message.get("name") == "read_file" and isinstance(message.get("content"), dict)),
                None,
            )
            if file_result is not None:
                return BrainResult(
                    action=AgentAction(
                        thought="The requested file Observation is available, so return its factual summary.",
                        action="final_answer",
                        answer=str(file_result.get("content", file_result)),
                        done=True,
                    ),
                    raw_output="test-brain-generic-file-summary-final",
                    model="test-brain",
                    duration_ms=0,
                )
            if search_result is not None:
                matches = search_result.get("matches", [])
                file_id = next(
                    (item.get("file_id") for item in matches if isinstance(item, dict) and item.get("name") == expected_name),
                    None,
                )
                if file_id:
                    return BrainResult(
                        action=AgentAction(
                            thought="The search Observation identified the requested file; read its returned file_id next.",
                            action="call_tool",
                            tool="read_file",
                            args={"file_id": file_id},
                        ),
                        raw_output="test-brain-generic-file-summary-read",
                        model="test-brain",
                        duration_ms=0,
                    )
            return BrainResult(
                action=AgentAction(
                    thought="Search the authorized directory before reading a file.",
                    action="call_tool",
                    tool="search_files",
                    args={"directory": expected_directory, "file_name": expected_name},
                ),
                raw_output="test-brain-generic-file-summary-search",
                model="test-brain",
                duration_ms=0,
            )
        if "华东销售复盘" in user_input and file_summary_tools <= allowed_tools:
            search_result = next(
                (message["content"] for message in tool_messages if message.get("name") == "search_files" and isinstance(message.get("content"), dict)),
                None,
            )
            file_result = next(
                (message["content"] for message in reversed(tool_messages) if message.get("name") == "read_file" and isinstance(message.get("content"), dict)),
                None,
            )
            if file_result is not None:
                return BrainResult(
                    action=AgentAction(
                        thought="The file Observation contains the requested conclusions, so summarize them for the user.",
                        action="final_answer",
                        answer=(
                            "《华东销售复盘》的三项主要结论：\n"
                            "1. 第二季度销售额同比增长 18%，企业订阅产品是新增收入主力。\n"
                            "2. 上海和杭州续约率达 92%，苏州续约偏低，需要补充驻场支持。\n"
                            "3. 线下渠道转化率 11%，低于线上研讨会的 19%，应把更多预算转向线上活动。"
                        ),
                        done=True,
                    ),
                    raw_output="test-brain-file-summary-final",
                    model="test-brain",
                    duration_ms=0,
                )
            if search_result is not None:
                matches = search_result.get("matches", [])
                file_id = next(
                    (item.get("file_id") for item in matches if isinstance(item, dict) and item.get("name") == "华东销售复盘.md"),
                    None,
                )
                if file_id:
                    return BrainResult(
                        action=AgentAction(
                            thought="The search Observation identified the requested file; read that file_id next.",
                            action="call_tool",
                            tool="read_file",
                            args={"file_id": file_id},
                        ),
                        raw_output="test-brain-file-summary-read",
                        model="test-brain",
                        duration_ms=0,
                    )
            return BrainResult(
                action=AgentAction(
                    thought="The task names a directory and file, so search the authorized workspace before reading.",
                    action="call_tool",
                    tool="search_files",
                    args={"directory": "季度报告", "file_name": "华东销售复盘.md"},
                ),
                raw_output="test-brain-file-summary-search",
                model="test-brain",
                duration_ms=0,
            )

        if last_tool_result is not None:
            return BrainResult(
                action=AgentAction(
                    thought="A tool returned the required result; answer the user.",
                    action="final_answer",
                    answer=str(last_tool_result),
                    done=True,
                ),
                raw_output="test-brain-final",
                model="test-brain",
                duration_ms=0,
            )

        if "记住" in user_input and "memory_store" in allowed_tools:
            key, _, value = user_input.partition("记住")
            return BrainResult(
                action=AgentAction(
                    thought="Save the stated fact to memory.",
                    action="call_tool",
                    tool="memory_store",
                    args={"key": "fact", "value": value.strip()},
                ),
                raw_output="test-brain-memory",
                model="test-brain",
                duration_ms=0,
            )

        if ("并行" in user_input or "parallel" in user_input.lower()) and {"calculator", "mock_search"} <= allowed_tools:
            return BrainResult(
                action=AgentAction(
                    thought="This teaching request explicitly demonstrates two independent reads.",
                    action="call_tools",
                    tool_calls=[
                        ToolCall(tool="calculator", args={"expression": "2 + 2"}),
                        ToolCall(tool="mock_search", args={"query": "agent"}),
                    ],
                ),
                raw_output="test-brain-parallel-tools",
                model="test-brain",
                duration_ms=0,
            )

        numbers = re.findall(r"-?\d+(?:\.\d+)?", user_input)
        if numbers and "calculator" in allowed_tools:
            expression = user_input.replace("计算", "").replace("calculate", "").replace("算", "").strip()
            return BrainResult(
                action=AgentAction(
                    thought="The task requires arithmetic.",
                    action="call_tool",
                    tool="calculator",
                    args={"expression": expression},
                ),
                raw_output="test-brain-calculator",
                model="test-brain",
                duration_ms=0,
            )

        if "agent" in user_input.lower() and "mock_search" in allowed_tools:
            return BrainResult(
                action=AgentAction(
                    thought="Search the local teaching corpus.",
                    action="call_tool",
                    tool="mock_search",
                    args={"query": "agent"},
                ),
                raw_output="test-brain-search",
                model="test-brain",
                duration_ms=0,
            )

        return BrainResult(
            action=AgentAction(
                thought="Return the available answer.",
                action="final_answer",
                answer=user_input or "No answer available.",
                done=True,
            ),
            raw_output="test-brain-direct",
            model="test-brain",
            duration_ms=0,
        )


class ClarificationBrain:
    """Deterministic teaching brain used to demonstrate ask_user continuation."""

    def decide(self, spec: MiniAgentSpec, context: list[dict]) -> BrainResult:
        user_messages = [
            m["content"]
            for m in context
            if m.get("role") == "user" and not str(m.get("content", "")).startswith("<agent_status>")
        ]
        if len(user_messages) == 1:
            return BrainResult(
                action=AgentAction(
                    thought="The request is underspecified; ask the learner for one detail.",
                    action="ask_user",
                    args={"question": "你希望我围绕什么主题继续？"},
                ),
                raw_output="clarification-brain-ask-user",
                model="clarification-brain",
                duration_ms=0,
            )
        return BrainResult(
            action=AgentAction(
                thought="The learner supplied the missing detail; finish the task.",
                action="final_answer",
                answer=f"已收到补充信息：{user_messages[-1]}",
                done=True,
            ),
            raw_output="clarification-brain-final",
            model="clarification-brain",
            duration_ms=0,
        )
