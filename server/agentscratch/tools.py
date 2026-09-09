"""Local sandbox tool runtime. Tools intentionally have no network or filesystem access."""

from __future__ import annotations

import ast
import hashlib
import operator
import re
from pathlib import Path
from typing import Any

_SAFE_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_SAFE_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_SEARCH_CORPUS = [
    "LLM agents combine a model, memory, tools, and a policy for choosing actions.",
    "Tool calling means the model selects a tool and fills its arguments.",
    "An agent loop repeats observation, decision, action, and observation until done.",
]
_TEACHING_RESOURCES = {
    "file-a": {
        "name": "教学文件 A.txt",
        "content": "项目 A 的可计入数值为 12。",
    },
    "file-b": {
        "name": "教学文件 B.txt",
        "content": "项目 B 的可计入数值为 30。",
    },
    "east-sales-review-q2": {
        "directory": "季度报告",
        "name": "华东销售复盘.md",
        "content": """# 华东销售复盘\n\n## 三项主要结论\n1. 第二季度华东区销售额同比增长 18%，其中企业订阅产品贡献了新增收入的主要部分。\n2. 上海与杭州的续约率达到 92%，说明重点客户的客户成功跟进有效；苏州的续约率较低，需要在下季度补充驻场支持。\n3. 线下渠道带来的线索转化率只有 11%，低于线上研讨会的 19%，下季度应把更多预算转向线上行业活动。\n\n## 后续行动\n销售负责人将在 7 月前提交苏州客户挽回计划，并调整渠道活动预算。""",
    },
    "north-sales-review-q2": {
        "directory": "季度报告",
        "name": "华北销售复盘.md",
        "content": "华北区第二季度销售复盘：重点关注北京区域的交付节奏和合作伙伴培训。",
    },
    "south-sales-review-q2": {
        "directory": "季度报告",
        "name": "华南销售复盘.md",
        "content": """# 华南销售复盘\n\n1. 广深两地第二季度销售额同比增长 14%，渠道伙伴贡献新增订单 46%。\n2. 新客户首月激活率为 81%，客服响应中位时间缩短至 6 分钟。\n3. 珠海区域的商机转化率仅 9%，下季度需要补充行业方案支持。""",
    },
    "west-sales-review-q2": {
        "directory": "季度报告",
        "name": "西部销售复盘.md",
        "content": """# 西部销售复盘\n\n1. 成都与重庆的订阅收入同比增长 21%，教育行业是主要增量。\n2. 重点客户续约率达到 88%，但首单平均周期延长到 37 天。\n3. 线上演示后的线索转化率为 16%，应继续投入区域线上活动。""",
    },
    "operations-review-q2": {
        "directory": "运营报告",
        "name": "客户运营复盘.md",
        "content": """# 客户运营复盘\n\n1. 高价值客户健康度达标率为 86%，较上季度提升 7 个百分点。\n2. 续费预警工单的 24 小时处理率为 94%。\n3. 新用户第七日留存率为 43%，需要优化首次配置引导。""",
    },
}
_TEXT_FILE_SUFFIXES = {".txt", ".md", ".csv", ".json", ".log"}


def teaching_resources() -> dict[str, dict[str, str]]:
    """Return a fresh, explicit teaching workspace for each Agent run."""
    return {file_id: dict(document) for file_id, document in _TEACHING_RESOURCES.items()}


def choose_teaching_workspace_directory() -> Path | None:
    """Ask the local desktop user to choose an explicit teaching root.

    A browser's directory picker intentionally hides absolute paths.  The
    teaching runtime needs that path to enforce a root boundary while a run is
    executing, so this small local-only API uses the native Windows dialog.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:  # pragma: no cover - depends on Python build
        raise RuntimeError("native folder picker is unavailable in this Python installation") from exc

    dialog_root = tk.Tk()
    try:
        dialog_root.withdraw()
        dialog_root.attributes("-topmost", True)
        dialog_root.update()
        selected = filedialog.askdirectory(
            parent=dialog_root,
            title="选择 AgentScratch 教学工作区",
            mustexist=True,
        )
    finally:
        dialog_root.destroy()
    if not selected:
        return None
    root = Path(selected).resolve(strict=True)
    if not root.is_dir():
        raise RuntimeError("selected teaching workspace is not a directory")
    return root


def teaching_workspace_tree(root_path: str, *, max_depth: int = 4, max_entries: int = 240) -> dict[str, Any]:
    """Return bounded metadata for a workspace, never file content.

    Links resolving outside the selected root are intentionally excluded so a
    convenient visual tree cannot accidentally imply a permission expansion.
    """
    try:
        root = Path(root_path).resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"teaching workspace directory does not exist: {root_path}") from exc
    if not root.is_dir():
        raise ValueError(f"teaching workspace is not a directory: {root_path}")

    entry_count = 0
    truncated = False

    def inside_root(path: Path) -> bool:
        try:
            path.resolve(strict=True).relative_to(root)
            return True
        except (OSError, ValueError):
            return False

    def children_of(directory: Path, depth: int) -> list[dict[str, Any]]:
        nonlocal entry_count, truncated
        if depth >= max_depth:
            return []
        try:
            children = sorted(directory.iterdir(), key=lambda item: (not item.is_dir(), item.name.casefold()))
        except OSError:
            return []
        entries: list[dict[str, Any]] = []
        for child in children:
            if entry_count >= max_entries:
                truncated = True
                break
            if not inside_root(child):
                continue
            entry_count += 1
            relative_path = child.relative_to(root).as_posix()
            if child.is_dir():
                entries.append({
                    "name": child.name,
                    "kind": "directory",
                    "relative_path": relative_path,
                    "children": children_of(child, depth + 1),
                })
                continue
            try:
                size_bytes = child.stat().st_size
            except OSError:
                size_bytes = 0
            entries.append({
                "name": child.name,
                "kind": "file",
                "relative_path": relative_path,
                "size_bytes": size_bytes,
                "readable": child.suffix.casefold() in _TEXT_FILE_SUFFIXES,
            })
        return entries

    return {
        "root_path": str(root),
        "name": root.name or str(root),
        "entries": children_of(root, 0),
        "entry_count": entry_count,
        "truncated": truncated,
        "max_depth": max_depth,
        "max_entries": max_entries,
    }


def teaching_workspace_index(root_path: str, *, max_entries: int = 40) -> dict[str, Any]:
    """Build a compact, content-free manifest for the model's Context.

    It gives a small local model enough metadata to plan a search without
    granting direct reads.  The returned paths remain relative to the
    selected root and the actual content still requires search -> file_id ->
    read_file.
    """
    tree = teaching_workspace_tree(root_path, max_depth=3, max_entries=max_entries)
    entries: list[dict[str, Any]] = []

    def collect(items: list[dict[str, Any]]) -> None:
        for item in items:
            record = {"path": item["relative_path"], "kind": item["kind"]}
            if item["kind"] == "file":
                record["readable"] = item.get("readable", False)
            entries.append(record)
            if item["kind"] == "directory":
                collect(item.get("children", []))

    collect(tree["entries"])
    return {"root": ".", "entries": entries, "truncated": tree["truncated"]}


def _eval_math(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_math(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_BIN_OPS:
        return _SAFE_BIN_OPS[type(node.op)](_eval_math(node.left), _eval_math(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_UNARY_OPS:
        return _SAFE_UNARY_OPS[type(node.op)](_eval_math(node.operand))
    raise ValueError("Expression contains unsupported operations")


def calculator(expression: str) -> Any:
    tree = ast.parse(expression, mode="eval")
    return _eval_math(tree)


def mock_search(query: str) -> list[str]:
    q = query.lower()
    return [item for item in _SEARCH_CORPUS if any(word in item.lower() for word in q.split())]


def read_document(file_id: str, *, memory: dict[str, Any]) -> dict[str, Any]:
    """Read only a resource explicitly placed in this run's workspace."""
    resources = memory.get("resources", {})
    if not isinstance(resources, dict) or file_id not in resources:
        raise PermissionError(f"resource is not available to this run: {file_id}")
    document = resources[file_id]
    if not isinstance(document, dict) or not isinstance(document.get("content"), str):
        raise ValueError(f"resource has invalid content: {file_id}")
    content = document["content"]
    if len(content) > 8_000:
        raise ValueError(f"resource exceeds teaching read limit: {file_id}")
    return {
        "file_id": file_id,
        "name": str(document.get("name", file_id)),
        "content": content,
        "characters": len(content),
    }


def search_files(directory: str | None, file_name: str | None = None, *, memory: dict[str, Any]) -> dict[str, Any]:
    """Search the run-scoped teaching workspace and return opaque file IDs.

    This models a real document connector: the model can discover metadata but
    must use a returned ``file_id`` to read content in a later action.
    """
    workspace_root = _workspace_root(memory)
    if workspace_root is not None:
        return _search_workspace_files(workspace_root, directory, file_name, memory)
    resources = memory.get("resources", {})
    if not isinstance(resources, dict):
        raise PermissionError("teaching workspace is not available to this run")
    normalized_directory = (directory or ".").strip().casefold() or "."
    normalized_name = (file_name or "").strip().casefold()
    matches = [
        {
            "file_id": file_id,
            "name": str(document.get("name", file_id)),
            "directory": str(document.get("directory", "")),
            "characters": len(str(document.get("content", ""))),
        }
        for file_id, document in resources.items()
        if isinstance(document, dict)
        and (normalized_directory == "." or str(document.get("directory", "")).casefold() == normalized_directory)
        and (not normalized_name or normalized_name in str(document.get("name", "")).casefold())
    ]
    discovered = memory.setdefault("discovered_file_ids", [])
    if not isinstance(discovered, list):
        discovered = []
        memory["discovered_file_ids"] = discovered
    for match in matches:
        if match["file_id"] not in discovered:
            discovered.append(match["file_id"])
    return {
        "directory": directory or ".",
        "file_name": file_name or "",
        "matches": matches[:20],
        "count": len(matches),
    }


def read_file(file_id: str, *, memory: dict[str, Any]) -> dict[str, Any]:
    """Read one previously discovered, run-authorized teaching file."""
    discovered = memory.get("discovered_file_ids", [])
    if not isinstance(discovered, list) or file_id not in discovered:
        raise PermissionError(f"file must be discovered with search_files before reading: {file_id}")
    workspace_root = _workspace_root(memory)
    if workspace_root is not None:
        return _read_workspace_file(workspace_root, file_id, memory)
    result = read_document(file_id, memory=memory)
    resource = memory["resources"][file_id]
    return {**result, "directory": str(resource.get("directory", ""))}


def _workspace_root(memory: dict[str, Any]) -> Path | None:
    workspace = memory.get("workspace", {})
    configured_root = workspace.get("root_path") if isinstance(workspace, dict) else None
    if not isinstance(configured_root, str) or not configured_root.strip():
        return None
    root = Path(configured_root).resolve(strict=True)
    if not root.is_dir():
        raise PermissionError("configured teaching workspace is not available")
    return root


def _workspace_directory(root: Path, directory: str | None) -> Path:
    normalized = directory.strip() if isinstance(directory, str) else ""
    if not normalized or normalized == ".":
        return root
    requested = Path(normalized)
    if requested.is_absolute():
        raise PermissionError("directory must be a relative path inside the teaching workspace")
    candidate = (root / requested).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise PermissionError("directory escapes the teaching workspace") from exc
    if not candidate.is_dir():
        raise FileNotFoundError(f"directory was not found in the teaching workspace: {directory}")
    return candidate


def _search_workspace_files(root: Path, directory: str | None, file_name: str | None, memory: dict[str, Any]) -> dict[str, Any]:
    target = _workspace_directory(root, directory)
    normalized_name = (file_name or "").strip().casefold()
    matches: list[dict[str, Any]] = []
    references = memory.setdefault("workspace_file_refs", {})
    if not isinstance(references, dict):
        references = {}
        memory["workspace_file_refs"] = references
    for path in sorted(target.rglob("*")):
        if not path.is_file() or path.suffix.casefold() not in _TEXT_FILE_SUFFIXES:
            continue
        if normalized_name and normalized_name not in path.name.casefold():
            continue
        relative = path.relative_to(root).as_posix()
        file_id = "workspace-" + hashlib.sha256(relative.encode("utf-8")).hexdigest()[:16]
        references[file_id] = relative
        matches.append({"file_id": file_id, "name": path.name, "directory": path.parent.relative_to(root).as_posix(), "characters": path.stat().st_size})
        if len(matches) >= 20:
            break
    discovered = memory.setdefault("discovered_file_ids", [])
    if not isinstance(discovered, list):
        discovered = []
        memory["discovered_file_ids"] = discovered
    for match in matches:
        if match["file_id"] not in discovered:
            discovered.append(match["file_id"])
    return {"directory": directory or ".", "file_name": file_name or "", "matches": matches, "count": len(matches), "source": "teaching_workspace"}


def _read_workspace_file(root: Path, file_id: str, memory: dict[str, Any]) -> dict[str, Any]:
    references = memory.get("workspace_file_refs", {})
    relative = references.get(file_id) if isinstance(references, dict) else None
    if not isinstance(relative, str):
        raise PermissionError(f"workspace file reference is not available: {file_id}")
    path = (root / relative).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise PermissionError("file escapes the teaching workspace") from exc
    if not path.is_file() or path.suffix.casefold() not in _TEXT_FILE_SUFFIXES:
        raise PermissionError("file is not a supported text resource")
    raw = path.read_bytes()
    if len(raw) > 8_000:
        raise ValueError("file exceeds teaching read limit: 8000 bytes")
    try:
        content = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        content = raw.decode("gb18030")
    return {"file_id": file_id, "name": path.name, "directory": path.parent.relative_to(root).as_posix(), "content": content, "characters": len(content), "source": "teaching_workspace"}


def extract_numbers(text: str) -> dict[str, Any]:
    """Extract numeric values from supplied text without accessing resources."""
    if len(text) > 8_000:
        raise ValueError("text exceeds teaching extraction limit")
    values = [float(value) if "." in value else int(value) for value in re.findall(r"-?\d+(?:\.\d+)?", text)]
    return {"values": values, "count": len(values)}


def memory_store(key: str, value: str, *, memory: dict[str, Any]) -> str:
    memory[key] = value
    return f"stored:{key}"


def memory_lookup(key: str, *, memory: dict[str, Any]) -> Any:
    if key not in memory:
        raise KeyError(f"memory key not found: {key}")
    return memory[key]


def final_answer(answer: str) -> str:
    return answer


def available_tool_names() -> list[str]:
    return [
        "calculator",
        "mock_search",
        "search_files",
        "read_file",
        "read_document",
        "extract_numbers",
        "memory_store",
        "memory_lookup",
        "final_answer",
    ]
