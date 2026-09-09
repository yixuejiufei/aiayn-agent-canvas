"""SQLite persistence for AgentScratch agents, runs, and event timelines."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from .schemas import AgentEvent, AgentRunState, MiniAgentSpec


def _database_path() -> str:
    configured = os.getenv("AGENTSCRATCH_DB_PATH", "")
    if configured:
        return configured
    return str(Path(__file__).resolve().parent.parent / "agentscratch.db")


class SQLiteStore:
    """Small synchronous repository suited to the local teaching runtime."""

    def __init__(self, path: str | None = None) -> None:
        self.path = path or _database_path()
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._initialize()

    def _initialize(self) -> None:
        with self._lock, self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS agents (
                    agent_id TEXT PRIMARY KEY,
                    spec_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    step INTEGER NOT NULL,
                    type TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    node TEXT NOT NULL,
                    data_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_events_run_step
                    ON events(run_id, step, timestamp);
                CREATE TABLE IF NOT EXISTS cases (
                    case_id TEXT PRIMARY KEY,
                    case_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runtime_settings (
                    setting_key TEXT PRIMARY KEY,
                    setting_value TEXT NOT NULL
                );
                """
            )

    def save_agent(self, agent_id: str, spec: MiniAgentSpec) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO agents(agent_id, spec_json, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(agent_id) DO UPDATE SET spec_json=excluded.spec_json
                """,
                (agent_id, spec.model_dump_json(), time.time()),
            )

    def get_agent(self, agent_id: str) -> MiniAgentSpec | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT spec_json FROM agents WHERE agent_id = ?", (agent_id,)
            ).fetchone()
        return MiniAgentSpec.model_validate_json(row["spec_json"]) if row else None

    def list_agents(self) -> list[tuple[str, MiniAgentSpec]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT agent_id, spec_json FROM agents ORDER BY created_at"
            ).fetchall()
        return [
            (row["agent_id"], MiniAgentSpec.model_validate_json(row["spec_json"]))
            for row in rows
        ]

    def save_case(self, case: dict[str, Any]) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO cases(case_id, case_json, created_at)
                VALUES (?, ?, ?)
                """,
                (
                    case["case_id"],
                    json.dumps(case, ensure_ascii=False),
                    time.time(),
                ),
            )

    def get_case(self, case_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT case_json FROM cases WHERE case_id = ?", (case_id,)
            ).fetchone()
        return json.loads(row["case_json"]) if row else None

    def list_cases(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT case_json FROM cases ORDER BY created_at"
            ).fetchall()
        return [json.loads(row["case_json"]) for row in rows]

    def save_run(self, run: AgentRunState) -> None:
        state_json = run.model_dump_json()
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO runs(run_id, agent_id, state_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    agent_id=excluded.agent_id,
                    state_json=excluded.state_json,
                    updated_at=excluded.updated_at
                """,
                (run.run_id, run.agent_id, state_json, time.time()),
            )
            self._connection.executemany(
                """
                INSERT OR IGNORE INTO events(
                    event_id, run_id, step, type, timestamp, node, data_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        event.event_id,
                        event.run_id,
                        event.step,
                        event.type,
                        event.timestamp,
                        event.node,
                        json.dumps(event.data, ensure_ascii=False, default=str),
                    )
                    for event in run.events
                ],
            )

    def get_run(self, run_id: str) -> AgentRunState | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT state_json FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return AgentRunState.model_validate_json(row["state_json"]) if row else None

    def list_events(self, run_id: str, after: int = 0) -> list[AgentEvent]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT event_id, run_id, step, type, timestamp, node, data_json
                FROM events
                WHERE run_id = ?
                ORDER BY rowid
                LIMIT -1 OFFSET ?
                """,
                (run_id, max(0, after)),
            ).fetchall()
        return [
            AgentEvent(
                event_id=row["event_id"],
                run_id=row["run_id"],
                step=row["step"],
                type=row["type"],
                timestamp=row["timestamp"],
                node=row["node"],
                data=json.loads(row["data_json"]),
            )
            for row in rows
        ]

    def event_cursor(self, run_id: str) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT COUNT(*) AS count FROM events WHERE run_id = ?", (run_id,)
            ).fetchone()
        return int(row["count"]) if row else 0

    def _tool_stats_reset_at(self) -> float:
        with self._lock:
            row = self._connection.execute(
                "SELECT setting_value FROM runtime_settings WHERE setting_key = 'tool_stats_reset_at'"
            ).fetchone()
        try:
            return float(row["setting_value"]) if row else 0.0
        except (TypeError, ValueError):
            return 0.0

    def clear_model_tool_stats(self) -> float:
        """Start a new metrics window without destroying runs or their replay."""
        reset_at = time.time()
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO runtime_settings(setting_key, setting_value) VALUES ('tool_stats_reset_at', ?)
                ON CONFLICT(setting_key) DO UPDATE SET setting_value=excluded.setting_value
                """,
                (str(reset_at),),
            )
        return reset_at

    def model_tool_stats(self) -> dict[str, Any]:
        """Aggregate model-attributed tool outcomes from the immutable timeline."""
        reset_at = self._tool_stats_reset_at()
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT run_id, step, type, timestamp, data_json
                FROM events
                WHERE timestamp >= ?
                ORDER BY run_id, rowid
                """,
                (reset_at,),
            ).fetchall()

        groups: dict[tuple[str, str], dict[str, Any]] = {}
        step_model: dict[tuple[str, int], tuple[str, str]] = {}
        run_models: dict[str, set[tuple[str, str]]] = {}
        completed_runs: set[str] = set()
        parsed_rows = [
            (row["run_id"], int(row["step"]), row["type"], float(row["timestamp"]), json.loads(row["data_json"]))
            for row in rows
        ]

        # A server can ignore the model field sent by the client (for example
        # an externally launched llama-server on a fixed model).  First learn
        # each run's requested -> server-reported mapping from successful
        # completions.  Later llm.error events have no response body, but can
        # still be attributed to that same real model instead of creating a
        # misleading second row for the requested alias.
        configured_by_step: dict[tuple[str, int], tuple[str, str]] = {}
        observed_models: dict[tuple[str, str, str], set[str]] = {}
        for run_id, step, event_type, _, data in parsed_rows:
            if event_type == "llm.start":
                configured_by_step[(run_id, step)] = (
                    str(data.get("provider", "unknown")), str(data.get("model", "unknown"))
                )
            elif event_type == "llm.end":
                provider, requested = configured_by_step.get((run_id, step), ("unknown", "unknown"))
                actual = str(data.get("model") or requested)
                observed_models.setdefault((run_id, provider, requested), set()).add(actual)

        def resolved_model(run_id: str, provider: str, requested: str, event_model: Any = None) -> str:
            if event_model:
                return str(event_model)
            candidates = observed_models.get((run_id, provider, requested), set())
            return next(iter(candidates)) if len(candidates) == 1 else requested

        def decision_reason(data: dict[str, Any]) -> str:
            """Read native reason_kind, with a best-effort historic fallback."""
            if data.get("reason_kind"):
                return str(data["reason_kind"])
            message = str(data.get("error", "")).lower()
            if "service unavailable" in message or "connection" in message or "timed out" in message:
                return "connection"
            if "routed to an unconnected" in message:
                return "route"
            if "outside its capability" in message or "tool is not allowed" in message:
                return "capability"
            if "without a matching outgoing" in message:
                return "workflow"
            if "model output" in message or "invalid response" in message:
                return "protocol"
            return "other"

        def bucket(key: tuple[str, str]) -> dict[str, Any]:
            if key not in groups:
                groups[key] = {
                    "provider": key[0], "model": key[1], "decisions": 0,
                    "decision_errors": 0,
                    "decision_error_breakdown": {},
                    "tool_calls_requested": 0, "tool_calls_validated": 0,
                    "tool_calls_started": 0,
                    "tool_calls_succeeded": 0, "tool_calls_failed": 0,
                    "tool_calls_blocked": 0,
                    "last_event_at": 0.0,
                }
            return groups[key]

        for run_id, step, event_type, timestamp, data in parsed_rows:
            key = step_model.get((run_id, step))
            if event_type == "llm.start":
                key = (str(data.get("provider", "unknown")), str(data.get("model", "unknown")))
                step_model[(run_id, step)] = key
                # The requested model is only a provisional identity. Wait
                # for llm.end or llm.error so this event cannot create a
                # phantom metrics row when a server reports a different id.
                continue
            elif event_type == "llm.end":
                provider = key[0] if key else "unknown"
                requested = key[1] if key else "unknown"
                key = (provider, resolved_model(run_id, provider, requested, data.get("model")))
                step_model[(run_id, step)] = key
                run_models.setdefault(run_id, set()).add(key)
                current = bucket(key)
                current["decisions"] += 1
                action = data.get("action", {})
                if action.get("action") == "call_tool":
                    current["tool_calls_requested"] += 1
                elif action.get("action") == "call_tools":
                    current["tool_calls_requested"] += len(action.get("tool_calls") or [])

            key = step_model.get((run_id, step))
            if key is None:
                continue
            if event_type != "llm.end":
                provider, requested = key
                key = (
                    provider,
                    resolved_model(
                        run_id,
                        provider,
                        requested,
                        data.get("model") if event_type == "llm.error" else None,
                    ),
                )
                step_model[(run_id, step)] = key
                run_models.setdefault(run_id, set()).add(key)
            current = bucket(key)
            current["last_event_at"] = max(timestamp, current["last_event_at"])
            if event_type == "llm.error":
                # Historic runs have no category; they were all generated by
                # the decision/validation path, so count them the same way.
                current["decision_errors"] += 1
                reason = decision_reason(data)
                breakdown = current["decision_error_breakdown"]
                breakdown[reason] = int(breakdown.get(reason, 0)) + 1
            elif event_type == "action.validate":
                action = data.get("action", {})
                if action.get("action") == "call_tool":
                    current["tool_calls_validated"] += 1
                elif action.get("action") == "call_tools":
                    current["tool_calls_validated"] += len(action.get("tool_calls") or [])
            elif event_type == "tool.start":
                current["tool_calls_started"] += 1
            elif event_type == "tool.end":
                current["tool_calls_succeeded"] += 1
            elif event_type == "tool.error":
                current["tool_calls_failed"] += 1
            elif event_type == "tool.rejected":
                current["tool_calls_blocked"] += len(data.get("calls") or [])
            elif event_type == "agent.finish":
                completed_runs.add(run_id)

        records: list[dict[str, Any]] = []
        for key, current in groups.items():
            attempted = current["tool_calls_succeeded"] + current["tool_calls_failed"]
            current["tool_execution_success_rate"] = (
                current["tool_calls_succeeded"] / attempted if attempted else None
            )
            requested = current["tool_calls_requested"]
            validated = current["tool_calls_validated"]
            current["proposal_validation_rate"] = (
                validated / requested if requested else None
            )
            current["plan_to_tool_success_rate"] = (
                current["tool_calls_succeeded"] / requested if requested else None
            )
            current["runs"] = sum(1 for models in run_models.values() if key in models)
            current["completed_runs"] = sum(
                1 for run_id in completed_runs if key in run_models.get(run_id, set())
            )
            records.append(current)

        records.sort(
            key=lambda item: (
                item["plan_to_tool_success_rate"] is None,
                -(item["plan_to_tool_success_rate"] or 0),
                -item["tool_calls_requested"],
            )
        )
        return {"reset_at": reset_at, "models": records}

    def close(self) -> None:
        with self._lock:
            self._connection.close()
