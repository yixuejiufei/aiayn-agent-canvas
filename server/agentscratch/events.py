"""Core event types emitted by the Agent interpreter."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

from .schemas import AgentEvent


@dataclass(slots=True)
class EventSink:
    run_id: str
    step: int = 0

    def emit(self, type_: str, node: str, data: dict | None = None) -> AgentEvent:
        return AgentEvent(
            event_id=str(uuid.uuid4()),
            run_id=self.run_id,
            step=self.step,
            type=type_,
            timestamp=time.time(),
            node=node,
            data=data or {},
        )

    def next_step(self) -> None:
        self.step += 1
