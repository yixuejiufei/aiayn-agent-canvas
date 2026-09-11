"""Public canvas application entry point."""

from __future__ import annotations

import os
from pathlib import Path

# The canvas owns the default 8081 managed llama.cpp service and its bundled
# runtime. Set configuration before the shared runtime singleton is imported.
os.environ.setdefault("AGENTSCRATCH_RUNTIME_ROOT", str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("AGENTSCRATCH_MANAGED_LLM_PORT", "8081")

from agentscratch.api import app

__all__ = ["app"]
