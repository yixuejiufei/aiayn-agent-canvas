"""Keep pytest runs away from the learner's durable local SQLite database."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


_test_database_dir = Path(tempfile.mkdtemp(prefix="agentscratch-pytest-"))
os.environ["AGENTSCRATCH_DB_PATH"] = str(_test_database_dir / "agentscratch-test.db")
