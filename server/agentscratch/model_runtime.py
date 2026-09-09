"""Safe local GGUF discovery and on-demand llama.cpp model serving.

Only files inside the configured model directory are considered.  The manager
never edits model files and never terminates a llama-server it did not start.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import httpx


_IGNORED_SUFFIXES = {".part", ".incomplete"}
_MIN_MODEL_BYTES = 100 * 1024 * 1024


class LocalModelRuntime:
    def __init__(self) -> None:
        self.model_root = Path(os.getenv("AGENTSCRATCH_MODELS_DIR", r"E:\AIModels")).resolve()
        self.llama_server = Path(
            os.getenv(
                "AGENTSCRATCH_LLAMA_SERVER",
                str(Path(__file__).resolve().parents[2] / "bin" / "llama-cpu" / "llama-server.exe"),
            )
        )
        self.llama_server_gpu = Path(
            os.getenv(
                "AGENTSCRATCH_LLAMA_SERVER_GPU",
                str(Path(__file__).resolve().parents[2] / "bin" / "llama-cuda" / "llama-server.exe"),
            )
        )
        self.host = "127.0.0.1"
        self.port = int(os.getenv("AGENTSCRATCH_MANAGED_LLM_PORT", "8081"))
        self.default_context_size = int(os.getenv("AGENTSCRATCH_DEFAULT_CONTEXT_SIZE", "8192"))
        self._process: subprocess.Popen[bytes] | None = None
        self._active: dict[str, Any] | None = None
        self._lock = threading.RLock()

    @staticmethod
    def _model_id(path: Path) -> str:
        stem = path.stem.lower()
        normalized = "".join(character if character.isalnum() else "-" for character in stem)
        return "-".join(part for part in normalized.split("-") if part)[:96]

    @staticmethod
    def _is_eligible_gguf(path: Path, size: int) -> bool:
        lower_name = path.name.lower()
        return not (
            path.suffix.lower() in _IGNORED_SUFFIXES
            or ".part" in lower_name
            or ".incomplete" in lower_name
            or size < _MIN_MODEL_BYTES
            or "mmproj" in lower_name
        )

    def discover(self) -> list[dict[str, Any]]:
        if not self.model_root.is_dir():
            return []
        profiles: list[dict[str, Any]] = []
        for path in self.model_root.rglob("*.gguf"):
            try:
                resolved = path.resolve()
                resolved.relative_to(self.model_root)
                size = resolved.stat().st_size
            except (OSError, ValueError):
                continue
            if not self._is_eligible_gguf(resolved, size):
                continue
            model_id = self._model_id(resolved)
            profiles.append(
                {
                    "id": model_id,
                    "label": resolved.stem,
                    "path": str(resolved),
                    "size_bytes": size,
                    "is_instruction_tuned": "instruct" in resolved.name.lower() or "chat" in resolved.name.lower(),
                    "active": bool(self._active and self._active["path"] == str(resolved)),
                }
            )
        # Filename collisions are uncommon but possible across quantizations;
        # preserve both by adding a deterministic suffix to the later record.
        used: set[str] = set()
        for profile in sorted(profiles, key=lambda item: item["label"].lower()):
            base_id = profile["id"]
            index = 2
            while profile["id"] in used:
                profile["id"] = f"{base_id}-{index}"
                index += 1
            used.add(profile["id"])
        return sorted(profiles, key=lambda item: item["label"].lower())

    def connection(self) -> tuple[str, str]:
        with self._lock:
            # A Python/API restart does not necessarily stop the child
            # llama-server on Windows.  Reconnect to that dedicated managed
            # port so the UI reports the actually loaded GGUF rather than the
            # unrelated qwen-tiny fallback on port 8080.
            self._adopt_running_server_locked()
            if self._active is not None:
                return f"http://{self.host}:{self.port}/v1", str(self._active["id"])
        return (
            os.getenv("AGENTSCRATCH_LLM_BASE_URL", "http://127.0.0.1:8080/v1").rstrip("/"),
            os.getenv("AGENTSCRATCH_LLM_MODEL", "qwen-tiny"),
        )

    def _adopt_running_server_locked(self) -> bool:
        """Reattach to a surviving AgentScratch llama-server, if one exists.

        This is deliberately observational: it never starts, stops, or
        reconfigures a process.  A later explicit model activation remains the
        only operation allowed to replace the serving model.
        """
        if self._active is not None:
            return True
        try:
            response = httpx.get(f"http://{self.host}:{self.port}/v1/models", timeout=0.35)
            response.raise_for_status()
            model_records = {
                str(item.get("id")): item
                for item in response.json().get("data", [])
                if isinstance(item, dict) and item.get("id")
            }
        except (httpx.HTTPError, ValueError, TypeError):
            return False
        for profile in self.discover():
            record = model_records.get(profile["id"])
            if record is not None:
                # The previous API process owns no longer exists, so `_process`
                # is intentionally left None.  The device is not reliably
                # recoverable from OpenAI-compatible /models metadata.
                metadata = record.get("meta", {})
                context_size = metadata.get("n_ctx") if isinstance(metadata, dict) else None
                self._active = {
                    **profile,
                    "device": "reconnected",
                    "context_size": int(context_size) if isinstance(context_size, int) and context_size > 0 else None,
                }
                return True
        return False

    def devices(self) -> list[dict[str, Any]]:
        """Expose only execution backends that can actually be launched."""
        gpu_server = self._gpu_server_path()
        return [
            {
                "id": "cpu",
                "label": "CPU",
                "available": self.llama_server.is_file(),
                "detail": "Bundled llama.cpp CPU runtime",
            },
            {
                "id": "gpu",
                "label": "GPU（CUDA）",
                "available": gpu_server is not None,
                "detail": str(gpu_server) if gpu_server else "Requires a CUDA-enabled llama.cpp llama-server.exe",
            },
        ]

    def _gpu_server_path(self) -> Path | None:
        """Find an explicitly configured or already-installed CUDA backend.

        LM Studio distributes llama.cpp backends in its user extension folder.
        Reusing that executable avoids a second CUDA installation while keeping
        the model process managed by AgentScratch.
        """
        candidates = [self.llama_server_gpu]
        lm_studio_backends = Path.home() / ".lmstudio" / "extensions" / "backends"
        if lm_studio_backends.is_dir():
            candidates.extend(sorted(
                lm_studio_backends.glob("llama.cpp-win-x86_64-nvidia-cuda*/llama-server.exe"),
                reverse=True,
            ))
        return next((candidate for candidate in candidates if candidate.is_file()), None)

    @staticmethod
    def _gpu_library_dirs(llama_server: Path) -> list[Path]:
        """Locate CUDA vendor DLLs shipped beside LM Studio backends."""
        backends_dir = llama_server.parent.parent
        vendor_dir = backends_dir / "vendor"
        if not vendor_dir.is_dir():
            return []
        return sorted({path.parent for path in vendor_dir.glob("*/cublas64_12.dll") if path.is_file()})

    def active_status(self) -> dict[str, Any]:
        base_url, model = self.connection()
        return {
            "base_url": base_url,
            "model": model,
            "managed": self._active is not None,
            "device": self._active.get("device", "cpu") if self._active else "external",
            "context_size": self._active.get("context_size") if self._active else None,
        }

    def ensure_selected(self, model_id: str | None, device: str = "cpu", context_size: int | None = None) -> bool:
        """Restore a canvas-selected GGUF after an API process restart.

        A workflow stores its selected profile id.  Without this guard a
        restarted API would silently fall back to an unrelated external model
        on port 8080 while still labelling the run with the selected id.
        ``False`` means the id belongs to an external service, so the existing
        configured connection remains authoritative.
        """
        if not model_id:
            return False
        profiles = {profile["id"]: profile for profile in self.discover()}
        if model_id not in profiles:
            return False
        self.activate(model_id, device, context_size=context_size)
        return True

    def activate(self, model_id: str, device: str = "cpu", context_size: int | None = None) -> dict[str, Any]:
        profiles = {profile["id"]: profile for profile in self.discover()}
        profile = profiles.get(model_id)
        if profile is None:
            raise ValueError("unknown or incomplete local GGUF model")
        if device not in {"cpu", "gpu"}:
            raise ValueError("unknown local execution device")
        effective_context_size = context_size if context_size is not None else self.default_context_size
        if not isinstance(effective_context_size, int) or not 256 <= effective_context_size <= 131072:
            raise ValueError("context size must be between 256 and 131072 tokens")
        llama_server = self.llama_server if device == "cpu" else self._gpu_server_path()
        if llama_server is None or not llama_server.is_file():
            if device == "gpu":
                raise RuntimeError(
                    "CUDA llama-server executable was not found. Install it at bin/llama-cuda/llama-server.exe "
                    "or set AGENTSCRATCH_LLAMA_SERVER_GPU."
                )
            raise RuntimeError(f"llama-server executable was not found: {llama_server}")

        with self._lock:
            if (
                self._active
                and self._active["path"] == profile["path"]
                and self._active.get("device") == device
                and self._active.get("context_size") == effective_context_size
                and self._process
                and self._process.poll() is None
            ):
                self._stop_orphaned_servers_locked(exclude_pid=self._process.pid)
                return {**profile, "base_url": self.connection()[0], "managed": True, "device": device, "context_size": effective_context_size}
            self._stop_managed_locked()
            self._stop_orphaned_servers_locked()
            command = [
                str(llama_server),
                "-m", profile["path"],
                "--alias", profile["id"],
                "--host", self.host,
                "--port", str(self.port),
                "--ctx-size", str(effective_context_size),
            ]
            if device == "gpu":
                # llama.cpp offloads as many transformer layers as the VRAM
                # allows; unsupported layers safely remain on the CPU.
                command.extend(["--n-gpu-layers", "99"])
            creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            environment = None
            if device == "gpu":
                # LM Studio keeps CUDA vendor libraries in a sibling directory
                # rather than directly beside llama-server.exe.  Prepend it
                # only for this child process, never altering global PATH.
                library_dirs = self._gpu_library_dirs(llama_server)
                if library_dirs:
                    environment = os.environ.copy()
                    environment["PATH"] = os.pathsep.join([*(str(path) for path in library_dirs), environment.get("PATH", "")])
            self._process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creation_flags,
                cwd=str(llama_server.parent),
                env=environment,
            )
            try:
                self._wait_until_ready(profile["id"])
            except Exception:
                self._stop_managed_locked()
                raise
            self._active = {**profile, "device": device, "context_size": effective_context_size}
            return {**profile, "base_url": self.connection()[0], "managed": True, "device": device, "context_size": effective_context_size}

    def _wait_until_ready(self, expected_id: str) -> None:
        endpoint = f"http://{self.host}:{self.port}/v1/models"
        deadline = time.monotonic() + 90
        last_error = "starting local model service"
        while time.monotonic() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError("llama-server exited while loading the selected model")
            try:
                response = httpx.get(endpoint, timeout=2)
                response.raise_for_status()
                model_ids = [item.get("id") for item in response.json().get("data", [])]
                if expected_id in model_ids:
                    return
                last_error = "llama-server started without the selected model alias"
            except httpx.HTTPError as exc:
                last_error = str(exc)
            time.sleep(0.5)
        raise RuntimeError(f"timed out loading local model: {last_error}")

    def _stop_managed_locked(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)
        self._process = None
        self._active = None

    def _stop_orphaned_servers_locked(self, exclude_pid: int | None = None) -> None:
        """Release stale AgentScratch llama-server processes on our managed port.

        The runtime's Python process may be restarted while a child model server
        survives.  Port 8081 is reserved for this manager, so any remaining
        llama-server listener there is a prior managed instance, not an
        arbitrary external model service (the fallback uses port 8080).
        """
        if os.name != "nt":
            return
        try:
            output = subprocess.run(
                ["netstat", "-ano", "-p", "tcp"],
                capture_output=True,
                text=True,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            ).stdout
        except OSError:
            return
        candidate_pids: set[int] = set()
        for line in output.splitlines():
            fields = line.split()
            if len(fields) < 5 or fields[0].upper() != "TCP" or fields[-2].upper() != "LISTENING":
                continue
            if not fields[1].endswith(f":{self.port}"):
                continue
            try:
                candidate_pids.add(int(fields[-1]))
            except ValueError:
                continue
        for pid in candidate_pids:
            if pid == exclude_pid:
                continue
            # 8081 is dedicated to the manager, but retain a final image-name
            # check before stopping a process.
            try:
                task = subprocess.run(
                    ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                    capture_output=True,
                    text=True,
                    check=False,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                ).stdout.lower()
            except OSError:
                continue
            if "llama-server.exe" not in task:
                continue
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/F"],
                capture_output=True,
                text=True,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )


local_model_runtime = LocalModelRuntime()
