from __future__ import annotations

import os
import signal
import threading
import traceback
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path

from .pipeline import UserFacingError, explain_failure, run_pipeline, run_subprocess


class AnalysisJob:
    def __init__(self, config: dict, output_root: Path):
        self.id = uuid.uuid4().hex[:10]
        self.config = config
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        image_stem = Path(config["image_path"]).stem
        safe_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in image_stem)[:80]
        self.output_dir = output_root / f"{safe_stem}_{stamp}_{self.id[:4]}"
        self.status = "queued"
        self.stage = "Waiting to start"
        self.progress = 0
        self.error = None
        self.technical_error = None
        self.result = None
        self.logs = deque(maxlen=500)
        self.created_at = datetime.now().isoformat(timespec="seconds")
        self.finished_at = None
        self.process = None
        self.cancel_requested = False
        self._lock = threading.RLock()

    def update(self, stage: str, progress: int) -> None:
        with self._lock:
            self.stage = stage
            self.progress = max(self.progress, int(progress))

    def append_log(self, line: str) -> None:
        line = line.rstrip()
        if line:
            with self._lock:
                self.logs.append(line)

    def set_process(self, process) -> None:
        with self._lock:
            self.process = process

    def cancel(self) -> None:
        with self._lock:
            self.cancel_requested = True
            process = self.process
        if process is not None and process.poll() is None:
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
            except ProcessLookupError:
                pass

    def to_dict(self, include_logs: bool = True) -> dict:
        with self._lock:
            payload = {
                "id": self.id,
                "image_name": Path(self.config['image_path']).name,
                "status": self.status,
                "stage": self.stage,
                "progress": self.progress,
                "error": self.error,
                "technical_error": self.technical_error,
                "result": self.result,
                "output_dir": str(self.output_dir),
                "created_at": self.created_at,
                "finished_at": self.finished_at,
            }
            if include_logs:
                payload["logs"] = list(self.logs)
            return payload


class JobManager:
    def __init__(self, output_root: Path):
        self.output_root = output_root.expanduser().resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, AnalysisJob] = {}
        self._lock = threading.RLock()
        self._active_job_id: str | None = None

    def create(self, config: dict) -> AnalysisJob:
        requested_root = config.get("output_dir")
        output_root = Path(requested_root).expanduser().resolve() if requested_root else self.output_root
        try:
            output_root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise UserFacingError(f"Output folder cannot be created: {exc}") from exc
        with self._lock:
            if self._active_job_id:
                active = self.jobs.get(self._active_job_id)
                if active and active.status in {"queued", "running", "cancelling"}:
                    raise UserFacingError(
                        "Another analysis is currently running. Wait for it to finish or cancel it first."
                    )
            job = AnalysisJob(config, output_root)
            self.jobs[job.id] = job
            self._active_job_id = job.id
        thread = threading.Thread(target=self._run, args=(job,), daemon=True)
        thread.start()
        return job

    def get(self, job_id: str) -> AnalysisJob | None:
        with self._lock:
            return self.jobs.get(job_id)

    def list(self) -> list[dict]:
        with self._lock:
            jobs = list(self.jobs.values())
        return [job.to_dict(include_logs=False) for job in reversed(jobs)]

    def _run(self, job: AnalysisJob) -> None:
        job.status = "running"
        job.update("Inspecting TIFF geometry", 3)

        def update(stage: str, progress: int) -> None:
            if job.cancel_requested:
                raise UserFacingError("Analysis cancelled by the user.")
            job.update(stage, progress)

        def run_command(command: list[str], label: str, progress: int) -> None:
            if job.cancel_requested:
                raise UserFacingError("Analysis cancelled by the user.")
            job.append_log(f"[{label}] starting")
            process = run_subprocess(command, APP_ROOT)
            job.set_process(process)
            output_lines = []
            assert process.stdout is not None
            for line in process.stdout:
                output_lines.append(line.rstrip())
                job.append_log(f"[{label}] {line.rstrip()}")
            return_code = process.wait()
            job.set_process(None)
            if job.cancel_requested:
                raise UserFacingError("Analysis cancelled by the user.")
            if return_code != 0:
                tail = "\n".join(output_lines[-30:])
                raise RuntimeError(f"{label} exited with code {return_code}\n{tail}")
            job.update(job.stage, progress)
            job.append_log(f"[{label}] complete")

        try:
            job.result = run_pipeline(job.config, job.output_dir, run_command, update)
            job.status = "completed"
            job.update("Complete", 100)
        except UserFacingError as exc:
            if job.cancel_requested:
                job.status = "cancelled"
                job.error = "The analysis was cancelled. Partial outputs remain in the run folder."
            else:
                job.status = "failed"
                job.error = str(exc)
            job.technical_error = str(exc)
            job.append_log(str(exc))
        except Exception as exc:
            job.status = "failed"
            job.technical_error = str(exc)
            job.error = explain_failure(str(exc))
            job.append_log(traceback.format_exc())
        finally:
            job.finished_at = datetime.now().isoformat(timespec="seconds")
            with self._lock:
                if self._active_job_id == job.id:
                    self._active_job_id = None

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if not job:
            return False
        if job.status not in {"queued", "running", "cancelling"}:
            return True
        job.status = "cancelling"
        job.stage = "Stopping the active process"
        job.cancel()
        return True

    def shutdown(self) -> None:
        with self._lock:
            jobs = list(self.jobs.values())
        for job in jobs:
            if job.status in {"queued", "running", "cancelling"}:
                job.cancel()


APP_ROOT = Path(__file__).resolve().parents[1]
