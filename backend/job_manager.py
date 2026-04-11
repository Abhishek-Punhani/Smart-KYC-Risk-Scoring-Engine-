from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from pipeline import run_pipeline


class JobManager:
    def __init__(self, base_dir: Path, max_workers: int = 2) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = Lock()
        self.jobs: dict[str, dict[str, Any]] = {}

    def create_run(
        self, dataset_path: Path, config: dict[str, Any] | None = None
    ) -> str:
        run_id = uuid4().hex[:12]
        run_dir = self.base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        with self._lock:
            self.jobs[run_id] = {
                "run_id": run_id,
                "status": "queued",
                "stage": "queued",
                "progress": 0,
                "started_at": datetime.utcnow().isoformat() + "Z",
                "finished_at": None,
                "error": None,
                "summary": None,
                "run_dir": str(run_dir),
            }

        self.executor.submit(self._execute, run_id, dataset_path, run_dir, config or {})
        return run_id

    def _execute(
        self, run_id: str, dataset_path: Path, run_dir: Path, config: dict[str, Any]
    ) -> None:
        try:
            self._update(run_id, status="running", stage="pipeline", progress=10)
            summary = run_pipeline(
                dataset_path=dataset_path, out_dir=run_dir, config=config
            )
            self._update(
                run_id,
                status="completed",
                stage="completed",
                progress=100,
                finished_at=datetime.utcnow().isoformat() + "Z",
                summary=summary,
            )
        except Exception as e:
            self._update(
                run_id,
                status="failed",
                stage="failed",
                progress=100,
                finished_at=datetime.utcnow().isoformat() + "Z",
                error=str(e),
            )

    def _update(self, run_id: str, **updates: Any) -> None:
        with self._lock:
            self.jobs[run_id].update(updates)

    def get(self, run_id: str) -> dict[str, Any] | None:
        return self.jobs.get(run_id)

    def list_runs(self) -> list[dict[str, Any]]:
        return sorted(self.jobs.values(), key=lambda x: x["started_at"], reverse=True)
