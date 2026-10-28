import fcntl
import json
from pathlib import Path
from typing import Dict, List, Optional, Literal
from pydantic import BaseModel


class ExperimentRun(BaseModel):
    """One (article_id, phase, model) run result."""
    article_id: str
    phase: int
    model: str
    status: Literal["ok", "failed"]
    error: Optional[str] = None


class ExperimentTracker:
    """Track completed (article, phase, model) runs."""

    def __init__(self, log_path: str, experiments_root: str):
        self.log_path = Path(log_path)
        self.experiments_root = Path(experiments_root)
        self.experiments: Dict[str, List[ExperimentRun]] = {}
        self._load()

    def _load(self):
        if not self.log_path.exists():
            self.experiments = {}
            return

        # Locked read: another batch may be mid-write inside its LOCK_EX block.
        with open(self.log_path, "r", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                raw = f.read()
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        try:
            data = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            data = {}
        self.experiments = {}
        for article_id, runs in data.items():
            self.experiments[article_id] = [
                ExperimentRun(**run) for run in runs
            ]

    def _atomic_append(self, run: ExperimentRun) -> None:
        """Atomically append run to the log file; safe under concurrent batches.

        Locks the file (fcntl.LOCK_EX), re-reads disk state, appends, writes
        back. Without the re-read, parallel processes would each save their
        own stale in-memory snapshot and overwrite each other's records.
        """
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a+", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.seek(0)
                raw = f.read()
                try:
                    data = json.loads(raw) if raw.strip() else {}
                except json.JSONDecodeError:
                    data = {}

                aid = run.article_id
                if aid not in data:
                    data[aid] = []
                data[aid].append(run.model_dump())

                # Refresh in-memory cache so is_completed reflects the latest disk state
                self.experiments = {
                    k: [ExperimentRun(**r) for r in runs] for k, runs in data.items()
                }

                f.seek(0)
                f.truncate()
                json.dump(data, f, indent=2)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def is_completed(self, article_id: str, phase: int, model: str) -> bool:
        if article_id not in self.experiments:
            return False

        for run in self.experiments[article_id]:
            if run.phase == phase and run.model == model and run.status == "ok":
                return True

        return False

    def record_success(self, article_id: str, phase: int, model: str):
        run = ExperimentRun(
            article_id=article_id,
            phase=phase,
            model=model,
            status="ok",
        )
        self._atomic_append(run)

    def record_failure(self, article_id: str, phase: int, model: str, error: str):
        run = ExperimentRun(
            article_id=article_id,
            phase=phase,
            model=model,
            status="failed",
            error=error,
        )
        self._atomic_append(run)

    def get_output_paths(self, article_id: str, phase: int, model: str) -> Dict[str, str]:
        output_dir = self.experiments_root / article_id / f"phase{phase}" / model

        return {
            "output_dir": str(output_dir),
            "output_file": str(output_dir / "output.json"),
            "log_file": str(output_dir / "log.json"),
        }

    def get_stats(self, phase: Optional[int] = None, model: Optional[str] = None) -> Dict:
        all_runs = []
        for runs in self.experiments.values():
            all_runs.extend(runs)
        if phase is not None:
            all_runs = [r for r in all_runs if r.phase == phase]
        if model is not None:
            all_runs = [r for r in all_runs if r.model == model]

        return {
            "total": len(all_runs),
            "ok": sum(1 for r in all_runs if r.status == "ok"),
            "failed": sum(1 for r in all_runs if r.status == "failed"),
        }