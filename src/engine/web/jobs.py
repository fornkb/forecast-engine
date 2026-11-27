"""Background jobs for slow work (AI calls, news and weather fetches).

A review can take a minute, so the page submits a job and polls for it.
Jobs live in memory: one local operator, one process. One job per client at a
time, so two clicks never pay for the same run twice.
"""

from __future__ import annotations

import threading
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from engine.reason.llm import LLMError
from engine.reason.schemas import ReasoningError

JobKind = Literal["review", "estimate", "onboard", "answers", "forecast", "backtest"]
LABEL = {
    "review": "Reviewing the forecast",
    "estimate": "Estimating cost",
    "onboard": "Drafting the company profile",
    "answers": "Applying your answers",
    "forecast": "Refreshing the forecast",
    "backtest": "Running the backtest",
}


class Busy(RuntimeError):
    pass


@dataclass
class Job:
    id: str
    client_id: str
    kind: JobKind
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    status: Literal["running", "done", "failed"] = "running"
    result: Any = None
    error: str = ""

    @property
    def label(self) -> str:
        return LABEL[self.kind]

    @property
    def seconds(self) -> float:
        end = self.finished_at or datetime.now(UTC)
        return round((end - self.started_at).total_seconds(), 1)


class JobRunner:
    def __init__(self, inline: bool = False, keep: int = 50) -> None:
        self.inline, self.keep = inline, keep
        self.jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._pool = None if inline else ThreadPoolExecutor(max_workers=2)

    def running(self, client_id: str) -> Job | None:
        return next(
            (j for j in self.jobs.values() if j.client_id == client_id and j.status == "running"),
            None,
        )

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def submit(self, client_id: str, kind: JobKind, work: Callable[[], Any]) -> Job:
        with self._lock:
            if (busy := self.running(client_id)) is not None:
                raise Busy(f"{busy.label} is still running for {client_id}.")
            job = Job(uuid4().hex[:10], client_id, kind)
            self.jobs[job.id] = job
            for old in list(self.jobs)[: max(0, len(self.jobs) - self.keep)]:
                if self.jobs[old].status != "running":
                    del self.jobs[old]
        if self._pool is None:
            self._execute(job, work)
        else:
            self._pool.submit(self._execute, job, work)
        return job

    @staticmethod
    def _execute(job: Job, work: Callable[[], Any]) -> None:
        try:
            job.result = work()
            job.status = "done"
        except (ReasoningError, LLMError, ValueError) as exc:
            job.error, job.status = str(exc), "failed"
        except Exception as exc:  # shown on the page instead of a silent dead job
            job.error = f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc(limit=4)}"
            job.status = "failed"
        finally:
            job.finished_at = datetime.now(UTC)
