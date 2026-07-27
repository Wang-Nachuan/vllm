# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Thread pool:
    Two queues (load, store) and two sets of threads:
      - Load-priority threads: drain the load queue first, then the store queue.
      - Store-priority threads: drain the store queue first, then the load queue.
    Load jobs are enqueued to the load queue; store jobs to the store queue.
"""

import threading
import time
from collections import deque
from collections.abc import Callable, Iterable

from vllm.logger import init_logger
from vllm.v1.kv_offload.tiering.base import JobId, JobResult, JobTiming

logger = init_logger(__name__)


class JobState:
    """
    Thread-safe completion tracker for a set of per-block I/O tasks.

    Each task calls task_done(success) when it finishes.
    """

    __slots__ = (
        "_job_id",
        "_n_tasks",
        "_completed",
        "_success",
        "_submitted_at",
        "_last_task_started_at",
        "_last_task_finished_at",
        "_lock",
    )

    def __init__(self, job_id: JobId, n_tasks: int, submitted_at: float) -> None:
        assert n_tasks > 0
        self._job_id: JobId = job_id
        self._n_tasks = n_tasks
        self._completed = 0
        self._success = True
        self._submitted_at = submitted_at
        self._last_task_started_at = submitted_at
        self._last_task_finished_at = submitted_at
        self._lock = threading.Lock()

    @property
    def job_id(self) -> JobId:
        return self._job_id

    def task_done(
        self,
        success: bool,
        started_at: float,
        finished_at: float,
    ) -> JobResult | None:
        """Record a task and return the result when the whole job completes."""
        with self._lock:
            self._completed += 1
            if not success:
                self._success = False
            if finished_at >= self._last_task_finished_at:
                self._last_task_started_at = started_at
                self._last_task_finished_at = finished_at
            if self._completed != self._n_tasks:
                return None
            return JobResult(
                job_id=self._job_id,
                success=self._success,
                timing=JobTiming(
                    submitted_at=self._submitted_at,
                    last_task_started_at=self._last_task_started_at,
                    last_task_finished_at=self._last_task_finished_at,
                ),
            )


class DualQueueThreadPool:
    """
    Thread pool with two task queues (load and store) and two thread groups.

    Load-priority threads drain the load queue first, then fall back to the
    store queue.  Store-priority threads do the reverse.  Both queues share
    a single condition variable.
    """

    def __init__(
        self,
        n_read_threads: int,
        n_write_threads: int,
        thread_name_prefix: str = "fs_secondary_tier",
    ) -> None:
        self._load_q: deque = deque()
        self._store_q: deque = deque()
        self._condition = threading.Condition(threading.Lock())
        self._stop = False
        self._threads: list[threading.Thread] = []
        self._finished_q: deque[JobResult] = deque()

        for i in range(n_read_threads):
            t = threading.Thread(
                target=self._worker,
                args=(True,),
                name=f"{thread_name_prefix}_l{i}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)

        for i in range(n_write_threads):
            t = threading.Thread(
                target=self._worker,
                args=(False,),
                name=f"{thread_name_prefix}_s{i}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)

    def enqueue_load(
        self,
        job_id: JobId,
        n_tasks: int,
        tasks: Iterable[Callable],
    ) -> None:
        """Enqueue load tasks for a job (high-priority for load-priority threads)."""
        tasks = tuple(tasks)
        assert len(tasks) == n_tasks
        state = JobState(job_id, n_tasks, time.monotonic())
        with self._condition:
            for fn in tasks:
                self._load_q.append((fn, state))
            self._condition.notify(n_tasks)

    def enqueue_store(
        self,
        job_id: JobId,
        n_tasks: int,
        tasks: Iterable[Callable],
    ) -> None:
        """Enqueue store tasks for a job (high-priority for store-priority threads)."""
        tasks = tuple(tasks)
        assert len(tasks) == n_tasks
        state = JobState(job_id, n_tasks, time.monotonic())
        with self._condition:
            for fn in tasks:
                self._store_q.append((fn, state))
            self._condition.notify(n_tasks)

    def get_finished(self) -> list[JobResult]:
        jobs = []
        while self._finished_q:
            jobs.append(self._finished_q.popleft())
        return jobs

    def shutdown(self, wait: bool = True) -> None:
        with self._condition:
            self._stop = True
            self._load_q.clear()
            self._store_q.clear()
            self._condition.notify_all()
        if wait:
            for t in self._threads:
                t.join()

    def _worker(self, load_priority: bool) -> None:
        # Wait for tasks, process from primary queue first, fall back to secondary.
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._stop or self._load_q or self._store_q
                )
                if self._stop:
                    return
                primary = self._load_q if load_priority else self._store_q
                secondary = self._store_q if load_priority else self._load_q
                task, state = primary.popleft() if primary else secondary.popleft()
            started_at = time.monotonic()
            try:
                task()
                success = True
            except Exception as exc:
                logger.error(
                    "Job %s block I/O failed: %s",
                    state.job_id,
                    exc,
                )
                success = False

            result = state.task_done(success, started_at, time.monotonic())
            if result is not None:
                self._finished_q.append(result)
