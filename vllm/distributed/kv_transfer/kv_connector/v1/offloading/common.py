# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, field

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)
from vllm.v1.kv_offload.worker.worker import TransferSpec

ReqId = str


@dataclass(frozen=True)
class LoadJobTiming:
    """Timing for one worker's completed cache-load transfer.

    Host timestamps use ``time.monotonic()`` and are therefore comparable
    across tensor-parallel worker processes on the same host. ``dma_elapsed_s``
    is measured by device events and excludes time queued behind earlier work.
    """

    worker_start_s: float
    worker_enqueue_s: float
    dma_elapsed_s: float
    completion_observed_s: float
    transfer_bytes: int


@dataclass
class TransferJob:
    """A transfer job bundling request context with transfer spec.

    Used for both loads and stores, keyed by scheduler-assigned job ID.
    The worker reports the job ID back when the transfer finishes,
    and the scheduler processes the completion.
    """

    req_id: ReqId
    transfer_spec: TransferSpec


@dataclass
class OffloadingConnectorMetadata(KVConnectorMetadata):
    # Keyed by scheduler-assigned job IDs.
    load_jobs: dict[int, TransferJob]
    store_jobs: dict[int, TransferJob]
    jobs_to_flush: set[int] | None = None


@dataclass
class OffloadingWorkerMetadata(KVConnectorWorkerMetadata):
    """Worker -> Scheduler metadata for completed transfer jobs.

    Each worker reports {job_id: 1} for newly completed transfer jobs
    (load or store). aggregate() sums counts across workers within a step.
    For load timing, aggregate() keeps the complete record from the worker
    that observed completion last, i.e. the tensor-parallel critical rank.
    Forced store-flush waits use the longest worker wait because the batch
    cannot proceed until every worker is ready.
    The scheduler accumulates across steps and processes
    a transfer completion only when count reaches num_workers.
    """

    completed_jobs: dict[int, int] = field(default_factory=dict)
    load_job_timings: dict[int, LoadJobTiming] = field(default_factory=dict)
    prefix_offload_wait_s: float = 0.0
    prefix_offload_wait_started_at_s: float | None = None

    def mark_completed(self, job_id: int) -> None:
        """Record a transfer job completion from this worker."""
        self.completed_jobs[job_id] = 1

    def record_prefix_offload_wait(
        self, started_at_s: float, duration_s: float
    ) -> None:
        """Record a forced store-flush wait on this model step."""
        duration_s = max(0.0, duration_s)
        if duration_s > self.prefix_offload_wait_s:
            self.prefix_offload_wait_s = duration_s
            self.prefix_offload_wait_started_at_s = started_at_s

    def record_load_job_timing(self, job_id: int, timing: LoadJobTiming) -> None:
        """Record timing for a cache-load job completed by this worker."""
        self.load_job_timings[job_id] = timing

    def aggregate(
        self, other: "KVConnectorWorkerMetadata"
    ) -> "KVConnectorWorkerMetadata":
        assert isinstance(other, OffloadingWorkerMetadata)

        merged = dict(self.completed_jobs)
        for job_id, v in other.completed_jobs.items():
            merged[job_id] = merged.get(job_id, 0) + v

        load_job_timings = dict(self.load_job_timings)
        for job_id, timing in other.load_job_timings.items():
            current = load_job_timings.get(job_id)
            if current is None or timing.completion_observed_s > (
                current.completion_observed_s
            ):
                load_job_timings[job_id] = timing

        self_offload_finished_at = (
            (self.prefix_offload_wait_started_at_s or 0.0)
            + self.prefix_offload_wait_s
        )
        other_offload_finished_at = (
            (other.prefix_offload_wait_started_at_s or 0.0)
            + other.prefix_offload_wait_s
        )
        if other_offload_finished_at > self_offload_finished_at:
            prefix_offload_wait_s = other.prefix_offload_wait_s
            prefix_offload_wait_started_at_s = (
                other.prefix_offload_wait_started_at_s
            )
        else:
            prefix_offload_wait_s = self.prefix_offload_wait_s
            prefix_offload_wait_started_at_s = self.prefix_offload_wait_started_at_s

        return OffloadingWorkerMetadata(
            completed_jobs=merged,
            load_job_timings=load_job_timings,
            prefix_offload_wait_s=prefix_offload_wait_s,
            prefix_offload_wait_started_at_s=prefix_offload_wait_started_at_s,
        )
