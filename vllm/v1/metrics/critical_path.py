# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
from dataclasses import dataclass, field
from typing import Literal

CriticalPathWaitReason = Literal[
    "owned_l2",
    "shared_l1",
    "shared_l2",
    "unattributed_load",
    "offload",
]
ModelForwardPhase = Literal["prefill", "decode"]


@dataclass(slots=True)
class CriticalPathMetrics:
    """Request-visible critical-path latency accumulated by the scheduler."""

    prefix_lookup_s: float = 0.0
    prefix_load_s: float = 0.0
    prefix_offload_s: float = 0.0
    prefill_s: float = 0.0
    decode_s: float = 0.0
    prefix_load_l2_admission_s: float = 0.0
    prefix_load_l2_io_queue_s: float = 0.0
    prefix_load_l2_ssd_read_s: float = 0.0
    prefix_load_l2_completion_wait_s: float = 0.0
    prefix_load_l1_prepare_dispatch_s: float = 0.0
    prefix_load_l1_h2d_dma_s: float = 0.0
    prefix_load_l1_control_wait_s: float = 0.0
    prefix_load_shared_l2_wait_s: float = 0.0
    prefix_load_shared_l1_wait_s: float = 0.0
    prefix_load_unattributed_s: float = 0.0
    _lookup_wait_started_at: float | None = field(default=None, repr=False)
    _lookup_wait_reason: CriticalPathWaitReason | None = field(
        default=None, repr=False
    )
    _prefix_load_started_at: float | None = field(default=None, repr=False)
    _prefix_load_excluded_intervals: list[tuple[float, float]] = field(
        default_factory=list, repr=False
    )
    _prefix_load_unpositioned_excluded_s: float = field(default=0.0, repr=False)
    _l1_worker_enqueue_s: float | None = field(default=None, repr=False)
    _l1_dma_elapsed_s: float | None = field(default=None, repr=False)
    _owned_l2_wait_intervals: list[tuple[float, float]] = field(
        default_factory=list, repr=False
    )

    def add_prefix_lookup(self, duration_s: float) -> None:
        self.prefix_lookup_s += max(0.0, duration_s)

    def add_synchronous_l2_admission(self, duration_s: float) -> None:
        """Charge in-lookup promotion setup to L2 admission, not lookup."""
        duration_s = max(0.0, duration_s)
        self.prefix_load_s += duration_s
        self.prefix_load_l2_admission_s += duration_s

    def start_lookup_wait(
        self, reason: CriticalPathWaitReason, now: float
    ) -> None:
        assert self._lookup_wait_started_at is None
        self._lookup_wait_started_at = now
        self._lookup_wait_reason = reason

    def finish_lookup_wait(self, now: float) -> None:
        if self._lookup_wait_started_at is None:
            return
        started_at = self._lookup_wait_started_at
        duration_s = max(0.0, now - started_at)
        if self._lookup_wait_reason == "shared_l1":
            self.prefix_load_s += duration_s
            self.prefix_load_shared_l1_wait_s += duration_s
        elif self._lookup_wait_reason == "shared_l2":
            self.prefix_load_s += duration_s
            self.prefix_load_shared_l2_wait_s += duration_s
        elif self._lookup_wait_reason == "owned_l2":
            self.prefix_load_s += duration_s
            self._owned_l2_wait_intervals.append((started_at, now))
        elif self._lookup_wait_reason == "unattributed_load":
            self.prefix_load_s += duration_s
            self.prefix_load_unattributed_s += duration_s
        else:
            assert self._lookup_wait_reason == "offload"
            self.prefix_offload_s += duration_s
        self._lookup_wait_started_at = None
        self._lookup_wait_reason = None

    def start_prefix_load(self, now: float) -> None:
        assert self._prefix_load_started_at is None
        self._prefix_load_started_at = now
        self._prefix_load_excluded_intervals = []
        self._prefix_load_unpositioned_excluded_s = 0.0
        self._l1_worker_enqueue_s = None
        self._l1_dma_elapsed_s = None

    def record_l1_load_timing(
        self,
        worker_enqueue_s: float,
        dma_elapsed_s: float,
    ) -> None:
        """Record the critical tensor-parallel rank for an owned L1 load."""
        self._l1_worker_enqueue_s = worker_enqueue_s
        self._l1_dma_elapsed_s = max(0.0, dma_elapsed_s)

    def finish_prefix_load(self, now: float) -> None:
        if self._prefix_load_started_at is None:
            return
        load_started_at = self._prefix_load_started_at
        positioned_excluded_s = self._overlap_s(
            self._prefix_load_excluded_intervals, load_started_at, now
        )
        excluded_s = (
            positioned_excluded_s + self._prefix_load_unpositioned_excluded_s
        )
        duration_s = max(
            0.0,
            now - load_started_at - excluded_s,
        )
        self.prefix_load_s += duration_s

        if self._l1_worker_enqueue_s is None or self._l1_dma_elapsed_s is None:
            self.prefix_load_unattributed_s += duration_s
        else:
            prepare_dispatch_s = min(
                duration_s,
                max(
                    0.0,
                    self._l1_worker_enqueue_s
                    - load_started_at
                    - self._overlap_s(
                        self._prefix_load_excluded_intervals,
                        load_started_at,
                        self._l1_worker_enqueue_s,
                    )
                    - self._prefix_load_unpositioned_excluded_s,
                ),
            )
            remaining_s = duration_s - prepare_dispatch_s
            dma_s = min(remaining_s, self._l1_dma_elapsed_s)
            self.prefix_load_l1_prepare_dispatch_s += prepare_dispatch_s
            self.prefix_load_l1_h2d_dma_s += dma_s
            self.prefix_load_l1_control_wait_s += remaining_s - dma_s

        self._prefix_load_started_at = None
        self._prefix_load_excluded_intervals = []
        self._prefix_load_unpositioned_excluded_s = 0.0
        self._l1_worker_enqueue_s = None
        self._l1_dma_elapsed_s = None

    @staticmethod
    def _overlap_s(
        intervals: list[tuple[float, float]], start: float, end: float
    ) -> float:
        return sum(
            max(0.0, min(interval_end, end) - max(interval_start, start))
            for interval_start, interval_end in intervals
        )

    def record_l2_promotion_timing(
        self,
        initiated_at: float,
        submitted_at: float | None,
        last_task_started_at: float | None,
        last_task_finished_at: float | None,
    ) -> None:
        """Split owned L2 wait using the promotion's critical I/O branch."""
        intervals = self._owned_l2_wait_intervals
        if not intervals:
            return
        self._owned_l2_wait_intervals = []

        if (
            submitted_at is None
            or last_task_started_at is None
            or last_task_finished_at is None
        ):
            self.prefix_load_unattributed_s += sum(
                max(0.0, end - start) for start, end in intervals
            )
            return

        submitted_at = max(initiated_at, submitted_at)
        last_task_started_at = max(submitted_at, last_task_started_at)
        last_task_finished_at = max(last_task_started_at, last_task_finished_at)
        self.prefix_load_l2_admission_s += self._overlap_s(
            intervals, initiated_at, submitted_at
        )
        self.prefix_load_l2_io_queue_s += self._overlap_s(
            intervals, submitted_at, last_task_started_at
        )
        self.prefix_load_l2_ssd_read_s += self._overlap_s(
            intervals, last_task_started_at, last_task_finished_at
        )
        self.prefix_load_l2_completion_wait_s += self._overlap_s(
            intervals, last_task_finished_at, float("inf")
        )

        attributed_s = self._overlap_s(intervals, initiated_at, float("inf"))
        total_s = sum(max(0.0, end - start) for start, end in intervals)
        self.prefix_load_unattributed_s += max(0.0, total_s - attributed_s)

    def add_prefix_offload(
        self, duration_s: float, started_at_s: float | None = None
    ) -> None:
        duration_s = max(0.0, duration_s)
        self.prefix_offload_s += duration_s
        if self._prefix_load_started_at is not None:
            if started_at_s is None:
                self._prefix_load_unpositioned_excluded_s += duration_s
            else:
                self._prefix_load_excluded_intervals.append(
                    (started_at_s, started_at_s + duration_s)
                )

    def add_model_forward(
        self, phase: ModelForwardPhase, duration_s: float
    ) -> None:
        duration_s = max(0.0, duration_s)
        if phase == "prefill":
            self.prefill_s += duration_s
        else:
            assert phase == "decode"
            self.decode_s += duration_s

    def finish_pending_waits(self, now: float) -> None:
        self.finish_lookup_wait(now)
        self.finish_prefix_load(now)
        if self._owned_l2_wait_intervals:
            self.prefix_load_unattributed_s += sum(
                max(0.0, end - start)
                for start, end in self._owned_l2_wait_intervals
            )
            self._owned_l2_wait_intervals = []


def write_critical_path_record(
    trace_path: str,
    request_id: str,
    server_e2e_s: float,
    metrics: CriticalPathMetrics,
) -> None:
    """Append one completed request record with a single atomic write."""
    server_e2e_s = max(0.0, server_e2e_s)
    non_offload_accounted_s = (
        metrics.prefix_lookup_s
        + metrics.prefix_load_s
        + metrics.prefill_s
        + metrics.decode_s
    )
    # A forced offload wait can overlap an asynchronously executing model
    # forward. Only the part that fits on the remaining request critical path
    # contributes to E2E latency.
    prefix_offload_s = min(
        metrics.prefix_offload_s,
        max(0.0, server_e2e_s - non_offload_accounted_s),
    )
    accounted_s = non_offload_accounted_s + prefix_offload_s
    record = {
        "request_id": request_id,
        "server_e2e_s": server_e2e_s,
        "prefix_lookup_s": metrics.prefix_lookup_s,
        "prefix_load_s": metrics.prefix_load_s,
        "prefix_offload_s": prefix_offload_s,
        "prefill_s": metrics.prefill_s,
        "decode_s": metrics.decode_s,
        "prefix_load_l2_admission_s": metrics.prefix_load_l2_admission_s,
        "prefix_load_l2_io_queue_s": metrics.prefix_load_l2_io_queue_s,
        "prefix_load_l2_ssd_read_s": metrics.prefix_load_l2_ssd_read_s,
        "prefix_load_l2_completion_wait_s": (
            metrics.prefix_load_l2_completion_wait_s
        ),
        "prefix_load_l1_prepare_dispatch_s": (
            metrics.prefix_load_l1_prepare_dispatch_s
        ),
        "prefix_load_l1_h2d_dma_s": metrics.prefix_load_l1_h2d_dma_s,
        "prefix_load_l1_control_wait_s": metrics.prefix_load_l1_control_wait_s,
        "prefix_load_shared_l2_wait_s": metrics.prefix_load_shared_l2_wait_s,
        "prefix_load_shared_l1_wait_s": metrics.prefix_load_shared_l1_wait_s,
        "prefix_load_unattributed_s": metrics.prefix_load_unattributed_s,
        # This is a request-visible residual. It includes queue/admission,
        # scheduler and executor overhead, input preparation, sampling, and
        # output processing inside the engine core.
        "queue_scheduler_s": max(0.0, server_e2e_s - accounted_s),
    }
    payload = (json.dumps(record, separators=(",", ":")) + "\n").encode()
    fd = os.open(trace_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(fd, payload)
    finally:
        os.close(fd)
