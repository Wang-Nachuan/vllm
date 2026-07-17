# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
from dataclasses import dataclass, field
from typing import Literal

CriticalPathWaitReason = Literal["load", "offload"]
ModelForwardPhase = Literal["prefill", "decode"]


@dataclass(slots=True)
class CriticalPathMetrics:
    """Request-visible critical-path latency accumulated by the scheduler."""

    prefix_lookup_s: float = 0.0
    prefix_load_s: float = 0.0
    prefix_offload_s: float = 0.0
    prefill_s: float = 0.0
    decode_s: float = 0.0
    _lookup_wait_started_at: float | None = field(default=None, repr=False)
    _lookup_wait_reason: CriticalPathWaitReason | None = field(
        default=None, repr=False
    )
    _prefix_load_started_at: float | None = field(default=None, repr=False)
    _prefix_load_excluded_s: float = field(default=0.0, repr=False)

    def add_prefix_lookup(self, duration_s: float) -> None:
        self.prefix_lookup_s += max(0.0, duration_s)

    def start_lookup_wait(
        self, reason: CriticalPathWaitReason, now: float
    ) -> None:
        assert self._lookup_wait_started_at is None
        self._lookup_wait_started_at = now
        self._lookup_wait_reason = reason

    def finish_lookup_wait(self, now: float) -> None:
        if self._lookup_wait_started_at is None:
            return
        duration_s = max(0.0, now - self._lookup_wait_started_at)
        if self._lookup_wait_reason == "load":
            self.prefix_load_s += duration_s
        else:
            assert self._lookup_wait_reason == "offload"
            self.prefix_offload_s += duration_s
        self._lookup_wait_started_at = None
        self._lookup_wait_reason = None

    def start_prefix_load(self, now: float) -> None:
        assert self._prefix_load_started_at is None
        self._prefix_load_started_at = now
        self._prefix_load_excluded_s = 0.0

    def finish_prefix_load(self, now: float) -> None:
        if self._prefix_load_started_at is None:
            return
        self.prefix_load_s += max(
            0.0,
            now - self._prefix_load_started_at - self._prefix_load_excluded_s,
        )
        self._prefix_load_started_at = None
        self._prefix_load_excluded_s = 0.0

    def add_prefix_offload(self, duration_s: float) -> None:
        duration_s = max(0.0, duration_s)
        self.prefix_offload_s += duration_s
        if self._prefix_load_started_at is not None:
            self._prefix_load_excluded_s += duration_s

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


def write_critical_path_record(
    trace_path: str,
    request_id: str,
    server_e2e_s: float,
    metrics: CriticalPathMetrics,
) -> None:
    """Append one completed request record with a single atomic write."""
    server_e2e_s = max(0.0, server_e2e_s)
    accounted_s = (
        metrics.prefix_lookup_s
        + metrics.prefix_load_s
        + metrics.prefix_offload_s
        + metrics.prefill_s
        + metrics.decode_s
    )
    record = {
        "request_id": request_id,
        "server_e2e_s": server_e2e_s,
        "prefix_lookup_s": metrics.prefix_lookup_s,
        "prefix_load_s": metrics.prefix_load_s,
        "prefix_offload_s": metrics.prefix_offload_s,
        "prefill_s": metrics.prefill_s,
        "decode_s": metrics.decode_s,
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
