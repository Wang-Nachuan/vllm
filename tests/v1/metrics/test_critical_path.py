# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from vllm.v1.metrics.critical_path import (
    CriticalPathMetrics,
    write_critical_path_record,
)

pytestmark = pytest.mark.cpu_test


def test_waits_and_forward_phases_are_disjoint():
    metrics = CriticalPathMetrics()
    metrics.start_lookup_wait("offload", 1.0)
    metrics.finish_lookup_wait(1.5)
    metrics.start_prefix_load(2.0)
    metrics.add_prefix_offload(0.25)
    metrics.finish_prefix_load(3.0)
    metrics.add_prefix_lookup(0.1)
    metrics.add_model_forward("prefill", 0.2)
    metrics.add_model_forward("decode", 0.3)

    assert metrics.prefix_lookup_s == pytest.approx(0.1)
    assert metrics.prefix_load_s == pytest.approx(0.75)
    assert metrics.prefix_offload_s == pytest.approx(0.75)
    assert metrics.prefill_s == pytest.approx(0.2)
    assert metrics.decode_s == pytest.approx(0.3)


def test_write_record_uses_external_id_and_residual(tmp_path):
    metrics = CriticalPathMetrics(prefill_s=1.0, decode_s=2.0)
    trace_path = tmp_path / "critical_path.jsonl"

    write_critical_path_record(
        str(trace_path),
        request_id="chatcmpl-mswea-task-1",
        server_e2e_s=5.0,
        metrics=metrics,
    )

    record = json.loads(trace_path.read_text())
    assert record["request_id"] == "chatcmpl-mswea-task-1"
    assert record["queue_scheduler_s"] == pytest.approx(2.0)
    assert "mixed_prefill_decode_s" not in record
