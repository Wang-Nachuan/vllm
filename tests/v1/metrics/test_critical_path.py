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
    metrics.record_l1_load_timing(worker_enqueue_s=2.1, dma_elapsed_s=0.4)
    metrics.add_prefix_offload(0.25, started_at_s=2.2)
    metrics.finish_prefix_load(3.0)
    metrics.add_prefix_lookup(0.1)
    metrics.add_model_forward("prefill", 0.2)
    metrics.add_model_forward("decode", 0.3)

    assert metrics.prefix_lookup_s == pytest.approx(0.1)
    assert metrics.prefix_load_s == pytest.approx(0.75)
    assert metrics.prefix_offload_s == pytest.approx(0.75)
    assert metrics.prefill_s == pytest.approx(0.2)
    assert metrics.decode_s == pytest.approx(0.3)
    assert metrics.prefix_load_l1_prepare_dispatch_s == pytest.approx(0.1)
    assert metrics.prefix_load_l1_h2d_dma_s == pytest.approx(0.4)
    assert metrics.prefix_load_l1_control_wait_s == pytest.approx(0.25)


def test_forced_offload_is_removed_from_its_actual_l1_phase():
    metrics = CriticalPathMetrics()
    metrics.start_prefix_load(1.0)
    metrics.record_l1_load_timing(worker_enqueue_s=1.5, dma_elapsed_s=0.25)
    metrics.add_prefix_offload(0.2, started_at_s=1.1)
    metrics.finish_prefix_load(2.0)

    assert metrics.prefix_load_s == pytest.approx(0.8)
    assert metrics.prefix_load_l1_prepare_dispatch_s == pytest.approx(0.3)
    assert metrics.prefix_load_l1_h2d_dma_s == pytest.approx(0.25)
    assert metrics.prefix_load_l1_control_wait_s == pytest.approx(0.25)


def test_l2_stages_are_clipped_to_owned_request_waits():
    metrics = CriticalPathMetrics()
    metrics.add_synchronous_l2_admission(0.05)
    metrics.start_lookup_wait("owned_l2", 1.1)
    metrics.finish_lookup_wait(1.4)
    metrics.start_lookup_wait("owned_l2", 1.5)
    metrics.finish_lookup_wait(2.1)

    metrics.record_l2_promotion_timing(
        initiated_at=1.0,
        submitted_at=1.2,
        last_task_started_at=1.6,
        last_task_finished_at=1.9,
    )

    assert metrics.prefix_load_s == pytest.approx(0.95)
    assert metrics.prefix_load_l2_admission_s == pytest.approx(0.15)
    assert metrics.prefix_load_l2_io_queue_s == pytest.approx(0.3)
    assert metrics.prefix_load_l2_ssd_read_s == pytest.approx(0.3)
    assert metrics.prefix_load_l2_completion_wait_s == pytest.approx(0.2)
    assert metrics.prefix_load_unattributed_s == 0.0


def test_shared_load_waits_are_separate_from_owned_io():
    metrics = CriticalPathMetrics()
    metrics.start_lookup_wait("shared_l1", 1.0)
    metrics.finish_lookup_wait(1.2)
    metrics.start_lookup_wait("shared_l2", 2.0)
    metrics.finish_lookup_wait(2.3)

    assert metrics.prefix_load_s == pytest.approx(0.5)
    assert metrics.prefix_load_shared_l1_wait_s == pytest.approx(0.2)
    assert metrics.prefix_load_shared_l2_wait_s == pytest.approx(0.3)


def test_mixed_cache_dependencies_are_reported_as_unattributed_load():
    metrics = CriticalPathMetrics()
    metrics.start_lookup_wait("unattributed_load", 1.0)
    metrics.finish_lookup_wait(1.25)

    assert metrics.prefix_load_s == pytest.approx(0.25)
    assert metrics.prefix_load_unattributed_s == pytest.approx(0.25)


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


def test_write_record_excludes_async_offload_overlap(tmp_path):
    metrics = CriticalPathMetrics(
        prefix_offload_s=3.0,
        prefill_s=4.0,
        decode_s=4.0,
    )
    trace_path = tmp_path / "critical_path.jsonl"

    write_critical_path_record(
        str(trace_path),
        request_id="chatcmpl-mswea-task-1",
        server_e2e_s=10.0,
        metrics=metrics,
    )

    record = json.loads(trace_path.read_text())
    assert record["prefix_offload_s"] == pytest.approx(2.0)
    assert record["queue_scheduler_s"] == 0.0
