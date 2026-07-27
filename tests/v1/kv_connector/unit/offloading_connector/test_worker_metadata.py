# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    LoadJobTiming,
    OffloadingWorkerMetadata,
)

pytestmark = pytest.mark.cpu_test


def test_aggregate_sums_counts():
    meta1 = OffloadingWorkerMetadata(completed_jobs={42: 1, 7: 1})
    meta2 = OffloadingWorkerMetadata(completed_jobs={42: 1, 7: 1})
    result = meta1.aggregate(meta2)
    assert result.completed_jobs == {42: 2, 7: 2}


def test_aggregate_disjoint_jobs():
    meta1 = OffloadingWorkerMetadata(completed_jobs={42: 1, 7: 1})
    meta2 = OffloadingWorkerMetadata(completed_jobs={43: 1, 8: 1})
    result = meta1.aggregate(meta2)
    assert result.completed_jobs == {42: 1, 7: 1, 43: 1, 8: 1}


def test_aggregate_multiple_workers():
    meta1 = OffloadingWorkerMetadata(completed_jobs={42: 1, 43: 1, 7: 1})
    meta2 = OffloadingWorkerMetadata(completed_jobs={42: 1, 7: 1, 8: 1})
    meta3 = OffloadingWorkerMetadata(completed_jobs={42: 1, 43: 1, 8: 1})
    result = meta1.aggregate(meta2).aggregate(meta3)
    assert result.completed_jobs == {42: 3, 43: 2, 7: 2, 8: 2}


def test_aggregate_uses_longest_forced_offload_wait():
    meta1 = OffloadingWorkerMetadata(
        prefix_offload_wait_s=0.25, prefix_offload_wait_started_at_s=1.0
    )
    meta2 = OffloadingWorkerMetadata(
        prefix_offload_wait_s=0.5, prefix_offload_wait_started_at_s=2.0
    )

    result = meta1.aggregate(meta2)

    assert result.prefix_offload_wait_s == 0.5
    assert result.prefix_offload_wait_started_at_s == 2.0


def test_aggregate_uses_latest_forced_offload_completion():
    longer_but_earlier = OffloadingWorkerMetadata(
        prefix_offload_wait_s=0.5, prefix_offload_wait_started_at_s=1.0
    )
    critical = OffloadingWorkerMetadata(
        prefix_offload_wait_s=0.25, prefix_offload_wait_started_at_s=2.0
    )

    result = longer_but_earlier.aggregate(critical)

    assert result.prefix_offload_wait_s == 0.25
    assert result.prefix_offload_wait_started_at_s == 2.0


def test_aggregate_load_timing_uses_latest_completion():
    earlier = LoadJobTiming(
        worker_start_s=1.0,
        worker_enqueue_s=1.1,
        dma_elapsed_s=0.2,
        completion_observed_s=1.4,
        transfer_bytes=100,
    )
    critical = LoadJobTiming(
        worker_start_s=1.05,
        worker_enqueue_s=1.2,
        dma_elapsed_s=0.25,
        completion_observed_s=1.6,
        transfer_bytes=200,
    )
    unrelated = LoadJobTiming(
        worker_start_s=2.0,
        worker_enqueue_s=2.1,
        dma_elapsed_s=0.1,
        completion_observed_s=2.3,
        transfer_bytes=300,
    )
    meta1 = OffloadingWorkerMetadata(
        completed_jobs={42: 1}, load_job_timings={42: earlier}
    )
    meta2 = OffloadingWorkerMetadata(
        completed_jobs={42: 1, 7: 1},
        load_job_timings={42: critical, 7: unrelated},
    )

    result = meta1.aggregate(meta2)

    assert result.completed_jobs == {42: 2, 7: 1}
    assert result.load_job_timings == {42: critical, 7: unrelated}
