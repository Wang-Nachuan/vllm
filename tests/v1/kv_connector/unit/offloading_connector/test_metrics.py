# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)


def test_build_kv_connector_stats_with_none():
    """Test that build_kv_connector_stats returns empty stats when given None."""
    stats = OffloadingConnector.build_kv_connector_stats(data=None)

    assert stats is not None
    assert isinstance(stats, OffloadingConnectorStats)
    assert len(stats.data) == 0
    assert stats.is_empty()


def test_build_kv_connector_stats_with_empty_dict():
    """Test that build_kv_connector_stats returns empty stats with empty dict."""
    stats = OffloadingConnector.build_kv_connector_stats(data={})

    assert stats is not None
    assert isinstance(stats, OffloadingConnectorStats)
    assert len(stats.data) == 0
    assert stats.is_empty()


def test_build_kv_connector_stats_reconstructs_offload_stats():
    """Test that OffloadingConnector stats are properly reconstructed with
    correct data."""
    serialized_data = {
        "CPU_to_GPU": [
            {"op_size": 16, "op_time": 1.0},
            {"op_size": 8, "op_time": 0.5},
        ],
        "GPU_to_CPU": [
            {"op_size": 1, "op_time": 0.1},
            {"op_size": 2, "op_time": 0.2},
        ],
    }

    stats = OffloadingConnector.build_kv_connector_stats(data=serialized_data)

    offload_connector_stats = stats
    assert isinstance(offload_connector_stats, OffloadingConnectorStats)
    assert offload_connector_stats.data["CPU_to_GPU"] == [
        {"op_size": 16, "op_time": 1.0},
        {"op_size": 8, "op_time": 0.5},
    ]
    assert offload_connector_stats.data["GPU_to_CPU"] == [
        {"op_size": 1, "op_time": 0.1},
        {"op_size": 2, "op_time": 0.2},
    ]


def test_aggregate_same_connector():
    """Test aggregating stats from the same connector type."""
    stats1 = OffloadingConnectorStats(
        data={
            "CPU_to_GPU": [
                {"op_size": 16, "op_time": 1.0},
                {"op_size": 8, "op_time": 0.5},
            ],
            "GPU_to_CPU": [
                {"op_size": 1, "op_time": 0.1},
                {"op_size": 2, "op_time": 0.2},
            ],
        }
    )

    stats2 = OffloadingConnectorStats(
        data={
            "CPU_to_GPU": [
                {"op_size": 3, "op_time": 0.2},
                {"op_size": 7, "op_time": 0.9},
            ],
            "GPU_to_CPU": [{"op_size": 16, "op_time": 2}],
        }
    )

    result = stats1.aggregate(stats2)

    assert result is stats1  # Should return self
    offload_connector_stats = result
    assert offload_connector_stats.data["CPU_to_GPU"] == [
        {"op_size": 16, "op_time": 1.0},
        {"op_size": 8, "op_time": 0.5},
        {"op_size": 3, "op_time": 0.2},
        {"op_size": 7, "op_time": 0.9},
    ]
    assert offload_connector_stats.data["GPU_to_CPU"] == [
        {"op_size": 1, "op_time": 0.1},
        {"op_size": 2, "op_time": 0.2},
        {"op_size": 16, "op_time": 2},
    ]


def test_aggregate_secondary_tier_stats():
    """Test that secondary tier counters are summed across observations."""
    stats1 = OffloadingConnectorStats()
    stats1.record_secondary_tier_stats(
        {
            "store": {
                "submitted_jobs": 2,
                "submitted_blocks": 6,
                "succeeded_jobs": 1,
                "succeeded_blocks": 3,
            },
            "promotion": {
                "submitted_jobs": 1,
                "submitted_blocks": 2,
            },
        }
    )

    stats2 = OffloadingConnectorStats()
    stats2.record_secondary_tier_stats(
        {
            "store": {
                "succeeded_jobs": 1,
                "succeeded_blocks": 3,
                "failed_jobs": 1,
                "failed_blocks": 2,
            },
            "promotion": {
                "succeeded_jobs": 1,
                "succeeded_blocks": 2,
            },
        }
    )

    reduced = stats1.aggregate(stats2).reduce()

    assert reduced["secondary_tier_store_submitted_jobs"] == 2
    assert reduced["secondary_tier_store_submitted_blocks"] == 6
    assert reduced["secondary_tier_store_succeeded_jobs"] == 2
    assert reduced["secondary_tier_store_succeeded_blocks"] == 6
    assert reduced["secondary_tier_store_failed_jobs"] == 1
    assert reduced["secondary_tier_store_failed_blocks"] == 2
    assert reduced["secondary_tier_promotion_submitted_jobs"] == 1
    assert reduced["secondary_tier_promotion_submitted_blocks"] == 2
    assert reduced["secondary_tier_promotion_succeeded_jobs"] == 1
    assert reduced["secondary_tier_promotion_succeeded_blocks"] == 2


def test_aggregate_tiering_lookup_stats():
    """Test that tiering lookup counters are summed across observations."""
    stats1 = OffloadingConnectorStats()
    stats1.record_tiering_lookup_stats(
        {
            "total_blocks": 4,
            "cpu_lookup_blocks": 4,
            "cpu_hit_ready_blocks": 1,
            "cpu_hit_pending_blocks": 1,
            "secondary_lookup_blocks": 2,
            "secondary_hit_ready_blocks": 2,
        }
    )

    stats2 = OffloadingConnectorStats()
    stats2.record_tiering_lookup_stats(
        {
            "total_blocks": 3,
            "secondary_hit_pending_blocks": 1,
            "secondary_hit_but_no_primary_slot_blocks": 1,
            "miss_blocks": 2,
        }
    )

    reduced = stats1.aggregate(stats2).reduce()

    assert reduced["tiering_lookup_total_blocks"] == 7
    assert reduced["tiering_lookup_cpu_lookup_blocks"] == 4
    assert reduced["tiering_lookup_cpu_hit_ready_blocks"] == 1
    assert reduced["tiering_lookup_cpu_hit_pending_blocks"] == 1
    assert reduced["tiering_lookup_secondary_lookup_blocks"] == 2
    assert reduced["tiering_lookup_secondary_hit_ready_blocks"] == 2
    assert reduced["tiering_lookup_secondary_hit_pending_blocks"] == 1
    assert reduced["tiering_lookup_secondary_hit_but_no_primary_slot_blocks"] == 1
    assert reduced["tiering_lookup_miss_blocks"] == 2


def test_aggregate_primary_eviction_stats():
    """Test that primary eviction counters are summed across observations."""
    stats1 = OffloadingConnectorStats()
    stats1.record_primary_eviction_stats(
        {
            "promotion": {"events": 2, "blocks": 2},
            "store": {"events": 1, "blocks": 3},
        }
    )

    stats2 = OffloadingConnectorStats()
    stats2.record_primary_eviction_stats(
        {
            "promotion": {"events": 1, "blocks": 1},
            "store": {"events": 2, "blocks": 4},
        }
    )

    reduced = stats1.aggregate(stats2).reduce()

    assert reduced["primary_eviction_promotion_events"] == 3
    assert reduced["primary_eviction_promotion_blocks"] == 3
    assert reduced["primary_eviction_store_events"] == 3
    assert reduced["primary_eviction_store_blocks"] == 7


def test_reduce():
    """Test that reduce() correctly reduces all nested connector stats."""
    stats = OffloadingConnectorStats(
        data={
            "CPU_to_GPU": [
                {"op_size": 16, "op_time": 1.0},
                {"op_size": 8, "op_time": 0.5},
                {"op_size": 3, "op_time": 0.2},
                {"op_size": 7, "op_time": 0.9},
            ],
            "GPU_to_CPU": [
                {"op_size": 1, "op_time": 0.1},
                {"op_size": 2, "op_time": 0.2},
                {"op_size": 16, "op_time": 2},
            ],
        }
    )

    reduced = stats.reduce()

    assert isinstance(reduced, dict)
    # Check that the stats were reduced (should have aggregated values)
    assert "CPU_to_GPU_total_bytes" in reduced
    assert "CPU_to_GPU_total_time" in reduced
    assert "GPU_to_CPU_total_bytes" in reduced
    assert "GPU_to_CPU_total_time" in reduced
    assert reduced["CPU_to_GPU_total_bytes"] == 34
    assert reduced["CPU_to_GPU_total_time"] == 2.6
    assert reduced["GPU_to_CPU_total_time"] == 2.3
    assert reduced["GPU_to_CPU_total_bytes"] == 19


def test_reset():
    """Test that reset() resets all nested connector stats."""
    offload_connector_stats = OffloadingConnectorStats(
        data={
            "CPU_to_GPU": [
                {"op_size": 3, "op_time": 0.2},
                {"op_size": 7, "op_time": 0.9},
            ],
            "GPU_to_CPU": [{"op_size": 16, "op_time": 2}],
        }
    )

    assert not offload_connector_stats.is_empty()

    offload_connector_stats.reset()

    # After reset, stats should be empty
    assert offload_connector_stats.is_empty()
    assert len(offload_connector_stats.data) == 0
