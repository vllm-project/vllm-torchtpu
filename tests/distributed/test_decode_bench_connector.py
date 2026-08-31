# SPDX-License-Identifier: Apache-2.0
"""Focused tests for the TPU DecodeBenchConnector implementation."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.decode_bench_connector import \
    DecodeBenchConnectorMetadata as UpstreamDecodeBenchConnectorMetadata
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec, MambaSpec,
                                        UniformTypeKVCacheSpecs)

from vllm_torchtpu.distributed.kv_transfer.decode_bench_connector import (
    DecodeBenchConnector, DecodeBenchConnectorMetadata,
    DecodeBenchConnectorScheduler, DecodeBenchConnectorWorker,
    _upstream_fill_slots_in_place, _uses_upstream_block_fill)

_MODULE = "vllm_torchtpu.distributed.kv_transfer.decode_bench_connector"


def _vllm_config(*,
                 dcp: int = 1,
                 pcp: int = 1,
                 block_size: int = 16,
                 **extra_config):
    transfer_config = SimpleNamespace(
        kv_connector="DecodeBenchConnector",
        is_kv_producer=False,
        get_from_extra_config=lambda key, default: extra_config.get(
            key, default),
    )
    return SimpleNamespace(
        kv_transfer_config=transfer_config,
        cache_config=SimpleNamespace(block_size=block_size,
                                     enable_prefix_caching=True,
                                     prefix_match_unit=None),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=dcp,
            prefill_context_parallel_size=pcp,
        ),
    )


def _hybrid_cache_config(num_blocks: int = 4) -> KVCacheConfig:
    attention = FullAttentionSpec(block_size=16,
                                  num_kv_heads=1,
                                  head_size=128,
                                  dtype=torch.bfloat16)
    recurrent = MambaSpec(
        block_size=32768,
        shapes=((1, 1), (1, 1, 1)),
        dtypes=(torch.float32, torch.bfloat16),
    )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["attn", "attn_alias"], attention),
            KVCacheGroupSpec(["kda.0"], recurrent),
            KVCacheGroupSpec(["kda.1"], recurrent),
            KVCacheGroupSpec(["kda.2"], recurrent),
        ],
    )


@pytest.mark.parametrize(
    "fill_std",
    [-0.1, float("nan"), float("inf"), -float("inf")])
def test_rejects_invalid_fill_std(fill_std):
    with pytest.raises(ValueError, match="fill_std"):
        DecodeBenchConnectorWorker(_vllm_config(fill_std=fill_std),
                                   _hybrid_cache_config())


def test_accepts_positive_fill_std():
    worker = DecodeBenchConnectorWorker(_vllm_config(fill_std=0.1),
                                        _hybrid_cache_config())
    assert worker.fill_std == pytest.approx(0.1)


def test_worker_requires_prefix_caching():
    config = _vllm_config()
    config.cache_config.enable_prefix_caching = False

    with pytest.raises(ValueError, match="requires prefix caching"):
        DecodeBenchConnectorWorker(config, _hybrid_cache_config())


def test_worker_rejects_raw_integer_cache_encoding():
    worker = DecodeBenchConnectorWorker(_vllm_config(), _hybrid_cache_config())

    with pytest.raises(TypeError, match="raw or integer cache encodings"):
        worker.register_kv_caches({
            "attn": (
                torch.zeros((4, 8, 1, 128), dtype=torch.uint8),
                torch.zeros((4, 2, 1, 128), dtype=torch.uint8),
            ),
        })

    assert worker.kv_caches is None


def test_scheduler_uses_each_group_block_size_and_cleans_up():
    cache_config = _hybrid_cache_config()
    scheduler = DecodeBenchConnectorScheduler(_vllm_config(), cache_config)
    request = SimpleNamespace(request_id="request-0", num_tokens=32768)

    assert scheduler.get_num_new_matched_tokens(request, 0) == (32767, False)
    blocks = MagicMock()
    blocks.get_block_ids.return_value = (
        list(range(2048)),
        [101],
        [102],
        [103],
    )
    scheduler.update_state_after_alloc(request, blocks, 32767)
    metadata = scheduler.build_connector_meta(MagicMock())

    selected, num_tokens = metadata.reqs_to_fill["request-0"]
    assert num_tokens == 32767
    assert len(selected[0]) == 2048
    assert selected[1:] == ([101], [102], [103])
    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    assert not scheduler.build_connector_meta(MagicMock()).reqs_to_fill

    scheduler.request_finished(request)
    assert scheduler.get_num_new_matched_tokens(request, 0) == (32767, False)


def test_scheduler_does_not_mark_request_filled_after_bad_allocation():
    scheduler = DecodeBenchConnectorScheduler(_vllm_config(),
                                              _hybrid_cache_config())
    request = SimpleNamespace(request_id="request-0", num_tokens=33)
    blocks = MagicMock()
    blocks.get_block_ids.return_value = ([0], [1], [2], [3])

    with pytest.raises(ValueError, match="through index 1"):
        scheduler.update_state_after_alloc(request, blocks, 32)
    assert scheduler.get_num_new_matched_tokens(request, 0) == (32, False)


def test_scheduler_selects_only_prefix_cache_suffix_blocks():
    attention = FullAttentionSpec(block_size=16,
                                  num_kv_heads=1,
                                  head_size=128,
                                  dtype=torch.bfloat16)
    cache_config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["attn"], attention)],
    )
    scheduler = DecodeBenchConnectorScheduler(_vllm_config(), cache_config)
    request = SimpleNamespace(request_id="request-0", num_tokens=64)
    blocks = MagicMock()
    blocks.get_block_ids.return_value = ([10, 11, 12, 13], )

    # The first 32 tokens came from the local prefix cache. The connector owns
    # tokens [32, 63), which touch only manager blocks 2 and 3.
    scheduler.update_state_after_alloc(request, blocks, 31)
    metadata = scheduler.build_connector_meta(MagicMock())

    assert metadata.reqs_to_fill["request-0"] == ((([12, 13]), ), 31)


def test_scheduler_scales_attention_manager_blocks_for_dcp():
    attention = FullAttentionSpec(block_size=16,
                                  num_kv_heads=1,
                                  head_size=128,
                                  dtype=torch.bfloat16)
    cache_config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["attn"], attention)],
    )
    scheduler = DecodeBenchConnectorScheduler(_vllm_config(dcp=2),
                                              cache_config)
    request = SimpleNamespace(request_id="request-0", num_tokens=64)
    blocks = MagicMock()
    blocks.get_block_ids.return_value = ([20, 21], )

    scheduler.update_state_after_alloc(request, blocks, 31)
    metadata = scheduler.build_connector_meta(MagicMock())

    assert metadata.reqs_to_fill["request-0"] == ((([21]), ), 31)


def test_scheduler_scales_hybrid_manager_blocks_for_pcp():
    attention = FullAttentionSpec(block_size=4096,
                                  num_kv_heads=1,
                                  head_size=128,
                                  dtype=torch.bfloat16)
    recurrent = MambaSpec(block_size=4096,
                          shapes=((1, 8), (1, 2, 2)),
                          dtypes=(torch.bfloat16, torch.float32))
    cache_config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["attn"], attention),
            KVCacheGroupSpec(["gdn"], recurrent),
        ],
    )
    scheduler = DecodeBenchConnectorScheduler(
        _vllm_config(pcp=8, block_size=4096), cache_config)
    request = SimpleNamespace(request_id="request-0", num_tokens=32768)
    blocks = MagicMock()
    blocks.get_block_ids.return_value = ([31], [47])

    scheduler.update_state_after_alloc(request, blocks, 32767)
    metadata = scheduler.build_connector_meta(MagicMock())

    assert metadata.reqs_to_fill["request-0"] == ((([31], [47])), 32767)


def test_scheduler_selects_hybrid_pcp_suffix_after_local_hit():
    attention = FullAttentionSpec(block_size=4096,
                                  num_kv_heads=1,
                                  head_size=128,
                                  dtype=torch.bfloat16)
    recurrent = MambaSpec(block_size=4096,
                          shapes=((1, 8), (1, 2, 2)),
                          dtypes=(torch.bfloat16, torch.float32))
    cache_config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["attn"], attention),
            KVCacheGroupSpec(["gdn"], recurrent),
        ],
    )
    scheduler = DecodeBenchConnectorScheduler(
        _vllm_config(pcp=8, block_size=4096), cache_config)
    request = SimpleNamespace(request_id="request-0", num_tokens=36864)
    blocks = MagicMock()
    blocks.get_block_ids.return_value = ([31, 32], [47, 48])

    scheduler.update_state_after_alloc(request, blocks, 4095)
    metadata = scheduler.build_connector_meta(MagicMock())

    assert metadata.reqs_to_fill["request-0"] == ((([32], [48])), 4095)


@pytest.mark.parametrize(
    ("unhashed_blocks", "expected_blocks"),
    [
        (([10], [21], [22], [23]), ([], [21], [22], [23])),
        (([], [], [], []), ([], [], [], [])),
    ],
)
def test_scheduler_selects_only_final_state_after_fully_local_hit(
        unhashed_blocks, expected_blocks):
    cache_config = _hybrid_cache_config()
    scheduler = DecodeBenchConnectorScheduler(_vllm_config(), cache_config)
    request = SimpleNamespace(request_id="request-0", num_tokens=32769)
    blocks = MagicMock()
    blocks.get_unhashed_block_ids_all_groups.return_value = unhashed_blocks

    scheduler.update_state_after_alloc(request, blocks, 0)
    metadata = scheduler.build_connector_meta(MagicMock())

    assert metadata.reqs_to_fill["request-0"] == (
        expected_blocks,
        0,
    )
    assert scheduler.get_num_new_matched_tokens(request, 32768) == (0, False)


@pytest.mark.parametrize("fill_std", [0.0, 0.2])
def test_upstream_fill_matches_selected_row_semantics(fill_std):
    cache = torch.arange(8 * 16 * 640,
                         dtype=torch.float32).reshape(8, 16, 1, 640)
    cache = cache.to(torch.bfloat16)
    before = cache.clone()
    selected = [1, 5]
    indices = torch.tensor(selected, dtype=torch.long)

    _upstream_fill_slots_in_place(cache, indices, mean=0.4, std=fill_std)

    samples = cache[selected].float()
    if fill_std == 0.0:
        torch.testing.assert_close(samples,
                                   torch.full_like(samples, 0.4),
                                   rtol=0.0,
                                   atol=0.002)
    else:
        assert samples.mean().item() == pytest.approx(0.4, abs=0.015)
        assert samples.std().item() == pytest.approx(fill_std, abs=0.015)
    untouched = torch.ones(cache.shape[0], dtype=torch.bool)
    untouched[selected] = False
    torch.testing.assert_close(cache[untouched], before[untouched])


@pytest.mark.parametrize("fill_std", [0.0, 0.2])
def test_worker_uses_upstream_fill_for_kimi_mla_layout(fill_std):
    attention_spec = FullAttentionSpec(block_size=16,
                                       num_kv_heads=1,
                                       head_size=640,
                                       dtype=torch.bfloat16)
    cache_config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["attn"], attention_spec)],
    )
    worker = DecodeBenchConnectorWorker(_vllm_config(fill_std=fill_std),
                                        cache_config)
    attention = torch.zeros((8, 16, 1, 640), dtype=torch.bfloat16)
    assert _uses_upstream_block_fill(attention)
    worker.register_kv_caches({"attn": attention})
    metadata = DecodeBenchConnectorMetadata(reqs_to_fill={
        "request-0": (([1, 5], ), 31),
    })

    with patch(f"{_MODULE}._upstream_fill_slots_in_place",
               wraps=_upstream_fill_slots_in_place) as upstream_fill, \
            patch(f"{_MODULE}._fill_slots_in_place") as constant_fill, \
            patch(f"{_MODULE}._normal_fill_slots_in_place") as normal_fill, \
            patch(f"{_MODULE}._synchronize_mutated"):
        worker.start_fill_kv(metadata)

    upstream_fill.assert_called_once()
    cache, indices = upstream_fill.call_args.args
    assert cache is attention
    assert indices.dtype == torch.long
    assert indices.tolist() == [1, 5]
    assert upstream_fill.call_args.kwargs == {
        "mean": 0.015,
        "std": fill_std,
    }
    constant_fill.assert_not_called()
    normal_fill.assert_not_called()


def test_worker_fills_attention_rows_and_only_compact_kda_slots():
    # Deliberately make manager capacity differ from the physical row count;
    # Kimi's HMA padding has this property in production.
    cache_config = _hybrid_cache_config(num_blocks=7)
    worker = DecodeBenchConnectorWorker(_vllm_config(), cache_config)

    # Physical attention rows are indexed directly by manager block ID even
    # when the physical row count differs from manager capacity. The alias
    # must still be donated once.
    attention = torch.zeros((8, 1, 1), dtype=torch.bfloat16)
    kda_caches = {
        f"kda.{index}": (
            torch.zeros((9, 1, 1), dtype=torch.float32),
            torch.zeros((9, 1, 1, 1), dtype=torch.bfloat16),
        )
        for index in range(3)
    }
    worker.register_kv_caches({
        "attn": attention,
        "attn_alias": attention,
        **kda_caches,
    })
    runner = MagicMock()
    runner._unified_kv_layout = False
    runner.ensure_mamba_slots_for_requests.return_value = {"a": 1, "b": 2}
    worker.register_runner(runner)

    metadata = DecodeBenchConnectorMetadata(
        reqs_to_fill={
            "a": (([1, 1], [7], [8], [9]), 31),
            # Fully local prefix hit: no attention rows need materialization, but
            # the newly assigned compact recurrent slot still does.
            "b": (([], [], [], []), 0),
        })
    calls: list[tuple[torch.Tensor, list[int], float]] = []

    def record_fill(cache, indices, fill_value):
        calls.append((cache, indices.tolist(), float(fill_value.item())))

    with patch(f"{_MODULE}._fill_slots_in_place", side_effect=record_fill), \
            patch(f"{_MODULE}._synchronize_mutated") as synchronize:
        worker.start_fill_kv(metadata)

    runner.ensure_mamba_slots_for_requests.assert_called_once_with(["a", "b"])
    attention_calls = [call for call in calls if call[0] is attention]
    assert len(attention_calls) == 1
    assert attention_calls[0][1] == [1]
    assert attention_calls[0][2] == pytest.approx(0.015)

    state_ids = {id(state) for cache in kda_caches.values() for state in cache}
    state_calls = [call for call in calls if id(call[0]) in state_ids]
    assert len(state_calls) == 6
    assert all(call[1] == [1, 2] for call in state_calls)
    synchronize.assert_called_once()
    synchronized = synchronize.call_args.args[0]
    assert len(synchronized) == 7
    assert len({id(tensor) for tensor in synchronized}) == 7


def test_worker_expands_manager_blocks_to_physical_kernel_rows():
    attention_spec = FullAttentionSpec(block_size=16,
                                       num_kv_heads=1,
                                       head_size=128,
                                       dtype=torch.bfloat16)
    cache_config = KVCacheConfig(
        num_blocks=6,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["attn"], attention_spec)],
    )
    worker = DecodeBenchConnectorWorker(_vllm_config(), cache_config)
    attention = torch.zeros((12, 1, 1), dtype=torch.bfloat16)
    worker.register_kv_caches({"attn": attention})
    runner = MagicMock()
    runner._unified_kv_layout = False
    runner._kernel_block_size_by_group_id = {0: 8}
    worker.register_runner(runner)
    metadata = DecodeBenchConnectorMetadata(reqs_to_fill={
        "request-0": (([1, 4], ), 31),
    })
    calls: list[list[int]] = []

    def record_fill(cache, indices, fill_value):
        del cache, fill_value
        calls.append(indices.tolist())

    with patch(f"{_MODULE}._fill_slots_in_place", side_effect=record_fill), \
            patch(f"{_MODULE}._synchronize_mutated"):
        worker.start_fill_kv(metadata)

    assert calls == [[2, 3, 8, 9]]


def test_worker_fills_composite_split_attention_cache_once_per_alias():
    layer_spec = FullAttentionSpec(block_size=16,
                                   num_kv_heads=1,
                                   head_size=128,
                                   dtype=torch.bfloat16)
    composite_spec = UniformTypeKVCacheSpecs(
        block_size=16, kv_cache_specs={"attn": layer_spec})
    cache_config = KVCacheConfig(
        num_blocks=6,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["attn", "attn_alias"], composite_spec)
        ],
    )
    worker = DecodeBenchConnectorWorker(_vllm_config(), cache_config)
    split_cache = (
        torch.zeros((6, 1, 1), dtype=torch.bfloat16),
        torch.zeros((6, 1, 1), dtype=torch.bfloat16),
    )
    worker.register_kv_caches({
        "attn": split_cache,
        "attn_alias": split_cache,
    })
    runner = MagicMock()
    runner._unified_kv_layout = False
    runner._kernel_block_size_by_group_id = {0: 16}
    worker.register_runner(runner)
    metadata = DecodeBenchConnectorMetadata(reqs_to_fill={
        "request-0": (([2], ), 15),
    })
    calls: list[tuple[torch.Tensor, list[int]]] = []

    def record_fill(cache, indices, fill_value):
        del fill_value
        calls.append((cache, indices.tolist()))

    with patch(f"{_MODULE}._fill_slots_in_place", side_effect=record_fill), \
            patch(f"{_MODULE}._synchronize_mutated"):
        worker.start_fill_kv(metadata)

    assert len(calls) == 2
    assert {id(cache)
            for cache, _ in calls} == {id(tensor)
                                       for tensor in split_cache}
    assert all(indices == [2] for _, indices in calls)


def test_worker_dispatches_gaussian_fill_with_independent_seeds():
    cache_config = _hybrid_cache_config(num_blocks=7)
    worker = DecodeBenchConnectorWorker(
        _vllm_config(fill_mean=0.4, fill_std=0.2), cache_config)
    attention = torch.zeros((8, 1, 1), dtype=torch.bfloat16)
    kda_caches = {
        f"kda.{index}": (
            torch.zeros((9, 1, 1), dtype=torch.float32),
            torch.zeros((9, 1, 1, 1), dtype=torch.bfloat16),
        )
        for index in range(3)
    }
    worker.register_kv_caches({
        "attn": attention,
        "attn_alias": attention,
        **kda_caches,
    })
    runner = MagicMock()
    runner._unified_kv_layout = False
    runner.ensure_mamba_slots_for_requests.return_value = {"request-0": 2}
    worker.register_runner(runner)
    metadata = DecodeBenchConnectorMetadata(reqs_to_fill={
        "request-0": (([1], [7], [8], [9]), 15),
    })
    calls: list[tuple[torch.Tensor, list[int], list[float], int]] = []

    def record_fill(cache, indices, params, seed):
        calls.append((cache, indices.tolist(), params.tolist(), seed.item()))

    with patch(f"{_MODULE}._normal_fill_slots_in_place",
               side_effect=record_fill), \
            patch(f"{_MODULE}._fill_slots_in_place") as constant_fill, \
            patch(f"{_MODULE}._synchronize_mutated"):
        worker.start_fill_kv(metadata)

    assert len(calls) == 7
    assert all(call[2] == pytest.approx([0.4, 0.2]) for call in calls)
    assert len({call[3] for call in calls}) == len(calls)
    constant_fill.assert_not_called()


@pytest.mark.parametrize(("ssm_dtype", "ssm_itemsize"), [
    (torch.bfloat16, 2),
    (torch.float32, 4),
])
def test_worker_aggregates_unified_attention_and_gdn_by_storage(
        ssm_dtype, ssm_itemsize):
    attention = FullAttentionSpec(block_size=8,
                                  num_kv_heads=1,
                                  head_size=8,
                                  dtype=torch.bfloat16)
    recurrent = MambaSpec(block_size=8,
                          shapes=((1, 8), (1, 2, 2)),
                          dtypes=(torch.bfloat16, ssm_dtype),
                          page_size_padded=attention.page_size_bytes)
    cache_config = KVCacheConfig(
        num_blocks=6,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["attn", "attn_alias"], attention),
            KVCacheGroupSpec(["gdn", "gdn_alias"], recurrent),
        ],
    )
    worker = DecodeBenchConnectorWorker(_vllm_config(block_size=8),
                                        cache_config)
    raw = torch.zeros((12, 4, 2, 8), dtype=torch.bfloat16)
    worker.register_kv_caches({
        "attn": raw,
        "attn_alias": raw,
        "gdn": [raw],
        "gdn_alias": [raw],
    })
    runner = MagicMock()
    runner._unified_kv_layout = True
    runner._pool_block_split = 2
    runner._kernel_block_size_by_group_id = {0: 4, 1: 4}
    worker.register_runner(runner)
    metadata = DecodeBenchConnectorMetadata(reqs_to_fill={
        "request-0": (([0, 2], [0, 2, 3]), 15),
    })

    layout = SimpleNamespace(ssm_bytes=16, conv_bytes=16)
    with patch(f"{_MODULE}.get_tpu_pooled_gdn_state_layout",
               return_value=layout), \
            patch(f"{_MODULE}._fill_hybrid_pool_in_place") as fill, \
            patch(f"{_MODULE}._synchronize_mutated") as synchronize:
        worker.start_fill_kv(metadata)

    runner.ensure_mamba_slots_for_requests.assert_not_called()
    fill.assert_called_once()
    cache, indices, fill_kinds, fill_value = fill.call_args.args
    assert cache is raw
    assert indices.tolist() == [2, 3]
    assert fill_kinds.tolist() == [3, 2]
    assert fill_value.item() == pytest.approx(0.015)
    assert fill.call_args.kwargs == {
        "split": 2,
        "ssm_bytes": 16,
        "ssm_itemsize": ssm_itemsize,
        "conv_bytes": 16,
    }
    synchronize.assert_called_once_with([raw])


def test_connector_wrapper_delegates_runner_registration():
    connector = DecodeBenchConnector(_vllm_config(), KVConnectorRole.WORKER,
                                     _hybrid_cache_config())
    runner = MagicMock()
    runner._unified_kv_layout = False

    connector.register_runner(runner)

    assert connector.connector_worker is not None
    assert connector.connector_worker.runner is runner


def test_connector_reuses_upstream_wire_metadata():
    assert DecodeBenchConnectorMetadata is UpstreamDecodeBenchConnectorMetadata
    connector = DecodeBenchConnector(_vllm_config(), KVConnectorRole.WORKER,
                                     _hybrid_cache_config())
    assert connector.connector_worker is not None
    connector.connector_worker.start_fill_kv = MagicMock()
    connector.bind_connector_metadata(
        DecodeBenchConnectorMetadata(reqs_to_fill={"request-0": (([1], ),
                                                                 15)}))

    connector.start_load_kv(MagicMock())

    metadata = connector.connector_worker.start_fill_kv.call_args.args[0]
    assert isinstance(metadata, DecodeBenchConnectorMetadata)
    assert metadata.reqs_to_fill == {"request-0": (([1], ), 15)}
