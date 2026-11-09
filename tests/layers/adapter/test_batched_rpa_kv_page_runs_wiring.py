"""Wiring of TPU_BATCHED_RPA_KV_PAGE_RUNS.

The startup check that every manager block's kernel pages sit at consecutive
pool indices, the binding of the attention layers, and the kernel entry. The
kernel equivalence is in tests/kernels/experimental/batched_rpa/test_page_runs.py
and test_batched_rpa_kv_page_runs_attention.py (TPU).
"""

import functools
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_torchtpu.block_major_pool import BlockMajorPoolLayout, flat_kernel_ids
from vllm_torchtpu.kernels.experimental.batched_rpa import (
    configs as batched_rpa_configs,
)
from vllm_torchtpu.layers.adapter import attention as adapter
from vllm_torchtpu.runner import kv_cache_manager
from vllm_torchtpu.runner.kv_cache_manager import KVCacheManager, kv_page_runs

_LANE = batched_rpa_configs.KVLayout.SEQ_ALONG_LANE
# Served geometry: 2176-token manager blocks of 128-token kernel pages, a
# 1744-block pool and a 1M-token table (482 manager blocks per request).
_BLOCK, _PAGE, _NUM_BLOCKS, _BLOCKS_PER_REQ = 2176, 128, 1744, 482
_RUNS = _BLOCK // _PAGE


def _block_table(block_size=_BLOCK, kernel_block_size=_PAGE):
    from vllm.v1.worker.block_table import BlockTable

    return BlockTable(
        block_size=block_size,
        max_num_reqs=4,
        max_num_blocks_per_req=_BLOCKS_PER_REQ,
        max_num_batched_tokens=64,
        pin_memory=False,
        device=torch.device("cpu"),
        kernel_block_size=kernel_block_size,
        cp_kv_cache_interleave_size=1,
    )


@pytest.mark.cpu_test
def test_served_block_table_keeps_each_manager_block_consecutive():
    table = _block_table()
    order = np.random.default_rng(0).permutation(np.arange(1, _NUM_BLOCKS))
    # Three requests of 61 shuffled manager blocks, added and then grown.
    rows = [order[61 * r : 61 * (r + 1)].tolist() for r in range(3)]
    for r, ids in enumerate(rows):
        table.add_row(ids[:40], r)
        table.append_row(ids[40:], r)
    pages_np = table.get_numpy_array()
    for r, ids in enumerate(rows):
        pages = pages_np[r, : len(ids) * _RUNS].reshape(len(ids), _RUNS)
        np.testing.assert_array_equal(
            pages, np.asarray(ids)[:, None] * _RUNS + np.arange(_RUNS)
        )
        # The block-major remap keeps them consecutive in every region.
        for pool_index in range(4):
            flat = flat_kernel_ids(
                torch.from_numpy(pages.ravel().copy()), pool_index, 4, _RUNS
            ).numpy()
            flat = flat.reshape(len(ids), _RUNS)
            np.testing.assert_array_equal(
                flat - flat[:, :1], np.broadcast_to(np.arange(_RUNS), flat.shape)
            )
    assert kv_page_runs(table, _BLOCK, _PAGE, _NUM_BLOCKS, _NUM_BLOCKS * _RUNS) == 17
    layout = BlockMajorPoolLayout(_NUM_BLOCKS, 4, _RUNS, {})
    for pool_index in range(4):
        runs = kv_page_runs(
            table,
            _BLOCK,
            _PAGE,
            _NUM_BLOCKS,
            _NUM_BLOCKS * 4 * _RUNS,
            layout,
            pool_index,
        )
        assert runs == _RUNS
    # The probe leaves its row empty.
    assert table.num_blocks_per_row[-1] == 0
    assert not pages_np[-1].any()


@pytest.mark.cpu_test
def test_page_run_check_refuses_pages_that_are_not_consecutive(monkeypatch):
    pages = _NUM_BLOCKS * _RUNS
    with pytest.raises(NotImplementedError, match="do not span several"):
        kv_page_runs(_block_table(_PAGE, _PAGE), _PAGE, _PAGE, _NUM_BLOCKS, pages)
    with pytest.raises(NotImplementedError, match="maps manager blocks to 34"):
        kv_page_runs(_block_table(2 * _BLOCK), _BLOCK, _PAGE, _NUM_BLOCKS, pages)
    with pytest.raises(NotImplementedError, match="pool holds"):
        kv_page_runs(_block_table(), _BLOCK, _PAGE, _NUM_BLOCKS, pages - _RUNS)
    layout = BlockMajorPoolLayout(_NUM_BLOCKS, 4, 2 * _RUNS, {})
    with pytest.raises(NotImplementedError, match="block-major pool splits"):
        kv_page_runs(_block_table(), _BLOCK, _PAGE, _NUM_BLOCKS, 4 * pages, layout)
    busy = _block_table()
    busy.append_row([5], 3)
    with pytest.raises(NotImplementedError, match="probe row is in use"):
        kv_page_runs(busy, _BLOCK, _PAGE, _NUM_BLOCKS, pages)
    # An expansion that strides a manager block's pages across the pool.
    strided = _block_table()
    monkeypatch.setattr(
        strided,
        "map_to_kernel_blocks",
        lambda ids, per_block, arange: (
            ids.reshape(-1, 1) + _NUM_BLOCKS * arange
        ).reshape(-1),
    )
    with pytest.raises(NotImplementedError, match="map to kernel pages"):
        kv_page_runs(strided, _BLOCK, _PAGE, _NUM_BLOCKS, pages)


def _batched_impl(**attrs):
    impl = object.__new__(adapter.PallasBatchedRPAAttentionBackendImpl)
    impl._pool_is_seq_along_lane = True
    impl.head_size = 256
    impl.sliding_window = None
    impl.kv_sharing_target_layer_name = None
    impl.kv_run_pages = None
    for key, value in attrs.items():
        setattr(impl, key, value)
    return impl


def _manager(monkeypatch, impls, *, pcp=1, dcp=1):
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    rebuilt = []
    runner = SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=pcp, decode_context_parallel_size=dcp
        ),
        vllm_config=None,
        input_batch=SimpleNamespace(block_table=[_block_table()]),
        _initialize_attention_kernels=lambda force: rebuilt.append(force),
    )
    layers = {name: SimpleNamespace(impl=impl) for name, impl in impls.items()}
    monkeypatch.setattr(
        kv_cache_manager, "get_layers_from_vllm_config", lambda *_: layers
    )
    spec = FullAttentionSpec(
        block_size=_BLOCK, num_kv_heads=1, head_size=256, dtype=torch.bfloat16
    )
    config = SimpleNamespace(
        num_blocks=_NUM_BLOCKS,
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec, layer_names=list(impls))],
    )
    pool = torch.empty((_NUM_BLOCKS * _RUNS, 1), dtype=torch.uint8)
    materialized = SimpleNamespace(
        bm_pool_layout=None, kv_caches=dict.fromkeys(impls, pool)
    )

    def bind():
        KVCacheManager(runner)._bind_kv_page_runs(config, {0: _PAGE}, materialized)

    return bind, rebuilt


@pytest.mark.cpu_test
def test_binding_hands_every_attention_layer_its_runs(monkeypatch):
    impls = {
        "model.layers.3.attn": _batched_impl(),
        "mtp.layers.0.attn": _batched_impl(),
    }
    bind, rebuilt = _manager(monkeypatch, impls)
    bind()
    assert [impl.kv_run_pages for impl in impls.values()] == [_RUNS, _RUNS]
    # The custom ops are rebuilt with the runs.
    assert rebuilt == [True]


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("impl", "kwargs", "match"),
    [
        (SimpleNamespace(), {}, "does not run the batched RPA kernel"),
        (_batched_impl(_pool_is_seq_along_lane=False), {}, "SEQ_ALONG_LANE"),
        (_batched_impl(head_size=64), {}, "head size other than 64"),
        (_batched_impl(sliding_window=1024), {}, "no sliding window"),
        (_batched_impl(kv_sharing_target_layer_name="a"), {}, "its own KV cache"),
        (_batched_impl(), {"pcp": 2}, "context parallelism"),
        (_batched_impl(), {"dcp": 2}, "context parallelism"),
    ],
)
def test_binding_refuses_layers_that_would_not_copy_runs(
    monkeypatch, impl, kwargs, match
):
    bind, rebuilt = _manager(monkeypatch, {"model.layers.3.attn": impl}, **kwargs)
    with pytest.raises(NotImplementedError, match=match):
        bind()
    assert not rebuilt


@pytest.mark.cpu_test
def test_kernel_entry_binds_page_runs(monkeypatch):
    seen = {}

    def capture(*args, rpa_func, **kwargs):
        seen["rpa_func"] = rpa_func

    monkeypatch.setattr(adapter, "_pallas_rpa_kernel_impl", capture)
    monkeypatch.delenv("TPU_BATCHED_RPA_DECODE_BLOCK", raising=False)
    monkeypatch.delenv("TPU_BATCHED_RPA_XLA_SCHEDULE", raising=False)
    call = functools.partial(
        adapter._pallas_rpa_kernel_batched,
        *([None] * 12),
        mesh=None,
        sliding_window=None,
        skip_kv_update=False,
        kv_layout=_LANE,
    )
    call(kv_run_pages=_RUNS)
    assert seen["rpa_func"].func is adapter.ragged_paged_attention_batched
    assert seen["rpa_func"].keywords == {"kv_run_pages": _RUNS}
    call()
    assert seen["rpa_func"] is adapter.ragged_paged_attention_batched


@pytest.mark.cpu_test
def test_build_rpa_kernel_binds_kv_run_pages(monkeypatch):
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.config.attention import AttentionConfig
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        set_vllm_model_wrapper_context,
    )

    captured = []

    class FakeOp:
        def register_fake(self, _fake_impl):
            pass

    def fake_jax_op(_name, wrapped_fn, **_kwargs):
        captured.append(wrapped_fn)
        return FakeOp()

    monkeypatch.setattr(adapter.PallasAttentionBackendImpl, "_kernel_registry", {})
    monkeypatch.setattr(
        adapter.PallasAttentionBackendImpl,
        "_select_kernel_mesh",
        staticmethod(lambda _mesh, _pcp: (object(), object(), None)),
    )
    monkeypatch.setattr(adapter.pallas, "jax_op", fake_jax_op)
    mesh = SimpleNamespace(shape={"attn_dp": 1, "expert": 1, "model": 1})
    config = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        attention_config=AttentionConfig(backend=AttentionBackendEnum.CUSTOM),
    )
    # HND (SEQ_ALONG_LANE pages), set on the config so that neither
    # VLLM_KV_CACHE_LAYOUT nor vLLM's env cache picks the layout.
    config.cache_config.kv_cache_layout = "LBHNC"

    def impl():
        with set_current_vllm_config(config):
            layer = adapter.PallasBatchedRPAAttentionBackendImpl(
                num_heads=2,
                head_size=128,
                scale=1.0,
                num_kv_heads=1,
                alibi_slopes=None,
                sliding_window=None,
                kv_cache_dtype="bfloat16",
            )
        layer.decode_query_size = 4
        return layer

    def build(layer):
        with set_vllm_model_wrapper_context(mesh=mesh, vllm_config=None):
            layer._build_rpa_kernel(None, None, None)
        return captured[-1].keywords

    plain, runs = impl(), impl()
    assert runs.kv_layout is _LANE
    runs.set_kv_run_pages(_RUNS)
    assert "kv_run_pages" not in build(plain)
    assert build(runs)["kv_run_pages"] == _RUNS
    # Layers with and without runs get separate kernels.
    assert len(captured) == 2

    def one_page_per_dma(
        *args,
        mesh,
        sliding_window,
        sm_scale,
        soft_cap,
        q_scale,
        k_scale,
        v_scale,
        skip_kv_update,
        use_causal_mask,
        kv_layout,
        decode_query_size=1,
    ):
        raise AssertionError("not called")

    other = impl()
    other.set_kv_run_pages(_RUNS)
    other._kernel_entry = one_page_per_dma
    other._kernel_op_prefix = "pallas::rpa_kernel_one_page_per_dma"
    with pytest.raises(NotImplementedError, match="one page per DMA"):
        build(other)
