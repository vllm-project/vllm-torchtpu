"""Host-only unit tests for the block-major unified KV pool.

Verifies block-ID remapping across pool consumers, region index derivation
for both layer-outermost and block-outermost vLLM placements, merged pool
materialization, and Raiden transfer/offload contract registration.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
)
from vllm.v1.worker.utils import AttentionGroup

from vllm_torchtpu.block_major_pool import (
    BlockMajorPoolLayout,
    flat_kernel_ids,
    flat_manager_ids,
    kernel_view,
    unified_block_major_enabled,
)
from vllm_torchtpu.distributed.kv_transfer.raiden import pool_manifest as rpm
from vllm_torchtpu.kv_cache_materializer import (
    layer_to_pool_index,
    materialize_kv_cache_tensors,
    resolve_block_major_pool_layout,
)


class FakeAttentionBackend:
    @staticmethod
    def get_kv_cache_shape(
        num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str
    ):
        return (num_blocks, block_size, num_kv_heads, 2, head_size)


BLOCK = 16
PAGE = BLOCK * 2 * 2 * 8 * torch.bfloat16.itemsize
ATTN = [f"model.layers.{i}.self_attn" for i in (0, 4, 8)]
MAMBA = [f"model.layers.{i}.mamba" for i in (1, 2, 3, 5, 6, 7, 9, 10, 11)]


def _specs():
    attn = FullAttentionSpec(
        block_size=BLOCK,
        num_kv_heads=2,
        head_size=8,
        dtype=torch.bfloat16,
        page_size_padded=PAGE,
    )
    mamba = MambaSpec(
        block_size=BLOCK,
        shapes=((2, 8), (2, 4, 8)),
        dtypes=(torch.bfloat16, torch.float32),
        page_size_padded=PAGE,
    )
    return attn, mamba


def _groups():
    attn, mamba = _specs()
    return [
        KVCacheGroupSpec(layer_names=ATTN, kv_cache_spec=attn),
        KVCacheGroupSpec(layer_names=MAMBA[0::3], kv_cache_spec=mamba),
        KVCacheGroupSpec(layer_names=MAMBA[1::3], kv_cache_spec=mamba),
        KVCacheGroupSpec(layer_names=MAMBA[2::3], kv_cache_spec=mamba),
    ]


def layer_outermost_config(num_blocks=4) -> KVCacheConfig:
    """Build a layer-outermost KVCacheConfig where each of the 3 regions spans
    `num_blocks * PAGE` contiguous bytes."""
    pool = num_blocks * PAGE
    tensors = [
        KVCacheTensor(
            size=3 * pool,
            layers=list(group.layer_names),
            layer_stride=pool,
            block_stride=PAGE,
        )
        for group in _groups()
    ]
    return KVCacheConfig(
        num_blocks=num_blocks, kv_cache_tensors=tensors, kv_cache_groups=_groups()
    )


def block_outermost_config(num_blocks=4) -> KVCacheConfig:
    """Build a block-outermost KVCacheConfig where each scheduler block spans
    3 consecutive region pages."""
    pool = num_blocks * PAGE
    tensors = [
        KVCacheTensor(
            size=3 * pool,
            layers=list(group.layer_names),
            layer_stride=PAGE,
            block_stride=3 * PAGE,
        )
        for group in _groups()
    ]
    return KVCacheConfig(
        num_blocks=num_blocks, kv_cache_tensors=tensors, kv_cache_groups=_groups()
    )


def _attn_groups(cfg):
    return [
        [
            AttentionGroup(
                backend=FakeAttentionBackend,
                layer_names=list(group.layer_names),
                kv_cache_spec=group.kv_cache_spec,
                kv_cache_group_id=gid,
            )
        ]
        for gid, group in enumerate(cfg.kv_cache_groups)
    ]


# --- id remaps ---------------------------------------------------------------


def test_flat_manager_ids():
    ids = torch.tensor([0, 1, 5], dtype=torch.int32)
    assert flat_manager_ids(ids, 2, 3).tolist() == [2, 5, 17]
    assert flat_manager_ids(ids, 0, 1) is ids


def test_flat_kernel_ids_split_one():
    ids = torch.tensor([0, 1, 5], dtype=torch.int32)
    assert flat_kernel_ids(ids, 2, 3, 1).tolist() == [2, 5, 17]


def test_flat_kernel_ids_keeps_kernel_blocks_consecutive():
    # split=4: manager block 1 is kernel blocks 4..7; in the merged pool
    # (P=3, p=2) manager block 1 is flat manager block 5, kernel 20..23.
    ids = torch.arange(4, 8, dtype=torch.int32)
    assert flat_kernel_ids(ids, 2, 3, 4).tolist() == [20, 21, 22, 23]
    # Padding stays inside the null manager block.
    assert flat_kernel_ids(torch.zeros(2, dtype=torch.int32), 2, 3, 4).tolist() == [
        8,
        8,
    ]


def test_flat_kernel_ids_matches_manager_ids():
    ids = torch.arange(0, 24, dtype=torch.int32)
    split, num_pools, p = 4, 3, 1
    expected = flat_manager_ids(ids // split, p, num_pools) * split + ids % split
    assert flat_kernel_ids(ids, p, num_pools, split).tolist() == expected.tolist()


# --- RPA entries -------------------------------------------------------------

_POOL_6D = (4, 3 * 2, BLOCK // 2, 2, 2, 8)


def _rpa_entry_args(pool):
    q = torch.zeros(8, 4, 8)
    ints = torch.zeros(4, dtype=torch.int32)
    return (pool, q, q, q, ints, ints, ints, ints, None, None, None, None)


@pytest.mark.parametrize(
    "entry_name",
    [
        "_pallas_rpa_kernel_default",
        "_pallas_rpa_kernel_batched",
    ],
)
@pytest.mark.parametrize("block_major", [True, False])
def test_rpa_entries_fold_the_merged_pool(entry_name, block_major):
    from vllm_torchtpu.layers.adapter import attention as adapter

    seen = {}

    def fake_attention(pool, query, *_args, **_kwargs):
        seen["pool_shape"] = tuple(pool.shape)
        return pool, query

    entry = getattr(adapter, entry_name)
    pool = torch.zeros(_POOL_6D if block_major else (4 * 3 * 2,) + _POOL_6D[2:])
    extra = {"kv_layout": None} if entry_name == "_pallas_rpa_kernel_batched" else {}
    with patch.object(adapter, "attention", fake_attention):
        new_pool, _ = entry(
            *_rpa_entry_args(pool),
            mesh=None,
            sliding_window=None,
            skip_kv_update=False,
            block_major=block_major,
            **extra,
        )
    assert seen["pool_shape"] == (4 * 3 * 2,) + _POOL_6D[2:]
    assert tuple(new_pool.shape) == tuple(pool.shape)


@pytest.mark.parametrize("block_major", [True, False])
def test_pcp_streaming_entry_folds_the_merged_pool(block_major):
    from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa import vllm_adapter

    seen = {}

    def fake_kernel(*, kv_cache, q, **_kwargs):
        seen["pool_shape"] = tuple(kv_cache.shape)
        return q, kv_cache

    entry = vllm_adapter.make_pcp_streaming_rpa_kernel(
        q_scale=None,
        k_scale=None,
        v_scale=None,
        mesh=None,
        sliding_window=None,
        skip_kv_update=False,
        cp_kv_cache_interleave_size=BLOCK,
        block_major=block_major,
    )
    pool = torch.zeros(_POOL_6D if block_major else (4 * 3 * 2,) + _POOL_6D[2:])
    with patch.object(vllm_adapter, "sharded_pcp_ragged_paged_attention", fake_kernel):
        new_pool, _ = entry(*_rpa_entry_args(pool)[:8])
    assert seen["pool_shape"] == (4 * 3 * 2,) + _POOL_6D[2:]
    assert tuple(new_pool.shape) == tuple(pool.shape)


# --- mamba row copy ----------------------------------------------------------


@pytest.mark.parametrize(
    "rows, block_bytes, budget, expected",
    [
        (10 * 16, 256 << 10, 8 << 20, 32),
        (10 * 16, 256 << 10, 16 << 20, 40),
        (10 * 16, 256 << 10, 64 << 20, 160),
        (10 * 1, 4 << 20, 8 << 20, 2),
        (10 * 1, 4 << 20, 4 << 20, 1),
        (7, 1, 3, 1),
    ],
)
def test_row_copy_step_rows_is_the_largest_fitting_divisor(
    rows, block_bytes, budget, expected
):
    from vllm_torchtpu.layers.adapter.custom_ops.mamba_state_copy_op import (
        _row_copy_step_rows,
    )

    assert _row_copy_step_rows(rows, block_bytes, budget) == expected


def test_row_copy_step_rows_rejects_an_oversized_kernel_block():
    from vllm_torchtpu.layers.adapter.custom_ops.mamba_state_copy_op import (
        _row_copy_step_rows,
    )

    with pytest.raises(ValueError, match="exceeds"):
        _row_copy_step_rows(160, 16 << 20, 8 << 20)


def test_copy_mamba_state_rows_sizes_the_step_from_the_pool():
    from vllm_torchtpu.layers.adapter.custom_ops import mamba_state_copy_op

    pool = torch.zeros(4, 3 * 2, BLOCK // 2, 2, 2, 8, dtype=torch.bfloat16)
    kernel_block_bytes = (BLOCK // 2) * 2 * 2 * 8 * 2
    seen = {}

    def fake_row_copy_fn(num_pools, split, step_rows):
        seen["key"] = (num_pools, split, step_rows)
        return lambda *_args: torch.tensor(0)

    with (
        patch.object(
            mamba_state_copy_op,
            "_copy_step_budget_bytes",
            lambda: 3 * kernel_block_bytes,
        ),
        patch.object(mamba_state_copy_op, "_row_copy_fn", fake_row_copy_fn),
    ):
        mamba_state_copy_op.copy_mamba_state_rows(
            pool, torch.tensor([1]), torch.tensor([2]), num_pools=3, split=2
        )
    assert seen["key"] == (3, 2, 3)


# --- selection ---------------------------------------------------------------


@pytest.mark.parametrize(
    "flag, unified, hybrid, expected",
    [
        (True, True, True, True),
        (False, True, True, False),
        (True, False, True, False),
        (True, True, False, False),
    ],
)
def test_unified_block_major_enabled(flag, unified, hybrid, expected):
    from vllm_torchtpu import envs

    cfg = SimpleNamespace(model_config=SimpleNamespace(is_hybrid=hybrid))
    with (
        patch.object(envs, "VLLM_TPU_BLOCK_MAJOR_KV", flag),
        patch(
            "vllm_torchtpu.platforms.tpu_block_size_utils.unified_kv_layout_enabled",
            return_value=unified,
        ),
    ):
        assert unified_block_major_enabled(cfg) is expected


# --- pool index derivation ---------------------------------------------------


@pytest.mark.parametrize("make", [layer_outermost_config, block_outermost_config])
def test_layer_to_pool_index_both_placements(make):
    indices = layer_to_pool_index(make())
    assert set(indices.values()) == {0, 1, 2}
    for i, name in enumerate(ATTN):
        assert indices[name] == i
    for group_layers in (MAMBA[0::3], MAMBA[1::3], MAMBA[2::3]):
        assert [indices[n] for n in group_layers] == [0, 1, 2]


def test_layer_to_pool_index_rejects_foreign_stride():
    cfg = layer_outermost_config()
    cfg.kv_cache_tensors[0].block_stride = 2 * PAGE
    with pytest.raises(AssertionError, match="layer-compact or block-compact"):
        layer_to_pool_index(cfg)


# --- materialization ---------------------------------------------------------


@pytest.mark.parametrize("make", [layer_outermost_config, block_outermost_config])
def test_block_major_materializes_one_merged_pool(make):
    cfg = make(num_blocks=4)
    kernel_block = BLOCK // 2
    out = materialize_kv_cache_tensors(
        kv_cache_config=cfg,
        attn_groups=_attn_groups(cfg),
        kernel_block_sizes=[kernel_block] * 4,
        device=torch.device("cpu"),
        cache_dtype="auto",
        block_major=True,
    )
    assert len(out.raw_tensors) == 1
    (pool,) = out.raw_tensors
    # One row per scheduler block: P * split kernel blocks of the kernel
    # block size.
    assert tuple(pool.shape) == (4, 3 * 2, kernel_block, 2, 2, 8)
    assert pool.nbytes == 4 * 3 * PAGE
    layout = out.bm_pool_layout
    assert layout == BlockMajorPoolLayout(
        num_blocks=4, num_pools=3, split=2, pool_index_by_layer=layer_to_pool_index(cfg)
    )
    # The allocation describes the pool exactly as the config-only
    # derivation the offload contract uses does.
    assert layout == resolve_block_major_pool_layout(cfg, kernel_block)
    assert layout.rows_per_block == pool.shape[1]
    assert out.layer_to_raw == {name: pool for name in ATTN + MAMBA}
    assert tuple(kernel_view(pool).shape) == (4 * 3 * 2, kernel_block, 2, 2, 8)
    for name in ATTN:
        assert out.kv_caches[name] is pool
    for name in MAMBA:
        assert out.kv_caches[name] == [pool]


def test_registration_view_leads_with_the_allocation_dim():
    """Verify the transfer plane can view the merged pool without changing `shape[0]`.

    Raiden reconstructs a typed `(num_blocks, num_pools, split, *page)` view over
    the pool's untyped storage and requires the leading dimension to match the
    allocated `num_blocks`.
    """
    cfg = layer_outermost_config(num_blocks=4)
    kernel_block = BLOCK // 2
    out = materialize_kv_cache_tensors(
        kv_cache_config=cfg,
        attn_groups=_attn_groups(cfg),
        kernel_block_sizes=[kernel_block] * 4,
        device=torch.device("cpu"),
        cache_dtype="auto",
        block_major=True,
    )
    (pool,) = out.raw_tensors
    layout = out.bm_pool_layout
    kernel_shape = (kernel_block, 2, 2, 8)
    registered = (
        torch.empty(0, dtype=pool.dtype)
        .set_(pool.untyped_storage())
        .view((layout.num_blocks, layout.num_pools, layout.split) + kernel_shape)
    )
    assert registered.shape[0] == pool.shape[0]
    assert registered.nbytes == pool.nbytes
    assert registered.reshape(pool.shape).data_ptr() == pool.data_ptr()


def test_layer_major_materialization_unchanged():
    cfg = layer_outermost_config(num_blocks=4)
    out = materialize_kv_cache_tensors(
        kv_cache_config=cfg,
        attn_groups=_attn_groups(cfg),
        kernel_block_sizes=[BLOCK] * 4,
        device=torch.device("cpu"),
        cache_dtype="auto",
    )
    assert out.bm_pool_layout is None
    assert len(out.raw_tensors) == 3
    assert tuple(out.raw_tensors[0].shape) == (4, BLOCK, 2, 2, 8)


# --- Stage-3 pool manifest ----------------------------------------------------

_GDN_GEOMETRY = rpm.GdnHeadGeometry(
    local_key_heads=1,
    local_value_heads=1,
    key_head_dim=2,
    value_head_dim=4,
    conv_kernel_size=3,
)


def _manifest(out):
    return rpm.build_qwen35_pool_manifest(
        named_kv_caches=out.kv_caches,
        kv_cache_groups=_groups(),
        raw_tensors=out.raw_tensors,
        gdn_geometry=_GDN_GEOMETRY,
        mamba_group_ordinal_by_layer={
            name: gid
            for gid, group in enumerate(_groups()[1:])
            for name in group.layer_names
        },
        block_major=out.bm_pool_layout,
    )


@pytest.mark.parametrize("make", [layer_outermost_config, block_outermost_config])
def test_block_major_manifest_keeps_the_layer_major_regions(make):
    """Verify that the block-major manifest preserves per-pool region descriptors
    and live bytes while consolidating storage into a single multi-region row stride."""
    cfg = make(num_blocks=4)
    kernel_block = BLOCK // 2
    kwargs = dict(
        kv_cache_config=cfg,
        attn_groups=_attn_groups(cfg),
        kernel_block_sizes=[kernel_block] * 4,
        device=torch.device("cpu"),
        cache_dtype="auto",
    )
    lm = _manifest(materialize_kv_cache_tensors(**kwargs))
    bm_out = materialize_kv_cache_tensors(**kwargs, block_major=True)
    bm = _manifest(bm_out)

    assert len(lm.storages) == 3
    assert bm.storages == [bm_out.raw_tensors[0]]
    assert [(p.tag, p.layer_name) for p in bm.pools] == [
        (p.tag, p.layer_name) for p in lm.pools
    ]
    regions = layer_to_pool_index(cfg)
    for lm_pool, bm_pool in zip(lm.pools, bm.pools):
        assert lm_pool.block_stride_bytes == PAGE
        assert bm_pool.block_stride_bytes == 3 * PAGE
        assert bm_pool.num_blocks == lm_pool.num_blocks == 4
        assert bm_pool.storage_index == 0
        assert bm_pool.base_offset_bytes == (
            regions[bm_pool.layer_name] * PAGE + lm_pool.base_offset_bytes
        )
        assert bm_pool.regions == lm_pool.regions
        assert bm_pool.dtype_tag == lm_pool.dtype_tag

    def without_stride(geometry):
        return {
            tag: {k: v for k, v in entry.items() if k != "block_stride_bytes"}
            for tag, entry in geometry.items()
        }

    assert without_stride(bm.geometry_by_tag()) == without_stride(lm.geometry_by_tag())
    rpm.verify_storage_binding(bm, bm_out.kv_caches, bm_out.raw_tensors)


# --- raiden geometry and contract --------------------------------------------

_KERNEL_SHAPE = (BLOCK // 2, 2, 2, 8)


def _patched_geometry(block_major: bool):
    """Patch the environment so the offload geometry resolves the test
    config against the fake backend with a kernel block of BLOCK // 2."""
    from vllm_torchtpu.offload import raiden_store

    backend = MagicMock()
    backend.is_ssm.return_value = False
    backend.get_kv_cache_shape.side_effect = lambda **kw: (
        kw["num_blocks"],
        kw["block_size"],
        2,
        2,
        8,
    )
    return [
        patch.object(raiden_store.envs, "VLLM_TPU_BLOCK_MAJOR_KV", block_major),
        patch(
            "vllm_torchtpu.platforms.tpu_block_size_utils.unified_kv_layout_enabled",
            return_value=True,
        ),
        patch(
            "vllm_torchtpu.platforms.tpu_platform.TpuPlatform._find_non_ssm_backend",
            return_value=backend,
        ),
        patch("vllm.v1.worker.utils.select_common_block_size", return_value=BLOCK // 2),
        patch("vllm.config.set_current_vllm_config"),
    ]


def _with(patches):
    import contextlib

    stack = contextlib.ExitStack()
    for p in patches:
        stack.enter_context(p)
    return stack


def test_geometry_reports_manager_blocks_under_block_major():
    from vllm_torchtpu.offload.raiden_store import resolve_kernel_geometry

    cfg = layer_outermost_config()
    with _with(_patched_geometry(False)):
        kernel, shape, dtype, device = resolve_kernel_geometry(MagicMock(), cfg)
    assert (kernel, shape, device) == (BLOCK // 2, _KERNEL_SHAPE, BLOCK)
    with _with(_patched_geometry(True)):
        kernel, shape, dtype, device = resolve_kernel_geometry(MagicMock(), cfg)
    assert (kernel, shape, device) == (BLOCK, (2,) + _KERNEL_SHAPE, BLOCK)


def test_unified_contract_folds_regions_into_the_row():
    from vllm_torchtpu.offload import block_major_layout as bml

    cfg = layer_outermost_config()
    with (
        _with(_patched_geometry(True)),
        patch.object(bml.tpu_envs, "VLLM_TPU_BLOCK_MAJOR_KV", True),
    ):
        contract = bml.resolve_block_major_contract(MagicMock(), cfg)
    assert contract.fragment_count == 3
    assert contract.fragment_row_bytes == PAGE
    assert contract.bundle_row_bytes == 3 * PAGE
    with (
        _with(_patched_geometry(True)),
        patch.object(bml.tpu_envs, "VLLM_TPU_BLOCK_MAJOR_KV", True),
    ):
        again = bml.resolve_block_major_contract(MagicMock(), block_outermost_config())
    assert again.logical_fingerprint == contract.logical_fingerprint


def test_worker_views_the_merged_pool_by_scheduler_block():
    from vllm.v1.kv_offload.base import (
        CanonicalKVCacheRef,
        CanonicalKVCaches,
        CanonicalKVCacheTensor,
    )

    from vllm_torchtpu.offload import block_major_layout as bml
    from vllm_torchtpu.offload.raiden_store import RaidenStoreOffloadingWorker

    cfg = layer_outermost_config(num_blocks=4)
    with (
        _with(_patched_geometry(True)),
        patch.object(bml.tpu_envs, "VLLM_TPU_BLOCK_MAJOR_KV", True),
    ):
        contract = bml.resolve_block_major_contract(MagicMock(), cfg)
        row = contract.bundle_row_bytes
        canonical = CanonicalKVCaches(
            tensors=[
                CanonicalKVCacheTensor(
                    tensor=torch.zeros(4, row, dtype=torch.int8), page_size_bytes=row
                )
            ],
            group_data_refs=[[CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=row)]],
        )
        worker = RaidenStoreOffloadingWorker(
            canonical,
            vllm_config=MagicMock(),
            kv_cache_config=cfg,
            host_blocks_to_allocate=8,
            controller_address="127.0.0.1:1",
            rank=0,
            block_major_contract=contract,
        )
    try:
        (view,) = worker._device_tensors[0]
        assert tuple(view.shape) == (4, 3, 2) + _KERNEL_SHAPE
        assert view.dtype == torch.bfloat16
    finally:
        worker.shutdown()


def test_register_kv_caches_canonicalizes_the_merged_pool():
    from vllm_torchtpu.offload.raiden_connector import TPURaidenOffloadingConnector

    cfg = layer_outermost_config(num_blocks=4)
    out = materialize_kv_cache_tensors(
        kv_cache_config=cfg,
        attn_groups=_attn_groups(cfg),
        kernel_block_sizes=[BLOCK // 2] * 4,
        device=torch.device("cpu"),
        cache_dtype="auto",
        block_major=True,
    )
    captured = {}
    fake = SimpleNamespace(
        connector_worker=SimpleNamespace(
            kv_cache_config=cfg,
            _init_worker=lambda canonical: captured.setdefault("c", canonical),
        )
    )
    TPURaidenOffloadingConnector.register_kv_caches(fake, out.kv_caches)
    canonical = captured["c"]
    assert len(canonical.tensors) == 1
    assert canonical.tensors[0].page_size_bytes == 3 * PAGE
    assert tuple(canonical.tensors[0].tensor.shape) == (4, 3 * PAGE)
    assert len(canonical.group_data_refs) == 4


# --- mamba row copy expansion ------------------------------------------------


def _row_copy_rows_fn(num_pools, split, step_rows):
    """Build `_rows_fn` for the geometry without registering a torch op."""
    from vllm_torchtpu.layers.adapter.custom_ops import mamba_state_copy_op

    captured = {}

    def fake_jax_op(_name, fn, **_kwargs):
        captured["rows_fn"] = fn
        return fn

    with patch.object(mamba_state_copy_op.pallas, "jax_op", fake_jax_op):
        mamba_state_copy_op._row_copy_fn.__wrapped__(num_pools, split, step_rows)
    return captured["rows_fn"]


def _gather_copy_blocks(pool, src, dst):
    out = np.array(pool)
    out[np.asarray(dst)] = np.asarray(pool)[np.asarray(src)]
    return jnp.asarray(out)


@pytest.mark.parametrize("step_rows", [2, 6], ids=["steps=3", "steps=1"])
def test_row_copy_expands_manager_pairs_into_whole_rows(step_rows):
    from vllm_torchtpu.kernels import pool_adapters

    num_pools, split = 3, 2
    pool = jnp.arange(4 * num_pools * split * 2 * 3, dtype=jnp.int32).reshape(
        4, num_pools * split, 2, 3
    )
    src = jnp.array([1, 3], dtype=jnp.int32)
    dst = jnp.array([2, 0], dtype=jnp.int32)
    reference = np.array(pool)
    reference[2] = np.array(pool)[1]
    reference[0] = np.array(pool)[3]

    rows_fn = _row_copy_rows_fn(num_pools, split, step_rows)
    with patch.object(pool_adapters, "copy_blocks", _gather_copy_blocks):
        new_pool, marker = rows_fn(pool, src, dst)

    assert new_pool.shape == pool.shape
    np.testing.assert_array_equal(np.array(new_pool), reference)
    assert int(marker) == 1


# --- runner binding ----------------------------------------------------------


def _kv_cache_manager(**runner_attrs):
    from vllm_torchtpu.runner.kv_cache_manager import KVCacheManager

    runner = SimpleNamespace(
        speculative_config=None,
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1, pipeline_parallel_size=1
        ),
        use_spmd=False,
        shared_kv_cache_layers={},
        vllm_config=object(),
    )
    for name, value in runner_attrs.items():
        setattr(runner, name, value)
    return KVCacheManager(runner)


@pytest.mark.parametrize(
    "runner_attrs, feature",
    [
        ({"speculative_config": object()}, "speculative decoding"),
        (
            {
                "parallel_config": SimpleNamespace(
                    decode_context_parallel_size=2, pipeline_parallel_size=1
                )
            },
            "decode context parallelism",
        ),
        (
            {
                "parallel_config": SimpleNamespace(
                    decode_context_parallel_size=1, pipeline_parallel_size=2
                )
            },
            "pipeline parallelism",
        ),
        ({"use_spmd": True}, "SPMD"),
    ],
)
def test_block_major_pool_refuses_unsupported_features(runner_attrs, feature):
    manager = _kv_cache_manager(**runner_attrs)
    with pytest.raises(NotImplementedError, match=feature):
        manager._check_block_major_pool_supported()


def test_block_major_pool_accepts_the_plain_config():
    _kv_cache_manager()._check_block_major_pool_supported()


def _bind_layers(monkeypatch, layers, shared=None):
    from vllm_torchtpu.runner import kv_cache_manager as kcm

    monkeypatch.setattr(kcm, "get_layers_from_vllm_config", lambda _cfg, _base: layers)
    layout = BlockMajorPoolLayout(
        num_blocks=4,
        num_pools=3,
        split=4,
        pool_index_by_layer={ATTN[2]: 2, MAMBA[1]: 1},
    )
    manager = _kv_cache_manager(shared_kv_cache_layers=shared or {})
    manager._bind_block_major_pool_layers(layout)
    return layout


def test_bind_block_major_pool_layers_hands_each_layer_its_region(monkeypatch):
    from vllm_torchtpu.layers.adapter.attention import PallasAttentionBackendImpl
    from vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op import (
        VllmGatedDeltaNetAttention,
    )

    attn_impl = MagicMock(spec=PallasAttentionBackendImpl)
    shared_impl = MagicMock(spec=PallasAttentionBackendImpl)
    gdn = MagicMock(spec=VllmGatedDeltaNetAttention)
    layers = {
        ATTN[2]: SimpleNamespace(impl=attn_impl),
        MAMBA[1]: gdn,
        "model.layers.12.self_attn": SimpleNamespace(impl=shared_impl),
    }
    layout = _bind_layers(
        monkeypatch, layers, shared={"model.layers.12.self_attn": ATTN[2]}
    )
    attn_impl.set_block_major_pool.assert_called_once_with(2, layout)
    shared_impl.set_block_major_pool.assert_called_once_with(2, layout)
    gdn.set_block_major_pool.assert_called_once_with(1, layout)


def test_bind_block_major_pool_layers_rejects_other_layer_types(monkeypatch):
    from vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op import (
        VllmGatedDeltaNetAttention,
    )

    class OtherImpl:
        pass

    layers = {
        ATTN[2]: SimpleNamespace(impl=OtherImpl()),
        MAMBA[1]: MagicMock(spec=VllmGatedDeltaNetAttention),
    }
    with pytest.raises(NotImplementedError, match=ATTN[2].replace(".", r"\.")):
        _bind_layers(monkeypatch, layers)
