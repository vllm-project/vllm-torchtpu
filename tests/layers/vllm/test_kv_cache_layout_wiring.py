"""Wiring of `VLLM_KV_CACHE_LAYOUT` onto the batched-RPA KV cache layout.

    unset / NHD -> HEAD_ALONG_SUBLANE, byte-identical to the inherited layout
    HND         -> SEQ_ALONG_LANE, tokens on lanes and head_dim in the words

Only the second makes an fp8 page smaller than a bf16 page when a rank holds a
single KV head (b/510425663).
"""
import contextlib

import pytest
import torch

from vllm_torchtpu.kernels.experimental.batched_rpa import \
    configs as batched_rpa_configs
from vllm_torchtpu.layers.vllm.attention import (
    PallasAttentionBackend, PallasBatchedRPAAttentionBackend,
    resolve_batched_rpa_kv_layout)
from vllm_torchtpu.platforms.tpu_block_size_utils import \
    validate_kv_layout_supports_unified_pool

BF16 = torch.bfloat16
FP8 = torch.float8_e4m3fn
KVLayout = batched_rpa_configs.KVLayout

BASE = PallasAttentionBackend
BATCHED = PallasBatchedRPAAttentionBackend

# (num_kv_heads, head_size); 1 KV head is the b/510425663 case (a TP=4 rank of
# a 4-KV-head model), 128 and 256 are the head widths in use on this backend.
HEAD_CASES = [(1, 128), (1, 256), (2, 128), (4, 128), (8, 128), (8, 256)]


@contextlib.contextmanager
def _layout_env(monkeypatch, value):
    """Select a layout for the duration of one test.

    `EngineCore` startup calls `enable_envs_cache()`, which freezes `vllm.envs`
    process-wide; `tests/entrypoints/` sorts before this file and would leave
    these tests asserting against a stale value. The cache is restored only
    after `monkeypatch.undo()`, so it can never re-cache a patched value.
    """
    from vllm import envs as vllm_envs
    was_cached = vllm_envs._is_envs_cache_enabled()
    vllm_envs.disable_envs_cache()
    if value is None:
        monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
    else:
        monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", value)
    try:
        yield
    finally:
        monkeypatch.undo()
        if was_cached:
            vllm_envs.enable_envs_cache()


@pytest.fixture
def nhd(monkeypatch):
    """Default selection. Explicitly unset so a dirty environment can't pass."""
    with _layout_env(monkeypatch, None):
        yield


@pytest.fixture
def hnd(monkeypatch):
    with _layout_env(monkeypatch, "HND"):
        yield


def _require_tpu():
    """Skip when no TPU is available. Only the kernel-agreement test needs one;
    everything else here is integer arithmetic."""
    import jax
    try:
        backend = jax.default_backend()
    except RuntimeError as exc:
        pytest.skip(f"no TPU backend available: {exc}")
    if backend != "tpu":
        pytest.skip(f"requires TPU backend, found {backend!r}")


def page_bytes(backend, num_kv_heads, head_size, dtype, block_size=128):
    return backend.get_kv_cache_page_size_bytes(block_size, num_kv_heads,
                                                head_size, dtype)


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------


def test_unset_selects_head_along_sublane(nhd):
    assert resolve_batched_rpa_kv_layout() is KVLayout.HEAD_ALONG_SUBLANE


def test_nhd_selects_head_along_sublane(monkeypatch):
    with _layout_env(monkeypatch, "NHD"):
        assert resolve_batched_rpa_kv_layout() is KVLayout.HEAD_ALONG_SUBLANE


def test_hnd_selects_seq_along_lane(hnd):
    assert resolve_batched_rpa_kv_layout() is KVLayout.SEQ_ALONG_LANE


# --------------------------------------------------------------------------
# The default must not move: this backend is behaviour-neutral without opt-in
# --------------------------------------------------------------------------


@pytest.mark.parametrize("num_kv_heads,head_size", HEAD_CASES)
@pytest.mark.parametrize("dtype", [BF16, FP8])
@pytest.mark.parametrize("block_size", [16, 128, 256])
def test_default_shape_matches_inherited(nhd, num_kv_heads, head_size, dtype,
                                         block_size):
    assert BATCHED.get_kv_cache_shape(7, block_size, num_kv_heads, head_size,
                                      dtype) == BASE.get_kv_cache_shape(
                                          7, block_size, num_kv_heads,
                                          head_size, dtype)


def test_default_block_size_unchanged(nhd):
    assert BATCHED.get_supported_kernel_block_sizes() == [256]


def test_seq_along_lane_requires_page_size_128(hnd):
    # `validate_inputs` in batched_rpa/configs.py rejects any other page size.
    assert BATCHED.get_supported_kernel_block_sizes() == [128]


# --------------------------------------------------------------------------
# The fix itself
# --------------------------------------------------------------------------


@pytest.mark.parametrize("num_kv_heads,head_size", HEAD_CASES)
def test_fp8_halves_the_page_at_every_head_count(hnd, num_kv_heads, head_size):
    bf16 = page_bytes(BATCHED, num_kv_heads, head_size, BF16)
    fp8 = page_bytes(BATCHED, num_kv_heads, head_size, FP8)
    assert bf16 == 2 * fp8, (num_kv_heads, head_size, bf16, fp8)


def test_single_kv_head_fp8_saves_nothing_in_the_default_layout(nhd):
    """The bug this wiring fixes, kept as a live guard so the two layouts
    cannot quietly converge."""
    assert (page_bytes(BATCHED, 1, 128,
                       BF16) == page_bytes(BATCHED, 1, 128, FP8))


@pytest.mark.parametrize("num_kv_heads,head_size", HEAD_CASES)
def test_bf16_pages_are_identical_across_layouts(hnd, num_kv_heads, head_size):
    """The layout change must move fp8 only, never bf16. `BASE` is layout-blind,
    so it stands in for the default layout while `hnd` is in effect."""
    assert (page_bytes(BATCHED, num_kv_heads, head_size,
                       BF16) == page_bytes(BASE, num_kv_heads, head_size,
                                           BF16))


# --------------------------------------------------------------------------
# Shape contracts the rest of the stack depends on
# --------------------------------------------------------------------------


@pytest.mark.parametrize("num_kv_heads,head_size", HEAD_CASES)
@pytest.mark.parametrize("dtype", [BF16, FP8])
def test_shape_matches_the_kernel_wrapper(hnd, num_kv_heads, head_size, dtype):
    """The backend and the kernel must agree, or `validate_inputs` rejects the
    cache the backend just allocated."""
    _require_tpu()
    jax_dtype = {BF16: "bfloat16", FP8: "float8_e4m3"}[dtype]
    import jax.numpy as jnp

    from vllm_torchtpu.kernels.experimental.batched_rpa import wrapper
    assert BATCHED.get_kv_cache_shape(7, 128, num_kv_heads, head_size,
                                      dtype) == wrapper.get_kv_cache_shape(
                                          7,
                                          128,
                                          num_kv_heads,
                                          head_size,
                                          jnp.dtype(jax_dtype),
                                          kv_layout=KVLayout.SEQ_ALONG_LANE)


@pytest.mark.parametrize("dtype", [BF16, FP8])
def test_page_size_bytes_follows_the_allocated_shape(hnd, dtype):
    """It used to hardcode the base class, which would have made the unified
    pool size pages the kernel does not allocate."""
    shape = BATCHED.get_kv_cache_shape(1, 128, 2, 128, dtype)
    numel = 1
    for dim in shape:
        numel *= dim
    itemsize = torch.empty((), dtype=dtype).element_size()
    assert page_bytes(BATCHED, 2, 128, dtype) == numel * itemsize


@pytest.mark.parametrize("dtype", [BF16, FP8])
def test_num_blocks_stays_at_dim_zero(hnd, dtype):
    """vLLM's offloading connector locates the block axis with
    `shape.index(num_blocks)`; both layouts must keep it at dim 0."""
    assert BATCHED.get_kv_cache_shape(4242, 128, 2, 128, dtype)[0] == 4242
    assert BATCHED.get_kv_cache_shape(4242, 128, 2, 128, "auto")[0] == 4242


# Both KV-head axis positions are pinned here, because the partition spec in
# `sharded_ragged_paged_attention` has to track them.


def test_kv_head_axis_is_dim_two_by_default(nhd):
    shape = BATCHED.get_kv_cache_shape(7, 128, 4, 128, BF16)
    assert shape[2] == 4 * 2 // 2  # ceil(Kx2 / packing)
    assert shape[1] == 128  # page_size


def test_kv_head_axis_is_dim_one_under_seq_along_lane(hnd):
    shape = BATCHED.get_kv_cache_shape(7, 128, 4, 128, BF16)
    assert shape[1] == 4 * 2  # Kx2 outright
    assert shape[4] == 128  # page_size innermost


def test_head_dim_64_is_refused(hnd):
    """head_dim==64 is rerouted to the hd64 kernel, which only reads
    HEAD_ALONG_SUBLANE pages."""
    with pytest.raises(NotImplementedError, match="head_dim==64"):
        BATCHED.get_kv_cache_shape(7, 128, 2, 64, BF16)


def test_head_dim_64_still_works_by_default(nhd):
    assert BATCHED.get_kv_cache_shape(7, 128, 2, 64, BF16) == \
        BASE.get_kv_cache_shape(7, 128, 2, 64, BF16)


# --------------------------------------------------------------------------
# Unified block pool
# --------------------------------------------------------------------------


def test_unified_pool_rejects_seq_along_lane(hnd):
    with pytest.raises(NotImplementedError, match="unified block pool"):
        validate_kv_layout_supports_unified_pool(BATCHED)


def test_unified_pool_allows_the_default_layout(nhd):
    validate_kv_layout_supports_unified_pool(BATCHED)


def test_unified_pool_ignores_other_backends(hnd):
    """Only the batched-RPA backend reads the layout; the guard must not fire
    for a backend that cannot select it."""
    validate_kv_layout_supports_unified_pool(BASE)


# --------------------------------------------------------------------------
# Coexistence with the longctx fork
# --------------------------------------------------------------------------
#
# `USE_BATCHED_RPA_LONGCTX` swaps in the longctx fork, which has its own
# seq-along-lane switch. `VLLM_KV_CACHE_LAYOUT` drives the mainline fork only.


@pytest.fixture
def longctx(monkeypatch):
    """Turn on the longctx fork for one test.

    Patched as a module attribute, not via `os.environ`: that shadows
    `envs.__getattr__`, so it survives
    `vllm_torchtpu.envs.enable_envs_cache()`, which has no way back.
    """
    from vllm_torchtpu import envs
    monkeypatch.setattr(envs, "USE_BATCHED_RPA_LONGCTX", True, raising=False)
    yield


def test_hnd_under_longctx_is_refused(hnd, longctx):
    with pytest.raises(NotImplementedError,
                       match="USE_BATCHED_RPA_SEQ_ON_LANE"):
        resolve_batched_rpa_kv_layout()


def test_longctx_block_sizes_are_untouched(nhd, longctx):
    """The mainline `[128]` must not leak into the longctx fork's list."""
    assert BATCHED.get_supported_kernel_block_sizes() == [
        128, 256, 512, 1024, 2048, 4096
    ]


def test_page_size_bytes_is_not_overridden():
    """Inherited on purpose: the base classmethod sizes itself from
    `cls.get_kv_cache_shape`, so an override would repeat the layout branch."""
    assert "get_kv_cache_page_size_bytes" not in vars(BATCHED)
