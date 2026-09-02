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

BF16 = torch.bfloat16
FP8 = torch.float8_e4m3fn
KVLayout = batched_rpa_configs.KVLayout

BASE = PallasAttentionBackend
BATCHED = PallasBatchedRPAAttentionBackend

# (num_kv_heads, head_size); 1 KV head is the b/510425663 case (a TP=4 rank of
# a 4-KV-head model), 128 and 256 are the head widths in use on this backend.
HEAD_CASES = [(1, 128), (1, 256), (2, 128), (4, 128), (8, 128), (8, 256)]
# Subset for properties that only need to show a value is carried through, not
# recomputed: one KV head (the b/510425663 case), several heads, and a wide head.
HEAD_CASES_CORE = [(1, 128), (2, 128), (8, 256)]


@contextlib.contextmanager
def _layout_env(monkeypatch, value):
    """Select a layout for the duration of one test.

    `EngineCore` startup calls `enable_envs_cache()`, which freezes `vllm.envs`
    process-wide; `tests/entrypoints/` sorts before this file and would leave
    these tests asserting against a stale value. The cache is restored only
    after `monkeypatch.undo()`, so it can never re-cache a patched value.

    `envs.seq_along_lane_enabled()` reads this env var live on each call, so
    setting it here is enough -- nothing on our side caches the derived value.
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


# Selection


@pytest.mark.parametrize("value,expected", [
    (None, KVLayout.HEAD_ALONG_SUBLANE),
    ("NHD", KVLayout.HEAD_ALONG_SUBLANE),
    ("HND", KVLayout.SEQ_ALONG_LANE),
])
def test_env_selects_layout(monkeypatch, value, expected):
    with _layout_env(monkeypatch, value):
        assert resolve_batched_rpa_kv_layout() is expected


# The default must not move: this backend is behaviour-neutral without opt-in


@pytest.mark.parametrize("num_kv_heads,head_size", HEAD_CASES)
@pytest.mark.parametrize("dtype", [BF16, FP8])
@pytest.mark.parametrize("block_size", [16, 256])
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


# The fix itself


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


# Shape contracts the rest of the stack depends on


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


@pytest.mark.parametrize("layout,kv_head_axis,kv_heads,page_axis",
                         [("NHD", 2, 4 * 2 // 2, 1), ("HND", 1, 4 * 2, 4)])
def test_kv_head_and_page_axis_positions(monkeypatch, layout, kv_head_axis,
                                         kv_heads, page_axis):
    with _layout_env(monkeypatch, layout):
        shape = BATCHED.get_kv_cache_shape(7, 128, 4, 128, BF16)
    assert shape[kv_head_axis] == kv_heads
    assert shape[page_axis] == 128  # page_size


def test_head_dim_64_is_refused_under_hnd_and_delegated_by_default(
        monkeypatch):
    """head_dim==64 is rerouted to the hd64 kernel, which only reads
    HEAD_ALONG_SUBLANE pages."""
    with _layout_env(monkeypatch, "HND"):
        with pytest.raises(NotImplementedError, match="head_dim==64"):
            BATCHED.get_kv_cache_shape(7, 128, 2, 64, BF16)
    with _layout_env(monkeypatch, None):
        assert BATCHED.get_kv_cache_shape(7, 128, 2, 64, BF16) == \
            BASE.get_kv_cache_shape(7, 128, 2, 64, BF16)


# Unified block pool


def test_unified_pool_accepts_seq_along_lane(hnd):
    """SEQ_ALONG_LANE can back the unified pool: the GDN custom op reshapes
    the attention pool into its own row view at the op boundary rather than
    requiring token-contiguous pages.

    This pins the removal of `validate_kv_layout_supports_unified_pool`, which
    rejected exactly this combination."""
    from vllm_torchtpu.platforms import tpu_block_size_utils
    assert not hasattr(tpu_block_size_utils,
                       "validate_kv_layout_supports_unified_pool")


@pytest.mark.parametrize("layout,reshapes", [("HND", True), (None, False)])
def test_pooled_gdn_state_view_follows_the_layout(monkeypatch, layout,
                                                  reshapes):
    """The pool reshape is keyed on the layout, not on a fork-local flag, so
    it fires for the mainline kernel too -- and no-ops by default."""
    import jax.numpy as jnp

    from vllm_torchtpu.layers.vllm.custom_ops.gdn_attention_op import \
        _pooled_gdn_state_view

    # (num_blocks, num_kv_heads_x2, packed_head_dim, kv_packing, page_size)
    state = jnp.zeros((1, 2, 64, 4, 128), dtype=jnp.float8_e4m3fn)
    with _layout_env(monkeypatch, layout):
        viewed = _pooled_gdn_state_view(state)
    assert (viewed.shape != state.shape) is reshapes
    assert viewed.size == state.size


# Coexistence with the longctx fork
#
# `USE_BATCHED_RPA_LONGCTX` swaps in the longctx fork. Both forks now take
# their layout from `VLLM_KV_CACHE_LAYOUT`; the longctx fork maps it onto its
# own `KVLayout` enum inside its wrapper.


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


def test_hnd_selects_seq_along_lane_under_longctx(hnd, longctx):
    """HND drives both forks. Previously refused, because the longctx fork
    took its layout from a separate flag."""
    assert resolve_batched_rpa_kv_layout() is KVLayout.SEQ_ALONG_LANE


def test_longctx_wrapper_reads_the_same_layout_env(hnd, longctx):
    """The longctx fork resolves HND onto its own `KVLayout` enum, which is a
    distinct class from the mainline one -- so the two must be compared by
    name, not identity."""
    from vllm_torchtpu import envs as tpu_envs
    from vllm_torchtpu.kernels.experimental.batched_rpa_longctx import \
        configs as longctx_configs
    assert tpu_envs.seq_along_lane_enabled()
    assert (longctx_configs.KVLayout.SEQ_ALONG_LANE.name ==
            KVLayout.SEQ_ALONG_LANE.name)
    assert longctx_configs.KVLayout is not KVLayout


def test_longctx_block_sizes_are_untouched(nhd, longctx):
    """The mainline `[128]` must not leak into the longctx fork's list."""
    assert BATCHED.get_supported_kernel_block_sizes() == [
        128, 256, 512, 1024, 2048, 4096
    ]


def test_page_size_bytes_is_not_overridden():
    """Inherited on purpose: the base classmethod sizes itself from
    `cls.get_kv_cache_shape`, so an override would repeat the layout branch."""
    assert "get_kv_cache_page_size_bytes" not in vars(BATCHED)


# The page size the memory planner actually sees
#
# `normalize_kv_cache_specs_for_tpu` runs over every spec the runner builds and
# is the last thing to touch `page_size_padded`, which vLLM divides into free
# memory to get `num_blocks`. It used to recompute that from a hardcoded
# `PallasAttentionBackend`, discarding the layout-aware value the runner had
# just set.


def _normalized_page_bytes(backend, num_kv_heads, head_size, dtype):
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    from vllm_torchtpu.kv_cache_spec_normalizer import \
        normalize_kv_cache_specs_for_tpu
    spec = FullAttentionSpec(
        block_size=128,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
        page_size_padded=page_bytes(backend, num_kv_heads, head_size, dtype),
        indexes_kv_by_block_stride=True,
    )
    normalized = normalize_kv_cache_specs_for_tpu(
        {"l": spec}, dtype, attention_backend_cls=backend)
    return normalized["l"].page_size_bytes


@pytest.mark.parametrize("num_kv_heads,head_size", HEAD_CASES_CORE)
def test_normalizer_preserves_the_fp8_halving(hnd, num_kv_heads, head_size):
    bf16 = _normalized_page_bytes(BATCHED, num_kv_heads, head_size, BF16)
    fp8 = _normalized_page_bytes(BATCHED, num_kv_heads, head_size, FP8)
    assert bf16 == 2 * fp8, (num_kv_heads, head_size, bf16, fp8)


@pytest.mark.parametrize("num_kv_heads,head_size", HEAD_CASES_CORE)
@pytest.mark.parametrize("dtype", [BF16, FP8])
def test_normalizer_agrees_with_the_allocated_page(hnd, num_kv_heads,
                                                   head_size, dtype):
    """The planner's per-block budget must equal what gets allocated, or blocks
    are counted against a page nobody builds."""
    assert (_normalized_page_bytes(BATCHED, num_kv_heads, head_size,
                                   dtype) == page_bytes(
                                       BATCHED, num_kv_heads, head_size,
                                       dtype))


@pytest.mark.parametrize("num_kv_heads,head_size", HEAD_CASES_CORE)
@pytest.mark.parametrize("dtype", [BF16, FP8])
def test_normalizer_leaves_the_default_layout_alone(nhd, num_kv_heads,
                                                    head_size, dtype):
    assert (_normalized_page_bytes(BATCHED, num_kv_heads, head_size,
                                   dtype) == page_bytes(
                                       BASE, num_kv_heads, head_size, dtype))


def test_layout_blind_backend_would_lose_the_halving(hnd):
    """Non-vacuousness: the base class the normalizer used to hardcode reports
    a bf16-sized page for fp8, which is the bug these tests guard."""
    assert (_normalized_page_bytes(BASE, 1, 128,
                                   FP8) == _normalized_page_bytes(
                                       BASE, 1, 128, BF16))


# Head width read off the cache
#
# The impl pads a layer narrower than the page it writes into. That width is
# `shape[-1]` under HEAD_ALONG_SUBLANE but `shape[2] * shape[3]` under
# SEQ_ALONG_LANE, whose last axis is page_size.


def _width(impl_cls, shape):
    import torch as _torch
    impl = impl_cls.__new__(impl_cls)
    return impl._cache_head_width(_torch.empty(shape, device="meta"))


@pytest.mark.parametrize("head_size", [80, 128, 256])
@pytest.mark.parametrize("layout", ["HND", None])
def test_cache_head_width_follows_the_layout(monkeypatch, layout, head_size):
    from vllm_torchtpu.layers.vllm.attention import \
        PallasBatchedRPAAttentionBackendImpl as BATCHED_IMPL
    with _layout_env(monkeypatch, layout):
        shape = BATCHED.get_kv_cache_shape(4, 128, 2, head_size, BF16)
        width = _width(BATCHED_IMPL, shape)
    if layout == "HND":
        # Last axis is page_size here, so the naive read would be wrong.
        assert width == shape[2] * shape[3] and shape[-1] == 128
    else:
        assert width == shape[-1]


def test_narrow_head_is_not_spuriously_padded(hnd):
    """head_size 80 stores 80 wide under SEQ_ALONG_LANE, so no padding is due.
    Reading `shape[-1]` would have padded q/k/v to 128 against an 80-wide
    cache."""
    from vllm_torchtpu.layers.vllm.attention import \
        PallasBatchedRPAAttentionBackendImpl as BATCHED_IMPL
    shape = BATCHED.get_kv_cache_shape(4, 128, 2, 80, BF16)
    assert _width(BATCHED_IMPL, shape) == 80
    assert shape[-1] > 80  # what the bug read


def test_cache_head_width_under_longctx_seq_along_lane(hnd, longctx):
    """The longctx fork reaches `_cache_head_width` too, and its pages are
    SEQ_ALONG_LANE under HND just like the mainline fork's.

    Regression guard for the rebase onto #608: that PR keyed this branch on
    `USE_BATCHED_RPA_SEQ_ON_LANE`, so resolving the width purely from the
    mainline layout would silently return `shape[-1]` here and pad q/k/v
    against a cache that is not that wide."""
    from vllm_torchtpu.layers.vllm.attention import \
        PallasBatchedRPAAttentionBackendImpl as BATCHED_IMPL

    # (num_blocks, num_kv_heads_x2, packed_head_dim, kv_packing, page_size)
    shape = (4, 2, 64, 4, 128)
    assert _width(BATCHED_IMPL, shape) == 256
    assert shape[-1] == 128  # what reading the last axis would have returned


# Migration off the removed flag


def test_removed_flag_is_rejected_at_import(monkeypatch):
    """A deployment still setting `USE_BATCHED_RPA_SEQ_ON_LANE` must fail
    loudly: silently ignoring it would halve KV cache capacity with no
    signal."""
    import importlib
    monkeypatch.setenv("USE_BATCHED_RPA_SEQ_ON_LANE", "1")
    import vllm_torchtpu.layers.common.attention_interface as ai
    with pytest.raises(ValueError, match="has been removed"):
        importlib.reload(ai)
    monkeypatch.delenv("USE_BATCHED_RPA_SEQ_ON_LANE")
    importlib.reload(ai)
