"""Wiring of `VLLM_KV_CACHE_LAYOUT` onto the batched-RPA KV cache layout.

    unset / NHD -> HEAD_ALONG_SUBLANE, byte-identical to the inherited layout
    HND         -> SEQ_ALONG_LANE, tokens on lanes and head_dim in the words

Only the second makes an fp8 page smaller than a bf16 page when a rank holds a
single KV head (b/510425663).
"""
import contextlib

import pytest
import torch
from vllm.v1.attention.backends.utils import get_kv_cache_layout

from vllm_torchtpu.kernels.experimental.batched_rpa import \
    configs as batched_rpa_configs
from vllm_torchtpu.layers.vllm.attention import (
    KV_LAYOUT_BY_VLLM_LAYOUT, PallasAttentionBackend,
    PallasBatchedRPAAttentionBackend)

BF16 = torch.bfloat16
FP8 = torch.float8_e4m3fn
KVLayout = batched_rpa_configs.KVLayout

BASE = PallasAttentionBackend
BATCHED = PallasBatchedRPAAttentionBackend

# (num_kv_heads, head_size); 1 KV head is the b/510425663 case (a TP=4 rank of
# a 4-KV-head model), 128 and 256 are the head widths in use on this backend.
HEAD_CASES = [(1, 128), (1, 256), (2, 128), (4, 128), (8, 128), (8, 256)]
# Subset for properties that only carry a value through rather than recompute it.
HEAD_CASES_CORE = [(1, 128), (2, 128), (8, 256)]


@contextlib.contextmanager
def _layout_env(monkeypatch, value):
    """Select a layout for the duration of one test.

    Three caches sit between the env var and `get_kv_cache_layout()`: the
    `vllm.envs` freeze an earlier `tests/entrypoints/` run leaves behind, the
    accessor's own `lru_cache`, and -- with the env unset -- the KV connector
    it falls through to, which needs an active config and resolves to "NHD".
    """
    from vllm import envs as vllm_envs
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.v1.attention.backends.utils import set_kv_cache_layout

    was_cached = vllm_envs._is_envs_cache_enabled()
    vllm_envs.disable_envs_cache()
    if value is None:
        monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
    else:
        monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", value)
    set_kv_cache_layout(None)
    try:
        with set_current_vllm_config(VllmConfig()):
            yield
    finally:
        monkeypatch.undo()
        set_kv_cache_layout(None)
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
        assert KV_LAYOUT_BY_VLLM_LAYOUT[get_kv_cache_layout()] is expected


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


# Pinned because `sharded_ragged_paged_attention`'s partition spec tracks them.


@pytest.mark.parametrize("layout,kv_head_axis,kv_heads,page_axis",
                         [("NHD", 2, 4 * 2 // 2, 1), ("HND", 1, 4 * 2, 4)])
def test_kv_head_and_page_axis_positions(monkeypatch, layout, kv_head_axis,
                                         kv_heads, page_axis):
    with _layout_env(monkeypatch, layout):
        shape = BATCHED.get_kv_cache_shape(7, 128, 4, 128, BF16)
    assert shape[kv_head_axis] == kv_heads
    assert shape[page_axis] == 128  # page_size


@pytest.mark.parametrize("layout", ["HND", None])
def test_head_dim_64_delegates_whatever_the_layout(monkeypatch, layout):
    """head_dim==64 is rerouted to the hd64 kernel, which reads only
    HEAD_ALONG_SUBLANE pages, so the shape comes from the base class.

    HND + head_dim 64 is refused at config time instead (#825): the pages are
    fine for the hd64 kernel, but `get_kv_cache_layout()` still answers "HND"
    process-wide, so `forward` reads a head width of 2 for a 64-wide head and
    skips the padding it owes."""
    with _layout_env(monkeypatch, layout):
        assert BATCHED.get_kv_cache_shape(7, 128, 2, 64, BF16) == \
            BASE.get_kv_cache_shape(7, 128, 2, 64, BF16)


def test_head_dim_64_delegates_on_longctx_too(hnd, longctx):
    """The longctx fork takes the same escape."""
    assert BATCHED.get_kv_cache_shape(7, 128, 2, 64, BF16) == \
        BASE.get_kv_cache_shape(7, 128, 2, 64, BF16)


# Unified block pool


def test_unified_pool_accepts_seq_along_lane(hnd):
    """SEQ_ALONG_LANE can back the unified pool: the GDN op reshapes at its
    own boundary rather than needing token-contiguous pages. Pins the removal
    of `validate_kv_layout_supports_unified_pool`."""
    from vllm_torchtpu.platforms import tpu_block_size_utils
    assert not hasattr(tpu_block_size_utils,
                       "validate_kv_layout_supports_unified_pool")


# Coexistence with the longctx fork


@pytest.fixture
def longctx(monkeypatch):
    """Turn on the longctx fork for one test.

    Patched as a module attribute, not via `os.environ`: that survives
    `enable_envs_cache()`, which has no way back.
    """
    from vllm_torchtpu import envs
    monkeypatch.setattr(envs, "USE_BATCHED_RPA_LONGCTX", True, raising=False)
    yield


def test_hnd_selects_seq_along_lane_under_longctx(hnd, longctx):
    """HND drives both forks. Previously refused, because the longctx fork
    took its layout from a separate flag."""
    assert (KV_LAYOUT_BY_VLLM_LAYOUT[get_kv_cache_layout()]
            is KVLayout.SEQ_ALONG_LANE)


def test_longctx_wrapper_reads_the_same_layout_env(hnd, longctx):
    """The longctx fork resolves HND onto its own `KVLayout` enum, which is a
    distinct class from the mainline one -- so the two must be compared by
    name, not identity."""
    from vllm_torchtpu.kernels.experimental.batched_rpa_longctx import \
        configs as longctx_configs
    assert (KV_LAYOUT_BY_VLLM_LAYOUT[get_kv_cache_layout()]
            is KVLayout.SEQ_ALONG_LANE)
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
    normalized = normalize_kv_cache_specs_for_tpu({"l": spec},
                                                  dtype,
                                                  attention_backend=backend)
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


@pytest.mark.parametrize("head_size", [80, 128, 256])
@pytest.mark.parametrize("layout", ["HND", None])
def test_page_carries_head_width_where_forward_reads_it(
        monkeypatch, layout, head_size):
    """`forward` reads `shape[2] * shape[3]` under HND and `shape[-1]`
    otherwise; pin that the page really is shaped that way.

    head_size 80 separates the two: SEQ_ALONG_LANE stores it 80 wide, so
    reading `shape[-1]` would pad q/k/v to 128 against an 80-wide cache."""
    _require_tpu()
    with _layout_env(monkeypatch, layout):
        shape = BATCHED.get_kv_cache_shape(4, 128, 2, head_size, BF16)
    if layout == "HND":
        assert shape[2] * shape[3] == head_size
        # page_size, which is what the naive read would have returned
        assert shape[-1] == 128
    else:
        assert shape[-1] >= head_size


def test_longctx_hnd_page_is_shaped_like_mainline(hnd, longctx):
    """Both forks put the head width in dims 2-3 under HND, so the single
    layout-keyed branch in `forward` covers both.

    Regression guard for the rebase onto #608, which keyed that branch on
    `USE_BATCHED_RPA_SEQ_ON_LANE`."""
    _require_tpu()
    shape = BATCHED.get_kv_cache_shape(4, 128, 2, 256, BF16)
    assert shape[2] * shape[3] == 256
    assert shape[-1] == 128


def test_no_layout_lookup_on_the_compiled_forward_path():
    """Nothing on the forward path may call `get_kv_cache_layout()`.

    It asks the KV connector, which needs a current vLLM config; a compiled
    forward has none, and Dynamo traces into the accessor so its `lru_cache`
    does not spare us. Callers there must use a value resolved at
    construction.

    A source check, because reproducing it needs a real engine -- it surfaced
    only in CI, as `ObservedAssertionErrorError` raised inside the Dynamo
    region."""
    import inspect

    from vllm_torchtpu.layers.vllm import attention as attn

    on_forward_path = [
        attn.PallasAttentionBackendImpl.forward,
        attn.PallasAttentionBackendImpl._validate_pcp_streaming_support,
        attn._pallas_rpa_kernel_batched,
    ]
    offenders = [
        fn.__qualname__ for fn in on_forward_path
        if "get_kv_cache_layout" in inspect.getsource(fn)
    ]
    assert not offenders, f"{offenders} resolve the KV layout at forward time"


def test_raiden_geometry_resolves_the_layout_under_a_config():
    """`resolve_kernel_geometry` runs in the EngineCore, where no config is set.

    Both calls resolve the KV cache layout, which falls through to the KV
    connector and asserts without one.
    """
    import ast
    import inspect
    import textwrap

    from vllm_torchtpu.offload import raiden_store

    tree = ast.parse(
        textwrap.dedent(inspect.getsource(
            raiden_store.resolve_kernel_geometry)))
    guarded = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue
        if not any(
                isinstance(i.context_expr, ast.Call)
                and getattr(i.context_expr.func, "id",
                            None) == "set_current_vllm_config"
                for i in node.items):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Call):
                f = inner.func
                guarded.add(f.attr if isinstance(f, ast.Attribute
                                                 ) else getattr(f, "id", None))

    assert {"select_common_block_size", "get_kv_cache_shape"} <= guarded


@pytest.mark.parametrize("num_kv_heads,head_size", HEAD_CASES_CORE)
@pytest.mark.parametrize("dtype", [BF16, FP8])
def test_default_layout_shape_needs_no_tpu_client(nhd, monkeypatch,
                                                  num_kv_heads, head_size,
                                                  dtype):
    """The shape must be arithmetic, not a device query.

    `resolve_kernel_geometry` calls this from the scheduler process, which owns
    no chip -- the workers hold them all. The batched wrapper reads lane and
    sublane geometry via `pltpu.get_tpu_info()`, so routing the default layout
    through it made engine startup die with
    `TPU initialization failed: open(/dev/vfio/N): Device or resource busy`.
    """
    from jax.experimental.pallas import tpu as pltpu

    def no_device(*_args, **_kwargs):
        raise AssertionError("get_kv_cache_shape queried the TPU")

    monkeypatch.setattr(pltpu, "get_tpu_info", no_device)
    assert BATCHED.get_kv_cache_shape(4, 128, num_kv_heads, head_size,
                                      dtype) == BASE.get_kv_cache_shape(
                                          4, 128, num_kv_heads, head_size,
                                          dtype)
