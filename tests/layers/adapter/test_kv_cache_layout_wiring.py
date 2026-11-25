# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Wiring of `VLLM_KV_CACHE_LAYOUT` onto the batched-RPA KV cache layout.

    unset / NHD -> HEAD_ALONG_SUBLANE, byte-identical to the inherited layout
    HND         -> SEQ_ALONG_LANE, tokens on lanes and head_dim in the words

Only the second makes an fp8 page smaller than a bf16 page when a rank holds a
single KV head.
"""

import contextlib
import functools
import math

import pytest
import torch
from vllm.config import get_current_vllm_config
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

from vllm_torchtpu.kernels.experimental.batched_rpa import (
    configs as batched_rpa_configs,
)
from vllm_torchtpu.kv_cache_materializer import materialize_shared_attention_cache
from vllm_torchtpu.layers.adapter.attention import (
    KV_LAYOUT_BY_VLLM_LAYOUT,
    PallasAttentionBackend,
    PallasBatchedRPAAttentionBackend,
)
from vllm_torchtpu.layers.core.attention_interface import uses_seq_along_lane_kv_page

BF16 = torch.bfloat16
FP8 = torch.float8_e4m3fn
KVLayout = batched_rpa_configs.KVLayout

BASE = PallasAttentionBackend
BATCHED = PallasBatchedRPAAttentionBackend

# (num_kv_heads, head_size); 1 KV head (a TP=4 rank of a 4-KV-head model) is the
# case where only SEQ_ALONG_LANE halves the fp8 page. 128 and 256 are the head
# widths in use on this backend.
HEAD_CASES = [(1, 128), (1, 256), (2, 128), (4, 128), (8, 128), (8, 256)]
# Subset for properties that only carry a value through rather than recompute it.
HEAD_CASES_CORE = [(1, 128), (2, 128), (8, 256)]


@contextlib.contextmanager
def _layout_env(monkeypatch, value):
    """Resolve legacy env aliases through the native config API."""
    from vllm import envs as vllm_envs
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.config.attention import AttentionConfig
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

    was_cached = vllm_envs._is_envs_cache_enabled()
    vllm_envs.disable_envs_cache()
    if value is None:
        monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
    else:
        monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", value)
    config = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        attention_config=AttentionConfig(backend=AttentionBackendEnum.CUSTOM),
    )
    resolve_kv_cache_layout(config, [["LBNHC", "LBHNC"]])
    try:
        with set_current_vllm_config(config):
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
    return backend.get_kv_cache_page_size_bytes(
        block_size, num_kv_heads, head_size, dtype
    )


# Selection


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, KVLayout.HEAD_ALONG_SUBLANE),
        ("NHD", KVLayout.HEAD_ALONG_SUBLANE),
        ("HND", KVLayout.SEQ_ALONG_LANE),
    ],
)
def test_env_selects_layout(monkeypatch, value, expected):
    with _layout_env(monkeypatch, value):
        assert (
            KV_LAYOUT_BY_VLLM_LAYOUT[
                get_current_vllm_config().cache_config.get_resolved_kv_cache_layout()
            ]
            is expected
        )


# The default must not move: this backend is behaviour-neutral without opt-in


@pytest.mark.parametrize("num_kv_heads,head_size", HEAD_CASES)
@pytest.mark.parametrize("dtype", [BF16, FP8])
@pytest.mark.parametrize("block_size", [16, 256])
def test_default_shape_matches_inherited(
    nhd, num_kv_heads, head_size, dtype, block_size
):
    assert BATCHED.get_kv_cache_shape(
        7, block_size, num_kv_heads, head_size, dtype
    ) == BASE.get_kv_cache_shape(7, block_size, num_kv_heads, head_size, dtype)


KERNEL_BLOCK_SIZES = [128, 256]


@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize("dtype", [BF16, FP8])
@pytest.mark.parametrize("head_dim", [64, 80, 256, 512])
def test_shared_pool_traces_without_vllm_config(monkeypatch, layout, dtype, head_dim):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from vllm.config import get_current_vllm_config_or_none

    from vllm_torchtpu.layers.adapter import attention as adapter

    packing = 4 // dtype.itemsize
    with _layout_env(monkeypatch, layout):
        native_shape = BATCHED.get_kv_cache_shape(8, 128, 2, head_dim, dtype)
        words_per_token = 2 * math.prod(native_shape[1:]) // (128 * packing)
        pool_shape = BATCHED.get_shared_pool_shape(
            4, 128, packing, words_per_token, [head_dim]
        )
        kv_layout = KV_LAYOUT_BY_VLLM_LAYOUT[
            get_current_vllm_config().cache_config.get_resolved_kv_cache_layout()
        ]
    assert get_current_vllm_config_or_none() is None

    def attention(kv_cache, query, key, value, metadata, mesh, **kwargs):
        assert kv_cache.shape == native_shape
        assert kwargs["kv_layout"] == kv_layout
        return kv_cache, query

    monkeypatch.setattr(adapter, "attention", attention)
    mesh = jax.sharding.Mesh(np.array(jax.devices("cpu")[:1]), ("model",))
    run = functools.partial(
        adapter._pallas_rpa_kernel_batched,
        sinks=None,
        q_scale=None,
        k_scale=1.0,
        v_scale=1.0,
        mesh=mesh,
        sliding_window=None,
        skip_kv_update=False,
        kv_layout=kv_layout,
    )
    jax_dtype = {BF16: jnp.bfloat16, FP8: jnp.float8_e4m3fn}[dtype]
    query_shape = (1, 4, head_dim)
    cache, output = jax.eval_shape(
        run,
        jax.ShapeDtypeStruct(pool_shape, jax_dtype),
        jax.ShapeDtypeStruct(query_shape, jnp.bfloat16),
        jax.ShapeDtypeStruct((1, 2, head_dim), jax_dtype),
        jax.ShapeDtypeStruct((1, 2, head_dim), jax_dtype),
        jax.ShapeDtypeStruct((1,), jnp.int32),
        jax.ShapeDtypeStruct((1,), jnp.int32),
        jax.ShapeDtypeStruct((2,), jnp.int32),
        jax.ShapeDtypeStruct((3,), jnp.int32),
    )
    assert cache.shape == pool_shape
    assert cache.dtype == jnp.dtype(jax_dtype)
    assert output.shape == query_shape


def test_batched_kernel_pages_under_nhd(nhd):
    assert BATCHED.get_supported_kernel_block_sizes() == KERNEL_BLOCK_SIZES


def test_batched_kernel_pages_under_hnd_without_pcp(hnd):
    assert BATCHED.get_supported_kernel_block_sizes() == KERNEL_BLOCK_SIZES


# The fix itself


@pytest.mark.parametrize("num_kv_heads,head_size", HEAD_CASES)
def test_fp8_halves_the_page_at_every_head_count(hnd, num_kv_heads, head_size):
    bf16 = page_bytes(BATCHED, num_kv_heads, head_size, BF16)
    fp8 = page_bytes(BATCHED, num_kv_heads, head_size, FP8)
    assert bf16 == 2 * fp8, (num_kv_heads, head_size, bf16, fp8)


def test_single_kv_head_fp8_saves_nothing_in_the_default_layout(nhd):
    """The bug this wiring fixes, kept as a live guard so the two layouts
    cannot quietly converge."""
    assert page_bytes(BATCHED, 1, 128, BF16) == page_bytes(BATCHED, 1, 128, FP8)


@pytest.mark.parametrize("num_kv_heads,head_size", HEAD_CASES)
def test_bf16_pages_are_identical_across_layouts(hnd, num_kv_heads, head_size):
    """The layout change must move fp8 only, never bf16. `BASE` is layout-blind,
    so it stands in for the default layout while `hnd` is in effect."""
    assert page_bytes(BATCHED, num_kv_heads, head_size, BF16) == page_bytes(
        BASE, num_kv_heads, head_size, BF16
    )


# Shape contracts the rest of the stack depends on


@pytest.mark.tpu_test
@pytest.mark.parametrize("num_kv_heads,head_size", HEAD_CASES)
@pytest.mark.parametrize("dtype", [BF16, FP8])
def test_shape_matches_the_kernel_wrapper(hnd, num_kv_heads, head_size, dtype):
    """The backend and the kernel must agree, or `validate_inputs` rejects the
    cache the backend just allocated."""
    _require_tpu()
    jax_dtype = {BF16: "bfloat16", FP8: "float8_e4m3"}[dtype]
    import jax.numpy as jnp

    from vllm_torchtpu.kernels.experimental.batched_rpa import wrapper

    assert BATCHED.get_kv_cache_shape(
        7, 128, num_kv_heads, head_size, dtype
    ) == wrapper.get_kv_cache_shape(
        7,
        128,
        num_kv_heads,
        head_size,
        jnp.dtype(jax_dtype),
        kv_layout=KVLayout.SEQ_ALONG_LANE,
    )


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


@pytest.mark.parametrize(
    "layout,kv_head_axis,kv_heads,page_axis",
    [("NHD", 2, 4 * 2 // 2, 1), ("HND", 1, 4 * 2, 4)],
)
def test_kv_head_and_page_axis_positions(
    monkeypatch, layout, kv_head_axis, kv_heads, page_axis
):
    with _layout_env(monkeypatch, layout):
        shape = BATCHED.get_kv_cache_shape(7, 128, 4, 128, BF16)
    assert shape[kv_head_axis] == kv_heads
    assert shape[page_axis] == 128  # page_size


@pytest.mark.parametrize("layout", ["HND", None])
def test_head_dim_64_delegates_whatever_the_layout(monkeypatch, layout):
    """head_dim==64 is rerouted to the hd64 kernel, which reads only
    HEAD_ALONG_SUBLANE pages, so the shape comes from the base class.

    HND + head_dim 64 is refused at config time instead (#825): the pages are
    fine for the hd64 kernel, but
    `get_current_vllm_config().cache_config.get_resolved_kv_cache_layout()` still
    answers "HND" process-wide, so `forward` reads a head width of 2 for a 64-wide head
    and skips the padding it owes."""
    with _layout_env(monkeypatch, layout):
        assert BATCHED.get_kv_cache_shape(
            7, 128, 2, 64, BF16
        ) == BASE.get_kv_cache_shape(7, 128, 2, 64, BF16)


# Unified block pool


def test_unified_pool_accepts_seq_along_lane(hnd):
    """SEQ_ALONG_LANE can back the unified pool: the GDN op reshapes at its
    own boundary rather than needing token-contiguous pages. Pins the removal
    of `validate_kv_layout_supports_unified_pool`."""
    from vllm_torchtpu.platforms import tpu_block_size_utils

    assert not hasattr(tpu_block_size_utils, "validate_kv_layout_supports_unified_pool")


def test_page_size_bytes_is_not_overridden():
    """Inherited on purpose: the base classmethod sizes itself from
    `cls.get_kv_cache_shape`, so an override would repeat the layout branch."""
    assert "get_kv_cache_page_size_bytes" not in vars(BATCHED)


# The page size the memory planner actually sees


def _normalized_page_bytes(backend, num_kv_heads, head_size, dtype):
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    from vllm_torchtpu.kv_cache_spec_normalizer import normalize_kv_cache_specs_for_tpu

    spec = FullAttentionSpec(
        block_size=128,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
        page_size_padded=page_bytes(backend, num_kv_heads, head_size, dtype),
    )
    normalized = normalize_kv_cache_specs_for_tpu(
        {"l": spec}, dtype, attention_backend=backend
    )
    return normalized["l"].page_size_bytes


@pytest.mark.parametrize("num_kv_heads,head_size", HEAD_CASES_CORE)
def test_normalizer_preserves_the_fp8_halving(hnd, num_kv_heads, head_size):
    bf16 = _normalized_page_bytes(BATCHED, num_kv_heads, head_size, BF16)
    fp8 = _normalized_page_bytes(BATCHED, num_kv_heads, head_size, FP8)
    assert bf16 == 2 * fp8, (num_kv_heads, head_size, bf16, fp8)


@pytest.mark.parametrize("num_kv_heads,head_size", HEAD_CASES_CORE)
@pytest.mark.parametrize("dtype", [BF16, FP8])
def test_normalizer_agrees_with_the_allocated_page(hnd, num_kv_heads, head_size, dtype):
    """The planner's per-block budget must equal what gets allocated, or blocks
    are counted against a page nobody builds."""
    assert _normalized_page_bytes(
        BATCHED, num_kv_heads, head_size, dtype
    ) == page_bytes(BATCHED, num_kv_heads, head_size, dtype)


@pytest.mark.parametrize("num_kv_heads,head_size", HEAD_CASES_CORE)
@pytest.mark.parametrize("dtype", [BF16, FP8])
def test_normalizer_leaves_the_default_layout_alone(
    nhd, num_kv_heads, head_size, dtype
):
    assert _normalized_page_bytes(
        BATCHED, num_kv_heads, head_size, dtype
    ) == page_bytes(BASE, num_kv_heads, head_size, dtype)


def test_layout_blind_backend_would_lose_the_halving(hnd):
    """Non-vacuousness: the base class the normalizer used to hardcode reports
    a bf16-sized page for fp8, which is the bug these tests guard."""
    assert _normalized_page_bytes(BASE, 1, 128, FP8) == _normalized_page_bytes(
        BASE, 1, 128, BF16
    )


# Head width read off the cache


@pytest.mark.tpu_test
@pytest.mark.parametrize("head_size", [80, 128, 256])
@pytest.mark.parametrize("layout", ["HND", None])
def test_page_carries_head_width_where_forward_reads_it(monkeypatch, layout, head_size):
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


def test_no_layout_lookup_on_the_compiled_forward_path():
    """Nothing on the forward path may call
    `get_current_vllm_config().cache_config.get_resolved_kv_cache_layout()`.

    It asks the KV connector, which needs a current vLLM config; a compiled
    forward has none, and Dynamo traces into the accessor so its `lru_cache`
    does not spare us. Callers there must use a value resolved at
    construction.

    A source check, because reproducing it needs a real engine -- it surfaced
    only in CI, as `ObservedAssertionErrorError` raised inside the Dynamo
    region."""
    import inspect

    from vllm_torchtpu.layers.adapter import attention as attn

    on_forward_path = [
        attn.PallasAttentionBackendImpl.forward,
        attn.PallasAttentionBackendImpl._validate_pcp_streaming_support,
        attn._pallas_rpa_kernel_batched,
    ]
    offenders = [
        fn.__qualname__
        for fn in on_forward_path
        if "get_resolved_kv_cache_layout" in inspect.getsource(fn)
    ]
    assert not offenders, f"{offenders} resolve the KV layout at forward time"


def test_raiden_geometry_resolves_the_layout_under_a_config():
    """Verify `resolve_attention_geometry` sets the vLLM config context before
    querying KV cache layout in the EngineCore process."""
    import ast
    import inspect
    import textwrap

    from vllm_torchtpu.offload import raiden_store

    tree = ast.parse(
        textwrap.dedent(inspect.getsource(raiden_store.resolve_attention_geometry))
    )
    guarded = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue
        if not any(
            isinstance(i.context_expr, ast.Call)
            and getattr(i.context_expr.func, "id", None) == "set_current_vllm_config"
            for i in node.items
        ):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Call):
                f = inner.func
                guarded.add(
                    f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
                )

    assert {"select_common_block_size", "get_kv_cache_shape"} <= guarded


@pytest.mark.parametrize("num_kv_heads,head_size", HEAD_CASES_CORE)
@pytest.mark.parametrize("dtype", [BF16, FP8])
def test_default_layout_shape_needs_no_tpu_client(
    nhd, monkeypatch, num_kv_heads, head_size, dtype
):
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
    assert BATCHED.get_kv_cache_shape(
        4, 128, num_kv_heads, head_size, dtype
    ) == BASE.get_kv_cache_shape(4, 128, num_kv_heads, head_size, dtype)


def test_shape_needs_no_config_for_the_probe_and_hd64(monkeypatch):
    """Two calls must answer without reading the KV layout at all.

    The resolved layout lives in the current vLLM config. vLLM's
    `OffloadingConnectorWorker.register_kv_caches` probes this
    method with no dtype and no config, and head_dim 64 routes to the hd64
    kernel, which ignores the layout. Resolving the layout before those two
    early returns made both raise `Current vLLM config is not set`.

    Deliberately no config fixture: the other tests here wrap the body in
    `set_current_vllm_config`, which is exactly what hides this.
    """

    def no_config():
        raise AssertionError("get_kv_cache_shape read the current vLLM config")

    monkeypatch.setattr(
        "vllm_torchtpu.layers.adapter.attention.get_current_vllm_config", no_config
    )
    assert BATCHED.get_kv_cache_shape(
        4, 128, 1, 256, "auto"
    ) == BASE.get_kv_cache_shape(4, 128, 1, 256, "auto")
    assert BATCHED.get_kv_cache_shape(4, 128, 1, 64, BF16) == BASE.get_kv_cache_shape(
        4, 128, 1, 64, BF16
    )


# The bound kernel must read the layout the layer allocated


def _batched_impl():
    from vllm_torchtpu.layers.adapter.attention import (
        PallasBatchedRPAAttentionBackendImpl,
    )

    return PallasBatchedRPAAttentionBackendImpl(
        num_heads=2,
        head_size=128,
        scale=1.0,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="bfloat16",
    )


def _build_rpa_kernel(impl, monkeypatch):
    """Run `_build_rpa_kernel` without compiling; return the bound entry."""
    from types import SimpleNamespace

    from vllm_torchtpu.layers.adapter.attention import PallasAttentionBackendImpl
    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        set_vllm_model_wrapper_context,
    )

    captured = {}

    class FakeOp:
        def register_fake(self, _fake_impl):
            pass

    def fake_jax_op(_name, wrapped_fn, **_kwargs):
        captured["fn"] = wrapped_fn
        return FakeOp()

    monkeypatch.setattr(PallasAttentionBackendImpl, "_kernel_registry", {})
    monkeypatch.setattr(
        PallasAttentionBackendImpl,
        "_select_kernel_mesh",
        staticmethod(lambda _mesh, _pcp: (object(), object(), None)),
    )
    monkeypatch.setattr(
        "vllm_torchtpu.layers.adapter.attention.pallas.jax_op", fake_jax_op
    )
    mesh = SimpleNamespace(shape={"attn_dp": 1, "expert": 1, "model": 1})
    with set_vllm_model_wrapper_context(mesh=mesh, vllm_config=None):
        impl._build_rpa_kernel(None, None, None)
    return captured["fn"]


def test_hnd_refuses_a_kernel_entry_that_ignores_the_layout(hnd, monkeypatch):
    """A tp=1 speculative draft runs `_pallas_rpa_kernel_local`, which takes no
    `kv_layout`."""
    from vllm_torchtpu.layers.adapter.attention import _pallas_rpa_kernel_local

    impl = _batched_impl()
    impl._kernel_entry = _pallas_rpa_kernel_local
    impl._kernel_op_prefix = "pallas::rpa_kernel_local"
    with pytest.raises(NotImplementedError, match="takes no kv_layout"):
        _build_rpa_kernel(impl, monkeypatch)


def test_hnd_binds_kv_layout_into_the_batched_kernel_entry(hnd, monkeypatch):
    impl = _batched_impl()
    assert impl._pool_is_seq_along_lane
    bound = _build_rpa_kernel(impl, monkeypatch)
    assert bound.keywords["kv_layout"] is KVLayout.SEQ_ALONG_LANE


def test_default_layout_accepts_a_layout_blind_kernel_entry(nhd, monkeypatch):
    """`_pallas_rpa_kernel_local` reads HEAD_ALONG_SUBLANE, the default layout."""
    from vllm_torchtpu.layers.adapter.attention import _pallas_rpa_kernel_local

    impl = _batched_impl()
    impl._kernel_entry = _pallas_rpa_kernel_local
    impl._kernel_op_prefix = "pallas::rpa_kernel_local"
    bound = _build_rpa_kernel(impl, monkeypatch)
    assert "kv_layout" not in bound.keywords


def _build_mixed_shared_pool(
    *,
    num_blocks: int,
    block_size: int,
    swa_heads: int,
    swa_hd: int,
    glob_heads: int,
    glob_hd: int,
    dtype: torch.dtype,
    sliding_window: int,
) -> tuple[torch.Tensor, tuple[int, ...], tuple[int, ...], int, int, int]:
    swa_b = page_bytes(BATCHED, swa_heads, swa_hd, dtype, block_size=block_size)
    glob_b = page_bytes(BATCHED, glob_heads, glob_hd, dtype, block_size=block_size)
    pool_b = max(swa_b, glob_b)
    swa_split, glob_split = pool_b // swa_b, pool_b // glob_b
    swa_native = BATCHED.get_kv_cache_shape(
        num_blocks * swa_split, block_size, swa_heads, swa_hd, dtype
    )
    glob_native = BATCHED.get_kv_cache_shape(
        num_blocks * glob_split, block_size, glob_heads, glob_hd, dtype
    )
    pool = materialize_shared_attention_cache(
        layer_specs=[
            (
                "swa",
                SlidingWindowSpec(
                    block_size=block_size,
                    num_kv_heads=swa_heads,
                    head_size=swa_hd,
                    dtype=dtype,
                    sliding_window=sliding_window,
                    page_size_padded=pool_b,
                ),
            ),
            (
                "glob",
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=glob_heads,
                    head_size=glob_hd,
                    dtype=dtype,
                    page_size_padded=pool_b,
                ),
            ),
        ],
        tensor_size=num_blocks * pool_b,
        num_blocks=num_blocks,
        kernel_block_size=block_size,
        attn_backend=BATCHED,
        device=torch.device("cpu"),
    )
    return pool, swa_native, glob_native, swa_split, glob_split, pool_b


@pytest.mark.parametrize("block_size", [128, 256])
@pytest.mark.parametrize(
    "swa_heads,swa_hd,glob_heads,glob_hd",
    [(2, 256, 1, 512), (2, 80, 1, 192), (2, 128, 2, 256)],
)
def test_shared_pool_seq_along_lane_traces_kernel_without_transpose(
    monkeypatch, block_size, swa_heads, swa_hd, glob_heads, glob_hd
):
    """Mixed-head-dim shared pools under SEQ_ALONG_LANE keep
    `(packing, block_size)` trailing so `_pallas_rpa_kernel_batched` reshapes
    without emitting a 5D `transpose`, including unequal (2x) native pages."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from vllm.config import get_current_vllm_config_or_none

    from vllm_torchtpu.layers.adapter import attention as adapter

    num_blocks = 8
    with _layout_env(monkeypatch, "HND"):
        pool, swa_native, glob_native, swa_split, glob_split, pool_b = (
            _build_mixed_shared_pool(
                num_blocks=num_blocks,
                block_size=block_size,
                swa_heads=swa_heads,
                swa_hd=swa_hd,
                glob_heads=glob_heads,
                glob_hd=glob_hd,
                dtype=FP8,
                sliding_window=1024,
            )
        )
        kv_layout = KV_LAYOUT_BY_VLLM_LAYOUT[
            get_current_vllm_config().cache_config.get_resolved_kv_cache_layout()
        ]
    assert get_current_vllm_config_or_none() is None
    assert pool.shape == (num_blocks, pool_b // (4 * block_size), 4, block_size)

    mesh = jax.sharding.Mesh(np.array(jax.devices("cpu")[:1]), ("model",))
    seen: list[tuple[int, ...]] = []

    def stub_attn(kv_cache, query, key, value, metadata, mesh, **kwargs):
        seen.append(kv_cache.shape)
        assert kwargs["kv_layout"] == kv_layout
        return kv_cache, query + metadata.block_tables.sum().astype(query.dtype)

    monkeypatch.setattr(adapter, "attention", stub_attn)
    for kv_h, hd, expected_native, split in (
        (swa_heads, swa_hd, swa_native, swa_split),
        (glob_heads, glob_hd, glob_native, glob_split),
    ):
        seen.clear()
        run = functools.partial(
            adapter._pallas_rpa_kernel_batched,
            sinks=None,
            q_scale=None,
            k_scale=1.0,
            v_scale=1.0,
            mesh=mesh,
            sliding_window=None,
            skip_kv_update=False,
            kv_layout=kv_layout,
        )
        jaxpr = jax.make_jaxpr(run)(
            jax.ShapeDtypeStruct(pool.shape, jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((1, kv_h * 2, hd), jnp.bfloat16),
            jax.ShapeDtypeStruct((1, kv_h, hd), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((1, kv_h, hd), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((1,), jnp.int32),
            jax.ShapeDtypeStruct((2,), jnp.int32),
            jax.ShapeDtypeStruct((2,), jnp.int32),
            jax.ShapeDtypeStruct((3,), jnp.int32),
        )
        assert seen == [expected_native]
        primitives = [eqn.primitive.name for eqn in jaxpr.jaxpr.eqns]
        assert "transpose" not in primitives, primitives
        assert ("mul" in primitives) == (split > 1), primitives


def test_get_shared_pool_shape_and_seq_along_lane_predicate(monkeypatch):
    assert uses_seq_along_lane_kv_page(KVLayout.SEQ_ALONG_LANE, 128)
    assert not uses_seq_along_lane_kv_page(KVLayout.SEQ_ALONG_LANE, 64)
    assert not uses_seq_along_lane_kv_page(KVLayout.HEAD_ALONG_SUBLANE, 128)
    assert not uses_seq_along_lane_kv_page(None, 128)
    assert BATCHED.get_shared_pool_shape(8, 128, 4, 256, [64]) == (8, 128, 4, 256)
    assert BATCHED.get_shared_pool_shape(8, 128, 4, 256, []) == (8, 128, 4, 256)

    with _layout_env(monkeypatch, "HND"):
        assert BATCHED.get_shared_pool_shape(8, 128, 4, 256, [256, 512]) == (
            8,
            256,
            4,
            128,
        )
        assert BASE.get_shared_pool_shape(8, 128, 4, 256, [256, 512]) == (
            8,
            128,
            4,
            256,
        )
        with pytest.raises(NotImplementedError, match="head_dim==64"):
            BATCHED.get_shared_pool_shape(8, 128, 4, 256, [64, 128])

    with _layout_env(monkeypatch, "NHD"):
        assert BATCHED.get_shared_pool_shape(8, 128, 4, 256, [64, 128]) == (
            8,
            128,
            4,
            256,
        )


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    "swa_heads,swa_hd,glob_heads,glob_hd",
    [(2, 256, 1, 512), (2, 80, 1, 160), (2, 128, 2, 256)],
)
def test_shared_pool_seq_along_lane_matches_separate_caches_on_tpu(
    hnd, swa_heads, swa_hd, glob_heads, glob_hd
):
    """Run `_pallas_rpa_kernel_batched` on TPU across equal and unequal (2x)
    4D `SEQ_ALONG_LANE` shared pools and verify bit-exact output parity."""
    _require_tpu()
    import jax
    import jax.numpy as jnp
    import numpy as np

    from vllm_torchtpu.layers.adapter import attention as adapter

    bs, nblk = 128, 8
    pool_t, swa_native, glob_native, swa_s, glob_s, _ = _build_mixed_shared_pool(
        num_blocks=nblk,
        block_size=bs,
        swa_heads=swa_heads,
        swa_hd=swa_hd,
        glob_heads=glob_heads,
        glob_hd=glob_hd,
        dtype=BF16,
        sliding_window=192,
    )
    shared = jnp.zeros(tuple(pool_t.shape), dtype=jnp.bfloat16)
    sep_swa = jnp.zeros(swa_native, dtype=jnp.bfloat16)
    sep_glob = jnp.zeros(glob_native, dtype=jnp.bfloat16)
    mesh = jax.sharding.Mesh(np.asarray(jax.devices("tpu")[:1]), ("model",))
    rng = np.random.default_rng(0)
    pre_len, tot_len = bs + 16, bs + 17

    def mk(h: int, d: int):
        return tuple(
            jnp.asarray(rng.standard_normal(s) * 0.5, jnp.bfloat16)
            for s in ((tot_len, h * 2, d), (tot_len, h, d), (tot_len, h, d))
        )

    swa_qkv, glob_qkv = mk(swa_heads, swa_hd), mk(glob_heads, glob_hd)
    swa_bt, glob_bt = jnp.asarray([1, 4], jnp.int32), jnp.asarray([2, 5], jnp.int32)

    def step(cache, qkv, bt, sl: slice, seq_l: int, dec: bool, sw):
        q, k, v = (a[sl] for a in qkv)
        dist = jnp.asarray((1, 1, 1) if dec else (0, 0, 1), jnp.int32)
        return adapter._pallas_rpa_kernel_batched(
            cache,
            q,
            k,
            v,
            jnp.asarray([seq_l], jnp.int32),
            bt,
            jnp.asarray([0, q.shape[0]], jnp.int32),
            dist,
            None,
            None,
            None,
            None,
            mesh=mesh,
            sliding_window=sw,
            sm_scale=q.shape[-1] ** -0.5,
            skip_kv_update=False,
            kv_layout=KVLayout.SEQ_ALONG_LANE,
        )

    shared, o1 = step(shared, swa_qkv, swa_bt, slice(0, pre_len), pre_len, False, 192)
    shared, o2 = step(
        shared, glob_qkv, glob_bt, slice(0, pre_len), pre_len, False, None
    )
    shared, o3 = step(
        shared, swa_qkv, swa_bt, slice(pre_len, tot_len), tot_len, True, 192
    )
    shared, o4 = step(
        shared, glob_qkv, glob_bt, slice(pre_len, tot_len), tot_len, True, None
    )
    sep_swa, r1 = step(
        sep_swa, swa_qkv, swa_bt * swa_s, slice(0, pre_len), pre_len, False, 192
    )
    sep_swa, r3 = step(
        sep_swa, swa_qkv, swa_bt * swa_s, slice(pre_len, tot_len), tot_len, True, 192
    )
    sep_glob, r2 = step(
        sep_glob, glob_qkv, glob_bt * glob_s, slice(0, pre_len), pre_len, False, None
    )
    sep_glob, r4 = step(
        sep_glob,
        glob_qkv,
        glob_bt * glob_s,
        slice(pre_len, tot_len),
        tot_len,
        True,
        None,
    )
    for got, ref in ((o1, r1), (o2, r2), (o3, r3), (o4, r4)):
        np.testing.assert_array_equal(
            np.asarray(got.astype(jnp.float32)), np.asarray(ref.astype(jnp.float32))
        )
