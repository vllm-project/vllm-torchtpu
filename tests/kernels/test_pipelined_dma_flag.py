# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

import inspect
import os
import sys
from unittest.mock import MagicMock, patch

import jax
import jax.extend.core
import jax.numpy as jnp
import numpy as np
import pytest

# Provide lightweight stubs for optional heavy runtime dependencies if not present
for mod_name in ("torch", "vllm", "vllm.utils"):
    if mod_name not in sys.modules:
        try:
            __import__(mod_name)
        except ImportError:
            sys.modules[mod_name] = MagicMock()

from vllm_torchtpu import envs  # noqa: E402
from vllm_torchtpu.kernels.ragged_paged_attention.v2 import (  # noqa: E402
    ragged_kv_cache_update,
)
from vllm_torchtpu.kernels.ragged_paged_attention.v3 import kernel  # noqa: E402


def test_env_var_default():
    """Verify that both flags default to False when unset."""
    with patch.dict(os.environ, {}, clear=True):
        assert envs.USE_RPA_PIPELINED_DMA_STAGING is False
        assert envs.USE_PIPELINED_DMA_STAGING is False


def test_env_var_toggle():
    """Verify that setting either flag toggles both flags."""
    # Test setting USE_RPA_PIPELINED_DMA_STAGING="1"
    with patch.dict(os.environ, {"USE_RPA_PIPELINED_DMA_STAGING": "1"}, clear=True):
        assert envs.USE_RPA_PIPELINED_DMA_STAGING is True
        assert envs.USE_PIPELINED_DMA_STAGING is True

    # Test setting USE_PIPELINED_DMA_STAGING="1"
    with patch.dict(os.environ, {"USE_PIPELINED_DMA_STAGING": "1"}, clear=True):
        assert envs.USE_RPA_PIPELINED_DMA_STAGING is True
        assert envs.USE_PIPELINED_DMA_STAGING is True

    # Test boolean string true
    with patch.dict(os.environ, {"USE_RPA_PIPELINED_DMA_STAGING": "true"}, clear=True):
        assert envs.USE_RPA_PIPELINED_DMA_STAGING is True
        assert envs.USE_PIPELINED_DMA_STAGING is True

    # Test explicit disable "0"
    with patch.dict(os.environ, {"USE_RPA_PIPELINED_DMA_STAGING": "0"}, clear=True):
        assert envs.USE_RPA_PIPELINED_DMA_STAGING is False
        assert envs.USE_PIPELINED_DMA_STAGING is False


def test_ragged_kv_cache_update_signature_and_wiring():
    """Verify ragged_kv_cache_update parameter defaults and env resolution."""
    sig = inspect.signature(ragged_kv_cache_update.kv_cache_update)
    assert "pipelined_dma" in sig.parameters
    assert sig.parameters["pipelined_dma"].default is None

    mock_kv = MagicMock()
    mock_new_kv = MagicMock()
    mock_new_kv.shape = (10, 2, 128)
    unwrapped_update = inspect.unwrap(ragged_kv_cache_update.kv_cache_update)

    # Verify when env var is False
    with (
        patch.dict(os.environ, {}, clear=True),
        patch.object(ragged_kv_cache_update, "_kv_cache_update") as mock_update,
    ):
        unwrapped_update(
            mock_new_kv, MagicMock(), mock_kv, MagicMock(), num_slices_per_block=4
        )
        mock_update.assert_called_once()
        _, kwargs = mock_update.call_args
        assert kwargs["pipelined_dma"] is False

    # Verify when env var is True
    with (
        patch.dict(os.environ, {"USE_RPA_PIPELINED_DMA_STAGING": "1"}, clear=True),
        patch.object(ragged_kv_cache_update, "_kv_cache_update") as mock_update,
    ):
        unwrapped_update(
            mock_new_kv, MagicMock(), mock_kv, MagicMock(), num_slices_per_block=4
        )
        mock_update.assert_called_once()
        _, kwargs = mock_update.call_args
        assert kwargs["pipelined_dma"] is True

    # Verify explicit override pipelined_dma=False when env var is True
    with (
        patch.dict(os.environ, {"USE_RPA_PIPELINED_DMA_STAGING": "1"}, clear=True),
        patch.object(ragged_kv_cache_update, "_kv_cache_update") as mock_update,
    ):
        unwrapped_update(
            mock_new_kv,
            MagicMock(),
            mock_kv,
            MagicMock(),
            num_slices_per_block=4,
            pipelined_dma=False,
        )
        mock_update.assert_called_once()
        _, kwargs = mock_update.call_args
        assert kwargs["pipelined_dma"] is False


def test_kv_cache_update_builds_block_specs_and_traces():
    """Verify _kv_cache_update builds pl.ANY BlockSpecs and traces cleanly."""
    import functools

    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl

    new_kv = jnp.zeros((16, 2, 128), dtype=jnp.bfloat16)
    kv_cache = jnp.zeros((64, 2, 128), dtype=jnp.bfloat16)
    slices = jnp.zeros((3, 4), dtype=jnp.int32)
    num_slices = jnp.array([2], dtype=jnp.int32)

    for pipelined_dma in (False, True):
        with patch.object(
            ragged_kv_cache_update.pl,
            "pallas_call",
            wraps=ragged_kv_cache_update.pl.pallas_call,
        ) as spy_pallas_call:
            out_shape = jax.eval_shape(
                functools.partial(
                    ragged_kv_cache_update.kv_cache_update,
                    num_slices_per_block=4,
                    pipelined_dma=pipelined_dma,
                ),
                new_kv,
                slices,
                kv_cache,
                num_slices,
            )
            assert out_shape.shape == kv_cache.shape
            assert out_shape.dtype == kv_cache.dtype
            spy_pallas_call.assert_called_once()
            _, call_kwargs = spy_pallas_call.call_args
            grid_spec = call_kwargs["grid_spec"]
            assert [spec.memory_space for spec in grid_spec.in_specs] == [
                pl.ANY,
                pl.ANY,
            ]
            assert [spec.memory_space for spec in grid_spec.out_specs] == [pl.ANY]


def test_kernel_v3_prepare_inputs_outputs_wiring():
    """Verify kernel.py prepare_inputs/prepare_outputs wiring to env flag."""
    sig_in = inspect.signature(kernel.prepare_inputs)
    assert "fuse_non_tiling_axis_swap" in sig_in.parameters
    assert sig_in.parameters["fuse_non_tiling_axis_swap"].default is None

    sig_out = inspect.signature(kernel.prepare_outputs)
    assert "fuse_non_tiling_axis_swap" in sig_out.parameters
    assert sig_out.parameters["fuse_non_tiling_axis_swap"].default is None


def test_kernel_v3_prepare_inputs_outputs_execution():
    """Verify prepare_inputs and prepare_outputs shape behavior under env toggle."""
    import jax.numpy as jnp

    max_tokens = 16
    q_heads = 8
    kv_heads = 2
    head_dim = 128
    q = jnp.zeros((max_tokens, q_heads, head_dim))
    k = jnp.zeros((max_tokens, kv_heads, head_dim))
    v = jnp.zeros((max_tokens, kv_heads, head_dim))

    # Default / False: returns [kv_heads, max_tokens, q_heads_per_kv_head, 1, head_dim]
    with patch.dict(os.environ, {}, clear=True):
        q_out, _ = kernel.prepare_inputs(q, k, v)
        assert q_out.shape == (2, 16, 4, 1, 128)
        out = kernel.prepare_outputs(q_out, 4, 128)
        assert out.shape == (16, 8, 128)

    # Flag enabled: returns [max_tokens, kv_heads, q_heads_per_kv_head, 1, head_dim]
    with patch.dict(os.environ, {"USE_RPA_PIPELINED_DMA_STAGING": "1"}, clear=True):
        q_out, _ = kernel.prepare_inputs(q, k, v)
        assert q_out.shape == (16, 2, 4, 1, 128)
        out = kernel.prepare_outputs(q_out, 4, 128)
        assert out.shape == (16, 8, 128)

    # Explicit override False when flag is True
    with patch.dict(os.environ, {"USE_RPA_PIPELINED_DMA_STAGING": "1"}, clear=True):
        q_out, _ = kernel.prepare_inputs(q, k, v, fuse_non_tiling_axis_swap=False)
        assert q_out.shape == (2, 16, 4, 1, 128)
        out = kernel.prepare_outputs(q_out, 4, 128, fuse_non_tiling_axis_swap=False)
        assert out.shape == (16, 8, 128)


@pytest.mark.parametrize("fuse", [None, True, False])
@pytest.mark.parametrize("env", ["0", "1"])
@pytest.mark.parametrize(
    "public, jitted",
    [
        ("ragged_paged_attention", "_ragged_paged_attention"),
        ("ragged_paged_attention_bundled", "_ragged_paged_attention_bundled"),
    ],
)
def test_kernel_v3_entry_points_resolve_flag_before_jit(public, jitted, env, fuse):
    """None reads USE_RPA_PIPELINED_DMA_STAGING outside the jit (R-4.3); an
    explicit value wins over the env."""
    with (
        patch.dict(os.environ, {"USE_RPA_PIPELINED_DMA_STAGING": env}, clear=True),
        patch.object(kernel, jitted) as mock_jitted,
    ):
        getattr(kernel, public)("q", sm_scale=0.5, fuse_non_tiling_axis_swap=fuse)
    expected = env == "1" if fuse is None else fuse
    mock_jitted.assert_called_once_with(
        "q", sm_scale=0.5, fuse_non_tiling_axis_swap=expected
    )


def _primitives_outside_kernels(jaxpr, found):
    for eqn in jaxpr.eqns:
        found.append(eqn.primitive.name)
        if eqn.primitive.name == "pallas_call":
            continue
        for sub in jax.extend.core.jaxprs_in_params(eqn.params):
            _primitives_outside_kernels(sub, found)
    return found


def _count_transposes_outside_kernels(fn, args):
    jaxpr = jax.make_jaxpr(fn)(*args)
    found = _primitives_outside_kernels(jaxpr.jaxpr, [])
    assert found.count("pallas_call") == 3  # decode, prefill, mixed
    assert jaxpr.out_avals[0].shape == args[0].shape
    return found.count("transpose")


def _zero_rpa_args(
    num_q_heads,
    num_kv_heads,
    *,
    q_dtype=jnp.bfloat16,
    kv_dtype=jnp.bfloat16,
    head_dim=128,
    page_size=16,
    num_layers=None,
):
    max_num_tokens, max_num_seqs, pages_per_seq, total_num_pages = 64, 4, 4, 32
    cache_shape = kernel.get_kv_cache_shape(
        total_num_pages, page_size, num_kv_heads, head_dim, kv_dtype
    )
    if num_layers is not None:
        cache_shape = (cache_shape[0], num_layers, *cache_shape[1:])
    return (
        jnp.zeros((max_num_tokens, num_q_heads, head_dim), q_dtype),
        jnp.zeros((max_num_tokens, num_kv_heads, head_dim), kv_dtype),
        jnp.zeros((max_num_tokens, num_kv_heads, head_dim), kv_dtype),
        jnp.zeros(cache_shape, kv_dtype),
        jnp.zeros((max_num_seqs,), jnp.int32),
        jnp.zeros((max_num_seqs * pages_per_seq,), jnp.int32),
        jnp.zeros((max_num_seqs + 1,), jnp.int32),
        jnp.zeros((3,), jnp.int32),
    )


_TRACE_KWARGS = dict(chunk_prefill_size=16, vmem_limit_bytes=64 * 1024 * 1024)


@pytest.mark.parametrize("tpu_version", [5, 6, 7])
@pytest.mark.parametrize("kv_dtype", [jnp.bfloat16, jnp.float8_e4m3fn, jnp.float8_e5m2])
@pytest.mark.parametrize(
    "num_q_heads, num_kv_heads", [(8, 2), (4, 4), (16, 1), (6, 2), (32, 8)]
)
def test_fused_axis_swap_traces_no_transposes(
    monkeypatch, num_q_heads, num_kv_heads, kv_dtype, tpu_version
):
    """With the swap fused into the kernel DMAs, XLA no longer transposes the
    queries before RPA v3 or the outputs after it. Traced on CPU, so every
    decode, prefill and mixed kernel body is built, DMA descriptors included."""
    monkeypatch.setattr(kernel, "get_tpu_version", lambda: tpu_version)
    args = _zero_rpa_args(num_q_heads, num_kv_heads, kv_dtype=kv_dtype)

    def transposes(fuse):
        return _count_transposes_outside_kernels(
            lambda *a: kernel.ragged_paged_attention(
                *a, fuse_non_tiling_axis_swap=fuse, **_TRACE_KWARGS
            ),
            args,
        )

    assert transposes(fuse=False) == 2
    assert transposes(fuse=True) == 0


@pytest.mark.parametrize(
    "q_dtype, kv_dtype, head_dim, page_size",
    [
        (jnp.float32, jnp.float32, 128, 16),
        (jnp.bfloat16, jnp.bfloat16, 64, 16),
        (jnp.bfloat16, jnp.bfloat16, 96, 32),
        (jnp.bfloat16, jnp.bfloat16, 256, 64),
        (jnp.bfloat16, jnp.float8_e4m3fn, 192, 128),
    ],
)
def test_fused_axis_swap_traces_no_transposes_across_layouts(
    monkeypatch, q_dtype, kv_dtype, head_dim, page_size
):
    """Same as above across query packing, padded head dims and page sizes."""
    monkeypatch.setattr(kernel, "get_tpu_version", lambda: 7)
    args = _zero_rpa_args(
        8, 2, q_dtype=q_dtype, kv_dtype=kv_dtype, head_dim=head_dim, page_size=page_size
    )
    for fuse, expected in ((False, 2), (True, 0)):
        assert (
            _count_transposes_outside_kernels(
                lambda *a, fuse=fuse: kernel.ragged_paged_attention(
                    *a, fuse_non_tiling_axis_swap=fuse, **_TRACE_KWARGS
                ),
                args,
            )
            == expected
        )


def test_bundled_fused_axis_swap_traces_no_transposes(monkeypatch):
    monkeypatch.setattr(kernel, "get_tpu_version", lambda: 7)
    args = _zero_rpa_args(8, 2, num_layers=3)

    def transposes(fuse):
        return _count_transposes_outside_kernels(
            lambda q, k, v, bundle, *md: kernel.ragged_paged_attention_bundled(
                q,
                k,
                v,
                bundle,
                jnp.int32(1),
                *md,
                fuse_non_tiling_axis_swap=fuse,
                **_TRACE_KWARGS,
            ),
            args,
        )

    assert transposes(fuse=False) == 2
    assert transposes(fuse=True) == 0


# (q_lens, kv_lens, distribution, chunk_prefill_size) per batch kind. Uneven
# lengths leave partial query blocks and blocks that start mid-tile.
_BATCHES = {
    "decode": ([1] * 8, [5, 17, 33, 64, 1, 90, 128, 47], (8, 8, 8), None),
    "prefill": ([37, 150, 5, 64], [37, 150, 5, 64], (0, 0, 4), None),
    "long_prefill": ([1000], [1000], (0, 0, 1), None),
    "chunked": ([32, 32, 32], [32, 75, 160], (0, 3, 3), 32),
    "mixed": (
        [1, 1, 1, 32, 32, 20, 100],
        [40, 9, 128, 64, 96, 20, 150],
        (3, 5, 7),
        32,
    ),
    # One-token sequences scheduled as mixed: multi-token blocks holding one row.
    "mixed_single_tokens": ([1, 1, 1], [1, 30, 77], (0, 0, 3), None),
}
_HEADS = [(8, 2), (4, 4), (16, 1), (6, 2), (32, 8)]
_KV_DTYPES = [jnp.bfloat16, jnp.float8_e4m3fn, jnp.float8_e5m2]


def _random_batch_spec(seed):
    rng = np.random.default_rng(seed)
    chunk = int(rng.choice([16, 32, 64]))
    num_decode, num_prefill, num_mixed = (
        int(rng.integers(0, 6)),
        int(rng.integers(0, 3)),
        int(rng.integers(1, 5)),
    )
    q_lens = (
        [1] * num_decode
        + [chunk] * num_prefill
        + [int(x) for x in rng.integers(1, 200, num_mixed)]
    )
    kv_lens = [q + int(rng.integers(0, 300)) for q in q_lens]
    distribution = (
        num_decode,
        num_decode + num_prefill,
        num_decode + num_prefill + num_mixed,
    )
    return q_lens, kv_lens, distribution, chunk


def _rpa_batch(
    spec,
    num_q_heads,
    num_kv_heads,
    *,
    q_dtype=jnp.bfloat16,
    kv_dtype=jnp.bfloat16,
    head_dim=128,
    page_size=16,
    seed=0,
):
    q_lens, kv_lens, distribution, chunk_prefill_size = spec
    num_seqs = len(q_lens)
    max_num_seqs = num_seqs + 2
    num_tokens = sum(q_lens)
    # Always leave padding rows past the last real token.
    max_num_tokens = 64 * (num_tokens // 64 + 1)
    pages_per_seq = -(-max(kv_lens) // page_size)
    total_num_pages = max_num_seqs * pages_per_seq + 8
    rng = np.random.default_rng(seed)

    def normal(*shape):
        return rng.standard_normal(shape, dtype=np.float32)

    cu_q_lens = np.zeros(max_num_seqs + 1, np.int32)
    cu_q_lens[1 : num_seqs + 1] = np.cumsum(q_lens)
    cu_q_lens[num_seqs + 1 :] = cu_q_lens[num_seqs]
    return dict(
        q=normal(max_num_tokens, num_q_heads, head_dim),
        k=normal(max_num_tokens, num_kv_heads, head_dim),
        v=normal(max_num_tokens, num_kv_heads, head_dim),
        kv_cache=normal(
            *kernel.get_kv_cache_shape(
                total_num_pages, page_size, num_kv_heads, head_dim, kv_dtype
            )
        ),
        kv_lens=np.pad(np.asarray(kv_lens, np.int32), (0, max_num_seqs - num_seqs)),
        page_indices=rng.permutation(total_num_pages)[
            : max_num_seqs * pages_per_seq
        ].astype(np.int32),
        cu_q_lens=cu_q_lens,
        distribution=np.asarray(distribution, np.int32),
        chunk_prefill_size=chunk_prefill_size,
        num_tokens=num_tokens,
        head_dim=head_dim,
        q_dtype=q_dtype,
        kv_dtype=kv_dtype,
    )


def _device_inputs(b):
    # Fresh device arrays per call: the kernel donates them.
    return (
        jnp.asarray(b["q"], b["q_dtype"]),
        jnp.asarray(b["k"], b["kv_dtype"]),
        jnp.asarray(b["v"], b["kv_dtype"]),
        jnp.asarray(b["kv_cache"], b["kv_dtype"]),
        jnp.asarray(b["kv_lens"]),
        jnp.asarray(b["page_indices"]),
        jnp.asarray(b["cu_q_lens"]),
        jnp.asarray(b["distribution"]),
    )


def _as_f32(x):
    return np.asarray(x.astype(jnp.float32))


def _require_tpu():
    if jax.default_backend() != "tpu":
        pytest.fail(f"Expected JAX TPU backend, got {jax.default_backend()}.")


_REF_KWARGS = (
    "sm_scale",
    "use_causal_mask",
    "skip_kv_update",
    "sliding_window",
    "soft_cap",
    "out_dtype",
    "q_scale",
    "k_scale",
    "v_scale",
)


def _assert_fused_matches_unfused_and_reference(b, **kwargs):
    """Fusing the query/output axis swap into the RPA v3 DMAs changes only how
    blocks move between HBM and VMEM, so outputs and caches stay bit-identical
    to the unfused path, and both match the reference."""
    _require_tpu()
    kwargs.setdefault("sm_scale", b["head_dim"] ** -0.5)
    kwargs.setdefault("chunk_prefill_size", b["chunk_prefill_size"])
    if b["kv_dtype"] not in (jnp.bfloat16, jnp.float32):
        kwargs.setdefault("k_scale", 0.5)
        kwargs.setdefault("v_scale", 0.5)
    n = b["num_tokens"]
    results = {}
    for fuse in (False, True):
        out, cache = kernel.ragged_paged_attention(
            *_device_inputs(b), fuse_non_tiling_axis_swap=fuse, **kwargs
        )
        results[fuse] = (_as_f32(out)[:n], _as_f32(cache))
    ref_out, ref_cache = kernel.ref_ragged_paged_attention(
        *_device_inputs(b), **{k: v for k, v in kwargs.items() if k in _REF_KWARGS}
    )

    np.testing.assert_array_equal(results[True][0], results[False][0])
    np.testing.assert_array_equal(results[True][1], results[False][1])
    for out, cache in results.values():
        np.testing.assert_allclose(out, _as_f32(ref_out), atol=0.2, rtol=0.2)
        np.testing.assert_array_equal(cache, _as_f32(ref_cache))


@pytest.mark.tpu_test
@pytest.mark.parametrize("kv_dtype", _KV_DTYPES)
@pytest.mark.parametrize("num_q_heads, num_kv_heads", _HEADS)
@pytest.mark.parametrize("batch", list(_BATCHES))
def test_fused_axis_swap_matches_unfused_and_reference(
    batch, num_q_heads, num_kv_heads, kv_dtype
):
    _assert_fused_matches_unfused_and_reference(
        _rpa_batch(_BATCHES[batch], num_q_heads, num_kv_heads, kv_dtype=kv_dtype)
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize("batch", ["prefill", "mixed"])
@pytest.mark.parametrize(
    "options",
    [
        dict(sliding_window=24),
        dict(soft_cap=10.0),
        dict(use_causal_mask=False),
        dict(sliding_window=48, soft_cap=30.0),
        dict(q_scale=0.5),
        dict(skip_kv_update=True),
    ],
    ids=lambda o: "-".join(f"{k}={v}" for k, v in o.items()),
)
def test_fused_axis_swap_attention_options(batch, options):
    kv_dtype = jnp.float8_e4m3fn if "q_scale" in options else jnp.bfloat16
    _assert_fused_matches_unfused_and_reference(
        _rpa_batch(_BATCHES[batch], 8, 2, kv_dtype=kv_dtype), **options
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    "block_sizes",
    [
        # (bq_sz, bkv_sz, bq_csz, bkv_csz); bq_sz == 1 takes the one-row copy
        # on prefill and mixed sequences too.
        (1, 64, 1, 32),
        (8, 64, 8, 32),
        (16, 128, 8, 64),
        (32, 256, 16, 128),
        (128, 128, 32, 64),
        (256, 512, 128, 256),
    ],
    ids=str,
)
@pytest.mark.parametrize("num_q_heads, num_kv_heads", [(8, 2), (16, 1)])
def test_fused_axis_swap_block_sizes(block_sizes, num_q_heads, num_kv_heads):
    _assert_fused_matches_unfused_and_reference(
        _rpa_batch(_BATCHES["mixed"], num_q_heads, num_kv_heads),
        p_block_sizes=block_sizes,
        m_block_sizes=block_sizes,
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize("d_block_sizes", [(1, 32, 1, 32), (1, 256, 1, 128)], ids=str)
def test_fused_axis_swap_decode_block_sizes(d_block_sizes):
    _assert_fused_matches_unfused_and_reference(
        _rpa_batch(_BATCHES["decode"], 8, 2), d_block_sizes=d_block_sizes
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize("page_size", [16, 32, 64, 128])
@pytest.mark.parametrize("batch", ["decode", "mixed"])
def test_fused_axis_swap_page_sizes(batch, page_size):
    _assert_fused_matches_unfused_and_reference(
        _rpa_batch(_BATCHES[batch], 8, 2, page_size=page_size)
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize("head_dim", [64, 96, 128, 192, 256])
def test_fused_axis_swap_head_dims(head_dim):
    _assert_fused_matches_unfused_and_reference(
        _rpa_batch(_BATCHES["mixed"], 8, 2, head_dim=head_dim)
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize("batch", list(_BATCHES))
def test_fused_axis_swap_float32(batch):
    """q_packing == 1: one query head per packed row."""
    _assert_fused_matches_unfused_and_reference(
        _rpa_batch(_BATCHES[batch], 8, 2, q_dtype=jnp.float32, kv_dtype=jnp.float32)
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize("seed", range(32))
def test_fused_axis_swap_random_batches(seed):
    """Random decode/prefill/mixed splits, lengths, heads and KV dtypes."""
    rng = np.random.default_rng(1000 + seed)
    num_q_heads, num_kv_heads = _HEADS[int(rng.integers(len(_HEADS)))]
    kv_dtype = _KV_DTYPES[int(rng.integers(len(_KV_DTYPES)))]
    _assert_fused_matches_unfused_and_reference(
        _rpa_batch(
            _random_batch_spec(seed),
            num_q_heads,
            num_kv_heads,
            kv_dtype=kv_dtype,
            seed=seed,
        )
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize("batch", list(_BATCHES))
def test_bundled_fused_axis_swap_matches_unfused(batch):
    """The bundled entry point forwards the flag: fused and unfused stay
    bit-identical, and only the target layer's slots change."""
    _require_tpu()
    num_layers, layer = 3, 1
    b = _rpa_batch(_BATCHES[batch], 8, 2)
    bundle = np.random.default_rng(1).standard_normal(
        (b["kv_cache"].shape[0], num_layers, *b["kv_cache"].shape[1:]),
        dtype=np.float32,
    )
    n = b["num_tokens"]
    results = {}
    for fuse in (False, True):
        q, k, v, _, *md = _device_inputs(b)
        out, new_bundle = kernel.ragged_paged_attention_bundled(
            q,
            k,
            v,
            jnp.asarray(bundle, jnp.bfloat16),
            jnp.int32(layer),
            *md,
            sm_scale=128**-0.5,
            chunk_prefill_size=b["chunk_prefill_size"],
            fuse_non_tiling_axis_swap=fuse,
        )
        results[fuse] = (_as_f32(out)[:n], _as_f32(new_bundle))
    np.testing.assert_array_equal(results[True][0], results[False][0])
    np.testing.assert_array_equal(results[True][1], results[False][1])
    untouched = [i for i in range(num_layers) if i != layer]
    np.testing.assert_array_equal(
        results[True][1][:, untouched],
        _as_f32(jnp.asarray(bundle[:, untouched], jnp.bfloat16)),
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize("batch", ["decode", "mixed"])
def test_env_flag_fuses_axis_swap(batch):
    """USE_RPA_PIPELINED_DMA_STAGING=1 with the default argument takes the
    fused path and matches the explicitly unfused result."""
    _require_tpu()
    b = _rpa_batch(_BATCHES[batch], 8, 2)
    kwargs = dict(sm_scale=128**-0.5, chunk_prefill_size=b["chunk_prefill_size"])
    n = b["num_tokens"]
    unfused = kernel.ragged_paged_attention(
        *_device_inputs(b), fuse_non_tiling_axis_swap=False, **kwargs
    )
    with (
        patch.dict(os.environ, {"USE_RPA_PIPELINED_DMA_STAGING": "1"}),
        patch.object(
            kernel, "_ragged_paged_attention", wraps=kernel._ragged_paged_attention
        ) as spy,
    ):
        fused = kernel.ragged_paged_attention(*_device_inputs(b), **kwargs)
    assert spy.call_args.kwargs["fuse_non_tiling_axis_swap"] is True
    np.testing.assert_array_equal(_as_f32(fused[0])[:n], _as_f32(unfused[0])[:n])
    np.testing.assert_array_equal(_as_f32(fused[1]), _as_f32(unfused[1]))
