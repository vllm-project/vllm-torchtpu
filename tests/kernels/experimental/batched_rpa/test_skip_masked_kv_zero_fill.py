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
"""`skip_masked_kv_zero_fill`: on SEQ_ALONG_LANE pages the decode kernel need not
zero its KV VMEM window, because unwritten V rows are masked before PV and keys
past a decode token's sequence or window are masked in QK. The mixed kernel
keeps the fill.

The refusal and the tpu7x lowering run on CPU; the bitwise check needs a TPU.
It first runs a kernel that fills most of VMEM with NaN, which the RPA kernels
would otherwise only see from whatever ran before them. Nothing places that
scratch where their KV windows land, so a pass is evidence, not proof.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, wrapper

_HND = configs.KVLayout.SEQ_ALONG_LANE
_FP8 = jnp.float8_e4m3fn
_PAGE, _PAGES_PER_SEQ, _MAX_SEQS = 128, 4, 8


def _serve(**kwargs):
    return configs.ServingConfigs(
        num_seqs=8,
        page_size=256,
        total_q_tokens=16,
        num_page_indices=8 * 64,
        dtype_q=jnp.bfloat16,
        dtype_kv=_FP8,
        dtype_out=jnp.bfloat16,
        skip_masked_kv_zero_fill=True,
        **kwargs,
    )


def test_needs_masked_v():
    assert _serve(kv_layout=_HND).mask_invalid_v
    with pytest.raises(ValueError, match="needs mask_invalid_v"):
        _serve(kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE)
    with pytest.raises(ValueError, match="needs mask_invalid_v"):
        _serve(kv_layout=_HND, mask_invalid_v=False)


def _args(tokens, q_heads, kv_heads, head_dim, num_pages):
    kv_shape = wrapper.get_kv_cache_shape(
        num_pages, _PAGE, kv_heads, head_dim, _FP8, kv_layout=_HND
    )
    return [
        ((tokens, q_heads, head_dim), jnp.bfloat16),
        ((tokens, kv_heads, head_dim), _FP8),
        ((tokens, kv_heads, head_dim), _FP8),
        (kv_shape, _FP8),
        ((_MAX_SEQS,), jnp.int32),
        ((_MAX_SEQS * _PAGES_PER_SEQ,), jnp.int32),
        ((_MAX_SEQS + 1,), jnp.int32),
        ((3,), jnp.int32),
    ]


def _attend(*a, sliding_window=None, skip=True):
    return wrapper.ragged_paged_attention(
        *a,
        sm_scale=a[0].shape[-1] ** -0.5,
        sliding_window=sliding_window,
        k_scale=1.0,
        v_scale=1.0,
        kv_layout=_HND,
        skip_empty_launch=True,
        skip_masked_kv_zero_fill=skip,
    )


@pytest.mark.parametrize("sliding_window", [None, 64])
def test_kernels_lower_for_tpu7x(sliding_window):
    """Decode and mixed kernels trace and pass Mosaic's verifier (no device)."""
    args = [jax.ShapeDtypeStruct(s, d) for s, d in _args(128, 8, 1, 512, 32)]
    mesh = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    with use_abstract_mesh(mesh):
        text = (
            jax.jit(lambda *a: _attend(*a, sliding_window=sliding_window))
            .trace(*args)
            .lower(lowering_platforms=("tpu",))
            .as_text()
        )
    assert "RPAd" in text and "RPAm" in text


def _fill_vmem_with_nan(x):
    """A kernel that leaves most of VMEM holding NaN, then returns x."""

    def body(x_ref, o_ref, scratch):
        scratch[...] = jnp.full(scratch.shape, jnp.nan, scratch.dtype)
        o_ref[...] = x_ref[...]

    return pl.pallas_call(
        body,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        scratch_shapes=[pltpu.VMEM((48 * 1024 * 1024 // 4 // 128, 128), jnp.float32)],
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=60 * 1024 * 1024),
    )(x)


@pytest.mark.tpu_test
@pytest.mark.parametrize("sliding_window", [None, 64])
@pytest.mark.parametrize(
    "kv_lens, q_lens",
    [
        ((100,), (1,)),  # one decode row, partial last page
        ((70, 241, 33), (1, 1, 1)),  # decode rows, most lanes left empty
        ((70, 241, 90), (1, 1, 40)),  # decode rows and a prefill chunk
        ((70, 90, 130), (1, 40, 30)),  # chunks with padded rows in their last q block
    ],
)
def test_outputs_bitwise_equal_on_nan_vmem(sliding_window, kv_lens, q_lens):
    if jax.default_backend() != "tpu":
        pytest.skip(f"Requires TPU backend, found {jax.default_backend()!r}.")
    rng = np.random.default_rng(0)
    tokens, num_pages = 64, _MAX_SEQS * _PAGES_PER_SEQ
    shapes = _args(tokens, 8, 1, 512, num_pages)
    q, k, v, cache = (
        jnp.asarray(rng.standard_normal(s).astype(np.float32), d) for s, d in shapes[:4]
    )
    n = len(kv_lens)
    lens = np.zeros(_MAX_SEQS, np.int32)
    lens[:n] = kv_lens
    cu = np.full(_MAX_SEQS + 1, sum(q_lens), np.int32)
    cu[: n + 1] = np.concatenate([[0], np.cumsum(q_lens)])
    decode = sum(1 for x in q_lens if x == 1)
    pages = rng.permutation(num_pages).astype(np.int32)
    dist = np.array([decode, decode, n], np.int32)
    host = [jnp.asarray(x) for x in (lens, pages, cu, dist)]

    def run(skip):
        if skip:  # leave NaN wherever the kernels' VMEM windows will be
            jax.block_until_ready(_fill_vmem_with_nan(jnp.zeros((8, 128))))
        # The cache is donated, so each run gets a fresh copy.
        out, new_cache = jax.jit(
            lambda *a: _attend(*a, sliding_window=sliding_window, skip=skip)
        )(q, k, v, jnp.copy(cache), *host)
        out = out[: sum(q_lens)]
        return out, np.asarray(new_cache.view(jnp.uint8))

    ref_out, ref_cache = run(False)
    assert np.isfinite(np.asarray(ref_out, np.float32)).all()
    for _ in range(2):
        out, new_cache = run(True)
        np.testing.assert_array_equal(
            np.asarray(out.view(jnp.uint16)), np.asarray(ref_out.view(jnp.uint16))
        )
        np.testing.assert_array_equal(new_cache, ref_cache)
