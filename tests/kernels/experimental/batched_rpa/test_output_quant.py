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
"""quantize_head_major_output against quantize_tensor of the token-major rows.

The pass runs on CPU in Pallas interpret mode and must match bitwise; the
wrapper's `output_quant_dtype` is checked through its lowering for an abstract
TPU7x device, and on a TPU against the unquantized output.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental.batched_rpa import configs as KBC
from vllm_torchtpu.kernels.experimental.batched_rpa import output_quant
from vllm_torchtpu.kernels.experimental.batched_rpa import wrapper as KB
from vllm_torchtpu.kernels.quantized_matmul.util import quantize_tensor

_FP8 = jnp.float8_e4m3fn


def _head_major(kv_heads, tokens, groups, head_dim, seed=0):
    """A query-buffer-shaped output with per-token magnitudes and a zero row."""
    pairs = -(-groups // 2)
    k0, k1 = jax.random.split(jax.random.PRNGKey(seed))
    mag = jax.random.uniform(k0, (1, tokens, 1, 1, 1), minval=0.01, maxval=4.0)
    o = jax.random.normal(k1, (kv_heads, tokens, pairs, 2, head_dim)) * mag
    return o.astype(jnp.bfloat16).at[:, 1].set(0)


def _reference(o, groups, head_dim):
    kv_heads, tokens, pairs, packing, _ = o.shape
    rows = o.reshape(kv_heads, tokens, pairs * packing, -1)[:, :, :groups, :head_dim]
    return quantize_tensor(rows.swapaxes(0, 1).reshape(tokens, -1), _FP8)


@pytest.mark.parametrize(
    "kv_heads,tokens,groups,head_dim",
    [
        pytest.param(8, 512, 2, 256, id="sliding-31b-tp2"),
        pytest.param(2, 256, 8, 512, id="global-31b-tp2"),
        pytest.param(8, 48, 2, 256, id="odd-bucket"),
        pytest.param(2, 20, 8, 512, id="padded-tokens"),
        pytest.param(3, 32, 3, 128, id="padding-head"),
    ],
)
def test_matches_quantize_tensor(kv_heads, tokens, groups, head_dim):
    o = _head_major(kv_heads, tokens, groups, head_dim)
    q, scale = output_quant.quantize_head_major_output(
        o,
        num_q_heads_per_kv_head=groups,
        head_dim=head_dim,
        quant_dtype=_FP8,
        interpret=True,
    )
    ref_q, ref_scale = _reference(o, groups, head_dim)
    assert q.dtype == _FP8 and scale.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(scale), np.asarray(ref_scale))
    np.testing.assert_array_equal(
        np.asarray(q.view(jnp.uint8)), np.asarray(ref_q.view(jnp.uint8))
    )


def test_block_tokens():
    assert output_quant.block_tokens(16384, 16) == 256
    assert output_quant.block_tokens(16384, 32) == 128
    assert output_quant.block_tokens(48, 16) == 16
    assert output_quant.block_tokens(64, 4) == 64


def test_rejects_unaligned_heads():
    o = jnp.zeros((2, 16, 1, 2, 192), jnp.bfloat16)
    with pytest.raises(ValueError, match="lane-aligned"):
        output_quant.quantize_head_major_output(
            o, num_q_heads_per_kv_head=2, head_dim=192, quant_dtype=_FP8
        )


_DECODE_BLOCKS = KBC.BlockSizes(bq_sz=1, bq_c_sz=1, bkv_sz=256, batch_size=8)
_PREFILL_BLOCKS = KBC.BlockSizes(bq_sz=64, bq_c_sz=16, bkv_sz=256, batch_size=2)
_PAGE_SIZE, _PAGES_PER_SEQ, _MAX_SEQS, _TOKENS = 16, 8, 16, 16


def _args(num_q_heads, num_kv_heads, head_dim):
    cache_shape = KB.get_kv_cache_shape(
        _MAX_SEQS * _PAGES_PER_SEQ, _PAGE_SIZE, num_kv_heads, head_dim, _FP8
    )
    rng = np.random.default_rng(0)
    kv = rng.standard_normal((2, _TOKENS, num_kv_heads, head_dim))
    return (
        jnp.asarray(
            rng.standard_normal((_TOKENS, num_q_heads, head_dim)), jnp.bfloat16
        ),
        jnp.asarray(kv[0], _FP8),
        jnp.asarray(kv[1], _FP8),
        jnp.asarray(rng.standard_normal(cache_shape), _FP8),
        jnp.asarray((100, 40) + (0,) * (_MAX_SEQS - 2), jnp.int32),
        jnp.arange(_MAX_SEQS * _PAGES_PER_SEQ, dtype=jnp.int32),
        jnp.asarray((0, 1, 13) + (13,) * (_MAX_SEQS - 2), jnp.int32),
        jnp.asarray((1, 1, 2), jnp.int32),
    )


def _call(*args, fp8_compute=False, **kwargs):
    return KB.ragged_paged_attention(
        *args,
        sm_scale=args[0].shape[-1] ** -0.5,
        k_scale=1.0,
        v_scale=1.0,
        decode_block_sizes=_DECODE_BLOCKS,
        prefill_block_sizes=None if fp8_compute else _PREFILL_BLOCKS,
        fp8_compute=fp8_compute,
        **kwargs,
    )


_GEOMETRIES = [
    pytest.param((8, 2, 256), False, id="bf16-8q2kv"),
    pytest.param((8, 2, 512), True, id="fp8-8q2kv-hd512"),
]


def _abstract_tpu():
    device = jax.sharding.AbstractDevice(
        device_kind="TPU7x", num_cores=1, platform="tpu"
    )
    return jax.sharding.use_abstract_mesh(
        jax.sharding.AbstractMesh((), (), abstract_device=device)
    )


@pytest.mark.parametrize("heads,fp8_compute", _GEOMETRIES)
def test_wrapper_returns_quantized_rows(heads, fp8_compute):
    args = _args(*heads)
    with _abstract_tpu():
        traced = jax.jit(
            lambda *a: _call(*a, fp8_compute=fp8_compute, output_quant_dtype=_FP8)
        ).trace(*args)
        text = traced.lower(lowering_platforms=("tpu",)).as_text()
    (q, scale), cache = traced.out_info
    assert (q.shape, q.dtype) == ((_TOKENS, heads[0] * heads[2]), _FP8)
    assert (scale.shape, scale.dtype) == ((_TOKENS, 1), jnp.float32)
    assert cache.shape == args[3].shape
    assert "rpa_output_quantize" in text


def test_wrapper_refuses_lse_output():
    args = _args(2, 1, 256)
    with _abstract_tpu(), pytest.raises(ValueError, match="output_quant_dtype"):
        jax.jit(lambda *a: _call(*a, output_quant_dtype=_FP8, return_lse=True)).trace(
            *args
        )


@pytest.mark.tpu_test
@pytest.mark.parametrize("heads,fp8_compute", _GEOMETRIES)
def test_wrapper_matches_quantized_output(heads, fp8_compute):
    if jax.devices()[0].platform != "tpu":
        pytest.skip("the batched RPA kernel needs a TPU")
    out, _ = _call(*_args(*heads), fp8_compute=fp8_compute)
    ref_q, ref_scale = quantize_tensor(out.reshape(_TOKENS, -1), _FP8)
    (q, scale), _ = _call(
        *_args(*heads), fp8_compute=fp8_compute, output_quant_dtype=_FP8
    )
    np.testing.assert_array_equal(np.asarray(scale), np.asarray(ref_scale))
    np.testing.assert_array_equal(
        np.asarray(q.view(jnp.uint8)), np.asarray(ref_q.view(jnp.uint8))
    )
