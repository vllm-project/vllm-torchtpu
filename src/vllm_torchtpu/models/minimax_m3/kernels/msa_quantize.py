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
"""Rotate and quantize a step's K/V heads into 4-bit cache rows in one Pallas
pass: `msa_fp4.quantize(msa_fp4.rotate_keys(x))` of every head of ``k`` and
``v``, written as the cache rows ``[K_0 | V_0 | ... | K_G-1 | V_G-1]``.

As XLA fusions (8K tokens, 4 heads, c64 twin j-9a9fccc2) the rotation wrote
f32 keys and values (16 MB each) for the quantize to read back, and the
interleave of the nibbles into rows was another copy: ~50 us per layer.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.models.minimax_m3.kernels import msa_fp4

BLOCK_TOKENS = 2048
# Steps of at least this many tokens use the kernel (`msa.write_fp4`); decode
# steps keep the XLA fusions (as `msa_dcp_heads.HEAD_PARALLEL_MIN_TOKENS`).
MIN_TOKENS = 2048
# Lanes of `quantize_kv_rows`' lane block (`index_k`): one 32-bit word per
# slot of a page, as `msa_write` reads per-slot values.
_LANES = 128


def _quantize(x, h_ref):
    """`msa_fp4.quantize(msa_fp4.rotate_keys(x))` op for op: (nibbles
    ``[Bt, D]``, scales ``[1, Bt]``). The rotation comes out ``[D, Bt]`` (the
    Hadamard matrix is symmetric), so the amax reduces across vregs, not
    lanes, and the scales come out lane-dense; only the nibbles are
    transposed back."""
    y = jax.lax.dot_general(
        h_ref[...], x, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
    )
    amax = jnp.max(jnp.abs(y), axis=0, keepdims=True)
    inv = jnp.where(amax > 0, msa_fp4.INT4_MAX / jnp.where(amax > 0, amax, 1.0), 0.0)
    q = jnp.clip(jnp.round(y * inv), -msa_fp4.INT4_MAX, msa_fp4.INT4_MAX)
    return (q.T.astype(jnp.int32) & 0xF).astype(jnp.uint8), amax / msa_fp4.INT4_MAX


def _kernel(h_ref, k_ref, v_ref, *refs, heads, head_dim):
    index = len(refs) == 4  # index keys in; their nibbles and the slot lanes out
    ik_ref, refs = (refs[0], refs[1:]) if index else (None, refs)
    rows_ref, scales_ref = refs[0], refs[-1]
    scales = []
    for h in range(heads):
        for i, x_ref in enumerate((k_ref, v_ref)):
            q, scale = _quantize(x_ref[:, h * head_dim : (h + 1) * head_dim], h_ref)
            col = (2 * h + i) * head_dim
            rows_ref[:, col : col + head_dim] = q
            scales.append(scale)
    if not index:
        scales_ref[...] = jnp.concatenate(scales, axis=0)
        return
    refs[1][...], index_scale = _quantize(ik_ref[...], h_ref)
    # Token-major lanes: the K/V scales from lane 0, the index key's in the
    # last lane, zeros between (8-row blocks transposed).
    zeros = jnp.zeros_like(index_scale)
    head = -(-len(scales) // 8) * 8
    first = jnp.concatenate(scales + [zeros] * (head - len(scales)))
    last = jnp.concatenate([zeros] * 7 + [index_scale])
    words = functools.partial(jax.lax.bitcast_convert_type, new_dtype=jnp.uint32)
    scales_ref[:, :head] = words(first.T)
    scales_ref[:, head : _LANES - 8] = jnp.zeros(
        (first.shape[1], _LANES - 8 - head), jnp.uint32
    )
    scales_ref[:, _LANES - 8 :] = words(last.T)


def quantize_kv_rows(
    k: jax.Array,
    v: jax.Array,
    index_k: jax.Array | None = None,
    *,
    interpret: bool = False,
) -> tuple[jax.Array, ...]:
    """bf16 ``k``/``v [T, G, D]`` -> (uint8 nibble rows ``[T, 2 * G * D]``,
    f32 scales ``[T, 2 * G]`` ordered ``[k_0, v_0, ..., k_G-1, v_G-1]``).

    With bf16 ``index_k [T, D]`` also its nibbles ``[T, D]`` and scales
    ``[T]``, from the same pass, and every scale's bits as one uint32 lane
    block ``[T, 128]`` (the K/V scales in lanes ``[0, 2 G)``, the index
    key's in the last lane, zeros elsewhere): `msa_write.SlotValues.lanes`
    of a write of the K/V scales and one folded-scale target. The scales
    returned are its columns.
    As XLA ops the lanes took a transposing fusion and a relayout of the
    whole block (~3.8 us at 8K tokens, anvil probe)."""
    num_tokens, heads, head_dim = k.shape
    block_t = min(num_tokens, BLOCK_TOKENS)
    rows = pl.BlockSpec((block_t, heads * head_dim), lambda i: (i, 0))
    in_specs = [pl.BlockSpec((head_dim, head_dim), lambda i: (0, 0)), rows, rows]
    out_specs = [pl.BlockSpec((block_t, 2 * heads * head_dim), lambda i: (i, 0))]
    out_shape = [jax.ShapeDtypeStruct((num_tokens, 2 * heads * head_dim), jnp.uint8)]
    inputs = [
        jnp.asarray(msa_fp4.hadamard(head_dim), jnp.bfloat16),
        k.reshape(num_tokens, -1),
        v.reshape(num_tokens, -1),
    ]
    if index_k is None:
        out_specs.append(pl.BlockSpec((2 * heads, block_t), lambda i: (0, i)))
        out_shape.append(jax.ShapeDtypeStruct((2 * heads, num_tokens), jnp.float32))
    else:
        if 2 * heads > _LANES - 8:
            raise ValueError(
                f"{heads} KV heads' scales and the index key's over {_LANES} lanes"
            )
        in_specs.append(pl.BlockSpec((block_t, head_dim), lambda i: (i, 0)))
        inputs.append(index_k)
        out_specs += [
            pl.BlockSpec((block_t, head_dim), lambda i: (i, 0)),
            pl.BlockSpec((block_t, _LANES), lambda i: (i, 0)),
        ]
        out_shape += [
            jax.ShapeDtypeStruct((num_tokens, head_dim), jnp.uint8),
            jax.ShapeDtypeStruct((num_tokens, _LANES), jnp.uint32),
        ]
    rows, *rest = pl.pallas_call(
        lambda *refs: _kernel(*refs, heads=heads, head_dim=head_dim),
        grid=(pl.cdiv(num_tokens, block_t),),
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=out_shape,
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
        name="minimax_m3_msa_quantize_kv_rows",
    )(*inputs)
    if index_k is None:
        return rows, rest[0].T
    index_rows, lanes = rest
    scales = jax.lax.bitcast_convert_type(lanes[:, : 2 * heads], jnp.float32)
    index_scales = jax.lax.bitcast_convert_type(lanes[:, -1], jnp.float32)
    return rows, scales, index_rows, index_scales, lanes
