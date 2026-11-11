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
"""EP one-hot MoE permute and combine over the valid routed prefix only.

The dense one-hot forms (``fused_moe_gmm``) multiply over every routed slot,
but under EP only the expert-sorted prefix of local slots (about 1/EP of them)
holds work. These kernels take the prefix length as a scalar and visit only
the 128-slot blocks that cover it, building each block's one-hot from the
inverse sort permutation (``argsort_revert_indices``, [tokens, topk]).
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_SLOT_BLOCK = 128
# Output columns per combine matmul: bounds the f32 partial product.
_COMBINE_COLS = 512


def can_use_bounded_onehot(num_tokens: int, num_slots: int, hidden_size: int) -> bool:
    """Shapes both kernels lay out without padding, within VMEM."""
    if num_tokens % 128 or num_slots % _SLOT_BLOCK or hidden_size % _COMBINE_COLS:
        return False
    tpu_info = pltpu.get_tpu_info()
    # Combine: f32 owner + bf16 output + double-buffered slot blocks.
    combine_bytes = (
        num_tokens * hidden_size * (4 + 2) + 2 * _SLOT_BLOCK * hidden_size * 2
    )
    # Permute: resident bf16 hidden + double-buffered output blocks.
    permute_bytes = num_tokens * hidden_size * 2 + 2 * _SLOT_BLOCK * hidden_size * 2
    return max(combine_bytes, permute_bytes) <= tpu_info.vmem_capacity_bytes * 0.8


def _num_blocks(valid_count, num_slots: int):
    valid = jnp.clip(valid_count, 0, num_slots)
    # One block even for an empty prefix keeps the dynamic grid non-empty.
    return jnp.maximum(1, pl.cdiv(valid, _SLOT_BLOCK))


def _permute_kernel(valid_ref, revert_t_ref, hidden_ref, out_hbm_ref, *, num_slots):
    num_tokens = hidden_ref.shape[0]
    topk = revert_t_ref.shape[0]

    def body(out_ref):
        slot0 = pl.program_id(0) * _SLOT_BLOCK
        slots = lax.broadcasted_iota(jnp.int32, (_SLOT_BLOCK, num_tokens), 0) + slot0
        onehot = jnp.zeros((_SLOT_BLOCK, num_tokens), jnp.float32)
        for k in range(topk):
            onehot += (revert_t_ref[k : k + 1, :] == slots).astype(jnp.float32)
        out_ref[...] = jnp.dot(
            onehot.astype(hidden_ref.dtype),
            hidden_ref[...],
            preferred_element_type=jnp.float32,
        ).astype(out_ref.dtype)

    pltpu.emit_pipeline(
        body,
        grid=(_num_blocks(valid_ref[0], num_slots),),
        in_specs=(),
        out_specs=(pl.BlockSpec((_SLOT_BLOCK, hidden_ref.shape[1]), lambda b: (b, 0)),),
        dimension_semantics=("arbitrary",),
    )(out_hbm_ref)


@jax.jit
def bounded_onehot_permute(
    hidden: jax.Array,  # [tokens, hidden]
    argsort_revert_indices: jax.Array,  # int32[tokens * topk]
    valid_count: jax.Array,  # int32[1]
) -> jax.Array:
    """Routed buffer [tokens * topk, hidden]: slot s holds its token's row.

    Rows from the first 128-slot block past ``valid_count`` on are left
    unwritten; the GMM never reads past the valid prefix.
    """
    num_tokens, hidden_size = hidden.shape
    num_slots = argsort_revert_indices.size
    revert_t = argsort_revert_indices.reshape(num_tokens, -1).T
    return pl.pallas_call(
        functools.partial(_permute_kernel, num_slots=num_slots),
        out_shape=jax.ShapeDtypeStruct((num_slots, hidden_size), hidden.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.VMEM),
                pl.BlockSpec(memory_space=pltpu.VMEM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9),
        ),
        name=f"bounded_onehot_permute-s_{num_slots}-t_{num_tokens}",
    )(valid_count, revert_t, hidden)


def _combine_kernel(
    valid_ref, revert_ref, weights_ref, routed_hbm_ref, out_ref, owner_ref, *, num_slots
):
    num_tokens, topk = revert_ref.shape
    valid = jnp.clip(valid_ref[0], 0, num_slots)
    owner_ref[...] = jnp.zeros_like(owner_ref)

    def body(routed_ref):
        slot0 = pl.program_id(0) * _SLOT_BLOCK
        rows = routed_ref[...]
        # The GMM leaves rows past the valid prefix unwritten: clear them so a
        # stale NaN cannot reach a token through a zero weight.
        row_slot = lax.broadcasted_iota(jnp.int32, rows.shape, 0) + slot0
        rows = jnp.where(row_slot < valid, rows, jnp.zeros_like(rows))
        slots = lax.broadcasted_iota(jnp.int32, (num_tokens, _SLOT_BLOCK), 1) + slot0
        combine = jnp.zeros((num_tokens, _SLOT_BLOCK), jnp.float32)
        for k in range(topk):
            combine += jnp.where(
                revert_ref[:, k : k + 1] == slots, weights_ref[:, k : k + 1], 0.0
            )
        combine = combine.astype(rows.dtype)
        for n in range(0, rows.shape[1], _COMBINE_COLS):
            cols = slice(n, n + _COMBINE_COLS)
            owner_ref[:, cols] += jnp.dot(
                combine, rows[:, cols], preferred_element_type=jnp.float32
            )

    pltpu.emit_pipeline(
        body,
        grid=(_num_blocks(valid_ref[0], num_slots),),
        in_specs=(
            pl.BlockSpec((_SLOT_BLOCK, routed_hbm_ref.shape[1]), lambda b: (b, 0)),
        ),
        out_specs=(),
        dimension_semantics=("arbitrary",),
    )(routed_hbm_ref)
    out_ref[...] = owner_ref[...].astype(out_ref.dtype)


@jax.jit
def bounded_onehot_combine(
    routed_out: jax.Array,  # [tokens * topk, hidden], expert-sorted
    argsort_revert_indices: jax.Array,  # int32[tokens * topk]
    weights: jax.Array,  # [tokens, topk], zero for non-local routes
    valid_count: jax.Array,  # int32[1]
) -> jax.Array:
    """out[t] = sum_k weights[t, k] * routed_out[revert[t, k]], over the valid prefix.

    The matmul runs in ``routed_out.dtype`` with f32 accumulation.
    """
    num_slots, hidden_size = routed_out.shape
    num_tokens = weights.shape[0]
    revert = argsort_revert_indices.reshape(num_tokens, -1)
    weights = weights.astype(jnp.float32)
    return pl.pallas_call(
        functools.partial(_combine_kernel, num_slots=num_slots),
        out_shape=jax.ShapeDtypeStruct((num_tokens, hidden_size), routed_out.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.VMEM),
                pl.BlockSpec(memory_space=pltpu.VMEM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
            scratch_shapes=[pltpu.VMEM((num_tokens, hidden_size), jnp.float32)],
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9),
        ),
        name=f"bounded_onehot_combine-s_{num_slots}-t_{num_tokens}",
    )(valid_count, revert, weights, routed_out)
