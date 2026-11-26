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
holds work. These kernels take the prefix length as a scalar: the permute
runs the 128-slot blocks that cover it, the combine the smallest power-of-two
multiple of 128 slots. Both build the one-hot from the inverse sort
permutation (``argsort_revert_indices``) laid out [topk, tokens].
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_SLOT_BLOCK = 128
# Hidden columns per pipeline step (measured on tpu7x: 2 steps of 3072 beat
# 1 of 6144 and 4 of 1536): the column chunks stream through VMEM
# while the previous chunk's matmul runs.
_COL_CHUNKS = (3072, 1536, 1024, 512)


def _vmem_bytes(num_tokens: int, num_slots: int, hidden_size: int, chunk: int) -> int:
    # Permute: resident hidden block + double-buffered 128-slot output blocks.
    permute = (num_tokens + 2 * _SLOT_BLOCK) * hidden_size * 2
    # Combine: one-hot scratch + double-buffered input and output column chunks.
    combine = num_slots * num_tokens * 2 + 2 * (num_slots + num_tokens) * chunk * 2
    return max(permute, combine)


def _col_chunk(num_tokens: int, num_slots: int, hidden_size: int) -> int | None:
    try:
        budget = pltpu.get_tpu_info().vmem_capacity_bytes * 0.8
    except ValueError:  # not a TPU (CPU tests)
        return None
    return next(
        (
            c
            for c in _COL_CHUNKS
            if hidden_size % c == 0
            and _vmem_bytes(num_tokens, num_slots, hidden_size, c) <= budget
        ),
        None,
    )


def _slot_caps(num_slots: int) -> list[int]:
    """Static prefix lengths, each compiled as one branch: 128, 256, ... slots."""
    caps, cap = [], _SLOT_BLOCK
    while cap < num_slots:
        caps.append(cap)
        cap *= 2
    return [*caps, num_slots]


def can_use_bounded_onehot(num_tokens: int, num_slots: int, hidden_size: int) -> bool:
    """Shapes both kernels lay out, within VMEM: whole 128-token lanes, or
    with TPU_MOE_ONEHOT_BOUNDED_SUBLANE_TOKENS past 256 tokens whole bf16
    sublane tiles (Mosaic masks the lane padding: the 320/416-token c48/c64
    decode buckets), and whole 128-slot blocks."""
    from vllm_torchtpu import envs

    sublane = envs.TPU_MOE_ONEHOT_BOUNDED_SUBLANE_TOKENS
    return (
        (num_tokens % 128 == 0 or (sublane and num_tokens > 256 and num_tokens % 16 == 0))
        and num_slots % _SLOT_BLOCK == 0
        and _col_chunk(num_tokens, num_slots, hidden_size) is not None
    )


def _run_capped(valid_ref, num_slots: int, branch) -> None:
    """branch(cap) for the smallest cap covering the valid prefix."""
    caps = _slot_caps(num_slots)
    valid = valid_ref[0]
    idx = sum((valid > c).astype(jnp.int32) for c in caps[:-1])
    lax.switch(idx, [functools.partial(branch, c) for c in caps])


def _permute_kernel(valid_ref, revert_t_ref, hidden_ref, out_hbm_ref):
    topk, num_tokens = revert_t_ref.shape
    num_slots, hidden_size = out_hbm_ref.shape
    # 0 * NaN == NaN: a non-finite token would reach every slot. x * 0 is 0
    # exactly for finite x (Mosaic's isfinite takes f32 only).
    hidden = hidden_ref[...]
    hidden_ref[...] = jnp.where(hidden * 0 == 0, hidden, jnp.zeros_like(hidden))

    def body(out_ref):
        slots = lax.broadcasted_iota(jnp.int32, (_SLOT_BLOCK, num_tokens), 0)
        slots += pl.program_id(0) * _SLOT_BLOCK
        onehot = jnp.zeros((_SLOT_BLOCK, num_tokens), jnp.float32)
        for k in range(topk):
            onehot += (revert_t_ref[k : k + 1, :] == slots).astype(jnp.float32)
        out_ref[...] = jnp.dot(
            onehot.astype(hidden_ref.dtype),
            hidden_ref[...],
            preferred_element_type=jnp.float32,
        ).astype(out_ref.dtype)

    # One 128-slot block at a time up to the valid prefix: power-of-two caps
    # (as the combine takes) ran up to 2x the blocks here.
    valid = jnp.clip(valid_ref[0], 0, num_slots)
    pltpu.emit_pipeline(
        body,
        grid=(jnp.maximum(1, pl.cdiv(valid, _SLOT_BLOCK)),),
        in_specs=(),
        out_specs=(pl.BlockSpec((_SLOT_BLOCK, hidden_size), lambda b: (b, 0)),),
    )(out_hbm_ref)


@jax.jit
def bounded_onehot_permute(
    hidden: jax.Array,  # [tokens, hidden]
    argsort_revert_indices: jax.Array,  # int32[tokens * topk]
    valid_count: jax.Array,  # int32[1]
) -> jax.Array:
    """Routed buffer [tokens * topk, hidden]: slot s holds its token's row,
    with non-finite values zeroed.

    Only the 128-slot blocks covering ``valid_count`` are written; the GMM
    never reads past the valid prefix.
    """
    num_tokens, hidden_size = hidden.shape
    num_slots = argsort_revert_indices.size
    revert_t = argsort_revert_indices.reshape(num_tokens, -1).T
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    return pl.pallas_call(
        _permute_kernel,
        out_shape=jax.ShapeDtypeStruct((num_slots, hidden_size), hidden.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            # The whole hidden block sits in VMEM: streaming its column chunks
            # from HBM ran 2x slower inside the served graph than alone.
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.VMEM),
                pl.BlockSpec(memory_space=pltpu.VMEM),
            ],
            out_specs=hbm,
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9),
        ),
        name=f"bounded_onehot_permute-s_{num_slots}-t_{num_tokens}",
    )(valid_count, revert_t, hidden)


def _combine_kernel(
    valid_ref, revert_t_ref, weights_t_ref, routed_hbm_ref, out_hbm_ref, combine_ref
):
    topk, num_tokens = revert_t_ref.shape
    num_slots, hidden_size = routed_hbm_ref.shape
    chunk = _col_chunk(num_tokens, num_slots, hidden_size)
    valid = valid_ref[0]

    def branch(cap):
        for s0 in range(0, cap, _SLOT_BLOCK):
            # Built slot-major like the permute's one-hot (tokens on lanes,
            # where revert_t has them), then transposed on the XLU.
            slots = lax.broadcasted_iota(jnp.int32, (_SLOT_BLOCK, num_tokens), 0) + s0
            combine_t = jnp.zeros((_SLOT_BLOCK, num_tokens), jnp.float32)
            for k in range(topk):
                combine_t += jnp.where(
                    revert_t_ref[k : k + 1, :] == slots,
                    weights_t_ref[k : k + 1, :].astype(jnp.float32),
                    0.0,
                )
            combine_ref[:, s0 : s0 + _SLOT_BLOCK] = combine_t.T.astype(
                combine_ref.dtype
            )

        def body(routed_ref, out_ref):
            rows = routed_ref[...]
            # The GMM leaves rows past the valid prefix unwritten: clear them so
            # a stale NaN cannot reach a token through a zero weight.
            row_slot = lax.broadcasted_iota(jnp.int32, rows.shape, 0)
            rows = jnp.where(row_slot < valid, rows, jnp.zeros_like(rows))
            out_ref[...] = jnp.dot(
                combine_ref[:, :cap], rows, preferred_element_type=jnp.float32
            ).astype(out_ref.dtype)

        pltpu.emit_pipeline(
            body,
            grid=(hidden_size // chunk,),
            in_specs=(pl.BlockSpec((cap, chunk), lambda c: (0, c)),),
            out_specs=(pl.BlockSpec((num_tokens, chunk), lambda c: (0, c)),),
        )(routed_hbm_ref, out_hbm_ref)

    _run_capped(valid_ref, num_slots, branch)


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
    revert_t = argsort_revert_indices.reshape(num_tokens, -1).T
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    return pl.pallas_call(
        _combine_kernel,
        out_shape=jax.ShapeDtypeStruct((num_tokens, hidden_size), routed_out.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[vmem, vmem, hbm],
            out_specs=hbm,
            scratch_shapes=[pltpu.VMEM((num_tokens, num_slots), routed_out.dtype)],
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9),
        ),
        name=f"bounded_onehot_combine-s_{num_slots}-t_{num_tokens}",
    )(valid_count, revert_t, weights.T, routed_out)
