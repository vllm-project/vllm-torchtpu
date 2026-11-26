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


# Output columns per shared-expert product (bounds its f32 temporary).
_SHARED_COLS = 512


def _vmem_bytes(
    num_tokens: int, num_slots: int, hidden_size: int, chunk: int, shared_inter: int
) -> int:
    # Permute: resident hidden block + double-buffered 128-slot output blocks.
    permute = (num_tokens + 2 * _SLOT_BLOCK) * hidden_size * 2
    # Combine: one-hot scratch + double-buffered input and output column chunks.
    combine = num_slots * num_tokens * 2 + 2 * (num_slots + num_tokens) * chunk * 2
    if shared_inter:
        # The shared expert's activation, double-buffered e4m3 weight and
        # (8-sublane padded) f32 scale column chunks, the f32 routed sum of a
        # chunk and the f32 temporaries of one _SHARED_COLS product.
        combine += num_tokens * shared_inter * 2 + 2 * (shared_inter + 32) * chunk
        combine += num_tokens * (chunk + 2 * _SHARED_COLS) * 4
    return max(permute, combine)


def _col_chunk(
    num_tokens: int, num_slots: int, hidden_size: int, shared_inter: int = 0
) -> int | None:
    try:
        budget = pltpu.get_tpu_info().vmem_capacity_bytes * 0.8
    except ValueError:  # not a TPU (CPU tests)
        return None
    return next(
        (
            c
            for c in _COL_CHUNKS
            if hidden_size % c == 0
            and (not shared_inter or c % _SHARED_COLS == 0)
            and _vmem_bytes(num_tokens, num_slots, hidden_size, c, shared_inter)
            <= budget
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


def can_use_bounded_onehot(
    num_tokens: int, num_slots: int, hidden_size: int, shared_inter: int = 0
) -> bool:
    """Shapes both kernels lay out, within VMEM: whole 128-token lanes, or
    past 256 tokens whole bf16 sublane tiles (Mosaic masks the lane padding:
    the 320/416-token c48/c64 decode buckets; at c48's 192 the bounded form
    gained nothing in the twin), and whole 128-slot blocks. ``shared_inter``:
    the intermediate size of a shared expert the combine adds, 0 for none."""
    return (
        (num_tokens % 128 == 0 or (num_tokens > 256 and num_tokens % 16 == 0))
        and num_slots % _SLOT_BLOCK == 0
        and shared_inter % 128 == 0
        and _col_chunk(num_tokens, num_slots, hidden_size, shared_inter) is not None
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


def _combine_kernel(valid_ref, revert_t_ref, weights_t_ref, routed_hbm_ref, *refs):
    shared = len(refs) == 5
    if shared:
        # [T, i_s] bf16 (VMEM), [i_s, H] e4m3 and [1, H] f32 (HBM).
        act_ref, w2s_hbm_ref, s2s_hbm_ref, out_hbm_ref, combine_ref = refs
    else:
        out_hbm_ref, combine_ref = refs
    topk, num_tokens = revert_t_ref.shape
    num_slots, hidden_size = routed_hbm_ref.shape
    chunk = _col_chunk(
        num_tokens, num_slots, hidden_size, act_ref.shape[1] if shared else 0
    )
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

        def routed_sum(routed_ref):
            rows = routed_ref[...]
            # The GMM leaves rows past the valid prefix unwritten: clear them so
            # a stale NaN cannot reach a token through a zero weight.
            row_slot = lax.broadcasted_iota(jnp.int32, rows.shape, 0)
            rows = jnp.where(row_slot < valid, rows, jnp.zeros_like(rows))
            return jnp.dot(
                combine_ref[:, :cap], rows, preferred_element_type=jnp.float32
            )

        def body(routed_ref, out_ref):
            out_ref[...] = routed_sum(routed_ref).astype(out_ref.dtype)

        def shared_body(routed_ref, w2s_ref, s2s_ref, out_ref):
            # Rounded as the unfused decode graph is: XLA's fusion of
            # fused_moe_gmm._add_shared_down keeps the projection in f32 up to
            # the one rounding of the sum (its excess precision drops the
            # bf16 converts). tc_combine rounds each step as SharedExpert
            # specifies; the two differ by at most one bf16 ulp.
            total = routed_sum(routed_ref)
            act = act_ref[...]
            for c0 in range(0, chunk, _SHARED_COLS):
                cols = slice(c0, c0 + _SHARED_COLS)
                y = jnp.dot(act, w2s_ref[:, cols], preferred_element_type=jnp.float32)
                routed = total[:, cols].astype(out_ref.dtype).astype(jnp.float32)
                out_ref[:, cols] = (routed + y * s2s_ref[:, cols]).astype(out_ref.dtype)

        routed_spec = pl.BlockSpec((cap, chunk), lambda c: (0, c))
        out_spec = pl.BlockSpec((num_tokens, chunk), lambda c: (0, c))
        if not shared:
            pltpu.emit_pipeline(
                body,
                grid=(hidden_size // chunk,),
                in_specs=(routed_spec,),
                out_specs=(out_spec,),
            )(routed_hbm_ref, out_hbm_ref)
            return
        pltpu.emit_pipeline(
            shared_body,
            grid=(hidden_size // chunk,),
            in_specs=(
                routed_spec,
                pl.BlockSpec((w2s_hbm_ref.shape[0], chunk), lambda c: (0, c)),
                pl.BlockSpec((1, chunk), lambda c: (0, c)),
            ),
            out_specs=(out_spec,),
        )(routed_hbm_ref, w2s_hbm_ref, s2s_hbm_ref, out_hbm_ref)

    _run_capped(valid_ref, num_slots, branch)


@jax.jit
def bounded_onehot_combine(
    routed_out: jax.Array,  # [tokens * topk, hidden], expert-sorted
    argsort_revert_indices: jax.Array,  # int32[tokens * topk]
    weights: jax.Array,  # [tokens, topk], zero for non-local routes
    valid_count: jax.Array,  # int32[1]
    shared_down: tuple[jax.Array, jax.Array, jax.Array] | None = None,
) -> jax.Array:
    """out[t] = sum_k weights[t, k] * routed_out[revert[t, k]], over the valid prefix.

    The matmul runs in ``routed_out.dtype`` with f32 accumulation.
    ``shared_down`` (activation [T, i_s] bf16, w2 [i_s, H] e4m3, s2 [1, H]
    f32) adds (act @ w2) * s2 in f32 to the bf16 sum before its one bf16
    rounding, as XLA evaluates the separate add: the routed sum is not
    written to HBM and read back.
    """
    num_slots, hidden_size = routed_out.shape
    num_tokens = weights.shape[0]
    revert_t = argsort_revert_indices.reshape(num_tokens, -1).T
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    shared_operands, shared_specs = (), []
    if shared_down is not None:
        act, w2s, s2s = shared_down
        shared_operands = (act, w2s, s2s.astype(jnp.float32).reshape(1, hidden_size))
        shared_specs = [vmem, hbm, hbm]
    return pl.pallas_call(
        _combine_kernel,
        out_shape=jax.ShapeDtypeStruct((num_tokens, hidden_size), routed_out.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[vmem, vmem, hbm, *shared_specs],
            out_specs=hbm,
            scratch_shapes=[pltpu.VMEM((num_tokens, num_slots), routed_out.dtype)],
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9),
        ),
        name=f"bounded_onehot_combine-s_{num_slots}-t_{num_tokens}"
        + ("-shared" if shared_down is not None else ""),
    )(valid_count, revert_t, weights.T, routed_out, *shared_operands)
