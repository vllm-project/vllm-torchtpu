"""Fused Pallas/Mosaic TPU decode kernel for GLM 5.3 Flash KDA layers (v1).

This is the kernel that does not exist anywhere in google3 today, and it is
34 of the model's 45 layers. The chunked *prefill* path is already covered by
`//third_party/py/tokamax/_src/ops/experimental/kda`; this is its decode
counterpart.

Design is driven by measurement. Measured on TPU7x, the recurrent state is
4 MiB per sequence per layer against ~7 FLOP per state element, i.e. 0.875
FLOP/byte against 2.410 TB/s of HBM -- roughly two orders of magnitude below
the ridge point. So the only thing that matters is moving the state exactly
once, with as little else going on as possible.

## What v0 got wrong

The first version used a grid of `(num_seqs, num_heads)` and, because the
state is `[d_k, d_v]` with `d_k` in sublanes while `q`/`k`/`g` arrive with
`d_k` in lanes, it moved those three vectors onto the sublane axis with a
masked-iota transpose. Measured, that version hit only **8.9% of roofline and
was 5.7x slower than plain XLA**.

Two compounding causes, both visible in the A/B:

  * Each masked-iota transpose builds a `[128, 128]` iota pair, compares,
    selects and reduces -- about 15x more element-work than the delta rule it
    was supporting, and it ran three times per grid step.
  * The grid had `num_seqs * 64` steps each moving only 64 KiB, so per-step
    overhead dominated.

That is the same failure the z.ai blog's Figure 4 describes: redundant
per-tile recomputation of gating and normalization swamping the real work.

## What v1 does instead

**The state is stored transposed, as `[d_v, d_k]`.** This single change makes
every operand fall out in its natural TPU layout and removes all transposes:

  * decay multiplies along `d_k`, which is now the lane axis, so the gate
    broadcasts directly;
  * `pred` and `out` become lane reductions with `k`/`q` as lane vectors;
  * `resid` emerges from a lane reduction already on the sublane axis, which
    is exactly what the rank-1 outer product needs.

**One grid step per sequence, vectorized across all 64 heads.** The head axis
becomes the leading (non-tiled) dimension, so the whole step is expressed as
batched elementwise work with no per-head loop, and the grid shrinks from
`num_seqs * 64` steps to `num_seqs`.

> Layout contract: `recurrent_state` is `[num_slots, num_heads, d_v, d_k]`,
> **not** the `[..., d_k, d_v]` of the HuggingFace reference. Since
> `d_k == d_v == 128` for GLM 5.3 Flash the shape tuple is unchanged and only
> the meaning of the last two axes differs, but prefill must write the state
> in this order. A single transpose at the prefill/decode boundary is cheap
> relative to prefill itself.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.glm53_kda import kda_decode_ref as ref


def _l2_normalize(x: jax.Array, eps: float = ref.L2NORM_EPS) -> jax.Array:
    """L2-normalize over the last (lane) axis, in fp32."""
    xf = x.astype(jnp.float32)
    return xf * jax.lax.rsqrt(jnp.sum(xf * xf, axis=-1, keepdims=True) + eps)


def select_decode_h_block(num_seqs: int, num_heads: int = ref.NUM_HEADS) -> int:
    """Selects optimal Pallas head block size `h_block` for KDA decode on TPU v7x."""
    if num_seqs <= 4 and num_heads % 32 == 0:
        return 32
    if num_seqs <= 16 and num_heads % 16 == 0:
        return 16
    return num_heads


def select_prefill_h_block(
    num_seqs: int, chunk_len: int, num_heads: int = ref.NUM_HEADS
) -> int:
    """Selects optimal Pallas head block size `h_block` for KDA chunked prefill."""
    del num_seqs
    if chunk_len <= 64 and num_heads % 32 == 0:
        return 32
    if chunk_len <= 256 and num_heads % 16 == 0:
        return 16
    if num_heads % 8 == 0:
        return 8
    return num_heads


def _kda_decode_kernel(
    # Scalar prefetch.
    state_indices_ref,
    # Inputs.
    q_norm_ref,  # [1, h_block, d_k] float32
    k_norm_ref,  # [1, h_block, d_k] float32
    v_ref,  # [1, h_block, d_v]
    decay_ref,  # [1, h_block, d_k] float32
    bk_ref,  # [1, h_block, d_k] float32
    state_ref,  # [1, h_block, d_v, d_k], aliased to the output
    # Outputs.
    out_ref,  # [1, h_block, d_v]
    new_state_ref,  # [1, h_block, d_v, d_k], same dtype as state_ref
    *,
    math_repeats: int,
    null_slot: int = -1,
):
    """One `(seq_idx, head_block)` decode tile in VMEM.

    With `null_slot >= 0`, a sequence whose state slot is `null_slot` is batch
    padding: its state block passes through unchanged and its output is zeros.
    That is what lets the kernel run directly on the whole slot cache, where
    padded entries all point at the (never assigned) null slot.
    """

    def _step():
        state = state_ref[0].astype(jnp.float32)
        out = jnp.zeros(out_ref.shape[1:], jnp.float32)

        if math_repeats:
            q = q_norm_ref[0].astype(jnp.float32)  # [h_block, d_k]
            k = k_norm_ref[0].astype(jnp.float32)  # [h_block, d_k]
            v = v_ref[0].astype(jnp.float32)  # [h_block, d_v]
            decay = decay_ref[0].astype(jnp.float32)  # [h_block, d_k]
            bk = bk_ref[0].astype(jnp.float32)[:, None, :]  # [h_block, 1, d_k]

            for _ in range(math_repeats):
                state = state * decay[:, None, :]
                pred = jnp.sum(k[:, None, :] * state, axis=2)  # [h_block, d_v]
                resid = v - pred
                state = state + resid[:, :, None] * bk
                out = jnp.sum(q[:, None, :] * state, axis=2)  # [h_block, d_v]

        new_state_ref[0] = state.astype(new_state_ref.dtype)
        out_ref[0] = out.astype(out_ref.dtype)

    if null_slot < 0:
        del state_indices_ref  # Consumed by the index maps, not the body.
        _step()
        return

    slot = state_indices_ref[pl.program_id(0)]

    @pl.when(slot != null_slot)
    def _execute():
        _step()

    @pl.when(slot == null_slot)
    def _passthrough():
        # The output block aliases the input block, so it must be written even
        # here: an unwritten VMEM buffer would flush stale data into the slot.
        new_state_ref[0] = state_ref[0]
        out_ref[0] = jnp.zeros(out_ref.shape[1:], out_ref.dtype)


@functools.partial(
    jax.jit,
    static_argnames=(
        "num_heads",
        "head_dim",
        "lower_bound",
        "qk_scale",
        "math_repeats",
        "h_block",
        "null_slot",
    ),
    donate_argnames=("recurrent_state",),
)
def kda_decode(
    q: jax.Array,  # [num_seqs, num_heads, d_k]
    k: jax.Array,  # [num_seqs, num_heads, d_k]
    v: jax.Array,  # [num_seqs, num_heads, d_v]
    g_raw: jax.Array,  # [num_seqs, num_heads, d_k]
    beta: jax.Array,  # [num_seqs, num_heads]
    a_log: jax.Array,  # [num_heads]
    dt_bias: jax.Array,  # [num_heads, d_k]
    recurrent_state: jax.Array,  # [num_slots, num_heads, d_v, d_k]
    state_indices: jax.Array,  # [num_seqs] int32
    *,
    num_heads: int = ref.NUM_HEADS,
    head_dim: int = ref.HEAD_DIM,
    lower_bound: float = ref.LOWER_BOUND,
    qk_scale: float = ref.QK_SCALE,
    math_repeats: int = 1,
    h_block: int = 0,
    null_slot: int = -1,
) -> tuple[jax.Array, jax.Array]:
    """`jax.jit`-wrapped KDA decode. See `_kda_decode_impl` for the semantics."""
    del num_heads, head_dim
    return _kda_decode_impl(
        q,
        k,
        v,
        g_raw,
        beta,
        a_log,
        dt_bias,
        recurrent_state,
        state_indices,
        lower_bound,
        qk_scale,
        math_repeats,
        h_block=h_block,
        null_slot=null_slot,
    )


def kda_decode_jax(
    q: jax.Array,  # [num_seqs, num_heads, d_k]
    k: jax.Array,  # [num_seqs, num_heads, d_k]
    v: jax.Array,  # [num_seqs, num_heads, d_v]
    g_raw: jax.Array,  # [num_seqs, num_heads, d_k]
    beta: jax.Array,  # [num_seqs, num_heads]
    a_log: jax.Array,  # [num_heads]
    dt_bias: jax.Array,  # [num_heads, d_k]
    recurrent_state: jax.Array,  # [num_slots, num_heads, d_v, d_k]
    state_indices: jax.Array,  # [num_seqs] int32
    lower_bound: float,
    qk_scale: float,
    h_block: int = 0,
    null_slot: int = -1,
) -> tuple[jax.Array, jax.Array]:
    """Un-jitted entry point, shaped for the torch_tpu Pallas bridge."""
    return _kda_decode_impl(
        q,
        k,
        v,
        g_raw,
        beta,
        a_log,
        dt_bias,
        recurrent_state,
        state_indices,
        lower_bound,
        qk_scale,
        math_repeats=1,
        h_block=h_block,
        null_slot=null_slot,
    )


def _kda_decode_impl(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g_raw: jax.Array,
    beta: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array,
    recurrent_state: jax.Array,
    state_indices: jax.Array,
    lower_bound: float,
    qk_scale: float,
    math_repeats: int,
    h_block: int = 0,
    null_slot: int = -1,
) -> tuple[jax.Array, jax.Array]:
    """Fused KDA decode: one token per sequence, tiled by `h_block`.

    Sequence `s` reads and writes state slot `state_indices[s]` of
    `recurrent_state` in place (the state is aliased to the output). The slots
    may index a dense, already-gathered batch or the serving layer's whole slot
    cache; in the latter case pass the cache's padding slot as `null_slot` (see
    `_kda_decode_kernel`) so padded entries neither advance nor emit anything.
    Slots must be unique apart from `null_slot`.
    """
    num_seqs, num_heads, head_dim = q.shape
    if h_block <= 0:
        h_block = select_decode_h_block(num_seqs, num_heads)
    num_h_blocks = num_heads // h_block

    q_norm = _l2_normalize(q) * qk_scale
    k_norm = _l2_normalize(k)
    bk = beta.astype(jnp.float32)[..., None] * k_norm
    g = g_raw.astype(jnp.float32) + dt_bias.astype(jnp.float32)[None, :, :]
    a = jnp.exp(a_log.astype(jnp.float32))[None, :, None]
    decay = jnp.exp(lower_bound * jax.nn.sigmoid(a * g))

    def seq_hb_map(s, hb, _):
        return (s, hb, 0)

    def state_map(s, hb, state_indices_ref):
        # Clamped so that a negative padding id can never address memory outside
        # the cache.
        return (jnp.maximum(state_indices_ref[s], 0), hb, 0, 0)

    def tok_spec():
        return pl.BlockSpec((1, h_block, head_dim), seq_hb_map)

    kernel = pl.pallas_call(
        functools.partial(
            _kda_decode_kernel,
            math_repeats=math_repeats,
            null_slot=null_slot,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(num_seqs, num_h_blocks),
            in_specs=[
                tok_spec(),  # q_norm
                tok_spec(),  # k_norm
                tok_spec(),  # v
                tok_spec(),  # decay
                tok_spec(),  # bk
                pl.BlockSpec((1, h_block, head_dim, head_dim), state_map),
            ],
            out_specs=[
                pl.BlockSpec((1, h_block, head_dim), seq_hb_map),
                pl.BlockSpec((1, h_block, head_dim, head_dim), state_map),
            ],
        ),
        out_shape=[
            jax.ShapeDtypeStruct((num_seqs, num_heads, head_dim), jnp.float32),
            jax.ShapeDtypeStruct(recurrent_state.shape, recurrent_state.dtype),
        ],
        input_output_aliases={6: 1},
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary", "parallel"),
        ),
    )

    return kernel(state_indices, q_norm, k_norm, v, decay, bk, recurrent_state)


def _kda_prefill_chunk_kernel(
    valid_lens_ref,  # SMEM: [num_seqs] int32
    q_norm_ref,  # VMEM: [1, chunk_len, h_block, d_k] float32
    k_norm_ref,  # VMEM: [1, chunk_len, h_block, d_k] float32
    v_ref,  # VMEM: [1, chunk_len, h_block, d_v]
    decay_ref,  # VMEM: [1, chunk_len, h_block, d_k] float32
    bk_ref,  # VMEM: [1, chunk_len, h_block, d_k] float32
    state_ref,  # VMEM: [1, h_block, d_v, d_k]
    out_ref,  # VMEM: [1, chunk_len, h_block, d_v]
    new_state_ref,  # VMEM: [1, h_block, d_v, d_k] (aliased to state_ref)
):
    """Fused VMEM multi-token prefill chunk kernel for `(seq_idx, head_block)`."""
    seq_idx = pl.program_id(0)
    valid_len = valid_lens_ref[seq_idx]
    chunk_len = q_norm_ref.shape[1]

    # Early-exit / skip execution if this is a padded dummy sequence (valid_len == 0)
    @pl.when(valid_len > 0)
    def _execute():
        state = state_ref[0].astype(jnp.float32)

        def _step_body(t, st):
            decay = decay_ref[0, t].astype(jnp.float32)
            q = q_norm_ref[0, t].astype(jnp.float32)
            k = k_norm_ref[0, t].astype(jnp.float32)
            v = v_ref[0, t].astype(jnp.float32)
            bk = bk_ref[0, t].astype(jnp.float32)[:, None, :]

            st_decayed = st * decay[:, None, :]
            pred = jnp.sum(k[:, None, :] * st_decayed, axis=2)
            resid = v - pred
            next_st = st_decayed + resid[:, :, None] * bk
            out_t = jnp.sum(q[:, None, :] * next_st, axis=2)

            out_ref[0, t] = out_t.astype(out_ref.dtype)
            return jnp.where(t < valid_len, next_st, st)

        final_st = jax.lax.fori_loop(0, chunk_len, _step_body, state)
        new_state_ref[0] = final_st.astype(new_state_ref.dtype)

    @pl.when(valid_len <= 0)
    def _passthrough():
        new_state_ref[0] = state_ref[0]


def kda_prefill_chunk_jax(
    q: jax.Array,  # [num_seqs, chunk_len, num_heads, d_k]
    k: jax.Array,  # [num_seqs, chunk_len, num_heads, d_k]
    v: jax.Array,  # [num_seqs, chunk_len, num_heads, d_v]
    g_raw: jax.Array,  # [num_seqs, chunk_len, num_heads, d_k]
    beta: jax.Array,  # [num_seqs, chunk_len, num_heads]
    a_log: jax.Array,  # [num_heads]
    dt_bias: jax.Array,  # [num_heads, d_k]
    recurrent_state: jax.Array,  # [num_seqs, num_heads, d_v, d_k]
    valid_lens: jax.Array,  # [num_seqs] int32
    lower_bound: float,
    qk_scale: float,
    h_block: int = 0,
) -> tuple[jax.Array, jax.Array]:
    """Un-jitted Pallas chunk prefill entry point for `torch_tpu`'s `jax_op`."""
    num_seqs, chunk_len, num_heads, head_dim = q.shape
    if h_block <= 0:
        h_block = select_prefill_h_block(num_seqs, chunk_len, num_heads)
    num_h_blocks = num_heads // h_block

    q_norm = _l2_normalize(q) * qk_scale
    k_norm = _l2_normalize(k)
    bk = beta.astype(jnp.float32)[..., None] * k_norm
    g = g_raw.astype(jnp.float32) + dt_bias.astype(jnp.float32)[None, None, :, :]
    a = jnp.exp(a_log.astype(jnp.float32))[None, None, :, None]
    decay = jnp.exp(lower_bound * jax.nn.sigmoid(a * g))

    def seq_chunk_map(s, hb, _):
        return (s, 0, hb, 0)

    def state_seq_map(s, hb, _):
        return (s, hb, 0, 0)

    def chunk_spec():
        return pl.BlockSpec((1, chunk_len, h_block, head_dim), seq_chunk_map)

    kernel = pl.pallas_call(
        _kda_prefill_chunk_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(num_seqs, num_h_blocks),
            in_specs=[
                chunk_spec(),  # q_norm
                chunk_spec(),  # k_norm
                chunk_spec(),  # v
                chunk_spec(),  # decay
                chunk_spec(),  # bk
                pl.BlockSpec((1, h_block, head_dim, head_dim), state_seq_map),
            ],
            out_specs=[
                chunk_spec(),
                pl.BlockSpec((1, h_block, head_dim, head_dim), state_seq_map),
            ],
        ),
        out_shape=[
            jax.ShapeDtypeStruct(
                (num_seqs, chunk_len, num_heads, head_dim), jnp.float32
            ),
            jax.ShapeDtypeStruct(recurrent_state.shape, recurrent_state.dtype),
        ],
        input_output_aliases={6: 1},
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel"),
        ),
    )
    return kernel(valid_lens, q_norm, k_norm, v, decay, bk, recurrent_state)
