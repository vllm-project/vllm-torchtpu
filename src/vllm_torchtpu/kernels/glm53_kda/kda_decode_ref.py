"""Reference implementation of ragged KDA decode for GLM 5.3 Flash.

This is the *correctness oracle* for the Pallas decode kernel. It is written to
be obviously-correct rather than fast: a literal transcription of the
token-by-token recurrence, with no fusion and no tiling.

The recurrence is transcribed from the HuggingFace reference
`recurrent_kimi_delta_attention` in
`//third_party/py/transformers/v5/models/glm5_next/modeling_glm5_next.py`, and
cross-checked against the JAX reference in
`//third_party/py/tokamax/_src/ops/experimental/kda/reference.py`.

For a single head, with state `S` of shape `[d_k, d_v]`:

    S    <- S * exp(g_t)[:, None]          # per-key-channel decay
    pred  = k_t @ S                        # [d_v]
    resid = v_t - pred                     # [d_v]
    S    <- S + (beta_t * k_t)[:, None] * resid[None, :]
    out_t = q_t @ S                        # note: the *updated* state

`beta` is folded into `k` rather than `resid`; this is exactly equivalent
because the update is a rank-1 outer product.

The gate is GLM's bounded "safe gate", which differs from Qwen3-Next's GDN gate:

    g = lower_bound * sigmoid(exp(A_log) * (g_raw + dt_bias))

with `lower_bound = -5.0`, giving `g` in `[-5, 0]`. Critically, `dt_bias` is
per `(head, key_channel)` and `A_log` is per `head` only, so `g` has shape
`[num_heads, d_k]` per token -- *not* the per-head scalar that GDN uses. That
difference is the entire reason a separate kernel is needed.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# GLM 5.3 Flash linear-attention hyperparameters, from
# `Glm5NextTextConfig` in configuration_glm5_next.py.
NUM_HEADS = 64
HEAD_DIM = 128
CONV_KERNEL_DIM = 4
LOWER_BOUND = -5.0
# `Glm5NextTextLinearAttention` scales queries by 1/sqrt(head_dim).
QK_SCALE = HEAD_DIM**-0.5
L2NORM_EPS = 1e-6


def activate_gate(
    g_raw: jax.Array,  # [..., num_heads, d_k]
    a_log: jax.Array,  # [num_heads]
    dt_bias: jax.Array,  # [num_heads, d_k]
    lower_bound: float = LOWER_BOUND,
) -> jax.Array:
    """GLM's bounded safe-gate. Returns log-decay in `[lower_bound, 0]`."""
    g = g_raw.astype(jnp.float32) + dt_bias.astype(jnp.float32)
    a = jnp.exp(a_log.astype(jnp.float32))[:, None]  # [num_heads, 1]
    return lower_bound * jax.nn.sigmoid(a * g)


def l2_normalize(x: jax.Array, eps: float = L2NORM_EPS) -> jax.Array:
    """L2-normalize over the last axis in fp32.

    KDA calls `chunk_kda(..., use_qk_l2norm_in_kernel=True)`, so q and k are
    normalized *inside* the kernel, before the scale is applied.
    """
    x_f = x.astype(jnp.float32)
    return x_f * jax.lax.rsqrt(jnp.sum(x_f * x_f, axis=-1, keepdims=True) + eps)


def causal_conv1d_update(
    conv_state: jax.Array,  # [conv_kernel_dim - 1, conv_dim]
    x: jax.Array,  # [conv_dim]
    weight: jax.Array,  # [conv_dim, conv_kernel_dim]
) -> tuple[jax.Array, jax.Array]:
    """Single-token depthwise causal conv1d update with SiLU.

    Mirrors `causal_conv1d_update(..., activation="silu")` in the HF model.
    Returns `(activated_output, new_conv_state)`.
    """
    # Window is the stored history followed by the new token.
    window = jnp.concatenate([conv_state, x[None, :]], axis=0)  # [k, conv_dim]
    out = jnp.sum(window * weight.T.astype(window.dtype), axis=0)  # [conv_dim]
    out = out * jax.nn.sigmoid(out)  # SiLU
    return out, window[1:]


def kda_decode_step(
    state: jax.Array,  # [num_heads, d_k, d_v] fp32
    q: jax.Array,  # [num_heads, d_k]
    k: jax.Array,  # [num_heads, d_k]
    v: jax.Array,  # [num_heads, d_v]
    g: jax.Array,  # [num_heads, d_k]  activated log-decay
    beta: jax.Array,  # [num_heads]
) -> tuple[jax.Array, jax.Array]:
    """One KDA decode step for all heads. Returns `(output, new_state)`.

    All arithmetic is fp32; the recurrent state is the numerically sensitive
    part of the model and the HF reference keeps it in fp32 throughout.
    """
    q = l2_normalize(q) * QK_SCALE
    k = l2_normalize(k)
    v = v.astype(jnp.float32)
    beta = beta.astype(jnp.float32)

    # Decay along the key axis: state is [d_k, d_v], g is per key-channel.
    state = state * jnp.exp(g)[..., None]
    # pred[h, dv] = sum_dk k[h, dk] * state[h, dk, dv]
    #
    # `precision=HIGHEST` is load-bearing, not decoration. On TPU the default
    # lowering sends this einsum to the MXU as bf16 passes even though both
    # operands are fp32, which measures 2.65e-3 relative error against an fp64
    # oracle -- versus 1.9e-7 for the Pallas kernel, which does the reduction
    # as explicit fp32 elementwise work. This is the same class of bug the
    # z.ai blog documents, where Triton's `tl.dot` silently defaulted to TF32
    # and corrupted KDA through state merging.
    pred = jnp.einsum(
        "hk,hkv->hv",
        k,
        state,
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    resid = v - pred
    # Rank-1 delta-rule update.
    state = state + (beta[:, None] * k)[..., None] * resid[:, None, :]
    out = jnp.einsum(
        "hk,hkv->hv",
        q,
        state,
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    return out, state


def ragged_kda_decode_ref(
    qkv: jax.Array,  # [num_tokens, conv_dim]  pre-conv, post in_proj
    beta: jax.Array,  # [num_tokens, num_heads]
    g_raw: jax.Array,  # [num_tokens, num_heads, d_k]
    conv_state: jax.Array,  # [num_slots, conv_kernel_dim - 1, conv_dim]
    recurrent_state: jax.Array,  # [num_slots, num_heads, d_k, d_v]
    conv_weight: jax.Array,  # [conv_dim, conv_kernel_dim]
    a_log: jax.Array,  # [num_heads]
    dt_bias: jax.Array,  # [num_heads, d_k]
    state_indices: jax.Array,  # [num_seqs] int32, slot per sequence
    query_start_loc: jax.Array,  # [num_seqs + 1] int32
    num_heads: int = NUM_HEADS,
    head_dim: int = HEAD_DIM,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Ragged KDA decode over a batch of sequences.

    Follows the vLLM/tpu_inference convention used by the GDN kernel: tokens for
    all sequences are flattened into one ragged array delimited by
    `query_start_loc`, and each sequence's recurrent + conv state lives in a
    slot addressed by `state_indices`. Slot 0 is conventionally a null slot for
    padded or invalid requests.

    Returns `(output, new_conv_state, new_recurrent_state)` where output has
    shape `[num_tokens, num_heads * d_v]`.
    """
    num_tokens = qkv.shape[0]
    proj = num_heads * head_dim
    out = jnp.zeros((num_tokens, proj), dtype=jnp.float32)
    num_seqs = state_indices.shape[0]

    for s in range(num_seqs):
        slot = state_indices[s]
        start = query_start_loc[s]
        end = query_start_loc[s + 1]
        cs = conv_state[slot]
        rs = recurrent_state[slot].astype(jnp.float32)

        # Python-level loop over a traced length is not possible, so the caller is
        # expected to pass concrete bounds. This reference is used with small,
        # statically-known shapes in tests.
        for t in range(int(start), int(end)):
            conv_out, cs = causal_conv1d_update(cs, qkv[t], conv_weight)
            # Layout matches HF: [q | k | v] concatenated along the feature axis.
            q = conv_out[:proj].reshape(num_heads, head_dim)
            k = conv_out[proj : 2 * proj].reshape(num_heads, head_dim)
            v = conv_out[2 * proj :].reshape(num_heads, head_dim)
            g = activate_gate(g_raw[t], a_log, dt_bias)
            step_out, rs = kda_decode_step(rs, q, k, v, g, beta[t])
            out = out.at[t].set(step_out.reshape(proj))

        conv_state = conv_state.at[slot].set(cs)
        recurrent_state = recurrent_state.at[slot].set(rs)

    return out, conv_state, recurrent_state
