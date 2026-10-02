"""TorchTPU adapter: GLM 5.3 Flash KDA decode as a native PyTorch op.

This is the bridge between the Pallas kernel (JAX) and the PyTorch serving
stack. `torch_tpu._internal.pallas.jax_op` registers a JAX function as a
`torch.library.custom_op`, so the kernel becomes callable from ordinary
PyTorch code and traceable by `torch.compile(backend="tpu")`.

## Why donation needs a wrapper

The recurrent state is by far the largest tensor in the problem -- 4 MiB per
sequence per layer, 136 MiB per sequence across all 34 KDA layers -- so
copying it once per decode step would cost more than the kernel itself. The
kernel therefore updates it in place via `input_output_aliases`, and the op
declares the state as donated.

Donation in `jax_op` leaves the caller's tensor *invalid* rather than mutated:
JAX values are immutable, so the buffer is consumed and a new tensor is
returned. That is a sharp edge for a KV-cache-style tensor that serving code
naturally holds a long-lived reference to. `kda_decode` below therefore
`set_`s the caller's tensor to the returned storage, so the cache object the
caller already owns stays valid and no copy occurs.

## State layout

`recurrent_state` is `[num_slots, num_heads, d_v, d_k]` -- transposed relative
to the HuggingFace reference's `[..., d_k, d_v]`. This is not cosmetic: it is
what removes every masked-iota transpose from the kernel and is worth 8.9x.
Since `d_k == d_v == 128` for GLM 5.3 Flash the shape tuple is unchanged, so
the two layouts are indistinguishable by shape alone -- prefill must write the
state in this order. See `kda_decode_kernel`'s module docstring.
"""

from __future__ import annotations

import collections

import torch
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.glm53_kda import kda_decode_kernel as kern
from vllm_torchtpu.kernels.glm53_kda import kda_decode_ref as ref

_STORAGE_KEEPALIVE: collections.deque[torch.UntypedStorage] = collections.deque(
    maxlen=2
)


# Argument 7 is `recurrent_state`. Donating it is what lets the kernel's
# `input_output_aliases` actually avoid a copy end to end; without it the
# bridge would hand Pallas a fresh buffer every call.
kda_decode_op = pallas.jax_op(
    "glm53_kda::decode",
    kern.kda_decode_jax,
    donate_argnums=(7,),
)


def kda_decode(
    q: torch.Tensor,  # [num_seqs, num_heads, d_k]
    k: torch.Tensor,  # [num_seqs, num_heads, d_k]
    v: torch.Tensor,  # [num_seqs, num_heads, d_v]
    g_raw: torch.Tensor,  # [num_seqs, num_heads, d_k]  pre-activation gate
    beta: torch.Tensor,  # [num_seqs, num_heads]
    a_log: torch.Tensor,  # [num_heads]
    dt_bias: torch.Tensor,  # [num_heads, d_k]
    recurrent_state: torch.Tensor,  # [num_slots, num_heads, d_v, d_k]
    state_indices: torch.Tensor,  # [num_seqs] int32
    lower_bound: float = ref.LOWER_BOUND,
    qk_scale: float = ref.QK_SCALE,
    h_block: int = 0,
    null_slot: int = -1,
) -> torch.Tensor:
    """One KDA decode step per sequence, with the state updated in place.

    `recurrent_state` may be the serving cache itself: sequence `s` reads and
    writes slot `state_indices[s]`, and with `null_slot >= 0` entries pointing
    at that slot are padding (state untouched, output zeros).
    """
    out, new_state = kda_decode_op(
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
        h_block,
        null_slot,
    )
    _STORAGE_KEEPALIVE.append(recurrent_state.untyped_storage())
    recurrent_state.set_(new_state)
    return out


def activate_gate(
    g_raw: torch.Tensor,  # [..., num_heads, d_k]
    a_log: torch.Tensor,  # [num_heads]
    dt_bias: torch.Tensor,  # [num_heads, d_k]
    lower_bound: float = ref.LOWER_BOUND,
) -> torch.Tensor:
    """GLM's bounded safe-gate, in PyTorch. Returns log-decay in [lower_bound, 0]."""
    g = g_raw.float() + dt_bias.float()
    a = torch.exp(a_log.float()).unsqueeze(-1)
    return lower_bound * torch.sigmoid(a * g)


kda_prefill_chunk_op = pallas.jax_op(
    "glm53_kda::prefill_chunk",
    kern.kda_prefill_chunk_jax,
    donate_argnums=(7,),
)


def kda_prefill_chunk(
    q: torch.Tensor,  # [num_seqs, chunk_len, num_heads, d_k]
    k: torch.Tensor,
    v: torch.Tensor,
    g_raw: torch.Tensor,  # [num_seqs, chunk_len, num_heads, d_k]
    beta: torch.Tensor,  # [num_seqs, chunk_len, num_heads]
    a_log: torch.Tensor,  # [num_heads]
    dt_bias: torch.Tensor,  # [num_heads, d_k]
    recurrent_state: torch.Tensor,  # [num_seqs, num_heads, d_v, d_k]
    valid_lens: torch.Tensor,  # [num_seqs] int32
    lower_bound: float = ref.LOWER_BOUND,
    qk_scale: float = ref.QK_SCALE,
    h_block: int = 0,
) -> torch.Tensor:
    """Runs one fused VMEM chunk of KDA prefill, updating `recurrent_state` in place."""
    out, new_state = kda_prefill_chunk_op(
        q,
        k,
        v,
        g_raw,
        beta,
        a_log,
        dt_bias,
        recurrent_state,
        valid_lens,
        lower_bound,
        qk_scale,
        h_block,
    )
    _STORAGE_KEEPALIVE.append(recurrent_state.untyped_storage())
    recurrent_state.set_(new_state)
    return out
