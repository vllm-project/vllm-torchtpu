"""vLLM-shaped KDA decode for TPU.

vLLM's KDA layer (`KimiGatedDeltaNetAttention`) dispatches its pure-decode
batches to `fused_recurrent_kda_packed_decode`, importing a Triton
implementation from `vllm.models.kimi_k3.{nvidia,amd}.ops` inside `_forward`.
This module provides the same callable backed by the Pallas kernel, so a TPU
override of that layer can swap the kernel without touching projections,
weight loading, `o_norm`, or the cache plumbing.

## Why this signature and not the kernel's

`kda_decode_torch.kda_decode` takes pre-split, pre-activated tensors. vLLM
hands the kernel a *packed* `mixed_qkv` and *raw* gate/beta projections, and
expects the gate activation to happen inside. Matching vLLM's signature here
rather than asking the layer to adapt is what keeps the override to a single
method.

## The one contract that matters

`initial_state` is `[num_slots, num_heads, d_v, d_k]`. That is vLLM's own
layout -- `fla/ops/kda.py` allocates the final state as
`q.new_empty(T, HV, V, K)` -- and it is also what the Pallas kernel needs to
keep `d_k` on the lane axis. Since `d_k == d_v == 128` for GLM 5.3 Flash the
two possible orderings are indistinguishable by shape, so this is asserted in
the tests against a multi-step reference rather than left to a comment.

## In place, not gathered

The kernel indexes the state cache through `state_indices` itself, so the
whole cache goes in and only the step's slots are read and written. Gathering
the active slots into a dense batch and scattering them back instead costs
three extra full passes over the active state (gather, copy, scatter; ~3.6
ms per step at batch 64 across the 34 KDA layers) and several extra eager
dispatches per layer.
"""

from __future__ import annotations

import torch

from vllm_torchtpu.kernels.glm53_kda import kda_decode_ref as ref
from vllm_torchtpu.layers.adapter.custom_ops import glm53_kda_decode_torch as kdt


def can_update_in_place(state: torch.Tensor) -> bool:
    """Whether the decode kernel may update `state` in place.

    The kernel donates the state buffer and `kda_decode` re-points the tensor
    at the result with `set_`. That is only sound when the tensor is the sole,
    whole view of its storage -- which the TPU runner's per-layer mamba caches
    are. For a view, `set_` would re-point the view alone and leave its base
    (and every sibling view) on the donated buffer.
    """
    return (
        state._base is None  # pylint: disable=protected-access
        and state.is_contiguous()
        and state.storage_offset() == 0
        and state.untyped_storage().nbytes() == state.numel() * state.element_size()
    )


def fused_recurrent_kda_packed_decode(
    mixed_qkv: torch.Tensor,  # [num_tokens, 3 * num_heads * head_dim]
    raw_g: torch.Tensor,  # [1, num_tokens, num_heads, head_dim]
    raw_beta: torch.Tensor,  # [1, num_tokens, num_heads]
    A_log: torch.Tensor,  # [num_heads]  pylint: disable=invalid-name
    dt_bias: torch.Tensor,  # [num_heads * head_dim]
    lower_bound: float | None,
    initial_state: torch.Tensor,  # [num_slots, num_heads, d_v, d_k]
    state_indices: torch.Tensor,  # [num_tokens] int32
    null_slot: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One decode step per token, state updated in place.

    Drop-in for vLLM's `fused_recurrent_kda_packed_decode`. Each token in a
    pure-decode batch is its own sequence, so there is no ragged structure to
    carry and `cu_seqlens` is implicit.

    The kernel reads and writes `initial_state[state_indices[t]]` directly, so
    the cache is neither gathered nor scattered. `null_slot >= 0` names the
    cache's padding slot: tokens mapped to it leave the slot untouched and
    produce zeros. If `initial_state` is a view (see `can_update_in_place`) the
    step falls back to gather -> kernel -> scatter, where `null_slot` is moot.

    Returns `(core_attn_out, state)` with `core_attn_out` shaped
    `[1, num_tokens, num_heads, head_dim]`, matching what the caller writes into
    `core_attn_out` before `o_norm`.
    """
    if lower_bound is None:
        # The unbounded softplus branch is a different recurrence, not a fallback:
        # on GLM's weights it collapses median retention from ~0.98 to ~0.07. GLM
        # 5.3 Flash always sets `linear_lower_bound`, so a missing bound means the
        # config was not plumbed through rather than that softplus was intended.
        raise ValueError(
            "KDA on TPU implements GLM's bounded safe gate, which requires "
            "`lower_bound` (config `linear_lower_bound`, e.g. -5.0). Got None, "
            "which would select the unbounded softplus gate."
        )

    num_tokens = mixed_qkv.shape[0]
    num_heads, head_dim = raw_g.shape[-2], raw_g.shape[-1]
    proj = num_heads * head_dim
    if state_indices.numel() < num_tokens:
        raise ValueError(
            f"KDA decode got {num_tokens} tokens but only "
            f"{state_indices.numel()} state indices."
        )

    # vLLM packs the conv output as [q | k | v], each `local_projection_size`
    # wide; see `KimiGatedDeltaNetAttention._forward`'s prefill split.
    q, k, v = (
        mixed_qkv[..., :proj],
        mixed_qkv[..., proj : 2 * proj],
        mixed_qkv[..., 2 * proj :],
    )
    shape = (num_tokens, num_heads, head_dim)
    q, k, v = q.reshape(shape), k.reshape(shape), v.reshape(shape)

    # `raw_beta` is pre-activation; the Triton path applies sigmoid internally
    # and so must this one.
    beta = torch.sigmoid(raw_beta.reshape(num_tokens, num_heads).float())

    kernel_args = (
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        raw_g.reshape(shape).contiguous(),
        beta,
        A_log.float(),
        dt_bias.reshape(num_heads, head_dim).float(),
    )
    kernel_kwargs = dict(lower_bound=float(lower_bound), qk_scale=head_dim**-0.5)
    if can_update_in_place(initial_state):
        if state_indices.dtype != torch.int32:
            state_indices = state_indices.to(torch.int32)
        out = kdt.kda_decode(
            *kernel_args,
            initial_state,
            state_indices,
            null_slot=null_slot,
            **kernel_kwargs,
        )
    else:
        idx = state_indices[:num_tokens].to(torch.long)
        active_state = initial_state.index_select(0, idx).contiguous()
        dense_indices = torch.arange(
            num_tokens, device=mixed_qkv.device, dtype=torch.int32
        )
        out = kdt.kda_decode(*kernel_args, active_state, dense_indices, **kernel_kwargs)
        initial_state.index_copy_(0, idx, active_state)
    # q/k L2 normalization and the q scale happen inside the kernel, matching
    # vLLM's `use_qk_l2norm_in_kernel=True`.
    return out.reshape(1, num_tokens, num_heads, head_dim), initial_state


def activate_gate(
    raw_g: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float = ref.LOWER_BOUND,
) -> torch.Tensor:
    """GLM's bounded safe gate, for references and tests. See `kda_decode_torch`."""
    return kdt.activate_gate(raw_g, a_log, dt_bias, lower_bound)
