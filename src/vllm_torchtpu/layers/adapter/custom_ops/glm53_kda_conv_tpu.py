"""Paged causal depthwise conv update for KDA decode on TPU.

vLLM's KDA decode path calls `causal_conv1d_update(..., conv_state_indices=...)`
to advance the short convolution that precedes the recurrence. Every
implementation of that in the tree is unusable here:

  * `ops/causal_conv1d.py` is Triton (CUDA/ROCm).
  * `ops/cpu/causal_conv1d.py::causal_conv1d_update_cpu` calls a C++ CPU op.
  * `causal_conv1d_update_torch` in that same file is pure PyTorch and would
    run on TPU, but it has **no `conv_state_indices` parameter** -- it assumes
    `conv_state` is already gathered and row `i` belongs to sequence `i`. vLLM
    passes the whole paged cache plus an index vector, so it cannot be used
    directly.

Hence this module: the paged gather/scatter that the torch fallback is missing.

## Why this is a handful of elementwise ops and not a conv

For GLM 5.3 Flash `short_conv_kernel_size == 4`, and the cache holds
`width - 1 == 3` past timesteps. During decode exactly one new token arrives,
so the window is `[state | x]` -- precisely `width` wide. The depthwise
convolution over a window of exactly its own width is a weighted sum along the
last axis, with no sliding. `F.conv1d` would work but forces a grouped-conv
lowering for what is a multiply-and-reduce; on TPU this stays in the VPU and
fuses with the surrounding gather.
"""

from __future__ import annotations

import torch


def causal_conv1d_update_paged(
    x: torch.Tensor,  # [num_tokens, dim] one decode token per sequence
    conv_state: torch.Tensor,  # [num_slots, dim, state_len]
    weight: torch.Tensor,  # [dim, width]
    bias: torch.Tensor | None = None,
    activation: str | None = "silu",
    conv_state_indices: torch.Tensor | None = None,  # [num_tokens] int32
) -> torch.Tensor:
    """Advances the paged conv cache by one token and returns the conv output.

    Mirrors `causal_conv1d_update`'s decode contract: `conv_state` is updated in
    place at the rows named by `conv_state_indices`, and rows not named are left
    untouched.

    Returns `[num_tokens, dim]`.
    """
    if activation not in (None, "silu", "swish"):
        raise ValueError(f"unsupported activation {activation!r}")

    state_len = conv_state.shape[-1]
    width = weight.shape[-1]
    if width != state_len + 1:
        # The single-token fast path below is only a plain weighted sum because the
        # window is exactly `width` wide. Anything else needs a real sliding conv,
        # and silently computing the wrong thing would be far worse than failing.
        raise ValueError(
            f"expected width == state_len + 1 for single-token decode, got "
            f"width={width}, state_len={state_len}."
        )

    if conv_state_indices is None:
        gathered = conv_state
        idx = None
    else:
        # vLLM hands these over as int32 (`decode_conv_indices`). `index_select`
        # tolerates int32 but `index_copy_` demands int64, so the gather would
        # succeed and only the scatter would fail -- convert once, up front, rather
        # than at each use.
        idx = conv_state_indices.to(torch.long)
        gathered = conv_state.index_select(0, idx)

    # [num_tokens, dim, width]: the three cached timesteps plus the new one.
    window = torch.cat([gathered, x.unsqueeze(-1).to(gathered.dtype)], dim=-1)

    if idx is None:
        conv_state.copy_(window[..., -state_len:])
    else:
        conv_state.index_copy_(0, idx, window[..., -state_len:])

    # fp32 accumulation: the conv feeds q/k/v, and q/k are L2-normalized straight
    # afterwards, so error here is not damped before it enters the recurrence.
    out = (window.float() * weight.float().unsqueeze(0)).sum(-1)
    if bias is not None:
        out = out + bias.float().unsqueeze(0)
    if activation in ("silu", "swish"):
        out = torch.nn.functional.silu(out)
    return out.to(x.dtype)


def causal_conv1d_fn_paged(
    x: torch.Tensor,  # [dim, num_tokens]
    weight: torch.Tensor,  # [dim, width]
    bias: torch.Tensor | None = None,
    activation: str | None = "silu",
    conv_states: torch.Tensor | None = None,  # [num_slots, dim, state_len]
    has_initial_state: torch.Tensor | None = None,  # [num_seqs] bool
    cache_indices: torch.Tensor | None = None,  # [num_seqs] int32
    query_start_loc: torch.Tensor | None = None,  # [num_seqs + 1] int32
) -> torch.Tensor:
    """Ragged causal depthwise conv for prefill, updating the paged cache.

    Counterpart to `causal_conv1d_update_paged` for the prefill path. Matches the
    `causal_conv1d_fn` contract used by `KimiGatedDeltaNetAttention._forward`:
    channel-major `[dim, num_tokens]` in and out, sequences delimited by
    `query_start_loc`, and `conv_states` written in place at `cache_indices`.

    `has_initial_state[s]` distinguishes a prompt resuming from a previous
    scheduler step (prepend the cached window) from a fresh one (prepend zeros).
    Getting this backwards is silent: a fresh sequence would inherit whatever
    stale window happened to be in its slot, which only shows up as a subtly
    wrong first few tokens.
    """
    if activation not in (None, "silu", "swish"):
        raise ValueError(f"unsupported activation {activation!r}")
    if conv_states is None or cache_indices is None or query_start_loc is None:
        raise ValueError(
            "conv_states, cache_indices and query_start_loc are all required: the "
            "prefill conv is paged and ragged."
        )

    dim, _ = x.shape
    state_len = conv_states.shape[-1]
    width = weight.shape[-1]
    if width != state_len + 1:
        raise ValueError(
            f"expected width == state_len + 1, got width={width}, "
            f"state_len={state_len}."
        )

    bounds = query_start_loc.tolist()
    num_seqs = len(bounds) - 1
    idx = cache_indices.to(torch.long)
    out = torch.empty_like(x)

    for s in range(num_seqs):
        start, end = int(bounds[s]), int(bounds[s + 1])
        if end <= start:
            continue
        slot = idx[s]
        tokens = x[:, start:end]  # [dim, L]

        if has_initial_state is not None and bool(has_initial_state[s]):
            prefix = conv_states[slot]
        else:
            prefix = torch.zeros(
                (dim, state_len), dtype=conv_states.dtype, device=x.device
            )
        full = torch.cat([prefix.to(x.dtype), tokens], dim=-1)  # [dim, L + W - 1]

        # Causal: output token t sees inputs (t - W + 1 .. t), which is exactly a
        # valid (unpadded) conv over `full`.
        conv = torch.nn.functional.conv1d(
            full.float().unsqueeze(0),
            weight.float().unsqueeze(1),
            bias.float() if bias is not None else None,
            padding=0,
            groups=dim,
        )[0]
        if activation in ("silu", "swish"):
            conv = torch.nn.functional.silu(conv)
        out[:, start:end] = conv.to(x.dtype)

        conv_states[slot] = full[:, -state_len:].to(conv_states.dtype)

    return out
