"""vLLM-shaped KDA prefill for TPU.

Implements `chunk_kda_with_fused_gate`, the second of the two kernels vLLM's
KDA layer dispatches to (the other, `fused_recurrent_kda_packed_decode`, lives
in `kda_vllm_decode`). Between them they cover the non-speculative path of
`KimiGatedDeltaNetAttention._forward`.

## This is a correctness placeholder, not the eventual kernel

The name upstream is `chunk_...` because the CUDA implementation is *chunked*:
it reformulates the recurrence so each chunk's contribution is a matmul, which
is what makes prefill fast on a tensor-core machine. This implementation is a
plain sequential scan -- one kernel launch per token -- and is therefore
**O(prompt length) launches**, roughly 0.1 ms each per layer.

That is deliberate and it is stated here rather than buried:

  * It is *correct*, and correctness is testable now against the float64 GLM
    reference. A chunked formulation is a much larger change whose numerics
    would need this exact scan as its oracle anyway.
  * Prefill throughput is where most of the blog's optimisation work lives, so
    the chunked version deserves to be designed against measurements rather
    than written speculatively alongside the bring-up.

Do not benchmark end-to-end prefill against this and conclude anything about
TPU. It exists to make the model *run*.

## Shapes

Mirrors the call site in `kimi_gdn_linear_attn.py`: `q/k/v/raw_g` are
`[1, num_tokens, num_heads, head_dim]`, `raw_beta` is
`[1, num_tokens, num_heads]`, and `cu_seqlens` gives the ragged sequence
boundaries. `initial_state` is already *gathered* by the caller
(`gather_initial_states`), so it is `[num_seqs, num_heads, d_v, d_k]` -- one
row per sequence in this batch, not the full paged cache.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from vllm_torchtpu.layers.adapter.custom_ops import glm53_kda_decode_torch as kdt

# With `kda_prefill_chunk_jax` tiling `num_heads` into 8-head blocks along a 2D
# grid `(num_seqs, num_heads // 8)`, `CHUNK_LEN = 64` uses only 4.51 MiB of
# scoped VMEM per block (well inside TPU v7x's 32.00 MiB limit) while cutting
# HBM state traffic and launch overhead by 64x.
CHUNK_LEN = 64
_VALID_T_CACHE: dict[tuple[int, int, str], torch.Tensor] = {}


def select_prefill_chunk_len(num_seqs: int, total_tokens: int) -> int:
    """Selects optimal VMEM chunk length `chunk_len` for `(num_seqs, total_tokens)`."""
    if num_seqs <= 0 or total_tokens <= 0:
        return CHUNK_LEN
    if total_tokens % num_seqs == 0:
        seq_len = total_tokens // num_seqs
        if seq_len % 256 == 0 and total_tokens <= 8192:
            return 256
        if seq_len % 128 == 0:
            return 128
        if seq_len % 64 == 0:
            return 64
        if seq_len % 32 == 0:
            return 32
        if seq_len % 16 == 0:
            return 16
    return CHUNK_LEN


def _get_cached_valid_t(
    num_seqs: int, chunk_len: int, device: torch.device
) -> torch.Tensor:
    key = (num_seqs, chunk_len, str(device))
    t = _VALID_T_CACHE.get(key)
    if t is None or t.device != device:
        t = torch.full((num_seqs,), chunk_len, dtype=torch.int32, device=device)
        _VALID_T_CACHE[key] = t
    return t


def chunk_kda_with_fused_gate(
    q: torch.Tensor,  # [1, num_tokens, num_heads, head_dim]
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,  # [1, num_tokens, num_heads, head_dim]
    raw_beta: torch.Tensor,  # [1, num_tokens, num_heads]
    A_log: torch.Tensor,  # [num_heads]  pylint: disable=invalid-name
    g_bias: torch.Tensor,  # [num_heads * head_dim]
    lower_bound: float | None,
    initial_state: torch.Tensor,  # [num_seqs, num_heads, d_v, d_k]
    output_final_state: bool = True,
    use_qk_l2norm_in_kernel: bool = True,
    cu_seqlens: torch.Tensor | None = None,  # [num_seqs + 1] int32
    chunk_len: int | None = None,
    h_block: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ragged-batch KDA prefill via fused VMEM chunked Pallas kernel."""
    if lower_bound is None:
        raise ValueError(
            "KDA on TPU implements GLM's bounded safe gate, which requires "
            "`lower_bound` (config `linear_lower_bound`, e.g. -5.0). Got None, "
            "which would select the unbounded softplus gate."
        )
    if not use_qk_l2norm_in_kernel:
        raise ValueError(
            "use_qk_l2norm_in_kernel=False is not supported: the TPU kernel "
            "always applies the q/k L2 normalization internally."
        )
    if cu_seqlens is None:
        raise ValueError("cu_seqlens is required to delimit the ragged batch.")

    num_heads, head_dim = q.shape[-2], q.shape[-1]
    num_seqs = initial_state.shape[0]
    if cu_seqlens.numel() != num_seqs + 1:
        raise ValueError(
            f"cu_seqlens has {cu_seqlens.numel()} entries, which delimits "
            f"{cu_seqlens.numel() - 1} sequences, but initial_state has "
            f"{num_seqs} rows."
        )

    q0, k0, v0 = q[0], k[0], v[0]
    g0 = raw_g[0]
    beta0 = torch.sigmoid(raw_beta[0].float())
    a_log = A_log.float()
    dt_bias = g_bias.reshape(num_heads, head_dim).float()

    state = initial_state
    total_tokens = q0.shape[0]
    bounds = [int(x) for x in cu_seqlens.tolist()]
    lens = [bounds[s + 1] - bounds[s] for s in range(num_seqs)]
    # The fast path reshapes the batch as `num_seqs` equal-length sequences. A
    # token total that merely divides evenly (e.g. lengths 100 + 28) is not
    # enough: it would mix tokens of different sequences into one chunk row.
    uniform = (
        num_seqs > 0
        and bounds[0] == 0
        and bounds[-1] == total_tokens
        and all(n == lens[0] for n in lens)
    )
    if chunk_len is None or chunk_len <= 0:
        chunk_len = (
            select_prefill_chunk_len(num_seqs, total_tokens) if uniform else CHUNK_LEN
        )

    if uniform and total_tokens > 0 and total_tokens % (num_seqs * chunk_len) == 0:
        seq_len = total_tokens // num_seqs
        num_chunks = seq_len // chunk_len
        valid_t = _get_cached_valid_t(num_seqs, chunk_len, q.device)
        if num_chunks == 1:
            chunk_out = kdt.kda_prefill_chunk(
                q0.view(num_seqs, chunk_len, num_heads, head_dim),
                k0.view(num_seqs, chunk_len, num_heads, head_dim),
                v0.view(num_seqs, chunk_len, num_heads, head_dim),
                g0.view(num_seqs, chunk_len, num_heads, head_dim),
                beta0.view(num_seqs, chunk_len, num_heads),
                a_log,
                dt_bias,
                state,
                valid_t,
                lower_bound=float(lower_bound),
                qk_scale=head_dim**-0.5,
                h_block=h_block,
            )
            out = chunk_out.view(1, total_tokens, num_heads, head_dim).to(q.dtype)
            return out, state if output_final_state else None

        q_seq = q0.view(num_seqs, num_chunks, chunk_len, num_heads, head_dim)
        k_seq = k0.view(num_seqs, num_chunks, chunk_len, num_heads, head_dim)
        v_seq = v0.view(num_seqs, num_chunks, chunk_len, num_heads, head_dim)
        g_seq = g0.view(num_seqs, num_chunks, chunk_len, num_heads, head_dim)
        b_seq = beta0.view(num_seqs, num_chunks, chunk_len, num_heads)
        out_chunks = []
        for c in range(num_chunks):
            c_out = kdt.kda_prefill_chunk(
                q_seq[:, c].contiguous(),
                k_seq[:, c].contiguous(),
                v_seq[:, c].contiguous(),
                g_seq[:, c].contiguous(),
                b_seq[:, c].contiguous(),
                a_log,
                dt_bias,
                state,
                valid_t,
                lower_bound=float(lower_bound),
                qk_scale=head_dim**-0.5,
                h_block=h_block,
            )
            out_chunks.append(c_out.to(q.dtype))
        out = torch.stack(out_chunks, dim=1).view(1, total_tokens, num_heads, head_dim)
        return out, state if output_final_state else None

    out = torch.empty_like(q0)
    max_len = max(lens) if lens else 0

    for offset in range(0, max_len, chunk_len):
        q_chunk = torch.zeros(
            (num_seqs, chunk_len, num_heads, head_dim),
            dtype=q0.dtype,
            device=q.device,
        )
        k_chunk = torch.zeros_like(q_chunk)
        v_chunk = torch.zeros_like(q_chunk)
        g_chunk = torch.zeros_like(q_chunk)
        beta_chunk = torch.zeros(
            (num_seqs, chunk_len, num_heads),
            dtype=torch.float32,
            device=q.device,
        )
        valid = []
        for s in range(num_seqs):
            rem = max(0, min(chunk_len, lens[s] - offset))
            valid.append(rem)
            if rem > 0:
                lo = bounds[s] + offset
                hi = lo + rem
                q_chunk[s, :rem] = q0[lo:hi]
                k_chunk[s, :rem] = k0[lo:hi]
                v_chunk[s, :rem] = v0[lo:hi]
                g_chunk[s, :rem] = g0[lo:hi]
                beta_chunk[s, :rem] = beta0[lo:hi]

        valid_t = torch.tensor(valid, dtype=torch.int32, device=q.device)
        chunk_out = kdt.kda_prefill_chunk(
            q_chunk,
            k_chunk,
            v_chunk,
            g_chunk,
            beta_chunk,
            a_log,
            dt_bias,
            state,
            valid_t,
            lower_bound=float(lower_bound),
            qk_scale=head_dim**-0.5,
            h_block=h_block,
        )
        for s in range(num_seqs):
            rem = valid[s]
            if rem > 0:
                lo = bounds[s] + offset
                out[lo : lo + rem] = chunk_out[s, :rem].to(out.dtype)

    return out.unsqueeze(0), state if output_final_state else None


def chunk_kda_prefill_gathered(
    qkv_rows: torch.Tensor,  # [num_rows, 3 * num_heads * head_dim]
    g_rows: torch.Tensor,  # [num_rows, num_heads, head_dim] raw gate
    beta_rows: torch.Tensor,  # [num_rows, num_heads] raw (pre-sigmoid) beta
    A_log: torch.Tensor,  # [num_heads]  pylint: disable=invalid-name
    g_bias: torch.Tensor,  # [num_heads * head_dim]
    lower_bound: float | None,
    recurrent_state: torch.Tensor,  # [num_slots, num_heads, d_v, d_k]
    slots: torch.Tensor,  # [num_seqs] int64
    has_initial_state: torch.Tensor,  # [num_seqs] bool
    chunk_rows: Sequence[torch.Tensor],  # per chunk: [num_seqs * chunk_len]
    chunk_valid: Sequence[torch.Tensor],  # per chunk: [num_seqs] int32
    chunk_len: int,
    num_heads: int,
    head_dim: int,
    out: torch.Tensor,  # [num_rows, num_heads, head_dim]
    num_prefills: int | None = None,
    num_tokens: int | None = None,
    prefill_start: int = 0,
) -> None:
    """Ragged KDA prefill driven by a precomputed row-gather plan.

    Same kernel and math as `chunk_kda_with_fused_gate`, but the caller has
    already laid every chunk out as row indices into the token tensors, so each
    chunk costs a few gathers and a single post-loop write instead of per-chunk
    scatter copies, and every shape depends only on `(num_seqs, chunk_len)`.

    `chunk_rows[c][s * chunk_len + j]` is the row holding token `c * chunk_len +
    j` of sequence `s`, or a row of zeros past the sequence's end; the kernel
    masks those positions with `chunk_valid[c]`, and their outputs land on rows
    the caller discards. Final states are scattered back to `recurrent_state` at
    `slots`; sequences with `has_initial_state == False` start from zeros rather
    than from whatever a reused slot last held.
    """
    if lower_bound is None:
        raise ValueError(
            "KDA on TPU implements GLM's bounded safe gate, which requires "
            "`lower_bound` (config `linear_lower_bound`, e.g. -5.0). Got None."
        )
    num_seqs = slots.numel()
    a_log = A_log.float()
    dt_bias = g_bias.reshape(num_heads, head_dim).float()

    gathered = recurrent_state.index_select(0, slots)
    state = torch.where(
        has_initial_state.view(-1, 1, 1, 1),
        gathered,
        torch.zeros_like(gathered),
    ).contiguous()
    shape = (num_seqs, chunk_len, num_heads, head_dim)
    # Reshape the projection tensor prior to chunk selection so slices along the
    # sequence dimension remain physically contiguous in memory (eliminates
    # per-chunk .split(proj, dim=-1).contiguous() layout transpositions).
    qkv_3d = qkv_rows.reshape(-1, 3, num_heads, head_dim)
    q_rows = qkv_3d[:, 0].contiguous()
    k_rows = qkv_3d[:, 1].contiguous()
    v_rows = qkv_3d[:, 2].contiguous()
    g_rows_contig = g_rows.reshape(-1, num_heads, head_dim).contiguous()

    chunk_outs: list[torch.Tensor] = []
    for rows, valid in zip(chunk_rows, chunk_valid):
        q = q_rows.index_select(0, rows).reshape(shape)
        k = k_rows.index_select(0, rows).reshape(shape)
        v = v_rows.index_select(0, rows).reshape(shape)
        g = g_rows_contig.index_select(0, rows).reshape(shape)
        beta = torch.sigmoid(beta_rows.index_select(0, rows).float()).reshape(
            num_seqs, chunk_len, num_heads
        )
        chunk_out = kdt.kda_prefill_chunk(
            q,
            k,
            v,
            g,
            beta,
            a_log,
            dt_bias,
            state,
            valid,
            lower_bound=float(lower_bound),
            qk_scale=head_dim**-0.5,
        )
        chunk_outs.append(chunk_out)

    if chunk_outs:
        # Concatenate along token axis: [s_pad, total_chunk_tokens, H, D]
        full_prefill_out = (
            chunk_outs[0] if len(chunk_outs) == 1 else torch.cat(chunk_outs, dim=1)
        ).to(out.dtype)
        if (
            (num_prefills == 1 or num_seqs == 1)
            and num_tokens is not None
            and prefill_start == 0
        ):
            # Contiguous slice write into output: zero scatter overhead
            out[:num_tokens] = full_prefill_out[0, :num_tokens]
        elif (num_prefills == 1 or num_seqs == 1) and num_tokens is not None:
            out[prefill_start : prefill_start + num_tokens] = full_prefill_out[
                0, :num_tokens
            ]
        else:
            # Single batched scatter after the chunk loop
            flattened_active_indices = torch.cat(
                [r.view(num_seqs, chunk_len) for r in chunk_rows], dim=1
            ).reshape(-1)
            out.index_copy_(
                0,
                flattened_active_indices,
                full_prefill_out.reshape(-1, num_heads, head_dim),
            )
    recurrent_state.index_copy_(0, slots, state)


def chunk_kda_with_fused_gate_token_by_token(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,  # pylint: disable=invalid-name
    g_bias: torch.Tensor,
    lower_bound: float,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Baseline 1-token-per-launch prefill retained for A/B benchmarking."""
    num_heads, head_dim = q.shape[-2], q.shape[-1]
    num_seqs = initial_state.shape[0]
    q0, k0, v0 = q[0], k[0], v[0]
    g0, beta0 = raw_g[0], raw_beta[0]
    a_log = A_log.float()
    dt_bias = g_bias.reshape(num_heads, head_dim).float()
    state = initial_state
    out = torch.empty_like(q0)
    bounds = cu_seqlens.tolist()
    for s in range(num_seqs):
        start, end = int(bounds[s]), int(bounds[s + 1])
        slot = torch.tensor([s], dtype=torch.int32, device=q.device)
        for t in range(start, end):
            step_out = kdt.kda_decode(
                q0[t : t + 1].contiguous(),
                k0[t : t + 1].contiguous(),
                v0[t : t + 1].contiguous(),
                g0[t : t + 1].contiguous(),
                torch.sigmoid(beta0[t : t + 1].float()),
                a_log,
                dt_bias,
                state,
                slot,
                lower_bound=float(lower_bound),
                qk_scale=head_dim**-0.5,
            )
            out[t] = step_out[0]
    return out.unsqueeze(0), state
