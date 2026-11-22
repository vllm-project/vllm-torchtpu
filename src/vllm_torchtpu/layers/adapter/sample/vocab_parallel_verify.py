# SPDX-License-Identifier: Apache-2.0
"""Non-greedy speculative verify on this rank's vocabulary shard.

With ``TPU_SAMPLER_VOCAB_PARALLEL_TARGET`` the target LM head returns each
rank's [rows, V/tp] shard of the logits and every full-vocabulary pass of the
verify runs on that shard. Only per-row values cross ranks:

1. the softmax statistics, all-gathered as f32[rows, 4]: the local max and
   the local sum of exp(z - local max) of the temperature-scaled row, and
   the raw logits' (max, global id) pair for the greedy fallback;
2. the top-p threshold search, one all-reduce of f32[2**radix_bits - 1,
   rows] per round (`top_k_top_p.vocab_parallel_topp_threshold`);
3. under real acceptance, one all-reduce of f32[2, draft rows]: the target
   probability of each draft (from the rank that owns its column) and the
   mass kept by top-p;
4. the draw, an exponential race on every shard whose (value, id) pairs go
   through the f32 pair exchange of `spec_decode.utils.vocab_parallel_argmax`.

Recovered rows (one a draft slot) and bonus rows (one a request) run as one
batch. Every rank draws the full-vocabulary uniforms of the default path from
the shared generator and slices its own columns, so each global column has its
own uniform, the same one on every rank, and the race over the union of the
shards is the full-vocabulary race. The accept decisions and the output
assembly are the rejection sampler's (`rejection_sampler.accepted_drafts`,
`rejection_sampler.assemble_rejection_output`).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from vllm_torchtpu.layers.adapter.sample.rejection_sampler import (
    TEMPERATURE_EPS,
    accepted_drafts,
    assemble_rejection_output,
)
from vllm_torchtpu.layers.adapter.sample.top_k_top_p import (
    vocab_parallel_topp_threshold,
)
from vllm_torchtpu.spec_decode.utils import (
    VocabParallelArgmax,
    vocab_parallel_max_argmax,
)


def _find_dynamic_slice():
    """torch_tpu's dynamic slice op (tpu::dynamic_slice, tpu::_dynamic_slice
    from torch_tpu 0.2.0), or None without torch_tpu."""
    for name in ("dynamic_slice", "_dynamic_slice"):
        try:
            return getattr(torch.ops.tpu, name)
        except (AttributeError, RuntimeError):
            continue
    return None


def register_dynamic_slice_fake() -> None:
    """Gives torch_tpu's dynamic slice op the fake kernel Dynamo traces with
    (torch_tpu registers only its TPU kernel) and resolves `_DYNAMIC_SLICE`.

    A no-op without torch_tpu and when a fake kernel is already registered."""
    global _DYNAMIC_SLICE
    _DYNAMIC_SLICE = _find_dynamic_slice()
    if _DYNAMIC_SLICE is None:
        return
    try:

        @torch.library.register_fake(_DYNAMIC_SLICE._qualified_op_name)
        def _(input, start_indices, slice_sizes):
            return input.new_empty(list(slice_sizes))

    except RuntimeError:
        pass


_DYNAMIC_SLICE = None
register_dynamic_slice_fake()


def _columns_from(u: torch.Tensor, start: torch.Tensor, width: int) -> torch.Tensor:
    """u[:, start:start + width] for a 0-d or one-element int ``start``: a
    dynamic slice of the last dimension (a gather of those columns when
    torch_tpu has no dynamic slice op)."""
    if u.device.type == "cpu":
        return u.narrow(1, int(start), width)
    start = start.to(torch.int32).reshape(())
    if _DYNAMIC_SLICE is None:
        columns = start + torch.arange(width, dtype=torch.int32, device=u.device)
        return torch.index_select(u, 1, columns)
    return _DYNAMIC_SLICE(u, [torch.zeros_like(start), start], [u.shape[0], width])


def shard_columns(
    u: torch.Tensor,
    vocab_start: torch.Tensor,
    width: int,
    tp_size: int,
    minor_slice: bool = False,
) -> torch.Tensor:
    """This rank's columns [vocab_start, vocab_start + width) of a [rows, V]
    full-vocabulary draw, V <= tp_size * width (columns past V, the padding
    of the last shards, come back as 0.5).

    ``vocab_start`` is a device int32 scalar (`utils.with_device_vocab_start`),
    so the program is the same on every rank. ``minor_slice``
    (TPU_SAMPLER_VP_UNIFORM_SLICE) takes the columns with a dynamic slice of
    the last dimension instead of the [rows, tp_size, width] view, which XLA
    relayouts when width is not a multiple of 128; same values."""
    rows, vocab = u.shape
    if vocab < tp_size * width:
        u = torch.nn.functional.pad(u, (0, tp_size * width - vocab), value=0.5)
    if minor_slice:
        return _columns_from(u, vocab_start, width)
    rank = (vocab_start // width).to(torch.int64).view(1)
    return torch.index_select(u.view(rows, tp_size, width), 1, rank).view(rows, width)


def race_on_shard(
    probs: torch.Tensor,
    kept: torch.Tensor,
    excluded: torch.Tensor,
    u: torch.Tensor,
    vocab_start: torch.Tensor,
    num_org_local: int,
    tp_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The exponential race of the rejection sampler's recovered token,
    p / E with E = -log(u), over the kept columns of every shard with the
    ``excluded`` id (-1: none) left out. ``probs``/``kept``/``u`` are this
    rank's [rows, V/tp] columns. Returns the winning value (f32 [rows], 0 when
    no column has mass) and its global id (int32 [rows]), ties to the lowest
    id."""
    f32 = torch.float32
    tiny = torch.finfo(f32).tiny
    width = probs.shape[-1]
    u = torch.clamp(u, min=tiny, max=1.0 - torch.finfo(f32).eps)
    exp_noise = -torch.log(u)
    column = torch.arange(width, device=probs.device).unsqueeze(0)
    not_excluded = column != (excluded - vocab_start).unsqueeze(-1)
    residual = torch.where(kept & not_excluded, probs, 0.0)
    info = VocabParallelArgmax(
        org_vocab_start=vocab_start, num_org_local=num_org_local, tp_size=tp_size
    )
    return vocab_parallel_max_argmax(residual / (exp_noise + tiny), info)


def sample_on_shard(
    logits: torch.Tensor,
    divisor: torch.Tensor,
    top_p: torch.Tensor,
    excluded: torch.Tensor,
    is_greedy: torch.Tensor,
    uniforms: Sequence[torch.Tensor],
    vocab_start: torch.Tensor,
    num_org_local: int,
    tp_size: int,
    radix_bits: int,
    draft_rows: int | None = None,
    uniforms_on_shard: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """One token a row from this rank's [rows, V/tp] logit shard; the same
    tokens on every rank.

    Each row samples p = softmax(logits / divisor) restricted to its top-p set
    with its ``excluded`` id (-1: none) left out, by the exponential race on
    ``uniforms``: full-vocabulary draws whose rows, stacked, are the logits
    rows, of which this rank uses its own columns. A greedy row, and a row
    with no mass left, takes the raw logits' argmax.

    With ``draft_rows``, the first that many rows are draft slots whose
    ``excluded`` id is their draft, and the second return value is the
    target probability of each draft after top-p, renormalised over the kept
    set (0 for a placeholder draft or one outside the set).

    With ``uniforms_on_shard`` the ``uniforms`` are already this rank's
    [rows, V/tp] columns.
    """
    from vllm.distributed import (
        tensor_model_parallel_all_gather,
        tensor_model_parallel_all_reduce,
    )

    f32 = torch.float32
    width = logits.shape[-1]
    device = logits.device
    scaled = logits.to(f32) / divisor.unsqueeze(-1)
    raw = logits
    if num_org_local < width:
        real = torch.arange(width, device=device) < num_org_local
        scaled = torch.where(real, scaled, float("-inf"))
        raw = torch.where(real, raw, float("-inf"))

    # Softmax statistics and the raw (max, id) pair, one exchange.
    local_max = scaled.amax(dim=-1)
    local_sum = torch.sum(torch.exp(scaled - local_max.unsqueeze(-1)), dim=-1)
    local_sum = torch.where(local_max == float("-inf"), 0.0, local_sum)
    raw_idx = torch.argmax(raw, dim=-1, keepdim=True)
    raw_max = torch.gather(raw, -1, raw_idx).to(f32)
    payload = torch.cat(
        [
            local_max.unsqueeze(-1),
            local_sum.unsqueeze(-1),
            raw_max,
            (raw_idx + vocab_start).to(f32),
        ],
        dim=-1,
    )
    rows = payload.shape[0]
    stats = tensor_model_parallel_all_gather(payload, dim=0).view(tp_size, rows, 4)
    row_max = stats[..., 0].amax(dim=0)
    row_sum = torch.sum(stats[..., 1] * torch.exp(stats[..., 0] - row_max), dim=0)
    best_rank = torch.argmax(stats[..., 2], dim=0, keepdim=True)
    raw_argmax = torch.gather(stats[..., 3], 0, best_rank).view(rows).to(torch.int32)
    probs = torch.exp(scaled - row_max.unsqueeze(-1)) / row_sum.unsqueeze(-1)

    # Top-p; rows with top_p >= 1 keep every column.
    threshold = vocab_parallel_topp_threshold(probs, top_p, radix_bits)
    kept = (probs >= threshold.unsqueeze(-1)) | (top_p >= 1.0).unsqueeze(-1)

    target_token_probs = None
    if draft_rows is not None:
        drafts = excluded[:draft_rows]
        local_col = drafts - vocab_start
        owned = (drafts >= 0) & (local_col >= 0) & (local_col < num_org_local)
        safe_col = local_col.clamp(min=0, max=width - 1).unsqueeze(-1)
        draft_probs = probs[:draft_rows]
        draft_kept = kept[:draft_rows]
        draft_prob = torch.where(
            owned & torch.gather(draft_kept, -1, safe_col).squeeze(-1),
            torch.gather(draft_probs, -1, safe_col).squeeze(-1),
            0.0,
        )
        kept_mass = torch.sum(torch.where(draft_kept, draft_probs, 0.0), dim=-1)
        summed = tensor_model_parallel_all_reduce(torch.stack([draft_prob, kept_mass]))
        target_token_probs = summed[0] / summed[1]

    if uniforms_on_shard:
        u = torch.cat(list(uniforms))
    else:
        u = torch.cat([shard_columns(x, vocab_start, width, tp_size) for x in uniforms])
    race_max, race_id = race_on_shard(
        probs, kept, excluded, u, vocab_start, num_org_local, tp_size
    )
    sampled = torch.where(is_greedy | ~(race_max > 0), raw_argmax, race_id)
    return sampled, target_token_probs


def vocab_parallel_spec_verify(
    shard_logits: torch.Tensor,
    target_logits_indices: torch.Tensor,
    bonus_logits_indices: torch.Tensor,
    draft_token_ids: torch.Tensor,
    num_draft_tokens: torch.Tensor,
    segment_ids: torch.Tensor,
    group_indices: torch.Tensor,
    temperatures: torch.Tensor,
    top_p: torch.Tensor,
    accept_u: torch.Tensor,
    recover_u: torch.Tensor,
    bonus_u: torch.Tensor,
    vocab_start: torch.Tensor,
    num_org_local: int,
    tp_size: int,
    max_draft_tokens: int,
    synthetic_conditional_rates: torch.Tensor | None,
    synthetic_mode: bool,
    radix_bits: int,
    bonus_greedy_temperature: float,
    verify_rows: int = 0,
    uniform_slice: bool = False,
) -> torch.Tensor:
    """Bonus sampling and draft rejection of one verify chunk from this rank's
    logit shard; the rejection sampler's [requests, max_draft_tokens + 1]
    output, the same on every rank.

    Args:
      shard_logits: [logits rows, V/tp] this rank's columns of the chunk's
        logits, padding columns (past ``num_org_local``) included.
      target_logits_indices / bonus_logits_indices: the draft slots' and the
        requests' rows of ``shard_logits``.
      draft_token_ids, segment_ids, group_indices: per draft slot.
      num_draft_tokens: per request.
      temperatures / top_p: [requests, 1] per request.
      accept_u: [draft slots] uniforms of the accept test.
      recover_u / bonus_u: [draft slots, V] and [requests, V] full-vocabulary
        uniforms; this rank uses its own columns.
      vocab_start: device int32 scalar, the global id of local column 0.
      num_org_local: local columns that are real vocabulary.
      tp_size: ranks the vocabulary is split over.
      synthetic_conditional_rates / synthetic_mode: synthetic acceptance.
      radix_bits: bits of the top-p threshold resolved per round.
      bonus_greedy_temperature: a bonus row at or below this temperature is
        greedy (SAMPLING_EPS of the default bonus sampler); a recovered row is
        greedy at temperature 0, as in the rejection sampler.
      verify_rows: static (TPU_SAMPLER_VERIFY_TRIM_ROWS); keep only the first
        that many draft slots, which hold every real one.
      uniform_slice: static (TPU_SAMPLER_VP_UNIFORM_SLICE); take this rank's
        uniform columns by a dynamic slice of the last dimension
        (`shard_columns`), before the verify_rows trim of ``recover_u``: a row
        slice of the full draw first would copy the trimmed rows at full width.
    """
    if verify_rows:
        target_logits_indices = target_logits_indices[:verify_rows]
        draft_token_ids = draft_token_ids[:verify_rows]
        segment_ids = segment_ids[:verify_rows]
        group_indices = group_indices[:verify_rows]
        accept_u = accept_u[:verify_rows]
        if not uniform_slice:
            recover_u = recover_u[:verify_rows]
    num_slots = draft_token_ids.shape[0]
    batch_size = num_draft_tokens.shape[0]
    f32 = torch.float32

    draft_token_ids = draft_token_ids.to(torch.int64)
    # A padding slot's segment id may be the request count.
    safe_segment_ids = segment_ids.to(torch.int64).clamp(min=0, max=batch_size - 1)
    req_temperatures = temperatures.squeeze(-1)
    req_top_p = top_p.squeeze(-1).to(f32)
    slot_temperatures = req_temperatures[safe_segment_ids]

    # One row a draft slot (its recovered token), then one a request (its
    # bonus token), with the rejection sampler's and the bonus sampler's
    # temperature scaling and greedy rule.
    logits = torch.index_select(
        shard_logits, 0, torch.cat([target_logits_indices, bonus_logits_indices])
    )
    divisor = torch.cat(
        [
            slot_temperatures.to(f32) + TEMPERATURE_EPS,
            torch.where(
                req_temperatures == 0,
                torch.ones_like(req_temperatures),
                req_temperatures,
            ).to(f32),
        ]
    )
    is_greedy = torch.cat(
        [slot_temperatures == 0, req_temperatures <= bonus_greedy_temperature]
    )
    excluded = torch.cat(
        [
            draft_token_ids.clamp(min=-1),
            torch.full(
                (batch_size,), -1, dtype=torch.int64, device=draft_token_ids.device
            ),
        ]
    )
    uniforms = (recover_u, bonus_u)
    if uniform_slice:
        width = shard_logits.shape[-1]
        uniforms = (
            shard_columns(recover_u, vocab_start, width, tp_size, True)[:num_slots],
            shard_columns(bonus_u, vocab_start, width, tp_size, True),
        )
    sampled, target_token_probs = sample_on_shard(
        logits,
        divisor,
        torch.cat([req_top_p[safe_segment_ids], req_top_p]),
        excluded,
        is_greedy,
        uniforms,
        vocab_start,
        num_org_local,
        tp_size,
        radix_bits,
        draft_rows=None if synthetic_mode else num_slots,
        uniforms_on_shard=uniform_slice,
    )
    # The raw argmax of a greedy draft row is its sampled token.
    accepted = accepted_drafts(
        draft_token_ids,
        is_greedy[:num_slots],
        sampled[:num_slots],
        target_token_probs,
        accept_u,
        group_indices,
        synthetic_conditional_rates,
        synthetic_mode,
    )
    return assemble_rejection_output(
        draft_token_ids,
        accepted,
        sampled[:num_slots],
        num_draft_tokens,
        sampled[num_slots:],
        segment_ids,
        group_indices,
        max_draft_tokens,
    )
