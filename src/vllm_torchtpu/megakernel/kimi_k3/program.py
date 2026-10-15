# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 as a megakernel program: the prefill role's :class:`KimiPrefiller`
or the decode role's :class:`KimiDecoder`, and what they support.

The K3 programs have no vLLM dependencies; ``megakernel.runner`` builds them
from vLLM's configuration and the standalone tools build them directly.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from vllm_torchtpu.megakernel.config import MegakernelEngineConfig
from vllm_torchtpu.megakernel.program import Capabilities

TP_SIZE = 32
DCP_SIZE = 32
BLOCK_SIZE = 256
HIDDEN_SIZE = 7168
VOCAB = 163840  # tokens; vocabulary-sharded 5120 per TP rank
DRAFT_TOKENS = 7  # DSpark: the verify is the anchor and 7 drafts
SPEC_BATCHES = (1, 2, 4, 8, 16)  # the batched speculation's compiled request counts
MAX_STEP_TOKENS = 8192  # the prefill's largest window
# Sampling parameters the ops implement (beyond greedy): temperature, top-p,
# top-k and a per-request seed, on both roles.
SAMPLING = frozenset({"temperature", "top_p", "top_k", "seed"})

_SPEC_SUPPORT = (
    f"3 or {DRAFT_TOKENS} drafts at batches 1-8, and 3 at batches "
    f"9-{max(SPEC_BATCHES)} through the compiled "
    f"speculation batches {SPEC_BATCHES} (up to --max-num-seqs), 0 anywhere"
)


def kernel_modules(*names: str) -> list[Any]:
    """Import ``vllm_torchtpu.megakernel.kimi_k3.kernels.<name>`` for each name."""
    return [
        importlib.import_module(f"vllm_torchtpu.megakernel.kimi_k3.kernels.{name}")
        for name in names
    ]


def supports_spec_tokens(batch: int, drafts: int) -> bool:
    """Widths supported by the single and batched verifier/drafter."""
    return drafts == 0 or (
        1 <= batch <= 16 and (drafts == 3 or (batch <= 8 and drafts == DRAFT_TOKENS))
    )


def spec_tokens_per_batch(config: MegakernelEngineConfig) -> tuple[int, ...] | None:
    """The drafts vLLM schedules at every decoding batch size: a dense table
    whose index ``b`` holds batch size ``b``'s count (``[0]`` unused), or
    None when the decoder does not speculate.

    vLLM's own schedule semantics (``build_dynamic_sd_schedule_lookup``): a
    range covers its batch sizes, a gap or the tail carries the previous
    range's count forward, every count clamps to ``num_speculative_tokens``,
    and without a schedule every batch size gets ``num_speculative_tokens``.
    Raises ``NotImplementedError`` for a schedule the kernels can't run.
    """
    if config.role != "decode" or not config.num_speculative_tokens:
        return None
    schedule = config.num_speculative_tokens_per_batch_size
    num_spec, max_seqs = config.num_speculative_tokens, config.max_num_seqs
    dense = [0] * (max_seqs + 1)
    if schedule is None:
        for b in range(1, max_seqs + 1):
            dense[b] = num_spec
    else:
        entries = sorted((int(lo), int(hi), int(k)) for lo, hi, k in schedule)
        previous_end = 0
        for lo, hi, k in entries:
            if lo <= 0 or hi < lo or k < 0:
                raise NotImplementedError(
                    "num_speculative_tokens_per_batch_size entries are "
                    f"(min_batch, max_batch, drafts) with 1 <= min <= max and "
                    f"drafts >= 0, not ({lo}, {hi}, {k})"
                )
            if lo <= previous_end:
                raise NotImplementedError(
                    "num_speculative_tokens_per_batch_size ranges must be "
                    f"non-overlapping: ({lo}, {hi}, {k})"
                )
            previous_end = hi
        if entries[0][0] != 1:
            raise NotImplementedError(
                "num_speculative_tokens_per_batch_size's first range must "
                f"start at batch size 1: {entries[0]}"
            )
        next_b, last_k = 1, 0
        for lo, hi, k in entries:
            for b in range(next_b, min(lo, max_seqs + 1)):  # a gap carries forward
                dense[b] = min(num_spec, last_k)
            for b in range(max(lo, next_b), min(hi, max_seqs) + 1):
                dense[b] = min(num_spec, k)
            next_b, last_k = max(next_b, hi + 1), k
            if next_b > max_seqs:
                break
        for b in range(next_b, max_seqs + 1):  # and so does the tail
            dense[b] = min(num_spec, last_k)
    for b in range(1, max_seqs + 1):
        k = dense[b]
        if k == 0:
            continue
        supported = supports_spec_tokens(b, k)
        if b > 1 and supported:
            # A batched verify holds b requests only through a compiled
            # speculation batch, itself no wider than --max-num-seqs.
            supported = any(b <= s <= max_seqs for s in SPEC_BATCHES)
        if not supported:
            given = (
                "None: num_speculative_tokens everywhere"
                if schedule is None
                else str(schedule)
            )
            raise NotImplementedError(
                f"DSpark on the K3 decoder supports {_SPEC_SUPPORT}; the "
                f"schedule gives {k} drafts at batch size {b} "
                f"(num_speculative_tokens_per_batch_size={given}; "
                f"--max-num-seqs {max_seqs})"
            )
    return tuple(dense)


def check(config: MegakernelEngineConfig) -> None:
    """Raise ``NotImplementedError`` for a configuration the K3 programs
    do not implement."""
    if config.role not in ("prefill", "decode"):
        raise NotImplementedError(
            f"the K3 megakernel serves one role of a P/D pair, not {config.role!r}"
        )
    if config.tensor_parallel_size != TP_SIZE:
        raise NotImplementedError(
            f"the K3 megakernel needs --tensor-parallel-size {TP_SIZE}"
        )
    if config.attention_dp not in (1, 2) or (
        config.attention_dp > 1 and config.role != "decode"
    ):
        raise NotImplementedError(
            "the K3 decoder runs attention_dp 1 or 2; the prefill runs 1"
        )
    dcp = DCP_SIZE // config.attention_dp
    if config.decode_context_parallel_size != dcp:
        raise NotImplementedError(
            f"the K3 megakernel needs --decode-context-parallel-size {dcp}"
            + (f" under attention_dp {config.attention_dp}" if dcp != DCP_SIZE else "")
        )
    if config.attention_dp > 1 and config.num_speculative_tokens:
        raise NotImplementedError(
            "the K3 decoder under attention DP does not speculate"
        )
    from vllm_torchtpu.megakernel.kimi_k3.layout import cache_layout

    layout = cache_layout(config.kv_cache_dtype)
    if config.block_size != layout.tokens_per_block:
        raise NotImplementedError(
            f"the K3 megakernel with cache dtype {config.kv_cache_dtype} requires "
            f"--block-size {layout.tokens_per_block}"
        )
    if config.cp_kv_cache_interleave_size not in (None, layout.tokens_per_block):
        raise NotImplementedError(
            f"the K3 megakernel requires --cp-kv-cache-interleave-size "
            f"{layout.tokens_per_block} with cache dtype {config.kv_cache_dtype}"
        )
    if config.role == "prefill" and config.max_num_batched_tokens > MAX_STEP_TOKENS:
        raise NotImplementedError(
            f"the K3 prefill takes at most {MAX_STEP_TOKENS} tokens per step: "
            f"--max-num-batched-tokens <= {MAX_STEP_TOKENS}"
        )
    if config.role == "decode" and config.num_speculative_tokens not in (
        0,
        3,
        DRAFT_TOKENS,
    ):
        raise NotImplementedError(
            "DSpark on the K3 decoder supports "
            f"num_speculative_tokens=0, 3 or {DRAFT_TOKENS}"
        )
    dense = spec_tokens_per_batch(config)  # validates the per-batch-size schedule
    if config.synthetic_acceptance_rates_by_width is not None:
        by_width = dict(config.synthetic_acceptance_rates_by_width)
        missing = sorted({k for k in dense or () if k} - set(by_width))
        if missing:
            raise NotImplementedError(
                f"synthetic_acceptance_lengths gives no length for the {missing} "
                f"draft widths the schedule runs (it has {sorted(by_width)})"
            )
        own = by_width.get(config.num_speculative_tokens)
        if own is not None and own != config.synthetic_acceptance_rates:
            raise NotImplementedError(
                f"synthetic_acceptance_lengths' {config.num_speculative_tokens}-draft "
                f"rates {list(own)} differ from --speculative-config's "
                f"{list(config.synthetic_acceptance_rates or ())}"
            )
    if config.role == "decode" and not config.async_scheduling:
        raise NotImplementedError(
            "the K3 decoder needs async scheduling (it keeps each request's next "
            "input on the device)"
        )


def capabilities(config: MegakernelEngineConfig) -> Capabilities:
    check(config)
    dense = spec_tokens_per_batch(config)
    speculating = dense is not None and any(dense)
    return Capabilities(
        roles=frozenset({config.role}),
        sampling=SAMPLING,
        speculation="dspark" if speculating else None,
        num_speculative_tokens=config.num_speculative_tokens if speculating else 0,
        max_step_tokens=MAX_STEP_TOKENS
        if config.role == "prefill"
        else config.max_num_batched_tokens,
        page_aligned_chunks=config.role == "prefill",
        block_copies=config.role == "prefill",
        # The decoder's plain op masks its logits (the prefill discards its
        # first tokens, so it only accepts the requests). DSpark's verify ops
        # do not mask: a step with a structured request runs plain instead.
        structured_outputs=True,
    )


def build(
    config: MegakernelEngineConfig,
    presharded: Path,
    draft_checkpoint: Path | None = None,
    *,
    tp_rank: int,
):
    """The program for ``config.role`` on TP rank ``tp_rank`` (the order the KV
    connector pairs devices in; ``draft_checkpoint``: the DSpark speculator --
    the prefill computes its ring, the decoder drafts with it and imports the
    ring the prefill wrote)."""
    check(config)
    if config.role == "prefill":
        from vllm_torchtpu.megakernel.kimi_k3.prefill import KimiPrefiller

        return KimiPrefiller(config, presharded, draft_checkpoint, tp_rank=tp_rank)
    from vllm_torchtpu.megakernel.kimi_k3.decode import KimiDecoder
    from vllm_torchtpu.megakernel.kimi_k3.decode_ops import (
        Control,
        DecodeOps,
        decode_batch_sizes,
    )

    dense = spec_tokens_per_batch(config)
    draft = None
    if draft_checkpoint is not None and dense is not None and any(dense):
        from vllm_torchtpu.megakernel.kimi_k3.draft import DsparkDraft

        draft = DsparkDraft(
            max_position=config.max_model_len + 16,
            checkpoint=str(draft_checkpoint),
            presharded=Path(presharded),
            synthetic_rates=config.synthetic_acceptance_rates,
            synthetic_rates_by_width=dict(config.synthetic_acceptance_rates_by_width)
            if config.synthetic_acceptance_rates_by_width is not None
            else None,
        )
    # Speculative post-states are temporary; acceptance commits one state
    # into each regular request slot (decode_ops.Control).
    sub_base = 2 * config.max_num_seqs + 1
    ops = DecodeOps(
        presharded,
        num_slots=Control.total_slots(sub_base) if draft else sub_base,
        batch_sizes=decode_batch_sizes(config.max_num_seqs),
    )
    return KimiDecoder(config, ops, draft, tp_rank=tp_rank)
