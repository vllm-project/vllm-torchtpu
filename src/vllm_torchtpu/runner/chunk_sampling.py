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
"""Per-chunk token sampling for ``TPUModelRunner.sample_tokens``.

``sample_tokens`` builds one ``StepContext`` per step and passes it, with each
chunk's logits, batch range and processors, to ``sample_plain_chunk`` or, for a
chunk that verifies drafts, to ``sample_spec_verify_chunk``. Both draw tokens
through ``_sample_chunk_logits``; a verify chunk hands the params of its bonus
draw to the rejection sampler.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm_torchtpu.runner import min_tokens as min_tokens_mask

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput

    from vllm_torchtpu.runner import chunk_processors
    from vllm_torchtpu.runner.speculative_decoding_manager import SpecDecodeMetadata
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


@dataclass(frozen=True)
class StepContext:
    """The sampling inputs that every chunk of a ``sample_tokens`` step shares."""

    top_tokens: bool
    all_greedy: bool
    sampling_generator: torch.Generator | None
    grammar_output: GrammarOutput | None
    scheduler_output: SchedulerOutput
    min_toks: min_tokens_mask.MinTokensState
    rep_pen: bool


@dataclass(frozen=True)
class ChunkSamplingParams:
    """A sampled chunk's padded ``[rows, 1]`` sampling params and the sampler
    variant that drew its tokens. The rejection sampler reuses them to verify
    the chunk's drafts."""

    temperatures: torch.Tensor
    top_k: torch.Tensor
    top_p: torch.Tensor
    topk_candidates: int
    apply_topk_topp: bool


def _sample_chunk_logits(
    runner: TPUModelRunner,
    step: StepContext,
    logits: torch.Tensor,
    *,
    cur_start_idx: int,
    cur_end_idx: int,
) -> tuple[torch.Tensor, ChunkSamplingParams | None]:
    """Samples one token per row of a chunk's ``[rows, vocab]`` logits.

    Returns the ``[rows, 1]`` token IDs and, for a sampled (not all-greedy)
    chunk, the padded sampling params that drew them.
    """
    if step.all_greedy:
        return runner._greedy_sample(logits), None
    temperatures, top_k, top_p = runner._build_padded_sampling_params(
        cur_start_idx, cur_end_idx, logits
    )
    params = ChunkSamplingParams(
        temperatures,
        top_k,
        top_p,
        runner._sampler_topk_candidates(
            cur_start_idx, cur_end_idx, step.grammar_output
        ),
        runner._sampler_applies_topk_topp(),
    )
    u = torch.rand_like(logits, dtype=torch.float32, generator=step.sampling_generator)
    token_ids = runner.sample_from_logits_func(
        logits,
        temperatures,
        u,
        top_k,
        top_p,
        all_greedy=False,
        topk_candidates=params.topk_candidates,
        apply_topk_topp=params.apply_topk_topp,
    )
    return token_ids, params


def sample_plain_chunk(
    runner: TPUModelRunner,
    step: StepContext,
    logits: torch.Tensor,
    *,
    cur_start_idx: int,
    cur_end_idx: int,
    processors: Sequence[chunk_processors.PlainChunkProcessor] = (),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask and sample a plain chunk ``[cur_start_idx, cur_end_idx)``.

    With ``step.top_tokens``, ``logits`` holds each row's greedy token instead.
    In a speculative step, no request of a plain chunk has drafts, so
    ``mask_logits`` reads the bitmask row of each request's next token.
    ``step.rep_pen`` is ``RepetitionPenaltyTracker.begin_step`` for this step.
    ``processors`` run after the ``min_tokens`` mask and the repetition
    penalty, in order (``chunk_processors.PlainChunkProcessor``).

    Returns ``(selected_token_ids, logprob_logits)``. Logprobs come from the
    logits before the min_tokens mask and the repetition penalty, as vLLM's
    default raw_logprobs mode precedes its logits processors (the TPU runner
    has no logprobs_mode).
    """
    if step.grammar_output is not None:
        logits = runner.structured_decoding_manager.mask_logits(
            logits, step.grammar_output, cur_start_idx, cur_end_idx
        )
    logprob_logits = logits
    min_tokens_rows = min_tokens_mask.plain_rows(
        step.min_toks, cur_start_idx, cur_end_idx
    )
    if min_tokens_rows:
        # _greedy_top_tokens_this_step keeps a step with a request below
        # min_tokens on the full logits.
        assert not step.top_tokens
        logits = runner._mask_min_tokens(logits, min_tokens_rows, step.min_toks)
    if step.rep_pen:
        # _greedy_top_tokens_this_step keeps a step with a penalized
        # request on the full logits.
        assert not step.top_tokens
        logits = runner._repetition_penalty.apply(
            logits,
            runner.input_batch,
            runner.requests,
            cur_start_idx,
            cur_end_idx,
        )
    if not step.top_tokens:
        for proc in processors:
            logits = proc.apply(logits)
    with runner._stage_timing.stage("sample"):
        if step.top_tokens:
            selected = logits
            for proc in processors:
                selected = proc.override_top_tokens(selected)
        else:
            selected, _ = _sample_chunk_logits(
                runner,
                step,
                logits,
                cur_start_idx=cur_start_idx,
                cur_end_idx=cur_end_idx,
            )
        for proc in processors:
            proc.on_sampled(selected)
    return selected, logprob_logits


def sample_spec_verify_chunk(
    runner: TPUModelRunner,
    step: StepContext,
    logits: torch.Tensor,
    md: SpecDecodeMetadata,
    cur_start_idx: int,
    cur_end_idx: int,
    processors: Sequence[chunk_processors.SpecChunkProcessor] = (),
) -> torch.Tensor:
    """Bonus-token sampling + draft rejection for one spec-verify chunk.

    Returns the rejection sampler's [num_reqs, 1 + num_spec_tokens]
    selected tokens for the chunk. With ``step.top_tokens``, ``logits`` holds
    each row's greedy token instead (no request is below min_tokens then).
    ``processors`` prepare the target and bonus rows and the drafts before
    the rejection sampler runs, then ``finish`` on its output
    (``chunk_processors.SpecChunkProcessor``).
    """
    assert md.draft_token_ids is not None, "deferred draft_token_ids never filled"
    assert runner.rejection_sampler is not None
    assert runner.speculative_config is not None
    max_draft_tokens = runner.speculative_config.num_speculative_tokens
    draft_token_ids = md.draft_token_ids
    target_logits: torch.Tensor | None = None
    target_argmax: torch.Tensor | None = None
    params: ChunkSamplingParams | None = None
    if step.top_tokens:
        bonus_token_ids, target_argmax = runner.spec_bonus_and_target_tokens(
            logits, md.bonus_logits_indices, md.target_logits_indices
        )
        for proc in processors:
            target_argmax, bonus_token_ids, draft_token_ids = proc.prepare_tokens(
                target_argmax,
                bonus_token_ids,
                draft_token_ids,
                md.segment_ids,
                md.group_indices,
                max_draft_tokens,
            )
    else:
        min_toks = step.min_toks
        min_tokens_rows = None
        if min_toks and any(cur_start_idx <= i < cur_end_idx for i in min_toks):
            draft_lengths = md.draft_lengths_cpu[: cur_end_idx - cur_start_idx]
            if step.grammar_output is None:
                # Mask before the gathers, so a greedy chunk keeps the fused
                # verify prelude.
                logits = runner._mask_min_tokens(
                    logits,
                    min_tokens_mask.verify_logits_rows(
                        min_toks, cur_start_idx, draft_lengths.tolist()
                    ),
                    min_toks,
                )
            else:
                # After the grammar mask, which the structured-output
                # restore needs to see.
                min_tokens_rows = min_tokens_mask.verify_rows(
                    min_toks, cur_start_idx, draft_lengths.tolist()
                )
        if step.all_greedy and step.grammar_output is None and not processors:
            bonus_token_ids, target_logits = runner.spec_bonus_and_target_logits(
                logits, md.bonus_logits_indices, md.target_logits_indices
            )
        else:
            bonus_logits, target_logits = runner.spec_gather_bonus_and_target_logits(
                logits, md.bonus_logits_indices, md.target_logits_indices
            )
            if step.grammar_output is not None:
                target_logits, bonus_logits = (
                    runner.structured_decoding_manager.mask_spec_logits(
                        target_logits,
                        bonus_logits,
                        step.grammar_output,
                        step.scheduler_output.scheduled_spec_decode_tokens,
                        md.draft_lengths_cpu,
                        cur_start_idx,
                        cur_end_idx,
                    )
                )
                draft_token_ids = (
                    runner.structured_decoding_manager.mask_invalid_draft_tokens(
                        draft_token_ids,
                        step.grammar_output,
                        cur_start_idx,
                        cur_end_idx,
                        draft_lengths_cpu=md.draft_lengths_cpu,
                    )
                )
            if min_tokens_rows is not None:
                target_rows, bonus_rows = min_tokens_rows
                target_logits = runner._mask_min_tokens(
                    target_logits, target_rows, min_toks
                )
                bonus_logits = runner._mask_min_tokens(
                    bonus_logits, bonus_rows, min_toks
                )
            for proc in processors:
                target_logits, bonus_logits, draft_token_ids = proc.prepare_logits(
                    target_logits,
                    bonus_logits,
                    draft_token_ids,
                    md.segment_ids,
                    md.group_indices,
                    max_draft_tokens,
                )
            bonus_token_ids, params = _sample_chunk_logits(
                runner,
                step,
                bonus_logits,
                cur_start_idx=cur_start_idx,
                cur_end_idx=cur_end_idx,
            )
            bonus_token_ids = bonus_token_ids.view(-1)
    # Greedy chunks (and the top-tokens path, whose steps are all greedy)
    # draw acceptance noise only for synthetic acceptance; sampled chunks
    # also draw the recovery noise and verify on their sampling params.
    do_sampling = params is not None
    accept_u = None
    if runner.rejection_sampler.synthetic_mode or do_sampling:
        accept_u = torch.rand(
            draft_token_ids.shape,
            dtype=torch.float32,
            device=logits.device,
            generator=step.sampling_generator,
        )
    temperatures = top_k = top_p = recover_u = None
    topk_candidates = 0
    apply_topk_topp = True
    if do_sampling:
        assert params is not None and target_logits is not None
        temperatures = params.temperatures[md.segment_ids]
        top_k = params.top_k[md.segment_ids]
        top_p = params.top_p[md.segment_ids]
        topk_candidates = params.topk_candidates
        apply_topk_topp = params.apply_topk_topp
        recover_u = torch.rand_like(
            target_logits, dtype=torch.float32, generator=step.sampling_generator
        )
    next_tokens = runner.rejection_sampler(
        draft_token_ids=draft_token_ids,
        num_draft_tokens=md.draft_lengths,
        target_logits=target_logits,
        bonus_token_ids=bonus_token_ids,
        segment_ids=md.segment_ids,
        group_indices=md.group_indices,
        max_draft_tokens=max_draft_tokens,
        temperatures=temperatures,
        top_k=top_k,
        top_p=top_p,
        accept_u=accept_u,
        recover_u=recover_u,
        do_sampling=do_sampling,
        topk_candidates=topk_candidates,
        target_argmax=target_argmax,
        apply_topk_topp=apply_topk_topp,
    )
    for proc in processors:
        proc.finish(next_tokens)
    return next_tokens
