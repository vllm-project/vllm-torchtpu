# SPDX-License-Identifier: Apache-2.0
"""Verbatim copies of the STACK-2 candidate fb1f4ef44's tpu_runner code (generated from `git show
fb1f4ef44:src/vllm_torchtpu/runner/tpu_runner.py`, bodies unchanged): `_assemble_async_spec_substitution` (the
VLLM_TPU_SPEC_TAIL_COMPILED port adds an early-return branch in front of it) and sample_tokens' inline join + bound pad
of the parked source (unchanged). The flag-off identity test runs the new method against these. Not imported by
serving code."""

import torch

from vllm_torchtpu.runner.tpu_runner_async_output import (  # noqa: F401
    assemble_spec_next_tokens,
    compute_num_rejected,
)
from vllm_torchtpu.spec_decode.utils import fit_rows  # noqa: F401


def old_assemble_async_spec_substitution(self, drafts, next_tokens_per_chunk, state):
    """Async-spec substitution assembler.

    From the already-proposed on-device ``drafts`` (the unified eagle3
    propose call in ``sample_tokens``), builds the per-request
    ``[bonus, draft_1..K]`` substitution source and the stride-``1+K``
    ``next_token_indices``, and computes the per-request rejected count.
    Returns ``(next_tokens_tpu_chunks, next_token_indices,
    spec_decode_num_rejected_tokens, num_draft_per_req)``.
    """
    from vllm_torchtpu.runner.tpu_runner_async_output import (
        assemble_spec_next_tokens,
        compute_num_rejected,
    )

    assert drafts is not None, (
        "async spec substitution requires the eagle3 drafts proposed "
        "earlier in sample_tokens"
    )
    drafts_per_chunk = self._drafts_per_chunk(drafts, state.num_reqs_list)
    # Bucketed drafters keep the sampler's padded rows (an exact-count slice
    # would compile a program per request count); others keep exact rows.
    bucketed = getattr(getattr(self, "drafter", None), "bucketed_drafts", False)
    next_tokens_tpu_chunks: list[torch.Tensor] = []
    next_token_indices: dict[int, int] = {}
    num_rejected_chunks: list[torch.Tensor] = []
    num_draft_per_req: dict[int, int] = {}
    req_offset = 0
    padded_req_offset = 0
    # Uniform `1+K` source span per request: req position `p` maps to source
    # `[p*(1+K) .. p*(1+K)+K]`, and `req_id_to_index_copy` stores `p` (the
    # index builder multiplies by the stride). All chunks carry their next
    # proposals, even if they did not verify drafts in the current step.
    for nt_chunk, num_reqs, md, drafts_chunk in zip(
        next_tokens_per_chunk,
        state.num_reqs_list,
        state.spec_decode_metadata_list,
        drafts_per_chunk,
    ):
        chunk_padded_num_reqs = nt_chunk.shape[0] if bucketed else num_reqs
        drafts_chunk = fit_rows(drafts_chunk, chunk_padded_num_reqs)
        src = assemble_spec_next_tokens(
            nt_chunk, drafts_chunk, None if bucketed else num_reqs
        )
        if md is not None:
            num_rejected_chunks.append(
                compute_num_rejected(
                    nt_chunk, fit_rows(md.draft_lengths, chunk_padded_num_reqs)
                )
            )
            for r in range(num_reqs):
                num_draft_per_req[req_offset + r] = int(md.draft_lengths_cpu[r])
        else:
            # No drafts were verified this step, so there is no rejection
            # correction or speculative output-placeholder rollback.
            num_rejected_chunks.append(
                torch.zeros(
                    chunk_padded_num_reqs, dtype=torch.int32, device=nt_chunk.device
                )
            )
        next_tokens_tpu_chunks.append(src)
        for r in range(num_reqs):
            next_token_indices[req_offset + r] = padded_req_offset + r
        req_offset += num_reqs
        padded_req_offset += chunk_padded_num_reqs
    spec_num_rejected = (
        torch.cat(num_rejected_chunks) if num_rejected_chunks else None
    )
    return (
        next_tokens_tpu_chunks,
        next_token_indices,
        spec_num_rejected,
        num_draft_per_req,
    )


def old_bounded_async_source(self, next_tokens_tpu_chunks, spec_num_rejected):
    if len(next_tokens_tpu_chunks) == 1:
        next_tokens_tpu = next_tokens_tpu_chunks[0]
    else:
        next_tokens_tpu = torch.cat(next_tokens_tpu_chunks, dim=0)
    # Bound the async-spec source length so the torch.compiled
    # _substitute_placeholder_token (next step) sees a fixed shape
    # instead of recompiling per real num_reqs. Append-pad to the
    # max stride-(1+K) span: the substitution only reads
    # [position*stride .. +n_sched-1] for real reqs (positions <
    # num_reqs), so the appended tail is never indexed -- real
    # positions are unchanged. Spec only (stride 1+K); pure-non-spec
    # async keeps its stride-1 source + existing precompile.
    if spec_num_rejected is not None:
        assert self.speculative_config is not None
        bound = self.max_num_reqs * (
            1 + self.speculative_config.num_speculative_tokens
        )
        if next_tokens_tpu.shape[0] < bound:
            next_tokens_tpu = torch.nn.functional.pad(
                next_tokens_tpu, (0, bound - next_tokens_tpu.shape[0])
            )
    return next_tokens_tpu
