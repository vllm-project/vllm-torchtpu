# SPDX-License-Identifier: Apache-2.0
"""Shape sets for the startup precompile of the spec-decode verify path.

Kept free of runner state so the sets can be unit-tested without a TPU.

Why these exist: on a spec-decode step ``compute_selected_logits`` gathers
``final_logits_indices`` -- ``padded_logits_length`` rows, a num-tokens bucket
-- out of the step's hidden states, whose row count is the step's own
num-tokens bucket. ``_precompile_compute_selected_logits`` only sweeps index
lengths from ``num_reqs_paddings`` (the non-spec path), and the spec-decode
real warmup only runs decode-only batches, so a *mixed* step (a prefill chunk
plus decodes carrying drafts) reaches ``(chunk bucket, padded_logits_length)``
pairs nobody compiled. Each one is a dynamo retrace plus a full XLA compile
(~1.5 s for GLM-5.2's vocab-parallel LM head) on the serving path, stalling
every running request.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence


def spec_verify_logits_lengths(
    num_tokens_paddings: Sequence[int],
    max_num_reqs: int,
    num_spec_tokens: int,
) -> list[int]:
    """Every ``padded_logits_length`` a spec verify step can produce.

    A step samples ``num_draft_tokens + 1`` positions per request, at most
    ``max_num_reqs * (K + 1)`` in total, padded up to a num-tokens bucket
    (``get_spec_decode_metadata``). Logits rows never exceed the largest
    bucket, hence the cap -- the same bound ``_precompile_rejection_sampler``
    uses.
    """
    paddings = sorted(num_tokens_paddings)
    if not paddings:
        return []
    cap = min(max_num_reqs * (num_spec_tokens + 1), paddings[-1])
    top = paddings[bisect.bisect_left(paddings, cap)]
    return [p for p in paddings if p <= top]


def num_reqs_sweep_pairs(
    num_tokens_paddings: Sequence[int],
    num_reqs_paddings: Sequence[int],
    max_num_reqs: int,
) -> set[tuple[int, int]]:
    """The ``(num_tokens, index_len)`` pairs the num_reqs sweep compiles.

    Mirrors the loop in ``_precompile_compute_selected_logits``.
    """
    pairs: set[tuple[int, int]] = set()
    for num_tokens in num_tokens_paddings:
        for num_reqs in num_reqs_paddings:
            pairs.add((num_tokens, num_reqs))
            if num_reqs >= min(num_tokens, max_num_reqs):
                break
    return pairs


def spec_selected_logits_pairs(
    num_tokens_paddings: Sequence[int],
    num_reqs_paddings: Sequence[int],
    max_num_reqs: int,
    num_spec_tokens: int,
) -> list[tuple[int, int]]:
    """``(num_tokens, padded_logits_length)`` pairs a spec step can hand
    ``compute_selected_logits`` that the num_reqs sweep does not compile.

    A step's sampled positions are a subset of its scheduled tokens, and both
    are padded with the same monotone bucket list, so the padded logits length
    never exceeds the padded token count.
    """
    covered = num_reqs_sweep_pairs(num_tokens_paddings, num_reqs_paddings, max_num_reqs)
    lengths = spec_verify_logits_lengths(
        num_tokens_paddings, max_num_reqs, num_spec_tokens
    )
    return [
        (num_tokens, length)
        for num_tokens in sorted(num_tokens_paddings)
        for length in lengths
        if length <= num_tokens and (num_tokens, length) not in covered
    ]
