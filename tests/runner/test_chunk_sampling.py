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
"""Tests for the per-chunk sampling helpers of ``TPUModelRunner``."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from vllm import SamplingParams

from vllm_torchtpu.layers.adapter.sample.rejection_sampler import RejectionSampler
from vllm_torchtpu.runner import min_tokens
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

NEG_INF = float("-inf")


@pytest.fixture(autouse=True)
def _eager_kernels():
    """Run the ``@torch.compile(backend="tpu")`` helpers eagerly on CPU."""
    with torch.compiler.set_stance("force_eager"):
        yield


def _runner(*, all_greedy: bool = True, spec_k: int | None = None) -> SimpleNamespace:
    """Sampling at temperature 0.7 with top-k 2; strict verify with ``spec_k``."""
    spec_config = (
        None
        if spec_k is None
        else SimpleNamespace(
            num_speculative_tokens=spec_k,
            rejection_sample_method="strict",
            synthetic_acceptance_rates=None,
        )
    )
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            req_ids=["r0"], all_greedy=all_greedy, no_top_k=True, no_top_p=True
        ),
        speculative_config=spec_config,
        rejection_sampler=(
            None
            if spec_config is None
            else RejectionSampler(spec_config, torch.device("cpu"))
        ),
        _stage_timing=MagicMock(),
        _greedy_sample=lambda x: torch.argmax(x, dim=-1, keepdim=True),
        _sampler_topk_candidates=lambda *_args: 0,
        _build_padded_sampling_params=lambda _start, _end, logits: (
            torch.full((logits.shape[0], 1), 0.7),
            torch.full((logits.shape[0], 1), 2, dtype=torch.int32),
            torch.ones((logits.shape[0], 1)),
        ),
    )
    for fn in (
        TPUModelRunner._sample_chunk_logits,
        TPUModelRunner._sample_plain_chunk,
        TPUModelRunner._sample_spec_verify_chunk,
        TPUModelRunner._mask_min_tokens,
        TPUModelRunner._apply_temperature,
        TPUModelRunner._sampler_applies_topk_topp,
        TPUModelRunner.spec_bonus_and_target_logits,
        TPUModelRunner.spec_bonus_and_target_tokens,
        TPUModelRunner.spec_gather_bonus_and_target_logits,
    ):
        setattr(runner, fn.__name__, fn.__get__(runner))
    runner.sample_from_logits_func = TPUModelRunner.sample_from_logits.__get__(runner)
    return runner


def _below_min_tokens(stop_token: int) -> min_tokens.MinTokensState:
    """Batch row 0 is a request with no output yet, below ``min_tokens=10``."""
    tracker = min_tokens.MinTokensTracker()
    tracker.add("r0", SamplingParams(min_tokens=10, stop_token_ids=[stop_token]))
    return tracker.batch_state({"r0": 0}, lambda _req_id: 0)


@pytest.mark.parametrize("all_greedy", [True, False], ids=["greedy", "sampled"])
@pytest.mark.parametrize("is_spec_step", [False, True], ids=["plain", "spec_step"])
def test_plain_chunk_masks_then_samples(is_spec_step: bool, all_greedy: bool) -> None:
    runner = _runner(all_greedy=all_greedy)
    grammar_calls: list[str] = []

    def block_99(logits: torch.Tensor) -> torch.Tensor:
        out = logits.clone()
        out[0, 99] = NEG_INF
        return out

    def mask_logits(logits, _grammar_output, _start, _end):
        grammar_calls.append("plain")
        return block_99(logits)

    def mask_spec_logits(target, logits, _grammar_output, spec_tokens, *_args):
        grammar_calls.append(f"spec:{spec_tokens}")
        return target, block_99(logits)

    runner.structured_decoding_manager = SimpleNamespace(
        mask_logits=mask_logits, mask_spec_logits=mask_spec_logits
    )
    # Row 0 prefers token 99 (grammar-blocked), then the stop token 12 (below
    # min_tokens), then token 40; every other logit is far below token 40, so
    # sampling picks it too.
    raw = torch.full((4, 128), -20.0)
    raw[0, 99], raw[0, 12], raw[0, 40] = 20.0, 9.0, 6.0

    selected, logprob_logits = runner._sample_plain_chunk(
        raw.clone(),
        cur_start_idx=0,
        cur_end_idx=1,
        top_tokens=False,
        all_greedy=all_greedy,
        sampling_generator=torch.Generator().manual_seed(0),
        grammar_output=object(),
        scheduler_output=SimpleNamespace(scheduled_spec_decode_tokens={"r0": [5]}),
        is_spec_step=is_spec_step,
        min_toks=_below_min_tokens(12),
    )

    assert int(selected[0, 0]) == 40
    assert grammar_calls == ["spec:{'r0': [5]}" if is_spec_step else "plain"]
    # Logprobs see the grammar mask but not the min_tokens mask.
    assert logprob_logits[0, 99].item() == NEG_INF
    assert logprob_logits[0, 12].item() == 9.0
    assert logprob_logits[0, 40].item() == 6.0


def test_plain_chunk_top_tokens_returns_the_greedy_tokens() -> None:
    runner = _runner()
    tokens = torch.tensor([[30], [0], [0], [0]])

    def sample(min_toks: min_tokens.MinTokensState):
        return runner._sample_plain_chunk(
            tokens,
            cur_start_idx=0,
            cur_end_idx=1,
            top_tokens=True,
            all_greedy=True,
            sampling_generator=None,
            grammar_output=None,
            scheduler_output=SimpleNamespace(scheduled_spec_decode_tokens={}),
            is_spec_step=False,
            min_toks=min_toks,
        )

    selected, logprob_values = sample({})
    assert selected is tokens and logprob_values is tokens
    # _greedy_top_tokens_this_step keeps a step with a request below min_tokens
    # on the full logits, so the top-tokens path never sees one.
    with pytest.raises(AssertionError):
        sample(_below_min_tokens(12))


def _verify_md(drafts: list[int], pad_tokens: int = 4, pad_reqs: int = 4):
    """One request with ``drafts``; rows 0..n-1 are targets, row n is its bonus."""
    n = len(drafts)
    pad = pad_tokens - n
    return SimpleNamespace(
        draft_token_ids=torch.tensor(drafts + [0] * pad, dtype=torch.int32),
        draft_lengths=torch.tensor([n] + [0] * (pad_reqs - 1), dtype=torch.int32),
        draft_lengths_cpu=np.array([n]),
        target_logits_indices=torch.tensor(
            list(range(n)) + [0] * pad, dtype=torch.int32
        ),
        bonus_logits_indices=torch.tensor(
            [n] + [0] * (pad_reqs - 1), dtype=torch.int32
        ),
        segment_ids=torch.tensor([0] * n + [1] * pad, dtype=torch.int64),
        group_indices=torch.tensor(list(range(n)) + [0] * pad, dtype=torch.int32),
    )


@pytest.mark.parametrize("mode", ["greedy", "top_tokens", "random"])
def test_spec_verify_chunk_accepts_drafts_and_samples_the_bonus(mode: str) -> None:
    runner = _runner(all_greedy=mode != "random", spec_k=2)
    fused = MagicMock(side_effect=runner.spec_bonus_and_target_logits)
    gathered = MagicMock(side_effect=runner.spec_gather_bonus_and_target_logits)
    runner.spec_bonus_and_target_logits = fused
    runner.spec_gather_bonus_and_target_logits = gathered
    # The target rows prefer the drafts 30 and 31 and the bonus row 32, far above
    # every other logit, so sampling at temperature 0.7 agrees with greedy.
    logits = torch.full((4, 128), -20.0)
    for row, tok in enumerate((30, 31, 32)):
        logits[row, tok] = 6.0
    values = torch.argmax(logits, -1, keepdim=True) if mode == "top_tokens" else logits

    out = runner._sample_spec_verify_chunk(
        values,
        _verify_md([30, 31]),
        None,
        SimpleNamespace(scheduled_spec_decode_tokens={}),
        0,
        1,
        mode != "random",
        torch.Generator().manual_seed(0),
        mode == "top_tokens",
        {},
    )

    assert out[0].tolist() == [30, 31, 32]
    # Only an all-greedy chunk on the full logits takes the fused verify prelude.
    expected_preludes = {"greedy": (1, 0), "top_tokens": (0, 0), "random": (0, 1)}
    assert (fused.call_count, gathered.call_count) == expected_preludes[mode]


@pytest.mark.parametrize("all_greedy", [True, False], ids=["greedy", "random"])
def test_spec_verify_chunk_applies_grammar_then_min_tokens(all_greedy: bool) -> None:
    runner = _runner(all_greedy=all_greedy, spec_k=2)
    seen_by_grammar: list[float] = []

    def mask_spec_logits(target, bonus, *_args):
        # The grammar runs first: it still sees the stop token's logit.
        seen_by_grammar.append(bonus[0, 12].item())
        bonus = bonus.clone()
        bonus[0, 32] = NEG_INF
        return target, bonus

    runner.structured_decoding_manager = SimpleNamespace(
        mask_spec_logits=mask_spec_logits,
        mask_invalid_draft_tokens=lambda draft_token_ids, *_args, **_kwargs: (
            draft_token_ids
        ),
    )
    # The bonus row prefers token 32 (grammar-blocked), then the stop token 12
    # (below min_tokens), then token 33.
    logits = torch.full((4, 128), -20.0)
    logits[0, 30] = logits[1, 31] = 6.0
    logits[2, 32], logits[2, 12], logits[2, 33] = 7.0, 6.5, 6.0

    out = runner._sample_spec_verify_chunk(
        logits,
        _verify_md([30, 31]),
        object(),
        SimpleNamespace(scheduled_spec_decode_tokens={"r0": [30, 31]}),
        0,
        1,
        all_greedy,
        torch.Generator().manual_seed(0),
        False,
        _below_min_tokens(12),
    )

    assert out[0].tolist() == [30, 31, 33]
    assert seen_by_grammar == [6.5]
