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

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from vllm import SamplingParams

from vllm_torchtpu.layers.adapter.sample.rejection_sampler import RejectionSampler
from vllm_torchtpu.layers.adapter.sample.top_k_top_p import CANDIDATE_BLOCK
from vllm_torchtpu.runner import min_tokens, tpu_runner
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


# Per-chunk processor hooks (``chunk_processors``).


def _grammar_blocking(token: int) -> SimpleNamespace:
    """A structured-decoding stub whose grammar masks ``token`` in row 0."""

    def block(logits: torch.Tensor) -> torch.Tensor:
        out = logits.clone()
        out[0, token] = NEG_INF
        return out

    def mask_logits(logits, *_args):
        return block(logits)

    def mask_spec_logits(target, logits, *_args):
        return target, block(logits)

    return SimpleNamespace(
        mask_logits=mask_logits,
        mask_spec_logits=mask_spec_logits,
        mask_invalid_draft_tokens=lambda draft_token_ids, *_args, **_kwargs: (
            draft_token_ids
        ),
    )


class _PlainRecorder:
    """Records every hook call; ``apply`` writes ``value`` to column ``col``."""

    def __init__(self, name: str, col: int, value: float, events: list[str]):
        self.name, self.col, self.value, self.events = name, col, value, events

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        # The grammar (99) and min_tokens (12) masks, then column 50 as the
        # previous processor left it.
        seen = tuple(logits[0, c].item() for c in (99, 12, 50))
        self.events.append(f"{self.name}.apply{seen}")
        out = logits.clone()
        out[0, self.col] = self.value
        return out

    def override_top_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        self.events.append(f"{self.name}.override:{int(token_ids[0, 0])}")
        return token_ids * 2 + int(self.value > 0)

    def on_sampled(self, token_ids: torch.Tensor) -> None:
        self.events.append(f"{self.name}.sampled:{int(token_ids[0, 0])}")


@pytest.mark.parametrize("all_greedy", [True, False], ids=["greedy", "sampled"])
@pytest.mark.parametrize("is_spec_step", [False, True], ids=["plain", "spec_step"])
def test_plain_chunk_processors_run_in_order_after_the_masks(
    is_spec_step: bool, all_greedy: bool
) -> None:
    runner = _runner(all_greedy=all_greedy)
    runner.structured_decoding_manager = _grammar_blocking(99)
    raw = torch.full((4, 128), -20.0)
    raw[0, 99], raw[0, 12], raw[0, 50], raw[0, 40] = 20.0, 9.0, -3.0, 6.0
    events: list[str] = []
    # p1 promotes token 50 over token 40; p2 masks it again.
    procs = [
        _PlainRecorder("p1", 50, 100.0, events),
        _PlainRecorder("p2", 50, NEG_INF, events),
    ]

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
        processors=procs,
    )

    assert events == [
        f"p1.apply({NEG_INF}, {NEG_INF}, -3.0)",
        f"p2.apply({NEG_INF}, {NEG_INF}, 100.0)",
        "p1.sampled:40",
        "p2.sampled:40",
    ]
    assert int(selected[0, 0]) == 40
    # Logprobs see none of the processors.
    assert logprob_logits[0, 50].item() == -3.0


def test_plain_chunk_top_tokens_overrides_in_order_then_on_sampled() -> None:
    runner = _runner()
    events: list[str] = []
    procs = [
        _PlainRecorder("p1", 50, 100.0, events),
        _PlainRecorder("p2", 50, NEG_INF, events),
    ]

    selected, _ = runner._sample_plain_chunk(
        torch.tensor([[30], [0], [0], [0]]),
        cur_start_idx=0,
        cur_end_idx=1,
        top_tokens=True,
        all_greedy=True,
        sampling_generator=None,
        grammar_output=None,
        scheduler_output=SimpleNamespace(scheduled_spec_decode_tokens={}),
        is_spec_step=False,
        min_toks={},
        processors=procs,
    )

    # Each processor rewrites the previous one's tokens; on_sampled then sees
    # the final tokens.
    assert events == [
        "p1.override:30",
        "p2.override:61",
        "p1.sampled:122",
        "p2.sampled:122",
    ]
    assert int(selected[0, 0]) == 122


class _SpecRecorder:
    """Records every hook call. Optionally rejects draft ``reject_draft``, or
    makes target row ``target[0]`` prefer token ``target[1]``, or makes the
    bonus row prefer token ``bonus``."""

    def __init__(
        self,
        name: str,
        events: list[str],
        *,
        reject_draft: int | None = None,
        target: tuple[int, int] | None = None,
        bonus: int | None = None,
    ):
        self.name, self.events = name, events
        self.reject_draft, self.target, self.bonus = reject_draft, target, bonus

    def _drafts(self, draft_token_ids: torch.Tensor) -> torch.Tensor:
        if self.reject_draft is None:
            return draft_token_ids
        return torch.where(
            draft_token_ids == self.reject_draft, -1, draft_token_ids
        ).to(draft_token_ids.dtype)

    def prepare_logits(self, target_logits, bonus_logits, draft_token_ids, *_args):
        self.events.append(f"{self.name}.prepare_logits")
        if self.target is not None:
            target_logits = target_logits.clone()
            target_logits[self.target[0], self.target[1]] = 50.0
        if self.bonus is not None:
            bonus_logits = bonus_logits.clone()
            bonus_logits[0, self.bonus] = 50.0
        return target_logits, bonus_logits, self._drafts(draft_token_ids)

    def prepare_tokens(self, target_argmax, bonus_token_ids, draft_token_ids, *_args):
        self.events.append(f"{self.name}.prepare_tokens")
        if self.target is not None:
            target_argmax = target_argmax.clone()
            target_argmax[self.target[0]] = self.target[1]
        if self.bonus is not None:
            bonus_token_ids = bonus_token_ids.clone()
            bonus_token_ids[0] = self.bonus
        return target_argmax, bonus_token_ids, self._drafts(draft_token_ids)

    def finish(self, next_tokens: torch.Tensor) -> None:
        self.events.append(f"{self.name}.finish:{next_tokens[0].tolist()}")


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ({"bonus": 33}, [30, 31, 33]),
        ({"target": (1, 34)}, [30, 34, -1]),
        ({"reject_draft": 31}, [30, 31, -1]),
    ],
    ids=["bonus_row", "target_row", "draft"],
)
@pytest.mark.parametrize("mode", ["greedy", "top_tokens", "random"])
def test_spec_verify_chunk_uses_what_the_processors_return(
    mode: str, change: dict, expected: list[int]
) -> None:
    runner = _runner(all_greedy=mode != "random", spec_k=2)
    fused = MagicMock(side_effect=runner.spec_bonus_and_target_logits)
    gathered = MagicMock(side_effect=runner.spec_gather_bonus_and_target_logits)
    runner.spec_bonus_and_target_logits = fused
    runner.spec_gather_bonus_and_target_logits = gathered
    logits = torch.full((4, 128), -20.0)
    for row, tok in enumerate((30, 31, 32)):
        logits[row, tok] = 6.0
    values = torch.argmax(logits, -1, keepdim=True) if mode == "top_tokens" else logits
    events: list[str] = []

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
        [_SpecRecorder("p1", events), _SpecRecorder("p2", events, **change)],
    )

    assert out[0].tolist() == expected
    prepare = "prepare_tokens" if mode == "top_tokens" else "prepare_logits"
    assert events == [
        f"p1.{prepare}",
        f"p2.{prepare}",
        f"p1.finish:{expected}",
        f"p2.finish:{expected}",
    ]
    # A chunk with processors never takes the fused greedy verify prelude.
    assert fused.call_count == 0
    assert gathered.call_count == (0 if mode == "top_tokens" else 1)


@pytest.mark.parametrize("all_greedy", [True, False], ids=["greedy", "random"])
def test_spec_verify_chunk_processors_run_after_grammar_and_min_tokens(
    all_greedy: bool,
) -> None:
    runner = _runner(all_greedy=all_greedy, spec_k=2)
    runner.structured_decoding_manager = _grammar_blocking(32)
    # The bonus row prefers token 32 (grammar-blocked), then the stop token 12
    # (below min_tokens), then token 33.
    logits = torch.full((4, 128), -20.0)
    logits[0, 30] = logits[1, 31] = 6.0
    logits[2, 32], logits[2, 12], logits[2, 33] = 7.0, 6.5, 6.0
    seen: list[tuple[float, float]] = []

    class _Recorder(_SpecRecorder):
        def prepare_logits(self, target_logits, bonus_logits, *args):
            seen.append((bonus_logits[0, 32].item(), bonus_logits[0, 12].item()))
            return super().prepare_logits(target_logits, bonus_logits, *args)

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
        [_Recorder("p1", [], bonus=35)],
    )

    assert seen == [(NEG_INF, NEG_INF)]
    assert out[0].tolist() == [30, 31, 35]


# Processors and the packed top-k candidate paths: the fused verifies have no
# processor hooks, and packed candidates are neither logits nor greedy tokens.


class _TpuTopTokens:
    """Greedy top tokens whose device reads as TPU, so the TPU-only fused
    verify branch is the one under test."""

    def __init__(self, values: torch.Tensor):
        self.values = values
        self.device = SimpleNamespace(type="tpu")


def test_spec_verify_chunk_top_tokens_with_processors_skips_the_fused_verify(
    monkeypatch,
) -> None:
    runner = _runner(spec_k=2)
    fused = MagicMock(side_effect=AssertionError("fused verify with processors"))
    monkeypatch.setattr(tpu_runner, "_fused_greedy_verify_from_top_tokens", fused)
    gather = runner.spec_bonus_and_target_tokens
    runner.spec_bonus_and_target_tokens = lambda tokens, *args: gather(
        tokens.values, *args
    )
    logits = torch.full((4, 128), -20.0)
    for row, tok in enumerate((30, 31, 32)):
        logits[row, tok] = 6.0
    tokens = _TpuTopTokens(torch.argmax(logits, -1, keepdim=True))
    args = (
        _verify_md([30, 31]),
        None,
        SimpleNamespace(scheduled_spec_decode_tokens={}),
        0,
        1,
        True,
        torch.Generator().manual_seed(0),
        True,
        {},
    )
    events: list[str] = []

    out = runner._sample_spec_verify_chunk(
        tokens, *args, [_SpecRecorder("p1", events, bonus=33)]
    )

    assert out[0].tolist() == [30, 31, 33]
    assert events == ["p1.prepare_tokens", "p1.finish:[30, 31, 33]"]
    # Without processors the same TPU chunk takes the fused verify.
    fused.side_effect = None
    fused.return_value = sentinel = torch.zeros((4, 3), dtype=torch.int64)
    assert runner._sample_spec_verify_chunk(tokens, *args) is sentinel


class _ReachedGather(Exception):
    """Raised by the full-logits gather: the chunk took the non-fused path."""


def test_spec_verify_chunk_with_processors_skips_the_candidate_verify() -> None:
    runner = _runner(all_greedy=False, spec_k=2)
    runner._sampler_topk_candidates = lambda *_args: 1
    runner._build_padded_sampling_params = (
        lambda _start, _end, logits, padded_num_reqs=None: (
            torch.full((padded_num_reqs or logits.shape[0], 1), 0.7),
            torch.full((padded_num_reqs or logits.shape[0], 1), 2, dtype=torch.int32),
            torch.ones((padded_num_reqs or logits.shape[0], 1)),
        )
    )
    fused = MagicMock(return_value=torch.zeros((4, 3), dtype=torch.int64))
    runner.rejection_sampler.fused_verify_and_sample_candidates = fused
    runner.spec_gather_bonus_and_target_logits = MagicMock(side_effect=_ReachedGather)
    args = (
        torch.zeros((4, 2 * CANDIDATE_BLOCK)),
        _verify_md([30, 31]),
        None,
        SimpleNamespace(scheduled_spec_decode_tokens={}),
        0,
        1,
        False,
        torch.Generator().manual_seed(0),
        False,
        {},
    )

    # A sampled chunk within the candidate blocks takes the fused verify ...
    assert runner._sample_spec_verify_chunk(*args) is fused.return_value
    # ... unless it has processors: they need the gathered logits.
    with pytest.raises(_ReachedGather):
        runner._sample_spec_verify_chunk(*args, [_SpecRecorder("p1", [])])
    assert fused.call_count == 1


def test_packed_topk_candidates_never_meet_a_processor() -> None:
    runner = _runner(all_greedy=False, spec_k=2)
    packed = torch.zeros((4, 64))
    events: list[str] = []

    with pytest.raises(AssertionError, match="reached a plain chunk"):
        runner._sample_plain_chunk(
            packed,
            cur_start_idx=0,
            cur_end_idx=1,
            top_tokens=False,
            all_greedy=False,
            sampling_generator=None,
            grammar_output=None,
            scheduler_output=SimpleNamespace(scheduled_spec_decode_tokens={}),
            is_spec_step=False,
            min_toks={},
            processors=[_PlainRecorder("p1", 50, 100.0, events)],
            topk_candidates_packed=True,
        )
    with pytest.raises(AssertionError, match="reached a verify chunk"):
        runner._sample_spec_verify_chunk(
            packed,
            _verify_md([30, 31]),
            None,
            SimpleNamespace(scheduled_spec_decode_tokens={}),
            0,
            1,
            False,
            None,
            False,
            {},
            [_SpecRecorder("p1", events)],
            True,
        )
    # Neither path ran a hook on the packed values.
    assert events == []


def _sample_packed(packed, temperatures, u, top_k, top_p):
    runner = SimpleNamespace(
        _apply_temperature=TPUModelRunner._apply_temperature.__get__(object())
    )
    fn = inspect.unwrap(TPUModelRunner.sample_from_packed_candidates)
    return fn(runner, packed, temperatures, u, top_k, top_p)


def _sample_packed_before_the_tie_rule(packed, temperatures, u, top_k, top_p):
    """sample_from_packed_candidates as it was before greedy rows took the
    lowest tied token id: the reference for every other row."""
    num_cand = packed.shape[-1] // 2
    cand_logits = packed[:, :num_cand]
    cand_token_ids = packed[:, num_cand:].to(torch.int64)
    is_greedy = temperatures <= tpu_runner.SAMPLING_EPS
    scaled = cand_logits / torch.where(temperatures == 0.0, 1.0, temperatures)
    masked = tpu_runner.apply_top_k_top_p(
        scaled, top_k, top_p, max_k=min(32, scaled.shape[-1])
    )
    tiny, eps = torch.finfo(u.dtype).tiny, torch.finfo(u.dtype).eps
    noisy = masked - torch.log(-torch.log(torch.clamp(u, min=tiny, max=1.0 - eps)))
    final = torch.where(is_greedy, cand_logits, noisy)
    return torch.gather(cand_token_ids, -1, torch.argmax(final, dim=-1, keepdim=True))


def test_packed_candidates_greedy_rows_take_the_lowest_tied_token_id() -> None:
    # The packed producer lists tied maxima in token-id order; the sampler
    # keeps the rule for any order, here token 50 before token 7.
    logits = torch.tensor([[1.0, 3.0, 2.0, 3.0], [1.0, 3.0, 2.0, 3.0]])
    token_ids = torch.tensor([[9.0, 50.0, 8.0, 7.0], [9.0, 50.0, 8.0, 7.0]])
    packed = torch.cat([logits, token_ids], dim=-1)
    temperatures = torch.tensor([[0.0], [0.7]])
    u = torch.full((2, 4), 0.5)
    top_k = torch.tensor([[1], [1]], dtype=torch.int32)
    top_p = torch.ones((2, 1))
    out = _sample_packed(packed, temperatures, u, top_k, top_p)
    # The greedy row takes the full vocabulary's first argmax among the tie.
    assert out[0].tolist() == [7]
    # The sampled row is unchanged.
    ref = _sample_packed_before_the_tie_rule(packed, temperatures, u, top_k, top_p)
    assert out[1].tolist() == ref[1].tolist()


@pytest.mark.tpu_test
@pytest.mark.parametrize("u_in_graph", [False, True], ids=["u_given", "u_in_graph"])
@pytest.mark.parametrize("rows", [8, 16, 32])
def test_compiled_packed_sampler_on_tpu_matches_the_host(rows, u_in_graph) -> None:
    """The compiled sample_from_packed_candidates gives the host's tokens on
    TPU at 256 candidates: greedy rows (one with a tie, the higher token id
    listed first) and top-k 1 sampled rows, whose token depends on neither
    the noise nor its last bits. `u_in_graph` runs the variant serving
    compiles, which draws its uniforms in-graph (u=None). The module runs
    every test under force_eager, so this one restores the default stance
    around the TPU call."""
    gen = torch.Generator().manual_seed(rows)
    num_cand = 256
    logits = torch.randn((rows, num_cand), generator=gen)
    token_ids = (
        torch.randperm(248320, generator=gen)[: rows * num_cand]
        .reshape(rows, num_cand)
        .to(torch.float32)
    )
    logits[0, 3] = logits[0, 9] = logits[0].max() + 1.0
    token_ids[0, 3], token_ids[0, 9] = 900.0, 12.0
    packed = torch.cat([logits, token_ids], dim=-1)
    temperatures = torch.full((rows, 1), 0.7)
    temperatures[: rows // 2] = 0.0
    u = torch.rand((rows, num_cand), generator=gen)
    top_k = torch.ones((rows, 1), dtype=torch.int32)
    top_p = torch.ones((rows, 1))
    host = _sample_packed(packed, temperatures, u, top_k, top_p)
    assert host[0].item() == 12
    runner = SimpleNamespace(
        _apply_temperature=TPUModelRunner._apply_temperature.__get__(object())
    )
    device = torch.device("tpu")
    with torch.compiler.set_stance("default"):
        out = TPUModelRunner.sample_from_packed_candidates(
            runner,
            packed.to(device),
            temperatures.to(device),
            None if u_in_graph else u.to(device),
            top_k.to(device),
            top_p.to(device),
        )
    assert torch.equal(out.cpu(), host)


@pytest.mark.parametrize("u_in_graph", [False, True], ids=["u_given", "u_in_graph"])
@pytest.mark.parametrize("rows", [8, 16])
def test_the_traced_packed_sampler_matches_the_host(rows, u_in_graph) -> None:
    """The CPU counterpart of the TPU test above: dynamo traces
    sample_from_packed_candidates whole (fullgraph, static shapes) under the
    default stance and the eager backend runs the captured graph, which gives
    the host's tokens for greedy rows (one with a tie, the higher token id
    listed first) and top-k 1 sampled rows. It covers the traced program, not
    its TPU lowering, which only the TPU test runs."""
    fn = inspect.unwrap(TPUModelRunner.sample_from_packed_candidates)
    gen = torch.Generator().manual_seed(rows)
    num_cand = 64
    logits = torch.randn((rows, num_cand), generator=gen)
    token_ids = (
        torch.randperm(4096, generator=gen)[: rows * num_cand]
        .reshape(rows, num_cand)
        .to(torch.float32)
    )
    logits[0, 3] = logits[0, 9] = logits[0].max() + 1.0
    token_ids[0, 3], token_ids[0, 9] = 900.0, 12.0
    packed = torch.cat([logits, token_ids], dim=-1)
    temperatures = torch.full((rows, 1), 0.7)
    temperatures[: rows // 2] = 0.0
    u = torch.rand((rows, num_cand), generator=gen)
    top_k = torch.ones((rows, 1), dtype=torch.int32)
    top_p = torch.ones((rows, 1))
    host = _sample_packed(packed, temperatures, u, top_k, top_p)
    assert host[0].item() == 12
    runner = SimpleNamespace(
        _apply_temperature=TPUModelRunner._apply_temperature.__get__(object())
    )
    torch._dynamo.reset()
    with torch.compiler.set_stance("default"):
        traced = torch.compile(fn, backend="eager", fullgraph=True, dynamic=False)
        out = traced(
            runner, packed, temperatures, None if u_in_graph else u, top_k, top_p
        )
    assert torch.equal(out, host)


@pytest.mark.parametrize("seed", range(20))
def test_packed_candidates_rows_without_a_greedy_tie_sample_as_before(seed) -> None:
    """Sampled rows, and greedy rows whose maximum is unique, give exactly the
    token they gave before the tie rule."""
    gen = torch.Generator().manual_seed(seed)
    rows, num_cand = 16, 64
    logits = torch.randn((rows, num_cand), generator=gen)
    token_ids = torch.randperm(4096, generator=gen)[: rows * num_cand]
    packed = torch.cat(
        [logits, token_ids.reshape(rows, num_cand).to(torch.float32)], dim=-1
    )
    temperatures = torch.where(torch.rand((rows, 1), generator=gen) < 0.3, 0.0, 0.7).to(
        torch.float32
    )
    u = torch.rand((rows, num_cand), generator=gen)
    top_k = torch.randint(1, 33, (rows, 1), generator=gen, dtype=torch.int32)
    top_p = torch.where(torch.rand((rows, 1), generator=gen) < 0.5, 0.95, 1.0)
    out = _sample_packed(packed, temperatures, u, top_k, top_p)
    ref = _sample_packed_before_the_tie_rule(packed, temperatures, u, top_k, top_p)
    assert torch.equal(out, ref)


def test_in_graph_draws_seed_the_device_generator_or_refuse(monkeypatch) -> None:
    """The candidate samplers draw in-graph from the TPU default generator;
    every rank seeds it alike, and a missing generator is an error, not a
    silent skip."""
    seeded = []

    class _Generator:
        def manual_seed(self, seed):
            seeded.append(seed)

    runner = SimpleNamespace(
        _sampling_generator=None,
        model_config=SimpleNamespace(seed=7),
        device=torch.device("cpu"),
        _sampler_draws_in_graph=lambda: True,
    )
    monkeypatch.setattr(
        torch, "tpu", SimpleNamespace(default_generators=[_Generator()]), raising=False
    )
    TPUModelRunner._get_sampling_generator(runner)
    assert seeded == [7]

    runner._sampling_generator = None
    monkeypatch.setattr(
        torch, "tpu", SimpleNamespace(default_generators=[]), raising=False
    )
    with pytest.raises(RuntimeError, match="TPU device 0"):
        TPUModelRunner._get_sampling_generator(runner)
