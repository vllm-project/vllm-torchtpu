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

import pytest
import torch
from sampling_test_utils import _sampling_runner, _sampling_step, _verify_md
from vllm import SamplingParams

from vllm_torchtpu.runner import chunk_sampling, min_tokens

NEG_INF = float("-inf")


@pytest.fixture(autouse=True)
def _eager_kernels():
    """Run the ``@torch.compile(backend="tpu")`` helpers eagerly on CPU."""
    with torch.compiler.set_stance("force_eager"):
        yield


def _below_min_tokens(stop_token: int) -> min_tokens.MinTokensState:
    """Batch row 0 is a request with no output yet, below ``min_tokens=10``."""
    tracker = min_tokens.MinTokensTracker()
    tracker.add("r0", SamplingParams(min_tokens=10, stop_token_ids=[stop_token]))
    return tracker.batch_state({"r0": 0}, lambda _req_id: 0)


@pytest.mark.parametrize("all_greedy", [True, False], ids=["greedy", "sampled"])
def test_plain_chunk_masks_then_samples(all_greedy: bool) -> None:
    runner = _sampling_runner(temperature=0.7)
    grammar_calls: list[str] = []

    def block_99(logits: torch.Tensor) -> torch.Tensor:
        out = logits.clone()
        out[0, 99] = NEG_INF
        return out

    def mask_logits(logits, _grammar_output, _start, _end):
        grammar_calls.append("plain")
        return block_99(logits)

    runner.structured_decoding_manager = SimpleNamespace(mask_logits=mask_logits)
    # Row 0 prefers token 99 (grammar-blocked), then the stop token 12 (below
    # min_tokens), then token 40; every other logit is far below token 40, so
    # sampling picks it too.
    raw = torch.full((4, 128), -20.0)
    raw[0, 99], raw[0, 12], raw[0, 40] = 20.0, 9.0, 6.0

    step = _sampling_step(
        all_greedy=all_greedy,
        sampling_generator=torch.Generator().manual_seed(0),
        grammar_output=object(),
        min_toks=_below_min_tokens(12),
    )
    selected, logprob_logits = chunk_sampling.sample_plain_chunk(
        runner, step, raw.clone(), cur_start_idx=0, cur_end_idx=1
    )

    assert int(selected[0, 0]) == 40
    assert grammar_calls == ["plain"]
    # Logprobs see the grammar mask but not the min_tokens mask.
    assert logprob_logits[0, 99].item() == NEG_INF
    assert logprob_logits[0, 12].item() == 9.0
    assert logprob_logits[0, 40].item() == 6.0


def test_plain_chunk_top_tokens_returns_the_greedy_tokens() -> None:
    tokens = torch.tensor([[30], [0], [0], [0]])

    selected, logprob_values = chunk_sampling.sample_plain_chunk(
        _sampling_runner(),
        _sampling_step(top_tokens=True),
        tokens,
        cur_start_idx=0,
        cur_end_idx=1,
    )

    assert selected is tokens and logprob_values is tokens


@pytest.mark.parametrize(
    "mode, top_k",
    [("greedy", 0), ("top_tokens", 0), ("random", 2), ("random", 0)],
    ids=["greedy", "top_tokens", "random_top_k", "random"],
)
def test_spec_verify_chunk_accepts_drafts_and_samples_the_bonus(
    mode: str, top_k: int
) -> None:
    runner = _sampling_runner(temperature=0.7, top_k=top_k, spec_k=2)
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

    step = _sampling_step(
        top_tokens=mode == "top_tokens",
        all_greedy=mode != "random",
        sampling_generator=torch.Generator().manual_seed(0),
    )
    out = chunk_sampling.sample_spec_verify_chunk(
        runner, step, values, _verify_md([[30, 31]]), 0, 1
    )

    assert out[0].tolist() == [30, 31, 32]
    # Only an all-greedy chunk on the full logits takes the fused verify prelude.
    expected_preludes = {"greedy": (1, 0), "top_tokens": (0, 0), "random": (0, 1)}
    assert (fused.call_count, gathered.call_count) == expected_preludes[mode]


@pytest.mark.parametrize("all_greedy", [True, False], ids=["greedy", "random"])
def test_spec_verify_chunk_applies_grammar_then_min_tokens(all_greedy: bool) -> None:
    runner = _sampling_runner(temperature=0.7, spec_k=2)
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

    step = _sampling_step(
        all_greedy=all_greedy,
        sampling_generator=torch.Generator().manual_seed(0),
        grammar_output=object(),
        scheduler_output=SimpleNamespace(scheduled_spec_decode_tokens={"r0": [30, 31]}),
        min_toks=_below_min_tokens(12),
    )
    out = chunk_sampling.sample_spec_verify_chunk(
        runner, step, logits, _verify_md([[30, 31]]), 0, 1
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
def test_plain_chunk_processors_run_in_order_after_the_masks(
    all_greedy: bool,
) -> None:
    runner = _sampling_runner(temperature=0.7)
    runner.structured_decoding_manager = _grammar_blocking(99)
    raw = torch.full((4, 128), -20.0)
    raw[0, 99], raw[0, 12], raw[0, 50], raw[0, 40] = 20.0, 9.0, -3.0, 6.0
    events: list[str] = []
    # p1 promotes token 50 over token 40; p2 masks it again.
    procs = [
        _PlainRecorder("p1", 50, 100.0, events),
        _PlainRecorder("p2", 50, NEG_INF, events),
    ]

    step = _sampling_step(
        all_greedy=all_greedy,
        sampling_generator=torch.Generator().manual_seed(0),
        grammar_output=object(),
        min_toks=_below_min_tokens(12),
    )
    selected, logprob_logits = chunk_sampling.sample_plain_chunk(
        runner, step, raw.clone(), cur_start_idx=0, cur_end_idx=1, processors=procs
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
    events: list[str] = []
    procs = [
        _PlainRecorder("p1", 50, 100.0, events),
        _PlainRecorder("p2", 50, NEG_INF, events),
    ]

    selected, _ = chunk_sampling.sample_plain_chunk(
        _sampling_runner(),
        _sampling_step(top_tokens=True),
        torch.tensor([[30], [0], [0], [0]]),
        cur_start_idx=0,
        cur_end_idx=1,
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
    runner = _sampling_runner(temperature=0.7, spec_k=2)
    fused = MagicMock(side_effect=runner.spec_bonus_and_target_logits)
    gathered = MagicMock(side_effect=runner.spec_gather_bonus_and_target_logits)
    runner.spec_bonus_and_target_logits = fused
    runner.spec_gather_bonus_and_target_logits = gathered
    logits = torch.full((4, 128), -20.0)
    for row, tok in enumerate((30, 31, 32)):
        logits[row, tok] = 6.0
    values = torch.argmax(logits, -1, keepdim=True) if mode == "top_tokens" else logits
    events: list[str] = []

    step = _sampling_step(
        top_tokens=mode == "top_tokens",
        all_greedy=mode != "random",
        sampling_generator=torch.Generator().manual_seed(0),
    )
    out = chunk_sampling.sample_spec_verify_chunk(
        runner,
        step,
        values,
        _verify_md([[30, 31]]),
        0,
        1,
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
    runner = _sampling_runner(temperature=0.7, spec_k=2)
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

    step = _sampling_step(
        all_greedy=all_greedy,
        sampling_generator=torch.Generator().manual_seed(0),
        grammar_output=object(),
        scheduler_output=SimpleNamespace(scheduled_spec_decode_tokens={"r0": [30, 31]}),
        min_toks=_below_min_tokens(12),
    )
    out = chunk_sampling.sample_spec_verify_chunk(
        runner,
        step,
        logits,
        _verify_md([[30, 31]]),
        0,
        1,
        [_Recorder("p1", [], bonus=35)],
    )

    assert seen == [(NEG_INF, NEG_INF)]
    assert out[0].tolist() == [30, 31, 35]


@pytest.mark.parametrize("filtered", [True, False], ids=["top_k_top_p", "neither"])
def test_spec_verify_chunk_verifies_on_the_params_of_its_bonus_draw(
    filtered: bool,
) -> None:
    """The rejection sampler gets each draft row's request params and the
    sampler variant of the chunk's bonus draw."""
    runner = _sampling_runner(spec_k=2)
    runner.input_batch.no_top_k = runner.input_batch.no_top_p = not filtered
    # Two requests at different temperatures; the bonus rows pad to 4.
    temperatures = torch.tensor([[0.5], [2.0], [1.0], [1.0]])
    top_k = torch.tensor(
        [[3], [5], [0], [0]] if filtered else [[0]] * 4, dtype=torch.int32
    )
    top_p = torch.tensor([[0.9], [0.8], [1.0], [1.0]] if filtered else [[1.0]] * 4)
    runner._build_padded_sampling_params = lambda *_args: (temperatures, top_k, top_p)
    candidates = 4 if filtered else 0
    runner._sampler_topk_candidates = lambda *_args: candidates
    bonus_variants = []

    def sample_bonus(logits, *_params, **variant):
        bonus_variants.append(variant)
        return torch.argmax(logits, dim=-1, keepdim=True)

    runner.sample_from_logits_func = sample_bonus
    runner.rejection_sampler = MagicMock(synthetic_mode=False)
    md = _verify_md([[30, 31], [40]])

    chunk_sampling.sample_spec_verify_chunk(
        runner, _sampling_step(all_greedy=False), torch.zeros(8, 128), md, 0, 2
    )

    verified = runner.rejection_sampler.call_args.kwargs
    assert verified["do_sampling"]
    assert torch.equal(verified["temperatures"], temperatures[md.segment_ids])
    assert torch.equal(verified["top_k"], top_k[md.segment_ids])
    assert torch.equal(verified["top_p"], top_p[md.segment_ids])
    (bonus,) = bonus_variants
    assert verified["topk_candidates"] == bonus["topk_candidates"] == candidates
    assert verified["apply_topk_topp"] is bonus["apply_topk_topp"] is filtered


@pytest.mark.parametrize(
    "top_k, accepted", [(0, False), (1, True)], ids=["near_uniform", "top_k_1"]
)
def test_sampled_verify_of_an_argmax_draft(top_k: int, accepted: bool) -> None:
    """Draft 30 is the target argmax, which greedy verify accepts. At
    temperature 100 the target is near uniform (the draft holds about 1/256
    of the mass), and the seeded noise rejects the draft; top-k 1 gives the
    argmax all the mass, so the chunk verifies as greedy does."""
    runner = _sampling_runner(temperature=100.0, top_k=top_k, spec_k=2)
    padded_params = runner._build_padded_sampling_params
    bounds = []

    def chunk_params(start, end, logits):
        bounds.append((start, end))
        return padded_params(start, end, logits)

    runner._build_padded_sampling_params = chunk_params
    # One request, at batch row 3: two draft rows, then its bonus row.
    logits = torch.zeros(4, 256)
    for row, tok in enumerate((30, 31, 32)):
        logits[row, tok] = 1.0
    md = _verify_md([[30, 31]], pad_reqs=2)

    def verify(all_greedy: bool) -> list[int]:
        step = _sampling_step(
            all_greedy=all_greedy, sampling_generator=torch.Generator().manual_seed(0)
        )
        out = chunk_sampling.sample_spec_verify_chunk(runner, step, logits, md, 3, 4)
        return out[0].tolist()

    greedy = verify(all_greedy=True)
    sampled = verify(all_greedy=False)

    assert greedy == [30, 31, 32]
    if accepted:
        assert sampled == greedy
    else:
        # Rejected at the first draft, which the recovered token never is.
        assert sampled[0] != 30 and sampled[1:] == [-1, -1]
    assert bounds == [(3, 4)]
