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
"""Tests for the per-chunk processor hooks of ``TPUModelRunner`` sampling."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from vllm import SamplingParams

from vllm_torchtpu.layers.adapter.sample.rejection_sampler import RejectionSampler
from vllm_torchtpu.runner import chunk_processors, min_tokens
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

NEG_INF = float("-inf")


@pytest.fixture(autouse=True)
def _eager_kernels():
    """Run the ``@torch.compile(backend="tpu")`` helpers eagerly on CPU."""
    with torch.compiler.set_stance("force_eager"):
        yield


def _runner(*, all_greedy: bool = True, spec_k: int | None = None) -> SimpleNamespace:
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


class _PlainRecorder:
    """Rewrites column ``col`` to ``value`` and records every hook call."""

    def __init__(self, name: str, col: int, value: float, events: list[str]):
        self.name, self.col, self.value, self.events = name, col, value, events

    def mask_logits(self, logits: torch.Tensor) -> torch.Tensor:
        # The grammar (column 99) and min_tokens (stop token 12) masks come first.
        seen = (logits[0, 99].item(), logits[0, 12].item())
        self.events.append(f"{self.name}.mask{seen}")
        out = logits.clone()
        out[0, self.col] = self.value
        return out

    def on_sampled(self, selected_token_ids: torch.Tensor) -> None:
        self.events.append(f"{self.name}.sampled:{int(selected_token_ids[0, 0])}")

    def override_top_tokens(self, selected_token_ids: torch.Tensor) -> torch.Tensor:
        self.events.append(f"{self.name}.override:{int(selected_token_ids[0, 0])}")
        return selected_token_ids * 2 + int(self.value > 0)


def test_plain_chunk_hooks_run_in_order_after_logprob_capture() -> None:
    runner = _runner()
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
    tracker = min_tokens.MinTokensTracker()
    tracker.add("r0", SamplingParams(min_tokens=2, stop_token_ids=[12]))
    min_toks = tracker.batch_state({"r0": 0}, lambda _req_id: 0)
    raw = torch.zeros((4, 128))
    raw[0, 99], raw[0, 12], raw[0, 50] = 20.0, 9.0, -3.0
    events: list[str] = []
    # The second processor masks token 50, which the first one promoted.
    procs = [
        _PlainRecorder("p1", 50, 100.0, events),
        _PlainRecorder("p2", 50, NEG_INF, events),
    ]

    for is_spec_step in (False, True):
        events.clear()
        selected, logprob_logits = runner._sample_plain_chunk(
            raw.clone(),
            cur_start_idx=0,
            cur_end_idx=1,
            top_tokens=False,
            all_greedy=True,
            sampling_generator=None,
            grammar_output=object(),
            scheduler_output=SimpleNamespace(scheduled_spec_decode_tokens={"r0": [5]}),
            is_spec_step=is_spec_step,
            min_toks=min_toks,
            processors=procs,
        )
        # p1 then p2 masked logits that already had the grammar and min_tokens
        # masks; sampling saw p2's mask, and on_sampled ran once each.
        assert events == [
            f"p1.mask({NEG_INF}, {NEG_INF})",
            f"p2.mask({NEG_INF}, {NEG_INF})",
            "p1.sampled:0",
            "p2.sampled:0",
        ]
        assert int(selected[0, 0]) == 0
        # Logprobs see the grammar mask but none of the processors.
        assert logprob_logits[0, 99].item() == NEG_INF
        assert logprob_logits[0, 12].item() == 9.0
        assert logprob_logits[0, 50].item() == -3.0
    assert grammar_calls == ["plain", "spec:{'r0': [5]}"]

    # Top tokens: each processor overrides the previous one's tokens instead.
    events.clear()
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
    assert events == ["p1.override:30", "p2.override:61"]
    assert int(selected[0, 0]) == 122


def test_stateless_processor_wraps_a_mask() -> None:
    proc = chunk_processors.StatelessPlainChunkProcessor(lambda lg: lg + 1)
    assert proc.mask_logits(torch.zeros(2)).tolist() == [1.0, 1.0]
    assert proc.on_sampled(torch.zeros(2)) is None
    with pytest.raises(NotImplementedError, match="top_tokens"):
        proc.override_top_tokens(torch.zeros((2, 1)))


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


class _SpecRecorder:
    def __init__(self, name: str, events: list[str], invalid_draft: int | None):
        self.name, self.events, self.invalid_draft = name, events, invalid_draft

    def _drafts(self, draft_token_ids: torch.Tensor) -> torch.Tensor:
        if self.invalid_draft is None:
            return draft_token_ids
        return torch.where(
            draft_token_ids == self.invalid_draft, -1, draft_token_ids
        ).to(draft_token_ids.dtype)

    def prepare_logits(self, target_logits, bonus_logits, draft_token_ids, *_args):
        self.events.append(f"{self.name}.prepare_logits")
        return target_logits, bonus_logits, self._drafts(draft_token_ids)

    def prepare_tokens(self, target_argmax, bonus_token_ids, draft_token_ids, *_args):
        self.events.append(f"{self.name}.prepare_tokens")
        return target_argmax, bonus_token_ids, self._drafts(draft_token_ids)

    def finish(self, next_tokens: torch.Tensor, max_draft_tokens: int) -> None:
        self.events.append(f"{self.name}.finish:{max_draft_tokens}")


@pytest.mark.parametrize("mode", ["greedy", "top_tokens", "random"])
def test_spec_verify_chunk_hooks_run_once_around_rejection(mode: str) -> None:
    runner = _runner(all_greedy=mode != "random", spec_k=2)
    fused, gathered = MagicMock(), MagicMock()
    fused.side_effect = runner.spec_bonus_and_target_logits
    gathered.side_effect = runner.spec_gather_bonus_and_target_logits
    runner.spec_bonus_and_target_logits = fused
    runner.spec_gather_bonus_and_target_logits = gathered
    logits = torch.full((4, 128), -20.0)
    for row, tok in enumerate((30, 31, 32)):
        logits[row, tok] = 6.0
    values = torch.argmax(logits, -1, keepdim=True) if mode == "top_tokens" else logits

    def verify(processors) -> torch.Tensor:
        return runner._sample_spec_verify_chunk(
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
            processors,
        )

    no_procs = verify(())
    events: list[str] = []
    # The second processor rejects draft 31, so the bonus token is dropped.
    out = verify([_SpecRecorder("p1", events, None), _SpecRecorder("p2", events, 31)])
    prepare = "prepare_tokens" if mode == "top_tokens" else "prepare_logits"
    assert events == [f"p1.{prepare}", f"p2.{prepare}", "p1.finish:2", "p2.finish:2"]
    if mode == "random":
        return
    assert no_procs[0].tolist() == [30, 31, 32]
    assert out[0].tolist() == [30, 31, -1]
    # Greedy chunks keep the fused verify prelude only without processors.
    if mode == "greedy":
        assert (fused.call_count, gathered.call_count) == (1, 1)
