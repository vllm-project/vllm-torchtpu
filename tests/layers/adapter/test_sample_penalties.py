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
"""Repetition / frequency / presence penalties on ``[rows, vocab]`` logits,
checked against vLLM's own ``apply_penalties``."""

import pytest
import torch
from vllm.model_executor.layers.utils import apply_penalties

from vllm_torchtpu.layers.adapter.sample import penalties

VOCAB = 32
NEG_INF = float("-inf")


@pytest.fixture
def eager():
    with torch.compiler.set_stance("force_eager"):
        yield


def _pad(rows, value):
    width = max(1, max(len(r) for r in rows))
    return torch.tensor([r + [value] * (width - len(r)) for r in rows])


def vllm_reference(logits, prompts, outputs, rep, pres, freq):
    """vLLM's apply_penalties on float32 logits, as its sampler calls it (the
    sampler upcasts first; padding is the vocabulary size)."""
    return apply_penalties(
        logits.to(torch.float32).clone(),
        _pad(prompts, VOCAB),
        _pad(outputs, VOCAB),
        torch.tensor(pres, dtype=torch.float32),
        torch.tensor(freq, dtype=torch.float32),
        torch.tensor(rep, dtype=torch.float32),
    )


def encoded_tables(prompts, outputs, width):
    """The serving layout: prompt ids, output ids as -(id + 2), -1 padding,
    split into [rows, width] tables (one per pass), flat as the runner
    uploads them."""
    rows = [p + [-t - 2 for t in o] for p, o in zip(prompts, outputs)]
    passes = max(1, -(-max(len(r) for r in rows) // width))
    padded = _pad([r + [-1] * (passes * width - len(r)) for r in rows], -1)
    return [
        padded[:, p * width : (p + 1) * width].to(torch.int32).reshape(-1)
        for p in range(passes)
    ]


def serving_path(logits, prompts, outputs, rep, pres, freq, width):
    tables = encoded_tables(prompts, outputs, width)
    seen, counts = penalties.penalty_counts(logits, tables[0])
    for table in tables[1:]:
        seen, counts = penalties.add_penalty_counts(seen, counts, table)
    params = torch.tensor(list(zip(rep, pres, freq)), dtype=torch.float32)
    return penalties.apply_penalty_counts(logits, seen, counts, params.reshape(-1))


def export_path(logits, prompts, outputs, rep, pres, freq):
    history = _pad([p + o for p, o in zip(prompts, outputs)], -1)
    return penalties.penalize_logits(
        logits,
        history,
        torch.tensor([len(p) for p in prompts], dtype=torch.int32),
        torch.tensor(rep),
        torch.tensor(pres),
        torch.tensor(freq),
    )


PROMPTS = [[1, 2, 2, 9], [3, 4], [5], [6, 7, 8]]
OUTPUTS = [[2, 7, 7, 7], [], [5, 5, 11], [8, 1, 1]]

CASES = {
    "repetition": ([1.3] * 4, [0.0] * 4, [0.0] * 4),
    "presence": ([1.0] * 4, [0.5] * 4, [0.0] * 4),
    "frequency": ([1.0] * 4, [0.0] * 4, [0.7] * 4),
    "per_row": ([1.3, 1.0, 0.8, 2.0], [0.0, 0.4, -0.3, 0.1], [0.2, 0.0, 0.5, -0.1]),
    "repetition_below_one": ([0.7] * 4, [0.0] * 4, [0.0] * 4),
    "neutral": ([1.0] * 4, [0.0] * 4, [0.0] * 4),
}


def _logits(dtype=torch.float32, masked=False):
    gen = torch.Generator().manual_seed(0)
    logits = torch.randn(4, VOCAB, generator=gen) * 3
    if masked:  # e.g. min_tokens or a grammar set some logits to -inf first
        logits[0, 2] = logits[2, 5] = logits[3, 0] = NEG_INF
    return logits.to(dtype)


@pytest.mark.usefixtures("eager")
class TestAgainstVllm:
    @pytest.mark.parametrize("case", list(CASES))
    @pytest.mark.parametrize("width", [64, 4, 3])  # 1, 3 and 4 passes
    def test_serving_path(self, case, width):
        rep, pres, freq = CASES[case]
        logits = _logits()
        got = serving_path(logits, PROMPTS, OUTPUTS, rep, pres, freq, width)
        want = vllm_reference(logits, PROMPTS, OUTPUTS, rep, pres, freq)
        torch.testing.assert_close(got, want)

    @pytest.mark.parametrize("case", list(CASES))
    def test_export_path(self, case):
        rep, pres, freq = CASES[case]
        logits = _logits()
        got = export_path(logits, PROMPTS, OUTPUTS, rep, pres, freq)
        torch.testing.assert_close(
            got, vllm_reference(logits, PROMPTS, OUTPUTS, rep, pres, freq)
        )

    def test_bfloat16_logits_are_penalized_in_float32(self):
        """vLLM's sampler penalizes float32 logits; a small additive penalty
        would round away if the result went back to bfloat16."""
        rep, pres, freq = CASES["per_row"]
        logits = _logits(torch.bfloat16)
        got = serving_path(logits, PROMPTS, OUTPUTS, rep, pres, freq, width=4)
        assert got.dtype == torch.float32
        torch.testing.assert_close(
            got, vllm_reference(logits, PROMPTS, OUTPUTS, rep, pres, freq)
        )
        small = serving_path(
            torch.full((1, VOCAB), 20.0, dtype=torch.bfloat16),
            [[]],
            [[3]],
            [1.0],
            [0.05],
            [0.0],
            width=4,
        )
        assert small[0, 3] < small[0, 4]  # 19.95 vs 20 survives
        assert torch.tensor(19.95).to(torch.bfloat16) == 20.0  # it would not

    def test_masked_logits_stay_masked(self):
        rep, pres, freq = CASES["per_row"]
        logits = _logits(masked=True)
        got = serving_path(logits, PROMPTS, OUTPUTS, rep, pres, freq, width=4)
        want = vllm_reference(logits, PROMPTS, OUTPUTS, rep, pres, freq)
        torch.testing.assert_close(got, want)
        assert torch.isneginf(got[0, 2]) and torch.isneginf(got[2, 5])

    def test_repeated_ids_count_once_for_presence_and_per_occurrence_for_frequency(
        self,
    ):
        logits = torch.zeros(1, VOCAB)
        got = serving_path(logits, [[4, 4]], [[6, 6, 6]], [1.0], [1.0], [0.5], width=2)
        assert got[0, 6].item() == pytest.approx(-1.0 - 1.5)
        assert got[0, 4].item() == 0.0  # prompt only: no presence/frequency

    def test_padding_rows_are_neutral(self):
        logits = _logits()
        table = torch.full((4 * 8,), penalties.PAD_TOKEN, dtype=torch.int32)
        seen, counts = penalties.penalty_counts(logits, table)
        params = torch.tensor([1.0, 0.0, 0.0] * 4)
        assert torch.equal(
            penalties.apply_penalty_counts(logits, seen, counts, params), logits
        )


@pytest.mark.tpu_test
def test_compiled_penalties_on_tpu_match_cpu():
    """scatter_add with repeated ids is the part that can differ on TPU."""
    import torch_tpu  # noqa: F401

    rep, pres, freq = CASES["per_row"]
    prompts = [p * 300 for p in PROMPTS]  # repeated ids, several passes
    outputs = [o * 200 for o in OUTPUTS]
    gen = torch.Generator().manual_seed(0)
    logits = torch.randn(4, 1024, generator=gen) * 3
    width = 512
    with torch.compiler.set_stance("force_eager"):
        ref = serving_path(logits, prompts, outputs, rep, pres, freq, width)
    tpu = torch.device("tpu")
    tables = [t.to(tpu) for t in encoded_tables(prompts, outputs, width)]
    seen, counts = penalties.penalty_counts(logits.to(tpu), tables[0])
    for table in tables[1:]:
        seen, counts = penalties.add_penalty_counts(seen, counts, table)
    params = torch.tensor(list(zip(rep, pres, freq)), dtype=torch.float32)
    params = params.reshape(-1).to(tpu)
    got = penalties.apply_penalty_counts(logits.to(tpu), seen, counts, params).cpu()
    torch.testing.assert_close(got, ref)
