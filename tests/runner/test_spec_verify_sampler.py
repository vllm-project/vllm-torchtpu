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
"""The runner hands the non-greedy verify to a model's spec_verify_sampler."""

from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.layers.adapter.sample.rejection_sampler import RejectionSampler
from vllm_torchtpu.runner import spec_verify_sampler as svs
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

K = 3
VOCAB = 64


class _Recorder:
    def __init__(self):
        self.calls: list[svs.SpecVerifyInputs] = []
        self.precompiles: list[tuple[svs.SpecVerifyInputs, list]] = []

    def sample(self, inputs):
        self.calls.append(inputs)
        return torch.full((inputs.temperatures.shape[0], K + 1), 7)

    def precompile(self, inputs, sampler_variants):
        self.precompiles.append((inputs, list(sampler_variants)))


class _Unfused(Exception):
    pass


def _unfused(*args, **kwargs):
    raise _Unfused


def _metadata(num_reqs=2, padded_num_reqs=4, num_tokens=8):
    lens = [K, 1][:num_reqs]
    return SimpleNamespace(
        draft_token_ids=torch.arange(num_tokens, dtype=torch.int32),
        draft_lengths=torch.tensor(lens + [0] * (padded_num_reqs - num_reqs)),
        draft_lengths_cpu=torch.tensor(lens),
        target_logits_indices=torch.zeros(num_tokens, dtype=torch.int32),
        bonus_logits_indices=torch.zeros(padded_num_reqs, dtype=torch.int32),
        segment_ids=torch.zeros(num_tokens, dtype=torch.int64),
        group_indices=torch.zeros(num_tokens, dtype=torch.int32),
    )


def _runner(sampler, no_top_k=True):
    runner = SimpleNamespace(
        _spec_verify_sampler=sampler,
        vocab_size=VOCAB,
        input_batch=SimpleNamespace(
            temperature_cpu_tensor=torch.tensor([1.0, 0.5]),
            top_k_cpu_tensor=torch.tensor([VOCAB, 5 if not no_top_k else VOCAB]),
            top_p_cpu_tensor=torch.tensor([0.9, 1.0]),
            no_top_k=no_top_k,
        ),
        rejection_sampler=RejectionSampler(),
        speculative_config=SimpleNamespace(num_speculative_tokens=K),
        _sampler_topk_candidates=lambda start, end, grammar: 0,
        _sampler_applies_topk_topp=lambda: True,
        spec_bonus_and_target_logits=_unfused,
        spec_gather_bonus_and_target_logits=_unfused,
    )
    for name in ("_sample_spec_verify_model", "_build_padded_sampling_params"):
        setattr(runner, name, getattr(TPUModelRunner, name).__get__(runner))
    return runner


def _verify(runner, md, *, all_greedy=False, processors=(), seed=0):
    return TPUModelRunner._sample_spec_verify_chunk(
        runner,
        torch.randn((8, VOCAB)),
        md,
        grammar_output=None,
        scheduler_output=SimpleNamespace(),
        cur_start_idx=0,
        cur_end_idx=2,
        all_greedy=all_greedy,
        sampling_generator=torch.Generator().manual_seed(seed),
        processors=processors,
    )


@pytest.mark.parametrize("no_top_k", [True, False])
def test_model_sampler_gets_the_runner_draws_in_order(no_top_k):
    sampler = _Recorder()
    md = _metadata()
    out = _verify(_runner(sampler, no_top_k), md, seed=3)
    assert torch.equal(out, torch.full((4, K + 1), 7))
    (inputs,) = sampler.calls
    # The unfused path draws the bonus noise, then accept, then recover.
    g = torch.Generator().manual_seed(3)
    assert torch.equal(inputs.bonus_u, torch.rand((4, VOCAB), generator=g))
    assert torch.equal(inputs.accept_u, torch.rand((8,), generator=g))
    assert torch.equal(inputs.recover_u, torch.rand((8, VOCAB), generator=g))
    # Per-request params padded to the request bucket, neutral padding.
    assert inputs.temperatures.shape == (4, 1)
    assert inputs.temperatures.view(-1).tolist() == [1.0, 0.5, 1.0, 1.0]
    assert inputs.top_k.view(-1).tolist() == [0, 0 if no_top_k else 5, 0, 0]
    assert inputs.any_top_k is (not no_top_k)
    assert inputs.max_draft_tokens == K
    assert inputs.synthetic_mode is False
    assert inputs.topk_candidates == 0 and inputs.apply_topk_topp is True
    assert inputs.draft_token_ids is md.draft_token_ids
    assert inputs.num_draft_tokens is md.draft_lengths


@pytest.mark.parametrize(
    "sampler,all_greedy,processors",
    [
        (None, False, ()),
        (_Recorder(), True, ()),
        (_Recorder(), False, (object(),)),
    ],
)
def test_runner_keeps_its_verify(sampler, all_greedy, processors):
    """No model sampler, an all-greedy chunk or chunk processors: the runner's
    own verify runs."""
    with pytest.raises(_Unfused):
        _verify(
            _runner(sampler), _metadata(), all_greedy=all_greedy, processors=processors
        )
    if sampler is not None:
        assert not sampler.calls


def test_model_spec_verify_sampler():
    attr = svs.SPEC_VERIFY_SAMPLER_ATTR
    assert svs.model_spec_verify_sampler(SimpleNamespace()) is None
    assert svs.model_spec_verify_sampler(SimpleNamespace(**{attr: None})) is None
    sampler = _Recorder()
    model = SimpleNamespace(**{attr: sampler})
    assert svs.model_spec_verify_sampler(model) is sampler
    with pytest.raises(TypeError):
        svs.model_spec_verify_sampler(SimpleNamespace(**{attr: True}))
    # An nn.Module declares the capability by setting the attribute.
    module = torch.nn.Linear(1, 1)
    assert not isinstance(module, svs.SupportsSpecVerifySampler)
    assert svs.model_spec_verify_sampler(module) is None
    module.spec_verify_sampler = sampler
    assert isinstance(module, svs.SupportsSpecVerifySampler)
    assert svs.model_spec_verify_sampler(module) is sampler


@pytest.mark.parametrize("blocks", [0, 2])
def test_precompile_hands_the_sampler_variants(blocks):
    sampler = _Recorder()
    runner = SimpleNamespace(
        _spec_verify_sampler=sampler,
        rejection_sampler=RejectionSampler(),
        speculative_config=SimpleNamespace(num_speculative_tokens=K),
        vocab_size=VOCAB,
        _topk_candidate_blocks=lambda: blocks,
    )
    num_tokens, num_reqs = 8, 2
    TPUModelRunner._precompile_spec_verify_model(
        runner,
        torch.zeros((num_tokens, VOCAB), dtype=torch.bfloat16),
        torch.zeros(num_reqs, dtype=torch.int32),
        torch.zeros(num_tokens, dtype=torch.int32),
        torch.zeros(num_tokens, dtype=torch.int32),
        torch.zeros(num_reqs, dtype=torch.int32),
        torch.zeros(num_tokens, dtype=torch.int64),
        torch.zeros(num_tokens, dtype=torch.int32),
        torch.zeros(num_tokens, dtype=torch.float32),
        torch.zeros((num_tokens, VOCAB), dtype=torch.float32),
    )
    ((inputs, variants),) = sampler.precompiles
    want = [(0, True), (0, False)] + ([(blocks, True)] if blocks else [])
    assert variants == want
    assert inputs.temperatures.shape == (num_reqs, 1)
    assert inputs.temperatures.dtype == torch.bfloat16
    assert inputs.bonus_u.shape == (num_reqs, VOCAB)
    assert inputs.max_draft_tokens == K
