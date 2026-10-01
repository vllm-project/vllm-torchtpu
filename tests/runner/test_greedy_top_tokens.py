# SPDX-License-Identifier: Apache-2.0
"""TPU_GREEDY_TOP_TOKENS: the vocabulary-parallel argmax against the
full-logits argmax, and when a step may take it."""

import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm import SamplingParams
from vllm.model_executor.layers import logits_processor as lp_module
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.sampling_params import StructuredOutputsParams

from vllm_torchtpu.layers.adapter.sample.rejection_sampler import RejectionSampler
from vllm_torchtpu.runner import greedy_top_tokens
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


class _HeadShard(torch.nn.Module):
    """One rank's slice of a vocab-parallel LM head (rows [start, start+n))."""

    def __init__(self, weight, start, tp_size, num_pad):
        super().__init__()
        self.weight = weight
        self.tp_size = tp_size
        self.shard_indices = SimpleNamespace(
            org_vocab_start_index=start, num_org_vocab_padding=num_pad
        )
        self.quant_method = SimpleNamespace(
            apply=lambda layer, x, bias=None: x @ layer.weight.T
        )


class _Owner(torch.nn.Module):
    def __init__(self, head):
        super().__init__()
        self.lm_head = head
        with patch.object(
            lp_module,
            "get_current_vllm_config",
            return_value=SimpleNamespace(model_config=None),
        ):
            self.logits_processor = LogitsProcessor(head.weight.shape[0])


def _shards(weight, tp_size):
    """Split [vocab, H] into tp_size shards, padding the tail with rows that
    would win the argmax if the padding mask were missing."""
    vocab, hidden = weight.shape
    per_rank = math.ceil(vocab / tp_size)
    pad_rows = torch.full((per_rank * tp_size - vocab, hidden), 1e4, dtype=weight.dtype)
    padded = torch.cat([weight, pad_rows])
    shards = []
    for rank in range(tp_size):
        start = rank * per_rank
        num_pad = max(0, start + per_rank - vocab) if start < vocab else per_rank
        shards.append(
            _HeadShard(padded[start : start + per_rank], start, tp_size, num_pad)
        )
    return shards


def _vocab_parallel_top_tokens(weight, hidden_states, tp_size, monkeypatch):
    """vLLM's get_top_tokens on every rank, the all-gather played back from
    the pairs each rank contributed."""
    shards = _shards(weight, tp_size)
    pairs = []

    def record(pair, dim):
        pairs.append(pair)
        return pair.repeat(1, tp_size)

    monkeypatch.setattr(lp_module, "tensor_model_parallel_all_gather", record)
    for shard in shards:
        greedy_top_tokens.top_tokens(_Owner(shard), hidden_states)
    monkeypatch.setattr(
        lp_module,
        "tensor_model_parallel_all_gather",
        lambda pair, dim: torch.cat(pairs, dim=-1),
    )
    return greedy_top_tokens.top_tokens(_Owner(shards[0]), hidden_states)


def _full_logits_tokens(weight, hidden_states):
    return torch.argmax(hidden_states @ weight.T, dim=-1, keepdim=True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
def test_top_tokens_match_full_logits_argmax(monkeypatch, dtype, tp_size):
    gen = torch.Generator().manual_seed(tp_size)
    weight = torch.randn(301, 32, generator=gen).to(dtype)
    hidden_states = torch.randn(17, 32, generator=gen).to(dtype)

    got = _vocab_parallel_top_tokens(weight, hidden_states, tp_size, monkeypatch)

    assert got.dtype == torch.int64 and got.shape == (17, 1)
    assert torch.equal(got, _full_logits_tokens(weight, hidden_states))


@pytest.mark.parametrize("tp_size", [1, 4])
def test_ties_resolve_to_the_lowest_token_id(monkeypatch, tp_size):
    # 16 tokens, 4 per rank. Row 0 ties ids 5 and 13 (ranks 1 and 3); row 1
    # ties ids 6 and 7 inside rank 1; row 2 ties every id.
    weight = torch.zeros(16, 3)
    weight[[5, 13], 0] = 2.0
    weight[[6, 7], 1] = 2.0
    hidden_states = torch.tensor([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])

    got = _vocab_parallel_top_tokens(weight, hidden_states, tp_size, monkeypatch)

    assert got.view(-1).tolist() == [5, 6, 0]
    assert torch.equal(got, _full_logits_tokens(weight, hidden_states))


def _batch(*params, max_num_logprobs=None):
    requests = {
        f"r{i}": SimpleNamespace(sampling_params=p) for i, p in enumerate(params)
    }
    input_batch = SimpleNamespace(
        all_greedy=all(p.temperature == 0 for p in params),
        max_num_logprobs=max_num_logprobs,
        req_ids=[*requests, None],
    )
    return input_batch, requests


def test_step_eligible_for_greedy_batches_only():
    greedy = SamplingParams(temperature=0)
    assert greedy_top_tokens.step_eligible(*_batch(greedy, greedy))
    assert not greedy_top_tokens.step_eligible(
        *_batch(greedy, SamplingParams(temperature=0.6))
    )
    assert not greedy_top_tokens.step_eligible(*_batch(greedy, max_num_logprobs=2))


def test_structured_output_falls_back_to_full_logits():
    grammar = SamplingParams(
        temperature=0, structured_outputs=StructuredOutputsParams(regex="[ab]+")
    )
    assert not greedy_top_tokens.step_eligible(
        *_batch(SamplingParams(temperature=0), grammar)
    )


def test_resolve_head_owner_finds_the_nested_language_model():
    owner = _Owner(_shards(torch.zeros(8, 2), 2)[0])
    model = torch.nn.Module()
    model.language_model = owner

    assert greedy_top_tokens.resolve_head_owner(model, dp_lockstep=False) is owner


def test_resolve_head_owner_refuses_unsupported_models():
    with pytest.raises(NotImplementedError, match="no module with an lm_head"):
        greedy_top_tokens.resolve_head_owner(torch.nn.Linear(2, 2), False)
    with pytest.raises(NotImplementedError, match="expert-parallel data"):
        greedy_top_tokens.resolve_head_owner(
            _Owner(_shards(torch.zeros(8, 2), 2)[0]), dp_lockstep=True
        )

    class TpuLogitsModel(_Owner):
        def compute_logits(self, hidden_states):
            return hidden_states

    TpuLogitsModel.compute_logits.__module__ = "vllm_torchtpu.models.fake"
    with pytest.raises(NotImplementedError, match="TPU-specific compute_logits"):
        greedy_top_tokens.resolve_head_owner(
            TpuLogitsModel(_shards(torch.zeros(8, 2), 1)[0]), False
        )


def _runner(weight, speculative_config=None):
    owner = _Owner(_shards(weight, 1)[0])
    runner = object.__new__(TPUModelRunner)
    runner.is_pooling_model = False
    runner.model = SimpleNamespace(compute_logits=lambda h: h @ weight.T)
    runner._greedy_head_owner = owner
    runner.speculative_config = speculative_config
    return runner


def test_runner_selected_top_tokens_match_selected_logits_argmax():
    gen = torch.Generator().manual_seed(0)
    weight = torch.randn(64, 8, generator=gen)
    hidden_states = torch.randn(12, 8, generator=gen)
    indices = torch.tensor([11, 0, 3, 3], dtype=torch.int32)
    runner = _runner(weight)

    with torch.compiler.set_stance("force_eager"):
        tokens = TPUModelRunner.compute_selected_top_tokens(
            runner, hidden_states, indices
        )
        logits = TPUModelRunner.compute_selected_logits(runner, hidden_states, indices)

    assert torch.equal(tokens, torch.argmax(logits, dim=-1, keepdim=True))


def test_runner_step_choice():
    greedy = SamplingParams(temperature=0)
    runner = _runner(torch.zeros(8, 2))
    runner.input_batch, runner.requests = _batch(greedy)
    assert runner._greedy_top_tokens_this_step()

    runner._greedy_head_owner = None
    assert not runner._greedy_top_tokens_this_step()

    spec = _runner(torch.zeros(8, 2), SimpleNamespace(num_speculative_tokens=3))
    spec.input_batch, spec.requests = _batch(greedy)
    assert spec._greedy_top_tokens_this_step()
    # A mixed greedy / sampled batch verifies on the full logits.
    spec.input_batch, spec.requests = _batch(greedy, SamplingParams(temperature=0.6))
    assert not spec._greedy_top_tokens_this_step()


K = 3


def _verify_chunk(monkeypatch, synthetic):
    """Three requests at k=3: all drafts accepted, rejected at the second
    draft, rejected at the first of two drafts. Rows are the vocab-parallel
    logits of a 4-rank head; one target row is an exact tie."""
    gen = torch.Generator().manual_seed(7)
    weight = torch.randn(96, 16, generator=gen)
    weight[20] *= 3
    weight[50] = weight[20]
    draft_lengths = [3, 3, 2]
    rows = sum(d + 1 for d in draft_lengths)
    hidden_states = torch.randn(rows, 16, generator=gen)
    hidden_states[4] = weight[20]
    logits = hidden_states @ weight.T
    assert torch.topk(logits[4], 2).indices.sort().values.tolist() == [20, 50]
    argmax = torch.argmax(logits, dim=-1)

    target_rows, bonus_rows, drafts, segments, groups = [], [], [], [], []
    row = 0
    for req, length in enumerate(draft_lengths):
        for pos in range(length):
            target_rows.append(row + pos)
            segments.append(req)
            groups.append(pos)
            token = int(argmax[row + pos])
            rejected = (req == 1 and pos == 1) or (req == 2 and pos == 0)
            drafts.append((token + 1) % 96 if rejected else token)
        bonus_rows.append(row + length)
        row += length + 1
    md = SimpleNamespace(
        draft_token_ids=torch.tensor(drafts, dtype=torch.int32),
        draft_lengths=torch.tensor(draft_lengths, dtype=torch.int32),
        target_logits_indices=torch.tensor(target_rows, dtype=torch.int32),
        bonus_logits_indices=torch.tensor(bonus_rows, dtype=torch.int32),
        segment_ids=torch.tensor(segments, dtype=torch.int64),
        group_indices=torch.tensor(groups, dtype=torch.int32),
    )
    spec_config = SimpleNamespace(
        num_speculative_tokens=K,
        rejection_sample_method="synthetic" if synthetic else "strict",
        synthetic_acceptance_rates=[0.9, 0.5, 0.2] if synthetic else None,
    )
    runner = _runner(weight, spec_config)
    runner.rejection_sampler = RejectionSampler(spec_config, torch.device("cpu"))
    tokens = _vocab_parallel_top_tokens(weight, hidden_states, 4, monkeypatch)
    return runner, logits, tokens, md


def _sample(runner, values, md, top_tokens, seed=0):
    with torch.compiler.set_stance("force_eager"):
        return TPUModelRunner._sample_spec_verify_chunk(
            runner,
            values,
            md,
            None,
            None,
            0,
            3,
            True,
            torch.Generator().manual_seed(seed),
            top_tokens,
        )


def test_verify_on_top_tokens_matches_full_logits(monkeypatch):
    runner, logits, tokens, md = _verify_chunk(monkeypatch, synthetic=False)

    full = _sample(runner, logits, md, top_tokens=False)
    top = _sample(runner, tokens, md, top_tokens=True)

    assert torch.equal(top, full)
    assert tokens[4].item() == 20  # the tie row, accepted as req 1's first draft
    argmax = torch.argmax(logits, dim=-1).tolist()
    drafts = md.draft_token_ids.tolist()
    assert full.tolist() == [
        [*drafts[0:3], argmax[3]],  # all accepted + bonus
        [drafts[3], argmax[5], -1, -1],  # rejected at the second draft
        [argmax[8], -1, -1, -1],  # rejected at the first draft
    ]


@pytest.mark.parametrize("seed", range(4))
def test_synthetic_verify_on_top_tokens_matches_full_logits(monkeypatch, seed):
    runner, logits, tokens, md = _verify_chunk(monkeypatch, synthetic=True)

    full = _sample(runner, logits, md, top_tokens=False, seed=seed)
    top = _sample(runner, tokens, md, top_tokens=True, seed=seed)

    assert torch.equal(top, full)


def test_rejection_sampler_refuses_argmax_for_sampling():
    with pytest.raises(ValueError, match="needs target_logits"):
        RejectionSampler()(
            draft_token_ids=torch.zeros(1, dtype=torch.int32),
            num_draft_tokens=torch.ones(1, dtype=torch.int32),
            target_logits=None,
            bonus_token_ids=torch.zeros(1, dtype=torch.int64),
            segment_ids=torch.zeros(1, dtype=torch.int64),
            group_indices=torch.zeros(1, dtype=torch.int32),
            max_draft_tokens=1,
            do_sampling=True,
            target_argmax=torch.zeros(1, dtype=torch.int64),
        )
