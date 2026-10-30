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
"""TPU_GREEDY_TOP_TOKENS: the vocabulary-parallel argmax against the
full-logits argmax, and when a step may take it."""

import logging
import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm import SamplingParams
from vllm import logger as vllm_logger
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
    runner._pp_is_last = True
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
    logits of a 4-rank head; one target row is an exact tie. Entries are
    quarters, so every logit is exact in fp32 whatever order a CPU's matmul
    sums in, and the tie holds on every host."""
    gen = torch.Generator().manual_seed(7)
    weight = torch.randint(-8, 9, (96, 16), generator=gen).float() / 4
    weight[20] *= 3
    weight[50] = weight[20]
    draft_lengths = [3, 3, 2]
    rows = sum(d + 1 for d in draft_lengths)
    hidden_states = torch.randint(-8, 9, (rows, 16), generator=gen).float() / 4
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


def test_log_step_path_logs_each_path_once():
    # Captured on the "vllm" logger at its configured level: a record that
    # does not reach it never reaches the server log either.
    if hasattr(vllm_logger, "_print_info_once"):
        vllm_logger._print_info_once.cache_clear()
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    logger = logging.getLogger("vllm")
    # Another test may have raised the level or called logging.disable();
    # either drops the record before any handler runs.
    old_level = logger.level
    old_disable = logging.root.manager.disable
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logging.disable(logging.NOTSET)
    try:
        # The runner's warmup steps hold only synthetic "__..__" requests.
        greedy_top_tokens.log_step_path(True, ["__spec_warmup_0__", None])
        greedy_top_tokens.log_step_path(False, ["__spec_warmup_b2_0__"])
        assert records == []
        for is_spec_step in (True, False, True, False, False):
            greedy_top_tokens.log_step_path(is_spec_step, ["__spec_warmup_1__", "r0"])
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
        logging.disable(old_disable)
    assert [r.getMessage() for r in records] == [
        "TPU_GREEDY_TOP_TOKENS: a speculative verify step took the top tokens",
        "TPU_GREEDY_TOP_TOKENS: a non-speculative step took the top tokens",
    ]


def _vocab_parallel_top_tokens_dim0(weight, hidden_states, tp_size, monkeypatch):
    """Play back tensor_model_parallel_all_gather honoring dim=0 (as
    DeviceCommunicatorBase.all_gather does on TPU)."""
    vocab = weight.shape[0]
    shards = _shards(weight, tp_size)
    owners = []
    for shard in shards:
        owner = _Owner(shard)
        owner.logits_processor.vocab_size = vocab
        owner.logits_processor.padded_vocab_size = shard.weight.shape[0] * tp_size
        greedy_top_tokens.resolve_head_owner(owner, dp_lockstep=False)
        owners.append(owner)

    recorded = []
    seen_dims = []

    def record_dim0(tensor, dim=-1):
        seen_dims.append(dim)
        recorded.append(tensor)
        if dim == 0:
            return tensor.repeat(tp_size, *([1] * (tensor.ndim - 1)))
        return tensor.repeat(1, tp_size)

    monkeypatch.setattr(lp_module, "tensor_model_parallel_all_gather", record_dim0)
    for owner in owners:
        greedy_top_tokens.top_tokens(owner, hidden_states)

    monkeypatch.setattr(
        lp_module,
        "tensor_model_parallel_all_gather",
        lambda tensor, dim=-1: torch.cat(recorded, dim=dim),
    )
    out = greedy_top_tokens.top_tokens(owners[0], hidden_states)
    if tp_size > 1:
        assert all(d == 0 for d in seen_dims)
        assert all(
            t.dtype == torch.float32 and t.shape == (hidden_states.shape[0], 2)
            for t in recorded
        )
    return out


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
def test_tpu_dim0_and_buffer_top_tokens_match_full_logits_argmax(
    monkeypatch, dtype, tp_size
):
    gen = torch.Generator().manual_seed(100 + tp_size)
    weight = torch.randn(301, 32, generator=gen).to(dtype)
    hidden_states = torch.randn(17, 32, generator=gen).to(dtype)

    got = _vocab_parallel_top_tokens_dim0(weight, hidden_states, tp_size, monkeypatch)

    assert got.dtype == torch.int64 and got.shape == (17, 1)
    assert torch.equal(got, _full_logits_tokens(weight, hidden_states))


@pytest.mark.parametrize("tp_size", [1, 4])
def test_bfloat16_negative_zero_and_positive_zero_ties(monkeypatch, tp_size):
    weight = torch.full((16, 2), -1.0, dtype=torch.bfloat16)
    # Row 0: token 1 has -0.0 (-1.0 * 0.0 + 0.0), token 5 has +0.0, rest are -2.0.
    weight[:, 1] = -2.0
    weight[1, 1] = torch.tensor(-0.0, dtype=torch.bfloat16)
    weight[5, 1] = torch.tensor(0.0, dtype=torch.bfloat16)
    hidden_states = torch.tensor([[0.0, 1.0]], dtype=torch.bfloat16)

    got = _vocab_parallel_top_tokens_dim0(weight, hidden_states, tp_size, monkeypatch)

    assert got.item() == 1
    assert torch.equal(got, _full_logits_tokens(weight, hidden_states))


def _vocab_parallel_topk_candidates_dim0(weight, hidden_states, tp_size, monkeypatch):
    """Play back tensor_model_parallel_all_gather honoring dim=0 for
    tpu_get_topk_candidates across all TP shards."""
    vocab = weight.shape[0]
    shards = _shards(weight, tp_size)
    owners = []
    for shard in shards:
        owner = _Owner(shard)
        owner.logits_processor.vocab_size = vocab
        owner.logits_processor.padded_vocab_size = shard.weight.shape[0] * tp_size
        greedy_top_tokens.resolve_head_owner(owner, dp_lockstep=False)
        owners.append(owner)

    recorded = []
    seen_dims = []

    def record_dim0(tensor, dim=-1):
        seen_dims.append(dim)
        recorded.append(tensor)
        if dim == 0:
            return tensor.repeat(tp_size, *([1] * (tensor.ndim - 1)))
        return tensor.repeat(1, tp_size)

    monkeypatch.setattr(lp_module, "tensor_model_parallel_all_gather", record_dim0)
    for owner in owners:
        greedy_top_tokens.topk_candidates(owner, hidden_states)

    monkeypatch.setattr(
        lp_module,
        "tensor_model_parallel_all_gather",
        lambda tensor, dim=-1: torch.cat(recorded, dim=dim),
    )
    out = greedy_top_tokens.topk_candidates(owners[0], hidden_states)
    k_local = greedy_top_tokens.LOCAL_TOPK_CANDIDATES
    if tp_size > 1:
        assert all(d == 0 for d in seen_dims)
        assert all(
            t.dtype == torch.float32 and t.shape == (hidden_states.shape[0], k_local, 2)
            for t in recorded
        )
    return out


@pytest.mark.parametrize("vocab_size", [301, 2100])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_tpu_topk_candidates_contain_global_top20(
    monkeypatch, vocab_size, dtype, tp_size
):
    gen = torch.Generator().manual_seed(200 + vocab_size + tp_size)
    # Give every row 20 clearly separated top tokens placed across up to 8
    # blocks of 64 per shard so both the 2-stage block reducer (shard_vocab >= 512)
    # and direct topk fallback (shard_vocab < 512) preserve the exact global Top-20.
    weight = (torch.randn(vocab_size, 16, generator=gen) * 0.1).to(dtype)
    hidden_states = torch.zeros(5, 16, dtype=dtype)
    per_rank = math.ceil(vocab_size / tp_size)
    for r in range(5):
        hidden_states[r, r % 16] = 1.0
        # Choose up to 8 blocks of 64 inside each shard
        eligible_ids = []
        for rank in range(tp_size):
            shard_start = rank * per_rank
            shard_end = min(vocab_size, (rank + 1) * per_rank)
            if shard_end <= shard_start:
                continue
            n_blocks = math.ceil((shard_end - shard_start) / 64)
            chosen_blks = torch.randperm(n_blocks, generator=gen)[: min(8, n_blocks)]
            for b in chosen_blks.tolist():
                b_start = shard_start + b * 64
                b_end = min(shard_end, b_start + 64)
                eligible_ids.extend(range(b_start, b_end))
        perm = torch.randperm(len(eligible_ids), generator=gen)[:20]
        top_ids = [eligible_ids[i] for i in perm.tolist()]
        for rank_idx, tid in enumerate(top_ids):
            weight[tid, r % 16] = float(50 - rank_idx)

    packed = _vocab_parallel_topk_candidates_dim0(
        weight, hidden_states, tp_size, monkeypatch
    )
    num_cand = greedy_top_tokens.LOCAL_TOPK_CANDIDATES * tp_size
    assert packed.dtype == torch.float32
    assert packed.shape == (5, 2 * num_cand)

    cand_logits = packed[:, :num_cand]
    cand_ids = packed[:, num_cand:].to(torch.int64)
    # Out-of-vocab padding tokens must never appear as valid unmasked candidates.
    valid_mask = cand_logits > -1e30
    assert int(cand_ids[valid_mask].max().item()) < vocab_size

    full_logits = (hidden_states @ weight.T).to(torch.float32)
    ref_top20_vals, ref_top20_ids = torch.topk(full_logits, k=20, dim=-1)
    cand_top20_vals, cand_top20_pos = torch.topk(cand_logits, k=20, dim=-1)
    cand_top20_ids = torch.gather(cand_ids, 1, cand_top20_pos)

    assert torch.equal(cand_top20_ids, ref_top20_ids)
    assert torch.allclose(cand_top20_vals, ref_top20_vals, atol=1e-4, rtol=1e-4)


def test_topk_candidates_step_eligible():
    greedy = SamplingParams(temperature=0)
    top20 = SamplingParams(temperature=0.6, top_k=20, top_p=0.95)
    top32 = SamplingParams(temperature=0.6, top_k=32, top_p=1.0)
    top64 = SamplingParams(temperature=0.6, top_k=64, top_p=0.95)
    no_topk = SamplingParams(temperature=0.6, top_k=-1, top_p=0.95)

    # Pure top-k <= 32 or mixed greedy + top-k <= 32 is eligible.
    assert greedy_top_tokens.topk_candidates_step_eligible(*_batch(top20, top32))
    assert greedy_top_tokens.topk_candidates_step_eligible(*_batch(greedy, top20))

    # Pure greedy uses greedy_top_tokens.step_eligible instead.
    assert not greedy_top_tokens.topk_candidates_step_eligible(*_batch(greedy, greedy))
    # Logprobs require full logits.
    assert not greedy_top_tokens.topk_candidates_step_eligible(
        *_batch(top20, max_num_logprobs=1)
    )
    # Unbounded top_k or top_k > 32 falls back to full logits.
    assert not greedy_top_tokens.topk_candidates_step_eligible(*_batch(top20, top64))
    assert not greedy_top_tokens.topk_candidates_step_eligible(*_batch(top20, no_topk))
    # Structured outputs or penalties fall back to full logits.
    grammar = SamplingParams(
        temperature=0.6,
        top_k=20,
        structured_outputs=StructuredOutputsParams(regex="[ab]+"),
    )
    assert not greedy_top_tokens.topk_candidates_step_eligible(*_batch(top20, grammar))
    penalized = SamplingParams(temperature=0.6, top_k=20, presence_penalty=0.5)
    assert not greedy_top_tokens.topk_candidates_step_eligible(
        *_batch(top20, penalized)
    )


def test_log_topk_candidates_step_path_logs_each_path_once():
    if hasattr(vllm_logger, "_print_info_once"):
        vllm_logger._print_info_once.cache_clear()
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    logger = logging.getLogger("vllm")
    old_level = logger.level
    old_disable = logging.root.manager.disable
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logging.disable(logging.NOTSET)
    try:
        greedy_top_tokens.log_topk_candidates_step_path(
            True, ["__spec_warmup_0__", None]
        )
        greedy_top_tokens.log_topk_candidates_step_path(False, ["__spec_warmup_b2_0__"])
        assert records == []
        for is_spec_step in (True, False, True, False):
            greedy_top_tokens.log_topk_candidates_step_path(
                is_spec_step, ["__spec_warmup_1__", "r0"]
            )
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
        logging.disable(old_disable)
    assert [r.getMessage() for r in records] == [
        "TPU_GREEDY_TOP_TOKENS: a speculative verify step took "
        "vocab-parallel top-32 candidates",
        "TPU_GREEDY_TOP_TOKENS: a non-speculative step took "
        "vocab-parallel top-32 candidates",
    ]


def test_runner_selected_logits_lengths_includes_spec_verify_buckets():
    runner = _runner(
        torch.zeros(64, 8),
        SimpleNamespace(
            num_speculative_tokens=3,
            use_eagle=lambda: True,
            method="mtp",
        ),
    )
    runner.max_num_reqs = 16
    runner.num_reqs_paddings = [1, 2, 4, 8, 16]
    runner.num_tokens_paddings = [16, 24, 32, 40, 48, 56, 64, 128]
    runner._pcp_enabled = False
    runner._mtp_enabled = False
    lengths = runner._selected_logits_lengths(64)
    for expected_len in (16, 24, 32, 40, 48, 56, 64):
        assert expected_len in lengths


def test_runner_sample_from_packed_candidates_and_verify_chunk(monkeypatch):
    gen = torch.Generator().manual_seed(42)
    weight = torch.randn(128, 16, generator=gen) * 0.1
    # Row 0 (req 0 draft 0): token 10 dominates
    # Row 1 (req 0 draft 1): token 20 dominates
    # Row 2 (req 0 bonus): token 30 dominates
    # Row 3 (req 1 draft 0): token 90 dominates
    # Row 4 (req 1 bonus): token 100 dominates
    hidden_states = torch.zeros(5, 16)
    for r, tok in enumerate([10, 20, 30, 90, 100]):
        hidden_states[r, r] = 1.0
        weight[tok, r] = 50.0

    spec_config = SimpleNamespace(
        num_speculative_tokens=2,
        rejection_sample_method="strict",
        synthetic_acceptance_rates=None,
    )
    runner = _runner(weight, spec_config)
    runner.vocab_size = 128
    runner.rejection_sampler = RejectionSampler(spec_config, torch.device("cpu"))
    runner.input_batch = SimpleNamespace(
        temperature_cpu_tensor=torch.full((2,), 0.6, dtype=torch.float32),
        top_k_cpu_tensor=torch.full((2,), 20, dtype=torch.int32),
        top_p_cpu_tensor=torch.full((2,), 0.95, dtype=torch.float32),
    )

    packed = _vocab_parallel_topk_candidates_dim0(weight, hidden_states, 2, monkeypatch)
    runner._greedy_head_owner.lm_head.tp_size = 2
    num_cand = packed.shape[-1] // 2

    # 1. Verify sample_from_packed_candidates picks [10, 20, 30, 90, 100].
    temps = torch.full((5, 1), 0.6, dtype=torch.float32)
    u = torch.full((5, num_cand), 0.5, dtype=torch.float32)
    top_k = torch.full((5, 1), 20, dtype=torch.int32)
    top_p = torch.full((5, 1), 0.95, dtype=torch.float32)
    with torch.compiler.set_stance("force_eager"):
        sampled = TPUModelRunner.sample_from_packed_candidates(
            runner, packed, temps, u, top_k, top_p
        )
    assert sampled.view(-1).tolist() == [10, 20, 30, 90, 100]

    # 2. Verify _sample_spec_verify_chunk with topk_candidates_packed=True:
    # req 0 drafts [10, 99] -> 10 accepted, 99 rejected & recovers 20
    # req 1 drafts [90] -> draft 0 (90) accepted + bonus 100
    md = SimpleNamespace(
        draft_token_ids=torch.tensor([10, 99, 90], dtype=torch.int32),
        draft_lengths=torch.tensor([2, 1], dtype=torch.int32),
        target_logits_indices=torch.tensor([0, 1, 3], dtype=torch.int32),
        bonus_logits_indices=torch.tensor([2, 4], dtype=torch.int32),
        segment_ids=torch.tensor([0, 0, 1], dtype=torch.int64),
        group_indices=torch.tensor([0, 1, 0], dtype=torch.int32),
    )
    with torch.compiler.set_stance("force_eager"):
        out = TPUModelRunner._sample_spec_verify_chunk(
            runner,
            packed,
            md,
            None,
            None,
            0,
            2,
            False,
            torch.Generator().manual_seed(0),
            top_tokens=False,
            topk_candidates_packed=True,
        )
    assert out.tolist() == [[10, 20, -1], [90, 100, -1]]
