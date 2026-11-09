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
"""Wiring of the draft propose split over DP (TPU_MTP_DRAFT_PROPOSE_DP_SPLIT):
when the proposer takes it, what it refuses, and the operands it passes."""

import inspect
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import vllm.distributed.parallel_state as parallel_state
import vllm.model_executor.layers.vocab_parallel_embedding as vpe

from vllm_torchtpu.layers.adapter import draft_propose_split as adapter
from vllm_torchtpu.layers.adapter import ragged_a2a_moe
from vllm_torchtpu.layers.core import draft_propose_split as split
from vllm_torchtpu.runner.utils import HiddenMajorLMHeadMethod
from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer

HIDDEN, COLS, DP = 256, 4096, 8


def _head(cols=COLS, dtype=torch.bfloat16):
    head = vpe.ParallelLMHead.__new__(vpe.ParallelLMHead)
    torch.nn.Module.__init__(head)
    head.weight = torch.nn.Parameter(torch.zeros(HIDDEN, cols, dtype=dtype))
    head.quant_method = HiddenMajorLMHeadMethod()
    head.tp_size = 1
    head.register_buffer("org_vocab_start_index", torch.tensor(0, dtype=torch.int32))
    head.register_buffer("org_vocab_end_index", torch.tensor(cols, dtype=torch.int32))
    return head


def _processor(**kw):
    return SimpleNamespace(**{"scale": 1.0, "soft_cap": None, "head_dtype": None, **kw})


@pytest.fixture
def dp_rank(monkeypatch):
    state = SimpleNamespace(rank=3, size=DP)
    monkeypatch.setattr(
        parallel_state,
        "get_dp_group",
        lambda: SimpleNamespace(rank_in_group=state.rank, world_size=state.size),
    )
    monkeypatch.setattr(
        ragged_a2a_moe,
        "get_dp_tp_mesh",
        lambda: SimpleNamespace(shape={split.DP_AXIS: state.size, split.TP_AXIS: 4}),
    )
    monkeypatch.setattr(adapter, "_get_op", lambda: None)
    return state


def _proposer(*, vocab_parallel=True, lockstep=True):
    speculative_config = SimpleNamespace(
        draft_tensor_parallel_size=4, method="mtp", draft_model_config=None
    )
    vllm_config = SimpleNamespace(
        speculative_config=speculative_config,
        parallel_config=SimpleNamespace(tensor_parallel_size=4),
    )
    proposer = Eagle3Proposer(
        runner=mock.MagicMock(_pcp_enabled=False, _mtp_enabled=True),
        vllm_config=vllm_config,
    )
    proposer._use_vocab_parallel_argmax = vocab_parallel
    proposer._dp_lockstep_sharded = lambda: lockstep
    proposer.draft_model = SimpleNamespace(lm_head=_head(), logits_processor=None)
    return proposer


@pytest.mark.cpu_test
def test_slice_start_follows_the_dp_rank(dp_rank):
    for rank in (0, 3, 7):
        dp_rank.rank = rank
        head = _head()
        adapter.enable_draft_propose_split(head, _processor())
        start = getattr(head, adapter.SLICE_START_ATTR)
        assert start.dtype == torch.int32
        assert start.tolist() == [rank * COLS // DP]
        assert adapter.SLICE_START_ATTR not in head.state_dict()


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "case",
    ["vocab_major", "scale", "soft_cap", "fp32_head", "fp32_weight", "dp1", "cols"],
)
def test_heads_the_split_cannot_serve_are_refused(dp_rank, case):
    head = _head(
        cols=COLS - 4 if case == "cols" else COLS,
        dtype=torch.float32 if case == "fp32_weight" else torch.bfloat16,
    )
    if case == "vocab_major":
        head.quant_method = vpe.UnquantizedEmbeddingMethod()
    if case == "dp1":
        dp_rank.size = 1
    processor = _processor(
        scale=0.5 if case == "scale" else 1.0,
        soft_cap=30.0 if case == "soft_cap" else None,
        head_dtype=torch.float32 if case == "fp32_head" else None,
    )
    with pytest.raises(NotImplementedError, match="TPU_MTP_DRAFT_PROPOSE_DP_SPLIT"):
        adapter.enable_draft_propose_split(head, processor)
    assert not hasattr(head, adapter.SLICE_START_ATTR)


@pytest.mark.cpu_test
def test_split_passes_rank_operands_and_declines_other_row_counts(monkeypatch, dp_rank):
    table = dict(split._XLA_HEAD_WINDOWS)
    table[(3, HIDDEN, COLS)] = 128
    monkeypatch.setattr(split, "_XLA_HEAD_WINDOWS", table)
    head = _head()
    adapter.enable_draft_propose_split(head, _processor())
    calls = []

    def op(*operands):
        calls.append([tuple(t.shape) for t in operands])
        return torch.full((1, 1, operands[0].shape[2]), 5, dtype=torch.int32)

    monkeypatch.setattr(adapter, "_get_op", lambda: op)
    hidden = torch.zeros(3, HIDDEN, dtype=torch.bfloat16)
    tokens = adapter.draft_split_tokens(head, hidden)
    assert tokens.tolist() == [5, 5, 5]
    assert calls == [
        [(1, 1, 3, HIDDEN), (1, 1, HIDDEN, COLS), (1, 1, 1), (1, 1, 1), (1, 1, 1)]
    ]
    assert adapter.draft_split_tokens(head, torch.zeros(4, HIDDEN)) is None
    assert adapter.draft_split_tokens(head, torch.zeros(3, HIDDEN)) is None  # fp32
    assert len(calls) == 1


@pytest.mark.cpu_test
@pytest.mark.parametrize("vocab_parallel,lockstep", [(False, True), (True, False)])
def test_proposer_refuses_the_split_without_its_preconditions(vocab_parallel, lockstep):
    proposer = _proposer(vocab_parallel=vocab_parallel, lockstep=lockstep)
    with pytest.raises(NotImplementedError, match="TPU_MTP_DRAFT_PROPOSE_DP_SPLIT"):
        proposer._enable_draft_propose_split()
    assert proposer._draft_propose_split is False


@pytest.mark.cpu_test
def test_proposer_enables_the_split_on_the_shared_head(monkeypatch):
    proposer = _proposer()
    seen = []
    monkeypatch.setattr(
        adapter, "enable_draft_propose_split", lambda head, proc: seen.append(head)
    )
    proposer._enable_draft_propose_split()
    assert proposer._draft_propose_split is True
    assert seen == [proposer.draft_model.lm_head]


@pytest.mark.cpu_test
def test_draft_tokens_come_from_the_split_unless_it_declines(monkeypatch):
    proposer = _proposer()
    head = proposer.draft_model.lm_head
    logits = torch.tensor([[0.0, 3.0, 3.0, 1.0], [2.0, 0.0, 1.0, 2.0]])
    processor_calls = []

    def processor(lm_head, hidden, skip_gather=False):
        processor_calls.append((lm_head, skip_gather))
        return logits.to(torch.bfloat16)

    proposer.draft_model.logits_processor = processor
    head.org_vocab_end_index.fill_(4)
    proposer._draft_propose_split = True
    propose = inspect.unwrap(Eagle3Proposer._draft_propose_token).__get__(proposer)
    hidden = torch.zeros(2, HIDDEN, dtype=torch.bfloat16)

    split_tokens = torch.tensor([7, 9], dtype=torch.int32)
    monkeypatch.setattr(adapter, "draft_split_tokens", lambda h, x: split_tokens)
    assert propose(hidden) is split_tokens
    assert processor_calls == []

    monkeypatch.setattr(adapter, "draft_split_tokens", lambda h, x: None)
    tokens = propose(hidden)
    assert tokens.dtype == torch.int32 and tokens.tolist() == [1, 0]
    assert processor_calls == [(head, True)]
