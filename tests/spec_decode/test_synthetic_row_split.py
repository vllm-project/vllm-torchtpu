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
"""spec_synthetic_row_split: the synthetic one-row verify with its request
rows split over the TP ranks gives the unsplit path's tokens, and with the
key unset the runner never reaches the split path."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import vllm.distributed

from vllm_torchtpu.layers.adapter.sample import rejection_sampler as rs
from vllm_torchtpu.runner import tpu_runner
from vllm_torchtpu.runner.speculative_decoding_manager import SpecDecodeMetadata
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

K = 3


def _batch(num_reqs: int, vocab: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn((num_reqs, vocab), generator=g) * 3
    lengths = torch.randint(0, K + 1, (num_reqs,), generator=g, dtype=torch.int32)
    num_tokens = int(lengths.sum()) + num_reqs
    segment_ids = torch.repeat_interleave(
        torch.arange(num_reqs), lengths.to(torch.int64) + 1
    )
    group_indices = torch.cat(
        [torch.arange(int(n) + 1) for n in lengths.tolist()]
    ).to(torch.int32)
    draft_ids = torch.randint(0, vocab, (num_tokens,), generator=g, dtype=torch.int32)
    accepted = torch.minimum(
        torch.randint(0, K + 1, (num_reqs,), generator=g), lengths.to(torch.int64)
    )
    recover_draft = torch.where(
        accepted < lengths.to(torch.int64),
        torch.randint(0, vocab, (num_reqs,), generator=g),
        torch.full((num_reqs,), -1),
    )
    temps = torch.rand((num_reqs, 1), generator=g)
    temps[::3] = 0.0  # greedy rows
    top_k = torch.zeros((num_reqs, 1), dtype=torch.int32)
    top_p = 0.5 + 0.5 * torch.rand((num_reqs, 1), generator=g)
    recover_u = torch.rand((num_reqs, vocab), generator=g)
    return dict(
        logits=logits,
        recover_draft=recover_draft,
        accepted_count=accepted,
        temperatures=temps,
        top_k=top_k,
        top_p=top_p,
        recover_u=recover_u,
        draft_token_ids=draft_ids,
        num_draft_tokens=lengths,
        segment_ids=segment_ids,
        group_indices=group_indices,
    )


def _args(b, logits):
    return (
        logits,
        b["recover_draft"],
        b["accepted_count"],
        b["temperatures"],
        b["top_k"],
        b["top_p"],
        b["recover_u"],
        b["draft_token_ids"],
        b["num_draft_tokens"],
        b["segment_ids"],
        b["group_indices"],
        K,
    )


@pytest.mark.parametrize("tp", [2, 4])
@pytest.mark.parametrize("apply_top_k", [True, False])
def test_split_rows_sample_the_unsplit_tokens(monkeypatch, tp, apply_top_k):
    num_reqs, vocab = 16, 384
    b = _batch(num_reqs, vocab)
    want = rs.synthetic_sample_and_assemble.__wrapped__(
        *_args(b, b["logits"]), apply_top_k=apply_top_k
    )
    per_rank = num_reqs // tp
    split = rs.synthetic_sample_rows_and_assemble.__wrapped__

    def run(rank, gathered):
        rest = _args(b, b["logits"][rank * per_rank : (rank + 1) * per_rank])
        return split(
            rest[0],
            torch.tensor([rank], dtype=torch.int32),
            *rest[1:],
            apply_top_k=apply_top_k,
        )

    # Pass 1: each rank's sampled block; pass 2: the gathered blocks.
    blocks = []

    def capture(x, dim):
        blocks.append(x.clone())
        return x.repeat(tp)

    monkeypatch.setattr(vllm.distributed, "tensor_model_parallel_all_gather", capture)
    for rank in range(tp):
        run(rank, None)
    gathered = torch.cat(blocks)
    monkeypatch.setattr(
        vllm.distributed, "tensor_model_parallel_all_gather", lambda x, dim: gathered
    )
    for rank in range(tp):
        assert torch.equal(run(rank, gathered), want)


@pytest.mark.parametrize("tp", [2, 8])
def test_logit_rows_are_the_rank_block_of_the_gathered_logits(monkeypatch, tp):
    num_rows, hidden, vocab, pad = 4 * tp, 8, 64 * tp, 3
    g = torch.Generator().manual_seed(1)
    hidden_states = torch.randn((num_rows + 2, hidden), generator=g)
    head = torch.randn((hidden, vocab), generator=g)
    indices = torch.randperm(num_rows + 2, generator=g)[:num_rows].to(torch.int32)
    full = hidden_states[indices.long()] @ head
    shard_vocab = vocab // tp

    for rank in range(tp):
        model = SimpleNamespace(
            compute_logits_shard=lambda x, r=None: None,
            logits_processor=SimpleNamespace(org_vocab_size=vocab - pad),
        )

        def shard_of(x, d):
            return x @ head[:, d * shard_vocab : (d + 1) * shard_vocab]

        model.compute_logits_shard = lambda x, rank=rank: shard_of(x, rank)

        def all_to_all(send, *_args, rank=rank):
            # Rank `rank` receives block `rank` of every rank d's send buffer.
            selected = hidden_states[indices.long()]
            return torch.stack(
                [
                    shard_of(selected, d).reshape(tp, num_rows // tp, shard_vocab)[
                        rank
                    ]
                    for d in range(tp)
                ]
            )

        from torch.distributed import _functional_collectives as funcol

        monkeypatch.setattr(funcol, "all_to_all_single", all_to_all)
        monkeypatch.setattr(
            vllm.distributed,
            "get_tp_group",
            lambda: SimpleNamespace(world_size=tp, device_group=None),
        )
        runner = SimpleNamespace(model=model)
        got = TPUModelRunner.compute_selected_logit_rows.__wrapped__(
            runner, hidden_states, indices
        )
        block = full[rank * (num_rows // tp) : (rank + 1) * (num_rows // tp)]
        assert torch.equal(got, block[:, : vocab - pad])


def _runner(additional, tp=8, rank_tensor=None):
    runner = TPUModelRunner.__new__(TPUModelRunner)
    runner.vllm_config = SimpleNamespace(additional_config=additional)
    runner.speculative_config = object()
    runner._pp_is_last = True
    runner.is_pooling_model = False
    runner.model = SimpleNamespace(compute_logits_shard=lambda x: x)
    runner.device = torch.device("cpu")
    runner._dp_lockstep_enabled = lambda: False
    runner._synthetic_row_rank = rank_tensor
    return runner


@pytest.fixture
def tp8(monkeypatch):
    monkeypatch.setattr(
        vllm.distributed, "get_tensor_model_parallel_world_size", lambda: 8
    )
    monkeypatch.setattr(vllm.distributed, "get_tensor_model_parallel_rank", lambda: 3)


@pytest.mark.parametrize("additional", [None, {}, {"spec_synthetic_row_split": False}])
def test_key_off_never_splits(tp8, additional):
    runner = _runner(additional)
    assert runner._synthetic_row_split_rank() is None
    runner._synthetic_row_rank = None
    assert not runner._synthetic_rows_split(64)

    # The verify keeps the unsplit LM head on the planned rows.
    rows = torch.arange(64, dtype=torch.int32)
    runner._plan_synthetic_verify = lambda md: rows
    runner.compute_selected_logits = mock.MagicMock(return_value="logits")
    runner.compute_selected_logit_rows = mock.MagicMock()
    md = SpecDecodeMetadata.__new__(SpecDecodeMetadata)
    assert runner._synthetic_verify_logits("hidden", md) == "logits"
    runner.compute_selected_logits.assert_called_once_with("hidden", rows)
    runner.compute_selected_logit_rows.assert_not_called()
    assert md.synthetic_row_rank is None


def test_key_on_splits_eligible_row_counts(tp8):
    runner = _runner({"spec_synthetic_row_split": True})
    rank = runner._synthetic_row_split_rank()
    assert rank.tolist() == [3]
    runner._synthetic_row_rank = rank
    # >= 4 rows per rank and a multiple of TP.
    assert [n for n in (8, 16, 24, 32, 40, 48, 64) if runner._synthetic_rows_split(n)] == [
        32,
        40,
        48,
        64,
    ]
    rows = torch.arange(32, dtype=torch.int32)
    runner._plan_synthetic_verify = lambda md: rows
    runner.compute_selected_logits = mock.MagicMock()
    runner.compute_selected_logit_rows = mock.MagicMock(return_value="rows")
    md = SpecDecodeMetadata.__new__(SpecDecodeMetadata)
    assert runner._synthetic_verify_logits("hidden", md) == "rows"
    assert md.synthetic_row_rank is rank


def test_key_on_without_model_shard_head_stays_off(tp8):
    runner = _runner({"spec_synthetic_row_split": True})
    runner.model = SimpleNamespace()
    assert runner._synthetic_row_split_rank() is None


def test_split_threshold_constant():
    assert tpu_runner._SYNTHETIC_ROW_SPLIT_MIN_ROWS == 4


def _input_batch():
    batch = SimpleNamespace(is_pooling_model=False, sampling_metadata="metadata")
    batch._make_sampling_metadata = mock.MagicMock(return_value="rebuilt")
    return batch


def test_skip_device_sampling_metadata():
    runner = TPUModelRunner.__new__(TPUModelRunner)
    runner.input_batch = _input_batch()
    original = runner.input_batch._make_sampling_metadata
    runner._skip_device_sampling_metadata()
    assert runner.input_batch._make_sampling_metadata() == "metadata"
    original.assert_not_called()
    # Idempotent; pooling batches are left alone.
    runner._skip_device_sampling_metadata()
    pooling = _input_batch()
    pooling.is_pooling_model = True
    runner.input_batch = pooling
    runner._skip_device_sampling_metadata()
    assert pooling._make_sampling_metadata() == "rebuilt"
