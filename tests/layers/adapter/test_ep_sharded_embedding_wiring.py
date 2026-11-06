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
"""Construction and dummy rows of the EP-sharded token embedding."""

from types import SimpleNamespace

import pytest
import torch
import vllm.distributed.parallel_state as parallel_state
import vllm.model_executor.layers.vocab_parallel_embedding as vpe

import vllm_torchtpu
import vllm_torchtpu.envs as envs
from vllm_torchtpu.layers.adapter import ep_sharded_embedding as ep_embed
from vllm_torchtpu.layers.adapter import ragged_a2a_moe

DP, TP = 2, 4
ROWS = 64  # per EP rank
VOCAB = ROWS * DP * TP
HIDDEN = 16


@pytest.fixture
def ranks(monkeypatch):
    vllm_torchtpu._patch_vllm_vocab_parallel_embedding()
    state = SimpleNamespace(ep_rank=5)
    monkeypatch.setattr(
        vpe, "get_tensor_model_parallel_rank", lambda: state.ep_rank % TP
    )
    monkeypatch.setattr(vpe, "get_tensor_model_parallel_world_size", lambda: TP)
    monkeypatch.setattr(
        parallel_state,
        "get_ep_group",
        lambda: SimpleNamespace(rank_in_group=state.ep_rank, world_size=DP * TP),
    )
    monkeypatch.setattr(
        parallel_state, "get_tp_group", lambda: SimpleNamespace(world_size=TP)
    )
    monkeypatch.setattr(ragged_a2a_moe, "get_dp_tp_mesh", lambda: None)
    return state


@pytest.mark.cpu_test
def test_embed_tokens_rows_follow_the_ep_rank(monkeypatch, ranks):
    monkeypatch.setattr(envs, "TPU_EMBED_EP_SHARD", True, raising=False)
    embed = vpe.VocabParallelEmbedding(VOCAB, HIDDEN, prefix="model.embed_tokens")
    assert embed.weight.shape == (ROWS, HIDDEN)
    assert embed.shard_indices.org_vocab_start_index == 5 * ROWS
    assert embed.shard_indices.org_vocab_end_index == 6 * ROWS
    assert getattr(embed, ep_embed.EP_SHARD_ATTR) == (5, DP * TP, TP)
    assert getattr(embed, ep_embed.EP_START_ATTR).tolist() == [5 * ROWS]
    # Checkpoint loading copies this rank's rows.
    full = torch.arange(VOCAB * HIDDEN, dtype=torch.float32).reshape(VOCAB, HIDDEN)
    embed.weight_loader(embed.weight, full)
    assert torch.equal(embed.weight.data, full[5 * ROWS : 6 * ROWS])
    # The TP size lookups are restored after construction.
    assert vpe.get_tensor_model_parallel_world_size() == TP


@pytest.mark.cpu_test
def test_lm_head_and_knob_off_keep_the_vocab_parallel_rows(monkeypatch, ranks):
    monkeypatch.setattr(envs, "TPU_EMBED_EP_SHARD", True, raising=False)
    head = vpe.ParallelLMHead(VOCAB, HIDDEN, prefix="lm_head")
    assert head.weight.shape == (VOCAB // TP, HIDDEN)
    assert not hasattr(head, ep_embed.EP_SHARD_ATTR)
    # Model code builds embed_tokens without a prefix.
    unnamed = vpe.VocabParallelEmbedding(VOCAB, HIDDEN)
    assert unnamed.weight.shape == (ROWS, HIDDEN)
    monkeypatch.setattr(envs, "TPU_EMBED_EP_SHARD", False, raising=False)
    embed = vpe.VocabParallelEmbedding(VOCAB, HIDDEN, prefix="model.embed_tokens")
    assert embed.weight.shape == (VOCAB // TP, HIDDEN)
    assert not hasattr(embed, ep_embed.EP_SHARD_ATTR)


@pytest.mark.cpu_test
def test_tensor_parallelism_is_required(monkeypatch, ranks):
    monkeypatch.setattr(envs, "TPU_EMBED_EP_SHARD", True, raising=False)
    monkeypatch.setattr(
        parallel_state, "get_tp_group", lambda: SimpleNamespace(world_size=1)
    )
    with pytest.raises(NotImplementedError):
        vpe.VocabParallelEmbedding(VOCAB, HIDDEN, prefix="model.embed_tokens")


@pytest.mark.cpu_test
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_dummy_rows_rebuild_the_vocab_parallel_table(monkeypatch, ranks, dtype):
    monkeypatch.setattr(envs, "TPU_EMBED_EP_SHARD", True, raising=False)
    rows = []
    for rank in range(DP * TP):
        ranks.ep_rank = rank
        embed = vpe.VocabParallelEmbedding(
            VOCAB, HIDDEN, params_dtype=dtype, prefix="model.embed_tokens"
        )
        ep_embed.restore_tp_dummy_rows(torch.nn.ModuleList([embed]))
        rows.append(embed.weight.data.clone())
    # Dummy loading draws each TP shard of the vocab-parallel table from the
    # same seed and shape, so that table repeats one draw TP times.
    generator = torch.Generator().manual_seed(1234)
    draw = torch.empty((VOCAB // TP, HIDDEN), dtype=dtype)
    draw.uniform_(-1e-3, 1e-3, generator=generator)
    assert torch.equal(torch.cat(rows), draw.repeat(TP, 1))
