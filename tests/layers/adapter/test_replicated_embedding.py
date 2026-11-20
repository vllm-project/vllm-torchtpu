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
"""TPU_EMBED_REPLICATED: local lookups match the shard-sum collectives.

A load-time replica of the vocab-parallel table, looked up locally, gives the
rows the all-reduce / reduce-scatter of shard partials gives. No TPU.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import vllm.distributed
from vllm.model_executor.layers import vocab_parallel_embedding as vpe

import vllm_torchtpu
from vllm_torchtpu.layers.adapter import attention
from vllm_torchtpu.layers.adapter import moe_prefill_sp as sp
from vllm_torchtpu.layers.adapter import replicated_embedding as rep

TP = 4
ORG_VOCAB = 29  # not a multiple of the padding: the last shard has pad rows
PAD_TO = 8
HIDDEN = 6
PAD_ROW_SENTINEL = 999.0


class _ShardEmbed(vpe.VocabParallelEmbedding):
    """Rank `rank`'s vocab shard of `table`, as upstream lays it out.

    Built without a TP group: bounds from upstream `_get_indices`.
    """

    def __init__(
        self,
        table: torch.Tensor,
        rank: int,
        *,
        added: int = 0,
        quant_method: object | None = None,
    ):
        torch.nn.Module.__init__(self)
        self.tp_size = TP
        self.use_fused_embedding = False
        self.quant_method = quant_method or vpe.UnquantizedEmbeddingMethod()
        self.org_vocab_size = ORG_VOCAB
        self.org_vocab_size_padded = vpe.pad_vocab_size(ORG_VOCAB, PAD_TO)
        self.num_embeddings_padded = vpe.pad_vocab_size(
            self.org_vocab_size_padded + added, PAD_TO
        )
        self.shard_indices = vpe.VocabParallelEmbedding._get_indices(  # noqa: SLF001 -- upstream shard-layout producer
            self.num_embeddings_padded,
            self.org_vocab_size_padded,
            ORG_VOCAB + added,
            ORG_VOCAB,
            rank,
            TP,
        )
        per = self.num_embeddings_padded // TP
        self.weight = torch.nn.Parameter(
            table[rank * per : (rank + 1) * per].clone(), requires_grad=False
        )
        idx = self.shard_indices
        for name in (
            "org_vocab_start_index",
            "org_vocab_end_index",
            "num_org_vocab_padding",
            "added_vocab_start_index",
            "added_vocab_end_index",
        ):
            self.register_buffer(
                name, torch.tensor(getattr(idx, name), dtype=torch.int32)
            )


def _padded_table() -> torch.Tensor:
    padded = vpe.pad_vocab_size(ORG_VOCAB, PAD_TO)
    table = torch.randn(padded, HIDDEN, dtype=torch.bfloat16)
    table[ORG_VOCAB:] = PAD_ROW_SENTINEL
    return table


def _shards(table: torch.Tensor) -> list[_ShardEmbed]:
    return [_ShardEmbed(table, r) for r in range(TP)]


def _shard_sum_reference(shards: list[_ShardEmbed], ids: torch.Tensor) -> torch.Tensor:
    """What the all-reduce / reduce-scatter of shard partials sums to."""
    vllm_torchtpu._patch_vllm_vocab_parallel_embedding()  # noqa: SLF001 -- installs the TPU mask helper used at runtime
    total = torch.zeros(ids.shape[0], HIDDEN, dtype=torch.float32)
    for shard in shards:
        masked, mask = vpe.get_masked_input_and_mask(
            ids,
            shard.org_vocab_start_index,
            shard.org_vocab_end_index,
            shard.num_org_vocab_padding,
            shard.added_vocab_start_index,
            shard.added_vocab_end_index,
        )
        part = torch.nn.functional.embedding(masked.long(), shard.weight)
        total += part.masked_fill(mask.unsqueeze(-1), 0).float()
    return total.to(torch.bfloat16)


def _arm_rank(shards: list[_ShardEmbed], rank: int) -> _ShardEmbed:
    rep.arm(shards[rank], lambda _shard: torch.cat([s.weight for s in shards], 0))
    return shards[rank]


def test_should_equal_shard_sum_when_every_token_is_looked_up_locally():
    shards = _shards(_padded_table())
    ids = torch.tensor([28, 0, 7, 8, 15, 16, 23, 24, 3, 28, 11], dtype=torch.int32)

    embed = _arm_rank(shards, rank=2)
    got = rep.lookup(rep.full_weight(embed), ids)

    torch.testing.assert_close(got, _shard_sum_reference(shards, ids), rtol=0, atol=0)


def test_should_register_the_rank_ordered_table_when_armed():
    table = _padded_table()
    embed = _arm_rank(_shards(table), rank=1)

    full = rep.full_weight(embed)

    assert full is not None
    assert full.data_ptr() != embed.weight.data_ptr()
    torch.testing.assert_close(full, table, rtol=0, atol=0)


def test_should_raise_when_the_gather_misses_rows():
    shards = _shards(_padded_table())

    with pytest.raises(ValueError, match="gathered embedding"):
        rep.arm(shards[0], lambda shard: torch.cat([shard] * (TP - 1), 0))
    assert rep.full_weight(shards[0]) is None


@pytest.mark.parametrize(
    ("added", "quantized", "tp", "eligible"),
    [
        (0, False, TP, True),
        (PAD_TO, False, TP, False),  # added vocab: shards are not in token order
        (0, True, TP, False),  # quantized table: no plain rows to replicate
        (0, False, TP * 2, False),  # sharded over a different group
    ],
)
def test_should_replicate_only_plain_rank_ordered_tables(
    added, quantized, tp, eligible
):
    table = torch.zeros(64, HIDDEN, dtype=torch.bfloat16)
    method = SimpleNamespace(embedding=None) if quantized else None
    embed = _ShardEmbed(table, 0, added=added, quant_method=method)

    assert rep.can_replicate(embed, tp) is eligible


@pytest.mark.parametrize("eligible", [True, False])
def test_should_gather_and_register_only_when_eligible(eligible):
    shards = _shards(_padded_table())
    gathers = []

    def _gather_rows(shard: torch.Tensor) -> torch.Tensor:
        gathers.append(tuple(shard.shape))
        return torch.cat([s.weight for s in shards], 0)

    armed = rep.arm_if_eligible(shards[0], TP if eligible else TP * 2, _gather_rows)

    assert armed is eligible
    assert (rep.full_weight(shards[0]) is not None) is eligible
    assert len(gathers) == (1 if eligible else 0)


def test_should_keep_the_masked_all_reduce_in_decode_forward_when_replicated():
    shards = _shards(_padded_table())
    ids = torch.tensor([5, 27, 9, 0], dtype=torch.int32)
    embed = _arm_rank(shards, rank=3)
    reference = _shard_sum_reference(shards, ids)
    reduced, retained = [], []

    def _all_reduce(partial):
        reduced.append(partial.clone())
        return reference.clone()

    def _retain(out, *buffers):
        retained.extend(id(b) for b in buffers)
        return out

    with (
        patch.object(
            vllm.distributed,
            "tensor_model_parallel_all_reduce",
            side_effect=_all_reduce,
        ),
        patch.object(attention, "_retain_buffers", side_effect=_retain),
    ):
        got = embed(ids)

    # The rank's own masked partial goes into the all-reduce, as without a replica.
    start = 3 * embed.weight.shape[0]
    expected_partial = torch.stack(
        [
            embed.weight[t - start] if start <= t < ORG_VOCAB else embed.weight[0] * 0
            for t in ids.tolist()
        ]
    )
    torch.testing.assert_close(reduced[0], expected_partial, rtol=0, atol=0)
    torch.testing.assert_close(got, reference, rtol=0, atol=0)
    assert id(rep.full_weight(embed)) in retained


class _KeepLayer:
    """A decoder layer stand-in carrying a keep list, recording its inputs."""

    def __init__(self, keep: list[torch.Tensor]):
        setattr(self, sp.KEEP_ATTR, keep)
        self.seen: list[torch.Tensor] = []

    def __call__(self, pos, x, res, scaling, *, gather_group=None):
        del gather_group
        self.seen.append(x)
        return x, x


def _run_sp_forward(embed, input_ids, rank_buf, events):
    num_tokens = input_ids.shape[0]
    layer = _KeepLayer([rank_buf])
    model = SimpleNamespace(
        embed_tokens=embed,
        layers=[layer],
        start_layer=0,
        end_layer=1,
        norm=lambda x, res: (x, res),
    )
    setattr(model, sp.KEEP_ATTR, True)
    setattr(model, sp.EMBED_RS_ATTR, True)

    def _rank_operand():
        events.append("rank_operand")
        return rank_buf

    def _retain(x, *keep):
        events.append(("retain", tuple(id(k) for k in keep)))
        return x

    with (
        patch.object(sp, "applies", return_value=True),
        patch.object(sp, "get_tensor_model_parallel_world_size", return_value=TP),
        patch.object(sp.fused_moe_ep, "rank_operand", side_effect=_rank_operand),
        patch.object(sp, "_retain_buffers", side_effect=_retain),
        patch.object(
            sp,
            "tensor_model_parallel_reduce_scatter",
            side_effect=AssertionError("reduce-scatter on a replicated table"),
        ),
        patch.object(sp.small_all_reduce_op, "retain", side_effect=lambda x: x),
        patch.object(sp, "_all_rows", side_effect=lambda x, n: x),
    ):
        sp.patched_model_forward(
            model, input_ids, torch.arange(num_tokens, dtype=torch.int32)
        )
    return layer.seen[0]


def test_should_look_up_only_own_rows_in_sequence_parallel_forward():
    shards = _shards(_padded_table())
    rank, num_tokens = 2, 8
    rows = num_tokens // TP
    input_ids = torch.tensor([28, 1, 9, 17, 25, 4, 12, 20], dtype=torch.int64)
    embed = _arm_rank(shards, rank=rank)

    got = _run_sp_forward(
        embed, input_ids, torch.tensor([[rank]], dtype=torch.int32), events=[]
    )

    own_ids = input_ids[rank * rows : (rank + 1) * rows]
    torch.testing.assert_close(
        got, _shard_sum_reference(shards, own_ids), rtol=0, atol=0
    )


def test_should_lift_the_decode_lookup_operands_in_sequence_parallel_forward():
    shards = _shards(_padded_table())
    embed = _arm_rank(shards, rank=1)
    events: list = []

    _run_sp_forward(
        embed,
        torch.arange(8, dtype=torch.int64),
        torch.tensor([[1]], dtype=torch.int32),
        events,
    )

    retained = {i for e in events if e != "rank_operand" for i in e[1]}
    assert {id(t) for t in rep.shard_operands(embed)} <= retained


def test_should_lift_rank_through_layer_keep_before_reading_rank_operand():
    shards = _shards(_padded_table())
    rank_buf = torch.tensor([[1]], dtype=torch.int32)
    events: list = []

    _run_sp_forward(
        _arm_rank(shards, rank=1),
        torch.arange(8, dtype=torch.int64),
        rank_buf,
        events,
    )

    first_rank = events.index("rank_operand")
    keeps_before = [e for e in events[:first_rank] if e != "rank_operand"]
    assert any(id(rank_buf) in ids for _tag, ids in keeps_before)


def test_should_not_touch_the_trace_when_the_table_is_not_replicated():
    embed = _shards(_padded_table())[0]
    out = torch.ones(2, HIDDEN, dtype=torch.bfloat16)

    with patch.object(
        attention, "_retain_buffers", side_effect=AssertionError("retained")
    ):
        got = rep.retain_table(embed, out)

    assert got is out


DECODE_IDS = [-1, 28, ORG_VOCAB, 31, 0, 13]  # 29..31 are padded tail rows


def _decode_forward(embed, ids):
    retained = []

    def _retain(out, *buffers):
        retained.extend(id(b) for b in buffers)
        return out

    def _no_all_reduce(partial):
        raise AssertionError("decode lookup from the replica must not all-reduce")

    with (
        patch.object(
            vllm.distributed,
            "tensor_model_parallel_all_reduce",
            side_effect=_no_all_reduce,
        ),
        patch.object(attention, "_retain_buffers", side_effect=_retain),
    ):
        got = embed(ids)
    return got, retained


def test_should_zero_out_of_vocab_rows_when_decode_reads_the_replica():
    shards = _shards(_padded_table())
    ids = torch.tensor(DECODE_IDS, dtype=torch.int32)
    embed = _arm_rank(shards, rank=1)
    rep.enable_decode_lookup(embed)

    got, retained = _decode_forward(embed, ids)

    torch.testing.assert_close(got, _shard_sum_reference(shards, ids), rtol=0, atol=0)
    assert torch.count_nonzero(got[[0, 2, 3]]) == 0
    assert id(embed.weight) in retained


def test_should_differ_from_the_plain_replica_lookup_on_padded_tail_ids():
    shards = _shards(_padded_table())
    tail_ids = torch.tensor([ORG_VOCAB, 31], dtype=torch.int32)
    embed = _arm_rank(shards, rank=0)

    plain = rep.lookup(rep.full_weight(embed), tail_ids)

    assert not torch.equal(plain, _shard_sum_reference(shards, tail_ids))


def test_should_not_read_the_replica_in_decode_until_enabled():
    embed = _arm_rank(_shards(_padded_table()), rank=2)

    assert rep.decode_lookup(embed, torch.tensor([3], dtype=torch.int32)) is None


@pytest.mark.parametrize("decode", [True, False])
def test_should_serve_decode_from_the_replica_only_when_armed_for_decode(decode):
    shards = _shards(_padded_table())
    ids = torch.tensor([ORG_VOCAB, 4], dtype=torch.int32)

    rep.arm_if_eligible(
        shards[0],
        TP,
        lambda _shard: torch.cat([s.weight for s in shards], 0),
        decode=decode,
    )

    assert (rep.decode_lookup(shards[0], ids) is not None) is decode
