# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_QWEN4_EXP_SP_EMBED_RS: the sequence-parallel graph's token embedding,
reduce-scattered to each rank's rows, is bitwise the all-reduced embedding
sliced by ``sp.shard``.

TP ranks are threads exchanging tensors through a barrier. The all-reduce path
runs vLLM's own ``VocabParallelEmbedding.forward``; the reduce-scatter path runs
``Qwen4ExpBackbone._sp_embed_shard`` through ``sequence_parallel``'s layout code.
"""

import threading
from types import SimpleNamespace as NS

import pytest
import torch
import torch.nn.functional as F
from vllm.model_executor.layers import vocab_parallel_embedding as vpe

from vllm_torchtpu.models.vllm.qwen4_exp import model as qwen4_model
from vllm_torchtpu.models.vllm.qwen4_exp import sequence_parallel as sp

TP = 8
PER_RANK = 16
VOCAB = TP * PER_RANK
HIDDEN = 32
TOKENS = 64


class ThreadTP:
    """Collectives over threads; each thread is one TP rank."""

    def __init__(self, size):
        self.size = size
        self.barrier = threading.Barrier(size)
        self.slots = [None] * size
        self.local = threading.local()

    @property
    def rank(self):
        return self.local.rank

    def exchange(self, x):
        self.barrier.wait()
        self.slots[self.rank] = x
        self.barrier.wait()
        values = list(self.slots)
        self.barrier.wait()
        return values

    def _sum(self, values):
        out = values[0]
        for v in values[1:]:
            out = out + v  # in the tensor's own dtype, as the wire sums it
        return out

    def all_reduce(self, x):
        return self._sum(self.exchange(x))

    def reduce_scatter(self, x):
        # A different summation order from the all-reduce on purpose.
        return self._sum(self.exchange(x)[::-1]).chunk(self.size)[self.rank].clone()

    def shard(self, x):
        return x.chunk(self.size, dim=0)[self.rank].clone()

    def run(self, fn):
        results, errors = [None] * self.size, []

        def body(rank):
            self.local.rank = rank
            try:
                results[rank] = fn(rank)
            except BaseException as exc:  # pragma: no cover - surfaced below
                errors.append(exc)
                self.barrier.abort()

        threads = [threading.Thread(target=body, args=(r,)) for r in range(self.size)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise errors[0]
        return results


def _table():
    g = torch.Generator().manual_seed(0)
    table = torch.randn(VOCAB, HIDDEN, generator=g).to(torch.bfloat16)
    table[3, :5] = -0.0  # signed zeros: both paths must agree on them
    table[PER_RANK, 1] = -0.0
    return table


def _embedding(table, rank):
    """A rank's vocab-parallel embedding, as vLLM lays it out (no added vocab)."""
    start, end = rank * PER_RANK, (rank + 1) * PER_RANK
    return NS(
        tp_size=TP,
        use_fused_embedding=False,
        reduce_results=True,
        parallel_group=None,
        weight=table[start:end].clone(),
        embedding_dim=HIDDEN,
        shard_indices=NS(
            org_vocab_start_index=start,
            org_vocab_end_index=end,
            num_org_vocab_padding=0,
            added_vocab_start_index=VOCAB,
            added_vocab_end_index=VOCAB,
        ),
        quant_method=NS(embedding=lambda layer, ids: F.embedding(ids, layer.weight)),
    )


@pytest.fixture
def tp8(monkeypatch, request):
    tp = ThreadTP(TP)
    sp.reset_microbatch()
    monkeypatch.setattr(sp, "reduce_scatter_rows", tp.reduce_scatter)
    monkeypatch.setattr(sp, "all_gather_rows", lambda x: torch.cat(tp.exchange(x)))
    monkeypatch.setattr(sp, "chunk_rows", tp.shard)
    monkeypatch.setattr(sp, "tp_world_size", lambda: TP)
    monkeypatch.setattr(sp, "_microbatch_decision", request.param)
    monkeypatch.setattr(vpe, "tensor_model_parallel_all_reduce", tp.all_reduce)
    # vLLM wraps the mask in torch.compile(dynamic=True); run it eager on CPU in
    # both paths (the same function, so the comparison stays like for like).
    eager = getattr(
        vpe.get_masked_input_and_mask,
        "_torchdynamo_orig_callable",
        vpe.get_masked_input_and_mask,
    )
    monkeypatch.setattr(vpe, "get_masked_input_and_mask", eager)
    monkeypatch.setattr(qwen4_model, "get_masked_input_and_mask", eager)
    monkeypatch.setattr(qwen4_model.envs, "VLLM_TPU_QWEN4_EXP_SP_EMBED_RS", True)
    yield tp
    sp.reset_microbatch()


def _ids():
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, VOCAB, (TOKENS,), generator=g)
    ids[:4] = torch.tensor([3, PER_RANK, PER_RANK - 1, VOCAB - 1])  # shard edges
    return ids


def _compare(tp, table, ids, rs_embedding=_embedding):
    """Per rank: (RS rows bitwise == sp.shard(AR rows), RS rows == table rows)."""

    def body(rank):
        emb = _embedding(table, rank)
        reference = sp.shard(vpe.VocabParallelEmbedding.forward(emb, ids))
        backbone = NS(embed_tokens=rs_embedding(table, rank))
        assert qwen4_model.Qwen4ExpBackbone._sp_embed_rs(backbone)
        mine = qwen4_model.Qwen4ExpBackbone._sp_embed_shard(backbone, ids)
        same = (
            mine.dtype == reference.dtype
            and mine.shape == reference.shape
            and torch.equal(mine.view(torch.int16), reference.view(torch.int16))
        )
        # By value: an owner's -0.0 plus the other ranks' +0.0 is +0.0, in the
        # all-reduce and the reduce-scatter alike (the bitwise check above).
        truth = sp.shard(table[ids])
        return same, torch.equal(mine, truth)

    return tp.run(body)


@pytest.mark.parametrize("tp8", [False, True], ids=["mb_off", "mb_on"], indirect=True)
def test_reduce_scattered_embedding_is_bitwise_the_all_reduced_shard(tp8):
    results = _compare(tp8, _table(), _ids())
    assert all(same for same, _ in results)
    assert all(truth for _, truth in results)


@pytest.mark.parametrize("tp8", [False, True], ids=["mb_off", "mb_on"], indirect=True)
def test_a_wrong_owner_mask_is_caught(tp8):
    """Mutation control: rank 0 also claims rank 1's first vocab row (an owner
    range one past its shard). That row then has two non-zero owners and the
    comparison must fail."""
    table, ids = _table(), _ids()

    def overlapping(table, rank):
        emb = _embedding(table, rank)
        if rank == 0:
            emb.shard_indices.org_vocab_end_index += 1
            emb.weight = table[: PER_RANK + 1].clone()
        return emb

    assert PER_RANK in ids.tolist()
    results = _compare(tp8, table, ids, rs_embedding=overlapping)
    assert not all(same for same, _ in results)


def test_flag_off_or_unsupported_layout_keeps_the_all_reduce(monkeypatch):
    emb = _embedding(_table(), 0)
    backbone = NS(embed_tokens=emb)
    monkeypatch.setattr(qwen4_model.envs, "VLLM_TPU_QWEN4_EXP_SP_EMBED_RS", False)
    assert not qwen4_model.Qwen4ExpBackbone._sp_embed_rs(backbone)
    monkeypatch.setattr(qwen4_model.envs, "VLLM_TPU_QWEN4_EXP_SP_EMBED_RS", True)
    assert qwen4_model.Qwen4ExpBackbone._sp_embed_rs(backbone)
    for field, value in [
        ("tp_size", 1),
        ("reduce_results", False),
        ("parallel_group", object()),
        ("use_fused_embedding", True),
        ("weight", emb.weight.float()),
    ]:
        changed = NS(**{**vars(emb), field: value})
        assert not qwen4_model.Qwen4ExpBackbone._sp_embed_rs(NS(embed_tokens=changed))
    # The pinned vLLM's embedding has no reduce_results field: it always reduces.
    pinned = NS(**{k: v for k, v in vars(emb).items() if k != "reduce_results"})
    assert qwen4_model.Qwen4ExpBackbone._sp_embed_rs(NS(embed_tokens=pinned))
