# SPDX-License-Identifier: Apache-2.0
"""q38 L4 X-D14-DRAFT-2 (a): the MTP draft's vocab-parallel embedding all-reduce
through the armed H8 decode all-reduce (VLLM_TPU_DRAFT_EMBED_H8, default off).

- ``VocabParallelEmbedding._tpu_partial`` is the TPU patched forward's tp > 1
  branch up to its all-reduce: summed over the TP ranks it is the full lookup,
  and patched_forward == all_reduce(_tpu_partial) with one all-reduce call.
- ``Qwen4ExpMultiTokenPredictor.embed`` sends the partial to the armed H8 op when
  the flag is on and H8 takes the tensor, to the stock all-reduce when H8 is not
  armed, and leaves the embedding's own forward alone when the flag is off.
- The flag is read once at build time and is part of the hashed compile key.
"""

import ast
import os
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest
import torch
import torch.nn.functional as F
from torch import nn

V, H, TP = 16, 8, 2


def _shard(rank, weight):
    import vllm.model_executor.layers.vocab_parallel_embedding as vpe

    import vllm_torchtpu

    vllm_torchtpu._patch_vllm_vocab_parallel_embedding()
    emb = object.__new__(vpe.VocabParallelEmbedding)
    nn.Module.__init__(emb)
    per = V // TP
    emb.tp_size = TP
    emb.weight = nn.Parameter(weight[rank * per:(rank + 1) * per].clone(), requires_grad=False)
    emb.quant_method = NS(embedding=lambda layer, ids: F.embedding(ids, layer.weight))
    emb.shard_indices = NS(
        org_vocab_start_index=rank * per,
        org_vocab_end_index=(rank + 1) * per,
        num_org_vocab_padding=0,
        added_vocab_start_index=V,
        added_vocab_end_index=V,
    )
    return emb


class _Rec:
    def __init__(self, scale=1):
        self.calls = []
        self.scale = scale

    def __call__(self, x):
        self.calls.append(x.clone())
        return x * self.scale


def test_partial_sums_to_the_full_lookup_and_matches_patched_forward(monkeypatch):
    import vllm.distributed as vd

    weight = torch.randn(V, H, dtype=torch.float32)
    ids = torch.tensor([0, 3, 7, 8, 12, 15, 9, 1])
    shards = [_shard(r, weight) for r in range(TP)]
    parts = [s._tpu_partial(ids) for s in shards]
    assert torch.equal(sum(parts), weight[ids])
    # Rows a rank does not own are exact zeros in its partial.
    owned0 = ids < V // TP
    assert torch.equal(parts[0][~owned0], torch.zeros_like(parts[0][~owned0]))
    assert torch.equal(parts[1][owned0], torch.zeros_like(parts[1][owned0]))
    rec = _Rec()
    monkeypatch.setattr(vd, "tensor_model_parallel_all_reduce", rec)
    for s, p in zip(shards, parts):
        rec.calls.clear()
        out = s(ids)  # the TPU patched forward
        assert len(rec.calls) == 1 and torch.equal(rec.calls[0], p) and torch.equal(out, p)


def _embed(emb, flag):
    from vllm_torchtpu.models.vllm.qwen4_exp import mtp

    return mtp.Qwen4ExpMultiTokenPredictor.embed(NS(embed_tokens=emb, embed_h8=flag), ids_bf16())


def ids_bf16():
    return torch.tensor([0, 5, 9, 14, 2, 11, 7, 8])


def test_draft_embed_routes_the_partial(monkeypatch):
    import vllm.distributed as vd

    from vllm_torchtpu.layers.adapter import tp_ar_decode

    weight = torch.randn(V, H).to(torch.bfloat16)
    emb = _shard(1, weight)
    partial = emb._tpu_partial(ids_bf16())
    stock = _Rec(scale=3)
    monkeypatch.setattr(vd, "tensor_model_parallel_all_reduce", stock)
    h8 = _Rec(scale=2)
    # Armed: flag on -> the H8 op, once, on the partial; the stock all-reduce is not called.
    monkeypatch.setattr(tp_ar_decode, "_ARMED", (h8, H))
    out = _embed(emb, True)
    assert len(h8.calls) == 1 and torch.equal(h8.calls[0], partial) and not stock.calls
    assert torch.equal(out, partial * 2)
    # Armed for another hidden size: H8 does not take it -> stock all-reduce of the partial.
    h8.calls.clear()
    monkeypatch.setattr(tp_ar_decode, "_ARMED", (h8, H + 1))
    out = _embed(emb, True)
    assert not h8.calls and len(stock.calls) == 1 and torch.equal(stock.calls[0], partial)
    assert torch.equal(out, partial * 3)
    # Not armed: stock all-reduce of the partial.
    stock.calls.clear()
    monkeypatch.setattr(tp_ar_decode, "_ARMED", None)
    out = _embed(emb, True)
    assert not h8.calls and len(stock.calls) == 1 and torch.equal(out, partial * 3)
    # Flag off, armed: the embedding's own forward (one stock all-reduce), never H8.
    stock.calls.clear()
    monkeypatch.setattr(tp_ar_decode, "_ARMED", (h8, H))
    out = _embed(emb, False)
    assert not h8.calls and len(stock.calls) == 1 and torch.equal(out, partial * 3)


def test_tp1_takes_the_plain_forward(monkeypatch):
    from vllm_torchtpu.layers.adapter import tp_ar_decode

    h8 = _Rec()
    monkeypatch.setattr(tp_ar_decode, "_ARMED", (h8, H))
    plain = nn.Embedding(V, H)
    out = _embed(plain, True)
    assert not h8.calls and torch.equal(out, plain(ids_bf16()))


def test_flag_default_and_build_time_read():
    env = {k: v for k, v in os.environ.items() if k != "VLLM_TPU_DRAFT_EMBED_H8"}
    code = "from vllm_torchtpu import envs; print('FLAG', envs.VLLM_TPU_DRAFT_EMBED_H8)"
    for extra, want in (({}, "FLAG False"), ({"VLLM_TPU_DRAFT_EMBED_H8": "1"}, "FLAG True")):
        out = subprocess.run([sys.executable, "-c", code], env={**env, **extra},
                             capture_output=True, text=True, timeout=600)
        assert out.returncode == 0, out.stderr[-2000:]
        assert [ln for ln in out.stdout.splitlines() if ln.startswith("FLAG ")] == [want], out.stdout
    here = os.path.dirname(os.path.abspath(__file__))
    src = open(os.path.join(here, "..", "..", "src", "vllm_torchtpu", "models", "vllm",
                            "qwen4_exp", "mtp.py")).read()
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "Qwen4ExpMultiTokenPredictor")
    meth = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    init_src = ast.get_source_segment(src, meth["__init__"])
    fuse_src = ast.get_source_segment(src, meth["fuse_inputs"])
    assert "self.embed_h8 = bool(vllm_torchtpu_envs.VLLM_TPU_DRAFT_EMBED_H8)" in init_src
    assert "self.embed(input_ids)" in fuse_src and "self.embed_tokens(" not in fuse_src
