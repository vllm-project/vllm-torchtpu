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
"""`ops.msa.gather_group_kv` with TPU_MINIMAX_M3_DCP_KV_GATHER_SHARE=1 returns
the same tensors as the full gather when the replicas of a KV head hold the
same k/v/index_q, for every rank of the DCP group."""

import ast
import importlib
import logging
import os
import pathlib
import types
import typing

import pytest
import torch


def _load_msa():
    """ops/msa.py, or, on a host without the TPU stack, just the definitions
    this test needs, read from its source."""
    try:
        return importlib.import_module("vllm_torchtpu.models.minimax_m3.ops.msa")
    except ImportError:
        pass
    path = pathlib.Path(__file__).parents[2] / (
        "src/vllm_torchtpu/models/minimax_m3/ops/msa.py"
    )
    tree = ast.parse(path.read_text())
    want = {"KV_GATHER_SHARE_ENV", "kv_gather_share_enabled", "gather_group_kv"}
    body = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in want)
        or (
            isinstance(node, (ast.Assign, ast.AnnAssign))
            and any(
                getattr(t, "id", None) in want
                for t in (node.targets if isinstance(node, ast.Assign) else [node.target])
            )
        )
    ]
    assert len(body) == len(want), [getattr(n, "name", n) for n in body]
    mod = types.ModuleType("msa_subset")
    mod.__dict__.update(
        os=os, torch=torch, Any=typing.Any, init_logger=logging.getLogger
    )
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), mod.__dict__)
    return mod


class _Group:
    """One all-gather of the DCP group: the first pass records each rank's
    input, the second returns their concatenation."""

    def __init__(self, inputs=None):
        self.inputs = [] if inputs is None else inputs
        self.recording = inputs is None

    def all_gather(self, x, dim):
        if self.recording:
            self.inputs.append(x.clone())
            shape = list(x.shape)
            shape[dim] *= DCP
            return torch.zeros(shape, dtype=x.dtype)
        return torch.cat(self.inputs, dim=dim)


DCP = 8


def _run(msa, ks, vs, iqs, replicas, share):
    rank_tensors = [torch.tensor([r], dtype=torch.int32) for r in range(DCP)]
    rec = _Group()
    for r in range(DCP):
        msa.gather_group_kv(
            ks[r], vs[r], iqs[r], rec, DCP, replicas, rank_tensors[r] if share else None
        )
    out = []
    for r in range(DCP):
        g = _Group(rec.inputs)
        out.append(
            msa.gather_group_kv(
                ks[r], vs[r], iqs[r], g, DCP, replicas,
                rank_tensors[r] if share else None,
            )
        )
    return out, rec.inputs


@pytest.mark.parametrize("tokens", [8192, 512, 24])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_share_equals_full_gather(monkeypatch, tokens, dtype):
    msa = _load_msa()
    replicas, d = 2, 128
    gen = torch.Generator().manual_seed(tokens)
    heads = DCP // replicas
    per_head = [
        [torch.randn(tokens, 1, d, generator=gen).to(dtype) for _ in range(3)]
        for _ in range(heads)
    ]
    ks = [per_head[r // replicas][0] for r in range(DCP)]
    vs = [per_head[r // replicas][1] for r in range(DCP)]
    iqs = [per_head[r // replicas][2] for r in range(DCP)]
    monkeypatch.setenv(msa.KV_GATHER_SHARE_ENV, "0")
    ref, ref_in = _run(msa, ks, vs, iqs, replicas, share=False)
    monkeypatch.setenv(msa.KV_GATHER_SHARE_ENV, "1")
    got, got_in = _run(msa, ks, vs, iqs, replicas, share=True)
    # Each rank sends 1 / replicas of the bytes.
    assert got_in[0].numel() * replicas == ref_in[0].numel()
    for r in range(DCP):
        for a, b in zip(ref[r], got[r]):
            assert a.shape == b.shape == (tokens, heads, d)
            assert a.is_contiguous() and b.is_contiguous()
            assert torch.equal(a, b)
    # Expected content: head h of every rank is per_head[h].
    for h in range(heads):
        for i in range(3):
            assert torch.equal(got[3][i][:, h], per_head[h][i][:, 0])


def test_share_falls_back_on_odd_tokens(monkeypatch):
    msa = _load_msa()
    monkeypatch.setenv(msa.KV_GATHER_SHARE_ENV, "1")
    t, d = 7, 128
    x = [torch.randn(t, 1, d) for _ in range(DCP)]
    out, inputs = _run(msa, x, x, x, 2, share=True)
    assert inputs[0].shape == (t, 3, d)  # the full gather
    assert out[0][0].shape == (t, DCP // 2, d)
