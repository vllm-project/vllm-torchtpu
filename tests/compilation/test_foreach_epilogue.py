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
"""TPU_FOREACH_MUTATION_EPILOGUE: the rewrite only touches plain write-back
epilogues, keeps pairs and order, falls back to the original when a
destination requires grad, and a compiled graph's results are unchanged."""

import pytest
import torch
from torch._dynamo.backends.common import aot_autograd

from vllm_torchtpu.compilation import foreach_epilogue as fe

LEAF = (
    "def _apply_mutations(orig_inputs, updated_inputs):\n"
    "    if orig_inputs[0].requires_grad: "
    "orig_inputs[0].detach().copy_(updated_inputs[0])\n"
    "    else: orig_inputs[0].copy_(updated_inputs[0])\n"
    "    if orig_inputs[3].requires_grad: "
    "orig_inputs[3].detach().copy_(updated_inputs[1])\n"
    "    else: orig_inputs[3].copy_(updated_inputs[1])\n"
    "    orig_inputs[5].copy_(updated_inputs[2])"
)


def test_parse_pairs_plain_and_leaf():
    assert fe.parse_pairs(LEAF) == [(0, 0), (3, 1), (5, 2)]


@pytest.mark.parametrize(
    "bad",
    [
        LEAF + "\n    with torch.no_grad(): orig_inputs[6].set_(updated_inputs[3])",
        LEAF + "\n    orig_inputs[6].as_strided_("
        "updated_inputs[3].size(), updated_inputs[3].stride(), 0)",
        LEAF.replace(
            "else: orig_inputs[3].copy_", "else: orig_inputs[4].copy_"
        ),  # mismatched branch
        LEAF + "\n    orig_inputs[0].copy_(updated_inputs[3])",  # same input twice
        "def other(orig_inputs, updated_inputs):\n"
        "    orig_inputs[0].copy_(updated_inputs[0])",
    ],
)
def test_anything_else_is_kept(bad):
    assert fe.rewrite(bad) is None


def test_rewrite_semantics_and_fallback():
    src = fe.rewrite(LEAF)
    assert src is not None and "torch._foreach_copy_" in src
    orig_calls = []
    g = {
        "torch": torch,
        "_h5b_original_apply_mutations": lambda o, u: orig_calls.append(1),
    }
    exec(compile(src, "<t>", "exec"), g)
    fn = g["_apply_mutations"]
    orig = [torch.zeros(3) for _ in range(6)]
    upd = [torch.full((3,), float(i + 1)) for i in range(3)]
    fn(orig, upd)
    assert (
        torch.equal(orig[0], upd[0])
        and torch.equal(orig[3], upd[1])
        and torch.equal(orig[5], upd[2])
    )
    assert all(torch.equal(orig[i], torch.zeros(3)) for i in (1, 2, 4))
    assert orig_calls == []
    orig[3] = torch.zeros(3, requires_grad=True)
    fn(orig, upd)
    assert orig_calls == [1]  # the original epilogue handles autograd
    orig[3] = torch.zeros(3)
    upd_rg = list(upd)
    upd_rg[1] = torch.ones(3, requires_grad=True)  # a source that requires grad
    fn(orig, upd_rg)
    assert orig_calls == [1, 1]


def _backend():
    return aot_autograd(
        fw_compiler=lambda gm, ex: gm, keep_inference_input_mutations=False
    )


def test_compiled_graph_identical_with_patch(monkeypatch):
    from torch._functorch._aot_autograd import subclass_codegen as scg

    monkeypatch.setattr(scg, "_compile_and_exec_source", scg._compile_and_exec_source)
    monkeypatch.setattr(fe, "_installed", False)
    stats0 = dict(fe.STATS)

    def f(x, idx, caches):
        for i, c in enumerate(caches):
            c.index_copy_(0, idx, x + i)
        return x.sum()

    def run(compiled):
        torch._dynamo.reset()
        caches = [torch.zeros(16, 4) for _ in range(5)]
        fn = torch.compile(f, backend=_backend(), fullgraph=True) if compiled else f
        outs = []
        with torch.no_grad():
            for step in range(4):
                idx = torch.tensor([step, step + 8])
                x = torch.randn(2, 4, generator=torch.Generator().manual_seed(step))
                outs.append(fn(x, idx, caches))
        return outs, caches

    ref_out, ref_caches = run(False)
    base_out, base_caches = run(True)
    fe.install()
    new_out, new_caches = run(True)
    assert fe.STATS["rewritten"] > stats0["rewritten"]
    assert fe.STATS["pairs"] - stats0["pairs"] >= 5
    for a, b, c in zip(ref_caches, base_caches, new_caches):
        assert torch.equal(a, b) and torch.equal(a, c)
    for a, b, c in zip(ref_out, base_out, new_out):
        assert torch.equal(a, b) and torch.equal(a, c)
