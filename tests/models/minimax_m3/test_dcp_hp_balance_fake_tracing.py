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
"""The 4-bit MSA DCP op over several KV heads traces with fake tensors and a
symbolic token count (as Dynamo runs it in ``capture_model``) with
TPU_MINIMAX_M3_DCP_KV_GATHER_HEAD_MAJOR on (the head-major scoring op
``minimax_m3_msa_dcp_fp4_score_hm_*``) and off, at the c64 DCP 8 and DCP 4
layouts, and gives the same output shapes both ways. Nothing lowers or
runs."""

import pytest
import torch

from tests.models.minimax_m3 import test_msa_index_reuse_fake_tracing as ft

pytest.importorskip("torch_tpu")

LAYOUTS = {
    "c64": ft.SERVED["c64"],
    "c64_dcp4": dict(ft.SERVED["c64"], dcp=4, g=2),
}
ENV = "TPU_MINIMAX_M3_DCP_KV_GATHER_HEAD_MAJOR"
# Scales unique to this file (op names carry the scale; a name registered
# twice in one process corrupts torch's op registry).
_SCALES = {
    (name, hm): ft.SCALE * (1.5 + 0.125 * i)
    for i, (name, hm) in enumerate((n, h) for n in LAYOUTS for h in (False, True))
}


@pytest.fixture(scope="module")
def build():
    from vllm_torchtpu.distributed import dcp as dcp_mod
    from vllm_torchtpu.models.minimax_m3.ops import msa as msa_ops

    real = dcp_mod.get_dcp_group
    mp = pytest.MonkeyPatch()

    def make(name, hm):
        key = ("hpbal", name, hm)
        if key in ft._KEEP:
            return ft._KEEP[key]
        shape = LAYOUTS[name]
        dcp_mod.get_dcp_group = lambda: ft._Group(shape["dcp"])
        mp.setenv(ENV, "1" if hm else "0")
        try:
            op = msa_ops.get_msa_dcp_op(
                _SCALES[name, hm],
                ft.TOPK,
                0,
                1,
                shape["dcp"],
                shape["reps"],
                fp4=True,
                decode_rows=4,
            )
        finally:
            dcp_mod.get_dcp_group = real
            mp.undo()
        ft._KEEP[key] = op
        return op

    yield make
    msa_ops.get_msa_dcp_op.cache_clear()


@pytest.mark.parametrize("name", sorted(LAYOUTS))
def test_head_major_score_op_symbolic(name, build, monkeypatch):
    from vllm_torchtpu.models.minimax_m3.ops import msa as msa_ops

    shape = LAYOUTS[name]
    off, on = build(name, False), build(name, True)
    calls = []
    real_hm = msa_ops.gather_group_kv_head_major

    def spy(*args):
        calls.append(args[0].shape[0])
        return real_hm(*args)

    mode, seed = ft._mode()
    with mode:
        num = seed.shape[0]
        want = ft._layer_call(off, shape, num, {})
        monkeypatch.setattr(msa_ops, "gather_group_kv_head_major", spy)

        def no_old(*args):
            raise AssertionError("token-major gather on the head-major path")

        monkeypatch.setattr(msa_ops, "gather_group_kv", no_old)
        got = ft._layer_call(on, shape, num, {})
    assert calls, "head-major gather not taken"
    assert got.shape == want.shape and got.dtype == want.dtype
    assert got.shape[0] == num and isinstance(got.shape[0], torch.SymInt)


@pytest.mark.parametrize("name", sorted(LAYOUTS))
def test_head_major_score_op_dynamo(name, build):
    """Dynamo over the whole op with head-major gather on: fullgraph,
    meta tensors, dynamic token dim."""
    shape = LAYOUTS[name]
    op = build(name, True)
    num = ft.TOKENS
    meta = ft._Metadata(
        *[t.to("meta") for t in ft._meta(shape)], ft._empty((3,), ft.I32).to("meta")
    )
    args = [
        ft._empty((num, shape["heads"], ft.HEAD_DIM), ft.BF16),
        ft._empty((num, 1, ft.HEAD_DIM), ft.BF16),
        ft._empty((num, 1, ft.HEAD_DIM), ft.BF16),
        ft._empty((num, 1, ft.HEAD_DIM), ft.BF16),
        ft._empty((num, ft.HEAD_DIM), ft.BF16),
    ]
    args = [a.to("meta") for a in args]
    for a in args:
        torch._dynamo.mark_dynamic(a, 0)
    caches = [ft._kv(shape).to("meta"), ft._index(shape).to("meta")]
    extra = dict(
        dcp_rank=ft._empty((1,), ft.I32).to("meta"),
        scale_cache=ft._scale(shape).to("meta"),
        hadamard=ft._empty((ft.HEAD_DIM, ft.HEAD_DIM), ft.BF16).to("meta"),
    )
    torch._dynamo.reset()

    def fn(q, k, v, iq, ik):
        return op(q, k, v, iq, ik, *caches, meta, **extra) + 1

    out = torch.compile(fn, backend="eager", fullgraph=True)(*args)
    assert out.shape == (num, shape["heads"], ft.HEAD_DIM)
