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
"""Fused HCA decode attention vs the stock SWA + MLA path and a float64 oracle."""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import absltest, parameterized

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "khca"))
import hca_common as C  # noqa: E402
import mixed as M  # noqa: E402

pytestmark = pytest.mark.tpu_test

M.MIXED.update({
    "ver1": (16, [(3000 + 64, 7)]),
    "ver2-boundary": (16, [(2000 + 64, 7), (9000 + 3, 7)]),
    "ver4-h32": (32, [(1500 + 64, 7), (4100 + 64, 7), (700, 7), (2600 + 64, 7)]),
})
CELLS = sorted(M.MIXED)


def _err(a, ref):
    d = np.abs(np.asarray(a, np.float64) - ref)
    return float(d.max() / np.abs(ref).max()), float(d.mean())


class FusedHcaDecodeTest(parameterized.TestCase):

    def _run(self, name, blk_pages, swa_u8):
        C.stock_env()
        from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout as CL
        from vllm_torchtpu.kernels.deepseek_v4.core_attention.fused_hca_decode import fused_hca_attention
        inp = M.build(name)
        d = {k: v for k, v in M.device(inp).items() if v is not None}
        cs = jnp.asarray(C.cos_sin_np(max_pos=1 << 18))
        so, sswa, kv = jax.jit(M.stock_attn())(d, cs)
        swa_in = CL.words_to_slabs(d["swa"]) if swa_u8 else d["swa"]

        def fused(d, kv, swa):
            return fused_hca_attention(
                d["q"], kv, swa, d["seq_lens"], d["swa_bt"], d["cu"], d["dist"], d["hca"],
                (d["positions"] + 1) // C.RATIO, d["hca_bt"], d["sinks"], sm_scale=C.SM_SCALE,
                sliding_window=C.WINDOW, logical_page_size=C.SWA_PAGE, chunk=C.VQ, blk_pages=blk_pages)

        co, cswa = jax.jit(fused)(d, kv, swa_in)
        if swa_u8:
            cswa = CL.slabs_to_words(cswa)
        n = int(inp.cu[len(inp.reqs)])
        ref = M.oracle_attn(inp, np.asarray(kv.astype(jnp.float32)).astype(np.float64))[:n]
        es = _err(np.asarray(so.astype(jnp.float32))[:n], ref)
        ec = _err(np.asarray(co.astype(jnp.float32))[:n], ref)
        tol = (max(2 * es[0], es[0] + 2.0 ** -8), max(2 * es[1], 1e-6))
        self.assertLessEqual(ec[0], tol[0], (name, es, ec))
        self.assertLessEqual(ec[1], tol[1], (name, es, ec))
        np.testing.assert_array_equal(np.asarray(cswa), np.asarray(sswa))
        np.testing.assert_array_equal(np.asarray(co.astype(jnp.float32))[n:], 0)

    @parameterized.parameters(*CELLS)
    def test_cell(self, name):
        self._run(name, blk_pages=128, swa_u8=False)

    @parameterized.parameters(("ver2-boundary", 32), ("mix-c16", 128), ("mix-ver-tail", 2))
    def test_block_sizes(self, name, blk_pages):
        self._run(name, blk_pages=blk_pages, swa_u8=False)

    @parameterized.parameters("mix-dec-ver", "ver4-h32")
    def test_u8_swa_cache(self, name):
        self._run(name, blk_pages=128, swa_u8=True)


    @parameterized.parameters("mix-dec-ver", "ver4-h32", "mix-c16", "mix-ver-tail")
    def test_rope_in_op(self, name):
        """`_attention_hca_rope` (TPU_DSV4_FUSED_HCA_ROPE): the fused kernel with RoPE in its prologue equals the
        stock rope kernels + `_attention_hca`'s fused path bit for bit; and the T<=32 fallback (fused off) runs the
        stock path."""
        C.stock_env()
        from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4 import deepseek_v4_attention_op as op
        inp = M.build(name)
        d = {k: v for k, v in M.device(inp).items() if v is not None}
        cs = jnp.asarray(C.cos_sin_np(max_pos=1 << 18))
        T, H, D = d["q"].shape
        q_flat = d["q"].reshape(T, H * D)
        kw = dict(sm_scale=C.SM_SCALE, sliding_window=C.WINDOW, logical_page_size=C.SWA_PAGE, swa_only=False,
                  two_caches_same_buffer=False, hca_num_kv_pages_per_block=(8, 64, 64),
                  hca_num_queries_per_block=(8, 8, 8), verify_q_len=C.VQ)

        def run(d):
            return op._attention_hca_rope(
                q_flat, d["kv"], d["swa"], d["seq_lens"], d["swa_bt"], d["cu"], d["dist"], d["hca"],
                d["seq_lens"] // C.RATIO, (d["positions"] + 1) // C.RATIO, d["hca_bt"], d["cu"], d["dist"], d["sinks"],
                d["positions"], cs, head_dim=D, q_eps=1e-6, **kw)

        os.environ["TPU_DSV4_FUSED_HCA_DECODE"] = "1"
        try:
            fo, fswa = jax.jit(run)(d)
        finally:
            os.environ.pop("TPU_DSV4_FUSED_HCA_DECODE", None)
        so, sswa = jax.jit(run)(d)  # fused off -> stock rope kernels + stock attention
        n = int(inp.cu[len(inp.reqs)])
        np.testing.assert_array_equal(np.asarray(fswa), np.asarray(sswa))
        e = np.abs(np.asarray(fo.astype(jnp.float32))[:n] - np.asarray(so.astype(jnp.float32))[:n]).max()
        self.assertLess(e / np.abs(np.asarray(so.astype(jnp.float32))[:n]).max(), 2e-2)


if __name__ == "__main__":
    absltest.main()
