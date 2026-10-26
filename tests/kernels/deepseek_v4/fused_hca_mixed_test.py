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
"""Fused HCA mixed-step attention (T > 32) vs the stock SWA + MLA-p + MLA-m path and a float64 oracle."""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import absltest, parameterized

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "kmlam"))
import mlam_common as MC  # noqa: E402

pytestmark = pytest.mark.tpu_test

MC.CELLS.update({
    "ut-dec-ver-pf": MC.MCell("ut-dec-ver-pf", 16, 64, ((3000 + 64, 1), (2000 + 64, 7), (9000 + 3, 40)), 4, 300),
    "ut-fresh": MC.MCell("ut-fresh", 16, 128, ((1500 + 64, 7), (300, 120)), 4, 300),
    "ut-two-blocks": MC.MCell("ut-two-blocks", 16, 64, ((140000 + 37, 50),), 2, 400),
    "ut-h32-pf": MC.MCell("ut-h32-pf", 32, 256, ((60000 + 64, 7), (25000 + 3, 200)), 4, 300),
    "ut-ver-only": MC.MCell("ut-ver-only", 32, 64, ((50000 + 64, 7), (70000 + 3, 7)), 4, 300),
})
CELLS = ["ut-dec-ver-pf", "ut-fresh", "ut-two-blocks", "ut-h32-pf", "ut-ver-only", "edge-small", "edge-sink"]


def _err(a, ref):
    d = np.abs(np.asarray(a, np.float64) - ref)
    return float(d.max() / np.abs(ref).max()), float(d.mean())


class FusedHcaMixedTest(parameterized.TestCase):

    def _run(self, name, **knobs):
        MC.stock_env()
        from vllm_torchtpu.kernels.deepseek_v4.core_attention.fused_hca_mixed import fused_hca_mixed_attention
        inp = MC.build(MC.CELLS[name])
        d = MC.device(inp)
        so, sswa = jax.jit(MC.stock_attn())(d)

        def fused(d):
            return fused_hca_mixed_attention(
                d["q"], d["kv"], d["swa"], d["seq_lens"], d["swa_bt"], d["cu"], d["dist"], d["hca"],
                (d["positions"] + 1) // MC.RATIO, d["hca_bt"], d["sinks"], sm_scale=MC.C.SM_SCALE,
                sliding_window=MC.W, logical_page_size=MC.P, chunk=MC.VQ, **knobs)

        co, cswa = jax.jit(fused)(d)
        n = int(inp.cu[len(inp.cell.reqs)])
        ref = MC.oracle(inp)[:n]
        es = _err(np.asarray(so.astype(jnp.float32))[:n], ref)
        ec = _err(np.asarray(co.astype(jnp.float32))[:n], ref)
        tol = (max(2 * es[0], es[0] + 2.0 ** -8), max(2 * es[1], 1e-6))
        self.assertLessEqual(ec[0], tol[0], (name, es, ec))
        self.assertLessEqual(ec[1], tol[1], (name, es, ec))
        np.testing.assert_array_equal(np.asarray(cswa), np.asarray(sswa))
        np.testing.assert_array_equal(np.asarray(co.astype(jnp.float32))[n:], 0)

    @parameterized.parameters(*CELLS)
    def test_cell(self, name):
        self._run(name)

    @parameterized.parameters(("ut-two-blocks", dict(kt=256, blk_pages=32)), ("ut-h32-pf", dict(kt=1024)),
                              ("ut-two-blocks", dict(kt=2048, kt_tail=512, stats128=True, bq=64)),
                              ("ut-h32-pf", dict(kt=2048, kt_tail=512, stats128=True, bq=32, swa_merge=True)),
                              ("ut-dec-ver-pf", dict(kt=1024, kt_tail=256, stats128=True, row_split=2)),
                              ("ut-h32-pf", dict(kt=2048, kt_tail=512, stats128=True, bq=32, swa_fc=True)),
                              ("ut-fresh", dict(kt=2048, kt_tail=512, stats128=True, bq=64, swa_fc=True)),
                              ("ut-dec-ver-pf", dict(edge_mask_only=False)), ("ut-fresh", dict(bq=8)))
    def test_knobs(self, name, knobs):
        self._run(name, **knobs)

    def test_env_routing(self):
        """TPU_DSV4_FUSED_HCA_MIXED=1 routes _attention_hca at T > 32 to the kernel (bit-identical to calling it)."""
        MC.stock_env()
        from vllm_torchtpu.kernels.deepseek_v4.core_attention.fused_hca_mixed import fused_hca_mixed_attention
        inp = MC.build(MC.CELLS["ut-h32-pf"])
        d = MC.device(inp)
        os.environ["TPU_DSV4_FUSED_HCA_MIXED"] = "1"
        try:
            ro, rswa = jax.jit(MC.stock_attn())(d)
        finally:
            os.environ.pop("TPU_DSV4_FUSED_HCA_MIXED")
        co, cswa = jax.jit(lambda d: fused_hca_mixed_attention(
            d["q"], d["kv"], d["swa"], d["seq_lens"], d["swa_bt"], d["cu"], d["dist"], d["hca"],
            (d["positions"] + 1) // MC.RATIO, d["hca_bt"], d["sinks"], sm_scale=MC.C.SM_SCALE,
            sliding_window=MC.W, logical_page_size=MC.P, chunk=MC.VQ, bq=1024 // 32, kt=2048,
            kt_tail=512, stats128=True, swa_fc=True))(d)
        np.testing.assert_array_equal(np.asarray(ro.astype(jnp.float32)), np.asarray(co.astype(jnp.float32)))
        np.testing.assert_array_equal(np.asarray(rswa), np.asarray(cswa))


if __name__ == "__main__":
    absltest.main()
