# SPDX-License-Identifier: Apache-2.0
"""STACK-D v5 composition (q38 L1-V5): GDN-REL-FETCH (VLLM_TPU_MK_GDN_INK=2) x X-GLUE-1 (VLLM_TPU_MK_GDN_CKPT_INK=1), CPU
interpret (the verifier's gap: GDN-REL-FETCH's bitwise proof never ran with X-GLUE on top).

Where the two meet: X-GLUE-1 writes the GDN checkpoints into the pool from inside the pool-GDN block (through the pool
gather's FP8 staging buffers), and GDN-REL-FETCH changes how the dense weight-stream refills are issued inside every layer
(`Tiles.rel_fetch`). Harness: X-GLUE-1's own pool-GDN segment step (tests/megakernel/test_glue_ckpt_ink.py `_run`: R = 2,
qwen38.TINY, the served pool layout, real write ids incl. null ids; the kernel's own pool output with the flag on), with
GDN-REL-FETCH applied as L1-GDNINK's test applies it (VLLM_TPU_MK_GDN_INK=2 and `rel_fetch=True` on the test tiles).
Limit (stated): no CPU harness runs the WHOLE step with the pool-GDN block (test_wholestep's whole step takes GDN states as
inputs, so X-GLUE-1 does not apply there); this is the level at which X-GLUE-1's own bitwise proof runs, and GDN-REL-FETCH's
whole-step bitwise proof (all leaves, DCP 1 / 2 / 4) is its own runner's.
Rules (stated before data), per case (16 rows with half / rewrite id patterns, 48 rows with the served chunk gather):
  * engagement: TwoRing.fetch_rel is called in the INK=2 build only; the X-GLUE-1 kernel (POOL_OUT) in both;
  * bitwise: INK=2 + X-GLUE-1 == INK=0 + X-GLUE-1: residual bytes and every pool byte (the in-kernel checkpoint writes) on
    both ranks; and INK=2 + X-GLUE-1 == today's path (X-GLUE-1 off: the kernel's checkpoints scattered), X-GLUE-1's own
    `_check` (blocks written = the live write ids, nothing else).
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from tests.megakernel import test_glue_ckpt_ink as GL

INK_ENV = "VLLM_TPU_MK_GDN_INK"
# (B, kb, pattern, chunk_gather, head_major)
CASES = {
    "b16_half_k512": (16, 512, "half", "0", False),
    "b16_rw_k128": (16, 128, "rw", "0", False),
    "b48_all_k128_cg": (48, 128, "all", "1", True),
}


@pytest.mark.parametrize("case", list(CASES))
def test_gdn_rel_fetch_with_glue_is_bitwise(monkeypatch, case):
    from vllm_torchtpu.megakernel import stream

    B, kb, pattern, cg, hm = CASES[case]
    kw = dict(kb=kb, pattern=pattern, chunk_gather=cg, head_major=hm, seed=5 + B)
    calls = {"fetch_rel": 0}
    real = stream.TwoRing.fetch_rel

    def fetch_rel(self, *a, **k):
        calls["fetch_rel"] += 1
        return real(self, *a, **k)

    monkeypatch.setattr(stream.TwoRing, "fetch_rel", fetch_rel)
    monkeypatch.delenv(INK_ENV, raising=False)
    base_tiles = GL.TILES
    today = GL._run(monkeypatch, B, on=False, **kw)  # INK=0, X-GLUE-1 off: the kernel's checkpoints scattered
    n0 = calls["fetch_rel"]
    glue0 = GL._run(monkeypatch, B, on=True, **kw)  # INK=0 + X-GLUE-1
    n1 = calls["fetch_rel"]
    monkeypatch.setenv(INK_ENV, "2")
    monkeypatch.setattr(GL, "TILES", dataclasses.replace(base_tiles, rel_fetch=True))
    try:
        glue2 = GL._run(monkeypatch, B, on=True, **kw)  # INK=2 + X-GLUE-1
    finally:
        monkeypatch.setattr(GL, "TILES", base_tiles)
        monkeypatch.delenv(INK_ENV, raising=False)
    n2 = calls["fetch_rel"]
    assert n0 == 0 and n1 == n0, ("fetch_rel ran without INK=2", n0, n1)
    assert n2 > n1, ("GDN-REL-FETCH not engaged: no fetch_rel call in the INK=2 build", n1, n2)
    assert np.array_equal(glue2["pool_in"], glue0["pool_in"])
    nres = int((GL._bits(glue2["res"]) != GL._bits(glue0["res"])).sum())
    npool = int((glue2["pool"] != glue0["pool"]).sum())
    print(f"V5 GDNINKxGLUE {case}: fetch_rel calls INK=2 {n2 - n1} (INK=0: 0); INK=2 + X-GLUE vs INK=0 + X-GLUE: residual "
          f"bytes differing {nres}, pool bytes differing {npool}", flush=True)
    assert nres == 0 and npool == 0, (case, nres, npool)
    GL._check(today, glue2, f"v5 {case} INK=2 + X-GLUE-1 vs today's scatter")
