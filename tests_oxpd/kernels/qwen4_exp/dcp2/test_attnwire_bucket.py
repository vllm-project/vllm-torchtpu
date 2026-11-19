# SPDX-License-Identifier: Apache-2.0
"""q38 L4 ATTNWIRE T1: L1-H3's open caveat (I) - the real attach path packs the draft attention kind's operands ONCE, at
max(driver.buckets) (`DraftAttn(..., max(driver.buckets))`), while every fused-loop pass builds the kernel at its own rows
count B (`DraftAttnKind.callable_for(block rows)`, the draft's loop bucket). No L1 CPU leg ran a kernel built at B < max
against operands packed at max.

1. B-invariance of the packed bytes, measured: each rank's operands through the real torch packer
   (`pack_draft_attn_operands`) at B = 8 / 16 / 32 / 48 / 64 vs at 64, sha256 per operand, with the production tiles
   (`segment_tiles(B)`); on TINY at (R, DCP) = (2, 1) / (2, 2) / (4, 4) and on the served geometry qwen38.FULL at TP 8,
   DCP 4 and 1. On TINY also: the production-tile pack equals the CPU-tile pack (L1's call-test tiles, hier 0: the
   production hier-2 all-reduce asserts R == 8, so CPU kernels build with L1's tiles) and L1's JAX packer.
2. The kernel built at each B (8 / 16 / 32 / 48 / 64 rows, DCP 1 / 2 / 4) against operands packed once at 64, vs L1's
   oracle of today's draft pass (test_l1h3_draft_attn_call._stock): every live row within 2 BF16 ulps (+ 1e-6 absolute; L1-H3's rule),
   >= 99.9 % of attention elements bitwise, finite, idle rows 0; every rank's pool after the pass byte-identical to today's.
Rules stated before data; each case prints one ATTNWIRE line.
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import attnwire_harness as aw  # noqa: E402
import test_l1h3_draft_attn_call as l1c  # noqa: E402

ROWS = [8, 16, 32, 48, 64]
PACK_AT = 64
GROUPS = [(2, 1), (2, 2), (4, 4)]


def _live(B):
    return max(1, B - 3)  # idle seats at every rows count


@pytest.mark.parametrize("R,D", GROUPS)
def test_pack_is_rows_invariant_tiny(R, D):
    from vllm_torchtpu.megakernel.models.qwen38_inputs import _rep, _rows
    from vllm_torchtpu.megakernel.models.qwen38_pack import pack_weights
    from vllm_torchtpu.megakernel import kernel, planner

    c = aw.case_b(500 + 10 * R + D, R, D, 1, 8, 5)
    ref, ref_sha = aw.pack_torch(c, R, D, PACK_AT)
    for B in ROWS:
        got, sha = aw.pack_torch(c, R, D, B)
        same = sha == ref_sha
        print(f"ATTNWIRE PACK TINY R={R} D={D} B={B} vs B={PACK_AT}: operands {len(sha[0])}, sha256 rank0 "
              f"{[s[:12] for s in sha[0]]}, equal {same}", flush=True)
        assert same, ("packed operands differ between rows counts", B, sha, ref_sha)
    # the CPU tiles (L1's call test) pack the same bytes, and so does L1's JAX packer
    cpu, cpu_sha = aw.pack_torch(c, R, D, PACK_AT, tiles=l1c.TILES)
    assert cpu_sha == ref_sha, ("production vs CPU tiles pack differently", cpu_sha, ref_sha)
    spec, ex = aw.spec_ex(c, R, D, PACK_AT, tiles=l1c.TILES)
    w = pack_weights(spec, ex, {"qkvg": c["qkvg"]},
                     {"qsa_qn": _rows(_rep(c["nq"][None], R), (0,), 1), "qsa_kn": _rows(_rep(c["nk"][None], R), (0,), 1)})
    names = list(planner.plan(spec, ex).w_names)
    # L1's JAX packer (the operands L1's KERN / CALL legs ran) vs the torch packer (the attach path): equal per operand after
    # dropping unit dims (the JAX fold keeps the [R, L] leading dims; the torch path folds them into the tile axis).
    rows = []
    for name, mine in zip(names, ref):
        theirs = np.asarray(kernel.fold(w[name]))
        mine = np.asarray(mine)
        sq_m = tuple(d for d in mine.shape if d != 1)
        sq_t = tuple(d for d in theirs.shape if d != 1)
        same = sq_m == sq_t and np.array_equal(mine.reshape(-1).view(np.uint8), theirs.reshape(-1).view(np.uint8))
        rows.append((name, mine.shape, theirs.shape, same))
    print(f"ATTNWIRE PACK TINY R={R} D={D}: production tiles == CPU tiles (sha256 per operand); torch packer vs L1's JAX packer "
          f"(torch shape, JAX shape, equal after dropping unit dims): {rows}", flush=True)
    assert all(r[3] for r in rows), ("torch packer differs from L1's JAX packer", rows)


@pytest.mark.parametrize("D", [4, 1])
def test_pack_is_rows_invariant_full_tp8(D):
    """The served geometry: qwen38.FULL, TP 8 (3 heads per rank), production tiles, random weights."""
    import torch

    from vllm_torchtpu.megakernel.models import qwen38

    R, g = 8, qwen38.FULL
    rng = np.random.default_rng(77 + D)
    nh = g.NH // R
    wq = [jnp.asarray(rng.normal(0, 0.02, (g.H, nh * 2 * g.HD)), jnp.bfloat16) for _ in range(R)]
    wkv = jnp.asarray(rng.normal(0, 0.02, (g.H, 2 * g.HD)), jnp.bfloat16)
    c = dict(g=g, wq=wq, wkv=wkv, nq=jnp.asarray(rng.normal(0, 0.05, (g.HD,)), jnp.float32),
             nk=jnp.asarray(rng.normal(0, 0.05, (g.HD,)), jnp.float32), pools=[np.zeros((1, 1), np.uint8)] * R,
             trig=jnp.zeros((8, 128), jnp.bfloat16), shared=np.zeros((1, 1), np.int32), main_slot=1)
    old = (aw.HD, aw.NH_RANK)
    aw.HD, aw.NH_RANK = g.HD, nh
    try:
        ref, ref_sha = aw.pack_torch(c, R, D, PACK_AT)
        for B in ROWS:
            got, sha = aw.pack_torch(c, R, D, B)
            same = sha == ref_sha
            print(f"ATTNWIRE PACK FULL TP8 D={D} B={B} vs B={PACK_AT}: shapes {[tuple(x.shape) for x in got]}, sha256 rank0 "
                  f"{[s[:12] for s in sha[0]]}, all ranks equal to B={PACK_AT}: {same}", flush=True)
            assert same, ("packed operands differ between rows counts", B)
    finally:
        aw.HD, aw.NH_RANK = old
    assert torch is not None


@pytest.mark.parametrize("B", ROWS)
@pytest.mark.parametrize("R,D", GROUPS)
def test_kernel_at_rows_b_with_operands_packed_at_max(rpa_stand_in, monkeypatch, R, D, B):
    l1c._interpret_keep_params(monkeypatch)
    l1c._route_any_dcp(monkeypatch)
    if D > 1:
        from vllm_torchtpu.models.vllm.qwen4_exp import pool_layout

        monkeypatch.setenv(pool_layout.MAIN_KV_DCP_ENV, str(D))
    live_n = _live(B)
    c = aw.case_b(900 + 100 * D + B, R, D, 1, B, live_n)
    w_max, sha_max = aw.pack_torch(c, R, D, PACK_AT)  # production tiles, ONCE at max (the attach path)
    _, sha_b = aw.pack_torch(c, R, D, B)
    _, sha_cpu = aw.pack_torch(c, R, D, PACK_AT, tiles=l1c.TILES)
    assert sha_cpu == sha_max, "the CPU-tile build would read a different packing"
    want, want_pools = l1c._stock(c, R, D)
    got, got_pools = aw.kernel_at(c, R, D, B, w_max, l1c.TILES)
    live = np.arange(c["n"]) < live_n
    st = aw.compare(got, want, live)
    byte_diff = [int((a != b).sum()) for a, b in zip(got_pools, want_pools)]
    changed = [int((b != np.asarray(p0).view(np.uint8)).sum()) for b, p0 in zip(want_pools, c["pools"])]
    print(f"ATTNWIRE BUCKET R={R} D={D} B={B} live={live_n} packed_at={PACK_AT} (pack bytes at B equal: {sha_b == sha_max}): "
          f"bitwise {st['exact']:.6f}, max ulps {st['max_ulps']:.1f}, beyond {aw.ULP_BOUND} ulps {st['bad']} (beyond 3: {st['bad3']}), scale "
          f"{st['scale']:.4f}; pool bytes differing {byte_diff} (today's pass changed {changed})", flush=True)
    assert st["finite"] and st["scale"] > 1e-3
    assert st["bad"] == 0, ("attention beyond 2 ulps", st)
    assert st["exact"] >= aw.EXACT_BOUND, ("bitwise share below 99.9 %", st)
    assert st["idle_zero"], "idle rows not 0"
    assert sum(changed) > 0, ("write control: today's pass wrote nothing", changed)
    assert byte_diff == [0] * R, ("pool bytes differ", byte_diff)
