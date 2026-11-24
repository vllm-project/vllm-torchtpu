#!/usr/bin/env python3
"""XFER-4: byte identity and cost of QWEN4_EXP_PD_REPACK_V2 against today's repack, on one TPU chip.

Runs in the infx-qwen38-flash-bench pod (BENCH=script) from the vllm-torchtpu tree root:

    PYTHONPATH=src python3 q38_xfer4/q38_xfer4_repack_v2_bench.py --sizes 4800,20000,81000,160000 --reps 10
    (writes $CDK_OUTPUT_DIR/q38f-bench/script/q38_xfer4_repack_v2.json and prints Q38_XFER4_REPACKV2 lines;
     also writes the recipe's run01_A/result.json via q38_script_result, ok = verdict PASS)

What runs (C): the served StagedConsumer methods themselves, on a shell consumer (object.__new__) holding
a real ``make_repack`` fn (TPU defaults: dma_row_gather_u8 and the aliased Pallas writer) with the served
S12 routed compact-GDN geometry (12 FA layers, 36 GDN owners; TP1 3,072 SSM + 128 conv rows -> TP8 384 + 16;
512-row staging and pool pages; TP8 + main-KV DCP4; rank 5):
  v1  ``_run_chunks(repack_chunks(...))``: today's served path (``_build_op``: jax_op + torch.compile);
  v2  ``_run_chunks(repack_programs_v2(..., 36))``: the flag's path (``_run_programs_v2``: one packed H2D and
      one compiled run per program layout), after ``warm_repack_v2`` compiled every layout.
Both start from the same pool (random bytes, null page 0 zero, as serving's) and staging (random, page 0 zero).

Gates, per size (all must hold):
  G1 identity  the v2 pool equals the v1 pool byte for byte, over every page;
  G2 reference v1's written rows equal an independent numpy reference (main KV: the rank's head pair, zeros
               after; compressed / ring / PLE rows unchanged; GDN: the staged SSM rows, then the rank's conv
               channels), so G1 is anchored to the right bytes;
  G3 mutant    a v2 landing with its last GDN owner dropped must FAIL G1 (the check bites).
Measured, not gated: programs per landing; host dispatch ms per landing (the HOST_PHASES kv_connector cost)
and dispatch-to-ready ms (a one-element read-back of the pool), medians over --reps, ABAB interleaved; the
warm compile seconds.
--cpu-smoke: a small geometry on the CPU with jax only (the fn called directly; no torch_tpu): G1-G3 on the
programs, to check the plumbing before the TPU run.
"""

import argparse
import json
import os
import statistics
import sys
import time
from types import SimpleNamespace

import numpy as np

ROWS, LANES = 4, 256


def geometry(cpu_smoke: bool):
    if cpu_smoke:
        return dict(stage_rows=128, dst_block=128, ratio=4, fa_layers=2, gdn_layers=5, dcp=2, tp=8, rank=5,
                    src=(16, 8, 8, 16), dst=(2, 1), conv_width=3, ring_rows=16, gdn_v2=5)
    return dict(stage_rows=512, dst_block=512, ratio=4, fa_layers=12, gdn_layers=36, dcp=4, tp=8, rank=5,
                src=(48, 16, 128, 128), dst=(6, 2), conv_width=3, ring_rows=64, gdn_v2=36)


def gdn_geoms(st, g):
    v, k, hk, hv = g["src"]
    vd, kd = g["dst"]
    cw = g["conv_width"]
    tok = ROWS * LANES

    def rows_for(vh, kh):
        ssm = -(-vh * hk * hv * 4 // tok)
        conv_bytes = cw * (2 * kh * hk + vh * hv) * 4
        conv = 1
        while conv * tok < conv_bytes:
            conv *= 2
        return ssm, conv

    s_ssm, s_conv = rows_for(v, k)
    d_ssm, d_conv = rows_for(vd, kd)
    s = st.GdnGeometry(ssm_tokens=s_ssm, conv_tokens=s_conv, v_heads=v, k_heads=k, head_k_dim=hk,
                       head_v_dim=hv, conv_width=cw, ssm_itemsize=4, conv_itemsize=4)
    d = st.GdnGeometry(ssm_tokens=d_ssm, conv_tokens=d_conv, v_heads=vd, k_heads=kd, head_k_dim=hk,
                       head_v_dim=hv, conv_width=cw, ssm_itemsize=4, conv_itemsize=4)
    return s, d


def plan(st, g, s, d, n_tokens, seed):
    """Row indices of one landing (the shapes StagedConsumer._routed_indices produces), with distinct random
    staging and pool pages, each layer's rows in contiguous runs (as served pages are)."""
    rng = np.random.default_rng(seed)
    sr, db, L = g["stage_rows"], g["dst_block"], g["fa_layers"]
    main_n = -(-n_tokens // g["dcp"])
    comp_n = -(-(-(-n_tokens // g["ratio"])) // 2)
    c_rows = d.ssm_tokens + s.conv_tokens
    gdn_pages = -(-c_rows // sr)
    need_src = L * (-(-main_n // sr) + -(-comp_n // sr) + 1) + 1 + g["gdn_layers"] * gdn_pages
    need_dst = L * (-(-main_n // db) + -(-comp_n // db) + 1) + 1 + g["gdn_layers"]
    src = list(rng.permutation(np.arange(1, need_src + 1)))
    dst = list(rng.permutation(np.arange(1, need_dst + 1)))

    def run_rows(pages_list, n, block):
        pages = [int(pages_list.pop()) for _ in range(-(-n // block))]
        t = np.arange(n, dtype=np.int64)
        return np.asarray(pages, np.int64)[t // block] * block + t % block

    main_s, main_d, copy_s, copy_d = [], [], [], []
    for _ in range(L):
        main_s.append(run_rows(src, main_n, sr))
        main_d.append(run_rows(dst, main_n, db))
    for _ in range(L):
        copy_s.append(run_rows(src, comp_n, sr))
        copy_d.append(run_rows(dst, comp_n, db))
        copy_s.append(run_rows(src, g["ring_rows"], sr))
        copy_d.append(run_rows(dst, g["ring_rows"], db))
    copy_s.append(run_rows(src, g["ring_rows"], sr))  # PLE
    copy_d.append(run_rows(dst, g["ring_rows"], db))
    gdn_s = np.asarray([[int(src.pop()) for _ in range(gdn_pages)] for _ in range(g["gdn_layers"])], np.int64)
    gdn_d = np.asarray([int(dst.pop()) for _ in range(g["gdn_layers"])], np.int64)

    def cat(parts):
        return np.concatenate([st.align_rows(p) for p in parts if len(p)])

    idx = (cat(main_s), cat(main_d), cat(copy_s), cat(copy_d), gdn_s, gdn_d)
    return idx, need_src + 1, need_dst + 1, gdn_pages


# --- the independent reference (numpy) -------------------------------------------------------------


def typed_f32(rows):
    return np.ascontiguousarray(np.transpose(rows, (0, 2, 1))).view(np.uint32).reshape(-1)


def untyped_f32(words, nrows):
    w = np.zeros(nrows * LANES, np.uint32)
    w[: len(words)] = words
    return np.ascontiguousarray(np.transpose(w.view(np.uint8).reshape(nrows, LANES, 4), (0, 2, 1)))


def reference_check(stage_rows, pool_rows, idx, rank, s, d, channels, g):
    """Mismatched destination rows of a landing against the numpy reference (0 to pass)."""
    main_s, main_d, copy_s, copy_d, gdn_s, gdn_d = idx
    h = rank.kv_head
    bad = 0
    want = np.zeros((len(main_s), ROWS, LANES), np.uint8)
    want[:, 0:2] = stage_rows[main_s][:, 2 * h : 2 * h + 2]
    bad += int((pool_rows[main_d] != want).reshape(len(main_d), -1).any(axis=1).sum())
    bad += int((pool_rows[copy_d] != stage_rows[copy_s]).reshape(len(copy_d), -1).any(axis=1).sum())
    sr, db = g["stage_rows"], g["dst_block"]
    c_rows = d.ssm_tokens + s.conv_tokens
    for pages, dp in zip(gdn_s.tolist(), gdn_d.tolist()):
        region = np.concatenate([stage_rows[p * sr : (p + 1) * sr] for p in pages])[:c_rows]
        ssm = region[: d.ssm_tokens]
        conv = typed_f32(region[d.ssm_tokens :])[: s.conv_width * s.conv_dim].reshape(s.conv_width, s.conv_dim)
        conv = untyped_f32(conv[:, np.asarray(channels)].reshape(-1), d.conv_tokens)
        exp = np.concatenate([ssm, conv])
        got = pool_rows[dp * db : dp * db + len(exp)]
        bad += int((got != exp).reshape(len(exp), -1).any(axis=1).sum())
    return bad


# --- runs ----------------------------------------------------------------------------------------------


def programs_for(st, idx, arm, g):
    if arm == "v1":
        return list(st.repack_chunks(*idx))
    if arm == "v2":
        return st.repack_programs_v2(*idx, g["gdn_v2"])
    if arm == "mutant":  # v2 without its last GDN owner: G1 must fail
        return st.repack_programs_v2(*idx[:4], idx[4][:-1], idx[5][:-1], g["gdn_v2"])
    raise ValueError(arm)


def run_cpu(st, g, sizes, seed):
    import jax.numpy as jnp

    s, d = gdn_geoms(st, g)
    rank = st.RankSlice(tp_size=g["tp"], tp_rank=g["rank"], num_kv_heads=2, main_dcp=g["dcp"])
    channels = st.conv_channels(s, d, rank)

    def write(pool, rows, page, logical, valid):
        page = jnp.where(valid, page, pool.shape[0])
        return pool.at[page, logical % pool.shape[1]].set(rows, mode="drop")

    gp = -(-(d.ssm_tokens + s.conv_tokens) // g["stage_rows"])  # staging pages per GDN owner (served: 2)
    fn = st.make_repack(src_block=g["stage_rows"], dst_block=g["dst_block"], src_gdn=s, dst_gdn=d,
                        kv_head=rank.kv_head, v_head_start=rank.tp_rank * d.v_heads, channels=channels,
                        write=write, compact_gdn=(g["stage_rows"], gp))
    out = []
    for n in sizes:
        idx, n_src, n_dst, gp_n = plan(st, g, s, d, n, seed + n)
        assert gp_n == gp
        rng = np.random.default_rng(seed + 2 * n)
        staging = rng.integers(0, 255, (n_src, g["stage_rows"], 1, ROWS, LANES)).astype(np.uint8)
        staging[0] = 0
        pool0 = rng.integers(0, 255, (n_dst, g["dst_block"], 1, ROWS, LANES)).astype(np.uint8)
        pool0[0] = 0
        pools = {}
        for arm in ("v1", "v2", "mutant"):
            p = jnp.asarray(pool0)
            for prog in programs_for(st, idx, arm, g):
                p = fn(p, jnp.asarray(staging), *(jnp.asarray(np.asarray(v), jnp.int32) for v in prog))
            pools[arm] = np.asarray(p)
        flat = lambda a: a.reshape(-1, ROWS, LANES)
        out.append(dict(
            tokens=n, programs={a: len(programs_for(st, idx, a, g)) for a in ("v1", "v2")},
            g1_identity=bool(np.array_equal(pools["v1"], pools["v2"])),
            g2_reference_mismatches=reference_check(flat(staging), flat(pools["v1"]), idx, rank, s, d, channels, g),
            g3_mutant_differs=not bool(np.array_equal(pools["v1"], pools["mutant"])),
        ))
    return out


def run_tpu(st, g, sizes, reps, seed):
    import torch
    from vllm.plugins import load_general_plugins

    # As XFER-3's TPU read bench (q38_xfer3_readbench.py, run on this image): the plugin registers the device.
    load_general_plugins()
    dev = torch.device("tpu")

    s, d = gdn_geoms(st, g)
    rank = st.RankSlice(tp_size=g["tp"], tp_rank=g["rank"], num_kv_heads=2, main_dcp=g["dcp"])
    channels = st.conv_channels(s, d, rank)
    fn = st.make_repack(src_block=g["stage_rows"], dst_block=g["dst_block"], src_gdn=s, dst_gdn=d,
                        kv_head=rank.kv_head, v_head_start=rank.tp_rank * d.v_heads, channels=channels,
                        compact_gdn=(g["stage_rows"], -(-(d.ssm_tokens + s.conv_tokens) // g["stage_rows"])))
    gp = -(-(d.ssm_tokens + s.conv_tokens) // g["stage_rows"])  # staging pages per GDN owner (served: 2)
    route = SimpleNamespace(stage_rows=g["stage_rows"], gdn_rows=gp * g["stage_rows"])

    shells = {}
    for a in ("v1", "v2", "mutant"):
        c = object.__new__(st.StagedConsumer)
        c._fn, c._op = fn, None
        c._gdn_names = ["gdn%d" % i for i in range(g["gdn_layers"])]
        c.route = route
        c._v2_gdn_pages = 0 if a == "v1" else g["gdn_v2"]
        shells[a] = c

    def ready(t):  # dispatch-to-ready: a data dependency on the pool
        return t.reshape(-1)[0:1].cpu()

    plans = {n: plan(st, g, s, d, n, seed + n) for n in sizes}
    assert all(p[3] == gp for p in plans.values())
    n_src = max(p[1] for p in plans.values())
    n_dst = max(p[2] for p in plans.values())
    f8 = torch.float8_e4m3fn
    results, warm_s = [], None
    for n in sizes:
        idx = plans[n][0]
        rng = np.random.default_rng(seed + 2 * n)
        staging_np = rng.integers(0, 255, (n_src, g["stage_rows"], 1, ROWS, LANES)).astype(np.uint8)
        staging_np[0] = 0
        pool_np = rng.integers(0, 255, (n_dst, g["dst_block"], 1, ROWS, LANES)).astype(np.uint8)
        pool_np[0] = 0
        staging = torch.from_numpy(staging_np).to(dev).view(f8)
        pools = {a: torch.from_numpy(pool_np).to(dev).view(f8) for a in shells}
        for a, c in shells.items():
            c.pool, c.staging = pools[a], staging
        if warm_s is None:
            warm_s = shells["v2"].warm_repack_v2()
            # one aliased v2 op per process (its op name is registered once), shared with the mutant
            shells["mutant"]._jop_v2 = shells["v2"]._jop_v2
            shells["mutant"]._ops_v2 = shells["v2"]._ops_v2
        progs = {a: programs_for(st, idx, a, g) for a in shells}
        # identity landing (each arm once, from the same pool)
        for a in shells:
            shells[a]._run_chunks(progs[a])
            ready(pools[a])
        host = {a: pools[a].cpu().contiguous().view(torch.uint8).numpy() for a in shells}
        flat = lambda x: x.reshape(-1, ROWS, LANES)
        r = dict(tokens=n, rows=dict(main=int(len(idx[0])), copy=int(len(idx[2])), gdn=int(len(idx[4]))),
                 programs={a: len(progs[a]) for a in ("v1", "v2")},
                 g1_identity=bool(np.array_equal(host["v1"], host["v2"])),
                 g2_reference_mismatches=reference_check(flat(staging_np), flat(host["v1"]), idx, rank, s, d,
                                                         channels, g),
                 g3_mutant_differs=not bool(np.array_equal(host["v1"], host["mutant"])))
        # timing: ABAB, repeated landings of the same load (idempotent writes)
        t = {"v1": ([], []), "v2": ([], [])}
        for _ in range(reps):
            for a in ("v1", "v2"):
                ready(pools[a])
                t0 = time.perf_counter()
                shells[a]._run_chunks(progs[a])
                t1 = time.perf_counter()
                ready(pools[a])
                t2 = time.perf_counter()
                t[a][0].append((t1 - t0) * 1e3)
                t[a][1].append((t2 - t0) * 1e3)
        for a in ("v1", "v2"):
            r[a + "_host_ms"] = round(statistics.median(t[a][0]), 3)
            r[a + "_ready_ms"] = round(statistics.median(t[a][1]), 3)
        results.append(r)
        print("Q38_XFER4_REPACKV2_SIZE " + json.dumps(r, sort_keys=True), flush=True)
        for c in shells.values():
            c.pool = c.staging = None
        del staging, pools, host
    return results, warm_s


def verdict(results):
    ok = all(r["g1_identity"] and r["g2_reference_mismatches"] == 0 and r["g3_mutant_differs"] for r in results)
    return "PASS" if ok and results else "FAIL"


def script_summary(out):
    """The one case the bench recipe's scorer needs: the verdict and the per-size gates and costs (no 'new' key,
    which the scorer would read as a concurrency)."""
    keep = ("tokens", "g1_identity", "g2_reference_mismatches", "g3_mutant_differs", "programs",
            "v1_host_ms", "v2_host_ms", "v1_ready_ms", "v2_ready_ms")
    return dict(bench="q38-xfer4-repackv2", verdict=out["verdict"], wall_s=out["wall_s"], warm_s=out["warm_s"],
                results=[{k: r.get(k) for k in keep} for r in out["results"]])


def selftest():
    good = [dict(g1_identity=True, g2_reference_mismatches=0, g3_mutant_differs=True)]
    assert verdict(good) == "PASS"
    assert verdict([dict(good[0], g1_identity=False)]) == "FAIL"
    assert verdict([dict(good[0], g2_reference_mismatches=3)]) == "FAIL"
    assert verdict([dict(good[0], g3_mutant_differs=False)]) == "FAIL"  # a mutant that passes: broken check
    assert verdict([]) == "FAIL"
    print("SELFTEST PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="4800,20000,81000,160000")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=38)
    ap.add_argument("--cpu-smoke", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--gdn-v2", type=int, default=0, help="GDN owners per v2 program (default: all owners)")
    ap.add_argument("--stage-rows", type=int, default=0,
                    help="staging page rows (default: the geometry's; the served routed decode uses 256, so each\n"
                         "GDN owner of 384 SSM + 128 conv rows spans 2 staging pages)")
    ap.add_argument("--ring-rows", type=int, default=0, help="ring / PLE rows per owner (default: the geometry's)")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    # BENCH=script contract (SHARD-TP's writer, vendored next to this file): the bench recipe scores the job from
    # $CDK_OUTPUT_DIR/q38f-bench/script/run01_A/result.json ({"ok", "cases": [...]}); without it a run that exits 0
    # still scores Failed (i2 j-1e435b93: verdict PASS, '==> exit 0', runs_ok 0). Clear it first, so a crash cannot
    # leave a stale one; write it last, ok = verdict PASS (FAIL or a crash fail the job, as the prereg reads them).
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import q38_script_result
    if os.environ.get("CDK_OUTPUT_DIR"):
        q38_script_result.clear()
    sys.path.insert(0, os.path.join(os.getcwd(), "src"))
    from vllm_torchtpu.models.vllm.qwen4_exp import pd_staging as st

    g = geometry(a.cpu_smoke)
    if a.gdn_v2:
        g["gdn_v2"] = a.gdn_v2
    if a.stage_rows:
        g["stage_rows"] = a.stage_rows
    if a.ring_rows:
        g["ring_rows"] = a.ring_rows
    sizes = [int(x) for x in a.sizes.split(",")]
    t0 = time.time()
    if a.cpu_smoke:
        results, warm_s = run_cpu(st, g, sizes, a.seed), None
    else:
        results, warm_s = run_tpu(st, g, sizes, a.reps, a.seed)
    out = dict(verdict=verdict(results), warm_s=warm_s, geometry=g, sizes=sizes, reps=a.reps,
               results=results, wall_s=round(time.time() - t0, 1))
    print("Q38_XFER4_REPACKV2 " + json.dumps(out, sort_keys=True), flush=True)
    dest = os.environ.get("CDK_OUTPUT_DIR")
    if dest and not a.cpu_smoke:
        d = os.path.join(dest, "q38f-bench", "script")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "q38_xfer4_repack_v2.json"), "w") as fh:
            json.dump(out, fh, indent=1, sort_keys=True)
    if dest:
        q38_script_result.write(out["verdict"] == "PASS", script_summary(out))


if __name__ == "__main__":
    main()
