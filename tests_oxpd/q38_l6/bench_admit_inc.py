#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""q38 L6 lever admitinc: CPU microbench of the geometry event (one finish + one admission) on a served-shaped
block-table set. Runs in a CDK CPU job only. Usage:
  python3 tests/q38_l6/bench_admit_inc.py [--n 4,9,11,25,52] [--events 120] [--max-len 262144]

Shape (I, reconstructed): 65 KV groups as in the served log's group list quoted by bench_engine_alloc.py
(37 x 512, 13 x 2048, 2 x 512, 13 x 4); a group's row width is ceil(max_len / size) for the 512 / 2048 groups
(use_max_model_len) and 4 for the 13 small groups; a request holds ceil(ctx / size) blocks (4 in a small group)
with ctx drawn so that one admission carries about 9-12K block ids (the HOSTSLOW read, lane_dechost_b.md
18:46Z: '9-12K block ids per admission'). The padded request count is the runner's bucket (8, then powers of 2).

Per N: two identical worlds, A (lever on) and B (lever off), real vLLM BlockTables under a MultiGroupBlockTable,
BT-INC on and upload reuse on (served). Each event: a finish (clear_row + move_row, InputBatch's
remove_request + condense) and an admission (add_row) through the MultiGroupBlockTable, then one staged pack at
the new count (N-1, then N again), between three decode steps (append_row with one block for the 41 groups of
size 512, every fourth also the 13 size-2048 groups, empty lists elsewhere, plus a BT-INC pack). Timed per call
with perf_counter_ns, arms alternating first. The H2D upload is excluded in both arms (same bytes: the whole
staged buffer). Identity: after every write both worlds' tables are equal, and after every pack both arms'
staged bytes equal a freshly built full pack. Last line: BENCH_DONE identity=PASS|FAIL.
"""

import argparse
import math
import os
import random
import statistics as st
import sys
import time
from types import SimpleNamespace

ap = argparse.ArgumentParser()
ap.add_argument("--n", default="4,9,11,25,52")
ap.add_argument("--events", type=int, default=120)
ap.add_argument("--max-len", type=int, default=262144)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--event-rate", type=float, default=0.028)  # full packs on 2.8 % of steps (host_levers_h1.md)
a = ap.parse_args()

for k in list(os.environ):
    if k.startswith("VLLM_TPU_L6_"):
        del os.environ[k]
os.environ["VLLM_TPU_L6_ADMIT_INC"] = "1"
os.environ["VLLM_TPU_BLOCK_TABLE_UPLOAD_REUSE"] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402
from vllm.v1.worker.block_table import BlockTable, MultiGroupBlockTable  # noqa: E402

from vllm_torchtpu.core import l6_ab  # noqa: E402
from vllm_torchtpu.layers.core import attention_metadata as am  # noqa: E402

CPU = torch.device("cpu")
am._STAGE_INC = True
am._ADMIT_CHECK[0] = 0
l6_ab.begin_step()
assert MultiGroupBlockTable.__dict__.get("_q38_admit_patch"), "batched row writes not installed"
SIZES = [512] * 37 + [2048] * 13 + [512] * 2 + [4] * 13
MAXR = 64


def arm(on):
    l6_ab._state["on"]["admitinc"] = on


def widths():
    return [4 if s == 4 else math.ceil(a.max_len / s) for s in SIZES]


def make_world():
    m = object.__new__(MultiGroupBlockTable)
    m.block_tables = [BlockTable(s, MAXR, w, 64, False, CPU, s, 1) for s, w in zip(SIZES, widths())]
    return m


def padded(n):
    return 8 if n <= 8 else 1 << (n - 1).bit_length()


class Ids:
    def __init__(self):
        self.next = 1

    def take(self, k):
        out = list(range(self.next, self.next + k))
        self.next += k
        return out


def req_blocks(rng, ids):
    ctx = rng.randint(100_000, 150_000)
    return tuple(ids.take(4 if s == 4 else math.ceil(ctx / s)) for s in SIZES)


def reference(m, ws, n, nrp):
    parts = []
    for bt, w in zip(m.block_tables, ws):
        rows = torch.zeros((nrp, w), dtype=torch.int32)
        rows[:n] = bt.get_cpu_tensor()[:n, :w]
        parts.append(rows.flatten())
    return torch.cat(parts)


def pack(runner, m, ws, n, nrp):
    entries = [(i, bt, w) for i, (bt, w) in enumerate(zip(m.block_tables, ws))]
    geom, off = [], 0
    for w in ws:
        geom.append((off, nrp, w))
        off += nrp * w
    key = (am._BLOCK_TABLE_VERSION[0], tuple(id(bt) for bt in m.block_tables), 0, n, tuple(geom))
    ctx = SimpleNamespace(position_ids_override=None, start_index=0, num_reqs=n)
    t0 = time.perf_counter_ns()
    out = am._pack_and_upload(runner, ctx, entries, nrp, geom, key)
    return out, (time.perf_counter_ns() - t0) / 1000.0


def same_tables(x, y):
    return all(np.array_equal(p.block_table.np, q.block_table.np) and np.array_equal(p.num_blocks_per_row, q.num_blocks_per_row)
               for p, q in zip(x.block_tables, y.block_tables))


def timed(fn, *args):
    t0 = time.perf_counter_ns()
    fn(*args)
    return (time.perf_counter_ns() - t0) / 1000.0


def med(x):
    return st.median(x) if x else float("nan")


def run_n(n_req, rng):
    nrp = padded(n_req)
    ws = widths()
    A, B = make_world(), make_world()
    ids_a, ids_b = Ids(), Ids()
    ra = SimpleNamespace(_block_table_stage_cpu=None, device=CPU)
    rb = SimpleNamespace(_block_table_stage_cpu=None, device=CPU)
    per_admit = []
    for row in range(n_req):
        blocks = req_blocks(rng, ids_a)
        ids_b.take(sum(len(b) for b in blocks))
        per_admit.append(sum(len(b) for b in blocks))
        arm(True)
        A.add_row(blocks, row)
        arm(False)
        B.add_row(blocks, row)
    ok = True
    arm(True)
    pa, _ = pack(ra, A, ws, n_req, nrp)
    arm(False)
    pb, _ = pack(rb, B, ws, n_req, nrp)
    ok &= torch.equal(pa, pb)
    T = {k: [] for k in ("pack_stock", "pack_admit", "fin_stock", "fin_batch", "adm_stock", "adm_batch",
                         "app_stock", "app_batch", "inc_stock", "inc_admit")}
    n = n_req
    for ev in range(a.events):
        first_a = ev % 2 == 0
        # three decode steps: append_row for a few requests, BT-INC pack (no count change)
        for d in range(3):
            rows = rng.sample(range(n), max(1, n // 8))
            for row in rows:
                k = 1 if (ev * 3 + d + row) % 4 == 0 else 0
                new = []
                for s, bt in zip(SIZES, A.block_tables):
                    room = bt.max_num_blocks_per_req - int(bt.num_blocks_per_row[row])
                    want = 1 if s == 512 else (k if s == 2048 else 0)
                    new.append(list(range(10**9 + ev * 1000 + d * 100 + row, 10**9 + ev * 1000 + d * 100 + row + want)) if room >= want else [])
                new = tuple(new)
                for on in ((True, False) if first_a else (False, True)):
                    arm(on)
                    T["app_batch" if on else "app_stock"].append(timed((A if on else B).append_row, new, row))
            for on in ((True, False) if first_a else (False, True)):
                arm(on)
                out, us = pack(ra if on else rb, A if on else B, ws, n, nrp)
                T["inc_admit" if on else "inc_stock"].append(us)
                ok &= torch.equal(out, reference(A if on else B, ws, n, nrp))
        # finish a random request (remove_request + condense), pack at n - 1
        row = rng.randrange(n)
        last = n - 1
        for on in ((True, False) if first_a else (False, True)):
            arm(on)
            m = A if on else B
            t = timed(m.clear_row, row)
            if row != last:
                t += timed(m.move_row, last, row)
            T["fin_batch" if on else "fin_stock"].append(t)
        n -= 1
        for on in ((True, False) if first_a else (False, True)):
            arm(on)
            out, us = pack(ra if on else rb, A if on else B, ws, n, nrp)
            T["pack_admit" if on else "pack_stock"].append(us)
            ok &= torch.equal(out, reference(A if on else B, ws, n, nrp))
        # admit a new request at row n, pack at n + 1
        blocks = req_blocks(rng, ids_a)
        ids_b.take(sum(len(b) for b in blocks))
        per_admit.append(sum(len(b) for b in blocks))
        for on in ((True, False) if first_a else (False, True)):
            arm(on)
            T["adm_batch" if on else "adm_stock"].append(timed((A if on else B).add_row, blocks, n))
        n += 1
        for on in ((True, False) if first_a else (False, True)):
            arm(on)
            out, us = pack(ra if on else rb, A if on else B, ws, n, nrp)
            T["pack_admit" if on else "pack_stock"].append(us)
            ok &= torch.equal(out, reference(A if on else B, ws, n, nrp))
        ok &= same_tables(A, B)
    stock_ev = med(T["pack_stock"]) * 2 + med(T["fin_stock"]) + med(T["adm_stock"])
    admit_ev = med(T["pack_admit"]) * 2 + med(T["fin_batch"]) + med(T["adm_batch"])
    mean = st.mean
    stock_ev_m = mean(T["pack_stock"]) * 2 + mean(T["fin_stock"]) + mean(T["adm_stock"])
    admit_ev_m = mean(T["pack_admit"]) * 2 + mean(T["fin_batch"]) + mean(T["adm_batch"])
    d_app = med(T["app_stock"]) - med(T["app_batch"])
    print(
        f"BENCH n={n_req} nrp={nrp} groups={len(SIZES)} ids_per_admit={st.mean(per_admit):.0f} "
        f"scratch_ints={nrp * sum(ws)} events={a.events} "
        f"pack_stock_us={med(T['pack_stock']):.1f} (mean {st.mean(T['pack_stock']):.1f}) "
        f"pack_admit_us={med(T['pack_admit']):.1f} (mean {st.mean(T['pack_admit']):.1f}) "
        f"finish_writes_stock_us={med(T['fin_stock']):.1f} finish_writes_batched_us={med(T['fin_batch']):.1f} "
        f"admit_add_row_stock_us={med(T['adm_stock']):.1f} admit_add_row_batched_us={med(T['adm_batch']):.1f} "
        f"append_row_stock_us={med(T['app_stock']):.2f} append_row_batched_us={med(T['app_batch']):.2f} "
        f"steady_inc_pack_stock_us={med(T['inc_stock']):.1f} steady_inc_pack_lever_us={med(T['inc_admit']):.1f} "
        f"event_stock_us={stock_ev:.1f} event_lever_us={admit_ev:.1f} event_delta_us={stock_ev - admit_ev:.1f} "
        f"per_step_delta_us_at_{a.event_rate:g}={(stock_ev - admit_ev) / 2 * a.event_rate:.1f} "
        f"event_stock_mean_us={stock_ev_m:.1f} event_lever_mean_us={admit_ev_m:.1f} "
        f"per_change_delta_mean_us={(stock_ev_m - admit_ev_m) / 2:.1f} "
        f"per_step_delta_mean_us_at_{a.event_rate:g}={(stock_ev_m - admit_ev_m) / 2 * a.event_rate:.1f} "
        f"append_delta_us_per_call={d_app:.2f} identity={'PASS' if ok else 'FAIL'}",
        flush=True,
    )
    return ok


def main():
    print(f"BENCH_ENV nproc {os.cpu_count()} torch {torch.__version__} numpy {np.__version__} "
          f"max_len {a.max_len} widths {sorted(set(widths()))}", flush=True)
    rng = random.Random(a.seed)
    ok = True
    for n in [int(x) for x in a.n.split(",")]:
        ok &= run_n(n, rng)
    st_ = am._ADMIT_STATS
    print(f"BENCH_ENGAGEMENT admit_packs={st_['inc']} batched_writes={st_['batch']} "
          f"btinc={am._STAGE_INC_STATS}", flush=True)
    ok &= st_["inc"] > 0 and st_["batch"] > 0
    print(f"BENCH_DONE identity={'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
