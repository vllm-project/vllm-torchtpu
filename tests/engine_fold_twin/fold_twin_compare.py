# SPDX-License-Identifier: Apache-2.0
"""Compare the fold-on and fold-off arms of fold_twin.py.

  python3 fold_twin_compare.py <work_dir>

Per dumped request and rank: every (KV-cache group, block-table column, raw
tensor) page, on vs off, in layer order. A page is "same" when its bytes are
identical; otherwise the differing byte range and the size of the difference
read as bf16 / f32 / fp8-e4m3. The first group in layer order that differs,
with every earlier one the same, names the first diverging layer. Then the
first output token's top-5 logprobs and the greedy tokens per request.
"""

import glob
import json
import math
import os
import re
import sys

import numpy as np


def layer_key(names):
    best = (10**6, "")
    for n in names:
        m = re.search(r"layers\.(\d+)\.", n)
        if m:
            best = min(best, (int(m.group(1)), n))
    return best


def load(path):
    z = np.load(path)
    out = {}
    for k in z.files:
        if k.startswith("meta_"):
            continue
        rid, g, c, t, n = k.split("|")
        out[(int(g[1:]), int(c[1:]), int(t[1:]))] = (z[k], int(n[1:]))
    return out


def views(a, b):
    res = {}
    for name, dt in (("bf16", "bfloat16"), ("f32", np.float32), ("e4m3", "float8_e4m3fn")):
        try:
            if isinstance(dt, str):
                import ml_dtypes  # noqa: F401
                dt = getattr(__import__("ml_dtypes"), dt)
            dt = np.dtype(dt)
            n = (len(a) // dt.itemsize) * dt.itemsize
            x = a[:n].view(dt).astype(np.float64)
            y = b[:n].view(dt).astype(np.float64)
        except Exception as e:  # noqa: BLE001
            res[name] = f"n/a ({e!r})"[:40]
            continue
        fin = np.isfinite(x) & np.isfinite(y)
        d = np.abs(x - y)[fin]
        ref = np.abs(y[fin]).max() if fin.any() else 0.0
        res[name] = (f"max|d| {d.max() if d.size else 0:.3g} / max|off| {ref:.3g}, "
                     f"diff elems {int((x != y).sum())}/{x.size}, "
                     f"nonfinite on/off {int((~np.isfinite(x)).sum())}/{int((~np.isfinite(y)).sum())}")
    return res


def compare_pages(work, rid, groups, A="on", B="off"):
    for on_path in sorted(glob.glob(os.path.join(work, f"pages_{A}_{rid}_r*.npz"))):
        rank = on_path.rsplit("_r", 1)[1].split(".")[0]
        off_path = os.path.join(work, f"pages_{B}_{rid}_r{rank}.npz")
        if not os.path.exists(off_path):
            print(f"  {rid} rank {rank}: no off dump")
            continue
        on, off = load(on_path), load(off_path)
        keys = sorted(set(on) | set(off),
                      key=lambda k: (layer_key(groups[k[0]]["layers"]), k[0], k[1], k[2]))
        first = None
        same = 0
        lines = []
        for k in keys:
            g, c, t = k
            gi = groups[g]
            lk = layer_key(gi["layers"])
            label = f"g{g:02d} {gi['spec'][:18]:18s} L{lk[0]:>3} {lk[1][:38]:38s} col {c:3d} t{t}"
            if k not in on or k not in off:
                lines.append(f"    {label}: only in {'on' if k in on else 'off'}")
                continue
            (a, n), (b, _) = on[k], off[k]
            size = max(len(a), len(b))
            a = np.pad(a, (0, size - len(a)))
            b = np.pad(b, (0, size - len(b)))
            if np.array_equal(a, b):
                same += 1
                continue
            dif = np.flatnonzero(a != b)
            v = views(a, b)
            lines.append(f"    {label}: DIFF bytes {dif.size}/{size} in [{dif[0]}, {dif[-1]}] "
                         f"(page {n}, written on/off {len(on[k][0])}/{len(off[k][0])})")
            for name, s in v.items():
                lines.append(f"        {name}: {s}")
            if first is None:
                first = label
        print(f"  {rid} rank {rank}: {same} pages same, {len(keys) - same} differ; "
              f"FIRST DIVERGING: {first}")
        for line in lines[:80]:
            print(line)


def compare_tokens(on, off):
    for pname in on["phases"]:
        a, b = on["phases"][pname], off["phases"].get(pname)
        if b is None:
            continue
        print(f"[{pname}] engine steps on/off {a['steps']}/{b['steps']}")
        if a.get("gsm") or b.get("gsm"):
            ga, gb = a.get("gsm") or {}, b.get("gsm") or {}
            print(f"  GSM acc on {ga.get('ok')}/{ga.get('n')} (empty {ga.get('empty')}) vs off "
                  f"{gb.get('ok')}/{gb.get('n')} (empty {gb.get('empty')})")
            continue
        for arm, ph in (("on", a), ("off", b)):
            for x in ph["sched"]:
                print(f"    {arm} step {x['step']}: {x['sched']} copies={x['copies']}")
        for rid in a["requests"]:
            ra, rb = a["requests"][rid], b["requests"][rid]
            n = min(len(ra["tokens"]), len(rb["tokens"]))
            firstdiff = next((i for i in range(n) if ra["tokens"][i] != rb["tokens"][i]), None)
            t5a = (ra["top5"] or [None])[0]
            t5b = (rb["top5"] or [None])[0]
            d = None
            if t5a and t5b:
                da, db = dict(t5a), dict(t5b)
                common = set(da) & set(db)
                d = max((abs(da[k] - db[k]) for k in common), default=float("nan"))
                d = f"top5 overlap {len(common)}/5, max|dlp| {d:.4g}"
            nf = sum(1 for x in ra["top1"] + rb["top1"] if x is None or not math.isfinite(x))
            print(f"  {rid} P={ra['prompt_len']}: tokens on {ra['tokens'][:8]} off "
                  f"{rb['tokens'][:8]}; first diff {firstdiff}; pos0 {d}; nonfinite {nf}")
            print(f"      top5[0] on  {t5a}")
            print(f"      top5[0] off {t5b}")


def main():
    work = sys.argv[1]
    gpath = os.path.join(work, "groups_on.json")
    if not os.path.exists(gpath):
        gpath = os.path.join(work, "groups_off.json")
    groups = json.load(open(gpath))["groups"]
    for arm in ("on", "off"):
        p = os.path.join(work, f"groups_{arm}.json")
        if os.path.exists(p):
            gs = json.load(open(p))
            print(f"groups_{arm}: {len(gs['groups'])} groups, num_blocks {gs['num_blocks']}, "
                  f"raws {gs['raws']}; ckpt groups "
                  f"{sum(1 for g in gs['groups'] if g['ckpt'])}")
    A = sys.argv[2] if len(sys.argv) > 2 else "on"
    B = sys.argv[3] if len(sys.argv) > 3 else "off"
    print(f"##### {A} vs {B} #####")
    try:
        on = json.load(open(os.path.join(work, f"arm_{A}.json")))
        off = json.load(open(os.path.join(work, f"arm_{B}.json")))
        print("fingerprint equal:", on["fingerprint"] == off["fingerprint"])
        compare_tokens(on, off)
    except FileNotFoundError as e:
        print("no arm json:", e)
    rids = sorted({os.path.basename(p).split("_")[2]
                   for p in glob.glob(os.path.join(work, f"pages_{A}_*_r*.npz"))})
    print("=== PAGES ===")
    for rid in rids:
        compare_pages(work, rid, groups, A, B)


if __name__ == "__main__":
    main()
