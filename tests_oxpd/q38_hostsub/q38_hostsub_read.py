#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""q38 L4 HOSTSUBS census reader (stdlib only; run with python3 -I).

Reads pod-log text (one arm's segment, or a whole log with --arm-start / the
J0-style `==> Q38_ARM_START runNN_<label>` markers) and prints, per arm, the
sub-phase census from the HOSTSUBS lines that runner/host_phases.py writes with
VLLM_TPU_HOST_PHASES_SUBS=1:
  HOSTSUBS rank=R steps=S per-step sub-phase host us (p50/p90/mean-per-step/n-steps) ~name=a/b/c/n ...
and, for context, the parent phases from the HOSTPHASES lines of the same ranks:
  HOSTPHASES rank=R steps=S ... total_mean=T name=p50/p90/mean ...

Statistic: for each (rank, sub-phase) the median over windows of mean-per-step
and of p50; then rank 0 and the rank maximum of those medians. Windows whose
first --skip-windows per rank (warm-up) are dropped. Sub-phases split a phase:
each is printed under the phase that contains it (static map below; unknown
names go to 'other'), with the phase's own rank-0 median mean and the
remainder (phase mean - sum of its sub means, I).

GC census (q38 L6-INSTR): GCCENSUS lines (VLLM_TPU_HOST_PHASES > 0)
  GCCENSUS rank=R steps=S window=N ... gc_us=<sum>/<collections>/<gen2> max_us=<us> g0=.. g1=.. g2=..
give, per rank, the median over windows (same warm-up skip) of the GC pause per
step (sum / N), collections per step and gen-2 collections per window.

P-METER (q38 L6, runner/l6_pmeter.py): PMETER lines (one per rank per window)
  PMETER rank=R host=H pid=P window=W seq0=S n=K t0_ns=T e=d,f,a,b,l;...;x;...
give per step (seq = S + entry index) the dequeue time (t0 + d us) and the forward enqueue (f), the previous-step copy wait
start / return (a / b) and the dp_loop enqueue (l) as us after the dequeue ('-' not stamped, 'x' meter off). Ranks are
aligned by seq; a step is read only if every rank of the arm has it (else it is dropped and counted). Premium per aligned
step = rank max - rank median of the step-relative duration (dequeue -> stamp; immune to clock offsets between hosts) and,
when every rank logged the same host, of the absolute stamp. The copy wait (b - a) gives the rank-mean wait, which in a
host-bound regime equals E[max] - E[mean] of the host cycle (H2 hunt_3 section 1.4 (d2)). THREADSTAT lines give per-thread
window deltas of on-CPU time, run-queue wait and nonvoluntary context switches.

Output lines (stable, grep-able):
  HOSTSUB arm=<a> ranks=<n> windows=<w> sub=<name> phase=<p> r0_mean=<us> r0_p50=<us> max_mean=<us> n_steps_med=<k>
  HOSTSUBPH arm=<a> phase=<p> r0_phase_mean=<us> subs_sum=<us> remainder=<us>
  HOSTGC arm=<a> ranks=<n> windows=<w> r0_us_per_step=<us> max_us_per_step=<us> r0_coll_per_step=<x> r0_gen2_per_window=<k> max_pause_us=<us>
  HOSTSUBSUM arm=<a> ranks=<n> windows=<w> subs=<k> verdict=OK|NO_DATA
  HOSTPMETER arm=<a> ranks=<n> hosts=<h> steps=<aligned> dropped=<d> stamp=<fwd|loop> rel_prem_p50= rel_prem_mean= abs_prem_p50= abs_prem_mean= laggard=<rank>:<share>
  HOSTPMWAIT arm=<a> steps=<k> wait_rankmean_mean= wait_rankmean_p50= wait_rankmin_mean= wait_r0_mean=
  HOSTPMSPIKE arm=<a> stamp=fwd thr_us=500 steps=<k> share=<x> mean_us=<x> top=<rank>:<count>
  HOSTTHREAD arm=<a> rank=<r> thread=<name> windows=<k> run_us_med= wait_us_med= nvcsw_med=
Exit 0 iff every requested arm has data.
"""

import argparse
import re
import statistics as st
import sys

SUB_RX = re.compile(r"HOSTSUBS rank=(-?\d+) steps=(\d+) per-step sub-phase host us \(p50/p90/mean-per-step/n-steps\)(.*)$")
PH_RX = re.compile(r"HOSTPHASES rank=(-?\d+) steps=(\d+) per-step host us \(p50/p90/mean-per-step\) total_mean=\d+(.*)$")
ARM_RX = re.compile(r"^==> Q38_ARM_START run\d+_(\S+)")
ARM_END_RX = re.compile(r"^==> Q38_ARM_END run\d+_(\S+)")
PM_RX = re.compile(r"PMETER rank=(-?\d+) host=(\S+) pid=\d+ window=(\d+) seq0=(\d+) n=\d+ t0_ns=(\S+) e=(\S*)")
TS_RX = re.compile(r"THREADSTAT rank=(-?\d+) seq=(\d+) n=\d+ ?(.*)$")
GC_RX = re.compile(r"GCCENSUS rank=(-?\d+) steps=(\d+) window=(\d+) .*gc_us=(\d+)/(\d+)/(\d+) max_us=(\d+)")
# phase_of() returns the FIRST entry whose key prefixes the name, so specific
# names come before their prefix (pi_seed_sync etc. sit in prepare_inputs, the
# other pi_ sub-marks in prep_host_and_uploads).
PHASE_OF = {
    "pi_seed_sync": "prepare_inputs", "pi_subst2": "prepare_inputs", "pi_collect": "prepare_inputs",
    "pi_cross": "prepare_inputs",
    "pi_": "prep_host_and_uploads", "am_": "prep_attention_metadata", "us_": "update_states",
    "kc_": "kv_connector", "pf_": "post_forward", "st_": "sample_tail", "fd_": "forward_dispatch",
    "add_request": "update_states",
    # q38 L6-INSTR (H1 rung 1): PR-0 pi_post and adapter sub-marks, the DH-E
    # draft sub-marks (rows supplied by L4-D) and the T-5 sample_setup census.
    "pp_": "prep_host_and_uploads", "ap_": "adapter_prepare_inputs",
    "dpi_": "dp_prepare_inputs", "df0_": "dp_forward0", "dc_": "dp_carries",
    "ss_": "sample_setup",
    # q38 L4-E DRAFT1B-BUILD log-only sub-marks (rows requested 11-14): ~dl_fold (the DH-B stage-A carry-loop
    # program, after a ~dc_rest boundary) and ~df_fold (the DH-D single draft program, after a ~df0_rest boundary);
    # df_ sits after df0_ because the first matching row wins. ~dc_fold is covered by dc_ above.
    "dl_": "dp_loop", "df_": "dp_forward0",
    # q38 L6 P-METER: the copy wait inside _modify_prev_results splits dp_entry (where it
    # runs with a KV connector); the admission census splits am_stage_bt into ~am_pack /
    # ~am_h2d / ~am_walk (already under am_).
    "mpr_": "dp_entry",
    # q38 L6 NODEFER census: the VFY call's split of vfy_rejection.
    "vr_": "vfy_rejection",
}


def phase_of(name):
    for k, v in PHASE_OF.items():
        if name.startswith(k):
            return v
    return "other"


def segments(lines, default="all"):
    """Yield (arm, line); lines outside any ARM_START/END pair go to `default`
    when no markers exist at all."""
    marked = any(ARM_RX.search(ln) for ln in lines)
    arm = None if marked else default
    for ln in lines:
        m = ARM_RX.search(ln)
        if m:
            arm = m.group(1)
            continue
        if ARM_END_RX.search(ln):
            arm = None if marked else default
            continue
        if arm is not None:
            yield arm, ln


def parse(lines, skip):
    data = {}  # arm -> {"subs": {(rank, name): [(mean, p50, n)]}, "ph": {(rank, name): [mean]}, "win": {rank: k}}
    for arm, ln in segments(lines):
        d = data.setdefault(arm, {"subs": {}, "ph": {}, "win": {}, "pwin": {}, "gc": {}, "gwin": {},
                                  "pm": {}, "pmhost": {}, "pmlines": {}, "ts": {}})
        if "PMETER rank=" in ln:
            m = PM_RX.search(ln)
            if m:
                r, w, seq0 = int(m.group(1)), int(m.group(3)), int(m.group(4))
                d["pmhost"][r] = m.group(2)
                d["pmlines"][r] = d["pmlines"].get(r, 0) + 1
                t0 = int(m.group(5)) // 1000 if m.group(5) != "-" else 0
                base = seq0  # the window's own first seq, never a per-rank running count
                for i, e in enumerate(m.group(6).split(";")):
                    seq = base + i
                    if e == "x" or not e:
                        continue
                    v = [None if x == "-" else int(x) for x in e.split(",")]
                    d["pm"].setdefault(r, {})[seq] = (t0 + v[0], v[1], v[2], v[3], v[4])
            continue
        if "THREADSTAT rank=" in ln:
            m = TS_RX.search(ln)
            if m:
                r = int(m.group(1))
                for kv in m.group(3).split():
                    name, _, vals = kv.partition("=")
                    if vals == "-" or "/" not in vals:
                        continue
                    run, wait, _sl, _vol, nvol = vals.split("/")
                    d["ts"].setdefault((r, name), []).append((int(run), int(wait), int(nvol)))
            continue
        m = GC_RX.search(ln)
        if m:
            r = int(m.group(1))
            d["gwin"][r] = d["gwin"].get(r, 0) + 1
            if d["gwin"][r] > skip:
                n = max(int(m.group(3)), 1)
                d["gc"].setdefault(r, []).append(
                    (int(m.group(4)) / n, int(m.group(5)) / n, int(m.group(6)), int(m.group(7))))
            continue
        m = SUB_RX.search(ln)
        if m:
            r = int(m.group(1))
            d["win"][r] = d["win"].get(r, 0) + 1
            if d["win"][r] <= skip:
                continue
            for kv in m.group(3).split():
                if not kv.startswith("~") or "=" not in kv:
                    continue
                k, v = kv[1:].split("=", 1)
                p50, _p90, mean, n = v.split("/")
                d["subs"].setdefault((r, k), []).append((float(mean), float(p50), int(n)))
            continue
        m = PH_RX.search(ln)
        if m:
            r = int(m.group(1))
            d["pwin"][r] = d["pwin"].get(r, 0) + 1
            if d["pwin"][r] <= skip:
                continue
            for kv in m.group(3).split():
                if "=" not in kv:
                    continue
                k, v = kv.split("=", 1)
                d["ph"].setdefault((r, k), []).append(float(v.split("/")[2]))
    return data


def _q50(xs):
    return st.median(xs) if xs else None


def _fm(x):
    return "NA" if x is None else f"{x:.1f}"


def pmeter_report(arm, d):
    """P-METER premium: per aligned step, rank max - rank median (see the module docstring)."""
    pm = d["pm"]
    if not pm:
        return []
    ranks = sorted(pm)
    one_host = len(set(d["pmhost"].values())) == 1
    seqs = sorted(set().union(*[set(v) for v in pm.values()]))
    # a step is read only if every rank of the arm logged it (a missing line or an 'x' drops it)
    aligned = [s for s in seqs if sum(s in pm[r] for r in ranks) == len(ranks)]
    out = []
    for stamp, k in (("fwd", 1), ("loop", 4)):
        rel_p, abs_p, lag = [], [], {}
        spikes = []
        for s in aligned:
            here = [r for r in ranks if s in pm[r]]
            vals = {r: pm[r][s][k] for r in here}
            if any(v is None for v in vals.values()):
                continue
            med = st.median(vals.values())
            top = max(here, key=lambda r: (vals[r], -r))
            prem = vals[top] - med
            rel_p.append(prem)
            lag[top] = lag.get(top, 0) + 1
            if stamp == "fwd" and prem > 500:
                spikes.append((prem, top))
            if one_host:
                ab = {r: pm[r][s][0] + vals[r] for r in here}
                abs_p.append(max(ab.values()) - st.median(ab.values()))
        lr = max(lag, key=lambda r: (lag[r], -r)) if lag else None
        out.append(f"HOSTPMETER arm={arm} ranks={len(ranks)} hosts={len(set(d['pmhost'].values()))} steps={len(aligned)} "
                   f"dropped={len(seqs) - len(aligned)} stamp={stamp} rel_prem_p50={_fm(_q50(rel_p))} "
                   f"rel_prem_mean={_fm(st.mean(rel_p) if rel_p else None)} abs_prem_p50={_fm(_q50(abs_p))} "
                   f"abs_prem_mean={_fm(st.mean(abs_p) if abs_p else None)} "
                   f"laggard={lr if lr is not None else '-'}:{(lag.get(lr, 0) / len(rel_p)) if rel_p else 0:.2f}")
        if stamp == "fwd":
            tops = {}
            for _, t in spikes:
                tops[t] = tops.get(t, 0) + 1
            tt = max(tops, key=lambda r: (tops[r], -r)) if tops else None
            out.append(f"HOSTPMSPIKE arm={arm} stamp=fwd thr_us=500 steps={len(spikes)} "
                       f"share={(len(spikes) / len(rel_p)) if rel_p else 0:.3f} "
                       f"mean_us={_fm(st.mean(p for p, _ in spikes) if spikes else None)} "
                       f"top={tt if tt is not None else '-'}:{tops.get(tt, 0)}")
    wm, wmin, w0 = [], [], []
    for s in aligned:
        w = {r: (pm[r][s][3] - pm[r][s][2]) for r in ranks
             if s in pm[r] and pm[r][s][2] is not None and pm[r][s][3] is not None}
        if len(w) != len(ranks):
            continue
        wm.append(st.mean(w.values()))
        wmin.append(min(w.values()))
        w0.append(w[ranks[0]])
    out.append(f"HOSTPMWAIT arm={arm} steps={len(wm)} wait_rankmean_mean={_fm(st.mean(wm) if wm else None)} "
               f"wait_rankmean_p50={_fm(_q50(wm))} wait_rankmin_mean={_fm(st.mean(wmin) if wmin else None)} "
               f"wait_r0_mean={_fm(st.mean(w0) if w0 else None)}")
    for (r, name), rows in sorted(d["ts"].items()):
        dl = [(b[0] - a[0], b[1] - a[1], b[2] - a[2]) for a, b in zip(rows, rows[1:])]
        if not dl:
            continue
        out.append(f"HOSTTHREAD arm={arm} rank={r} thread={name.rsplit(':', 1)[0]} windows={len(dl)} "
                   f"run_us_med={st.median(x[0] for x in dl) / 1000:.0f} wait_us_med={st.median(x[1] for x in dl) / 1000:.0f} "
                   f"nvcsw_med={st.median(x[2] for x in dl):.0f}")
    return out


def report(data, arms):
    out, ok = [], True
    for arm in arms:
        d = data.get(arm)
        if d:
            out.extend(pmeter_report(arm, d))
        if not d or not d["subs"]:
            out.append(f"HOSTSUBSUM arm={arm} ranks=0 windows=0 subs=0 verdict=NO_DATA")
            ok = False
            continue
        ranks = sorted({r for r, _ in d["subs"]})
        names = sorted({k for _, k in d["subs"]}, key=lambda k: (phase_of(k), k))
        windows = max(len(v) for v in d["subs"].values())
        r0 = 0 if 0 in ranks else ranks[0]
        by_phase = {}
        for k in names:
            meds = {r: st.median(x[0] for x in d["subs"][(r, k)]) for r in ranks if (r, k) in d["subs"]}
            p50s = {r: st.median(x[1] for x in d["subs"][(r, k)]) for r in ranks if (r, k) in d["subs"]}
            nst = st.median(x[2] for x in d["subs"].get((r0, k), [(0, 0, 0)]))
            r0m = meds.get(r0, float("nan"))
            ph = phase_of(k)
            by_phase[ph] = by_phase.get(ph, 0.0) + (0.0 if r0m != r0m else r0m)
            out.append(f"HOSTSUB arm={arm} ranks={len(ranks)} windows={windows} sub={k} phase={ph} "
                       f"r0_mean={r0m:.0f} r0_p50={p50s.get(r0, float('nan')):.0f} "
                       f"max_mean={max(meds.values()):.0f} n_steps_med={nst:.0f}")
        for ph, s in sorted(by_phase.items()):
            pm = d["ph"].get((r0, ph))
            if pm:
                pmm = st.median(pm)
                out.append(f"HOSTSUBPH arm={arm} phase={ph} r0_phase_mean={pmm:.0f} subs_sum={s:.0f} remainder={pmm - s:.0f}")
            else:
                out.append(f"HOSTSUBPH arm={arm} phase={ph} r0_phase_mean=n/a subs_sum={s:.0f} remainder=n/a")
        if d["gc"]:
            g0 = 0 if 0 in d["gc"] else min(d["gc"])
            per = {r: st.median(x[0] for x in v) for r, v in d["gc"].items()}
            out.append(f"HOSTGC arm={arm} ranks={len(d['gc'])} windows={max(len(v) for v in d['gc'].values())} "
                       f"r0_us_per_step={per[g0]:.1f} max_us_per_step={max(per.values()):.1f} "
                       f"r0_coll_per_step={st.median(x[1] for x in d['gc'][g0]):.2f} "
                       f"r0_gen2_per_window={st.median(x[2] for x in d['gc'][g0]):.0f} "
                       f"max_pause_us={max(x[3] for v in d['gc'].values() for x in v)}")
        out.append(f"HOSTSUBSUM arm={arm} ranks={len(ranks)} windows={windows} subs={len(names)} verdict=OK")
    return out, ok


def selftest():
    def sub(r, s, items):
        return (f"(Worker_TP{r}_EP{r} pid=1) INFO [host_phases.py:170] HOSTSUBS rank={r} steps={s} per-step sub-phase host us "
                "(p50/p90/mean-per-step/n-steps) " + " ".join(f"~{k}={v}" for k, v in items))

    def ph(r, s, items):
        return (f"(Worker_TP{r}_EP{r} pid=1) INFO [host_phases.py:150] HOSTPHASES rank={r} steps={s} per-step host us "
                "(p50/p90/mean-per-step) total_mean=999 " + " ".join(f"{k}={v}" for k, v in items))
    lines = ["==> Q38_ARM_START run01_d14 env=[x]"]
    for w, s in enumerate((64, 128, 192)):
        for r in (0, 1):
            base = 100 + 10 * w + r
            lines.append(sub(r, s, [("pf_logits", f"{base}/{base+5}/{base}/64"), ("pf_counters", "20/22/20/64"),
                                    ("fd_whole_call", f"{300+r}/310/{300+r}/64")]))
            lines.append(ph(r, s, [("post_forward", f"350/360/{350+w}"), ("forward_dispatch", "540/560/540")]))
    # A HOSTSUBS line between two arms belongs to no arm (dropped).
    lines += ["==> Q38_ARM_END run01_d14 rc=0", sub(0, 256, [("pf_logits", "900/900/900/64")]),
              "==> Q38_ARM_START run02_base env=[x]", "nothing here", "==> Q38_ARM_END run02_base rc=0"]
    data = parse(lines, skip=1)
    out, ok = report(data, ["d14", "base"])
    checks = [
        ("NO_DATA for an arm without HOSTSUBS", any("arm=base" in x and "verdict=NO_DATA" in x for x in out) and not ok),
        ("warm-up window skipped: r0 pf_logits median of 110, 120 = 115",
         any("sub=pf_logits" in x and "r0_mean=115 " in x for x in out)),
        ("rank max over ranks: pf_logits max_mean 116", any("sub=pf_logits" in x and "max_mean=116 " in x for x in out)),
        ("phase map pf_ -> post_forward", any("sub=pf_counters phase=post_forward" in x for x in out)),
        ("phase map fd_ -> forward_dispatch", any("sub=fd_whole_call phase=forward_dispatch" in x for x in out)),
        ("remainder = phase mean (median 351.5) - subs (115 + 20)",
         any("HOSTSUBPH arm=d14 phase=post_forward r0_phase_mean=352 subs_sum=135 remainder=216" in x for x in out)),
        ("windows counted after skip = 2", any("HOSTSUBSUM arm=d14 ranks=2 windows=2 subs=3 verdict=OK" in x for x in out)),
        ("no markers: whole log is one arm 'all'", report(parse([sub(0, 64, [("st_async", "5/6/5/64")])] * 2, 0), ["all"])[1]),
        ("HOSTPHASES ~ tokens never parsed as subs", not any("sub=~" in x for x in out)),
    ]
    # q38 L6-INSTR rows: one fixture sub per new prefix (and the specific pi_
    # names), each must land in its parent phase; plus the GC census summary.
    inst = [("pi_seed_sync", "prepare_inputs"), ("pi_collect", "prepare_inputs"), ("pi_h2d", "prep_host_and_uploads"),
            ("pp_fused", "prep_host_and_uploads"), ("ap_fused", "adapter_prepare_inputs"),
            ("dpi_seed", "dp_prepare_inputs"), ("df0_call", "dp_forward0"), ("dc_call", "dp_carries"),
            ("ss_draft_chunks", "sample_setup"), ("dl_fold", "dp_loop"), ("df_fold", "dp_forward0")]

    def gcl(r, s, us, n, g2, mx):
        return (f"(Worker_TP{r}_EP{r} pid=1) INFO [host_phases.py:240] GCCENSUS rank={r} steps={s} window=200 "
                f"cyclic GC pauses (sum_us/collections/gen2_collections) gc_us={us}/{n}/{g2} max_us={mx} "
                f"g0={us}/{n} g1=0/0 g2=0/{g2}")
    lines2 = []
    for w, s in enumerate((200, 400, 600)):
        for r in (0, 1):
            lines2.append(sub(r, s, [(k, f"{10 + i}/{12 + i}/{10 + i}/200") for i, (k, _) in enumerate(inst)]))
            lines2.append(gcl(r, s, 2000 * (w + 1) + 1000 * r, 40, 2 * w, 300 + w))
    out2, ok2 = report(parse(lines2, skip=1), ["all"])
    checks += [(f"row {k} -> {ph}", ok2 and any(f"sub={k} phase={ph} " in x for x in out2)) for k, ph in inst]
    # GC: windows 2..3 after the skip; r0 sum 4000, 6000 us / 200 steps -> median 25.0 us per step; r1 30.0;
    # 40 collections / 200 steps = 0.20; gen2 2, 4 -> median 3; longest pause 302 (the skipped window had 300).
    checks.append(("GC census per step and rank max", any(
        x.startswith("HOSTGC arm=all ranks=2 windows=2 r0_us_per_step=25.0 max_us_per_step=30.0 "
                     "r0_coll_per_step=0.20 r0_gen2_per_window=3 max_pause_us=302") for x in out2)))
    out += out2
    # q38 L6 P-METER rows: mpr_ / am_ sub-marks land in their phases.
    out3, ok3 = report(parse([sub(0, 64, [("mpr_wait", "9/9/9/64"), ("vr_call", "8/8/8/64"), ("am_pack", "5/5/5/64"), ("am_h2d", "4/4/4/64"),
                                         ("am_walk", "3/3/3/64")])] * 2, 0), ["all"])
    checks += [(f"row {k} -> {ph}", ok3 and any(f"sub={k} phase={ph} " in x for x in out3))
               for k, ph in (("mpr_wait", "dp_entry"), ("vr_call", "vfy_rejection"), ("am_pack", "prep_attention_metadata"),
                             ("am_h2d", "prep_attention_metadata"), ("am_walk", "prep_attention_metadata"))]
    # P-METER premium: 8 ranks, windows of 100 steps (seq 1..100, 101..200). Rank 3's forward enqueue is 1,500 us late
    # on every 10th step of the second window only; rank 2 lost its first window's line; rank 5 had the meter off at
    # seq 155. So 99 steps align (101..200 minus 155), 101 are dropped, and 10 spikes are recovered on rank 3.
    def pml(r, host, seq0, w, ents, shift=0):
        t0 = (1_000_000 + seq0 * 7000 + 5 * r + shift) * 1000
        parts = []
        for i in range(w):
            sq = seq0 + i
            if sq in ents.get("x", ()):
                parts.append("x")
                continue
            dq = 1_000_000 + sq * 7000 + 5 * r + shift
            f = 800 + r + (1500 if ents.get("spike") and r == 3 and sq > 100 and sq % 10 == 0 else 0)
            a = 900 + r
            parts.append(f"{dq - t0 // 1000},{f},{a},{a + 300 - 10 * r},{2000 + 2 * r}")
        return (f"(Worker_TP{r}_EP{r} pid=1) INFO 11-14 05:00:00 [l6_pmeter.py:200] PMETER rank={r} host={host} pid=9 "
                f"window={w} seq0={seq0} n={w} t0_ns={t0} e=" + ";".join(parts))
    lines4 = []
    for seq0 in (1, 101):
        for r in range(8):
            if r == 2 and seq0 == 1:
                continue
            lines4.append(pml(r, "h1", seq0, 100, {"spike": True, "x": (155,) if r == 5 else ()}))
    lines4 += ["(Worker_TP0_EP0 pid=1) INFO x [l6_pmeter.py:201] THREADSTAT rank=0 seq=100 n=2 MainThread:11=1000000/500000/10/5/2 out_thr:12=-",
               "(Worker_TP0_EP0 pid=1) INFO x [l6_pmeter.py:201] THREADSTAT rank=0 seq=200 n=2 MainThread:11=3000000/800000/30/9/6 out_thr:12=-"]
    out4, _ = report(parse(lines4, 0), ["all"])
    checks += [
        ("pmeter fwd: aligned 99, dropped 101, rel p50 3.5 / mean 154.5, abs 21.0 / 169.5, laggard rank 7",
         "HOSTPMETER arm=all ranks=8 hosts=1 steps=99 dropped=101 stamp=fwd rel_prem_p50=3.5 rel_prem_mean=154.5 "
         "abs_prem_p50=21.0 abs_prem_mean=169.5 laggard=7:0.90" in out4),
        ("pmeter spike recovered: 10 steps on rank 3, 1498.5 us",
         "HOSTPMSPIKE arm=all stamp=fwd thr_us=500 steps=10 share=0.101 mean_us=1498.5 top=3:10" in out4),
        ("pmeter loop: rel 7.0, abs 24.5",
         "HOSTPMETER arm=all ranks=8 hosts=1 steps=99 dropped=101 stamp=loop rel_prem_p50=7.0 rel_prem_mean=7.0 "
         "abs_prem_p50=24.5 abs_prem_mean=24.5 laggard=7:1.00" in out4),
        ("pmeter wait: rank mean 265, min 230, r0 300",
         "HOSTPMWAIT arm=all steps=99 wait_rankmean_mean=265.0 wait_rankmean_p50=265.0 wait_rankmin_mean=230.0 "
         "wait_r0_mean=300.0" in out4),
        ("threadstat window delta", "HOSTTHREAD arm=all rank=0 thread=MainThread windows=1 run_us_med=2000 wait_us_med=300 nvcsw_med=4" in out4),
    ]
    # Two hosts with a 1 s clock offset: the absolute premium is not read, the step-relative one is unaffected.
    lines5 = [pml(r, "h1" if r < 4 else "h2", 1, 100, {}, shift=0 if r < 4 else 1_000_000) for r in range(8)]
    out5, _ = report(parse(lines5, 0), ["all"])
    checks.append(("two hosts: abs premium NA, rel premium 3.5",
                   "HOSTPMETER arm=all ranks=8 hosts=2 steps=100 dropped=0 stamp=fwd rel_prem_p50=3.5 rel_prem_mean=3.5 "
                   "abs_prem_p50=NA abs_prem_mean=NA laggard=7:1.00" in out5))
    out += out3 + out4 + out5
    bad = [n for n, c in checks if not c]
    for n in bad:
        print(f"SELFTEST FAIL {n}")
    print("\n".join(out))
    print(f"HOSTSUB_SELFTEST {'PASS' if not bad else 'FAIL'} ({len(checks) - len(bad)}/{len(checks)} checks)")
    return 0 if not bad else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="*")
    ap.add_argument("--arm", action="append", default=[])
    ap.add_argument("--skip-windows", type=int, default=1)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    lines = []
    for p in a.logs:
        opener = open
        if p.endswith(".gz"):
            import gzip
            opener = gzip.open
        with opener(p, "rt", errors="replace") as f:
            lines.extend(ln.rstrip("\n") for ln in f)
    data = parse(lines, a.skip_windows)
    arms = a.arm or sorted(data)
    out, ok = report(data, arms)
    print("\n".join(out))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
