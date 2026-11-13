#!/usr/bin/env python3
"""Compare a repro run against the handoff v3 reference numbers.

  scripts/compare_results.py prefill runs/<thread>/<run>/summary.json
  scripts/compare_results.py decode  runs/<thread>/<run>/          (dir with result_*.json)

Tolerance: +/-5% by default (--tol 0.05). Different node / cluster / spot jitter is expected
to move numbers by a few percent; anything beyond that is worth investigating.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Handoff v3, section 6.1 (commit cd4eedb, node gke-tpu-6b80bbef-2xq5, 2026-10-12 20:58 UTC).
PREFILL_REF = {  # label: (input tok/s per node, mean TTFT ms)
    "128k_c4": (186461.8, 2792.8),
    "128k_c8": (197315.9, 4973.0),
    "128k_c16": (196687.8, 8659.9),
    "100k_c8": (195977.4, 3819.0),
    "100k_c16": (195191.6, 6675.3),
    "32k_c8": (188896.0, 1339.7),
}
# Handoff v3, section 6.2. NOTE: these came from the w384 tree with stock `vllm bench serve` and
# prefix caching OFF; the current driver uses its own client + prefix caching with 8 shared prefixes.
DECODE_REF = {  # file: (concurrency, step ms, steady-state tok/s/node @tau=3.5)
    "result_dp8_hbm0p90_64x1024_verified.json": (64, 37.02, 6108.5),
    "result_dp8_4x_c256.json": (256, 65.06, 14223.4),
    "result_dp8_8x_c512.json": (512, None, None),  # 2-wave in the reference; wall 8,586.2
}


def pct(new: float, ref: float) -> str:
    return f"{(new - ref) / ref * 100:+.1f}%"


def flag(new: float, ref: float, tol: float, higher_is_better: bool = True) -> str:
    d = (new - ref) / ref
    if abs(d) <= tol:
        return "OK"
    return "BETTER" if (d > 0) == higher_is_better else "WORSE"


def prefill(path: Path, tol: float) -> None:
    s = json.loads(path.read_text())
    print(f"prefill run: commit={s.get('git_commit_head')} node={s.get('node')} cluster={s.get('cluster')}")
    print(f"{'workload':10} {'tok/s/node':>12} {'ref':>12} {'diff':>8} {'per chip':>9} {'TTFT ms':>9} {'ref':>8}  verdict")
    for label, (ref_tps, ref_ttft) in PREFILL_REF.items():
        r = s.get("results", {}).get(label)
        if not r:
            print(f"{label:10} {'(missing)':>12}")
            continue
        tps = r["total_input_tokens"] / r["duration"]
        ttft = r.get("mean_ttft_ms", 0.0)
        fails = r.get("num_prompts", r.get("completed", 0)) - r.get("completed", 0)
        print(f"{label:10} {tps:12,.1f} {ref_tps:12,.1f} {pct(tps, ref_tps):>8} {tps/4:9,.0f} {ttft:9,.1f} {ref_ttft:8,.1f}  "
              f"{flag(tps, ref_tps, tol)}{'  FAILED_REQS=' + str(fails) if fails else ''}")


def decode(d: Path, tol: float) -> None:
    for fn, (c, ref_step, ref_tps) in DECODE_REF.items():
        p = d / fn
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        tau = r.get("natural_mean_accept_len") or 1.0
        int_steps_s = r.get("steady_state_interior_step_throughput_steps_s")
        step = (c * 1000.0 / int_steps_s) if int_steps_s else (r["mean_tpot_ms"] * tau)
        ss = r.get("steady_state_interior_throughput_at_accept_3p5_tok_s")
        print(f"\n{fn}  (c={c}, completed={r.get('completed')}/{r.get('num_prompts')}, failed={r.get('failed')}, "
              f"single_wave={r.get('single_wave_verified')}, natural tau={tau:.4f})")
        if ref_step:
            print(f"  step ms          : {step:8.2f}   ref {ref_step:8.2f}   {pct(step, ref_step):>8}  {flag(step, ref_step, tol, False)}")
        else:
            print(f"  step ms          : {step:8.2f}   (reference was 2-wave; no step ref)")
        if ss and ref_tps:
            print(f"  tok/s/node @3.5  : {ss:8,.1f}   ref {ref_tps:8,.1f}   {pct(ss, ref_tps):>8}  {flag(ss, ref_tps, tol)}"
                  f"   per chip {ss/4:,.1f}")
        elif ss:
            print(f"  tok/s/node @3.5  : {ss:8,.1f}   per chip {ss/4:,.1f}")
        print(f"  wall tok/s @3.5  : {r.get('wall_clock_throughput_at_accept_3p5_tok_s')}")
        print(f"  gates            : 2+2 passed={r.get('passed_2p2')}  USA states={r.get('num_us_states_matched')}/50 "
              f"(known failing in handoff)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["prefill", "decode"])
    ap.add_argument("path")
    ap.add_argument("--tol", type=float, default=0.05)
    a = ap.parse_args()
    p = Path(a.path)
    if a.kind == "prefill":
        prefill(p if p.is_file() else p / "summary.json", a.tol)
    else:
        decode(p if p.is_dir() else p.parent, a.tol)


if __name__ == "__main__":
    main()
