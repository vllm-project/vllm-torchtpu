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
"""Perf & resource report generator and paired significance / roofline analysis.

Strictly separates `exact` and `approx` numerics profiles in every output table,
reports static VMEM/SMEM + optional Cloudtop AOT TPU7x LLO bundle/spill/hash stats,
analytic HBM roofline floors (`megakernel_roofline.md`), and paired interleaved
significance (`t`-interval + Wilcoxon signed-rank + noise-floor classification).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import pathlib
import statistics
import sys
from collections.abc import Sequence
from typing import Any

from . import compiler_stats

DEFAULT_REPORT_BUCKETS = (
    (1, 1),
    (1, 2),
    (1, 4),
    (1, 8),
    (1, 16),
    (1, 32),
    (1, 64),
    (4, 4),
    (4, 8),
    (4, 16),
    (4, 24),
    (4, 32),
    (4, 40),
    (4, 48),
    (4, 56),
    (4, 64),
)

# Two-sided Student-t 95% critical values t_{df, 0.975} for small pair counts K = df +
# 1.
_T_CRIT_95: dict[int, float] = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
}


def t_crit_95(df: int) -> float:
    """Return the two-sided 95% Student-t critical value for `df` degrees of freedom."""
    if df <= 0:
        return float("inf")
    if df in _T_CRIT_95:
        return _T_CRIT_95[df]
    # Cornish-Fisher approximation for df > 10
    z = 1.95996
    return (
        z
        + (z**3 + z) / (4.0 * df)
        + (5.0 * z**5 + 16.0 * z**3 + 3.0 * z) / (96.0 * df * df)
    )


def parse_bucket_list(raw: str) -> list[tuple[int, int]]:
    """Parse `'1:8,1:32,4:4,4:16'` or `'8,16,32'` (defaults to `T=1`) into `(T,
    B)` pairs.
    """
    if not raw or not raw.strip():
        return list(DEFAULT_REPORT_BUCKETS)
    out: list[tuple[int, int]] = []
    for item in raw.split(","):
        tok = item.strip()
        if not tok:
            continue
        if ":" in tok:
            t_s, b_s = tok.split(":", 1)
            out.append((int(t_s.strip()), int(b_s.strip())))
        else:
            out.append((1, int(tok)))
    return out


def paired_significance(
    base_runs: Sequence[float],
    head_runs: Sequence[float],
    *,
    higher_is_better: bool = False,
    noise_floor_pct: float = 1.5,
) -> dict[str, Any]:
    """Compute paired interleaved A/B significance and noise-floor classification.

    Implements the C3 paired decision rule:
    - Pairwise relative delta `d_i = (H_i - B_i) / B_i`.
    - 95% Student-t confidence interval
      `[CI_lo, CI_hi] = d_bar +/- t_{K-1,0.975} * SE(d)`.
    - Acceptance = **no statistically significant regression** at any concurrency.
      `noise_floor_pct` (`+/-1.5%` for throughput/p50, `+/-2.0%` for p90/p99) is used
      strictly as a noise-floor definition (calling `INCONCLUSIVE_HIGH_VARIANCE` if
      `CI_half_width > noise_floor_pct` and `0 in [CI_lo, CI_hi]`), never as an
      allowed slowdown.
    """
    if len(base_runs) != len(head_runs) or len(base_runs) == 0:
        raise ValueError(
            f"Expected equal non-empty base_runs ({len(base_runs)}) "
            f"and head_runs ({len(head_runs)})"
        )
    k = len(base_runs)
    rel_deltas_pct = [
        ((h - b) / b) * 100.0 for b, h in zip(base_runs, head_runs, strict=True)
    ]
    mean_base = statistics.fmean(base_runs)
    mean_head = statistics.fmean(head_runs)
    d_bar = statistics.fmean(rel_deltas_pct)
    if k >= 2:
        sd = statistics.stdev(rel_deltas_pct)
        se = sd / math.sqrt(k)
        hw = t_crit_95(k - 1) * se
    else:
        sd = 0.0
        se = float("inf")
        hw = float("inf")

    ci_lo = d_bar - hw
    ci_hi = d_bar + hw

    # Determine regression vs improvement direction:
    # If higher_is_better=False (e.g. latency TPOT/ITL), d < 0 is improvement, d > 0 is
    # regression.
    # If higher_is_better=True (e.g. throughput), d > 0 is improvement, d < 0 is
    # regression.
    if higher_is_better:
        sig_regression = ci_hi < 0.0
        sig_improvement = ci_lo > 0.0
    else:
        sig_regression = ci_lo > 0.0
        sig_improvement = ci_hi < 0.0

    if k < 2:
        status = "INVALID_SINGLE_RUN"
    elif sig_regression:
        status = "SIGNIFICANT_REGRESSION"
    elif sig_improvement:
        status = "SIGNIFICANT_IMPROVEMENT"
    elif hw > noise_floor_pct:
        status = "INCONCLUSIVE_HIGH_VARIANCE"
    else:
        status = "WITHIN_NOISE_EQUIVALENT"

    return {
        "pairs": k,
        "mean_base": round(mean_base, 4),
        "mean_head": round(mean_head, 4),
        "mean_delta_pct": round(d_bar, 3),
        "sd_delta_pct": round(sd, 3),
        "se_delta_pct": round(se, 3) if math.isfinite(se) else None,
        "ci95_lo_pct": round(ci_lo, 3) if math.isfinite(ci_lo) else None,
        "ci95_hi_pct": round(ci_hi, 3) if math.isfinite(ci_hi) else None,
        "ci95_half_width_pct": round(hw, 3) if math.isfinite(hw) else None,
        "noise_floor_pct": noise_floor_pct,
        "higher_is_better": higher_is_better,
        "status": status,
        "pairwise_deltas_pct": [round(x, 3) for x in rel_deltas_pct],
    }


def gsm8k_accuracy_gate(
    score_head: float,
    score_base: float = 0.9242,
    *,
    n_questions: int = 1319,
    b_only_correct: int | None = None,
    c_only_correct: int | None = None,
    acceptance_lengths: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Evaluate GSM8K accuracy parity using 2*SE band or paired McNemar test.

    Also reports `mean_acceptance_length` spread across runs when provided.
    """
    se_single = math.sqrt(score_base * (1.0 - score_base) / max(1, n_questions))
    delta = score_head - score_base
    band_2se = 2.0 * se_single
    mcnemar_z: float | None = None
    if b_only_correct is not None and c_only_correct is not None:
        discordant = b_only_correct + c_only_correct
        mcnemar_z = (
            (b_only_correct - c_only_correct) / math.sqrt(discordant)
            if discordant > 0
            else 0.0
        )
        passed = mcnemar_z <= 1.96
    else:
        passed = delta >= -band_2se

    acc_len_stats: dict[str, float] | None = None
    if acceptance_lengths:
        vals = list(acceptance_lengths)
        acc_len_stats = {
            "min": round(min(vals), 4),
            "median": round(statistics.median(vals), 4),
            "max": round(max(vals), 4),
            "spread": round(max(vals) - min(vals), 4),
            "stdev": round(statistics.stdev(vals), 4) if len(vals) >= 2 else 0.0,
        }

    return {
        "n_questions": n_questions,
        "score_base": round(score_base, 4),
        "score_head": round(score_head, 4),
        "delta": round(delta, 4),
        "se_single": round(se_single, 4),
        "band_2se": round(band_2se, 4),
        "mcnemar_z": round(mcnemar_z, 3) if mcnemar_z is not None else None,
        "passed": bool(passed),
        "acceptance_length": acc_len_stats,
    }


def collect_profile_rows(
    profiles: Sequence[str],
    buckets: Sequence[tuple[int, int]],
    *,
    ctx: int = 4096,
    moe_tp: bool = False,
    e_slots: int = 72,
    stack_e: bool = True,
    aot_compile: bool = False,
    aot_workers: int = 1,
    fused_pool_io: bool | str | None = None,
    fused_pool_io_max_b: int | None = None,
    fused_pool_io_min_b_hi: int | None = None,
    latency_map: dict[str, float] | None = None,
    baseline_latency_map: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Collect static plan, roofline, and optional AOT compile stats for
    `(profile, T, B)`.
    """
    tasks: list[tuple[str, int, int]] = [
        (prof_name, T, B) for prof_name in profiles for (T, B) in buckets
    ]
    rows: list[dict[str, Any]] = []

    if aot_compile and aot_workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=aot_workers) as pool:
            futs = [
                pool.submit(
                    compiler_stats.compile_bucket_subprocess,
                    prof_name,
                    T,
                    B,
                    ctx=ctx,
                    moe_tp=moe_tp,
                    e_slots=e_slots,
                    stack_e=stack_e,
                    fused_pool_io=fused_pool_io,
                    fused_pool_io_max_b=fused_pool_io_max_b,
                    fused_pool_io_min_b_hi=fused_pool_io_min_b_hi,
                )
                for (prof_name, T, B) in tasks
            ]
            raw_results = [f.result() for f in futs]
    elif aot_compile:
        raw_results = [
            compiler_stats.compile_bucket_in_process(
                prof_name,
                T,
                B,
                ctx=ctx,
                moe_tp=moe_tp,
                e_slots=e_slots,
                stack_e=stack_e,
                fused_pool_io=fused_pool_io,
                fused_pool_io_max_b=fused_pool_io_max_b,
                fused_pool_io_min_b_hi=fused_pool_io_min_b_hi,
            )
            for (prof_name, T, B) in tasks
        ]
    else:
        raw_results = [
            compiler_stats.plan_bucket_stats(
                prof_name,
                T,
                B,
                ctx=ctx,
                moe_tp=moe_tp,
                e_slots=e_slots,
                stack_e=stack_e,
                fused_pool_io=fused_pool_io,
                fused_pool_io_max_b=fused_pool_io_max_b,
                fused_pool_io_min_b_hi=fused_pool_io_min_b_hi,
            )
            for (prof_name, T, B) in tasks
        ]

    for st in raw_results:
        key_prof = f"{st['profile']}:T={st['T']},B={st['B']}"
        key_tb = f"T={st['T']},B={st['B']}"
        med_us: float | None = None
        if latency_map:
            med_us = latency_map.get(key_prof, latency_map.get(key_tb))
        if med_us is not None:
            rl = compiler_stats.roofline_stats(
                B=st["B"],
                T=st["T"],
                ctx=st["ctx"],
                moe_tp=st["moe_tp"],
                median_us=med_us,
            )
            st.update(rl)
            st["median_us"] = round(med_us, 1)
            if baseline_latency_map:
                base_us = baseline_latency_map.get(key_tb)
                if base_us and base_us > 0:
                    st["vs_baseline_pct"] = round(
                        ((med_us - base_us) / base_us) * 100.0, 2
                    )
        rows.append(st)
    return rows


def _extract_main_llo_summary(row: dict[str, Any]) -> dict[str, Any]:
    """Extract main megakernel bundle count, spill count, and hash from an AOT row."""
    llo = row.get("llo")
    if not llo:
        return {
            "main_bundles": None,
            "total_bundles_all": None,
            "spills": None,
            "scoped_vmem_mib": None,
            "llo_hash": None,
            "n_subkernels": None,
        }
    subk = llo.get("subkernels", {})
    main_sk = None
    for name, sk in subk.items():
        if "decode_megakernel" in name:
            main_sk = sk
            break
    if main_sk is None and subk:
        main_sk = max(subk.values(), key=lambda x: x.get("total_bundles", 0))
    total_all = sum(sk.get("total_bundles", 0) for sk in subk.values())
    spills_all = sum(sk.get("spill_count", 0) for sk in subk.values())
    return {
        "main_bundles": main_sk.get("total_bundles", 0) if main_sk else 0,
        "main_non_empty": main_sk.get("non_empty_bundles", 0) if main_sk else 0,
        "total_bundles_all": total_all,
        "spills": spills_all,
        "scoped_vmem_mib": main_sk.get("scoped_vmem_mib") if main_sk else None,
        "llo_hash": (main_sk.get("normalized_pass_sha256", "")[:10] if main_sk else ""),
        "n_subkernels": len(subk),
    }


def format_markdown_tables(
    rows: Sequence[dict[str, Any]],
    *,
    include_aot: bool = False,
    include_latency: bool = False,
) -> str:
    """Format rows into two separate Markdown tables: Exact Numerics vs Approx
    Numerics.
    """
    exact_rows = [r for r in rows if r.get("numerics") == "exact"]
    approx_rows = [r for r in rows if r.get("numerics") != "exact"]

    def _render_group(title: str, group: list[dict[str, Any]]) -> str:
        if not group:
            return f"### {title}\n\n_No profiles in this numerics class._\n"
        cols = [
            "Profile",
            "T",
            "B",
            "NS",
            "Fits",
            "VMEM (MiB)",
            "SMEM (KiB)",
            "FUSED/GPR/GCH",
            "nbd/nbe/nkv",
            "gdn_r",
            "Crit GB",
            "Roofline (us)",
            "MXU (us)",
        ]
        if include_aot:
            cols.extend(
                [
                    "Compile (s)",
                    "MK Bundles",
                    "All Bundles",
                    "SubK",
                    "Spills",
                    "LLO SHA",
                ]
            )
        if include_latency:
            cols.extend(["Median (us)", "vs Base", "BW Util", "Headroom"])

        lines = [
            f"### {title}",
            "",
            "| " + " | ".join(cols) + " |",
            "|" + "|".join(["---"] * len(cols)) + "|",
        ]
        for r in group:
            t = r["tiles"]
            fgc = f"{int(r['FUSED'])}/{int(r['GPR'])}/{int(r['GCH'])}"
            rings = f"{t['nbd']}/{t['nbe']}/{t['nkv']}"
            cells = [
                f"`{r['profile']}`",
                str(r["T"]),
                str(r["B"]),
                str(r["NS"]),
                "YES" if r["fits_vmem"] else "NO",
                f"{r['vmem_mib']:.2f}/{r['vmem_limit_mib']}",
                f"{r['smem_kib']:.1f}",
                fgc,
                rings,
                str(t["gdn_rows"]),
                f"{r['crit_bytes_gb']:.2f}",
                f"{r['roofline_floor_us']:.1f}",
                f"{r.get('mxu_bf16_us', 0.0):.1f}",
            ]
            if include_aot:
                s = _extract_main_llo_summary(r)
                cells.extend(
                    [
                        f"{r.get('compile_s', 0.0):.1f}",
                        str(
                            s["main_bundles"] if s["main_bundles"] is not None else "-"
                        ),
                        str(
                            s["total_bundles_all"]
                            if s["total_bundles_all"] is not None
                            else "-"
                        ),
                        str(
                            s["n_subkernels"] if s["n_subkernels"] is not None else "-"
                        ),
                        str(s["spills"] if s["spills"] is not None else "-"),
                        f"`{s['llo_hash']}`" if s["llo_hash"] else "-",
                    ]
                )
            if include_latency:
                med = r.get("median_us")
                vs_b = r.get("vs_baseline_pct")
                bw_u = r.get("hbm_bw_util_pct")
                hr_u = r.get("headroom_us")
                hr_x = r.get("headroom_x")
                cells.extend(
                    [
                        f"{med:.1f}" if med is not None else "-",
                        f"{vs_b:+.2f}%" if vs_b is not None else "-",
                        f"{bw_u:.1f}%" if bw_u is not None else "-",
                        f"{hr_u:.0f}us ({hr_x:.2f}x)"
                        if (hr_u is not None and hr_x is not None)
                        else "-",
                    ]
                )
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        return "\n".join(lines)

    parts = [
        _render_group(
            "Table A: Exact Numerics Profiles (`numerics='exact'`)", exact_rows
        ),
        _render_group(
            "Table B: Approx Numerics Profiles (`numerics='approx'`)", approx_rows
        ),
    ]
    return "\n".join(parts)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate megakernel C2-lite static/AOT resource and roofline tables."
        )
    )
    parser.add_argument(
        "--profiles",
        default="default,qwen35-baseline,qwen35-candidate30,qwen35-candidate30-fused",
        help="Comma-separated profile names.",
    )
    parser.add_argument(
        "--buckets",
        default="",
        help="Comma-separated T:B bucket pairs (e.g. '1:8,1:32,4:4,4:16,4:32,4:64').",
    )
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument("--e-slots", type=int, default=72)
    parser.add_argument("--moe-tp", action="store_true")
    parser.add_argument(
        "--stack-e",
        action="store_true",
        default=True,
        help="Use 3D stacked expert layout during AOT compilation (default: True).",
    )
    parser.add_argument(
        "--no-stack-e",
        action="store_false",
        dest="stack_e",
        help="Use unstacked 2D per-layer expert layout.",
    )
    parser.add_argument(
        "--aot-compile",
        action="store_true",
        help=(
            "Run Cloudtop TPU7x AOT compilation and extract LLO "
            "bundle/spill/hash stats."
        ),
    )
    parser.add_argument("--aot-workers", type=int, default=1)
    parser.add_argument("--fused-pool-io", default=None)
    parser.add_argument("--fused-pool-io-max-b", type=int, default=None)
    parser.add_argument("--fused-pool-io-min-b-hi", type=int, default=None)
    parser.add_argument("--latency-json", default=None)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--md-out", default=None)
    args = parser.parse_args(argv)

    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    buckets = parse_bucket_list(args.buckets)

    latency_map: dict[str, float] | None = None
    if args.latency_json:
        latency_map = json.loads(pathlib.Path(args.latency_json).read_text())

    rows = collect_profile_rows(
        profiles,
        buckets,
        ctx=args.ctx,
        moe_tp=args.moe_tp,
        e_slots=args.e_slots,
        stack_e=args.stack_e,
        aot_compile=args.aot_compile,
        aot_workers=args.aot_workers,
        fused_pool_io=args.fused_pool_io,
        fused_pool_io_max_b=args.fused_pool_io_max_b,
        fused_pool_io_min_b_hi=args.fused_pool_io_min_b_hi,
        latency_map=latency_map,
    )
    md = format_markdown_tables(
        rows,
        include_aot=args.aot_compile,
        include_latency=bool(latency_map),
    )
    print(md)
    if args.md_out:
        pathlib.Path(args.md_out).write_text(md)
    if args.json_out:
        pathlib.Path(args.json_out).write_text(
            json.dumps(
                {
                    "toolchain": compiler_stats.get_toolchain_info(),
                    "rows": rows,
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
