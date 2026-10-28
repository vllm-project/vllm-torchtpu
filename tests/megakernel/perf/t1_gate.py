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
"""T1 Perf Gate Stage 1 (Cloudtop AOT full normalized LLO/Mosaic diff) & Stage
2 evaluator.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import pathlib
import sys
from collections.abc import Sequence
from typing import Any

from . import compiler_stats, report

DEFAULT_GATE_BUCKETS = "1:8,1:32,4:4,4:16,4:32,4:64"
DEFAULT_GATE_PROFILES = "default,qwen35-candidate30-fused"


def run_stage1_comparison(
    base_wt: str | pathlib.Path,
    head_wt: str | pathlib.Path,
    *,
    base_sha: str = "base",
    head_sha: str = "head",
    profiles: Sequence[str] = ("default", "qwen35-candidate30-fused"),
    buckets: Sequence[tuple[int, int]] = (
        (1, 8),
        (1, 32),
        (4, 4),
        (4, 16),
        (4, 32),
        (4, 64),
    ),
    workers: int = 2,
    stack_e: bool = True,
    out_dir: str | pathlib.Path | None = None,
) -> dict[str, Any]:
    """Compile `(profile, T, B)` in `base_wt` and `head_wt` and diff full LLO + Mosaic
    dumps.
    """
    out_path = pathlib.Path(out_dir).resolve() if out_dir else None
    if out_path:
        out_path.mkdir(parents=True, exist_ok=True)

    tasks: list[tuple[str, str, pathlib.Path, str, int, int, pathlib.Path | None]] = []
    for prof_name in profiles:
        for T, B in buckets:
            b_dump = (
                out_path / "dumps" / f"base_{prof_name}_t{T}_b{B}" if out_path else None
            )
            h_dump = (
                out_path / "dumps" / f"head_{prof_name}_t{T}_b{B}" if out_path else None
            )
            tasks.append(
                ("base", base_sha, pathlib.Path(base_wt), prof_name, T, B, b_dump)
            )
            tasks.append(
                ("head", head_sha, pathlib.Path(head_wt), prof_name, T, B, h_dump)
            )

    compiled_map: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        fut_to_key = {
            pool.submit(
                compiler_stats.compile_bucket_subprocess,
                prof_name,
                T,
                B,
                worktree=wt,
                stack_e=stack_e,
                dump_dir=d_dir,
            ): (side, prof_name, T, B)
            for (side, _sha, wt, prof_name, T, B, d_dir) in tasks
        }
        for fut, key in fut_to_key.items():
            compiled_map[key] = fut.result()

    bucket_diffs: list[dict[str, Any]] = []
    any_invalid_tc = False
    any_diff = False

    for prof_name in profiles:
        for T, B in buckets:
            b_res = compiled_map[("base", prof_name, T, B)]
            h_res = compiled_map[("head", prof_name, T, B)]
            d = compiler_stats.diff_compiled_buckets(
                b_res,
                h_res,
                diff_out_dir=(out_path / "diffs") if out_path else None,
            )
            for side_res in (b_res, h_res):
                dd = side_res.get("dump_dir")
                if dd and pathlib.Path(dd).is_dir():
                    for f in pathlib.Path(dd).glob("*.txt"):
                        f.unlink(missing_ok=True)
            if d["status"] == "INVALID_TOOLCHAIN":
                any_invalid_tc = True
            elif not d["exact_match"]:
                any_diff = True
            bucket_diffs.append(
                {
                    "profile": prof_name,
                    "numerics": b_res.get("numerics", "unknown"),
                    "T": T,
                    "B": B,
                    "base_FUSED": b_res.get("FUSED"),
                    "head_FUSED": h_res.get("FUSED"),
                    "base_vmem_mib": b_res.get("vmem_mib"),
                    "head_vmem_mib": h_res.get("vmem_mib"),
                    "base_smem_kib": b_res.get("smem_kib"),
                    "head_smem_kib": h_res.get("smem_kib"),
                    "base_compile_s": b_res.get("compile_s"),
                    "head_compile_s": h_res.get("compile_s"),
                    "base_llo_sha": b_res.get("llo", {}).get(
                        "full_llo_mosaic_sha256", ""
                    )[:12],
                    "head_llo_sha": h_res.get("llo", {}).get(
                        "full_llo_mosaic_sha256", ""
                    )[:12],
                    **d,
                }
            )

    if any_invalid_tc:
        overall = "INVALID_TOOLCHAIN"
    elif any_diff:
        overall = "DIFF"
    else:
        overall = "EQUAL"

    sample_tc = (
        next(iter(compiled_map.values())).get("toolchain", {}) if compiled_map else {}
    )
    summary = {
        "base_sha": base_sha,
        "head_sha": head_sha,
        "overall_status": overall,
        "toolchain": sample_tc,
        "buckets": bucket_diffs,
    }
    if out_path:
        (out_path / "stage1_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
    return summary


def format_stage1_markdown(summary: dict[str, Any]) -> str:
    """Format Stage 1 LLO + Mosaic diff summary into Exact and Approx Markdown
    tables.
    """
    tc = summary.get("toolchain", {})
    header = [
        "## T1 Perf Gate — Stage 1 (Cloudtop TPU7x AOT Full LLO + Mosaic Diff)",
        f"- **Base:** `{summary['base_sha']}`",
        f"- **Head:** `{summary['head_sha']}`",
        f"- **Overall Stage 1 Verdict:** **`{summary['overall_status']}`**",
        f"- **Toolchain:** `jax={tc.get('jax')}`, `jaxlib={tc.get('jaxlib')}`, "
        f"`libtpu={tc.get('libtpu')}` (`{tc.get('libtpu_so')}`), "
        f"`topology={tc.get('topology_name')}`",
        "",
    ]

    exact_b = [b for b in summary["buckets"] if b.get("numerics") == "exact"]
    approx_b = [b for b in summary["buckets"] if b.get("numerics") != "exact"]

    def _render_table(title: str, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return f"### {title}\n\n_None._\n"
        cols = [
            "Profile",
            "T",
            "B",
            "Status",
            "Mosaic",
            "HLO",
            "VMEM (B->H MiB)",
            "MK Bundles (B->H)",
            "dBundles",
            "Base LLO SHA",
            "Head LLO SHA",
            "Changed Subkernels",
        ]
        lines = [
            f"### {title}",
            "",
            "| " + " | ".join(cols) + " |",
            "|" + "|".join(["---"] * len(cols)) + "|",
        ]
        for r in rows:
            changed_sk = [
                f"`{k}` ({v['delta_total_bundles']:+d}b, {v['changed_passes_count']}p)"
                for k, v in r.get("subkernels", {}).items()
                if not v.get("full_llo_match")
            ]
            sk_str = ", ".join(changed_sk) if changed_sk else "0 (all identical)"
            cells = [
                f"`{r['profile']}`",
                str(r["T"]),
                str(r["B"]),
                f"**{r['status']}**",
                "EQ" if r.get("mosaic_match") else "DIFF",
                "EQ" if r.get("hlo_match") else "DIFF",
                f"{r['base_vmem_mib']:.2f}->{r['head_vmem_mib']:.2f}",
                f"{r['base_primary_bundles']}->{r['head_primary_bundles']}",
                f"{r['delta_primary_bundles']:+d}",
                f"`{r['base_llo_sha']}`",
                f"`{r['head_llo_sha']}`",
                sk_str,
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        return "\n".join(lines)

    body = [
        _render_table("Exact Numerics Buckets (`numerics='exact'`)", exact_b),
        _render_table("Approx Numerics Buckets (`numerics='approx'`)", approx_b),
    ]
    return "\n".join(header + body)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="T1 Perf Gate Stage 1 runner.")
    parser.add_argument("--base-wt", required=True)
    parser.add_argument("--head-wt", required=True)
    parser.add_argument("--base-sha", default="base")
    parser.add_argument("--head-sha", default="head")
    parser.add_argument("--profiles", default=DEFAULT_GATE_PROFILES)
    parser.add_argument("--buckets", default=DEFAULT_GATE_BUCKETS)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--mode",
        choices=("auto", "refactor", "intentional"),
        default="auto",
        help="In 'refactor' mode, any Stage 1 LLO diff fails with exit code 1.",
    )
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)

    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    buckets = report.parse_bucket_list(args.buckets)
    summary = run_stage1_comparison(
        args.base_wt,
        args.head_wt,
        base_sha=args.base_sha,
        head_sha=args.head_sha,
        profiles=profiles,
        buckets=buckets,
        workers=args.workers,
        out_dir=args.out_dir,
    )
    md = format_stage1_markdown(summary)
    print(md)
    if args.out_dir:
        (pathlib.Path(args.out_dir) / "stage1_report.md").write_text(
            md, encoding="utf-8"
        )

    if summary["overall_status"] == "INVALID_TOOLCHAIN":
        return 2
    if args.mode == "refactor" and summary["overall_status"] != "EQUAL":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
