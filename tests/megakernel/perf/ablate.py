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
"""Per-flag leave-one-out and add-one-in ablation runner (`ablate.py`).

Evaluates each opt-in performance flag in `qwen35-candidate30-fused` (leave-one-out)
or `qwen35-baseline` (add-one-in), reporting VMEM/SMEM deltas, numerics classification
(`exact` vs `approx`), and optional Cloudtop TPU7x AOT LLO bundle deltas.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
from collections.abc import Sequence
from typing import Any

from . import compiler_stats, report

# Canonical list of opt-in candidate30 flags with their module scope, default (off)
# value,
# candidate30 (on) value, and numerics class.
ABLATION_FLAGS: tuple[dict[str, Any], ...] = (
    {
        "id": "tiles.dense_acc_pair",
        "scope": "core",
        "field": "dense_acc_pair",
        "off": False,
        "on": True,
        "numerics": "exact",
    },
    {
        "id": "tiles.dmm_unroll",
        "scope": "core",
        "field": "dmm_unroll",
        "off": 0,
        "on": 8,
        "numerics": "exact",
    },
    {
        "id": "tiles.rel_fetch",
        "scope": "core",
        "field": "rel_fetch",
        "off": False,
        "on": True,
        "numerics": "exact",
    },
    {
        "id": "tiles.gqa.kv_sub",
        "scope": "gqa",
        "field": "kv_sub",
        "off": 0,
        "on": 1024,
        "numerics": "approx",
    },
    {
        "id": "tiles.gqa.kv_ering",
        "scope": "gqa",
        "field": "kv_ering",
        "off": False,
        "on": True,
        "numerics": "exact",
    },
    {
        "id": "tiles.gqa.pair_kv",
        "scope": "gqa",
        "field": "pair_kv",
        "off": False,
        "on": True,
        "numerics": "exact",
    },
    {
        "id": "tiles.gqa.gq_pack",
        "scope": "gqa",
        "field": "gq_pack",
        "off": False,
        "on": True,
        "numerics": "exact",
    },
    {
        "id": "tiles.gqa.gqa_qv",
        "scope": "gqa",
        "field": "gqa_qv",
        "off": False,
        "on": True,
        "numerics": "exact",
    },
    {
        "id": "tiles.gdn.gdn_head_major",
        "scope": "gdn",
        "field": "gdn_head_major",
        "off": False,
        "on": True,
        "numerics": "exact",
    },
    {
        "id": "tiles.gdn.conv_stage",
        "scope": "gdn",
        "field": "conv_stage",
        "off": False,
        "on": True,
        "numerics": "exact",
    },
    {
        "id": "tiles.moe.ecopy_opt",
        "scope": "moe",
        "field": "ecopy_opt",
        "off": 0,
        "on": 1,
        "numerics": "exact",
    },
    {
        "id": "tiles.moe.pipe_ep",
        "scope": "moe",
        "field": "pipe_ep",
        "off": False,
        "on": True,
        "numerics": "exact",
    },
    {
        "id": "tiles.moe.r2_direct_store",
        "scope": "moe",
        "field": "r2_direct_store",
        "off": False,
        "on": True,
        "numerics": "exact",
    },
    {
        "id": "tiles.moe.route_vec",
        "scope": "moe",
        "field": "route_vec",
        "off": False,
        "on": True,
        "numerics": "approx",
    },
    {
        "id": "tiles.moe.topk_lanes",
        "scope": "moe",
        "field": "topk_lanes",
        "off": False,
        "on": True,
        "numerics": "exact",
    },
    {
        "id": "tiles.moe.moe_unroll",
        "scope": "moe",
        "field": "moe_unroll",
        "off": False,
        "on": True,
        "numerics": "exact",
    },
    {
        "id": "fused_pool_io",
        "scope": "profile",
        "field": "fused_pool_io",
        "off": False,
        "on": "auto",
        "numerics": "exact",
    },
    {
        "id": "fused_pool_io_all_b",
        "scope": "profile",
        "field": "fused_pool_io_max_b",
        "off": 24,
        "on": 64,
        "numerics": "exact",
    },
)


def make_ablated_profile(
    base_profile_name: str,
    flag_spec: dict[str, Any],
    *,
    mode: str = "leave-one-out",
) -> Any:
    """Construct a synthetic `Profile` with one flag toggled according to `mode`."""
    from vllm_torchtpu.megakernel.profile import Profile, get_profile

    base_prof = get_profile(base_profile_name)
    target_val = flag_spec["off"] if mode == "leave-one-out" else flag_spec["on"]
    scope = flag_spec["scope"]
    field = flag_spec["field"]
    flag_id = flag_spec["id"]

    # Determine effective numerics of the ablated profile:
    if mode == "add-one-in":
        eff_numerics = (
            "approx"
            if (base_prof.numerics == "approx" or flag_spec["numerics"] == "approx")
            else "exact"
        )
    else:
        # In leave-one-out from candidate30, both kv_sub and route_vec are approx,
        # so removing only one still leaves the other approx flag active.
        eff_numerics = base_prof.numerics

    if scope == "profile":
        if field == "fused_pool_io":
            return dataclasses.replace(
                base_prof,
                name=f"{base_prof.name}[{flag_id}={target_val}]",
                numerics=eff_numerics,
                fused_pool_io=target_val,
            )
        if field == "fused_pool_io_max_b":
            return dataclasses.replace(
                base_prof,
                name=f"{base_prof.name}[max_b={target_val}]",
                numerics=eff_numerics,
                fused_pool_io="auto",
                fused_pool_io_max_b=int(target_val),
            )

    base_rule = base_prof.tiles

    def _wrapped_tiles(B: int, *, moe_tp: bool = False, R: int = 8):
        t = base_rule(B, moe_tp=moe_tp, R=R) if callable(base_rule) else base_rule
        if scope == "core":
            return dataclasses.replace(t, **{field: target_val})
        return t.with_modules({scope: {field: target_val}})

    return Profile(
        name=f"{base_prof.name}[{'-' if mode == 'leave-one-out' else '+'}{flag_id}]",
        model=base_prof.model,
        numerics=eff_numerics,
        tiles=_wrapped_tiles,
        buckets=base_prof.buckets,
        verify_buckets=base_prof.verify_buckets,
        fused_pool_io=base_prof.fused_pool_io,
        fused_pool_io_max_b=base_prof.fused_pool_io_max_b,
        fused_pool_io_min_b_hi=base_prof.fused_pool_io_min_b_hi,
        pool_gather=base_prof.pool_gather,
        description=f"Ablation ({mode}) on {flag_id}",
    )


def run_ablation_table(
    *,
    mode: str = "leave-one-out",
    buckets: Sequence[tuple[int, int]] = ((1, 8), (4, 8), (4, 32), (4, 64)),
    flags: Sequence[str] | None = None,
    aot_compile: bool = False,
    stack_e: bool = True,
) -> list[dict[str, Any]]:
    """Run leave-one-out or add-one-in ablations across `buckets`."""
    base_name = (
        "qwen35-candidate30-fused" if mode == "leave-one-out" else "qwen35-baseline"
    )
    selected_flags = [
        f
        for f in ABLATION_FLAGS
        if (not flags or f["id"] in flags or f["field"] in flags)
    ]

    # Collect baseline stats per bucket first
    base_by_tb: dict[tuple[int, int], dict[str, Any]] = {}
    for T, B in buckets:
        if aot_compile:
            st = compiler_stats.compile_bucket_in_process(
                base_name, T, B, stack_e=stack_e
            )
        else:
            st = compiler_stats.plan_bucket_stats(base_name, T, B, stack_e=stack_e)
        base_by_tb[(T, B)] = st

    results: list[dict[str, Any]] = []
    for fspec in selected_flags:
        prof = make_ablated_profile(base_name, fspec, mode=mode)
        for T, B in buckets:
            base_st = base_by_tb[(T, B)]
            if aot_compile:
                st = compiler_stats.compile_bucket_in_process(
                    prof, T, B, stack_e=stack_e
                )
            else:
                st = compiler_stats.plan_bucket_stats(prof, T, B, stack_e=stack_e)
            dvmem = round(st["vmem_mib"] - base_st["vmem_mib"], 2)
            dsmem = round(st["smem_kib"] - base_st["smem_kib"], 1)
            row: dict[str, Any] = {
                "mode": mode,
                "base_profile": base_name,
                "flag_id": fspec["id"],
                "flag_numerics": fspec["numerics"],
                "variant_numerics": st["numerics"],
                "T": T,
                "B": B,
                "fits_vmem": st["fits_vmem"],
                "vmem_mib": st["vmem_mib"],
                "base_vmem_mib": base_st["vmem_mib"],
                "delta_vmem_mib": dvmem,
                "smem_kib": st["smem_kib"],
                "delta_smem_kib": dsmem,
                "FUSED": st["FUSED"],
                "GPR": st["GPR"],
                "GCH": st["GCH"],
            }
            if aot_compile and "llo" in st and "llo" in base_st:
                s_cur = report._extract_main_llo_summary(st)
                s_base = report._extract_main_llo_summary(base_st)
                row["main_bundles"] = s_cur["main_bundles"]
                row["base_main_bundles"] = s_base["main_bundles"]
                row["delta_main_bundles"] = (s_cur["main_bundles"] or 0) - (
                    s_base["main_bundles"] or 0
                )
                row["total_bundles_all"] = s_cur["total_bundles_all"]
                row["delta_total_bundles_all"] = (s_cur["total_bundles_all"] or 0) - (
                    s_base["total_bundles_all"] or 0
                )
                row["spills"] = s_cur["spills"]
                row["llo_hash"] = s_cur["llo_hash"]
                row["compile_s"] = st.get("compile_s")
            results.append(row)
    return results


def format_ablation_markdown(
    rows: Sequence[dict[str, Any]], *, include_aot: bool = False
) -> str:
    """Format ablation rows into two Markdown tables separated by flag numerics
    class.
    """
    exact_rows = [r for r in rows if r["flag_numerics"] == "exact"]
    approx_rows = [r for r in rows if r["flag_numerics"] != "exact"]

    def _render(title: str, group: list[dict[str, Any]]) -> str:
        if not group:
            return f"### {title}\n\n_None._\n"
        cols = [
            "Flag",
            "Mode",
            "Flag Numerics",
            "Profile Numerics",
            "T",
            "B",
            "Fits",
            "VMEM (MiB)",
            "dVMEM (MiB)",
            "SMEM (KiB)",
            "dSMEM (KiB)",
            "FUSED/GPR/GCH",
        ]
        if include_aot:
            cols.extend(
                [
                    "MK Bundles",
                    "dMK Bundles",
                    "All Bundles",
                    "dAll Bundles",
                    "Spills",
                    "LLO SHA",
                ]
            )
        lines = [
            f"### {title}",
            "",
            "| " + " | ".join(cols) + " |",
            "|" + "|".join(["---"] * len(cols)) + "|",
        ]
        for r in group:
            fgc = f"{int(r['FUSED'])}/{int(r['GPR'])}/{int(r['GCH'])}"
            cells = [
                f"`{r['flag_id']}`",
                r["mode"],
                f"`{r['flag_numerics']}`",
                f"`{r['variant_numerics']}`",
                str(r["T"]),
                str(r["B"]),
                "YES" if r["fits_vmem"] else "NO",
                f"{r['vmem_mib']:.2f}",
                f"{r['delta_vmem_mib']:+.2f}",
                f"{r['smem_kib']:.1f}",
                f"{r['delta_smem_kib']:+.1f}",
                fgc,
            ]
            if include_aot:
                cells.extend(
                    [
                        str(r.get("main_bundles", "-")),
                        f"{r.get('delta_main_bundles', 0):+d}"
                        if "delta_main_bundles" in r
                        else "-",
                        str(r.get("total_bundles_all", "-")),
                        f"{r.get('delta_total_bundles_all', 0):+d}"
                        if "delta_total_bundles_all" in r
                        else "-",
                        str(r.get("spills", "-")),
                        f"`{r.get('llo_hash', '-')}`",
                    ]
                )
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        return "\n".join(lines)

    return "\n".join(
        [
            _render(
                "Exact-Numerics Flag Ablations (`flag_numerics='exact'`)", exact_rows
            ),
            _render(
                "Approx-Numerics Flag Ablations (`flag_numerics='approx'`)", approx_rows
            ),
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run leave-one-out or add-one-in per-flag ablations."
    )
    parser.add_argument(
        "--mode",
        choices=("leave-one-out", "add-one-in"),
        default="leave-one-out",
    )
    parser.add_argument(
        "--buckets",
        default="1:8,4:8,4:32,4:64",
        help="Comma-separated T:B bucket pairs.",
    )
    parser.add_argument(
        "--flags",
        default="",
        help="Optional comma-separated subset of flag IDs or field names.",
    )
    parser.add_argument("--aot-compile", action="store_true")
    parser.add_argument("--stack-e", action="store_true", default=True)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--md-out", default=None)
    args = parser.parse_args(argv)

    buckets = report.parse_bucket_list(args.buckets)
    flags = [f.strip() for f in args.flags.split(",") if f.strip()] or None
    rows = run_ablation_table(
        mode=args.mode,
        buckets=buckets,
        flags=flags,
        aot_compile=args.aot_compile,
        stack_e=args.stack_e,
    )
    md = format_ablation_markdown(rows, include_aot=args.aot_compile)
    print(md)
    if args.md_out:
        pathlib.Path(args.md_out).write_text(md)
    if args.json_out:
        pathlib.Path(args.json_out).write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
