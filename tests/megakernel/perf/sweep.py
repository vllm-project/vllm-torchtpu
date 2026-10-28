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
"""VMEM + Cloudtop AOT LLO prefiltered tile/ring sweep runner (`sweep.py`).

Sweeps `(nbd, nbe, nkv, gdn_rows, fused_pool_io_max_b)` across target `(T, B)` buckets:
1. Phase 1 (CPU static VMEM/SMEM filter): prunes points exceeding 62 MiB VMEM.
2. Phase 2 (Optional `--aot-top-k K` Cloudtop AOT compile): compiles top-K candidates,
   verifies 0 register spills, and ranks by scheduled bundle count and ring depth.
3. Emits a candidate `TileRule` snippet (`B`-only signature per review item 8).
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import pathlib
import sys
from collections.abc import Sequence
from typing import Any

from . import compiler_stats, report


def make_candidate_profile(
    base_profile_name: str,
    *,
    nbd: int,
    nbe: int,
    nkv: int,
    gdn_rows: int = 8,
    hier: int | None = None,
    fused_pool_io_max_b: int | None = None,
) -> Any:
    """Create a candidate `Profile` overriding `(nbd, nbe, nkv, gdn_rows, hier)`."""
    from vllm_torchtpu.megakernel.profile import Profile, get_profile

    base_prof = get_profile(base_profile_name)
    base_rule = base_prof.tiles

    def _rule(B: int, *, moe_tp: bool = False, R: int = 8):
        t = base_rule(B, moe_tp=moe_tp, R=R) if callable(base_rule) else base_rule
        kw: dict[str, Any] = {"nbd": nbd, "nbe": nbe}
        if hier is not None:
            kw["hier"] = hier
        t = dataclasses.replace(t, **kw)
        return t.with_modules(
            {
                "gqa": {"nkv": nkv},
                "gdn": {"gdn_rows": gdn_rows},
            }
        )

    tag = f"nbd{nbd}_nbe{nbe}_nkv{nkv}_gr{gdn_rows}"
    if fused_pool_io_max_b is not None:
        tag += f"_fmax{fused_pool_io_max_b}"
    return Profile(
        name=f"{base_prof.name}[{tag}]",
        model=base_prof.model,
        numerics=base_prof.numerics,
        tiles=_rule,
        buckets=base_prof.buckets,
        verify_buckets=base_prof.verify_buckets,
        fused_pool_io=base_prof.fused_pool_io,
        fused_pool_io_max_b=(
            fused_pool_io_max_b
            if fused_pool_io_max_b is not None
            else base_prof.fused_pool_io_max_b
        ),
        fused_pool_io_min_b_hi=base_prof.fused_pool_io_min_b_hi,
        pool_gather=base_prof.pool_gather,
        description=f"Sweep candidate {tag}",
    )


def sweep_bucket(
    base_profile_name: str,
    T: int,
    B: int,
    *,
    nbd_values: Sequence[int] = (7, 8, 10, 12, 14),
    nbe_values: Sequence[int] = (10, 12, 14, 16, 20),
    nkv_values: Sequence[int] = (3, 4),
    gdn_rows_values: Sequence[int] = (8,),
    fused_max_b_values: Sequence[int | None] = (None, 64),
    max_vmem_mib: float = 59.5,
    aot_top_k: int = 0,
    aot_workers: int = 2,
    stack_e: bool = True,
) -> list[dict[str, Any]]:
    """Sweep grid on `(T, B)`, filter by VMEM budget, and optionally
    AOT-compile top-K.
    """
    base_st = compiler_stats.plan_bucket_stats(base_profile_name, T, B, stack_e=stack_e)

    fitting: list[dict[str, Any]] = []
    for nbd, nbe, nkv, gdn_r, fmax in itertools.product(
        nbd_values, nbe_values, nkv_values, gdn_rows_values, fused_max_b_values
    ):
        prof = make_candidate_profile(
            base_profile_name,
            nbd=nbd,
            nbe=nbe,
            nkv=nkv,
            gdn_rows=gdn_r,
            fused_pool_io_max_b=fmax,
        )
        st = compiler_stats.plan_bucket_stats(
            prof,
            T,
            B,
            stack_e=stack_e,
            fused_pool_io_max_b=fmax,
        )
        if not st["fits_vmem"] or st["vmem_mib"] > max_vmem_mib:
            continue
        # Heuristic ring-depth score: prefer 1-launch FUSED=1, deeper expert ring nbe,
        # deeper dense ring nbd, deeper KV ring nkv
        ring_score = (
            (100.0 if st["FUSED"] else 0.0)
            + 2.0 * nbe
            + 1.5 * nbd
            + 3.0 * nkv
            + (0.5 * gdn_r)
        )
        st["candidate_params"] = {
            "nbd": nbd,
            "nbe": nbe,
            "nkv": nkv,
            "gdn_rows": gdn_r,
            "fused_pool_io_max_b": fmax,
        }
        st["delta_vmem_mib"] = round(st["vmem_mib"] - base_st["vmem_mib"], 2)
        st["ring_score"] = round(ring_score, 2)
        fitting.append(st)

    fitting.sort(key=lambda r: (-r["ring_score"], r["vmem_mib"]))

    if aot_top_k > 0 and fitting:
        top_slice = fitting[:aot_top_k]

        def _compile_one(cand: dict[str, Any]) -> dict[str, Any]:
            cp = cand["candidate_params"]
            prof = make_candidate_profile(
                base_profile_name,
                nbd=cp["nbd"],
                nbe=cp["nbe"],
                nkv=cp["nkv"],
                gdn_rows=cp["gdn_rows"],
                fused_pool_io_max_b=cp["fused_pool_io_max_b"],
            )
            compiled = compiler_stats.compile_bucket_in_process(
                prof,
                T,
                B,
                stack_e=stack_e,
                fused_pool_io_max_b=cp["fused_pool_io_max_b"],
            )
            s = report._extract_main_llo_summary(compiled)
            cand["compile_s"] = compiled.get("compile_s")
            cand["main_bundles"] = s["main_bundles"]
            cand["total_bundles_all"] = s["total_bundles_all"]
            cand["spills"] = s["spills"]
            cand["llo_hash"] = s["llo_hash"]
            return cand

        # Run sequentially in-process or via ThreadPool if requested
        for i, cand in enumerate(top_slice):
            top_slice[i] = _compile_one(cand)

        # Re-rank compiled top-K: 0 spills first, fewer total bundles across subkernels,
        # higher ring_score
        top_slice.sort(
            key=lambda r: (
                r.get("spills", 0),
                r.get("total_bundles_all", 10**9),
                -r["ring_score"],
            )
        )
        fitting[:aot_top_k] = top_slice

    return fitting


def format_sweep_markdown(
    candidates: Sequence[dict[str, Any]], *, limit: int = 15
) -> str:
    """Format top sweep candidates into Markdown tables separated by numerics class."""
    exact_c = [c for c in candidates if c.get("numerics") == "exact"][:limit]
    approx_c = [c for c in candidates if c.get("numerics") != "exact"][:limit]

    def _render(title: str, group: list[dict[str, Any]]) -> str:
        if not group:
            return f"### {title}\n\n_No matching candidates._\n"
        has_aot = any("main_bundles" in r for r in group)
        cols = [
            "Rank",
            "T",
            "B",
            "nbd",
            "nbe",
            "nkv",
            "gdn_r",
            "fmax_b",
            "FUSED/GPR/GCH",
            "VMEM (MiB)",
            "dVMEM",
            "SMEM (KiB)",
            "Ring Score",
        ]
        if has_aot:
            cols.extend(["MK Bundles", "All Bundles", "Spills", "LLO SHA"])
        lines = [
            f"### {title}",
            "",
            "| " + " | ".join(cols) + " |",
            "|" + "|".join(["---"] * len(cols)) + "|",
        ]
        for idx, r in enumerate(group, 1):
            cp = r["candidate_params"]
            fgc = f"{int(r['FUSED'])}/{int(r['GPR'])}/{int(r['GCH'])}"
            cells = [
                str(idx),
                str(r["T"]),
                str(r["B"]),
                str(cp["nbd"]),
                str(cp["nbe"]),
                str(cp["nkv"]),
                str(cp["gdn_rows"]),
                str(cp["fused_pool_io_max_b"] or "default"),
                fgc,
                f"{r['vmem_mib']:.2f}/{r['vmem_limit_mib']}",
                f"{r['delta_vmem_mib']:+.2f}",
                f"{r['smem_kib']:.1f}",
                f"{r['ring_score']:.1f}",
            ]
            if has_aot:
                cells.extend(
                    [
                        str(r.get("main_bundles", "-")),
                        str(r.get("total_bundles_all", "-")),
                        str(r.get("spills", "-")),
                        f"`{r.get('llo_hash', '-')}`",
                    ]
                )
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        return "\n".join(lines)

    return "\n".join(
        [
            _render("Exact Numerics Candidates (`numerics='exact'`)", exact_c),
            _render("Approx Numerics Candidates (`numerics='approx'`)", approx_c),
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep megakernel ring depths and fused_pool_io_max_b "
            "under VMEM/LLO constraints."
        )
    )
    parser.add_argument("--profile", default="qwen35-candidate30-fused")
    parser.add_argument("--buckets", default="4:24,4:32,4:40,4:48,4:56,4:64")
    parser.add_argument("--nbd", default="7,8,10,12,14")
    parser.add_argument("--nbe", default="12,14,16,20")
    parser.add_argument("--nkv", default="3,4")
    parser.add_argument("--gdn-rows", default="8")
    parser.add_argument("--fused-max-b", default="default,64")
    parser.add_argument("--max-vmem-mib", type=float, default=59.5)
    parser.add_argument("--aot-top-k", type=int, default=0)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--md-out", default=None)
    args = parser.parse_args(argv)

    buckets = report.parse_bucket_list(args.buckets)
    nbd_vals = [int(x) for x in args.nbd.split(",") if x.strip()]
    nbe_vals = [int(x) for x in args.nbe.split(",") if x.strip()]
    nkv_vals = [int(x) for x in args.nkv.split(",") if x.strip()]
    gr_vals = [int(x) for x in args.gdn_rows.split(",") if x.strip()]
    fmax_vals = [
        None if x.strip() in ("default", "none", "None") else int(x.strip())
        for x in args.fused_max_b.split(",")
        if x.strip()
    ]

    all_rows: list[dict[str, Any]] = []
    md_sections: list[str] = []
    for T, B in buckets:
        cands = sweep_bucket(
            args.profile,
            T,
            B,
            nbd_values=nbd_vals,
            nbe_values=nbe_vals,
            nkv_values=nkv_vals,
            gdn_rows_values=gr_vals,
            fused_max_b_values=fmax_vals,
            max_vmem_mib=args.max_vmem_mib,
            aot_top_k=args.aot_top_k,
        )
        all_rows.extend(cands[: args.limit])
        md_sections.append(
            f"## Bucket `T={T}, B={B}`\n\n"
            + format_sweep_markdown(cands, limit=args.limit)
        )

    full_md = "\n".join(md_sections)
    print(full_md)
    if args.md_out:
        pathlib.Path(args.md_out).write_text(full_md)
    if args.json_out:
        pathlib.Path(args.json_out).write_text(json.dumps(all_rows, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
