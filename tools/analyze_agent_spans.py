#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
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
"""Standalone E2E CUJ-2 Agentic Wastage & Turn Waterfall Analyzer.

Reads JSONL span logs produced by `VLLM_TPU_AGENTIC_SPAN_LOG` and generates:
1. Per-turn 7-category token attribution waterfall
2. Cross-replica `routing_loss` vs `eviction` adjudication (`session_evicted_at_ms`)
3. Per-turn DRAM recall slowdown attribution (`dram_tokens`, `dram_ms`)
4. Sub-agent fork lineage (`gen_ai.session.parent_id`)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

CATEGORIES = (
    "cache_hit",
    "new_input",
    "affinity_honored_eviction",
    "prefix_mutation_loss",
    "unreused_reply_loss",
    "routing_loss",
    "cross_session_opportunity",
)


def analyze_spans(spans: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("=" * 96)
    lines.append("CUJ-2 E2E AGENTIC SERVING WASTAGE & WATERFALL REPORT")
    lines.append("=" * 96)

    # Track last-seen replica and eviction timestamps per session for cross-replica join
    session_last_replica: dict[str, str] = {}
    session_evicted_at: dict[tuple[str, str], int] = {}

    totals = {cat: 0 for cat in CATEGORIES}
    total_padding = 0
    total_dram_tokens = 0
    total_dram_ms = 0.0

    header = (
        f"{'Turn':<6} {'Session (Parent)':<28} {'Replica':<16} "
        f"{'Hit':>5} {'New':>5} {'Evict':>6} {'Mutat':>6} {'Route':>6} "
        f"{'DRAM':>10} {'Root Cause Diagnosis'}"
    )
    lines.append(header)
    lines.append("-" * 96)

    for idx, sp in enumerate(spans, start=1):
        sid = str(sp.get("gen_ai.session.id", "stateless"))
        parent = sp.get("gen_ai.session.parent_id")
        turn_id = sp.get("gen_ai.session.turn_id", idx)
        replica = str(sp.get("gen_ai.server.replica_id", "unknown"))

        counts = {
            cat: int(sp.get(f"gen_ai.usage.input_tokens.{cat}", 0))
            for cat in CATEGORIES
        }
        for cat, val in counts.items():
            totals[cat] += val

        evicted_ms = sp.get("vllm.kv_cache.session_evicted_at_ms")
        if evicted_ms is not None:
            session_evicted_at[(replica, sid)] = int(evicted_ms)

        dram_toks = int(sp.get("vllm.kv_cache.dram_tokens", 0) or 0)
        dram_ms = float(sp.get("vllm.kv_cache.dram_ms", 0.0) or 0.0)
        pad_toks = int(sp.get("vllm.tpu.shape_padding_tokens", 0) or 0)
        total_padding += pad_toks
        total_dram_tokens += dram_toks
        total_dram_ms += dram_ms

        diag_parts: list[str] = []
        if parent:
            diag_parts.append(f"Forked from {parent}")
        if counts["prefix_mutation_loss"] > 0:
            mut_idx = sp.get("gen_ai.usage.mutation_offset_token_idx")
            mut_loss = counts["prefix_mutation_loss"]
            diag_parts.append(f"Prompt mutated at tok #{mut_idx} (-{mut_loss} toks)")
        if counts["unreused_reply_loss"] > 0:
            diag_parts.append(
                f"Previous reply not reused (-{counts['unreused_reply_loss']} toks)"
            )
        if counts["affinity_honored_eviction"] > 0:
            age = sp.get("vllm.kv_cache.time_since_eviction_ms")
            age_str = f" ({age}ms ago)" if age is not None else ""
            ev_loss = counts["affinity_honored_eviction"]
            diag_parts.append(f"HBM capacity eviction (-{ev_loss} toks{age_str})")
        if counts["routing_loss"] > 0:
            prev_rep = session_last_replica.get(sid)
            if prev_rep and (prev_rep, sid) in session_evicted_at:
                diag_parts.append(
                    f"Replica hop ({prev_rep}->{replica}), "
                    f"but {prev_rep} already evicted session"
                )
            else:
                r_loss = counts["routing_loss"]
                diag_parts.append(f"Unpinned replica hop to {replica} (-{r_loss} toks)")
        if dram_toks > 0:
            diag_parts.append(f"DRAM recall {dram_toks} toks (+{dram_ms:.1f}ms TTFT)")
        if not diag_parts:
            diag_parts.append(
                "Optimal prefix cache hit"
                if counts["cache_hit"] > 0
                else "Cold start turn"
            )

        session_last_replica[sid] = replica
        sess_label = f"{sid} ({parent})" if parent else sid
        dram_label = f"{dram_toks}t/{dram_ms:.1f}ms" if dram_toks else "-"

        evict_cnt = counts["affinity_honored_eviction"]
        mut_cnt = counts["prefix_mutation_loss"]
        lines.append(
            f"T{turn_id:<5} {sess_label[:28]:<28} {replica[:16]:<16} "
            f"{counts['cache_hit']:>5} {counts['new_input']:>5} "
            f"{evict_cnt:>6} {mut_cnt:>6} "
            f"{counts['routing_loss']:>6} {dram_label:>10} {' | '.join(diag_parts)}"
        )

    lines.append("-" * 96)
    grand_total = sum(totals.values())
    wasted = (
        totals["affinity_honored_eviction"]
        + totals["prefix_mutation_loss"]
        + totals["unreused_reply_loss"]
        + totals["routing_loss"]
        + totals["cross_session_opportunity"]
    )
    waste_pct = (100.0 * wasted / grand_total) if grand_total else 0.0
    lines.append(
        f"TOTAL INPUT TOKENS: {grand_total} | Cache Hit: {totals['cache_hit']} | "
        f"New Input: {totals['new_input']} | "
        f"Preventable Recompute Waste: {wasted} ({waste_pct:.1f}%)"
    )
    lines.append(
        f"HARDWARE OVERHEAD : TPU Shape Padding: {total_padding} toks | "
        f"DRAM Recall: {total_dram_tokens} toks ({total_dram_ms:.1f} ms)"
    )
    lines.append("=" * 96)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze CUJ-2 JSONL spans")
    parser.add_argument(
        "span_log",
        nargs="?",
        default=str(
            Path(__file__).resolve().parent / "fixtures" / "cuj2_golden_spans.jsonl"
        ),
    )
    args = parser.parse_args()
    spans = [
        json.loads(line)
        for line in Path(args.span_log).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sys.stdout.write(analyze_spans(spans) + "\n")


if __name__ == "__main__":
    main()
