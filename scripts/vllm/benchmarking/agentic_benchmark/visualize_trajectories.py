#!/usr/bin/env python3
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
"""Interactive trajectory and turn waterfall visualizer for agentic RL benchmarks.

This tool renders an interactive waterfall timeline displaying multi-turn trajectories,
turn-by-turn latency segmentation (model generation vs. environment/tool idle time),
batch launch boundaries, per-turn metrics (prompt/gen tokens, TTFT, TPOT, tok/s),
and a multi-run comparison dropdown to switch seamlessly between benchmark runs
(e.g., comparing different trace files or engine flags such as DP_SCHED_BATCH_PREFILL).

Input formats supported:
1. One or more trajectory JSONL files from benchmark_agentic.py
   (--save-trajectory-file).
2. Responses JSONL from benchmark_agentic.py (--save-responses-file).
3. Distributed RL inference_metrics.jsonl files.
4. Raw log files with [DEBUG_INFERENCE][Turn] and [DEBUG_INFERENCE][Trajectory] lines.
5. Built-in --demo mode generating realistic sample data across multiple runs.
"""

import argparse
import http.server
import json
import os
import re
import socketserver
import webbrowser
from typing import Any

_TEMPLATE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "waterfall_template.html"
)


def generate_single_demo_dataset(
    seed: int = 42,
    num_batches: int = 3,
    concurrency: int = 4,
    group_size: int = 4,
    time_between_batches: float = 20.0,
    has_tool_time: bool = True,
    model_factor: float = 1.0,
    run_name: str = "Demo Run",
    dp_sched_prefill: bool = False,
) -> dict[str, Any]:
    """Generates a synthetic multi-batch agentic benchmark trajectory dataset."""
    import random

    rng = random.Random(seed)

    batch_launch_times = [
        round(b * time_between_batches, 3) for b in range(num_batches)
    ]
    trajectories = []
    turns_all = []
    traj_idx = 0

    for b in range(num_batches):
        b_launch = batch_launch_times[b]
        for g in range(1, concurrency + 1):
            g_idx = b * concurrency + g
            prompt_len = rng.randint(4000, 8000)
            for s in range(group_size):
                traj_idx += 1
                traj_id = f"b{b}_g{g_idx}_s{s}"
                num_turns = rng.randint(6, 12)
                t_cur = b_launch + rng.uniform(0.05, 0.4)
                traj_turns = []
                tot_out = 0
                tot_model = 0.0
                tot_tool = 0.0
                t_start = t_cur

                for turn in range(1, num_turns + 1):
                    # Turn 1 prefill is longer on cache miss
                    is_miss = s == 0 and turn == 1
                    base_ttft = (
                        rng.uniform(0.4, 0.8) if is_miss else rng.uniform(0.05, 0.15)
                    )
                    if dp_sched_prefill and turn > 1:
                        # DP batch prefill batches prefills together,
                        # reducing tail TTFT under load
                        base_ttft *= 0.82
                    ttft_s = base_ttft * model_factor

                    gen_toks = rng.randint(150, 450)
                    tpot_ms = rng.uniform(4.5, 7.5) * model_factor
                    decode_s = (gen_toks - 1) * (tpot_ms / 1000.0)
                    model_time_s = round(ttft_s + decode_s, 3)

                    if has_tool_time and turn < num_turns:
                        tool_time_s = round(rng.uniform(0.8, 3.2), 3)
                    else:
                        tool_time_s = 0.0

                    turn_start = round(t_cur, 3)
                    turn_end = round(turn_start + model_time_s, 3)
                    tot_out += gen_toks
                    tot_model += model_time_s
                    tot_tool += tool_time_s

                    t_rec = {
                        "type": "turn",
                        "batch_idx": b,
                        "group_idx": g_idx,
                        "stream_idx": s,
                        "traj_id": traj_id,
                        "turn": turn,
                        "num_turns": num_turns,
                        "start_time_s": turn_start,
                        "end_time_s": turn_end,
                        "model_time_s": model_time_s,
                        "tool_time_s": tool_time_s,
                        "prompt_tokens": prompt_len + (turn - 1) * 200,
                        "output_tokens": gen_toks,
                        "ttft_ms": round(ttft_s * 1000.0, 1),
                        "tpot_ms": round(tpot_ms, 2),
                        "total_time_ms": round(model_time_s * 1000.0, 1),
                        "success": True,
                    }
                    traj_turns.append(t_rec)
                    turns_all.append(t_rec)
                    t_cur = turn_end + tool_time_s

                traj_rec = {
                    "type": "trajectory",
                    "traj_id": traj_id,
                    "batch_idx": b,
                    "group_idx": g_idx,
                    "stream_idx": s,
                    "num_turns": num_turns,
                    "start_time_s": round(t_start, 3),
                    "end_time_s": round(t_cur, 3),
                    "duration_s": round(t_cur - t_start, 3),
                    "model_time_s": round(tot_model, 3),
                    "tool_time_s": round(tot_tool, 3),
                    "prompt_tokens": prompt_len,
                    "output_tokens": tot_out,
                    "status": "COMPLETED",
                }
                trajectories.append(traj_rec)

    total_duration = max((t["end_time_s"] for t in trajectories), default=1.0)
    meta = {
        "run_name": run_name,
        "model": "Qwen/Qwen3.5-397B-A17B-FP8",
        "num_batches": num_batches,
        "concurrency": concurrency,
        "group_size": group_size,
        "time_between_batches_sec": time_between_batches,
        "batch_launch_times": batch_launch_times,
        "total_trajectories": len(trajectories),
        "total_turns": len(turns_all),
        "total_duration_sec": total_duration,
        "dp_sched_batch_prefill": dp_sched_prefill,
        "trace_file": "gs://wenxindong-vm/rl/mlperf2026/agentic_benchmark/gbs1024_trace_file_tool_time.jsonl"
        if has_tool_time
        else "gbs1024_trace_file.jsonl",
    }
    return {
        "meta": meta,
        "trajectories": trajectories,
        "turns": turns_all,
    }


def generate_demo_runs() -> dict[str, dict[str, Any]]:
    """Generates synthetic runs comparing different benchmark configurations."""
    return {
        "Demo: Tool Time (DP_SCHED=false)": generate_single_demo_dataset(
            seed=42,
            has_tool_time=True,
            dp_sched_prefill=False,
            model_factor=1.0,
            run_name="Demo: Tool Time (DP_SCHED=false)",
        ),
        "Demo: Tool Time (DP_SCHED=true)": generate_single_demo_dataset(
            seed=43,
            has_tool_time=True,
            dp_sched_prefill=True,
            model_factor=0.92,
            run_name="Demo: Tool Time (DP_SCHED=true)",
        ),
        "Demo: Baseline (No Tool Time)": generate_single_demo_dataset(
            seed=44,
            has_tool_time=False,
            dp_sched_prefill=False,
            model_factor=1.0,
            run_name="Demo: Baseline (No Tool Time)",
        ),
    }


def parse_metrics_file(path: str, run_name: str | None = None) -> dict[str, Any]:
    """Parses a trajectory metrics JSONL or text log file into structured records."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input metrics file not found: {path}")

    trajectories: dict[str, dict[str, Any]] = {}
    turns: list[dict[str, Any]] = []
    base_name = run_name or os.path.splitext(os.path.basename(path))[0]
    meta: dict[str, Any] = {
        "source_file": os.path.abspath(path),
        "run_name": base_name,
        "batch_launch_times": [],
    }

    with open(path, encoding="utf-8", errors="replace") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Format 1: JSONL file
    is_jsonl = False
    for line in lines[:10]:
        if line.startswith("{") and line.endswith("}"):
            is_jsonl = True
            break

    if is_jsonl:
        for line in lines:
            try:
                rec = json.loads(line)
            except Exception:
                continue

            r_type = rec.get("type")
            if r_type == "benchmark_meta":
                meta.update(rec)
            elif r_type == "trajectory":
                traj_id = rec.get("traj_id")
                if traj_id:
                    trajectories[traj_id] = rec
            elif r_type == "turn" or ("turn" in rec and "group_idx" in rec):
                # Turn record
                b_idx = rec.get("batch_idx", 0)
                g_idx = rec.get("group_idx", 0)
                s_idx = rec.get("stream_idx", 0)
                traj_id = rec.get("traj_id") or f"b{b_idx}_g{g_idx}_s{s_idx}"
                t_num = rec.get("turn", 1)
                num_turns = rec.get("num_turns", 1)
                model_time = rec.get("model_time_s")
                if model_time is None and "total_time_ms" in rec:
                    model_time = round(rec["total_time_ms"] / 1000.0, 4)

                turn_rec = {
                    "type": "turn",
                    "batch_idx": b_idx,
                    "group_idx": g_idx,
                    "stream_idx": s_idx,
                    "traj_id": traj_id,
                    "turn": t_num,
                    "num_turns": num_turns,
                    "start_time_s": rec.get("start_time_s", 0.0),
                    "end_time_s": rec.get("end_time_s", model_time or 0.0),
                    "model_time_s": model_time or 0.0,
                    "tool_time_s": rec.get("tool_time_s", 0.0),
                    "prompt_tokens": rec.get("input_history_tokens")
                    or rec.get("prompt_tokens", 0),
                    "output_tokens": rec.get("output_tokens")
                    or rec.get("completion_tokens", 0),
                    "ttft_ms": rec.get("ttft_ms"),
                    "tpot_ms": rec.get("tpot_ms"),
                    "total_time_ms": rec.get("total_time_ms"),
                    "success": rec.get("success", True),
                    "error": rec.get("error"),
                }
                turns.append(turn_rec)

    # Format 2: Fallback parser for [DEBUG_INFERENCE] text logs
    if not turns:
        turn_re = re.compile(
            r"\[DEBUG_INFERENCE\]\[Turn\]\s+(?:prompt_id=(?P<pid>[^\s,]+)|traj_id=(?P<tid>[^\s,]+)).*?"
            r"step_index=(?P<step>\d+).*?"
            r"(?:model_time_sec=(?P<mtime>[\d\.]+)|latency=(?P<lat>[\d\.]+)).*?"
            r"(?:prompt_tokens=(?P<ptok>\d+))?.*?"
            r"(?:completion_tokens=(?P<ctok>\d+))?",
            re.IGNORECASE,
        )
        for line in lines:
            m = turn_re.search(line)
            if m:
                tid = m.group("tid") or m.group("pid") or "traj_0"
                step = int(m.group("step"))
                mtime = float(m.group("mtime") or m.group("lat") or 0.0)
                ptok = int(m.group("ptok") or 0)
                ctok = int(m.group("ctok") or 0)
                turns.append(
                    {
                        "type": "turn",
                        "batch_idx": 0,
                        "group_idx": 0,
                        "stream_idx": 0,
                        "traj_id": tid,
                        "turn": step + 1,
                        "num_turns": step + 1,
                        "start_time_s": 0.0,
                        "end_time_s": mtime,
                        "model_time_s": mtime,
                        "tool_time_s": 0.0,
                        "prompt_tokens": ptok,
                        "output_tokens": ctok,
                        "success": True,
                    }
                )

    # Synthesize trajectory summary records if missing
    traj_turns_map: dict[str, list[dict[str, Any]]] = {}
    for t in turns:
        traj_turns_map.setdefault(t["traj_id"], []).append(t)

    for traj_id, t_list in traj_turns_map.items():
        if traj_id not in trajectories:
            first = t_list[0]
            t_start = min((x["start_time_s"] for x in t_list), default=0.0)
            t_end = max((x["end_time_s"] for x in t_list), default=0.0)
            tot_model = sum(x["model_time_s"] for x in t_list)
            tot_tool = sum(x.get("tool_time_s", 0.0) for x in t_list)
            tot_out = sum(x.get("output_tokens", 0) for x in t_list)
            b_idx = first.get("batch_idx", 0)
            trajectories[traj_id] = {
                "type": "trajectory",
                "traj_id": traj_id,
                "batch_idx": b_idx,
                "group_idx": first.get("group_idx", 0),
                "stream_idx": first.get("stream_idx", 0),
                "num_turns": len(t_list),
                "start_time_s": round(t_start, 4),
                "end_time_s": round(t_end, 4),
                "duration_s": round(max(0.0, t_end - t_start), 4),
                "model_time_s": round(tot_model, 4),
                "tool_time_s": round(tot_tool, 4),
                "prompt_tokens": first.get("prompt_tokens", 0),
                "output_tokens": tot_out,
                "status": "COMPLETED"
                if all(x.get("success", False) for x in t_list)
                else "FAILED",
            }

    # Discover batch launch times if not explicitly recorded
    batches_seen = sorted(set(t.get("batch_idx", 0) for t in trajectories.values()))
    if len(batches_seen) > 1 and len(meta.get("batch_launch_times", [])) <= 1:
        launch_times = []
        for b in batches_seen:
            b_trajs = [tr for tr in trajectories.values() if tr.get("batch_idx") == b]
            launch_times.append(
                min((tr.get("start_time_s", 0.0) for tr in b_trajs), default=0.0)
            )
        meta["batch_launch_times"] = launch_times
        meta["num_batches"] = len(batches_seen)

    total_duration = max(
        (tr.get("end_time_s", 0.0) for tr in trajectories.values()), default=1.0
    )
    meta["total_duration_sec"] = round(total_duration, 3)
    meta["total_trajectories"] = len(trajectories)
    meta["total_turns"] = len(turns)

    return {
        "meta": meta,
        "trajectories": list(trajectories.values()),
        "turns": turns,
    }


def generate_html(
    runs_data: dict[str, Any] | dict[str, dict[str, Any]],
    title: str = "Agentic RL Trajectory Turn Waterfall",
) -> str:
    """Generates a standalone, dependency-free HTML waterfall visualization."""
    if "trajectories" in runs_data and "turns" in runs_data:
        single_name = runs_data.get("meta", {}).get("run_name") or "Default Run"
        all_runs: dict[str, dict[str, Any]] = {single_name: runs_data}
    else:
        all_runs = runs_data  # type: ignore

    runs_json = json.dumps(all_runs)

    with open(_TEMPLATE_PATH, encoding="utf-8") as f:
        html_template = f.read()
    return html_template.replace("__TITLE__", title).replace("__RUNS_JSON__", runs_json)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize multi-turn agentic RL trajectories and turn latencies "
            "across one or more benchmark runs."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        default=[],
        help=(
            "One or more input trajectory metrics JSONL files, or "
            "'Run Label=path/to/file.jsonl'."
        ),
    )
    parser.add_argument(
        "--run",
        "-r",
        action="append",
        dest="explicit_runs",
        default=[],
        help=(
            "Explicitly named run in format 'Run Label=path/to/file.jsonl' "
            "(can be specified multiple times)."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="trajectory_waterfall.html",
        help="Output HTML file path (default: trajectory_waterfall.html).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Agentic RL Trajectory Turn Waterfall",
        help="Title displayed in the visualization report.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help=(
            "Generate synthetic multi-run demo data to preview the "
            "comparison visualization."
        ),
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start a local HTTP server to view the generated visualization.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for local HTTP server (default: 8080).",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the generated HTML report in the default web browser.",
    )

    args = parser.parse_args()

    runs_data: dict[str, dict[str, Any]] = {}

    # 1. Parse explicitly named runs via -r / --run
    for item in args.explicit_runs:
        if "=" in item:
            name, path = item.rsplit("=", 1)
        else:
            name, path = os.path.splitext(os.path.basename(item))[0], item
        print(f"Loading named run '{name}' from: {path}")
        runs_data[name] = parse_metrics_file(path, run_name=name)

    # 2. Parse positional inputs
    for item in args.inputs:
        if "=" in item:
            name, path = item.rsplit("=", 1)
        else:
            name, path = os.path.splitext(os.path.basename(item))[0], item
        print(f"Loading run '{name}' from: {path}")
        runs_data[name] = parse_metrics_file(path, run_name=name)

    # 3. Fallback to demo mode if no inputs provided or --demo requested
    if args.demo or not runs_data:
        if not runs_data and not args.demo:
            print(
                "No input files provided. Generating interactive multi-run "
                "demo data (--demo)..."
            )
        demo_runs = generate_demo_runs()
        runs_data.update(demo_runs)

    html_content = generate_html(runs_data, title=args.title)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(
        f"Generated trajectory waterfall visualization: {os.path.abspath(args.output)}"
    )
    print(f"Total runs embedded: {len(runs_data)} -> {list(runs_data.keys())}")
    for name, rdata in runs_data.items():
        print(
            f"  - [{name}]: {len(rdata['trajectories'])} trajectories, "
            f"{len(rdata['turns'])} turns"
        )

    if args.open:
        try:
            webbrowser.open(f"file://{os.path.abspath(args.output)}")
        except webbrowser.Error as e:
            print(f"Could not open a browser: {e}")

    if args.serve:
        out_dir = os.path.dirname(os.path.abspath(args.output)) or "."
        out_file = os.path.basename(args.output)
        os.chdir(out_dir)
        handler = http.server.SimpleHTTPRequestHandler
        print(f"Serving visualization at http://localhost:{args.port}/{out_file}")
        with socketserver.TCPServer(("", args.port), handler) as httpd:
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\nServer stopped.")


if __name__ == "__main__":
    main()
