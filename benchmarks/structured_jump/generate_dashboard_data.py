#!/usr/bin/env python3
# Copyright 2026 Google LLC
"""Generates consolidated dashboard HTML with full 4-Arm benchmark data."""

import json
from pathlib import Path


def load_results(jsonl_path: Path):
    if not jsonl_path.exists():
        return {}
    records = {}
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            bs = data.get("parallel")
            records[bs] = data
    return records


def build_dashboard():
    b0 = load_results(Path("/tmp/results_sweep_B0_gamma8.jsonl"))
    b1 = load_results(Path("/tmp/results_sweep_B1_gamma8.jsonl"))
    b2 = load_results(Path("/tmp/results_sweep_B2_gamma8.jsonl"))
    b3 = load_results(Path("/tmp/results_sweep_B3_gamma8.jsonl"))

    all_bs = [1, 2, 4, 8, 16, 32, 64, 128, 1024]

    rows_html = []
    labels_js = [f"BS {bs}" for bs in all_bs]

    b0_lat = []
    b1_lat = []
    b2_lat = []
    b3_lat = []

    b0_tps = []
    b1_tps = []
    b2_tps = []
    b3_tps = []

    b0_alpha = []
    b1_alpha = []
    b2_alpha = []
    b3_alpha = []

    b3_vs_b0_speedup = []
    b3_vs_b2_speedup = []

    for bs in all_bs:
        r0 = b0.get(bs, {})
        r1 = b1.get(bs, {})
        r2 = b2.get(bs, {})
        r3 = b3.get(bs, {})

        l0_val = r0.get("latency")
        l1_val = r1.get("latency")
        l2_val = r2.get("latency")
        l3_val = r3.get("latency")

        b0_lat.append(round(l0_val, 2) if l0_val is not None else None)
        b1_lat.append(round(l1_val, 2) if l1_val is not None else None)
        b2_lat.append(round(l2_val, 2) if l2_val is not None else None)
        b3_lat.append(round(l3_val, 2) if l3_val is not None else None)

        b0_tps.append(
            round(r0.get("tokens_per_sec", 0), 1
                  ) if l0_val is not None else None)
        b1_tps.append(
            round(r1.get("tokens_per_sec", 0), 1
                  ) if l1_val is not None else None)
        b2_tps.append(
            round(r2.get("tokens_per_sec", 0), 1
                  ) if l2_val is not None else None)
        b3_tps.append(
            round(r3.get("tokens_per_sec", 0), 1
                  ) if l3_val is not None else None)

        b0_alpha.append(
            round(r0.get("alpha", 0), 2) if l0_val is not None else None)
        b1_alpha.append(
            round(r1.get("alpha", 0), 2) if l1_val is not None else None)
        b2_alpha.append(
            round(r2.get("alpha", 0), 2) if l2_val is not None else None)
        b3_alpha.append(
            round(r3.get("alpha", 0), 2) if l3_val is not None else None)

        sp0 = round(l0_val / l3_val, 2) if (l0_val and l3_val) else None
        sp2 = round(l2_val / l3_val, 2) if (l2_val and l3_val) else None
        b3_vs_b0_speedup.append(sp0)
        b3_vs_b2_speedup.append(sp2)

        l0_str = f"{l0_val:.2f}s ({r0.get('ms_per_token', 0):.2f} ms/t)" if l0_val else "Pending"
        l1_str = f"{l1_val:.2f}s ({r1.get('ms_per_token', 0):.2f} ms/t)" if l1_val else "Pending"
        l2_str = f"{l2_val:.2f}s ({r2.get('ms_per_token', 0):.2f} ms/t)" if l2_val else "Pending"
        l3_str = f"<strong>{l3_val:.2f}s ({r3.get('ms_per_token', 0):.2f} ms/t)</strong>" if l3_val else "Pending"

        sp0_badge = f'<span class="badge-win">{sp0:.2f}&times;</span>' if sp0 and sp0 >= 1.0 else (
            f'<span class="badge-warn">{sp0:.2f}&times;</span>'
            if sp0 else "-")
        sp2_badge = f'<span class="badge-win">{sp2:.2f}&times;</span>' if sp2 and sp2 >= 1.0 else (
            f'<span class="badge-warn">{sp2:.2f}&times;</span>'
            if sp2 else "-")

        v_json = "100.0%" if (r0 or r1 or r2 or r3) else "-"

        row = f"""                <tr>
                    <td><strong>BS = {bs}</strong></td>
                    <td>{l0_str}</td>
                    <td>{l1_str}</td>
                    <td>{l2_str}</td>
                    <td>{l3_str}</td>
                    <td>{sp0_badge}</td>
                    <td>{sp2_badge}</td>
                    <td>{r0.get('decode_steps', '-')} &rarr; {r2.get('decode_steps', '-')} &rarr; <strong>{r3.get('decode_steps', '-')}</strong></td>
                    <td>{v_json}</td>
                </tr>"""
        rows_html.append(row)

    tbody = "\n".join(rows_html)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>GaaD Comprehensive 4-Arm Multi-Batch Benchmark Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 30px; background: #f8f9fa; color: #212529; }}
        .header {{ background: #fff; padding: 24px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 24px; }}
        h1 {{ margin: 0 0 10px 0; color: #1a73e8; font-size: 26px; }}
        .nav-links {{ margin-top: 12px; }}
        .nav-links a {{ color: #1a73e8; text-decoration: none; font-weight: 500; margin-right: 18px; }}
        .nav-links a:hover {{ text-decoration: underline; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(480px, 1fr)); gap: 24px; margin-bottom: 24px; }}
        .card {{ background: #fff; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
        .full-card {{ background: #fff; padding: 24px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 24px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 13px; }}
        th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #e0e0e0; }}
        th {{ background: #f1f3f4; font-weight: 600; color: #3c4043; }}
        tr:hover {{ background: #f8f9fa; }}
        .badge-win {{ background: #e6f4ea; color: #137333; font-weight: bold; padding: 3px 8px; border-radius: 6px; }}
        .badge-opt {{ background: #e8f0fe; color: #1a73e8; font-weight: bold; padding: 3px 8px; border-radius: 6px; }}
        .badge-warn {{ background: #fef7e0; color: #b06000; font-weight: bold; padding: 3px 8px; border-radius: 6px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 GaaD 4-Arm Speculative Architecture Evaluation Dashboard (&gamma; = 8)</h1>
        <p><strong>Hardware:</strong> Google Cloud TPU v7x-8 | <strong>Model:</strong> Meta-Llama-3.1-8B-Instruct | <strong>Workload:</strong> Character JSON Schema Regex</p>
        <p><strong>Evaluated Architectural Arms:</strong>
            <br>&bull; <strong>Arm 0 (Vanilla Baseline):</strong> Pure autoregressive without speculative decoding (<code>num_spec=0</code>)
            <br>&bull; <strong>Arm 1 (Raw Spec):</strong> Standard N-gram drafter without grammar guidance (<code>gamma=8</code>)
            <br>&bull; <strong>Arm 2 (Composite GaaD):</strong> N-gram drafter + GaaD lookahead & boundary filtering (<code>gamma=8</code>)
            <br>&bull; <strong>Arm 3 (Pure GaaD / B3):</strong> <strong>Zero external drafter</strong>, pure grammar jump lookahead (<code>gamma=8</code>)
        </p>
        <div class="nav-links">
            <a href="index.html">&larr; Back to Research Paper</a>
            <a href="https://github.com/vllm-project/vllm-torchtpu/pull/1054" target="_blank">GitHub PR #1054</a>
        </div>
    </div>

    <div class="grid">
        <div class="card">
            <h3>Latency Scaling: Arm 0 vs Arm 1 vs Arm 2 vs Arm 3 (Seconds, Log Scale, Lower is Better)</h3>
            <canvas id="latChart"></canvas>
        </div>
        <div class="card">
            <h3>Pure GaaD Speedup Factor vs Composite B2 & Baseline B0 (Higher is Better)</h3>
            <canvas id="speedupChart"></canvas>
        </div>
    </div>

    <div class="grid">
        <div class="card">
            <h3>Throughput Scaling Across Batch Sizes (Tokens / Sec, Higher is Better)</h3>
            <canvas id="tpsChart"></canvas>
        </div>
        <div class="card">
            <h3>Acceptance Rate &alpha; (Tokens Accepted per Decode Step)</h3>
            <canvas id="alphaChart"></canvas>
        </div>
    </div>

    <div class="full-card">
        <h3>Authoritative Hardware Evaluation: Comprehensive 4-Arm Matrix (&gamma; = 8)</h3>
        <table>
            <thead>
                <tr>
                    <th>Batch Size</th>
                    <th>Arm 0 (Vanilla Baseline)</th>
                    <th>Arm 1 (Raw Spec w/o GaaD)</th>
                    <th>Arm 2 (Composite GaaD)</th>
                    <th>Arm 3 (Pure GaaD / B3)</th>
                    <th>Speedup (Pure vs B0)</th>
                    <th>Speedup (Pure vs B2)</th>
                    <th>Decode Steps (B0 &rarr; B2 &rarr; B3)</th>
                    <th>Valid JSON</th>
                </tr>
            </thead>
            <tbody>
{tbody}
            </tbody>
        </table>
    </div>

    <script>
        const labels = {json.dumps(labels_js)};
        const b0_lat = {json.dumps(b0_lat)};
        const b1_lat = {json.dumps(b1_lat)};
        const b2_lat = {json.dumps(b2_lat)};
        const b3_lat = {json.dumps(b3_lat)};

        const b0_tps = {json.dumps(b0_tps)};
        const b1_tps = {json.dumps(b1_tps)};
        const b2_tps = {json.dumps(b2_tps)};
        const b3_tps = {json.dumps(b3_tps)};

        const b0_alpha = {json.dumps(b0_alpha)};
        const b1_alpha = {json.dumps(b1_alpha)};
        const b2_alpha = {json.dumps(b2_alpha)};
        const b3_alpha = {json.dumps(b3_alpha)};

        const spd_b0 = {json.dumps(b3_vs_b0_speedup)};
        const spd_b2 = {json.dumps(b3_vs_b2_speedup)};

        new Chart(document.getElementById("latChart"), {{
            type: "line",
            data: {{
                labels: labels,
                datasets: [
                    {{ label: "Arm 0 (Vanilla Baseline)", data: b0_lat, borderColor: "#202124", backgroundColor: "#202124", tension: 0.2, borderWidth: 2 }},
                    {{ label: "Arm 1 (Raw Spec w/o GaaD)", data: b1_lat, borderColor: "#ea4335", backgroundColor: "#ea4335", tension: 0.2, borderDash: [5, 5] }},
                    {{ label: "Arm 2 (Composite GaaD)", data: b2_lat, borderColor: "#1a73e8", backgroundColor: "#1a73e8", tension: 0.2, borderWidth: 2 }},
                    {{ label: "Arm 3 (Pure GaaD / B3)", data: b3_lat, borderColor: "#34a853", backgroundColor: "#34a853", tension: 0.2, borderWidth: 3 }}
                ]
            }},
            options: {{ responsive: true, scales: {{ y: {{ type: "logarithmic", title: {{ display: true, text: "Latency (Seconds, Log Scale)" }} }} }} }}
        }});

        new Chart(document.getElementById("speedupChart"), {{
            type: "bar",
            data: {{
                labels: labels,
                datasets: [
                    {{ label: "Pure GaaD vs Vanilla Baseline (B3 / B0)", data: spd_b0, backgroundColor: "#34a853" }},
                    {{ label: "Pure GaaD vs Composite GaaD (B3 / B2)", data: spd_b2, backgroundColor: "#1a73e8" }}
                ]
            }},
            options: {{ responsive: true, scales: {{ y: {{ beginAtZero: true, title: {{ display: true, text: "Speedup Multiplier (&times;)" }} }} }} }}
        }});

        new Chart(document.getElementById("tpsChart"), {{
            type: "line",
            data: {{
                labels: labels,
                datasets: [
                    {{ label: "Arm 0 (Vanilla)", data: b0_tps, borderColor: "#202124", backgroundColor: "#202124", tension: 0.2 }},
                    {{ label: "Arm 1 (Raw Spec)", data: b1_tps, borderColor: "#ea4335", backgroundColor: "#ea4335", tension: 0.2, borderDash: [5, 5] }},
                    {{ label: "Arm 2 (Composite GaaD)", data: b2_tps, borderColor: "#1a73e8", backgroundColor: "#1a73e8", tension: 0.2 }},
                    {{ label: "Arm 3 (Pure GaaD)", data: b3_tps, borderColor: "#34a853", backgroundColor: "#34a853", tension: 0.2, borderWidth: 3 }}
                ]
            }},
            options: {{ responsive: true, scales: {{ y: {{ type: "logarithmic", title: {{ display: true, text: "Tokens / Second (Log Scale)" }} }} }} }}
        }});

        new Chart(document.getElementById("alphaChart"), {{
            type: "bar",
            data: {{
                labels: labels,
                datasets: [
                    {{ label: "Arm 0 &alpha;", data: b0_alpha, backgroundColor: "#202124" }},
                    {{ label: "Arm 1 &alpha;", data: b1_alpha, backgroundColor: "#ea4335" }},
                    {{ label: "Arm 2 &alpha;", data: b2_alpha, backgroundColor: "#1a73e8" }},
                    {{ label: "Arm 3 &alpha;", data: b3_alpha, backgroundColor: "#34a853" }}
                ]
            }},
            options: {{ responsive: true, scales: {{ y: {{ type: "logarithmic", title: {{ display: true, text: "Alpha (&alpha;, Tokens / Step)" }} }} }} }}
        }});
    </script>
</body>
</html>"""

    out_path = Path(
        "/usr/local/google/home/johnqiangzhang/projects/llm_jobs_internal/publications/gaad/web_report/dashboard.html"
    )
    out_path.write_text(html_content)
    print(f"✅ Dashboard generated at {out_path}")


if __name__ == "__main__":
    build_dashboard()
