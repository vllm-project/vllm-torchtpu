# SPDX-License-Identifier: Apache-2.0
"""Greedy-decoding identity check for a static expert layout.

Starts the decode server the way the server-steps harness does (the same
launcher, TP1/DP8/EP8, DSpark K5, buckets [288, 2048]), sends a fixed set of
prompts at temperature 0 and records the completions. Run once without a
layout and once with one; the completions must be byte-identical for the
prompts sent one at a time (a single in-flight request gives both arms the
same batch composition and the same speculative acceptance). The concurrent
batch is informational: C384-style serving is not run-to-run deterministic
(batch composition and acceptance timing), so it reports a match rate only.

    greedy_identity.py --out DIR [--prompts 32] [--tokens 128] [--compare DIR_A]

Prompts: the first `--prompt-tokens` ids of the frozen decode input's windows
(scripts/dsv41_bench/server_steps, 98K-token prompts). The layout is selected
by the environment the harness uses (TPU_EXPERT_PLACEMENT=<name|path>).
"""

import argparse
import concurrent.futures
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
STEPS = HERE.parent / "server_steps"
sys.path.insert(0, str(STEPS))
from run import bootstrap_ports  # noqa: E402

BASE = "http://127.0.0.1:8000"


def _complete(prompt, tokens, seed=0):
    r = requests.post(
        f"{BASE}/v1/completions",
        json={
            "model": "deepseek-v41",
            "prompt": prompt,
            "max_tokens": tokens,
            "temperature": 0,
            "seed": seed,
            "logprobs": 0,
        },
        timeout=600,
    )
    r.raise_for_status()
    choice = r.json()["choices"][0]
    return {
        "text": choice["text"],
        "tokens": (choice.get("logprobs") or {}).get("tokens"),
        "finish": choice.get("finish_reason"),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--input",
        type=Path,
        default=Path(os.environ.get("DSV41_WORK", os.path.expanduser("~/dsv41-work")))
        / "inputs/dsv41-decode-inputs.json",
        help="the frozen decode input (inside dsv41-docker, pass the host path)",
    )
    ap.add_argument("--prompts", type=int, default=32)
    ap.add_argument("--prompt-tokens", type=int, default=2048)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--startup-timeout", type=float, default=7200)
    ap.add_argument(
        "--compare",
        type=Path,
        default=None,
        help="another run's directory to compare against",
    )
    ap.add_argument(
        "--no-server", action="store_true", help="use a server already listening"
    )
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    with open(a.input) as handle:
        data = json.load(handle)
    prompts = [row[: a.prompt_tokens] for row in data["prompts"][: a.prompts]]

    env = os.environ.copy()
    env.update(
        TP="1",
        DP="8",
        CHUNK="2048",
        MAX_SEQS="48",
        MAX_MODEL_LEN="98816",
        KV_BYTES="auto",
        HOST="127.0.0.1",
        PORT="8000",
        USE_PHASED_PROFILER="0",
    )
    env.setdefault("MOE_FUSED_EP_PREPROCESS_BACKEND", "topology")
    env["PYTHONPATH"] = str(STEPS) + os.pathsep + env.get("PYTHONPATH", "")
    server = None
    if not a.no_server:
        bootstrap_ports(env, a.out)
        cmd = [
            "bash",
            str(STEPS.parent / "serve_cached_prefill.sh"),
            "--compilation-config",
            json.dumps({"compile_sizes": [288, 2048]}),
        ]
        # The child keeps its own copy of the descriptor; ours closes here.
        with open(a.out / "server.log", "w") as log:
            server = subprocess.Popen(
                cmd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        started = time.monotonic()
        while True:
            if server.poll() is not None:
                raise SystemExit(
                    f"server exited during startup rc={server.returncode}; "
                    f"see {a.out}/server.log"
                )
            if time.monotonic() - started > a.startup_timeout:
                raise SystemExit("server not healthy before the deadline")
            try:
                if requests.get(f"{BASE}/health", timeout=2).ok:
                    break
            except requests.RequestException:
                pass
            time.sleep(5)
        print(f"server healthy after {time.monotonic() - started:.0f}s", flush=True)
    try:
        # Two warm-up requests so the first measured prompt sees a compiled path.
        for p in prompts[:2]:
            _complete(p, 8)
        t0 = time.monotonic()
        sequential = [_complete(p, a.tokens) for p in prompts]
        t1 = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as pool:
            concurrent_ = list(pool.map(lambda p: _complete(p, a.tokens), prompts))
        t2 = time.monotonic()
        result = {
            "layout": env.get("TPU_EXPERT_PLACEMENT", ""),
            "prompts": len(prompts),
            "prompt_tokens": a.prompt_tokens,
            "max_tokens": a.tokens,
            "sequential": sequential,
            "concurrent": concurrent_,
            "sequential_seconds": t1 - t0,
            "concurrent_seconds": t2 - t1,
        }
        (a.out / "identity.json").write_text(json.dumps(result, indent=1) + "\n")
        print(
            f"wrote {a.out}/identity.json: {len(prompts)} prompts, "
            f"sequential {t1 - t0:.0f}s, concurrent {t2 - t1:.0f}s"
        )
        if a.compare is not None:
            other = json.loads((a.compare / "identity.json").read_text())
            for mode in ("sequential", "concurrent"):
                same = sum(
                    x["text"] == y["text"]
                    for x, y in zip(result[mode], other[mode], strict=True)
                )
                print(
                    f"{mode}: {same}/{len(prompts)} completions identical to "
                    f"{a.compare}"
                )
            seq_same = sum(
                x["text"] == y["text"]
                for x, y in zip(result["sequential"], other["sequential"], strict=True)
            )
            if seq_same != len(prompts):
                print("SEQUENTIAL MISMATCH", file=sys.stderr)
                return 1
        return 0
    finally:
        if server is not None:
            os.killpg(os.getpgid(server.pid), signal.SIGTERM)
            try:
                server.wait(120)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(server.pid), signal.SIGKILL)


if __name__ == "__main__":
    sys.exit(main())
