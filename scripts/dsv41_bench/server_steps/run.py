# SPDX-License-Identifier: Apache-2.0
"""Run an isolated server with audited startup and natural serving measurements."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests
from bench_cached_extend import assert_no_compilation, summarize_startup


def cache_inventory(root: Path) -> dict:
    files = [p for p in root.rglob("*") if p.is_file()]
    return {
        "files": len(files),
        "bytes": sum(p.stat().st_size for p in files),
    }


def stop(process: subprocess.Popen | None) -> None:
    if process is None:
        return
    # Descendants can retain the log pipe even after the launcher exits.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--cache-mode", choices=["cold", "reuse"], required=True)
    parser.add_argument("--startup-timeout", type=float, default=3000)
    parser.add_argument(
        "--scenario", choices=["encoder_extend", "decode"], default="encoder_extend"
    )
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.rounds is None:
        args.rounds = 1 if args.scenario == "decode" else 5
    if args.rounds < 1:
        parser.error("The integrated launcher requires at least one formal window")
    if args.profile and args.scenario == "decode":
        parser.error("Decode profiling must be captured separately from timing")

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    args.out.mkdir(parents=True, exist_ok=False)
    before = cache_inventory(args.cache_dir)
    if args.cache_mode == "cold" and before["files"]:
        parser.error("Cold startup requires a new, empty cache directory")
    if args.cache_mode == "reuse" and not before["files"]:
        parser.error("Reuse startup requires a populated cache directory")
    try:
        response = requests.get("http://127.0.0.1:8000/health", timeout=2)
    except requests.RequestException:
        pass
    else:
        if response.ok:
            parser.error("Port 8000 already has a healthy server")

    here = Path(__file__).resolve().parent
    env = os.environ.copy()
    for key, subdir in [
        ("VLLM_CACHE_ROOT", "vllm"),
        ("JAX_COMPILATION_CACHE_DIR", "jax"),
        ("TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT", "torchtpu"),
    ]:
        path = args.cache_dir / subdir
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path.resolve())
    env["TORCH_TPU_TIER2_COMPILATION_CACHE"] = "disabled"
    env["TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE"] = "disabled"
    env["KV_BYTES"] = "auto"
    decode = args.scenario == "decode"
    buckets = [128, 512] if decode else [128, 256, 2048]
    env.update(
        TP="1",
        DP="8",
        CHUNK="512" if decode else "2048",
        MAX_SEQS="16" if decode else "1",
    )
    if decode:
        env["MAX_MODEL_LEN"] = "66048"
    env["PYTHONPATH"] = str(here) + os.pathsep + env.get("PYTHONPATH", "")
    server_command = [
        "bash",
        str(here.parent / "serve_cached_prefill.sh"),
        "--worker-extension-cls",
        "server_step_worker.ServerStepWorker",
        "--compilation-config",
        json.dumps({"compile_sizes": buckets}),
    ]
    if env.get("ARTIFACT_LOCAL_DIR"):
        server_command += [
            "--profiler-config",
            json.dumps(
                {
                    "profiler": "torch",
                    "torch_profiler_dir": str(
                        Path(env["ARTIFACT_LOCAL_DIR"]) / f"{args.scenario}/profile"
                    ),
                }
            ),
        ]
    manifest = {
        "scenario": args.scenario,
        "compile_sizes": buckets,
        "cache_mode": args.cache_mode,
        "cache_before": before,
        "cache_dir": str(args.cache_dir),
        "tier2": "disabled (public and internal)",
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["torch", "torch-tpu", "jax", "jaxlib", "libtpu"]
        },
        "startup_scope": "server launch through HTTP health readiness",
        "cache_hit_claim": "Inspect server logs; populated cache is not proof of hits",
    }
    import vllm

    def source_identity(root: Path) -> dict:
        return {
            "commit": subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
            ).strip(),
            "diff_sha256": hashlib.sha256(
                subprocess.check_output(
                    ["git", "-C", str(root), "diff", "HEAD", "--", "."]
                )
            ).hexdigest(),
        }

    model = Path(env["MODEL_PATH"])
    identity = {
        "scenario": args.scenario,
        "compile_sizes": buckets,
        "packages": manifest["packages"],
        "torchtpu": source_identity(here.parents[2]),
        "vllm": source_identity(Path(vllm.__file__).resolve().parent),
        "model": str(model.resolve()),
        "model_config_sha256": hashlib.sha256(
            (model / "config.json").read_bytes()
        ).hexdigest(),
        "parallelism": "TP1/DP8/EP8/PP1 on one v7x-8 host",
    }
    identity_path = args.cache_dir / "benchmark-identity.json"
    if args.cache_mode == "reuse":
        if (
            not identity_path.is_file()
            or json.loads(identity_path.read_text()) != identity
        ):
            parser.error("Cache identity differs; use a new cold cache directory")
    else:
        identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    manifest["identity"] = identity
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    server = None
    started = time.monotonic()
    timing = {"launch_wall_ns": time.time_ns(), "completed": False}
    try:
        with (args.out / "server.log").open("w") as log:
            server = subprocess.Popen(
                server_command,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            deadline = started + args.startup_timeout
            while True:
                if server.poll() is not None:
                    raise RuntimeError(
                        f"Server exited during startup: {server.returncode}"
                    )
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "Server did not become healthy before the deadline"
                    )
                try:
                    response = requests.get("http://127.0.0.1:8000/health", timeout=2)
                    ready = response.ok
                except requests.RequestException:
                    ready = False
                if ready:
                    break
                time.sleep(2)
            timing.update(
                ready_wall_ns=time.time_ns(),
                startup_seconds=time.monotonic() - started,
                cache_after_startup=cache_inventory(args.cache_dir),
            )
            (args.out / "startup.json").write_text(json.dumps(timing, indent=2) + "\n")
            base = [
                sys.executable,
                str(here / ("bench_decode.py" if decode else "bench_cached_extend.py")),
                "client",
                "--input",
                str(args.input),
            ]
            if decode:
                base += ["--server-log", str(args.out / "server.log")]
            with (args.out / "client-fixed.log").open("w") as client_log:
                subprocess.run(
                    [
                        *base,
                        "--out",
                        str(args.out / "fixed"),
                        "--rounds",
                        str(args.rounds),
                    ],
                    env=env,
                    stdout=client_log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            intervals = [
                json.loads(
                    (args.out / "fixed" / f"round-{index}-summary.json").read_text()
                )["audit_wall_interval_ns"]
                for index in range(1, args.rounds + 1)
            ]
            assert_no_compilation((args.out / "server.log").read_text(), intervals)
            if args.profile:
                with (args.out / "client-profile.log").open("w") as profile_log:
                    subprocess.run(
                        [
                            *base,
                            "--out",
                            str(args.out / "fixed-profile"),
                            "--rounds",
                            "1",
                            "--profile",
                        ],
                        env=env,
                        stdout=profile_log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
            timing["completed"] = True
    finally:
        stop(server)
        if (args.out / "server.log").exists():
            (args.out / "startup-phases.json").write_text(
                json.dumps(
                    summarize_startup(
                        (args.out / "server.log").read_text(errors="replace")
                    ),
                    indent=2,
                )
                + "\n"
            )
        timing["finished_wall_ns"] = time.time_ns()
        (args.out / "startup.json").write_text(json.dumps(timing, indent=2) + "\n")


if __name__ == "__main__":
    main()
