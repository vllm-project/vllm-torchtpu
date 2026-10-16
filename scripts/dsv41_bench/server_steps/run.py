# SPDX-License-Identifier: Apache-2.0
"""Run an isolated server with audited startup and natural serving measurements."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests
from bench_cached_extend import (
    assert_no_compilation,
    load_frozen_input,
    summarize_startup,
)
from server_step_worker import DECODE_SCENARIO


def source_identity(
    root: Path, package: str, *, build_commit: str | None = None
) -> dict:
    if any((parent / ".git").exists() for parent in (root, *root.parents)):
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

    # CI excludes .git; its image records the build SHA. The installed vLLM
    # Git dependency instead records its upstream SHA in direct_url.json.
    distribution = importlib.metadata.distribution(package)
    if build_commit is not None:
        if re.fullmatch(r"[0-9a-f]{40}", build_commit) is None:
            raise ValueError("Buildkite image source requires a full Git commit SHA")
        return {
            "commit": build_commit,
            "package_version": distribution.version,
            "provenance": "Buildkite CI image",
        }
    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text is None:
        raise ValueError(f"{package} requires Git source provenance for the benchmark")
    direct_url = json.loads(direct_url_text)
    if "vcs_info" not in direct_url or not direct_url["vcs_info"].get("commit_id"):
        raise ValueError(f"{package} direct_url.json has no VCS commit identity")
    return {
        "commit": direct_url["vcs_info"]["commit_id"],
        "package_version": distribution.version,
        "direct_url": direct_url,
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


def bootstrap_ports(env: dict[str, str], out: Path) -> None:
    """Use checked non-ephemeral ports for the eight co-located TPU workers."""
    for base in range(19000, 20000, 10):
        sockets = []
        try:
            for port in range(base, base + 10):
                sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
                sockets.append(sock)
                sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
                sock.bind(("::", port))
            ports = list(range(base, base + 10))
            env["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = ",".join(
                f"localhost:{port}" for port in ports[:8]
            )
            env["MASTER_ADDR"] = env["TORCH_TPU_DP_MASTER_ADDR"] = "127.0.0.1"
            env["MASTER_PORT"] = str(ports[8])
            env["TORCH_TPU_DP_MASTER_PORT"] = str(ports[9])
            (out / "bootstrap-ports.json").write_text(json.dumps(ports))
            return
        except OSError:
            continue
        finally:
            for sock in sockets:
                sock.close()
    raise RuntimeError("No available ten-port bootstrap block")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
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
    if args.scenario == "encoder_extend":
        load_frozen_input(args.input)

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        response = requests.get("http://127.0.0.1:8000/health", timeout=2)
    except requests.RequestException:
        pass
    else:
        if response.ok:
            parser.error("Port 8000 already has a healthy server")

    here = Path(__file__).resolve().parent
    env = os.environ.copy()
    compiler_scratch = tempfile.TemporaryDirectory(prefix="dsv41-compiler-")
    env["VLLM_CACHE_ROOT"] = compiler_scratch.name
    # Do not inherit persistent cache paths from the task or the host.
    for key in (
        "JAX_COMPILATION_CACHE_DIR",
        "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT",
        "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT",
    ):
        env.pop(key, None)
    env.update(
        VLLM_DISABLE_COMPILE_CACHE="1",
        JAX_ENABLE_COMPILATION_CACHE="false",
        TORCHINDUCTOR_FORCE_DISABLE_CACHES="1",
        TORCHINDUCTOR_AUTOGRAD_CACHE="0",
        TORCH_TPU_TIER2_COMPILATION_CACHE="disabled",
        TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE="disabled",
    )
    env["KV_BYTES"] = "auto"
    decode = args.scenario == "decode"
    # Steady target verification is exactly local_concurrency * (K + 1) rows;
    # padding it to 384 runs every layer on 25% padding rows.
    verify_rows = DECODE_SCENARIO.local_concurrency * (
        DECODE_SCENARIO.dspark_tokens + 1
    )
    buckets = [verify_rows, DECODE_SCENARIO.chunk_tokens] if decode else [2048]
    env.update(
        TP="1",
        DP="8",
        CHUNK=str(DECODE_SCENARIO.chunk_tokens) if decode else "2048",
        MAX_SEQS=str(DECODE_SCENARIO.local_concurrency) if decode else "1",
    )
    if decode:
        env["MAX_MODEL_LEN"] = str(
            DECODE_SCENARIO.prefix_tokens + DECODE_SCENARIO.output_tokens
        )
    else:
        env.pop("MAX_MODEL_LEN", None)
    env["USE_PHASED_PROFILER"] = "0"
    env["PYTHONPATH"] = str(here) + os.pathsep + env.get("PYTHONPATH", "")
    server_command = [
        "bash",
        str(here.parent / "serve_cached_prefill.sh"),
        "--worker-extension-cls",
        "server_step_worker.ServerStepWorker",
        "--compilation-config",
        json.dumps({"compile_sizes": buckets}),
    ]
    if args.profile:
        server_command += [
            "--profiler-config",
            json.dumps(
                {
                    "profiler": "torch",
                    "torch_profiler_dir": str(args.out / "profile"),
                }
            ),
        ]
    manifest = {
        "scenario": args.scenario,
        "compile_sizes": buckets,
        "cache_mode": "disabled",
        "serialization": "enabled",
        "tier2": "disabled (public and internal)",
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["torch", "torch-tpu", "jax", "jaxlib", "libtpu"]
        },
        "startup_scope": "server launch through HTTP health readiness",
    }
    import vllm

    model = Path(env["MODEL_PATH"])
    identity = {
        "scenario": args.scenario,
        "compile_sizes": buckets,
        "packages": manifest["packages"],
        "torchtpu": source_identity(
            here.parents[2],
            "vllm-torchtpu",
            build_commit=env.get("BUILDKITE_COMMIT"),
        ),
        "vllm": source_identity(Path(vllm.__file__).resolve().parent, "vllm"),
        "model": str(model.resolve()),
        "model_weights": env.get("MODEL_URI") or env["MODEL_PATH"],
        "model_config_sha256": hashlib.sha256(
            (model / "config.json").read_bytes()
        ).hexdigest(),
        "parallelism": "TP1/DP8/EP8/PP1 on one v7x-8 host",
    }
    manifest["identity"] = identity
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    bootstrap_ports(env, args.out)
    server = None
    started = time.monotonic()
    timing = {
        "launch_wall_ns": time.time_ns(),
        "completed": False,
        "formal_status": "pending",
        "profile_status": "pending" if args.profile else "disabled",
    }
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
                startup_log = (args.out / "server.log").read_text(errors="replace")
                if any(
                    error in startup_log
                    for error in [
                        "WorkerProc failed to start",
                        "Engine core initialization failed",
                    ]
                ):
                    raise RuntimeError("Worker startup failed; see server.log")
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
            timing["formal_status"] = "failed"
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
            timing["formal_status"] = "passed"
            if args.profile:
                timing["profile_status"] = "failed"
                with (args.out / "client-profile.log").open("w") as profile_log:
                    subprocess.run(
                        [
                            *base,
                            "--out",
                            str(args.out / "fixed-profile"),
                            *(
                                ["--profile-only", "--reuse-prefix"]
                                if decode
                                else ["--profile-only"]
                            ),
                        ],
                        env=env,
                        stdout=profile_log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                timing["profile_status"] = "passed"
            timing["completed"] = True
    finally:
        stop(server)
        compiler_scratch.cleanup()
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
