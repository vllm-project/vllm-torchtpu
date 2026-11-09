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
from server_step_worker import SCENARIOS


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
    parser.add_argument("--scenario", choices=list(SCENARIOS), default="encoder_extend")
    parser.add_argument("--host-rank", type=int, choices=[0, 1], default=0)
    parser.add_argument("--head-address")
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--compile-sizes", type=int, nargs="+")
    args = parser.parse_args()
    multihost = args.scenario == "decode_dp16"
    worker = multihost and args.host_rank == 1
    if multihost != bool(args.head_address) or (args.host_rank and not multihost):
        parser.error("decode_dp16 requires --head-address and two host ranks")
    if multihost and args.profile:
        parser.error("The CI multihost launcher runs formal measurements only")
    scenario = SCENARIOS[args.scenario]
    decode = scenario.name == "decode"
    if args.rounds is None:
        args.rounds = 1 if decode else 5
    if args.rounds < 1:
        parser.error("The integrated launcher requires at least one formal window")
    if args.scenario == "encoder_extend":
        load_frozen_input(args.input)

    shutdown_requested = False

    def terminate(signum, frame):
        nonlocal shutdown_requested
        if worker and signum == signal.SIGTERM:
            shutdown_requested = True
            return
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "launcher.pid").write_text(str(os.getpid()))
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
    env["MOE_FUSED_EP_PREPROCESS_BACKEND"] = "topology" if decode else "native"
    # Steady target verification is exactly local_concurrency * (K + 1) rows;
    # padding it to 384 runs every layer on 25% padding rows.
    verify_rows = scenario.local_concurrency * (scenario.dspark_tokens + 1)
    # Target and context-KV buckets; draft queries derive their own ladder
    # from request buckets and the draft block size.
    buckets = [verify_rows, scenario.chunk_tokens] if decode else [128, 2048]
    if args.compile_sizes is not None:
        buckets = args.compile_sizes
        if buckets != sorted(set(buckets)) or buckets[0] <= 0:
            parser.error("--compile-sizes must be positive, unique and increasing")
        if buckets[-1] < scenario.chunk_tokens:
            parser.error("--compile-sizes must cover the scheduler token budget")
    env.update(
        TP="1",
        DP=str(scenario.dp_size),
        CHUNK=str(scenario.chunk_tokens),
        MAX_SEQS=str(scenario.local_concurrency),
    )
    if decode:
        env["MAX_MODEL_LEN"] = str(scenario.prefix_tokens + scenario.output_tokens)
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
    if decode and not worker:
        server_command += ["--enable-log-requests", "--max-log-len", "0"]
    if multihost:
        # Native MP owns bootstrap; do not inherit a single-host SliceBuilder.
        for key in (
            "MEGASCALE_COORDINATOR_ADDRESS",
            "MEGASCALE_NUM_SLICES",
            "MEGASCALE_PORT",
            "MEGASCALE_SLICE_ID",
            "TPU_PROCESS_ADDRESSES",
            "TPU_PROCESS_PORT",
            "TPU_WORKER_HOSTNAMES",
            "TORCH_TPU_SLICEBUILDER_ADDRESSES",
        ):
            env.pop(key, None)
        head = socket.gethostbyname(args.head_address)
        env.update(
            TPU_MULTIHOST_BACKEND="mp",
            TPU_SKIP_MDS_QUERY="1",
            TORCH_TPU_DP_MASTER_ADDR=head,
            TORCH_TPU_DP_MASTER_PORT="19700",
            MASTER_ADDR=head,
            MASTER_PORT="19700",
            TORCH_TPU_BASE_PORT="19000",
            HOST="127.0.0.1",
            HEADLESS=str(int(worker)),
            TPU_DEEPSEEK_V41_ENGRAM_HBM="first",
            MOE_FUSED_EP_KERNEL_IMPL="push",
            MOE_FUSED_EP_PUSH_SC_PREPUSH="2",
            MOE_FUSED_EP_PUSH_TC_EXCHANGE="1",
            TPU_MOE_EPLB_DYNAMIC="1",
            TPU_RECLAIM_COMPILE_MEMORY="1",
            LD_PRELOAD="",
            ENGRAM_CONFIG='{"cpu_offload":true,"dp_shared_memory":false}',
        )
        server_command += [
            "--data-parallel-size-local",
            "8",
            "--data-parallel-address",
            head,
            "--data-parallel-rpc-port",
            "19500",
        ]
        # On the API host, even start-rank=0 selects hybrid LB and rejects
        # headless remote engines. Only the headless worker needs the offset.
        if worker:
            server_command += ["--data-parallel-start-rank", "8"]
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
        "engram_transfer": "raw_dma",
        "tier2": "disabled (public and internal)",
        "packages": {
            name: importlib.metadata.version(name)
            for name in [
                "torch",
                "torch-tpu",
                "jax",
                "jaxlib",
                "libtpu",
                "tpu-sync-torch",
            ]
        },
        "startup_scope": "server launch through HTTP health readiness",
    }
    if multihost:
        manifest.update(
            host_rank=args.host_rank,
            hostname=socket.gethostname(),
            engram_hbm="first",
            engram_dp_shared_memory=False,
            moe_kernel=env["MOE_FUSED_EP_KERNEL_IMPL"],
            moe_push_sc_prepush=int(env["MOE_FUSED_EP_PUSH_SC_PREPUSH"]),
            moe_push_tc_exchange=env["MOE_FUSED_EP_PUSH_TC_EXCHANGE"] == "1",
            moe_eplb_dynamic=env["TPU_MOE_EPLB_DYNAMIC"] == "1",
            moe_preprocess=env["MOE_FUSED_EP_PREPROCESS_BACKEND"],
            allocator="glibc",
            reclaim_compile_memory=True,
        )
    import vllm

    model = Path(env["MODEL_PATH"])
    identity = {
        "scenario": args.scenario,
        "engram_transfer": "raw_dma",
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
        "parallelism": (
            "TP1/DP16/EP16/PP1 on two v7x hosts (8 physical chips)"
            if multihost
            else "TP1/DP8/EP8/PP1 on one v7x-8 host"
        ),
    }
    manifest["identity"] = identity
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not multihost:
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
            if worker:
                # The host launcher stops this process only after the head client
                # finishes. An earlier serving exit is a failed benchmark.
                while not shutdown_requested:
                    if server.poll() is not None:
                        raise RuntimeError(f"Worker server exited: {server.returncode}")
                    time.sleep(2)
                timing.update(completed=True, shutdown_requested=True)
                return
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
                base += [
                    "--scenario",
                    args.scenario,
                    "--server-log",
                    str(args.out / "server.log"),
                ]
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
            if multihost:
                # Keep the head alive until the host coordinator has stopped
                # the worker. Killing the DP master first looks like a peer crash.
                (args.out / "formal-complete").touch()
                deadline = time.monotonic() + 240
                while not (args.out / "shutdown-permitted").exists():
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "Host coordinator did not finish worker shutdown"
                        )
                    time.sleep(2)
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
