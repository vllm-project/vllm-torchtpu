#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Shared pieces of the phase 3 AgentX harness benches.

Used by p3_p_leg_bench.py and p3_d_leg_bench.py (offline vLLM engine benches)
and by the kernel benches under tests/kernels/test_p3_*.py. It provides:

  * environment_id(): image, libtpu, torch_tpu, jax, vLLM, source sha and
    dirty-tree hash, topology, job id. Every bench prints it and writes it to
    env.json in its run directory.
  * Timeline: absolute marks for the queue, init, load, compile, warmup and
    timed windows, reported separately.
  * run_dir(): one output directory per run, never shared between pods.
  * SITECUSTOMIZE: source of a sitecustomize.py that the engine processes load
    at interpreter start (they are spawned and inherit PYTHONPATH). It patches
    the dummy loader (synthetic weights that survive the fp8 cast), the fused
    EP MoE routing (hash or trace replay), the scheduler (one JSONL record per
    engine step with computed tokens, running count, spec tokens and sampled
    tokens) and the TPU model runner (load, compile and warmup phase markers).

Nothing here touches the TPU unless environment_id(devices=True) is asked for,
so the engine benches can call it in the parent process before the workers
spawn.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import platform
import socket
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_ROOT = "/workspace/tmp/p3-harness/jobs"


def job_id() -> str:
    return (
        os.environ.get("CDK_JOB_ID")
        or os.environ.get("JS_NAME")
        or f"local-{socket.gethostname()}"
    )


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            capture_output=True, text=True, timeout=60, check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def source_identity() -> dict:
    """Source sha and a hash of the uncommitted diff.

    A --map-dir snapshot or an export copy has no .git; the export step writes
    P3_SNAPSHOT.json next to pyproject.toml with the same keys, and P3_SOURCE_SHA
    in the environment overrides both.
    """
    out = {"source_sha": os.environ.get("P3_SOURCE_SHA", ""), "source_dirty_sha": ""}
    snap = REPO_ROOT / "P3_SNAPSHOT.json"
    if snap.exists():
        try:
            data = json.loads(snap.read_text())
            out["source_sha"] = out["source_sha"] or data.get("source_sha", "")
            out["source_dirty_sha"] = data.get("source_dirty_sha", "")
            out["snapshot"] = data.get("snapshot", "")
        except (OSError, ValueError):
            pass
    if not out["source_sha"] and (REPO_ROOT / ".git").exists():
        out["source_sha"] = _git("rev-parse", "HEAD")
        diff = _git("diff", "HEAD", "--", "src", "scripts", "tests")
        out["source_dirty_sha"] = hashlib.sha256(diff.encode()).hexdigest()[:12] if diff else ""
    out["source_sha"] = out["source_sha"] or "unknown"
    return out


def package_versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for dist in ("torch-tpu", "libtpu", "jax", "jaxlib", "vllm", "vllm-torchtpu",
                 "torch", "tpu-sync-torch", "tpu-raiden-torch"):
        try:
            out[dist] = version(dist)
        except PackageNotFoundError:
            out[dist] = ""
    return out


def topology() -> dict:
    env = os.environ
    return {
        "tpu_topology": env.get("TPU_TOPOLOGY", ""),
        "chips_per_host_bounds": env.get("TPU_CHIPS_PER_HOST_BOUNDS", ""),
        "host_bounds": env.get("TPU_HOST_BOUNDS", ""),
        "process_bounds": env.get("TPU_PROCESS_BOUNDS", ""),
        "worker_id": env.get("TPU_WORKER_ID", ""),
        "num_hosts": env.get("P3_NUM_HOSTS", ""),
        "accelerator": env.get("P3_ACCELERATOR", "tpu7x"),
    }


def environment_id(devices: bool = False) -> dict:
    """The environment id every bench prints and records.

    devices=True adds jax.devices() (id, coords, core); it opens the TPU, so the
    offline engine benches never ask for it in the parent process.
    """
    env = {
        "image": os.environ.get("P3_IMAGE") or os.environ.get("SERVER_IMAGE")
        or os.environ.get("IMAGE") or "",
        "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS", ""),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "job": job_id(),
        "utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    env.update(package_versions())
    env.update(source_identity())
    env["topology"] = topology()
    if devices:
        import jax

        env["devices"] = [
            {"id": d.id, "coords": list(getattr(d, "coords", ())),
             "core": getattr(d, "core_on_chip", None), "kind": d.device_kind,
             "process": d.process_index}
            for d in jax.devices()
        ]
        env["jax_process_count"] = jax.process_count()
    return env


def environment_id_string(env: dict) -> str:
    """One line: the fields that decide whether two numbers are comparable."""
    t = env.get("topology", {})
    image = env.get("image", "")
    if "@sha256:" in image:
        image = image.split("@sha256:")[0].rsplit("/", 1)[-1] + "@" + image.split("@sha256:")[1][:12]
    elif "/" in image:
        image = image.rsplit("/", 1)[-1]
    return (
        f"image={image or 'unknown'};libtpu={env.get('libtpu', '')};"
        f"torch_tpu={env.get('torch-tpu', '')};jax={env.get('jax', '')};"
        f"vllm={env.get('vllm', '')};sha={env.get('source_sha', '')[:9]}"
        f"{'+' + env['source_dirty_sha'] if env.get('source_dirty_sha') else ''};"
        f"topo={t.get('tpu_topology') or t.get('chips_per_host_bounds') or '?'}"
        f"x{t.get('num_hosts') or '1'}host"
    )


def run_dir(bench: str, tag: str = "", root: str | None = None) -> str:
    """<root>/<job-id>/<bench>[-tag]-<utc>/ created fresh; never shared."""
    root = root or os.environ.get("P3_OUT_ROOT") or DEFAULT_OUT_ROOT
    if not os.path.isdir(os.path.dirname(root.rstrip("/")) or "/"):
        root = os.path.join(os.getcwd(), "p3-harness-out")
    name = bench + (f"-{tag}" if tag else "") + "-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = os.path.join(root, job_id(), name)
    os.makedirs(path, exist_ok=True)
    return path


def write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)


def write_env(out_dir: str, extra: dict | None = None, devices: bool = False) -> dict:
    env = environment_id(devices=devices)
    if extra:
        env.update(extra)
    env["id"] = environment_id_string(env)
    write_json(os.path.join(out_dir, "env.json"), env)
    print(f"[p3] ENVIRONMENT_ID {env['id']}", flush=True)
    print(f"[p3] out_dir {out_dir}", flush=True)
    return env
