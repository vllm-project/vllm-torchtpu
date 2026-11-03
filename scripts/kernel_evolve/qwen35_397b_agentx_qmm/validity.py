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
"""Validity check for the Qwen3.5-397B AgentX campaign on kernels/quantized_matmul.

The campaign scores the steps that carry a 16k-token prefill chunk, so its score
reaches the FP8 dense linears at one token bucket (16384 rows). The server pads
every batch to one of 15 buckets, and the decode and short-prefill steps run the
others. This check holds a candidate to every bucket, without a device:

- `same_cell`: the arm's benchmark config launches the frozen cell. The target
  serves the arm's own tree, so nothing else stops a candidate that edits a
  cell-keyed serve flag in that config.
- `qmm_aot_buckets`: the arm's `xla_quantized_matmul` (the function the
  per-channel FP8 linear path calls, `layers/adapter/linear_common.py`) compiles
  ahead of time for one tpu7x core at every bucket and every dense-linear shape
  the model serves.
- `qmm_numerics_buckets`: on the CPU (Pallas code in TPU interpret mode), at
  every bucket up to 3072 rows and every shape, its output is finite, an
  all-zero row gives zeros, and each row is within MAX_ROW_ERR of the product
  with unquantized activations. Golden's own worst row there is 0.034 (fp8
  activation rounding); a broken tile or tail is far above it. The larger
  buckets are compile-only: the CPU takes too long on them, and the score runs
  the largest one on the TPU.

    python3 validity.py --arm <checkout> [--job-id <id>] \
        --python <python with jax 0.11.2, libtpu 0.0.50>

Prints one JSON value on stdout (the harness's validity contract); everything
else goes to stderr. A child that cannot run at all says nothing about the arm,
so the script then prints nothing and exits non-zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

CONFIG = "qwen3.5-397b-fp8-tp8-dp1-ep-agentx"
CELL = "7f7b02900fa6"
CHECKS = ("same_cell", "qmm_aot_buckets", "qmm_numerics_buckets")

# The server's token paddings for this config (tpu_platform.py's log line in the
# AgentX job j-094ffbf0-e12d-4011-b1dc's server log).
BUCKETS = (4, 8, 16, 32, 48, 64, 128, 256, 512, 1024, 2048, 3072, 4096, 8192, 16384)
NUMERIC_BUCKETS = tuple(b for b in BUCKETS if b <= 3072)
# (k, n) of every FP8 dense linear per TP8 rank, as the weight reaches the
# function: GDN in_proj_qkvz and attention qkv (4096 -> 2560), GDN out_proj and
# attention o_proj (1024 -> 4096), shared-expert gate_up (4096 -> 256) and
# down (128 -> 4096). From the server log's "FP8 linear weights requantized"
# lines and the model's config.json.
SHAPES = ((4096, 2560), (1024, 4096), (4096, 256), (128, 4096))
TOPOLOGY = "tpu7x:2x2x1"
MAX_ROW_ERR = 0.06

SHIM = """#!/bin/bash
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="${PATH#"$here/bin:"}"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$here/stub"
exec python3 "$@"
"""


def same_cell(arm: Path) -> tuple[bool, str]:
    from perfkit import cells
    from perfkit.targets.vllm_torchtpu import VllmTorchTpu

    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "bin").mkdir()
        (Path(tmp) / "stub" / "vllm").mkdir(parents=True)
        (Path(tmp) / "stub" / "vllm" / "__init__.py").write_text("")
        shim = Path(tmp) / "bin" / "python3"
        shim.write_text(SHIM)
        shim.chmod(0o755)
        spec = {
            "name": "qwen35-agentx-qmm-validity",
            "adapter": "vllm_torchtpu",
            "repo": str(arm),
            "env": {"PATH": f"{tmp}/bin:/usr/local/bin:/usr/bin:/bin"},
        }
        record, why = cells.resolve(VllmTorchTpu(spec), CONFIG, refresh=True)
    if record is None:
        return False, f"cannot derive the arm's cell for {CONFIG}: {why}"
    key = record["cell_key"]
    if key != CELL:
        return False, (
            f"the arm serves cell {key}, not the frozen {CELL}: "
            f"{json.dumps(record['fields'])}"
        )
    return True, f"{CONFIG} in this tree launches cell {CELL}"


def _inputs(np, jnp, rows: int, k: int, n: int, quantize_tensor):
    """bf16 activations (row 0 all zero, row 1 large) and a per-channel fp8
    weight quantized the way the server's loader does (scale over k)."""
    rng = np.random.default_rng(rows * 7919 + k * 31 + n)
    x = rng.standard_normal((rows, k)).astype(np.float32)
    x[0] = 0.0
    if rows > 1:
        x[1] *= 64.0
    w = (rng.standard_normal((k, n)) * 0.02).astype(np.float32)
    w_q, w_s = quantize_tensor(jnp.asarray(w), jnp.float8_e4m3fn, dim=0)
    return jnp.asarray(x).astype(jnp.bfloat16), w_q, w_s.reshape(-1).astype(jnp.float32)


def _child(arm: Path) -> dict:
    """Runs under the jax python: both kernel checks on the arm's code."""
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.setdefault("ALLOW_MULTIPLE_LIBTPU_LOAD", "1")
    os.environ.pop("LIBTPU_INIT_ARGS", None)
    os.environ.setdefault("TPU_ACCELERATOR_TYPE", "tpu7x-8")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(arm / "src"))
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax._src import mesh as mesh_lib
    from jax.experimental import topologies
    from jax.experimental.pallas import tpu as pltpu

    from vllm_torchtpu.kernels.quantized_matmul import util

    origin = Path(util.__file__).resolve()
    if not origin.is_relative_to(arm):
        raise RuntimeError(f"the kernel imported from {origin}, outside the arm {arm}")
    # Golden's quantizer, for the inputs only: the candidate's may differ.
    quantize_tensor = util.quantize_tensor
    fn = jax.jit(util.xla_quantized_matmul)

    aot = []
    device = topologies.get_topology_desc(TOPOLOGY, platform="tpu").devices[0]
    sharding = jax.sharding.SingleDeviceSharding(device)
    core = mesh_lib.AbstractMesh(
        (),
        (),
        abstract_device=mesh_lib.AbstractDevice(
            device_kind="TPU7x", num_cores=1, platform="tpu"
        ),
    )
    for k, n in SHAPES:
        for rows in BUCKETS:
            try:
                args = (
                    jax.ShapeDtypeStruct((rows, k), jnp.bfloat16, sharding=sharding),
                    jax.ShapeDtypeStruct((k, n), jnp.float8_e4m3fn, sharding=sharding),
                    jax.ShapeDtypeStruct((n,), jnp.float32, sharding=sharding),
                )
                with mesh_lib.use_abstract_mesh(core):
                    lowered = fn.trace(*args).lower(lowering_platforms=("tpu",))
                lowered.compile()
            except Exception as e:  # noqa: BLE001 - reported as this shape's failure
                aot.append(
                    f"rows {rows} k {k} n {n}: {type(e).__name__}: {str(e)[:300]}"
                )

    numerics, worst = [], 0.0
    for k, n in SHAPES:
        for rows in NUMERIC_BUCKETS:
            x, w_q, w_s = _inputs(np, jnp, rows, k, n, quantize_tensor)
            try:
                with pltpu.force_tpu_interpret_mode():
                    got = np.asarray(fn(x, w_q, w_s).astype(jnp.float32))
            except Exception as e:  # noqa: BLE001 - reported as this shape's failure
                numerics.append(
                    f"rows {rows} k {k} n {n}: {type(e).__name__}: {str(e)[:300]}"
                )
                continue
            if got.shape != (rows, n):
                numerics.append(f"rows {rows} k {k} n {n}: output shape {got.shape}")
                continue
            ref = np.asarray(x.astype(jnp.float32)) @ (
                np.asarray(w_q.astype(jnp.float32)) * np.asarray(w_s)[None]
            )
            if not np.isfinite(got).all():
                numerics.append(f"rows {rows} k {k} n {n}: non-finite output")
                continue
            if np.abs(got[0]).max() != 0.0:
                numerics.append(
                    f"rows {rows} k {k} n {n}: the all-zero row gives "
                    f"max |out| {np.abs(got[0]).max():.3g}"
                )
                continue
            err = (
                (
                    np.linalg.norm(got[1:] - ref[1:], axis=1)
                    / np.linalg.norm(ref[1:], axis=1)
                )
                if rows > 1
                else np.zeros(1)
            )
            worst = max(worst, float(err.max()))
            if err.max() > MAX_ROW_ERR:
                r = int(err.argmax()) + 1
                numerics.append(
                    f"rows {rows} k {k} n {n}: row {r} relative error "
                    f"{err.max():.4f} > {MAX_ROW_ERR}"
                )
    return {"aot": aot, "numerics": numerics, "worst_row_err": worst}


def kernel_checks(arm: Path, python: str) -> list[dict] | None:
    """The two kernel checks, from a child under `python`; None when the child
    could not run at all."""
    if not python:
        print(
            "validity: no --python and no $PERFKIT_PYTHON for the kernel checks",
            file=sys.stderr,
        )
        return None
    r = subprocess.run(
        [python, str(Path(__file__).resolve()), "--child", str(arm)],
        capture_output=True,
        text=True,
        timeout=3600,
    )
    sys.stderr.write(r.stderr[-4000:])
    if r.returncode != 0:
        print(f"validity: the kernel child exited {r.returncode}", file=sys.stderr)
        return None
    try:
        got = json.loads(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        print(
            f"validity: the kernel child printed no result: {r.stdout[-500:]!r}",
            file=sys.stderr,
        )
        return None
    shapes = ", ".join(f"{k}x{n}" for k, n in SHAPES)
    passes = {
        "aot": f"compiles for {TOPOLOGY} at {len(BUCKETS)} buckets x shapes {shapes}",
        "numerics": (
            f"finite, zero row zero, worst row error {got['worst_row_err']:.4f} "
            f"<= {MAX_ROW_ERR} at buckets {NUMERIC_BUCKETS[0]}-{NUMERIC_BUCKETS[-1]} "
            f"x shapes {shapes}"
        ),
    }
    out = []
    for name, key in (("qmm_aot_buckets", "aot"), ("qmm_numerics_buckets", "numerics")):
        fails = got[key]
        out.append(
            {
                "name": name,
                "pass": not fails,
                "detail": passes[key]
                if not fails
                else "; ".join(fails[:5])
                + (f"; and {len(fails) - 5} more" if len(fails) > 5 else ""),
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--job-id", default="")
    ap.add_argument("--python", default=os.environ.get("PERFKIT_PYTHON", ""))
    a = ap.parse_args()
    sys.path.insert(
        0, os.environ.get("PERFKIT_HOME", str(Path.home() / "tpu-perf-forge"))
    )
    from perfkit import scorer

    arm = Path(a.arm).expanduser().resolve()
    tree = scorer.git_tree(arm)
    if not tree:
        print(f"validity: {arm} is not a git tree", file=sys.stderr)
        return 2
    ok, detail = same_cell(arm)
    print(f"validity: same_cell {'PASS' if ok else 'FAIL'}: {detail}", file=sys.stderr)
    kernel = kernel_checks(arm, a.python)
    if kernel is None:
        return 2
    for c in kernel:
        print(
            f"validity: {c['name']} {'PASS' if c['pass'] else 'FAIL'}: {c['detail']}",
            file=sys.stderr,
        )
    run = os.environ.get("AMPLIO_RUN_ID", "local")
    out = {
        "validity_hash": "sha256:"
        + hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "job_id": a.job_id or f"{run}-validity-{int(time.time())}",
        "tree": tree,
        "checks": [{"name": "same_cell", "pass": ok, "detail": detail}, *kernel],
    }
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        print(json.dumps(_child(Path(sys.argv[2]).resolve())))
        sys.exit(0)
    sys.exit(main())
