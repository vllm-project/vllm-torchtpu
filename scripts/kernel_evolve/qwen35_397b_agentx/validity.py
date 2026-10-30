"""Validity check for the Qwen3.5-397B AgentX layer-run campaign: the arm serves the frozen cell.

forge's serve_e2e.py refuses a local arm whose tree would launch another cell
(`_assert_same_cell`). An arm it measures on Cloud DevKit takes its serve command
from the target's repo, and this campaign's target names the arm's own tree, so
nothing in the scorer stops a candidate that edits a cell-keyed serve flag in its
benchmark config. This check re-derives the arm's cell key from the command its
tree would launch, with perfkit's own definition of a cell, and fails a tree that
moved one. Environment exports in the config script are tree code and stay free.

    python3 validity.py --arm <checkout> [--job-id <id>]

The router top-k kernel's checks run in a child process under `--python` (default
$PERFKIT_PYTHON), which needs jax 0.11.2 and libtpu 0.0.50 and opens no TPU:

- `router_topk_aot_buckets`: the arm's `vllm_torchtpu.kernels.router_topk.router_topk`,
  the function the MoE router's torch op wraps, compiles ahead of time for one tpu7x
  core at every token bucket the server pads a batch to (ROUTER_BUCKETS), and each
  program launches a Mosaic kernel. The frozen config only reaches the 16384 bucket.
- `router_topk_interpret_buckets`: at every bucket, in TPU interpret mode, it returns
  exactly what the documented algorithm returns (`_reference`, the same as
  `layers/core/rowmax_topk.py`, the torch path served off-TPU): values descending,
  ties to the lowest expert id, NaN/-inf/-FLT_MAX scores clamped, and an all-hazard
  row named by NaN weights.

    python3 validity.py --arm <checkout> [--job-id <id>] [--python <python with jax>]

Prints one JSON value on stdout (harness/internal/campaign/validity/validity.go);
anything else goes to stderr. Needs no device: it runs the arm's
`run_benchmarks.sh --dry-run`, with an empty vllm package of its own on the path.
A child that cannot run (no such python, no jax) says nothing about the arm, so the
script then prints nothing and exits non-zero.
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
CHECKS = ("same_cell", "router_topk_aot_buckets", "router_topk_interpret_buckets")

# The server's token paddings for this config (tpu_platform.py's log line in the
# AgentX job's server log): every row count the router top-k is called with.
ROUTER_BUCKETS = (4, 8, 16, 32, 48, 64, 128, 256, 512, 1024, 2048, 3072, 4096, 8192, 16384)
EXPERTS = 512
TOPK = 10
TOPOLOGY = "tpu7x:2x2x1"
NEG = -3.0e38

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
        spec = {"name": "qwen35-agentx-validity", "adapter": "vllm_torchtpu", "repo": str(arm),
                "env": {"PATH": f"{tmp}/bin:/usr/local/bin:/usr/bin:/bin"}}
        record, why = cells.resolve(VllmTorchTpu(spec), CONFIG, refresh=True)
    if record is None:
        return False, f"cannot derive the arm's cell for {CONFIG}: {why}"
    key = record["cell_key"]
    if key != CELL:
        return False, (f"the arm serves cell {key}, not the frozen {CELL}: "
                       f"{json.dumps(record['fields'])}")
    return True, f"{CONFIG} in this tree launches cell {CELL}"


def _reference(np, x, k):
    """The kernel's documented algorithm in numpy: k passes of row max, lowest
    column attaining it, mask it; hazards clamped to NEG; NaN weights for a row
    with no score above NEG."""
    n = x.shape[1]
    neg = np.float32(NEG)
    cur = np.where(x > neg, x, neg)
    masked = np.finfo(np.float32).min
    iota = np.arange(n, dtype=np.int32)
    vals, ids = [], []
    for _ in range(k):
        m = cur.max(axis=1, keepdims=True)
        idx = np.where(cur == m, iota, n - 1).min(axis=1)
        vals.append(m[:, 0])
        ids.append(idx)
        cur = np.where(iota == idx[:, None], masked, cur)
    w = np.stack(vals, 1).astype(np.float32)
    w = np.where(w[:, :1] <= neg, np.float32(np.nan), w)
    return w, np.stack(ids, 1).astype(np.int32)


def _scores(np, rows: int):
    """Softmax router scores, with the hazards in the first rows: an all-NaN row,
    an all -inf row, ties at the maximum, and a row mixing NaN, -inf and -FLT_MAX
    with real scores."""
    rng = np.random.default_rng(rows)
    logits = rng.standard_normal((rows, EXPERTS)).astype(np.float32)
    x = np.exp(logits - logits.max(axis=1, keepdims=True))
    x = (x / x.sum(axis=1, keepdims=True)).astype(np.float32)
    special = [np.full(EXPERTS, np.nan, np.float32), np.full(EXPERTS, -np.inf, np.float32)]
    tie = x[min(2, rows - 1)].copy()
    tie[[7, 100, 300, 511]] = tie.max() + np.float32(0.25)
    tie[[3, 200]] = np.float32(0.125)
    special.append(tie)
    mixed = x[min(3, rows - 1)].copy()
    mixed[::5] = np.nan
    mixed[1::7] = -np.inf
    mixed[2::11] = np.finfo(np.float32).min
    special.append(mixed)
    for r, row in enumerate(special[:rows]):
        x[r] = row
    return x


def _router_child(arm: Path) -> dict:
    """Runs under the jax python: both router checks on the arm's kernel."""
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.setdefault("ALLOW_MULTIPLE_LIBTPU_LOAD", "1")
    os.environ.setdefault("TPU_ACCELERATOR_TYPE", "tpu7x-8")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(arm / "src"))
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax._src import mesh as mesh_lib
    from jax.experimental import topologies
    from jax.experimental.pallas import tpu as pltpu

    import vllm_torchtpu.kernels.router_topk as rk

    origin = Path(rk.__file__).resolve()
    if not origin.is_relative_to(arm):
        raise RuntimeError(f"the router kernel imported from {origin}, outside the arm {arm}")
    fn = jax.jit(rk.router_topk, static_argnums=(1,))

    aot = []
    device = topologies.get_topology_desc(TOPOLOGY, platform="tpu").devices[0]
    sharding = jax.sharding.SingleDeviceSharding(device)
    core = mesh_lib.AbstractMesh((), (), abstract_device=mesh_lib.AbstractDevice(
        device_kind="TPU7x", num_cores=1, platform="tpu"))
    for rows in ROUTER_BUCKETS:
        try:
            arg = jax.ShapeDtypeStruct((rows, EXPERTS), jnp.float32, sharding=sharding)
            with mesh_lib.use_abstract_mesh(core):
                lowered = fn.trace(arg, TOPK).lower(lowering_platforms=("tpu",))
            lowered.compile()
            if not lowered.as_text().count("tpu_custom_call"):
                aot.append(f"rows {rows}: the program launches no Mosaic kernel")
        except Exception as e:  # noqa: BLE001 - reported as this bucket's failure
            aot.append(f"rows {rows}: {type(e).__name__}: {str(e)[:300]}")

    interp = []
    for rows in ROUTER_BUCKETS:
        x = _scores(np, rows)
        want_w, want_i = _reference(np, x, TOPK)
        try:
            with pltpu.force_tpu_interpret_mode():
                got_w, got_i = fn(jnp.asarray(x), TOPK)
            got_w, got_i = np.asarray(got_w), np.asarray(got_i)
        except Exception as e:  # noqa: BLE001 - reported as this bucket's failure
            interp.append(f"rows {rows}: {type(e).__name__}: {str(e)[:300]}")
            continue
        if got_w.shape != want_w.shape or got_i.shape != want_i.shape:
            interp.append(f"rows {rows}: shapes {got_w.shape}/{got_i.shape}, want {want_w.shape}")
            continue
        bad = np.nonzero((got_i != want_i).any(axis=1)
                         | ~((got_w == want_w) | (np.isnan(got_w) & np.isnan(want_w))).all(axis=1))[0]
        if bad.size:
            r = int(bad[0])
            interp.append(f"rows {rows}: {bad.size} rows differ, first row {r}: ids "
                          f"{got_i[r].tolist()} want {want_i[r].tolist()}")
    return {"aot": aot, "interpret": interp}


def router_checks(arm: Path, python: str) -> list[dict] | None:
    """The two router checks, from a child under `python`; None when the child
    could not run at all."""
    if not python:
        print("validity: no --python and no $PERFKIT_PYTHON for the router checks", file=sys.stderr)
        return None
    r = subprocess.run([python, str(Path(__file__).resolve()), "--router-child", str(arm)],
                       capture_output=True, text=True, timeout=1800)
    sys.stderr.write(r.stderr[-4000:])
    if r.returncode != 0:
        print(f"validity: the router child exited {r.returncode}", file=sys.stderr)
        return None
    try:
        got = json.loads(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        print(f"validity: the router child printed no result: {r.stdout[-500:]!r}", file=sys.stderr)
        return None
    buckets = f"{len(ROUTER_BUCKETS)} buckets, rows {ROUTER_BUCKETS[0]}-{ROUTER_BUCKETS[-1]}"
    out = []
    for name, key, ok in (("router_topk_aot_buckets", "aot", f"compiles for {TOPOLOGY} at {buckets}"),
                          ("router_topk_interpret_buckets", "interpret",
                           f"matches the reference in TPU interpret mode at {buckets}")):
        fails = got[key]
        out.append({"name": name, "pass": not fails,
                    "detail": ok if not fails else "; ".join(fails[:5])
                    + (f"; and {len(fails) - 5} more" if len(fails) > 5 else "")})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--job-id", default="")
    ap.add_argument("--python", default=os.environ.get("PERFKIT_PYTHON", ""))
    a = ap.parse_args()
    sys.path.insert(0, os.environ.get("PERFKIT_HOME", str(Path.home() / "tpu-perf-forge")))
    from perfkit import scorer

    arm = Path(a.arm).expanduser().resolve()
    tree = scorer.git_tree(arm)
    if not tree:
        print(f"validity: {arm} is not a git tree", file=sys.stderr)
        return 2
    ok, detail = same_cell(arm)
    print(f"validity: same_cell {'PASS' if ok else 'FAIL'}: {detail}", file=sys.stderr)
    router = router_checks(arm, a.python)
    if router is None:
        return 2
    for c in router:
        print(f"validity: {c['name']} {'PASS' if c['pass'] else 'FAIL'}: {c['detail']}",
              file=sys.stderr)
    out = {"validity_hash": "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "job_id": a.job_id or f"{os.environ.get('AMPLIO_RUN_ID', 'local')}-validity-{int(time.time())}",
           "tree": tree,
           "checks": [{"name": "same_cell", "pass": ok, "detail": detail}, *router]}
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--router-child":
        print(json.dumps(_router_child(Path(sys.argv[2]).resolve())))
        sys.exit(0)
    sys.exit(main())
