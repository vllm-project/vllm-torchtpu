"""Validity check for the Qwen3.5-397B AgentX layer-run campaign: the arm serves the frozen cell.

forge's serve_e2e.py refuses a local arm whose tree would launch another cell
(`_assert_same_cell`). An arm it measures on Cloud DevKit takes its serve command
from the target's repo, and this campaign's target names the arm's own tree, so
nothing in the scorer stops a candidate that edits a cell-keyed serve flag in its
benchmark config. This check re-derives the arm's cell key from the command its
tree would launch, with perfkit's own definition of a cell, and fails a tree that
moved one. Environment exports in the config script are tree code and stay free.

    python3 validity.py --arm <checkout> [--job-id <id>]

Prints one JSON value on stdout (harness/internal/campaign/validity/validity.go);
anything else goes to stderr. Needs no device: it runs the arm's
`run_benchmarks.sh --dry-run`, with an empty vllm package of its own on the path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

CONFIG = "qwen3.5-397b-fp8-tp8-dp1-ep-agentx"
CELL = "7f7b02900fa6"
CHECKS = ("same_cell",)

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--job-id", default="")
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
    out = {"validity_hash": "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "job_id": a.job_id or f"{os.environ.get('AMPLIO_RUN_ID', 'local')}-validity-{int(time.time())}",
           "tree": tree,
           "checks": [{"name": "same_cell", "pass": ok, "detail": detail}]}
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
