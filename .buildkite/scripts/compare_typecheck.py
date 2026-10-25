#!/usr/bin/env python3
"""Compare two source revisions inside one pinned CI image, without changing gates."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
from collections import Counter
from pathlib import Path


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def inside():
    output = Path("/comparison")
    manifest = json.loads((output / "manifest.json").read_text())
    subprocess.run(
        ["uv", "pip", "install", "--system", "--quiet", "pyrefly==1.3.0"],
        check=True,
    )
    packages = sorted(
        (dist.metadata["Name"], dist.version)
        for dist in importlib.metadata.distributions()
    )
    save_json(output / "packages.json", packages)
    workspace = Path("/workspace/vllm-torchtpu")
    os.chdir("/tmp")
    backup = Path(tempfile.mkdtemp(prefix="image-source-")) / "original"
    shutil.move(workspace, backup)
    diagnostics = {}
    statuses = {}
    baseline_hashes = {}
    try:
        for case in ("base", "head"):
            archive = output / f"{case}.tar"
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            assert digest == manifest[case]["archive_sha256"]
            workspace.mkdir()
            with tarfile.open(archive) as source:
                source.extractall(workspace, filter="data")
            baseline_hashes[case] = hashlib.sha256(
                (workspace / ".pyrefly-baseline-full.json").read_bytes()
            ).hexdigest()
            command = [
                "pyrefly",
                "check",
                "--site-package-path",
                sysconfig.get_paths()["purelib"],
                "--baseline",
                ".pyrefly-baseline-full.json",
                "--output-format",
                "json",
                "--color",
                "never",
            ]
            print(f"Checking {case}: {manifest[case]['commit']}", flush=True)
            result = subprocess.run(
                command, cwd=workspace, capture_output=True, text=True
            )
            (output / f"{case}.json").write_text(result.stdout)
            (output / f"{case}.log").write_text(result.stderr)
            if result.returncode not in (0, 1):
                raise RuntimeError(f"{case}: checker exited {result.returncode}")
            errors = json.loads(result.stdout)["errors"]
            diagnostics[case] = Counter(
                (e["path"], e["name"], e["description"])
                for e in errors
                if e["severity"] == "error"
            )
            statuses[case] = result.returncode
            print(f"{case}: {sum(diagnostics[case].values())} errors", flush=True)
            shutil.rmtree(workspace)
    finally:
        if workspace.exists():
            shutil.rmtree(workspace)
        shutil.move(backup, workspace)
    assert baseline_hashes["base"] == baseline_hashes["head"], "Baselines differ"
    new = diagnostics["head"] - diagnostics["base"]
    fixed = diagnostics["base"] - diagnostics["head"]
    summary = {
        "manifest": manifest,
        "baseline_sha256": baseline_hashes["base"],
        "exit_codes": statuses,
        "base_errors": sum(diagnostics["base"].values()),
        "head_errors": sum(diagnostics["head"].values()),
        "new_errors": [{"diagnostic": k, "count": v} for k, v in sorted(new.items())],
        "fixed_errors": [
            {"diagnostic": k, "count": v} for k, v in sorted(fixed.items())
        ],
    }
    save_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    # This diagnostic job does not replace or suppress the full type-check gate.
    return bool(new)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inside", action="store_true")
    parser.add_argument("--image")
    parser.add_argument("--base")
    parser.add_argument("--head")
    parser.add_argument("--output", default="perf_eval_results/typecheck-comparison")
    args = parser.parse_args()
    if args.inside:
        return inside()
    if not all((args.image, args.base, args.head)):
        parser.error("--image, --base and --head are required")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"image": args.image}
    for case, ref in (("base", args.base), ("head", args.head)):
        commit = subprocess.check_output(["git", "rev-parse", ref], text=True).strip()
        archive = output / f"{case}.tar"
        subprocess.run(["git", "archive", "-o", str(archive), commit], check=True)
        manifest[case] = {
            "commit": commit,
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        }
    subprocess.run(["docker", "pull", "-q", args.image], check=True)
    manifest["image_inspect"] = json.loads(
        subprocess.check_output(
            [
                "docker",
                "image",
                "inspect",
                "--format",
                "{{json .RepoDigests}}",
                args.image,
            ],
            text=True,
        )
    )
    save_json(output / "manifest.json", manifest)
    command = [
        "docker",
        "run",
        "--rm",
        "--init",
        "-w",
        "/tmp",
        "-e",
        "JAX_PLATFORMS=cpu",
        "-e",
        "PYTHONUNBUFFERED=1",
        "-v",
        f"{output}:/comparison",
        "-v",
        f"{Path(__file__).resolve()}:/compare_typecheck.py:ro",
        args.image,
        "python3",
        "/compare_typecheck.py",
        "--inside",
    ]
    try:
        return subprocess.run(command).returncode
    finally:
        for case in ("base", "head"):
            (output / f"{case}.tar").unlink()


if __name__ == "__main__":
    sys.exit(main())
