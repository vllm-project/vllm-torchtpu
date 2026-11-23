#!/usr/bin/env python3
"""Summarize an fnrec recording directory for the vllm-torchtpu experiment.

Reads the per-process `fn.*.txt` files and `pytest.*.jsonl` session files
fnrec writes, and prints whether each thing the experiment checks held.
Not for merging.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys


def parse_process(path: pathlib.Path) -> dict:
    proc = {
        "file": path.name,
        "header": {},
        "root": None,
        "tests_root": None,
        "functions": set(),
        "lines": 0,
        "packages": set(),
        "end": None,
        "errors": [],
    }
    for raw in path.read_text(errors="replace").splitlines():
        if not raw:
            continue
        parts = raw.split("\t")
        tag = parts[0]
        if tag == "#start":
            for field in parts[1:]:
                key, _, value = field.partition("=")
                proc["header"][key] = value
        elif tag == "#root" and len(parts) > 1:
            proc["root"] = parts[1]
        elif tag == "#tests" and len(parts) > 1:
            proc["tests_root"] = parts[1]
        elif tag == "#ln":
            proc["lines"] += 1
        elif tag == "#pkg" and len(parts) > 1:
            proc["packages"].add(parts[1])
        elif tag == "#end":
            proc["end"] = dict(
                field.partition("=")[::2] for field in parts[1:] if "=" in field
            )
        elif tag == "#error":
            proc["errors"].append(raw)
        elif not tag.startswith("#") and len(parts) == 3:
            proc["functions"].add((parts[0], parts[1]))
    return proc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=pathlib.Path)
    ap.add_argument("--baseline-seconds", type=float, default=0.0)
    ap.add_argument("--recorded-seconds", type=float, default=0.0)
    ap.add_argument("--baseline-exit", type=int, default=0)
    ap.add_argument("--recorded-exit", type=int, default=0)
    ap.add_argument("--json", type=pathlib.Path)
    args = ap.parse_args(argv)

    procs = [parse_process(p) for p in sorted(args.dir.glob("fn.*.txt"))]
    pids = {p["header"].get("pid") for p in procs}
    roots = {p["root"] for p in procs if p["root"]}
    tests_roots = {p["tests_root"] for p in procs if p["tests_root"]}

    package_funcs: set[tuple[str, str]] = set()
    test_funcs: set[tuple[str, str]] = set()
    for p in procs:
        for filename, qualname in p["functions"]:
            if p["root"] and filename.startswith(p["root"]):
                package_funcs.add((filename, qualname))
            elif p["tests_root"] and filename.startswith(p["tests_root"]):
                test_funcs.add((filename, qualname))

    children = [p for p in procs if p["header"].get("ppid") in pids]
    forks = [p for p in procs if p["header"].get("origin", "").startswith("fork")]
    recording = [p for p in procs if p["root"]]
    ended = [p for p in recording if p["end"] is not None]
    errors = sum(int(p["end"].get("errors", 0)) for p in ended) + sum(
        len(p["errors"]) for p in procs
    )

    sessions = collections.Counter()
    for f in sorted(args.dir.glob("pytest.*.jsonl")):
        for line in f.read_text(errors="replace").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("event") != "session":
                continue
            for key in ("selected", "passed", "failed", "skipped", "errors"):
                if isinstance(row.get(key), int):
                    sessions[key] += row[key]

    by_file = collections.Counter()
    for filename, _ in package_funcs:
        root = next((r for r in roots if filename.startswith(r)), "")
        by_file[filename[len(root):]] += 1

    overhead = (
        (args.recorded_seconds - args.baseline_seconds) / args.baseline_seconds * 100
        if args.baseline_seconds > 0
        else None
    )

    checks = {
        "records vllm_torchtpu functions": bool(package_funcs)
        and any("vllm_torchtpu" in r for r in roots),
        "records test-file functions": bool(test_funcs),
        "records child processes": bool(children),
        "no recorder errors": errors == 0,
        "recording processes exited cleanly": bool(recording)
        and len(ended) == len(recording),
        "tests pass with and without the recorder": args.baseline_exit == 0
        and args.recorded_exit == 0,
        "slowdown under 50%": overhead is not None and overhead < 50,
    }

    print("Processes seen:        ", len(procs))
    print("  recording (root set):", len(recording))
    print("  child processes:     ", len(children), f"(forks: {len(forks)})")
    print("  clean exits:         ", f"{len(ended)}/{len(recording)}")
    print("Package root(s):       ", ", ".join(sorted(roots)) or "(none)")
    print("Tests root(s):         ", ", ".join(sorted(tests_roots)) or "(none)")
    print("vllm_torchtpu functions recorded:", len(package_funcs))
    print("test-file functions recorded:    ", len(test_funcs))
    print("lines recorded (#ln):            ", sum(p["lines"] for p in procs))
    print("recorder errors:                 ", errors)
    print("pytest sessions:                 ", dict(sessions) or "(no session files)")
    print(
        "time: baseline %.1fs, recorded %.1fs, slowdown %s"
        % (
            args.baseline_seconds,
            args.recorded_seconds,
            f"{overhead:.0f}%" if overhead is not None else "n/a",
        )
    )
    print("Top recorded files:")
    for name, count in by_file.most_common(12):
        print(f"  {count:5d}  {name}")
    print()
    for name, ok in checks.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "processes": len(procs),
                    "recording": len(recording),
                    "children": len(children),
                    "clean_exits": len(ended),
                    "package_functions": len(package_funcs),
                    "test_functions": len(test_funcs),
                    "errors": errors,
                    "sessions": dict(sessions),
                    "overhead_percent": overhead,
                    "top_files": by_file.most_common(30),
                    "checks": checks,
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
