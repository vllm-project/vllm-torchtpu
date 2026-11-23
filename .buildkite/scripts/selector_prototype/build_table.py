"""Fold fnrec recordings into a table: CI step -> the functions it ran.

PROTOTYPE. The process-file reader and the health rules are adapted from
vllm-project/ci-infra (Apache-2.0), buildkite/ci_selector/ci_selector/coverage/
model.py at 04dd673a92 (`read_process`, `Stamp.thin`). Simplified: one build,
no Buildkite API, no signing, no line records.

Input is the directory the recording steps uploaded, laid out as
`fnrec/<step key>/shard<N>/` with a `meta.json` and the recorder's files.

    python3 build_table.py ARTIFACTS_DIR --commit SHA --out table.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from changed_funcs import import_time_names, read_at  # noqa: E402

# Where each recorded root lives in the repo. The recorder's root is the
# package directory itself, so its repo path keeps the package name.
PACKAGE_PREFIX = "src/vllm_torchtpu/"
TESTS_PREFIX = "tests/"
TABLE_VERSION = 1


def read_process(path: pathlib.Path) -> dict | None:
    """One fnrec process file, read to EOF. None if it never found our package."""
    root = tests_root = None
    raw: list[tuple[str, str]] = []
    clean_exit = False
    errors = 0
    for line in path.read_text(errors="replace").splitlines():
        parts = line.split("\t")
        tag = parts[0]
        if tag == "#start":
            meta = dict(p.split("=", 1) for p in parts[1:] if "=" in p)
            root = root or meta.get("root") or None
        elif tag == "#root":
            root = parts[1] if len(parts) > 1 and parts[1] else root
        elif tag == "#tests":
            tests_root = parts[1] if len(parts) > 1 and parts[1] else None
        elif tag in ("#stat", "#end"):
            meta = dict(p.split("=", 1) for p in parts[1:] if "=" in p)
            errors += int(meta.get("errors", 0) or 0)
            clean_exit = clean_exit or tag == "#end"
        elif tag == "#error":
            errors += 1
        elif not tag.startswith("#") and len(parts) == 3:
            raw.append((parts[0], parts[1]))
    if root is None:
        return None
    roots = [(root.rstrip("/") + "/", PACKAGE_PREFIX)]
    if tests_root:
        roots.append((tests_root.rstrip("/") + "/", TESTS_PREFIX))
    functions: dict[str, set[str]] = defaultdict(set)
    for filename, qualname in raw:
        for base, prefix in roots:
            if filename.startswith(base):
                functions[prefix + filename[len(base):]].add(qualname)
                break
    return {"functions": functions, "clean_exit": clean_exit, "errors": errors}


def read_sessions(shard_dir: pathlib.Path) -> dict[str, int]:
    """Test outcomes from fnrec_pytest's session lines."""
    totals: dict[str, int] = defaultdict(int)
    for f in shard_dir.glob("pytest.*.jsonl"):
        for line in f.read_text(errors="replace").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("event") != "session":
                continue
            totals["sessions"] += 1
            for key in ("selected", "passed", "failed", "errors", "skipped"):
                if isinstance(row.get(key), int):
                    totals[key] += row[key]
    return dict(totals)


def thin_reasons(health: dict) -> list[str]:
    """Why a row is too weak to skip a step on. Empty means healthy.

    Upstream's rule: a weak row may still say "run", never "skip".
    """
    reasons = []
    if not health["n_functions"]:
        reasons.append("recorded no functions")
    missing = sorted(set(range(health["shards_expected"])) - set(health["shards_seen"]))
    if missing:
        reasons.append(f"shards missing: {missing}")
    if any(rc not in (0, 5) for rc in health["exit_codes"]):
        reasons.append(f"a shard failed (exit codes {health['exit_codes']})")
    if health["kind"] == "pytest":
        if health["tests"].get("failed") or health["tests"].get("errors"):
            reasons.append("tests failed")
        if not health["tests"].get("passed"):
            reasons.append("no test passed")
    elif any(rc != 0 for rc in health["exit_codes"]):
        # A script step has no test count: its exit code is the verdict.
        reasons.append("the step's command failed")
    if not health["clean_exits"]:
        reasons.append("no process exited cleanly")
    return reasons


def build(artifacts: pathlib.Path, repo: pathlib.Path, commit: str) -> dict:
    by_step: dict[str, list[pathlib.Path]] = defaultdict(list)
    for meta_file in sorted(artifacts.rglob("meta.json")):
        meta = json.loads(meta_file.read_text())
        by_step[meta["step"]].append(meta_file.parent)

    rows = {}
    for step, shard_dirs in sorted(by_step.items()):
        functions: dict[str, set[str]] = defaultdict(set)
        health = {
            "shards_seen": [],
            "shards_expected": 0,
            "exit_codes": [],
            "processes": 0,
            "clean_exits": 0,
            "recorder_errors": 0,
            "tests": defaultdict(int),
            "commits": set(),
            "kind": "pytest",
        }
        for shard_dir in shard_dirs:
            meta = json.loads((shard_dir / "meta.json").read_text())
            health["kind"] = meta.get("kind", "pytest")
            health["shards_seen"].append(int(meta["shard"]))
            health["shards_expected"] = max(
                health["shards_expected"], int(meta["shards"])
            )
            # A peer pod torn down once the head finished exits by signal;
            # only pods marked judged speak for the step.
            if meta.get("judged", 1):
                health["exit_codes"].append(int(meta["exit_code"]))
            if meta.get("commit"):
                health["commits"].add(meta["commit"])
            for proc_file in shard_dir.glob("fn.*.txt"):
                proc = read_process(proc_file)
                if proc is None:
                    continue
                health["processes"] += 1
                health["clean_exits"] += proc["clean_exit"]
                health["recorder_errors"] += proc["errors"]
                for path, names in proc["functions"].items():
                    functions[path] |= names
            for key, value in read_sessions(shard_dir).items():
                health["tests"][key] += value

        # Module and class bodies, read off the source at the recording
        # commit: entering one only means the step imported the file.
        import_time = {}
        for path in functions:
            source = read_at(repo, commit, path)
            if source is None:
                continue
            try:
                import_time[path] = sorted(import_time_names(source, path) & functions[path])
            except Exception:
                pass

        health["n_files"] = len(functions)
        health["n_functions"] = sum(len(v) for v in functions.values())
        health["tests"] = dict(health["tests"])
        health["commits"] = sorted(health["commits"])
        health["shards_seen"] = sorted(health["shards_seen"])
        health["thin"] = thin_reasons(health)
        rows[step] = {
            "functions": {p: sorted(n) for p, n in sorted(functions.items())},
            "import_time": import_time,
            "health": health,
        }
    return {"version": TABLE_VERSION, "commit": commit, "rows": rows}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("artifacts", type=pathlib.Path)
    ap.add_argument("--commit", required=True, help="commit the recordings ran at")
    ap.add_argument("--repo", type=pathlib.Path, default=pathlib.Path.cwd())
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("table.json"))
    args = ap.parse_args(argv)
    table = build(args.artifacts, args.repo, args.commit)
    if not table["rows"]:
        sys.exit(f"no recordings (meta.json files) under {args.artifacts}")
    args.out.write_text(json.dumps(table, indent=1, sort_keys=True))
    for step, row in table["rows"].items():
        h = row["health"]
        status = "healthy" if not h["thin"] else "THIN: " + "; ".join(h["thin"])
        print(
            f"{step}: {h['n_functions']} functions in {h['n_files']} files, "
            f"shards {h['shards_seen']}/{h['shards_expected']}, "
            f"{h['processes']} processes, tests {h['tests']}, {status}"
        )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
