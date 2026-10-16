#!/usr/bin/env python3
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
"""Run pyrefly and require its errors to match a baseline one for one.

pyrefly matches a baseline entry on file, error kind and message, and a single
entry hides every error with that key in the file. On its own, that lets a
fixed error come back, or a second copy of a listed error appear, without
failing the check.

This script runs pyrefly with an empty baseline and compares counts per key:

- More errors than entries fails: the extra errors are new, even though the
  same message is already listed for the file.
- More entries than errors fails too: those errors were fixed, and the spare
  entries would hide a later error with the same message. Regenerate the
  baseline to drop them.

``pyrefly check --update-baseline`` writes one entry per error, so a freshly
regenerated baseline always passes.

Usage:

    tools/check_pyrefly_baseline.py --baseline FILE [PYREFLY_ARG ...]

Arguments it does not know are passed to ``pyrefly check``.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import subprocess
import sys
import tempfile

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

Key = tuple[str, str, str]


def error_key(error: dict) -> Key:
    """(repo-relative path, error kind, message), the key pyrefly matches on."""
    path = pathlib.Path(error["path"])
    if path.is_absolute():
        path = path.relative_to(REPO_ROOT)
    return (path.as_posix(), error["name"], error["concise_description"])


def compare(
    errors: list[dict], entries: list[dict]
) -> tuple[dict[Key, tuple[int, int]], dict[Key, tuple[int, int]]]:
    """Keys with more errors than entries, and keys with more entries than
    errors. Each maps to (entries, errors)."""
    found = collections.Counter(error_key(e) for e in errors)
    listed = collections.Counter(error_key(e) for e in entries)
    extra = {k: (listed[k], n) for k, n in found.items() if n > listed[k]}
    stale = {k: (n, found[k]) for k, n in listed.items() if n > found[k]}
    return extra, stale


def run_pyrefly(pyrefly_args: list[str]) -> list[dict]:
    """Every error pyrefly reports, ignoring all baselines."""
    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False
    ) as empty_baseline:
        empty_baseline.write('{"errors": []}\n')
    try:
        result = subprocess.run(
            [
                "pyrefly",
                "check",
                "--baseline",
                empty_baseline.name,
                "--output-format",
                "json",
                *pyrefly_args,
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        os.unlink(empty_baseline.name)
    # pyrefly exits 1 when it reports errors, which is the normal case here.
    # Any other code means it could not run, and stdout is not JSON.
    if result.returncode not in (0, 1):
        raise RuntimeError(f"pyrefly exited {result.returncode}:\n{result.stderr}")
    report = json.loads(result.stdout)
    errors = report["errors"] if isinstance(report, dict) else report
    return [e for e in errors if e.get("severity", "error") == "error"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", required=True, help="baseline JSON file")
    parser.add_argument(
        "--update-command",
        default="pyrefly check --update-baseline",
        help="command that regenerates this baseline, shown on failure",
    )
    args, pyrefly_args = parser.parse_known_args()

    errors = run_pyrefly(pyrefly_args)
    with open(REPO_ROOT / args.baseline) as f:
        entries = json.load(f)["errors"]
    extra, stale = compare(errors, entries)

    if extra:
        lines: dict[Key, list[int]] = collections.defaultdict(list)
        for e in errors:
            lines[error_key(e)].append(e["line"])
        print(f"pyrefly: errors not covered by {args.baseline}. Fix them:")
        for (path, name, message), (listed, found) in sorted(extra.items()):
            at = ", ".join(str(n) for n in sorted(lines[(path, name, message)]))
            print(f"  {path} [{name}] {message}")
            print(f"    listed {listed}, found {found}, at lines {at}")
    if stale:
        print(
            f"pyrefly: entries in {args.baseline} that match no error, because"
            f" the errors were fixed. Delete them with `{args.update_command}`:"
        )
        for (path, name, message), (listed, found) in sorted(stale.items()):
            print(f"  {path} [{name}] {message}")
            print(f"    listed {listed}, found {found}")
    if extra or stale:
        return 1
    print(
        f"pyrefly: {len(errors)} errors, each matched by one entry in {args.baseline}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
