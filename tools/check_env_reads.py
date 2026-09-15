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
"""Fail when a new environment variable is read outside ``envs.py``.

``envs.py`` is meant to read as the complete list of knobs this package
understands. Every direct ``os.environ.get`` / ``os.getenv`` elsewhere makes
that false: the knob becomes undiscoverable, its default lives at the read
site, and each site hand-rolls its own parsing.

This is a ratchet, not a wall. ``ENV_READ_BASELINE`` records the reads that
already existed, so the existing ones stay put until someone migrates them,
while a newly added read fails the hook. Migrating a read means deleting its
line from the baseline; the hook checks for stale entries too, so the
baseline cannot drift upward silently.

Run directly to see the current state:

    tools/check_env_reads.py            # check
    tools/check_env_reads.py --update   # rewrite the baseline
"""
from __future__ import annotations

import argparse
import ast
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PACKAGE = REPO_ROOT / "src" / "vllm_torchtpu"
BASELINE_PATH = REPO_ROOT / "tools" / "env_reads_baseline.txt"

# `envs.py` is the registry itself; `env_override.py` layers derived defaults
# on top of it before anything reads them. Both must touch os.environ.
# Matched by full path, not basename, so a future `foo/envs.py` elsewhere in
# the package does not silently inherit the exemption.
EXEMPT_PATHS = {
    PACKAGE / "envs.py",
    PACKAGE / "env_override.py",
}

# Process bootstrap, not user configuration. A name belongs here when this
# package (or torchrun / Ray / the TPU runtime) writes it to os.environ at
# startup so child processes inherit it, and later reads it back. An envs.py
# entry would misdescribe such a name: "unset" does not mean "use the
# default", it means "not computed yet", and the read sites need semantics no
# builder has:
#
#   __init__.py             os.environ["LOCAL_RANK"] must raise when unset;
#                           a worker silently becoming rank 0 is worse.
#   model_loader_patches.py chains LOCAL_RANK -> RANK -> "0".
#   tpu_worker.py           falls back to pc.world_size, a runtime value, and
#                           os.environ["TORCH_TPU_DP_MASTER_PORT"] must raise.
#   tpu_platform.py         os.environ.pop("TORCH_TPU_DP_SIZE") consumes the
#                           value; the registry has no equivalent.
#   raiden_store.py         checks whether PYTHONHASHSEED is set, not its
#                           value, and it is a CPython knob, not one of ours.
#
# The TORCH_TPU_* names and TPU_NUM_HOSTS are computed from the slice shape
# by tpu_platform.py, tpu_mp_multihost.py and the Ray executors, written to
# os.environ, and read back by the workers. An operator may pre-seed some of
# them (GKE sets TORCH_TPU_TOPOLOGY); the package still owns the value.
#
# Only names actually read in this package are listed. Adding a new bootstrap
# read fails this hook with a message pointing here, which is the moment to
# decide whether it belongs in envs.py instead.
BOOTSTRAP_VARS = {
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "PYTHONHASHSEED",
    "RANK",
    "TORCH_TPU_DP_MASTER_ADDR",
    "TORCH_TPU_DP_MASTER_PORT",
    "TORCH_TPU_DP_SIZE",
    "TORCH_TPU_SLICEBUILDER_ADDRESSES",
    "TORCH_TPU_TOPOLOGY",
    "TORCH_TPU_XPROF_SESSION_ID",
    "TPU_NUM_HOSTS",
    "WORLD_SIZE",
}


def _os_aliases(tree: ast.Module) -> set[str]:
    """Local names bound to the ``os`` module: ``os``, plus any
    ``import os as _os`` alias, which this package does use."""
    aliases = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "os":
                    aliases.add(alias.asname or "os")
    return aliases


def _attr_path(node: ast.AST) -> tuple[str, ...]:
    """Dotted name of an attribute chain, e.g. ``os.environ.get``."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return tuple(reversed(parts))


def _literal_or_dynamic(node: ast.AST | None) -> str:
    """A string literal's value, or ``<dynamic>`` for anything else."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return "<dynamic>"


def _read_name(node: ast.AST, aliases: set[str]) -> str | None:
    """The variable this node reads, or None if it is not an env read.

    Three spellings count, and all three are used in this package:
    ``os.getenv(X)``, ``os.environ.get(X)`` and ``os.environ[X]``. The
    subscript form is only a read in a Load context -- ``os.environ[X] = v``
    and ``del os.environ[X]`` are writes, which this hook does not police.
    """
    if isinstance(node, ast.Call):
        path = _attr_path(node.func)
        if not path or path[0] not in aliases:
            return None
        if path[1:] in (("getenv", ), ("environ", "get")):
            return _literal_or_dynamic(node.args[0] if node.args else None)
        return None

    if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
        path = _attr_path(node.value)
        if len(path) == 2 and path[0] in aliases and path[1] == "environ":
            return _literal_or_dynamic(node.slice)
    return None


def find_reads() -> list[str]:
    """Every non-exempt env read, as sorted ``relpath:VAR`` entries."""
    found: set[str] = set()
    for path in sorted(PACKAGE.rglob("*.py")):
        if path in EXEMPT_PATHS:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        aliases = _os_aliases(tree)
        if not aliases:
            continue
        for node in ast.walk(tree):
            name = _read_name(node, aliases)
            if name is None or name in BOOTSTRAP_VARS:
                continue
            rel = path.relative_to(REPO_ROOT).as_posix()
            found.add(f"{rel}:{name}")
    return sorted(found)


def load_baseline() -> list[str]:
    if not BASELINE_PATH.exists():
        return []
    return sorted(line.strip()
                  for line in BASELINE_PATH.read_text().splitlines()
                  if line.strip() and not line.startswith("#"))


def write_baseline(entries: list[str]) -> None:
    header = (
        "# Environment reads that bypass envs.py, as of the last update.\n"
        "# Generated by tools/check_env_reads.py --update.\n"
        "# Shrink this file by moving a knob into envs.py; never grow it.\n")
    BASELINE_PATH.write_text(header + "\n".join(entries) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update",
                        action="store_true",
                        help="rewrite the baseline from the current tree")
    # pre-commit passes the staged filenames; this hook always scans the
    # whole package, so accept and ignore them.
    parser.add_argument("filenames", nargs="*", help=argparse.SUPPRESS)
    args = parser.parse_args()

    found = find_reads()
    if args.update:
        write_baseline(found)
        print(f"baseline updated: {len(found)} entries")
        return 0

    baseline = load_baseline()
    added = sorted(set(found) - set(baseline))
    removed = sorted(set(baseline) - set(found))

    if added:
        print("New environment reads bypass envs.py:\n")
        for entry in added:
            path, _, var = entry.rpartition(":")
            print(f"  {path}  reads {var}")
        print("\nAdd an entry to src/vllm_torchtpu/envs.py and read it as "
              "`envs.YOUR_VAR` instead, so the knob is discoverable, typed "
              "and has one documented default.\nIf this is genuine process "
              "bootstrap (set per worker at spawn), add it to BOOTSTRAP_VARS "
              "in tools/check_env_reads.py.")
        return 1

    if removed:
        print("These baseline entries no longer exist (nice):\n")
        for entry in removed:
            print(f"  {entry}")
        print("\nRun `tools/check_env_reads.py --update` to shrink the "
              "baseline, and commit it.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
