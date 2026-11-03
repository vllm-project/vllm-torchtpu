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
"""Run pytest in isolated/bounded subprocess chunks and merge JUnit XML.

When Buildkite Test Engine (`bktec`) assigns many test files to a single shard:
1. On CPU, running multiple Pallas-interpret / XLA:CPU modules inside one
   long-lived `pytest` process accumulates LLVM MCJIT / XLA:CPU native arenas
   until the container hits cgroup OOM (`exit code 137`).
2. On TPU (`has_vfio=True`), worker/engine-spawning tests (`tests/entrypoints`,
   `spawns_tpu_workers`, `prepare_tpu_environment`, `torch.distributed.run`)
   must run in isolated child processes *before* any in-process `torch_tpu` or
   `jax` test claims `/dev/vfio`, with `/dev/vfio` release verification between
   active chunks so consecutive `START_SESSION` calls do not race GCE teardown.

Running isolated/bounded chunks in fresh child `pytest` processes reclaims 100%
of native heap and closes `/dev/vfio` file descriptors at every chunk boundary
while producing a single merged JUnit XML file at `--junit-xml` for `bktec`.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import fcntl
import functools
import hashlib
import importlib.util
import os
import signal
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.distributed.tpu_test_utils import (  # noqa: E402
    has_tpu_vfio_devices,
    wait_for_vfio_idle,
)

_SPAWN_SYMBOLS: frozenset[str] = frozenset(
    {
        "spawns_tpu_workers",
        "prepare_tpu_environment",
        "set_tpu_launch_env",
        "run_in_isolated_process",
        "run_tpu_worker_session",
    }
)

_CPU_HEAVY_IMPORT_SUBSTRINGS: tuple[str, ...] = (
    "pallas",
    "streamindex",
    "pooled_qsa",
    "qsa_rpa",
    "block_sparse_gqa_prefill",
    "dcp2_harness",
    "test_qwen4_exp_shared_stream_indexer",
)


def _normalize_selector_path(selector: str, repo_root: Path) -> tuple[str, Path]:
    file_part = selector.split("::", 1)[0]
    candidate = Path(file_part)
    resolved = candidate if candidate.is_absolute() else (repo_root / candidate)
    try:
        rel_posix = resolved.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        rel_posix = Path(os.path.normpath(file_part)).as_posix().lstrip("./")
    return rel_posix, resolved


@functools.lru_cache(maxsize=2048)
def _analyze_test_file(resolved_path: str) -> tuple[bool, bool]:
    """Inspect AST (ignoring comments/docstrings) for spawn & Pallas usage."""
    path = Path(resolved_path)
    if not path.is_file():
        return False, False
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=resolved_path)
    except (OSError, SyntaxError):
        return False, False

    code_symbols: set[str] = set()
    seq_str_literals: set[str] = set()
    import_names: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            code_symbols.add(node.id)
        elif isinstance(node, ast.Attribute):
            code_symbols.add(node.attr)
        elif isinstance(node, (ast.List, ast.Tuple)):
            for elt in node.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    seq_str_literals.add(elt.value)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                import_names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod:
                import_names.add(mod)
            for alias in node.names:
                import_names.add(alias.name)
                if mod:
                    import_names.add(f"{mod}.{alias.name}")

    spawns_workers = bool(code_symbols & _SPAWN_SYMBOLS) or (
        "torch.distributed.run" in seq_str_literals
    )
    cpu_heavy_kernel = any(
        token in imp for imp in import_names for token in _CPU_HEAVY_IMPORT_SUBSTRINGS
    )
    return spawns_workers, cpu_heavy_kernel


def classify_selector(
    selector: str,
    *,
    has_vfio: bool,
    repo_root: Path | None = None,
) -> str:
    """Classify selector as ``spawn_isolated``, ``cpu_heavy_isolated``, or ``batch``."""
    root = _REPO_ROOT if repo_root is None else repo_root
    rel_posix, resolved = _normalize_selector_path(selector, root)
    spawns_workers, cpu_heavy_kernel = _analyze_test_file(str(resolved.resolve()))

    if has_vfio:
        if rel_posix.startswith(("tests/entrypoints/", "tests/e2e/")) or spawns_workers:
            return "spawn_isolated"
        return "batch"

    if rel_posix.startswith("tests/kernels/") and cpu_heavy_kernel:
        return "cpu_heavy_isolated"
    return "batch"


def plan_chunks(
    selectors: list[str],
    chunk_size: int,
    *,
    has_vfio: bool,
    repo_root: Path | None = None,
) -> list[list[str]]:
    """Order and partition selectors into safe subprocess chunks."""
    if chunk_size <= 0:
        raise ValueError(
            f"--chunk-size must be a positive integer >= 1, got {chunk_size}; "
            "pass e.g. --chunk-size 6."
        )
    root = _REPO_ROOT if repo_root is None else repo_root
    clean = [s for s in selectors if s.strip()]
    if not clean:
        return []

    if has_vfio:
        spawn_items: list[tuple[str, str]] = []
        batch_items: list[tuple[str, str]] = []
        for sel in clean:
            rel_posix, _ = _normalize_selector_path(sel, root)
            kind = classify_selector(sel, has_vfio=True, repo_root=root)
            if kind == "spawn_isolated":
                spawn_items.append((rel_posix, sel))
            else:
                batch_items.append((rel_posix, sel))
        spawn_items.sort(key=lambda pair: pair[0])
        batch_items.sort(key=lambda pair: pair[0])
        ordered_batch = [sel for _, sel in batch_items]
        return [[sel] for _, sel in spawn_items] + [
            ordered_batch[i : i + chunk_size]
            for i in range(0, len(ordered_batch), chunk_size)
        ]

    heavy_by_file: dict[str, list[str]] = defaultdict(list)
    batch_items = []
    for sel in clean:
        rel_posix, _ = _normalize_selector_path(sel, root)
        kind = classify_selector(sel, has_vfio=False, repo_root=root)
        if kind == "cpu_heavy_isolated":
            heavy_by_file[rel_posix].append(sel)
        else:
            batch_items.append((rel_posix, sel))

    chunks: list[list[str]] = []
    for rel_posix in sorted(heavy_by_file):
        file_sels = heavy_by_file[rel_posix]
        for i in range(0, len(file_sels), chunk_size):
            chunks.append(file_sels[i : i + chunk_size])

    batch_items.sort(key=lambda pair: pair[0])
    ordered_batch = [sel for _, sel in batch_items]
    chunks.extend(
        ordered_batch[i : i + chunk_size]
        for i in range(0, len(ordered_batch), chunk_size)
    )
    return chunks


def _fsync_directory(dir_path: str) -> None:
    with contextlib.suppress(OSError):
        dir_fd = os.open(dir_path, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)


def _write_merged_junit_xml(chunk_xmls: list[str], output_path: str) -> None:
    root = ET.Element("testsuites")
    for xml_path in chunk_xmls:
        if not os.path.exists(xml_path):
            continue
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError:
            continue
        elem = tree.getroot()
        if elem.tag == "testsuites":
            for child in list(elem):
                root.append(child)
        elif elem.tag == "testsuite":
            root.append(elem)

    abs_out = os.path.abspath(output_path)
    out_dir = os.path.dirname(abs_out) or "."
    os.makedirs(out_dir, exist_ok=True)
    lock_digest = hashlib.sha256(abs_out.encode("utf-8")).hexdigest()[:24]
    lock_path = os.path.join(
        tempfile.gettempdir(), f"run_pytest_chunked_{lock_digest}.lock"
    )
    with open(lock_path, "a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=out_dir,
                prefix=f".{os.path.basename(abs_out)}.",
                suffix=".tmp",
            )
            try:
                with os.fdopen(fd, "wb") as f:
                    ET.ElementTree(root).write(
                        f, encoding="utf-8", xml_declaration=True
                    )
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, abs_out)
                _fsync_directory(out_dir)
            except BaseException:
                with contextlib.suppress(OSError):
                    os.unlink(tmp_path)
                raise
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if "--" in raw_args:
        sep = raw_args.index("--")
        wrapper_args, post_sep_files = raw_args[:sep], raw_args[sep + 1 :]
    else:
        wrapper_args, post_sep_files = raw_args, []

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=6,
        help="Maximum number of batch test selectors per pytest subprocess (e.g. 6).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-test pytest --timeout in seconds (e.g. 600).",
    )
    parser.add_argument(
        "--mark",
        type=str,
        default="",
        help="Pytest -m marker expression (e.g. 'not tpu_test and not nightly').",
    )
    parser.add_argument(
        "--junit-xml",
        type=str,
        default="",
        help="Output path for the merged JUnit XML report.",
    )
    parser.add_argument(
        "selectors",
        nargs="*",
        help="Test file or nodeid selectors to execute in bounded chunks.",
    )
    args = parser.parse_intermixed_args(wrapper_args)

    if args.chunk_size <= 0:
        raise ValueError(
            f"--chunk-size must be a positive integer >= 1, got {args.chunk_size}; "
            "pass e.g. --chunk-size 6."
        )

    files = [f for f in [*args.selectors, *post_sep_files] if f.strip()]
    if not files:
        if args.junit_xml:
            _write_merged_junit_xml([], args.junit_xml)
        return 5

    has_vfio = has_tpu_vfio_devices()
    has_pytest_timeout = importlib.util.find_spec("pytest_timeout") is not None
    settle_s = float(os.environ.get("PYTEST_CHUNK_TPU_SETTLE_SECONDS", "6.0"))
    chunks = plan_chunks(files, args.chunk_size, has_vfio=has_vfio)
    any_collected = False
    first_failure_rc: int | None = None

    with tempfile.TemporaryDirectory(prefix="pytest_chunks_") as tmpdir:
        chunk_xmls: list[str] = []
        for idx, chunk in enumerate(chunks):
            chunk_xml = os.path.join(tmpdir, f"chunk_{idx:04d}.xml")
            chunk_xmls.append(chunk_xml)
            cmd = [
                sys.executable,
                "-m",
                "pytest",
                "-sv",
            ]
            if args.timeout > 0 and has_pytest_timeout:
                cmd.append(f"--timeout={args.timeout}")
            if args.mark:
                cmd.extend(["-m", args.mark])
            if args.junit_xml:
                cmd.append(f"--junit-xml={chunk_xml}")
            cmd.extend(chunk)

            print(
                f"[run_pytest_chunked] chunk {idx + 1}/{len(chunks)} "
                f"({len(chunk)} selectors): {chunk[0]}"
                + (f" .. {chunk[-1]}" if len(chunk) > 1 else ""),
                flush=True,
            )
            proc = subprocess.Popen(cmd, start_new_session=True)
            try:
                rc = proc.wait()
            except BaseException:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                raise
            finally:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(proc.pid, signal.SIGKILL)

            if rc == 0:
                any_collected = True
            elif rc == 5:
                pass
            else:
                norm_rc = rc if rc > 0 else (128 + (-rc))
                print(
                    f"[run_pytest_chunked] chunk {idx + 1}/{len(chunks)} "
                    f"failed with exit code {norm_rc}",
                    flush=True,
                )
                if first_failure_rc is None:
                    first_failure_rc = norm_rc

            if has_vfio and rc != 5 and idx + 1 < len(chunks):
                wait_for_vfio_idle(timeout_s=30.0, settle_s=settle_s)

        if args.junit_xml:
            _write_merged_junit_xml(chunk_xmls, args.junit_xml)

    if first_failure_rc is not None:
        return first_failure_rc
    return 0 if any_collected else 5


if __name__ == "__main__":
    sys.exit(main())
