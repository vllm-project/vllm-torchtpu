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
"""Run pytest in bounded file-count subprocess chunks and merge JUnit XML.

When Buildkite Test Engine (`bktec`) assigns 100+ test files with short or
unrecorded historical durations to a single shard, running all of them inside
one long-lived `pytest` process accumulates LLVM MCJIT / XLA CPU / TorchDynamo
and PJRT allocations until the pod hits cgroup OOM (`exit code 137`).

Running bounded chunks in fresh child `pytest` processes reclaims 100% of native
heap and closes `/dev/vfio` file descriptors at every chunk boundary while
producing a single merged JUnit XML file at `--junit-xml` for `bktec`.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import signal
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET


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
    fd, tmp_path = tempfile.mkstemp(
        dir=out_dir,
        prefix=f".{os.path.basename(abs_out)}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "wb") as f:
            ET.ElementTree(root).write(f, encoding="utf-8", xml_declaration=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, abs_out)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


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
        default=12,
        help="Maximum number of test selectors per pytest subprocess (e.g. 12).",
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
            "pass e.g. --chunk-size 12."
        )

    files = [f for f in [*args.selectors, *post_sep_files] if f.strip()]
    if not files:
        if args.junit_xml:
            _write_merged_junit_xml([], args.junit_xml)
        return 5

    chunks = [
        files[i : i + args.chunk_size] for i in range(0, len(files), args.chunk_size)
    ]
    has_vfio = os.path.exists("/dev/vfio")
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
                f"--timeout={args.timeout}",
            ]
            if args.mark:
                cmd.extend(["-m", args.mark])
            if args.junit_xml:
                cmd.append(f"--junit-xml={chunk_xml}")
            cmd.extend(chunk)

            print(
                f"[run_pytest_chunked] chunk {idx + 1}/{len(chunks)} "
                f"({len(chunk)} files)",
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

            if has_vfio and idx + 1 < len(chunks):
                time.sleep(2.0)

        if args.junit_xml:
            _write_merged_junit_xml(chunk_xmls, args.junit_xml)

    if first_failure_rc is not None:
        return first_failure_rc
    return 0 if any_collected else 5


if __name__ == "__main__":
    sys.exit(main())
