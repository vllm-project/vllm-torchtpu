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
"""Pytest configuration and shared fixtures for the TPU megakernel test suite.

Megakernel multi-rank CPU interpret tests require 8 virtual XLA host devices
(`--xla_force_host_platform_device_count=8`), whereas other suites in the
repository (such as `tests/experimental/export/test_checkpoint.py`) assume a
single host CPU device per process. To avoid polluting the global XLA host
device count when collected alongside non-megakernel tests in a shared pytest
session:
- Direct invocations targeting only `tests/megakernel*` configure `XLA_FLAGS`
  in `pytest_configure` before JAX initializes its CPU backend and run in-process.
- Mixed invocations leave the parent process's `XLA_FLAGS` untouched and
  transparently stream each `tests/megakernel*` module through an isolated
  8-CPU-device subprocess whenever the parent process has fewer than 8 CPU
  devices.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

from tests.megakernel.megakernel_fakes import TINY_TILES
from vllm_torchtpu.megakernel.exec_spec import ExecSpec

_PLUGIN_NAME = "vllm_megakernel_cpu_xla"
_ENV_ACTIVE = "_VLLM_MK_SUBPROC_ACTIVE"
_ENV_NODEIDS = "_VLLM_MK_SUBPROC_NODEIDS"
_ENV_STREAM_FD = "_VLLM_MK_SUBPROC_FD"

_MK_XLA_FLAGS = (
    (
        "--xla_force_host_platform_device_count",
        "--xla_force_host_platform_device_count=8",
    ),
    (
        "--xla_llvm_disable_expensive_passes",
        "--xla_llvm_disable_expensive_passes=true",
    ),
)


def _with_mk_xla_flags(raw_flags: str) -> str:
    flags = raw_flags
    for prefix, flag in _MK_XLA_FLAGS:
        if prefix not in flags:
            flags = f"{flags} {flag}".strip()
    return flags


def _mk_roots() -> tuple[Path, ...]:
    tests_dir = Path(__file__).resolve().parent.parent
    return (
        (tests_dir / "megakernel").resolve(),
        (tests_dir / "megakernel_runtime").resolve(),
    )


def _is_mk_path(path: Path) -> bool:
    resolved = path.resolve()
    return any(resolved == root or root in resolved.parents for root in _mk_roots())


def _xla_cpu_initialized() -> bool:
    xb = sys.modules.get("jax._src.xla_bridge")
    if xb is None:
        return False
    backends = getattr(xb, "_backends", None)
    return isinstance(backends, dict) and "cpu" in backends


def _targets_only_megakernel(config: pytest.Config) -> bool:
    if not config.args:
        return False
    cwd = Path(str(config.invocation_params.dir)).resolve()
    for raw in config.args:
        path_part = raw.split("::", 1)[0]
        if not path_part or not _is_mk_path(cwd / path_part):
            return False
    return True


def _serialize_report(report: pytest.TestReport) -> dict[str, Any]:
    longrepr: Any
    if report.longrepr is None:
        longrepr = None
    elif isinstance(report.longrepr, tuple) and len(report.longrepr) == 3:
        longrepr = [
            str(report.longrepr[0]),
            int(report.longrepr[1]),
            str(report.longrepr[2]),
        ]
    else:
        longrepr = report.longreprtext
    data: dict[str, Any] = {
        "when": report.when,
        "outcome": report.outcome,
        "longrepr": longrepr,
        "duration": report.duration,
        "sections": [[str(k), str(v)] for k, v in report.sections],
    }
    if hasattr(report, "wasxfail"):
        data["wasxfail"] = str(report.wasxfail)
    return data


def _deserialize_report(item: pytest.Item, raw: dict[str, Any]) -> pytest.TestReport:
    lr_raw = raw["longrepr"]
    longrepr: None | tuple[str, int, str] | str
    if lr_raw is None:
        longrepr = None
    elif isinstance(lr_raw, list) and len(lr_raw) == 3:
        longrepr = (str(lr_raw[0]), int(lr_raw[1]), str(lr_raw[2]))
    else:
        longrepr = str(lr_raw)
    rep = pytest.TestReport(
        nodeid=item.nodeid,
        location=item.location,
        keywords=item.keywords,
        outcome=raw["outcome"],
        longrepr=longrepr,
        when=raw["when"],
        sections=[(str(s[0]), str(s[1])) for s in raw.get("sections", ())],
        duration=float(raw.get("duration", 0.0)),
    )
    if "wasxfail" in raw:
        rep.wasxfail = raw["wasxfail"]  # type: ignore[attr-defined]
    return rep


class _ChildStreamPlugin:
    def __init__(self, nodeids_file: str, stream_fd: int) -> None:
        with open(nodeids_file, encoding="utf-8") as f:
            self._wanted: list[str] = json.load(f)
        self._wanted_set = set(self._wanted)
        self._stream = os.fdopen(stream_fd, "w", encoding="utf-8", buffering=1)
        self._pending: dict[str, list[dict[str, Any]]] = {}

    def pytest_collection_modifyitems(
        self, config: pytest.Config, items: list[pytest.Item]
    ) -> None:
        by_id: dict[str, list[pytest.Item]] = {}
        for it in items:
            if it.nodeid in self._wanted_set:
                by_id.setdefault(it.nodeid, []).append(it)
        ordered: list[pytest.Item] = []
        for nid in self._wanted:
            bucket = by_id.get(nid)
            if bucket:
                ordered.append(bucket.pop(0))
        items[:] = ordered

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        self._pending.setdefault(report.nodeid, []).append(_serialize_report(report))

    def pytest_runtest_logfinish(
        self, nodeid: str, location: tuple[str, int | None, str]
    ) -> None:
        payload = {
            "nodeid": nodeid,
            "reports": self._pending.pop(nodeid, []),
        }
        self._stream.write(json.dumps(payload) + "\n")
        self._stream.flush()

    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        self._stream.close()


class _ModuleRunner:
    def __init__(
        self,
        proc: subprocess.Popen[bytes],
        reader: io.TextIOWrapper,
        log_path: Path,
        nodeids_path: Path,
        remaining: int,
    ) -> None:
        self.proc = proc
        self.reader = reader
        self.log_path = log_path
        self.nodeids_path = nodeids_path
        self.remaining = remaining
        self.buffered: dict[str, list[dict[str, Any]]] = {}

    def next_reports(self, nodeid: str) -> list[dict[str, Any]] | None:
        if nodeid in self.buffered:
            return self.buffered.pop(nodeid)
        while True:
            line = self.reader.readline()
            if not line:
                return None
            msg = json.loads(line)
            if msg["nodeid"] == nodeid:
                return msg["reports"]
            self.buffered[msg["nodeid"]] = msg["reports"]

    def close(self, *, terminate: bool = False) -> tuple[int, str]:
        try:
            self.reader.close()
        finally:
            if terminate and self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
            try:
                rc = self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                rc = self.proc.wait()
            out = (
                self.log_path.read_text(encoding="utf-8", errors="replace")
                if self.log_path.is_file()
                else ""
            )
            self.log_path.unlink(missing_ok=True)
            self.nodeids_path.unlink(missing_ok=True)
        return rc, out


class _ParentIsolationPlugin:
    def __init__(self) -> None:
        self._has_8_cpu: bool | None = None
        self._runners: dict[Path, _ModuleRunner] = {}

    def _check_inproc_8_cpu(self) -> bool:
        if self._has_8_cpu is None:
            import jax

            self._has_8_cpu = len(jax.devices("cpu")) >= 8
        return self._has_8_cpu

    def _get_or_start_runner(self, item: pytest.Item, mod_path: Path) -> _ModuleRunner:
        runner = self._runners.get(mod_path)
        if runner is not None:
            return runner
        session_items = item.session.items
        try:
            start = session_items.index(item)
        except ValueError:
            start = 0
        wanted = [
            it.nodeid
            for it in session_items[start:]
            if Path(it.path).resolve() == mod_path
        ]
        fd_tmp, nodeids_str = tempfile.mkstemp(prefix="mk_nodeids_", suffix=".json")
        with os.fdopen(fd_tmp, "w", encoding="utf-8") as f:
            json.dump(wanted, f)
        r_fd, w_fd = os.pipe()
        fd_log, log_str = tempfile.mkstemp(prefix="mk_subproc_", suffix=".log")
        env = dict(os.environ)
        env["XLA_FLAGS"] = _with_mk_xla_flags(env.get("XLA_FLAGS", ""))
        env[_ENV_ACTIVE] = "1"
        env[_ENV_NODEIDS] = nodeids_str
        env[_ENV_STREAM_FD] = str(w_fd)
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            f"--rootdir={item.config.rootpath}",
            "--tb=short",
            "-q",
        ]
        confcutdir = getattr(item.config.option, "confcutdir", None)
        if confcutdir:
            cmd.append(f"--confcutdir={confcutdir}")
        cmd.append(str(mod_path))
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(item.config.invocation_params.dir),
                env=env,
                stdout=fd_log,
                stderr=subprocess.STDOUT,
                pass_fds=(w_fd,),
            )
        finally:
            os.close(w_fd)
            os.close(fd_log)
        reader = os.fdopen(r_fd, "r", encoding="utf-8")
        runner = _ModuleRunner(
            proc=proc,
            reader=reader,
            log_path=Path(log_str),
            nodeids_path=Path(nodeids_str),
            remaining=len(wanted),
        )
        self._runners[mod_path] = runner
        return runner

    def pytest_runtest_protocol(
        self, item: pytest.Item, nextitem: pytest.Item | None
    ) -> bool | None:
        mod_path = Path(item.path).resolve()
        if not _is_mk_path(mod_path):
            return None
        if self._check_inproc_8_cpu():
            return None

        runner = self._get_or_start_runner(item, mod_path)
        ihook = item.ihook
        ihook.pytest_runtest_logstart(nodeid=item.nodeid, location=item.location)
        try:
            raw_reports = runner.next_reports(item.nodeid)
        except BaseException:
            self._runners.pop(mod_path, None)
            runner.close(terminate=True)
            raise
        runner.remaining -= 1
        rc = 0
        crash_out = ""
        if raw_reports is None or runner.remaining <= 0:
            self._runners.pop(mod_path, None)
            rc, crash_out = runner.close(terminate=raw_reports is None)

        if raw_reports:
            for raw in raw_reports:
                ihook.pytest_runtest_logreport(report=_deserialize_report(item, raw))
        else:
            msg = (
                f"Megakernel 8-CPU-device subprocess exited early (rc={rc}) "
                f"before reporting {item.nodeid}:\n{crash_out[-4000:]}"
            )
            for when, outcome, lr in (
                ("setup", "passed", None),
                ("call", "failed", msg),
                ("teardown", "passed", None),
            ):
                ihook.pytest_runtest_logreport(
                    report=pytest.TestReport(
                        nodeid=item.nodeid,
                        location=item.location,
                        keywords=item.keywords,
                        outcome=outcome,  # type: ignore[arg-type]
                        longrepr=lr,
                        when=when,  # type: ignore[arg-type]
                    )
                )
        ihook.pytest_runtest_logfinish(nodeid=item.nodeid, location=item.location)
        return True

    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        for mod_path in list(self._runners):
            self._runners.pop(mod_path).close(terminate=True)


def register_megakernel_cpu_plugin(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "slow: multi-rank or large-batch CPU interpret tests"
    )
    config.addinivalue_line("markers", "golden: bit-exact golden hash regression tests")
    if config.pluginmanager.has_plugin(_PLUGIN_NAME):
        return
    if os.environ.get(_ENV_ACTIVE) == "1":
        config.pluginmanager.register(
            _ChildStreamPlugin(
                os.environ[_ENV_NODEIDS], int(os.environ[_ENV_STREAM_FD])
            ),
            _PLUGIN_NAME,
        )
        return
    if _targets_only_megakernel(config) and not _xla_cpu_initialized():
        os.environ["XLA_FLAGS"] = _with_mk_xla_flags(os.environ.get("XLA_FLAGS", ""))
    config.pluginmanager.register(_ParentIsolationPlugin(), _PLUGIN_NAME)


def pytest_configure(config: pytest.Config) -> None:
    register_megakernel_cpu_plugin(config)


@pytest.fixture
def fake_ex() -> ExecSpec:
    return ExecSpec(B=4, T=1, R=2, ctx=256, tiles=TINY_TILES, interpret=True)
