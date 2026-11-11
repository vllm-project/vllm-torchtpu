# SPDX-License-Identifier: Apache-2.0
"""The host-tier telemetry's container-memory reader (`raiden_connector._cgroup_memory`).

CP3's serving container printed mem_current=n/a: the reader only tried the v2 files at the cgroup root, which a
container that sees the host hierarchy does not have. These tests build fake cgroup trees for each layout the
reader must handle and check the bytes and the reported source.
"""

import pytest

from vllm_torchtpu.offload.raiden_connector import _cgroup_memory

POD = "/kubepods.slice/kubepods-burstable.slice/pod1234/cri-containerd-abcd.scope"


def _write(base, files):
    base.mkdir(parents=True, exist_ok=True)
    for name, value in files.items():
        (base / name).write_text(f"{value}\n")


def _proc(tmp_path, text):
    p = tmp_path / "proc_self_cgroup"
    p.write_text(text)
    return str(p)


def test_v2_namespaced_reads_the_root(tmp_path):
    root = tmp_path / "cg"
    _write(root, {"memory.current": 600, "memory.peak": 700, "memory.max": 966367641600})
    out = _cgroup_memory(str(root), _proc(tmp_path, "0::/\n"))
    assert out == {"current": 600, "peak": 700, "max": 966367641600, "source": "v2:root"}


def test_v2_host_view_reads_the_own_cgroup_not_the_root(tmp_path):
    root = tmp_path / "cg"
    # the host root has no memory.current (as in CP3's serving container); the pod's cgroup does
    _write(root, {"memory.stat": "anon 1"})
    _write(root / POD.lstrip("/"), {"memory.current": 612, "memory.peak": 680, "memory.max": "max"})
    out = _cgroup_memory(str(root), _proc(tmp_path, f"0::{POD}\n"))
    assert out == {"current": 612, "peak": 680, "max": None, "source": "v2:own"}


def test_v2_own_path_preferred_over_a_readable_root(tmp_path):
    root = tmp_path / "cg"
    _write(root, {"memory.current": 1, "memory.peak": 1, "memory.max": 1})
    _write(root / POD.lstrip("/"), {"memory.current": 5, "memory.peak": 6, "memory.max": 7})
    out = _cgroup_memory(str(root), _proc(tmp_path, f"0::{POD}\n"))
    assert out["source"] == "v2:own" and (out["current"], out["peak"], out["max"]) == (5, 6, 7)


def test_v1_own_memory_controller_path(tmp_path):
    root = tmp_path / "cg"
    _write(root / "memory" / POD.lstrip("/"), {
        "memory.usage_in_bytes": 300, "memory.max_usage_in_bytes": 400, "memory.limit_in_bytes": 900})
    proc = _proc(tmp_path, f"12:cpu,cpuacct:{POD}\n7:memory:{POD}\n0::/\n")
    out = _cgroup_memory(str(root), proc)
    assert out == {"current": 300, "peak": 400, "max": 900, "source": "v1:own"}


def test_v1_root_and_unlimited_limit(tmp_path):
    root = tmp_path / "cg"
    _write(root / "memory", {
        "memory.usage_in_bytes": 30, "memory.max_usage_in_bytes": 40,
        "memory.limit_in_bytes": 9223372036854771712})
    out = _cgroup_memory(str(root), _proc(tmp_path, "7:memory:/\n"))
    assert out == {"current": 30, "peak": 40, "max": None, "source": "v1:root"}


def test_nothing_readable(tmp_path):
    out = _cgroup_memory(str(tmp_path / "absent"), str(tmp_path / "no_proc"))
    assert out == {"current": None, "peak": None, "max": None, "source": "none"}


@pytest.mark.parametrize("text", ["garbage\n", "", "0::\n"])
def test_unparseable_proc_falls_back_to_the_root(tmp_path, text):
    root = tmp_path / "cg"
    _write(root, {"memory.current": 9, "memory.peak": 10, "memory.max": 11})
    assert _cgroup_memory(str(root), _proc(tmp_path, text))["source"] == "v2:root"
