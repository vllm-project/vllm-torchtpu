# SPDX-License-Identifier: Apache-2.0
"""q38 HOSTTIER-D D1: container memory telemetry (core/memstat.py)."""

import json
import threading
import types

import pytest

from vllm_torchtpu import envs
from vllm_torchtpu.compilation import tpu_compiler
from vllm_torchtpu.core import memstat

V2_STAT = """anon 472446402560
file 85899345920
kernel 4294967296
kernel_stack 104857600
pagetables 1073741824
sock 268435456
shmem 77309411328
file_mapped 2147483648
file_dirty 0
active_file 6442450944
inactive_file 2147483648
slab 2684354560
"""

V1_STAT = """cache 85899345920
rss 472446402560
shmem 77309411328
mapped_file 2147483648
active_file 6442450944
inactive_file 2147483648
"""


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_flag_default_off_and_compile_ignored():
    assert envs.environment_variables["VLLM_TPU_MEMSTAT_INTERVAL_S"]() == 0
    assert "VLLM_TPU_MEMSTAT_INTERVAL_S" in tpu_compiler._TPU_COMPILE_ENV_IGNORED


def test_flag_reads_env(monkeypatch):
    monkeypatch.setenv("VLLM_TPU_MEMSTAT_INTERVAL_S", "120")
    assert envs.VLLM_TPU_MEMSTAT_INTERVAL_S == 120
    monkeypatch.setenv("VLLM_TPU_MEMSTAT_INTERVAL_S", "-1")
    with pytest.raises(Exception):
        envs.VLLM_TPU_MEMSTAT_INTERVAL_S


def test_cgroup_v2(tmp_path):
    _write(tmp_path / "memory.stat", V2_STAT)
    _write(tmp_path / "memory.current", "558345748480\n")
    _write(tmp_path / "memory.max", "966367641600\n")
    out = memstat.read_cgroup(str(tmp_path))
    assert out["version"] == 2
    assert out["current"] == 558345748480
    assert out["max"] == 966367641600
    assert out["peak"] is None  # memory.peak absent on older kernels
    assert out["anon"] == 472446402560
    assert out["file"] == 85899345920
    assert out["shmem"] == 77309411328
    assert out["kernel"] == 4294967296
    assert out["slab"] == 2684354560
    assert out["sock"] == 268435456
    assert out["file_mapped"] == 2147483648
    assert out["active_file"] == 6442450944
    assert out["inactive_file"] == 2147483648
    assert out["pagetables"] == 1073741824


def test_cgroup_v2_unlimited_max(tmp_path):
    _write(tmp_path / "memory.stat", V2_STAT)
    _write(tmp_path / "memory.max", "max\n")
    assert memstat.read_cgroup(str(tmp_path))["max"] is None


def test_cgroup_v1_maps_names(tmp_path):
    _write(tmp_path / "memory" / "memory.stat", V1_STAT)
    _write(tmp_path / "memory" / "memory.usage_in_bytes", "558345748480\n")
    _write(tmp_path / "memory" / "memory.limit_in_bytes", "966367641600\n")
    out = memstat.read_cgroup(str(tmp_path))
    assert out["version"] == 1
    assert out["current"] == 558345748480
    assert out["anon"] == 472446402560
    assert out["file"] == 85899345920
    assert out["shmem"] == 77309411328
    assert out["file_mapped"] == 2147483648
    assert out["kernel"] is None


def test_tmpfs_only_and_used_bytes(tmp_path):
    mounts = tmp_path / "mounts"
    mounts.write_text(
        "overlay / overlay rw 0 0\n"
        "tmpfs /dev/shm tmpfs rw,nosuid 0 0\n"
        "tmpfs /tmp tmpfs rw 0 0\n"
        "/dev/sda1 /vllm-cache ext4 rw 0 0\n"
        "tmpfs /root/.cache tmpfs rw 0 0\n"
        "tmpfs /tmp tmpfs rw 0 0\n"
    )
    sizes = {"/dev/shm": (1000, 400), "/tmp": (1000, 990), "/root/.cache": (500, 500)}

    def fake_statvfs(path):
        blocks, bfree = sizes[path]
        return types.SimpleNamespace(f_blocks=blocks, f_bfree=bfree, f_frsize=4096)

    out = memstat.read_tmpfs(str(mounts), fake_statvfs)
    assert out == {"/dev/shm": 600 * 4096, "/tmp": 10 * 4096, "/root/.cache": 0}


def _proc(root, pid, name, rss_kb, anon_kb, file_kb, shmem_kb):
    _write(
        root / str(pid) / "status",
        f"Name:\t{name}\nVmRSS:\t{rss_kb} kB\nRssAnon:\t{anon_kb} kB\n"
        f"RssFile:\t{file_kb} kB\nRssShmem:\t{shmem_kb} kB\n",
    )


def test_procs_sorted_bytes_and_totals(tmp_path):
    _proc(tmp_path, 10, "python3", 1000, 900, 80, 20)
    _proc(tmp_path, 11, "VllmWorker-0", 5000, 4000, 600, 400)
    _proc(tmp_path, 12, "kthreadish", 0, 0, 0, 0)  # no RSS: skipped
    (tmp_path / "self").mkdir()  # non-numeric entries are ignored
    out = memstat.read_procs(str(tmp_path), top=1)
    assert out["n"] == 2
    assert out["totals"] == {
        "rss": 6000 * 1024,
        "anon": 4900 * 1024,
        "file": 680 * 1024,
        "shmem": 420 * 1024,
    }
    assert [p["name"] for p in out["top"]] == ["VllmWorker-0"]
    assert out["top"][0]["anon"] == 4000 * 1024


def test_snapshot_reports_failed_part_and_keeps_others(tmp_path):
    _proc(tmp_path / "proc", 7, "python3", 10, 10, 0, 0)
    snap = memstat.snapshot(
        cgroup_root=str(tmp_path / "missing"),
        mounts_path=str(tmp_path / "missing_mounts"),
        proc_root=str(tmp_path / "proc"),
    )
    assert snap["cgroup"] is None and snap["tmpfs"] is None
    assert snap["procs"]["n"] == 1
    assert len(snap["errors"]) == 2
    assert snap["errors"][0].startswith("cgroup: ")


def test_format_line_is_prefixed_json():
    line = memstat.format_line({"t": 1.0, "errors": [], "cgroup": {"anon": 5}})
    assert line.startswith("Q38_MEMSTAT ")
    assert json.loads(line[len("Q38_MEMSTAT "):])["cgroup"]["anon"] == 5


@pytest.fixture
def fresh(monkeypatch):
    monkeypatch.setattr(memstat, "_thread", None)
    stop = threading.Event()
    yield stop
    stop.set()
    t = memstat._thread
    if t is not None:
        t.join(timeout=5)


def _cfg(rank_local=None, rank=0, size_local=1):
    pc = types.SimpleNamespace(
        data_parallel_rank_local=rank_local,
        data_parallel_rank=rank,
        data_parallel_size_local=size_local,
    )
    return types.SimpleNamespace(parallel_config=pc)


def test_off_by_default_starts_nothing(monkeypatch, fresh):
    monkeypatch.delenv("VLLM_TPU_MEMSTAT_INTERVAL_S", raising=False)
    assert memstat.maybe_start_memstat(_cfg(0), emit=lambda s: None, stop=fresh) is None
    assert memstat._thread is None


def test_only_local_dp_rank_zero_logs(monkeypatch, fresh):
    monkeypatch.setenv("VLLM_TPU_MEMSTAT_INTERVAL_S", "3600")
    assert memstat.maybe_start_memstat(_cfg(1), emit=lambda s: None, stop=fresh) is None
    # Without a local rank, the rank modulo the local DP size decides (O4: 4 per pod).
    assert memstat.maybe_start_memstat(_cfg(None, rank=5, size_local=4), emit=lambda s: None, stop=fresh) is None
    assert memstat._thread is None


def test_on_emits_first_line_at_start_and_is_idempotent(monkeypatch, fresh):
    monkeypatch.setenv("VLLM_TPU_MEMSTAT_INTERVAL_S", "3600")
    lines = []
    got = threading.Event()

    def emit(s):
        lines.append(s)
        got.set()

    t = memstat.maybe_start_memstat(_cfg(None, rank=4, size_local=4), emit=emit, stop=fresh)
    assert t is not None and t.daemon
    assert got.wait(10)
    assert lines[0].startswith("Q38_MEMSTAT ")
    assert memstat.maybe_start_memstat(_cfg(0), emit=emit, stop=fresh) is t
