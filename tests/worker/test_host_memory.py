"""TPU_HOST_MEMORY_LOG_S: cgroup v1/v2 parsing, the largest-process scan, the
report line, OOM-kill warnings and the one-reporter-per-host lock (all against
a fake sysfs / procfs)."""

import os

from vllm_torchtpu.worker import host_memory

_GIB = 1 << 30


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def _fake_cgroup_v2(root, current, limit="max", oom_kill=0, peak=None):
    _write(f"{root}/memory.current", f"{current}\n")
    _write(f"{root}/memory.max", f"{limit}\n")
    if peak is not None:
        _write(f"{root}/memory.peak", f"{peak}\n")
    _write(
        f"{root}/memory.stat",
        f"anon {600 * _GIB}\nfile {80 * _GIB}\nkernel 123\nshmem {30 * _GIB}\n"
        f"file_mapped {20 * _GIB}\nunevictable {10 * _GIB}\n",
    )
    _write(
        f"{root}/memory.events",
        f"low 0\nhigh 0\nmax 7\noom 1\noom_kill {oom_kill}\noom_group_kill 0\n",
    )


def _fake_process(proc, pid, name, rss_kb=None):
    lines = [f"Name:\t{name}", "State:\tS (sleeping)", f"Pid:\t{pid}"]
    if rss_kb is not None:
        lines.append(f"VmRSS:\t{rss_kb} kB")
    _write(f"{proc}/{pid}/status", "\n".join(lines) + "\n")


def _self_cgroup(tmp_path, text="0::/\n"):
    """A fake /proc/self/cgroup; the default is a private cgroup namespace."""
    path = str(tmp_path / "self_cgroup")
    _write(path, text)
    return path


class _RecordingLogger:
    def __init__(self):
        self.calls = []

    def info(self, msg, *args):
        self.calls.append(("info", msg % args))

    def warning(self, msg, *args):
        self.calls.append(("warning", msg % args))

    def debug(self, msg, *args, **kwargs):
        self.calls.append(("debug", msg % args))


def test_reads_cgroup_v2(tmp_path):
    root = str(tmp_path / "cg")
    _fake_cgroup_v2(root, 800 * _GIB, limit=900 * _GIB, oom_kill=2, peak=850 * _GIB)
    memory = host_memory.read_cgroup_memory(root, _self_cgroup(tmp_path))
    assert memory == host_memory.CgroupMemory(
        current=800 * _GIB,
        limit=900 * _GIB,
        peak=850 * _GIB,
        anon=600 * _GIB,
        file=80 * _GIB,
        shmem=30 * _GIB,
        oom_kills=2,
        file_mapped=20 * _GIB,
        unevictable=10 * _GIB,
    )


def test_cgroup_v2_without_limit_or_peak(tmp_path):
    root = str(tmp_path / "cg")
    _fake_cgroup_v2(root, 5 * _GIB)  # memory.max is "max", no memory.peak
    memory = host_memory.read_cgroup_memory(root, _self_cgroup(tmp_path))
    assert memory is not None
    assert memory.limit is None
    assert memory.peak is None
    line = host_memory.format_report(memory, [])
    assert line.startswith("Host memory: cgroup 5.0 GiB; anon 600.0")
    assert "%" not in line


def test_reads_cgroup_v1(tmp_path):
    root = str(tmp_path / "cg")
    _write(f"{root}/memory/memory.usage_in_bytes", f"{100 * _GIB}\n")
    _write(f"{root}/memory/memory.limit_in_bytes", f"{200 * _GIB}\n")
    _write(f"{root}/memory/memory.max_usage_in_bytes", f"{150 * _GIB}\n")
    _write(
        f"{root}/memory/memory.stat",
        f"cache 1\nrss 2\nshmem 3\ntotal_cache {10 * _GIB}\n"
        f"total_rss {70 * _GIB}\ntotal_shmem {5 * _GIB}\n"
        f"total_mapped_file {4 * _GIB}\ntotal_unevictable {2 * _GIB}\n",
    )
    _write(
        f"{root}/memory/memory.oom_control",
        "oom_kill_disable 0\nunder_oom 0\noom_kill 1\n",
    )
    memory = host_memory.read_cgroup_memory(
        root, _self_cgroup(tmp_path, "12:memory:/\n0::/\n")
    )
    assert memory == host_memory.CgroupMemory(
        current=100 * _GIB,
        limit=200 * _GIB,
        peak=150 * _GIB,
        anon=70 * _GIB,
        file=10 * _GIB,
        shmem=5 * _GIB,
        oom_kills=1,
        file_mapped=4 * _GIB,
        unevictable=2 * _GIB,
    )


def test_cgroup_v1_no_limit_is_none(tmp_path):
    root = str(tmp_path / "cg")
    _write(f"{root}/memory/memory.usage_in_bytes", "4096\n")
    _write(f"{root}/memory/memory.limit_in_bytes", "9223372036854771712\n")
    memory = host_memory.read_cgroup_memory(root, _self_cgroup(tmp_path))
    assert memory is not None
    assert memory.current == 4096
    assert memory.limit is None
    assert memory.oom_kills is None


def test_cgroup_v2_without_namespace_reads_own_path(tmp_path):
    # The mount root is the host's top cgroup, which has no memory.current.
    root = str(tmp_path / "cg")
    _write(f"{root}/cgroup.controllers", "cpu memory\n")
    _fake_cgroup_v2(f"{root}/kubepods/pod1/ctr", 42 * _GIB, limit=900 * _GIB)
    self_cgroup = _self_cgroup(tmp_path, "0::/kubepods/pod1/ctr\n")
    memory = host_memory.read_cgroup_memory(root, self_cgroup)
    assert memory is not None
    assert (memory.current, memory.limit) == (42 * _GIB, 900 * _GIB)


def test_cgroup_v1_without_namespace_prefers_own_path(tmp_path):
    # The v1 root reports the whole host; the container's own path wins.
    root = str(tmp_path / "cg")
    _write(f"{root}/memory/memory.usage_in_bytes", f"{1500 * _GIB}\n")
    own = f"{root}/memory/kubepods/pod1/ctr"
    _write(f"{own}/memory.usage_in_bytes", f"{10 * _GIB}\n")
    _write(f"{own}/memory.limit_in_bytes", f"{900 * _GIB}\n")
    self_cgroup = _self_cgroup(
        tmp_path, "12:cpu,cpuacct:/kubepods/pod1/ctr\n5:memory:/kubepods/pod1/ctr\n"
    )
    memory = host_memory.read_cgroup_memory(root, self_cgroup)
    assert memory is not None
    assert (memory.current, memory.limit) == (10 * _GIB, 900 * _GIB)


def test_no_cgroup(tmp_path):
    assert (
        host_memory.read_cgroup_memory(
            str(tmp_path / "missing"), str(tmp_path / "no_self_cgroup")
        )
        is None
    )
    assert (
        host_memory.format_report(None, []) == "Host memory: no readable memory cgroup"
    )


def test_largest_processes(tmp_path):
    proc = str(tmp_path / "proc")
    _fake_process(proc, 2608, "VLLM::EngineCor", rss_kb=40 * 1024 * 1024)
    _fake_process(proc, 18578, "ray::RayWorkerP", rss_kb=98 * 1024 * 1024)
    _fake_process(proc, 18579, "ray::RayWorkerP", rss_kb=97 * 1024 * 1024)
    _fake_process(proc, 2, "kthreadd")  # kernel thread: no VmRSS
    _fake_process(proc, 77, "bash", rss_kb=4)
    _write(f"{proc}/self/status", "Name:\tnot-a-pid\nVmRSS:\t999999999 kB\n")
    _write(f"{proc}/meminfo", "MemTotal: 1 kB\n")
    assert host_memory.largest_processes(proc, count=3) == [
        (18578, "ray::RayWorkerP", 98 * _GIB),
        (18579, "ray::RayWorkerP", 97 * _GIB),
        (2608, "VLLM::EngineCor", 40 * _GIB),
    ]
    assert host_memory.largest_processes(str(tmp_path / "missing")) == []


def test_format_report():
    memory = host_memory.CgroupMemory(
        current=810 * _GIB,
        limit=900 * _GIB,
        peak=850 * _GIB,
        anon=700 * _GIB,
        file=80 * _GIB,
        shmem=30 * _GIB,
        oom_kills=0,
        file_mapped=12 * _GIB,
        unevictable=8 * _GIB,
    )
    line = host_memory.format_report(
        memory,
        [(18578, "ray::RayWorkerP", 98 * _GIB), (2608, "VLLM::EngineCor", 40 * _GIB)],
    )
    assert line == (
        "Host memory: cgroup 810.0 GiB of 900.0 GiB (90.0%); peak 850.0 GiB; "
        "anon 700.0, file 80.0, mapped 12.0, shmem 30.0, unevictable 8.0 GiB; "
        "oom_kill 0; largest RSS: "
        "pid 18578 ray::RayWorkerP 98.0 GiB, pid 2608 VLLM::EngineCor 40.0 GiB"
    )


def test_one_reporter_per_host_and_takeover(tmp_path):
    lock = str(tmp_path / host_memory.LOCK_FILE_NAME)
    first = host_memory.HostMemoryReporter(60, lock)
    second = host_memory.HostMemoryReporter(60, lock)
    assert first.try_acquire()
    assert first.try_acquire()  # idempotent for the holder
    assert not second.try_acquire()
    assert not second.holds_lock
    first.release()  # what the kernel does when the holder's process exits
    assert second.try_acquire()
    assert not first.try_acquire()
    second.release()


def test_warns_when_oom_kills_grow(tmp_path, monkeypatch):
    root = str(tmp_path / "cg")
    proc = str(tmp_path / "proc")
    _fake_process(proc, 10, "python3", rss_kb=1024)
    recorder = _RecordingLogger()
    monkeypatch.setattr(host_memory, "logger", recorder)
    reporter = host_memory.HostMemoryReporter(
        60,
        str(tmp_path / "lock"),
        cgroup_root=root,
        self_cgroup=_self_cgroup(tmp_path),
        proc=proc,
    )

    _fake_cgroup_v2(root, 700 * _GIB, limit=900 * _GIB, oom_kill=0)
    reporter.report_once()
    _fake_cgroup_v2(root, 899 * _GIB, limit=900 * _GIB, oom_kill=1)
    reporter.report_once()
    reporter.report_once()

    levels = [level for level, _ in recorder.calls]
    assert levels == ["info", "warning", "info"]
    assert "oom_kill 1" in recorder.calls[1][1]
    assert recorder.calls[1][1].endswith(
        "(the cgroup OOM killer fired 1 time(s) since the last report)"
    )


def test_takeover_after_an_oom_kill_warns(tmp_path, monkeypatch):
    root = str(tmp_path / "cg")
    recorder = _RecordingLogger()
    monkeypatch.setattr(host_memory, "logger", recorder)
    _fake_cgroup_v2(root, 500 * _GIB, limit=900 * _GIB, oom_kill=1)
    reporter = host_memory.HostMemoryReporter(
        60,
        str(tmp_path / "lock"),
        cgroup_root=root,
        self_cgroup=_self_cgroup(tmp_path),
        proc=str(tmp_path / "proc"),
    )
    reporter.report_once()
    assert recorder.calls[0][0] == "warning"


def test_maybe_start_is_off_at_zero_and_starts_once(monkeypatch):
    started = []

    class _FakeThread:
        def __init__(self, target, args, name, daemon):
            self.name = name
            self.daemon = daemon

        def start(self):
            started.append((self.name, self.daemon))

    monkeypatch.setattr(host_memory.threading, "Thread", _FakeThread)
    monkeypatch.setattr(host_memory, "_started", False)
    host_memory.maybe_start(0)
    assert started == []
    host_memory.maybe_start(30)
    host_memory.maybe_start(30)
    assert started == [("tpu-host-memory", True)]


def test_logs_through_the_vllm_logger_tree():
    # Only loggers under "vllm" get vLLM's handler; any other name loses its
    # INFO lines to Python's last-resort handler (WARNING and up only).
    assert host_memory.logger.name == "vllm.vllm_torchtpu.worker.host_memory"
