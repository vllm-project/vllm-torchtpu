"""The per-worker socket binding narrows the inherited CPU set to the chip's socket and never widens it."""
import builtins
import io
from unittest import mock

from vllm_torchtpu.worker import tpu_worker


def _run(inherited, node_cpulist, node=1):
    applied = []
    real_open = builtins.open

    def fake_open(path, *a, **k):
        if str(path).endswith(f"/node{node}/cpulist"):
            return io.StringIO(node_cpulist)
        return real_open(path, *a, **k)

    with mock.patch.object(tpu_worker, "_tpu_chip_numa_node", return_value=("0000:c0:00.0", node)), \
         mock.patch("os.sched_getaffinity", return_value=set(inherited)), \
         mock.patch("os.sched_setaffinity", side_effect=lambda pid, cpus: applied.append(set(cpus))), \
         mock.patch("builtins.open", side_effect=fake_open), \
         mock.patch("ctypes.CDLL", side_effect=OSError("no libnuma")):
        tpu_worker._bind_worker_to_chip_socket(4)
    return applied


def test_unrestricted_worker_gets_the_whole_socket():
    applied = _run(inherited=range(0, 224), node_cpulist="56-111,168-223\n")
    assert applied == [set(range(56, 112)) | set(range(168, 224))]


def test_reservation_is_kept_by_intersection():
    # a launcher that reserved cpus 0-7 and 56-63 for itself: the worker keeps only its socket's share of the rest
    reserved_out = set(range(0, 224)) - set(range(0, 8)) - set(range(56, 64))
    applied = _run(inherited=reserved_out, node_cpulist="56-111,168-223\n")
    assert applied == [(set(range(64, 112)) | set(range(168, 224)))]


def test_no_cpu_on_the_socket_leaves_the_inherited_set():
    # pinned by the launcher to the other socket entirely: nothing is applied, nothing widened
    applied = _run(inherited=range(0, 56), node_cpulist="56-111,168-223\n")
    assert applied == []


def test_parse_cpulist():
    assert tpu_worker._parse_cpulist("0-3,8,10-11\n") == [
        0, 1, 2, 3, 8, 10, 11
    ]
