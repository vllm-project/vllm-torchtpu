# SPDX-License-Identifier: Apache-2.0
from vllm_torchtpu.worker import cpu_pinning


def test_cpulist_parses_ranges_and_singles():
    assert cpu_pinning._cpulist("0-2,5,112-113") == [0, 1, 2, 5, 112, 113]


def _node0(monkeypatch, cores=56):
    monkeypatch.setattr(
        cpu_pinning,
        "physical_cores",
        lambda node: [(c, c + 112) for c in range(cores)],
    )


def test_numa_mode_gives_the_whole_node(monkeypatch):
    _node0(monkeypatch)
    assert len(cpu_pinning.plan("numa", 0, 1, 4, reserve=2)) == 112


def test_slices_are_disjoint_whole_cores_and_keep_the_reserve(monkeypatch):
    _node0(monkeypatch)
    slices = [cpu_pinning.plan("slice", 0, slot, 4, reserve=2) for slot in range(4)]
    assert all(len(s) == 26 for s in slices)  # 13 cores with their SMT siblings
    union = set().union(*slices)
    assert len(union) == 4 * 26
    assert all((c + 112 in s) == (c in s) for s in slices for c in range(56))
    assert not union & {54, 55, 166, 167}  # the reserved last two cores


def test_physical_cores_intersects_initial_cpuset(monkeypatch):
    sysfs = {
        "/sys/devices/system/node/node0/cpulist": "0-3,8-11",
        "/sys/devices/system/cpu/cpu0/topology/thread_siblings_list": "0,8",
        "/sys/devices/system/cpu/cpu1/topology/thread_siblings_list": "1,9",
        "/sys/devices/system/cpu/cpu2/topology/thread_siblings_list": "2,10",
        "/sys/devices/system/cpu/cpu3/topology/thread_siblings_list": "3,11",
        "/sys/devices/system/cpu/cpu8/topology/thread_siblings_list": "0,8",
        "/sys/devices/system/cpu/cpu9/topology/thread_siblings_list": "1,9",
        "/sys/devices/system/cpu/cpu10/topology/thread_siblings_list": "2,10",
        "/sys/devices/system/cpu/cpu11/topology/thread_siblings_list": "3,11",
    }
    monkeypatch.setattr(cpu_pinning, "_INITIAL_CPUS", None)
    monkeypatch.setattr(cpu_pinning, "_read", lambda p: sysfs[p])
    monkeypatch.setattr(cpu_pinning, "allowed_cpus", lambda: {2, 3, 10})
    assert cpu_pinning.physical_cores(0) == [(2, 10), (3,)]


def _mock_subhost_pci(monkeypatch, slice_idx: int, allowed: set[int]):
    # 8-chip host: node 1 has 8 TPU PCI functions (0000:80..87); each 4-chip
    # sub-host slice sees 4 VFIO functions on node 1 (80..83 or 84..87).
    all_bdfs = [f"0000:8{i}:00.0" for i in range(8)]
    my_bdfs = all_bdfs[slice_idx * 4 : (slice_idx + 1) * 4]
    sysfs = {"/sys/devices/system/node/node1/cpulist": "60-119,180-239"}
    for bdf in all_bdfs:
        base = f"/sys/bus/pci/devices/{bdf}"
        sysfs[f"{base}/vendor"] = "0x1ae0"
        sysfs[f"{base}/device"] = "0x007e"
        sysfs[f"{base}/numa_node"] = "1"
    for c in range(60, 120):
        sib = f"{c},{c + 120}"
        sysfs[f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list"] = sib
        sysfs[
            f"/sys/devices/system/cpu/cpu{c + 120}/topology/thread_siblings_list"
        ] = sib

    monkeypatch.setattr(cpu_pinning, "_INITIAL_CPUS", None)
    monkeypatch.setattr(cpu_pinning, "_read", lambda p: sysfs[p])
    monkeypatch.setattr(cpu_pinning, "allowed_cpus", lambda: set(allowed))
    monkeypatch.setattr(
        cpu_pinning, "all_chip_devices", lambda: [(1, bdf) for bdf in my_bdfs]
    )
    monkeypatch.setattr(
        cpu_pinning.glob,
        "glob",
        lambda pat: (
            [f"/sys/bus/pci/devices/{bdf}" for bdf in all_bdfs]
            if pat == "/sys/bus/pci/devices/*"
            else []
        ),
    )
    return my_bdfs


def test_subhost_shared_cpus_offsets_slots_across_co_tenants(monkeypatch):
    full_host_cpus = set(range(240))
    bdfs_a = _mock_subhost_pci(monkeypatch, slice_idx=0, allowed=full_host_cpus)
    peers_a = cpu_pinning.node_peers(1)
    slots_a = len(peers_a)
    slices_a = [
        cpu_pinning.plan("slice", 1, peers_a.index((1, bdf)), slots_a, reserve=4)
        for bdf in bdfs_a
    ]

    bdfs_b = _mock_subhost_pci(monkeypatch, slice_idx=1, allowed=full_host_cpus)
    peers_b = cpu_pinning.node_peers(1)
    slots_b = len(peers_b)
    slices_b = [
        cpu_pinning.plan("slice", 1, peers_b.index((1, bdf)), slots_b, reserve=4)
        for bdf in bdfs_b
    ]

    assert slots_a == 8 and slots_b == 8
    union_a = set().union(*slices_a)
    union_b = set().union(*slices_b)
    assert not (union_a & union_b)


def test_subhost_isolated_cpuset_uses_local_vfio_slots(monkeypatch):
    # Container B already has a private cpuset on node 1 (90..119, 210..239).
    private_cpus = set(range(90, 120)) | set(range(210, 240))
    bdfs_b = _mock_subhost_pci(monkeypatch, slice_idx=1, allowed=private_cpus)
    peers_b = cpu_pinning.node_peers(1)
    assert peers_b == [(1, bdf) for bdf in bdfs_b]
    slices_b = [
        cpu_pinning.plan("slice", 1, slot, len(peers_b), reserve=2)
        for slot in range(len(peers_b))
    ]
    assert all(s <= private_cpus for s in slices_b)
    assert all(len(s) == 14 for s in slices_b)
