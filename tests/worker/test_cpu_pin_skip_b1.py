# SPDX-License-Identifier: Apache-2.0
"""q38 SERVE r0pin: VLLM_TPU_CPU_PIN_SKIP_CPUS in the b1 form's slice plan.

Run with Q38_FORM=b1 (the key exists only in _forms/b1); skipped otherwise.
"""
import pytest

from vllm_torchtpu.worker import cpu_pinning

pytestmark = pytest.mark.skipif(
    not hasattr(cpu_pinning, "skip_cpus"), reason="needs Q38_FORM=b1 (r0pin key)"
)

CORES = 56  # one node: CPUs 0-55 with SMT siblings 112-167, as on the decode host
SLOTS, RESERVE = 4, 4


def _node0(monkeypatch):
    monkeypatch.setattr(
        cpu_pinning, "physical_cores", lambda node: [(c, c + 112) for c in range(CORES)]
    )
    monkeypatch.setattr(
        cpu_pinning, "node_cpus", lambda node: set(range(CORES)) | set(range(112, 112 + CORES))
    )


def test_empty_skip_is_the_old_plan(monkeypatch):
    _node0(monkeypatch)
    for slot in range(SLOTS):
        new = cpu_pinning.slice_plan(0, slot, SLOTS, RESERVE, skip=set())
        old = {c for core in [(c, c + 112) for c in range(slot * 13, slot * 13 + 13)] for c in core}
        assert new == old
    assert 0 in cpu_pinning.slice_plan(0, 0, SLOTS, RESERVE, skip=set())
    assert cpu_pinning.reserved_cpus(0, SLOTS, RESERVE, skip=set()) == {
        52, 53, 54, 55, 164, 165, 166, 167
    }


def test_skip_cpu0_shifts_slices_keeps_size_and_frees_the_core(monkeypatch):
    _node0(monkeypatch)
    skip = {0, 112}
    slices = [cpu_pinning.slice_plan(0, s, SLOTS, RESERVE, skip=skip) for s in range(SLOTS)]
    assert all(len(s) == 26 for s in slices)  # 13 cores with SMT siblings, unchanged
    union = set().union(*slices)
    assert len(union) == SLOTS * 26  # disjoint
    assert not union & skip
    assert slices[0] == {c for c in range(1, 14)} | {c + 112 for c in range(1, 14)}
    reserve = cpu_pinning.reserved_cpus(0, SLOTS, RESERVE, skip=skip)
    assert reserve == {53, 54, 55, 165, 166, 167}  # the remainder shrinks by the skipped core
    assert not reserve & skip and not reserve & union


def test_skip_reads_the_env(monkeypatch):
    monkeypatch.setenv("VLLM_TPU_CPU_PIN_SKIP_CPUS", "0,112")
    assert cpu_pinning.skip_cpus() == {0, 112}
    monkeypatch.setenv("VLLM_TPU_CPU_PIN_SKIP_CPUS", "")
    assert cpu_pinning.skip_cpus() == set()
    monkeypatch.delenv("VLLM_TPU_CPU_PIN_SKIP_CPUS", raising=False)
    assert cpu_pinning.skip_cpus() == set()


def test_skip_that_leaves_too_few_cores_refuses(monkeypatch):
    _node0(monkeypatch)
    skip = set(range(0, 5))  # 5 cores gone: 51 < 4 slots x 13
    with pytest.raises(cpu_pinning.SlicePinError, match="R3"):
        cpu_pinning.slice_plan(0, 0, SLOTS, RESERVE, skip=skip)


def test_skip_range_list_parses(monkeypatch):
    monkeypatch.setenv("VLLM_TPU_CPU_PIN_SKIP_CPUS", "0-23,112-135")
    assert cpu_pinning.skip_cpus() == set(range(24)) | set(range(112, 136))
    monkeypatch.setenv("VLLM_TPU_CPU_PIN_SKIP_CPUS", "0-23")
    assert cpu_pinning.skip_cpus() == set(range(24))


def test_skip_without_siblings_drops_the_whole_core(monkeypatch):
    # the served value "0" (recipe env lists cannot carry a comma) == "0,112"
    _node0(monkeypatch)
    for slot in range(SLOTS):
        assert cpu_pinning.slice_plan(0, slot, SLOTS, 2, skip={0}) == cpu_pinning.slice_plan(
            0, slot, SLOTS, 2, skip={0, 112}
        )
    assert cpu_pinning.slice_plan(0, 0, SLOTS, 2, skip={0}) == set(range(1, 14)) | set(range(113, 126))


def test_arm_c_reserve24_skip_0_23_gives_eight_core_slices(monkeypatch):
    # SERVE-NODE0 arm c: RESERVE=24, skip "0-23" (== "0-23,112-135"): ranks 0-3 on cores 24-55
    _node0(monkeypatch)
    want = [set(range(24 + 8 * s, 32 + 8 * s)) | set(range(136 + 8 * s, 144 + 8 * s)) for s in range(SLOTS)]
    for skip in (set(range(24)), set(range(24)) | set(range(112, 136))):
        assert [cpu_pinning.slice_plan(0, s, SLOTS, 24, skip=skip) for s in range(SLOTS)] == want
        assert cpu_pinning.reserved_cpus(0, SLOTS, 24, skip=skip) == set()


def test_irq_dump_lines_round_trip(tmp_path):
    import base64
    import gzip
    import hashlib

    a = tmp_path / "interrupts"
    b = tmp_path / "softirqs"
    a.write_bytes(("CPU0 CPU1\n" + "".join(" %d: %d %d x\n" % (i, i * 7, i * 11) for i in range(3000))).encode())
    b.write_bytes(b"NET_RX: 5 6\n")
    lines = cpu_pinning.irq_dump_lines(0, 3, paths=(str(a), str(b)), chunk=500, now=12.5)
    for p in (a, b):
        mine = [x for x in lines if (" src=%s " % p) in x]
        n = int(mine[0].split(" part=1/")[1].split()[0])
        assert len(mine) == n and all(x.startswith("CPU IRQ DUMP rank=0 seq=3 t=12.500 src=") for x in mine)
        raw = gzip.decompress(base64.b64decode("".join(x.split(" b64=", 1)[1] for x in mine)))
        assert raw == p.read_bytes() and ("md5=%s" % hashlib.md5(raw).hexdigest()) in mine[0]
        assert all(len(x.split(" b64=", 1)[1]) <= 500 for x in mine)
    assert len([x for x in lines if (" src=%s " % a) in x]) > 1  # chunked
    missing = cpu_pinning.irq_dump_lines(0, 0, paths=(str(tmp_path / "nope"),))
    assert len(missing) == 1 and "UNREADABLE" in missing[0]
