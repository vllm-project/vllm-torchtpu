# SPDX-License-Identifier: Apache-2.0
"""q38 SV-PAIR: strict VLLM_TPU_CPU_PIN=slice (refusal rows R1-R6) and VLLM_TPU_CPU_PIN_ENGINE.

The topology is the served decode host measured in Car E2 c212's decode log (j-b5cd8aed, the eight `CPU pin:` lines
under mode numa): node 0 = CPUs 0-55 + 112-167, node 1 = 56-111 + 168-223 (SMT pairs (c, c + 112)), four TPU devices
per node, rank r holding device r. The kernel (thread list, masks, cpuset) is a fake bound to the module's `os`
name, so nothing here touches the real process affinity.

Q38_SVPIN_BASE must name the base tree's worker/cpu_pinning.py (K3b 5ef1f2a184) for the numa / off identity test."""
import asyncio
import importlib.util
import os
import sys
import types

import pytest

from vllm_torchtpu.worker import cpu_pinning

NODE = {0: list(range(0, 56)) + list(range(112, 168)), 1: list(range(56, 112)) + list(range(168, 224))}
DEVS = [(0, "0000:00:0%d.0" % i) for i in range(4)] + [(1, "0000:80:0%d.0" % i) for i in range(4)]
ALL = set(range(224))
# Expected slice plans (physical cores; each comes with its sibling c + 112): = scratch/serve2/cpupin/plan_sim.py output.
WANT = {
    0: {0: [(0, 13), (14, 27), (28, 41), (42, 55)], 1: [(56, 69), (70, 83), (84, 97), (98, 111)]},
    2: {0: [(0, 12), (13, 25), (26, 38), (39, 51)], 1: [(56, 68), (69, 81), (82, 94), (95, 107)]},
    4: {0: [(0, 12), (13, 25), (26, 38), (39, 51)], 1: [(56, 68), (69, 81), (82, 94), (95, 107)]},
    8: {0: [(0, 11), (12, 23), (24, 35), (36, 47)], 1: [(56, 67), (68, 79), (80, 91), (92, 103)]},
}


def cores_to_cpus(lo, hi):
    return {c for c in range(lo, hi + 1)} | {c + 112 for c in range(lo, hi + 1)}


class Kernel:
    """Threads with affinity masks; the cpuset the kernel intersects with; failure injections."""

    def __init__(self, tids=(100, 101, 102), allowed=ALL, eperm=(), vanish=(), drop=None, spawn_once=False):
        self.allowed = set(allowed)
        self.masks = {t: set(self.allowed) for t in tids}
        self.main = tids[0]
        self.eperm, self.vanish, self.drop = set(eperm), set(vanish), dict(drop or {})
        self.spawn_once, self.spawned = spawn_once, False
        self.set_calls = []

    def listdir(self, path):
        assert path == "/proc/self/task", path
        return [str(t) for t in sorted(self.masks)]

    def sched_getaffinity(self, tid):
        tid = self.main if tid == 0 else tid
        if tid in self.vanish:
            raise ProcessLookupError(3, "No such process")
        return set(self.masks[tid])

    def sched_setaffinity(self, tid, cpus):
        tid = self.main if tid == 0 else tid
        if tid in self.vanish:
            raise ProcessLookupError(3, "No such process")
        if tid in self.eperm:
            raise PermissionError(1, "Operation not permitted")
        self.set_calls.append((tid, frozenset(cpus)))
        new = (set(cpus) & self.allowed) - self.drop.get(tid, set())
        if not new:
            raise OSError(22, "Invalid argument")
        old = self.masks[tid]
        self.masks[tid] = new
        if self.spawn_once and not self.spawned:  # a not-yet-pinned thread spawns one with its old mask
            self.spawned = True
            self.masks[999] = set(old)

    def getpid(self):
        return 4242


class FakeOS:
    def __init__(self, kernel):
        self._k = kernel

    def __getattr__(self, name):
        if name in ("listdir", "sched_getaffinity", "sched_setaffinity", "getpid"):
            return getattr(self._k, name)
        return getattr(os, name)


def fake_read_factory(outside_sibling=None):
    def fake_read(path):
        if path.startswith("/sys/devices/system/node/node") and path.endswith("/cpulist"):
            c = NODE[int(path.split("/node/node")[1].split("/")[0])]
            return "%d-%d,%d-%d" % (c[0], c[55], c[56], c[111])
        if "/topology/thread_siblings_list" in path:
            cpu = int(path.split("/cpu/cpu")[1].split("/")[0])
            a = cpu % 112
            if outside_sibling is not None and a == outside_sibling[0]:
                return "%d,%d" % (a, outside_sibling[1])
            return "%d,%d" % (a, a + 112)
        raise OSError(path)

    return fake_read


class Rec:
    def __init__(self):
        self.records = []

    def info(self, msg, *args):
        self.records.append(("info", msg, args))

    def warning(self, msg, *args):
        self.records.append(("warning", msg, args))

    def exception(self, msg, *args):
        self.records.append(("exception", msg, args))


def rig(monkeypatch, mod, kernel, mine, devs=DEVS, outside_sibling=None):
    monkeypatch.setattr(mod, "os", FakeOS(kernel))
    monkeypatch.setattr(mod, "_read", fake_read_factory(outside_sibling))
    monkeypatch.setattr(mod, "my_chip_devices", lambda: list(mine))
    monkeypatch.setattr(mod, "all_chip_devices", lambda: sorted(devs))
    rec = Rec()
    monkeypatch.setattr(mod, "logger", rec)
    return rec


def env(monkeypatch, pin, reserve=None, engine=None):
    monkeypatch.setenv("VLLM_TPU_CPU_PIN", pin)
    monkeypatch.delenv("VLLM_TPU_CPU_PROBE_S", raising=False)
    if reserve is None:
        monkeypatch.delenv("VLLM_TPU_CPU_PIN_RESERVE", raising=False)
    else:
        monkeypatch.setenv("VLLM_TPU_CPU_PIN_RESERVE", str(reserve))
    if engine is None:
        monkeypatch.delenv("VLLM_TPU_CPU_PIN_ENGINE", raising=False)
    else:
        monkeypatch.setenv("VLLM_TPU_CPU_PIN_ENGINE", engine)


def load_base():
    path = os.environ.get("Q38_SVPIN_BASE", "")
    assert path and os.path.isfile(path), "Q38_SVPIN_BASE must name the base tree's worker/cpu_pinning.py"
    spec = importlib.util.spec_from_file_location("q38_svpin_base_cpu_pinning", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- the plan on the measured topology -------------------------------------------------------------------------


def test_slice_plan_matches_the_measured_table(monkeypatch):
    monkeypatch.setattr(cpu_pinning, "_read", fake_read_factory())
    for reserve, nodes in WANT.items():
        seen = set()
        for node, ranges in nodes.items():
            for slot, (lo, hi) in enumerate(ranges):
                got = cpu_pinning.slice_plan(node, slot, 4, reserve)
                assert got == cores_to_cpus(lo, hi), (reserve, node, slot)
                assert got == cpu_pinning.plan("slice", node, slot, 4, reserve)
                assert not got & seen
                seen |= got
    assert cpu_pinning.reserved_cpus(0, 4, 2) == cores_to_cpus(52, 55)
    assert cpu_pinning.reserved_cpus(1, 4, 2) == cores_to_cpus(108, 111)
    assert cpu_pinning.reserved_cpus(0, 4, 0) == set()


# --- numa / off: byte-identical behaviour to the base tree ----------------------------------------------------


@pytest.mark.parametrize("mode", ["numa", "off"])
def test_numa_and_off_match_the_base_tree(monkeypatch, mode):
    base = load_base()
    env(monkeypatch, mode)
    results = []
    for mod in (base, cpu_pinning):
        per_rank = []
        for rank in range(8):
            kernel = Kernel()
            rec = rig(monkeypatch, mod, kernel, [DEVS[rank]])
            mod.apply(rank, rank)
            per_rank.append((kernel.set_calls, {t: sorted(m) for t, m in kernel.masks.items()}, rec.records))
        results.append(per_rank)
    assert results[0] == results[1]
    if mode == "numa":
        assert all(set(m) == set(NODE[DEVS[r][0]]) for r, pr in enumerate(results[1]) for m in pr[1].values())
    else:
        assert all(not pr[0] and not pr[2] for pr in results[1])


# --- slice: the happy path --------------------------------------------------------------------------------------


def test_slice_pins_every_thread_to_its_slot(monkeypatch):
    env(monkeypatch, "slice")
    for rank in range(8):
        kernel = Kernel()
        rec = rig(monkeypatch, cpu_pinning, kernel, [DEVS[rank]])
        cpu_pinning.apply(rank, rank)
        node, slot = DEVS[rank][0], rank % 4
        want = cores_to_cpus(*WANT[2][node][slot])
        assert all(m == want for m in kernel.masks.values())
        (level, msg, args), = rec.records
        assert level == "info" and "mode slice" in msg and args[2] == node and args[3] == slot and args[4] == 4 and args[5] == 13


# --- refusal rows -----------------------------------------------------------------------------------------------


def refuses(monkeypatch, row, mine, kernel=None, devs=DEVS, outside_sibling=None, reserve=None):
    env(monkeypatch, "slice", reserve=reserve)
    kernel = kernel or Kernel()
    before = {t: set(m) for t, m in kernel.masks.items()}
    rig(monkeypatch, cpu_pinning, kernel, mine, devs=devs, outside_sibling=outside_sibling)
    with pytest.raises(cpu_pinning.SlicePinError, match=row):
        cpu_pinning.apply(0, 0)
    return kernel, before


def test_r1_two_devices_on_the_node_refuse(monkeypatch):
    kernel, before = refuses(monkeypatch, "R1", [DEVS[0], DEVS[1]])
    assert kernel.masks == before and not kernel.set_calls


def test_r1_devices_on_two_nodes_refuse(monkeypatch):
    refuses(monkeypatch, "R1", [DEVS[0], DEVS[4]])


def test_r1_no_device_or_unknown_node_refuses(monkeypatch):
    refuses(monkeypatch, "R1", [])
    refuses(monkeypatch, "R1", [(-1, "0000:00:00.0")])


def test_r2_device_missing_from_its_node_refuses(monkeypatch):
    kernel, before = refuses(monkeypatch, "R2", [(0, "0000:00:09.0")])
    assert kernel.masks == before and not kernel.set_calls


def test_r3_too_few_cores_for_slots_and_reserve_refuses(monkeypatch):
    kernel, before = refuses(monkeypatch, "R3", [DEVS[0]], reserve=53)
    assert kernel.masks == before and not kernel.set_calls


def test_r4_sibling_outside_the_node_refuses(monkeypatch):
    kernel, before = refuses(monkeypatch, "R4", [DEVS[0]], outside_sibling=(5, 60))
    assert kernel.masks == before and not kernel.set_calls


def test_r5_plan_outside_the_allowed_cpus_refuses_before_pinning(monkeypatch):
    kernel = Kernel(allowed=ALL - {3})
    kernel2, before = refuses(monkeypatch, "R5", [DEVS[0]], kernel=kernel)
    assert kernel2.masks == before and not kernel2.set_calls


def test_r6_read_back_mismatch_refuses(monkeypatch):
    # the kernel silently drops CPU 7 from thread 101's mask (a cpuset the plan does not see)
    refuses(monkeypatch, "R6", [DEVS[0]], kernel=Kernel(drop={101: {7}}))


def test_r6_refused_setaffinity_refuses(monkeypatch):
    refuses(monkeypatch, "R6", [DEVS[0]], kernel=Kernel(eperm={102}))


def test_r6_tolerates_exited_threads_and_catches_new_ones(monkeypatch):
    env(monkeypatch, "slice")
    kernel = Kernel(vanish={102}, spawn_once=True)
    rig(monkeypatch, cpu_pinning, kernel, [DEVS[5]])
    cpu_pinning.apply(5, 5)
    want = cores_to_cpus(*WANT[2][1][1])
    assert kernel.masks[999] == want and kernel.masks[100] == want and kernel.masks[101] == want


# --- VLLM_TPU_CPU_PIN_ENGINE --------------------------------------------------------------------------------------


class FakeEngineCore:
    def __init__(self, kernel):
        self.during = {t: set(m) for t, m in kernel.masks.items()}


class FakeChat:
    async def create_chat_completion(self, request):
        return ("chat", request)


class FakeCompletion:
    async def create_completion(self, request):
        return ("completion", request)


def fake_vllm(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.core", types.SimpleNamespace(EngineCore=FakeEngineCore))
    monkeypatch.setitem(
        sys.modules, "vllm.entrypoints.openai.chat_completion.serving", types.SimpleNamespace(OpenAIServingChat=FakeChat)
    )
    monkeypatch.setitem(
        sys.modules, "vllm.entrypoints.openai.completion.serving", types.SimpleNamespace(OpenAIServingCompletion=FakeCompletion)
    )
    for cls, name in ((FakeEngineCore, "__init__"), (FakeChat, "create_chat_completion"), (FakeCompletion, "create_completion")):
        monkeypatch.setattr(cls, name, cls.__dict__[name])
    for flag in ("_q38_cpu_pin_engine",):
        monkeypatch.delattr(FakeEngineCore, flag, raising=False)
    monkeypatch.setattr(cpu_pinning, "_ENGINE_DONE", {})


def test_engine_option_off_patches_and_imports_nothing(monkeypatch):
    fake_vllm(monkeypatch)
    for engine in (None, "off", ""):
        env(monkeypatch, "slice", engine=engine)
        kernel = Kernel()
        rig(monkeypatch, cpu_pinning, kernel, [DEVS[0]])
        init0, chat0, comp0 = FakeEngineCore.__init__, FakeChat.create_chat_completion, FakeCompletion.create_completion
        assert cpu_pinning.patch_engine_core_cpu_pin() is None and cpu_pinning.patch_api_server_cpu_pin() is None
        assert (FakeEngineCore.__init__, FakeChat.create_chat_completion, FakeCompletion.create_completion) == (init0, chat0, comp0)
        FakeEngineCore(kernel)
        assert not kernel.set_calls and cpu_pinning._ENGINE_DONE == {}


def test_engine_option_needs_slice_and_a_valid_value(monkeypatch):
    fake_vllm(monkeypatch)
    rig(monkeypatch, cpu_pinning, Kernel(), [DEVS[0]])
    env(monkeypatch, "numa", engine="reserved")
    with pytest.raises(cpu_pinning.SlicePinError, match="needs VLLM_TPU_CPU_PIN=slice"):
        cpu_pinning.patch_engine_core_cpu_pin()
    for bad in ("reserved:x", "slice", "reserved:"):
        env(monkeypatch, "slice", engine=bad)
        with pytest.raises(cpu_pinning.SlicePinError, match="expected off, reserved"):
            cpu_pinning.engine_target()


def test_engine_target_is_the_reserve_and_disjoint_from_every_slice(monkeypatch):
    rig(monkeypatch, cpu_pinning, Kernel(), [DEVS[0]])
    env(monkeypatch, "slice", engine="reserved")
    both = cpu_pinning.engine_target()
    assert both == cores_to_cpus(52, 55) | cores_to_cpus(108, 111)
    env(monkeypatch, "slice", engine="reserved:1")
    assert cpu_pinning.engine_target() == cores_to_cpus(108, 111)
    slices = set().union(*(cores_to_cpus(*r) for node in (0, 1) for r in WANT[2][node]))
    assert not both & slices


def test_engine_option_refuses_an_empty_or_foreign_reserve(monkeypatch):
    rig(monkeypatch, cpu_pinning, Kernel(), [DEVS[0]])
    env(monkeypatch, "slice", reserve=0, engine="reserved")
    with pytest.raises(cpu_pinning.SlicePinError, match="no reserved cores"):
        cpu_pinning.engine_target()
    env(monkeypatch, "slice", engine="reserved:2")
    with pytest.raises(cpu_pinning.SlicePinError, match="node 2 has no TPU devices"):
        cpu_pinning.engine_target()


def test_engine_core_is_pinned_after_its_executor_and_only_once(monkeypatch):
    fake_vllm(monkeypatch)
    env(monkeypatch, "slice", engine="reserved")
    kernel = Kernel()
    rec = rig(monkeypatch, cpu_pinning, kernel, [DEVS[0]])
    cpu_pinning.patch_engine_core_cpu_pin()
    core = FakeEngineCore(kernel)
    target = cores_to_cpus(52, 55) | cores_to_cpus(108, 111)
    assert all(m == ALL for m in core.during.values())  # the executor (and its workers) ran unpinned
    assert all(m == target for m in kernel.masks.values())
    n = len(kernel.set_calls)
    FakeEngineCore(kernel)
    assert len(kernel.set_calls) == n
    assert any(r[0] == "info" and "role=%s" in r[1] and r[2][0] == "engine_core" for r in rec.records)


def test_api_server_is_pinned_once_at_its_first_request(monkeypatch):
    fake_vllm(monkeypatch)
    env(monkeypatch, "slice", engine="reserved:0")
    kernel = Kernel()
    rig(monkeypatch, cpu_pinning, kernel, [DEVS[0]])
    cpu_pinning.patch_api_server_cpu_pin()
    assert not kernel.set_calls  # nothing at patch time
    assert asyncio.run(FakeChat().create_chat_completion("r1")) == ("chat", "r1")
    assert all(m == cores_to_cpus(52, 55) for m in kernel.masks.values())
    n = len(kernel.set_calls)
    assert asyncio.run(FakeCompletion().create_completion("r2")) == ("completion", "r2")
    assert asyncio.run(FakeChat().create_chat_completion("r3")) == ("chat", "r3")
    assert len(kernel.set_calls) == n


# --- real-tree facts (need the vllm_torchtpu package and vLLM) ------------------------------------------------------


def test_engine_key_is_registered_and_compile_ignored():
    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation import tpu_compiler

    assert envs.environment_variables["VLLM_TPU_CPU_PIN_ENGINE"]() == os.environ.get("VLLM_TPU_CPU_PIN_ENGINE", "off")
    assert "VLLM_TPU_CPU_PIN_ENGINE" in tpu_compiler._TPU_COMPILE_ENV_IGNORED
    assert "VLLM_TPU_CPU_PIN_ENGINE" not in tpu_compiler._tpu_compile_env_factors()


def test_hook_targets_exist_in_vllm():
    import inspect

    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
    from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
    from vllm.v1.engine.core import EngineCore, EngineCoreProc

    assert callable(EngineCore.__init__) and issubclass(EngineCoreProc, EngineCore)
    assert inspect.iscoroutinefunction(OpenAIServingChat.create_chat_completion)
    assert inspect.iscoroutinefunction(OpenAIServingCompletion.create_completion)
