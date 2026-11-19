# SPDX-License-Identifier: Apache-2.0
"""q38 L6 GC-ADMIT-FREEZE: lever gcadmit (core/l6_gcadmit.py) and its hook in TPUModelRunner.execute_model.
Host-only; CDK CPU pod. Real gc module and vLLM's CachedRequestState; the runner is built with __new__ and
its _update_states replaced by one that builds the admitted requests' state."""

import gc
import os
import pathlib
import weakref
from types import SimpleNamespace as NS

import pytest
from vllm.v1.worker.gpu_input_batch import CachedRequestState  # imported before any fixture freeze

ENVS = ("VLLM_TPU_HOST_GC", "VLLM_TPU_HOST_AB_PERIOD_MS", "VLLM_TPU_D1N_AB_PERIOD_MS", "VLLM_TPU_SERVED_AB")
_UNFREEZE = gc.unfreeze  # the real one, for the fixture (tests below wrap gc.unfreeze to watch for calls)


@pytest.fixture
def ga(monkeypatch):
    from vllm_torchtpu.core import l6_ab, l6_gcadmit

    for k in list(os.environ):
        if k.startswith("VLLM_TPU_L6_") or k in ENVS:
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(l6_ab, "_rank_world", lambda: (0, 1))
    lines = []
    for mod in (l6_ab, l6_gcadmit):
        monkeypatch.setattr(mod.logger, "info", lambda fmt, *a: lines.append(fmt % a))
        monkeypatch.setattr(mod.logger, "warning", lambda fmt, *a: lines.append("WARNING " + fmt % a))
        monkeypatch.setattr(mod.logger, "error", lambda fmt, *a: lines.append("ERROR " + fmt % a))
    th = gc.get_threshold()
    gc.enable()
    _UNFREEZE()  # test isolation only: the lever itself never unfreezes
    gc.collect()
    gc.freeze()  # the worker's warm-up freeze_gc_heap(): the lever starts from a frozen heap

    def setup(**env):
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        l6_ab._resolve()
        l6_gcadmit._resolve()
        return l6_ab, l6_gcadmit

    yield setup, lines, monkeypatch
    _UNFREEZE()
    gc.enable()
    gc.set_threshold(*th)
    for k in list(os.environ):
        if k.startswith("VLLM_TPU_L6_"):
            monkeypatch.delenv(k, raising=False)
    l6_ab._resolve()
    l6_gcadmit._resolve()


def _frozen(*objs):
    """Each object is GC-tracked and in no collected generation (gc.get_objects() lists generations 0-2 only)."""
    live = {id(o) for o in gc.get_objects()}
    return [gc.is_tracked(o) and id(o) not in live for o in objs]


def _state(req_id, n=64):
    return CachedRequestState(req_id=req_id, prompt_token_ids=list(range(1000, 1000 + n)), mm_features=[],
                              sampling_params=None, generator=None, block_ids=([1, 2, 3],),
                              num_computed_tokens=0, output_token_ids=[])


def _so(*req_ids):
    return NS(scheduled_new_reqs=[NS(req_id=r) for r in req_ids], total_num_scheduled_tokens=0,
              num_scheduled_tokens={r: 1 for r in req_ids})


class _Cycle:
    def __init__(self):
        self.me = self
        self.payload = [object() for _ in range(4)]


def test_admission_step_freezes_the_new_state_and_other_steps_do_not(ga):
    setup, lines, _ = ga
    ab, ga_ = setup(VLLM_TPU_L6_GC_ADMIT_FREEZE="1")
    runner = NS(warmup_done=True)
    st = _state("a")
    ga_.after_update_states(runner, _so("a"))
    assert _frozen(st, st.prompt_token_ids, st.output_token_ids, st.block_ids[0]) == [True] * 4
    assert ab._state["counts"].get("gcadmit.freeze") == 1
    later = [object()]  # created after the freeze, on steps without admissions
    for _ in range(3):
        ga_.after_update_states(runner, _so())
    assert _frozen(later) == [False]
    assert ab._state["counts"].get("gcadmit.freeze") == 1
    ga_.after_update_states(runner, _so("b", "c"))
    assert _frozen(later) == [True]
    assert ab._state["counts"].get("gcadmit.freeze") == 2
    assert not [ln for ln in lines if ln.startswith(("ERROR", "WARNING"))], lines


def test_waits_for_warm_up(ga):
    setup, lines, _ = ga
    ab, ga_ = setup(VLLM_TPU_L6_GC_ADMIT_FREEZE="1")
    runner = NS(warmup_done=False)
    st = _state("w")
    ga_.after_update_states(runner, _so("w"))
    assert _frozen(st) == [False] and "gcadmit.freeze" not in ab._state["counts"]
    runner.warmup_done = True
    ga_.after_update_states(runner, _so("w2"))
    assert _frozen(st) == [True]


@pytest.mark.parametrize("case", ["d1e_freeze", "gcfreeze"])
def test_refused_next_to_another_freeze_owner(ga, case):
    setup, lines, _ = ga
    env = {"VLLM_TPU_HOST_GC": "freeze"} if case == "d1e_freeze" else {"VLLM_TPU_L6_GC_FREEZE": "ab:1",
                                                                       "VLLM_TPU_L6_AB_STEPS": "4",
                                                                       "VLLM_TPU_L6_AB_CONFIRM": "1"}
    ab, ga_ = setup(VLLM_TPU_L6_GC_ADMIT_FREEZE="1", **env)
    runner = NS(warmup_done=True)
    probe = [object()]
    for i in range(3):
        ab.begin_step()
        ga_.after_update_states(runner, _so(f"r{i}"))
    assert _frozen(probe) == [False]
    assert "gcadmit.freeze" not in ab._state["counts"]
    assert sum(ln.startswith("ERROR q38 L6 gcadmit refused") for ln in lines) == 1, lines


def test_never_unfreezes_across_windows_errors_and_the_cap(ga):
    """ab:1 with 4-step windows: an admission in window 1 (on) freezes; admissions in window 2 (off) neither
    freeze nor thaw; then an error and the cap turn the lever off, still without gc.unfreeze()."""
    setup, lines, mp = ga
    calls = []
    mp.setattr(gc, "unfreeze", lambda: (calls.append("unfreeze"), _UNFREEZE()))
    ab, ga_ = setup(VLLM_TPU_L6_GC_ADMIT_FREEZE="ab:1", VLLM_TPU_L6_AB_STEPS="4", VLLM_TPU_L6_AB_CONFIRM="1")
    runner = NS(warmup_done=True)
    for _ in range(4):  # steps 1-3 window 0 (off), step 4 window 1 (on)
        ab.begin_step()
    st = _state("on")
    ga_.after_update_states(runner, _so("on"))
    assert _frozen(st) == [True]
    for _ in range(4):  # window 2: off
        ab.begin_step()
    assert not ab.on("gcadmit")
    off = _state("off")
    ga_.after_update_states(runner, _so("off"))
    assert _frozen(st, off) == [True, False]
    assert ab._state["counts"].get("gcadmit.freeze") is None  # the window-2 counts (window 1 froze once)
    mp.setattr(ga_.gc, "freeze", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    for _ in range(4):  # window 3: on again, the freeze raises
        ab.begin_step()
    ga_.after_update_states(runner, _so("err"))
    assert ga_._st["off"]
    assert sum(ln.startswith("ERROR q38 L6 gcadmit off") for ln in lines) == 1, lines
    assert _frozen(st, off) == [True, False]
    assert calls == [], calls


def test_counts_freeze_us_reclaimed_n_frozen_and_the_line_cadence(ga):
    """Each freeze reports m = what it moved (the permanent generation grows by exactly m, read here by the test;
    the lever itself never calls gc.get_freeze_count()); n_frozen sums m; lines at freezes 1-3 and every CHECK-th."""
    setup, lines, _ = ga
    ab, ga_ = setup(VLLM_TPU_L6_GC_ADMIT_FREEZE="1", VLLM_TPU_L6_GC_ADMIT_FREEZE_CHECK="2")
    runner = NS(warmup_done=True)
    keep, moved = [], []
    _state("warm")  # first-use caches of the state class outside the measured freezes
    gc.collect()
    for i in range(5):
        keep.append([_state(f"k{i}") for _ in range(10)])
        _Cycle()  # one unreachable cycle per step: reclaimed by the pre-freeze collection, not frozen
        so = _so(f"k{i}")
        before = gc.get_freeze_count()
        ga_.after_update_states(runner, so)
        moved.append(gc.get_freeze_count() - before)
    c = ab._state["counts"]
    assert c["gcadmit.freeze"] == 5
    assert c["gcadmit.us"] >= 0 and c["gcadmit.reclaimed"] >= 5 * 2, c
    gl = [ln for ln in lines if ln.startswith("q38 L6 GCADMIT ")]
    assert [ln.split()[3] for ln in gl] == ["freeze=1", "freeze=2", "freeze=3", "freeze=4"], gl  # 5 is not a 2nd
    this = [int(ln.split("frozen_this=")[1].split()[0]) for ln in gl]
    assert this == moved[:4], (this, moved)
    assert min(moved) >= 10 * 3, moved  # ten request states, each at least the state and its two lists
    assert c["gcadmit.n_frozen"] == sum(moved), (c, moved)
    assert gl[3].endswith(f"frozen_this={moved[3]} frozen_cum={sum(moved[:4])}"), gl
    assert keep and so


def test_pre_freeze_collection_reclaims_cyclic_garbage_instead_of_pinning_it(ga):
    """The leak guard: cyclic garbage created after one freeze is collected by the next admission's
    gc.collect(), not moved into the permanent generation where it could never be reclaimed."""
    setup, lines, _ = ga
    ab, ga_ = setup(VLLM_TPU_L6_GC_ADMIT_FREEZE="1")
    gc.disable()  # no automatic collection between the garbage and the freeze (served D1e raises gen-0 to 50000)
    runner = NS(warmup_done=True)
    ga_.after_update_states(runner, _so("first"))
    refs = []
    for _ in range(20):
        c = _Cycle()
        refs.append(weakref.ref(c))
        del c
    assert all(r() is not None for r in refs)  # unreachable but uncollected
    ga_.after_update_states(runner, _so("second"))
    assert all(r() is None for r in refs), "frozen cyclic garbage"


def test_cap_stops_freezing_without_unfreezing(ga):
    setup, lines, mp = ga
    calls = []
    mp.setattr(gc, "unfreeze", lambda: (calls.append("unfreeze"), _UNFREEZE()))
    ab, ga_ = setup(VLLM_TPU_L6_GC_ADMIT_FREEZE="1", VLLM_TPU_L6_GC_ADMIT_FREEZE_CAP="2500",
                    VLLM_TPU_L6_GC_ADMIT_FREEZE_CHECK="1")
    runner = NS(warmup_done=True)
    keep = []
    for i in range(10):
        keep.append([[j] for j in range(1000)])  # 1001 live tracked objects per admission
        ga_.after_update_states(runner, _so(f"c{i}"))
    c = ab._state["counts"]
    # objects frozen in total about 1001 / 2002 / 3003 (plus a few step objects) after freezes 1 / 2 / 3: capped
    # at the third
    assert c["gcadmit.freeze"] == 3 and c["gcadmit.capped"] == 1, c
    assert _frozen(keep[0][0], keep[2][0], keep[3][0], keep[9][0]) == [True, True, False, False]
    assert sum(ln.startswith("WARNING q38 L6 gcadmit capped") for ln in lines) == 1, lines
    assert calls == []


def test_never_reads_gc_get_freeze_count_on_the_step_path(ga):
    """gc.get_freeze_count() walks the whole permanent generation (117-132 ms at 1.64 M frozen objects, M:
    j-698978c0): the lever must not call it, on any freeze or at the cap."""
    setup, lines, mp = ga
    calls = []

    def boom():
        calls.append(1)
        raise AssertionError("gc.get_freeze_count() on the step path")

    ab, ga_ = setup(VLLM_TPU_L6_GC_ADMIT_FREEZE="1", VLLM_TPU_L6_GC_ADMIT_FREEZE_CAP="5000",
                    VLLM_TPU_L6_GC_ADMIT_FREEZE_CHECK="1")
    mp.setattr(gc, "get_freeze_count", boom)
    runner = NS(warmup_done=True)
    keep = []
    for i in range(20):
        keep.append([[j] for j in range(500)])  # 501 live tracked objects per admission
        ga_.after_update_states(runner, _so(f"g{i}"))
    c = ab._state["counts"]
    assert calls == []
    assert c.get("gcadmit.capped") == 1 and c["gcadmit.freeze"] >= 5, c
    assert sum(ln.startswith("WARNING q38 L6 gcadmit capped") for ln in lines) == 1, lines
    assert not [ln for ln in lines if ln.startswith("ERROR")], lines


def test_ab_refused_when_the_worker_never_begins_l6_steps(ga):
    setup, lines, _ = ga
    ab, ga_ = setup(VLLM_TPU_L6_GC_ADMIT_FREEZE="ab:1", VLLM_TPU_L6_AB_STEPS="4", VLLM_TPU_L6_AB_CONFIRM="1")
    runner = NS(warmup_done=True)
    probe = [object()]
    for i in range(3):
        ga_.after_update_states(runner, _so(f"n{i}"))  # no l6_ab.begin_step(): step_seq() stays 0
    assert _frozen(probe) == [False]
    assert sum("ab: windows need the worker's l6_ab.begin_step" in ln for ln in lines
               if ln.startswith("ERROR")) == 1, lines


def test_flag_off_is_a_silent_no_op(ga):
    setup, lines, _ = ga
    for val in ("", "0", "off"):
        ab, ga_ = setup(VLLM_TPU_L6_GC_ADMIT_FREEZE=val)
        assert not ga_.ON
        probe = [object()]
        ga_.after_update_states(NS(warmup_done=True), _so("x"))
        assert _frozen(probe) == [False]
    assert lines == [], lines


# ---------------------------------------------------------------- the hook in TPUModelRunner.execute_model


def _runner(monkeypatch):
    from vllm_torchtpu.runner import tpu_runner
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    monkeypatch.setattr(tpu_runner, "has_kv_transfer_group", lambda: False)
    r = TPUModelRunner.__new__(TPUModelRunner)
    r.execute_model_state = None
    r._pp_wave = None
    r.input_batch = NS(num_reqs=0)
    r.scheduler_config = NS(async_scheduling=False)
    r.requests = {}
    r.warmup_done = True
    order = []

    def update_states(so):  # the base _update_states builds each admitted request's state
        order.append("update_states")
        for nrd in so.scheduled_new_reqs:
            r.requests[nrd.req_id] = _state(nrd.req_id)

    r._update_states = update_states
    return r, order


def test_execute_model_freezes_the_state_update_states_built(ga):
    setup, lines, mp = ga
    from vllm_torchtpu.runner import tpu_runner

    ab, ga_ = setup(VLLM_TPU_L6_GC_ADMIT_FREEZE="1")
    r, order = _runner(mp)
    real = ga_.after_update_states
    mp.setattr(ga_, "after_update_states", lambda runner, so: (order.append("gcadmit"), real(runner, so)))
    out = r.execute_model(_so("e0"))
    assert out is tpu_runner.EMPTY_MODEL_RUNNER_OUTPUT
    assert order == ["update_states", "gcadmit"], order
    st = r.requests["e0"]
    assert _frozen(st, st.prompt_token_ids) == [True, True]
    r.execute_model(_so())  # a step without admissions: nothing frozen
    probe = [object()]
    r.execute_model(_so())
    assert _frozen(probe) == [False]


def test_execute_model_never_calls_the_lever_with_the_flag_off(ga):
    setup, lines, mp = ga
    from vllm_torchtpu.runner import tpu_runner

    ab, ga_ = setup()
    assert not ga_.ON
    r, order = _runner(mp)
    mp.setattr(ga_, "after_update_states", lambda runner, so: order.append("gcadmit"))
    outs = [r.execute_model(_so(f"f{i}")) for i in range(3)]
    assert all(o is tpu_runner.EMPTY_MODEL_RUNNER_OUTPUT for o in outs)
    assert order == ["update_states"] * 3, order
    assert _frozen(r.requests["f0"], r.requests["f2"]) == [False, False]


def test_hook_is_one_guarded_statement_right_after_update_states():
    from vllm_torchtpu.runner import tpu_runner

    src = pathlib.Path(tpu_runner.__file__).read_text()
    want = ("        self._update_states(scheduler_output)\n"
            "        if _l6_gcadmit.ON:  # q38 L6 GC-ADMIT-FREEZE: freeze this step's admitted state "
            "(core/l6_gcadmit.py)\n"
            "            _l6_gcadmit.after_update_states(self, scheduler_output)\n")
    assert src.count(want) == 1
    assert src.count("_l6_gcadmit.after_update_states(") == 1
    i = src.index(want)
    assert i < src.index('        host_phases.mark("update_states")\n', i)  # inside the update_states phase


def test_lever_is_registered_and_compile_ignored():
    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation.tpu_compiler import _TPU_COMPILE_ENV_IGNORED
    from vllm_torchtpu.core import l6_ab

    assert l6_ab.LEVERS["gcadmit"] == "VLLM_TPU_L6_GC_ADMIT_FREEZE"
    for name in ("VLLM_TPU_L6_GC_ADMIT_FREEZE", "VLLM_TPU_L6_GC_ADMIT_FREEZE_CAP",
                 "VLLM_TPU_L6_GC_ADMIT_FREEZE_CHECK"):
        assert name in envs.environment_variables and name in _TPU_COMPILE_ENV_IGNORED, name


def test_import_resolves_to_the_tree_under_test():
    """Guard against a staging that silently imports the pod's installed package."""
    import vllm_torchtpu

    want = os.environ.get("Q38_L6_EXPECT_TREE")
    if not want:
        pytest.skip("Q38_L6_EXPECT_TREE not set (the CPU harness sets it)")
    got = os.path.realpath(vllm_torchtpu.__file__)
    assert got.startswith(os.path.realpath(want) + os.sep), (got, want)
