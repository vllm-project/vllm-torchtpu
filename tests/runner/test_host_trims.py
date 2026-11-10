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
"""The batch-1 host trims (``runner.host_trims``): the rule that turns them
on, the fast-call guard memo, the AOT guard skip for draft programs, the
connector predicate cache, the connector stats cadence, the DP-size cache and
the process-wide ``tpu_runner`` settings. CPU only: compiled functions use
the eager backend.
"""

from __future__ import annotations

import sys
import types

import pytest
import torch

from vllm_torchtpu import _qx_fastcall as qx
from vllm_torchtpu import utils as vu
from vllm_torchtpu.runner import fast_forward_context as ffc
from vllm_torchtpu.runner import host_trims, runner_options


def _fn_scale(x, n):
    return x * n + 1


def _fn_shape(x):
    # dynamic=False bakes the size into the graph: a wrong cache entry gives
    # a wrong value even on the eager backend.
    return x + x.shape[0]


def _wrapped(fn):
    torch._dynamo.reset()
    compiled = torch.compile(fn, backend="eager", fullgraph=True, dynamic=False)
    return qx._wrap(compiled, fn)


@pytest.fixture
def memo_on(monkeypatch):
    # The memo sits on the direct dispatch, which tpu_compile.fast_compiled_call
    # arms; both are off by default.
    monkeypatch.setitem(qx._CONFIG, "fast", True)
    monkeypatch.setitem(qx._MEMO, "on", True)
    for k in qx._MEMO_STATS:
        monkeypatch.setitem(qx._MEMO_STATS, k, 0)
    yield


def test_memo_key_covers_metadata_scalars_and_identity():
    a = torch.zeros(4, 8)
    obj = object()
    k1 = qx._memo_key((a, 3, None, "s", obj, [a, 2]))
    assert k1 == qx._memo_key((torch.ones(4, 8), 3, None, "s", obj, [a, 2]))
    # Tensor values are not part of the key; metadata is.
    assert k1 != qx._memo_key((torch.zeros(4, 9), 3, None, "s", obj, [a, 2]))
    assert k1 != qx._memo_key((torch.zeros(8, 4).t(), 3, None, "s", obj, [a, 2]))
    assert k1 != qx._memo_key((a.to(torch.bfloat16), 3, None, "s", obj, [a, 2]))
    assert k1 != qx._memo_key((a.requires_grad_(False), 4, None, "s", obj, [a, 2]))
    assert k1 != qx._memo_key((a, 3, None, "s", object(), [a, 2]))
    assert k1 != qx._memo_key((a, 3, None, "s", obj, [a, 3]))
    assert k1 != qx._memo_key((a, True, None, "s", obj, [a, 2]))  # bool vs int
    assert qx._memo_key(([0] * (qx._MEMO_MAX_SEQ + 1),)) is None


def test_memo_default_off_leaves_dispatch_unchanged(monkeypatch):
    monkeypatch.setitem(qx._CONFIG, "fast", True)
    monkeypatch.setitem(qx._MEMO, "on", False)
    w = _wrapped(_fn_scale)
    x = torch.arange(6.0)
    for _ in range(4):
        torch.testing.assert_close(w(x, 2), _fn_scale(x, 2))
    assert w._qx_fast_state["memo"] == {}


def test_memo_hits_and_results_match_eager(memo_on):
    w = _wrapped(_fn_scale)
    xs = [torch.arange(6.0), torch.arange(6.0) + 3, torch.arange(10.0)]
    for _ in range(3):
        for x in xs:
            for n in (2, 5):
                torch.testing.assert_close(w(x, n), _fn_scale(x, n))
    assert qx._MEMO_STATS["hit"] > 0
    assert qx._MEMO_STATS["store"] >= 2
    # Same shape and n -> one memo entry serves both tensor values.
    keys = w._qx_fast_state["memo"]
    assert all(len(k) == 2 for k in keys)


def test_memo_respects_scalar_value_and_shape(memo_on):
    w = _wrapped(_fn_scale)
    x6, x10 = torch.arange(6.0), torch.arange(10.0)
    for _ in range(2):
        torch.testing.assert_close(w(x6, 2), _fn_scale(x6, 2))
        torch.testing.assert_close(w(x6, 7), _fn_scale(x6, 7))
        torch.testing.assert_close(w(x10, 2), _fn_scale(x10, 2))
    # After warm-up every repeat is a memo hit, and still right.
    hits = qx._MEMO_STATS["hit"]
    for n in (2, 7):
        torch.testing.assert_close(w(x6, n), _fn_scale(x6, n))
    torch.testing.assert_close(w(x10, 2), _fn_scale(x10, 2))
    assert qx._MEMO_STATS["hit"] == hits + 3


def test_memo_never_reuses_an_entry_across_shapes(memo_on):
    w = _wrapped(_fn_shape)
    x6, x10 = torch.zeros(6), torch.zeros(10)
    for _ in range(3):
        torch.testing.assert_close(w(x6), x6 + 6)
        torch.testing.assert_close(w(x10), x10 + 10)
    assert qx._MEMO_STATS["hit"] >= 2


def test_memo_cleared_when_cache_entries_change(memo_on):
    w = _wrapped(_fn_scale)
    x = torch.arange(6.0)
    w(x, 2)
    w(x, 2)
    w(x, 2)
    st = w._qx_fast_state
    assert st["memo"], "warm signature should be memoized"
    count_before = st["memo_count"]
    # A new specialization recompiles: entry count changes, the memo is reset.
    w(torch.arange(12.0), 3)
    assert st["count"] != count_before
    assert st["memo"] == {}
    torch.testing.assert_close(w(x, 2), _fn_scale(x, 2))


class _DS:
    def __init__(self, evaluate):
        self.evaluate_guards = evaluate


class _AOTFn:
    def __init__(self):
        self._guard_check_enabled = True

    def disable_guard_check(self):
        self._guard_check_enabled = False


def _module(evaluate):
    m = types.SimpleNamespace()
    m.compilation_config = types.SimpleNamespace(dynamic_shapes_config=_DS(evaluate))
    m.aot_compiled_fn = _AOTFn()
    return m


def test_aot_skip_guards_matches_vllm_load_policy(monkeypatch):
    host_trims._reset_for_tests()
    monkeypatch.setattr(host_trims, "AOT_SKIP_GUARDS", True)
    m = _module(evaluate=False)
    assert host_trims.maybe_skip_aot_guards(m) is True
    assert m.aot_compiled_fn._guard_check_enabled is False
    assert host_trims.maybe_skip_aot_guards(m) is False  # once per module
    keep = _module(evaluate=True)
    assert host_trims.maybe_skip_aot_guards(keep) is False
    assert keep.aot_compiled_fn._guard_check_enabled is True
    unknown = types.SimpleNamespace(aot_compiled_fn=_AOTFn())
    assert host_trims.maybe_skip_aot_guards(unknown) is False
    assert unknown.aot_compiled_fn._guard_check_enabled is True


def test_aot_skip_guards_off_by_default(monkeypatch):
    host_trims._reset_for_tests()
    monkeypatch.setattr(host_trims, "AOT_SKIP_GUARDS", False)
    m = _module(evaluate=False)
    assert host_trims.maybe_skip_aot_guards(m) is False
    assert m.aot_compiled_fn._guard_check_enabled is True


def test_aot_skip_guards_waits_for_compiled_fn(monkeypatch):
    host_trims._reset_for_tests()
    monkeypatch.setattr(host_trims, "AOT_SKIP_GUARDS", True)
    m = types.SimpleNamespace(
        compilation_config=types.SimpleNamespace(dynamic_shapes_config=_DS(False))
    )
    assert host_trims.maybe_skip_aot_guards(m) is False  # not compiled yet
    m.aot_compiled_fn = _AOTFn()
    assert host_trims.maybe_skip_aot_guards(m) is True


def test_connector_env_cache(monkeypatch):
    tc = pytest.importorskip("vllm_torchtpu.distributed.kv_transfer.tpu_connector")
    monkeypatch.setattr(tc, "_env_predicate_cache", {})
    monkeypatch.setenv("TPU_KV_RESHARD_TRANSPORT", "raiden")
    monkeypatch.setattr(host_trims, "ENABLED", False)
    assert tc._use_raiden_stage3_transport() is True
    monkeypatch.setenv("TPU_KV_RESHARD_TRANSPORT", "zmq")
    assert tc._use_raiden_stage3_transport() is False  # off: follows env
    monkeypatch.setattr(host_trims, "ENABLED", True)
    monkeypatch.setenv("TPU_KV_RESHARD_TRANSPORT", "raiden")
    assert tc._use_raiden_stage3_transport() is True
    monkeypatch.setenv("TPU_KV_RESHARD_TRANSPORT", "zmq")
    assert tc._use_raiden_stage3_transport() is True  # on: first answer kept
    glm = tc._use_raiden_glm_admission()
    assert tc._use_raiden_glm_admission() is glm
    assert set(tc._env_predicate_cache) == {"stage3", "glm"}


# --- KV connector stats cadence ----------------------------------------------


def _get_finished(monkeypatch, every, samples=((), ())):
    from vllm_torchtpu.runner import tpu_runner as tr

    calls = {"stats": 0}

    class Conn:
        def get_transfer_results(self, finished):
            return types.SimpleNamespace(
                finished_sending=set(), finished_recving=set(), failed_recving=set()
            )

        def get_block_ids_with_load_errors(self):
            return set()

        def build_connector_worker_meta(self):
            return None

        def get_kv_connector_stats(self):
            calls["stats"] += 1
            return {"n": calls["stats"]}

        def clear_connector_metadata(self):
            pass

    conn = Conn()
    monkeypatch.setattr(tr, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(tr, "get_kv_transfer_group", lambda: conn)
    monkeypatch.setattr(host_trims, "CONN_STATS_EVERY", every)
    runner = types.SimpleNamespace(
        _stage_timing=types.SimpleNamespace(take_samples=lambda: samples),
        kv_cache_config=types.SimpleNamespace(kv_cache_groups=[1]),
    )
    so = types.SimpleNamespace(finished_req_ids=set())
    outs = [
        tr.TPUModelRunner.get_finished_kv_transfers(runner, so)[6] for _ in range(8)
    ]
    return outs, calls["stats"]


def test_conn_stats_every_step_by_default(monkeypatch):
    outs, n = _get_finished(monkeypatch, 1)
    assert n == 8 and all(o is not None for o in outs)


def test_conn_stats_throttled(monkeypatch):
    outs, n = _get_finished(monkeypatch, 4)
    assert n == 2
    assert [o is not None for o in outs] == [False, False, False, True] * 2


def test_conn_stats_polled_when_runner_timings_pending(monkeypatch):
    outs, n = _get_finished(monkeypatch, 4, samples=([1.0], []))
    assert n == 8 and all(o is not None for o in outs)


# --- process-wide tpu_runner settings ------------------------------------------


def test_switch_interval_applied_from_the_first_step(monkeypatch):
    orig = sys.getswitchinterval()
    monkeypatch.setattr(host_trims, "SWITCH_US", 200)
    monkeypatch.setattr(host_trims, "GC_FREEZE", False)
    monkeypatch.setattr(
        host_trims, "_hook", {"n": 0, "applied": False, "frozen": False}
    )
    try:
        host_trims.on_step()
        assert abs(sys.getswitchinterval() - 200e-6) < 1e-9
        host_trims.on_step()
        assert abs(sys.getswitchinterval() - 200e-6) < 1e-9
    finally:
        sys.setswitchinterval(orig)


def test_gc_freeze_after_n_steps(monkeypatch):
    import gc as gc_mod

    orig = gc_mod.get_threshold()
    monkeypatch.setattr(host_trims, "SWITCH_US", 0)
    monkeypatch.setattr(host_trims, "GC_FREEZE", True)
    monkeypatch.setattr(host_trims, "GC_FREEZE_AFTER_STEPS", 3)
    monkeypatch.setattr(host_trims, "GC_FREEZE_GEN0_THRESHOLD", 12345)
    monkeypatch.setattr(
        host_trims, "_hook", {"n": 0, "applied": False, "frozen": False}
    )
    try:
        host_trims.on_step()
        host_trims.on_step()
        assert gc_mod.get_threshold()[0] == orig[0] and not host_trims._hook["frozen"]
        host_trims.on_step()
        assert host_trims._hook["frozen"] and gc_mod.get_threshold()[0] == 12345
        assert gc_mod.get_freeze_count() > 0
    finally:
        gc_mod.unfreeze()
        gc_mod.set_threshold(*orig)


def test_dp_size_cache(monkeypatch):
    pc1 = types.SimpleNamespace(data_parallel_size=4)
    pc2 = types.SimpleNamespace(data_parallel_size=2)
    monkeypatch.delenv("TORCH_TPU_DP_SIZE", raising=False)
    monkeypatch.setattr(vu, "_dp_size_cache", [None, 0])
    monkeypatch.setattr(vu, "_DP_SIZE_CACHE", [False])
    assert vu.get_dp_size(pc1) == 4
    vu._DP_SIZE_CACHE[0] = True
    assert vu.get_dp_size(pc1) == 4
    pc1.data_parallel_size = 8  # cached for the same config object
    assert vu.get_dp_size(pc1) == 4
    assert vu.get_dp_size(pc2) == 2  # another config object refreshes
    monkeypatch.setenv("TORCH_TPU_DP_SIZE", "3")
    assert vu.get_dp_size(types.SimpleNamespace(data_parallel_size=1)) == 3


# --- the batch-1 rule ------------------------------------------------------------


def _config(max_num_seqs, kv_role=None, runner=None):
    kv = None
    if kv_role is not None:
        kv = types.SimpleNamespace(
            is_kv_producer=kv_role in ("kv_producer", "kv_both"),
            is_kv_consumer=kv_role in ("kv_consumer", "kv_both"),
        )
    additional = {} if runner is None else {"tpu_runner": runner}
    return types.SimpleNamespace(
        scheduler_config=types.SimpleNamespace(max_num_seqs=max_num_seqs),
        kv_transfer_config=kv,
        additional_config=additional,
    )


@pytest.fixture
def restore_trims():
    yield
    host_trims.configure(_config(256))
    host_trims._configured = False


@pytest.mark.parametrize(
    "max_num_seqs, kv_role, expected",
    [
        (1, None, True),
        (1, "kv_consumer", True),
        (1, "kv_both", True),
        (1, "kv_producer", False),
        (2, None, False),
        (2, "kv_consumer", False),
        (256, None, False),
    ],
)
def test_rule_is_batch1_decode(max_num_seqs, kv_role, expected):
    assert host_trims.rule(_config(max_num_seqs, kv_role)) is expected


def test_configure_turns_every_trim_on_and_off(restore_trims):
    host_trims.configure(_config(1, "kv_consumer"))
    assert host_trims.ENABLED and host_trims.AOT_SKIP_GUARDS
    assert host_trims.CONN_STATS_EVERY == host_trims.CONN_STATS_EVERY_ON == 16
    assert ffc.enabled() and qx._MEMO["on"] and vu._DP_SIZE_CACHE[0]
    host_trims.configure(_config(2, "kv_consumer"))
    assert not host_trims.ENABLED and not host_trims.AOT_SKIP_GUARDS
    assert host_trims.CONN_STATS_EVERY == 1
    assert not ffc.enabled() and not qx._MEMO["on"] and not vu._DP_SIZE_CACHE[0]


def test_old_environment_switches_are_ignored(monkeypatch, restore_trims):
    for var in (
        "VLLM_TPU_T3_HOST",
        "VLLM_TPU_T3_GUARD_MEMO",
        "VLLM_TPU_T3_AOT_SKIP_GUARDS",
        "VLLM_TPU_T3_ENV_CACHE",
        "VLLM_TPU_T3_CONN_STATS",
        "VLLM_TPU_T3_DP_CACHE",
        "VLLM_TPU_FAST_FORWARD_CONTEXT",
    ):
        monkeypatch.setenv(var, "1")
    monkeypatch.setenv("VLLM_TPU_T3_SWITCH_US", "200")
    monkeypatch.setenv("VLLM_TPU_T3_GC", "freeze")
    host_trims.configure(_config(4))
    assert not host_trims.ENABLED and not ffc.enabled() and not qx._MEMO["on"]
    assert host_trims.CONN_STATS_EVERY == 1 and not host_trims.STEP_HOOK


def test_process_settings_come_from_the_tpu_runner_section(restore_trims):
    host_trims.configure(_config(1))
    assert host_trims.SWITCH_US == 0 and not host_trims.GC_FREEZE
    assert not host_trims.STEP_HOOK
    host_trims.configure(_config(4, runner={"switch_interval_us": 200}))
    assert host_trims.SWITCH_US == 200 and host_trims.STEP_HOOK
    assert not host_trims.ENABLED  # the settings do not depend on the rule
    host_trims.configure(_config(4, runner={"gc_freeze": True}))
    assert host_trims.GC_FREEZE and host_trims.STEP_HOOK


@pytest.mark.parametrize(
    "raw, match",
    [
        ({"switch_interval_us": 200.0}, "must be a int"),
        ({"switch_interval_us": True}, "must be a int"),
        ({"switch_interval_us": -1}, ">= 0"),
        ({"gc_freeze": 1}, "must be a bool"),
    ],
)
def test_process_settings_parse_strictly(raw, match):
    with pytest.raises(ValueError, match=match):
        runner_options.parse(raw)


def test_scheduler_step_configures_its_process_once(monkeypatch, restore_trims):
    calls = []
    monkeypatch.setattr(host_trims, "_configured", False)
    real = host_trims.configure
    monkeypatch.setattr(
        host_trims, "configure", lambda cfg: (calls.append(cfg), real(cfg))
    )
    steps = []
    monkeypatch.setattr(host_trims, "on_step", lambda: steps.append(1))
    cfg = _config(4, runner={"switch_interval_us": 200})
    host_trims.on_scheduler_step(cfg)
    host_trims.on_scheduler_step(cfg)
    assert calls == [cfg] and len(steps) == 2


def test_the_runner_applies_the_rule_when_it_is_built():
    # Building a TPUModelRunner needs a TPU, so check the call in its source.
    import inspect

    from vllm_torchtpu.runner import tpu_runner as tr

    assert "host_trims.configure(self.vllm_config)" in inspect.getsource(
        tr.TPUModelRunner.__init__
    )
