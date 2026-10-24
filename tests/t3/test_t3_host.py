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
"""IX track 3 host trims (all default off): the qx fast-call guard memo
(VLLM_TPU_T3_GUARD_MEMO), the AOT guard skip for draft programs
(VLLM_TPU_T3_AOT_SKIP_GUARDS) and the connector env-predicate cache
(VLLM_TPU_T3_ENV_CACHE). CPU only: compiled functions use the eager backend.
"""

from __future__ import annotations

import sys
import types

import pytest
import torch

from vllm_torchtpu import _qx_fastcall as qx
from vllm_torchtpu.runner import t3_host

# A module global read inside a compiled function: Dynamo guards on its value.
_G = 1


def _fn_scale(x, n):
    return x * n + 1


def _fn_global(x):
    return x + _G


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
    monkeypatch.setattr(qx, "_MEMO_MODE", "1")
    for k in qx._MEMO_STATS:
        monkeypatch.setitem(qx._MEMO_STATS, k, 0)
    yield


@pytest.fixture
def memo_check(monkeypatch):
    monkeypatch.setattr(qx, "_MEMO_MODE", "check")
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
    monkeypatch.setattr(qx, "_MEMO_MODE", "")
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


def test_memo_mode_env_parsing():
    assert qx._parse_memo_mode(None) == ""
    for off in ("", "0", "false", "OFF", " no "):
        assert qx._parse_memo_mode(off) == ""
    for on in ("1", "true", "on"):
        assert qx._parse_memo_mode(on) == "1"
    assert qx._parse_memo_mode("Check") == "check"


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


def test_check_mode_counts_agreement(memo_check):
    w = _wrapped(_fn_scale)
    x = torch.arange(6.0)
    for _ in range(5):
        torch.testing.assert_close(w(x, 2), _fn_scale(x, 2))
    assert qx._MEMO_STATS["check_ok"] >= 2
    assert qx._MEMO_STATS["check_mismatch"] == 0


def test_check_mode_catches_unkeyed_global_change(memo_check, monkeypatch):
    """A guard on state outside the arguments (a module global) is invisible
    to the memo key. CHECK mode must detect it and still return the right
    answer; this is the documented limitation the CHECK run validates."""
    w = _wrapped(_fn_global)
    x = torch.arange(4.0)
    monkeypatch.setattr(sys.modules[__name__], "_G", 1)
    for _ in range(3):
        torch.testing.assert_close(w(x), x + 1)
    monkeypatch.setattr(sys.modules[__name__], "_G", 5)
    out = w(x)
    torch.testing.assert_close(out, x + 5)
    assert qx._MEMO_STATS["check_mismatch"] >= 1


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
    t3_host._reset_for_tests()
    monkeypatch.setattr(t3_host, "AOT_SKIP_GUARDS", True)
    m = _module(evaluate=False)
    assert t3_host.maybe_skip_aot_guards(m) is True
    assert m.aot_compiled_fn._guard_check_enabled is False
    assert t3_host.maybe_skip_aot_guards(m) is False  # once per module
    keep = _module(evaluate=True)
    assert t3_host.maybe_skip_aot_guards(keep) is False
    assert keep.aot_compiled_fn._guard_check_enabled is True
    unknown = types.SimpleNamespace(aot_compiled_fn=_AOTFn())
    assert t3_host.maybe_skip_aot_guards(unknown) is False
    assert unknown.aot_compiled_fn._guard_check_enabled is True


def test_aot_skip_guards_off_by_default(monkeypatch):
    t3_host._reset_for_tests()
    monkeypatch.setattr(t3_host, "AOT_SKIP_GUARDS", False)
    m = _module(evaluate=False)
    assert t3_host.maybe_skip_aot_guards(m) is False
    assert m.aot_compiled_fn._guard_check_enabled is True


def test_aot_skip_guards_waits_for_compiled_fn(monkeypatch):
    t3_host._reset_for_tests()
    monkeypatch.setattr(t3_host, "AOT_SKIP_GUARDS", True)
    m = types.SimpleNamespace(
        compilation_config=types.SimpleNamespace(dynamic_shapes_config=_DS(False))
    )
    assert t3_host.maybe_skip_aot_guards(m) is False  # not compiled yet
    m.aot_compiled_fn = _AOTFn()
    assert t3_host.maybe_skip_aot_guards(m) is True


def test_connector_env_cache(monkeypatch):
    tc = pytest.importorskip("vllm_torchtpu.distributed.kv_transfer.tpu_connector")
    monkeypatch.setattr(tc, "_t3_env_cache", {})
    monkeypatch.setenv("TPU_KV_RESHARD_TRANSPORT", "raiden")
    monkeypatch.setattr(tc, "_T3_ENV_CACHE", False)
    assert tc._use_raiden_stage3_transport() is True
    monkeypatch.setenv("TPU_KV_RESHARD_TRANSPORT", "zmq")
    assert tc._use_raiden_stage3_transport() is False  # off: follows env
    monkeypatch.setattr(tc, "_T3_ENV_CACHE", True)
    monkeypatch.setenv("TPU_KV_RESHARD_TRANSPORT", "raiden")
    assert tc._use_raiden_stage3_transport() is True
    monkeypatch.setenv("TPU_KV_RESHARD_TRANSPORT", "zmq")
    assert tc._use_raiden_stage3_transport() is True  # on: first answer kept
    glm = tc._use_raiden_glm_admission()
    assert tc._use_raiden_glm_admission() is glm
    assert set(tc._t3_env_cache) == {"stage3", "glm"}
