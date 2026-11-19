# SPDX-License-Identifier: Apache-2.0
"""q38 L6-HOST x L5 SAB co-run: L5's served A/B (megakernel_runtime/served_ab.py, whole-step
ON / OFF windows of P driver calls) and L6's host-lever classes (core/l6_ab.py, step windows of
S steps) run on one step stream without changing each other, and every SAB window holds equal
shares of each L6 class. Host-only; CDK CPU pod."""

import os
from types import SimpleNamespace as NS

import pytest

P, DS = 4000, 64          # SAB period and confirmation delay (calls)
S, M, DL = 200, 4, 16     # L6 step window, classes, confirmation delay
N = 2 * P + 3 * S + 37    # steps simulated (two full SAB windows plus a ragged tail)


def _runner(j):
    ids = ["r1", "r2", "r3"]
    return NS(warmup_done=True, input_batch=NS(req_ids=ids, num_computed_tokens_cpu=[100 + 4 * j] * 3))


@pytest.fixture
def env(monkeypatch, tmp_path):
    from vllm_torchtpu.core import l6_ab
    from vllm_torchtpu.megakernel_runtime import served_ab

    for k in list(os.environ):
        if k.startswith("VLLM_TPU_L6_") or k in ("VLLM_TPU_HOST_AB_PERIOD_MS", "VLLM_TPU_D1N_AB_PERIOD_MS",
                                                  "VLLM_TPU_SERVED_AB"):
            monkeypatch.delenv(k, raising=False)
    for mod in (l6_ab, served_ab):
        monkeypatch.setattr(mod.logger, "info", lambda *a, **k: None)
        monkeypatch.setattr(mod.logger, "error", lambda *a, **k: None)
    monkeypatch.setattr(l6_ab, "_rank_world", lambda: (0, 1))
    yield l6_ab, served_ab, tmp_path, monkeypatch
    for k in list(os.environ):
        if k.startswith("VLLM_TPU_L6_") or k == "VLLM_TPU_SERVED_AB":
            monkeypatch.delenv(k, raising=False)
    l6_ab._resolve()


def _arm_l6(l6_ab, monkeypatch, tmp_path, sab_spec):
    monkeypatch.setenv("VLLM_TPU_L6_AB_STEPS", str(S))
    monkeypatch.setenv("VLLM_TPU_L6_AB_CONFIRM", str(DL))
    monkeypatch.setenv("VLLM_TPU_L6_AB_CLASSES", str(M))
    monkeypatch.setenv("VLLM_TPU_L6_ENVS_CACHE", "ab:1,3")
    monkeypatch.setenv("VLLM_TPU_L6_FAST_FWD_CTX", "ab:2,3")
    monkeypatch.setenv("VLLM_TPU_L6_AB_DIR", str(tmp_path / "l6"))
    if sab_spec:
        monkeypatch.setenv("VLLM_TPU_SERVED_AB", sab_spec)
    else:
        monkeypatch.delenv("VLLM_TPU_SERVED_AB", raising=False)
    l6_ab._resolve()
    assert l6_ab.REFUSED is False


def _stream(l6_ab, served_ab, tmp_path, monkeypatch, sab_on, l6_on, sab_spec=f"whole:{P}:{DS}"):
    sab = served_ab.ServedAB(P, DS, 0, 1, str(tmp_path / "sab")) if sab_on else None
    if l6_on:
        _arm_l6(l6_ab, monkeypatch, tmp_path, sab_spec if sab_on else "")
    rec = []
    for j in range(1, N + 1):
        r = _runner(j)
        whole = None
        if sab is not None:
            sab.tick(r)
            whole = sab.whole_on()
        cls = None
        levers = None
        if l6_on:
            l6_ab.begin_step(fp_fn=lambda j=j: f"fp{j}")
            cls = l6_ab.current_class()
            levers = (l6_ab.on("envcache"), l6_ab.on("fctx"))
        rec.append((whole, cls, levers))
    return rec


def test_each_mechanism_is_unchanged_by_the_other(env):
    l6_ab, served_ab, tmp, mp = env
    both = _stream(l6_ab, served_ab, tmp / "a", mp, True, True)
    sab_only = _stream(l6_ab, served_ab, tmp / "b", mp, True, False)
    l6_only = _stream(l6_ab, served_ab, tmp / "c", mp, False, True)
    assert [x[0] for x in both] == [x[0] for x in sab_only]          # SAB's whole-step arm per step
    assert [x[1:] for x in both] == [x[1:] for x in l6_only]          # L6's class and lever states per step
    # both mechanisms actually move: SAB has OFF steps, L6 has all four lever combinations
    assert any(w is False for w, _, _ in both) and any(w is True for w, _, _ in both)
    assert {lv for _, _, lv in both} == {(False, False), (True, False), (False, True), (True, True)}


def test_every_sab_window_holds_equal_l6_class_shares(env):
    l6_ab, served_ab, tmp, mp = env
    rec = _stream(l6_ab, served_ab, tmp, mp, True, True)
    # SAB window k covers its calls k*P .. k*P+P-1 (n counts from 0 = step 1); check the full ones
    for k in range(N // P):
        window = rec[k * P:(k + 1) * P]
        counts = {}
        for _, cls, _ in window:
            counts[cls] = counts.get(cls, 0) + 1
        assert sorted(counts) == ["0", "1", "2", "3"] and set(counts.values()) == {P // M}, (k, counts)
        # and inside the SAB OFF part (after the confirmation delay) the classes are balanced too
        off = [cls for w, cls, _ in window[DS:] if w is False]
        if off:
            share = {c: off.count(c) / len(off) for c in "0123"}
            assert max(share.values()) - min(share.values()) <= 0.10, (k, share)
