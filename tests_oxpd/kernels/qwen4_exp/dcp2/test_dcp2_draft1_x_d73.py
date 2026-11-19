# SPDX-License-Identifier: Apache-2.0
"""q38 STACK-D v4 composed check: X-D14-DRAFT-1 (VLLM_TPU_QSA_DRAFT_NO_INDEX_STATE) x L2 D73
(VLLM_TPU_QSA_FUSED_KV_WRITE). D83S-BUILD for L4-E / L1, 2026-11-13.

With DRAFT-1 on, the MTP draft's ``_noidx`` steps call ``qsa_cache_writes(index_state=False)``,
whose early return (main K/V through ``write_main_cache``) comes before D73's fused-writer
decision, so the draft never takes D73's branch (C). The pair had never been run together.

1. Bitwise, DRAFT-1 on, D73's env on vs off, over DRAFT-1's 8-step closed-loop draft
   (tests/kernels/qwen4_exp/dcp2/test_dcp2_draft_no_index_state.py's harness: step 0 the rows
   step over a verify chunk, steps 1-7 one token per request, each step's draft token feeding
   the next) at DCP 1 (the replicated ops) and DCP 2 (both ranks of the VLLM_TPU_QSA_MTP_KV_DCP
   ops): draft tokens, per-rank outputs / LSEs / selected tokens and every pool byte (the
   draft's main K/V pages and its untouched index pages) are identical at every step. Controls:
   main K/V was written, the index pages kept their initial bytes, the tokens are not degenerate.
2. Engagement (traced with make_jaxpr, real TPU7x info; the fused writer's kernels stubbed and
   counted, they are not what is measured): over the same 3 draft passes, D73's env routes the
   index-state (DRAFT-1 off) layer onto the fused writer (fused_layer_write x3, write_main_cache
   x0, the concurrent ring read x3) and the env-off layer onto the stock writers (x0 / x3 / x0),
   so the env bites in this harness and in this process; with DRAFT-1 on the draft takes
   write_main_cache x3 and the fused writer x0 under either env.
Mutation: tests/q38_d1xd73/run_cpu.sh moves the early return after D73's decision (the draft
takes D73's choice and skips write_main_cache with the env on): both items must fail.
Same harness as ``test_dcp2_draft_no_index_state`` (CPU, Pallas interpret, RPA stand-in).
"""

import collections
import functools
import os

import jax
import numpy as np
import pytest
from dcp2_harness import raw
from test_dcp2_draft_no_index_state import (
    QUERY_LENS,
    STEPS,
    _bytes,
    _initial_pools,
    _one_step,
    _setup,
    _step_inputs,
    _three_passes,
    tpu7x_info,  # noqa: F401 (fixture)
)
from test_dcp2_op import partial_kernels, stub_select  # noqa: F401 (fixture)

from vllm_torchtpu.kernels.qwen4_exp import fused_kv_write, pooled_qsa

D73 = "VLLM_TPU_QSA_FUSED_KV_WRITE"
DRAFT1 = True  # the _noidx steps (VLLM_TPU_QSA_DRAFT_NO_INDEX_STATE=1 on the torch side)


def _d73(monkeypatch, on):
    if on:
        monkeypatch.setenv(D73, "1")
    else:
        monkeypatch.delenv(D73, raising=False)


def run_d73_lockstep(variant, monkeypatch, seed=0):
    d = _setup(seed)
    start = _initial_pools(d, variant)
    pools = {False: list(start), True: list(start)}
    initial = [raw(p) for p in start]
    outside = np.ones(len(initial[0]), bool)
    outside[d.index_pages] = False
    tokens, history = None, []
    for i in range(STEPS):
        inp = _step_inputs(d, i, tokens)
        got = {}
        for env in (False, True):
            _d73(monkeypatch, env)
            got[env] = _one_step(d, variant, DRAFT1, inp, pools[env])
        _d73(monkeypatch, False)
        assert got[False][2] == got[True][2], f"step {i}: draft tokens differ (D73 off {got[False][2]} on {got[True][2]})"
        for r, (off, on) in enumerate(zip(got[False][1], got[True][1])):
            for name in off:
                np.testing.assert_array_equal(_bytes(on[name]), _bytes(off[name]),
                                              err_msg=f"step {i} rank {r}: {name} differs")
        for r in range(len(initial)):
            p_off, p_on = raw(got[False][0][r]), raw(got[True][0][r])
            np.testing.assert_array_equal(p_on[outside], p_off[outside],
                                          err_msg=f"step {i} rank {r}: the draft's main K/V bytes differ")
            np.testing.assert_array_equal(p_on, p_off, err_msg=f"step {i} rank {r}: pool bytes differ")
            np.testing.assert_array_equal(p_on[d.index_pages], initial[r][d.index_pages],
                                          err_msg=f"step {i} rank {r}: index state written under D73")
        pools = {env: got[env][0] for env in (False, True)}
        tokens = got[False][2]
        history.append(tokens)
    for r in range(len(initial)):
        for env in (False, True):
            assert not np.array_equal(raw(pools[env][r])[outside], initial[r][outside]), \
                f"rank {r} D73 {'on' if env else 'off'}: no main K/V was written"
    flat = [t for step in history for t in step if t >= 0]
    assert len(set(flat)) > 1, f"degenerate draft tokens {history}"
    return history


def _origin_line():
    return "D1XD73_ORIGIN pooled_qsa=%s fused_kv_write=%s" % (
        os.path.realpath(pooled_qsa.__file__), os.path.realpath(fused_kv_write.__file__))


def test_eight_step_draft1_bitwise_d73_on_vs_off_dcp2(interpret, rpa_stand_in, monkeypatch, partial_kernels):
    monkeypatch.setattr(pooled_qsa, "select_blocks", stub_select)
    history = run_d73_lockstep("dcp", monkeypatch)
    print("D1XD73_BITWISE dcp=2 steps", len(history), "tokens", history)
    print(_origin_line())


def test_eight_step_draft1_bitwise_d73_on_vs_off_dcp1(interpret, rpa_stand_in, monkeypatch):
    monkeypatch.setattr(pooled_qsa, "select_blocks", stub_select)
    history = run_d73_lockstep("replicated", monkeypatch)
    print("D1XD73_BITWISE dcp=1 steps", len(history), "tokens", history)
    print(_origin_line())


@pytest.mark.parametrize("variant", ["dcp", "replicated"])
def test_d73_env_bites_but_the_draft1_draft_bypasses_it(tpu7x_info, rpa_stand_in, monkeypatch, variant):
    monkeypatch.setattr(pooled_qsa, "select_blocks", stub_select)
    calls = collections.Counter()

    def counting(name, fn):
        def call(*args, **kwargs):
            calls[name] += 1
            return fn(*args, **kwargs)
        return call

    monkeypatch.setattr(pooled_qsa, "write_main_cache", counting("write_main_cache", pooled_qsa.write_main_cache))
    # The fused writer's kernels are stubbed (counted, the pool passes through; the batched ring
    # reader stands in for the concurrent one): this item counts which writer path each layer takes.
    monkeypatch.setattr(pooled_qsa, "fused_layer_write", counting("fused_layer_write", lambda pool, **_: pool))
    monkeypatch.setattr(pooled_qsa, "read_ring_and_tags_concurrent",
                        counting("ring_read_concurrent", pooled_qsa.read_ring_and_tags))
    d = _setup()
    pool = _initial_pools(d, variant)[0]
    got = {}
    for draft1 in (False, True):
        for env in (False, True):
            _d73(monkeypatch, env)
            calls.clear()
            jax.make_jaxpr(functools.partial(_three_passes, d, variant, draft1))(pool)
            got[(draft1, env)] = tuple(calls[k] for k in ("fused_layer_write", "write_main_cache",
                                                         "ring_read_concurrent"))
            print(f"D1XD73_ENGAGE variant={variant} draft1={'on' if draft1 else 'off'} d73={'on' if env else 'off'} "
                  f"fused_layer_write={got[(draft1, env)][0]} write_main_cache={got[(draft1, env)][1]} "
                  f"ring_read_concurrent={got[(draft1, env)][2]}")
    _d73(monkeypatch, False)
    passes = 3
    assert got[(False, True)] == (passes, 0, passes), got  # D73 engages on an index-state layer here
    assert got[(False, False)] == (0, passes, 0), got
    assert got[(True, True)] == (0, passes, 0), got  # the DRAFT-1 draft bypasses D73
    assert got[(True, False)] == (0, passes, 0), got
    assert len(QUERY_LENS) == 5
