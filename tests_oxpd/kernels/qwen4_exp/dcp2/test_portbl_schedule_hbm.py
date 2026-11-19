# SPDX-License-Identifier: Apache-2.0
"""q38 PORT-BL UNION: VLLM_TPU_RPA_SCHEDULE_HBM_O1S (Benliang Wang's VLLM_TPU_RPA_SCHEDULE_HBM, 2ce02bb1a on
benliangw/q38fh-integ-pin-5b44702aea, scoped to the O1s target attend).

  SB  the schedule kernel's outputs with ``hbm_outputs`` on equal those with it off, leaf by leaf and bit for bit,
      for the exact schedule the served decode call builds (captured from a real ``ragged_paged_attention`` call
      at served dims; the schedule kernel runs in Pallas interpret mode on CPU).
  SE  engagement / scope: with the key on, every batched-RPA call of ``pooled_qsa_request_attend_dcp_step`` (the
      per-row decode, and the union calls when the union key is on too) asks for ``schedule_in_hbm``, and the
      wrapper hands ``hbm_outputs=True`` to the schedule kernel; with the key off none does; the other DCP attend
      (``pooled_qsa_shard_attend_dcp_step``) never does, key on or off.
"""

from __future__ import annotations


import jax
import jax.numpy as jnp
import numpy as np
import pytest
from dcp2_harness import rank_operand

import portbl_union_harness as H
from vllm_torchtpu.kernels.experimental.batched_rpa import schedule, wrapper
from vllm_torchtpu.kernels.qwen4_exp import pooled_qsa, qsa_rpa, qsa_union_meta
from test_portbl_union_o1s import K_SCALE, SERVED, V_SCALE

KEY = "VLLM_TPU_RPA_SCHEDULE_HBM_O1S"
# The wrapper without its jax.jit, so a schedule captured from a real call holds concrete arrays.
EAGER_RPA = wrapper.ragged_paged_attention.__wrapped__


class _Stop(Exception):
    pass


@pytest.fixture
def served(monkeypatch, interpret):
    for k, v in SERVED.items():
        monkeypatch.setenv(k, v)
    for k in (KEY, "VLLM_TPU_QSA_DECODE_RPA_UNION_O1S"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(qsa_union_meta, "union_sc_metadata_staged", H.sc_standin)
    qsa_rpa.UNION_TRACED.clear()
    qsa_rpa.SCHEDULE_HBM_TRACED.clear()
    jax.clear_caches()
    return monkeypatch


@pytest.mark.parametrize("case", ["80", "96", "128", "128half"])
def test_sb_hbm_typed_schedule_is_the_same_schedule(served, case):
    """Every RPA call shape of the served buckets: one call of 80 / 96 / 128 rows; 192 and 256 rows run 128-row
    chunks, the last of 192 half live ("128half")."""
    rows = int(case[:3].rstrip("h"))
    step = H.make_step(seed=rows + 1, tokens=rows)
    if case.endswith("half"):
        step["lengths"] = jnp.where(jnp.arange(rows) < 64, step["lengths"], 0)
    pool = H.random_pool(rows + 1, step["npages"])
    captured = []
    real = schedule.generate_rpa_metadata

    def spy(*args, **kwargs):
        captured.append((args, kwargs))
        raise _Stop

    served.setattr(schedule, "generate_rpa_metadata", spy)
    served.setattr(wrapper, "ragged_paged_attention", EAGER_RPA)
    blocks = jnp.asarray(step["blocks"])
    table = step["main_table"][step["req"]]
    with pytest.raises(_Stop):
        qsa_rpa.decode_attention(step["q"], step["gate"], blocks, step["lengths"], table, pool, kv_heads=1,
                                 ratio=4, k_scale=K_SCALE, v_scale=V_SCALE, dcp=(rank_operand(1), 4),
                                 return_lse=True, schedule_hbm=True)
    args, kwargs = captured[-1]
    assert kwargs["hbm_outputs"] is True
    outs = {}
    for flag in (False, True):
        outs[flag] = real(*args, **{**kwargs, "hbm_outputs": flag})
    la, lb = jax.tree.leaves(outs[False]), jax.tree.leaves(outs[True])
    same = [bool(np.array_equal(np.asarray(a), np.asarray(b))) for a, b in zip(la, lb)]
    steps = int(np.asarray(outs[False].actual_steps).reshape(-1)[0])
    print(f"PBLU SB rows {case}: {len(la)} schedule leaves, equal {sum(same)}/{len(same)}; actual steps {steps}; "
          f"shapes {[tuple(np.shape(x)) for x in la]}")
    assert len(la) == len(lb) and all(same) and steps > 0


def _record_rpa(served, standin):
    seen = []
    inner = H.STANDINS[standin]

    def rec(*args, **kwargs):
        seen.append(("union" if kwargs.get("selection_bits") is not None else "rows",
                     kwargs.get("schedule_in_hbm", False)))
        return inner(*args, **kwargs)

    served.setattr(wrapper, "ragged_paged_attention", rec)
    return seen


@pytest.mark.parametrize("union", [False, True])
def test_se_key_reaches_every_o1s_rpa_call_and_no_other(served, union):
    step = H.make_step(seed=3, tokens=96)
    pool = H.random_pool(3, step["npages"])
    if union:
        served.setenv("VLLM_TPU_QSA_DECODE_RPA_UNION_O1S", "640")
    seen_by = {}
    for key in (False, True):
        if key:
            served.setenv(KEY, "1")
        else:
            served.delenv(KEY, raising=False)
        jax.clear_caches()
        seen = _record_rpa(served, "keyset")
        pooled_qsa.pooled_qsa_request_attend_dcp_step(
            step["q"], step["gate"], step["lengths"], step["req"], step["starts"], step["main_table"],
            step["context"], pool, step["gathered"], rank_operand(0), kv_heads=1, compress_ratio=4,
            token_topk=2048, dcp_size=4, k_scale=K_SCALE, v_scale=V_SCALE, max_query_len=4)
        o1s = list(seen)
        seen.clear()
        pooled_qsa.pooled_qsa_shard_attend_dcp_step(
            step["q"], step["gate"], step["lengths"], step["req"], step["starts"], step["main_table"],
            step["context"], pool, jnp.asarray(step["blocks"]), rank_operand(0), kv_heads=1, compress_ratio=4,
            token_topk=2048, dcp_size=4, k_scale=K_SCALE, v_scale=V_SCALE, max_query_len=4, merged=True)
        other = list(seen)
        seen_by[key] = (o1s, other)
        print(f"PBLU SE union {union} key {key}: O1s attend RPA calls (kind, schedule_in_hbm) {o1s}; "
              f"other DCP attend {other}")
        kinds = {k for k, _ in o1s}
        assert kinds == ({"rows", "union"} if union else {"rows"}), o1s  # traced: both lax.cond branches
        assert all(h is key for _, h in o1s), o1s
        assert other and all(h is False for _, h in other), other
        # The trace-time engagement marker names the O1s program, and only with the key on.
        assert ((96, 64) in qsa_rpa.SCHEDULE_HBM_TRACED) is key, qsa_rpa.SCHEDULE_HBM_TRACED
        qsa_rpa.SCHEDULE_HBM_TRACED.clear()
    # The wrapper hands the flag to the schedule kernel.
    captured = []

    def spy(*args, **kwargs):
        captured.append(kwargs.get("hbm_outputs"))
        raise _Stop

    served.setattr(schedule, "generate_rpa_metadata", spy)
    served.setattr(wrapper, "ragged_paged_attention", EAGER_RPA)
    for flag in (False, True):
        with pytest.raises(_Stop):
            qsa_rpa.decode_attention(step["q"], step["gate"], jnp.asarray(step["blocks"]), step["lengths"],
                                     step["main_table"][step["req"]], pool, kv_heads=1, ratio=4,
                                     k_scale=K_SCALE, v_scale=V_SCALE, dcp=(rank_operand(0), 4), return_lse=True,
                                     **({"schedule_hbm": True} if flag else {}))
    assert captured == [False, True], captured
