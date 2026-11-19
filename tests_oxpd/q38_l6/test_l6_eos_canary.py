# SPDX-License-Identifier: Apache-2.0
"""q38 L6 end-token canary (core/l6_eos_canary.py): host-only tests; run in a CDK CPU pod."""

import inspect
import os

import pytest

from vllm_torchtpu.core import l6_ab
from vllm_torchtpu.core import l6_eos_canary as eos

END = 248044


@pytest.fixture
def act(monkeypatch):
    """l6_ab active (one uniform lever set) and this process rank 0; module state restored after."""
    for k in list(os.environ):
        if k.startswith("VLLM_TPU_L6_") or k == "VLLM_TPU_HOST_AB_PERIOD_MS":
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("VLLM_TPU_L6_ENVS_CACHE", "1")
    l6_ab._resolve()
    l6_ab.begin_step()
    monkeypatch.setattr(eos, "IDS", frozenset({END}))
    monkeypatch.setattr(eos, "_rank", [0])
    yield monkeypatch
    for k in list(os.environ):
        if k.startswith("VLLM_TPU_L6_"):
            monkeypatch.delenv(k, raising=False)
    l6_ab._resolve()


def test_parse_ids():
    assert eos._parse("248044") == frozenset({248044})
    assert eos._parse("248044+248043") == eos._parse("248044,248043") == frozenset({248044, 248043})
    assert eos._parse("") == eos._parse("0") == eos._parse("  ") == frozenset()
    with pytest.raises(ValueError):
        eos._parse("-1")


def test_counts_tokens_and_end_tokens_on_rank0(act):
    eos.count([[1, END, 3], [END], [], [7, 8]])
    c = l6_ab._state["counts"]
    assert c.get("eos.tok") == 6 and c.get("eos.end") == 2, c
    eos.count([[5]])
    assert c.get("eos.tok") == 7 and c.get("eos.end") == 2, c


def test_other_ranks_count_nothing(act):
    act.setattr(eos, "_rank", [3])
    eos.count([[END, END]])
    assert "eos.tok" not in l6_ab._state["counts"] and "eos.end" not in l6_ab._state["counts"]


def test_inactive_l6_ab_counts_nothing(act):
    act.delenv("VLLM_TPU_L6_ENVS_CACHE", raising=False)
    l6_ab._resolve()
    assert not l6_ab.ACTIVE
    eos.count([[END]])
    assert l6_ab._state["counts"] == {}


def test_counts_reach_the_window_line(act):
    lines = []
    act.setattr(l6_ab.logger, "info", lambda fmt, *a: lines.append(fmt % a))
    clock = {"t": 7 * 20000 * 1_000_000}
    act.setattr(l6_ab, "_now_ns", lambda: clock["t"])
    l6_ab._resolve()
    l6_ab.begin_step()
    eos.count([[END, 2, 3]])
    clock["t"] += 21000 * 1_000_000
    l6_ab.begin_step()
    assert any("role=worker" in ln and "eos.end=1" in ln and "eos.tok=3" in ln for ln in lines), lines


def test_runner_counts_after_the_discards_and_before_the_rollback():
    """The hook sits in _modify_prev_results after the discarded requests' tokens are cleared (so they are not
    counted) and before the rollback loop; it is guarded by the flag."""
    from vllm_torchtpu.runner import tpu_runner

    src = inspect.getsource(tpu_runner.TPUModelRunner._modify_prev_results)
    i_clear = src.index("valid_sampled_token_ids[i].clear()")
    i_hook = src.index("_l6_eos.count(valid_sampled_token_ids)")
    i_loop = src.index("for pre_req_idx, req_state, seq_len, req_id in pre_request_seq_lens:")
    assert i_clear < i_hook < i_loop
    assert "if _l6_eos.ON:" in src[i_clear:i_hook]


def test_flag_registered_and_compile_ignored():
    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation import tpu_compiler

    assert eos.ENV in envs.environment_variables
    assert eos.ENV in tpu_compiler._TPU_COMPILE_ENV_IGNORED


def test_solo_phase_steps_are_counted_apart(act):
    """A step where exactly one request produced tokens also counts eos.tok1 / eos.end1 (IX: empty batch-1 answers
    reproduce with one request in flight); steps with two or more active requests do not."""
    eos.count([[END, 5], [], []])          # solo: one active request
    eos.count([[1], [END]])                # two active requests
    eos.count([[7, 8, END]])               # solo
    c = l6_ab._state["counts"]
    assert c.get("eos.tok") == 7 and c.get("eos.end") == 3, c
    assert c.get("eos.tok1") == 5 and c.get("eos.end1") == 2, c
