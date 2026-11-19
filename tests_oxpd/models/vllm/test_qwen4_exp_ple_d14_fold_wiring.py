# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_PLE_D14_FOLD wiring in Qwen4ExpPooledPLE.bind_pool (q38 L1-PLE; CPU).

The fold changes only the body of the bounded speculative conv op (see
tests/kernels/test_qwen4_exp_ple_d14_fold.py for its bitwise proof). Here:
with VLLM_TPU_PLE_DECODE_BOUNDED set, the flag builds the bounded op with
``fold=True`` (its own op name); without the bounded op the flag is inert and
says so; unset, the builder sees exactly the D07-b2 arguments. A decode step
through forward_with_history (the harness of the D07-b2 wiring test: pool region
I/O as plain XLA stand-ins on a 2-D pool, where the fold's geometry rule keeps
the bounded path) gives a bitwise-equal PLE addend, hidden state and pool with
and without the flag.
"""

import functools
import logging

import jax
import pytest
import torch

from tests.models.vllm import test_qwen4_exp_ple_decode_bounded_wiring as w
from vllm_torchtpu.kernels.qwen4_exp import pool as pool_mod

FOLD = "VLLM_TPU_PLE_D14_FOLD"


def _adapter(monkeypatch, env):
    from vllm_torchtpu.models.vllm.qwen4_exp import pooled_ple

    monkeypatch.delenv(FOLD, raising=False)
    monkeypatch.setattr(pooled_ple, "_FOLD_NOTED", [])
    built = []

    def spec_builder(dilation, state_len, bounded=0, *rest):  # q38 STACK-D v5: rest = (min_rows, fold) (L1-V5)
        built.append((bounded,) + tuple(rest))
        kw = dict(dilation=dilation, state_len=state_len, bounded=bounded)
        if rest and rest[0]:
            kw["bounded_min_rows"] = int(rest[0])
        if len(rest) > 1 and rest[1]:
            kw["fold"] = True
        fn = jax.jit(functools.partial(pool_mod.pooled_short_conv_speculative, **kw))

        def run(x, weight, pool, starts, ids, initial, read_ids, ckpt_ids, windowed):
            new_pool, output = fn(
                *map(w._to_jax, (x, weight, pool, starts, ids, initial, read_ids, ckpt_ids, windowed))
            )
            pool.copy_(w._to_torch(new_pool))
            return w._to_torch(output)

        return run

    # The D07-b2 harness installs its own spec builder; ours replaces it after.
    real_setattr = monkeypatch.setattr

    def setattr_hook(*args, **kwargs):
        if len(args) >= 3 and args[0] is pooled_ple and args[1] == "build_pooled_speculative_short_conv_op":
            args = (args[0], args[1], spec_builder) + tuple(args[3:])
        return real_setattr(*args, **kwargs)

    monkeypatch.setattr = setattr_hook
    try:
        adapted, pool, _ = w._adapter(monkeypatch, env)
    finally:
        monkeypatch.setattr = real_setattr
    return adapted, pool, built


@pytest.mark.parametrize("bounded", ["1", "2"])
def test_fold_flag_builds_the_bounded_op_with_fold(monkeypatch, bounded):
    adapted, pool, built = _adapter(monkeypatch, {w.FLAG: bounded, FOLD: "1"})
    assert built == [(0,), (int(bounded), 0, True)], built  # q38 STACK-D v5: (bounded, min_rows, fold)
    assert adapted._speculative_conv_bounded is not None


def test_fold_flag_unset_builds_exactly_the_d07b2_ops(monkeypatch):
    adapted, pool, built = _adapter(monkeypatch, {w.FLAG: "1"})
    assert built == [(0,), (1,)], built


def test_fold_flag_without_the_bounded_op_is_inert(monkeypatch):
    records = []

    class _Keep(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    log = logging.getLogger("vllm.vllm_torchtpu.models.vllm.qwen4_exp.pooled_ple")
    h = _Keep(level=logging.INFO)
    log.addHandler(h)
    try:
        adapted, pool, built = _adapter(monkeypatch, {FOLD: "1"})
    finally:
        log.removeHandler(h)
    assert built == [(0,)], built
    assert adapted._speculative_conv_bounded is None
    assert any("q38 PLE-D14-FOLD inert" in m for m in records), records


def test_fold_flag_logs_its_bind_line_once(monkeypatch):
    records = []

    class _Keep(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    log = logging.getLogger("vllm.vllm_torchtpu.models.vllm.qwen4_exp.pooled_ple")
    h = _Keep(level=logging.INFO)
    log.addHandler(h)
    old = log.level
    log.setLevel(logging.INFO)
    try:
        adapted, pool, built = _adapter(monkeypatch, {w.FLAG: "1", FOLD: "1"})
        adapted.bind_pool(pool)  # a second bind in the same process logs nothing more
    finally:
        log.removeHandler(h)
        log.setLevel(old)
    lines = [m for m in records if "q38 PLE-D14-FOLD on" in m]
    print("Q38 L1-PLE bind engagement:", lines)
    assert lines == [
        "q38 PLE-D14-FOLD on: ple bounded conv op (VLLM_TPU_PLE_DECODE_BOUNDED=1) built "
        "with the fused conv + checkpoint program"
    ], records


@pytest.mark.parametrize("tokens,live", [(16, 4), (32, 5)])
def test_fold_flag_step_is_bitwise_without_it(monkeypatch, tokens, live):
    results = []
    for env in ({w.FLAG: "1"}, {w.FLAG: "1", FOLD: "1"}):
        torch.manual_seed(1234)  # _randomize_ple draws from the global generator: same weights in both arms
        adapted, pool, built = _adapter(monkeypatch, env)
        step = w._decode_step(tokens * 10 + live, count=8, tokens=tokens, live=live)
        results.append(w._forward(adapted, pool, step, uniform_windows=True))
        monkeypatch.undo()
    base, fold = results
    assert torch.count_nonzero(base[0][: live * w.WINDOW]) > 0
    for got, want, what in zip(fold, base, ("PLE addend", "hidden after PLE", "pool")):
        w._same(got, want, what)


def test_fold_op_builds_through_jax_op():
    """The fold op builds through torch_tpu's real pallas.jax_op next to the
    D07-b2 ops (distinct op names)."""
    from vllm_torchtpu.layers.adapter.custom_ops import ple_op

    ops = [ple_op.build_pooled_speculative_short_conv_op(3, 9, *b) for b in ((1,), (1, 0, True), (2, 0, True))]  # q38 STACK-D v5
    assert all(callable(op) for op in ops)
    assert len({id(op) for op in ops}) == 3
