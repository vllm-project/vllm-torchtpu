# SPDX-License-Identifier: Apache-2.0
"""q38 L6 lever defer (runner/tpu_runner.py l6_defer_ok + the sample_tokens switch). Host-only; CDK CPU pod."""

import ast
import pathlib
from types import SimpleNamespace as NS

import pytest


@pytest.fixture
def tr(monkeypatch):
    import vllm_torchtpu.runner.tpu_runner as tr

    monkeypatch.setattr(tr, "has_kv_transfer_group", lambda: True)
    return tr


def _runner(role="kv_consumer", drafter=True):
    return NS(vllm_config=NS(kv_transfer_config=NS(kv_role=role)), _is_async_drafter=drafter)


def test_only_a_pure_consumer_with_a_device_seeded_draft_defers(tr, monkeypatch):
    tok = [object()]
    assert tr.l6_defer_ok(_runner(), tok, []) is True
    assert tr.l6_defer_ok(_runner("kv_producer"), tok, []) is False
    assert tr.l6_defer_ok(_runner("kv_both"), tok, []) is False
    assert tr.l6_defer_ok(_runner(drafter=False), tok, []) is False
    assert tr.l6_defer_ok(_runner(), [], []) is False                 # nothing sampled this step
    assert tr.l6_defer_ok(_runner(), tok, [3]) is False               # a partial prefill discards a token
    monkeypatch.setattr(tr, "has_kv_transfer_group", lambda: False)
    assert tr.l6_defer_ok(_runner(), tok, []) is False                # no connector: the stock flag owns it


def test_sample_tokens_switch_sits_before_the_early_wait():
    import vllm_torchtpu.runner.tpu_runner as tr

    src = pathlib.Path(tr.__file__).read_text()
    fn = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "sample_tokens")
    body = ast.unparse(fn)
    i0 = body.index("defer_prev_wait = self.scheduler_config.async_scheduling and envs.VLLM_TPU_DEFER_PREV_RESULTS_WAIT")
    i1 = body.index("_l6_ab.on('defer')")
    i2 = body.index("l6_defer_ok(self, combined_selected_tokens, discard_sampled_tokens_req_indices)")
    i3 = body.index("host_phases.mark('token_d2h_bookkeeping')")
    i4 = body.index("if self.scheduler_config.async_scheduling and (not defer_prev_wait):")
    i5 = body.index("if defer_prev_wait:")
    assert i0 < i1 < i2 < i3 < i4 < i5
    seg = body[i1:i3]
    assert "defer_prev_wait = True" in seg and "_l6_ab.note('defer', 'deferred')" in seg
