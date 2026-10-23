# SPDX-License-Identifier: Apache-2.0
"""ix-unified-port switches that restore what the old trees ran.

Each defaults to this branch's (gated) behaviour:
- TPU_PP_QUAD_COST=0: the PP chunk scheduler without PR #2243's quadratic cost (linear
  keys, waiting work from num_computed_tokens, no pending pairs, no live target ceiling).
- TPU_PP_INFLIGHT_COST=0: no in-flight step costing (4ec32d152 / 06f7b9bf9).
- TPU_SCHED_PAW_PROMOTE_WARMEST=0: FCFS admission, no warmest-first promotion.
- TPU_SCHED_DECODE_INFLIGHT_THROTTLE=1 (default 0): the final-lineage decode admission
  throttle (_decode_inflight_reserved_blocks, fp2q-r5 0447ff58b).
"""
import inspect
import inspect

import pytest

from vllm_torchtpu.core.prefix_aware_scheduler import PrefixAwareScheduler


# ---------------- TPU_SCHED_PAW_PROMOTE_WARMEST ----------------


@pytest.mark.parametrize("val,expect", [(None, True), ("1", True), ("0", False), ("off", False)])
def test_promote_switch(monkeypatch, val, expect):
    if val is None:
        monkeypatch.delenv("TPU_SCHED_PAW_PROMOTE_WARMEST", raising=False)
    else:
        monkeypatch.setenv("TPU_SCHED_PAW_PROMOTE_WARMEST", val)
    s = PrefixAwareScheduler.__new__(PrefixAwareScheduler)
    assert s._promote_warmest_on() is expect


def test_pas_schedule_guards_the_promotion():
    src = inspect.getsource(PrefixAwareScheduler.schedule)
    assert "if self._promote_warmest_on():\n                self._select_waiting_queue_for_scheduling()" in src
