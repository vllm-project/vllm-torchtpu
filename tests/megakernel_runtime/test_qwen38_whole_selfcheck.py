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
"""whole=1 self-check: a server that never makes a whole-step call says so, loudly, early."""

from types import SimpleNamespace

from vllm_torchtpu.megakernel_runtime import qwen38_driver as drv


def _run(monkeypatch, *, steps, calls_per_step, program=object(), warm_at=0):
    errors, infos = [], []
    monkeypatch.setattr(drv.logger, "error", lambda *a: errors.append(a[0] % a[1:]))
    monkeypatch.setattr(drv.logger, "info", lambda *a: infos.append(a[0] % a[1:]))
    d = SimpleNamespace(whole_calls=0, whole_program=program, whole_dry=False, stats={"whole_b4": 0})
    runner = SimpleNamespace(warmup_done=False)
    for i in range(steps):
        runner.warmup_done = i >= warm_at
        d.whole_calls += calls_per_step
        drv.Qwen38SegmentDriver._whole_self_check(d, runner)
    return errors, infos


def test_no_whole_step_call_after_warmup_is_an_error(monkeypatch):
    errors, _ = _run(monkeypatch, steps=drv.WHOLE_SELF_CHECK_STEPS + 50, calls_per_step=0,
                     program=None)
    assert len(errors) == 1 and "SELF-CHECK FAILED" in errors[0] and "disabled" in errors[0]


def test_whole_step_calls_pass_once(monkeypatch):
    errors, infos = _run(monkeypatch, steps=3 * drv.WHOLE_SELF_CHECK_STEPS, calls_per_step=1)
    assert not errors and sum("self-check:" in m for m in infos) == 1


def test_warmup_calls_do_not_count(monkeypatch):
    """Whole-step calls during warmup only, none after: still an error."""
    msgs = []
    monkeypatch.setattr(drv.logger, "error", lambda *a: msgs.append(a[0] % a[1:]))
    d = SimpleNamespace(whole_calls=0, whole_program=object(), whole_dry=False, stats={})
    runner = SimpleNamespace(warmup_done=False)
    for _ in range(10):  # warmup: calls happen, not counted
        d.whole_calls += 1
        drv.Qwen38SegmentDriver._whole_self_check(d, runner)
    runner.warmup_done = True
    for _ in range(drv.WHOLE_SELF_CHECK_STEPS):
        drv.Qwen38SegmentDriver._whole_self_check(d, runner)
    assert len(msgs) == 1 and "nothing eligible" in msgs[0], msgs


def test_dry_mode_reports_without_error(monkeypatch):
    msgs, errs = [], []
    monkeypatch.setattr(drv.logger, "error", lambda *a: errs.append(a[0] % a[1:]))
    monkeypatch.setattr(drv.logger, "info", lambda *a: msgs.append(a[0] % a[1:]))
    d = SimpleNamespace(whole_calls=0, whole_program=object(), whole_dry=True,
                        stats={"dry_whole_b4": 150, "stock_whole_ctx": 50})
    runner = SimpleNamespace(warmup_done=True)
    for _ in range(drv.WHOLE_SELF_CHECK_STEPS):
        drv.Qwen38SegmentDriver._whole_self_check(d, runner)
    assert not errs and len(msgs) == 1 and "wholedry" in msgs[0] and "150 dry" in msgs[0]
