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
"""Megakernel failures are fatal unless VLLM_TPU_MK_ALLOW_FALLBACK=1.

za8d: the whole-step trace raised (PLE_TRIM AttributeError), the driver fell back to
the stock path with the MTP tail still attached, and the server answered garbage
(GSM8K 0.004). These tests force the same failure points on a stub driver.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.megakernel_runtime import qwen38_driver as drv
from vllm_torchtpu.megakernel_runtime.qwen38_segment_call import T_VERIFY


def _driver(whole_program):
    d = drv.Qwen38SegmentDriver.__new__(drv.Qwen38SegmentDriver)
    d.__dict__.update(
        program=object(), shadow=None, whole_program=whole_program, whole=False,
        seq=0, buckets=(4,), whole_above="stock", whole_dry=False, mk_calls=10,
        segx_left=0, h0check_left=0, warm=False, check_left=0, wholecheck_left=0,
        whole_shadow=None, whole_calls=0, whole_max_rows=32, whole_cap=1 << 20,
        stats={},
    )
    d._note = lambda *a, **k: None
    d._count = lambda *a, **k: None
    d._whole_contexts = lambda runner: contextlib.nullcontext()
    return d


def _runner():
    tail, mk_draft = object(), object()
    draft = SimpleNamespace(model=SimpleNamespace(mk_tail=tail, mk_draft=mk_draft))
    return SimpleNamespace(
        drafter=SimpleNamespace(draft_model=draft),
        input_batch=SimpleNamespace(num_computed_tokens_cpu=np.array([100])),
        warmup_done=True,
    ), draft


def _aot_failure(**_):
    raise AttributeError("'VocabParallelEmbedding' object has no attribute 'reduce_results'")


@pytest.fixture
def step(monkeypatch):
    monkeypatch.setattr(
        drv.MegakernelDriver, "_inspect_step_detail",
        staticmethod(lambda *a: (True, 1, T_VERIFY, "")),
    )


def test_failed_whole_step_trace_raises_by_default(monkeypatch, step):
    monkeypatch.setattr(envs, "VLLM_TPU_MK_ALLOW_FALLBACK", False)
    d, (runner, draft) = _driver(_aot_failure), _runner()
    with pytest.raises(drv.MegakernelSetupError, match="megakernel failed") as info:
        d.try_forward(runner, torch.zeros(4, dtype=torch.int32), torch.zeros(4), None, None)
    assert isinstance(info.value.__cause__, AttributeError)
    # Nothing keeps running on its own: the target kernels and the draft kernels are off.
    assert d.whole_program is None and d.program is None
    assert draft.model.mk_tail is None and draft.model.mk_draft is None


def test_fallback_only_with_explicit_opt_in(monkeypatch, step):
    monkeypatch.setattr(envs, "VLLM_TPU_MK_ALLOW_FALLBACK", True)
    d, (runner, draft) = _driver(_aot_failure), _runner()
    ok, out = d.try_forward(runner, torch.zeros(4, dtype=torch.int32), torch.zeros(4), None, None)
    assert (ok, out) == (False, None)
    assert draft.model.mk_tail is None and draft.model.mk_draft is None


def test_failed_whole_step_setup_raises_by_default(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_TPU_MK_ALLOW_FALLBACK", False)
    d = drv.Qwen38SegmentDriver.__new__(drv.Qwen38SegmentDriver)
    d.__dict__.update(segments=SimpleNamespace(layers=[]), whole_program=object())
    with pytest.raises(drv.MegakernelSetupError, match="whole-step: setup failed"):
        d._attach_whole(SimpleNamespace(model=None))
    assert d.whole_program is None
