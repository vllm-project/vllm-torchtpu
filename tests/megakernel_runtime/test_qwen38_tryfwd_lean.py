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
"""VLLM_TPU_MK_TRYFWD_LEAN: the Qwen3.8 megakernel try_forward with less host
bookkeeping makes the same decisions, calls the same programs with the same
arguments and keeps the same counters as the default path."""

import contextlib
import random
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.megakernel_runtime import qwen38_driver as drv

D = drv.Qwen38SegmentDriver
BUCKETS = (4, 8, 16, 24, 32, 48, 64)


def _driver(
    log,
    *,
    whole_above="stock",
    check_left=0,
    wholecheck_left=0,
    whole_max_rows=32,
    whole_cap=262144,
    whole_dry=False,
):
    d = object.__new__(D)

    def program(tag):
        def call(*, input_ids, positions, **kw):
            log.append((tag, int(input_ids.shape[0]), tuple(sorted(kw))))
            return torch.full((int(input_ids.shape[0]), 2), float(len(log)))

        return call

    d.__dict__.update(
        program=program("segment"),
        whole=True,
        seq=0,
        buckets=BUCKETS,
        whole_program=program("whole"),
        whole_above=whole_above,
        whole_max_rows=whole_max_rows,
        whole_cap=whole_cap,
        whole_dry=whole_dry,
        mk_calls=10,
        segx_left=0,
        h0check_left=0,
        warm=False,
        check_left=check_left,
        shadow=object() if check_left else None,
        check_rows=(),
        check_warm=set(),
        wholecheck_left=wholecheck_left,
        whole_shadow=object() if wholecheck_left else None,
        wholecheck_rows=(8, 16),
        wholecheck_warm=set(),
        whole_calls=0,
        stats={},
        sync=False,
        _whole_checked=True,
        calls_total=0,
    )

    def shadow_step(runner, input_ids, positions, kw, real, n_req):
        d.check_left -= 1
        log.append(("shadow", int(input_ids.shape[0]), real, n_req))
        return torch.zeros(int(input_ids.shape[0]), 2)

    def whole_shadow_step(runner, input_ids, positions, kw, rows, real, n_req):
        d.wholecheck_left -= 1
        log.append(("wholeshadow", rows, real, n_req))
        return torch.zeros(rows, 2)

    d._shadow_step = shadow_step
    d._whole_shadow_step = whole_shadow_step
    d._whole_contexts = lambda runner: contextlib.nullcontext()
    d._serve_contexts = lambda runner: contextlib.nullcontext()
    return d


def _runner():
    return SimpleNamespace(
        _attn_metadata_builder_ctx=SimpleNamespace(num_reqs=1),
        mamba_slot_read_offsets=torch.zeros(4),
        _cached_comb_key=(1, 1, 1),
        query_start_loc_np=np.zeros(8192, np.int32),
        input_batch=SimpleNamespace(num_computed_tokens_cpu=np.zeros(256, np.int32)),
        warmup_done=True,
    )


def _step(rng, runner):
    """One random step: mostly uniform verify steps, some mixed / prefill / odd rows,
    contexts around the 64-token real-traffic edge and the whole-step cap."""
    n = rng.choice((1, 2, 3, 4, 6, 8, 12, 16))
    rows = next((b for b in BUCKETS if b >= 4 * n), 4 * n)
    if rng.random() < 0.05:
        rows = rng.choice((5, 12, 128))
    qsl = np.arange(0, 4 * (n + 1), 4, dtype=np.int32)
    kind = rng.random()
    if kind < 0.05:
        qsl[-1] += 3  # mixed
    elif kind < 0.08:
        qsl = np.arange(0, n + 1, dtype=np.int32)  # t=1
    windowed = n if rng.random() > 0.03 else 0  # prefill
    ctx = rng.choice((10, 60, 70, 5000, 262140, 262141, 300000))
    runner._attn_metadata_builder_ctx.num_reqs = n
    runner._cached_comb_key = (n, windowed, n)
    runner.query_start_loc_np = np.concatenate([qsl, np.full(64, qsl[-1], np.int32)])
    runner.input_batch.num_computed_tokens_cpu = np.full(256, ctx, np.int32)
    runner.warmup_done = rng.random() > 0.05
    return torch.zeros(rows, dtype=torch.int32)


def _run(monkeypatch, lean, seed, **cfg):
    monkeypatch.setattr(envs, "VLLM_TPU_MK_TRYFWD_LEAN", lean, raising=False)
    monkeypatch.setattr(drv.logger, "info", lambda *a, **k: None)
    log: list = []
    d, runner, rng = _driver(log, **cfg), _runner(), random.Random(seed)
    trace = []
    for _ in range(600):
        ids = _step(rng, runner)
        ok, out = D.try_forward(
            d, runner, ids, ids, None, None, model_kwargs={"context_lens": None}
        )
        trace.append((ok, None if out is None else out.tolist()))
    state = dict(
        stats=d.stats,
        counters=getattr(d, "counters", {}),
        whole=d.whole_calls,
        mk=d.mk_calls,
        seq=d.seq,
        check_warm=d.check_warm,
        wholecheck_warm=d.wholecheck_warm,
        check_left=d.check_left,
        wholecheck_left=d.wholecheck_left,
    )
    return trace, log, state


CONFIGS = [
    {},
    {"whole_above": "segment"},
    {"check_left": 7},
    {"wholecheck_left": 9},
    {"whole_dry": True},
    {"whole_max_rows": 16, "whole_cap": 4096},
]


@pytest.mark.parametrize(
    "cfg", CONFIGS, ids=lambda c: ",".join(f"{k}={v}" for k, v in c.items()) or "served"
)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_lean_matches_default(monkeypatch, cfg, seed):
    base = _run(monkeypatch, False, seed, **cfg)
    lean = _run(monkeypatch, True, seed, **cfg)
    assert lean == base
    trace, log, state = base
    assert any(not ok for ok, _ in trace)
    # wholedry serves every step stock by design.
    assert any(ok for ok, _ in trace) != bool(cfg.get("whole_dry"))


def test_note_running_total_reseeds_after_flag_flip(monkeypatch):
    """Turning the flag off and on again mid-run must not keep a stale total."""
    logged = []
    monkeypatch.setattr(drv, "STEP_MIX_EVERY", 5)
    monkeypatch.setattr(drv, "STEP_MIX_INTERVAL_S", 0)
    monkeypatch.setattr(drv.logger, "info", lambda *a, **k: logged.append(dict(a[1])))
    d = SimpleNamespace(stats={})
    for i in range(40):
        monkeypatch.setattr(
            envs, "VLLM_TPU_MK_TRYFWD_LEAN", (i // 7) % 2 == 1, raising=False
        )
        D._note(d, f"k{i % 3}")
    assert [sum(s.values()) for s in logged] == [5, 10, 15, 20, 25, 30, 35, 40]


def test_lean_is_cheaper(monkeypatch):
    """Host cost of a served whole-step try_forward (programs stubbed); report only."""
    monkeypatch.setattr(drv.logger, "info", lambda *a, **k: None)
    res = {}
    for lean in (False, True):
        monkeypatch.setattr(envs, "VLLM_TPU_MK_TRYFWD_LEAN", lean, raising=False)
        d, runner = _driver([]), _runner()
        d.whole_program = lambda **kw: None
        runner._attn_metadata_builder_ctx.num_reqs = 4
        runner._cached_comb_key = (4, 4, 4)
        runner.query_start_loc_np = np.concatenate(
            [np.arange(0, 20, 4, dtype=np.int32), np.full(8192, 16, np.int32)]
        )
        runner.input_batch.num_computed_tokens_cpu = np.full(256, 5000, np.int32)
        ids = torch.zeros(16, dtype=torch.int32)
        best = 1e9
        for _ in range(3):
            t = time.perf_counter()
            for _ in range(5000):
                D.try_forward(d, runner, ids, ids, None, None, model_kwargs={})
            best = min(best, (time.perf_counter() - t) / 5000 * 1e6)
        res[lean] = best
        assert d.stats.get("whole_b16", 0) > 0
    print(f"try_forward host us/call: default {res[False]:.1f}, lean {res[True]:.1f}")
