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
"""q38 DEC-FB-3 VLLM_TPU_SPEC_VERIFY_PRECOMPILE_SERVED_ONLY (CPU).

The real rejection-sampler precompile runs on a fake runner whose programs
record what they are warmed with. With the knob on and a VPS head, the sweep
warms only the VPS verify, at the fallback width a chunk whose requests all
have an in-range top_k dispatches (computed by the real `_vps_fallback_rows`),
at every bucket pair the full sweep visits; the greedy / separate sampling
programs, the top-token verify, the top-k-candidates variant and the fused
verify are not warmed. Without a VPS head it warms only the fused verify. With
neither it changes nothing, and with the knob off the sweep is unchanged.
"""

import contextlib
from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.runner import tpu_runner as tr
from vllm_torchtpu.runner import vps_wiring

PADS = [8, 16, 32, 48, 64, 128, 256]
REQS = [1, 2, 4, 8, 16, 32, 64]
K = 3
FB_MAX = 8
KNOB = "VLLM_TPU_SPEC_VERIFY_PRECOMPILE_SERVED_ONLY"


class _Sampler:
    synthetic_mode = False
    synthetic_conditional_rates = None

    def __init__(self, log):
        self.log = log

    def __call__(self, **kw):
        t, n = int(kw["draft_token_ids"].shape[0]), int(kw["num_draft_tokens"].shape[0])
        if not kw.get("do_sampling"):
            kind = "rs_greedy"
        elif kw.get("topk_candidates"):
            kind = "rs_topk"
        else:
            kind = "rs_sample"
        self.log.append((kind, t, n, None))
        return torch.zeros(1)


def _runner(monkeypatch, *, vps, fb_path=True, tiegap=0):
    log = []
    r = SimpleNamespace(
        _is_async_drafter=True,
        speculative_config=SimpleNamespace(num_speculative_tokens=K),
        rejection_sampler=_Sampler(log),
        num_tokens_paddings=list(PADS),
        num_reqs_paddings=list(REQS),
        max_num_reqs=REQS[-1],
        vocab_size=16,
        _hidden_states_dtype=torch.float32,
        device=torch.device("cpu"),
        _greedy_head_owner=object(),  # the top-token verify is reachable when the knob is off
        _precompile_timed=lambda name: contextlib.nullcontext(),
        _vps_cand_draws=False,
        _vps_fb_path=fb_path,
        _vps_fb_max=FB_MAX,
        _vps_tiegap=tiegap,
    )
    if vps:
        r._vps_head = SimpleNamespace(shard_width=8, tp_size=2, full_vocab=16, shard_offset=0)
    r._topk_candidate_blocks = lambda: 2  # the top-k-candidates variant is reachable when off
    r._spec_verify_bucket_reachable = lambda t, n: tr.TPUModelRunner._spec_verify_bucket_reachable(r, t, n)

    def greedy_prelude(lg, b, t):
        log.append(("greedy_prelude", int(lg.shape[0]), int(b.shape[0]), None))
        return torch.zeros(b.shape[0], dtype=torch.int32), lg

    def ng_prelude(lg, b, t):
        log.append(("ng_prelude", int(lg.shape[0]), int(b.shape[0]), None))
        return None, lg

    r.spec_bonus_and_target_logits = greedy_prelude
    r.spec_gather_bonus_and_target_logits = ng_prelude
    r._precompile_top_token_verify = lambda t, *a: log.append(("top_token", int(t), int(a[5].shape[0]), None))
    r._precompile_verify_fused = lambda lg, b, *a: log.append(("fused", int(lg.shape[0]), int(b.shape[0]), None))
    r._precompile_vps_verify = lambda *a, **k: tr.TPUModelRunner._precompile_vps_verify(r, *a, **k)

    def vps_plain(local, off, *args):
        log.append(("vps", int(local.shape[0]), int(args[0].shape[0]), None))
        return torch.zeros(1)

    def vps_fb(local, off, *rest):
        log.append(("vps_fb", int(local.shape[0]), int(rest[0].shape[0]), int(rest[-3])))
        return torch.zeros(1), torch.zeros(1)

    monkeypatch.setattr(tr, "_vps_verify_compiled", vps_plain)
    monkeypatch.setattr(tr, "_vps_verify_fb_compiled", vps_fb)
    monkeypatch.setattr(tr, "synchronize_tensors", lambda *a, **k: None)
    return r, log


def _sweep(monkeypatch, *, knob, vps, fused, fb_path=True, tiegap=0, cap=None):
    for name, val in ((KNOB, knob), ("VLLM_TPU_VERIFY_FUSED", fused),
                      ("VLLM_TPU_SPEC_VERIFY_PRECOMPILE_MAX_SEQS", cap)):
        if val is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, str(val))
    r, log = _runner(monkeypatch, vps=vps, fb_path=fb_path, tiegap=tiegap)
    tr.TPUModelRunner._precompile_rejection_sampler(r)
    return log


def _kinds(log):
    return {e[0] for e in log}


def _pairs(log, kind):
    return [(t, n) for k, t, n, _ in log if k == kind]


def _served_width(tiegap):
    return vps_wiring.fallback_slots(0, tiegap, max(FB_MAX, tiegap))


@pytest.mark.parametrize("tiegap", [0, 2])
def test_served_only_warms_only_the_vps_served_width(monkeypatch, tiegap):
    off = _sweep(monkeypatch, knob=None, vps=True, fused=1, tiegap=tiegap)
    on = _sweep(monkeypatch, knob=1, vps=True, fused=1, tiegap=tiegap)
    # with the knob off every region is warmed (so the knob has something to skip)
    assert {"rs_greedy", "rs_sample", "rs_topk", "top_token", "fused", "vps_fb"} <= _kinds(off)
    assert len({e[3] for e in off if e[0] == "vps_fb"}) >= 2
    # with it on: only the VPS verify, only at the served width, at every bucket pair the sweep visits
    assert _kinds(on) == {"vps_fb"}
    assert {e[3] for e in on} == {_served_width(tiegap)}
    served = _served_width(tiegap)
    assert [(t, n) for k, t, n, rows in off if k == "vps_fb" and rows == served] == _pairs(on, "vps_fb")
    assert len(_pairs(on, "vps_fb")) == len(_pairs(off, "rs_greedy")) > 10


def _batch_runner(top_k, temperature, tiegap):
    n = len(top_k)
    return SimpleNamespace(
        input_batch=SimpleNamespace(
            top_k_cpu_tensor=torch.tensor(top_k, dtype=torch.int32),
            temperature_cpu_tensor=torch.tensor(temperature, dtype=torch.float32),
        ),
        vocab_size=151936,
        speculative_config=SimpleNamespace(num_speculative_tokens=K),
        _vps_tiegap=tiegap,
        _vps_fb_max=FB_MAX,
    ), n


@pytest.mark.parametrize("tiegap", [0, 2])
def test_served_width_is_what_a_served_chunk_dispatches(monkeypatch, tiegap):
    """The width warmed is the one the real chunk dispatch picks for served
    traffic (top_k 20 <= VP_KC, temperature 0.6); a chunk with an unset top_k
    needs fallback rows, gets another width, and compiles lazily."""
    on = _sweep(monkeypatch, knob=1, vps=True, fused=1, tiegap=tiegap)
    (warmed,) = {e[3] for e in on}
    r, n = _batch_runner([20] * 5, [0.6] * 5, tiegap)
    assert tr.TPUModelRunner._vps_fallback_rows(r, 0, n) == warmed
    r2, n2 = _batch_runner([20, 20, -1], [0.6, 0.6, 0.6], tiegap)
    assert tr.TPUModelRunner._vps_fallback_rows(r2, 0, n2) != warmed


def test_served_only_without_vps_warms_only_the_fused_verify(monkeypatch):
    off = _sweep(monkeypatch, knob=None, vps=False, fused=1)
    on = _sweep(monkeypatch, knob=1, vps=False, fused=1)
    assert _kinds(on) == {"fused"}
    assert _pairs(on, "fused") == _pairs(off, "fused")
    assert len(_pairs(on, "fused")) > 10


def test_served_only_without_fused_keeps_the_plain_vps_program(monkeypatch):
    """A VPS head is the served path even with the fused verify off; without
    the fallback path the VPS program has one width."""
    off = _sweep(monkeypatch, knob=None, vps=True, fused=None, fb_path=False)
    on = _sweep(monkeypatch, knob=1, vps=True, fused=None, fb_path=False)
    assert _kinds(on) == {"vps"}
    assert _pairs(on, "vps") == _pairs(off, "vps")
    assert "rs_greedy" in _kinds(off)


def test_served_only_is_a_no_op_without_vps_or_fused(monkeypatch):
    off = _sweep(monkeypatch, knob=None, vps=False, fused=None)
    on = _sweep(monkeypatch, knob=1, vps=False, fused=None)
    assert on == off
    assert "rs_sample" in _kinds(on)


def test_served_only_composes_with_the_cap(monkeypatch):
    off_capped = _sweep(monkeypatch, knob=None, vps=True, fused=1, cap=24)
    on_capped = _sweep(monkeypatch, knob=1, vps=True, fused=1, cap=24)
    on_full = _sweep(monkeypatch, knob=1, vps=True, fused=1)
    served = _served_width(0)
    assert _pairs(on_capped, "vps_fb") == [(t, n) for k, t, n, rows in off_capped if k == "vps_fb" and rows == served]
    assert len(_pairs(on_capped, "vps_fb")) < len(_pairs(on_full, "vps_fb"))


@pytest.mark.parametrize("vps", [True, False])
def test_knob_off_and_zero_keep_the_full_sweep(monkeypatch, vps):
    off = _sweep(monkeypatch, knob=None, vps=vps, fused=1)
    assert _sweep(monkeypatch, knob=0, vps=vps, fused=1) == off
    assert {"rs_greedy", "rs_sample", "rs_topk", "top_token", "fused"} <= _kinds(off)
