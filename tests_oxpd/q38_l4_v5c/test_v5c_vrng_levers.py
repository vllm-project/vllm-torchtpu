# SPDX-License-Identifier: Apache-2.0
"""q38 L4 v5 composition (q38-l4-v5c with AGREE and VRNG-VPS): VRNG-VPS's VFY composition with VPSHOST's sampsnap and
statsasync on (786f7bac2's branches in the `_vfy*` copies). CDK CPU pod only. V5COMP-BUILD (helper of L4-E).

Driven with VRNG x VFY's own harness (tests/q38_l4_samp2v/test_samp2_vfy.py: the runner's real methods on a fake
runner, compiled programs as their eager bodies, VRNG registered on the runner's agreement clock); the levers are
armed through the real core/l6_ab.py (uniform "1"), the snapshot / async copies are the tree's runner/l4_vpshost.py
(device event faked).

V1  8 engine steps at 8 / 32 rows, fallback rows 0 / 8: the VFY step with VRNG (one composed in-graph-draw program) and
    the stock D83 chain with VRNG (its VRNG program) give equal tokens, generator state (untouched by both), fallback
    stats, read-offset buffer and counters; both take the snapshot for the parameter and num_active uploads on every
    step; statsasync issues and drains the same copies and logs the same VPS_FALLBACK lines on the same steps.
Run: python -m pytest -q -p no:cacheprovider -p no:logging -s tests/q38_l4_v5c/test_v5c_vrng_levers.py
"""

import collections
import os
import sys
from types import SimpleNamespace as NS

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "q38_l4_samp2v"))
import test_samp2_vfy as sv  # noqa: E402
from test_samp2_vfy import rt  # noqa: E402,F401  (the harness fixture: eager programs, levers unset, log capture)
import vllm_torchtpu.runner.tpu_runner as tr  # noqa: E402
from vllm_torchtpu.core import l6_ab  # noqa: E402
from vllm_torchtpu.runner import l4_vpshost  # noqa: E402

LEVER_ENV = {"VLLM_TPU_L4_SAMP_SNAP": "1", "VLLM_TPU_L4_VPS_STATS_ASYNC": "1", "VLLM_TPU_VPS_STATS_EVERY": "2"}
COUNTS: dict = {}


class CountingSnap(l4_vpshost.SampSnap):
    def __init__(self):
        super().__init__()
        self.calls = collections.Counter()

    def params(self, *a, **k):
        self.calls["params"] += 1
        return super().params(*a, **k)

    def num_active(self, *a, **k):
        self.calls["num_active"] += 1
        return super().num_active(*a, **k)


class CountingAsync(l4_vpshost.AsyncCounts):
    def __init__(self):
        super().__init__()
        self.issued, self.drained = [], []

    def issue(self, tag, dev):
        self.issued.append(tag)
        super().issue(tag, dev)

    def ready(self, wait=False):
        out = super().ready(wait)
        self.drained += [t for t, _ in out]
        return out


class Batch:
    def __init__(self, B):
        self.req_ids = [f"req{i}" for i in range(B)]
        self.temperature_cpu_tensor = torch.full((B,), 0.6)
        self.top_k_cpu_tensor = torch.full((B,), 20, dtype=torch.int32)
        self.top_p_cpu_tensor = torch.full((B,), 0.95)

    def add_request(self, *a, **k):
        return None


class Lines:
    def __init__(self):
        self.tag, self.rec = None, collections.defaultdict(list)

    def info(self, fmt, *a):
        msg = fmt % a if a else fmt
        if msg.startswith("VPS_FALLBACK"):
            self.rec[self.tag].append(msg)

    def warning(self, *a, **k):
        return None

    warning_once = info_once = error = debug = warning


def _levers_runner(B, vfy):
    ctl = sv.mkctl()
    ctl.warm = sv.AlwaysWarm()
    r = sv.Runner(B, mode=1 if vfy else 0, vrng=ctl)
    if vfy:
        sv.warm_all(r)
    r._samp_snap = CountingSnap()
    r._vps_fb_async = CountingAsync()
    r.input_batch = Batch(B)
    r.input_batch.generators = {}
    return r, ctl


@pytest.mark.parametrize("F", [0, 8])
@pytest.mark.parametrize("R", [8, 32])
def test_v1_vrng_vfy_with_levers_equals_the_stock_vrng_chain(rt, monkeypatch, R, F):
    B = R
    lines = Lines()
    monkeypatch.setattr(tr, "logger", lines)
    monkeypatch.setattr(l4_vpshost, "_event", lambda: NS(query=lambda: True, synchronize=lambda: None))
    saved = {k: os.environ.get(k) for k in LEVER_ENV}
    os.environ.update(LEVER_ENV)
    l6_ab._resolve()
    l6_ab.begin_step()
    try:
        assert l6_ab.on("sampsnap") and l6_ab.on("statsasync")
        base, cb = _levers_runner(B, vfy=False)
        vfy, cv = _levers_runner(B, vfy=True)
        for i in range(8):
            hidden, idx, md = sv.problem(5000 + 10 * R + F + i, B, R)
            groups = sv.slot_groups(300 + i, 3, 64, R)
            gb, gv = torch.Generator().manual_seed(9), torch.Generator().manual_seed(9)
            g0 = gb.get_state().clone()
            lines.tag = "base"
            nb = len(lines.rec["base"])
            base._fused_read_offsets, base._fused_read_offsets_applied = (groups, R), False
            sv.tick(base)
            ob = base._sample_spec_verify_vps(base.compute_selected_local_logits(hidden, idx), md, 0, R, gb, F)
            lb = lines.rec["base"][nb:]
            lines.tag = "vfy"
            nv = len(lines.rec["vfy"])
            ov = sv.vfy_step(vfy, hidden, idx, md, groups, R, F, gv)
            lv = lines.rec["vfy"][nv:]
            vfy.__dict__.pop("_vfy_seed", None)
            assert torch.equal(ob, ov), f"tokens differ at step {i}"
            assert torch.equal(gb.get_state(), g0) and torch.equal(gv.get_state(), g0), i  # in-graph draws only
            assert torch.equal(base._vps_fb_stats, vfy._vps_fb_stats), (i, base._vps_fb_stats, vfy._vps_fb_stats)
            assert torch.equal(base.mamba_slot_read_offsets, vfy.mamba_slot_read_offsets), i
            assert cb.counter_word() == cv.counter_word() and lb == lv, (i, lb, lv)
        assert cb.vrng_calls == cv.vrng_calls == 8 and vfy._vfy.stats["vfy_vrng"] == 8, (
            cb.vrng_calls, cv.vrng_calls, dict(vfy._vfy.stats))
        assert base._samp_snap.calls == vfy._samp_snap.calls and base._samp_snap.calls["num_active"] == 8, (
            dict(base._samp_snap.calls), dict(vfy._samp_snap.calls))
        assert base._samp_snap.calls["params"] == 8
        assert base._vps_fb_async.issued == vfy._vps_fb_async.issued == [2, 4, 6, 8]
        assert base._vps_fb_async.drained == vfy._vps_fb_async.drained == [2, 4, 6]
        assert lines.rec["base"] == lines.rec["vfy"] and len(lines.rec["vfy"]) == 3
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        l6_ab._resolve()
    COUNTS[f"v1_R{R}_F{F}"] = 8
    print(f"\nV5C_VRNGLEV V1 R={R} F={F} steps=8 VFY+VRNG == stock+VRNG with sampsnap+statsasync: tokens / stats / "
          f"offsets / counters equal, generator untouched; snap {dict(vfy._samp_snap.calls)}; async issued "
          f"{vfy._vps_fb_async.issued} drained {vfy._vps_fb_async.drained}; vrng calls {cv.vrng_calls}")


def test_zz_counts():
    print("\nV5C_VRNGLEV_COUNTS " + " ".join(f"{k}={v}" for k, v in sorted(COUNTS.items())))
