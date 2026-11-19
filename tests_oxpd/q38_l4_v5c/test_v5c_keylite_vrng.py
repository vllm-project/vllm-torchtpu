# SPDX-License-Identifier: Apache-2.0
"""q38 L4 v5 composition (q38-l4-v5c): KEYLITE x VRNG-VPS on VFY-ONE steps (composition fix 3: the VFY warm check
takes VRNG's meta-draw key call `kcall`). CDK CPU pod only. V5COMP-BUILD (helper of L4-E).

Driven with VRNG x VFY's harness (tests/q38_l4_samp2v/test_samp2_vfy.py: the runner's real methods on a fake runner,
compiled programs as eager bodies, VRNG on the runner's agreement clock) after the REAL VFY warm-up
(`_vfy_precompile`, no AlwaysWarm), with VLLM_TPU_VPS_INGRAPH_RNG_AB=alt:2 (window k = n // 2: VRNG at engine steps
1, 4-5, 8-9, ..., the base draws at 2-3, 6-7, ...).

W1  300 engine steps at one warmed steady shape, KEYLITE off and on: every VRNG step runs the composed VRNG VFY
    program, every off-window step the base VFY program; no step is cold on either controller (no program runs at a
    key the warm-up did not record: no compile after warm-up); with KEYLITE the lite memo is shared by both paths
    (one entry: their full keys are equal) and the full-key audit (every 256 calls) never turns KEYLITE off.
Run: python -m pytest -q -p no:cacheprovider -p no:logging -s tests/q38_l4_v5c/test_v5c_keylite_vrng.py
"""

import os
import sys
from types import SimpleNamespace as NS

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "q38_l4_samp2v"))
import test_samp2_vfy as sv  # noqa: E402
from test_samp2_vfy import rt  # noqa: E402,F401
import vllm_torchtpu.runner.tpu_runner as tr  # noqa: E402
from vllm_torchtpu.runner import vfy_one, vps_wiring  # noqa: E402

STEPS = 300
COUNTS: dict = {}


def _step_inputs(r, R, seed):
    """sv.test_served_step_after_warmup's steady step at R live requests (a warmed shape), fresh per seed."""
    B = tr._get_padded_num_reqs_with_upper_limit(R, r.max_num_reqs)
    T = tr._get_padded_token_len(r.num_tokens_paddings, R * (sv.K + 1))
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(T, sv.HID, generator=g).bfloat16()
    idx = torch.randperm(T, generator=g).to(torch.int32)
    md = NS(bonus_logits_indices=(torch.arange(B) * (sv.K + 1) + sv.K).clamp(max=T - 1).to(torch.int32),
            target_logits_indices=torch.arange(T, dtype=torch.int32),
            draft_token_ids=torch.randint(0, sv.V, (T,), generator=g, dtype=torch.int32),
            draft_lengths=torch.tensor([sv.K] * R + [0] * (B - R), dtype=torch.int32), draft_lengths_cpu=[sv.K] * R,
            segment_ids=(torch.arange(T) // sv.K).clamp(max=R).to(torch.int64),
            group_indices=(torch.arange(T) % sv.K).to(torch.int32))
    ng, widths = r._read_offset_fold_geometry()
    groups = [torch.randint(1, 4096, (widths[0],), generator=g, dtype=torch.int32) for _ in range(ng)]
    return hidden, idx, md, groups


@pytest.mark.parametrize("keylite", [False, True])
def test_w1_vrng_window_switching_with_keylite_never_runs_cold(rt, monkeypatch, keylite):
    monkeypatch.setattr(tr.envs, "VLLM_TPU_SPEC_VERIFY_PRECOMPILE_SERVED_ONLY", False, raising=False)
    ctl = sv.mkctl(ab="alt:2")
    r = sv.WarmRunner(ctl)
    r._vfy.keylite = keylite
    r._vfy_precompile()
    assert len(r._vfy.warm) > 0 and len(ctl.warm) > 0
    seen = sv.count_programs(monkeypatch)
    R = 5
    F = vps_wiring.fallback_variants(r._vps_fb_max, r._vps_tiegap)[0]
    gen = torch.Generator().manual_seed(8)
    want_vrng = want_base = 0
    for i in range(STEPS):
        n = i + 1
        on = (n // 2) % 2 == 0
        hidden, idx, md, groups = _step_inputs(r, R, 7000 + i)
        r._fused_read_offsets = (groups, R)
        r._fused_read_offsets_applied = False
        r._vfy.step_mode = "vfy"
        sv.tick(r)
        r._sample_spec_verify_vps(vfy_one.VfyChunk(hidden, idx), md, 0, R, gen, F)
        r.__dict__.pop("_vfy_seed", None)
        want_vrng += on
        want_base += not on
        assert r._vfy.stats["cold"] == 0 and ctl.stats["cold"] == 0, (n, dict(r._vfy.stats), dict(ctl.stats))
        assert seen["vfy_rng"] == want_vrng and seen["vfy"] == want_base, (n, on, seen)
    assert seen["chain"] == 0 and seen["chain_rng"] == 0, seen
    assert ctl.vrng_calls == want_vrng > 0 and want_base > 0
    if keylite:
        s = r._vfy.stats
        assert not r._vfy.keylite_off and s["keylite_audit"] >= 1, dict(s)
        assert s["keylite_miss"] == 1 and len(r._vfy.lite) == 1, (dict(s), len(r._vfy.lite))  # one shared entry
        assert s["keylite_hit"] + s["keylite_miss"] + s["keylite_audit"] == STEPS, dict(s)
    COUNTS[f"w1_keylite{int(keylite)}"] = STEPS
    print(f"\nV5C_KEYLITE_VRNG W1 keylite={int(keylite)} steps={STEPS} vrng_vfy_programs={seen['vfy_rng']} "
          f"base_vfy_programs={seen['vfy']} cold=0 (VFY) / 0 (VRNG); chain programs 0; "
          f"keylite hit={r._vfy.stats['keylite_hit']} miss={r._vfy.stats['keylite_miss']} "
          f"audit={r._vfy.stats['keylite_audit']} off={r._vfy.keylite_off}")


def test_zz_counts():
    print("\nV5C_KEYLITE_VRNG_COUNTS " + " ".join(f"{k}={v}" for k, v in sorted(COUNTS.items())))
