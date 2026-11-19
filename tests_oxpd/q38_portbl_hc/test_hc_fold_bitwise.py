# SPDX-License-Identifier: Apache-2.0
"""q38 PORT-BL HC: BW and ENG legs (CPU, Pallas TPU interpret mode; 0 chips).

TPU_QWEN4_HC_DECODE_FOLD and TPU_QWEN4_HC_DECODE_TILE_ROWS (ported from Benliang Wang <benliangw@google.com>, commits
d523c2ce0 / 28419c4a5 on benliangw/q38fh-integ-pin-5b44702aea), flag on against flag off through the REAL stage loop
(Qwen4ExpBackbone._forward_impl), the REAL decoder-layer forward and REAL GatedResidual modules at the served HC
geometry (hc 4 x 2560, low rank 320), 4 layers with PLE at layer 1 plus the final mixer, every fused op built by
vllm_adapter's own builders and run as the real kernel in interpret mode (hc_fold_stage.py).

Arms (env at build and at run):
  served gates  TPU_QWEN4_HC_DECODE_FUSED=1, UP_T=1, VLLM_TPU_HC_DECODE_FUSED_MAX_ROWS=128, VLLM_TPU_HC_XLA_MAX_ROWS=256
                (the v8 record env); OFF = that, ON = that + TPU_QWEN4_HC_DECODE_FOLD=1.
                fused rows 64 / 80 / 96 / 112 / 128: the fold engages (3 hand-offs: layer 1 -> 2, 2 -> 3, 3 -> mixer).
                XLA rows 160 / 192 / 224 / 256: an exact no-op (no fused call, no deferral; the traced graph is
                compared op for op in test_hc_fold_units.py).
  gates unset   FUSED=1, UP_T=1, no HCGATE / HCX (160-256 take the fused path): NEITHER vs FOLD + TILE_ROWS=512, and
                NEITHER vs TILE_ROWS=512 alone, at 128 / 160 / 192 / 256.

Every arm compares, with torch.equal: the final mixer's output, the pre-mixer streams (the MTP draft's aux input) and
every layer's attention and MoE inputs (both HC mixes' outputs). ENG (engagement): the kernel calls each arm made
(plain mixes, combine prologues with / without the injection gate), the layers' defer flags, and the kernel tile.

FMA note (why the bitwise legs pin the XLA:CPU ISA): unfolded, a layer's mlp combine runs as torch ops
(bf16(f32(residual) + f32(block) * injection)); folded, the same expression runs inside the interpret-mode kernel,
which XLA:CPU compiles with FMA contraction allowed (fma(block, injection, residual) rounds once). The proof job runs
the bitwise legs with XLA_FLAGS=--xla_cpu_max_isa=<an ISA without FMA> and the FMA oracle test shows, at the default
ISA, that every differing element is exactly the contraction's rounding. Q38_PBL_HC_BW=report prints instead of
asserting bitwise equality (engagement is asserted in both modes).
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hc_fold_stage as S  # noqa: E402

STRICT = os.environ.get("Q38_PBL_HC_BW", "strict") == "strict"
DIMS = S.DIMS[os.environ.get("Q38_PBL_HC_DIMS", "real")]
LAYERS = int(os.environ.get("Q38_PBL_HC_LAYERS", "4"))
ROWS_FUSED = [int(r) for r in os.environ.get("Q38_PBL_HC_ROWS_FUSED", "64,80,96,112,128").split(",")]
ROWS_XLA = [int(r) for r in os.environ.get("Q38_PBL_HC_ROWS_XLA", "160,192,224,256").split(",")]
ROWS_TILE = [int(r) for r in os.environ.get("Q38_PBL_HC_ROWS_TILE", "128,160,192,256").split(",")]

SERVED = dict(S.SERVED)
SERVED_FOLD = dict(S.SERVED, **{S.FOLD: "1"})
UNGATED = {S.FUSED: "1", S.UPT: "1"}
UNGATED_FOLD_TILE = dict(UNGATED, **{S.FOLD: "1", S.TILE: "512"})
UNGATED_TILE = dict(UNGATED, **{S.TILE: "512"})
ARMS = {"served_off": SERVED, "served_on": SERVED_FOLD, "ungated_off": UNGATED,
        "ungated_fold_tile": UNGATED_FOLD_TILE, "ungated_tile": UNGATED_TILE}
_STAGES: dict = {}


def stage(arm):
    if arm not in _STAGES:
        _STAGES[arm] = S.build_stage(ARMS[arm], DIMS, layers=LAYERS)
    return _STAGES[arm]


def _line(tag, rows, ok, det, on, off, extra=""):
    print(f"\nPBL_HC {tag} rows {rows}: bitwise {'yes' if ok else 'NO'} mixed {det['mixed']} hidden {det['hidden']} "
          f"blocks {det['blocks'][0]}/{det['blocks'][1]} unequal {det['blocks'][2]} | calls on {S.counts(on['calls'])} "
          f"off {S.counts(off['calls'])} | defers on {sum(d['defer'] for d in on['defers'])} "
          f"off {sum(d['defer'] for d in off['defers'])}{extra}")


def test_same_weights_in_every_arm():
    ref = stage("served_off")[0].state_dict()
    for arm in ARMS:
        sd = stage(arm)[0].state_dict()
        assert sd.keys() == ref.keys(), arm
        assert all(torch.equal(sd[k], ref[k]) for k in ref), arm
    names = {arm: stage(arm)[1] for arm in ARMS}
    print("\nPBL_HC BUILT " + " | ".join(f"{a}: {sorted(set(n for n in v if 'hc_decode' in n))}" for a, v in names.items()))
    # The fold builds the final mixer's no-injection combine-mix op, UP_T variant; nothing else changes.
    for arm, env in ARMS.items():
        noinj = [n for n in names[arm] if "hc_decode_combine" in n and "_noinj" in n]
        if env.get(S.FOLD):
            assert noinj and all("_upt" in n for n in noinj), (arm, noinj)
        else:
            assert not noinj, (arm, noinj)
        assert all("_upt" in n for n in names[arm] if "hc_decode" in n), (arm, names[arm])


@pytest.mark.parametrize("rows", ROWS_FUSED)
def test_bw_served_gates_fused_rows(rows):
    off = S.run(stage("served_off")[0], SERVED, rows, DIMS)
    on = S.run(stage("served_on")[0], SERVED_FOLD, rows, DIMS)
    ok, det = S.compare(on, off)
    want_on, handoffs = S.expected(LAYERS, S.PLE_AT, True)
    want_off, _ = S.expected(LAYERS, S.PLE_AT, False)
    _line("BW served-gates fused", rows, ok, det, on, off, f" | hand-offs on {handoffs}")
    # ENG: the fold ran (and only there); the kernel read up.T (UP_T) in every call; one tile (<= 128 rows).
    assert S.counts(on["calls"]) == want_on, S.counts(on["calls"])
    assert S.counts(off["calls"]) == want_off, S.counts(off["calls"])
    assert sum(d["defer"] for d in on["defers"]) == handoffs and not any(d["defer"] for d in off["defers"])
    # decode_fold on layer i = the hand-off INTO layer i came from the fold (every layer after the first, except PLE).
    assert [d["decode_fold"] for d in on["defers"]] == [i >= 1 and i != S.PLE_AT for i in range(LAYERS)]
    assert all(c["up_t"] and c["tile"] == rows and c["max_tile"] == 0 for c in on["calls"] + off["calls"])
    if STRICT:
        assert ok, det


@pytest.mark.parametrize("rows", ROWS_XLA)
def test_bw_served_gates_xla_rows_are_a_noop(rows):
    off = S.run(stage("served_off")[0], SERVED, rows, DIMS)
    on = S.run(stage("served_on")[0], SERVED_FOLD, rows, DIMS)
    ok, det = S.compare(on, off)
    _line("BW served-gates xla", rows, ok, det, on, off, " | hand-offs on 0 (expected)")
    assert on["calls"] == [] and off["calls"] == []          # HCGATE / HCX: no fused HC at all
    assert not any(d["defer"] or d["pending"] or d["decode_fold"] for d in on["defers"] + off["defers"])
    assert ok, det   # the XLA path on both sides: torch ops only, so bitwise at any ISA


@pytest.mark.parametrize("rows", ROWS_TILE)
def test_bw_ungated_fold_plus_tile(rows):
    off = S.run(stage("ungated_off")[0], UNGATED, rows, DIMS)
    on = S.run(stage("ungated_fold_tile")[0], UNGATED_FOLD_TILE, rows, DIMS)
    ok, det = S.compare(on, off)
    want_on, handoffs = S.expected(LAYERS, S.PLE_AT, True)
    tiles_on = sorted({c["tile"] for c in on["calls"]})
    tiles_off = sorted({c["tile"] for c in off["calls"]})
    _line("BW ungated fold+tile512", rows, ok, det, on, off,
          f" | hand-offs on {handoffs} | tiles on {tiles_on} off {tiles_off}")
    assert S.counts(on["calls"]) == want_on, S.counts(on["calls"])
    assert S.counts(off["calls"]) == S.expected(LAYERS, S.PLE_AT, False)[0]
    assert tiles_on == [S.hc_decode.decode_kernel_tile(rows, 512)] and all(c["max_tile"] == 512 for c in on["calls"])
    assert tiles_off == [S.hc_decode.decode_token_tile(rows)] and all(c["max_tile"] == 0 for c in off["calls"])
    assert all(c["up_t"] for c in on["calls"] + off["calls"])
    if STRICT:
        assert ok, det


@pytest.mark.parametrize("rows", ROWS_TILE)
def test_bw_ungated_tile_alone(rows):
    off = S.run(stage("ungated_off")[0], UNGATED, rows, DIMS)
    on = S.run(stage("ungated_tile")[0], UNGATED_TILE, rows, DIMS)
    ok, det = S.compare(on, off)
    tiles_on = sorted({c["tile"] for c in on["calls"]})
    _line("BW ungated tile512-only", rows, ok, det, on, off, f" | tiles on {tiles_on}")
    assert S.counts(on["calls"]) == S.counts(off["calls"]) == S.expected(LAYERS, S.PLE_AT, False)[0]
    assert tiles_on == [S.hc_decode.decode_kernel_tile(rows, 512)]
    # The tile alone changes no arithmetic boundary (rows are independent): bitwise at any ISA.
    assert ok, det


def _fma_oracle(prev, block, inj, h):
    """Per element: h_nofma = bf16(f32(f32(block * inj) + prev)), h_fma = bf16(f32(prev + block * inj exact))."""
    p = prev.float().numpy().astype(np.float32)
    b = block.float().numpy().astype(np.float32)
    g = inj.numpy().astype(np.float32)
    hidden = b.shape[1]
    nofma, fma = [], []
    for k in range(S.HC):
        pk = p[:, k * hidden:(k + 1) * hidden]
        prod32 = (b * g[:, k:k + 1]).astype(np.float32)
        nofma.append((prod32 + pk).astype(np.float32))
        exact = pk.astype(np.float64) + b.astype(np.float64) * g[:, k:k + 1].astype(np.float64)
        fma.append(exact.astype(np.float32))
    to_bf16 = lambda a: torch.from_numpy(np.concatenate(a, axis=1)).to(torch.bfloat16)  # noqa: E731
    return to_bf16(nofma), to_bf16(fma)


@pytest.mark.parametrize("rows", [96, 128])
def test_fma_oracle_at_one_handoff(rows):
    """One mlp -> next attn hand-off at served dims: the XLA-path combine (GatedResidual.combine, torch ops) against the
    kernel prologue's combined streams (the fold's), with a per-element FMA oracle."""
    st, _ = stage("served_on")
    mlp, attn = st.layers[2].mlp_hyper_connection, st.layers[3].attn_hyper_connection
    g = torch.Generator().manual_seed(5000 + rows)
    h = DIMS["hidden"]
    resid = torch.randn(rows, S.HC * h, generator=g).bfloat16()
    moe = torch.randn(rows, h, generator=g).bfloat16()
    gate = (2.0 * torch.sigmoid(torch.randn(rows, S.HC, generator=g) / S.HC)).float()
    with S.env_state(**SERVED_FOLD), torch.no_grad():
        h_xla = mlp.combine(moe, (resid, gate))
        mixed_x, (_, inj_x) = attn.mix(h_xla)
        del S.CALLS[:]
        mixed_k, (h_k, inj_k) = mlp.combine_mix(moe, (resid, gate), attn, prefill_fused=False)
        prologue = S.counts(S.CALLS)
    h_nofma, h_fma = _fma_oracle(resid, moe, gate, h_k)
    diff = ~(h_k.view(torch.int16) == h_xla.view(torch.int16))
    xla_is_nofma = torch.equal(h_xla, h_nofma)
    k_is_fma_where_diff = bool(torch.equal(h_k[diff], h_fma[diff])) if diff.any() else True
    fma_sites = int((h_fma.view(torch.int16) != h_nofma.view(torch.int16)).sum())
    print(f"\nPBL_HC FMA rows {rows}: h differs at {int(diff.sum())} of {diff.numel()} elements; torch combine == "
          f"no-FMA oracle {xla_is_nofma}; kernel == FMA oracle at every differing element {k_is_fma_where_diff}; "
          f"FMA-sensitive elements {fma_sites}; mixed equal {torch.equal(mixed_k, mixed_x)} injection equal "
          f"{torch.equal(inj_k, inj_x)}; XLA_FLAGS={os.environ.get('XLA_FLAGS', '')!r}; calls {prologue}")
    assert prologue == {"mix": 0, "prologue_inj": 1, "prologue_noinj": 0}
    assert xla_is_nofma            # the unfolded combine is IEEE mul-then-add
    assert k_is_fma_where_diff     # the only difference the kernel can show is the contraction's rounding
    if STRICT:
        assert not diff.any() and torch.equal(mixed_k, mixed_x) and torch.equal(inj_k, inj_x)
