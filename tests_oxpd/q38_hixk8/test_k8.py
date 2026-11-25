# SPDX-License-Identifier: Apache-2.0
"""q38 HIX-KERN K8 REC-SKIP (VLLM_TPU_MK_KERN_REC_SKIP), unrolled form: under X-K-LIVE mode 1 the whole-state GDN recurrence skips pad
requests. CPU interpret, the sidecar test's harness (16 rows, T 4, R 2, 3 GDN layers + the QSA o_proj layer), pool path (SIDECAR off, the
served form after probe C).

K8-P   live masks [1,1,1,1], [1,1,1,0], [1,1,0,0], [1,0,0,0], default interpreter AND uninitialized_memory='nan' (INTERPRET-INIT): residual
       rows, rec_ckpt rows and conv_ckpt slots of the LIVE requests bit-identical between REC_SKIP 0 and 1; at [1,1,1,1] every byte equal.
       Engagement: the flag-on recurrence is traced (GDN_KSKIP_TRACES), the flag-off one is not.
K8-C1  control (planted predicate mutant): live slot 0 treated as pad must move live residual rows.
K8-C2  control (planted stale-VMEM mutant, nan init): a live request's first row reads the PREVIOUS slot's recb rows (a pad slot left
       stale) must move live rows; proves the nan-init leg sees a stale read.
K8-R   refusals: REC_SKIP without X-K-LIVE mode 1, with mode 2, and with GDN-REC-WSHM (GDN_INK bit 1) raise at bind; the bind line format.
"""

from __future__ import annotations

import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from tests.megakernel import test_hix700_sidecar as S

from vllm_torchtpu.megakernel.blocks.gdn import recurrence as REC  # noqa: E402

NAN = pltpu.InterpretParams(uninitialized_memory="nan")


def _seg(monkeypatch, val: str, nan: bool = False, **kw):
    monkeypatch.setenv("VLLM_TPU_MK_KERN_REC_SKIP", val)
    if nan:
        real = S.ExecSpec
        monkeypatch.setattr(S, "ExecSpec", lambda *a, **k: real(*a, **dict(k, interpret=NAN)))
    try:
        return S._step(monkeypatch, **kw)
    finally:
        monkeypatch.delenv("VLLM_TPU_MK_KERN_REC_SKIP", raising=False)
        if nan:
            monkeypatch.setattr(S, "ExecSpec", real)


def _live_diff(a, b, live) -> tuple[int, int, int]:
    rows = np.repeat(np.asarray(live, bool), S.T)
    slots = np.asarray(live, bool)
    nres = S._ndiff(a["res"][:, rows], b["res"][:, rows])
    rec_rows = np.repeat(rows, S.VH)  # rec [R, NL, B * vh, KD, VD]
    nrec = S._ndiff(a["rec"][:, :, rec_rows], b["rec"][:, :, rec_rows])
    nconv = S._ndiff(a["conv"][:, :, slots], b["conv"][:, :, slots])
    return nres, nrec, nconv


def _kd_predicate(monkeypatch):
    """K8 with every request computed under the SAME branch structure: req_live OR live_rows[s T] < 2 (always True at run time, data-
    dependent at trace time). It isolates the skip from XLA:CPU's dynamic-branch codegen (K8-N / K8-N2 controls)."""
    import jax.numpy as jnp
    import vllm_torchtpu.megakernel.live_rows as LR

    real = LR.req_live
    monkeypatch.setattr(LR, "req_live", lambda ctx, s: jnp.logical_or(real(ctx, s), ctx.extra[LR.LIVE_ROWS][s * ctx.ex.T] < 2))
    return real


@pytest.mark.parametrize("nan", [False, True])
@pytest.mark.parametrize("live", [[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]])
def test_k8_pool_live_rows_bitwise(monkeypatch, live, nan):
    """K8-P: (1) the SKIP is bitwise: K8 vs KD (same branch structure, nothing skipped) -> live residual / rec / conv bytes equal;
    (2) against the flag-off gdn_rec: live residual and conv bytes equal; the recurrent-state checkpoint difference is exactly XLA:CPU's
    dynamic-branch codegen set (K8 vs flag-off == KD vs flag-off, printed), never the skip."""
    raws, rb, hi, wb, NL = S._pools(11)
    REC.GDN_KSKIP_TRACES.clear()
    a = _seg(monkeypatch, "0", nan=nan, side=False, raws=raws, rb=rb, hi=hi, x_seed=3, live=live)
    assert not REC.GDN_KSKIP_TRACES, "the flag-off step traced the K8 recurrence"
    b = _seg(monkeypatch, "1", nan=nan, side=False, raws=raws, rb=rb, hi=hi, x_seed=3, live=live)
    print(f"Q38K8 K8-P engagement live={live} nan={nan}: flag-on recurrences traced {len(REC.GDN_KSKIP_TRACES)} "
          f"{sorted(set(REC.GDN_KSKIP_TRACES))}", flush=True)
    assert REC.GDN_KSKIP_TRACES, "REC_SKIP=1 traced no flag-on recurrence: the check would be vacuous"
    real = _kd_predicate(monkeypatch)
    kd = _seg(monkeypatch, "1", nan=nan, side=False, raws=raws, rb=rb, hi=hi, x_seed=3, live=live)
    import vllm_torchtpu.megakernel.live_rows as LR
    monkeypatch.setattr(LR, "req_live", real)
    d_skip = _live_diff(kd, b, live)
    d_off = _live_diff(a, b, live)
    d_ctl = _live_diff(a, kd, live)
    print(f"Q38K8 K8-P live={live} nan={nan}: SKIP (K8 vs KD) live residual / rec / conv bytes differing {d_skip}; vs flag-off {d_off}; "
          f"KD vs flag-off {d_ctl}", flush=True)
    assert d_skip == (0, 0, 0)
    assert d_off[0] == 0 and d_off[2] == 0 and d_off == d_ctl


def test_k8_control_skipping_a_live_request_moves_live_rows(monkeypatch):
    live = [1, 1, 1, 0]
    raws, rb, hi, wb, NL = S._pools(11)
    a = _seg(monkeypatch, "0", side=False, raws=raws, rb=rb, hi=hi, x_seed=3, live=live)
    import jax.numpy as jnp
    import vllm_torchtpu.megakernel.live_rows as LR

    real = LR.req_live
    monkeypatch.setattr(LR, "req_live", lambda ctx, s: jnp.logical_xor(real(ctx, s), s == 0))  # live slot 0 treated as pad
    m = _seg(monkeypatch, "1", side=False, raws=raws, rb=rb, hi=hi, x_seed=3, live=live)
    d = _live_diff(a, m, live)
    print(f"Q38K8 K8-C1 control (live slot 0 skipped): live residual / rec / conv bytes differing {d} (want residual > 0)", flush=True)
    assert d[0] > 0


def test_k8_control_stale_vmem_read_is_seen_under_nan_init(monkeypatch):
    live = [1, 0, 1, 0]
    raws, rb, hi, wb, NL = S._pools(11)
    a = _seg(monkeypatch, "0", nan=True, side=False, raws=raws, rb=rb, hi=hi, x_seed=3, live=live)
    src = open(REC.__file__).read()
    old = "            src = b if t_ == 0 else b - 1\n            q = q * lax.rsqrt"
    assert src.count(old) == 1, src.count(old)  # the K8 body holds this exact text once (gdn_rec's VER branch is followed by "else:")
    import types

    mut = types.ModuleType("rec_mut")
    code = src.replace("            src = b if t_ == 0 else b - 1\n            q = q * lax.rsqrt",
                        "            src = (b - T if s_ > 0 else b) if t_ == 0 else b - 1\n            q = q * lax.rsqrt", 1)
    assert code != src
    mut.__dict__["__name__"] = REC.__name__
    mut.__dict__["__package__"] = REC.__package__
    exec(compile(code, REC.__file__, "exec"), mut.__dict__)
    monkeypatch.setattr(REC, "gdn_rec_kskip", mut.__dict__["gdn_rec_kskip"])
    m = _seg(monkeypatch, "1", nan=True, side=False, raws=raws, rb=rb, hi=hi, x_seed=3, live=live)
    d = _live_diff(a, m, live)
    nan_live = int(np.isnan(np.asarray(m["res"], np.float32)[:, np.repeat(np.asarray(live, bool), S.T)]).sum())
    print(f"Q38K8 K8-C2 control (live request reads the previous, pad slot's stale recb rows; nan init): live residual / rec / conv "
          f"bytes differing {d}; NaN in live residual rows {nan_live} (want differing > 0)", flush=True)
    assert d[0] > 0


def test_k8_refusals_and_bind_line(monkeypatch):
    from vllm_torchtpu.megakernel_runtime import qwen38_k8 as K

    monkeypatch.setenv("VLLM_TPU_MK_KERN_REC_SKIP", "1")
    monkeypatch.delenv("VLLM_TPU_MK_LIVE_ROWS", raising=False)
    monkeypatch.delenv("VLLM_TPU_MK_LIVE_GDN_COMPUTE", raising=False)
    with pytest.raises(ValueError, match="needs X-K-LIVE mode 1"):
        K.k8_check()
    monkeypatch.setenv("VLLM_TPU_MK_LIVE_ROWS", "1")
    monkeypatch.setenv("VLLM_TPU_MK_LIVE_GDN_COMPUTE", "1")
    with pytest.raises(ValueError, match="needs X-K-LIVE mode 1"):
        K.k8_check()
    monkeypatch.delenv("VLLM_TPU_MK_LIVE_GDN_COMPUTE", raising=False)
    monkeypatch.setenv("VLLM_TPU_MK_GDN_INK", "3")
    with pytest.raises(ValueError, match="GDN-REC-WSHM"):
        K.k8_check()
    monkeypatch.setenv("VLLM_TPU_MK_GDN_INK", "2")
    assert K.k8_bind_line(5) == "q38 HIX-KERN K8 bind rank=5: rec_skip=1 live_rows_mode=1 gdn_ink=2"
    monkeypatch.setenv("VLLM_TPU_MK_KERN_REC_SKIP", "0")
    monkeypatch.delenv("VLLM_TPU_MK_LIVE_ROWS", raising=False)
    assert K.k8_bind_line(0) == "q38 HIX-KERN K8 bind rank=0: rec_skip=0 live_rows_mode=0 gdn_ink=2"


def _rec_diff_detail(a, b):
    """(count, [(rank, layer, row, |diff in bf16 ulp| max)]) of differing recurrent-state elements (rec [R, NL, B * vh, KD, VD] bf16)."""
    import ml_dtypes  # noqa: F401  (bf16 view)

    x = np.ascontiguousarray(np.asarray(a["rec"])).view(np.uint16).astype(np.int32)
    y = np.ascontiguousarray(np.asarray(b["rec"])).view(np.uint16).astype(np.int32)
    idx = np.argwhere(x != y)
    out = sorted({(int(r), int(l), int(row)) for r, l, row, *_ in idx})
    return int(len(idx)), out[:12], int(np.abs(x - y).max()) if len(idx) else 0


def test_k8_numerics_control_branch_structure(monkeypatch):
    """K8-N control (HIX-LEAD-10 step 3): is the CPU state difference the SKIP or the conditional-branch codegen of XLA:CPU?
    A = flag-off gdn_rec; A2 = flag-off again (determinism); K = K8 at live [1,1,1,1] (every branch taken); KT = K8 with req_live forced
    True (the same branch structure, predicates constant). If A vs A2 = 0, K vs KT = 0 and A vs K = A vs KT > 0, the difference comes from
    the branch structure on XLA:CPU, not from skipping (no pad request exists at [1,1,1,1])."""
    live = [1, 1, 1, 1]
    raws, rb, hi, wb, NL = S._pools(11)
    A = _seg(monkeypatch, "0", side=False, raws=raws, rb=rb, hi=hi, x_seed=3, live=live)
    A2 = _seg(monkeypatch, "0", side=False, raws=raws, rb=rb, hi=hi, x_seed=3, live=live)
    K = _seg(monkeypatch, "1", side=False, raws=raws, rb=rb, hi=hi, x_seed=3, live=live)
    import jax.numpy as jnp
    import vllm_torchtpu.megakernel.live_rows as LR

    monkeypatch.setattr(LR, "req_live", lambda ctx, s: jnp.bool_(True))
    KT = _seg(monkeypatch, "1", side=False, raws=raws, rb=rb, hi=hi, x_seed=3, live=live)
    rows = {"A-A2": _rec_diff_detail(A, A2), "K-KT": _rec_diff_detail(K, KT), "A-K": _rec_diff_detail(A, K), "A-KT": _rec_diff_detail(A, KT)}
    res = {k: S._ndiff(x["res"], y["res"]) for k, (x, y) in {"A-A2": (A, A2), "K-KT": (K, KT), "A-K": (A, K), "A-KT": (A, KT)}.items()}
    for k, v in rows.items():
        print(f"Q38K8 K8-N {k}: rec elements differing {v[0]} at (rank, layer, row) {v[1]} max |ulp| {v[2]}; residual bytes differing {res[k]}",
              flush=True)
    assert rows["A-A2"][0] == 0 and res["A-A2"] == 0


def test_k8_numerics_control_dynamic_true_predicate(monkeypatch):
    """K8-N2: KD = K8 with a predicate that is always True at run time but data-dependent at trace time (req_live OR live_rows[s T] < 2;
    the mask holds 0 / 1). If KD == K (6 elements) and KD != A, a real (unfoldable) branch is what moves the XLA:CPU state numerics."""
    live = [1, 1, 1, 1]
    raws, rb, hi, wb, NL = S._pools(11)
    A = _seg(monkeypatch, "0", side=False, raws=raws, rb=rb, hi=hi, x_seed=3, live=live)
    K = _seg(monkeypatch, "1", side=False, raws=raws, rb=rb, hi=hi, x_seed=3, live=live)
    import jax.numpy as jnp
    import vllm_torchtpu.megakernel.live_rows as LR

    real = LR.req_live
    monkeypatch.setattr(LR, "req_live", lambda ctx, s: jnp.logical_or(real(ctx, s), ctx.extra[LR.LIVE_ROWS][s * ctx.ex.T] < 2))
    KD = _seg(monkeypatch, "1", side=False, raws=raws, rb=rb, hi=hi, x_seed=3, live=live)
    for k, (x, y) in {"A-KD": (A, KD), "K-KD": (K, KD)}.items():
        v = _rec_diff_detail(x, y)
        print(f"Q38K8 K8-N2 {k}: rec elements differing {v[0]} at (rank, layer, row) {v[1]} max |ulp| {v[2]}; residual bytes differing "
              f"{S._ndiff(x['res'], y['res'])}", flush=True)
