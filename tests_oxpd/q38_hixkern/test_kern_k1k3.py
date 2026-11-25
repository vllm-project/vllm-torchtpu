# SPDX-License-Identifier: Apache-2.0
"""q38 HIX-KERN K1 (VLLM_TPU_MK_KERN_SIDE_PREFETCH) and K3 (VLLM_TPU_MK_KERN_ROUTE_VEC): bitwise against the flag-off kernel (DEV sanity).

K3-U  `kern_route.vec_list` equals `dedup_local`'s scalar walk (a Python model of dedup.py dedup_local: first LIVE pick in (b, k) order,
      this rank's experts only, pad rows neither add nor mark) on 400 random pick tables (MR 16, TOPK 10, ER 64, me 0..7, random live
      masks, repeated experts within and across rows).
K3-K  the 16-row pool-GDN segment kernel (CPU interpret, the sidecar test's harness: real per-rank GDN shapes, MoE in every layer),
      ROUTE_VEC 0 vs 1 in one process: residual, rec_ckpt and conv_ckpt bit-identical, all live and with X-K-LIVE (3 live); control: a
      mutant of vec_list that drops the list's last expert must move the residual.
K1-K  the sidecar 2-step chain (side_on = 1, stale pool, random offsets), SIDE_PREFETCH 0 vs 1 in one process: residual and the sidecar
      it leaves bit-identical, all live and 3 live; and with the sidecar off the key refuses at bind (`kern_check`).
Run: Q38_OXPD_FORM=1 JAX_PLATFORMS=cpu PYTHONPATH=src:<tests shim> pytest -q -s --noconftest tests_oxpd/q38_hixkern/test_kern_k1k3.py
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.megakernel import test_hix700_sidecar as S  # the sidecar test's harness (imports the kernel stack and the vllm.envs stub)

import jax.numpy as jnp  # noqa: E402

from vllm_torchtpu.megakernel.blocks.moe import kern_route as KR  # noqa: E402


def _ref_list(ids: np.ndarray, live: np.ndarray, me: int, ER: int, TOPK: int, B: int) -> list:
    seen, out = set(), []
    for b in range(B):
        for k in range(TOPK):
            e = int(ids[b, k])
            if not live[b]:
                continue
            if e // ER == me and e not in seen:
                out.append(e)
            seen.add(e)
    return out


def test_k3_unit_vec_list_equals_scalar_walk():
    rng = np.random.default_rng(7)
    MR, TOPK, ER, B = 16, 10, 64, 16
    for t in range(400):
        me = int(rng.integers(0, 8))
        hot = rng.integers(0, 512, 24)  # a small hot set so experts repeat within and across rows
        ids = np.zeros((MR, 128), np.int32)
        for b in range(B):
            ids[b, :TOPK] = rng.choice(hot if rng.random() < 0.7 else np.arange(512), TOPK, replace=False)
        live = rng.random(MR) < (0.5 if t % 3 else 1.0)
        el, n = KR.vec_list(jnp.asarray(ids), jnp.asarray(np.broadcast_to(live[:, None], (MR, 128)).astype(np.int32)), me, ER, TOPK)  # K3-MOSAIC: lane-dense int32 mask
        el, n = np.asarray(el)[0], int(np.asarray(n)[0, 0])
        ref = _ref_list(ids, live, me, ER, TOPK, B)
        assert n == len(ref) and el[:n].tolist() == ref, (t, me, n, el[:n].tolist(), ref)
    print("Q38KERN K3-U 400/400 random tables: vec_list == dedup_local walk", flush=True)


def _seg(monkeypatch, key: str, val: str, **kw):
    monkeypatch.setenv(key, val)
    try:
        return S._step(monkeypatch, **kw)
    finally:
        monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize("live", [None, [1, 1, 1, 0]])
def test_k3_kernel_bitwise(monkeypatch, live):
    raws, rb, hi, wb, NL = S._pools(11)
    KR.KERN_ROUTE_TRACES.clear()
    a = _seg(monkeypatch, "VLLM_TPU_MK_KERN_ROUTE_VEC", "0", side=False, raws=raws, rb=rb, hi=hi, x_seed=3, live=live)
    assert not KR.KERN_ROUTE_TRACES, KR.KERN_ROUTE_TRACES
    b = _seg(monkeypatch, "VLLM_TPU_MK_KERN_ROUTE_VEC", "1", side=False, raws=raws, rb=rb, hi=hi, x_seed=3, live=live)
    print(f"Q38KERN K3-K engagement: flag-on routes traced {len(KR.KERN_ROUTE_TRACES)} {sorted(set(KR.KERN_ROUTE_TRACES))}", flush=True)
    assert KR.KERN_ROUTE_TRACES, "ROUTE_VEC=1 traced no flag-on route: the check would be vacuous"
    d = (S._ndiff(a["res"], b["res"]), S._ndiff(a["rec"], b["rec"]), S._ndiff(a["conv"], b["conv"]))
    print(f"Q38KERN K3-K live={live}: residual / rec / conv bytes differing {d}", flush=True)
    assert d == (0, 0, 0)
    if live is None:  # control: a list missing its last expert must move the residual (the check sees the list)
        real = KR.vec_list

        def drop_last(ids, lv, me, ER, TOPK):
            el, n = real(ids, lv, me, ER, TOPK)
            return el, jnp.maximum(n - 1, 0)

        monkeypatch.setattr(KR, "vec_list", drop_last)
        c = _seg(monkeypatch, "VLLM_TPU_MK_KERN_ROUTE_VEC", "1", side=False, raws=raws, rb=rb, hi=hi, x_seed=3, live=live)
        nctl = S._ndiff(a["res"], c["res"])
        print(f"Q38KERN K3-K control (last expert dropped): residual bytes differing {nctl} (want > 0)", flush=True)
        assert nctl > 0


@pytest.mark.parametrize("live", [None, [1, 1, 1, 0]])
def test_k1_chain_bitwise(monkeypatch, live):
    rng = np.random.default_rng(31)
    NL = S.qwen38.SEGMENT_LAYERS - 1
    off = rng.integers(0, S.T, (NL, S.NS)).astype(np.int32)
    c = S._chain(monkeypatch, off)  # flag off: got = sidecar step on the stale pool
    kw = dict(side=True, raws=c["raws"], rb=c["rb2"], hi=c["hi"], x_seed=8, side_in=c["side_in"], side_on=1, side_off=off, live=live)
    S.sc.KERN_PF_TRACES.clear()
    a = _seg(monkeypatch, "VLLM_TPU_MK_KERN_SIDE_PREFETCH", "0", **kw)
    assert not S.sc.KERN_PF_TRACES, S.sc.KERN_PF_TRACES
    b = _seg(monkeypatch, "VLLM_TPU_MK_KERN_SIDE_PREFETCH", "1", **kw)
    print(f"Q38KERN K1-K engagement: prefetch sites traced {len(S.sc.KERN_PF_TRACES)} {sorted(set(S.sc.KERN_PF_TRACES))}", flush=True)
    assert S.sc.KERN_PF_TRACES, "SIDE_PREFETCH=1 traced no prefetch site: the check would be vacuous"
    d = (S._ndiff(a["res"], b["res"]), S._ndiff(a["rec"], b["rec"]), S._ndiff(a["conv"], b["conv"]))
    print(f"Q38KERN K1-K live={live}: residual / sidecar rec / conv bytes differing {d}", flush=True)
    assert d == (0, 0, 0)
    if live is None:
        nref = S._ndiff(b["res"], c["ref"]["res"])
        print(f"Q38KERN K1-K vs the flag-off pool reference: residual bytes differing {nref}", flush=True)
        assert nref == 0
        monkeypatch.setattr(S.sc, "_KERN_PF_AHEAD", 0)  # control: prefetch the WRONG plane (o instead of o + 1) must move the residual
        m = _seg(monkeypatch, "VLLM_TPU_MK_KERN_SIDE_PREFETCH", "1", **kw)
        nctl = S._ndiff(m["res"], a["res"])
        print(f"Q38KERN K1-K control (wrong plane prefetched): residual bytes differing {nctl} (want > 0)", flush=True)
        assert nctl > 0


def test_k1_refuses_without_sidecar(monkeypatch):
    from vllm_torchtpu.megakernel_runtime import qwen38_kern as K

    monkeypatch.setenv("VLLM_TPU_MK_KERN_SIDE_PREFETCH", "1")
    monkeypatch.delenv("VLLM_TPU_MK_GDN_SIDECAR", raising=False)
    with pytest.raises(ValueError, match="needs VLLM_TPU_MK_GDN_SIDECAR=1"):
        K.kern_bind_line(0)
    monkeypatch.setenv("VLLM_TPU_MK_GDN_SIDECAR", "1")
    line = K.kern_bind_line(3)
    assert line == "q38 HIX-KERN bind rank=3: side_prefetch=1 sidecar=1 route_vec=0 stock_kv=0", line
