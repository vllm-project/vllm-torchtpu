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
"""On-TPU v7x-8 (2x2x1, 8 TensorCores) FULL_EP parity & latency benchmark."""

from __future__ import annotations

import dataclasses
import gc
import hashlib
import json
import pathlib
import sys
import time
from typing import Any

_THIS_DIR = str(pathlib.Path(__file__).resolve().parent)
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SRC_DIR = str(_REPO_ROOT / "src")
for _p in (_SRC_DIR, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from frozen_utils import load_frozen_qwen35  # noqa: E402
from jax import lax  # noqa: E402
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from vllm_torchtpu import megakernel as mk  # noqa: E402
from vllm_torchtpu.megakernel import (  # noqa: E402
    cache,
    kernel,
    planner,
)
from vllm_torchtpu.megakernel.blocks.gdn import GDNBlock  # noqa: E402
from vllm_torchtpu.megakernel.cache import should_fuse_pool_io  # noqa: E402
from vllm_torchtpu.megakernel.exec_spec import ExecSpec  # noqa: E402
from vllm_torchtpu.megakernel.models import qwen35  # noqa: E402
from vllm_torchtpu.megakernel.models.qwen35_profile import (  # noqa: E402
    baseline_tiles,
)
from vllm_torchtpu.megakernel.ops import (  # noqa: E402
    BF16,
    F32,
    FP8,
    I32,
    rup,
)
from vllm_torchtpu.megakernel.profile import get_profile  # noqa: E402

# Qwen3.5 baseline tile rule (bit-exact with the frozen kernel); the bisect
# bench below swaps this module global for a flagged variant.
frozen_parity_tiles = baseline_tiles


def _has_tpu_v7x8() -> bool:
    try:
        return len(jax.devices("tpu")) >= 8
    except RuntimeError:
        return False


def _load_frozen_mk():
    frozen_mk, _ = load_frozen_qwen35()
    return frozen_mk


def _load_frozen_pool_io(frozen_mk):
    del frozen_mk
    _, frozen_pio = load_frozen_qwen35()
    return frozen_pio


def _init_shared_weights_on_tpu(
    mesh: Mesh,
    g: qwen35.Qwen35Geometry,
    ex: ExecSpec,
    f_lay: dict[str, tuple[int, int]],
    f_S: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Allocate 60-layer hybrid FULL_EP (att_every=4) weights on TPU HBM once."""
    ms = qwen35.qwen35_spec(g)
    p = planner.plan(ms, ex)
    nt, tk = p.nt, p.tk
    l_lay = p.smalls_layout
    l_S = p.smalls_width

    R = ex.R
    L, H, E, HD, VD = g.L, g.H, g.E, g.HD, g.VD
    nh = g.nh(R)
    G = g.G(R)
    vh = g.vh(R)
    qkv = g.QKV(R)
    nli = g.NLI(R)
    nai = g.NAI(R)
    in_s_w = max(nli, nai)
    isr, ir, ew = g.ISR(R), g.IR(R), g.EW(R)
    ngdn = g.NGDN
    natt = g.NATT
    koff = nh * 2 * HD

    out_specs = {
        "lin_in": P("tp"),
        "lin_out": P("tp"),
        "att_in": P("tp"),
        "att_out": P("tp"),
        "router": P("tp"),
        "sh13": P("tp"),
        "sh2": P("tp"),
        "e13": P("tp"),
        "e2": P("tp"),
        "es": P("tp"),
        "smalls_lib": P("tp"),
        "smalls_f": P("tp"),
        "final_norm": P("tp"),
    }

    def _local_init():
        me = lax.axis_index("tp")
        kv_grp = lax.div(me, G)
        root = jax.random.PRNGKey(20261013)

        def key_for(tag: int, rank_dep: Any = None):
            k = jax.random.fold_in(root, tag)
            if rank_dep is not None:
                k = jax.random.fold_in(k, rank_dep)
            return k

        lin_in = (
            jax.random.normal(
                key_for(101, me),
                (1, ngdn, nt["lin_in"], tk["lin_in"], nli),
                dtype=BF16,
            )
            * jnp.bfloat16(0.125)
        ).astype(FP8)
        lin_out = (
            jax.random.normal(
                key_for(102, me),
                (1, ngdn, nt["lin_out"], tk["lin_out"], H),
                dtype=BF16,
            )
            * jnp.bfloat16(0.125)
        ).astype(FP8)

        att_q = (
            jax.random.normal(
                key_for(1, me),
                (1, natt, nt["att_in"], tk["att_in"], koff),
                dtype=BF16,
            )
            * jnp.bfloat16(0.125)
        ).astype(FP8)
        att_kv = (
            jax.random.normal(
                key_for(2, kv_grp),
                (1, natt, nt["att_in"], tk["att_in"], 2 * HD),
                dtype=BF16,
            )
            * jnp.bfloat16(0.125)
        ).astype(FP8)
        att_in = jnp.concatenate([att_q, att_kv], axis=-1)

        att_out = (
            jax.random.normal(
                key_for(3, me),
                (1, natt, nt["att_out"], tk["att_out"], H),
                dtype=BF16,
            )
            * jnp.bfloat16(0.125)
        ).astype(FP8)

        router = jax.random.normal(
            key_for(4),
            (1, L, nt["router"], tk["router"], E),
            dtype=BF16,
        ) * jnp.bfloat16(0.05)

        sh13 = (
            jax.random.normal(
                key_for(5, me),
                (1, L, nt["sh13"], tk["sh13"], 2 * isr),
                dtype=BF16,
            )
            * jnp.bfloat16(0.125)
        ).astype(FP8)
        sh2 = (
            jax.random.normal(
                key_for(6, me),
                (1, L, nt["sh2"], tk["sh2"], H),
                dtype=BF16,
            )
            * jnp.bfloat16(0.125)
        ).astype(FP8)

        k_e13_base = key_for(7, me)

        def _gen_e13(_, lyr):
            kl = jax.random.fold_in(k_e13_base, lyr)
            w = (
                jax.random.normal(kl, (ew, nt["e13"], tk["e13"], 2 * ir), dtype=BF16)
                * jnp.bfloat16(0.125)
            ).astype(FP8)
            return None, w

        _, e13_layers = lax.scan(_gen_e13, None, jnp.arange(L, dtype=I32))
        e13 = e13_layers[None, ...]

        k_e2_base = key_for(8, me)

        def _gen_e2(_, lyr):
            kl = jax.random.fold_in(k_e2_base, lyr)
            w = (
                jax.random.normal(kl, (ew, nt["e2"], tk["e2"], H), dtype=BF16)
                * jnp.bfloat16(0.125)
            ).astype(FP8)
            return None, w

        _, e2_layers = lax.scan(_gen_e2, None, jnp.arange(L, dtype=I32))
        e2 = e2_layers[None, ...]

        es = (
            jnp.abs(
                jax.random.normal(key_for(9, me), (1, L, ew, 1, 2 * ir + H), dtype=F32)
            )
            * 0.008
            + 0.004
        )

        in_s_parts = [
            jnp.abs(jax.random.normal(key_for(13, me), (1, L, koff), dtype=F32)) * 0.008
            + 0.004,
            jnp.abs(jax.random.normal(key_for(14, kv_grp), (1, L, 2 * HD), dtype=F32))
            * 0.008
            + 0.004,
        ]
        if in_s_w > nai:
            in_s_parts.append(
                jnp.abs(
                    jax.random.normal(key_for(103, me), (1, L, in_s_w - nai), dtype=F32)
                )
                * 0.008
                + 0.004
            )
        alog_vh = jax.random.normal(key_for(105, me), (1, L, vh), dtype=F32) * 0.1 - 1.0
        dtb_vh = jax.random.normal(key_for(106, me), (1, L, vh), dtype=F32) * 0.1
        fields = {
            "in_norm": jax.random.normal(key_for(10), (1, L, H), dtype=F32) * 0.02,
            "post_norm": jax.random.normal(key_for(11), (1, L, H), dtype=F32) * 0.02,
            "sh_gate": jax.random.normal(key_for(12), (1, L, H), dtype=F32) * 0.05,
            "in_s": jnp.concatenate(in_s_parts, axis=-1),
            "out_s": jnp.abs(jax.random.normal(key_for(15, me), (1, L, H), dtype=F32))
            * 0.008
            + 0.004,
            "conv": jax.random.normal(key_for(104, me), (1, L, 4 * qkv), dtype=F32)
            * 0.05,
            "alog": jnp.pad(alog_vh, ((0, 0), (0, 0), (0, 128 - vh))),
            "dtb": jnp.pad(dtb_vh, ((0, 0), (0, 0), (0, 128 - vh))),
            "gnorm": jnp.ones((1, L, VD), dtype=F32)
            + jax.random.normal(key_for(107, me), (1, L, VD), dtype=F32) * 0.02,
            "qn": jax.random.normal(key_for(16, me), (1, L, HD), dtype=F32) * 0.02,
            "kn": jax.random.normal(key_for(17, kv_grp), (1, L, HD), dtype=F32) * 0.02,
            "sh13_s": jnp.abs(
                jax.random.normal(key_for(18, me), (1, L, 2 * isr), dtype=F32)
            )
            * 0.008
            + 0.004,
            "sh2_s": jnp.abs(jax.random.normal(key_for(19, me), (1, L, H), dtype=F32))
            * 0.008
            + 0.004,
        }

        sm_lib = jnp.zeros((1, L, 1, l_S), dtype=F32)
        sm_f = jnp.zeros((1, L, 1, f_S), dtype=F32)
        for name, (lo, w) in l_lay.items():
            fo, _ = f_lay[name]
            val = fields[name][:, :, :w]
            sm_lib = sm_lib.at[:, :, 0, lo : lo + w].set(val)
            sm_f = sm_f.at[:, :, 0, fo : fo + w].set(val)

        final_norm = jax.random.normal(key_for(20), (1, 1, H), dtype=F32) * 0.02

        return {
            "lin_in": lin_in,
            "lin_out": lin_out,
            "att_in": att_in,
            "att_out": att_out,
            "router": router,
            "sh13": sh13,
            "sh2": sh2,
            "e13": e13,
            "e2": e2,
            "es": es,
            "smalls_lib": sm_lib,
            "smalls_f": sm_f,
            "final_norm": final_norm,
        }

    init_fn = jax.jit(
        jax.shard_map(
            _local_init,
            mesh=mesh,
            in_specs=(),
            out_specs=out_specs,
            check_vma=False,
        )
    )
    raw = init_fn()
    jax.block_until_ready(raw["e13"])

    w_lib = {
        "lin_in": raw["lin_in"],
        "lin_out": raw["lin_out"],
        "att_in": raw["att_in"],
        "att_out": raw["att_out"],
        "router": raw["router"],
        "sh13": raw["sh13"],
        "sh2": raw["sh2"],
        "e13": raw["e13"],
        "e2": raw["e2"],
        "es": raw["es"],
        "smalls": raw["smalls_lib"],
        "final_norm": raw["final_norm"],
    }
    w_f = {
        "lin_in": raw["lin_in"],
        "lin_out": raw["lin_out"],
        "att_in": raw["att_in"],
        "att_out": raw["att_out"],
        "router": raw["router"],
        "sh13": raw["sh13"],
        "sh2": raw["sh2"],
        "e13": raw["e13"],
        "e2": raw["e2"],
        "es": raw["es"],
        "smalls": raw["smalls_f"],
        "final_norm": raw["final_norm"],
    }
    return w_lib, w_f


def _init_bucket_inputs_on_tpu(
    mesh: Mesh,
    g: qwen35.Qwen35Geometry,
    ex: ExecSpec,
) -> tuple[jax.Array, jax.Array, dict[str, Any]]:
    """Allocate `(pos, hid, states)` for a single `(T, B)` hybrid bucket on TPU HBM."""
    rep = NamedSharding(mesh, P())
    R, B, T, M, NS, ctx = ex.R, ex.B, ex.T, ex.M, ex.NS, ex.ctx
    G = g.G(R)
    ngdn, natt, H, HD, KD, VD = g.NGDN, g.NATT, g.H, g.HD, g.KD, g.VD
    qkv = g.QKV(R)
    vh = g.vh(R)
    conv_shape = (
        (ngdn, rup(3 * B, 8), qkv) if T == 1 else (ngdn, NS, rup(3 + T, 8), qkv)
    )

    base_pos = jnp.int32(2048) + jnp.arange(NS, dtype=I32) * jnp.int32(16)
    if T == 1:
        pos_host = base_pos
    else:
        pos_host = (base_pos[:, None] + jnp.arange(T, dtype=I32)[None, :]).reshape(B)
    pos = jax.device_put(pos_host, rep)

    st_specs = {
        "conv": P("tp"),
        "rec": P("tp"),
        "kc": P("tp"),
        "vc": P("tp"),
    }

    def _local_bucket():
        me = lax.axis_index("tp")
        kv_grp = lax.div(me, G)
        root = jax.random.PRNGKey(99000 + T * 100 + B)
        k_hid = jax.random.fold_in(root, 1)
        k_kc_base = jax.random.fold_in(jax.random.fold_in(root, 2), kv_grp)
        k_vc_base = jax.random.fold_in(jax.random.fold_in(root, 3), kv_grp)
        k_conv = jax.random.fold_in(jax.random.fold_in(root, 4), me)
        k_rec_base = jax.random.fold_in(jax.random.fold_in(root, 5), me)

        hid_b = jax.random.normal(k_hid, (B, H), dtype=BF16) * jnp.bfloat16(0.1)
        hid_m = jnp.pad(hid_b, ((0, M - B), (0, 0)))

        conv = jax.random.normal(k_conv, conv_shape, dtype=F32) * jnp.float32(0.05)

        def _gen_rec(_, lyr):
            kr = jax.random.fold_in(k_rec_base, lyr)
            rec_l = jax.random.normal(kr, (NS * vh, KD, VD), dtype=BF16) * jnp.bfloat16(
                0.02
            )
            return None, rec_l

        _, rec = lax.scan(_gen_rec, None, jnp.arange(ngdn, dtype=I32))

        def _gen_kv(_, lyr):
            kk = jax.random.fold_in(k_kc_base, lyr)
            kv = jax.random.fold_in(k_vc_base, lyr)
            kc_l = jax.random.normal(kk, (NS, ctx, HD), dtype=BF16) * jnp.bfloat16(0.08)
            vc_l = jax.random.normal(kv, (NS, ctx, HD), dtype=BF16) * jnp.bfloat16(0.08)
            return None, (kc_l, vc_l)

        _, (kc, vc) = lax.scan(_gen_kv, None, jnp.arange(natt, dtype=I32))
        st: dict[str, Any] = {
            "conv": conv,
            "rec": rec,
            "kc": kc,
            "vc": vc,
        }
        return hid_m, st

    init_fn = jax.jit(
        jax.shard_map(
            _local_bucket,
            mesh=mesh,
            in_specs=(),
            out_specs=(P("tp"), st_specs),
            check_vma=False,
        )
    )
    hid, st = init_fn()
    jax.block_until_ready(hid)
    return pos, hid, st


def _count_routed_experts(
    hid: jax.Array,
    router_w: jax.Array,
    B: int,
    H: int,
    E: int,
    TOPK: int,
    R: int,
) -> tuple[float, float]:
    """Compute mean distinct routed experts hit across all 60 layers."""
    hid_0 = np.asarray(jax.device_get(kernel.unfold(hid, R)[0, :B])).astype(np.float32)
    router_0 = np.asarray(jax.device_get(router_w[0])).astype(np.float32)
    L = router_0.shape[0]
    w_r = router_0.reshape(L, H, E)
    logits = np.einsum("bh,lhe->lbe", hid_0, w_r)
    top_idx = np.argpartition(-logits, TOPK, axis=-1)[:, :, :TOPK]
    er = E // R
    uniq_global = [len(np.unique(top_idx[lyr])) for lyr in range(L)]
    uniq_per_rank = [
        np.mean(
            [len(np.unique(top_idx[lyr][(top_idx[lyr] // er) == r])) for r in range(R)]
        )
        for lyr in range(L)
    ]
    return float(np.mean(uniq_global)), float(np.mean(uniq_per_rank))


def _time_step_fn(
    step_fn,
    pos: jax.Array,
    hid: jax.Array,
    weights: dict[str, Any],
    states: dict[str, Any],
    *,
    T: int,
    warmup: int = 15,
    batches: int = 10,
    steps_per_batch: int = 10,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Warm up and measure step latency over `batches * steps_per_batch` iterations."""
    for _ in range(warmup):
        out_hid, out_st = step_fn(pos, hid, weights, states)
        if T == 1:
            states = out_st
    jax.block_until_ready(out_hid)

    batch_us = []
    for _ in range(batches):
        t0 = time.perf_counter()
        for _ in range(steps_per_batch):
            out_hid, out_st = step_fn(pos, hid, weights, states)
            if T == 1:
                states = out_st
        jax.block_until_ready(out_hid)
        dt_us = (time.perf_counter() - t0) * 1e6 / steps_per_batch
        batch_us.append(dt_us)

    arr = np.asarray(batch_us, dtype=np.float64)
    stats = {
        "median_us": float(np.median(arr)),
        "mean_us": float(np.mean(arr)),
        "p10_us": float(np.percentile(arr, 10)),
        "p90_us": float(np.percentile(arr, 90)),
        "min_us": float(np.min(arr)),
    }
    return stats, states


def test_m2_tpu_v7x8_full_ep_parity_and_latency():
    """Run FULL_EP (L=60, att_every=4, R=8, ctx=4096) on-TPU v7x-8 benchmark."""
    if not _has_tpu_v7x8():
        pytest.skip("Requires 8 TPU TensorCores (v7x-8)")
    tpu_devs = jax.devices("tpu")
    frozen_mk = _load_frozen_mk()
    mesh = Mesh(tpu_devs[:8], ("tp",))
    rep = NamedSharding(mesh, P())

    g = dataclasses.replace(qwen35.FULL_EP, L=60, att_every=4)
    ex_init = ExecSpec(B=4, T=1, R=8, ctx=4096, stack_e=True)
    c_f_init = frozen_mk.MKConfig(
        R=8,
        B=4,
        T=1,
        L=60,
        att_every=4,
        moe_tp=False,
        ctx=4096,
        stack_e=True,
    )
    f_lay, f_S = frozen_mk.smalls_layout(c_f_init)

    t_w0 = time.perf_counter()
    w_lib, w_f = _init_shared_weights_on_tpu(mesh, g, ex_init, f_lay, f_S)
    dt_w = time.perf_counter() - t_w0
    print(
        f"\n[M2 TPU v7x-8 Bench] Allocated shared 60-layer hybrid FULL_EP "
        f"(att_every=4) weights on 8 TPU cores in {dt_w:.2f}s"
    )

    buckets = [
        (1, 4),
        (1, 8),
        (1, 16),
        (1, 32),
        (1, 64),
        (4, 4),
        (4, 8),
        (4, 16),
        (4, 32),
        (4, 64),
    ]
    results = []

    for T, B in buckets:
        ex = ExecSpec(
            B=B,
            T=T,
            R=8,
            ctx=4096,
            stack_e=True,
            tiles=frozen_parity_tiles(B),
        )
        c_f = frozen_mk.MKConfig(
            R=8,
            B=B,
            T=T,
            L=60,
            att_every=4,
            moe_tp=False,
            ctx=4096,
            stack_e=True,
        )
        tiles_f = frozen_mk.default_tiles(c_f)
        if B > 48:
            tiles_f = dataclasses.replace(tiles_f, nbe=6)
        donate = T == 1
        conv_key = "conv" if T == 1 else "conv_ckpt"
        rec_key = "rec" if T == 1 else "rec_ckpt"

        # --- 1. Library megakernel: compile, parity snapshot, and latency ---
        pos, hid_lib, st_lib = _init_bucket_inputs_on_tpu(mesh, g, ex)
        uniq_glob, uniq_rank = _count_routed_experts(
            hid_lib, w_lib["router"], B, g.H, g.E, g.TOPK, 8
        )
        step_lib, info_lib = qwen35.make_step(mesh, g, ex, donate_states=donate)
        vmem_lib = info_lib["vmem_scratch_bytes"]
        comp_lib = step_lib.lower(pos, hid_lib, w_lib, st_lib).compile()
        lib_sha = hashlib.sha256(comp_lib.as_text().encode("utf-8")).hexdigest()[:16]
        del comp_lib

        out_hid_lib, out_st_lib = step_lib(pos, hid_lib, w_lib, st_lib)
        jax.block_until_ready(out_hid_lib)
        if donate:
            st_lib = out_st_lib
        elif T > 1:
            acc_dev = jax.device_put((jnp.arange(ex.NS, dtype=I32) % T) + 1, rep)
            vh = g.vh(8)
            acc_st = jax.jit(
                lambda s, a, _t=T, _vh=vh: GDNBlock.accept(s, s, a, T=_t, vh=_vh)
            )(out_st_lib, acc_dev)
            jax.block_until_ready(acc_st["rec"])
            assert acc_st["conv"].shape == st_lib["conv"].shape
            assert acc_st["rec"].shape == st_lib["rec"].shape
            del acc_st, acc_dev

        hid_lib_np = np.asarray(jax.device_get(kernel.unfold(out_hid_lib, 8)))
        conv_unfold_lib = kernel.unfold(out_st_lib[conv_key], 8)
        rec_unfold_lib = kernel.unfold(out_st_lib[rec_key], 8)
        kc_unfold_lib = kernel.unfold(out_st_lib["kc"], 8)
        vc_unfold_lib = kernel.unfold(out_st_lib["vc"], 8)
        conv_snap_lib = np.asarray(jax.device_get(conv_unfold_lib[:, :4, ...]))
        rec_snap_lib = np.asarray(jax.device_get(rec_unfold_lib[:, :4, :, :16, :64]))
        kc_snap_lib = np.asarray(
            jax.device_get(kc_unfold_lib[:, :4, 0, 2048 : 2048 + T, :])
        )
        vc_snap_lib = np.asarray(
            jax.device_get(vc_unfold_lib[:, :4, 0, 2048 : 2048 + T, :])
        )
        del conv_unfold_lib, rec_unfold_lib, kc_unfold_lib, vc_unfold_lib, out_st_lib

        lib_stats, st_lib = _time_step_fn(step_lib, pos, hid_lib, w_lib, st_lib, T=T)
        del pos, hid_lib, st_lib, out_hid_lib, step_lib
        gc.collect()

        # --- 2. Frozen megakernel: compile, parity snapshot, and latency ---
        pos_f, hid_f, st_f = _init_bucket_inputs_on_tpu(mesh, g, ex)
        raw_step_f, info_f = frozen_mk.make_step(mesh, c_f, tiles_f)
        vmem_f = info_f["vmem_scratch_bytes"]
        step_f = jax.jit(raw_step_f, donate_argnums=(3,) if donate else ())
        comp_f = step_f.lower(pos_f, hid_f, w_f, st_f).compile()
        f_sha = hashlib.sha256(comp_f.as_text().encode("utf-8")).hexdigest()[:16]
        del comp_f

        out_hid_f, out_st_f = step_f(pos_f, hid_f, w_f, st_f)
        jax.block_until_ready(out_hid_f)
        if donate:
            st_f = out_st_f
        hid_f_np = np.asarray(jax.device_get(kernel.unfold(out_hid_f, 8)))
        conv_unfold_f = kernel.unfold(out_st_f["conv"], 8)
        rec_unfold_f = kernel.unfold(out_st_f["rec"], 8)
        kc_unfold_f = kernel.unfold(out_st_f["kc"], 8)
        vc_unfold_f = kernel.unfold(out_st_f["vc"], 8)
        conv_snap_f = np.asarray(jax.device_get(conv_unfold_f[:, :4, ...]))
        rec_snap_f = np.asarray(jax.device_get(rec_unfold_f[:, :4, :, :16, :64]))
        kc_snap_f = np.asarray(
            jax.device_get(kc_unfold_f[:, :4, 0, 2048 : 2048 + T, :])
        )
        vc_snap_f = np.asarray(
            jax.device_get(vc_unfold_f[:, :4, 0, 2048 : 2048 + T, :])
        )
        del conv_unfold_f, rec_unfold_f, kc_unfold_f, vc_unfold_f, out_st_f

        frozen_stats, st_f = _time_step_fn(step_f, pos_f, hid_f, w_f, st_f, T=T)
        del pos_f, hid_f, st_f, out_hid_f, step_f, raw_step_f
        gc.collect()

        # --- 3. Parity verification ---
        live_lib = hid_lib_np[:, :B, :].astype(np.float32)
        live_f = hid_f_np[:, :B, :].astype(np.float32)
        assert np.all(np.isfinite(live_lib)), (
            f"Non-finite output in library at T={T}, B={B}"
        )
        assert np.std(live_lib) > 1e-4, (
            f"Trivial zero output in library at T={T}, B={B}"
        )

        hid_bit_exact = bool(np.array_equal(hid_lib_np[:, :B, :], hid_f_np[:, :B, :]))
        conv_bit_exact = bool(np.array_equal(conv_snap_lib, conv_snap_f))
        rec_bit_exact = bool(np.array_equal(rec_snap_lib, rec_snap_f))
        kc_bit_exact = bool(np.array_equal(kc_snap_lib, kc_snap_f))
        vc_bit_exact = bool(np.array_equal(vc_snap_lib, vc_snap_f))
        max_abs_diff = float(np.max(np.abs(live_lib - live_f)))
        denom = float(np.linalg.norm(live_lib) * np.linalg.norm(live_f))
        cos_sim = float(np.sum(live_lib * live_f) / denom) if denom > 0 else 1.0

        delta_pct = (
            (lib_stats["median_us"] - frozen_stats["median_us"])
            / frozen_stats["median_us"]
            * 100.0
        )
        row = {
            "T": T,
            "B": B,
            "NS": B // T,
            "GCH": B > 16,
            "uniq_experts_global": round(uniq_glob, 1),
            "uniq_experts_per_rank": round(uniq_rank, 1),
            "lib_median_us": round(lib_stats["median_us"], 1),
            "frozen_median_us": round(frozen_stats["median_us"], 1),
            "delta_pct": round(delta_pct, 2),
            "lib_mean_us": round(lib_stats["mean_us"], 1),
            "frozen_mean_us": round(frozen_stats["mean_us"], 1),
            "vmem_lib_mib": round(vmem_lib / (1024 * 1024), 2),
            "vmem_frozen_mib": round(vmem_f / (1024 * 1024), 2),
            "hid_bit_exact": hid_bit_exact,
            "conv_bit_exact": conv_bit_exact,
            "rec_bit_exact": rec_bit_exact,
            "kc_bit_exact": kc_bit_exact,
            "vc_bit_exact": vc_bit_exact,
            "max_abs_diff": max_abs_diff,
            "cos_sim": cos_sim,
            "lib_sha256": lib_sha,
            "frozen_sha256": f_sha,
            "sha256_match": lib_sha == f_sha,
        }
        results.append(row)
        print(
            f"[Bucket T={T} B={B:2d} (NS={B // T:2d}, GCH={B > 16}, "
            f"experts={row['uniq_experts_global']:.1f}/{g.E} global, "
            f"{row['uniq_experts_per_rank']:.1f}/{g.ER(8)} per-rank)] "
            f"lib={row['lib_median_us']:6.1f} us vs "
            f"frozen={row['frozen_median_us']:6.1f} us ({row['delta_pct']:+.2f}%) | "
            f"VMEM {row['vmem_lib_mib']} vs {row['vmem_frozen_mib']} MiB | "
            f"bit_exact(hid={hid_bit_exact}, conv={conv_bit_exact}, "
            f"rec={rec_bit_exact}, kc={kc_bit_exact}, vc={vc_bit_exact}) "
            f"sha_match={row['sha256_match']} "
            f"max_diff={max_abs_diff:.3e} cos={cos_sim:.8f}",
            flush=True,
        )

    print("\n=== M2_TPU_V7X8_BENCH_JSON_START ===")
    print(json.dumps(results, indent=2))
    print("=== M2_TPU_V7X8_BENCH_JSON_END ===", flush=True)

    for r in results:
        T, B = r["T"], r["B"]
        assert r["hid_bit_exact"], (
            f"Hidden state not bit-exact at T={T}, B={B}: max_diff={r['max_abs_diff']}"
        )
        assert r["conv_bit_exact"], f"Conv state not bit-exact at T={T}, B={B}"
        assert r["rec_bit_exact"], f"Rec state not bit-exact at T={T}, B={B}"
        assert r["kc_bit_exact"], f"KC state not bit-exact at T={T}, B={B}"
        assert r["vc_bit_exact"], f"VC state not bit-exact at T={T}, B={B}"

    # --- 4. Also measure M1 (att_every=1) B=64 (T=1, T=4) for PR #1837 ---
    g_m1 = dataclasses.replace(qwen35.FULL_EP, L=60, att_every=1)
    p_m1 = planner.plan(qwen35.qwen35_spec(g_m1), ex_init)
    m1_lay = p_m1.smalls_layout
    m1_S = p_m1.smalls_width

    def _build_m1_weights(att_in_15, att_out_15, sm_f_all):
        att_in_60 = jnp.tile(att_in_15, (1, 4, 1, 1, 1))
        att_out_60 = jnp.tile(att_out_15, (1, 4, 1, 1, 1))
        sm_m1 = jnp.zeros((1, 60, 1, m1_S), dtype=F32)
        for name, (lo, w) in m1_lay.items():
            fo, _ = f_lay[name]
            sm_m1 = sm_m1.at[:, :, 0, lo : lo + w].set(sm_f_all[:, :, 0, fo : fo + w])
        lin_in_0 = jnp.zeros((1, 1, 8, 512, g.NLI(8)), dtype=FP8)
        lin_out_0 = jnp.zeros((1, 1, 2, 512, g.H), dtype=FP8)
        return att_in_60, att_out_60, sm_m1, lin_in_0, lin_out_0

    att_in_60, att_out_60, sm_m1, lin_in_0, lin_out_0 = jax.jit(
        jax.shard_map(
            _build_m1_weights,
            mesh=mesh,
            in_specs=(P("tp"), P("tp"), P("tp")),
            out_specs=(P("tp"), P("tp"), P("tp"), P("tp"), P("tp")),
            check_vma=False,
        )
    )(w_lib["att_in"], w_lib["att_out"], w_f["smalls"])
    jax.block_until_ready(att_in_60)

    del w_lib["lin_in"], w_lib["lin_out"], w_f["lin_in"], w_f["lin_out"]
    w_m1_lib = {
        "att_in": att_in_60,
        "att_out": att_out_60,
        "router": w_lib["router"],
        "sh13": w_lib["sh13"],
        "sh2": w_lib["sh2"],
        "e13": w_lib["e13"],
        "e2": w_lib["e2"],
        "es": w_lib["es"],
        "smalls": sm_m1,
        "final_norm": w_lib["final_norm"],
    }
    w_m1_f = {
        **w_m1_lib,
        "lin_in": lin_in_0,
        "lin_out": lin_out_0,
        "smalls": w_f["smalls"],
    }

    m1_results = []
    for T, B in [(1, 64), (4, 64)]:
        ex_m1 = ExecSpec(
            B=B,
            T=T,
            R=8,
            ctx=4096,
            stack_e=True,
            tiles=frozen_parity_tiles(B),
        )
        c_f_m1 = frozen_mk.MKConfig(
            R=8,
            B=B,
            T=T,
            L=60,
            att_every=1,
            moe_tp=False,
            ctx=4096,
            stack_e=True,
        )
        tiles_f_m1 = frozen_mk.default_tiles(c_f_m1)
        if B > 48:
            tiles_f_m1 = dataclasses.replace(tiles_f_m1, nbe=6)
        pos_m1, hid_m1, st_2 = _init_bucket_inputs_on_tpu(mesh, g, ex_m1)

        def _tile_kv(kc_15, vc_15, _b=B, _t=T):
            kc_60 = jnp.tile(kc_15, (4, 1, 1, 1))
            vc_60 = jnp.tile(vc_15, (4, 1, 1, 1))
            conv_0 = (
                jnp.zeros((1, rup(3 * _b, 8), g.QKV(8)), dtype=F32)
                if _t == 1
                else jnp.zeros((1, _b // _t, rup(3 + _t, 8), g.QKV(8)), dtype=F32)
            )
            rec_0 = jnp.zeros((1, (_b // _t) * g.vh(8), g.KD, g.VD), dtype=BF16)
            return kc_60, vc_60, conv_0, rec_0

        kc_60, vc_60, conv_0, rec_0 = jax.jit(
            jax.shard_map(
                _tile_kv,
                mesh=mesh,
                in_specs=(P("tp"), P("tp")),
                out_specs=(P("tp"), P("tp"), P("tp"), P("tp")),
                check_vma=False,
            )
        )(st_2["kc"], st_2["vc"])
        del st_2
        st_m1_lib = {"kc": kc_60, "vc": vc_60}
        step_m1_lib, info_m1_lib = qwen35.make_step(
            mesh, g_m1, ex_m1, donate_states=(T == 1)
        )
        out_hid_m1_lib, out_st_m1_lib = step_m1_lib(pos_m1, hid_m1, w_m1_lib, st_m1_lib)
        jax.block_until_ready(out_hid_m1_lib)
        if T == 1:
            st_m1_lib = out_st_m1_lib
        hid_m1_lib_np = np.asarray(jax.device_get(kernel.unfold(out_hid_m1_lib, 8)))
        del out_st_m1_lib
        m1_lib_stats, st_m1_lib = _time_step_fn(
            step_m1_lib, pos_m1, hid_m1, w_m1_lib, st_m1_lib, T=T
        )
        del st_m1_lib, out_hid_m1_lib, step_m1_lib
        gc.collect()

        pos_m1_f, hid_m1_f, st_2_f = _init_bucket_inputs_on_tpu(mesh, g, ex_m1)
        kc_60_f, vc_60_f, conv_0_f, rec_0_f = jax.jit(
            jax.shard_map(
                _tile_kv,
                mesh=mesh,
                in_specs=(P("tp"), P("tp")),
                out_specs=(P("tp"), P("tp"), P("tp"), P("tp")),
                check_vma=False,
            )
        )(st_2_f["kc"], st_2_f["vc"])
        del st_2_f
        st_m1_f = {
            "conv": conv_0_f,
            "rec": rec_0_f,
            "kc": kc_60_f,
            "vc": vc_60_f,
        }
        raw_step_m1_f, info_m1_f = frozen_mk.make_step(mesh, c_f_m1, tiles_f_m1)
        step_m1_f = jax.jit(raw_step_m1_f, donate_argnums=(3,) if T == 1 else ())
        out_hid_m1_f, out_st_m1_f = step_m1_f(pos_m1_f, hid_m1_f, w_m1_f, st_m1_f)
        jax.block_until_ready(out_hid_m1_f)
        if T == 1:
            st_m1_f = out_st_m1_f
        hid_m1_f_np = np.asarray(jax.device_get(kernel.unfold(out_hid_m1_f, 8)))
        del out_st_m1_f
        m1_f_stats, st_m1_f = _time_step_fn(
            step_m1_f, pos_m1_f, hid_m1_f, w_m1_f, st_m1_f, T=T
        )
        del (
            pos_m1,
            hid_m1,
            pos_m1_f,
            hid_m1_f,
            st_m1_f,
            out_hid_m1_f,
            step_m1_f,
            raw_step_m1_f,
        )
        gc.collect()

        m1_bit_exact = bool(
            np.array_equal(hid_m1_lib_np[:, :B, :], hid_m1_f_np[:, :B, :])
        )
        m1_delta = (
            (m1_lib_stats["median_us"] - m1_f_stats["median_us"])
            / m1_f_stats["median_us"]
            * 100.0
        )
        m1_row = {
            "T": T,
            "B": B,
            "NS": B // T,
            "lib_median_us": round(m1_lib_stats["median_us"], 1),
            "frozen_median_us": round(m1_f_stats["median_us"], 1),
            "delta_pct": round(m1_delta, 2),
            "lib_mean_us": round(m1_lib_stats["mean_us"], 1),
            "frozen_mean_us": round(m1_f_stats["mean_us"], 1),
            "vmem_lib_mib": round(info_m1_lib["vmem_scratch_bytes"] / (1024 * 1024), 2),
            "vmem_frozen_mib": round(
                info_m1_f["vmem_scratch_bytes"] / (1024 * 1024), 2
            ),
            "hid_bit_exact": m1_bit_exact,
        }
        m1_results.append(m1_row)
        print(
            f"[M1 (att_every=1) Bucket T={T} B={B:2d}] "
            f"lib={m1_row['lib_median_us']:6.1f} us vs "
            f"frozen={m1_row['frozen_median_us']:6.1f} us "
            f"({m1_row['delta_pct']:+.2f}%) | "
            f"VMEM {m1_row['vmem_lib_mib']} vs {m1_row['vmem_frozen_mib']} MiB | "
            f"bit_exact={m1_bit_exact}",
            flush=True,
        )
        assert m1_bit_exact, f"M1 B={B}, T={T} not bit-exact"

    print("\n=== M1_B64_TPU_V7X8_BENCH_JSON_START ===")
    print(json.dumps(m1_results, indent=2))
    print("=== M1_B64_TPU_V7X8_BENCH_JSON_END ===", flush=True)


def _init_pooled_bucket_on_tpu(
    mesh: Mesh,
    g: qwen35.Qwen35Geometry,
    ex: ExecSpec,
) -> tuple[
    jax.Array,
    jax.Array,
    tuple[jax.Array, ...],
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Allocate `(pos, hid, raw_pools, page_table, ckpt_indices, read_offsets)`."""
    rep = NamedSharding(mesh, P())
    R, B, T, M, NS, ctx = ex.R, ex.B, ex.T, ex.M, ex.NS, ex.ctx
    G = g.G(R)
    H = g.H
    G_PER_ATT = max(1, g.att_every - 1)
    pages_per_seq = ctx // mk.cache_exec(ex).page_size
    kv_pages_total = NS * pages_per_seq
    n_ckpt_slots = G_PER_ATT * NS * (T + 1)
    base_slot = (kv_pages_total + 4) // 5
    base_page = base_slot * 5
    gdn_pages = n_ckpt_slots * 5
    min_num_pages = base_page + gdn_pages
    assert mk.cache_exec(ex).num_pages >= min_num_pages, (
        mk.cache_exec(ex).num_pages,
        min_num_pages,
    )

    pt_cols = max(mk.pool.resolved_max_pages_per_seq(ex), pages_per_seq)
    pt_np = np.zeros((NS, pt_cols), dtype=np.int32)
    for s in range(NS):
        pt_np[s, :pages_per_seq] = np.arange(
            s * pages_per_seq, (s + 1) * pages_per_seq, dtype=np.int32
        )

    ckpt_np = np.zeros((G_PER_ATT, NS, T + 1), dtype=np.int32)
    for sub in range(G_PER_ATT):
        for s in range(NS):
            for k in range(T + 1):
                ckpt_np[sub, s, k] = base_slot + (sub * NS + s) * (T + 1) + k
    roff_np = np.zeros((G_PER_ATT, NS), dtype=np.int32)

    pt = jax.device_put(jnp.asarray(pt_np), rep)
    ckpt = jax.device_put(jnp.asarray(ckpt_np), rep)
    roff = jax.device_put(jnp.asarray(roff_np), rep)

    base_p = 65536 if ctx > 4096 else 2048
    base_pos = jnp.int32(base_p) + jnp.arange(NS, dtype=I32) * jnp.int32(16)
    if T == 1:
        pos_host = base_pos
    else:
        pos_host = (base_pos[:, None] + jnp.arange(T, dtype=I32)[None, :]).reshape(B)
    pos = jax.device_put(pos_host, rep)

    raw_spec = tuple(P("tp") for _ in range(g.NATT))

    def _local_pool_init():
        me = lax.axis_index("tp")
        kv_grp = lax.div(me, G)
        root = jax.random.PRNGKey(88000 + T * 100 + B + (ctx // 1024))
        k_hid = jax.random.fold_in(root, 1)
        hid_b = jax.random.normal(k_hid, (B, H), dtype=BF16) * jnp.bfloat16(0.1)
        hid_m = jnp.pad(hid_b, ((0, M - B), (0, 0)))

        k_kv_base = jax.random.fold_in(root, 10 + kv_grp)
        k_gdn_base = jax.random.fold_in(root, 100 + me)
        kv_tile = jax.random.normal(
            k_kv_base,
            (1, 16, 2, 64, 4, mk.cache_exec(ex).page_size),
            dtype=BF16,
        ) * jnp.bfloat16(0.08)
        gdn_tile = jax.random.normal(
            k_gdn_base,
            (1, gdn_pages, 2, 64, 2, mk.cache_exec(ex).page_size),
            dtype=BF16,
        ) * jnp.bfloat16(0.02)

        pools = []
        assert mk.cache_exec(ex).num_pages % 16 == 0, mk.cache_exec(ex).num_pages
        n_rep = mk.cache_exec(ex).num_pages // 16
        for g_idx in range(g.NATT):
            scale = jnp.bfloat16(1.0 - 0.01 * g_idx)
            kv_p = jnp.tile(
                (kv_tile * scale).astype(FP8),
                (1, n_rep, 1, 1, 1, 1),
            )
            gdn_bf16 = gdn_tile * scale
            gdn_fp8 = (
                lax.bitcast_convert_type(gdn_bf16, FP8)
                .swapaxes(-2, -1)
                .reshape(1, gdn_pages, 2, 64, 4, mk.cache_exec(ex).page_size)
            )
            pool_g = lax.dynamic_update_slice(
                kv_p, gdn_fp8, (0, base_page, 0, 0, 0, 0)
            ).reshape(
                mk.cache_exec(ex).num_pages, 2, 64, 4, mk.cache_exec(ex).page_size
            )
            pools.append(pool_g)
        return hid_m, tuple(pools)

    init_fn = jax.jit(
        jax.shard_map(
            _local_pool_init,
            mesh=mesh,
            in_specs=(),
            out_specs=(P("tp"), raw_spec),
            check_vma=False,
        )
    )
    hid, raw_pools = init_fn()
    jax.block_until_ready(hid)
    return pos, hid, raw_pools, pt, ckpt, roff


def _time_pooled_step_fn(
    step_fn,
    pos: jax.Array,
    hid: jax.Array,
    weights: dict[str, Any],
    raw_pools: tuple[jax.Array, ...],
    pt: jax.Array,
    ckpt: jax.Array,
    roff: jax.Array,
    *,
    warmup: int = 15,
    batches: int = 10,
    steps_per_batch: int = 10,
) -> tuple[dict[str, float], tuple[jax.Array, ...]]:
    """Warm up and measure pooled step latency over `batches * steps_per_batch`."""
    for _ in range(warmup):
        out_hid, raw_pools = step_fn(pos, hid, weights, raw_pools, pt, ckpt, roff)
    jax.block_until_ready(out_hid)

    batch_us = []
    for _ in range(batches):
        t0 = time.perf_counter()
        for _ in range(steps_per_batch):
            out_hid, raw_pools = step_fn(pos, hid, weights, raw_pools, pt, ckpt, roff)
        jax.block_until_ready(out_hid)
        dt_us = (time.perf_counter() - t0) * 1e6 / steps_per_batch
        batch_us.append(dt_us)

    arr = np.asarray(batch_us, dtype=np.float64)
    stats = {
        "median_us": float(np.median(arr)),
        "mean_us": float(np.mean(arr)),
        "p10_us": float(np.percentile(arr, 10)),
        "p90_us": float(np.percentile(arr, 90)),
        "min_us": float(np.min(arr)),
    }
    return stats, raw_pools


def test_m3_tpu_v7x8_paged_eplb_tp_bench():
    """Run M3 TPU v7x-8 benchmark: PallasWriter, stack_e=False, EPLB, and moe_tp."""
    if not _has_tpu_v7x8():
        pytest.skip("Requires 8 TPU TensorCores (v7x-8)")
    tpu_devs = jax.devices("tpu")
    frozen_mk = _load_frozen_mk()
    frozen_pio = _load_frozen_pool_io(frozen_mk)
    mesh = Mesh(tpu_devs[:8], ("tp",))

    g = dataclasses.replace(qwen35.FULL_EP, L=60, att_every=4)
    ex_init = ExecSpec(B=4, T=1, R=8, ctx=4096, stack_e=True)
    c_f_init = frozen_mk.MKConfig(
        R=8,
        B=4,
        T=1,
        L=60,
        att_every=4,
        moe_tp=False,
        ctx=4096,
        stack_e=True,
    )
    f_lay, f_S = frozen_mk.smalls_layout(c_f_init)

    t_w0 = time.perf_counter()
    w_lib, w_f = _init_shared_weights_on_tpu(mesh, g, ex_init, f_lay, f_S)
    dt_w = time.perf_counter() - t_w0
    print(
        f"\n[M3 TPU v7x-8 Bench] Allocated shared 60-layer hybrid FULL_EP "
        f"weights on 8 TPU cores in {dt_w:.2f}s",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # Part 1: PallasWriter (make_pooled_step) across (T, B) + ctx=128K bucket
    # -----------------------------------------------------------------------
    pooled_buckets = [
        (1, 4, 4096, 4096),
        (1, 16, 4096, 4096),
        (1, 32, 4096, 4096),
        (1, 64, 4096, 4096),
        (4, 4, 4096, 4096),
        (4, 16, 4096, 4096),
        (4, 64, 4096, 4096),
        (1, 4, 131072, 4608),  # 128K context (1,024 pages/seq, pos=65536)
    ]
    pooled_results = []

    for T, B, ctx, num_pages in pooled_buckets:
        ex = ExecSpec(
            B=B,
            T=T,
            R=8,
            ctx=ctx,
            stack_e=True,
            tiles=frozen_parity_tiles(B),
            cache=mk.CacheExec(
                paged_kv=True,
                num_pages=num_pages,
                pool_gather=False,
                fused_pool_io=False,
            ),
        )
        c_f = frozen_mk.MKConfig(
            R=8,
            B=B,
            T=T,
            L=60,
            att_every=4,
            moe_tp=False,
            ctx=ctx,
            paged_kv=True,
            num_pages=num_pages,
            max_pages_per_seq=mk.pool.resolved_max_pages_per_seq(ex),
            stack_e=True,
        )
        tiles_f = frozen_mk.default_tiles(c_f)
        # Override 1 for B > 48 on frozen baseline: use nbe=6 to match default_tiles(B)
        # so the 64-row MoE scratch fits within the 64 MiB VMEM budget.
        if B > 48:
            tiles_f = dataclasses.replace(tiles_f, nbe=6)

        # --- Library PallasWriter ---
        pos, hid_lib, raw_lib, pt, ckpt, roff = _init_pooled_bucket_on_tpu(mesh, g, ex)
        step_lib, info_lib = cache.PallasWriter().make_step(mesh, g, ex, donate=True)
        vmem_lib = info_lib["vmem_scratch_bytes"]
        smem_lib = info_lib["smem_scratch_bytes"]
        out_hid_lib, raw_lib = step_lib(pos, hid_lib, w_lib, raw_lib, pt, ckpt, roff)
        jax.block_until_ready(out_hid_lib)
        hid_lib_np = np.asarray(jax.device_get(kernel.unfold(out_hid_lib, 8)))
        # Snapshot updated KV pages and GDN checkpoint slots across all NS sequences
        # and attention pools in uint8 so raw FP8 bytes (including 0x7f/0xff in
        # bitcast F32/BF16 GDN states) are compared bit-for-bit.
        pos_host_np = np.asarray(jax.device_get(pos))
        pt_host_np = np.asarray(jax.device_get(pt))
        ckpt_host_np = np.asarray(jax.device_get(ckpt))
        kv_page_ids = jnp.asarray(
            [
                int(
                    pt_host_np[
                        b // T, int(pos_host_np[b]) // mk.cache_exec(ex).page_size
                    ]
                )
                for b in range(B)
            ],
            dtype=jnp.int32,
        )
        gdn_page_ids = jnp.asarray(
            [
                int(ckpt_host_np[sub, s, t_k]) * 5 + p_off
                for sub in range(3)
                for s in range(ex.NS)
                for t_k in range(1, 1 + T)
                for p_off in range(5)
            ],
            dtype=jnp.int32,
        )
        raw_lib_kv = [
            np.asarray(
                jax.device_get(
                    lax.bitcast_convert_type(
                        kernel.unfold(raw_lib[g_i], 8)[:, kv_page_ids], jnp.uint8
                    )
                )
            )
            for g_i in range(g.NATT)
        ]
        raw_lib_gdn = [
            np.asarray(
                jax.device_get(
                    lax.bitcast_convert_type(
                        kernel.unfold(raw_lib[g_i], 8)[:, gdn_page_ids], jnp.uint8
                    )
                )
            )
            for g_i in range(g.NATT)
        ]
        raw_lib_sha256 = [
            hashlib.sha256(
                np.asarray(
                    jax.device_get(lax.bitcast_convert_type(raw_lib[g_i], jnp.uint8))
                ).tobytes()
            ).hexdigest()
            for g_i in range(g.NATT)
        ]
        lib_stats, raw_lib = _time_pooled_step_fn(
            step_lib, pos, hid_lib, w_lib, raw_lib, pt, ckpt, roff
        )
        del pos, hid_lib, raw_lib, pt, ckpt, roff, out_hid_lib, step_lib
        gc.collect()

        # --- Frozen pool_io (fused_pool_io=False) ---
        # Override 2 for NS > 32 on frozen baseline: frozen pool_io.gather_gdn_states
        # omitted vmem_limit_bytes on pltpu.CompilerParams, so at NS=64 (T=1, B=64,
        # 37.12 MiB VMEM) it hits Pallas's 32 MiB default cap unless vmem_limit_bytes
        # is set to 62 MiB (matching scatter_pool_states in qwen35_pool.py).
        pos_f, hid_f, raw_f, pt_f, ckpt_f, roff_f = _init_pooled_bucket_on_tpu(
            mesh, g, ex
        )
        raw_step_f, info_f = frozen_pio.make_pooled_step(
            mesh, c_f, tiles_f, fused_pool_io=False
        )
        vmem_f = info_f["vmem_scratch_bytes"]
        step_f = jax.jit(
            lambda p_, h_, w_, r_, pt_, ck_, ro_, _fn=raw_step_f: _fn(
                p_, h_, w_, r_, pt_, ck_, ro_
            ),
            donate_argnums=(3,),
        )
        orig_cp = frozen_pio.pltpu.CompilerParams
        vmem_lim = min(tiles_f.vmem_mib, 62) << 20
        if (B // T) > 32:
            frozen_pio.pltpu.CompilerParams = (
                lambda _orig=orig_cp, _lim=vmem_lim, **kw: _orig(
                    **{
                        "vmem_limit_bytes": _lim,
                        **kw,
                    }
                )
            )
        try:
            out_hid_f, raw_f = step_f(pos_f, hid_f, w_f, raw_f, pt_f, ckpt_f, roff_f)
            jax.block_until_ready(out_hid_f)
        finally:
            frozen_pio.pltpu.CompilerParams = orig_cp
        hid_f_np = np.asarray(jax.device_get(kernel.unfold(out_hid_f, 8)))
        raw_f_kv = [
            np.asarray(
                jax.device_get(
                    lax.bitcast_convert_type(
                        kernel.unfold(raw_f[g_i], 8)[:, kv_page_ids], jnp.uint8
                    )
                )
            )
            for g_i in range(g.NATT)
        ]
        raw_f_gdn = [
            np.asarray(
                jax.device_get(
                    lax.bitcast_convert_type(
                        kernel.unfold(raw_f[g_i], 8)[:, gdn_page_ids], jnp.uint8
                    )
                )
            )
            for g_i in range(g.NATT)
        ]
        raw_f_sha256 = [
            hashlib.sha256(
                np.asarray(
                    jax.device_get(lax.bitcast_convert_type(raw_f[g_i], jnp.uint8))
                ).tobytes()
            ).hexdigest()
            for g_i in range(g.NATT)
        ]
        frozen_stats, raw_f = _time_pooled_step_fn(
            step_f, pos_f, hid_f, w_f, raw_f, pt_f, ckpt_f, roff_f
        )
        del (
            pos_f,
            hid_f,
            raw_f,
            pt_f,
            ckpt_f,
            roff_f,
            out_hid_f,
            step_f,
            raw_step_f,
        )
        gc.collect()

        live_lib = hid_lib_np[:, :B, :].astype(np.float32)
        live_f = hid_f_np[:, :B, :].astype(np.float32)
        assert np.all(np.isfinite(live_lib)), (
            f"Non-finite pooled output at T={T}, B={B}, ctx={ctx}"
        )
        hid_bit_exact = bool(
            np.array_equal(
                hid_lib_np[:, :B, :].view(np.uint8),
                hid_f_np[:, :B, :].view(np.uint8),
            )
        )
        pool_kv_bit_exact = all(
            np.array_equal(a, b) for a, b in zip(raw_lib_kv, raw_f_kv)
        )
        pool_gdn_bit_exact = all(
            np.array_equal(a, b) for a, b in zip(raw_lib_gdn, raw_f_gdn)
        )
        pool_full_sha256_match = raw_lib_sha256 == raw_f_sha256
        max_abs_diff = float(np.max(np.abs(live_lib - live_f)))
        delta_pct = (
            (lib_stats["median_us"] - frozen_stats["median_us"])
            / frozen_stats["median_us"]
            * 100.0
        )
        row = {
            "mode": "pallas_writer",
            "T": T,
            "B": B,
            "NS": B // T,
            "ctx": ctx,
            "max_pages_per_seq": mk.pool.resolved_max_pages_per_seq(ex),
            "lib_median_us": round(lib_stats["median_us"], 1),
            "frozen_median_us": round(frozen_stats["median_us"], 1),
            "delta_pct": round(delta_pct, 2),
            "lib_mean_us": round(lib_stats["mean_us"], 1),
            "frozen_mean_us": round(frozen_stats["mean_us"], 1),
            "vmem_lib_mib": round(vmem_lib / (1024 * 1024), 2),
            "vmem_frozen_mib": round(vmem_f / (1024 * 1024), 2),
            "smem_lib_kib": round(smem_lib / 1024.0, 2),
            "hid_bit_exact": hid_bit_exact,
            "pool_kv_bit_exact": pool_kv_bit_exact,
            "pool_gdn_bit_exact": pool_gdn_bit_exact,
            "pool_full_sha256_match": pool_full_sha256_match,
            "max_abs_diff": max_abs_diff,
        }
        pooled_results.append(row)
        pt_cols = mk.pool.resolved_max_pages_per_seq(ex)
        print(
            f"[M3 PallasWriter T={T} B={B:2d} ctx={ctx:6d} (pt_cols={pt_cols:4d})] "
            f"lib={row['lib_median_us']:6.1f} us vs "
            f"frozen={row['frozen_median_us']:6.1f} us ({row['delta_pct']:+.2f}%) | "
            f"VMEM {row['vmem_lib_mib']} MiB | SMEM {row['smem_lib_kib']} KiB | "
            f"bit_exact(hid={hid_bit_exact}, kv={pool_kv_bit_exact}, "
            f"gdn={pool_gdn_bit_exact}, sha={pool_full_sha256_match})",
            flush=True,
        )
        assert hid_bit_exact, (
            f"PallasWriter hid not bit-exact at T={T}, B={B}, ctx={ctx}"
        )
        assert pool_kv_bit_exact, (
            f"PallasWriter KV pool not bit-exact at T={T}, B={B}, ctx={ctx}"
        )
        assert pool_gdn_bit_exact, (
            f"PallasWriter GDN pool not bit-exact at T={T}, B={B}, ctx={ctx}"
        )
        assert pool_full_sha256_match, (
            f"PallasWriter full pool SHA-256 mismatch at T={T}, B={B}, ctx={ctx}"
        )
        assert row["delta_pct"] <= 2.0, (
            f"PallasWriter latency regression {row['delta_pct']:+.2f}% > +2.0%"
        )

    # Also compile & verify HLO for (64, 1024) page_table on TPU (B=64, T=1, ctx=128K)
    ex_128k_b64 = ExecSpec(
        B=64,
        T=1,
        R=8,
        ctx=131072,
        stack_e=True,
        tiles=frozen_parity_tiles(64),
        cache=mk.CacheExec(
            paged_kv=True,
            num_pages=4608,
            max_pages_per_seq=1024,
            pool_gather=False,
            fused_pool_io=False,
        ),
    )
    step_128k_b64, info_128k_b64 = cache.PallasWriter().make_step(
        mesh, g, ex_128k_b64, donate=True
    )
    assert info_128k_b64["smem_shapes"]["page_table"] == ((64, 1024), "int32")
    assert info_128k_b64["smem_scratch_bytes"] <= 1024 * 1024
    rep_s = NamedSharding(mesh, P())
    tp_s = NamedSharding(mesh, P("tp"))
    lowered_128k_b64 = step_128k_b64.lower(
        jax.ShapeDtypeStruct((64,), jnp.int32, sharding=rep_s),
        jax.ShapeDtypeStruct((8 * 64, g.H), BF16, sharding=tp_s),
        w_lib,
        tuple(
            jax.ShapeDtypeStruct((8 * 4608, 2, 64, 4, 128), FP8, sharding=tp_s)
            for _ in range(g.NATT)
        ),
        jax.ShapeDtypeStruct((64, 1024), jnp.int32, sharding=rep_s),
        jax.ShapeDtypeStruct((3, 64, 2), jnp.int32, sharding=rep_s),
        jax.ShapeDtypeStruct((3, 64), jnp.int32, sharding=rep_s),
    )
    compiled_128k_b64 = lowered_128k_b64.compile()
    ok_128k, msg_128k = cache.check_pooled_step_hlo(
        compiled_128k_b64.as_text(), page_size=128
    )
    assert ok_128k, msg_128k
    del step_128k_b64, lowered_128k_b64, compiled_128k_b64
    gc.collect()

    # -----------------------------------------------------------------------
    # Part 2: stack_e=False (per-layer expert views) at (1, 16) and (4, 16)
    # -----------------------------------------------------------------------
    unstacked_results = []
    e13_unstacked = tuple(w_lib["e13"][:, lyr] for lyr in range(g.L))
    del w_lib["e13"], w_f["e13"]
    gc.collect()
    e2_unstacked = tuple(w_lib["e2"][:, lyr] for lyr in range(g.L))
    del w_lib["e2"], w_f["e2"]
    gc.collect()
    w_lib_unstacked = {
        **w_lib,
        "e13": e13_unstacked,
        "e2": e2_unstacked,
    }
    w_f_unstacked = {
        **w_f,
        "e13": e13_unstacked,
        "e2": e2_unstacked,
    }

    # Verify & time PallasWriter with stack_e=False on TPU at T=1, B=16 vs frozen
    ex_pw_u = ExecSpec(
        B=16,
        T=1,
        R=8,
        ctx=4096,
        stack_e=False,
        tiles=frozen_parity_tiles(16),
        cache=mk.CacheExec(
            paged_kv=True, num_pages=4096, pool_gather=False, fused_pool_io=False
        ),
    )
    c_f_pw_u = frozen_mk.MKConfig(
        R=8,
        B=16,
        T=1,
        L=60,
        att_every=4,
        moe_tp=False,
        ctx=4096,
        paged_kv=True,
        num_pages=4096,
        max_pages_per_seq=mk.pool.resolved_max_pages_per_seq(ex_pw_u),
        stack_e=False,
    )
    tiles_f_pw_u = frozen_mk.default_tiles(c_f_pw_u)

    pos_pu, hid_pu, raw_pu, pt_pu, ckpt_pu, roff_pu = _init_pooled_bucket_on_tpu(
        mesh, g, ex_pw_u
    )
    pw_u_step, info_pw_u = cache.PallasWriter().make_step(mesh, g, ex_pw_u, donate=True)
    out_pu, raw_pu = pw_u_step(
        pos_pu, hid_pu, w_lib_unstacked, raw_pu, pt_pu, ckpt_pu, roff_pu
    )
    jax.block_until_ready(out_pu)
    hid_pu_np = np.asarray(jax.device_get(kernel.unfold(out_pu, 8)))
    pos_pu_np = np.asarray(jax.device_get(pos_pu))
    pt_pu_np = np.asarray(jax.device_get(pt_pu))
    ckpt_pu_np = np.asarray(jax.device_get(ckpt_pu))
    pu_kv_ids = jnp.asarray(
        [
            int(pt_pu_np[b, int(pos_pu_np[b]) // mk.cache_exec(ex_pw_u).page_size])
            for b in range(16)
        ],
        dtype=jnp.int32,
    )
    pu_gdn_ids = jnp.asarray(
        [
            int(ckpt_pu_np[sub, s, 1]) * 5 + p_off
            for sub in range(3)
            for s in range(16)
            for p_off in range(5)
        ],
        dtype=jnp.int32,
    )
    raw_pu_kv = [
        np.asarray(
            jax.device_get(
                lax.bitcast_convert_type(
                    kernel.unfold(raw_pu[g_i], 8)[:, pu_kv_ids], jnp.uint8
                )
            )
        )
        for g_i in range(g.NATT)
    ]
    raw_pu_gdn = [
        np.asarray(
            jax.device_get(
                lax.bitcast_convert_type(
                    kernel.unfold(raw_pu[g_i], 8)[:, pu_gdn_ids], jnp.uint8
                )
            )
        )
        for g_i in range(g.NATT)
    ]
    raw_pu_sha256 = [
        hashlib.sha256(
            np.asarray(
                jax.device_get(lax.bitcast_convert_type(raw_pu[g_i], jnp.uint8))
            ).tobytes()
        ).hexdigest()
        for g_i in range(g.NATT)
    ]
    pw_u_stats, raw_pu = _time_pooled_step_fn(
        pw_u_step, pos_pu, hid_pu, w_lib_unstacked, raw_pu, pt_pu, ckpt_pu, roff_pu
    )
    del pos_pu, hid_pu, raw_pu, pt_pu, ckpt_pu, roff_pu, out_pu, pw_u_step
    gc.collect()

    pos_puf, hid_puf, raw_puf, pt_puf, ckpt_puf, roff_puf = _init_pooled_bucket_on_tpu(
        mesh, g, ex_pw_u
    )
    raw_step_puf, info_puf = frozen_pio.make_pooled_step(
        mesh, c_f_pw_u, tiles_f_pw_u, fused_pool_io=False
    )
    step_puf = jax.jit(
        lambda p_, h_, w_, r_, pt_, ck_, ro_, _fn=raw_step_puf: _fn(
            p_, h_, w_, r_, pt_, ck_, ro_
        ),
        donate_argnums=(3,),
    )
    out_puf, raw_puf = step_puf(
        pos_puf, hid_puf, w_f_unstacked, raw_puf, pt_puf, ckpt_puf, roff_puf
    )
    jax.block_until_ready(out_puf)
    hid_puf_np = np.asarray(jax.device_get(kernel.unfold(out_puf, 8)))
    raw_puf_kv = [
        np.asarray(
            jax.device_get(
                lax.bitcast_convert_type(
                    kernel.unfold(raw_puf[g_i], 8)[:, pu_kv_ids], jnp.uint8
                )
            )
        )
        for g_i in range(g.NATT)
    ]
    raw_puf_gdn = [
        np.asarray(
            jax.device_get(
                lax.bitcast_convert_type(
                    kernel.unfold(raw_puf[g_i], 8)[:, pu_gdn_ids], jnp.uint8
                )
            )
        )
        for g_i in range(g.NATT)
    ]
    raw_puf_sha256 = [
        hashlib.sha256(
            np.asarray(
                jax.device_get(lax.bitcast_convert_type(raw_puf[g_i], jnp.uint8))
            ).tobytes()
        ).hexdigest()
        for g_i in range(g.NATT)
    ]
    puf_stats, raw_puf = _time_pooled_step_fn(
        step_puf, pos_puf, hid_puf, w_f_unstacked, raw_puf, pt_puf, ckpt_puf, roff_puf
    )
    del (
        pos_puf,
        hid_puf,
        raw_puf,
        pt_puf,
        ckpt_puf,
        roff_puf,
        out_puf,
        step_puf,
        raw_step_puf,
    )
    gc.collect()

    pu_hid_exact = bool(
        np.array_equal(
            hid_pu_np[:, :16, :].view(np.uint8),
            hid_puf_np[:, :16, :].view(np.uint8),
        )
    )
    pu_kv_exact = all(np.array_equal(a, b) for a, b in zip(raw_pu_kv, raw_puf_kv))
    pu_gdn_exact = all(np.array_equal(a, b) for a, b in zip(raw_pu_gdn, raw_puf_gdn))
    pu_sha256_exact = raw_pu_sha256 == raw_puf_sha256
    pu_delta = (
        (pw_u_stats["median_us"] - puf_stats["median_us"])
        / puf_stats["median_us"]
        * 100.0
    )
    pu_row = {
        "mode": "pallas_writer_stack_e_false",
        "T": 1,
        "B": 16,
        "lib_median_us": round(pw_u_stats["median_us"], 1),
        "frozen_median_us": round(puf_stats["median_us"], 1),
        "delta_pct": round(pu_delta, 2),
        "vmem_lib_mib": round(info_pw_u["vmem_scratch_bytes"] / (1024 * 1024), 2),
        "vmem_frozen_mib": round(info_puf["vmem_scratch_bytes"] / (1024 * 1024), 2),
        "hid_bit_exact": pu_hid_exact,
        "pool_kv_bit_exact": pu_kv_exact,
        "pool_gdn_bit_exact": pu_gdn_exact,
        "pool_full_sha256_match": pu_sha256_exact,
    }
    unstacked_results.append(pu_row)
    print(
        f"[M3 PallasWriter+stack_e=False T=1 B=16] "
        f"lib={pu_row['lib_median_us']:6.1f} us vs "
        f"frozen={pu_row['frozen_median_us']:6.1f} us "
        f"({pu_row['delta_pct']:+.2f}%) | "
        f"bit_exact(hid={pu_hid_exact}, kv={pu_kv_exact}, "
        f"gdn={pu_gdn_exact}, sha={pu_sha256_exact})",
        flush=True,
    )
    assert pu_hid_exact and pu_kv_exact and pu_gdn_exact and pu_sha256_exact
    assert pu_row["delta_pct"] <= 2.0

    for T, B in [(1, 16), (4, 16)]:
        ex_u = ExecSpec(
            B=B,
            T=T,
            R=8,
            ctx=4096,
            stack_e=False,
            tiles=frozen_parity_tiles(B),
        )
        c_f_u = frozen_mk.MKConfig(
            R=8,
            B=B,
            T=T,
            L=60,
            att_every=4,
            moe_tp=False,
            ctx=4096,
            stack_e=False,
        )
        tiles_f_u = frozen_mk.default_tiles(c_f_u)
        donate = T == 1

        pos_u, hid_u, st_u = _init_bucket_inputs_on_tpu(mesh, g, ex_u)
        step_u_lib, info_u_lib = qwen35.make_step(mesh, g, ex_u, donate_states=donate)
        out_hid_u_lib, st_u_out = step_u_lib(pos_u, hid_u, w_lib_unstacked, st_u)
        jax.block_until_ready(out_hid_u_lib)
        if donate:
            st_u = st_u_out
        hid_u_lib_np = np.asarray(jax.device_get(kernel.unfold(out_hid_u_lib, 8)))
        del st_u_out
        u_lib_stats, st_u = _time_step_fn(
            step_u_lib, pos_u, hid_u, w_lib_unstacked, st_u, T=T
        )
        del pos_u, hid_u, st_u, out_hid_u_lib, step_u_lib
        gc.collect()

        pos_uf, hid_uf, st_uf = _init_bucket_inputs_on_tpu(mesh, g, ex_u)
        raw_step_uf, info_uf = frozen_mk.make_step(mesh, c_f_u, tiles_f_u)
        step_uf = jax.jit(raw_step_uf, donate_argnums=(3,) if donate else ())
        out_hid_uf, st_uf_out = step_uf(pos_uf, hid_uf, w_f_unstacked, st_uf)
        jax.block_until_ready(out_hid_uf)
        if donate:
            st_uf = st_uf_out
        hid_uf_np = np.asarray(jax.device_get(kernel.unfold(out_hid_uf, 8)))
        del st_uf_out
        u_f_stats, st_uf = _time_step_fn(
            step_uf, pos_uf, hid_uf, w_f_unstacked, st_uf, T=T
        )
        del pos_uf, hid_uf, st_uf, out_hid_uf, step_uf, raw_step_uf
        gc.collect()

        u_bit_exact = bool(
            np.array_equal(
                hid_u_lib_np[:, :B, :].view(np.uint8),
                hid_uf_np[:, :B, :].view(np.uint8),
            )
        )
        u_delta = (
            (u_lib_stats["median_us"] - u_f_stats["median_us"])
            / u_f_stats["median_us"]
            * 100.0
        )
        u_row = {
            "mode": "stack_e_false",
            "T": T,
            "B": B,
            "lib_median_us": round(u_lib_stats["median_us"], 1),
            "frozen_median_us": round(u_f_stats["median_us"], 1),
            "delta_pct": round(u_delta, 2),
            "vmem_lib_mib": round(info_u_lib["vmem_scratch_bytes"] / (1024 * 1024), 2),
            "vmem_frozen_mib": round(info_uf["vmem_scratch_bytes"] / (1024 * 1024), 2),
            "hid_bit_exact": u_bit_exact,
        }
        unstacked_results.append(u_row)
        print(
            f"[M3 stack_e=False T={T} B={B:2d}] "
            f"lib={u_row['lib_median_us']:6.1f} us vs "
            f"frozen={u_row['frozen_median_us']:6.1f} us "
            f"({u_row['delta_pct']:+.2f}%) | bit_exact={u_bit_exact}",
            flush=True,
        )
        assert u_bit_exact, f"stack_e=False not bit-exact at T={T}, B={B}"
        assert u_row["delta_pct"] <= 2.0, (
            f"stack_e=False latency regression {u_row['delta_pct']:+.2f}% > +2.0%"
        )

    del e13_unstacked, e2_unstacked, w_lib_unstacked, w_f_unstacked
    del w_lib["es"], w_f["es"]
    gc.collect()
    jax.clear_caches()

    # -----------------------------------------------------------------------
    # Part 3: EPLB (e_slots=72, ecopy_opt=0 & ecopy_opt=1 Po2C) at (1, 16), (4, 16)
    # -----------------------------------------------------------------------
    g_eplb = dataclasses.replace(g, e_slots=72)
    ecopy_host = qwen35.encode_eplb_ecopy(g_eplb, 8)
    p_base = planner.plan(qwen35.qwen35_spec(g), ExecSpec(B=16, T=1, R=8, ctx=4096))
    p_eplb = planner.plan(
        qwen35.qwen35_spec(g_eplb), ExecSpec(B=16, T=1, R=8, ctx=4096)
    )
    c_f_e0 = frozen_mk.MKConfig(
        R=8, B=16, T=1, L=60, att_every=4, moe_tp=False, e_slots=72, ctx=4096
    )
    f_lay_e, f_S_e = frozen_mk.smalls_layout(c_f_e0)

    sm_lib_host = np.asarray(jax.device_get(w_lib["smalls"]))
    sm_eplb_np = np.zeros((8, 60, 1, p_eplb.smalls_width), dtype=np.float32)
    for name, (o0, w0) in p_base.smalls_layout.items():
        oe, _ = p_eplb.smalls_layout[name]
        sm_eplb_np[:, :, 0, oe : oe + w0] = sm_lib_host[:, :, 0, o0 : o0 + w0]
    oe_ec, w_ec = p_eplb.smalls_layout["ecopy"]
    sm_eplb_np[:, :, 0, oe_ec : oe_ec + w_ec] = np.asarray(ecopy_host)[None, :, :]

    sm_fe_np = np.zeros((8, 60, 1, f_S_e), dtype=np.float32)
    for name, (oe, we) in p_eplb.smalls_layout.items():
        fo, _ = f_lay_e[name]
        sm_fe_np[:, :, 0, fo : fo + we] = sm_eplb_np[:, :, 0, oe : oe + we]

    sm_lib_eplb = jax.device_put(jnp.asarray(sm_eplb_np), NamedSharding(mesh, P("tp")))
    sm_f_eplb = jax.device_put(jnp.asarray(sm_fe_np), NamedSharding(mesh, P("tp")))

    def _init_eplb_experts():
        me = lax.axis_index("tp")
        root = jax.random.PRNGKey(20261014)
        k_e13_base = jax.random.fold_in(jax.random.fold_in(root, 7), me)
        k_e2_base = jax.random.fold_in(jax.random.fold_in(root, 8), me)
        k_es = jax.random.fold_in(jax.random.fold_in(root, 9), me)

        e13_tile = (
            jax.random.normal(k_e13_base, (1, 2, 72, 8, 512, 2048), dtype=BF16)
            * jnp.bfloat16(0.125)
        ).astype(FP8)
        e2_tile = (
            jax.random.normal(k_e2_base, (1, 2, 72, 4, 256, 4096), dtype=BF16)
            * jnp.bfloat16(0.125)
        ).astype(FP8)
        es = (
            jnp.abs(jax.random.normal(k_es, (1, 60, 72, 1, 6144), dtype=F32)) * 0.008
            + 0.004
        )
        return (
            jnp.tile(e13_tile, (1, 30, 1, 1, 1, 1)),
            jnp.tile(e2_tile, (1, 30, 1, 1, 1, 1)),
            es,
        )

    e13_72, e2_72, es_72 = jax.jit(
        jax.shard_map(
            _init_eplb_experts,
            mesh=mesh,
            in_specs=(),
            out_specs=(P("tp"), P("tp"), P("tp")),
            check_vma=False,
        )
    )()
    jax.block_until_ready(e13_72)

    w_lib_eplb = {
        **w_lib,
        "smalls": sm_lib_eplb,
        "e13": e13_72,
        "e2": e2_72,
        "es": es_72,
    }
    w_f_eplb = {
        **w_f,
        "smalls": sm_f_eplb,
        "e13": e13_72,
        "e2": e2_72,
        "es": es_72,
    }

    eplb_results = []
    for ecopy_opt in (0, 1):
        for T, B in [(1, 16), (4, 16)]:
            tiles_lib_e = frozen_parity_tiles(B).with_blocks(
                {"moe": {"ecopy_opt": ecopy_opt}}
            )
            ex_e = ExecSpec(B=B, T=T, R=8, ctx=4096, stack_e=True, tiles=tiles_lib_e)
            c_f_e = frozen_mk.MKConfig(
                R=8,
                B=B,
                T=T,
                L=60,
                att_every=4,
                moe_tp=False,
                e_slots=72,
                ctx=4096,
                stack_e=True,
            )
            tiles_f_e = dataclasses.replace(
                frozen_mk.default_tiles(c_f_e), ecopy_opt=ecopy_opt
            )
            donate = T == 1

            pos_e, hid_e, st_e = _init_bucket_inputs_on_tpu(mesh, g_eplb, ex_e)
            step_e_lib, info_e_lib = qwen35.make_step(
                mesh, g_eplb, ex_e, donate_states=donate
            )
            out_hid_e_lib, st_e_out = step_e_lib(pos_e, hid_e, w_lib_eplb, st_e)
            jax.block_until_ready(out_hid_e_lib)
            if donate:
                st_e = st_e_out
            hid_e_lib_np = np.asarray(jax.device_get(kernel.unfold(out_hid_e_lib, 8)))
            del st_e_out
            e_lib_stats, st_e = _time_step_fn(
                step_e_lib, pos_e, hid_e, w_lib_eplb, st_e, T=T
            )
            del pos_e, hid_e, st_e, out_hid_e_lib, step_e_lib
            gc.collect()

            pos_ef, hid_ef, st_ef = _init_bucket_inputs_on_tpu(mesh, g_eplb, ex_e)
            raw_step_ef, info_ef = frozen_mk.make_step(mesh, c_f_e, tiles_f_e)
            step_ef = jax.jit(raw_step_ef, donate_argnums=(3,) if donate else ())
            out_hid_ef, st_ef_out = step_ef(pos_ef, hid_ef, w_f_eplb, st_ef)
            jax.block_until_ready(out_hid_ef)
            if donate:
                st_ef = st_ef_out
            hid_ef_np = np.asarray(jax.device_get(kernel.unfold(out_hid_ef, 8)))
            del st_ef_out
            e_f_stats, st_ef = _time_step_fn(
                step_ef, pos_ef, hid_ef, w_f_eplb, st_ef, T=T
            )
            del pos_ef, hid_ef, st_ef, out_hid_ef, step_ef, raw_step_ef
            gc.collect()

            e_bit_exact = bool(
                np.array_equal(
                    hid_e_lib_np[:, :B, :].view(np.uint8),
                    hid_ef_np[:, :B, :].view(np.uint8),
                )
            )
            e_delta = (
                (e_lib_stats["median_us"] - e_f_stats["median_us"])
                / e_f_stats["median_us"]
                * 100.0
            )
            e_row = {
                "mode": f"eplb_e72_opt{ecopy_opt}",
                "e_slots": 72,
                "ecopy_opt": ecopy_opt,
                "T": T,
                "B": B,
                "lib_median_us": round(e_lib_stats["median_us"], 1),
                "frozen_median_us": round(e_f_stats["median_us"], 1),
                "delta_pct": round(e_delta, 2),
                "vmem_lib_mib": round(
                    info_e_lib["vmem_scratch_bytes"] / (1024 * 1024), 2
                ),
                "vmem_frozen_mib": round(
                    info_ef["vmem_scratch_bytes"] / (1024 * 1024), 2
                ),
                "hid_bit_exact": e_bit_exact,
            }
            eplb_results.append(e_row)
            print(
                f"[M3 EPLB (e_slots=72, ecopy_opt={ecopy_opt}) T={T} B={B:2d}] "
                f"lib={e_row['lib_median_us']:6.1f} us vs "
                f"frozen={e_row['frozen_median_us']:6.1f} us "
                f"({e_row['delta_pct']:+.2f}%) | bit_exact={e_bit_exact}",
                flush=True,
            )
            assert e_bit_exact, (
                f"EPLB ecopy_opt={ecopy_opt} not bit-exact at T={T}, B={B}"
            )
            assert e_row["delta_pct"] <= 2.0, (
                f"EPLB latency regression {e_row['delta_pct']:+.2f}% > +2.0%"
            )

    del w_lib_eplb, w_f_eplb, e13_72, e2_72, es_72, sm_lib_eplb, sm_f_eplb
    gc.collect()
    jax.clear_caches()

    # -----------------------------------------------------------------------
    # Part 4: TP MoE (moe_tp=True, FULL_TP) at (1, 16) and (4, 16)
    # -----------------------------------------------------------------------
    g_tp = dataclasses.replace(g, moe_tp=True, e_slots=0)

    def _init_tp_experts():
        me = lax.axis_index("tp")
        root = jax.random.PRNGKey(20261015)
        k_e13_base = jax.random.fold_in(jax.random.fold_in(root, 7), me)
        k_e2_base = jax.random.fold_in(jax.random.fold_in(root, 8), me)
        k_es = jax.random.fold_in(jax.random.fold_in(root, 9), me)

        e13_tile = (
            jax.random.normal(k_e13_base, (1, 2, 512, 1, 4096, 256), dtype=BF16)
            * jnp.bfloat16(0.125)
        ).astype(FP8)
        e2_tile = (
            jax.random.normal(k_e2_base, (1, 2, 512, 1, 128, 4096), dtype=BF16)
            * jnp.bfloat16(0.125)
        ).astype(FP8)
        es = (
            jnp.abs(jax.random.normal(k_es, (1, 60, 512, 1, 4352), dtype=F32)) * 0.008
            + 0.004
        )
        return (
            jnp.tile(e13_tile, (1, 30, 1, 1, 1, 1)),
            jnp.tile(e2_tile, (1, 30, 1, 1, 1, 1)),
            es,
        )

    e13_tp, e2_tp, es_tp = jax.jit(
        jax.shard_map(
            _init_tp_experts,
            mesh=mesh,
            in_specs=(),
            out_specs=(P("tp"), P("tp"), P("tp")),
            check_vma=False,
        )
    )()
    jax.block_until_ready(e13_tp)

    w_lib_tp = {**w_lib, "e13": e13_tp, "e2": e2_tp, "es": es_tp}
    w_f_tp = {**w_f, "e13": e13_tp, "e2": e2_tp, "es": es_tp}

    tp_results = []
    for T, B in [(1, 16), (4, 16)]:
        ex_tp = ExecSpec(
            B=B,
            T=T,
            R=8,
            ctx=4096,
            stack_e=True,
            tiles=frozen_parity_tiles(B, moe_tp=True),
        )
        c_f_tp = frozen_mk.MKConfig(
            R=8,
            B=B,
            T=T,
            L=60,
            att_every=4,
            moe_tp=True,
            ctx=4096,
            stack_e=True,
        )
        tiles_f_tp = frozen_mk.default_tiles(c_f_tp)
        donate = T == 1

        pos_t, hid_t, st_t = _init_bucket_inputs_on_tpu(mesh, g_tp, ex_tp)
        step_t_lib, info_t_lib = qwen35.make_step(
            mesh, g_tp, ex_tp, donate_states=donate
        )
        out_hid_t_lib, st_t_out = step_t_lib(pos_t, hid_t, w_lib_tp, st_t)
        jax.block_until_ready(out_hid_t_lib)
        if donate:
            st_t = st_t_out
        hid_t_lib_np = np.asarray(jax.device_get(kernel.unfold(out_hid_t_lib, 8)))
        del st_t_out
        t_lib_stats, st_t = _time_step_fn(step_t_lib, pos_t, hid_t, w_lib_tp, st_t, T=T)
        del pos_t, hid_t, st_t, out_hid_t_lib, step_t_lib
        gc.collect()

        pos_tf, hid_tf, st_tf = _init_bucket_inputs_on_tpu(mesh, g_tp, ex_tp)
        raw_step_tf, info_tf = frozen_mk.make_step(mesh, c_f_tp, tiles_f_tp)
        step_tf = jax.jit(raw_step_tf, donate_argnums=(3,) if donate else ())
        out_hid_tf, st_tf_out = step_tf(pos_tf, hid_tf, w_f_tp, st_tf)
        jax.block_until_ready(out_hid_tf)
        if donate:
            st_tf = st_tf_out
        hid_tf_np = np.asarray(jax.device_get(kernel.unfold(out_hid_tf, 8)))
        del st_tf_out
        t_f_stats, st_tf = _time_step_fn(step_tf, pos_tf, hid_tf, w_f_tp, st_tf, T=T)
        del pos_tf, hid_tf, st_tf, out_hid_tf, step_tf, raw_step_tf
        gc.collect()

        t_bit_exact = bool(
            np.array_equal(
                hid_t_lib_np[:, :B, :].view(np.uint8),
                hid_tf_np[:, :B, :].view(np.uint8),
            )
        )
        t_delta = (
            (t_lib_stats["median_us"] - t_f_stats["median_us"])
            / t_f_stats["median_us"]
            * 100.0
        )
        t_row = {
            "mode": "moe_tp_true",
            "T": T,
            "B": B,
            "lib_median_us": round(t_lib_stats["median_us"], 1),
            "frozen_median_us": round(t_f_stats["median_us"], 1),
            "delta_pct": round(t_delta, 2),
            "vmem_lib_mib": round(info_t_lib["vmem_scratch_bytes"] / (1024 * 1024), 2),
            "vmem_frozen_mib": round(info_tf["vmem_scratch_bytes"] / (1024 * 1024), 2),
            "hid_bit_exact": t_bit_exact,
        }
        tp_results.append(t_row)
        print(
            f"[M3 TP MoE (moe_tp=True) T={T} B={B:2d}] "
            f"lib={t_row['lib_median_us']:6.1f} us vs "
            f"frozen={t_row['frozen_median_us']:6.1f} us "
            f"({t_row['delta_pct']:+.2f}%) | bit_exact={t_bit_exact}",
            flush=True,
        )
        assert t_bit_exact, f"moe_tp=True not bit-exact at T={T}, B={B}"
        assert t_row["delta_pct"] <= 2.0, (
            f"moe_tp=True latency regression {t_row['delta_pct']:+.2f}% > +2.0%"
        )

    m3_summary = {
        "pallas_writer": pooled_results,
        "stack_e_false": unstacked_results,
        "eplb": eplb_results,
        "moe_tp": tp_results,
    }
    print("\n=== M3_TPU_V7X8_BENCH_JSON_START ===")
    print(json.dumps(m3_summary, indent=2))
    print("=== M3_TPU_V7X8_BENCH_JSON_END ===", flush=True)


def _check_pooled_hlo_or_fail(
    compiled_text: str, spec: Any, ex: ExecSpec, *, fused: bool, tag: str
) -> None:
    """Pool-donation HLO guard: kernel layout + no full-pool HBM/VMEM copies."""
    g = spec.config
    ok, msg = cache.check_pooled_step_hlo(
        compiled_text,
        page_size=mk.cache_exec(ex).page_size,
        kvh=max(1, int(g.NKV) // ex.R),
        hd=int(g.HD),
        min_pool_pages=16,
        fused_pool_io=fused,
        kernel_names=spec.cache_backend.kernel_names,
    )
    assert ok, f"HLO check failed for {tag}: {msg}"


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(
        np.dot(a.ravel(), b.ravel())
        / (np.linalg.norm(a.ravel()) * np.linalg.norm(b.ravel()) + 1e-12)
    )


def test_m5_tpu_v7x8_profile_ablation_bench():
    """M5 profile bench: `default` vs `qwen35-candidate30` (+ GPR / fused ablation).

    Per bucket, every variant is built through `Profile.exec_for` with donated
    pools, passes the pool-donation HLO guard (fused variants included: no
    separate gather/scatter and no full-pool copy), and is timed:
      - `default`: frozen-parity profile (split pool I/O).
      - `c30`: `qwen35-candidate30` as checked in (GPR auto, split).
      - `c30_gpr_off`: c30 with `pool_gather=False` (only where c30 used GPR).
      - `c30_fused`: c30 with experimental `fused_pool_io=True` at the buckets
        the `"auto"` policy would fuse (B <= 24 or B >= 56).
    Numerics: `default` is the frozen-parity reference; c30 variants are
    `approx` and are reported as cosine similarity vs `default`.
    """
    if not _has_tpu_v7x8():
        pytest.skip("Requires 8 TPU TensorCores (v7x-8)")
    tpu_devs = jax.devices("tpu")
    frozen_mk = _load_frozen_mk()
    mesh = Mesh(tpu_devs[:8], ("tp",))

    g = dataclasses.replace(qwen35.FULL_EP, L=60, att_every=4)
    spec = qwen35.qwen35_spec(g)
    ex_init = ExecSpec(B=4, T=1, R=8, ctx=4096, stack_e=True)
    c_f_init = frozen_mk.MKConfig(
        R=8,
        B=4,
        T=1,
        L=60,
        att_every=4,
        moe_tp=False,
        ctx=4096,
        stack_e=True,
    )
    f_lay, f_S = frozen_mk.smalls_layout(c_f_init)
    w_lib, w_f = _init_shared_weights_on_tpu(mesh, g, ex_init, f_lay, f_S)
    del w_f
    gc.collect()

    p_def = get_profile("qwen35-baseline")
    p_c30 = get_profile("qwen35-candidate30")
    base = ExecSpec(
        R=8, ctx=4096, stack_e=True, cache=mk.CacheExec(paged_kv=True, num_pages=4096)
    )
    buckets = [(1, 4), (1, 16), (1, 32), (1, 64), (4, 4), (4, 16), (4, 64)]
    rows = []

    def _run(ex: ExecSpec, fused: bool, tag: str):
        pos, hid, raw, pt, ckpt, roff = _init_pooled_bucket_on_tpu(mesh, g, ex)
        step, info = cache.make_pooled_step(
            mesh, spec, ex, donate_pools=True, fused_pool_io=fused
        )
        comp = step.lower(pos, hid, w_lib, raw, pt, ckpt, roff).compile()
        _check_pooled_hlo_or_fail(comp.as_text(), spec, ex, fused=fused, tag=tag)
        del comp
        out, raw = step(pos, hid, w_lib, raw, pt, ckpt, roff)
        jax.block_until_ready(out)
        hid_np = np.asarray(jax.device_get(kernel.unfold(out, 8)))[:, : ex.B, :].astype(
            np.float32
        )
        assert np.all(np.isfinite(hid_np)), tag
        stats, raw = _time_pooled_step_fn(step, pos, hid, w_lib, raw, pt, ckpt, roff)
        gpr = bool(info["plan"].block_params["gdn"].GPR)
        del pos, hid, raw, pt, ckpt, roff, out, step
        gc.collect()
        return hid_np, stats, gpr, info["vmem_scratch_bytes"]

    def _record(name, ex, fused, row, hid_ref, med_ref, tag) -> bool:
        hid_v, st_v, gpr_v, vmem_v = _run(ex, fused, f"{name} {tag}")
        med = st_v["median_us"]
        row[f"{name}_median_us"] = round(med, 1)
        row[f"{name}_delta_pct"] = round((med - med_ref) / med_ref * 100.0, 2)
        row[f"{name}_cos"] = round(_cos(hid_ref, hid_v), 6)
        row[f"{name}_vmem_mib"] = round(vmem_v / (1 << 20), 2)
        return gpr_v

    for T, B in buckets:
        tag = f"T={T},B={B}"
        ex_def = p_def.exec_for(base, B=B, T=T, g=g)
        hid_def, st_def, _, _ = _run(ex_def, False, f"default {tag}")
        med_def = st_def["median_us"]
        row: dict[str, Any] = {"T": T, "B": B, "default_median_us": round(med_def, 1)}
        ref = (row, hid_def, med_def, tag)
        ex_c30 = p_c30.exec_for(base, B=B, T=T, g=g)
        row["c30_gpr"] = _record("c30", ex_c30, False, *ref)
        if row["c30_gpr"]:
            ex_off = mk.with_cache(ex_c30, pool_gather=False)
            _record("c30_gpr_off", ex_off, False, *ref)
        if should_fuse_pool_io(spec, ex_c30, fused_pool_io="auto"):
            _record("c30_fused", ex_c30, True, *ref)
        rows.append(row)
        print(f"[M5 profile bench {tag}] {json.dumps(row)}", flush=True)

    print("\n=== M5_TPU_V7X8_BENCH_JSON_START ===")
    print(json.dumps({"m5_profile_bench": rows}, indent=2))
    print("=== M5_TPU_V7X8_BENCH_JSON_END ===", flush=True)


# Block owning each bisect flag (`Tiles.blocks`).
_BISECT_FLAG_BLOCK = {"gqa_qv": "gqa", "moe_unroll": "moe"}


@pytest.mark.parametrize(
    "variant", ["default", "gqa_qv", "moe_unroll", "gqa_qv+moe_unroll"]
)
def test_m5_tpu_v7x8_frozen_parity_bisect(variant: str, monkeypatch):
    """Re-run the M2 frozen-parity bench with single M5 flags enabled.

    `default` must be bit-exact vs frozen; the other variants report which
    always-on-at-M5 code paths change TPU numerics (now opt-in flags).
    """
    if not _has_tpu_v7x8():
        pytest.skip("Requires 8 TPU TensorCores (v7x-8)")
    flags = {} if variant == "default" else dict.fromkeys(variant.split("+"), True)
    monkeypatch.setitem(
        globals(),
        "frozen_parity_tiles",
        lambda B, **kw: baseline_tiles(B, **kw).with_blocks(
            {_BISECT_FLAG_BLOCK[f]: {f: v} for f, v in flags.items()}
        ),
    )
    print(f"\n=== M5_BISECT variant={variant} flags={flags}", flush=True)
    try:
        test_m2_tpu_v7x8_full_ep_parity_and_latency()
        print(f"=== M5_BISECT variant={variant} BIT_EXACT", flush=True)
    except AssertionError as exc:
        print(f"=== M5_BISECT variant={variant} NOT_EXACT: {exc}", flush=True)
        if variant == "default":
            raise


# Fused vs split pool I/O at T=4 (verify), TPU bitwise. These configs include
# every case that deviates on CPU interpret (bounded XFAILs in
# test_m3_cache_eplb.py); on TPU hidden output and every pool page must match
# bytewise. First measured in job j-42e743ba-11c1-45c5-bec8 (libtpu 0.0.50).
_FUSED_T4_TPU_CASES = [
    # (tiles, B, seed): "c30f" = qwen35-candidate30-fused exec_for tiles,
    # "test" = the CPU fused test's tiles (head-major GDN, conv_stage off).
    ("c30f", 8, 0),
    ("c30f", 8, 2),
    ("c30f", 8, 4),
    ("c30f", 24, 0),
    ("c30f", 24, 1),
    ("c30f", 24, 2),
    ("test", 8, 0),
    ("test", 8, 2),
    ("test", 8, 4),
]


@pytest.mark.parametrize("tiles,B,seed", _FUSED_T4_TPU_CASES)
def test_fused_vs_split_t4_tpu_bitwise(tiles: str, B: int, seed: int):
    """Fused pool I/O is bitwise equal to split on TPU (hidden + every pool page)."""
    if not _has_tpu_v7x8():
        pytest.skip("Requires 8 TPU TensorCores (v7x-8)")
    from test_m3_cache_eplb import (
        _pool_diff_summary,
        _pool_page_kinds,
        _tiny_exec,
        _tiny_geom,
    )

    T = 4
    mesh = Mesh(np.array(jax.devices()[:2]), ("tp",))
    tp = NamedSharding(mesh, P("tp"))
    g = _tiny_geom(L=4, att_every=4)
    spec = qwen35.qwen35_spec(g)
    num_pages = 512 if B <= 8 else 2048
    if tiles == "test":
        ex0 = _tiny_exec(
            R=2, B=B, T=T, ctx=256, paged_kv=True, num_pages=num_pages, stack_e=False
        )
        ex = mk.with_cache(
            dataclasses.replace(
                ex0,
                tiles=ex0.resolved_tiles.with_blocks(
                    {"gdn": {"conv_stage": False, "gdn_head_major": True}}
                ),
                interpret=False,
            ),
            pool_gather=False,
        )
        key = 710 + T * 10 + B + 1000 * seed
    else:
        prof = get_profile("qwen35-candidate30-fused")
        base = ExecSpec(
            R=2,
            B=B,
            T=T,
            ctx=256,
            cache=mk.CacheExec(paged_kv=True, num_pages=num_pages),
        )
        ex = prof.exec_for(base, B=B, T=T, g=g)
        t = ex.resolved_tiles
        t = dataclasses.replace(
            t,
            bank=512 << 10,
            nbd=10,
            nbe=max(6, min(t.nbe, 12)),
            ebank=256 << 10,
            vmem_mib=32,
        ).with_blocks(
            {"gqa": {"kv_blk": 256, "kv_sub": 128 if t.block("gqa").kv_sub else 0}}
        )
        ex = mk.with_cache(
            dataclasses.replace(ex, tiles=t, interpret=False), pool_gather=False
        )
        key = 810 + B + 1000 * seed
    assert should_fuse_pool_io(spec, ex, fused_pool_io=True)
    pos, hid_l, w_l, st_l = qwen35.random_logical_inputs(jax.random.key(key), g, ex)
    pos_list = []
    for s_i in range(ex.NS):
        base_p = 126 if s_i == 0 else (20 + 10 * s_i)
        pos_list.extend(base_p + t_i for t_i in range(T))
    pos = jnp.asarray(pos_list, dtype=jnp.int32)
    st_l = dict(st_l, conv=st_l["conv"].astype(jnp.bfloat16).astype(jnp.float32))
    _, hid_mk, w_mk, _ = qwen35.pack_inputs(
        pos, hid_l, w_l, st_l, g, ex, mesh=mesh, build_default_pool=True
    )
    pt, ckpt, roff, _ = mk.default_pool_layout(g, ex)

    def _fresh_pools():
        return tuple(
            jax.device_put(np.asarray(r).copy(), tp)
            for r in mk.states_to_pool(g, ex, st_l, pt, ckpt, roff)
        )

    step_s, _ = mk.PallasWriter(fused_pool_io=False).make_step(
        mesh, spec, ex, donate=False
    )
    step_f, _ = mk.PallasWriter(fused_pool_io=True).make_step(
        mesh, spec, ex, donate=True
    )
    pools_in = [np.asarray(r).view(np.uint8) for r in _fresh_pools()]
    out_s, raw_s = step_s(pos, hid_mk, w_mk, _fresh_pools(), pt, ckpt, roff)
    out_f, raw_f = step_f(pos, hid_mk, w_mk, _fresh_pools(), pt, ckpt, roff)
    hs = np.asarray(out_s).view(np.uint8)
    hf = np.asarray(out_f).view(np.uint8)
    # Non-trivial: the split step wrote the pools and produced a finite output.
    assert np.isfinite(np.asarray(out_s, dtype=np.float32)).all()
    assert any(
        (np.asarray(r).view(np.uint8) != p).any() for r, p in zip(raw_s, pools_in)
    )
    kinds = _pool_page_kinds(pt, ckpt, g_per_att=g.att_every - 1)
    summary = _pool_diff_summary(raw_f, raw_s, mk.cache_exec(ex).num_pages, kinds)
    print(
        f"[FUSED_T4_TPU tiles={tiles} B={B} seed={seed}] "
        f"hidden_equal={bool((hs == hf).all())} pool_diff={summary}",
        flush=True,
    )
    np.testing.assert_array_equal(hf, hs)
    assert not summary, summary
