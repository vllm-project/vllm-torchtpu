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
"""`qwen3_next_patch._norm_fp8_gate_and_se_jax` with the DN1 prologue kernels
(TPU_SE_PREQUANT_KERNEL; R1 push lever DN1, agent DNS).

Given the residual sum ``res_out`` it returns ``(logits, se_out, h_bf16)``, meant
bit for bit as `_gemma_rms_norm_jax` + `_fp8_gate_and_se_jax` compute them:

* mode 1 (DN1b): the x absmax + fp8 convert run as one kernel pass
  (`norm_quant_rows`); everything after it is `_fp8_gate_and_se_jax`'s own
  statements, in order.
* mode 2 (DN1c, pre-MoE half): norm, x quant, mm1, silu*u and the inter quant
  run as one kernel (`norm_quant_mm1_inter`); the gate GEMV, the scale product
  and mm2 stay XLA, so XLA keeps fusing mm2 with the MoE combine add and the
  next layer's residual sum as today.
* mode 3: mode 2 with the gate GEMV also in the kernel.
* mode 4 (MO3b): mode 2 whose kernel also emits the MoE route quant (MOE spec v1,
  K1) in place of h_bf16; the caller stages it for the fused-EP MoE.

The normed rows ``h_bf16`` are still returned as an XLA expression (the MoE
route quant re-derives them from ``res_out``, as it does today). Returns None
when the row count or widths do not tile; the caller then runs today's path.
"""

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.experimental.se_prequant import kernel as K

FP8 = jnp.float8_e4m3fn
# Pallas interpret mode, for host tests only (they set it). Serving never infers it from
# the JAX backend: torch_tpu traces these ops through its own bridge.
INTERPRET = False
# Mode 4's x input spelling and scoped VMEM (r1pdnskb5 picks): "mustfuse_cvt" (per-layer must-fuse
# group, as mode 2 keeps it), "f32fuse" (the bf16->f32 convert fused into the producer, no must-fuse) or
# "bf16" (the kernel converts in-kernel; with the barrier the pod pays a bf16 relayout copy, r1pdnskb8).
# Default "bf16" (r1pdnskb8 + r1pdnse5): in the pod the must-fuse never formed, so "mustfuse_cvt" fell back to a
# standalone f32 convert (34.1 us/call); bf16 rows behind the barrier cost a bf16 relayout copy (22.7 us/call).
MODE4_XIN = "bf16"
MODE4_VMEM = 56 * 2**20
# MO3b option 1: the kernel reads res_out behind an optimization barrier, so x = f32(bf16 res_out)
# exactly and its convert cannot become a second output of the out_proj producer (r1pdnse4c: that
# cost the producer +24 ms/round, displaced its fused row sumsq and broke bit identity). r and
# h_bf16 still come from the un-barriered res_out, so the producer keeps its fused sumsq.
MODE4_BARRIER = True


def _mustfuse_on() -> bool:
    from vllm_torchtpu import envs

    return bool(envs.TPU_SE_PREQUANT_MUSTFUSE)


def _x_input(mode: int, fuse_id, interpret: bool, vmem_mib: int = 0) -> dict:
    """The kernel's x-input options: mode 2/3 keep the must-fuse group of r1pdnse3 (when the
    layer has an id); mode 4 uses MODE4_XIN / MODE4_VMEM. ``vmem_mib`` (> 0) overrides the
    kernel's scoped VMEM limit (TPU_SE_PREQUANT_VMEM_MIB)."""
    if interpret:
        return {}
    opts = {}
    if vmem_mib:
        opts["vmem_limit"] = int(vmem_mib) * 2**20
    if mode == 4:
        opts.setdefault("vmem_limit", MODE4_VMEM)
        if fuse_id is not None and _mustfuse_on():
            # TPU_SE_PREQUANT_MUSTFUSE: the per-layer must-fuse group of the convert and the
            # kernel call; qwen3_next_patch pushes MUST_FUSE onto the callee's ops so it
            # survives torch_tpu's embedding (r1pdns_mfinner_aot).
            opts.update(xin="mustfuse_cvt", fuse_x=True, fuse_id=str(fuse_id))
        elif MODE4_XIN == "f32fuse":
            opts.update(xin="f32fuse", fuse_x=True)
        elif MODE4_XIN == "bf16":
            pass  # bf16 res_out (behind the barrier) converted in-kernel; r1pdnskb8: a relayout copy, no f32 x
        elif fuse_id is not None:
            opts.update(xin="mustfuse_cvt", fuse_x=True, fuse_id=str(fuse_id))
        return opts
    if fuse_id is not None:
        opts.update(xin="mustfuse_cvt", fuse_x=True, fuse_id=str(fuse_id))
    return opts


def row_tile(rows: int) -> int:
    """The kernels' row tile: 256 or 128 dividing ``rows``, else 0 (today's path).

    r1pdnskb3 [M]: bit-identical at 256/128/64/32, but the pre-MoE kernel costs
    69 / 77 / 125 / 229 us per call at 4,352 rows against 106 us of the XLA ops it
    replaces, so tiles below 128 would be a slowdown."""
    for tm in (256, 128):
        if rows % tm == 0:
            return tm
    return 0


def inter_chunk(n_inter: int) -> int:
    for nc in (256, 128):
        if n_inter % nc == 0:
            return nc
    return 0


def norm_fp8_gate_and_se(
    res_out: jax.Array,
    norm_w_plus_one: jax.Array,
    w_gate_gu_fp8: jax.Array,
    s_gate_gu: jax.Array,
    w_se_gate_bf16: jax.Array,
    w_down_fp8: jax.Array,
    s_down: jax.Array,
    *,
    eps: float,
    mode: int,
    fuse_id: int | None = None,
    vmem_mib: int = 0,
):
    rows = res_out.shape[0]
    n_inter = w_down_fp8.shape[0]
    n_gate = w_gate_gu_fp8.shape[1] - 2 * n_inter
    tm = row_tile(rows)
    nc = inter_chunk(n_inter)
    if (
        mode not in (1, 2, 3, 4)
        or not tm
        or res_out.dtype != jnp.bfloat16
        or w_gate_gu_fp8.dtype != FP8
        or (mode >= 2 and (not nc or n_gate % 128))
    ):
        return None
    interpret = INTERPRET
    # `_gemma_rms_norm_jax`, with the mean square kept as r for the kernel.
    x_f32 = res_out.astype(jnp.float32)
    var = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
    r = jax.lax.rsqrt(var + jnp.float32(eps))
    h_bf16 = (x_f32 * r * norm_w_plus_one.astype(jnp.float32)).astype(jnp.bfloat16)
    if mode == 1:
        h_q, h_scale = K.norm_quant_rows(res_out, r, norm_w_plus_one, tm=tm, interpret=interpret)
        seg_logit = jnp.dot(
            h_bf16,
            w_se_gate_bf16.reshape(-1),
            preferred_element_type=jnp.float32,
        )[:, None]
        seg_sig = jax.nn.sigmoid(seg_logit)
        mm1 = jax.lax.dot_general(
            h_q,
            w_gate_gu_fp8,
            dimension_numbers=(((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        mm1_deq = mm1 * (h_scale * s_gate_gu[None, :])
        router_logits = mm1_deq[:, :n_gate].astype(jnp.bfloat16)
        gu = mm1_deq[:, n_gate:]
        g = gu[:, :n_inter]
        u = gu[:, n_inter:]
        inter = (jax.nn.silu(g) * u).astype(jnp.bfloat16).astype(jnp.float32)
        amax2 = jnp.max(jnp.abs(inter), axis=-1, keepdims=True)
        sc2 = amax2 * jnp.float32(1.0 / 448.0)
        sc2 = jnp.where(sc2 == 0.0, 1.0, sc2)
        sc2_inv = jnp.float32(1.0) / sc2
        q_inter = jnp.clip(inter * sc2_inv, -448.0, 448.0).astype(FP8)
        s_inter = sc2 * seg_sig
    else:
        outs = K.norm_quant_mm1_inter(
            jax.lax.optimization_barrier(res_out) if mode == 4 and MODE4_BARRIER else res_out,
            r,
            norm_w_plus_one,
            w_gate_gu_fp8,
            s_gate_gu,
            w_se_gate_bf16 if mode == 3 else None,
            n_gate=n_gate,
            tm=tm,
            nc=nc,
            seg=mode == 3,
            moe_q=mode == 4,
            interpret=interpret,
            # A per-layer must-fuse id: the kernel call is fused with the bf16->f32 convert
            # of res_out, so it reads res_out in the out_proj fusion's T(16,128) tiling instead
            # of through a relayout copy (r1pdnse1: +11.2 ms/round of copies).
            **_x_input(mode, fuse_id, interpret, vmem_mib),
        )
        router_logits, q_inter, sc2 = outs[0], outs[1], outs[2]
        moe_q = (outs[3], outs[4]) if mode == 4 else None
        if mode == 3:  # the gate GEMV ran in the kernel, on the same VMEM rows
            seg_logit = outs[3]
        else:
            seg_logit = jnp.dot(
                h_bf16,
                w_se_gate_bf16.reshape(-1),
                preferred_element_type=jnp.float32,
            )[:, None]
        s_inter = sc2 * jax.nn.sigmoid(seg_logit)
    mm2 = jax.lax.dot_general(
        q_inter,
        w_down_fp8,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    se_out = (mm2 * (s_inter * s_down[None, :])).astype(jnp.bfloat16)
    if mode == 4:
        # MO3b: the MoE route quant's (q [T, H/128, 128] fp8, scale [T, 1] f32) instead of
        # h_bf16, which nobody reads once the MoE takes the staged rows.
        return router_logits, se_out, moe_q[0], moe_q[1]
    return router_logits, se_out, h_bf16
