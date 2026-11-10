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
# ruff: noqa: E501
"""Router + shared-expert prologue kernels (R1 push, lever DN1).

``qwen3_next_patch._norm_fp8_gate_and_se_jax`` computes, per row of the
post-attention residual stream ``res`` [T, H] bf16 (GemmaRMSNorm already
reduced to ``r = rsqrt(var + eps)`` by the producer fusion):

    h     = bf16(f32(res) * r * w)                       # the norm
    h_q   = fp8(clip(f32(h) / s, +-448)),  s = rowmax|h| / 448 (1 if 0)
    mm1   = (h_q @ W_gu) * (s * s_gu)                   # [T, n_gate + 2 I] f32
    logits = bf16(mm1[:, :n_gate])
    inter = bf16(silu(g) * u)                            # g, u = the next 2 I cols
    q_i   = fp8(clip(f32(inter) / s2, +-448)), s2 = rowmax|inter| / 448 (1 if 0)

XLA does not keep ``h`` anywhere: every consumer re-reads ``res`` and redoes
the norm, so the x quantization alone is two HBM passes (absmax 16.0 us,
convert 27.4 us per call on base13), and mm1's f32 output is written and read
back for silu*u and the second absmax.

* :func:`norm_quant_rows` (DN1b): ``h_q, s`` in ONE pass over ``res``.
* :func:`norm_quant_mm1_inter` (DN1c, pre-MoE half): the whole chain above in
  one kernel with ``W_gu`` resident in VMEM; returns ``logits, q_i, s2``.

The elementwise math is written op for op as XLA's fusions compute it
(``x * r`` then ``* w``; ``amax * f32(1/448)``; ``1 / s``; ``g * (1 / (exp(-g) + 1)) * u``),
so the outputs are meant to be bit-identical to the XLA path. That is a claim
for the TPU bench to prove (se_bench.py), not an assumption: the absmax is
order-free, but the reciprocal, exp, the fp8 convert and the MXU accumulation
order of mm1 are lowered by Mosaic, not by XLA.

``seg=True`` also returns the shared-expert gate logit ``h @ w_seg`` [T, 1] f32
from the same VMEM tile (an MXU dot; bit identity with XLA's GEMV is again for
the bench to show).
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

FP8 = jnp.float8_e4m3fn
FP8_MAX = 448.0
SEG_LANES = 128

# Mutants for the bench (must FAIL bit identity): a wrong x-quant scale.
MUTANTS = ("none", "x447")
# How the kernel spells the scale and the fp8 cast, so that Mosaic reproduces the
# arithmetic XLA's TPU backend actually runs for `_fp8_gate_and_se_jax` (r1pdnsprobe1
# decides; CPU interpret cannot tell them apart):
#   scale  "mul": amax * f32(1/448) as written; "div": amax / 448.
#   cast   "direct": f32 -> fp8; "viabf16": f32 -> bf16 -> fp8 (double rounding);
#          "preround": RNE onto the fp8 grid in f32 first, then the (exact) cast.
SCALES = ("mul", "div")
CASTS = ("direct", "viabf16", "preround")
SCALE = "mul"
CAST = "direct"
# XLA's TPU codegen drops an f32 -> bf16 -> f32 round trip inside a fusion (excess
# precision; r1pdnsprobe1 [M]: XLA's x.astype(bf16).astype(fp8) == x.astype(fp8) on the
# whole sweep, and its x absmax differs from the bf16-rounded one in every row). So in
# serving the x absmax and the x quant see the UNROUNDED f32 norm, and (hypothesis, the
# re-bench decides) the inter absmax sees the unrounded silu*u while q_inter reads the
# bf16 inter XLA materializes. "none" reproduces that; "bf16" is the literal program.
ROUNDS = ("none", "bf16")
XROUND = "none"
IROUND = "none"


def _scale_of(amax, mutant: str, scale: str):
    den = 447.0 if mutant == "x447" else FP8_MAX
    if scale == "div":
        return amax / jnp.float32(den)
    return amax * jnp.float32(1.0 / den)


def round_to_fp8_grid(x):
    """RNE of |x| <= 448 onto the e4m3fn grid in f32 (spacing 2^(e-3) for normals,
    2^-9 below 2^-6): add and subtract 2^(spacing + 23) per element."""
    ax = jnp.abs(x)
    e = jnp.maximum((jax.lax.bitcast_convert_type(ax, jnp.int32) >> 23) - 127, -6)
    magic = jax.lax.bitcast_convert_type(((e - 3 + 23 + 127) << 23).astype(jnp.int32), jnp.float32)
    r = (ax + magic) - magic
    return jnp.where(x < 0, -r, r)


def _cast_fp8(y, cast: str):
    if cast == "viabf16":
        return y.astype(jnp.bfloat16).astype(FP8)
    if cast == "preround":
        return round_to_fp8_grid(y).astype(FP8)
    return y.astype(FP8)


def _quant_rows(v_f32, mutant="none", scale=None, cast=None):
    """`_fp8_gate_and_se_jax`'s per-row fp8 quantization of an f32 tile."""
    amax = jnp.max(jnp.abs(v_f32), axis=-1, keepdims=True)
    s = _scale_of(amax, mutant, scale or SCALE)
    s = jnp.where(s == 0.0, jnp.float32(1.0), s)
    s_inv = jnp.float32(1.0) / s
    q = _cast_fp8(jnp.clip(v_f32 * s_inv, -FP8_MAX, FP8_MAX), cast or CAST)
    return q, s


def _norm_tile_f32(x_ref, r_ref, w_ref, xin="bf16"):
    """`_gemma_rms_norm_jax` after the mean, before its bf16 cast: f32(x) * r * w.
    xin "u32": x_ref holds row pairs packed in u32 ([tm/2, H], the bf16 T(16,128)(2,1)
    memory read as u32 T(8,128)); "u32s": the same with the halves swapped."""
    x = x_ref[...]
    if xin in ("u32", "u32s"):
        x = pltpu.bitcast(x, jnp.bfloat16)
    return (x.astype(jnp.float32) * r_ref[...]) * w_ref[...]


XINS = ("bf16", "f32fuse", "u32", "u32s", "mustfuse", "mustfuse_cvt", "mustfuse_meta")
MOE_SCALES = ("div", "mul")


def _x_operand(res, xin):
    """The kernel's x operand for each input mode (see _norm_tile_f32)."""
    t, hidden = res.shape
    if xin == "f32fuse":
        return res.astype(jnp.float32)  # fused into the call (allow_input_fusion), never materialized
    if xin in ("mustfuse_cvt", "mustfuse_meta"):
        return res.astype(jnp.float32)
    if xin in ("u32", "u32s"):
        pairs = res.reshape(t // 2, 2, hidden)
        if xin == "u32s":
            pairs = pairs[:, ::-1, :]
        return jax.lax.bitcast_convert_type(jnp.swapaxes(pairs, 1, 2), jnp.uint32)
    return res


def _xquant_input(h_f32, xround):
    return h_f32.astype(jnp.bfloat16).astype(jnp.float32) if xround == "bf16" else h_f32


def _nq_kernel(x_ref, r_ref, w_ref, *rest, seg, mutant, scale, cast, xround):
    if seg:
        wseg_ref, q_ref, s_ref, seg_ref = rest
    else:
        q_ref, s_ref = rest
    h_f32 = _norm_tile_f32(x_ref, r_ref, w_ref)
    h = h_f32.astype(jnp.bfloat16)
    q, s = _quant_rows(_xquant_input(h_f32, xround), mutant, scale, cast)
    q_ref[...] = q
    s_ref[...] = s
    if seg:
        seg_ref[...] = jnp.dot(h, wseg_ref[...], preferred_element_type=jnp.float32)[:, :1]


def hidden_ok(hidden: int) -> bool:
    return hidden % 128 == 0


def _check_rows(t: int, tm: int):
    if t % tm or tm % 32:
        raise ValueError(f"rows {t} must be a multiple of tm {tm}, and tm of 32 (fp8 tiles)")


def _seg_operand(w_seg: jax.Array, hidden: int) -> jax.Array:
    """w_seg [H] as an [H, 128] bf16 MXU operand with the gate in column 0."""
    col = w_seg.reshape(hidden, 1).astype(jnp.bfloat16)
    return jnp.concatenate([col, jnp.zeros((hidden, SEG_LANES - 1), jnp.bfloat16)], axis=1)


@functools.partial(jax.jit, static_argnames=("tm", "seg", "mutant", "interpret", "scale", "cast", "xround"))
def norm_quant_rows(
    res: jax.Array,
    r: jax.Array,
    w: jax.Array,
    w_seg: jax.Array | None = None,
    *,
    tm: int = 256,
    seg: bool = False,
    mutant: str = "none",
    interpret: bool = False,
    scale: str | None = None,
    cast: str | None = None,
    xround: str | None = None,
):
    """DN1b. res [T, H] bf16, r [T, 1] f32 (rsqrt(var + eps)), w [H] (1 + weight).

    Returns ``(h_q [T, H] fp8, s [T, 1] f32)``, plus ``seg [T, 1] f32`` when
    ``seg``."""
    assert mutant in MUTANTS, mutant
    scale, cast, xround = scale or SCALE, cast or CAST, xround or XROUND
    assert scale in SCALES and cast in CASTS and xround in ROUNDS, (scale, cast, xround)
    t, hidden = res.shape
    _check_rows(t, tm)
    row = lambda i: (i, 0)  # noqa: E731
    fixed = lambda i: (0, 0)  # noqa: E731
    in_specs = [
        pl.BlockSpec((tm, hidden), row),
        pl.BlockSpec((tm, 1), row),
        pl.BlockSpec((1, hidden), fixed),
    ]
    args = [res, r.reshape(t, 1).astype(jnp.float32), w.reshape(1, hidden).astype(jnp.float32)]
    out_shape = [
        jax.ShapeDtypeStruct((t, hidden), FP8),
        jax.ShapeDtypeStruct((t, 1), jnp.float32),
    ]
    out_specs = [pl.BlockSpec((tm, hidden), row), pl.BlockSpec((tm, 1), row)]
    if seg:
        in_specs.append(pl.BlockSpec((hidden, SEG_LANES), fixed))
        args.append(_seg_operand(w_seg, hidden))
        out_shape.append(jax.ShapeDtypeStruct((t, 1), jnp.float32))
        out_specs.append(pl.BlockSpec((tm, 1), row))
    return pl.pallas_call(
        functools.partial(_nq_kernel, seg=seg, mutant=mutant, scale=scale, cast=cast, xround=xround),
        grid=(t // tm,),
        in_specs=in_specs,
        out_specs=tuple(out_specs),
        out_shape=tuple(out_shape),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",), vmem_limit_bytes=48 * 2**20
        ),
        name="se_norm_quant%s%s" % ("_seg" if seg else "", "" if mutant == "none" else "_mut" + mutant),
        interpret=pltpu.InterpretParams() if interpret else False,
    )(*args)


def _moe_quant(h_f32, moe_scale, moe_mutant):
    """MO3b (MOE spec v1, measured K1 in r1pmoe6): the MoE route quant as the pod computes it.
    amax in bf16 (= bf16(max|h32|), rounding is monotonic); scale = f32(amax) / 448 with the
    usable guard (0 where not finite or 0); q = fp8(h32 * f32(bf16(1 / scale))) on the UNROUNDED
    h32, one RNE cast; 0 where unusable."""
    amax = jnp.max(jnp.abs(h_f32), axis=-1, keepdims=True).astype(jnp.bfloat16).astype(jnp.float32)
    den = jnp.float32(447.0 if moe_mutant else FP8_MAX)
    sc = amax / den if moe_scale == "div" else amax * (jnp.float32(1.0) / den)
    usable = jnp.isfinite(sc) & (sc > 0)
    sc = jnp.where(usable, sc, jnp.float32(0.0))
    sinv = jnp.where(usable, jnp.float32(1.0) / jnp.where(usable, sc, jnp.float32(1.0)), jnp.float32(0.0))
    sinv = sinv.astype(jnp.bfloat16).astype(jnp.float32)
    q = jnp.where(usable, h_f32 * sinv, jnp.float32(0.0)).astype(FP8)
    return q, sc


def _k1_kernel(x_ref, r_ref, w_ref, wgu_ref, sgu_ref, *rest, n_gate, n_inter, nc, seg, mutant, scale, cast,
               xround, iround, xin, moe_q, moe_scale, moe_mutant):
    rest = list(rest)
    inter_vm = rest.pop()
    if moe_q:
        qm_ref, sm_ref = rest[-2], rest[-1]
        rest = rest[:-2]
    if seg:
        wseg_ref, lg_ref, qi_ref, s2_ref, seg_ref = rest
    else:
        lg_ref, qi_ref, s2_ref = rest
    h_f32 = _norm_tile_f32(x_ref, r_ref, w_ref, xin)
    if moe_q:
        qm, sm = _moe_quant(h_f32, moe_scale, moe_mutant)
        qm_ref[...] = qm.reshape(qm_ref.shape)
        sm_ref[...] = sm
    h = h_f32.astype(jnp.bfloat16)
    hq, s = _quant_rows(_xquant_input(h_f32, xround), mutant, scale, cast)
    if seg:
        seg_ref[...] = jnp.dot(h, wseg_ref[...], preferred_element_type=jnp.float32)[:, :1]

    def deq(c0, width):
        acc = jnp.dot(hq, wgu_ref[:, c0 : c0 + width], preferred_element_type=jnp.float32)
        return acc * (s * sgu_ref[:, c0 : c0 + width])

    lg_ref[...] = deq(0, n_gate).astype(lg_ref.dtype)
    amax2 = None
    for j in range(n_inter // nc):
        g = deq(n_gate + j * nc, nc)
        u = deq(n_gate + n_inter + j * nc, nc)
        sig = jnp.float32(1.0) / (jnp.exp(-g) + jnp.float32(1.0))
        v = (g * sig) * u
        vb = v.astype(jnp.bfloat16)
        inter_vm[:, j * nc : (j + 1) * nc] = vb
        av = v if iround == "none" else vb.astype(jnp.float32)
        m = jnp.max(jnp.abs(av), axis=-1, keepdims=True)
        amax2 = m if amax2 is None else jnp.maximum(amax2, m)
    # q_inter reads the bf16 inter (XLA materializes it); the absmax is per IROUND.
    s2 = _scale_of(amax2, "none", scale)
    s2 = jnp.where(s2 == 0.0, jnp.float32(1.0), s2)
    s2_inv = jnp.float32(1.0) / s2
    qi_ref[...] = _cast_fp8(jnp.clip(inter_vm[...].astype(jnp.float32) * s2_inv, -FP8_MAX, FP8_MAX), cast)
    s2_ref[...] = s2


@functools.partial(
    jax.jit,
    static_argnames=("n_gate", "tm", "nc", "seg", "mutant", "interpret", "weight_buffers", "scale", "cast",
                     "xround", "iround", "fuse_x", "xin", "fuse_id", "moe_q", "moe_scale", "moe_mutant",
                     "vmem_limit"),
)
def norm_quant_mm1_inter(
    res: jax.Array,
    r: jax.Array,
    w: jax.Array,
    w_gu: jax.Array,
    s_gu: jax.Array,
    w_seg: jax.Array | None = None,
    *,
    n_gate: int,
    tm: int = 256,
    nc: int = 256,
    seg: bool = False,
    mutant: str = "none",
    interpret: bool = False,
    weight_buffers: int = 1,
    scale: str | None = None,
    cast: str | None = None,
    xround: str | None = None,
    iround: str | None = None,
    fuse_x: bool = False,
    xin: str = "bf16",
    fuse_id: str = "1",
    moe_q: bool = False,
    moe_scale: str = "div",
    moe_mutant: bool = False,
    vmem_limit: int = 56 * 2**20,
):
    """DN1c pre-MoE half. w_gu [H, n_gate + 2 I] fp8 (K-major), s_gu [n_gate + 2 I] f32.

    Returns ``(logits [T, n_gate] bf16, q_inter [T, I] fp8, s2 [T, 1] f32)``,
    plus ``seg [T, 1] f32`` when ``seg``. ``s2`` is the inter scale after the
    zero guard; the caller multiplies in sigmoid(seg) as today."""
    assert mutant in MUTANTS, mutant
    scale, cast = scale or SCALE, cast or CAST
    xround, iround = xround or XROUND, iround or IROUND
    assert scale in SCALES and cast in CASTS and xround in ROUNDS and iround in ROUNDS and xin in XINS
    assert moe_scale in MOE_SCALES and hidden_ok(res.shape[1]), (moe_scale, res.shape)
    t, hidden = res.shape
    n_gu = w_gu.shape[1]
    n_inter = (n_gu - n_gate) // 2
    _check_rows(t, tm)
    if n_inter % nc or nc % 128 or n_gate % 128:
        raise ValueError(f"n_inter {n_inter} / chunk {nc} / n_gate {n_gate} must tile by 128")
    row = lambda i: (i, 0)  # noqa: E731
    fixed = lambda i: (0, 0)  # noqa: E731
    wmode = pl.Buffered(weight_buffers) if weight_buffers != 2 else None
    in_specs = [
        pl.BlockSpec((tm // 2, hidden) if xin in ("u32", "u32s") else (tm, hidden), row),
        pl.BlockSpec((tm, 1), row),
        pl.BlockSpec((1, hidden), fixed),
        pl.BlockSpec((hidden, n_gu), fixed, pipeline_mode=wmode),
        pl.BlockSpec((1, n_gu), fixed),
    ]
    fuse_x = fuse_x or xin == "f32fuse"
    args = [
        _x_operand(res, xin),
        r.reshape(t, 1).astype(jnp.float32),
        w.reshape(1, hidden).astype(jnp.float32),
        w_gu,
        s_gu.reshape(1, n_gu).astype(jnp.float32),
    ]
    out_shape = [
        jax.ShapeDtypeStruct((t, n_gate), jnp.bfloat16),
        jax.ShapeDtypeStruct((t, n_inter), FP8),
        jax.ShapeDtypeStruct((t, 1), jnp.float32),
    ]
    out_specs = [
        pl.BlockSpec((tm, n_gate), row),
        pl.BlockSpec((tm, n_inter), row),
        pl.BlockSpec((tm, 1), row),
    ]
    if seg:
        in_specs.append(pl.BlockSpec((hidden, SEG_LANES), fixed))
        args.append(_seg_operand(w_seg, hidden))
        out_shape.append(jax.ShapeDtypeStruct((t, 1), jnp.float32))
        out_specs.append(pl.BlockSpec((tm, 1), row))
    if moe_q:
        # MO3b: the MoE route quant's rows [T, H/128, 128] fp8 (per-token (32,128) tiles, the
        # layout the fused-EP plan kernel takes) and its row scale [T, 1] f32.
        lanes = hidden // 128
        out_shape.append(jax.ShapeDtypeStruct((t, lanes, 128), FP8))
        out_specs.append(pl.BlockSpec((tm, lanes, 128), lambda i: (i, 0, 0)))
        out_shape.append(jax.ShapeDtypeStruct((t, 1), jnp.float32))
        out_specs.append(pl.BlockSpec((tm, 1), row))
    call = pl.pallas_call(
        functools.partial(
            _k1_kernel, n_gate=n_gate, n_inter=n_inter, nc=nc, seg=seg, mutant=mutant, scale=scale, cast=cast,
            xround=xround, iround=iround, xin="bf16" if xin == "mustfuse" else xin,
            moe_q=moe_q, moe_scale=moe_scale, moe_mutant=moe_mutant,
        ),
        grid=(t // tm,),
        in_specs=in_specs,
        out_specs=tuple(out_specs),
        out_shape=tuple(out_shape),
        scratch_shapes=[pltpu.VMEM((tm, n_inter), jnp.bfloat16)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=vmem_limit,
            # fuse_x: let XLA fuse res's producer side (in serving: the relayout from the
            # out_proj fusion's T(16,128) output) into this call instead of a separate copy.
            allow_input_fusion=[True] + [False] * (len(in_specs) - 1) if fuse_x else None,
        ),
        name="se_k1_tm%d%s%s%s" % (tm, "_seg" if seg else "", "_moeq" if moe_q else "",
                                   "" if mutant == "none" else "_mut" + mutant),
        interpret=pltpu.InterpretParams() if interpret else False,
    )
    if xin.startswith("mustfuse") and not interpret:
        # A MUST_FUSE call: XLA must form one fusion around the Mosaic call (and, for
        # mustfuse_cvt, the bf16->f32 convert of x), which lets the call accept x in the
        # producer's large second-minor tiling T(16,128) instead of a relayout copy.
        from jax.experimental.xla_metadata import must_fuse_call

        if xin == "mustfuse_cvt":
            return must_fuse_call(fuse_id)(lambda x, *rest: call(x.astype(jnp.float32), *rest))(res, *args[1:])
        if xin == "mustfuse_meta":
            # The same grouping as op-level frontend attributes on the convert and the call (no
            # call wrapper): a hypothesis for the pod, where the call-wrapper form did not fuse.
            from jax.experimental.xla_metadata import set_xla_metadata

            with set_xla_metadata(MUST_FUSE=fuse_id):
                return call(res.astype(jnp.float32), *args[1:])
        return must_fuse_call(fuse_id)(call)(*args)
    return call(*args)
