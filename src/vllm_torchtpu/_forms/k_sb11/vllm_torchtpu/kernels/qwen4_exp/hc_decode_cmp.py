# SPDX-License-Identifier: Apache-2.0
"""Decode HyperConnection mix with every HBM transfer issued at entry.

The same unit as ``hc_decode.py`` (the previous HC's combine, the per-branch
RMS norm, the low-rank down/SiLU/up gate, the sigmoid-weighted branch mean and
the combine's injection gate), laid out after serving-roofline's ``hc``
template (bottomup/templates/hc.py, v2-cmp-int):

* Rows and weights are copied by hand, all started at entry and waited where
  first used, one stream's 2,560 columns at a time: the stream's input rows,
  then its down-projection slab, then the up-projection slabs. Each stream's
  combined rows go back to HBM as soon as they are formed, so only the last
  writes are exposed.
* The inject rows are copied below the down rows ([down; inject; unused] in
  one [384, W] VMEM buffer), so the injection logits come from the down
  projection's second MXU column tile and cost no VPU work.
* The combine and norm run on 16-row pieces in 256-lane chunks, the sum of
  squares kept in 128 lanes; the norm weight plus one is formed once.
* The mean's sigmoid is ``(1 + tanh(u / 2)) / 2`` with the half folded into
  the gate (halving is exact in bf16 and in the f32 dot) and into the mean's
  scale: one EUP op an element.

Rounding boundaries are the served decode kernel's:

* prologue: h = bf16(f32(residual) + f32(block) * injection_prev[:, b])
* norm = bf16(f32(h) * rsqrt(mean(h^2) + eps) * (1 + f32(w))) per branch
* gate = bf16(silu(f32(bf16(norm @ down^T) / hc)))
* mixed = bf16(sum_b(sigmoid(f32(bf16(gate @ up_b^T))) * f32(norm_b)) / hc)
* injection = 2 * sigmoid(f32(bf16(norm @ inject^T)) / hc), kept in f32

It differs from ``hc_decode.py`` in the summation orders (the sum of squares,
the dots, the injection logits now on the MXU) and in the sigmoid's
implementation: ulp-level differences that can move a bf16 rounding.

The input layouts are the served ones (``down`` [lowrank, W], ``up``
[W, lowrank], ``inject`` [hc, W]), so the module's parameters and their first
use order are unchanged.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.qwen4_exp.hc_decode import hc_decode_mix

# Row counts the kernel takes: whole 16-row pieces up to 256 rows. Other
# counts run ``hc_decode.py``.
MAX_TOKENS = 256
PIECE = 16
CHUNK = 256
LANES = 128


# (lookahead, overlap, up_whole): MiB of copies kept in flight ahead of the
# one being waited (a large number issues all at entry), the next stream's norm
# in one block with the previous down dot, one up dot for all streams. The
# default is the fastest form timed on TPU7x (j-283f92fc, tools/hc_cmp/bench.py):
# 15.3 us a call at 256 rows against 17.9 with a 4 MiB budget and 21.8 with
# every copy at entry and no overlap.
DEFAULT_OPTS = (("lookahead", 8), ("overlap", True), ("up_whole", True))
# q38 HC-HIGHROWS-2 (CMP-S, TPU_QWEN4_HC_DECODE_COMPARE_SIGMOID): the same
# schedule with the mix term's sigmoid taken as the served kernel takes it,
# jax.nn.sigmoid(bf16(u)).
SIGMOID_OPTS = DEFAULT_OPTS + (("sigmoid", True),)


def cmp_applies(tokens: int) -> bool:
    """Whether ``tokens`` rows run this kernel (a concrete count)."""
    return PIECE <= tokens <= MAX_TOKENS and tokens % PIECE == 0


class _NoCopy:
    """A copy left out by ``mode="compute"``."""

    def start(self):
        pass

    def wait(self):
        pass


def _lane_column(values, column: int):
    """``values[:, column:column + 1]`` as a masked lane reduction (exact)."""
    lanes = jax.lax.broadcasted_iota(jnp.int32, values.shape, 1)
    return jnp.sum(jnp.where(lanes == column, values, 0.0), axis=1, keepdims=True)


def _kernel(
    *refs,
    tokens: int,
    branches: int,
    hidden: int,
    lowrank: int,
    eps: float,
    prologue: bool,
    inject: bool,
    mode: str,
    lookahead: int,
    overlap: bool,
    up_whole: bool,
    sigmoid: bool = False,
):
    # mode (timing only): "full"; "dma" runs the copies and no compute;
    # "compute" runs the compute and no copies.
    compute = mode != "dma"
    refs = list(refs)
    if prologue:
        res_hbm, blk_hbm, pinj_hbm = refs[:3]
        refs = refs[3:]
    else:
        res_hbm = refs.pop(0)
        blk_hbm = pinj_hbm = None
    w_hbm, down_hbm, up_hbm = refs[:3]
    refs = refs[3:]
    injw_hbm = refs.pop(0) if inject else None
    mixed_hbm = refs.pop(0)
    injo_hbm = refs.pop(0) if inject else None
    h_hbm = refs.pop(0) if prologue else None
    (res_vm, w_vm, w1_vm, dn_vm, up_vm, xn_vm, y_vm, acc_vm, mixed_vm) = refs[:9]
    refs = refs[9:]
    if prologue:
        blk_vm, pinj_vm = refs[:2]
        refs = refs[2:]
    injo_vm = refs.pop(0) if inject else None
    (sems,) = refs

    rows = tokens
    width = branches * hidden
    pieces = rows // PIECE
    chunks = hidden // CHUNK

    def stream(b):
        return pl.ds(b * hidden, hidden)

    # Semaphores: 0..hc-1 input rows by stream, hc..2hc-1 down slabs,
    # 2hc..3hc-1 up slabs, 3hc block, 3hc+1 prev injection, 3hc+2 norm weight,
    # 3hc+3 inject rows, 3hc+4..4hc+3 combined rows out, 4hc+4 mixed out,
    # 4hc+5 injection out.
    s_rows, s_dn, s_up = 0, branches, 2 * branches
    s_blk, s_pinj, s_w, s_inj = (3 * branches + i for i in range(4))
    s_h = 3 * branches + 4
    s_mixed, s_injo = 4 * branches + 4, 4 * branches + 5

    def copy(src, dst, sem):
        if mode == "compute":
            return _NoCopy()
        return pltpu.make_async_copy(src, dst, sems.at[sem])

    rows_in = [
        copy(res_hbm.at[:, stream(b)], res_vm.at[:, stream(b)], s_rows + b)
        for b in range(branches)
    ]
    dn_in = [
        copy(
            down_hbm.at[:, stream(b)],
            dn_vm.at[pl.ds(0, lowrank), stream(b)],
            s_dn + b,
        )
        for b in range(branches)
    ]
    up_in = [
        copy(up_hbm.at[stream(b), :], up_vm.at[stream(b), :], s_up + b)
        for b in range(branches)
    ]
    w_in = copy(w_hbm, w_vm, s_w)
    blk_in = copy(blk_hbm, blk_vm, s_blk) if prologue else None
    pinj_in = copy(pinj_hbm, pinj_vm, s_pinj) if prologue else None
    inj_in = (
        copy(injw_hbm, dn_vm.at[pl.ds(lowrank, branches), :], s_inj)
        if inject
        else None
    )

    # Copies in consumption order. All in flight at once share the HBM port,
    # so the first stream's rows would land late: a copy starts once the
    # copies started ahead of it and not yet waited hold less than
    # ``lookahead`` MiB.
    T, H, W, R = rows, hidden, width, lowrank
    queue = [(rows_in[0], T * H * 2)]
    if prologue:
        queue += [(blk_in, T * H * 2), (pinj_in, T * LANES * 4)]
    queue += [(w_in, W * 2)]
    if inject:
        queue += [(inj_in, branches * W * 2)]
    queue += [(dn_in[0], R * H * 2)]
    for b in range(1, branches):
        queue += [(rows_in[b], T * H * 2), (dn_in[b], R * H * 2)]
    queue += [(c, H * LANES * 3 * 2) for c in up_in]
    budget = lookahead << 20
    state = dict(started=0, waited=-1)

    def pump():
        while state["started"] < len(queue):
            i = state["started"]
            held = sum(n for _, n in queue[state["waited"] + 1 : i])
            if i > state["waited"] + 1 and held + queue[i][1] > budget:
                break
            queue[i][0].start()
            state["started"] += 1

    def need(c):
        i = next(j for j, (q, _) in enumerate(queue) if q is c)
        while state["started"] <= i:
            queue[state["started"]][0].start()
            state["started"] += 1
        c.wait()
        state["waited"] = max(state["waited"], i)
        pump()

    pump()

    need(w_in)
    if compute:
        # The norm weight plus one, f32, broadcast over a piece's 16 rows.
        w1_vm[...] = jnp.broadcast_to(
            1.0 + w_vm[...].astype(jnp.float32)[None, :], (PIECE, width)
        )
    if prologue:
        need(blk_in)
        need(pinj_in)

    def piece(p, b):
        r0 = p if isinstance(p, int) else pl.multiple_of(p * PIECE, PIECE)
        prs = pl.ds(r0, PIECE)
        if prologue:
            ib = jnp.broadcast_to(_lane_column(pinj_vm[prs, :], b), (PIECE, CHUNK))
        ss = None
        for k in range(chunks):
            cols = pl.ds(b * hidden + k * CHUNK, CHUNK)
            if prologue:
                blk = blk_vm[prs, pl.ds(k * CHUNK, CHUNK)].astype(jnp.float32)
                hb = (res_vm[prs, cols].astype(jnp.float32) + blk * ib).astype(
                    jnp.bfloat16
                )
                res_vm[prs, cols] = hb
            else:
                hb = res_vm[prs, cols]
            for q in range(CHUNK // LANES):
                vq = hb[:, q * LANES : (q + 1) * LANES].astype(jnp.float32)
                ss = vq * vq if ss is None else ss + vq * vq
        rs = jax.lax.rsqrt(jnp.sum(ss, axis=1, keepdims=True) / hidden + eps)
        rb = jnp.broadcast_to(rs, (PIECE, CHUNK))
        for k in range(chunks):
            cols = pl.ds(b * hidden + k * CHUNK, CHUNK)
            v = res_vm[prs, cols].astype(jnp.float32)
            xn_vm[prs, cols] = (v * rb * w1_vm[:, cols]).astype(jnp.bfloat16)

    def down_dot(c):
        if inject and c == 0:
            need(inj_in)
        need(dn_in[c])
        if not compute:
            return
        part = jax.lax.dot_general(
            xn_vm[:, stream(c)],
            dn_vm[:, stream(c)],
            dimension_numbers=(((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        if c == 0:
            y_vm[...] = part
        else:
            y_vm[...] += part

    h_out = []
    for b in range(branches):
        need(rows_in[b])
        if overlap and b >= 1:
            # Stream b's norm, unrolled, in one block with stream b - 1's down
            # dot: the scheduler packs the VPU work beside the MXU work.
            down_dot(b - 1)
            if compute:
                for p in range(pieces):
                    piece(p * PIECE, b)
        elif compute:
            jax.lax.fori_loop(
                0, pieces, lambda p, c, b=b: (piece(p, b), c)[1], jnp.int32(0)
            )
        if prologue:
            out = copy(res_vm.at[:, stream(b)], h_hbm.at[:, stream(b)], s_h + b)
            out.start()
            h_out.append(out)
        if not overlap:
            down_dot(b)
    if overlap:
        down_dot(branches - 1)

    # Gate on the low-rank columns, halved (exact): the up dot then gives
    # u / 2, and 2 sigmoid(bf16(u)) = 1 + tanh(bf16(u / 2)).
    if compute:
        y = y_vm[:, pl.ds(0, lowrank)]
        z = y.astype(jnp.bfloat16).astype(jnp.float32) / branches
        z = z.astype(jnp.bfloat16).astype(jnp.float32)
        gate_half = (z * (0.5 * jax.nn.sigmoid(z))).astype(jnp.bfloat16)

    if inject and compute:
        # The inject rows' logits sit in lanes lowrank % 128 + j of the
        # 128-lane tile that holds column lowrank.
        tile0 = lowrank // LANES * LANES
        lane0 = lowrank - tile0
        logits = y_vm[:, pl.ds(tile0, LANES)]
        gates = 2.0 * jax.nn.sigmoid(
            logits.astype(jnp.bfloat16).astype(jnp.float32) / branches
        )
        shape = (rows, branches)
        lanes = jax.lax.broadcasted_iota(jnp.int32, shape, 1)
        injection = jnp.zeros(shape, jnp.float32)
        for j in range(branches):
            injection = jnp.where(
                lanes == j, _lane_column(gates, lane0 + j), injection
            )
        injo_vm[...] = injection
    if inject:
        injo_out = copy(injo_vm, injo_hbm, s_injo)
        injo_out.start()

    def mix_term(u_half, b):
        u = u_half.astype(jnp.bfloat16).astype(jnp.float32)
        if sigmoid:
            # CMP-S: u + u = bf16(u_full) exactly (halving is exact), so the term is
            # exactly 2 x the served kernel's sigmoid(bf16(u_full)) x norm; the mean's
            # 0.5 / hc below is unchanged.
            norm_b = xn_vm[:, stream(b)].astype(jnp.float32)
            return 2.0 * jax.nn.sigmoid(u + u) * norm_b
        return (1.0 + jnp.tanh(u)) * xn_vm[:, stream(b)].astype(jnp.float32)

    if up_whole:
        for b in range(branches):
            need(up_in[b])
    for b in range(branches):
        if not up_whole:
            need(up_in[b])
        if not compute:
            continue
        if up_whole:
            if b == 0:
                u_all = jax.lax.dot_general(
                    gate_half,
                    up_vm[...],
                    dimension_numbers=(((1,), (1,)), ((), ())),
                    preferred_element_type=jnp.float32,
                )
            term = mix_term(u_all[:, b * hidden : (b + 1) * hidden], b)
        else:
            u_half = jax.lax.dot_general(
                gate_half,
                up_vm[stream(b), :],
                dimension_numbers=(((1,), (1,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            term = mix_term(u_half, b)
        if b == 0:
            acc_vm[...] = term
        elif b < branches - 1:
            acc_vm[...] += term
        else:
            mixed_vm[...] = ((acc_vm[...] + term) * (0.5 / branches)).astype(
                jnp.bfloat16
            )
    mixed_out = copy(mixed_vm, mixed_hbm, s_mixed)
    mixed_out.start()

    for out in h_out:
        out.wait()
    if inject:
        injo_out.wait()
    mixed_out.wait()


def hc_decode_mix_auto(
    x,
    weight,
    down,
    up,
    inject_weight=None,
    *,
    branches: int,
    hidden: int,
    eps: float,
    block=None,
    prev_injection=None,
    max_tile=0,
    interpret=False,
    sigmoid=False,
):
    """``hc_decode_mix_cmp`` where ``cmp_applies``, else ``hc_decode_mix``.

    ``sigmoid`` (CMP-S) runs the cmp schedule with the served sigmoid
    (``SIGMOID_OPTS``).

    The choice is made on the concrete row count of the traced bucket, so the
    torch-side gate (``decode_token_tile``) is unchanged.
    """
    kwargs = dict(
        branches=branches,
        hidden=hidden,
        eps=eps,
        block=block,
        prev_injection=prev_injection,
        interpret=interpret,
    )
    if cmp_applies(x.shape[0]):
        opts = SIGMOID_OPTS if sigmoid else None
        return hc_decode_mix_cmp(
            x, weight, down, up, inject_weight, _opts=opts, **kwargs
        )
    return hc_decode_mix(
        x, weight, down, up, inject_weight, max_tile=max_tile, **kwargs
    )


def hc_decode_mix_cmp(
    x,
    weight,
    down,
    up,
    inject_weight=None,
    *,
    branches: int,
    hidden: int,
    eps: float,
    block=None,
    prev_injection=None,
    interpret=False,
    _mode="full",
    _opts=None,
):
    """``hc_decode.hc_decode_mix`` for ``cmp_applies`` row counts.

    Same arguments and returns: ``(mixed, injection)`` (``injection`` None
    without ``inject_weight``), plus the combined hyper input with ``block``.
    """
    return _hc_decode_mix_cmp(
        x,
        weight,
        down,
        up,
        inject_weight,
        block,
        prev_injection,
        branches=branches,
        hidden=hidden,
        eps=eps,
        interpret=interpret,
        mode=_mode,
        opts=_opts or DEFAULT_OPTS,
    )


@functools.partial(
    jax.jit, static_argnames=("branches", "hidden", "eps", "interpret", "mode", "opts")
)
def _hc_decode_mix_cmp(
    x,
    weight,
    down,
    up,
    inject_weight,
    block,
    prev_injection,
    *,
    branches,
    hidden,
    eps,
    interpret,
    mode,
    opts,
):
    tokens, width = x.shape
    lowrank = down.shape[0]
    prologue = block is not None
    inject = inject_weight is not None
    down_rows = -(-(lowrank + (branches if inject else 0)) // LANES) * LANES
    if (
        not cmp_applies(tokens)
        or width != branches * hidden
        or hidden % CHUNK
        or weight.shape != (width,)
        or down.shape != (lowrank, width)
        or up.shape != (width, lowrank)
        or (inject and inject_weight.shape != (branches, width))
        or (prologue and block.shape != (tokens, hidden))
        or (prologue and prev_injection.shape != (tokens, branches))
        or (prologue and prev_injection.dtype != jnp.float32)
        or any(
            a.dtype != jnp.bfloat16
            for a in (x, weight, down, up)
            + ((inject_weight,) if inject else ())
            + ((block,) if prologue else ())
        )
    ):
        raise ValueError("HC decode mix (cmp) requires BF16 16-row pieces")

    any_space = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    n_in = (3 if prologue else 1) + (4 if inject else 3)
    out_shape = [jax.ShapeDtypeStruct((tokens, hidden), jnp.bfloat16)]
    if inject:
        out_shape.append(jax.ShapeDtypeStruct((tokens, branches), jnp.float32))
    if prologue:
        out_shape.append(jax.ShapeDtypeStruct((tokens, width), jnp.bfloat16))
    scratch = [
        pltpu.VMEM((tokens, width), jnp.bfloat16),  # res_vm (rows in, h)
        pltpu.VMEM((width,), jnp.bfloat16),  # w_vm
        pltpu.VMEM((PIECE, width), jnp.float32),  # w1_vm
        pltpu.VMEM((down_rows, width), jnp.bfloat16),  # dn_vm [down; inject]
        pltpu.VMEM((width, lowrank), jnp.bfloat16),  # up_vm
        pltpu.VMEM((tokens, width), jnp.bfloat16),  # xn_vm
        pltpu.VMEM((tokens, down_rows), jnp.float32),  # y_vm
        pltpu.VMEM((tokens, hidden), jnp.float32),  # acc_vm
        pltpu.VMEM((tokens, hidden), jnp.bfloat16),  # mixed_vm
    ]
    if prologue:
        scratch += [
            pltpu.VMEM((tokens, hidden), jnp.bfloat16),  # blk_vm
            pltpu.VMEM((tokens, branches), jnp.float32),  # pinj_vm
        ]
    if inject:
        scratch.append(pltpu.VMEM((tokens, branches), jnp.float32))  # injo_vm
    scratch.append(pltpu.SemaphoreType.DMA((4 * branches + 6,)))
    args = [x, block, prev_injection] if prologue else [x]
    args += [weight, down, up] + ([inject_weight] if inject else [])
    outs = pl.pallas_call(
        functools.partial(
            _kernel,
            tokens=tokens,
            branches=branches,
            hidden=hidden,
            lowrank=lowrank,
            eps=eps,
            prologue=prologue,
            inject=inject,
            mode=mode,
            **dict(opts),
        ),
        in_specs=[any_space] * n_in,
        out_specs=[any_space] * len(out_shape),
        out_shape=out_shape,
        scratch_shapes=scratch,
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=56 << 20),
        interpret=interpret,
        # The served kernels' names with a suffix: profile classifiers that
        # match the served prefixes file these under the same rows.
        name=f"qwen4exp_hc_decode{'_combine' if prologue else ''}_cmp",
    )(*args)
    mixed = outs[0]
    injection = outs[1] if inject else None
    if prologue:
        return mixed, injection, outs[-1]
    return mixed, injection
