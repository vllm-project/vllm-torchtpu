# SPDX-License-Identifier: Apache-2.0
"""Prefill HyperConnection mix in serving-roofline's prefill ``hc`` form.

The unit of ``hc_prefill_fused.py`` (optionally the previous HC's combine,
the per-branch RMS norm, the low-rank down/SiLU/up gate, the sigmoid-weighted
branch mean and the combine's injection gate), laid out after
serving-roofline v2-cmp-int's composed prefill connection
(bottomup/compose/prefill_blocks.py ``HCRow`` at the defaults of
bottomup/compose/prefill.py: 256-row tiles, form 7, ``mix=tanh,swap=1,
one_body=1,edges=1``):

* Skewed row tiles. Grid step ``r`` runs row tile ``r``'s combine, norm, down
  projection, gate and injection, and row tile ``r - 1``'s up projections and
  mean, in one straight-line body. The two chains are independent, so the
  scheduler packs the MXU's dots of one beside the vector work of the other.
  The normed rows and the gate of a row tile wait for the next step in
  two-slot buffers indexed by the tile's parity. Step 0 and the drain step
  ``R`` run only their own half.
* The down projection takes the row tile as the MXU's pushed operand
  (``y^T = [down; inject] . norm^T``, one dot a 1,280-column slab, then
  transposed): fewer pushes than pushing the 384-column weight.
* The inject rows are copied below the down rows ([down; inject; unused] in one
  [384, W] VMEM buffer), so the injection logits come from the down
  projection's MXU pass. The up projection is taken as ``up^T`` [lowrank, W]:
  XLA keeps the [W, lowrank] checkpoint tensor column-major, so the transpose
  is a bitcast where the row-major tensor needs a relayout copy each call, and
  its dots push the weight untransposed. The weights are copied into VMEM once
  a call (no per-call weight concatenation in HBM).
* The combine and norm run on 16-row pieces in 256-lane chunks, the block
  rows' chunk read once for the four streams, the injection broadcast once a
  piece, the sum of squares kept in 128 lanes; the norm weight plus one is
  formed once.
* The mean's sigmoid is ``(1 + tanh(u / 2)) / 2`` with the half folded into
  the gate (halving is exact in bf16 and in the f32 dot) and into the mean's
  scale: one EUP op an element.

Rounding boundaries:

* prologue: h = bf16(f32(residual) + f32(block) * injection_prev[:, b])
* norm = bf16(f32(h) * rsqrt(mean(h^2) + eps) * (1 + f32(w))) per branch (the
  torch path's and the decode kernel's single rounding; ``hc_prefill_fused``
  rounds twice)
* gate = bf16(silu(f32(bf16(bf16(norm @ down^T) / hc))))
* mixed = bf16(sum_b(sigmoid(f32(bf16(gate @ up_b^T))) * f32(norm_b)) / hc)
* injection = 2 * sigmoid(f32(bf16(norm @ inject^T)) / hc), kept in f32

Inputs and outputs are ``hc_prefill_mix``'s, so the two are interchangeable.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

BF16, F32 = jnp.bfloat16, jnp.float32
PIECE = 16  # rows of a combine/norm piece: one bf16 sublane tile
CHUNK = 256  # lanes of a combine/norm chunk
LANES = 128
MIX_COLS = 512  # mixed-row columns an up-projection pass covers
ROW_TILE = 256
DOWN_SLAB = 1280  # contraction columns a down-projection dot covers (swap)


def rup(n: int, k: int) -> int:
    return (n + k - 1) // k * k


def cmp_rows_ok(tokens: int, token_tile: int, partial: bool = False) -> bool:
    """Whether ``tokens`` rows run in ``token_tile`` tiles (with ``partial`` the
    last one may hold any multiple of 16 rows)."""
    if partial:
        return tokens >= token_tile and tokens % PIECE == 0
    return tokens >= token_tile and tokens % token_tile == 0


def _lane_column(values, column: int):
    """``values[:, column:column + 1]`` as a masked lane reduction (exact)."""
    lanes = jax.lax.broadcasted_iota(jnp.int32, values.shape, 1)
    return jnp.sum(jnp.where(lanes == column, values, 0.0), axis=1, keepdims=True)


def _kernel(
    *refs,
    branches: int,
    hidden: int,
    lowrank: int,
    eps: float,
    prologue: bool,
    n_tiles: int,
    swap: bool,
    unroll: bool,
):
    refs = list(refs)
    x_ref = refs.pop(0)
    blk_ref = refs.pop(0) if prologue else None
    pinj_ref = refs.pop(0) if prologue else None
    w_hbm, down_hbm, upt_hbm, injw_hbm = refs[:4]
    refs = refs[4:]
    mixed_ref, inj_ref = refs[:2]
    refs = refs[2:]
    res_ref = refs.pop(0) if prologue else None
    w_vm, w1_vm, dn_vm, upt_vm, xn2, g2, acc_vm, sems = refs

    width = branches * hidden
    rt = xn2.shape[1]
    mix_cols = acc_vm.shape[1]
    inj_tile = lowrank // LANES * LANES
    inj_lane = lowrank - inj_tile
    i = pl.program_id(0)
    copies = [
        pltpu.make_async_copy(w_hbm, w_vm, sems.at[0]),
        pltpu.make_async_copy(down_hbm, dn_vm.at[pl.ds(0, lowrank)], sems.at[1]),
        pltpu.make_async_copy(injw_hbm, dn_vm.at[pl.ds(lowrank, branches)], sems.at[2]),
        pltpu.make_async_copy(upt_hbm, upt_vm, sems.at[3]),
    ]
    # Rows of the combined hyper input: written by the prologue and read back
    # by the norm's second pass.
    src = res_ref if prologue else x_ref

    def piece(r0):
        """Combine and norm of rows ``r0 .. r0 + 16`` into ``xn``'s rows."""
        rows = pl.ds(r0, PIECE)
        nb = hidden // CHUNK
        if prologue:
            inj = pinj_ref[rows, :]
            ib = [
                jnp.broadcast_to(_lane_column(inj, b), (PIECE, CHUNK))
                for b in range(branches)
            ]
        ss = [None] * branches
        for k in range(nb):
            c0 = k * CHUNK
            bk = blk_ref[rows, pl.ds(c0, CHUNK)].astype(F32) if prologue else None
            for b in range(branches):
                cols = pl.ds(b * hidden + c0, CHUNK)
                hb = x_ref[rows, cols]
                if prologue:
                    hb = (hb.astype(F32) + bk * ib[b]).astype(BF16)
                    res_ref[rows, cols] = hb
                for q in range(CHUNK // LANES):
                    vq = hb[:, q * LANES : (q + 1) * LANES].astype(F32)
                    ss[b] = vq * vq if ss[b] is None else ss[b] + vq * vq
        return rows, ss

    def norm_out(xn, rows, ss):
        nb = hidden // CHUNK
        for b in range(branches):
            rs = jax.lax.rsqrt(jnp.sum(ss[b], axis=1, keepdims=True) / hidden + eps)
            rb = jnp.broadcast_to(rs, (PIECE, CHUNK))
            for k in range(nb):
                cols = pl.ds(b * hidden + k * CHUNK, CHUNK)
                v = src[rows, cols].astype(F32)
                xn[rows, cols] = (v * rb * w1_vm[:, cols]).astype(BF16)

    def combine_norm(xn, straight: bool):
        if straight:
            for j in range(rt // PIECE):
                rows, ss = piece(j * PIECE)
                norm_out(xn, rows, ss)
            return

        def body(j, carry):
            rows, ss = piece(pl.multiple_of(j * PIECE, PIECE))
            norm_out(xn, rows, ss)
            return carry

        jax.lax.fori_loop(0, rt // PIECE, body, 0)

    def down_dot(xn):
        """[down; inject; unused] . norm^T of a row tile: f32 [rt, nd]."""
        if not swap:
            return jax.lax.dot_general(
                xn[...], dn_vm[...], (((1,), (1,)), ((), ())), preferred_element_type=F32
            )
        slab = DOWN_SLAB if width % DOWN_SLAB == 0 else width
        yt = None
        for s in range(width // slab):
            cols = pl.ds(s * slab, slab)
            part = jax.lax.dot_general(
                dn_vm[:, cols], xn[:, cols], (((1,), (1,)), ((), ())),
                preferred_element_type=F32,
            )
            yt = part if yt is None else yt + part
        return yt.T

    def half_a(slot, first: bool):
        """Row tile ``i``: combine, norm, down dot, gate (kept) and injection."""
        xn = xn2.at[slot]
        combine_norm(xn, straight=unroll and not first)
        if first:
            copies[1].wait()
            copies[2].wait()
        down = down_dot(xn).astype(BF16)
        z = (down[:, :lowrank] / branches).astype(BF16).astype(F32)
        # silu(z) / 2, exact in bf16: the up dot then gives u / 2.
        g2[slot] = (z * (0.5 * jax.nn.sigmoid(z))).astype(BF16)
        logits = down[:, inj_tile : inj_tile + LANES].astype(F32)
        shape = (rt, branches)
        lanes = jax.lax.broadcasted_iota(jnp.int32, shape, 1)
        inj = jnp.zeros(shape, F32)
        for j in range(branches):
            inj = jnp.where(lanes == j, _lane_column(logits, inj_lane + j), inj)
        inj_ref[...] = 2.0 * jax.nn.sigmoid(inj / branches)

    def half_b(slot):
        """Row tile ``i - 1``: the up dots on the kept gate and the mean."""
        xn = xn2.at[slot]
        gate = g2[slot]
        for c0 in range(0, hidden, mix_cols):
            for b in range(branches):
                start = b * hidden + c0
                u = jnp.dot(gate, upt_vm[:, pl.ds(start, mix_cols)], preferred_element_type=F32)
                # 2 sigmoid(bf16(u)) = 1 + tanh(bf16(u) / 2), u / 2 from the halved gate.
                term = (1.0 + jnp.tanh(u.astype(BF16).astype(F32))) * xn[
                    :, pl.ds(start, mix_cols)
                ].astype(F32)
                if b == 0:
                    acc_vm[...] = term
                else:
                    acc_vm[...] += term
            mixed_ref[:, pl.ds(c0, mix_cols)] = (acc_vm[...] * (0.5 / branches)).astype(BF16)

    @pl.when(i == 0)
    def _first():
        for copy in copies:
            copy.start()
        copies[0].wait()
        w1_vm[...] = jnp.broadcast_to(1.0 + w_vm[...].astype(F32)[None, :], w1_vm.shape)
        half_a(0, first=True)

    @pl.when(i == 1)
    def _wait_up():
        copies[3].wait()

    parity = i % 2

    @pl.when((i > 0) & (i < n_tiles))
    def _body():
        half_b(1 - parity)
        half_a(parity, first=False)

    @pl.when(i == n_tiles)
    def _drain():
        half_b(1 - parity)


def hc_prefill_mix_cmp(
    x,
    weight,
    down,
    up,
    inject,
    *,
    branches: int,
    hidden: int,
    eps: float,
    token_tile: int = ROW_TILE,
    block=None,
    prev_injection=None,
    interpret=False,
    partial=False,
    swap=True,
    unroll=True,
    norm_impl=None,
    scopes=0,
):
    """``hc_prefill_mix``'s contract in the form above. ``norm_impl`` and
    ``scopes`` are accepted for call compatibility and ignored. ``swap`` False
    pushes the weight in the down dot; ``unroll`` False loops over the norm's
    pieces (smaller code, no overlap with the up dots).

    Returns ``(mixed, injection)``, plus the combined hyper input when
    ``block`` is given.
    """
    del norm_impl, scopes
    return _hc_prefill_mix_cmp(
        x,
        weight,
        down,
        up,
        inject,
        block,
        prev_injection,
        branches=branches,
        hidden=hidden,
        eps=eps,
        token_tile=token_tile,
        interpret=interpret,
        partial=partial,
        swap=swap,
        unroll=unroll,
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "branches", "hidden", "eps", "token_tile", "interpret", "partial", "swap", "unroll"
    ),
)
def _hc_prefill_mix_cmp(
    x,
    weight,
    down,
    up,
    inject,
    block,
    prev_injection,
    *,
    branches,
    hidden,
    eps,
    token_tile,
    interpret,
    partial,
    swap,
    unroll,
):
    tokens, width = x.shape
    lowrank = down.shape[0]
    prologue = block is not None
    nd = rup(lowrank + branches, LANES)
    if (
        not cmp_rows_ok(tokens, token_tile, partial)
        or token_tile % PIECE
        or width != branches * hidden
        or hidden % CHUNK
        or lowrank % PIECE
        or lowrank % LANES + branches > LANES
        or weight.shape != (width,)
        or down.shape != (lowrank, width)
        or up.shape != (width, lowrank)
        or inject.shape != (branches, width)
        or (prologue and block.shape != (tokens, hidden))
        or (prologue and prev_injection.shape != (tokens, branches))
        or (prologue and prev_injection.dtype != jnp.float32)
        or any(
            a.dtype != jnp.bfloat16
            for a in (x, weight, down, up, inject) + ((block,) if prologue else ())
        )
    ):
        raise ValueError("HC prefill cmp mix requires BF16 16-row pieces and whole tiles")
    n_tiles = pl.cdiv(tokens, token_tile)
    last = n_tiles - 1
    mix_cols = MIX_COLS if hidden % MIX_COLS == 0 else CHUNK

    def rows(cols):
        return pl.BlockSpec((token_tile, cols), lambda i: (jnp.minimum(i, last), 0))

    hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    in_specs = [rows(width)]
    if prologue:
        in_specs += [rows(hidden), rows(branches)]
    in_specs += [hbm] * 4
    # Row tile r's mixed rows are written at step r + 1.
    out_specs = [
        pl.BlockSpec((token_tile, hidden), lambda i: (jnp.maximum(i - 1, 0), 0)),
        rows(branches),
    ]
    out_shape = [
        jax.ShapeDtypeStruct((tokens, hidden), jnp.bfloat16),
        jax.ShapeDtypeStruct((tokens, branches), jnp.float32),
    ]
    if prologue:
        out_specs.append(rows(width))
        out_shape.append(jax.ShapeDtypeStruct((tokens, width), jnp.bfloat16))
    scratch = [
        pltpu.VMEM((width,), jnp.bfloat16),
        pltpu.VMEM((PIECE, width), jnp.float32),
        pltpu.VMEM((nd, width), jnp.bfloat16),
        pltpu.VMEM((lowrank, width), jnp.bfloat16),
        pltpu.VMEM((2, token_tile, width), jnp.bfloat16),
        pltpu.VMEM((2, token_tile, lowrank), jnp.bfloat16),
        pltpu.VMEM((token_tile, mix_cols), jnp.float32),
        pltpu.SemaphoreType.DMA((4,)),
    ]
    args = [x] + ([block, prev_injection] if prologue else [])
    args += [weight, down, up.T, inject]
    outs = pl.pallas_call(
        functools.partial(
            _kernel,
            branches=branches,
            hidden=hidden,
            lowrank=lowrank,
            eps=eps,
            prologue=prologue,
            n_tiles=n_tiles,
            swap=swap,
            unroll=unroll,
        ),
        grid=(n_tiles + 1,),
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=out_shape,
        scratch_shapes=scratch,
        compiler_params=pltpu.CompilerParams(
            # Weights are fetched at step 0 and the skew carries a row tile's
            # normed rows and gate to the next step: steps run in order.
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=62 << 20,
        ),
        interpret=interpret,
        name=f"qwen4exp_hc_prefill_cmp{'_combine' if prologue else ''}_inj_m{token_tile}",
    )(*args)
    return tuple(outs)
