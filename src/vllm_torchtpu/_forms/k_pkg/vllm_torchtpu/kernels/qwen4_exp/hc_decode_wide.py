# SPDX-License-Identifier: Apache-2.0
"""Decode HyperConnection mix for 160 to 511 rows (q38 HC-HIGHROWS).

``TPU_QWEN4_HC_DECODE_WIDE`` (default 0, off) runs this kernel in place of
``hc_decode.py`` for decode steps of 160 to 511 rows, a multiple of 16; every
other row count keeps ``hc_decode.py``. At those rows ``hc_decode.py`` with
``TPU_QWEN4_HC_DECODE_TILE_ROWS=512`` runs one tile of all the rows: its row
inputs land before the body starts and its outputs leave after it ends, with no
overlap, and its f32 intermediates span the whole tile.

The schedule follows Benliang Wang's ``hc_decode_cmp.py``
(``TPU_QWEN4_HC_DECODE_COMPARE``, 16 to 256 rows; credit): every row and weight
copy is issued by hand in consumption order under a byte budget, and the
combined rows go back to HBM as soon as they are formed. It is laid out so it
fits VMEM up to 511 rows:

* Rows travel in row blocks of at most 128 rows (all streams, one contiguous
  copy per block) ahead of the norm weight, the down rows and the up rows.
* Phase A (vector unit), block by block as its rows land: the previous HC's
  combine and the per-branch RMS norm (mode 1 also forms the injection gate);
  the block's combined rows are written back at once.
* Phase B1 (matrix unit): the down projection over all rows.
* Phase B2: the up projection, the sigmoid-weighted branch mean and the mixed
  rows, one chunk of at most 128 rows at a time, so the f32 intermediates stay
  bounded; each chunk's mixed rows are written back at once.

Modes (static):

* 1, served order: every expression and rounding of ``hc_decode.py`` on the
  same row pieces (32-row pieces with a 16-row tail, the injection gate on the
  vector unit in its summation order, one down projection over all rows,
  ``jax.nn.sigmoid``). It aims to be bit-identical to ``hc_decode.py``; that is
  decided on device.
* 2, cmp math: ``hc_decode_cmp.py``'s math. The inject rows sit below the down
  rows, so the injection logits come from the down projection's matrix pass;
  the down projection is accumulated stream by stream; the sum of squares is
  kept in 128-lane partials over 16-row pieces; the mean's sigmoid is
  ``(1 + tanh(u / 2)) / 2`` with the half folded into the gate. It differs from
  ``hc_decode.py`` at the ulp level and can move a bf16 rounding.

* 3, W3 (q38 HC-HIGHROWS-2): mode 1's math and rounding, every expression
  of ``hc_decode.py`` unchanged, on a row-block software pipeline. The down
  rows are copied right after the first row block (budget ``W3_LOOKAHEAD_MIB``),
  and block i-1's full-K down projection and gate are emitted, unrolled, in one
  region with block i's combine / norm / injection pieces, so the scheduler can
  run the matrix pass beside the vector work. A row split keeps the served
  order (``TPU_QWEN4_HC_DECODE_TILE_ROWS=256`` is bit-identical to 512 on
  device at 288-496 rows); bit identity with ``hc_decode.py`` is decided on
  device.

``timing`` (bench only): ``"dma"`` runs the copies and no compute,
``"compute"`` the compute and no copies.

Schedule v2 (q38 HCK, ``TPU_QWEN4_HC_DECODE_W1_SCHED=1``; mode 1 only, default off;
kernel names gain ``s``), two static switches, both on under the key:

* ``sched_dma`` (data movement only): every copy is started at kernel entry
  (budget ``SCHED_LOOKAHEAD_MIB`` instead of ``lookahead``: today's queue order and
  copies, every wait where it was). No compute changes.
* ``sched_onepass``: phase B2 runs over the fewest row blocks of at most
  ``ONEPASS_ROWS`` rows (one pass up to 320 rows), so each branch's up weights
  stream through the matrix unit once; every expression of mode 1 is unchanged
  (the down projection stays one dot over all rows).
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.qwen4_exp.hc_decode import hc_decode_mix

MIN_TOKENS = 160
# hyperconnection._decode_fused takes the fused decode path below 512 rows.
MAX_TOKENS = 511
ROW_UNIT = 16
BLOCK_ROWS = 128
SERVED_PIECE = 32  # hc_decode.py's _ROW_CHUNK
CMP_PIECE = 16  # hc_decode_cmp.py's PIECE
CHUNK = 256
LANES = 128
MODES = (1, 2, 3)
TIMINGS = ("full", "dma", "compute")
LOOKAHEAD_MIB = 8
# Mode 3: block 0's rows, the down rows and block 1's rows in flight at once.
W3_LOOKAHEAD_MIB = 12
VMEM_LIMIT = 56 << 20
# q38 HCK schedule v2 (TPU_QWEN4_HC_DECODE_W1_SCHED).
SCHED_LOOKAHEAD_MIB = 64
ONEPASS_ROWS = 320
# q38 HCK floor key (TPU_QWEN4_HC_DECODE_WIDE_MIN_ROWS): mode 1 below MIN_TOKENS, down to 64 rows, when every row
# block holds at least two 32-row pieces (mode 1's served-piece loop asserts it).
FLOOR_CHOICES = (64, 80, 96, 112, 128, 144)


def wide_applies(tokens: int, min_rows=None) -> bool:
    """Whether ``tokens`` rows run this kernel (a concrete count).

    ``min_rows`` (q38 HCK floor key, one of ``FLOOR_CHOICES``; mode 1 only) lowers
    the floor from ``MIN_TOKENS``: below it, ``tokens >= min_rows`` and every row
    block must hold at least two 32-row pieces. Unset, MIN_TOKENS-MAX_TOKENS as
    before."""
    if tokens % ROW_UNIT or tokens > MAX_TOKENS:
        return False
    if tokens >= MIN_TOKENS:
        return True
    if min_rows is None:
        return False
    if min_rows not in FLOOR_CHOICES:
        raise ValueError(f"HC decode mix (wide): floor {min_rows!r} not in {FLOOR_CHOICES}")
    return tokens >= min_rows and all(rows // SERVED_PIECE >= 2 for _, rows in row_blocks(tokens))


def row_blocks(tokens: int, max_rows: int = BLOCK_ROWS) -> tuple:
    """``(offset, rows)`` blocks of at most ``max_rows`` rows, as equal as they
    can be in whole 32-row pieces, a 16-row tail (rows % 32 == 16) closing the
    last block. These are hc_decode.py's row pieces at its offsets (448 rows:
    128, 128, 96, 96; 240: 128 and 112)."""
    count = -(-tokens // max_rows)
    pieces, tail = divmod(tokens, SERVED_PIECE)
    base, extra = divmod(pieces, count)
    blocks, offset = [], 0
    for i in range(count):
        rows = (base + (1 if i < extra else 0)) * SERVED_PIECE
        if i == count - 1:
            rows += tail
        blocks.append((offset, rows))
        offset += rows
    return tuple(blocks)


def _lane_column(values, column):
    """``values[:, column:column + 1]`` as a masked lane reduction (exact)."""
    lanes = jax.lax.broadcasted_iota(jnp.int32, values.shape, 1)
    return jnp.sum(jnp.where(lanes == column, values, 0.0), axis=1, keepdims=True)


class _NoCopy:
    """A copy left out by ``timing="compute"``."""

    def start(self):
        pass

    def wait(self):
        pass


def _kernel(
    *refs,
    tokens: int,
    branches: int,
    hidden: int,
    lowrank: int,
    eps: float,
    prologue: bool,
    inject: bool,
    mode: int,
    timing: str,
    lookahead: int,
    fold: bool = False,
    sched_dma: bool = False,
    sched_onepass: bool = False,
):
    compute = timing != "dma"
    # INJ-MXU: the logits come from the down projection, not the vector unit.
    vpu_inject = inject and not fold
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
    res_vm, w_vm, dn_vm, up_vm, xn_vm, gate_vm, mixed_vm = refs[:7]
    refs = refs[7:]
    injw_vm = refs.pop(0) if vpu_inject and mode in (1, 3) else None
    w1_vm = refs.pop(0) if mode == 2 else None
    y_vm = refs.pop(0) if mode == 2 else None
    if prologue:
        blk_vm, pinj_vm = refs[:2]
        refs = refs[2:]
    else:
        blk_vm = pinj_vm = None
    injo_vm = refs.pop(0) if inject else None
    (sems,) = refs

    width = branches * hidden
    blocks = row_blocks(tokens)
    # HCK one-pass: phase B2's row blocks (phase A keeps ``blocks``).
    b2_blocks = row_blocks(tokens, ONEPASS_ROWS) if sched_onepass else blocks
    nt = (((1,), (1,)), ((), ()))

    def stream(b):
        return pl.ds(b * hidden, hidden)

    sem_ids = iter(range(sems.shape[0]))

    def copy(src, dst):
        sem = next(sem_ids)
        if timing == "compute":
            return _NoCopy()
        return pltpu.make_async_copy(src, dst, sems.at[sem])

    w_in = copy(w_hbm, w_vm)
    if inject:
        # Mode 2 and fold: the inject rows go below the down rows
        # ([down; inject; unused] / [down; inject; zero]).
        inj_dst = (
            injw_vm
            if vpu_inject and mode in (1, 3)
            else dn_vm.at[pl.ds(lowrank, branches), :]
        )
        inj_in = copy(injw_hbm, inj_dst)
    pinj_in = copy(pinj_hbm, pinj_vm) if prologue else None
    rows_in, blk_in = [], []
    for r0, rb in blocks:
        rows = pl.ds(r0, rb)
        rows_in.append(copy(res_hbm.at[rows, :], res_vm.at[rows, :]))
        blk_in.append(copy(blk_hbm.at[rows, :], blk_vm.at[rows, :]) if prologue else None)
    dn_in = copy(down_hbm, dn_vm.at[pl.ds(0, lowrank), :])
    up_in = copy(up_hbm, up_vm)

    # Copies in consumption order: a copy starts once the copies started ahead
    # of it and not yet waited hold less than ``lookahead`` MiB, or when it is
    # the one being waited (hc_decode_cmp.py's pump).
    queue = [(w_in, width * 2)]
    if vpu_inject:
        queue.append((inj_in, branches * width * 2))
    if prologue:
        queue.append((pinj_in, tokens * LANES * 4))
    for i, (_, rb) in enumerate(blocks):
        queue.append((rows_in[i], rb * width * 2))
        if prologue:
            queue.append((blk_in[i], rb * hidden * 2))
        if mode == 3 and i == 0:
            # W3: the down rows right after the first block; its full-K dot
            # runs beside block 1's vector work.
            queue.append((dn_in, lowrank * width * 2))
    if mode != 3:
        if fold:
            # Fold: the inject rows are first read by the down projection.
            queue.append((inj_in, branches * width * 2))
        queue.append((dn_in, lowrank * width * 2))
    queue.append((up_in, width * 3 * LANES * 2))
    budget = (W3_LOOKAHEAD_MIB if mode == 3 else lookahead) << 20
    if sched_dma:
        # HCK: every copy in flight from kernel entry (all scratch is allocated).
        budget = SCHED_LOOKAHEAD_MIB << 20
    # Fold: the inject copy (and what follows it) waits until the zero rows
    # under it are stored.
    hold = next(j for j, (q, _) in enumerate(queue) if q is inj_in) if fold else None
    state = dict(started=0, waited=-1, done=set(), hold=hold)

    def pump():
        while state["started"] < len(queue):
            i = state["started"]
            if state["hold"] is not None and i >= state["hold"]:
                break
            held = sum(n for _, n in queue[state["waited"] + 1 : i])
            if i > state["waited"] + 1 and held + queue[i][1] > budget:
                break
            queue[i][0].start()
            state["started"] += 1

    def need(c):
        # A copy is waited once (mode 3 needs the down rows at block 1 and
        # again before the last block's down projection).
        i = next(j for j, (q, _) in enumerate(queue) if q is c)
        if i in state["done"]:
            return
        state["done"].add(i)
        while state["started"] <= i:
            queue[state["started"]][0].start()
            state["started"] += 1
        c.wait()
        state["waited"] = max(state["waited"], i)
        pump()

    pump()
    if fold:
        if compute:
            # The 128-row tile from row ``lowrank`` is zeroed once (an aligned
            # store), then the inject copy fills its first ``branches`` rows.
            dn_vm[pl.ds(lowrank, dn_vm.shape[0] - lowrank), :] = jnp.zeros(
                (dn_vm.shape[0] - lowrank, width), dn_vm.dtype
            )
        state["hold"] = None
        pump()
    need(w_in)
    if vpu_inject and mode in (1, 3):
        need(inj_in)
    if prologue:
        need(pinj_in)
    if compute and mode == 2:
        # The norm weight plus one, f32, broadcast over a piece's 16 rows.
        w1_vm[...] = jnp.broadcast_to(
            1.0 + w_vm[...].astype(jnp.float32)[None, :], (CMP_PIECE, width)
        )

    # ---------------------------------------------------------------- phase A
    def served_rows(offset, rows):
        """hc_decode.py's normalize_rows, unchanged: the combine prologue, the
        per-branch norm and (with inject) the injection gate for ``rows`` rows;
        the combined rows overwrite the residual rows in place."""
        weight = w_vm[...].astype(jnp.float32)
        if prologue:
            block = blk_vm[pl.ds(offset, rows), :].astype(jnp.float32)
            prev_inj = pinj_vm[pl.ds(offset, rows), :]
        inj_terms = [None] * branches
        for b in range(branches):
            start = b * hidden
            if prologue:
                h = (
                    res_vm[pl.ds(offset, rows), pl.ds(start, hidden)].astype(
                        jnp.float32
                    )
                    + block * _lane_column(prev_inj, b)
                ).astype(jnp.bfloat16)
                res_vm[pl.ds(offset, rows), pl.ds(start, hidden)] = h
                values = h.astype(jnp.float32)
            else:
                values = res_vm[pl.ds(offset, rows), pl.ds(start, hidden)].astype(
                    jnp.float32
                )
            variance = jnp.sum(values * values, axis=-1, keepdims=True) / hidden
            normalized = values * jax.lax.rsqrt(variance + eps)
            norm = normalized * (1.0 + weight[None, start : start + hidden])
            norm = norm.astype(jnp.bfloat16)
            xn_vm[pl.ds(offset, rows), pl.ds(start, hidden)] = norm
            if vpu_inject:
                norm32 = norm.astype(jnp.float32)
                for j in range(branches):
                    row = injw_vm[pl.ds(j, 1), pl.ds(start, hidden)]
                    term = norm32 * row.astype(jnp.float32)
                    inj_terms[j] = term if inj_terms[j] is None else inj_terms[j] + term
        if vpu_inject:
            shape = (rows, branches)
            lanes = jax.lax.broadcasted_iota(jnp.int32, shape, 1)
            logits = jnp.zeros(shape, jnp.float32)
            for j in range(branches):
                column = jnp.sum(inj_terms[j], axis=1, keepdims=True)
                logits = jnp.where(lanes == j, column, logits)
            logits = logits.astype(jnp.bfloat16).astype(jnp.float32)
            injo_vm[pl.ds(offset, rows), :] = 2.0 * jax.nn.sigmoid(logits / branches)

    def cmp_piece(r0, b):
        """hc_decode_cmp.py's piece, unchanged: stream ``b`` of the 16 rows
        from ``r0``."""
        prs = pl.ds(r0, CMP_PIECE)
        if prologue:
            ib = jnp.broadcast_to(_lane_column(pinj_vm[prs, :], b), (CMP_PIECE, CHUNK))
        ss = None
        for k in range(hidden // CHUNK):
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
        rb = jnp.broadcast_to(rs, (CMP_PIECE, CHUNK))
        for k in range(hidden // CHUNK):
            cols = pl.ds(b * hidden + k * CHUNK, CHUNK)
            v = res_vm[prs, cols].astype(jnp.float32)
            xn_vm[prs, cols] = (v * rb * w1_vm[:, cols]).astype(jnp.bfloat16)

    def down_block(i):
        """W3: hc_decode.py's down projection and gate, unchanged, on block
        ``i``'s rows (full K; a row split keeps the served order)."""
        rows = pl.ds(*blocks[i])
        down_acc = jax.lax.dot_general(
            xn_vm[rows, :],
            dn_vm[...],
            dimension_numbers=nt,
            preferred_element_type=jnp.float32,
        )
        gate = (down_acc.astype(jnp.bfloat16) / branches).astype(jnp.bfloat16)
        gate_vm[rows, :] = jax.nn.silu(gate.astype(jnp.float32)).astype(jnp.bfloat16)

    h_out = []
    for i, (r0, rb) in enumerate(blocks):
        need(rows_in[i])
        if prologue:
            need(blk_in[i])
        if mode == 3:
            # Block i-1's matrix pass, then block i's served pieces, unrolled
            # in one region (hc_decode_cmp.py's overlap, on row blocks).
            if i >= 1:
                need(dn_in)
                if compute:
                    down_block(i - 1)
            if compute:
                pieces, tail = divmod(rb, SERVED_PIECE)
                for p in range(pieces):
                    served_rows(r0 + p * SERVED_PIECE, SERVED_PIECE)
                if tail:
                    served_rows(r0 + pieces * SERVED_PIECE, tail)
        elif compute and mode == 1:
            pieces, tail = divmod(rb, SERVED_PIECE)
            assert pieces >= 2 and tail in (0, 16), (rb, pieces, tail)

            def served_piece(index, carry, r0=r0):
                served_rows(
                    pl.multiple_of(r0 + index * SERVED_PIECE, SERVED_PIECE), SERVED_PIECE
                )
                return carry

            jax.lax.fori_loop(0, pieces, served_piece, jnp.int32(0))
            if tail:
                served_rows(r0 + pieces * SERVED_PIECE, tail)
        elif compute:
            for b in range(branches):

                def piece(p, carry, r0=r0, b=b):
                    cmp_piece(pl.multiple_of(r0 + p * CMP_PIECE, CMP_PIECE), b)
                    return carry

                jax.lax.fori_loop(0, rb // CMP_PIECE, piece, jnp.int32(0))
        if prologue:
            out = copy(res_vm.at[pl.ds(r0, rb), :], h_hbm.at[pl.ds(r0, rb), :])
            out.start()
            h_out.append(out)

    # ---------------------------------------------------------------- phase B1
    if inject and (mode == 2 or fold):
        need(inj_in)
    need(dn_in)
    if compute and mode == 3:
        down_block(len(blocks) - 1)
    elif compute and mode == 1:
        # hc_decode.py's down projection and gate, unchanged (one dot).
        down_acc = jax.lax.dot_general(
            xn_vm[...],
            dn_vm[...],
            dimension_numbers=nt,
            preferred_element_type=jnp.float32,
        )
        if fold:
            # Fold: the inject rows' columns sit in the last 128-lane tile.
            tile0 = lowrank // LANES * LANES
            lane0 = lowrank - tile0
            inj_tile = down_acc[:, tile0 : tile0 + LANES]
            down_acc = down_acc[:, :lowrank]
        gate = (down_acc.astype(jnp.bfloat16) / branches).astype(jnp.bfloat16)
        gate_vm[...] = jax.nn.silu(gate.astype(jnp.float32)).astype(jnp.bfloat16)
        if fold:
            # The served injection gate on the folded logits: each column
            # extracted exactly (one non-zero term), then bf16, 2 sigmoid(. / 4).
            shape = (tokens, branches)
            lanes = jax.lax.broadcasted_iota(jnp.int32, shape, 1)
            logits = jnp.zeros(shape, jnp.float32)
            for j in range(branches):
                column = _lane_column(inj_tile, lane0 + j)
                logits = jnp.where(lanes == j, column, logits)
            logits = logits.astype(jnp.bfloat16).astype(jnp.float32)
            injo_vm[...] = 2.0 * jax.nn.sigmoid(logits / branches)
    elif compute:
        # hc_decode_cmp.py: the down projection stream by stream, the inject
        # rows' logits in its second 128-lane column tile.
        for c in range(branches):
            part = jax.lax.dot_general(
                xn_vm[:, stream(c)],
                dn_vm[:, stream(c)],
                dimension_numbers=nt,
                preferred_element_type=jnp.float32,
            )
            if c == 0:
                y_vm[...] = part
            else:
                y_vm[...] += part
        # Gate on the low-rank columns, halved (exact): the up dot then gives
        # u / 2, and 2 sigmoid(bf16(u)) = 1 + tanh(bf16(u / 2)).
        y = y_vm[:, pl.ds(0, lowrank)]
        z = y.astype(jnp.bfloat16).astype(jnp.float32) / branches
        z = z.astype(jnp.bfloat16).astype(jnp.float32)
        gate_vm[...] = (z * (0.5 * jax.nn.sigmoid(z))).astype(jnp.bfloat16)
        if inject:
            tile0 = lowrank // LANES * LANES
            lane0 = lowrank - tile0
            logits = y_vm[:, pl.ds(tile0, LANES)]
            gates = 2.0 * jax.nn.sigmoid(
                logits.astype(jnp.bfloat16).astype(jnp.float32) / branches
            )
            shape = (tokens, branches)
            lanes = jax.lax.broadcasted_iota(jnp.int32, shape, 1)
            injection = jnp.zeros(shape, jnp.float32)
            for j in range(branches):
                injection = jnp.where(
                    lanes == j, _lane_column(gates, lane0 + j), injection
                )
            injo_vm[...] = injection
    if inject:
        injo_out = copy(injo_vm, injo_hbm)
        injo_out.start()

    # ---------------------------------------------------------------- phase B2
    need(up_in)
    mixed_out = []
    for c0, rc in b2_blocks:
        rows = pl.ds(c0, rc)
        if compute and mode in (1, 3):
            # hc_decode.py's up projection and branch mean, unchanged, on rc rows.
            mixed = None
            for b in range(branches):
                start = b * hidden
                logits = (
                    jax.lax.dot_general(
                        gate_vm[rows, :],
                        up_vm[pl.ds(start, hidden), :],
                        dimension_numbers=nt,
                        preferred_element_type=jnp.float32,
                    )
                    .astype(jnp.bfloat16)
                    .astype(jnp.float32)
                )
                weighted = jax.nn.sigmoid(logits) * xn_vm[
                    rows, pl.ds(start, hidden)
                ].astype(jnp.float32)
                mixed = weighted if mixed is None else mixed + weighted
            mixed_vm[rows, :] = (mixed / branches).astype(jnp.bfloat16)
        elif compute:
            # hc_decode_cmp.py's up projection (one dot) and mean, on rc rows.
            u_all = jax.lax.dot_general(
                gate_vm[rows, :],
                up_vm[...],
                dimension_numbers=nt,
                preferred_element_type=jnp.float32,
            )
            acc = None
            for b in range(branches):
                u = u_all[:, b * hidden : (b + 1) * hidden]
                u = u.astype(jnp.bfloat16).astype(jnp.float32)
                term = (1.0 + jnp.tanh(u)) * xn_vm[rows, stream(b)].astype(jnp.float32)
                if b == 0:
                    acc = term
                elif b < branches - 1:
                    acc = acc + term
                else:
                    mixed_vm[rows, :] = ((acc + term) * (0.5 / branches)).astype(
                        jnp.bfloat16
                    )
        out = copy(mixed_vm.at[rows, :], mixed_hbm.at[rows, :])
        out.start()
        mixed_out.append(out)

    for out in h_out:
        out.wait()
    if inject:
        injo_out.wait()
    for out in mixed_out:
        out.wait()


def sem_count(tokens: int, prologue: bool, inject: bool) -> int:
    """DMA semaphores the kernel takes: one per copy."""
    blocks = len(row_blocks(tokens))
    count = 3 + blocks * 2  # w, down, up; rows in and mixed out per block
    if inject:
        count += 2  # inject rows in, injection out
    if prologue:
        count += 1 + blocks * 2  # prev injection; block rows in, h out per block
    return count


def hc_decode_mix_wide(
    x,
    weight,
    down,
    up,
    inject_weight=None,
    *,
    branches: int,
    hidden: int,
    eps: float,
    mode: int,
    block=None,
    prev_injection=None,
    interpret=False,
    timing: str = "full",
    lookahead: int = LOOKAHEAD_MIB,
    fold: bool = False,
    sched_dma: bool = False,
    sched_onepass: bool = False,
    min_rows=None,
):
    """``hc_decode.hc_decode_mix`` for ``wide_applies`` row counts.

    Same arguments and returns: ``(mixed, injection)`` (``injection`` None
    without ``inject_weight``), plus the combined hyper input with ``block``.

    ``fold`` (q38 INJ-MXU, ``TPU_QWEN4_HC_DECODE_INJECT_FOLD``; mode 1 only, and
    only calls with ``inject_weight`` change; kernel name ``..._w1f``): mode 1
    with the injection logits taken from the single down projection. The inject
    rows are copied below the down rows (rows ``lowrank`` to ``lowrank +
    branches``, the rest of the 128-row tile zero, as
    ``hc_prefill_fused.extend_down`` and mode 2 lay them out), the vector-unit
    injection terms are dropped, and after the down projection each logit column
    is extracted exactly from its 128-lane tile with ``_lane_column``; then the
    served bf16 rounding and ``2 sigmoid(. / branches)``. The logits' f32
    summation order changes (numerics-changing, bounded by
    reports/prereg/q38_injmxu_bound_LOCKED.md); the gate path, mixed and the
    combined rows keep mode 1's expressions.

    ``sched_dma`` / ``sched_onepass`` (q38 HCK, ``TPU_QWEN4_HC_DECODE_W1_SCHED``;
    mode 1 only; kernel name gains ``s``): see the module docstring.

    ``min_rows`` (q38 HCK floor key, ``TPU_QWEN4_HC_DECODE_WIDE_MIN_ROWS``; mode 1
    only): accept the row counts ``wide_applies(tokens, min_rows)`` admits below
    MIN_TOKENS. It changes only the accepted range, never the kernel body.
    """
    return _hc_decode_mix_wide(
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
        mode=mode,
        interpret=interpret,
        timing=timing,
        lookahead=lookahead,
        fold=fold,
        sched_dma=sched_dma,
        sched_onepass=sched_onepass,
        min_rows=min_rows,
    )


def kernel_name(
    prologue: bool, mode: int, timing: str = "full", fold: bool = False, sched: bool = False
) -> str:
    """The Pallas call's name: the served kernels' names plus a suffix, so
    profile classifiers that match the served prefixes keep these rows."""
    suffix = "" if timing == "full" else f"_{timing}"
    tag = f"{mode}f" if fold else f"{mode}"
    if sched:
        tag += "s"
    return f"qwen4exp_hc_decode{'_combine' if prologue else ''}_w{tag}{suffix}"


@functools.partial(
    jax.jit,
    static_argnames=(
        "branches",
        "hidden",
        "eps",
        "mode",
        "interpret",
        "timing",
        "lookahead",
        "fold",
        "sched_dma",
        "sched_onepass",
        "min_rows",
    ),
)
def _hc_decode_mix_wide(
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
    mode,
    interpret,
    timing,
    lookahead,
    fold=False,
    sched_dma=False,
    sched_onepass=False,
    min_rows=None,
):
    tokens, width = x.shape
    lowrank = down.shape[0]
    prologue = block is not None
    inject = inject_weight is not None
    if fold and mode != 1:
        raise ValueError(f"HC decode mix (wide): fold needs mode 1, got mode {mode}")
    if (sched_dma or sched_onepass) and mode != 1:
        raise ValueError(f"HC decode mix (wide): the HCK schedule needs mode 1, got mode {mode}")
    if min_rows is not None and mode != 1:
        raise ValueError(f"HC decode mix (wide): the HCK floor needs mode 1, got mode {mode}")

    # Without the injection gate there is nothing to fold: mode 1 unchanged.
    fold = bool(fold) and inject
    if (
        mode not in MODES
        or timing not in TIMINGS
        or not wide_applies(tokens, min_rows)
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
        raise ValueError(
            "HC decode mix (wide) requires BF16 rows that are a multiple of 16 "
            f"in [{MIN_TOKENS if min_rows is None else min_rows}, {MAX_TOKENS}] and mode 1, 2 or 3"
        )

    if sched_dma or sched_onepass:
        # q38 HCK trace-time engagement line (prints when this shape is traced, i.e. on a cold compile; the
        # key's runtime line in vllm_adapter prints at op build): the served op compiled the schedule kernel.
        b2 = len(row_blocks(tokens, ONEPASS_ROWS) if sched_onepass else row_blocks(tokens))
        print(
            f"q38 HCK W1_SCHED traced: rows {tokens}, kernel {kernel_name(prologue, mode, timing, fold, True)}, "
            f"sched_dma {int(bool(sched_dma))}, sched_onepass {int(bool(sched_onepass))}, b2 blocks {b2}",
            flush=True,
        )
    if tokens < MIN_TOKENS:
        # q38 HCK floor key trace-time engagement line (cold compile): the wide kernel below MIN_TOKENS.
        print(
            f"q38 HCK WIDE_MIN_ROWS traced: rows {tokens}, kernel "
            f"{kernel_name(prologue, mode, timing, fold, bool(sched_dma or sched_onepass))}, floor {min_rows}, "
            f"blocks {len(row_blocks(tokens))}",
            flush=True,
        )
    any_space = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    n_in = (3 if prologue else 1) + (4 if inject else 3)
    out_shape = [jax.ShapeDtypeStruct((tokens, hidden), jnp.bfloat16)]
    if inject:
        out_shape.append(jax.ShapeDtypeStruct((tokens, branches), jnp.float32))
    if prologue:
        out_shape.append(jax.ShapeDtypeStruct((tokens, width), jnp.bfloat16))
    if mode in (1, 3) and not fold:
        down_rows = lowrank
    else:
        # [down; inject; unused], a whole number of 128-row tiles.
        down_rows = -(-(lowrank + (branches if inject else 0)) // LANES) * LANES
    scratch = [
        pltpu.VMEM((tokens, width), jnp.bfloat16),  # res_vm (rows in, then h)
        pltpu.VMEM((width,), jnp.bfloat16),  # w_vm
        pltpu.VMEM((down_rows, width), jnp.bfloat16),  # dn_vm
        pltpu.VMEM((width, lowrank), jnp.bfloat16),  # up_vm
        pltpu.VMEM((tokens, width), jnp.bfloat16),  # xn_vm
        pltpu.VMEM((tokens, lowrank), jnp.bfloat16),  # gate_vm
        pltpu.VMEM((tokens, hidden), jnp.bfloat16),  # mixed_vm
    ]
    if inject and mode in (1, 3) and not fold:
        scratch.append(pltpu.VMEM((branches, width), jnp.bfloat16))  # injw_vm
    if mode == 2:
        scratch.append(pltpu.VMEM((CMP_PIECE, width), jnp.float32))  # w1_vm
        scratch.append(pltpu.VMEM((tokens, down_rows), jnp.float32))  # y_vm
    if prologue:
        scratch += [
            pltpu.VMEM((tokens, hidden), jnp.bfloat16),  # blk_vm
            pltpu.VMEM((tokens, branches), jnp.float32),  # pinj_vm
        ]
    if inject:
        scratch.append(pltpu.VMEM((tokens, branches), jnp.float32))  # injo_vm
    scratch.append(pltpu.SemaphoreType.DMA((sem_count(tokens, prologue, inject),)))
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
            timing=timing,
            lookahead=lookahead,
            fold=fold,
            **(
                dict(sched_dma=sched_dma, sched_onepass=sched_onepass)
                if sched_dma or sched_onepass
                else {}
            ),
        ),
        in_specs=[any_space] * n_in,
        out_specs=[any_space] * len(out_shape),
        out_shape=out_shape,
        scratch_shapes=scratch,
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=VMEM_LIMIT),
        interpret=interpret,
        name=(
            kernel_name(prologue, mode, timing, fold, True)
            if sched_dma or sched_onepass
            else kernel_name(prologue, mode, timing, fold)
        ),
    )(*args)
    mixed = outs[0]
    injection = outs[1] if inject else None
    if prologue:
        return mixed, injection, outs[-1]
    return mixed, injection


def hc_decode_mix_wide_auto(
    x,
    weight,
    down,
    up,
    inject_weight=None,
    *,
    branches: int,
    hidden: int,
    eps: float,
    mode: int,
    block=None,
    prev_injection=None,
    max_tile=0,
    interpret=False,
    fold=False,
    sched=False,
    min_rows=None,
):
    """``hc_decode_mix_wide`` where ``wide_applies``, else ``hc_decode_mix``.

    The choice is made on the concrete row count of the traced bucket, so the
    torch-side gate (``decode_token_tile``) is unchanged. ``fold`` (INJ-MXU)
    reaches only the wide kernel; other row counts keep ``hc_decode_mix``.
    ``sched`` (q38 HCK, TPU_QWEN4_HC_DECODE_W1_SCHED; mode 1) turns on both
    schedule-v2 switches of the wide kernel; other row counts are unchanged.
    ``min_rows`` (q38 HCK floor key; mode 1) lowers the wide floor; it is passed
    on only below MIN_TOKENS, so calls at MIN_TOKENS and above are unchanged.
    """
    kwargs = dict(
        branches=branches,
        hidden=hidden,
        eps=eps,
        block=block,
        prev_injection=prev_injection,
        interpret=interpret,
    )
    if wide_applies(x.shape[0], min_rows):
        if sched:
            kwargs.update(sched_dma=True, sched_onepass=True)
        if x.shape[0] < MIN_TOKENS:
            kwargs.update(min_rows=min_rows)
        return hc_decode_mix_wide(
            x, weight, down, up, inject_weight, mode=mode, fold=fold, **kwargs
        )
    return hc_decode_mix(
        x, weight, down, up, inject_weight, max_tile=max_tile, **kwargs
    )
