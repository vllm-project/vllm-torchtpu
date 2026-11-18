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
"""DCP merge of attention partials as a reduce-scatter by recursive halving,
merging over the log-sum-exps at every step (one Pallas kernel).

Same contract as ``lse_merge.small_lse_merge`` (see there for the merge
itself): every core holds a partial ``o [T, H, D]`` and its log-sum-exp
``lse [T, H]`` for all heads, and core r returns heads [r * hl, (r + 1) * hl)
of the merge, head-major.

Why a second kernel: ``lse_merge`` exchanges the partials in one all-to-all.
On a 2x2x2 tpu7x slice (no wraparound) an all-to-all forwards each byte over
1.71 links on average; for a 4 MiB partial per core it measured 59-61 us per
call. Here every core, at each of ``log2(n)`` steps, sends half of the head
blocks it still holds to the one core whose label differs from its own in one
bit (one link away), receives that core's copy of the half it keeps, and
merges the two with the log-sum-exp rule, so the data halves at every step
and every byte crosses one link: 32.6-32.9 us per call at the same shape
(j-9fc7b5b5).

Labels: ``labels[r]`` is the hypercube label of mesh index r. Two cores whose
labels differ in one bit must be one link apart. ``hypercube_labels`` derives
them from each core's chip coordinates and core-on-chip index: one bit per
chip axis of extent 2 and one bit for the core on the chip.

Parts: the T tokens split into ``parts`` chunks. All parts process the
on-chip core bit first (``first_bit``), then the chip axes in an order rotated
by the part index, so that at one step different parts load different links.
Within part p the head blocks are kept in "slot" order: slot s holds the block
of the core whose label, with its bits read in part p's order (first processed
bit most significant), is s. The blocks a core holds before step k are then
one contiguous slot range.

Merge of two partials (o1, l1), (o2, l2) of one (token, head):
L = log(exp(l1) + exp(l2)), o = exp(l1 - L) o1 + exp(l2 - L) o2, where a term
with weight 0 contributes 0 (its partial may be NaN). Repeated over the steps
this is the merge over all cores. The log-sum-exps travel as float32 rows
(heads in sublanes, tokens in lanes); the transposes into and out of that
layout are matmuls against an identity matrix with the float32 operand split
into three bfloat16 terms (exact to about 2**-24 relative).

Precision: the merged partial is rounded to the partial's dtype (bf16) after
every step, so a result carries up to log2(n) bf16 roundings. The
two-collective merge it replaces rounds the weighted partial to bf16 and sums
in bf16 inside XLA's reduce-scatter; at n=16 this kernel's error measured
below that one (err/tol 24-42 vs 36-56 against a float64 oracle).

Example (GLM-5.2 decode, DCP 16 on 2x2x2, a 64-token bucket, 4 parts of 16
tokens): per part, step 0 sends 8 head blocks of bf16 [4, 16, 512] (512 KB) to
the other core of the chip, step 1 sends 4 blocks over one chip axis, step 2
two blocks, step 3 one block; each core returns bf16 [4, 64, 512].
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.small_collectives.common import LANES

# Barrier semaphore group, distinct from every other collective kernel in the
# tree (0, 5, 7, 9, 11, 13, 15, 17, 19, 21).
COLLECTIVE_ID = 23
# The input's -inf (and NaN) log-sum-exp is replaced by this finite value so
# that it can pass through the transposing matmuls; any value below VALID
# (half of it, out of reach of matmul rounding) means "no keys".
EMPTY = -1e30
VALID = EMPTY / 2
SUBLANES = 8
# Test-only faults: the partner's half never merged in, the wrong half kept,
# weights not divided by their sum, part p merged with part 0's arrivals.
FAULTS = ("keep_own", "wrong_half", "unnormalized", "stale_part")


def hypercube_labels(places):
    """Labels for ``small_lse_merge_tree`` from (chip coords, core on chip) per
    mesh index (``small_all_reduce_op._chip_places``), or None when the cores
    do not form a hypercube: every chip axis of extent 1 or 2, at most 2 cores
    per chip. Bit order: the chip axes of extent 2 in coordinate order, then
    the core. Returns (labels, core_bit or None)."""
    chips = [tuple(c) for c, _ in places]
    cores = [int(k) for _, k in places]
    dims = len(chips[0])
    extents = [len({c[d] for c in chips}) for d in range(dims)]
    if any(e > 2 for e in extents) or len(set(cores)) > 2:
        return None
    axes = [d for d in range(dims) if extents[d] == 2]
    lows = [min(c[d] for c in chips) for d in range(dims)]
    core_bit = len(axes) if len(set(cores)) == 2 else None
    labels = []
    for c, k in zip(chips, cores):
        lab = sum((c[d] - lows[d]) << i for i, d in enumerate(axes))
        if core_bit is not None:
            lab |= (k - min(cores)) << core_bit
        labels.append(lab)
    if sorted(labels) != list(range(len(places))):
        return None
    return tuple(labels), core_bit


def default_orders(bits, parts, first=None):
    """Bit order per part: ``first`` (e.g. the on-chip core bit) first, then
    the other bits rotated by the part index."""
    rest = [b for b in range(bits) if b != first]
    orders = []
    for p in range(parts):
        r = p % len(rest)
        orders.append(tuple(([first] if first is not None else []) + rest[r:] + rest[:r]))
    return tuple(orders)


def _split3(x):
    """x (f32) as three bf16 terms whose f32 sum is x to ~2**-24 relative."""
    hi = x.astype(jnp.bfloat16)
    r = x - hi.astype(jnp.float32)
    mid = r.astype(jnp.bfloat16)
    lo = (r - mid.astype(jnp.float32)).astype(jnp.bfloat16)
    return hi, mid, lo


def _nt(a_bf16, b, dims=(((1,), (1,)), ((), ()))):
    """a @ b.T in f32 with b split into three bf16 terms (a is exact in bf16)."""
    out = None
    for term in _split3(b):
        y = lax.dot_general(a_bf16, term, dims, preferred_element_type=jnp.float32)
        out = y if out is None else out + y
    return out


def small_lse_merge_tree(
    output,
    lse,
    *,
    num_cores,
    axis,
    labels,
    rank=None,
    parts=4,
    orders=None,
    first_bit=None,
    collective_id=COLLECTIVE_ID,
    interpret=None,
    _fault=None,
):
    """Merge of the DCP partials of ``num_cores`` cores, inside a shard_map
    over mesh axis ``axis``: core r returns its heads, head-major.

    Args:
      output: [T, H, D] (bf16 in production; any float), this core's partial
        for all heads. T / parts a multiple of 16 rows and at most 128.
      lse: [T, H] f32, its log-sum-exp; -inf or NaN where the core saw no keys.
      num_cores: cores on ``axis``, a power of 2.
      axis: the mesh axis, in DCP rank order.
      labels: hypercube label of each mesh index (see the module docstring).
      rank: int32 [1], this core's index on ``axis`` as data, or None.
      parts: token chunks with their own bit orders (pipelined).
      orders: bit order per part; None for ``default_orders(bits, parts,
        first_bit)``.
      first_bit: the label bit every part processes first (the on-chip core).
      interpret: passed to ``pl.pallas_call``.
      _fault: tests only, one of FAULTS: a deliberately wrong kernel that the
        accuracy test must reject.

    Returns:
      [H / num_cores, T, D] in ``output.dtype``.
    """
    if _fault is not None and _fault not in FAULTS:
        raise ValueError(f"unknown fault {_fault}")
    n = num_cores
    bits = n.bit_length() - 1
    if 1 << bits != n or sorted(labels) != list(range(n)):
        raise ValueError(f"labels {labels} are not a bijection onto 0..{n - 1}")
    tokens, heads, dim = output.shape
    hl = heads // n
    tc = tokens // parts
    if tokens % parts or tc % 16 or tc > LANES or heads % n or dim % LANES:
        raise ValueError(f"no layout for {output.shape} in {parts} parts over {n}")
    orders = orders or default_orders(bits, parts, first_bit)
    assert len(orders) == parts and all(sorted(o) == list(range(bits)) for o in orders)
    data_rank = rank is not None
    inv = {lab: r for r, lab in enumerate(labels)}

    def plabel(lab, order):
        return sum(((lab >> b) & 1) << (bits - 1 - k) for k, b in enumerate(order))

    # slot_owner[p][s]: mesh index whose block sits in slot s of part p.
    slot_owner = [[0] * n for _ in range(parts)]
    for p in range(parts):
        for r in range(n):
            slot_owner[p][plabel(labels[r], orders[p])] = r
    hp = max(SUBLANES, hl)  # lse rows per slot, padded to a sublane tile

    def kernel(*refs):
        refs = list(refs)
        me = refs.pop(0)[0] if data_rank else lax.axis_index(axis)
        (x_hbm, lse_ref, o_ref, xbuf, cur_o, cur_l, recv_o, recv_l,
         in_sems, send_sem, recv_sems) = refs

        def lookup(table):
            """table[me] for a static per-mesh-index table, as a scalar."""
            v = jnp.int32(table[0])
            for r in range(1, n):
                v = jnp.where(me == r, jnp.int32(table[r]), v)
            return v

        myp = [lookup([plabel(labels[r], orders[p]) for r in range(n)]) for p in range(parts)]
        partner = [[lookup([inv[labels[r] ^ (1 << orders[p][k])] for r in range(n)])
                    for k in range(bits)] for p in range(parts)]

        def dev(index):
            return {axis: index} if data_rank else (index,)

        def in_copy(p):
            return pltpu.make_async_copy(
                x_hbm.at[pl.ds(p * tc, tc)], xbuf.at[p], in_sems.at[p])

        for p in range(parts):
            in_copy(p).start()
        barrier = pltpu.get_barrier_semaphore()
        for j in range(n):
            pl.semaphore_signal(barrier, 1, device_id=dev(jnp.int32(j)),
                                device_id_type=pl.DeviceIdType.MESH)

        # lse [T, H] -> per part [H, 128] (heads in rows, tokens in lanes), by
        # an identity matmul: lseT[h, t] = sum_j I[h, j] lse[t, j].
        lse_all = lse_ref[...]
        lse_all = jnp.where(lse_all > EMPTY, lse_all, EMPTY)  # -inf, NaN -> EMPTY
        eye_h = (lax.broadcasted_iota(jnp.int32, (heads, heads), 0)
                 == lax.broadcasted_iota(jnp.int32, (heads, heads), 1)).astype(jnp.bfloat16)

        def relayout(p):
            in_copy(p).wait()
            xt = pltpu.einshape("thd->htd", xbuf[p])  # [H, tc, D]
            lp = lse_all[p * tc:(p + 1) * tc]  # [tc, H]
            lp = jnp.concatenate([lp, jnp.full((LANES - tc, heads), EMPTY, jnp.float32)], 0)
            lt = _nt(eye_h, lp)  # [H, 128]
            for s in range(n):
                r = slot_owner[p][s]
                cur_o[p, s] = xt[r * hl:(r + 1) * hl]
                block = lt[r * hl:(r + 1) * hl]
                if hp > hl:
                    block = jnp.concatenate(
                        [block, jnp.full((hp - hl, LANES), EMPTY, jnp.float32)], 0)
                cur_l[p, s] = block

        relayout(0)
        pl.semaphore_wait(barrier, n)
        for p in range(1, parts):
            relayout(p)

        def send(p, k):
            half = n >> (k + 1)
            held = (myp[p] >> (bits - k)) << (bits - k)
            mybit = (myp[p] >> (bits - 1 - k)) & 1
            start = held + (1 - mybit) * half
            for src, dst in ((cur_o, recv_o), (cur_l, recv_l)):
                pltpu.make_async_remote_copy(
                    src_ref=src.at[p, pl.ds(start, half)],
                    dst_ref=dst.at[p, pl.ds(n - (n >> k), half)],
                    send_sem=send_sem,
                    recv_sem=recv_sems.at[p, k],
                    device_id=dev(partner[p][k]),
                    device_id_type=pl.DeviceIdType.MESH,
                ).start()

        eye_t = (lax.broadcasted_iota(jnp.int32, (tc, LANES), 0)
                 == lax.broadcasted_iota(jnp.int32, (tc, LANES), 1)).astype(jnp.bfloat16)

        def merge(p, k):
            half = n >> (k + 1)
            off = n - (n >> k)  # this step's rows in recv_o / recv_l
            for dst in (recv_o, recv_l):
                pltpu.make_async_copy(dst.at[p, pl.ds(off, half)],
                                      dst.at[p, pl.ds(off, half)],
                                      recv_sems.at[p, k]).wait()
            held = (myp[p] >> (bits - k)) << (bits - k)
            mybit = (myp[p] >> (bits - 1 - k)) & 1
            if _fault == "wrong_half":
                mybit = 1 - mybit
            keep = held + mybit * half
            src_p = 0 if _fault == "stale_part" else p
            for i in range(half):
                s = keep + i
                l1 = cur_l[p, s]  # [hp, 128]
                l2 = recv_l[src_p, off + i]
                m = jnp.maximum(l1, l2)
                e1 = jnp.where(l1 > VALID, jnp.exp(l1 - m), 0.0)
                e2 = jnp.where(l2 > VALID, jnp.exp(l2 - m), 0.0)
                tot = e1 + e2
                safe = jnp.where(tot > 0, tot, 1.0)
                w1, w2 = (e1, e2) if _fault == "unnormalized" else (e1 / safe, e2 / safe)
                if _fault == "keep_own":
                    w1, w2 = jnp.where(e1 > 0, 1.0, 0.0), jnp.zeros_like(e2)
                cur_l[p, s] = jnp.where(tot > 0, m + jnp.log(safe), EMPTY)
                c1 = _nt(eye_t, w1)  # [tc, hp]: c1[t, h] = w1[h, t]
                c2 = _nt(eye_t, w2)
                o1 = cur_o[p, s]  # [hl, tc, D]
                o2 = recv_o[src_p, off + i]
                rows = []
                for h in range(hl):
                    a, b = c1[:, h:h + 1], c2[:, h:h + 1]
                    y = (jnp.where(a > 0, o1[h].astype(jnp.float32) * a, 0.0)
                         + jnp.where(b > 0, o2[h].astype(jnp.float32) * b, 0.0))
                    rows.append(y.astype(cur_o.dtype)[None])
                cur_o[p, s] = jnp.concatenate(rows, 0)

        for k in range(bits):
            for p in range(parts):
                if k:
                    merge(p, k - 1)
                send(p, k)
        for p in range(parts):
            merge(p, bits - 1)
            o_ref[:, pl.ds(p * tc, tc), :] = cur_o[p, myp[p]]
        # every byte this core sent: sum over steps of half blocks, o and lse
        for p in range(parts):
            for k in range(bits):
                half = n >> (k + 1)
                off = n - (n >> k)
                for dst in (recv_o, recv_l):
                    pltpu.make_async_copy(dst.at[p, pl.ds(off, half)],
                                          dst.at[p, pl.ds(off, half)], send_sem).wait()

    scalars, in_specs = [], []
    if data_rank:
        scalars.append(jnp.reshape(rank, (-1,))[:1].astype(jnp.int32))
        in_specs.append(pl.BlockSpec(memory_space=pltpu.SMEM))
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    in_specs += [pl.BlockSpec(memory_space=pl.ANY), vmem]
    scratch = [
        pltpu.VMEM((parts, tc, heads, dim), output.dtype),  # xbuf
        pltpu.VMEM((parts, n, hl, tc, dim), output.dtype),  # cur_o, slot order
        pltpu.VMEM((parts, n, hp, LANES), jnp.float32),  # cur_l
        pltpu.VMEM((parts, n - 1, hl, tc, dim), output.dtype),  # recv_o, all steps
        pltpu.VMEM((parts, n - 1, hp, LANES), jnp.float32),  # recv_l
        pltpu.SemaphoreType.DMA((parts,)),
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA((parts, bits)),
    ]
    params = {"collective_id": collective_id}
    if interpret is None:
        params["vmem_limit_bytes"] = pltpu.get_tpu_info().vmem_capacity_bytes * 3 // 4
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((hl, tokens, dim), output.dtype),
        in_specs=in_specs,
        out_specs=vmem,
        scratch_shapes=scratch,
        compiler_params=pltpu.CompilerParams(**params),
        interpret=interpret,
        name=f"small_lse_merge_tree_n{n}_t{tokens}_p{parts}",
    )(*scalars, output, lse)
