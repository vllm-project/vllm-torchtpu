# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Decode replica slots for the GLM-5.2 megakernel (GLM5_MK_DECODE_REPLICAS=k).

A decode step's time follows the busiest rank's distinct-expert visits. Each rank also holds k
copies of other ranks' experts, one per "slot". After routing, which every rank computes
identically, the kernel runs a deterministic greedy. It moves a hit expert from its owner onto a
slot holder when that leaves the holder with fewer visits than the owner. The same token rows are
multiplied either way, so the combine carries the same bytes; only the f32 summation order of
the reduce-scatter changes.

Everything here is host-side and in kernel expert index space: kernel expert j lives on mesh
index j // ER at local row j % ER. The megakernel's router columns are permuted so that is true
after any EPLB placement.

- `place`: the popularity placement. Take the most-likely-hit experts first, LPT onto the
  least-loaded rank that is not the owner. This is probe 31's "pop" (replicas.py), which gave
  1.28 ms/step modelled at R=16, k=2.
- `table`: the smalls rows the kernel reads (rep_e, rep_s).
- `gather_plans` and `make_gather_fn`: fill the slot tensors from the owners' shards. There is
  one ragged_all_to_all per slot index q, and each (source, holder) pair moves at most one expert
  row. It is read straight out of the source shard, so there is no send-side staging copy.
"""

from __future__ import annotations

import numpy as np

# Module-level on purpose: sharded_jax_callable resolves the string annotations of the gather
# function against these globals (as tpu_eplb.make_exchange_fn does).
import jax


def hit_probability(load: np.ndarray | None, E: int, ntok: int) -> np.ndarray:
    """Per-expert load (pairs, any scale) -> P(hit in a step of ntok tokens), probe 31's
    q = 1 - (1 - p)^ntok with p the expert's load share. None / all-zero -> uniform."""
    if load is None or float(np.sum(load)) <= 0:
        return np.full(E, 1.0 - (1.0 - 1.0 / E) ** ntok)
    p = np.asarray(load, np.float64) / float(np.sum(load))
    return 1.0 - (1.0 - p) ** ntok


def place(q: np.ndarray, R: int, ER: int, k: int) -> list[tuple[int, int]]:
    """Kernel-space hit probability [E] -> replica entries [(slot, expert)] in greedy order,
    slot = holder * k + index on the holder. At most R * k entries, holder != owner.
    Deterministic: stable order, ties to the lowest rank."""
    E = R * ER
    owner = np.arange(E) // ER
    load = np.array([q[owner == r].sum() for r in range(R)], np.float64)
    used = np.zeros(R, np.int64)
    out: list[tuple[int, int]] = []
    for e in np.argsort(-q, kind="stable"):
        if len(out) == R * k:
            break
        cand = [r for r in range(R) if r != owner[e] and used[r] < k]
        if not cand:
            continue
        r = min(cand, key=lambda r_: (load[r_], r_))
        out.append((int(r * k + used[r]), int(e)))
        used[r] += 1
        load[r] += q[e] / 2
        load[owner[e]] -= q[e] / 2
    return out


def table(entries: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    """-> (rep_e, rep_s) f32 [128], entry i in lane i, -1 = empty (the smalls fields)."""
    assert len(entries) <= 64, len(entries)
    rep_e = np.full(128, -1.0, np.float32)
    rep_s = np.full(128, -1.0, np.float32)
    for i, (s, e) in enumerate(entries):
        rep_e[i] = e
        rep_s[i] = s
    return rep_e, rep_s


def slot_experts(entries: list[tuple[int, int]], R: int, k: int) -> np.ndarray:
    """[R * k] kernel expert held in each slot, -1 empty."""
    out = np.full(R * k, -1, np.int64)
    for s, e in entries:
        out[s] = e
    return out


def greedy(hit: np.ndarray, entries: list[tuple[int, int]], R: int, ER: int, k: int):
    """Host model of the kernel's assignment. hit bool [E] -> (loads [R], moved [len(entries)])."""
    load = np.bincount(np.flatnonzero(hit) // ER, minlength=R)
    code = hit.astype(np.int64).copy()
    moved = np.zeros(len(entries), np.int64)
    for _ in range(3):
        for i, (s, e) in enumerate(entries):
            r, a = s // k, e // ER
            if code[e] == 1 and load[r] + 1 < load[a]:
                load[a] -= 1
                load[r] += 1
                code[e] = 2
                moved[i] = 1
    return load, moved


def gather_plans(entries: list[tuple[int, int]], R: int, ER: int, k: int) -> list[tuple]:
    """One ragged_all_to_all plan per slot index q: (in_off, send_sz, out_off, recv_sz), each
    [R, R] int32, row = the mesh index it is fed to (sender rows for the first three, receiver
    rows for recv_sz), in units of whole expert rows."""
    se = slot_experts(entries, R, k)
    plans = []
    for q in range(k):
        io = np.zeros((R, R), np.int32)
        ss = np.zeros((R, R), np.int32)
        oo = np.zeros((R, R), np.int32)
        for h in range(R):
            e = se[h * k + q]
            if e < 0:
                continue
            src = int(e) // ER
            io[src, h] = int(e) % ER
            ss[src, h] = 1
            oo[src, h] = q
        plans.append((io, ss, oo, ss.T.copy()))
    return plans


def make_gather_fn(mesh, axis_name):
    """(shard [ER, ...], slots [k, ...], in_off, send_sz, out_off, recv_sz) -> slots with the
    plan's rows written; rows no chunk lands on keep their input value."""
    import jax.numpy as jnp
    from jax import lax
    from jax.sharding import PartitionSpec as P

    spec = P(axis_name)

    def body(w_l, o_l, io, ss, oo, rs):
        dt = w_l.dtype
        if jnp.dtype(dt).itemsize == 1 and dt != jnp.int8:       # bits, not values
            w_l = lax.bitcast_convert_type(w_l, jnp.int8)
            o_l = lax.bitcast_convert_type(o_l, jnp.int8)
        out = lax.ragged_all_to_all(w_l, o_l, io[0], ss[0], oo[0], rs[0], axis_name=axis_name)
        return out if out.dtype == dt else lax.bitcast_convert_type(out, dt)

    def gather(
        w: jax.Array,
        slots: jax.Array,
        in_off: jax.Array,
        send_sz: jax.Array,
        out_off: jax.Array,
        recv_sz: jax.Array,
    ) -> jax.Array:
        return jax.shard_map(
            body, mesh=mesh, in_specs=(spec,) * 6, out_specs=spec, check_vma=False
        )(w, slots, in_off, send_sz, out_off, recv_sz)

    return gather


def make_gather_stk_fn(mesh, axis_name):
    """make_gather_fn into row `row` of a stacked [n_layers, k, ...] slot tensor (GLM5_MK_STACK_W):
    (stack, shard [ER, ...], row [1], in_off, send_sz, out_off, recv_sz) -> stack with that row's
    plan rows written. Donate the stack: the update is an in-place DUS of one row."""
    import jax.numpy as jnp
    from jax import lax
    from jax.sharding import PartitionSpec as P

    spec, sspec = P(axis_name), P(None, axis_name)

    def body(s_l, w_l, row, io, ss, oo, rs):
        dt = w_l.dtype
        if jnp.dtype(dt).itemsize == 1 and dt != jnp.int8:       # bits, not values
            w_l = lax.bitcast_convert_type(w_l, jnp.int8)
            s_l = lax.bitcast_convert_type(s_l, jnp.int8)
        o_l = lax.dynamic_index_in_dim(s_l, row[0], 0, keepdims=False)
        out = lax.ragged_all_to_all(w_l, o_l, io[0], ss[0], oo[0], rs[0], axis_name=axis_name)
        s_l = lax.dynamic_update_index_in_dim(s_l, out, row[0], 0)
        return s_l if s_l.dtype == dt else lax.bitcast_convert_type(s_l, dt)

    def gather_stk(
        stk: jax.Array,
        w: jax.Array,
        row: jax.Array,
        in_off: jax.Array,
        send_sz: jax.Array,
        out_off: jax.Array,
        recv_sz: jax.Array,
    ) -> jax.Array:
        return jax.shard_map(
            body, mesh=mesh, in_specs=(sspec, spec, P()) + (spec,) * 4, out_specs=sspec,
            check_vma=False)(stk, w, row, in_off, send_sz, out_off, recv_sz)

    return gather_stk


def load_profile(path: str) -> dict[int, np.ndarray]:
    """EPLB log npz (`layers`, `logical_load` [n, E]) -> {layer: logical load}."""
    d = np.load(path)
    return {int(l): np.asarray(v, np.float64) for l, v in zip(d["layers"], d["logical_load"])}
