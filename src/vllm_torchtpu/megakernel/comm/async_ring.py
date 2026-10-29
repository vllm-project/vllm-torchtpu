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
"""Split-phase non-blocking `AllReduceRing` execution (`ARPIPE` / `AR0S`)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl

from ..errors import MegakernelConfigError
from ..ops import BF16, F32, I32, rnd
from .params import CommParams
from .ring import AllReduceRing


@dataclass(frozen=True)
class AllReduceHandle:
    """In-flight state for split-phase `AllReduceRing` execution."""

    slot: Any
    round_idx: int
    val: jax.Array


def _active_rounds(cp: CommParams) -> tuple[int, ...]:
    return (0, 2) if (cp.HIER == 4 and cp.N_HOSTS == 1) else cp.AR_ROUNDS


def slot_for_step(step_idx: int | jax.Array) -> tuple[Any, Any]:
    """Map monotonic collective `step_idx` to alternating `(slot, first)`."""
    idx = jnp.asarray(step_idx, dtype=I32)
    return lax.rem(idx, I32(2)), idx < I32(2)


def start_allreduce(
    ring: AllReduceRing,
    x: jax.Array,
    *,
    step_idx: int | jax.Array | None = None,
    slot: Any = None,
    first: Any = None,
) -> AllReduceHandle:
    """Pack `x` and launch Round 0 remote DMAs without blocking on `wait_recv`."""
    if step_idx is not None:
        if slot is not None or first is not None:
            raise MegakernelConfigError("pass step_idx or (slot, first), not both")
        slot, first = slot_for_step(step_idx)
    elif slot is None or first is None:
        raise MegakernelConfigError("start_allreduce needs step_idx or (slot, first)")
    val = ring.pack_slot(slot, x, first)
    if ring.comm:
        with ring.sc("ar_comm"):
            for d in ring.ar_copies(slot, 0):
                d.start()
    return AllReduceHandle(slot=slot, round_idx=0, val=val)


def _transition_round(
    ring: AllReduceRing, slot: Any, prev_rnd: int, next_rnd: int
) -> None:
    cp, scr, me, sc = ring.cp, ring.scr, ring.me, ring.sc
    HIER = cp.HIER
    with sc("ar_comm"):
        for d in ring.ar_copies(slot, prev_rnd):
            d.wait_recv()
    with sc("ar_sum"):
        if HIER == 4 and prev_rnd == 0:
            lr = lax.rem(me, cp.HIER_GROUP)
            scr["ag_send"][slot] = jnp.sum(
                scr["recv"][slot].astype(F32), axis=0
            ).astype(BF16)
            if cp.N_HOSTS > 1:
                scr["ih_recv"][slot, me // cp.HIER_GROUP] = scr["ag_send"][slot]
            else:
                scr["ag_recv"][slot, lr] = scr["ag_send"][slot]
        elif HIER == 4 and prev_rnd == 1:
            scr["s2_send"][slot] = jnp.sum(
                scr["ih_recv"][slot].astype(F32), axis=0
            ).astype(BF16)
            scr["ag_recv"][slot, lax.rem(me, cp.HIER_GROUP)] = scr["s2_send"][slot]
        elif HIER == 3:
            bi = lax.rem(me >> next_rnd, 2)
            scr["send"][next_rnd, slot] = (
                scr["recv"][prev_rnd, slot, 0].astype(F32)
                + scr["recv"][prev_rnd, slot, 1].astype(F32)
            ).astype(BF16)
            scr["recv"][next_rnd, slot, bi] = scr["send"][next_rnd, slot]
        elif HIER in (1, 2) and prev_rnd == 0:
            scr["ag_send"][slot] = (
                scr["recv"][slot, 0].astype(F32) + scr["recv"][slot, 1].astype(F32)
            ).astype(BF16)
            scr["ag_recv"][slot, me // 2] = scr["ag_send"][slot]
        elif HIER == 2 and prev_rnd == 1:
            scr["s2_send"][slot] = jnp.sum(
                scr["ag_recv"][slot].astype(F32), axis=0
            ).astype(BF16)
            scr["s2_recv"][slot, lax.rem(me, 2)] = scr["s2_send"][slot]
        else:
            scr["ag_send"][slot] = jnp.sum(
                scr["recv"][slot].astype(F32), axis=0
            ).astype(BF16)
            scr["ag_recv"][slot, me] = scr["ag_send"][slot]
    with sc("ar_comm"):
        for d in ring.ar_copies(slot, next_rnd):
            d.start()


def advance_allreduce(ring: AllReduceRing, handle: AllReduceHandle) -> AllReduceHandle:
    """Finish the active ICI round and launch the next round non-blockingly."""
    if not ring.comm:
        return handle
    rounds = _active_rounds(ring.cp)
    pos = rounds.index(handle.round_idx)
    if pos + 1 >= len(rounds):
        return handle
    next_rnd = rounds[pos + 1]
    _transition_round(ring, handle.slot, handle.round_idx, next_rnd)
    return AllReduceHandle(slot=handle.slot, round_idx=next_rnd, val=handle.val)


def wait_allreduce(ring: AllReduceRing, handle: AllReduceHandle) -> jax.Array:
    """Complete all remaining ICI rounds on `handle` and return the F32 sum."""
    if not ring.comm:
        return rnd(handle.val)
    cp, scr, sc = ring.cp, ring.scr, ring.sc
    slot = handle.slot
    rounds = _active_rounds(cp)
    pos = rounds.index(handle.round_idx)
    for idx in range(pos, len(rounds) - 1):
        _transition_round(ring, slot, rounds[idx], rounds[idx + 1])
    with sc("ar_comm"):
        for d in ring.ar_copies(slot, rounds[-1]):
            d.wait_recv()
    with sc("ar_sum"):
        if cp.HIER == 4:
            return jnp.concatenate(
                [scr["ag_recv"][slot, j] for j in range(cp.HIER_GROUP)], axis=-1
            ).astype(F32)
        if cp.HIER == 3:
            last = cp.N_HIER3 - 1
            return scr["recv"][last, slot, 0].astype(F32) + scr["recv"][
                last, slot, 1
            ].astype(F32)
        if cp.HIER == 1:
            return jnp.sum(scr["ag_recv"][slot].astype(F32), axis=0)
        if cp.HIER == 2:
            return jnp.concatenate(
                [scr["s2_recv"][slot, 0], scr["s2_recv"][slot, 1]], axis=-1
            ).astype(F32)
        if cp.RSAG:
            return jnp.concatenate(
                [scr["ag_recv"][slot, j] for j in range(cp.R)], axis=-1
            ).astype(F32)
        return jnp.sum(scr["recv"][slot].astype(F32), axis=0)


def drain_async_sends(ring: AllReduceRing, num_steps: int | jax.Array) -> None:
    """Drain in-flight sends for `min(num_steps, 2)` alternating slots."""
    if not ring.comm:
        return
    steps = jnp.asarray(num_steps, dtype=I32)
    for slot_i in (0, 1):

        @pl.when(steps >= I32(slot_i + 1))
        def _(s=slot_i):
            for rnd_i in _active_rounds(ring.cp):
                for d in ring.ar_copies(s, rnd_i):
                    d.wait_send()
