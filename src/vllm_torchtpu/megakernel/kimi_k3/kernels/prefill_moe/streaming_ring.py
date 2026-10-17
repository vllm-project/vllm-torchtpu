# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Latent ring progress controlled by the shared-expert pipeline.

The protocol follows prefill_megakernel._all_gather_kernel. The caller supplies
the ring step explicitly so shared-expert products can advance it between
compute tiles. Ref adapters transport packed token rows without an HBM copy.
"""

import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.megakernel.kimi_k3.kernels import prefill_megakernel as k


def ring_step(
    tables,
    x,
    out,
    load_sems,
    send_sems,
    recv_sems,
    credits,
    store_sems,
    payload,
    received,
    *,
    step,
    regions,
    buffers,
    striped=False,
):
    """Ring all-gather of every rank's rows. ``striped``: the rows are SP
    stripes (``sp_token_index``) and land in window order; otherwise owner
    o's rows land at ``[o rows, (o + 1) rows)``."""
    tile = 0
    rank = lax.axis_index("k3_sp")
    positions = [tables[f, 0, rank] for f in range(3)]
    local_first = tile * payload.shape[0]

    def owner(pos, family):
        return tables[family, 1, pos % 32]

    def peer(pos, family):
        r = owner(pos, family)
        return (r,)

    pl.when(step == 0)(lambda: k._ring_enter(tables, rank))

    @pl.when(step == 0)
    def load():
        for s, (begin, size, col, width) in enumerate(regions):
            rows, cols = pl.ds(begin, size), pl.ds(col, width)
            pltpu.make_async_copy(
                x.at[pl.ds(local_first + begin, size), cols],
                payload.at[rows, cols],
                load_sems.at[s],
            ).start()

    for s, (begin, size, col, width) in enumerate(regions):
        family = (s // 2) % 3
        direction = -1 if s % 2 == 0 else 1
        pos = positions[family]
        target = peer(pos + direction, jnp.int32(family))
        source = peer(pos - direction, jnp.int32(family))
        rows, cols = pl.ds(begin, size), pl.ds(col, width)

        def remote(index, rows=rows, cols=cols, s=s, target=target):
            return pltpu.make_async_remote_copy(
                payload.at[rows, cols],
                received.at[index % buffers, rows, cols],
                send_sems.at[s],
                recv_sems.at[index % buffers, s],
                device_id=target,
                device_id_type=pl.DeviceIdType.MESH,
            )

        def store(
            index,
            begin=begin,
            size=size,
            cols=cols,
            s=s,
            pos=pos,
            family=family,
            direction=direction,
        ):
            owner_mesh = owner(pos - direction * index, jnp.int32(family))
            dest = tables[family, 2, owner_mesh]
            if not striped:
                return [
                    pltpu.make_async_copy(
                        payload.at[pl.ds(begin, size), cols],
                        out.at[
                            pl.ds(
                                dest * x.shape[0] + tile * payload.shape[0] + begin,
                                size,
                            ),
                            cols,
                        ],
                        store_sems.at[s],
                    )
                ]
            stripe = k.window_stripe(x.shape[0])
            return [
                pltpu.make_async_copy(
                    payload.at[pl.ds(row, count), cols],
                    out.at[pl.ds(k.sp_token_index(dest, row, stripe), count), cols],
                    store_sems.at[s],
                )
                for _, row, count in k._stripe_pieces(begin, size, stripe)
            ]

        @pl.when(step == 0)
        def seed(begin=begin, size=size, rows=rows, cols=cols, s=s):
            pltpu.make_async_copy(
                x.at[pl.ds(local_first + begin, size), cols],
                payload.at[rows, cols],
                load_sems.at[s],
            ).wait()

        @pl.when(step > 0)
        def receive(
            remote=remote, store=store, rows=rows, cols=cols, s=s, source=source
        ):
            remote(step - 1).wait()
            for copy in store(step - 1):
                copy.wait()
            payload.at[rows, cols][...] = received.at[(step - 1) % buffers, rows, cols][
                ...
            ]
            pl.semaphore_signal(
                credits.at[s], 1, device_id=source, device_id_type=pl.DeviceIdType.MESH
            )

        for copy in store(step):
            copy.start()

        @pl.when(step < 31)
        def forward(remote=remote, s=s):
            @pl.when(step >= buffers)
            def acquire():
                pl.semaphore_wait(credits.at[s], 1)

            remote(step).start()

        @pl.when(step == 31)
        def finish(store=store, s=s):
            for copy in store(step):
                copy.wait()
            pl.semaphore_wait(credits.at[s], buffers)
