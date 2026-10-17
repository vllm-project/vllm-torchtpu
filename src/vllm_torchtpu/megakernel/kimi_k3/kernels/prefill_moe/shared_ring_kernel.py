# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared-expert work interleaved with the latent ring."""


def make_shared_ring(local_rows, *, budget=1):
    import contextlib

    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    from .packed_layout import _PackedView, row_regions
    from .streaming_ring import ring_step

    hidden, middle, width = (7168, 3072, 3584)
    rows = 2 * local_rows
    nm, ng, nd = (rows // 256, middle // 512, hidden // 512)
    gu_steps = nm * ng
    total = nm * (ng + nd)

    row_shape = (2, width // 2)
    view = _PackedView
    regions = row_regions(local_rows, width // 2)
    streams = len(regions)

    def scope(n):
        return contextlib.nullcontext()

    def body(
        tables,
        layer,
        x,
        h,
        gu,
        down,
        gathered,
        y,
        loads,
        sends,
        receives,
        credits,
        stores,
        payload,
        received,
        hv,
        mid,
        wg,
        wu,
        wd,
        wsems,
        hsem,
        outbuf,
        osem,
        state,
    ):
        t = pl.program_id(0)
        gu = gu.at[layer[0]]
        down = down.at[layer[0]]

        def advance(stop):
            completed = (state[0] < 32) & (stop == 32)

            @pl.loop(state[0], stop)
            def step(s):
                ring_step(
                    tables,
                    view(x),
                    view(gathered),
                    loads,
                    sends,
                    receives,
                    credits,
                    stores,
                    view(payload),
                    view(received),
                    step=s,
                    regions=regions,
                    buffers=2,
                )

            state[0] = stop

            @pl.when(completed)
            def ring_completed():
                with scope("shared_ring_complete"):
                    state[1] = stop

        def gu_load(n, slot):
            for ref, offset in ((wg, 0), (wu, middle)):
                pltpu.make_async_copy(
                    gu.at[:, pl.ds(n * 512 + offset, 512)], ref.at[slot], wsems.at[slot]
                ).start()

        def down_load(n, slot):
            pltpu.make_async_copy(
                down.at[:, pl.ds(n * 512, 512)], wd.at[slot], wsems.at[slot]
            ).start()

        @pl.when(t == 0)
        def init():
            state[0] = 0
            pltpu.make_async_copy(h, hv, hsem).start()
            gu_load(jnp.int32(0), jnp.int32(0))
            pltpu.make_async_copy(h, hv, hsem).wait()

        @pl.when((t % budget == 0) & (state[0] < 32))
        def progress():
            advance(state[0] + 1)

        @pl.when(t < gu_steps)
        def gate_up():
            n, m = (t // nm, t % nm)
            slot = n % 2

            @pl.when(m == 0)
            def load_weights():
                pltpu.make_async_copy(wg.at[slot], wg.at[slot], wsems.at[slot]).wait()
                pltpu.make_async_copy(wu.at[slot], wu.at[slot], wsems.at[slot]).wait()

                @pl.when(n + 1 < ng)
                def prefetch():
                    gu_load(n + 1, 1 - slot)

            lhs = hv.at[pl.ds(m * 256, 256)][...]
            gate = jnp.dot(lhs, wg.at[slot][...], preferred_element_type=jnp.float32)
            up = jnp.dot(lhs, wu.at[slot][...], preferred_element_type=jnp.float32)
            value = (
                4 * jnp.tanh(gate / 4) * jax.nn.sigmoid(gate) * (25 * jnp.tanh(up / 25))
            )
            mid.at[pl.ds(m * 256, 256), pl.ds(n * 512, 512)][...] = value.astype(
                jnp.bfloat16
            )

            @pl.when(t == gu_steps - 1)
            def begin_down():
                with scope("shared_mid_ready"):
                    down_load(jnp.int32(0), jnp.int32(0))

        @pl.when(t >= gu_steps)
        def output():
            n, m = ((t - gu_steps) // nm, (t - gu_steps) % nm)
            slot = n % 2

            @pl.when(m == 0)
            def load_weights():
                pltpu.make_async_copy(wd.at[slot], wd.at[slot], wsems.at[slot]).wait()

                @pl.when(n + 1 < nd)
                def prefetch():
                    down_load(n + 1, 1 - slot)

            lhs = mid.at[pl.ds(m * 256, 256)][...]
            outbuf[...] = jnp.dot(
                lhs, wd.at[slot][...], preferred_element_type=jnp.float32
            )
            copy = pltpu.make_async_copy(
                outbuf, y.at[pl.ds(m * 256, 256), pl.ds(n * 512, 512)], osem
            )
            copy.start()
            copy.wait()

            @pl.when(t == gu_steps)
            def first_output():
                with scope("shared_first_output_ready"):
                    state[1] = t

        @pl.when(t == total - 1)
        def finish():
            with scope("shared_ring_drain"):
                advance(jnp.int32(32))
            with scope("shared_all_done"):
                state[1] = t

    return pl.pallas_call(
        body,
        out_shape=(
            jax.ShapeDtypeStruct((local_rows * 32, *row_shape), jnp.bfloat16),
            jax.ShapeDtypeStruct((rows, hidden), jnp.float32),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(total,),
            in_specs=(pl.BlockSpec(memory_space=pltpu.HBM),) * 4,
            out_specs=(pl.BlockSpec(memory_space=pltpu.HBM),) * 2,
            scratch_shapes=(
                pltpu.SemaphoreType.DMA((streams,)),
                pltpu.SemaphoreType.DMA((streams,)),
                pltpu.SemaphoreType.DMA((2, streams)),
                pltpu.SemaphoreType.REGULAR((streams,)),
                pltpu.SemaphoreType.DMA((streams,)),
                pltpu.VMEM((local_rows, *row_shape), jnp.bfloat16),
                pltpu.VMEM((2, local_rows, *row_shape), jnp.bfloat16),
                pltpu.VMEM((rows, hidden), jnp.bfloat16),
                pltpu.VMEM((rows, middle), jnp.bfloat16),
                pltpu.VMEM((2, hidden, 512), jnp.bfloat16),
                pltpu.VMEM((2, hidden, 512), jnp.bfloat16),
                pltpu.VMEM((2, middle, 512), jnp.bfloat16),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA,
                pltpu.VMEM((256, 512), jnp.float32),
                pltpu.SemaphoreType.DMA,
                pltpu.SMEM((2,), jnp.int32),
            ),
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=43,
            dimension_semantics=("arbitrary",),
            disable_bounds_checks=True,
            vmem_limit_bytes=96 * 1024**2,
        ),
        name=f"shared_ring_{'all'}_{'stream'}_b{budget}_m{256}_n{512}",
    )
