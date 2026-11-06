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
"""SparseCore group gather, current/ring selection and exact pooled rounding."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental import xla_metadata
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from .index_pool_prepare import _canonical_bf16_words


def _mean_words(values):
    # TPU's four-member reduction combines stride-two lanes first.
    summed = (values[0] + values[2]) + (values[1] + values[3])
    bits = plsc.bitcast(summed, jnp.int32)
    # The original reduce includes its +0 initializer, even for all -0 rows.
    bits = jnp.where((bits & 0x7FFFFFFF) == 0, 0, bits)
    value = plsc.bitcast(bits, jnp.float32) * jnp.float32(0.25)
    bits = plsc.bitcast(value, jnp.int32)
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF
    return _canonical_bf16_words(rounded)


def _contiguous_body(source, output, *, mesh, groups, count, width, lanes):
    worker = jax.lax.axis_index(("core", "subcore"))

    def body(raw_ref, out_ref):
        def pairs(pair):
            def columns(column):
                col = column * lanes

                def mean(group):
                    a = raw_ref[2 * group, pl.ds(col, lanes)]
                    b = raw_ref[2 * group + 1, pl.ds(col, lanes)]
                    words = (
                        a & 0xFFFF,
                        (a >> 16) & 0xFFFF,
                        b & 0xFFFF,
                        (b >> 16) & 0xFFFF,
                    )
                    return _mean_words(
                        [
                            plsc.bitcast(_canonical_bf16_words(w) << 16, jnp.float32)
                            for w in words
                        ]
                    )

                out_ref[pair, pl.ds(col, lanes)] = mean(2 * pair) | (
                    mean(2 * pair + 1) << 16
                )

            plsc.parallel_loop(0, width // lanes)(columns)

        plsc.parallel_loop(0, groups // 2)(pairs)

    pltpu.emit_pipeline(
        body,
        grid=(count // (4 * groups * mesh.size),),
        in_specs=pl.BlockSpec(
            (2 * groups, width), lambda b: (b * mesh.size + worker, 0)
        ),
        out_specs=pl.BlockSpec(
            (groups // 2, width), lambda b: (b * mesh.size + worker, 0)
        ),
    )(source.bitcast(jnp.int32), output.bitcast(jnp.int32))


def pool_index_groups_sc_async(
    raw: jax.Array,
    ring: jax.Array,
    starts: jax.Array,
    context: jax.Array,
    slots: jax.Array,
    req: jax.Array,
    ratio: int,
    *,
    scheduling_group_id: int | None = None,
) -> jax.Array:
    """Fuse all current/ring group pooling into one pipelined SC call."""
    assert raw.shape == (8192, 128) and raw.dtype == jnp.bfloat16
    assert ratio == 4 and len(context) == 8
    return _window_sc(
        raw,
        ring,
        starts,
        context,
        slots,
        req,
        ratio,
        scheduling_group_id=scheduling_group_id,
    )


def _window_sc(raw, ring, starts, context, slots, req, ratio, *, scheduling_group_id):
    hardware = pltpu.get_tpu_info()
    info = hardware.sparse_core
    assert info is not None
    lanes, count, width = info.num_lanes, *raw.shape
    groups = lanes
    mesh = plsc.VectorSubcoreMesh(
        num_cores=info.num_cores,
        num_subcores=info.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    size = pl.cdiv(count, ratio) + 2 * len(context)
    tiles = pl.cdiv(size, groups)
    assert tiles // mesh.size >= 2 and width % lanes == 0
    padded_groups = tiles * groups
    waves = pl.cdiv(tiles, mesh.size)
    metadata_width = 4 * groups
    # first = ratio*g + starts[r]%ratio - context[r]%ratio - 2*ratio*r.
    # Include every group/member, native row alignment and both endpoints.
    span = ratio * (groups - 1) + 3 * (ratio - 1) + 2 * ratio * (len(context) - 1)
    window = (
        pl.cdiv(span + hardware.num_sublanes, hardware.num_sublanes)
        * hardware.num_sublanes
    )
    window_rows = window // 2
    ring_words = ring.size // 2
    ring_rows = ring.shape[1]
    named_bytes = 4 * (
        waves * metadata_width + ring_words + 2 * window_rows * width + groups * width
    )
    if named_bytes > info.vmem_capacity_bytes:
        raise ValueError("SC compression windows exceed available per-worker VMEM")
    base = starts[:-1] // ratio + 2 * jnp.arange(len(context))
    # The last request also owns padding rows. Do not discard it based on
    # validity: those rows are part of the original dense pooling ABI.
    owned = starts[1:] > starts[:-1]
    owned = owned.at[-1].set(owned[-1] | (starts[-1] < count))
    aligned = jnp.all(~owned | ((starts[:-1] % ratio == 0) & (context % ratio == 0)))

    def make_metadata(_):
        candidates = jnp.arange(padded_groups, dtype=jnp.int32)
        owners = jnp.sum(candidates[:, None] >= base[None, 1:], axis=1)
        first = (
            starts[owners]
            + (candidates - base[owners]) * ratio
            - context[owners] % ratio
        )
        lo = jnp.min(jnp.clip(first, 0, count - 1).reshape(tiles, groups), axis=1)
        window_start = jnp.clip(
            lo // hardware.num_sublanes * hardware.num_sublanes, 0, count - window
        )
        encoded_slot = jnp.where(context[owners] > 0, slots[owners] + 1, 0)
        return jnp.stack(
            (
                first.reshape(tiles, groups),
                starts[owners].reshape(tiles, groups),
                encoded_slot.reshape(tiles, groups),
                jnp.broadcast_to(window_start[:, None], (tiles, groups)),
            ),
            axis=1,
        ).reshape(-1)

    metadata = jax.lax.cond(
        aligned,
        lambda _: jnp.zeros((tiles * metadata_width,), jnp.int32),
        make_metadata,
        None,
    )
    choice = jnp.full((lanes,), aligned, jnp.int32)

    def window_body(
        source,
        history,
        meta,
        output,
        meta_buf,
        ring_buf,
        raw_buf,
        out_buf,
        meta_sem,
        ring_sem,
        raw_sem,
        out_sem,
    ):
        worker = jax.lax.axis_index(("core", "subcore"))
        source = source.bitcast(jnp.int32)
        history = history.bitcast(jnp.int32).reshape(-1, width)
        output = output.bitcast(jnp.int32)
        history_copy = pltpu.make_async_copy(history, ring_buf, ring_sem)
        history_copy.start()
        for wave in range(waves):
            tile = wave * mesh.size + worker

            @pl.when(tile < tiles)
            def start_meta(*, wave=wave, tile=tile):
                pltpu.make_async_copy(
                    meta.at[pl.ds(tile * metadata_width, metadata_width)],
                    meta_buf.at[pl.ds(wave * metadata_width, metadata_width)],
                    meta_sem.at[wave],
                ).start()

        for wave in range(waves):
            tile = wave * mesh.size + worker

            @pl.when(tile < tiles)
            def wait_meta(*, wave=wave, tile=tile):
                pltpu.make_async_copy(
                    meta.at[pl.ds(tile * metadata_width, metadata_width)],
                    meta_buf.at[pl.ds(wave * metadata_width, metadata_width)],
                    meta_sem.at[wave],
                ).wait()

        def read_copy(wave, buffer):
            offset = wave * metadata_width + 3 * groups
            start = meta_buf[pl.ds(offset, lanes)][0] // 2
            return pltpu.make_async_copy(
                source.at[pl.ds(start, window_rows)],
                raw_buf.at[buffer],
                raw_sem.at[buffer],
            )

        def write_copy(wave, buffer):
            tile = wave * mesh.size + worker
            return pltpu.make_async_copy(
                out_buf.at[buffer],
                output.at[pl.ds(tile * (groups // 2), groups // 2)],
                out_sem.at[buffer],
            )

        read_copy(0, 0).start()
        history_copy.wait()

        def step(wave, _):
            tile = wave * mesh.size + worker
            buffer = wave % 2

            @pl.when(tile < tiles)
            def process():
                @pl.when(wave >= 2)
                def retire():
                    write_copy(wave - 2, buffer).wait()

                read_copy(wave, buffer).wait()

                @pl.when((wave + 1) * mesh.size + worker < tiles)
                def prefetch():
                    read_copy(wave + 1, 1 - buffer).start()

                offset = wave * metadata_width
                firsts = meta_buf[pl.ds(offset, lanes)]
                begins = meta_buf[pl.ds(offset + groups, lanes)]
                states = meta_buf[pl.ds(offset + 2 * groups, lanes)]
                start = meta_buf[pl.ds(offset + 3 * groups, lanes)][0] // 2

                def columns(column):
                    col = column * lanes

                    def mean(group):
                        values = []
                        for member in range(ratio):
                            index = firsts[group] + member
                            clipped = jnp.clip(index, 0, count - 1)
                            now = raw_buf[
                                buffer, clipped // 2 - start, pl.ds(col, lanes)
                            ]
                            now = (now >> ((clipped % 2) * 16)) & 0xFFFF
                            slot = jnp.maximum(states[group] - 1, 0)
                            old = ring_buf[
                                slot * (ring_rows // 2) + member // 2, pl.ds(col, lanes)
                            ]
                            old = (old >> ((member % 2) * 16)) & 0xFFFF
                            word = jnp.where(
                                index >= begins[group],
                                now,
                                jnp.where(states[group] > 0, old, 0),
                            )
                            values.append(
                                plsc.bitcast(
                                    _canonical_bf16_words(word) << 16, jnp.float32
                                )
                            )
                        return _mean_words(values)

                    for pair in range(groups // 2):
                        out_buf[buffer, pair, pl.ds(col, lanes)] = mean(2 * pair) | (
                            mean(2 * pair + 1) << 16
                        )

                plsc.parallel_loop(0, width // lanes)(columns)
                write_copy(wave, buffer).start()

        # Keep the five-wave DMA pipeline in a device loop. Unrolling both
        # pooling modes exceeds the installed SC executable instruction limit.
        jax.lax.fori_loop(0, waves, step, None)

        owned = tiles // mesh.size + (worker < tiles % mesh.size).astype(jnp.int32)
        for buffer in range(2):
            last = owned - 1 - ((owned - 1 - buffer) % 2)
            write_copy(last, buffer).wait()

    def kernel(source, history, meta, choice, output, *scratch):
        *window_scratch, choice_buf = scratch
        pltpu.sync_copy(choice, choice_buf)
        contiguous = choice_buf[pl.ds(0, lanes)][0] != 0

        @pl.when(contiguous)
        def contiguous_body():
            _contiguous_body(
                source,
                output,
                mesh=mesh,
                groups=groups,
                count=count,
                width=width,
                lanes=lanes,
            )

            # Both branches initialize the same compact buffer. This is only
            # an output tail; no nonexistent fifth raw tile may be prefetched.
            @pl.when(jax.lax.axis_index(("core", "subcore")) == 0)
            def tail():
                out_buf, out_sem = window_scratch[3], window_scratch[7]

                def rows(row):
                    def columns(column):
                        out_buf[0, row, pl.ds(column * lanes, lanes)] = jnp.zeros(
                            (lanes,), jnp.int32
                        )

                    plsc.parallel_loop(0, width // lanes)(columns)

                plsc.parallel_loop(0, groups // 2)(rows)
                pltpu.make_async_copy(
                    out_buf.at[0],
                    output.bitcast(jnp.int32).at[
                        pl.ds(count // (2 * ratio), groups // 2)
                    ],
                    out_sem.at[0],
                ).start()
                pltpu.make_async_copy(
                    out_buf.at[0],
                    output.bitcast(jnp.int32).at[
                        pl.ds(count // (2 * ratio), groups // 2)
                    ],
                    out_sem.at[0],
                ).wait()

        @pl.when(~contiguous)
        def windows():
            window_body(source, history, meta, output, *window_scratch)

    compact = pl.kernel(
        kernel,
        mesh=mesh,
        out_type=jax.ShapeDtypeStruct((padded_groups, width), raw.dtype),
        scratch_types=(
            pltpu.VMEM((waves * metadata_width,), jnp.int32),
            pltpu.VMEM((ring_words // width, width), jnp.int32),
            pltpu.VMEM((2, window_rows, width), jnp.int32),
            pltpu.VMEM((2, groups // 2, width), jnp.int32),
            pltpu.SemaphoreType.DMA((waves,)),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.VMEM((lanes,), jnp.int32),
        ),
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=True,
            needs_layout_passes=True,
            disable_bounds_checks=True,
        ),
        name="qsa_index_window_mean_sc",
    )(raw, ring, metadata, choice)
    if scheduling_group_id is not None:
        compact = xla_metadata.set_xla_metadata(
            compact, _scheduling_group_id=scheduling_group_id
        )
    local = jnp.arange(count, dtype=jnp.int32)
    logical = context[req] + local - starts[req]
    selected = base[req] + logical // ratio - context[req] // ratio
    return jax.lax.cond(
        aligned,
        lambda _: jnp.repeat(compact[: count // ratio], ratio, axis=0),
        lambda _: compact[selected],
        None,
    )
