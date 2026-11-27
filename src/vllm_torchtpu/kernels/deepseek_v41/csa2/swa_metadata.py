# SPDX-License-Identifier: Apache-2.0
"""CSA2 SWA metadata of several block tables in one scalar-core kernel.

`prepare_swa_metadata` as XLA is ~25 small ops per call, and its page lookup
gathers every padded request's window from a 4 MiB table in HBM. At decode
only a few of the request rows are live. This kernel DMAs one aligned slice
of each live request's table row and writes the metadata from it, so the
work follows the live requests, not the padded batch.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .cache import check_swa_metadata_args, prepare_swa_metadata

# Table entries DMA'd per live request and table: a 128-aligned slice that
# holds the request's (at most 128) window pages wherever the window starts.
_CHUNK = 256


def _smem_words(groups: int, requests: int, tokens: int, columns: int) -> int:
    """Scalar-memory words of the kernel: table slices, outputs, scalar prefetch."""
    slices = groups * requests * _CHUNK
    outputs = groups * requests * (2 + columns) + 2 * requests + groups * tokens
    # Slice starts and minimum per request; starts; positions.
    scalars = 2 * requests + (requests + 1) + tokens
    return slices + outputs + scalars


def _admitted(
    groups: int,
    requests: int,
    tokens: int,
    width: int,
    columns: int,
    interpret: bool,
) -> bool:
    if width % 128 or width < _CHUNK or columns > _CHUNK - 128:
        return False
    # The interpreter runs off TPU, where there is no chip to ask: assume 1 MiB.
    capacity = 1 << 20 if interpret else pltpu.get_tpu_info().smem_capacity_bytes
    # Half of scalar memory, the rest left to the compiler.
    return 4 * _smem_words(groups, requests, tokens, columns) <= capacity // 2


@functools.partial(
    jax.jit,
    static_argnames=(
        "page_size",
        "page_stride",
        "row_offset",
        "non_causal",
        "onehot_requests",
        "interpret",
    ),
)
def prepare_swa_metadata_tables(
    tables: tuple[jax.Array, ...],
    starts: jax.Array,
    positions: jax.Array,
    minimum_visible: jax.Array,
    *,
    page_size: int,
    page_stride: int,
    row_offset: int,
    non_causal: bool = False,
    onehot_requests: bool = False,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """`prepare_swa_metadata` for G flat [R * W] block tables: metadata
    [G, R, 2 + C], the shared bounds [R, 2] and records [G, T].

    Equal to one `prepare_swa_metadata` call per table under its contract
    (starts monotonic from 0, each request's positions contiguous). Shapes
    the kernel does not admit take those calls, where ``onehot_requests``
    applies: rows narrower than one slice (256 entries) or not whole
    128-entry tiles, windows over 128 pages, or scalar memory past its
    budget. ``interpret`` runs the kernel in the Pallas
    interpreter (CPU tests).
    """
    groups, tokens = len(tables), positions.shape[0]
    requests = starts.shape[0] - 1
    width = check_swa_metadata_args(
        tables[0],
        starts,
        positions,
        minimum_visible,
        page_size,
        page_stride,
        row_offset,
    )
    if any(table.shape != tables[0].shape for table in tables):
        raise ValueError("SWA group block tables must share one shape")
    columns = (tokens + 128 + page_size - 1) // page_size + 1
    if not _admitted(groups, requests, tokens, width, columns, interpret):
        results = [
            prepare_swa_metadata(
                table,
                starts,
                positions,
                minimum_visible,
                page_size=page_size,
                page_stride=page_stride,
                row_offset=row_offset,
                non_causal=non_causal,
                onehot_requests=onehot_requests,
            )
            for table in tables
        ]
        return (
            jnp.stack([result[0] for result in results]),
            results[0][1],
            jnp.stack([result[2] for result in results]),
        )
    metadata, bounds, records = _swa_metadata_call(
        tables,
        starts,
        positions,
        minimum_visible,
        width=width,
        columns=columns,
        page_size=page_size,
        page_stride=page_stride,
        row_offset=row_offset,
        back=128 if non_causal else 127,
        interpret=interpret,
    )
    return (
        metadata.reshape(groups, requests, 2 + columns),
        bounds.reshape(requests, 2),
        records.reshape(groups, tokens),
    )


def _swa_metadata_call(
    tables,
    starts,
    positions,
    minimum,
    *,
    width,
    columns,
    page_size,
    page_stride,
    row_offset,
    back,
    interpret,
):
    groups, tokens = len(tables), positions.shape[0]
    requests = starts.shape[0] - 1

    def kernel(starts_ref, pos_ref, min_ref, *refs):
        table_refs = refs[:groups]
        md_ref, bd_ref, rec_ref = refs[groups : groups + 3]
        bufs = refs[groups + 3 : 2 * groups + 3]
        base_ref, sem = refs[2 * groups + 3 :]

        def row(g, r):
            return (g * requests + r) * (2 + columns)

        def window(r):
            count = starts_ref[r + 1] - starts_ref[r]
            first = pos_ref[jnp.minimum(starts_ref[r], tokens - 1)]
            live = count > 0
            left = jnp.where(live, jnp.maximum(first - back, 0), 0)
            right = jnp.where(live, first + count, 0)
            return live, left, right, left // page_size

        def copy(r, g):
            start = pl.multiple_of(r * width + base_ref[r], 128)
            return pltpu.make_async_copy(
                table_refs[g].at[pl.ds(start, _CHUNK)],
                bufs[g].at[pl.ds(pl.multiple_of(r * _CHUNK, _CHUNK), _CHUNK)],
                sem.at[0],
            )

        def issue(r, carry):
            live, _, _, first_page = window(r)

            @pl.when(live)
            def _():
                start = jnp.minimum(first_page, width - columns)
                base_ref[r] = jnp.minimum(start // 128 * 128, width - _CHUNK)
                for g in range(groups):
                    copy(r, g).start()

            return carry

        def fill(r, carry):
            live, left, right, first_page = window(r)
            bd_ref[2 * r] = left
            bd_ref[2 * r + 1] = right
            for g in range(groups):
                md_ref[row(g, r)] = first_page * page_size
                md_ref[row(g, r) + 1] = min_ref[r]

            @pl.when(live)
            def _():
                for g in range(groups):
                    copy(r, g).wait()
                # Slice entry i holds column base + i; window column j is
                # column first_page + j (past the row end: masked below).
                at = r * _CHUNK - base_ref[r]
                for j in range(columns):
                    page = first_page + j
                    visible = (page < width) & (page * page_size < right)
                    entry = jnp.minimum(at + page, r * _CHUNK + _CHUNK - 1)
                    for g in range(groups):
                        physical = (
                            bufs[g][entry] * (page_stride // page_size)
                            + row_offset // page_size
                        )
                        md_ref[row(g, r) + 2 + j] = jnp.where(visible, physical, 0)

                def token(t, c):
                    # Under the contract a live token's page lies in its
                    # request's window; the clamp keeps any other position
                    # inside the request's own slice.
                    position = pos_ref[t]
                    entry = jnp.clip(
                        at + position // page_size,
                        r * _CHUNK,
                        r * _CHUNK + _CHUNK - 1,
                    )
                    for g in range(groups):
                        rec_ref[g * tokens + t] = (
                            row_offset
                            + bufs[g][entry] * page_stride
                            + position % page_size
                        )
                    return c

                jax.lax.fori_loop(starts_ref[r], starts_ref[r + 1], token, 0)

            @pl.when(jnp.logical_not(live))
            def _():
                for j in range(columns):
                    for g in range(groups):
                        md_ref[row(g, r) + 2 + j] = 0

            return carry

        def padding(t, carry):
            for g in range(groups):
                rec_ref[g * tokens + t] = -1
            return carry

        jax.lax.fori_loop(0, requests, issue, 0, unroll=8)
        jax.lax.fori_loop(0, requests, fill, 0, unroll=8)
        live_tokens = jnp.minimum(starts_ref[requests], tokens)
        jax.lax.fori_loop(live_tokens, tokens, padding, 0)

    smem = functools.partial(pl.BlockSpec, memory_space=pltpu.SMEM)
    # Without the constraint XLA prefetches each 4 MiB table into VMEM first.
    tables = tuple(pltpu.with_memory_space_constraint(t, pltpu.HBM) for t in tables)
    return pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((groups * requests * (2 + columns),), jnp.int32),
            jax.ShapeDtypeStruct((requests * 2,), jnp.int32),
            jax.ShapeDtypeStruct((groups * tokens,), jnp.int32),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=(1,),
            # The tables stay in HBM: only the live slices are read.
            in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)] * groups,
            # Flat: a 2-D output in scalar memory pads its minor dimension and
            # costs ~1 us more to write back.
            out_specs=(
                smem((groups * requests * (2 + columns),), lambda *_: (0,)),
                smem((requests * 2,), lambda *_: (0,)),
                smem((groups * tokens,), lambda *_: (0,)),
            ),
            scratch_shapes=[pltpu.SMEM((requests * _CHUNK,), jnp.int32)] * groups
            + [pltpu.SMEM((requests,), jnp.int32), pltpu.SemaphoreType.DMA((1,))],
        ),
        name="csa2_swa_metadata",
        interpret=interpret,
    )(starts, positions, minimum, *tables)
