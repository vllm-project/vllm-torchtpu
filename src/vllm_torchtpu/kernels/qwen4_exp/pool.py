# SPDX-License-Identifier: Apache-2.0
"""Typed state regions in Qwen4Exp's native attention-shaped pool.

Only requested regions cross HBM. Type reinterpretation and lane reshaping
take place inside Pallas; the persistent pool never gets a typed XLA view.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import envs
from vllm_torchtpu.kernels import pool_adapters, typed_ldst
from vllm_torchtpu.kernels.qwen4_exp.cache_regions import region_rows
from vllm_torchtpu.kernels.qwen4_exp.history_rows import (
    gather_history_rows,
    history_row_kernel_enabled,
)
from vllm_torchtpu.kernels.qwen4_exp.ple import short_conv


def read_region(pool, ids, *, elements, dtype, tok0=0):
    ntok, rows = region_rows(pool, elements, dtype)
    values = pool_adapters.gather_region(
        pool,
        jnp.maximum(ids, 0),
        tok0=tok0,
        ntok=ntok,
        out_dtype=dtype,
    )
    return values.reshape(len(ids), rows * pool.shape[-1])[:, :elements]


def _last_selected_row(selected):
    """For each row, the last row at or before it where ``selected`` holds (0 if none).

    As a block index this repeats the previous grid step's index on every
    unselected row, and the Pallas pipeline neither fetches nor writes back a
    block whose index is unchanged from the step before.
    """
    rows = jnp.arange(selected.shape[0], dtype=jnp.int32)
    return jax.lax.cummax(jnp.where(selected, rows, 0))


def read_selected_region(pool, ids, selected, *, elements, dtype):
    """`read_region` that moves bytes only for the rows where ``selected`` holds.

    Unselected rows of the result are not written: they hold whatever the
    output buffer held, and callers must mask them. The selected rows are
    byte-identical to `read_region`'s.
    """
    ntok, rows = region_rows(pool, elements, dtype)
    count = ids.shape[0]
    assert ntok <= pool.shape[1], "state region exceeds its manager page"
    payload = tuple(pool.shape[2:-1])
    lanes = pool.shape[-1]
    owner = _last_selected_row(selected)
    blocks = jnp.maximum(ids, 0)[owner]
    pad = (0,) * (len(payload) + 1)

    def kernel(blocks_ref, owner_ref, source, out):
        @pl.when(owner_ref[pl.program_id(0)] == pl.program_id(0))
        def load():
            out[...] = typed_ldst.load_typed(source.at[0], view_dtype=dtype)[None]

    values = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(count,),
            in_specs=[
                pl.BlockSpec(
                    (1, ntok) + payload + (lanes,),
                    lambda i, blocks, owner: (blocks[i], 0) + pad,
                )
            ],
            out_specs=pl.BlockSpec(
                (1, rows, lanes), lambda i, blocks, owner: (owner[i], 0, 0)
            ),
        ),
        out_shape=jax.ShapeDtypeStruct((count, rows, lanes), dtype),
    )(blocks, owner, pool)
    return values.reshape(count, rows * lanes)[:, :elements]


def write_region(pool, values, ids, *, tok0=0, skip_idle_source=False):
    """Write unique positive manager IDs; idle/null IDs leave all bytes intact.

    With ``skip_idle_source`` the value rows of idle IDs are not fetched
    either: their source block index repeats the last written row's.
    """
    count, elements = values.shape
    ntok, rows = region_rows(pool, elements, values.dtype)
    assert tok0 + ntok <= pool.shape[1], "state region exceeds its manager page"
    lanes = pool.shape[-1]
    values = jnp.pad(values, ((0, 0), (0, rows * lanes - elements))).reshape(
        count, rows, lanes
    )
    payload = tuple(pool.shape[2:])

    def kernel(indices, *refs):
        source, old, target = refs[-3:]

        @pl.when(indices[pl.program_id(0)] > 0)
        def store():
            def copy(encoded, semaphore):
                typed_ldst.store_typed(encoded, source[0])
                pltpu.async_copy(
                    encoded,
                    target.at[indices[pl.program_id(0)], pl.ds(tok0, ntok)],
                    semaphore,
                ).wait()

            pl.run_scoped(
                copy, pltpu.VMEM((ntok,) + payload, pool.dtype), pltpu.SemaphoreType.DMA
            )

    if skip_idle_source:
        prefetch = (ids, _last_selected_row(ids > 0))
        source_spec = pl.BlockSpec(
            (1, rows, lanes), lambda i, indices, src: (src[i], 0, 0)
        )
    else:
        prefetch = (ids,)
        source_spec = pl.BlockSpec((1, rows, lanes), lambda i, indices: (i, 0, 0))
    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(prefetch),
            grid=(count,),
            in_specs=[source_spec, pl.BlockSpec(memory_space=pltpu.HBM)],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
        ),
        out_shape=jax.ShapeDtypeStruct(pool.shape, pool.dtype),
        input_output_aliases={len(prefetch) + 1: 0},
    )(*prefetch, values, pool)


def pooled_short_conv(
    x: jax.Array,
    weight: jax.Array,
    pool: jax.Array,
    starts: jax.Array,
    ids: jax.Array,
    initial: jax.Array,
    *,
    dilation: int,
    state_len: int,
) -> tuple[jax.Array, jax.Array]:
    channels = x.shape[-1]
    history = read_region(
        pool, ids, elements=state_len * channels, dtype=jnp.bfloat16
    ).reshape(len(ids), state_len, channels)
    active = (ids > 0) & (starts[1:] > starts[:-1])
    slots = jnp.where(active, jnp.arange(len(ids), dtype=jnp.int32), -1)
    output, updated = short_conv(
        x,
        weight,
        history,
        starts,
        slots,
        initial,
        dilation=dilation,
        request_owned_history=True,
    )
    pool = write_region(pool, updated.reshape(len(ids), -1), jnp.where(active, ids, 0))
    return pool, output


def speculative_history_windows(
    x: jax.Array,
    history: jax.Array,
    starts: jax.Array,
    initial: jax.Array,
    window: int,
) -> jax.Array:
    """Each request's history after its first ``window`` chunk tokens.

    Entry ``[r, t]`` holds the ``state_len`` inputs ending at chunk token
    ``t`` of request ``r`` (oldest first), reading positions before the chunk
    from ``history`` (zero without initial state), exactly the window
    ``short_conv`` would carry had the chunk ended at token ``t``.
    """
    token_count = x.shape[0]
    state_len = history.shape[1]
    steps = jnp.arange(window, dtype=jnp.int32)[:, None]
    # [window, state_len] chunk-relative source positions.
    positions = steps + 1 - state_len + jnp.arange(state_len, dtype=jnp.int32)[None]
    clipped = jnp.clip(positions + state_len, 0, state_len - 1)
    if history_row_kernel_enabled():
        rows = jnp.arange(history.shape[0], dtype=jnp.int32)[:, None, None]
        past = gather_history_rows(history, rows * state_len + clipped[None])
    else:
        past = history[:, clipped]
    past = jnp.where(initial[:, None, None, None], past, 0)
    current = x[
        jnp.clip(
            starts[:-1, None, None].astype(jnp.int32) + positions[None],
            0,
            max(token_count - 1, 0),
        )
    ]
    return jnp.where((positions >= 0)[None, ..., None], current, past.astype(x.dtype))


def pooled_short_conv_speculative(
    x: jax.Array,
    weight: jax.Array,
    pool: jax.Array,
    starts: jax.Array,
    ids: jax.Array,
    initial: jax.Array,
    read_ids: jax.Array,
    ckpt_ids: jax.Array,
    windowed: jax.Array,
    *,
    dilation: int,
    state_len: int,
) -> tuple[jax.Array, jax.Array]:
    """PLE history with per-position checkpoints for speculative verification.

    Mirrors the pooled GDN state contract. ``read_ids`` is each request's
    checkpoint of its last accepted token (``ckpt_ids[r, read_offset]``).
    Windowed requests (decodes and verify windows, the runner's leading
    segment) write the history after chunk token ``t`` to ``ckpt_ids[r, t]``,
    so the next step resumes from whichever token was accepted last. Other
    requests (prefill/mixed) write their final history to ``ckpt_ids[r, 0]``,
    which is where a read offset of zero points.

    Only the first ``min(R, T)`` request rows are read and written; see
    `step_request_rows`. With VLLM_TPU_PLE_SKIP_IDLE_ROWS the history of
    inactive rows is not read (every use of it is masked by ``active``) and
    the checkpoint write fetches only the value rows it writes.
    """
    channels = x.shape[-1]
    starts, ids, initial, read_ids, ckpt_ids, windowed = step_request_rows(
        x.shape[0], starts, ids, initial, read_ids, ckpt_ids, windowed
    )
    count, window = ckpt_ids.shape
    lengths = starts[1:] - starts[:-1]
    active = (ids > 0) & (lengths > 0)
    skip_idle = envs.VLLM_TPU_PLE_SKIP_IDLE_ROWS
    if skip_idle:
        history = read_selected_region(
            pool, read_ids, active, elements=state_len * channels, dtype=jnp.bfloat16
        )
    else:
        history = read_region(
            pool, read_ids, elements=state_len * channels, dtype=jnp.bfloat16
        )
    history = history.reshape(count, state_len, channels)
    slots = jnp.where(active, jnp.arange(count, dtype=jnp.int32), -1)
    output, final = short_conv(
        x,
        weight,
        history,
        starts,
        slots,
        initial,
        dilation=dilation,
        request_owned_history=True,
    )
    if x.shape[0] <= count * window:
        values, targets = token_checkpoints(
            x, history, starts, initial, active, ckpt_ids, windowed
        )
    else:
        values, targets = request_checkpoints(
            x, history, final, starts, initial, active, ckpt_ids, windowed
        )
    pool = write_region(pool, values, targets, skip_idle_source=skip_idle)
    return pool, output


def step_request_rows(token_count, starts, ids, initial, read_ids, ckpt_ids, windowed):
    """The per-request inputs cut to the rows a step of ``token_count`` tokens can use.

    Every request with tokens has at least one, and the step metadata puts
    them ahead of the padded rows, whose starts repeat the last one. So with
    T tokens and R > T rows, rows T and later are empty (``starts[T] ==
    starts[-1]``) and dropping them leaves every output and every non-null
    checkpoint write unchanged, while the history read and the per-token
    gathers address T rows instead of R. T is the static token bucket.
    """
    count = ckpt_ids.shape[0]
    if not 0 < token_count < count:
        return starts, ids, initial, read_ids, ckpt_ids, windowed
    n = token_count
    return (
        starts[: n + 1],
        ids[:n],
        initial[:n],
        read_ids[:n],
        ckpt_ids[:n],
        windowed[:n],
    )


def request_checkpoints(x, history, final, starts, initial, active, ckpt_ids, windowed):
    """(values, targets) for every (request, window position) pair."""
    count, window = ckpt_ids.shape
    lengths = starts[1:] - starts[:-1]
    per_token = speculative_history_windows(x, history, starts, initial, window)
    steps = jnp.arange(window, dtype=jnp.int32)[None]
    # Non-windowed requests keep only their final history, in checkpoint 0.
    values = jnp.where(
        windowed[:, None, None, None],
        per_token,
        jnp.broadcast_to(final[:, None], per_token.shape),
    )
    write = active[:, None] & jnp.where(
        windowed[:, None], steps < lengths[:, None], steps == 0
    )
    targets = jnp.where(write, ckpt_ids, 0).reshape(-1)
    return values.reshape(count * window, -1), targets


def token_checkpoints(x, history, starts, initial, active, ckpt_ids, windowed):
    """The same written (target, value) pairs as `request_checkpoints`, one row per token.

    A checkpoint is the window ending at one chunk token, so decode and verify
    buckets (T <= count * window) build T rows instead of count * window.
    Windowed tokens past the window are dropped as before; a non-windowed
    request writes its last token's window (its final history) to column 0.
    """
    token_count = x.shape[0]
    count, window = ckpt_ids.shape
    state_len = history.shape[1]
    lengths = starts[1:] - starts[:-1]
    t = jnp.arange(token_count, dtype=jnp.int32)
    req = jnp.minimum(jnp.sum(t[:, None] >= starts[None, 1:], axis=1), count - 1)
    rel = t - starts[req]
    positions = rel[:, None] + 1 - state_len + jnp.arange(state_len, dtype=jnp.int32)[None]
    clipped = jnp.clip(positions + state_len, 0, state_len - 1)
    if history_row_kernel_enabled():
        past = gather_history_rows(history, req[:, None] * state_len + clipped)
    else:
        past = history[req[:, None], clipped]
    past = jnp.where(initial[req][:, None, None], past, 0)
    current = x[jnp.clip(starts[req][:, None] + positions, 0, max(token_count - 1, 0))]
    values = jnp.where((positions >= 0)[..., None], current, past.astype(x.dtype))
    win = windowed[req]
    write = (
        (t < starts[-1])
        & active[req]
        & jnp.where(win, rel < window, rel == lengths[req] - 1)
    )
    column = jnp.where(win, jnp.clip(rel, 0, window - 1), 0)
    targets = jnp.where(write, ckpt_ids[req, column], 0)
    return values.reshape(token_count, -1), targets


# q38 PLE-WRDEFER on the R form (port of L1's PLE-WRDEFER, q38-L1-plewd 2c0da1bd8, as served on
# q38-L1-c64-w3 2bc75911f): VLLM_TPU_PLE_WRDEFER = K (default 0 = off, hashed) keeps K checkpoint
# DMAs in flight. It sits below everything else and rebinds ``write_region`` here, so with the flag
# unset every line above, and with them the stock writer's Mosaic payload and debug locations, is
# unchanged; callers resolve ``write_region`` at call time. Only ``skip_idle_source=False`` takes the
# ring: with VLLM_TPU_PLE_SKIP_IDLE_ROWS (``skip_idle_source=True``) the stock writer runs as before.
_write_region_stock = write_region
_WRDEFER_SEEN: set = set()


def wrdefer_depth() -> int:
    """VLLM_TPU_PLE_WRDEFER: checkpoint DMAs the PLE writer keeps in flight (0 = stock writer)."""
    return int(envs.VLLM_TPU_PLE_WRDEFER or "0")


def write_region(pool, values, ids, *, tok0=0, skip_idle_source=False):
    """Write unique positive manager IDs; idle/null IDs leave all bytes intact.

    The stock writer (one grid step and one waited DMA per row) unless
    VLLM_TPU_PLE_WRDEFER = K > 0 and ``skip_idle_source`` is off (q38 PLE-WRDEFER):
    ``write_region_ring``.
    """
    depth = wrdefer_depth()
    if depth and not skip_idle_source:
        return write_region_ring(pool, values, ids, tok0=tok0, depth=depth)
    return _write_region_stock(
        pool, values, ids, tok0=tok0, skip_idle_source=skip_idle_source
    )


def _note_wrdefer(count: int, ntok: int, depth: int) -> None:
    """Trace-time engagement line, once per traced (rows, region, depth)."""
    key = (count, ntok, depth)
    if key in _WRDEFER_SEEN:
        return
    _WRDEFER_SEEN.add(key)
    from vllm_torchtpu.logger import init_logger

    init_logger(__name__).info(
        "q38 PLE-WRDEFER traced: %d rows x %d pool tokens, %d checkpoint DMAs in flight, "
        "one grid step",
        count,
        ntok,
        depth,
    )


def _when(condition):
    """``pl.when`` that also takes a static Python bool (static rows of the ring)."""
    if isinstance(condition, bool):
        return (lambda f: f()) if condition else (lambda f: None)
    return pl.when(condition)


def write_region_ring(pool, values, ids, *, tok0=0, depth=4):
    """``write_region`` with ``depth`` checkpoint DMAs in flight (q38 PLE-WRDEFER).

    One grid step walks the rows. Row ``i`` uses ring slot ``s = i % depth``: its
    values are DMA'd HBM -> VMEM slot ``s`` (started ``depth`` rows ahead), stored
    through the same typed view as the stock writer (``typed_ldst.store_typed``) into
    the slot's pool-layout buffer, and DMA'd to ``pool[ids[i], tok0 : tok0 + ntok]``.
    Row ``i`` first waits row ``i - depth``'s write (the slot's previous user), and
    the last ``depth`` writes are drained at the end, so every started DMA is waited
    exactly once before the kernel returns. Rows with ``ids <= 0`` start and wait
    nothing and leave every byte intact. Written ids are unique and positive, so no
    two DMAs touch the same bytes and their completion order cannot change the
    result: the pool is bitwise the stock writer's. ``depth = 1`` keeps one write in
    flight (the stock order, plus the read-ahead of the next row's values).
    """
    count, elements = values.shape
    ntok, rows = region_rows(pool, elements, values.dtype)
    assert tok0 + ntok <= pool.shape[1], "state region exceeds its manager page"
    if count == 0:
        return pool
    depth = max(1, min(int(depth), count))
    lanes = pool.shape[-1]
    values = jnp.pad(values, ((0, 0), (0, rows * lanes - elements))).reshape(
        count, rows, lanes
    )
    payload = tuple(pool.shape[2:])
    _note_wrdefer(count, ntok, depth)

    def kernel(indices, source, old, target, staged, encoded, rsem, wsem):
        def read(i, slot):
            return pltpu.make_async_copy(source.at[i], staged.at[slot], rsem.at[slot])

        def write(i, slot):
            return pltpu.make_async_copy(
                encoded.at[slot],
                target.at[indices[i], pl.ds(tok0, ntok)],
                wsem.at[slot],
            )

        def row(i, slot):
            # i: row (traced in the loop, static in the tail); slot = i % depth, static.
            @_when(i >= depth)
            def _():
                @pl.when(indices[i - depth] > 0)
                def _():
                    write(i - depth, slot).wait()

            @pl.when(indices[i] > 0)
            def _():
                read(i, slot).wait()
                typed_ldst.store_typed(encoded.at[slot], staged[slot])
                write(i, slot).start()

            @_when(i + depth < count)
            def _():
                @pl.when(indices[i + depth] > 0)
                def _():
                    read(i + depth, slot).start()

        for i in range(depth):  # read-ahead of the first ring
            @pl.when(indices[i] > 0)
            def _():
                read(i, i).start()

        def chunk(c, carry):
            for slot in range(depth):
                row(c * depth + slot, slot)
            return carry

        full = count // depth
        jax.lax.fori_loop(0, full, chunk, 0)
        for slot in range(count - full * depth):
            row(full * depth + slot, slot)
        for i in range(count - depth, count):  # drain
            @pl.when(indices[i] > 0)
            def _():
                write(i, i % depth).wait()

    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(1,),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[
                pltpu.VMEM((depth, rows, lanes), values.dtype),
                pltpu.VMEM((depth, ntok) + payload, pool.dtype),
                pltpu.SemaphoreType.DMA((depth,)),
                pltpu.SemaphoreType.DMA((depth,)),
            ],
        ),
        out_shape=jax.ShapeDtypeStruct(pool.shape, pool.dtype),
        input_output_aliases={2: 0},
        name="q38_ple_wrdefer",
    )(ids, values, pool)
