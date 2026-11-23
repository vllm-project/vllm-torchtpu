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


# q38 PLE-UNIFORM-COND on the R form (layer U of rple_design_v1; port of L1's D07-b2
# ``bounded_request_rows`` / ``uniform_window_conv``, q38-L1-c64-w3 2bc75911f, credit L1):
# VLLM_TPU_PLE_UNIFORM_COND (default off, hashed). It sits below everything else and rebinds
# ``pooled_short_conv_speculative`` here, so with the flag unset every line above is unchanged
# and the stock function runs as before. With the flag set, a token-path bucket whose shape
# allows it (T % window == 0, n = T // window <= rows, state_len == (taps - 1) * dilation,
# VLLM_TPU_PLE_SKIP_IDLE_ROWS off) chooses ON THE DEVICE, by ``lax.cond``, between the uniform
# path (history for the n leading rows, taps and checkpoint windows by static slices) and the
# stock path (the same equations as the stock function); every other bucket runs the stock
# function with no cond. The checkpoint write stays outside the cond (one writer: the
# VLLM_TPU_PLE_WRDEFER dispatcher), so the pool is only read inside it.
_pooled_short_conv_speculative_stock = pooled_short_conv_speculative
_UNIFORM_SEEN: set = set()


def uniform_cond_enabled() -> bool:
    """VLLM_TPU_PLE_UNIFORM_COND: the device-side uniform-window choice (q38 PLE-UNIFORM-COND)."""
    return bool(envs.VLLM_TPU_PLE_UNIFORM_COND)


def bounded_request_rows(token_count: int, count: int, window: int) -> int:
    """Request rows a uniform-window bucket can fill (q38 L1 D07-b2).

    A bucket of ``token_count`` tokens whose live requests each hold ``window``
    tokens has at most ceil(token_count / window) of them, and they come first:
    the runner's metadata lists the scheduled requests in order and pads the
    tail with empty rows up to max-num-seqs (``count``).
    """
    return min(count, -(-token_count // window))


def uniform_window_pred(starts: jax.Array, rows: int, window: int) -> jax.Array:
    """pred = (L <= rows) & all(lengths[:L] == window) & all(lengths[L:] == 0), with
    L = starts[-1] // window, the live row count of a uniform bucket: True iff every
    token sits in one of the L leading rows, each holding exactly ``window`` tokens.
    (With L counted as the rows that hold tokens the tail check would be implied by the
    lead check; with this L each check carries weight: a short request after full ones,
    e.g. lengths [4, 4, 1, 1], passes the lead check and fails only the tail check.)"""
    lengths = starts[1:] - starts[:-1]
    live = starts[-1] // window
    lead = jnp.arange(lengths.shape[0], dtype=starts.dtype) < live
    lead_ok = jnp.all(jnp.where(lead, lengths == window, True))
    tail_ok = jnp.all(jnp.where(lead, True, lengths == 0))
    return (live <= rows) & lead_ok & tail_ok


def _note_uniform(kind: str, token_count: int, count: int, rows: int, window: int, why: str = "") -> None:
    """Trace-time engagement line, once per traced shape and kind."""
    key = (kind, token_count, count, rows, window, why)
    if key in _UNIFORM_SEEN:
        return
    _UNIFORM_SEEN.add(key)
    from vllm_torchtpu.logger import init_logger

    if kind == "cond":
        init_logger(__name__).info(
            "q38 PLE-UNIFORM-COND traced: %d tokens, device-side cond between the uniform path "
            "(%d of %d request rows, window %d, static slices) and the stock path",
            token_count,
            rows,
            count,
            window,
        )
    else:
        init_logger(__name__).info(
            "q38 PLE-UNIFORM-COND static fallback: %d tokens, %d request rows, window %d (%s); "
            "stock path, no cond",
            token_count,
            count,
            window,
            why,
        )


def uniform_window_conv(x, weight, history, starts, ids, initial, ckpt_ids, windowed, *, dilation):
    """``short_conv`` (request-owned history) plus ``token_checkpoints`` for a
    bucket of exactly ``count`` requests x ``window`` tokens whose live
    requests each hold ``window`` tokens and come first (q38 L1 D07-b2).
    Returns (output, values, targets).

    Token ``t`` is chunk token ``t % window`` of request ``t // window``, so
    every row the two functions gather is a static slice of
    ``E[r] = concat(history[r] masked by initial[r], x[r * window : (r + 1) * window])``:
    tap ``k`` of token ``(r, j)`` reads ``E[r, k * dilation + j]``, and its
    checkpoint is ``E[r, j + 1 : j + 1 + state_len]``. The f32 products and
    their tap order, the BF16 rounding points and the masks are those of
    ``short_conv``, so live tokens' outputs and every written (target, value)
    pair are bitwise equal. Padded tokens (past ``starts[-1]``) read other
    rows than there, but their outputs are masked to zero and they write
    nothing in both.
    """
    token_count, channels = x.shape
    count, window = ckpt_ids.shape
    state_len = history.shape[1]
    taps = weight.shape[-1]
    assert token_count == count * window, (token_count, count, window)
    assert state_len == (taps - 1) * dilation, (state_len, taps, dilation)
    lengths = starts[1:] - starts[:-1]
    active = (ids > 0) & (lengths > 0)
    past = jnp.where(initial[:, None, None], history, jnp.zeros((), history.dtype))
    extended = jnp.concatenate(
        [past.astype(x.dtype), x.reshape(count, window, channels)], axis=1
    )
    accum = jnp.zeros((token_count, channels), jnp.float32)
    for tap in range(taps):
        value = extended[:, tap * dilation : tap * dilation + window].reshape(
            token_count, channels
        )
        accum = accum + value.astype(jnp.float32) * weight[:, tap].astype(jnp.float32)
    pre = _ple_bf16(accum)
    output = _ple_bf16(jax.nn.silu(pre.astype(jnp.float32)))
    t = jnp.arange(token_count, dtype=jnp.int32)
    req = t // window
    rel = t % window
    valid = (t < starts[-1]) & active[req]
    output = jnp.where(valid[:, None], output, 0)

    values = jnp.stack(
        [extended[:, j + 1 : j + 1 + state_len] for j in range(window)], axis=1
    ).reshape(token_count, state_len * channels)
    win = windowed[req]
    write = valid & jnp.where(win, rel < window, rel == lengths[req] - 1)
    column = jnp.where(win, jnp.clip(rel, 0, window - 1), 0)
    targets = jnp.where(write, ckpt_ids[req, column], 0)
    return output, values, targets


def _ple_bf16(x):
    from vllm_torchtpu.kernels.qwen4_exp.ple import _bf16

    return _bf16(x)


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
    """The stock ``pooled_short_conv_speculative`` unless VLLM_TPU_PLE_UNIFORM_COND is set
    (q38 PLE-UNIFORM-COND): then a token-path bucket of exactly n = T // window request
    rows chooses on the device between ``uniform_window_conv`` (pred True) and the stock
    equations (pred False); live tokens' outputs and every written checkpoint are bitwise
    the stock function's either way."""
    args = (x, weight, pool, starts, ids, initial, read_ids, ckpt_ids, windowed)
    kw = dict(dilation=dilation, state_len=state_len)
    if not uniform_cond_enabled():
        return _pooled_short_conv_speculative_stock(*args, **kw)
    token_count, channels = x.shape
    count0, window = ckpt_ids.shape
    rows = token_count // window
    why = (
        "VLLM_TPU_PLE_SKIP_IDLE_ROWS" if envs.VLLM_TPU_PLE_SKIP_IDLE_ROWS
        else "T % window != 0" if token_count % window
        else "state_len != (taps - 1) * dilation" if state_len != (weight.shape[-1] - 1) * dilation
        else "request path (T // window > rows)" if not 0 < rows <= count0
        else ""
    )
    if why:
        _note_uniform("fallback", token_count, count0, rows, window, why)
        return _pooled_short_conv_speculative_stock(*args, **kw)
    starts, ids, initial, read_ids, ckpt_ids, windowed = step_request_rows(
        token_count, starts, ids, initial, read_ids, ckpt_ids, windowed
    )
    count = ckpt_ids.shape[0]
    _note_uniform("cond", token_count, count, rows, window)
    pred = uniform_window_pred(starts, rows, window)

    def uniform(_):
        history = read_region(
            pool, read_ids[:rows], elements=state_len * channels, dtype=jnp.bfloat16
        ).reshape(rows, state_len, channels)
        return uniform_window_conv(
            x, weight, history, starts[: rows + 1], ids[:rows], initial[:rows],
            ckpt_ids[:rows], windowed[:rows], dilation=dilation,
        )

    def stock(_):
        # The stock function's equations (token path, SKIP_IDLE_ROWS off), without its write.
        lengths = starts[1:] - starts[:-1]
        active = (ids > 0) & (lengths > 0)
        history = read_region(
            pool, read_ids, elements=state_len * channels, dtype=jnp.bfloat16
        ).reshape(count, state_len, channels)
        slots = jnp.where(active, jnp.arange(count, dtype=jnp.int32), -1)
        output, _final = short_conv(
            x,
            weight,
            history,
            starts,
            slots,
            initial,
            dilation=dilation,
            request_owned_history=True,
        )
        values, targets = token_checkpoints(
            x, history, starts, initial, active, ckpt_ids, windowed
        )
        return output, values, targets

    output, values, targets = jax.lax.cond(pred, uniform, stock, None)
    pool = write_region(pool, values, targets)
    return pool, output


# q38 PLE-CS (VLLM_TPU_PLE_CHANNEL_SHARD, default "0" = off, hashed): the decode engine's PLE
# short conv and its per-token checkpoint windows on one HC stream's channel slice per TP rank;
# the caller all-gathers the conv output over TP (models/vllm/qwen4_exp/pooled_ple.py). This
# section sits below everything else and only adds functions, so with the flag unset every line
# above, and with them the stock kernels' Mosaic payloads and debug locations, is unchanged.
#
# The history region keeps the full [state_len, channels] BF16 layout, so prefill, the PD
# transfer, page seeding and the host tier see no change. In a pool of 1024-byte tokens a region
# of 10240 channels holds window row w in tokens [20 w, 20 w + 20), and stream s (2560 channels)
# of that row is the run of 5 tokens at 20 w + 5 s (``cache_regions.region_rows`` geometry). The
# stream reader and writers move whole tokens of those runs through the same typed views
# (``typed_ldst.load_typed`` / ``store_typed``) as the full-region reader and writers; that view
# maps each pool token to its own two BF16 rows, so a stream view is bit for bit the slice of the
# full view. A rank writes only its stream's tokens and the other streams of its pool pages go
# stale, so every PLE conv call of the engine must take this path (pooled_ple binds it for the
# engine's lifetime). ``rank`` is an int32[1] operand, so every TP rank traces one program.
import math as _math

_CS_SEEN: set = set()
# Request rows one grid step of the stream reader fetches (state_len DMAs each).
CS_READ_ROWS = 4


def channel_shard_streams() -> int:
    """VLLM_TPU_PLE_CHANNEL_SHARD: HC streams the PLE conv channels split into (0 = stock op)."""
    return int(envs.VLLM_TPU_PLE_CHANNEL_SHARD or "0")


def stream_geometry(pool, channels: int, streams: int) -> tuple[int, int, int]:
    """(row_tokens, stream_tokens, stream_rows) of a [state_len, channels] BF16 history region:
    pool tokens per window row, pool tokens per stream run, and BF16 rows of ``pool.shape[-1]``
    lanes in one run's typed view."""
    token_bytes = _math.prod(pool.shape[2:]) * jnp.dtype(pool.dtype).itemsize
    lanes = pool.shape[-1]
    row_bytes = channels * 2
    if channels % streams or row_bytes % (token_bytes * streams) or token_bytes % (lanes * 2):
        raise ValueError(
            f"PLE-CS: {channels} BF16 channels in {streams} streams are not whole "
            f"{token_bytes}-byte pool tokens of {lanes} lanes"
        )
    row_tokens = row_bytes // token_bytes
    stream_tokens = row_tokens // streams
    stream_rows = stream_tokens * token_bytes // (lanes * 2)
    assert stream_rows * lanes == channels // streams, (stream_rows, lanes, channels, streams)
    return row_tokens, stream_tokens, stream_rows


def read_region_stream(pool, ids, stream, *, state_len, channels, streams, selected=None):
    """[len(ids), state_len, channels // streams] BF16: stream ``stream[0]`` of each id's history
    region, bit for bit ``read_region(pool, ids, elements=state_len * channels, dtype=bf16)
    .reshape(-1, state_len, channels)[:, :, s * w : (s + 1) * w]`` (ids <= 0 read page 0, as
    there). With ``selected``, rows where it is False move no bytes and read zero (callers mask
    them, as for ``read_selected_region``). One grid step fetches CS_READ_ROWS rows: it starts
    every DMA of its rows (one per window row), waits them all, then stores the typed views."""
    count = ids.shape[0]
    row_tokens, stream_tokens, stream_rows = stream_geometry(pool, channels, streams)
    assert state_len * row_tokens <= pool.shape[1], "state region exceeds its manager page"
    lanes = pool.shape[-1]
    payload = tuple(pool.shape[2:])
    if count == 0:
        return jnp.zeros((0, state_len, channels // streams), jnp.bfloat16)
    per = min(CS_READ_ROWS, count)
    padded = pl.cdiv(count, per) * per
    blocks = jnp.pad(jnp.maximum(ids, 0).astype(jnp.int32), (0, padded - count))
    sel = jnp.ones((count,), jnp.int32) if selected is None else selected.astype(jnp.int32)
    sel = jnp.pad(sel, (0, padded - count))

    def kernel(blocks_ref, sel_ref, stream_ref, source, out, staged, sem):
        base = pl.program_id(0) * per
        first = stream_ref[0] * stream_tokens

        def copy(j, w):
            return pltpu.make_async_copy(
                source.at[blocks_ref[base + j], pl.ds(w * row_tokens + first, stream_tokens)],
                staged.at[j, w],
                sem,
            )

        for j in range(per):
            @pl.when(sel_ref[base + j] > 0)
            def _():
                for w in range(state_len):
                    copy(j, w).start()

        for j in range(per):
            @pl.when(sel_ref[base + j] > 0)
            def _():
                for w in range(state_len):
                    copy(j, w).wait()

        for j in range(per):
            @pl.when(sel_ref[base + j] > 0)
            def _():
                for w in range(state_len):
                    out[j, w] = typed_ldst.load_typed(staged.at[j, w], view_dtype=jnp.bfloat16)

            @pl.when(sel_ref[base + j] <= 0)
            def _():
                out[j] = jnp.zeros((state_len, stream_rows, lanes), jnp.bfloat16)

    values = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=(padded // per,),
            in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
            out_specs=pl.BlockSpec(
                (per, state_len, stream_rows, lanes), lambda i, *_: (i, 0, 0, 0)
            ),
            scratch_shapes=[
                pltpu.VMEM((per, state_len, stream_tokens) + payload, pool.dtype),
                pltpu.SemaphoreType.DMA,
            ],
        ),
        out_shape=jax.ShapeDtypeStruct((padded, state_len, stream_rows, lanes), jnp.bfloat16),
        name="q38_ple_cs_read",
    )(blocks, sel, stream.astype(jnp.int32).reshape(1), pool)
    return values[:count].reshape(count, state_len, stream_rows * lanes)


def write_region_stream(
    pool, values, ids, stream, *, state_len, channels, streams, skip_idle_source=False
):
    """``write_region`` of each id's stream run: row i of ``values`` ([count, state_len * w]
    BF16, one stream's window) lands at stream ``stream[0]``'s tokens of page ``ids[i]``'s full
    history region; ids <= 0 and every byte outside those tokens stay intact. The stock
    writer's structure (one grid step per row; the row's DMAs waited before the next row)."""
    count, elements = values.shape
    row_tokens, stream_tokens, stream_rows = stream_geometry(pool, channels, streams)
    lanes = pool.shape[-1]
    assert elements == state_len * stream_rows * lanes, (elements, state_len, stream_rows)
    assert state_len * row_tokens <= pool.shape[1], "state region exceeds its manager page"
    if count == 0:
        return pool
    payload = tuple(pool.shape[2:])
    values = values.reshape(count, state_len, stream_rows, lanes)
    stream = stream.astype(jnp.int32).reshape(1)

    def kernel(indices, stream_ref, *refs):
        source, old, target = refs[-3:]
        target_id = indices[pl.program_id(0)]

        @pl.when(target_id > 0)
        def store():
            first = stream_ref[0] * stream_tokens

            def copy(encoded, semaphore):
                def dma(w):
                    return pltpu.make_async_copy(
                        encoded.at[w],
                        target.at[target_id, pl.ds(w * row_tokens + first, stream_tokens)],
                        semaphore,
                    )

                for w in range(state_len):
                    typed_ldst.store_typed(encoded.at[w], source[0, w])
                for w in range(state_len):
                    dma(w).start()
                for w in range(state_len):
                    dma(w).wait()

            pl.run_scoped(
                copy,
                pltpu.VMEM((state_len, stream_tokens) + payload, pool.dtype),
                pltpu.SemaphoreType.DMA,
            )

    block = (1, state_len, stream_rows, lanes)
    if skip_idle_source:
        prefetch = (ids, stream, _last_selected_row(ids > 0))
        source_spec = pl.BlockSpec(block, lambda i, indices, st, src: (src[i], 0, 0, 0))
    else:
        prefetch = (ids, stream)
        source_spec = pl.BlockSpec(block, lambda i, indices, st: (i, 0, 0, 0))
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
        name="q38_ple_cs_write",
    )(*prefetch, values, pool)


def write_region_stream_ring(pool, values, ids, stream, *, state_len, channels, streams, depth=4):
    """``write_region_stream`` with ``depth`` rows of checkpoint DMAs in flight: the ring of
    ``write_region_ring`` (q38 PLE-WRDEFER), each row writing its state_len stream runs on its
    slot's semaphore and waiting all of them before the slot is reused or at the drain. Written
    ids are unique and positive, so no two DMAs touch the same bytes and the pool is bitwise
    ``write_region_stream``'s."""
    count, elements = values.shape
    row_tokens, stream_tokens, stream_rows = stream_geometry(pool, channels, streams)
    lanes = pool.shape[-1]
    assert elements == state_len * stream_rows * lanes, (elements, state_len, stream_rows)
    assert state_len * row_tokens <= pool.shape[1], "state region exceeds its manager page"
    if count == 0:
        return pool
    depth = max(1, min(int(depth), count))
    payload = tuple(pool.shape[2:])
    values = values.reshape(count, state_len, stream_rows, lanes)
    stream = stream.astype(jnp.int32).reshape(1)

    def kernel(indices, stream_ref, source, old, target, staged, encoded, rsem, wsem):
        first = stream_ref[0] * stream_tokens

        def read(i, slot):
            return pltpu.make_async_copy(source.at[i], staged.at[slot], rsem.at[slot])

        def write(i, slot, w):
            return pltpu.make_async_copy(
                encoded.at[slot, w],
                target.at[indices[i], pl.ds(w * row_tokens + first, stream_tokens)],
                wsem.at[slot],
            )

        def wait_writes(i, slot):
            for w in range(state_len):
                write(i, slot, w).wait()

        def row(i, slot):
            @_when(i >= depth)
            def _():
                @pl.when(indices[i - depth] > 0)
                def _():
                    wait_writes(i - depth, slot)

            @pl.when(indices[i] > 0)
            def _():
                read(i, slot).wait()
                for w in range(state_len):
                    typed_ldst.store_typed(encoded.at[slot, w], staged[slot, w])
                for w in range(state_len):
                    write(i, slot, w).start()

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
                wait_writes(i, i % depth)

    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(1,),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[
                pltpu.VMEM((depth, state_len, stream_rows, lanes), values.dtype),
                pltpu.VMEM((depth, state_len, stream_tokens) + payload, pool.dtype),
                pltpu.SemaphoreType.DMA((depth,)),
                pltpu.SemaphoreType.DMA((depth,)),
            ],
        ),
        out_shape=jax.ShapeDtypeStruct(pool.shape, pool.dtype),
        input_output_aliases={3: 0},
        name="q38_ple_cs_wrdefer",
    )(ids, stream, values, pool)


def write_region_stream_any(pool, values, ids, stream, *, skip_idle_source=False, **geometry):
    """The stream writer VLLM_TPU_PLE_WRDEFER selects, as ``write_region`` does for the full
    region: the ring with K > 0 and ``skip_idle_source`` off, else the stock structure."""
    depth = wrdefer_depth()
    if depth and not skip_idle_source:
        return write_region_stream_ring(pool, values, ids, stream, depth=depth, **geometry)
    return write_region_stream(
        pool, values, ids, stream, skip_idle_source=skip_idle_source, **geometry
    )


def _cs_inputs(x, weight, rank, streams, split):
    """(x, weight) cut to stream ``rank // split``, that stream as int32[1], and the part
    ``rank % split`` of the stream's output this rank contributes to the all-gather."""
    width = x.shape[-1] // streams
    r = rank.astype(jnp.int32).reshape(-1)[0]
    stream, part = r // split, r % split
    xs = jax.lax.dynamic_slice_in_dim(x, stream * width, width, axis=1)
    ws = jax.lax.dynamic_slice_in_dim(weight, stream * width, width, axis=0)
    return xs, ws, stream.reshape(1), part


def _cs_piece(output, part, split):
    """This rank's all-gather piece: part ``part`` of ``split`` of its stream's output."""
    if split == 1:
        return output
    width = output.shape[-1] // split
    return jax.lax.dynamic_slice_in_dim(output, part * width, width, axis=1)


def _note_cs(kind, token_count, count, channels, streams, split, skip_idle) -> None:
    """Trace-time engagement line, once per traced shape and kind."""
    depth = 0 if skip_idle else wrdefer_depth()
    key = (kind, token_count, count, channels, streams, split, depth, bool(skip_idle))
    if key in _CS_SEEN:
        return
    _CS_SEEN.add(key)
    from vllm_torchtpu.logger import init_logger

    init_logger(__name__).info(
        "q38 PLE-CS traced: %s, %d tokens, %d request rows, stream of %d of %d channels, "
        "all-gather piece %d channels, writer %s%s",
        kind,
        token_count,
        count,
        channels // streams,
        channels,
        channels // (streams * split),
        f"ring depth {depth}" if depth else "stock",
        ", idle rows skipped" if skip_idle else "",
    )


def pooled_short_conv_cs(
    x: jax.Array,
    weight: jax.Array,
    pool: jax.Array,
    starts: jax.Array,
    ids: jax.Array,
    initial: jax.Array,
    rank: jax.Array,
    *,
    dilation: int,
    state_len: int,
    streams: int,
    split: int,
) -> tuple[jax.Array, jax.Array]:
    """``pooled_short_conv`` on stream ``rank // split`` (q38 PLE-CS). Returns (pool, piece):
    piece = part ``rank % split`` of the stream's output, so the TP all-gather of the pieces in
    rank order is the stock output bit for bit; the pool gets the stream's tokens of the stock
    write and keeps every other byte."""
    channels = x.shape[-1]
    xs, ws, stream, part = _cs_inputs(x, weight, rank, streams, split)
    geometry = dict(state_len=state_len, channels=channels, streams=streams)
    _note_cs("pooled", x.shape[0], len(ids), channels, streams, split, False)
    history = read_region_stream(pool, ids, stream, **geometry)
    active = (ids > 0) & (starts[1:] > starts[:-1])
    slots = jnp.where(active, jnp.arange(len(ids), dtype=jnp.int32), -1)
    output, updated = short_conv(
        xs,
        ws,
        history,
        starts,
        slots,
        initial,
        dilation=dilation,
        request_owned_history=True,
    )
    pool = write_region_stream_any(
        pool, updated.reshape(len(ids), -1), jnp.where(active, ids, 0), stream, **geometry
    )
    return pool, _cs_piece(output, part, split)


def pooled_short_conv_speculative_cs(
    x: jax.Array,
    weight: jax.Array,
    pool: jax.Array,
    starts: jax.Array,
    ids: jax.Array,
    initial: jax.Array,
    read_ids: jax.Array,
    ckpt_ids: jax.Array,
    windowed: jax.Array,
    rank: jax.Array,
    *,
    dilation: int,
    state_len: int,
    streams: int,
    split: int,
) -> tuple[jax.Array, jax.Array]:
    """The stock ``pooled_short_conv_speculative`` (token and request paths,
    VLLM_TPU_PLE_SKIP_IDLE_ROWS, VLLM_TPU_PLE_WRDEFER) on stream ``rank // split`` (q38 PLE-CS).
    Returns (pool, piece) as ``pooled_short_conv_cs``: every written checkpoint gets the stream's
    tokens of the stock checkpoint and no other byte of the pool changes."""
    channels = x.shape[-1]
    xs, ws, stream, part = _cs_inputs(x, weight, rank, streams, split)
    starts, ids, initial, read_ids, ckpt_ids, windowed = step_request_rows(
        x.shape[0], starts, ids, initial, read_ids, ckpt_ids, windowed
    )
    count, window = ckpt_ids.shape
    lengths = starts[1:] - starts[:-1]
    active = (ids > 0) & (lengths > 0)
    skip_idle = envs.VLLM_TPU_PLE_SKIP_IDLE_ROWS
    geometry = dict(state_len=state_len, channels=channels, streams=streams)
    token_path = x.shape[0] <= count * window
    _note_cs(
        "speculative token path" if token_path else "speculative request path",
        x.shape[0], count, channels, streams, split, skip_idle,
    )
    history = read_region_stream(
        pool, read_ids, stream, selected=active if skip_idle else None, **geometry
    )
    slots = jnp.where(active, jnp.arange(count, dtype=jnp.int32), -1)
    output, final = short_conv(
        xs,
        ws,
        history,
        starts,
        slots,
        initial,
        dilation=dilation,
        request_owned_history=True,
    )
    if token_path:
        values, targets = token_checkpoints(
            xs, history, starts, initial, active, ckpt_ids, windowed
        )
    else:
        values, targets = request_checkpoints(
            xs, history, final, starts, initial, active, ckpt_ids, windowed
        )
    pool = write_region_stream_any(
        pool, values, targets, stream, skip_idle_source=skip_idle, **geometry
    )
    return pool, _cs_piece(output, part, split)
