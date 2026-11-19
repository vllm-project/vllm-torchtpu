# SPDX-License-Identifier: Apache-2.0
"""Typed state regions in Qwen4Exp's native attention-shaped pool.

Only requested regions cross HBM. Type reinterpretation and lane reshaping
take place inside Pallas; the persistent pool never gets a typed XLA view.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels import pool_adapters, typed_ldst
from vllm_torchtpu.kernels.qwen4_exp.cache_regions import region_rows
from vllm_torchtpu.kernels.qwen4_exp.ple import _bf16, short_conv


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


def write_region(pool, values, ids, *, tok0=0):
    """Write unique positive manager IDs; idle/null IDs leave all bytes intact."""
    count, elements = values.shape
    ntok, rows = region_rows(pool, elements, values.dtype)
    assert tok0 + ntok <= pool.shape[1], "state region exceeds its manager page"
    lanes = pool.shape[-1]
    values = jnp.pad(values, ((0, 0), (0, rows * lanes - elements))).reshape(
        count, rows, lanes
    )
    payload = tuple(pool.shape[2:])

    def kernel(indices, source, old, target):
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

    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(count,),
            in_specs=[
                pl.BlockSpec((1, rows, lanes), lambda i, indices: (i, 0, 0)),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
        ),
        out_shape=jax.ShapeDtypeStruct(pool.shape, pool.dtype),
        input_output_aliases={2: 0},
    )(ids, values, pool)


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
    past_matmul: bool = False,
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
        past_matmul=past_matmul,
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
    past = history[:, jnp.clip(positions + state_len, 0, state_len - 1)]
    past = jnp.where(initial[:, None, None, None], past, 0)
    current = x[
        jnp.clip(
            starts[:-1, None, None].astype(jnp.int32) + positions[None],
            0,
            max(token_count - 1, 0),
        )
    ]
    return jnp.where((positions >= 0)[None, ..., None], current, past.astype(x.dtype))


def bounded_request_rows(token_count: int, count: int, window: int) -> int:
    """Request rows a uniform-window bucket can fill (q38 L1 D07-b2).

    A bucket of ``token_count`` tokens whose live requests each hold ``window``
    tokens has at most ceil(token_count / window) of them, and they come first:
    the runner's metadata lists the scheduled requests in order and pads the
    tail with empty rows up to max-num-seqs (``count``).
    """
    return min(count, -(-token_count // window))


_BOUNDED_SEEN: set = set()


def _note_bounded(mode: int, token_count: int, count: int, rows: int, window: int) -> None:
    """Trace-time engagement line, once per traced shape (a compiled program
    that comes from a cache is not traced again, so it logs nothing)."""
    key = (mode, token_count, count, rows, window)
    if key in _BOUNDED_SEEN:
        return
    _BOUNDED_SEEN.add(key)
    from vllm_torchtpu.logger import init_logger

    init_logger(__name__).info(
        "q38 PLE_DECODE_BOUNDED=%d traced: %d tokens, history for %d of %d "
        "request rows (window %d)%s",
        mode,
        token_count,
        rows,
        count,
        window,
        ", static windows" if mode >= 2 and token_count == rows * window else "",
    )


_GATED_SEEN: set = set()


def _note_bounded_gated(mode: int, token_count: int, min_rows: int) -> None:
    """Trace-time engagement line of the min-rows gate (q38 L1-GLUEFIX), once per
    traced shape, as `_note_bounded`."""
    key = (mode, token_count, min_rows)
    if key in _GATED_SEEN:
        return
    _GATED_SEEN.add(key)
    from vllm_torchtpu.logger import init_logger

    init_logger(__name__).info(
        "q38 PLE_DECODE_BOUNDED=%d gated off: %d tokens < VLLM_TPU_PLE_DECODE_BOUNDED_MIN_ROWS=%d "
        "(default conv traced)",
        mode,
        token_count,
        min_rows,
    )


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
    bounded: int = 0,
    bounded_min_rows: int = 0,
    fold: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """PLE history with per-position checkpoints for speculative verification.

    Mirrors the pooled GDN state contract. ``read_ids`` is each request's
    checkpoint of its last accepted token (``ckpt_ids[r, read_offset]``).
    Windowed requests (decodes and verify windows, the runner's leading
    segment) write the history after chunk token ``t`` to ``ckpt_ids[r, t]``,
    so the next step resumes from whichever token was accepted last. Other
    requests (prefill/mixed) write their final history to ``ckpt_ids[r, 0]``,
    which is where a read offset of zero points.

    ``bounded`` (VLLM_TPU_PLE_DECODE_BOUNDED, q38 L1 D07-b2) is only for
    buckets whose live requests each hold exactly ``window`` tokens and come
    first (the Qwen3.8 megakernel steps: every request a 4-token verify
    window, n_req * 4 <= rows). 1: read history and compute only for the
    bounded_request_rows leading rows instead of all max-num-seqs rows.
    2: also take each token's conv taps and checkpoint window by static
    slices (uniform_window_conv) instead of gathers, when the bucket is
    exactly rows * window tokens. Live tokens' outputs and every written
    checkpoint are bitwise those of ``bounded=0``; padded tokens' outputs
    stay zero and they still write nothing.

    ``bounded_min_rows`` (VLLM_TPU_PLE_DECODE_BOUNDED_MIN_ROWS, q38 L1-GLUEFIX):
    a bucket of fewer tokens traces the ``bounded=0`` computation (the same
    equations as the default op); 0 = every bucket bounded.

    ``fold`` (VLLM_TPU_PLE_D14_FOLD, q38 L1-PLE; with ``bounded``): a bucket of
    exactly rows * window tokens runs the conv and its checkpoint writes as one
    Pallas program (ple_d14_fold); other buckets keep the bounded path.
    """
    channels = x.shape[-1]
    count, window = ckpt_ids.shape
    token_path = x.shape[0] <= count * window
    if bounded and token_path and x.shape[0] < bounded_min_rows:
        _note_bounded_gated(bounded, x.shape[0], bounded_min_rows)
        bounded = 0
    if bounded and token_path:
        rows = bounded_request_rows(x.shape[0], count, window)
        _note_bounded(bounded, x.shape[0], count, rows, window)
        starts, ids, initial = starts[: rows + 1], ids[:rows], initial[:rows]
        read_ids, ckpt_ids, windowed = read_ids[:rows], ckpt_ids[:rows], windowed[:rows]
        count = rows
        if fold:  # q38 L1-PLE (PLE-INTO-D14)
            from vllm_torchtpu.kernels.qwen4_exp import ple_d14_fold

            if ple_d14_fold.fold_applies(
                pool, x, weight, ckpt_ids, dilation=dilation, state_len=state_len
            ):
                return ple_d14_fold.uniform_window_conv_fold(
                    x, weight, pool, starts, ids, initial, read_ids, ckpt_ids, windowed,
                    dilation=dilation,
                )
        if (
            bounded >= 2
            and x.shape[0] == rows * window
            and state_len == (weight.shape[-1] - 1) * dilation
        ):
            history = read_region(
                pool, read_ids, elements=state_len * channels, dtype=jnp.bfloat16
            ).reshape(count, state_len, channels)
            output, values, targets = uniform_window_conv(
                x, weight, history, starts, ids, initial, ckpt_ids, windowed,
                dilation=dilation,
            )
            return write_region(pool, values, targets), output
    history = read_region(
        pool, read_ids, elements=state_len * channels, dtype=jnp.bfloat16
    ).reshape(count, state_len, channels)
    lengths = starts[1:] - starts[:-1]
    active = (ids > 0) & (lengths > 0)
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
    if token_path:
        values, targets = token_checkpoints(
            x, history, starts, initial, active, ckpt_ids, windowed
        )
    else:
        values, targets = request_checkpoints(
            x, history, final, starts, initial, active, ckpt_ids, windowed
        )
    pool = write_region(pool, values, targets)
    return pool, output


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
    past = history[req[:, None], jnp.clip(positions + state_len, 0, state_len - 1)]
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


def uniform_window_conv(x, weight, history, starts, ids, initial, ckpt_ids, windowed, *, dilation):
    """``short_conv`` (request-owned history) plus ``token_checkpoints`` for a
    bucket of exactly ``count`` requests x ``window`` tokens whose live
    requests each hold ``window`` tokens and come first (q38 L1 D07-b2,
    VLLM_TPU_PLE_DECODE_BOUNDED=2). Returns (output, values, targets).

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
    pre = _bf16(accum)
    output = _bf16(jax.nn.silu(pre.astype(jnp.float32)))
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


# q38 L1 PLE-WRDEFER (VLLM_TPU_PLE_WRDEFER = K, default 0 = off, hashed): the checkpoint
# writer with K DMAs in flight. It sits below everything else and rebinds ``write_region``
# here, so with the flag unset every line above, and with them the stock writer's Mosaic
# payload and debug locations, is unchanged; callers resolve ``write_region`` at call time.
_write_region_stock = write_region
_WRDEFER_SEEN: set = set()


def wrdefer_depth() -> int:
    """VLLM_TPU_PLE_WRDEFER: checkpoint DMAs the PLE writer keeps in flight (0 = stock writer)."""
    from vllm_torchtpu import envs

    return int(envs.VLLM_TPU_PLE_WRDEFER or "0")


def write_region(pool, values, ids, *, tok0=0):
    """Write unique positive manager IDs; idle/null IDs leave all bytes intact.

    The stock writer (one grid step and one waited DMA per row) unless
    VLLM_TPU_PLE_WRDEFER = K > 0 (q38 L1 PLE-WRDEFER): ``write_region_ring``.
    """
    depth = wrdefer_depth()
    if depth:
        return write_region_ring(pool, values, ids, tok0=tok0, depth=depth)
    return _write_region_stock(pool, values, ids, tok0=tok0)


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
    """``write_region`` with ``depth`` checkpoint DMAs in flight (q38 L1 PLE-WRDEFER).

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
