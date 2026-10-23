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
    """
    channels = x.shape[-1]
    count, window = ckpt_ids.shape
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
    if x.shape[0] <= count * window:
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
