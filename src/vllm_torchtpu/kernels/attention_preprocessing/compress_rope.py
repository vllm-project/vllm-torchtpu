# SPDX-License-Identifier: Apache-2.0
"""V4.1 compressor for integration: projection, pooling, RMSNorm, RoPE, CSA2.

Same arithmetic as ``compress_serial._compress_kernel``. Differences:

* RoPE comes from a per-step ``[tokens, rotary_dim]`` block of packed
  ``cos_sin_cache`` rows at each token's group start
  (``rope_gather.gather_rows``), read as an ordinary tiled input, as in
  ``qa_rope`` / ``qb_rope``;
* block sizes come from a ``CompressConfig`` (tuning table or
  ``default_config``), so any token count that is a multiple of 16 compiles;
  other counts are zero-padded with ``valid=0``;
* the BF16 weight is copied into VMEM once, in column chunks with one
  semaphore each, on its own DMA priority. Only the first token tile waits,
  chunk by chunk, so its matmul starts after the first chunk instead of the
  whole 5-10 MiB weight;
* ratio 2 reads and writes the compressor state ring itself (epic's
  ``save_compress_norm``): each FP32 ``[2*512]`` ring record is one
  ``(8, 128)`` tile of the shared INT32 ``[rows, 128]`` pool, 4 KiB
  contiguous. A record is staged as one ``(8, 128)`` VMEM tile and moved by
  one DMA. No ``kv_score`` reaches HBM and no caller-side ring gather is
  needed.

Ring records are compacted outside the kernel (``_compact``): a scalar loop
step costs tens of ns even when it issues nothing, and a prefill tile writes
only its request's last ring-capacity rows.
"""

import functools
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.attention_preprocessing import rope_gather
from vllm_torchtpu.kernels.attention_preprocessing.compress_serial import (
    _encode_csa2,
)
from vllm_torchtpu.kernels.attention_preprocessing.serial import _LANE, _lane_gather
from vllm_torchtpu.kernels.attention_preprocessing.tuned_block_sizes import (
    COMPRESS_ROPE_TUNING,
    CompressConfig,
)

_DIM = 512
_ROW_ALIGN = 16  # BF16 activation tiles are (16, 128)
_POOL_LANES = 128  # INT32 words per pool row (512 bytes)
_RECORD_ROWS = 2 * _DIM // _POOL_LANES  # pool rows per FP32 ratio-2 record


def _compress_rope_kernel(*refs, eps, ratio, tb, chunk, banks):
    refs = list(refs)
    # Ratio-2-only refs; ratio 1 never reads them.
    unused: Any = None
    write_table = write_offsets = read_table = read_offsets = unused
    shifted_ref = previous_ref = write_stage = read_stage = ring_sems = unused
    if ratio == 2:
        write_table, write_offsets, read_table, read_offsets = refs[:4]
        del refs[:4]
    x_ref, weight_hbm, gamma_ref, has_prev_ref, valid_ref, rope_rows_ref = refs[:6]
    del refs[:6]
    if ratio == 2:
        del refs[:banks]  # aliased state inputs, written through the outputs
    latent_out_ref, words_out_ref, scales_out_ref = refs[:3]
    del refs[:3]
    pools = [pool.bitcast(jnp.float32) for pool in refs[:banks]] if ratio == 2 else []
    del refs[: len(pools)]
    weight_ref, weight_sems = refs[:2]
    del refs[:2]
    if ratio == 2:
        shifted_ref, previous_ref, write_stage, read_stage, ring_sems = refs
        # Rows 8.. of the shifted buffer are this tile's FP32 projection.
        product_ref = shifted_ref.at[pl.ds(8, tb)]
    else:
        (product_ref,) = refs
    tile = pl.program_id(0)
    first_row = tile * tb
    chunks = weight_ref.shape[1] // chunk

    def weight_copy(c):
        columns = pl.ds(c * chunk, chunk)
        return pltpu.make_async_copy(
            weight_hbm.at[:, columns], weight_ref.at[:, columns], weight_sems.at[c]
        )

    def record_copy(stage, slot, pool, base, read, sem):
        hbm = pool.at[pl.ds(base, _RECORD_ROWS), :]
        if read:
            return pltpu.make_async_copy(hbm, stage.at[slot], sem)
        return pltpu.make_async_copy(stage.at[slot], hbm, sem)

    def move_piece(stage, slot, row, c, read):
        """Piece ``c`` of a token row <-> sublane ``c`` of its staged record.

        FP32 VMEM tiles are (8, 128): a dynamic token row is reached through
        its 8-aligned window and a sublane mask (E2003 otherwise).
        """
        lanes = pl.ds(c * _POOL_LANES, _POOL_LANES)
        window = pl.ds(pl.multiple_of(row // 8 * 8, 8), 8)
        mine = jax.lax.broadcasted_iota(jnp.int32, (8, _POOL_LANES), 0) == row % 8
        if read:
            piece = stage[slot, pl.ds(c, 1), :]
            previous_ref[window, lanes] = jnp.where(
                mine, piece, previous_ref[window, lanes]
            )
        else:
            # Select the row on its bits: exact, and keeps -0.0 for the ring.
            bits = pltpu.bitcast(product_ref[window, lanes], jnp.int32)
            picked = jnp.sum(jnp.where(mine, bits, 0), axis=0, keepdims=True)
            stage[slot, pl.ds(c, 1), :] = pltpu.bitcast(picked, jnp.float32)

    def ring_start(table, offsets, stage, sem, read):
        """Start this tile's record DMAs; writes first stage their token rows."""
        start = offsets[tile]

        def body(i, carry):
            row, record, slot = table[2 * i] - first_row, table[2 * i + 1], i - start
            if not read:
                for c in range(_RECORD_ROWS):
                    move_piece(stage, slot, row, c, read)
            if banks == 1:
                record_copy(
                    stage, slot, pools[0], record * _RECORD_ROWS, read, sem
                ).start()
            else:
                lo_records = pools[0].shape[0] // _RECORD_ROWS
                in_hi = record >= lo_records

                @pl.when(~in_hi)
                def _():
                    base = record * _RECORD_ROWS
                    record_copy(stage, slot, pools[0], base, read, sem).start()

                @pl.when(in_hi)
                def _():
                    base = (record - lo_records) * _RECORD_ROWS
                    record_copy(stage, slot, pools[1], base, read, sem).start()

            return carry

        jax.lax.fori_loop(start, offsets[tile + 1], body, 0)

    def ring_wait(offsets, stage, sem, read):
        """Wait for every record DMA of this tile (equal 4 KiB transfers)."""

        def body(i, carry):
            record_copy(stage, 0, pools[0], 0, read, sem).wait()
            return carry

        jax.lax.fori_loop(offsets[tile], offsets[tile + 1], body, 0)

    def unstage_reads():
        start = read_offsets[tile]

        def body(i, carry):
            row, slot = read_table[2 * i] - first_row, i - start
            for c in range(_RECORD_ROWS):
                move_piece(read_stage, slot, row, c, True)
            return carry

        jax.lax.fori_loop(start, read_offsets[tile + 1], body, 0)

    if ratio == 2:
        ring_start(read_table, read_offsets, read_stage, ring_sems.at[0], True)

    # The weight streams on its own priority, so the row tiles and ring DMAs
    # (priority 0) never queue behind it.
    @pl.when(tile == 0)
    def _():
        for c in range(chunks):
            weight_copy(c).start(priority=1)

    if ratio == 2:
        # Row 7 holds the previous tile's last row; rows 8.. hold this tile, so
        # rows 7..tb+6 are each token's in-chunk predecessor. Copy it before
        # this tile's matmul overwrites row tb+7.
        @pl.when(tile == 0)
        def _():
            shifted_ref[pl.ds(7, 1), :] = jnp.zeros((1, 2 * _DIM), jnp.float32)

        @pl.when(tile != 0)
        def _():
            shifted_ref[pl.ds(7, 1), :] = shifted_ref[pl.ds(tb + 7, 1), :]

    # Only the first tile waits for the weight, chunk by chunk. Later tiles run
    # one full-width dot with no control flow between chunks (which would drain
    # the MXU per chunk). 256-column chunks and the full width accumulate
    # identically, so every tile matches the single-dot pipelined kernel.
    @pl.when(tile == 0)
    def _():
        for c in range(chunks):
            weight_copy(c).wait()
            columns = pl.ds(c * chunk, chunk)
            product_ref[:, columns] = jnp.dot(
                x_ref[...], weight_ref[:, columns], preferred_element_type=jnp.float32
            )

    @pl.when(tile != 0)
    def _():
        product_ref[...] = jnp.dot(
            x_ref[...], weight_ref[...], preferred_element_type=jnp.float32
        )

    if ratio == 2:
        # A request's ring read lands before any of its writes are issued: the
        # read is in its first tile, its writes in that tile or later ones.
        ring_wait(read_offsets, read_stage, ring_sems.at[0], True)
        unstage_reads()
        ring_start(write_table, write_offsets, write_stage, ring_sems.at[1], False)
    product = product_ref[...]
    if ratio == 2:
        in_chunk = shifted_ref[pl.ds(7, tb), :]
        # Rows without a predecessor in the chunk or a ring read are never
        # valid (the caller's contract), so their unread rows are discarded.
        previous = jnp.where(has_prev_ref[...] != 0, in_chunk, previous_ref[...])
        score_prev, score_cur = previous[:, _DIM:], product[:, _DIM:]
        peak = jnp.maximum(score_prev, score_cur)
        e_prev = jnp.exp(score_prev - peak)
        e_cur = jnp.exp(score_cur - peak)
        total = e_prev + e_cur
        values = previous[:, :_DIM] * (e_prev / total) + product[:, :_DIM] * (
            e_cur / total
        )
    else:
        values = product
    inv_rms = jax.lax.rsqrt(jnp.mean(values * values, axis=-1, keepdims=True) + eps)
    latent = (values * inv_rms * gamma_ref[...].astype(jnp.float32)[None, :]).astype(
        jnp.bfloat16
    )
    valid = valid_ref[...] != 0
    latent = jnp.where(valid, latent, jnp.zeros_like(latent))
    latent_out_ref[...] = latent
    cos, sin = rope_gather.coefficients(rope_rows_ref[...])
    tail = latent[:, -_LANE:].astype(jnp.float32)
    partner = _lane_gather(tail, jnp.arange(_LANE) ^ 1)
    rotated_tail = (tail * cos + partner * sin).astype(jnp.bfloat16)
    # Invalid rows have zero latents, but 0*sin can be -0.0 depending on the
    # row's angle; force +0 so their (never published) words are 0.
    rotated_tail = jnp.where(valid, rotated_tail, 0)
    rotated = jnp.concatenate((latent[:, :-_LANE], rotated_tail), axis=-1).astype(
        jnp.float32
    )
    words, scales = _encode_csa2(rotated)
    words_out_ref[...] = words
    scales_out_ref[...] = scales
    if ratio == 2:
        # The next tile overwrites the write stage.
        ring_wait(write_offsets, write_stage, ring_sems.at[1], False)


def _largest_divisor(n, cap, multiple):
    for d in range(min(n, cap) // multiple * multiple, 0, -multiple):
        if n % d == 0:
            return d
    return None


def default_config(tokens: int) -> CompressConfig:
    """Shape-derived tile; ``tokens`` must already be a multiple of 16."""
    tb = tokens if tokens <= 256 else _largest_divisor(tokens, 256, _ROW_ALIGN)
    assert tb is not None  # 16 always divides
    return CompressConfig(token_block=tb)


def select_config(tokens: int, hidden: int, ratio: int) -> CompressConfig:
    from vllm_torchtpu.kernels.quantized_matmul.tuned_block_sizes import (
        get_tpu_version,
    )

    tuned = COMPRESS_ROPE_TUNING.get((get_tpu_version(), tokens, hidden, ratio))
    return tuned or default_config(tokens)


def check_config(config: CompressConfig, tokens: int, width: int) -> None:
    tb, chunk = config.token_block, config.weight_chunk
    problems = []
    if tb <= 0 or tb % _ROW_ALIGN or tokens % tb:
        problems.append(f"token_block {tb} must be a multiple of 16 dividing {tokens}")
    if chunk <= 0 or chunk % _LANE or width % chunk:
        problems.append(f"weight_chunk {chunk} must be 128-aligned and divide {width}")
    if problems:
        raise ValueError("Invalid CompressConfig: " + "; ".join(problems))


def vmem_bytes(config: CompressConfig, hidden: int, ratio: int) -> int:
    """Upper estimate of the call's VMEM (lane padding included)."""
    tb, width = config.token_block, ratio * _DIM
    weight = hidden * width * 2  # BF16, single copy
    x_tiles = 2 * tb * hidden * 2
    flags = 2 * 2 * tb * _LANE * 4  # has_prev, valid tiles padded to a lane
    outputs = 2 * tb * (_DIM * 2 + _LANE * 4 + _LANE * 4)
    rope_rows = 2 * tb * _LANE * 4  # double-buffered input tile, lane-padded
    if ratio == 1:
        scratch = tb * width * 4  # product tile
    else:  # shifted product, ring-read rows, write and read record stages
        scratch = (tb + 8) * width * 4 + 3 * tb * width * 4
    return weight + x_tiles + flags + outputs + rope_rows + scratch


def _compact(records, total_records, tiles, tb):
    """Valid ``records`` as flat ``(token row, record)`` pairs in token order,
    plus each tile's ``[offsets[t], offsets[t+1])`` pair range.

    One scatter and no gather: on TPU each lowers to a SparseCore offload of
    several us. Out-of-range records drop.
    """
    n = records.shape[0]
    valid = (records >= 0) & (records < total_records)
    slots = jnp.where(valid, jnp.cumsum(valid, dtype=jnp.int32) - 1, n)
    pairs = jnp.stack((jnp.arange(n, dtype=jnp.int32), records.astype(jnp.int32)), 1)
    table = jnp.zeros((n, 2), jnp.int32).at[slots].set(pairs, mode="drop")
    per_tile = valid.reshape(tiles, tb).sum(axis=1, dtype=jnp.int32)
    offsets = jnp.concatenate((jnp.zeros((1,), jnp.int32), jnp.cumsum(per_tile)))
    return table.reshape(-1), offsets.astype(jnp.int32)


def fused_compress_rope(
    x: jax.Array,
    weight: jax.Array,
    gamma: jax.Array,
    rope_rows: jax.Array,
    has_prev: jax.Array,
    valid: jax.Array,
    state: jax.Array | tuple[jax.Array, jax.Array] | None = None,
    ring_write: jax.Array | None = None,
    ring_read: jax.Array | None = None,
    *,
    eps: float = 1e-6,
    ratio: int = 2,
    config: CompressConfig | None = None,
):
    """Return ``(latent BF16, CSA2 words INT32, scales FP32 [T,1], state)``.

    ``rope_rows`` is FP32 ``[T, rotary_dim]``: ``rope_gather.gather_rows`` of
    the model's ``cos_sin_cache`` at each token's group start
    (``pos // ratio * ratio``), the RoPE position of its compressed record.
    Ratio-1 layers share the token-position rows of ``qa_rope``/``qb_rope``;
    ratio-2 layers share one group-start block per step.
    ``has_prev`` / ``valid`` are INT32 ``[T, 1]``.

    Ratio 2 only (``state`` is returned as None for ratio 1): ``state`` is
    the compressor state pool, INT32 ``[rows, 128]`` or its two banks (bank
    0's records first), updated in place. Record ``r`` is the FP32 ``[1024]``
    row in pool rows ``[8r, 8r+8)``. ``ring_write`` / ``ring_read`` are INT32
    ``[T]`` records, -1 for none: the row's projection is stored at
    ``ring_write`` (unique records), and rows with ``has_prev == 0`` pool
    with the stored row at ``ring_read`` (read before any write). Valid rows
    with ``has_prev == 0`` must have a ``ring_read`` record.
    """
    tokens, hidden = x.shape
    width = ratio * _DIM
    if ratio not in (1, 2):
        raise ValueError("V4.1 compression ratio must be 1 or 2")
    if x.dtype != jnp.bfloat16 or weight.dtype != jnp.bfloat16:
        raise ValueError("Compressor projection uses BF16 inputs and weights")
    if weight.shape != (hidden, width) or gamma.shape != (_DIM,):
        raise ValueError("Weight must be [hidden, ratio*512]; gamma [512]")
    if has_prev.shape != (tokens, 1) or valid.shape != (tokens, 1):
        raise ValueError("has_prev and valid must be INT32 [tokens, 1]")
    rope_gather.check_rope_rows(rope_rows, tokens, _DIM)
    banks = _check_ring(state, ring_write, ring_read, tokens, ratio)
    pools = () if state is None else (state if isinstance(state, tuple) else (state,))
    padded = -(-tokens // _ROW_ALIGN) * _ROW_ALIGN
    if padded != tokens:
        extra = padded - tokens
        x = jnp.pad(x, ((0, extra), (0, 0)))
        rope_rows = jnp.pad(rope_rows, ((0, extra), (0, 0)))
        has_prev = jnp.pad(has_prev, ((0, extra), (0, 0)), constant_values=1)
        valid = jnp.pad(valid, ((0, extra), (0, 0)))
        if ratio == 2:
            ring_write = jnp.pad(ring_write, (0, extra), constant_values=-1)
            ring_read = jnp.pad(ring_read, (0, extra), constant_values=-1)
    if config is None:
        config = select_config(padded, hidden, ratio)
    check_config(config, padded, width)
    tb = config.token_block
    tiles = padded // tb
    rotary_dim = rope_rows.shape[1]

    once = pl.Buffered(buffer_count=1)
    row = lambda t, *_: (t, 0)  # noqa: E731
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    prefetch = []
    scratch = [
        pltpu.VMEM((hidden, width), jnp.bfloat16),
        pltpu.SemaphoreType.DMA((width // config.weight_chunk,)),
    ]
    if ratio == 2:
        records = sum(pool.shape[0] for pool in pools) // _RECORD_ROWS
        prefetch = [
            *_compact(ring_write.astype(jnp.int32), records, tiles, tb),
            *_compact(ring_read.astype(jnp.int32), records, tiles, tb),
        ]
        stage = pltpu.VMEM((tb, _RECORD_ROWS, _POOL_LANES), jnp.float32)
        scratch += [
            pltpu.VMEM((tb + 8, width), jnp.float32),
            pltpu.VMEM((tb, width), jnp.float32),
            stage,
            stage,
            pltpu.SemaphoreType.DMA((2,)),
        ]
    else:
        scratch.append(pltpu.VMEM((tb, width), jnp.float32))
    operands = [*prefetch, x, weight, gamma, has_prev, valid, rope_rows, *pools]
    first_pool = len(operands) - len(pools)
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=len(prefetch),
        grid=(tiles,),
        in_specs=(
            pl.BlockSpec((tb, hidden), row),
            hbm,
            pl.BlockSpec(gamma.shape, lambda t, *_: (0,), once),
            pl.BlockSpec((tb, 1), row),
            pl.BlockSpec((tb, 1), row),
            pl.BlockSpec((tb, rotary_dim), row),
            *(hbm for _ in pools),
        ),
        out_specs=(
            pl.BlockSpec((tb, _DIM), row),
            pl.BlockSpec((tb, _LANE), row),
            pl.BlockSpec((tb, 1), row),
            *(hbm for _ in pools),
        ),
        scratch_shapes=tuple(scratch),
    )
    outputs = pl.pallas_call(
        functools.partial(
            _compress_rope_kernel,
            eps=eps,
            ratio=ratio,
            tb=tb,
            chunk=config.weight_chunk,
            banks=banks,
        ),
        grid_spec=grid_spec,
        out_shape=(
            jax.ShapeDtypeStruct((padded, _DIM), jnp.bfloat16),
            jax.ShapeDtypeStruct((padded, _LANE), jnp.int32),
            jax.ShapeDtypeStruct((padded, 1), jnp.float32),
            *(jax.ShapeDtypeStruct(pool.shape, pool.dtype) for pool in pools),
        ),
        input_output_aliases={first_pool + b: 3 + b for b in range(len(pools))},
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=config.vmem_limit_bytes,
        ),
        name=f"compress_rope_ratio{ratio}_t{tb}_c{config.weight_chunk}",
    )(*operands)
    latent, words, scales = (o[:tokens] for o in outputs[:3])
    new_state = outputs[3:]
    if not new_state:
        return latent, words, scales, None
    return latent, words, scales, new_state if banks == 2 else new_state[0]


def _check_ring(state, ring_write, ring_read, tokens, ratio):
    """Validate the ratio-2 state arguments; return the pool bank count."""
    if ratio == 1:
        if state is not None or ring_write is not None or ring_read is not None:
            raise ValueError("Ratio 1 keeps no compressor state")
        return 0
    if state is None or ring_write is None or ring_read is None:
        raise ValueError("Ratio 2 needs the state pool and ring records")
    pools = state if isinstance(state, tuple) else (state,)
    if len(pools) not in (1, 2) or any(
        pool.dtype != jnp.int32
        or pool.ndim != 2
        or pool.shape[1] != _POOL_LANES
        or pool.shape[0] % _RECORD_ROWS
        for pool in pools
    ):
        raise ValueError("State must be one or two INT32 [8k, 128] pool banks")
    for records in (ring_write, ring_read):
        if records.shape != (tokens,) or not jnp.issubdtype(records.dtype, jnp.integer):
            raise ValueError("Ring records must be integer [tokens]")
    return len(pools)


def supported(x, weight, gamma, rope_rows, ratio) -> bool:
    """Whether ``fused_compress_rope`` accepts these concrete lowering inputs."""
    if ratio not in (1, 2) or x.ndim != 2 or x.shape[0] == 0:
        return False
    if x.dtype != jnp.bfloat16 or weight.dtype != jnp.bfloat16:
        return False
    if weight.shape != (x.shape[1], ratio * _DIM) or gamma.shape != (_DIM,):
        return False
    try:
        rope_gather.check_rope_rows(rope_rows, x.shape[0], _DIM)
    except ValueError:
        return False
    return True
