# SPDX-License-Identifier: Apache-2.0
"""Torch ops for the fused DeepSeek-V4.1 attention-preprocessing kernels.

``pallas.jax_op`` requires every parameter and the return of a bridged
function to be an annotated array, so each kernel mode gets its own explicit
signature (no ``*args``, keyword parameters or ``Any`` return); epsilon and
the compression ratio are bound per op.
"""

import dataclasses
import functools
import inspect
from typing import Any, cast

import jax
import jax.numpy as jnp
import torch
from jax.sharding import PartitionSpec as P
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels import swa_metadata as swa_metadata_kernel
from vllm_torchtpu.kernels.attention_preprocessing import (
    compress_rope as compress_kernel,
)
from vllm_torchtpu.kernels.attention_preprocessing import qa_rope as qa_kernel
from vllm_torchtpu.kernels.attention_preprocessing import qb_rope as qb_kernel
from vllm_torchtpu.kernels.attention_preprocessing import rope_gather
from vllm_torchtpu.kernels.experimental import decode_preparation
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import get_pallas_op_mesh

_HEAD_DIM = 512
_ops: dict[str, Any] = {}


@torch.compiler.assume_constant_result
def compressor_layout_admitted(region: str, **geometry: int) -> bool:
    """Check static cache geometry once, outside the captured tensor graph.

    Only cache layout scalars enter this boundary, never token dimensions or
    tensors. Fusion policy is frozen by admit; fallback diagnostics must execute
    here at trace time because Dynamo cannot capture logger.warning.
    """
    layout = dataclasses.replace(
        decode_preparation.DEFAULT_COMPRESSOR_LAYOUT, **geometry
    )
    return decode_preparation.admit(
        region, decode_preparation.compressor_layout_problem(layout), backend=False
    )


def _get(name, build, donate=()):
    """``pallas::dsv41_<name>``, registered from ``build() -> (fn, fake)`` once."""
    op = _ops.get(name)
    if op is None:
        function, fake = build()
        op = pallas.jax_op(
            f"pallas::dsv41_{name}",
            function,
            mesh=get_pallas_op_mesh(),
            donate_argnums=donate,
            input_partition_specs=cast(
                Any, (P(),) * len(inspect.signature(function).parameters)
            ),
        )
        op.register_fake(fake)
        _ops[name] = op
    return op


def _tag(eps: float) -> str:
    return float(eps).hex().replace(".", "_").replace("+", "p").replace("-", "m")


def rope_rows() -> Any:
    """``(cos_sin_cache, positions) -> [T, rotary_dim]`` FP32 rows, clamped."""
    return _get(
        "rope_rows",
        lambda: (
            rope_gather.gather_rows,
            lambda cache, pos: cache.new_empty((pos.shape[0], cache.shape[1])),
        ),
    )


def _swa_records_bridge(
    window_size: int, page_size: int, page_stride: int, row_offset: int
):
    def records(
        query_start_loc: jax.Array,
        positions: jax.Array,
        window_starts: jax.Array,
        block_table: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        # Keep the slice inside the op so split graphs consume inputs in the
        # same order as the fallback, including oversized replay-start buffers.
        result = swa_metadata_kernel.swa_attention_records(
            query_start_loc,
            positions,
            window_starts[: query_start_loc.shape[0] - 1],
            block_table,
            window_size=window_size,
            page_size=page_size,
            page_stride=page_stride,
            row_offset=row_offset,
        )
        return tuple(result)

    return records


def _swa_records_fake(window_size: int):
    def fake(query_start_loc, positions, window_starts, block_table):
        shape = positions.shape
        return (
            positions.new_empty(shape, dtype=torch.int32),
            positions.new_empty(shape, dtype=torch.bool),
            positions.new_empty((shape[0], window_size), dtype=torch.int32),
        )

    return fake


@torch.compiler.assume_constant_result
def _register_swa_records(
    window_size: int, page_size: int, page_stride: int, row_offset: int
) -> str:
    # A layout may first appear inside compiled forward. Register in Python
    # setup so Dynamo never traces the Pallas registration machinery.
    name = f"swa_records_w{window_size}_p{page_size}_s{page_stride}_o{row_offset}"
    _get(
        name,
        lambda: (
            _swa_records_bridge(window_size, page_size, page_stride, row_offset),
            _swa_records_fake(window_size),
        ),
    )
    return name


def swa_records(
    window_size: int, page_size: int, page_stride: int, row_offset: int
) -> Any:
    """``(starts, positions, window_starts, table) -> slots, valid, records``."""
    name = _register_swa_records(window_size, page_size, page_stride, row_offset)
    # Dynamo's traced _ops dictionary cannot see registration's Python writes.
    return getattr(torch.ops.pallas, f"dsv41_{name}").default


def _qa(x, weight, scale, gamma_qr, gamma_kv, rows, *, eps, emit_qr):
    if not x.shape[0]:  # empty dummy batches
        return (
            jnp.zeros((0, gamma_qr.shape[0]), jnp.bfloat16),
            jnp.zeros((0, gamma_kv.shape[0]), jnp.bfloat16),
        )
    return qa_kernel.fused_qa_kv_norm_rope(
        x, weight, scale, gamma_qr, gamma_kv, rows, eps=eps, emit_qr=emit_qr
    )


def _qa_bridge(emit_qr: bool, eps: float):
    run = functools.partial(_qa, eps=eps, emit_qr=emit_qr)
    if emit_qr:

        def qa_kv_norm_rope(
            x: jax.Array,
            weight: jax.Array,
            weight_scale: jax.Array,
            gamma_qr: jax.Array,
            gamma_kv: jax.Array,
            rope_rows: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            return run(x, weight, weight_scale, gamma_qr, gamma_kv, rope_rows)

        return qa_kv_norm_rope

    def kv_norm_rope(
        x: jax.Array,
        weight: jax.Array,
        weight_scale: jax.Array,
        gamma_qr: jax.Array,
        gamma_kv: jax.Array,
        rope_rows: jax.Array,
    ) -> jax.Array:
        return run(x, weight, weight_scale, gamma_qr, gamma_kv, rope_rows)[1]

    return kv_norm_rope


def _qa_fake(emit_qr):
    def fake(x, weight, weight_scale, gamma_qr, gamma_kv, rope_rows):
        kv = x.new_empty((x.shape[0], gamma_kv.shape[0]))
        return (x.new_empty((x.shape[0], gamma_qr.shape[0])), kv) if emit_qr else kv

    return fake


def qa_rope(emit_qr: bool, eps: float) -> Any:
    """``(x, w, scale, gamma_qr, gamma_kv, rope_rows) -> (qr, kv)`` or ``kv``.

    KV leaves rotated; ``emit_qr=False`` projects only the KV columns.
    """
    return _get(
        f"qa_kv_norm_rope{'' if emit_qr else '_kv'}_e{_tag(eps)}",
        lambda: (_qa_bridge(emit_qr, eps), _qa_fake(emit_qr)),
    )


def _qb(
    qr: jax.Array, weight: jax.Array, weight_scale: jax.Array, rope_rows: jax.Array
) -> jax.Array:
    if not qr.shape[0]:
        return jnp.zeros((0, weight.shape[1] // _HEAD_DIM, _HEAD_DIM), jnp.bfloat16)
    return qb_kernel.fused_qb_rope(qr, weight, weight_scale, rope_rows)


def _qb_fake(qr, weight, weight_scale, rope_rows):
    return qr.new_empty((qr.shape[0], weight.shape[1] // _HEAD_DIM, _HEAD_DIM))


def qb_rope() -> Any:
    """``(qr, w, scale, rope_rows) -> q [T, heads, 512]`` with RoPE applied."""
    return _get("qb_rope", lambda: (_qb, _qb_fake))


def _compress(ratio, eps, x, weight, gamma, rows, has_prev, valid, *state_and_ring):
    if not x.shape[0]:
        state = state_and_ring[0] if state_and_ring else None
        return (
            jnp.zeros((0, 512), jnp.bfloat16),
            jnp.zeros((0, 128), jnp.int32),
            jnp.zeros((0, 1), jnp.float32),
            state,
        )
    return compress_kernel.fused_compress_rope(
        x, weight, gamma, rows, has_prev, valid, *state_and_ring, ratio=ratio, eps=eps
    )


def _compress_bridge(ratio: int, banks: int, eps: float):
    run = functools.partial(_compress, ratio, eps)
    if ratio == 1:

        def compress_rope_ratio1(
            x: jax.Array,
            weight: jax.Array,
            gamma: jax.Array,
            rope_rows: jax.Array,
            has_prev: jax.Array,
            valid: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            latent, words, scales, _ = run(x, weight, gamma, rope_rows, has_prev, valid)
            return latent, words, scales

        return compress_rope_ratio1
    if banks == 1:

        def compress_rope_ratio2(
            x: jax.Array,
            weight: jax.Array,
            gamma: jax.Array,
            rope_rows: jax.Array,
            has_prev: jax.Array,
            valid: jax.Array,
            state: jax.Array,
            ring_write: jax.Array,
            ring_read: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
            return run(
                x,
                weight,
                gamma,
                rope_rows,
                has_prev,
                valid,
                state,
                ring_write,
                ring_read,
            )

        return compress_rope_ratio2

    def compress_rope_ratio2_banked(
        x: jax.Array,
        weight: jax.Array,
        gamma: jax.Array,
        rope_rows: jax.Array,
        has_prev: jax.Array,
        valid: jax.Array,
        state_low: jax.Array,
        state_high: jax.Array,
        ring_write: jax.Array,
        ring_read: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        latent, words, scales, state = run(
            x,
            weight,
            gamma,
            rope_rows,
            has_prev,
            valid,
            (state_low, state_high),
            ring_write,
            ring_read,
        )
        assert isinstance(state, tuple)
        low, high = state
        return latent, words, scales, low, high

    return compress_rope_ratio2_banked


def _compress_fake(banks):
    def fake(x, weight, gamma, rope_rows, has_prev, valid, *state_and_ring):
        latent = x.new_empty((x.shape[0], 512), dtype=torch.bfloat16)
        words = x.new_empty((x.shape[0], 128), dtype=torch.int32)
        scales = x.new_empty((x.shape[0], 1), dtype=torch.float32)
        states = tuple(torch.empty_like(s) for s in state_and_ring[:banks])
        return latent, words, scales, *states

    return fake


def compress_rope(ratio: int, banks: int, eps: float) -> Any:
    """``(x, w, gamma, rope_rows, has_prev, valid[, *state, write, read])``.

    Returns ``(latent, CSA2 words, scales, *updated state banks)``; ratio 2
    updates its ``banks`` compressor state pools in place (donated).
    """
    if (ratio, banks) not in ((1, 0), (2, 1), (2, 2)):
        raise ValueError("Use ratio 1 without state, or ratio 2 with one/two banks")
    return _get(
        f"compress_rope_r{ratio}_b{banks}_e{_tag(eps)}",
        lambda: (_compress_bridge(ratio, banks, eps), _compress_fake(banks)),
        donate=tuple(range(6, 6 + banks)),
    )


def _state_records(low: jax.Array, high: jax.Array) -> int:
    return (low.shape[0] + high.shape[0]) // compress_kernel._RECORD_ROWS


def _admit_compressor(region, buckets, x, weight, inputs, *page_tables) -> bool:
    """One gate for both bridges; token dispatch stays inside JAX."""
    tokens = inputs[3].shape[0]
    problem = decode_preparation.compressor_inputs_problem(*inputs, *page_tables)
    if problem is None and x.shape[0] != tokens:
        problem = f"x has {x.shape[0]} rows for {tokens} positions"
    if problem is None and weight.shape != (x.shape[1], 2 * _HEAD_DIM):
        problem = f"weight {list(weight.shape)} is not [hidden, {2 * _HEAD_DIM}]"
    if problem is None and tokens in buckets:
        # The precomputed ring tables cover one whole, unpadded token tile.
        tile = compress_kernel.select_config(tokens, x.shape[1], 2).token_block
        if tile != tokens or tokens % compress_kernel._ROW_ALIGN:
            problem = f"compressor tile {tile} does not cover {tokens} tokens"
    return decode_preparation.admit(region, problem, tokens=tokens, buckets=buckets)


def compress_rope_packed(
    eps: float,
    layout: decode_preparation.CompressorLayout = (
        decode_preparation.DEFAULT_COMPRESSOR_LAYOUT
    ),
) -> Any:
    """Registered decode metadata + ratio-2 compressor, with two donated banks."""
    from vllm_torchtpu.kernels.quantized_matmul.tuned_block_sizes import (
        get_tpu_version,
    )

    # Buckets were measured on v7 only.
    if jax.default_backend() != "tpu" or get_tpu_version() != 7:
        return None

    def build():
        def run(
            table: jax.Array,
            starts: jax.Array,
            kv_starts: jax.Array,
            positions: jax.Array,
            minimum: jax.Array,
            x: jax.Array,
            weight: jax.Array,
            gamma: jax.Array,
            rows: jax.Array,
            low: jax.Array,
            high: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
            inputs = (table, starts, kv_starts, positions, minimum)
            if not _admit_compressor(
                "compressor_state",
                decode_preparation.COMPRESSOR_STATE_TOKENS,
                x,
                weight,
                inputs,
            ):
                return _compress_native_metadata(
                    *inputs, x, weight, gamma, rows, low, high, eps, layout
                )
            prev, valid, wp, wo, rp, ro = decode_preparation.compressor_state_packed(
                *inputs, total_records=_state_records(low, high), layout=layout
            )
            latent, words, scales, state = compress_kernel.fused_compress_rope(
                x,
                weight,
                gamma,
                rows,
                prev,
                valid,
                (low, high),
                eps=eps,
                ring_tables=(wp, wo, rp, ro),
            )
            assert isinstance(state, tuple)
            return latent, words, scales, *state, valid[:, 0] != 0

        def fake(
            table,
            starts,
            kv_starts,
            positions,
            minimum,
            x,
            weight,
            gamma,
            rows,
            low,
            high,
        ):
            return (
                x.new_empty((x.shape[0], 512), dtype=torch.bfloat16),
                x.new_empty((x.shape[0], 128), dtype=torch.int32),
                x.new_empty((x.shape[0], 1), dtype=torch.float32),
                torch.empty_like(low),
                torch.empty_like(high),
                x.new_empty((x.shape[0],), dtype=torch.bool),
            )

        return run, fake

    return _get(
        f"compress_rope_packed_{layout.tag}_e{_tag(eps)}", build, donate=(9, 10)
    )


def compress_rope_addresses(
    eps: float,
    layout: decode_preparation.CompressorLayout = (
        decode_preparation.DEFAULT_COMPRESSOR_LAYOUT
    ),
) -> Any:
    """Registered decode state/main/index addresses and ratio-2 compressor.

    Main records are physical rows of the main KV cache; index slots are
    logical index-cache slots. Both paths return the same contract.
    """
    from vllm_torchtpu.kernels.quantized_matmul.tuned_block_sizes import (
        get_tpu_version,
    )

    # Buckets were measured on v7 only.
    if jax.default_backend() != "tpu" or get_tpu_version() != 7:
        return None

    def build():
        def run(
            table: jax.Array,
            starts: jax.Array,
            kv_starts: jax.Array,
            positions: jax.Array,
            minimum: jax.Array,
            x: jax.Array,
            weight: jax.Array,
            gamma: jax.Array,
            rows: jax.Array,
            low: jax.Array,
            high: jax.Array,
            main: jax.Array,
            index: jax.Array,
        ) -> tuple[
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
        ]:
            inputs = (table, starts, kv_starts, positions, minimum)
            if not _admit_compressor(
                "compressor_addresses",
                decode_preparation.COMPRESSOR_ADDRESS_TOKENS,
                x,
                weight,
                inputs,
                main,
                index,
            ):
                result = _compress_native_metadata(
                    *inputs, x, weight, gamma, rows, low, high, eps, layout
                )
                return (
                    *result,
                    *_native_main_index(
                        kv_starts, positions, main, index, result[-1], layout
                    ),
                )
            prev, valid, wp, wo, rp, ro, main_records, index_slots = (
                decode_preparation.compressor_addresses_packed(
                    *inputs,
                    main,
                    index,
                    total_records=_state_records(low, high),
                    layout=layout,
                )
            )
            latent, words, scales, state = compress_kernel.fused_compress_rope(
                x,
                weight,
                gamma,
                rows,
                prev,
                valid,
                (low, high),
                eps=eps,
                ring_tables=(wp, wo, rp, ro),
            )
            assert isinstance(state, tuple)
            return (
                latent,
                words,
                scales,
                *state,
                valid[:, 0] != 0,
                main_records,
                index_slots,
            )

        def fake(
            table,
            starts,
            kv_starts,
            positions,
            minimum,
            x,
            weight,
            gamma,
            rows,
            low,
            high,
            main,
            index,
        ):
            return (
                x.new_empty((x.shape[0], 512), dtype=torch.bfloat16),
                x.new_empty((x.shape[0], 128), dtype=torch.int32),
                x.new_empty((x.shape[0], 1), dtype=torch.float32),
                torch.empty_like(low),
                torch.empty_like(high),
                x.new_empty((x.shape[0],), dtype=torch.bool),
                x.new_empty((x.shape[0],), dtype=torch.int32),
                x.new_empty((x.shape[0],), dtype=torch.int32),
            )

        return run, fake

    return _get(
        f"compress_rope_addresses_{layout.tag}_e{_tag(eps)}", build, donate=(9, 10)
    )


def _native_main_index(kv_starts, positions, main, index, valid, layout):
    """XLA form of the fused address outputs, for any token count."""
    token = jnp.arange(positions.size, dtype=jnp.int32)
    request = jnp.minimum(
        jnp.sum(token[:, None] >= kv_starts[None, 1:], axis=1), main.shape[0] - 1
    )
    live = token < kv_starts[-1]
    logical = positions // 2

    def page(table, capacity):
        column = jnp.minimum(logical // capacity, table.shape[1] - 1)
        return table[request, column], logical % capacity

    block, slot = page(main, layout.main_capacity)
    main_rows = layout.main_offset + block * layout.main_stride + slot
    records = jnp.where(live & (block >= 0) & valid, main_rows, -1)
    block, slot = page(index, layout.index_capacity)
    return records, jnp.where(live, block * layout.index_capacity + slot, -1)


def _compress_native_metadata(
    table,
    starts,
    kv_starts,
    positions,
    minimum,
    x,
    weight,
    gamma,
    rows,
    low,
    high,
    eps,
    layout,
):
    """Keep arbitrary token counts behind the dynamic Torch bridge signature."""
    prev, valid, write, read = decode_preparation._compressor_state_addresses(
        table, starts, kv_starts, positions, minimum, layout
    )
    latent, words, scales, state = _compress(
        2,
        eps,
        x,
        weight,
        gamma,
        rows,
        prev[:, None],
        valid[:, None],
        (low, high),
        write,
        read,
    )
    assert isinstance(state, tuple)
    return latent, words, scales, *state, valid != 0
