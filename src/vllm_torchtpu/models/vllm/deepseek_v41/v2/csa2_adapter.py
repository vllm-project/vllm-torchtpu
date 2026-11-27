# SPDX-License-Identifier: Apache-2.0
"""TP1 CSA2 bridges, retaining stage-scoped gathered KV ownership."""

import functools
import math
from typing import TYPE_CHECKING, Any, cast

import jax
import jax.numpy as jnp
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.deepseek_v4.o_projection import wo_a_projection
from vllm_torchtpu.kernels.deepseek_v41 import output_projection
from vllm_torchtpu.kernels.deepseek_v41.csa2 import cache as cache_kernels
from vllm_torchtpu.kernels.deepseek_v41.csa2.cache import (
    gather_csa2_kv_sc,
    prepare_main_records,
    prepare_swa_metadata,
    publish_csa2_kv,
)
from vllm_torchtpu.kernels.deepseek_v41.csa2.csa2 import csa2
from vllm_torchtpu.kernels.deepseek_v41.csa2.pipeline import (
    csa2_attend_rows,
    csa2_gather_chunk_size,
    csa2_pipeline_chunks,
)
from vllm_torchtpu.kernels.mla.paged_swa import swa_word_pages, write_swa_word_pool

if TYPE_CHECKING:
    from .cache import PagedCacheRows

_ops: dict[str, Any] = {}


def _publish_main(
    raw: jax.Array,
    scales: jax.Array,
    records: jax.Array,
    words: jax.Array,
    word_scales: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    return publish_csa2_kv(raw, scales, records, words, word_scales)


def _publish_main_banked(
    raw: jax.Array,
    high: jax.Array,
    scales: jax.Array,
    high_scales: jax.Array,
    records: jax.Array,
    words: jax.Array,
    word_scales: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    banks, scale = publish_csa2_kv(
        (raw, high), (scales, high_scales), records, words, word_scales
    )
    return *banks, *scale


def _publish_swa(
    raw: jax.Array, records: jax.Array, values: jax.Array, page_size: int, shift: int
) -> jax.Array:
    return write_swa_word_pool(raw, records, values, page_size=page_size, shift=shift)


def _publish_swa_banked(
    raw: jax.Array,
    high: jax.Array,
    records: jax.Array,
    values: jax.Array,
    page_size: int,
    shift: int,
) -> tuple[jax.Array, jax.Array]:
    return write_swa_word_pool(
        (raw, high), records, values, page_size=page_size, shift=shift
    )


def _inverse_rope(rope_rows: jax.Array) -> jax.Array:
    """CSA2's inverse-rotation table from the step's ``[T, 64]`` RoPE rows.

    Q/K already carry RoPE. The rows are the ones the layer's Q-A/Q-B kernels
    read (``DeepseekV41Attention._rope_rows``), so CSA2 needs no gather of its
    own.
    """
    table = rope_rows.astype(jnp.float32)
    tokens = table.shape[0]
    cos = jnp.concatenate(
        (jnp.ones((tokens, 64)), jnp.repeat(table[:, :32], 2, axis=1)), axis=1
    )
    sin = jnp.repeat(table[:, 32:], 2, axis=1)
    sin *= jnp.where(jnp.arange(64) % 2 == 0, 1.0, -1.0)
    sin = jnp.concatenate((jnp.zeros((tokens, 64)), sin), axis=1)
    return jnp.stack((cos, sin), axis=1)


def _attend(
    q: jax.Array,
    words: jax.Array,
    scales: jax.Array,
    raw: jax.Array,
    high: jax.Array | None,
    starts: jax.Array,
    positions: jax.Array,
    sink: jax.Array,
    rope_rows: jax.Array,
    metadata: jax.Array,
    bounds: jax.Array,
    counts: jax.Array,
    page_size: int,
    ratio: int,
    scale: float,
    non_causal: bool,
    bq: int,
) -> jax.Array:
    return csa2(
        q,
        words,
        scales,
        swa_word_pages(raw if high is None else (raw, high), page_size),
        starts,
        positions,
        sink.astype(jnp.float32),
        _inverse_rope(rope_rows),
        swa_metadata=metadata,
        swa_bounds=bounds if non_causal else None,
        main_lengths=counts,
        compress_ratio=ratio,
        scale=scale,
        bq=bq,
    )


def _attend_swa(
    q: jax.Array,
    raw: jax.Array,
    high: jax.Array | None,
    starts: jax.Array,
    positions: jax.Array,
    sink: jax.Array,
    rope_rows: jax.Array,
    metadata: jax.Array,
    bounds: jax.Array,
    page_size: int,
    scale: float,
    non_causal: bool,
    bq: int,
) -> jax.Array:
    """Attention over the SWA window alone: the layers whose compress ratio is 0."""
    return csa2(
        q,
        None,
        None,
        swa_word_pages(raw if high is None else (raw, high), page_size),
        starts,
        positions,
        sink.astype(jnp.float32),
        _inverse_rope(rope_rows),
        swa_metadata=metadata,
        swa_bounds=bounds if non_causal else None,
        scale=scale,
        bq=bq,
    )


def _gather_attend(
    q: jax.Array,
    raw: jax.Array,
    high: jax.Array | None,
    scales: jax.Array,
    high_scales: jax.Array | None,
    selected: jax.Array,
    pages: jax.Array,
    main_starts: jax.Array,
    swa: jax.Array,
    swa_high: jax.Array | None,
    starts: jax.Array,
    positions: jax.Array,
    sink: jax.Array,
    rope_rows: jax.Array,
    metadata: jax.Array,
    bounds: jax.Array,
    main_page_size: int,
    swa_page_size: int,
    ratio: int,
    scale: float,
    non_causal: bool,
    bq: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    def chunk_records(start, end):
        # Records are per row: a chunk's request boundaries are the batch's,
        # shifted and clipped to the chunk.
        return prepare_main_records(
            selected[start:end],
            pages,
            jnp.clip(main_starts - start, 0, end - start),
            page_size=main_page_size,
            onehot_requests=envs.TPU_ONEHOT_TABLE_LOOKUP,
        )

    return csa2_pipeline_chunks(
        q,
        raw if high is None else (raw, high),
        scales if high_scales is None else (scales, high_scales),
        chunk_records,
        swa_word_pages(swa if swa_high is None else (swa, swa_high), swa_page_size),
        starts,
        positions,
        sink.astype(jnp.float32),
        _inverse_rope(rope_rows),
        metadata,
        bounds if non_causal else None,
        chunk_size=csa2_gather_chunk_size(q.shape[0]),
        bq=bq,
        scale=scale,
        compress_ratio=ratio,
        deep_gather=envs.TPU_DEEPSEEK_V41_CSA2_DEEP_GATHER,
    )


def _select_gather_attend(
    index_q: jax.Array,
    index_banks: tuple[jax.Array, ...],
    index_table: jax.Array,
    weights: jax.Array,
    index_lengths: jax.Array,
    candidates: jax.Array,
    requests: jax.Array,
    batch_max: jax.Array,
    index_starts: jax.Array,
    lengths: jax.Array,
    swa_valid: jax.Array,
    q: jax.Array,
    raw: jax.Array,
    high: jax.Array | None,
    scales: jax.Array,
    high_scales: jax.Array | None,
    pages: jax.Array,
    main_starts: jax.Array,
    swa: jax.Array,
    swa_high: jax.Array | None,
    starts: jax.Array,
    positions: jax.Array,
    sink: jax.Array,
    rope_rows: jax.Array,
    metadata: jax.Array,
    bounds: jax.Array,
    layout_args: tuple[int, ...],
    topk: int,
    block_size: int,
    score_width: int,
    main_page_size: int,
    swa_page_size: int,
    ratio: int,
    scale: float,
    non_causal: bool,
    bq: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Index selection, KV gather and attention of a source layer in one program.

    Each prefill indexer tile's selection is gathered and attended while
    later tiles are still scored (TensorCore) and selected (SparseCore), so
    the gather overlaps scoring instead of following the whole selection;
    decode gathers and attends the whole selection at once. Returns the
    attention output, the main records [T,512] for stage reuse, and the
    gathered words, scales and valid counts, as `_gather_attend` does.
    """
    from .indexer_fp8 import select_fp8_indexer_tiles

    tokens = q.shape[0]
    swa_pages = swa_word_pages(
        swa if swa_high is None else (swa, swa_high), swa_page_size
    )
    word_banks = raw if high is None else (raw, high)
    scale_banks = scales if high_scales is None else (scales, high_scales)
    rope = _inverse_rope(rope_rows)
    sink = sink.astype(jnp.float32)
    kwargs = dict(
        swa_metadata=metadata,
        swa_bounds=bounds if non_causal else None,
        scale=scale,
        compress_ratio=ratio,
        bq=bq,
    )

    def mask(start, selected):
        end = start + selected.shape[0]
        valid = (
            (selected >= 0)
            & (selected < lengths[start:end])
            & swa_valid[start:end, None]
        )
        return jnp.where(valid, selected, -1)

    def consume(start, selected, state, following):
        if isinstance(selected, tuple):
            # Resolved by the SparseCore selection, as below.
            records, counts = selected
        else:
            selected = mask(start, selected)
        if (
            state is None
            and not isinstance(selected, tuple)
            and (selected.shape[0] == tokens)
        ):
            output, words, gathered_scales, counts = _gather_attend(
                q,
                raw,
                high,
                scales,
                high_scales,
                selected,
                pages,
                main_starts,
                swa,
                swa_high,
                starts,
                positions,
                sink,
                rope_rows,
                metadata,
                bounds,
                main_page_size,
                swa_page_size,
                ratio,
                scale,
                non_causal,
                bq,
            )
            return (output, words, gathered_scales, counts), following
        if not isinstance(selected, tuple):
            records, counts = prepare_main_records(
                selected,
                pages,
                jnp.clip(main_starts - start, 0, selected.shape[0]),
                page_size=main_page_size,
                onehot_requests=envs.TPU_ONEHOT_TABLE_LOOKUP,
            )
        rows = records.shape[0]
        if following is not None:
            # Records before the pipeline's next step, so the gather is
            # issued ahead of it and runs beside it instead of after it.
            records, following = jax.lax.optimization_barrier((records, following))

        def attend(out, owners, offset, tile_counts):
            return csa2_attend_rows(
                q,
                (owners[0].reshape(tokens, 512, 128), owners[1].reshape(tokens, 512)),
                tile_counts,
                out,
                swa_pages,
                starts,
                positions,
                sink,
                rope,
                offset=offset,
                kv_owner=True,
                **kwargs,
            )

        if state is None:
            out, owners = None, None
            all_counts = jnp.zeros((tokens,), jnp.int32)
        else:
            out, *owners, all_counts, pending = state
            # Attention runs one tile behind the gather, on rows complete in
            # the owners, and reads them before this tile's gather updates
            # them in place.
            out = attend(out, owners, start - pending.shape[0], pending)
            owners, out = jax.lax.optimization_barrier((tuple(owners), out))
        # Tiles gather into whole-T owners, which attention and later layers
        # read in place: no per-tile copy into a reuse buffer.
        owners = gather_csa2_kv_sc(
            word_banks,
            scale_banks,
            records,
            owners,
            deep=True,
            rows=None if owners else tokens * 512,
            row_offset=start * 512,
        )
        all_counts = jax.lax.dynamic_update_slice_in_dim(all_counts, counts, start, 0)
        if start + rows < tokens:
            return (out, *owners, all_counts, counts), following
        out = attend(out, owners, start, counts)
        kv = (owners[0].reshape(tokens, 512, 128), owners[1].reshape(tokens, 512))
        return (out, *kv, all_counts), following

    selected, _, (output, words, gathered_scales, counts) = select_fp8_indexer_tiles(
        index_q,
        index_banks,
        index_table,
        weights,
        index_lengths,
        candidates,
        requests,
        batch_max,
        index_starts,
        layout_args,
        topk,
        block_size,
        False,
        False,
        True,
        score_width,
        consume=consume,
        records=(
            pages,
            cache_kernels._requests(
                main_starts, tokens, onehot=envs.TPU_ONEHOT_TABLE_LOOKUP
            ),
            main_page_size,
        ),
    )
    return output, mask(0, selected), words, gathered_scales, counts


def _fake_gather_attend(q, raw, high, scales, high_scales, *_):
    return (
        torch.empty_like(q),
        raw.new_empty((q.shape[0], 512, 128)),
        scales.new_empty((q.shape[0], 512)),
        raw.new_empty((q.shape[0],)),
    )


def _fake_select_gather_attend(index_q, index_banks, index_table, weights, *rest):
    q, raw, high, scales = rest[7:11]
    return (
        torch.empty_like(q),
        raw.new_empty((q.shape[0], 512)),
        *_fake_gather_attend(q, raw, high, scales, None)[1:],
    )


def _metadata(
    table: jax.Array,
    starts: jax.Array,
    positions: jax.Array,
    minimum: jax.Array,
    page_size: int,
    page_stride: int,
    row_offset: int,
    non_causal: bool,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    return prepare_swa_metadata(
        table,
        starts,
        positions,
        minimum,
        page_size=page_size,
        page_stride=page_stride,
        row_offset=row_offset,
        non_causal=non_causal,
        onehot_requests=envs.TPU_ONEHOT_TABLE_LOOKUP,
    )


def _metadata_tables(
    tables: tuple[jax.Array, ...],
    starts: jax.Array,
    positions: jax.Array,
    minimum: jax.Array,
    page_size: int,
    page_stride: int,
    non_causal: bool,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """``_metadata`` for G [R,W] block tables at row offset 0, in one program.

    The tables stay separate inputs: stacking them would copy every full
    table each stage. Only the page lookups depend on the table; XLA shares
    the request, window and bound computation across the G calls.
    """
    results = [
        _metadata(
            table, starts, positions, minimum, page_size, page_stride, 0, non_causal
        )
        for table in tables
    ]
    metadata = jnp.stack([result[0] for result in results])
    records = jnp.stack([result[2] for result in results])
    return metadata, results[0][1], records


def _fake_metadata_tables(tables, starts, positions, minimum, page_size, *_):
    columns = (positions.shape[0] + 128 + page_size - 1) // page_size + 1
    requests = tables[0].shape[0]
    return (
        tables[0].new_empty((len(tables), requests, 2 + columns)),
        tables[0].new_empty((requests, 2)),
        positions.new_empty((len(tables), positions.shape[0])),
    )


def _fake_metadata(table, starts, positions, minimum, page_size, *_):
    columns = (positions.shape[0] + 128 + page_size - 1) // page_size + 1
    return (
        table.new_empty((table.shape[0], 2 + columns)),
        table.new_empty((table.shape[0], 2)),
        positions.new_empty(positions.shape),
    )


def _prepare_publish_swa(
    raw: jax.Array,
    table: jax.Array,
    starts: jax.Array,
    positions: jax.Array,
    minimum: jax.Array,
    values: jax.Array,
    page_stride: int,
    row_offset: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    metadata, bounds, records = _metadata(
        table, starts, positions, minimum, 64, page_stride, row_offset, False
    )
    owner = write_swa_word_pool(raw, records, values, page_size=64)
    return owner, metadata, bounds, records


def _prepare_publish_swa_banked(
    raw: jax.Array,
    high: jax.Array,
    table: jax.Array,
    starts: jax.Array,
    positions: jax.Array,
    minimum: jax.Array,
    values: jax.Array,
    page_stride: int,
    row_offset: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    metadata, bounds, records = _metadata(
        table, starts, positions, minimum, 64, page_stride, row_offset, False
    )
    owner = write_swa_word_pool((raw, high), records, values, page_size=64)
    return *owner, metadata, bounds, records


def _fake_prepare_publish_swa(raw, table, starts, positions, minimum, values, *_):
    return torch.empty_like(raw), *_fake_metadata(table, starts, positions, minimum, 64)


def _fake_prepare_publish_swa_banked(
    raw, high, table, starts, positions, minimum, values, *_
):
    return (
        torch.empty_like(raw),
        torch.empty_like(high),
        *_fake_metadata(table, starts, positions, minimum, 64),
    )


def _project(output: jax.Array, weight: jax.Array, scales: jax.Array) -> jax.Array:
    return wo_a_projection(output, weight, scales.reshape(-1), None)


def _project_out(
    output: jax.Array,
    wa: jax.Array,
    sa: jax.Array,
    wb: jax.Array,
    sb: jax.Array,
) -> jax.Array:
    # wo_a then wo_b. On v7 with the DSv4.1 weights the fused kernel runs at
    # every token count, in token tiles beyond one VMEM-resident tile, and is
    # bitwise equal to the two-kernel chain, which other contracts still run.
    from vllm_torchtpu.layers.adapter.linear_common import _quantized_matmul_jax

    sa = sa.reshape(-1)
    if output_projection.is_supported(output, wa, sa, wb, sb):
        return output_projection.fused_output_projection(output, wa, sa, wb, sb)
    projected = _quantized_matmul_jax(wo_a_projection(output, wa, sa, None), wb, sb)
    # Every path returns the fused kernel's padded row count.
    rows = output_projection.padded_rows(output.shape[0]) - output.shape[0]
    return jnp.pad(projected, ((0, rows), (0, 0)))


def _manual_bridge(function, mesh, tensor_args, out_specs):
    from jax.sharding import PartitionSpec as P

    @functools.wraps(function)
    def call(*args):
        # Scalars are static trailing arguments. Optional banks remain part of
        # the array pytree, matching the native attention bridge contract.
        return jax.shard_map(
            lambda *arrays: function(*arrays, *args[tensor_args:]),
            mesh=mesh,
            in_specs=(P(),) * tensor_args,
            out_specs=out_specs,
            check_vma=False,
        )(*args[:tensor_args])

    return call


def initialize_csa2_ops() -> None:
    if _ops:
        return
    from jax.sharding import PartitionSpec as P
    from torch_tpu._internal import pallas

    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        get_pallas_op_mesh,
        get_vllm_model_wrapper_context,
    )

    mesh = get_vllm_model_wrapper_context().mesh
    definitions = (
        ("metadata", _metadata, (P(),) * 4 + (None,) * 4, _fake_metadata),
        (
            "metadata_tables",
            _metadata_tables,
            (P(),) * 4 + (None,) * 3,
            _fake_metadata_tables,
        ),
        (
            "prepare_publish_swa",
            _prepare_publish_swa,
            (P(),) * 6 + (None,) * 2,
            _fake_prepare_publish_swa,
        ),
        (
            "prepare_publish_swa_banked",
            _prepare_publish_swa_banked,
            (P(),) * 7 + (None,) * 2,
            _fake_prepare_publish_swa_banked,
        ),
        (
            "publish_main",
            _publish_main,
            (P(),) * 5,
            lambda r, s, *_: (torch.empty_like(r), torch.empty_like(s)),
        ),
        (
            "publish_main_banked",
            _publish_main_banked,
            (P(),) * 7,
            lambda r, h, s, hs, *_: tuple(torch.empty_like(x) for x in (r, h, s, hs)),
        ),
        (
            "publish_swa",
            _publish_swa,
            (P(),) * 3 + (None,) * 2,
            lambda r, *_: torch.empty_like(r),
        ),
        (
            "publish_swa_banked",
            _publish_swa_banked,
            (P(),) * 4 + (None,) * 2,
            lambda r, h, *_: (torch.empty_like(r), torch.empty_like(h)),
        ),
        (
            "attend",
            _attend,
            (P(),) * 12 + (None,) * 5,
            lambda q, *_: torch.empty_like(q),
        ),
        (
            "attend_swa",
            _attend_swa,
            (P(),) * 9 + (None,) * 4,
            lambda q, *_: torch.empty_like(q),
        ),
        (
            "gather_attend",
            _gather_attend,
            (P(),) * 16 + (None,) * 6,
            _fake_gather_attend,
        ),
        (
            "select_gather_attend",
            _select_gather_attend,
            (P(),) * 26 + (None,) * 10,
            _fake_select_gather_attend,
        ),
        (
            "project",
            _project,
            (P(),) * 3,
            lambda x, w, *_: x.new_empty((x.shape[0], w.shape[1])),
        ),
        (
            "project_out",
            _project_out,
            (P(),) * 5,
            lambda x, wa, sa, wb, sb: x.new_empty(
                (output_projection.padded_rows(x.shape[0]), wb.shape[1])
            ),
        ),
    )
    output_counts = {
        "prepare_publish_swa": 4,
        "prepare_publish_swa_banked": 5,
        "metadata": 3,
        "metadata_tables": 3,
        "publish_main": 2,
        "publish_main_banked": 4,
        "publish_swa_banked": 2,
        "gather_attend": 4,
        "select_gather_attend": 5,
    }
    donated = {
        "prepare_publish_swa": (0,),
        "prepare_publish_swa_banked": (0, 1),
        "publish_main": (0, 1),
        "publish_main_banked": (0, 1, 2, 3),
        "publish_swa": (0,),
        "publish_swa_banked": (0, 1),
    }
    for name, function, specs, fake in definitions:
        count = output_counts.get(name, 1)
        out_specs = P() if count == 1 else (P(),) * count
        bridge = _manual_bridge(
            function, mesh, sum(spec is not None for spec in specs), out_specs
        )
        op = pallas.jax_op(
            "pallas::dsv41v2_csa2_" + name,
            bridge,
            # Carry ownership through the Torch boundary as well as the inner
            # JAX kernel; otherwise publication can copy the entire KV pool.
            donate_argnums=donated.get(name, ()),
            mesh=get_pallas_op_mesh(),
            input_partition_specs=cast(tuple[tuple[str, ...], ...], cast(Any, specs)),
        )
        op.register_fake(fake)
        _ops[name] = op


def publish_main(
    cache: "PagedCacheRows",
    records: torch.Tensor,
    words: torch.Tensor,
    scales: torch.Tensor,
) -> None:
    """Publish the fused compressor's CSA2 words/scales at physical ``records``."""
    assert cache.scale is not None
    args = (records.int(), words, scales.reshape(-1))
    if cache.raw_hi is None:
        raw, scale = _ops["publish_main"](cache.raw, cache.scale, *args)
        cache.raw.copy_(raw)
        cache.scale.copy_(scale)
    else:
        assert cache.scale_hi is not None
        raw, high, scale, high_scale = _ops["publish_main_banked"](
            cache.raw, cache.raw_hi, cache.scale, cache.scale_hi, *args
        )
        cache.raw.copy_(raw)
        cache.raw_hi.copy_(high)
        cache.scale.copy_(scale)
        cache.scale_hi.copy_(high_scale)


def publish_swa(
    cache: "PagedCacheRows",
    records: torch.Tensor,
    values: torch.Tensor,
    shift: int = 0,
) -> None:
    """Publish ``values`` at ``records + shift`` (see `write_swa_word_pool`)."""
    if cache.raw_hi is None:
        cache.raw.copy_(
            _ops["publish_swa"](cache.raw, records.int(), values, cache.shape[1], shift)
        )
    else:
        low, high = _ops["publish_swa_banked"](
            cache.raw, cache.raw_hi, records.int(), values, cache.shape[1], shift
        )
        cache.raw.copy_(low)
        cache.raw_hi.copy_(high)


def _attention_bq(tokens: int) -> int:
    return 6 if tokens <= 384 else 16


def attention(
    q: torch.Tensor,
    swa: "PagedCacheRows",
    main: "PagedCacheRows",
    records: torch.Tensor,
    pages: torch.Tensor | None,
    main_starts: torch.Tensor | None,
    starts: torch.Tensor,
    positions: torch.Tensor,
    sink: torch.Tensor,
    rope_rows: torch.Tensor,
    metadata: torch.Tensor,
    bounds: torch.Tensor,
    ratio: int,
    scale: float,
    non_causal: bool,
) -> torch.Tensor:
    from . import attention_kernels

    if not q.shape[0]:
        return q
    bq = _attention_bq(q.shape[0])
    if not records.shape[1]:
        return _ops["attend_swa"](
            q,
            swa.raw,
            swa.raw_hi,
            starts.int(),
            positions.int(),
            sink,
            rope_rows,
            metadata.int(),
            bounds.int(),
            swa.shape[1],
            scale,
            non_causal,
            bq,
        )
    if pages is None:
        # Already physical records: a single synthetic sequence preserves IDs.
        page_size = (
            math.gcd(main.raw.shape[0], main.raw_hi.shape[0])
            if main.raw_hi is not None
            else main.raw.shape[0]
        )
        pages = torch.arange(
            main.total_rows // page_size, dtype=torch.int32, device=records.device
        )[None, :]
        main_starts = torch.stack((starts[0] * 0, starts[-1]))
    else:
        page_size = main.shape[1]
    assert main_starts is not None and main.scale is not None
    cache = attention_kernels._active_gathered_kv
    layout = ("csa2", page_size)
    gathered = cache.get(main, records, layout) if cache is not None else None
    if gathered is None:
        output, words, scales, counts = _ops["gather_attend"](
            q,
            main.raw,
            main.raw_hi,
            main.scale,
            main.scale_hi,
            records.int(),
            pages.int(),
            main_starts.int(),
            swa.raw,
            swa.raw_hi,
            starts.int(),
            positions.int(),
            sink,
            rope_rows,
            metadata.int(),
            bounds.int(),
            page_size,
            swa.shape[1],
            ratio,
            scale,
            non_causal,
            bq,
        )
        if cache is not None:
            cache.put(main, records, layout, (words, scales), valid_counts=counts)
        return output
    assert cache is not None and cache.valid_counts is not None
    words, scales = gathered
    return _ops["attend"](
        q,
        words,
        scales,
        swa.raw,
        swa.raw_hi,
        starts.int(),
        positions.int(),
        sink,
        rope_rows,
        metadata.int(),
        bounds.int(),
        cache.valid_counts,
        swa.shape[1],
        ratio,
        scale,
        non_causal,
        bq,
    )


def select_attention(
    index_q: torch.Tensor,
    index_cache: Any,
    index_table: torch.Tensor,
    weights: torch.Tensor,
    index_lengths: torch.Tensor,
    requests: torch.Tensor,
    batch_max: torch.Tensor,
    index_starts: torch.Tensor,
    lengths: torch.Tensor,
    swa_valid: torch.Tensor,
    q: torch.Tensor,
    swa: "PagedCacheRows",
    main: "PagedCacheRows",
    pages: torch.Tensor,
    main_starts: torch.Tensor,
    starts: torch.Tensor,
    positions: torch.Tensor,
    sink: torch.Tensor,
    rope_rows: torch.Tensor,
    metadata: torch.Tensor,
    bounds: torch.Tensor,
    *,
    topk: int,
    block_size: int,
    score_width: int,
    ratio: int,
    scale: float,
    non_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A source layer's index selection and attention (`_select_gather_attend`).

    The index arguments are those of the packed FP8 indexer's dense selection.
    Returns the attention output and the main records, whose gathered KV is
    published to the stage's gathered-KV cache for the layers that reuse it.
    """
    from . import attention_kernels

    assert main.scale is not None
    output, records, words, scales, counts = _ops["select_gather_attend"](
        index_q,
        index_cache.banks,
        index_table.int(),
        weights,
        index_lengths.flatten().int(),
        index_lengths.new_empty((index_q.shape[0], 0), dtype=torch.int32),
        requests.int(),
        batch_max.reshape(1).int(),
        index_starts.int(),
        lengths.int(),
        swa_valid,
        q,
        main.raw,
        main.raw_hi,
        main.scale,
        main.scale_hi,
        pages.int(),
        main_starts.int(),
        swa.raw,
        swa.raw_hi,
        starts.int(),
        positions.int(),
        sink,
        rope_rows,
        metadata.int(),
        bounds.int(),
        index_cache._layout_args(),
        topk,
        block_size,
        score_width,
        main.shape[1],
        swa.shape[1],
        ratio,
        scale,
        non_causal,
        _attention_bq(q.shape[0]),
    )
    cache = attention_kernels._active_gathered_kv
    if cache is not None:
        cache.put(
            main, records, ("csa2", main.shape[1]), (words, scales), valid_counts=counts
        )
    return output, records


def project(
    output: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    return _ops["project"](output, weight, scales)


def project_out(
    output: torch.Tensor,
    wa: torch.Tensor,
    sa: torch.Tensor,
    wb: torch.Tensor,
    sb: torch.Tensor,
) -> torch.Tensor:
    """``wo_b(wo_a(output))``; ``wb`` is the canonical ``[k, n]`` FP8 weight
    with a per-channel ``[n]`` scale. Rows past ``output``'s are zero
    padding (`output_projection.padded_rows`)."""
    return _ops["project_out"](output, wa, sa, wb, sb)


def prepare_swa(
    table: torch.Tensor,
    starts: torch.Tensor,
    positions: torch.Tensor,
    minimum: torch.Tensor,
    cache: "PagedCacheRows",
    non_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _ops["metadata"](
        table.int(),
        starts.int(),
        positions.int(),
        minimum.int(),
        cache.shape[1],
        cache.page_stride,
        cache.row_offset,
        non_causal,
    )


def prepare_swa_tables(
    tables: tuple[torch.Tensor, ...],
    starts: torch.Tensor,
    positions: torch.Tensor,
    minimum: torch.Tensor,
    cache: "PagedCacheRows",
    non_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``prepare_swa`` for G [R,W] tables at row offset 0: [G,...] metadata
    and records, with the table-independent bounds shared."""
    return _ops["metadata_tables"](
        tuple(table.int() for table in tables),
        starts.int(),
        positions.int(),
        minimum.int(),
        cache.shape[1],
        cache.page_stride,
        non_causal,
    )


def prepare_publish_swa(
    table: torch.Tensor,
    starts: torch.Tensor,
    positions: torch.Tensor,
    minimum: torch.Tensor,
    cache: "PagedCacheRows",
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Publish into the shared word pool while creating the shared metadata."""
    args = (
        table.int(),
        starts.int(),
        positions.int(),
        minimum.int(),
        values,
        cache.page_stride,
        cache.row_offset,
    )
    if cache.raw_hi is None:
        raw, metadata, bounds, records = _ops["prepare_publish_swa"](cache.raw, *args)
        cache.raw.copy_(raw)
    else:
        raw, high, metadata, bounds, records = _ops["prepare_publish_swa_banked"](
            cache.raw, cache.raw_hi, *args
        )
        cache.raw.copy_(raw)
        cache.raw_hi.copy_(high)
    return metadata, bounds, records
