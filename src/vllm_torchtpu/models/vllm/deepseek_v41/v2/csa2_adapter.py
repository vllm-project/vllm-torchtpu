# SPDX-License-Identifier: Apache-2.0
"""TP1 CSA2 bridges, retaining stage-scoped gathered KV ownership."""

import functools
import math
from typing import TYPE_CHECKING, Any, cast

import jax
import jax.numpy as jnp
import torch

from vllm_torchtpu.kernels.deepseek_v4.o_projection import wo_a_projection
from vllm_torchtpu.kernels.deepseek_v41.csa2 import csa2
from vllm_torchtpu.kernels.deepseek_v41.csa2_cache import (
    publish_csa2_kv,
)
from vllm_torchtpu.kernels.deepseek_v41.csa2_metadata import (
    prepare_main_records,
    prepare_swa_metadata,
)
from vllm_torchtpu.kernels.deepseek_v41.csa2_pipeline import csa2_pipeline
from vllm_torchtpu.kernels.deepseek_v41.tuned_block_sizes import (
    csa2_attention_bq,
    csa2_gather_chunk_size,
)
from vllm_torchtpu.kernels.mla.paged_swa import swa_word_pages, write_swa_word_pool

if TYPE_CHECKING:
    from .cache import PagedCacheRows

_ops: dict[str, Any] = {}


def _publish_main(
    raw: jax.Array, scales: jax.Array, records: jax.Array, values: jax.Array
) -> tuple[jax.Array, jax.Array]:
    return publish_csa2_kv(raw, scales, records, values)


def _publish_main_banked(
    raw: jax.Array,
    high: jax.Array,
    scales: jax.Array,
    high_scales: jax.Array,
    records: jax.Array,
    values: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    words, scale = publish_csa2_kv((raw, high), (scales, high_scales), records, values)
    return *words, *scale


def _publish_swa(
    raw: jax.Array, records: jax.Array, values: jax.Array, page_size: int
) -> jax.Array:
    return write_swa_word_pool(raw, records, values, page_size=page_size)


def _publish_swa_banked(
    raw: jax.Array,
    high: jax.Array,
    records: jax.Array,
    values: jax.Array,
    page_size: int,
) -> tuple[jax.Array, jax.Array]:
    return write_swa_word_pool((raw, high), records, values, page_size=page_size)


def _inverse_rope(cos_sin: jax.Array, positions: jax.Array) -> jax.Array:
    # Q/K already carry RoPE; CSA2 consumes this small inverse-rotation table.
    table = cos_sin[positions].astype(jnp.float32)
    cos = jnp.concatenate(
        (jnp.ones((positions.shape[0], 64)), jnp.repeat(table[:, :32], 2, axis=1)),
        axis=1,
    )
    sin = jnp.repeat(table[:, 32:], 2, axis=1)
    sin *= jnp.where(jnp.arange(64) % 2 == 0, 1.0, -1.0)
    sin = jnp.concatenate((jnp.zeros((positions.shape[0], 64)), sin), axis=1)
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
    cos_sin: jax.Array,
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
        _inverse_rope(cos_sin, positions),
        swa_metadata=metadata,
        swa_bounds=bounds if non_causal else None,
        main_lengths=counts,
        compress_ratio=ratio,
        scale=scale,
        bq=csa2_attention_bq(
            q.shape[0],
            q.shape[1],
            starts.shape[0] - 1,
            page_size,
            non_causal,
            default_bq=bq,
        ),
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
    cos_sin: jax.Array,
    metadata: jax.Array,
    bounds: jax.Array,
    main_page_size: int,
    swa_page_size: int,
    ratio: int,
    scale: float,
    non_causal: bool,
    bq: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    records, counts = prepare_main_records(
        selected, pages, main_starts, page_size=main_page_size
    )
    chunk_size = csa2_gather_chunk_size(q.shape[0])
    output, words, scales = csa2_pipeline(
        q,
        raw if high is None else (raw, high),
        scales if high_scales is None else (scales, high_scales),
        records,
        swa_word_pages(swa if swa_high is None else (swa, swa_high), swa_page_size),
        starts,
        positions,
        sink.astype(jnp.float32),
        _inverse_rope(cos_sin, positions),
        metadata,
        bounds if non_causal else None,
        counts,
        chunk_size=chunk_size,
        bq=csa2_attention_bq(
            min(q.shape[0], chunk_size),
            q.shape[1],
            starts.shape[0] - 1,
            swa_page_size,
            non_causal,
            default_bq=bq,
        ),
        scale=scale,
        compress_ratio=ratio,
    )
    return output, words, scales, counts


def _fake_gather_attend(q, raw, high, scales, high_scales, *_):
    return (
        torch.empty_like(q),
        raw.new_empty((q.shape[0], 512, 128)),
        scales.new_empty((q.shape[0], 512)),
        raw.new_empty((q.shape[0],)),
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
    )


def _fake_metadata(table, starts, positions, minimum, page_size, *_):
    columns = (positions.shape[0] + 128 + page_size - 1) // page_size + 1
    return (
        table.new_empty((table.shape[0], 2 + columns)),
        table.new_empty((table.shape[0], 2)),
        positions.new_empty(positions.shape),
    )


def _project(output: jax.Array, weight: jax.Array, scales: jax.Array) -> jax.Array:
    return wo_a_projection(output, weight, scales.reshape(-1), None)


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
        get_vllm_model_wrapper_context,
    )

    mesh = get_vllm_model_wrapper_context().mesh
    definitions = (
        ("metadata", _metadata, (P(),) * 4 + (None,) * 4, _fake_metadata),
        (
            "publish_main",
            _publish_main,
            (P(),) * 4,
            lambda r, s, *_: (torch.empty_like(r), torch.empty_like(s)),
        ),
        (
            "publish_main_banked",
            _publish_main_banked,
            (P(),) * 6,
            lambda r, h, s, hs, *_: tuple(torch.empty_like(x) for x in (r, h, s, hs)),
        ),
        (
            "publish_swa",
            _publish_swa,
            (P(),) * 3 + (None,),
            lambda r, *_: torch.empty_like(r),
        ),
        (
            "publish_swa_banked",
            _publish_swa_banked,
            (P(),) * 4 + (None,),
            lambda r, h, *_: (torch.empty_like(r), torch.empty_like(h)),
        ),
        (
            "attend",
            _attend,
            (P(),) * 12 + (None,) * 5,
            lambda q, *_: torch.empty_like(q),
        ),
        (
            "gather_attend",
            _gather_attend,
            (P(),) * 16 + (None,) * 6,
            _fake_gather_attend,
        ),
        (
            "project",
            _project,
            (P(),) * 3,
            lambda x, w, *_: x.new_empty((x.shape[0], w.shape[1])),
        ),
    )
    output_counts = {
        "metadata": 3,
        "publish_main": 2,
        "publish_main_banked": 4,
        "publish_swa_banked": 2,
        "gather_attend": 4,
    }
    donated = {
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
            mesh=mesh,
            input_partition_specs=cast(Any, specs),
        )
        op.register_fake(fake)
        _ops[name] = op


def publish_main(
    cache: "PagedCacheRows", records: torch.Tensor, values: torch.Tensor
) -> None:
    assert cache.scale is not None
    if cache.raw_hi is None:
        raw, scale = _ops["publish_main"](cache.raw, cache.scale, records.int(), values)
        cache.raw.copy_(raw)
        cache.scale.copy_(scale)
    else:
        assert cache.scale_hi is not None
        raw, high, scale, high_scale = _ops["publish_main_banked"](
            cache.raw, cache.raw_hi, cache.scale, cache.scale_hi, records.int(), values
        )
        cache.raw.copy_(raw)
        cache.raw_hi.copy_(high)
        cache.scale.copy_(scale)
        cache.scale_hi.copy_(high_scale)


def publish_swa(
    cache: "PagedCacheRows", records: torch.Tensor, values: torch.Tensor
) -> None:
    if cache.raw_hi is None:
        cache.raw.copy_(
            _ops["publish_swa"](cache.raw, records.int(), values, cache.shape[1])
        )
    else:
        low, high = _ops["publish_swa_banked"](
            cache.raw, cache.raw_hi, records.int(), values, cache.shape[1]
        )
        cache.raw.copy_(low)
        cache.raw_hi.copy_(high)


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
    cos_sin: torch.Tensor,
    metadata: torch.Tensor,
    bounds: torch.Tensor,
    ratio: int,
    scale: float,
    non_causal: bool,
) -> torch.Tensor:
    from . import attention_kernels

    if not q.shape[0]:
        return q
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
    bq = 6 if q.shape[0] <= 384 else 16
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
            cos_sin,
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
        cos_sin,
        metadata.int(),
        bounds.int(),
        cache.valid_counts,
        swa.shape[1],
        ratio,
        scale,
        non_causal,
        bq,
    )


def project(
    output: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    return _ops["project"](output, weight, scales)


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
