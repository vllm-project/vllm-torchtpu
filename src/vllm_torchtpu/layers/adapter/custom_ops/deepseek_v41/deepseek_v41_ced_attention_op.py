"""V4.1 decoder SWA state and native CSA integration."""

import functools
from typing import cast

import jax
import jax.numpy as jnp
import torch

from vllm_torchtpu.distributed.pallas_shapes import get_kernel_mesh

from .deepseek_v41_cache import gather_cache_values, masked_row_indices

_attention_op = None


def swa_attention_state(q, pool, records, scale):
    """Compute the BF16 SWA state consumed by the native CSA merger."""
    tokens, heads, dim = q.shape
    tile = 64
    state = (
        jnp.full((tokens, heads), -1e30, jnp.float32),
        jnp.zeros((tokens, heads), jnp.float32),
        jnp.zeros(q.shape, jnp.float32),
    )
    width = records.shape[1]
    if not width:
        return state
    records = jnp.pad(records, ((0, 0), (0, -width % tile)), constant_values=-1)

    # Gather BF16 rows once outside; keep FP32 scratch tiled.
    gathered = gather_cache_values(
        pool, jnp.maximum(records, 0).reshape(-1), dim
    ).reshape(tokens, records.shape[1], dim)

    def step(index, carry):
        maximum, denominator, numerator = carry
        addresses = jax.lax.dynamic_slice_in_dim(records, index * tile, tile, axis=1)
        valid = addresses >= 0
        kv = jax.lax.dynamic_slice_in_dim(gathered, index * tile, tile, axis=1)
        kv = jnp.where(valid[..., None], kv, 0)
        scores = (
            jnp.einsum(
                "thd,tkd->thk",
                q.astype(jnp.float32),
                kv.astype(jnp.float32),
                preferred_element_type=jnp.float32,
            )
            * scale
        )
        scores = jnp.where(valid[:, None], scores, -jnp.inf)
        new_max = jnp.maximum(maximum, scores.max(-1))
        correction = jnp.exp(maximum - new_max)
        exponentials = jnp.exp(scores - new_max[..., None])
        denominator = denominator * correction + exponentials.sum(-1)
        numerator = numerator * correction[..., None] + jnp.einsum(
            "thk,tkd->thd",
            exponentials.astype(jnp.bfloat16),
            kv,
            preferred_element_type=jnp.float32,
        )
        return new_max, denominator, numerator

    return jax.lax.fori_loop(0, records.shape[1] // tile, step, state)


def ced_attention_jax(
    q: jax.Array,
    swa: jax.Array,
    main_nope: jax.Array,
    main_rope: jax.Array,
    swa_slots: jax.Array,
    selected: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    sink: jax.Array,
    *,
    scale: float,
    logical_main_page_size: int,
    page_size: int,
    page_stride: int,
    row_offset: int,
) -> jax.Array:
    from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4 import (
        deepseek_v4_attention_op as attention_ops,
    )

    swa_records = masked_row_indices(swa_slots, page_size, page_stride, row_offset)
    maximum, denominator, numerator = swa_attention_state(q, swa, swa_records, scale)
    if selected.shape[1] == 0:
        denominator += jnp.exp(sink.astype(jnp.float32)[None] - maximum)
        return (numerator / denominator[..., None]).astype(q.dtype)
    return attention_ops._attention_csa_from_swa(
        q,
        main_nope,
        main_rope,
        selected,
        page_indices,
        cu_q_lens,
        distribution,
        sink.astype(jnp.float32),
        numerator.astype(q.dtype),
        denominator,
        maximum,
        sm_scale=scale,
        logical_main_page_size=logical_main_page_size,
        attention_kernel_batch_size=min(4, max(1, 4096 // selected.shape[1])),
    )


def initialize_ced_attention():
    global _attention_op
    if _attention_op is not None:
        return
    from jax.sharding import PartitionSpec as P
    from torch_tpu._internal import pallas

    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        get_vllm_model_wrapper_context,
    )

    mesh = get_vllm_model_wrapper_context().mesh
    specs = (P(None, "model", None),) + (P(),) * 8 + (P("model"),)

    def attend(
        q: jax.Array,
        swa: jax.Array,
        main_nope: jax.Array,
        main_rope: jax.Array,
        swa_slots: jax.Array,
        selected: jax.Array,
        page_indices: jax.Array,
        cu_q_lens: jax.Array,
        distribution: jax.Array,
        sink: jax.Array,
        scale: float,
        logical_main_page_size: int,
        page_size: int,
        page_stride: int,
        row_offset: int,
    ) -> jax.Array:
        return jax.shard_map(
            functools.partial(
                ced_attention_jax,
                scale=scale,
                logical_main_page_size=logical_main_page_size,
                page_size=page_size,
                page_stride=page_stride,
                row_offset=row_offset,
            ),
            mesh=mesh,
            in_specs=specs,
            out_specs=P(None, "model", None),
            check_vma=False,
        )(
            q,
            swa,
            main_nope,
            main_rope,
            swa_slots,
            selected,
            page_indices,
            cu_q_lens,
            distribution,
            sink,
        )

    _attention_op = pallas.jax_op(
        "pallas::deepseek_v41_ced_attention",
        attend,
        mesh=get_kernel_mesh(mesh),
        input_partition_specs=cast(tuple[tuple[str, ...], ...], specs + (None,) * 5),
    )
    _attention_op.register_fake(lambda q, *args: torch.empty_like(q))


def ced_attention(
    q, swa, main, slots, selected, metadata, sink, scale, num_states, *, layout
):
    assert _attention_op is not None
    return _attention_op(
        q,
        swa,
        *main,
        slots.int(),
        selected.int(),
        metadata.block_tables,
        metadata.query_start_loc,
        metadata.request_distribution,
        sink,
        scale,
        num_states,
        *layout.geometry,
    )
