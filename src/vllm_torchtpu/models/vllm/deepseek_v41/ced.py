"""Decoder replay planning and its bounded SWA state for shared CSA."""

import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import torch

from .cache import gather_cache_values

_attention_op = None


@dataclass(frozen=True)
class DecoderReplayPlan:
    indices: tuple[int, ...]
    query_start_loc: tuple[int, ...]
    replay_starts: tuple[int, ...]
    padded_num_tokens: int


def plan_decoder_replay(metadata, window, num_tokens, compile_sizes):
    """Keep the same prompt-tail window across every chunk and later decode.

    The encoder publishes global KV for the complete chunk before this plan
    narrows the decoder batch. Prefix caching must be disabled: a cached prefix
    may not contain the encoder outputs required to rebuild decoder SWA.
    """
    bounds = metadata.query_start_loc_cpu
    lengths = metadata.seq_lens_cpu
    prompts = metadata.prompt_lens_cpu
    if bounds is None or lengths is None or prompts is None:
        return None  # Profiling/warmup executes both stages at the full bucket.
    if window < 1 or len(bounds) != len(lengths) + 1 or len(prompts) != len(lengths):
        raise ValueError("Invalid decoder replay scheduling metadata")
    if bounds[0] != 0 or bounds[-1] > num_tokens:
        raise ValueError("Query boundaries exceed the input token bucket")
    indices, replay_starts, output_bounds = [], [], [0]
    for start, end, length, prompt in zip(bounds, bounds[1:], lengths, prompts):
        count = end - start
        if count < 0 or length < count or prompt < 0:
            raise ValueError("Invalid scheduled query or prompt length")
        replay_start = max(0, prompt - window)
        skip = min(count, max(0, replay_start - (length - count)))
        indices.extend(range(start + skip, end))
        replay_starts.append(replay_start)
        output_bounds.append(len(indices))
    kept = len(indices)
    # Each stage dispatches by its own token bucket. Never introduce a second,
    # independent symbolic token dimension into vLLM's guard-free full graph.
    buckets = sorted({num_tokens, *(int(n) for n in compile_sizes if int(n) > 0)})
    padded = next(n for n in buckets if n >= kept) if kept else 0
    return DecoderReplayPlan(
        tuple(indices), tuple(output_bounds), tuple(replay_starts), padded
    )


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
    swa_records: jax.Array,
    selected: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    sink: jax.Array,
    *,
    scale: float,
    logical_main_page_size: int,
) -> jax.Array:
    from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4 import (
        deepseek_v4_attention_op as attention_ops,
    )

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
        swa_records: jax.Array,
        selected: jax.Array,
        page_indices: jax.Array,
        cu_q_lens: jax.Array,
        distribution: jax.Array,
        sink: jax.Array,
        scale: float,
        logical_main_page_size: int,
    ) -> jax.Array:
        return jax.shard_map(
            functools.partial(
                ced_attention_jax,
                scale=scale,
                logical_main_page_size=logical_main_page_size,
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
            swa_records,
            selected,
            page_indices,
            cu_q_lens,
            distribution,
            sink,
        )

    _attention_op = pallas.jax_op(
        "pallas::deepseek_v41_ced_attention",
        attend,
        mesh=mesh,
        input_partition_specs=specs + (None, None),
    )
    _attention_op.register_fake(lambda q, *args: torch.empty_like(q))


def ced_attention(q, swa, main, records, selected, metadata, sink, scale, num_states):
    assert _attention_op is not None
    return _attention_op(
        q,
        swa.raw,
        *main,
        records.int(),
        selected.int(),
        metadata.block_tables,
        metadata.query_start_loc,
        metadata.request_distribution,
        sink,
        scale,
        num_states,
    )
