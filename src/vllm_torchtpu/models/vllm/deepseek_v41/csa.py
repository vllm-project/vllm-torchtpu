"""V4.1 cache ownership and index selection feeding the native V4 FP8 CSA."""

import functools
import math
from dataclasses import replace

import jax
import jax.numpy as jnp
import torch
from vllm.v1.kv_cache_interface import MLAAttentionSpec

from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout
from vllm_torchtpu.kernels.deepseek_v4.core_attention import sparse_mla

from .cache import gather_cache_values, scatter_cache_words

CSA_RECORD_BYTES = 640  # 448 FP8 values, 64 E8M0 scales, 64 BF16 RoPE values.

_write_op = None
_attention_op = None


def packed_csa_spec(spec: MLAAttentionSpec) -> MLAAttentionSpec:
    """Reserve the upstream page geometry while storing 640-byte FP8 payloads.

    MLA page unification scales block sizes by integer ratios. Natural FP8
    pages (640 bytes/state) do not divide the BF16/state pages in V4.1. Keep
    the original bytes/state reservation, including after block-size changes;
    this adapter currently saves no physical cache allocation.
    """
    if spec.num_states % 4:
        raise ValueError("Native CSA requires a multiple of four KV states per page")
    if spec.state_content_size_bytes < CSA_RECORD_BYTES:
        raise ValueError("Native CSA needs at least 640 bytes per reserved KV state")
    return replace(
        spec, dtype=torch.uint8, state_content_bytes=spec.state_content_size_bytes
    )


def pack_csa_values(values: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Encode already-normalized/rotated V4.1 KV in the V4 lane-periodic layout."""
    assert values.ndim == 2 and values.shape[1] == 512
    assert values.dtype == jnp.bfloat16
    groups = values[:, :448].astype(jnp.float32).reshape(-1, 7, 64)
    amax = jnp.maximum(jnp.abs(groups).max(axis=1), 1e-4)
    exponent = jnp.ceil(jnp.log2(amax / 448.0)).astype(jnp.int32)
    scale_bytes = (exponent + 127).astype(jnp.uint8)
    # Construct an exact power of two: exp2's approximate result can sit just
    # below it, making extraction of the FP32 exponent select the wrong E8M0.
    scales = jax.lax.bitcast_convert_type(
        scale_bytes.astype(jnp.uint32) << 23, jnp.float32
    )
    quantized = (groups / scales[:, None]).astype(jnp.float8_e4m3fn)
    fp8_bytes = jax.lax.bitcast_convert_type(quantized, jnp.uint8).reshape(-1, 448)
    slabs = jnp.concatenate((fp8_bytes, scale_bytes), axis=1).reshape(-1, 4, 128)
    return csa_cache_layout.slabs_to_words(slabs), csa_cache_layout.rope_words(
        values[:, 448:]
    )


def write_csa_values(
    pool: jax.Array,
    nope_records: jax.Array,
    rope_records: jax.Array,
    values: jax.Array,
) -> jax.Array:
    nope, rope = pack_csa_values(values)
    pool = scatter_cache_words(pool, nope_records, nope)
    return scatter_cache_words(pool, rope_records, rope)


class PackedCSACache:
    """Two packed regions per upstream-owned page in one shared word-row pool.

    A page with T compressed tokens stores T NoPE rows, followed by T/4 RoPE
    rows and any allocator padding. Page stride and offset still come from
    vLLM, including overlays with indexer/state/SWA allocations. No strided
    whole-cache tensor is copied.
    """

    def __init__(self, raw, layout):
        self.raw = raw
        self.shape, self.dtype, stride_bytes, offset_bytes = layout
        if raw.dtype != torch.int32 or raw.ndim != 2 or raw.shape[1] != 128:
            raise ValueError("CSA cache requires int32 word rows with 128 lanes")
        if (
            self.dtype != torch.uint8
            or len(self.shape) != 3
            or self.shape[2] < CSA_RECORD_BYTES
        ):
            raise ValueError("CSA cache requires at least 640 bytes per KV state")
        if self.shape[1] % 4 or stride_bytes % 512 or offset_bytes % 512:
            raise ValueError("CSA pages require four-token and 512-byte alignment")
        if raw.shape[0] * 4 > torch.iinfo(torch.int32).max:
            raise ValueError("CSA RoPE record indices must fit int32")
        self.page_stride = stride_bytes // 512
        self.row_offset = offset_bytes // 512

    def numel(self):
        return math.prod(self.shape)

    def row_indices(self, slots):
        return (
            self.row_offset
            + slots // self.shape[1] * self.page_stride
            + slots % self.shape[1]
        )

    def rope_indices(self, slots):
        return (
            self.row_offset + slots // self.shape[1] * self.page_stride + self.shape[1]
        ) * 4 + slots % self.shape[1]

    def write(self, slots, values):
        nope_records = self.row_indices(slots.clamp(min=0)).masked_fill(slots < 0, -1)
        rope_records = self.rope_indices(slots.clamp(min=0)).masked_fill(slots < 0, -1)
        assert _write_op is not None
        self.raw.copy_(
            _write_op(self.raw, nope_records.int(), rope_records.int(), values)
        )


def compact_csa_records(nope_records, rope_records):
    """Native CSA masks a valid prefix; reindex may leave holes in its top-k."""
    valid = (nope_records >= 0) & (rope_records >= 0)
    order = jnp.argsort(~valid, axis=-1, stable=True)
    nope = jnp.take_along_axis(jnp.where(valid, nope_records, -1), order, axis=1)
    rope = jnp.take_along_axis(jnp.where(valid, rope_records, -1), order, axis=1)
    padding = (0, -nope.shape[1] % 128)
    return (
        jnp.pad(nope, ((0, 0), padding), constant_values=-1),
        jnp.pad(rope, ((0, 0), padding), constant_values=-1),
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


def csa_attention_jax(
    q: jax.Array,
    swa: jax.Array,
    main: jax.Array,
    swa_records: jax.Array,
    nope_records: jax.Array,
    rope_records: jax.Array,
    sink: jax.Array,
    scale: float,
) -> jax.Array:
    maximum, denominator, numerator = swa_attention_state(q, swa, swa_records, scale)
    if nope_records.shape[1] == 0:
        denominator += jnp.exp(sink.astype(jnp.float32)[None] - maximum)
        return (numerator / denominator[..., None]).astype(q.dtype)
    nope_records, rope_records = compact_csa_records(nope_records, rope_records)
    # The model has already applied the source layer's block table and causal
    # mask. Both regions live in the upstream shared pool, at distinct offsets.
    tokens = q.shape[0]
    return sparse_mla.sparse_ragged_paged_attention(
        q,
        main[:, None, :],
        main[:, None, :],
        nope_records,
        jnp.zeros((1,), jnp.int32),
        jnp.array([0, tokens], jnp.int32),
        jnp.array([0, 0, 1], jnp.int32),
        sink.astype(jnp.float32),
        numerator.astype(q.dtype),
        denominator,
        maximum,
        sm_scale=scale,
        physical_indices=(nope_records, rope_records),
        attention_kernel_batch_size=min(4, max(1, 4096 // nope_records.shape[1])),
        vmem_limit_bytes=32 * 1024 * 1024,
    )


def initialize_csa():
    global _write_op, _attention_op
    if _write_op is not None:
        return
    from jax.sharding import PartitionSpec as P
    from torch_tpu._internal import pallas

    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        get_vllm_model_wrapper_context,
    )

    mesh = get_vllm_model_wrapper_context().mesh
    write = jax.shard_map(
        write_csa_values,
        mesh=mesh,
        in_specs=(P(), P(), P(), P()),
        out_specs=P(),
        check_vma=False,
    )

    def attend(
        q: jax.Array,
        swa: jax.Array,
        main: jax.Array,
        swa_records: jax.Array,
        nope_records: jax.Array,
        rope_records: jax.Array,
        sink: jax.Array,
        scale: float,
    ) -> jax.Array:
        return jax.shard_map(
            functools.partial(csa_attention_jax, scale=scale),
            mesh=mesh,
            in_specs=(P(),) * 7,
            out_specs=P(),
            check_vma=False,
        )(q, swa, main, swa_records, nope_records, rope_records, sink)

    _write_op = pallas.jax_op(
        "pallas::deepseek_v41_pack_csa_fp8",
        write,
        mesh=mesh,
        donate_argnums=(0,),
        input_partition_specs=(P(), P(), P(), P()),
    )
    _write_op.register_fake(lambda pool, *args: torch.empty_like(pool))
    _attention_op = pallas.jax_op(
        "pallas::deepseek_v41_native_csa_fp8",
        attend,
        mesh=mesh,
        input_partition_specs=(P(), P(), P(), P(), P(), P(), P(), None),
    )
    _attention_op.register_fake(lambda q, *args: torch.empty_like(q))


def csa_attention(q, swa, main, swa_records, nope_records, rope_records, sink, scale):
    assert _attention_op is not None
    return _attention_op(
        q,
        swa.raw,
        main.raw,
        swa_records.int(),
        nope_records.int(),
        rope_records.int(),
        sink,
        scale,
    )
