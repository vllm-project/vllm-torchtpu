"""V4.1 cache ownership and index selection feeding the native V4 FP8 CSA."""

import functools
import math

import jax
import jax.numpy as jnp
import torch

from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout
from vllm_torchtpu.kernels.deepseek_v4.core_attention import sparse_mla

from .cache import scatter_cache_words

CSA_RECORD_BYTES = 640  # 448 FP8 values, 64 E8M0 scales, 64 BF16 RoPE values.

_write_op = None
_attention_op = None


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


def write_csa_values(pool, nope_records, rope_records, values):
    nope, rope = pack_csa_values(values)
    pool = scatter_cache_words(pool, nope_records, nope)
    return scatter_cache_words(pool, rope_records, rope)


class PackedCSACache:
    """Two packed regions per upstream-owned page in one shared word-row pool.

    A page with T compressed tokens stores T NoPE rows, followed by T/4 RoPE
    rows. Page stride and offset still come from vLLM, including overlays with
    indexer/state/SWA allocations. No strided whole-cache tensor is copied.
    """

    def __init__(self, raw, layout):
        self.raw = raw
        self.shape, self.dtype, stride_bytes, offset_bytes = layout
        if raw.dtype != torch.int32 or raw.ndim != 2 or raw.shape[1] != 128:
            raise ValueError("CSA cache requires int32 word rows with 128 lanes")
        if self.dtype != torch.uint8 or self.shape[2:] != (CSA_RECORD_BYTES,):
            raise ValueError("CSA cache requires 640 bytes per compressed token")
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

    def __getitem__(self, indices):
        if self.raw.device.type != "cpu":
            raise RuntimeError("Packed CSA reads must use the native CSA kernel")
        blocks, rows = indices
        slots = blocks * self.shape[1] + rows
        nope = self.raw[self.row_indices(slots).long()].long()
        slab = torch.stack([(nope >> (8 * i)) & 255 for i in range(4)], dim=-2)
        record = slab.to(torch.uint8).flatten(-2)
        fp8 = record[..., :448].contiguous().view(torch.float8_e4m3fn).float()
        scales = torch.pow(2.0, record[..., 448:].int() - 127)
        nope = (fp8.unflatten(-1, (7, 64)) * scales[..., None, :]).flatten(-2)
        rope = self.raw.reshape(-1, 32)[self.rope_indices(slots).long()].long()
        bits = torch.cat((rope & 65535, (rope >> 16) & 65535), dim=-1)
        rope = bits.to(torch.int16).view(torch.bfloat16)
        return torch.cat((nope.to(torch.bfloat16), rope), dim=-1)

    def write(self, slots, values):
        nope_records = self.row_indices(slots.clamp(min=0)).masked_fill(slots < 0, -1)
        rope_records = self.rope_indices(slots.clamp(min=0)).masked_fill(slots < 0, -1)
        if self.raw.device.type == "cpu":
            valid = slots >= 0
            values = values[valid].to(torch.bfloat16)
            groups = values[:, :448].float().reshape(-1, 7, 64)
            scales = torch.pow(
                2.0, torch.ceil(torch.log2(groups.abs().amax(1).clamp_min(1e-4) / 448))
            )
            fp8 = (groups / scales[:, None]).to(torch.float8_e4m3fn)
            scale_bytes = (scales.view(torch.int32) >> 23).to(torch.uint8)
            slabs = (
                torch.cat((fp8.view(torch.uint8).reshape(-1, 448), scale_bytes), dim=1)
                .reshape(-1, 4, 128)
                .int()
            )
            words = slabs[:, 0] | (slabs[:, 1] << 8) | (slabs[:, 2] << 16)
            words |= slabs[:, 3] << 24
            self.raw.index_copy_(0, nope_records[valid].long(), words)
            bits = values[:, 448:].contiguous().view(torch.int16).int() & 65535
            rope = bits[:, :32] | (bits[:, 32:] << 16)
            self.raw.reshape(-1, 32).index_copy_(0, rope_records[valid].long(), rope)
            return
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


def csa_attention_jax(
    q, swa, main, swa_records, nope_records, rope_records, sink, scale
):
    from .attention_kernels import paged_attention_state_jax

    maximum, denominator, numerator = paged_attention_state_jax(
        q, swa, swa_records, scale
    )
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

    def attend(q, swa, main, swa_records, nope_records, rope_records, sink, scale):
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
