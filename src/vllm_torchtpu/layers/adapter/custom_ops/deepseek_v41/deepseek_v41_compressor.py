"""V4.1 C1/C2 compression and compressed KV publication."""

import jax
import jax.numpy as jnp
import torch
from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v41 import compressor as compressor_upstream

from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout
from vllm_torchtpu.layers.adapter.normalization import rms_norm
from vllm_torchtpu.layers.adapter.upstream_bindings import device_hooks

from .deepseek_v41_backend import AttentionBackend
from .deepseek_v41_cache import (
    NativeCacheBinding,
    cache_slots,
    read_cache,
    scatter_cache_words,
    token_requests,
    write_cache,
)

_write_op = None


def save_compress_norm(
    kv_score,
    positions,
    state_cache,
    slot_mapping,
    query_start_loc,
    token_to_req_indices,
    norm_weight,
    eps,
    ratio,
    latent,
    *,
    state_layout=None,
):
    """Tensor implementation of the compressor_upstream fused compression operation.

    Read the previous row before publishing ring writes. For a long prefill,
    previous rows come directly from this chunk and only the final ring-sized
    suffix is stored, so writes never race aliases from earlier ring wraps.
    """
    count = kv_score.shape[0]
    if slot_mapping is not None:
        slot_mapping = slot_mapping[:count]
    if token_to_req_indices is not None:
        token_to_req_indices = token_to_req_indices[:count]
    dim = norm_weight.numel()
    values = kv_score[:, :dim].float()
    valid = slot_mapping >= 0
    if ratio == 2:
        capacity = (
            state_layout.page_size if state_layout is not None else state_cache.shape[1]
        )
        cache = state_cache
        rows = torch.arange(kv_score.shape[0], device=kv_score.device)
        requests = token_to_req_indices.long().clamp(max=query_start_loc.numel() - 2)
        starts = query_start_loc[requests]
        ends = query_start_loc[requests + 1]
        previous_slots = (
            slot_mapping.clamp(min=0) // capacity * capacity
            + (positions - 1) % capacity
        )
        previous = torch.where(
            (rows > starts)[:, None],
            kv_score[(rows - 1).clamp(min=0)].float(),
            read_cache(cache, previous_slots, layout=state_layout),
        )
        scores = torch.stack((previous[:, dim:], kv_score[:, dim:].float()), 1)
        pair = torch.stack((previous[:, :dim], values), 1)
        values = (pair * scores.softmax(1)).sum(1)
        # All reads above refer to the old ring. Keep unique physical writes.
        publish = valid & (rows >= ends - capacity)
        write_cache(cache, slot_mapping, kv_score.float(), publish, layout=state_layout)
        valid = valid & (positions % 2 == 1)
    elif ratio != 1:
        raise ValueError(f"V4.1 compression ratio must be 1 or 2, got {ratio}")
    result = rms_norm(values, norm_weight, eps)
    latent.copy_(result.masked_fill(~valid[:, None], 0).to(latent.dtype))


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
    nope_cache: jax.Array,
    rope_cache: jax.Array,
    records: jax.Array,
    values: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """A logical compressed slot names matching rows in two native arrays."""
    nope, rope = pack_csa_values(values)
    updated_nope = scatter_cache_words(nope_cache.reshape(-1, 128), records, nope)
    updated_rope = scatter_cache_words(rope_cache.reshape(-1, 128), records, rope)
    return updated_nope.reshape(nope_cache.shape), updated_rope.reshape(
        rope_cache.shape
    )


def write_compressed_cache(caches, slots, values, valid, num_states):
    nope, rope = caches
    records = (
        (slots // num_states * nope.shape[1] + slots % num_states)
        .masked_fill(~valid, -1)
        .int()
    )
    assert _write_op is not None
    updated_nope, updated_rope = _write_op(nope, rope, records, values)
    nope.copy_(updated_nope)
    rope.copy_(updated_rope)


def initialize_compressor_ops():
    global _write_op
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
        in_specs=(P(),) * 4,
        out_specs=(P(), P()),
        check_vma=False,
    )
    _write_op = pallas.jax_op(
        "pallas::deepseek_v41_write_compressed_kv",
        write,
        mesh=mesh,
        donate_argnums=(0, 1),
        input_partition_specs=(P(),) * 4,
    )
    _write_op.register_fake(
        lambda nope, rope, *args: (torch.empty_like(nope), torch.empty_like(rope))
    )


class CompressorStateCache(
    NativeCacheBinding, compressor_upstream.CompressorStateCache
):
    """Upstream state ownership with a shared native cache binding."""

    def get_attn_backend(self):
        return AttentionBackend


class DeepseekCompressor(compressor_upstream.DeepseekCompressor):
    """Inherit weights and state ownership; consume the shared TPU metadata."""

    def __init__(self, *args, **kwargs):
        with device_hooks(
            compressor_upstream, CompressorStateCache=CompressorStateCache
        ):
            super().__init__(*args, **kwargs)

    def project(self, hidden_states):
        from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR

        layer = self.fused_wkv_wgate
        weight = layer.weight.float()
        if not getattr(layer, WEIGHT_FLIPPED_ATTR, False):
            weight = weight.t()
        # Upstream accumulates the unquantized KV/score projection in FP32.
        # Keep that precision through pooling and normalization.
        return torch.mm(hidden_states.float(), weight)

    def forward(self, kv_score, positions):
        metadata = get_forward_context().attn_metadata
        if not isinstance(metadata, dict):
            return None
        if self.state_cache is None:
            state_cache = query_start_loc = requests = None
            state_layout = None
            source = self._static_forward_context[self.k_cache_prefix]
            slots = cache_slots(
                metadata[self.k_cache_prefix],
                positions,
                rows_per_page=source.cache_num_states,
            )
        else:
            state_metadata = metadata[self.state_cache.prefix]
            binding = (
                self.state_cache.native() if self.state_cache.has_native_cache else None
            )
            state_cache = self.state_cache.kv_cache
            state_layout = binding.layout if binding is not None else None
            slots = cache_slots(
                state_metadata,
                positions,
                ring=True,
                rows_per_page=(
                    state_layout.page_size
                    if state_layout is not None
                    else state_cache.shape[1]
                ),
            )
            query_start_loc = state_metadata.query_start_loc
            requests = token_requests(state_metadata, positions)
        latent = torch.empty(
            (kv_score.shape[0], self.head_dim),
            dtype=torch.bfloat16,
            device=kv_score.device,
        )
        save_compress_norm(
            kv_score,
            positions,
            state_cache,
            slots,
            query_start_loc,
            requests,
            self.norm.weight,
            self.rms_norm_eps,
            self.compress_ratio,
            latent,
            state_layout=state_layout,
        )
        return latent
