# SPDX-License-Identifier: Apache-2.0
"""Scheduled target-cache input diagnostics, owned independently of inference."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch_tpu._internal import pallas
from torch_tpu._internal.stream.streams import TpuEvent
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.attention import Attention
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec, MLAAttentionSpec
from vllm.v1.worker.utils import extract_layer_index

from vllm_torchtpu.block_major_pool import flat_kernel_ids
from vllm_torchtpu.layers.adapter.attention import PallasAttentionBackendImpl
from vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op import (
    VllmGatedDeltaNetAttention,
)
from vllm_torchtpu.layers.adapter.custom_ops.kv_cache_nan_flags_op import (
    attention_nan_op,
    dense_state_nan_op,
    pooled_state_nan_op,
)
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

logger = init_logger(__name__)


@dataclass
class CacheNaNLayer:
    name: str
    layer_id: int
    inspect: Callable[[AttentionMetadata, torch.Tensor], torch.Tensor]


@dataclass
class KVCacheNaNReadback:
    round_id: int
    req_ids: tuple[str, ...]
    layer_ids: tuple[int, ...]
    flags_cpu: torch.Tensor
    ready_event: TpuEvent

    def consume(self) -> None:
        self.ready_event.synchronize()
        if self.flags_cpu.any().item():
            logger.error(
                "KV cache NaN/Inf detected: role=target round=%d "
                "layer_ids=%s isnonfinite=%s req_ids=%s",
                self.round_id,
                list(self.layer_ids),
                json.dumps(self.flags_cpu.tolist()),
                list(self.req_ids),
            )


class KVCacheNaNDetector:
    def __init__(self, layers: tuple[CacheNaNLayer, ...], device: torch.device):
        self.layers = layers
        self.layer_ids = tuple(sorted({layer.layer_id for layer in layers}))
        self._slots = {layer_id: i for i, layer_id in enumerate(self.layer_ids)}
        self.flags = torch.zeros(len(self.layer_ids), dtype=torch.bool, device=device)
        self.armed_tpu = torch.zeros(1, dtype=torch.bool, device=device)
        self._armed = False
        self._round_id = 0
        self._req_ids: tuple[str, ...] = ()

    def arm(self) -> None:
        self.armed_tpu.fill_(True)
        self._armed = True
        self._round_id = 0

    def begin_round(self, req_ids: list[str]) -> None:
        self._round_id += 1
        self._req_ids = tuple(req_ids)
        self.flags.zero_()

    def record_chunk(self, metadata: dict[str, AttentionMetadata]) -> None:
        for layer in self.layers:
            slot = self._slots[layer.layer_id]
            self.flags[slot].logical_or_(
                layer.inspect(metadata[layer.name], self.armed_tpu)
            )

    def readback(self) -> KVCacheNaNReadback | None:
        if not self._armed:
            return None
        flags_cpu = self.flags.to("cpu", non_blocking=True)
        event = TpuEvent()
        event.record()
        return KVCacheNaNReadback(
            self._round_id, self._req_ids, self.layer_ids, flags_cpu, event
        )


def _attention_inspector(runner, name, module, spec):
    impl = module.impl
    source = module.kv_cache
    region = impl._bm_region
    # Dense bundled attention uses strided per-layer views. Read its owning
    # bundle directly so the diagnostic never materializes a whole KV view.
    bundle_index = runner._kv_cache_bundle_layer_index.get(name)
    if bundle_index is not None:
        source = runner._kv_cache_bundle
        assert source is not None, "Bundled attention must own a KV bundle"
    block_major = region is not None or bundle_index is not None
    seq_along_lane = impl._pool_is_seq_along_lane and spec.head_size != 64
    page_shape = source.shape[2:] if block_major else source.shape[1:]
    page_size = page_shape[-1] if seq_along_lane else page_shape[0]
    with set_current_vllm_config(runner.vllm_config):
        native_shape = module.get_attn_backend().get_kv_cache_shape(
            1,
            page_size,
            spec.num_kv_heads,
            spec.head_size,
            spec.dtype,
        )
    op = attention_nan_op(
        tuple(native_shape),
        spec.num_kv_heads,
        spec.head_size,
        seq_along_lane,
        block_major,
    )

    def inspect(meta, armed):
        cache = module.kv_cache
        tables = meta.block_tables
        if region is not None:
            index, layout = region
            tables = flat_kernel_ids(tables, index, layout.num_pools, layout.split)
        if bundle_index is not None:
            cache = runner._kv_cache_bundle
            tables = flat_kernel_ids(tables, bundle_index, cache.shape[1], 1)
        tables = tables.reshape(meta.seq_lens.shape[0], -1)
        return op(cache, tables, meta.seq_lens, meta.query_start_loc, armed)

    return inspect


def _gdn_inspector(runner, module):
    pooled = len(module.kv_cache) == 1
    if pooled:
        op = pooled_state_nan_op(
            module.conv1d.weight.shape[0],
            module.num_v_heads // module.tp_size,
            module.head_k_dim,
            module.head_v_dim,
            module.conv_kernel_size,
            runner.cache_config.block_size,
            pallas.pallas.TORCH_TO_JAX_DTYPE_MAP[module.get_state_dtype()[1]],
            module._bm_kernel,
        )
    else:
        op = dense_state_nan_op()

    def inspect(meta, armed):
        slots = meta.mamba_state_indices
        if slots is None:
            slots = meta.block_tables.reshape(meta.seq_lens.shape[0], -1)[:, 0]
        if pooled:
            slots = module._bm_state_ids(slots)
            return op(
                module.kv_cache[0],
                slots,
                meta.seq_lens,
                meta.query_start_loc,
                armed,
                meta.mamba_slot_read_offsets,
                meta.mamba_ckpt_indices,
            )
        return op(
            *module.kv_cache,
            slots,
            meta.seq_lens,
            meta.query_start_loc,
            armed,
            meta.mamba_slot_read_offsets,
        )

    return inspect


def install_kv_cache_nan_detector(runner: "TPUModelRunner") -> KVCacheNaNDetector:
    """Validate supported decode geometry once, then bind read-only inspectors."""
    parallel = runner.parallel_config
    if (
        parallel.pipeline_parallel_size != 1
        or parallel.prefill_context_parallel_size != 1
        or parallel.decode_context_parallel_size != 1
        or runner.use_spmd
        or runner.is_pooling_model
    ):
        raise ValueError(
            "KV cache NaN/Inf detection requires a causal LM, PP=PCP=DCP=1 "
            "and non-SPMD execution"
        )
    if getattr(runner, "_two_slot_ckpts", False):
        # The inspectors read the checkpoint a read offset names; two-slot
        # checkpoints keep a read code instead.
        raise ValueError(
            "KV cache NaN/Inf detection does not support TPU_GDN_TWO_SLOT_CKPT"
        )
    if (
        runner.scheduler_config.async_scheduling
        and parallel.tensor_parallel_size > 1
        and runner.vllm_config.kv_transfer_config is None
    ):
        raise ValueError(
            "Async multi-rank KV cache NaN/Inf detection requires all-rank output "
            "consumption through a KV connector"
        )
    config = runner.kv_cache_config
    assert config is not None, "KV cache allocation must precede detector installation"
    context = runner.vllm_config.compilation_config.static_forward_context
    target_modules = set(runner.model.modules())
    layers = []
    for group in config.kv_cache_groups:
        for name in group.layer_names:
            module = context[name]
            if module not in target_modules:
                continue
            spec = group.kv_cache_spec
            if isinstance(spec, FullAttentionSpec) and not spec.dtype.is_floating_point:
                raise ValueError(
                    "KV cache NaN/Inf detection requires floating FA cache storage: "
                    f"layer={name} dtype={spec.dtype}"
                )
            if (
                isinstance(spec, FullAttentionSpec)
                and not isinstance(spec, MLAAttentionSpec)
                and isinstance(module, Attention)
                and isinstance(module.impl, PallasAttentionBackendImpl)
            ):
                inspect = _attention_inspector(runner, name, module, spec)
            elif isinstance(spec, MambaSpec) and isinstance(
                module, VllmGatedDeltaNetAttention
            ):
                inspect = _gdn_inspector(runner, module)
            else:
                raise ValueError(
                    f"KV cache NaN/Inf detection does not support layer {name}: "
                    f"{type(spec).__name__} / {type(module).__name__}"
                )
            layers.append(CacheNaNLayer(name, extract_layer_index(name), inspect))
    if not layers:
        raise ValueError("KV cache NaN/Inf detection found no target cache layers")
    layers.sort(key=lambda layer: (layer.layer_id, layer.name))
    detector = KVCacheNaNDetector(tuple(layers), runner.device)
    logger.info(
        "KV cache NaN/Inf detection installed: role=target layer_ids=%s",
        list(detector.layer_ids),
    )
    return detector
