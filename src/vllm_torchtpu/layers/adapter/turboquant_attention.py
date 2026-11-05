# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TurboQuant KV-cache attention backend for TPU (fused Pallas RPA v3 kernel).

Selected for `--kv-cache-dtype turboquant_{k8v4,4bit_nc,k3v4_nc,3bit_nc}`.
Each layer's KV cache is `u8[num_blocks, num_kv_heads, page_words_128, 128]`
(`LBNHC`), updated and VREG-dequantized inside
`vllm_torchtpu.kernels.experimental.turboquant_rpa_v3`.
"""

import dataclasses
import functools
import math

import jax
import numpy as np
import torch
from jax.experimental import shard_map
from jax.sharding import PartitionSpec as P
from vllm.config import VllmConfig, get_current_vllm_config_or_none
from vllm.v1.attention.backend import MultipleOf
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheSpec, KVQuantMode
from vllm.v1.kv_cache_layout import KVCacheLayout as VllmKVCacheLayout

from vllm_torchtpu.kernels.experimental.batched_rpa import (
    configs as batched_rpa_configs,
)
from vllm_torchtpu.kernels.experimental.turboquant_rpa_v3.kernel import (
    ragged_paged_attention,
)
from vllm_torchtpu.layers.adapter.attention import (
    PallasAttentionBackend,
    PallasAttentionBackendImpl,
)
from vllm_torchtpu.layers.core import turboquant as tq
from vllm_torchtpu.layers.core.turboquant import (
    TURBOQUANT_KV_CACHE_DTYPES,
    is_turboquant_kv_cache_dtype,
)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_MODEL_AXIS_NAME = "model"

# Active preset in this process: `get_kv_cache_shape` and friends may receive
# the storage dtype (`torch.uint8`) rather than the preset string.
_active_preset: str | None = None


def set_active_turboquant_preset(preset: str) -> None:
    global _active_preset
    preset = preset.lower().strip()
    if preset not in TURBOQUANT_KV_CACHE_DTYPES:
        raise ValueError(
            f"Unknown TurboQuant KV cache dtype {preset!r}; expected one of "
            f"{TURBOQUANT_KV_CACHE_DTYPES}."
        )
    _active_preset = preset


def _resolve_preset(
    cache_dtype: str | torch.dtype | KVQuantMode | None = None,
) -> str | None:
    if isinstance(cache_dtype, KVQuantMode):
        return cache_dtype.name.lower() if cache_dtype.is_turboquant else None
    if isinstance(cache_dtype, torch.dtype):
        if cache_dtype is not torch.uint8:
            return None
    elif isinstance(cache_dtype, str):
        if is_turboquant_kv_cache_dtype(cache_dtype):
            return cache_dtype.lower().strip()
        if cache_dtype.lower().strip() != "auto":
            return None
    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is not None:
        cfg_dtype = getattr(
            getattr(vllm_config, "cache_config", None), "cache_dtype", None
        )
        return (
            cfg_dtype.lower().strip()
            if isinstance(cfg_dtype, str) and is_turboquant_kv_cache_dtype(cfg_dtype)
            else None
        )
    return _active_preset


def _supported_block_sizes(preset: str | None) -> tuple[int, ...]:
    return (16, 32, 64) if preset == "turboquant_k8v4" else (16, 64)


@functools.cache
def get_tq_layout(preset: str, head_dim: int) -> tq.TQLayout:
    from vllm.model_executor.layers.quantization.turboquant.config import (
        TurboQuantConfig,
    )

    return tq.TQLayout.from_tq_config(
        TurboQuantConfig.from_cache_dtype(preset, head_dim)
    )


@functools.cache
def get_tq_tables(preset: str, head_dim: int) -> tq.TQTables:
    from vllm.model_executor.layers.quantization.turboquant.centroids import (
        get_centroids,
    )

    layout = get_tq_layout(preset, head_dim)
    bits = layout.value_bits if layout.is_key_fp8 else layout.key_bits
    with torch.device("cpu"):
        centroids = get_centroids(head_dim, bits).cpu().numpy().astype(np.float32)
    return tq.make_tables(layout, centroids)


def validate_turboquant_head_dim(head_dim: int) -> None:
    # Walsh-Hadamard rotation requires a power of two; RPA v3 VREG unpacking
    # folds head_dim in 128-lane chunks.
    if head_dim <= 0 or head_dim % 128 != 0 or head_dim & (head_dim - 1):
        raise NotImplementedError(
            "TurboQuant KV cache on TPU currently supports head_dim in "
            f"{{128, 256, 512, ...}}; got head_dim={head_dim}."
        )


def _tq_rpa_kernel_entry(
    kv_cache: jax.Array,
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    request_distribution: jax.Array,
    sinks: jax.Array | None,
    q_scale: float | None,
    k_scale: float | None,
    v_scale: float | None,
    *,
    mesh: jax.sharding.Mesh,
    sliding_window: int | None,
    sm_scale: float | None = None,
    soft_cap: float | None = None,
    skip_kv_update: bool,
    use_causal_mask: bool = True,
    tq_preset: str,
) -> tuple[jax.Array, jax.Array]:
    """Same signature as `_pallas_rpa_kernel_default`, plus `tq_preset`."""
    if sinks is not None:
        raise NotImplementedError("TurboQuant KV cache does not support sinks.")
    if not use_causal_mask:
        raise NotImplementedError("TurboQuant KV cache requires a causal mask.")
    if q_scale is not None or k_scale is not None or v_scale is not None:
        raise NotImplementedError("TurboQuant KV cache takes no q/k/v scales.")

    head_dim = query.shape[-1]
    fn = functools.partial(
        tq.tq_ragged_paged_attention,
        layout=get_tq_layout(tq_preset, head_dim),
        tables=get_tq_tables(tq_preset, head_dim),
        rpa_fn=ragged_paged_attention,
        sm_scale=head_dim**-0.5 if sm_scale is None else sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        skip_kv_update=skip_kv_update,
    )
    qkv_spec = P(None, _MODEL_AXIS_NAME, None)
    cache_spec = P(None, _MODEL_AXIS_NAME, None, None)
    return shard_map.shard_map(
        fn,
        mesh=mesh,
        in_specs=(qkv_spec, qkv_spec, qkv_spec, cache_spec, P(), P(), P(), P()),
        out_specs=(cache_spec, qkv_spec),
        check_rep=False,
    )(
        query,
        key,
        value,
        kv_cache,
        seq_lens,
        block_tables,
        query_start_loc,
        request_distribution,
    )


@register_backend(AttentionBackendEnum.TURBOQUANT)
class PallasTurboQuantAttentionBackend(PallasAttentionBackend):
    supported_kv_cache_dtypes = list(TURBOQUANT_KV_CACHE_DTYPES)  # type: ignore[assignment]

    @staticmethod
    def get_name() -> str:
        return "TURBOQUANT"

    @staticmethod
    def get_impl_cls() -> type["PallasTurboQuantAttentionBackendImpl"]:
        return PallasTurboQuantAttentionBackendImpl

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype) -> bool:
        return is_turboquant_kv_cache_dtype(kv_cache_dtype)

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str | torch.dtype | KVQuantMode = "auto",
    ) -> tuple[int, ...]:
        preset = _resolve_preset(cache_dtype_str)
        if preset is None:
            return (num_blocks, num_kv_heads, block_size, 128)
        layout = get_tq_layout(preset, head_size)
        page_words = layout.page_words_128(block_size)
        assert layout.page_size_from_words_128(page_words) == block_size
        return (num_blocks, num_kv_heads, page_words, 128)

    @classmethod
    def get_kv_cache_page_size_bytes(
        cls,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str | torch.dtype | KVQuantMode = "auto",
    ) -> int:
        return math.prod(
            cls.get_kv_cache_shape(
                1, block_size, num_kv_heads, head_size, cache_dtype_str
            )
        )

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        page_bytes = cls.get_kv_cache_page_size_bytes(
            spec.block_size,
            spec.num_kv_heads,
            spec.head_size,
            spec.kv_quant_mode if spec.kv_quant_mode.is_turboquant else spec.dtype,
        )
        return dataclasses.replace(
            spec, num_head_slots=1, state_content_bytes=page_bytes // spec.block_size
        )

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[VllmKVCacheLayout, ...]:
        return (VllmKVCacheLayout.LBNHC,)

    @staticmethod
    def get_supported_kernel_block_sizes(
        kv_cache_spec: KVCacheSpec | None = None,
    ) -> list[int | MultipleOf]:
        preset = _resolve_preset(
            kv_cache_spec.kv_quant_mode
            if isinstance(kv_cache_spec, AttentionSpec)
            else None
        )
        return list(_supported_block_sizes(preset))

    @staticmethod
    def get_page_size(vllm_config: VllmConfig) -> int:
        page_size = min(PallasAttentionBackend.get_page_size(vllm_config), 64)
        preset = _resolve_preset(
            getattr(getattr(vllm_config, "cache_config", None), "cache_dtype", None)
        )
        return page_size if page_size in _supported_block_sizes(preset) else 16

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        return (1, 0, 2, 3, 4) if include_num_layers_dimension else (0, 1, 2, 3)


class PallasTurboQuantAttentionBackendImpl(PallasAttentionBackendImpl):
    """Reuses the Pallas impl's forward, op registration and cache writeback;
    only the kernel entry differs."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        kv_cache_dtype = kwargs.get("kv_cache_dtype") or args[6]
        if not is_turboquant_kv_cache_dtype(kv_cache_dtype):
            raise ValueError(
                f"{type(self).__name__} requires a TurboQuant kv_cache_dtype, "
                f"got {kv_cache_dtype!r}."
            )
        if self.sinks is not None:
            raise NotImplementedError("TurboQuant KV cache does not support sinks.")
        validate_turboquant_head_dim(self.head_size)
        if self._bm_kernel:
            raise NotImplementedError(
                "TurboQuant KV cache does not support VLLM_TPU_BLOCK_MAJOR_KV=1."
            )
        if self.kv_layout is not batched_rpa_configs.KVLayout.HEAD_ALONG_SUBLANE:
            raise NotImplementedError(
                "TurboQuant KV cache requires VLLM_KV_CACHE_LAYOUT=NHD."
            )
        preset = str(kv_cache_dtype).lower().strip()
        set_active_turboquant_preset(preset)
        get_tq_tables(preset, self.head_size)
        self.tq_preset = preset
        self._kernel_entry = functools.partial(  # type: ignore[misc]
            _tq_rpa_kernel_entry, tq_preset=preset
        )
        self._kernel_op_prefix = f"pallas::tq_rpa_kernel_{preset}"  # type: ignore[misc]


def check_turboquant_config(vllm_config: VllmConfig) -> None:
    """Platform-level validation, called from `TpuPlatform.check_and_update_config`."""
    cache_config = vllm_config.cache_config
    preset = cache_config.cache_dtype.lower().strip()
    set_active_turboquant_preset(preset)
    supported_block_sizes = _supported_block_sizes(preset)
    if (
        cache_config.user_specified_block_size
        and cache_config.block_size not in supported_block_sizes
    ):
        raise ValueError(
            f"TurboQuant KV cache ({preset}) supports block_size in "
            f"{supported_block_sizes}, got block_size={cache_config.block_size}."
        )

    model_config = vllm_config.model_config
    if model_config is not None:
        if model_config.use_mla:
            raise NotImplementedError(
                "TurboQuant KV cache does not support MLA models."
            )
        validate_turboquant_head_dim(model_config.get_head_size())
        if not cache_config.user_specified_block_size:
            cache_config.block_size = (  # type: ignore[assignment]
                PallasTurboQuantAttentionBackend.get_page_size(vllm_config)
            )
        if model_config.is_hybrid:
            if cache_config.enable_prefix_caching:
                logger.warning_once(
                    "TurboQuant KV cache on hybrid Mamba+Attention models uses the "
                    "compact Mamba state layout rather than the unified KV pool; "
                    "disabling prefix caching (enable_prefix_caching=False, "
                    "mamba_cache_mode='none')."
                )
                cache_config.enable_prefix_caching = False
            if cache_config.mamba_cache_mode == "align":
                cache_config.mamba_cache_mode = "none"
            cache_config.mamba_block_size = model_config.max_model_len

    parallel = vllm_config.parallel_config
    if parallel.prefill_context_parallel_size > 1:
        raise NotImplementedError("TurboQuant KV cache does not support PCP.")
    if parallel.decode_context_parallel_size > 1:
        raise NotImplementedError("TurboQuant KV cache does not support DCP.")

    if cache_config.kv_cache_dtype_skip_layers:
        logger.warning_once(
            "TurboQuant on TPU does not support kv_cache_dtype_skip_layers yet; "
            "quantizing every attention layer (ignoring %s). Expect lower "
            "accuracy than upstream, especially for turboquant_k3v4_nc and "
            "turboquant_3bit_nc.",
            tuple(cache_config.kv_cache_dtype_skip_layers),
        )
        cache_config.kv_cache_dtype_skip_layers = []

    logger.info_once(
        "TurboQuant KV cache (%s) on TPU uses fused Pallas RPA v3 VREG "
        "dequantization with per-step quantized page updates.",
        preset,
    )
