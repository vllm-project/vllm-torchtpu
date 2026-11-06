# Copyright 2026 Google LLC
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
"""Model-local QSA, BF16 checkpoint loading and explicit typed cache binding."""

import math
from collections.abc import Iterable
from dataclasses import dataclass, replace
from functools import partial
from typing import cast

import torch
from torch import nn
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention.attention import set_default_quant_scales
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
)
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.model_executor.parameter import BasevLLMParameter
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.attention.backend import MultipleOf
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheSpec,
    MLAAttentionSpec,
    get_kv_quant_mode,
)

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.topk_backend import TopKBackend
from vllm_torchtpu.layers.adapter.attention import (
    TPU_HEAD_SIZE_ALIGNMENT,
    PallasAttentionBackend,
)
from vllm_torchtpu.layers.adapter.custom_ops.qsa_op import build_compact_pooled_qsa_op
from vllm_torchtpu.layers.adapter.quantization.unquantized import (
    VllmUnquantizedLinearMethod,
)
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata

from .config import require_unified_pool
from .qsa_geometry import rope_axis_assignment, validate_qsa_geometry


@dataclass(frozen=True)
class QSAMetadata:
    """Logical query geometry; cache owners and page tables are per-layer metadata."""

    query_start_loc: torch.Tensor
    context_lens: torch.Tensor
    decode_cu: torch.Tensor | None = None


def make_rope_cache(max_positions: int, *, rotary_dim: int, theta: float):
    """BF16 cos/sin rows, padded to TPU's 128-column gather alignment."""
    # Match upstream RoPE frequencies on CPU before moving the BF16 cache.
    with torch.device("cpu"):
        rotary = RotaryEmbedding(
            head_size=rotary_dim,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_positions,
            base=theta,
            is_neox_style=True,
            dtype=torch.bfloat16,
        )
        cache = rotary.cos_sin_cache
        return torch.nn.functional.pad(cache, (0, (-rotary_dim) % 128))


QSA_BLOCK_SIZE = 128


def get_qsa_cache_specs(config, tp_size, cache_dtype):
    """Pure planner contract; model admission validates this geometry once."""
    fp8 = cache_dtype in ("fp8", "fp8_e4m3")
    head_dim = int(config.head_dim)
    index_dim = int(config.indexer_head_dim)
    ratio = int(config.indexer_compress_ratio)
    ring_rows = (ratio + 7) // 8 * 8
    # 8192 B/slot covers the observed T(2,128)/T(8,128) tag layouts for
    # 2..65 slots. Retain that placement allowance and budget larger raw
    # payloads explicitly; actual TPU allocation remains independently checked.
    axes = rope_axis_assignment(
        config.rope_parameters, int(head_dim * config.partial_rotary_factor)
    )
    tag_fields = 4 if axes is not None else 2
    ring_bytes = max(
        8192, ring_rows * (((index_dim + 127) // 128 * 128) * 2 + 4 * tag_fields)
    )
    main = FullAttentionSpec(
        block_size=QSA_BLOCK_SIZE,
        num_kv_heads=max(int(config.num_key_value_heads) // tp_size, 1),
        head_size=head_dim,
        dtype=torch.float8_e4m3fn if fp8 else torch.bfloat16,
        kv_quant_mode=get_kv_quant_mode(cache_dtype),
    )
    auxiliary = {
        "compressed_k": MLAAttentionSpec(
            block_size=QSA_BLOCK_SIZE,
            num_kv_heads=1,
            head_size=index_dim,
            state_content_bytes=((index_dim + 127) // 128 * 128) * 2,
            dtype=torch.bfloat16,
            tokens_per_state=ratio,
        ),
        "raw_ring": CircularBufferSpec(
            block_size=ratio,
            num_kv_heads=1,
            head_size=index_dim + 8 // 2,
            head_size_v=0,
            dtype=torch.bfloat16,
            page_size_padded=ring_bytes,
        ),
    }
    packing = 4 if fp8 else 2
    main = replace(
        main,
        num_head_slots=(2 * main.num_kv_heads + packing - 1) // packing,
        state_content_bytes=4 * ((head_dim + 127) // 128 * 128),
    )
    return main, auxiliary


class Qwen4ExpPooledQSABackend(PallasAttentionBackend):
    """Native pool geometry with compression groups as virtual RPA pages."""

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str | torch.dtype = "auto",
    ) -> tuple[int, ...]:
        if isinstance(cache_dtype_str, torch.dtype):
            dtype = cache_dtype_str
        elif cache_dtype_str in ("auto", "bfloat16"):
            dtype = torch.bfloat16
        elif cache_dtype_str in ("fp8", "fp8_e4m3"):
            dtype = torch.float8_e4m3fn
        else:
            raise ValueError("QSA pool requires BF16 or FP8 E4M3 cache storage")
        packing = torch.int32.itemsize // dtype.itemsize
        dimension = (
            math.ceil(head_size / TPU_HEAD_SIZE_ALIGNMENT) * TPU_HEAD_SIZE_ALIGNMENT
        )
        return (
            num_blocks,
            block_size,
            math.ceil(2 * num_kv_heads / packing),
            packing,
            dimension,
        )

    @staticmethod
    def get_name():
        return "QWEN4_EXP_TPU_POOLED_QSA"

    @staticmethod
    def get_supported_kernel_block_sizes(
        kv_cache_spec: KVCacheSpec | None = None,
    ) -> list[MultipleOf | int]:
        return [MultipleOf(QSA_BLOCK_SIZE)]

    @staticmethod
    def get_min_page_size(vllm_config):
        from .pool_layout import pooled_cache_specs

        return pooled_cache_specs(vllm_config)[0].block_size

    @classmethod
    def supported_kv_cache_layouts(cls):
        from vllm.v1.kv_cache_interface import KVCacheLayout

        return (KVCacheLayout.BLHNC,)


class _Indexer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.q_layernorm = GemmaRMSNorm(
            config.indexer_head_dim, eps=config.rms_norm_eps
        )
        self.k_layernorm = GemmaRMSNorm(
            config.indexer_head_dim, eps=config.rms_norm_eps
        )


_INPUT_PROJECTION_NAMES = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "indexer.index_qk_proj.weight",
)


class Qwen4ExpQSA(nn.Module, AttentionLayerBase):
    """Config-derived gated attention with a fully replicated MQA indexer.

    Checkpoint names are relative to ``...self_attn.``. No quant_config enters
    these BF16 projections. Both cache stores stay owned by the native allocator;
    the custom op donates state then copy_ makes writes visible to that owner.
    Profiling/capture uses explicitly bound scratch owners and shared metadata.
    """

    supports_dcp = False
    _k_scale: torch.Tensor
    _v_scale: torch.Tensor
    _k_scale_cpu: torch.Tensor
    _v_scale_cpu: torch.Tensor
    _k_scale_float: float
    _v_scale_float: float

    def __init__(
        self,
        *,
        config,
        tp_rank: int,
        tp_size: int,
        max_model_len: int,
        device=None,
        cache_dtype: str = "auto",
        prefix: str = "",
        topk_backend: TopKBackend | None = None,
        topk_backend_key: str | None = None,
    ):
        super().__init__()
        validate_qsa_geometry(config, tp_size=tp_size, cache_dtype=cache_dtype)
        if not 0 <= tp_rank < tp_size:
            raise ValueError("QSA requires an in-range TP rank")
        self.tp_rank, self.tp_size, self.prefix = tp_rank, tp_size, prefix
        require_unified_pool()
        self.topk_backend = topk_backend
        self.topk_backend_key = topk_backend_key
        self.fp8_kv = cache_dtype in ("fp8", "fp8_e4m3")
        self.cache_dtype = cache_dtype
        self.config = config
        self.hidden_size = int(config.hidden_size)
        self.head_dim = int(config.head_dim)
        self.total_num_heads = int(config.num_attention_heads)
        self.total_num_kv_heads = int(config.num_key_value_heads)
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = max(self.total_num_kv_heads // tp_size, 1)
        self.index_head_dim = int(config.indexer_head_dim)
        self.index_n_heads = int(config.indexer_n_heads)
        self.compress_ratio = int(config.indexer_compress_ratio)
        self.token_topk = int(config.indexer_budget)
        self.max_selected_tokens = self.token_topk + self.compress_ratio - 1
        self.input_sizes = (
            2 * self.num_heads * self.head_dim,
            self.num_kv_heads * self.head_dim,
            self.num_kv_heads * self.head_dim,
            (self.index_n_heads + 1) * self.index_head_dim,
        )
        with torch.device(device or "cpu"), set_default_torch_dtype(torch.bfloat16):
            # Keep Q independent so its MXU work can overlap asynchronous
            # index selection. Only the small K/V/index projections are packed.
            self.q_proj = ReplicatedLinear(
                self.hidden_size,
                self.input_sizes[0],
                bias=False,
                params_dtype=torch.bfloat16,
                quant_config=None,
                prefix=prefix + ".q_proj",
                return_bias=False,
                disable_tp=True,
            )
            self.input_proj = MergedColumnParallelLinear(
                self.hidden_size,
                list(self.input_sizes[1:]),
                bias=False,
                params_dtype=torch.bfloat16,
                quant_config=None,
                prefix=prefix + ".input_proj",
                return_bias=False,
                disable_tp=True,
            )
            # Upstream RowParallelLinear cannot take explicit TP geometry.
            # Reuse its constituent mechanisms: local unquantized linear and
            # the same TP reduction; loader slices columns exactly once.
            self.o_proj = ReplicatedLinear(
                self.num_heads * self.head_dim,
                self.hidden_size,
                bias=False,
                params_dtype=torch.bfloat16,
                quant_config=None,
                prefix=prefix + ".o_proj",
                return_bias=False,
                disable_tp=True,
            )
            self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.indexer = _Indexer(config)
        self.register_buffer(
            "rope_cache",
            make_rope_cache(
                max_model_len,
                rotary_dim=int(self.head_dim * self.config.partial_rotary_factor),
                theta=float(config.rope_parameters["rope_theta"]),
            ).to(device=device),
            persistent=False,
        )
        with torch.device(device or "cpu"):
            set_default_quant_scales(self, register_buffer=True)
        self.cache_owners: tuple[torch.Tensor, ...] = ()
        self.register_buffer("output_weight_tiles", None, persistent=False)
        self.register_buffer("k_norm_multiplier", None, persistent=False)
        self._loaded: set[str] = set()
        self.rope_axes = rope_axis_assignment(
            config.rope_parameters,
            int(self.head_dim * self.config.partial_rotary_factor),
        )
        self._op = partial(
            build_compact_pooled_qsa_op(
                topk_backend=self.topk_backend,
                topk_backend_key=self.topk_backend_key,
                rope_axes=self.rope_axes,
            ),
            compress_ratio=self.compress_ratio,
            token_topk=self.token_topk,
            norm_eps=float(self.q_norm.variance_epsilon),
            rotary_dim=int(self.head_dim * self.config.partial_rotary_factor),
        )
        self._projected_op = None

    def get_attn_backend(self):
        return Qwen4ExpPooledQSABackend

    def bind_pool(self, main_pool, index_pool):
        self.cache_owners = (main_pool, index_pool)

    def get_kv_cache_spec(self, vllm_config):
        return get_qsa_cache_specs(self.config, self.tp_size, self.cache_dtype)[0]

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        parameters = dict(self.named_parameters())
        global_shapes = {
            "q_proj.weight": (
                2 * self.total_num_heads * self.head_dim,
                self.hidden_size,
            ),
            "k_proj.weight": (
                self.total_num_kv_heads * self.head_dim,
                self.hidden_size,
            ),
            "v_proj.weight": (
                self.total_num_kv_heads * self.head_dim,
                self.hidden_size,
            ),
            "o_proj.weight": (self.hidden_size, self.total_num_heads * self.head_dim),
            "q_norm.weight": (self.head_dim,),
            "k_norm.weight": (self.head_dim,),
            "indexer.index_qk_proj.weight": (
                (self.index_n_heads + 1) * self.index_head_dim,
                self.hidden_size,
            ),
            "indexer.q_layernorm.weight": (self.index_head_dim,),
            "indexer.k_layernorm.weight": (self.index_head_dim,),
        }
        loaded = set()
        for name, weight in weights:
            if name in ("_k_scale", "_v_scale"):
                target = name
                if weight.numel() != 1:
                    raise ValueError(f"QSA {name} requires one per-tensor scale")
                value = float(weight.detach().float().cpu().item())
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"QSA {name} requires a finite positive scale")
                if target in self._loaded or target in loaded:
                    raise ValueError(f"Duplicate QSA checkpoint scale: {target}")
                # Keep the serialized buffer and host scalar consistent. The
                # compiled kernel specializes on this immutable layer scale.
                if target == "_k_scale":
                    self._k_scale.fill_(value)
                    self._k_scale_cpu.fill_(value)
                    self._k_scale_float = value
                else:
                    self._v_scale.fill_(value)
                    self._v_scale_cpu.fill_(value)
                    self._v_scale_float = value
                loaded.add(target)
                continue
            if (
                name not in global_shapes
                or weight.shape != global_shapes[name]
                or weight.dtype != torch.bfloat16
            ):
                raise ValueError(f"Invalid BF16 QSA checkpoint tensor: {name}")
            with torch.no_grad():
                if name in _INPUT_PROJECTION_NAMES:
                    shard = _INPUT_PROJECTION_NAMES.index(name)
                    width = self.input_sizes[shard]
                    if shard == 0:
                        rank = self.tp_rank
                    elif shard in (1, 2):
                        rank = self.tp_rank // max(
                            self.tp_size // self.total_num_kv_heads, 1
                        )
                    else:
                        rank = 0
                    local_weight = weight.narrow(0, rank * width, width)
                    if shard == 0:
                        self.q_proj.weight_loader(self.q_proj.weight, local_weight)
                    else:
                        parameter = self.input_proj.weight
                        cast(BasevLLMParameter, parameter).weight_loader(
                            parameter, local_weight, shard - 1
                        )
                elif name == "o_proj.weight":
                    parameter = parameters[name]
                    width = self.num_heads * self.head_dim
                    self.o_proj.weight_loader(
                        parameter,
                        weight[:, self.tp_rank * width : (self.tp_rank + 1) * width],
                    )
                else:
                    parameters[name].copy_(weight)
            loaded.add(name)
        self._loaded.update(loaded)
        return loaded

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        if self.k_norm_multiplier is not None:
            # Rebuild after dtype/device conversion; never retain a rounded BF16 scale.
            self.k_norm_multiplier = 1.0 + self.k_norm.weight.detach().float()
        return result

    def finalize_weights(self, *, block_size: int | None = None):
        expected = set(dict(self.named_parameters())) - {"input_proj.weight"}
        expected.update(_INPUT_PROJECTION_NAMES)
        missing = expected - self._loaded
        if missing:
            raise ValueError(f"Missing QSA checkpoint tensors: {sorted(missing)}")
        self.k_norm_multiplier = 1.0 + self.k_norm.weight.detach().float()
        # Shard checkpoint columns before preparing the existing K-major BF16
        # linear path. Its idempotent hook also handles the loader's later pass.
        self.o_proj.quant_method = VllmUnquantizedLinearMethod()
        self.o_proj.quant_method.process_weights_after_loading(self.o_proj)
        from vllm_torchtpu.kernels.qwen4_exp.tuned_block_sizes import (
            output_projection_tile,
        )

        weight = cast(torch.Tensor, self.o_proj.weight)
        tile = (
            output_projection_tile(tuple(weight.shape))
            if self._supports_projected_output()
            else None
        )
        if tile is None:
            self.output_weight_tiles = None
            self._projected_op = None
        if tile is not None and self.output_weight_tiles is None:
            self.output_weight_tiles = (
                torch.nn.functional.pad(weight.detach(), (0, -weight.shape[1] % tile))
                .reshape(weight.shape[0], math.ceil(weight.shape[1] / tile), tile)
                .permute(1, 0, 2)
                .contiguous()
            )

        if self.output_weight_tiles is not None and self._projected_op is None:
            self._projected_op = partial(
                build_compact_pooled_qsa_op(
                    topk_backend=self.topk_backend,
                    topk_backend_key=self.topk_backend_key,
                    project_output=True,
                    rope_axes=self.rope_axes,
                ),
                compress_ratio=self.compress_ratio,
                token_topk=self.token_topk,
                norm_eps=float(self.q_norm.variance_epsilon),
                rotary_dim=int(self.head_dim * self.config.partial_rotary_factor),
            )

    def _supports_projected_output(self) -> bool:
        # HC must follow the row-parallel projection's cross-shard reduction.
        # The fused entry has no collective and therefore requires unsharded K.
        return self.tp_size == 1 and self.o_proj.weight.dtype == torch.bfloat16

    def can_prefetch_output(self, hidden: torch.Tensor) -> bool:
        """Use a stable Torch ABI; plan fusion from actual arrays inside JAX."""
        return (
            len(self.cache_owners) == 2
            and self._supports_projected_output()
            and self.output_weight_tiles is not None
            and hidden.device.type == "tpu"
            and hidden.ndim == 2
            and hidden.shape[1] == self.hidden_size
            and hidden.dtype == torch.bfloat16
        )

    def forward_with_indices(
        self,
        hidden,
        positions,
        metadata: QSAMetadata,
        residual=None,
        hc_gate=None,
        hc_token_tile=0,
        hc_min_tokens=1024,
    ):
        assert len(self.cache_owners) == 2, (
            "Bind main and index pools before QSA execution"
        )
        per_layer = cast(
            dict[str, AttentionMetadata], get_forward_context().attn_metadata
        )
        key, value, index_qk = self.input_proj(hidden).split(
            self.input_sizes[1:], dim=-1
        )
        qgate = self.q_proj(hidden)
        projected = residual is not None
        if projected:
            assert hc_gate is not None and self.can_prefetch_output(hidden)
        op = self._projected_op if projected else self._op
        assert op is not None
        output, selected = op(
            qgate,
            key,
            value,
            index_qk,
            self.q_norm.weight,
            self.k_norm.weight,
            self.indexer.q_layernorm.weight,
            self.indexer.k_layernorm.weight,
            positions,
            self.rope_cache,
            *self.cache_owners,
            metadata.query_start_loc,
            metadata.context_lens,
            cast(torch.Tensor, per_layer[self.prefix + ".raw_ring"].block_tables),
            cast(torch.Tensor, per_layer[self.prefix].block_tables),
            cast(torch.Tensor, per_layer[self.prefix].block_tables),
            cast(torch.Tensor, per_layer[self.prefix].request_distribution),
            k_norm_multiplier=self.k_norm_multiplier,
            decode_cu=metadata.decode_cu,
            output_weight=self.output_weight_tiles if projected else None,
            residual=residual,
            hc_gate=hc_gate,
            original_output_weight=self.o_proj.weight if projected else None,
            hc_token_tile=hc_token_tile,
            hc_min_tokens=hc_min_tokens,
            k_scale=cast(float, self._k_scale_float) if self.fp8_kv else 1.0,
            v_scale=cast(float, self._v_scale_float) if self.fp8_kv else 1.0,
        )
        if projected:
            return output, selected
        output = self.o_proj(output)
        if self.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output)
        return output, selected

    def forward(
        self,
        hidden,
        positions,
        metadata: QSAMetadata,
        residual=None,
        hc_gate=None,
        hc_token_tile=0,
        hc_min_tokens=1024,
    ):
        return self.forward_with_indices(
            hidden, positions, metadata, residual, hc_gate, hc_token_tile, hc_min_tokens
        )[0]
