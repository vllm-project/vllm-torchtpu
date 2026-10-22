# SPDX-License-Identifier: Apache-2.0
"""Model-local QSA, BF16 checkpoint loading and explicit typed cache binding."""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from functools import partial
from typing import cast

import torch
from torch import nn
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.attention.attention import set_default_quant_scales
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear, ReplicatedLinear
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.model_executor.parameter import BasevLLMParameter
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.attention.backend import AttentionBackend, MultipleOf
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheSpec,
    MLAAttentionSpec,
    get_kv_quant_mode,
)

from vllm_torchtpu.layers.adapter.custom_ops.qsa_op import build_qsa_op


@dataclass(frozen=True)
class QSAMetadata:
    """The model adapter supplies one shared ragged geometry and group-native pages.

    ``context_lens`` is the logical length BEFORE this chunk, independent of
    the RoPE ``positions`` argument.
    Padded requests use state_slots=-1 and
    repeated query offsets. Stable slots are neither batch rows nor page IDs.
    Admission establishes <=8192 tokens/step and context within the model
    position capacity. RoPE covers [0, max_model_len). All real/scratch cache
    owners must be bound before forward.
    """

    query_start_loc: torch.Tensor
    context_lens: torch.Tensor
    state_slots: torch.Tensor
    main_block_table: torch.Tensor
    compressed_block_table: torch.Tensor


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
    ring_bytes = max(8192, ring_rows * (((index_dim + 127) // 128 * 128) * 2 + 8))
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


class Qwen4ExpQSABackend(AttentionBackend):
    supported_dtypes = [torch.bfloat16]
    supported_kv_cache_dtypes = ["auto", "bfloat16", "fp8", "fp8_e4m3"]

    @staticmethod
    def get_name():
        return "QWEN4_EXP_TPU_QSA"

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError("QSA is constructed by its model-owned layer")

    @staticmethod
    def get_builder_cls():
        # The model adapter enriches common group metadata with stable slots
        # and the *other group's* block table before constructing QSAMetadata.
        from vllm_torchtpu.layers.core.attention_metadata import (
            AttentionMetadataBuilder,
        )

        return AttentionMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes(
        kv_cache_spec: KVCacheSpec | None = None,
    ) -> list[MultipleOf | int]:
        return [QSA_BLOCK_SIZE]

    @staticmethod
    def get_min_page_size(vllm_config):
        return QSA_BLOCK_SIZE

    @classmethod
    def is_sparse(cls):
        return True

    @staticmethod
    def get_kv_cache_shape(
        num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"
    ):
        return (2, num_blocks, block_size, num_kv_heads, head_size)


class _Indexer(nn.Module):
    def __init__(self, config, prefix):
        super().__init__()
        self.index_qk_proj = ReplicatedLinear(
            config.hidden_size,
            (config.indexer_n_heads + config.indexer_kv_heads)
            * config.indexer_head_dim,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=None,
            prefix=prefix + ".index_qk_proj",
            return_bias=False,
            disable_tp=True,
        )
        self.q_layernorm = GemmaRMSNorm(
            config.indexer_head_dim, eps=config.rms_norm_eps
        )
        self.k_layernorm = GemmaRMSNorm(
            config.indexer_head_dim, eps=config.rms_norm_eps
        )


_KV_SCALE_NAMES = {
    "k_proj.k_scale": "_k_scale",
    "k_proj.output_scale": "_k_scale",
    "attn.k_scale": "_k_scale",
    "attn._k_scale": "_k_scale",
    "k_scale": "_k_scale",
    "_k_scale": "_k_scale",
    "v_proj.v_scale": "_v_scale",
    "v_proj.output_scale": "_v_scale",
    "attn.v_scale": "_v_scale",
    "attn._v_scale": "_v_scale",
    "v_scale": "_v_scale",
    "_v_scale": "_v_scale",
}


class Qwen4ExpQSA(nn.Module, AttentionLayerBase):
    """Config-derived gated attention with a fully replicated MQA indexer.

    Checkpoint names are relative to ``...self_attn.``. No quant_config enters
    these BF16 projections. Typed cache tensors stay owned by Task4's manager;
    the custom op donates state then copy_ makes writes visible to that owner.
    Profiling/capture uses explicitly bound scratch owners and shared metadata.
    """

    supports_dcp = False
    k_proj: ColumnParallelLinear
    v_proj: ColumnParallelLinear

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
    ):
        super().__init__()
        if tp_size <= 0:
            raise ValueError(f"QSA requires positive tp_size, got {tp_size}")
        if not 0 <= tp_rank < tp_size:
            raise ValueError("QSA requires an in-range TP rank")
        self.tp_rank, self.tp_size, self.prefix = tp_rank, tp_size, prefix
        self.fp8_kv = cache_dtype in ("fp8", "fp8_e4m3")
        self.cache_dtype = cache_dtype
        self.config = config
        self.hidden_size = int(config.hidden_size)
        self.head_dim = int(config.head_dim)
        self.total_num_heads = int(config.num_attention_heads)
        self.total_num_kv_heads = int(config.num_key_value_heads)
        if self.total_num_heads <= 0 or self.total_num_heads % tp_size:
            raise ValueError(
                f"QSA num_attention_heads={self.total_num_heads} must be positive "
                f"and divisible by tp_size={tp_size}"
            )
        if self.total_num_kv_heads <= 0 or (
            self.total_num_kv_heads % tp_size and tp_size % self.total_num_kv_heads
        ):
            raise ValueError(
                f"QSA num_key_value_heads={self.total_num_kv_heads} must be positive "
                f"and shard or replicate evenly across tp_size={tp_size}"
            )
        if int(config.indexer_kv_heads) != 1:
            raise ValueError(
                f"QSA indexer_kv_heads={config.indexer_kv_heads} must be 1 "
                "for the replicated MQA indexer"
            )
        if (
            self.head_dim <= 0
            or int(config.indexer_head_dim) <= 0
            or int(config.indexer_n_heads) <= 0
        ):
            raise ValueError(
                "QSA head_dim, indexer_head_dim and indexer_n_heads must be positive"
            )
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = max(self.total_num_kv_heads // tp_size, 1)
        self.index_head_dim = int(config.indexer_head_dim)
        self.index_n_heads = int(config.indexer_n_heads)
        self.compress_ratio = int(config.indexer_compress_ratio)
        self.token_topk = int(config.indexer_budget)
        self.max_selected_tokens = self.token_topk + self.compress_ratio - 1
        self.block_size = QSA_BLOCK_SIZE
        # Eight-row alignment is the current TPU allocation policy, not the
        # logical compression ratio. Only compress_ratio lanes hold history.
        self.ring_size = (self.compress_ratio + 7) // 8 * 8
        kv_size = min(tp_size, self.total_num_kv_heads)
        kv_rank = tp_rank // max(tp_size // self.total_num_kv_heads, 1)
        with torch.device(device or "cpu"), set_default_torch_dtype(torch.bfloat16):
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                2 * self.total_num_heads * self.head_dim,
                bias=False,
                params_dtype=torch.bfloat16,
                quant_config=None,
                prefix=prefix + ".q_proj",
                return_bias=False,
                tp_rank=tp_rank,
                tp_size=tp_size,
            )
            for name in ("k_proj", "v_proj"):
                setattr(
                    self,
                    name,
                    ColumnParallelLinear(
                        self.hidden_size,
                        self.total_num_kv_heads * self.head_dim,
                        bias=False,
                        params_dtype=torch.bfloat16,
                        quant_config=None,
                        prefix=prefix + "." + name,
                        return_bias=False,
                        tp_rank=kv_rank,
                        tp_size=kv_size,
                    ),
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
            self.indexer = _Indexer(config, prefix + ".indexer")
        self.register_buffer(
            "rope_cache",
            make_rope_cache(
                max_model_len,
                rotary_dim=int(self.head_dim * config.partial_rotary_factor),
                theta=float(config.rope_parameters["rope_theta"]),
            ).to(device=device),
            persistent=False,
        )
        with torch.device(device or "cpu"):
            set_default_quant_scales(self, register_buffer=True)
        self.cache_owners: tuple[torch.Tensor, ...] = ()
        self._loaded = set()
        self._op = partial(
            build_qsa_op(),
            compress_ratio=self.compress_ratio,
            token_topk=self.token_topk,
            norm_eps=float(self.q_norm.variance_epsilon),
            rotary_dim=int(self.head_dim * config.partial_rotary_factor),
        )

    def get_attn_backend(self):
        return Qwen4ExpQSABackend

    def get_kv_cache_spec(self, vllm_config):
        return get_qsa_cache_specs(self.config, self.tp_size, self.cache_dtype)[0]

    def get_auxiliary_cache_specs(self):
        return get_qsa_cache_specs(self.config, self.tp_size, self.cache_dtype)[1]

    def get_cache_owner_layouts(self, pages, slots):
        """Typed owner order shared by allocation, binding and the custom op."""
        auxiliary = (
            (
                "compressed_k",
                (
                    pages,
                    self.block_size // self.compress_ratio,
                    1,
                    (self.index_head_dim + 127) // 128 * 128,
                ),
                torch.bfloat16,
                0,
            ),
            (
                "raw_ring",
                (slots, self.ring_size, self.index_head_dim),
                torch.bfloat16,
                0,
            ),
            # Native low/high bit planes retain the int64 payload width.
            ("rope_tags", (slots, self.ring_size, 2), torch.int32, -1),
        )
        packing = 4 if self.fp8_kv else 2
        return (
            (
                "head_packed_kv",
                (
                    pages,
                    self.block_size,
                    (2 * self.num_kv_heads + packing - 1) // packing,
                    packing,
                    (self.head_dim + 127) // 128 * 128,
                ),
                torch.float8_e4m3fn if self.fp8_kv else torch.bfloat16,
                0,
            ),
            *auxiliary,
        )

    def bind_cache_owners(self, kv_cache: Sequence[torch.Tensor]) -> None:
        if len(kv_cache) != 4:
            raise ValueError("QSA requires main KV, compressed K, ring and tags")
        layouts = self.get_cache_owner_layouts(
            kv_cache[0].shape[0], kv_cache[2].shape[0]
        )
        for owner, (role, shape, dtype, _) in zip(kv_cache, layouts):
            if owner.shape != shape or owner.dtype != dtype:
                raise ValueError(f"QSA {role} requires shape={shape}, dtype={dtype}")
        self.cache_owners = tuple(kv_cache)

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
            if name in _KV_SCALE_NAMES:
                target = _KV_SCALE_NAMES[name]
                if weight.numel() != 1:
                    raise ValueError(f"QSA {name} requires one per-tensor scale")
                value = float(weight.detach().float().cpu().item())
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"QSA {name} requires a finite positive scale")
                if target in self._loaded or target in loaded:
                    raise ValueError(f"Duplicate QSA checkpoint scale: {target}")
                # Keep the serialized buffer and host scalar consistent. The
                # compiled kernel specializes on this immutable layer scale.
                getattr(self, target).fill_(value)
                getattr(self, target + "_cpu").fill_(value)
                setattr(self, target + "_float", value)
                loaded.add(target)
                continue
            if (
                name not in global_shapes
                or weight.shape != global_shapes[name]
                or weight.dtype != torch.bfloat16
            ):
                raise ValueError(f"Invalid BF16 QSA checkpoint tensor: {name}")
            parameter = parameters[name]
            with torch.no_grad():
                if name in ("q_proj.weight", "k_proj.weight", "v_proj.weight"):
                    cast(BasevLLMParameter, parameter).weight_loader(parameter, weight)
                elif name == "o_proj.weight":
                    width = self.num_heads * self.head_dim
                    self.o_proj.weight_loader(
                        parameter,
                        weight[:, self.tp_rank * width : (self.tp_rank + 1) * width],
                    )
                else:
                    parameter.copy_(weight)
            loaded.add(name)
        self._loaded.update(loaded)
        return loaded

    def finalize_weights(self):
        missing = set(dict(self.named_parameters())) - self._loaded
        if missing:
            raise ValueError(f"Missing QSA checkpoint tensors: {sorted(missing)}")

    def forward_with_indices(self, hidden, positions, metadata: QSAMetadata):
        assert len(self.cache_owners) == 4, (
            "Task4 must bind real or profiling scratch QSA owners"
        )
        output, selected = self._op(
            self.q_proj(hidden),
            self.k_proj(hidden),
            self.v_proj(hidden),
            self.indexer.index_qk_proj(hidden),
            self.q_norm.weight,
            self.k_norm.weight,
            self.indexer.q_layernorm.weight,
            self.indexer.k_layernorm.weight,
            positions,
            self.rope_cache,
            *self.cache_owners,
            metadata.query_start_loc,
            metadata.context_lens,
            metadata.state_slots,
            metadata.main_block_table,
            metadata.compressed_block_table,
            **(
                {"k_scale": self._k_scale_float, "v_scale": self._v_scale_float}
                if self.fp8_kv
                else {}
            ),
        )
        output = self.o_proj(output)
        if self.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output)
        return output, selected

    def forward(self, hidden, positions, metadata: QSAMetadata):
        return self.forward_with_indices(hidden, positions, metadata)[0]
