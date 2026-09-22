# SPDX-License-Identifier: Apache-2.0
"""Model-local QSA, BF16 checkpoint loading and explicit typed cache binding."""
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               ReplicatedLinear)
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import (CircularBufferSpec, FullAttentionSpec,
                                        MLAAttentionSpec)

from vllm_torchtpu.kernels.qwen4_exp.qsa import build_qsa_op


@dataclass(frozen=True)
class QSAMetadata:
    """Task4 adapter supplies one shared ragged geometry and group-native pages.

    ``context_lens`` is the logical length BEFORE this chunk, independent of
    the RoPE ``positions`` argument. Padded requests use state_slots=-1 and
    repeated query offsets. Stable slots are neither batch rows nor page IDs.
    Admission establishes <=128 tokens/step, <=8192 context and RoPE positions
    in [0,8192). All real/scratch cache owners must be bound before forward.
    """
    query_start_loc: torch.Tensor
    context_lens: torch.Tensor
    state_slots: torch.Tensor
    main_block_table: torch.Tensor
    compressed_block_table: torch.Tensor


def make_rope_cache(max_positions: int = 8192):
    """One-time CPU BF16 trig preparation using checkpoint's theta=1e7."""
    # CPU initialization is intentional, including when constructing on TPU.
    with torch.device("cpu"):
        inv_freq = 1.0 / (10000000.0
                          **(torch.arange(0, 64, 2, dtype=torch.float32) / 64))
        angles = torch.arange(max_positions,
                              dtype=torch.float32)[:, None] * inv_freq[None]
        return torch.stack((angles.cos(), angles.sin()), 1).to(torch.bfloat16)


class Qwen4ExpQSABackend(AttentionBackend):
    supported_dtypes = [torch.bfloat16]
    supported_kv_cache_dtypes = ["auto", "bfloat16"]

    @staticmethod
    def get_name():
        return "QWEN4_EXP_TPU_QSA"

    @staticmethod
    def get_impl_cls():
        return Qwen4ExpQSA

    @staticmethod
    def get_builder_cls():
        # The model adapter enriches common group metadata with stable slots
        # and the *other group's* block table before constructing QSAMetadata.
        from vllm_torchtpu.layers.core.attention_metadata import \
            AttentionMetadataBuilder
        return AttentionMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [128]

    @staticmethod
    def get_min_page_size(vllm_config):
        return 128

    @classmethod
    def is_sparse(cls):
        return True

    @staticmethod
    def get_kv_cache_shape(num_blocks,
                           block_size,
                           num_kv_heads,
                           head_size,
                           cache_dtype_str="auto"):
        return (2, num_blocks, block_size, num_kv_heads, head_size)


class _NormWeight(nn.Module):

    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(width, dtype=torch.bfloat16),
                                   requires_grad=False)


class _Indexer(nn.Module):

    def __init__(self, prefix):
        super().__init__()
        self.index_qk_proj = ReplicatedLinear(2560,
                                              640,
                                              bias=False,
                                              params_dtype=torch.bfloat16,
                                              quant_config=None,
                                              prefix=prefix + ".index_qk_proj",
                                              return_bias=False,
                                              disable_tp=True)
        self.q_layernorm = _NormWeight(128)
        self.k_layernorm = _NormWeight(128)


class Qwen4ExpQSA(nn.Module, AttentionLayerBase):
    """Q24/KV2/D256 gated attention with a fully replicated Q4/K1 indexer.

    Checkpoint names are relative to ``...self_attn.``. No quant_config enters
    these BF16 projections. Five cache tensors stay owned by Task4's manager;
    the custom op donates state then copy_ makes writes visible to that owner.
    Profiling/capture uses explicitly bound scratch owners and shared metadata.
    """
    supports_dcp = False

    def __init__(self,
                 *,
                 tp_rank: int,
                 tp_size: int = 8,
                 device=None,
                 prefix: str = ""):
        super().__init__()
        if tp_size not in (1, 2, 4, 8) or not 0 <= tp_rank < tp_size:
            raise ValueError("QSA requires TP1/2/4/8 and an in-range rank")
        self.tp_rank, self.tp_size, self.prefix = tp_rank, tp_size, prefix
        self.num_heads, self.num_kv_heads = 24 // tp_size, max(2 // tp_size, 1)
        kv_size, kv_rank = min(tp_size, 2), tp_rank // max(tp_size // 2, 1)
        with torch.device(device or "cpu"):
            self.q_proj = ColumnParallelLinear(2560,
                                               12288,
                                               bias=False,
                                               params_dtype=torch.bfloat16,
                                               quant_config=None,
                                               prefix=prefix + ".q_proj",
                                               return_bias=False,
                                               tp_rank=tp_rank,
                                               tp_size=tp_size)
            for name in ("k_proj", "v_proj"):
                setattr(
                    self, name,
                    ColumnParallelLinear(2560,
                                         512,
                                         bias=False,
                                         params_dtype=torch.bfloat16,
                                         quant_config=None,
                                         prefix=prefix + "." + name,
                                         return_bias=False,
                                         tp_rank=kv_rank,
                                         tp_size=kv_size))
            # Upstream RowParallelLinear cannot take explicit TP geometry.
            # Reuse its constituent mechanisms: local unquantized linear and
            # the same TP reduction; loader slices columns exactly once.
            self.o_proj = ReplicatedLinear(self.num_heads * 256,
                                           2560,
                                           bias=False,
                                           params_dtype=torch.bfloat16,
                                           quant_config=None,
                                           prefix=prefix + ".o_proj",
                                           return_bias=False,
                                           disable_tp=True)
            self.q_norm, self.k_norm = _NormWeight(256), _NormWeight(256)
            self.indexer = _Indexer(prefix + ".indexer")
        self.register_buffer("rope_cache",
                             make_rope_cache().to(device=device),
                             persistent=False)
        self.kv_cache = ()
        self._loaded = set()
        self._op = build_qsa_op()

    def get_attn_backend(self):
        return Qwen4ExpQSABackend

    def get_kv_cache_spec(self, vllm_config):
        return FullAttentionSpec(block_size=128,
                                 num_kv_heads=self.num_kv_heads,
                                 head_size=256,
                                 dtype=torch.bfloat16)

    def get_auxiliary_cache_specs(self):
        """Planner discovers side owners without registering public GPU classes.

        Raw ring spec conservatively budgets both typed fields and R8 padding;
        only lanes 0..3 have logical history meaning. It still uses the native
        one-block-per-request CircularBufferSpec lifecycle, at logical R4.
        Int64 tag padding is shape-dependent: [2,8] lowers to 2048B with
        T(2,128), whereas [65,8] lowers to 8192B with T(8,128) and reversed
        minor axes. 8192 bytes/slot bounds raw+tags for both observed layouts over the
        admitted 2..65 slots (requests plus null). This is a native placement
        allowance, not a guarantee about arbitrary compiler layouts; Task4
        must account actual physical buffers independently.
        """
        return {
            "compressed_k":
            MLAAttentionSpec(block_size=128,
                             num_kv_heads=1,
                             head_size=128,
                             dtype=torch.bfloat16,
                             tokens_per_state=4),
            "raw_ring":
            CircularBufferSpec(block_size=4,
                               num_kv_heads=1,
                               head_size=132,
                               head_size_v=0,
                               dtype=torch.bfloat16,
                               page_size_padded=8192),
        }

    def bind_kv_cache(self, kv_cache):
        if len(kv_cache) != 5:
            raise ValueError(
                "QSA binds main K, main V, compressed K, raw ring, raw positions"
            )
        k, v, c, r, p = kv_cache
        if (k.ndim != 4 or k.shape[1:] != (128, self.num_kv_heads, 256)
                or v.shape != k.shape or c.ndim != 4
                or c.shape[1:] != (32, 1, 128) or r.ndim != 3
                or r.shape[1:] != (8, 128) or p.shape != r.shape[:2]
                or p.dtype != torch.int64 or any(x.dtype != torch.bfloat16
                                                 for x in (k, v, c, r))):
            raise ValueError(
                "QSA requires typed P128 BF16 caches, R8 raw storage and int64 positions"
            )
        self.kv_cache = tuple(kv_cache)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        parameters = dict(self.named_parameters())
        global_shapes = {
            "q_proj.weight": (12288, 2560),
            "k_proj.weight": (512, 2560),
            "v_proj.weight": (512, 2560),
            "o_proj.weight": (2560, 6144),
            "q_norm.weight": (256, ),
            "k_norm.weight": (256, ),
            "indexer.index_qk_proj.weight": (640, 2560),
            "indexer.q_layernorm.weight": (128, ),
            "indexer.k_layernorm.weight": (128, ),
        }
        loaded = set()
        for name, weight in weights:
            if name not in global_shapes or weight.shape != global_shapes[
                    name] or weight.dtype != torch.bfloat16:
                raise ValueError(f"Invalid BF16 QSA checkpoint tensor: {name}")
            parameter = parameters[name]
            with torch.no_grad():
                if name in ("q_proj.weight", "k_proj.weight", "v_proj.weight"):
                    parameter.weight_loader(parameter, weight)
                elif name == "o_proj.weight":
                    width = self.num_heads * 256
                    self.o_proj.weight_loader(
                        parameter, weight[:, self.tp_rank *
                                          width:(self.tp_rank + 1) * width])
                else:
                    parameter.copy_(weight)
            loaded.add(name)
        self._loaded.update(loaded)
        return loaded

    def finalize_weights(self):
        missing = set(dict(self.named_parameters())) - self._loaded
        if missing:
            raise ValueError(
                f"Missing QSA checkpoint tensors: {sorted(missing)}")

    def forward_with_indices(self, hidden, positions, metadata: QSAMetadata):
        assert len(
            self.kv_cache
        ) == 5, "Task4 must bind real or profiling scratch QSA owners"
        output, selected = self._op(
            self.q_proj(hidden), self.k_proj(hidden), self.v_proj(hidden),
            self.indexer.index_qk_proj(hidden), self.q_norm.weight,
            self.k_norm.weight, self.indexer.q_layernorm.weight,
            self.indexer.k_layernorm.weight, positions, self.rope_cache,
            *self.kv_cache, metadata.query_start_loc, metadata.context_lens,
            metadata.state_slots, metadata.main_block_table,
            metadata.compressed_block_table)
        output = self.o_proj(output)
        if self.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output)
        return output, selected

    def forward(self, hidden, positions, metadata: QSAMetadata):
        return self.forward_with_indices(hidden, positions, metadata)[0]
