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
"""Context-parallel zigzag prefill program and custom ops for Gemma 4."""

from __future__ import annotations

import copy
import dataclasses
import functools
import re
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.distributed import _functional_collectives as funcol
from torch_tpu._internal import pallas
from vllm.compilation.backends import set_model_tag
from vllm.compilation.decorators import support_torch_compile
from vllm.config import get_current_vllm_config, set_current_vllm_config
from vllm.config.utils import getattr_iter
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.model_executor.layers import vocab_parallel_embedding
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.models.gemma4 import Gemma4ForCausalLM, Gemma4Model
from vllm.model_executor.models.gemma4_mm import Gemma4ForConditionalGeneration
from vllm.model_executor.models.utils import maybe_prefix

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter.attention import (
    PallasBatchedRPAAttentionBackendImpl,
    native_kv_cache_view,
)
from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8LinearMethodTPU
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.gemma4 import cp_prefill
from vllm_torchtpu.models.vllm.gemma4.layers import down_proj_from_gate_up
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    get_vllm_model_wrapper_context,
)

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig

logger = init_logger(__name__)

_REPLICA = "cp_replica"
_REPLICATED_TENSOR = re.compile(
    r"layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj|experts)\."
)

_orig_causal_lm_init = Gemma4ForCausalLM.__init__
_orig_causal_lm_forward = Gemma4ForCausalLM.forward
_orig_mm_forward = Gemma4ForConditionalGeneration.forward
_orig_model_load_weights = Gemma4Model.load_weights


@dataclasses.dataclass(frozen=True)
class LayerOps:
    prologue: Any
    full_attention: Any = None
    sliding_exchange: Any = None
    sliding_attention: Any = None


_OPS: dict[cp_prefill.LayerConfig, LayerOps] = {}


def _register(name: str, fn: Any, fake: Any, *, donate: bool, **bound: Any) -> Any:
    donate_argnums = (0,) if donate and not envs.TPU_KERNEL_ITER_MODE else None
    op = pallas.jax_op(
        f"cp_prefill::{name}_{len(_OPS)}",
        functools.partial(fn, **bound),
        donate_argnums=donate_argnums,
    )
    op.register_fake(fake)
    return op


def _layer_fake_outputs(
    kv_cache: torch.Tensor, q_hbm: torch.Tensor, cfg: cp_prefill.LayerConfig
) -> tuple[torch.Tensor, ...]:
    rows = q_hbm.shape[1]
    width = cfg.num_heads * cfg.head_dim
    if cfg.fp8_out_rows:
        fp8 = kv_cache.new_empty((rows, width), dtype=torch.float8_e4m3fn)
        scales = kv_cache.new_empty((rows, 1), dtype=torch.float32)
        return torch.empty_like(kv_cache), fp8, scales
    out = kv_cache.new_empty((rows, cfg.num_heads, cfg.head_dim), dtype=q_hbm.dtype)
    return torch.empty_like(kv_cache), out


def build_ops(cfg: cp_prefill.LayerConfig) -> LayerOps:
    """Build and cache PyTorch custom ops for ``cfg``."""
    if cfg not in _OPS:
        _OPS[cfg] = _build_ops(cfg)
    return _OPS[cfg]


def _build_ops(cfg: cp_prefill.LayerConfig) -> LayerOps:
    cache_view = functools.partial(
        native_kv_cache_view,
        num_kv_heads=cfg.local_kv_heads,
        head_dim=cfg.head_dim,
        kv_layout=cfg.kv_layout,
    )
    d, heads, kv_heads = cfg.head_dim, cfg.num_heads, cfg.num_kv_heads
    group, slots = heads // kv_heads, -(-2 * kv_heads // 4)

    def fake_prologue(
        qkv: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        q_norm_weight: torch.Tensor,
        k_norm_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        del cos, sin, q_norm_weight, k_norm_weight
        rows = qkv.shape[0]
        q_hbm = qkv.new_empty((kv_heads, rows, group // 2, 2, d), dtype=torch.bfloat16)
        new_kv = qkv.new_empty((rows, slots, 4, d), dtype=torch.float8_e4m3fn)
        if cfg.fp8_compute:
            q_fp8 = qkv.new_empty(
                (kv_heads, rows, group // 4, 4, d), dtype=torch.float8_e4m3fn
            )
            return q_hbm, new_kv, q_fp8
        return q_hbm, new_kv

    prologue_op = _register(
        "prologue", cp_prefill.qkv_prologue, fake_prologue, donate=False, cfg=cfg
    )

    if cfg.sliding_window is None:
        if cfg.fp8_compute:

            def fake_full_fp8(
                kv_cache: torch.Tensor,
                q_hbm: torch.Tensor,
                q_fp8: torch.Tensor,
                kv_rows: torch.Tensor,
                seq_lens: torch.Tensor,
                block_tables: torch.Tensor,
                rank: torch.Tensor,
            ) -> tuple[torch.Tensor, ...]:
                del q_fp8, kv_rows, seq_lens, block_tables, rank
                return _layer_fake_outputs(kv_cache, q_hbm, cfg)

            full_op = _register(
                "full_attention_fp8_q",
                cp_prefill.full_attention_layer_fp8_q,
                fake_full_fp8,
                donate=True,
                cfg=cfg,
                cache_view=cache_view,
            )
        else:

            def fake_full(
                kv_cache: torch.Tensor,
                q_hbm: torch.Tensor,
                kv_rows: torch.Tensor,
                seq_lens: torch.Tensor,
                block_tables: torch.Tensor,
                rank: torch.Tensor,
            ) -> tuple[torch.Tensor, ...]:
                del kv_rows, seq_lens, block_tables, rank
                return _layer_fake_outputs(kv_cache, q_hbm, cfg)

            full_op = _register(
                "full_attention",
                cp_prefill.full_attention_layer,
                fake_full,
                donate=True,
                cfg=cfg,
                cache_view=cache_view,
            )
        return LayerOps(prologue=prologue_op, full_attention=full_op)

    sliding_window = cfg.sliding_window

    def fake_exchange(
        qkv: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        q_norm_weight: torch.Tensor,
        k_norm_weight: torch.Tensor,
        seq_lens: torch.Tensor,
        rank: torch.Tensor,
        page_size: int,
    ) -> torch.Tensor:
        del cos, sin, q_norm_weight, k_norm_weight, seq_lens, rank
        halo_pages = cp_prefill.num_halo_pages(sliding_window, page_size)
        rows = (3 * halo_pages + 1) * page_size
        return qkv.new_empty((rows, slots * 4 * d), dtype=torch.int8)

    exchange_op = _register(
        "sliding_exchange",
        cp_prefill.sliding_exchange_rows_from_qkv,
        fake_exchange,
        donate=False,
        cfg=cfg,
    )

    if cfg.fp8_compute:

        def fake_sliding_fp8(
            kv_cache: torch.Tensor,
            q_hbm: torch.Tensor,
            new_kv: torch.Tensor,
            q_fp8: torch.Tensor,
            gathered: torch.Tensor,
            seq_lens: torch.Tensor,
            block_tables: torch.Tensor,
            rank: torch.Tensor,
        ) -> tuple[torch.Tensor, ...]:
            del new_kv, q_fp8, gathered, seq_lens, block_tables, rank
            return _layer_fake_outputs(kv_cache, q_hbm, cfg)

        sliding_op = _register(
            "sliding_attention_fp8_q",
            cp_prefill.sliding_attention_layer_fp8_q,
            fake_sliding_fp8,
            donate=True,
            cfg=cfg,
            cache_view=cache_view,
        )
    else:

        def fake_sliding(
            kv_cache: torch.Tensor,
            q_hbm: torch.Tensor,
            new_kv: torch.Tensor,
            gathered: torch.Tensor,
            seq_lens: torch.Tensor,
            block_tables: torch.Tensor,
            rank: torch.Tensor,
        ) -> tuple[torch.Tensor, ...]:
            del new_kv, gathered, seq_lens, block_tables, rank
            return _layer_fake_outputs(kv_cache, q_hbm, cfg)

        sliding_op = _register(
            "sliding_attention",
            cp_prefill.sliding_attention_layer,
            fake_sliding,
            donate=True,
            cfg=cfg,
            cache_view=cache_view,
        )
    return LayerOps(
        prologue=prologue_op,
        sliding_exchange=exchange_op,
        sliding_attention=sliding_op,
    )


def _layer_config(
    attn: Any,
    *,
    num_heads: int,
    num_kv_heads: int,
    world: int,
    prologue_eps: float,
    o_proj: Any = None,
    min_rows: int = 0,
) -> cp_prefill.LayerConfig:
    impl = attn.impl
    if not isinstance(impl, PallasBatchedRPAAttentionBackendImpl):
        raise NotImplementedError(
            "Context-parallel prefill attention requires --attention-backend CUSTOM, "
            f"got {type(impl).__name__}."
        )
    if impl.rpa_kernel is None and impl.rpa_dcp_cache_kernel is None:
        raise RuntimeError(
            f"{attn.layer_name}: prepare_cp_prefill_attention requires "
            "initialize_kernel to run first."
        )
    if impl.kv_cache_quantized_dtype != torch.float8_e4m3fn:
        raise NotImplementedError(
            "Context-parallel prefill attention requires an fp8_e4m3 KV cache."
        )
    unsupported = [
        what
        for what, present in (
            ("attention sinks", impl.sinks is not None),
            ("KV sharing", impl.kv_sharing_target_layer_name is not None),
            ("DCP", impl.dcp_world_size > 1),
            ("PCP streaming", impl._pcp_streaming),
            ("block-major KV pool", impl._bm_kernel),
        )
        if present
    ]
    if unsupported:
        raise NotImplementedError(
            "Context-parallel prefill attention does not support "
            f"{', '.join(unsupported)}."
        )
    if 2 * num_kv_heads % 4:
        raise NotImplementedError(
            f"{num_kv_heads} KV heads do not fill whole 4-byte FP8 words."
        )
    if impl.sliding_window is not None and world < 3:
        raise ValueError(
            f"Sliding-window CP requires world >= 3 for halo exchange, got {world}."
        )
    if impl.num_heads * world != num_heads:
        raise ValueError(
            f"{num_heads} query heads do not match {world} ranks x {impl.num_heads}."
        )
    if (num_heads // num_kv_heads) % 2 or impl.head_size % 128:
        raise NotImplementedError(
            "CP QKV prologue requires an even query-to-KV head ratio and "
            f"128-aligned head_size, got {num_heads // num_kv_heads} and "
            f"{impl.head_size}."
        )
    if impl.fp8_compute and (num_heads // num_kv_heads) % 4:
        raise ValueError(
            f"TPU_FP8_ATTENTION: {num_heads} query heads per {num_kv_heads} KV heads "
            "are not divisible by 4."
        )

    fp8_compute = impl.fp8_compute or (
        envs.TPU_FP8_ATTENTION and impl._fp8_compute_supported(num_heads, num_kv_heads)
    )
    fp8_out_rows = (
        o_proj is not None
        and min_rows > 0
        and isinstance(o_proj.quant_method, VllmFp8LinearMethodTPU)
        and o_proj.bias is None
        and o_proj.weight.dtype == torch.float8_e4m3fn
        and o_proj.quant_method.supports_quantized_input(o_proj, min_rows)
    )
    cfg = cp_prefill.LayerConfig(
        world=world,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=impl.head_size,
        sliding_window=impl.sliding_window,
        sm_scale=impl.scale,
        soft_cap=impl.logits_soft_cap,
        k_scale=attn._k_scale_float,
        v_scale=attn._v_scale_float,
        fp8_compute=fp8_compute,
        kv_layout=impl.kv_layout,
        prologue_eps=prologue_eps,
        fp8_out_rows=fp8_out_rows,
        unmasked_interior=True,
    )
    if cfg.local_kv_heads != impl.num_kv_heads:
        raise ValueError(
            f"{num_kv_heads} KV heads across {world} ranks give "
            f"{cfg.local_kv_heads} per rank, but cache holds {impl.num_kv_heads}."
        )
    return cfg


def prepare_cp_prefill_attention(
    attn: Any,
    *,
    num_heads: int,
    num_kv_heads: int,
    world: int,
    prologue_eps: float,
    o_proj: Any = None,
    min_rows: int = 0,
) -> None:
    """Build and attach CP prefill attention custom ops onto ``attn``."""
    if get_vllm_model_wrapper_context().kv_cache_bundle is not None:
        raise NotImplementedError(
            "Context-parallel prefill attention does not support bundled KV cache."
        )
    cfg = _layer_config(
        attn,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        world=world,
        prologue_eps=prologue_eps,
        o_proj=o_proj,
        min_rows=min_rows,
    )
    if cfg not in _OPS:
        logger.info_once(
            "Context-parallel prefill attention on %d ranks: %d query / %d KV heads, "
            "sliding window %s, %s QK/PV, fp8_out=%s.",
            world,
            num_heads,
            num_kv_heads,
            cfg.sliding_window,
            "fp8" if cfg.fp8_compute else "bf16",
            cfg.fp8_out_rows,
        )
    attn._cp_prefill_config = cfg
    attn._cp_prefill_ops = build_ops(cfg)


def _placeholder_output(
    like: torch.Tensor, cfg: cp_prefill.LayerConfig
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    rows = like.shape[0]
    width = cfg.num_heads * cfg.head_dim
    if cfg.fp8_out_rows:
        fp8 = like.new_zeros((rows, width), dtype=torch.float8_e4m3fn)
        return fp8, like.new_ones((rows, 1), dtype=torch.float32)
    return like.new_ones((rows, width))


def _waited(t: torch.Tensor) -> torch.Tensor:
    return funcol.wait_tensor(t) if isinstance(t, funcol.AsyncCollectiveTensor) else t


def _all_gather_rows(x: torch.Tensor) -> torch.Tensor:
    return _waited(funcol.all_gather_single(x, 0, get_tp_group().device_group))


def cp_prefill_attention_from_qkv(
    attn: Any,
    qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    plan: cp_prefill.ZigzagPlan,
    rank: torch.Tensor,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run zigzag CP prefill attention from fused ``qkv`` rows on ``rank``."""
    cfg: cp_prefill.LayerConfig | None = attn.__dict__.get("_cp_prefill_config")
    if cfg is None or cfg.world != plan.world:
        raise RuntimeError(
            f"{attn.layer_name}: call prepare_cp_prefill_attention for world="
            f"{plan.world} before tracing."
        )
    if not isinstance(rank, torch.Tensor):
        raise TypeError("rank must be passed as a device tensor.")
    rows = qkv.shape[0]
    if rows != 2 * plan.piece_tokens:
        raise ValueError(f"{rows} rows do not match 2 * {plan.piece_tokens}")
    attn_metadata, _, kv_cache, _ = get_attention_context(attn.layer_name)
    if kv_cache.numel() == 0:
        return _placeholder_output(qkv, cfg)

    ops: LayerOps = attn._cp_prefill_ops
    seq_lens, block_tables = attn_metadata.seq_lens, attn_metadata.block_tables
    weights = (q_norm_weight, k_norm_weight)
    if cfg.sliding_window is not None:
        exchanged = ops.sliding_exchange(
            qkv, cos, sin, *weights, seq_lens, rank, plan.page_size
        )
        gathered = _all_gather_rows(exchanged)
        q_hbm, new_kv, *q_fp8 = ops.prologue(qkv, cos, sin, *weights)
        new_kv_cache, *out = ops.sliding_attention(
            kv_cache, q_hbm, new_kv, *q_fp8, gathered, seq_lens, block_tables, rank
        )
    else:
        q_hbm, new_kv, *q_fp8 = ops.prologue(qkv, cos, sin, *weights)
        kv_rows = new_kv.view(torch.int8).reshape(rows, -1)
        gathered = _all_gather_rows(kv_rows)
        new_kv_cache, *out = ops.full_attention(
            kv_cache, q_hbm, *q_fp8, gathered, seq_lens, block_tables, rank
        )
    kv_cache.copy_(new_kv_cache)
    if cfg.fp8_out_rows:
        return out[0], out[1]
    return out[0].reshape(rows, cfg.num_heads * cfg.head_dim)


class _ReplicaAttention(nn.Module):
    def __init__(self, attn: Any, quant_config: Any, prefix: str) -> None:
        super().__init__()
        self.qkv_proj = QKVParallelLinear(
            attn.hidden_size,
            attn.head_dim,
            attn.total_num_heads,
            attn.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
            disable_tp=True,
        )
        self.o_proj = RowParallelLinear(
            attn.total_num_heads * attn.head_dim,
            attn.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
            disable_tp=True,
        )


class _ReplicaMLP(nn.Module):
    def __init__(
        self, mlp: Any, hidden_size: int, quant_config: Any, prefix: str
    ) -> None:
        super().__init__()
        intermediate_size = mlp.down_proj.input_size
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
            disable_tp=True,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
            disable_tp=True,
        )


class _ReplicaLayer(nn.Module):
    def __init__(self, layer: Any, config: Any, quant_config: Any, prefix: str) -> None:
        super().__init__()
        self.self_attn = _ReplicaAttention(
            layer.self_attn, quant_config, f"{prefix}.self_attn"
        )
        self.mlp = _ReplicaMLP(
            layer.mlp, layer.hidden_size, quant_config, f"{prefix}.mlp"
        )
        if layer.enable_moe_block:
            self.experts = FusedMoEFactory(
                num_experts=config.num_experts,
                top_k=config.top_k_experts,
                hidden_size=config.hidden_size,
                intermediate_size=getattr_iter(
                    config, ("moe_intermediate_size", "expert_intermediate_size")
                ),
                renormalize=True,
                quant_config=quant_config,
                tp_size=1,
                dp_size=1,
                prefix=f"{prefix}.experts",
                custom_routing_function=layer.router.routing_function,
                activation="gelu_tanh",
            )


class Gemma4ProjectionReplica(nn.Module):
    """Unsharded projection and embedding replica for Gemma 4 CP prefill."""

    def __init__(self, model: Gemma4Model, quant_config: Any, prefix: str) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            _ReplicaLayer(layer, model.config, quant_config, f"{prefix}.layers.{i}")
            for i, layer in enumerate(model.layers)
        )
        self.embed_tokens = vocab_parallel_embedding.VocabParallelEmbedding(
            model.config.vocab_size,
            model.config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
            disable_tp=True,
        )


def _rank_rows(x: torch.Tensor, plan: cp_prefill.ZigzagPlan, rank: int) -> torch.Tensor:
    size = plan.piece_tokens
    return torch.cat([x[p * size : (p + 1) * size] for p in plan.rank_pieces(rank)])


def _token_order(rows: torch.Tensor, plan: cp_prefill.ZigzagPlan) -> torch.Tensor:
    size = plan.piece_tokens
    blocks = []
    for piece in range(plan.num_pieces):
        rank, slot = plan.piece_owner(piece)
        start = (2 * rank + slot) * size
        blocks.append(rows[start : start + size])
    return torch.cat(blocks)


def _rotary_rows(
    rope: Any, x: torch.Tensor, positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cos_sin = rope._match_cos_sin_cache_dtype(x).index_select(0, positions)
    cos, sin = cos_sin.chunk(2, dim=-1)
    return cos.contiguous(), sin.contiguous()


def _replica_o_projection(o_proj: Any, attended: Any) -> torch.Tensor:
    if not isinstance(attended, tuple):
        return o_proj(attended)[0]
    x_q, x_scale = attended
    return o_proj.quant_method.apply_quantized_input(o_proj, x_q, x_scale)


def _context_parallel_layer(
    layer: Any,
    replica: Any,
    positions: torch.Tensor,
    x: torch.Tensor,
    plan: cp_prefill.ZigzagPlan,
    rank: torch.Tensor,
) -> torch.Tensor:
    attn = layer.self_attn
    residual = x
    qkv, _ = replica.self_attn.qkv_proj(layer.input_layernorm(x))
    cos, sin = _rotary_rows(attn.rotary_emb, qkv, positions)
    attended = cp_prefill_attention_from_qkv(
        attn.attn,
        qkv,
        cos,
        sin,
        attn.q_norm.weight,
        attn.k_norm.weight,
        plan,
        rank,
    )
    out = _replica_o_projection(replica.self_attn.o_proj, attended)
    x = layer.post_attention_layernorm(out) + residual
    residual = x

    gate_up, _ = replica.mlp.gate_up_proj(layer.pre_feedforward_layernorm(x))
    out = down_proj_from_gate_up(replica.mlp, gate_up, act_fn=layer.mlp.act_fn)
    if layer.enable_moe_block:
        experts = replica.experts(
            layer.pre_feedforward_layernorm_2(residual), layer.router(residual)
        )
        out = layer.post_feedforward_layernorm_1(out)
        out = out + layer.post_feedforward_layernorm_2(experts)
    return (layer.post_feedforward_layernorm(out) + residual) * layer.layer_scalar


def _context_parallel_forward(
    model: Any,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    plan: cp_prefill.ZigzagPlan,
    rank: torch.Tensor,
    rows: torch.Tensor,
) -> torch.Tensor:
    replica: Gemma4ProjectionReplica = model._modules[_REPLICA]
    x = replica.embed_tokens(input_ids.index_select(0, rows)) * model.normalizer
    local_positions = positions.index_select(0, rows)
    for layer, replica_layer in zip(model.layers, replica.layers):
        x = _context_parallel_layer(
            layer, replica_layer, local_positions, x, plan, rank
        )
    return _token_order(_all_gather_rows(model.norm(x)), plan)


@support_torch_compile(dynamic_arg_dims={"input_ids": 0, "positions": 0})
class Gemma4ContextParallelProgram(nn.Module):
    """Compiled zigzag CP prefill program for a single token bucket."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        model: Gemma4Model,
        num_tokens: int,
    ) -> None:
        del prefix
        super().__init__()
        self.model = model
        self.plan = cp_prefill.make_zigzag_plan(
            num_tokens,
            get_tensor_model_parallel_world_size(),
            vllm_config.cache_config.block_size,
        )
        rank = get_tensor_model_parallel_rank()
        self.register_buffer(
            "rank", torch.tensor([rank], dtype=torch.int32), persistent=False
        )
        self.register_buffer(
            "rows",
            _rank_rows(torch.arange(num_tokens, dtype=torch.int32), self.plan, rank),
            persistent=False,
        )

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return _context_parallel_forward(
            self.model, input_ids, positions, self.plan, self.rank, self.rows
        )


def _check_context_parallel(
    model: Gemma4Model, vllm_config: VllmConfig, cp_min_tokens: int
) -> None:
    what = "TPU_GEMMA4_CP_MIN_TOKENS"
    parallel = vllm_config.parallel_config
    unsupported = [
        name
        for name, present in (
            ("PP", parallel.pipeline_parallel_size > 1),
            ("DP", parallel.data_parallel_size > 1),
            ("LoRA", vllm_config.lora_config is not None),
            ("speculative decoding", vllm_config.speculative_config is not None),
            ("prefix caching", vllm_config.cache_config.enable_prefix_caching),
            ("kv_sharing_fast_prefill", model.fast_prefill_enabled),
            ("auxiliary hidden states", bool(model.aux_hidden_state_layers)),
        )
        if present
    ]
    for layer in model.layers:
        attn = layer.self_attn
        moe = layer.enable_moe_block
        for name, present in (
            ("KV sharing", attn.is_kv_shared_layer),
            ("per-layer embeddings", layer.per_layer_input_gate is not None),
            ("attention biases", attn.o_proj.bias is not None),
            ("EP weight filter", moe and parallel.enable_ep_weight_filter),
            ("EPLB", moe and parallel.enable_eplb),
            ("non-neox rotary", not attn.rotary_emb.is_neox_style),
        ):
            if present and name not in unsupported:
                unsupported.append(name)
    if unsupported:
        raise ValueError(f"{what} does not support {', '.join(unsupported)}")
    if parallel.tensor_parallel_size < 3:
        raise ValueError(
            f"{what} requires TP >= 3, got {parallel.tensor_parallel_size}"
        )

    sizes = sorted(int(s) for s in vllm_config.compilation_config.compile_sizes)
    solo = envs.VLLM_TPU_SOLO_PREFILL_OVER_TOKENS
    above = [s for s in sizes if s >= solo] if solo > 0 else []
    if len(above) < 2 or not above[0] < cp_min_tokens <= above[1]:
        raise ValueError(
            f"{what}={cp_min_tokens} requires VLLM_TPU_SOLO_PREFILL_OVER_TOKENS=N "
            "whose token bucket is the last bucket below it; got "
            f"N={solo}, buckets {sizes}"
        )


def _build_context_parallel_programs(
    model: Gemma4Model, vllm_config: VllmConfig, cp_min_tokens: int
) -> dict[int, Gemma4ContextParallelProgram]:
    config = vllm_config.compilation_config
    sizes = sorted(int(s) for s in config.compile_sizes)
    kept_tp_sizes = [s for s in sizes if s < cp_min_tokens]
    config.compile_sizes = list(kept_tp_sizes)
    outer_config = get_current_vllm_config().compilation_config
    if outer_config is not config and outer_config.compile_sizes:
        outer_config.compile_sizes = list(kept_tp_sizes)

    programs: dict[int, Gemma4ContextParallelProgram] = {}
    for size in (s for s in sizes if s >= cp_min_tokens):
        program_config = copy.copy(vllm_config)
        program_config.compilation_config = copy.copy(config)
        program_config.compilation_config.compile_sizes = [size]
        program_config.compilation_config.traced_files = set()
        program_config.compilation_config.cache_dir = ""
        program_config.compilation_config.local_cache_dir = ""
        with (
            set_current_vllm_config(program_config),
            set_model_tag(f"gemma4_cp{size}"),
        ):
            programs[size] = Gemma4ContextParallelProgram(
                vllm_config=program_config, model=model, num_tokens=size
            )
    return programs


def _patched_causal_lm_init(
    self: Gemma4ForCausalLM, *, vllm_config: VllmConfig, prefix: str = ""
) -> None:
    _orig_causal_lm_init(self, vllm_config=vllm_config, prefix=prefix)
    cp_min_tokens = envs.TPU_GEMMA4_CP_MIN_TOKENS
    if cp_min_tokens <= 0:
        return
    model: Any = self.model
    _check_context_parallel(model, vllm_config, cp_min_tokens)
    replica = Gemma4ProjectionReplica(
        model, vllm_config.quant_config, f"{maybe_prefix(prefix, 'model')}.{_REPLICA}"
    )
    for layer, copied in zip(model.layers, replica.layers):
        pairs = [
            (layer.self_attn.qkv_proj, copied.self_attn.qkv_proj),
            (layer.self_attn.o_proj, copied.self_attn.o_proj),
            (layer.mlp.gate_up_proj, copied.mlp.gate_up_proj),
            (layer.mlp.down_proj, copied.mlp.down_proj),
        ]
        if layer.enable_moe_block:
            pairs.append((layer.experts.routed_experts, copied.experts.routed_experts))
        for tp_module, module in pairs:
            if type(module.quant_method) is not type(tp_module.quant_method):
                raise ValueError(
                    f"{module.prefix} uses {type(module.quant_method).__name__}, "
                    f"but TP shard uses {type(tp_module.quant_method).__name__}"
                )
    if type(replica.embed_tokens.quant_method) is not type(
        model.embed_tokens.quant_method
    ):
        raise ValueError(
            "Embedding replica quant_method "
            f"{type(replica.embed_tokens.quant_method).__name__} does not match "
            f"{type(model.embed_tokens.quant_method).__name__}"
        )
    setattr(model, _REPLICA, replica)
    self._cp_programs = _build_context_parallel_programs(
        model, vllm_config, cp_min_tokens
    )
    self._cp_attention_prepared = False
    logger.info(
        "Built Gemma 4 context-parallel programs for token buckets %s; "
        "TP model compiles %s.",
        sorted(self._cp_programs),
        vllm_config.compilation_config.compile_sizes,
    )


def _profiling(model: Any) -> bool:
    attn = model.layers[0].self_attn.attn
    return get_attention_context(attn.layer_name)[2].numel() == 0


def _check_solo_prompt(num_tokens: int) -> None:
    context = get_vllm_model_wrapper_context()
    query_start_loc, seq_lens = context.query_start_loc_cpu, context.seq_lens_cpu
    if seq_lens is None or query_start_loc is None:
        return
    if len(seq_lens) != 1 or seq_lens[0] != query_start_loc[1]:
        raise RuntimeError(
            f"Context-parallel step of {num_tokens} tokens requires a single "
            "whole prompt without cached tokens; got query_start_loc="
            f"{query_start_loc.tolist()}, seq_lens={seq_lens.tolist()}"
        )


def _prepare_context_parallel_attention(model: Any, cp_buckets: Iterable[int]) -> None:
    world = get_tensor_model_parallel_world_size()
    min_rows = min(cp_buckets) // world
    replica: Gemma4ProjectionReplica = model._modules[_REPLICA]
    for layer, replica_layer in zip(model.layers, replica.layers):
        attn = layer.self_attn
        prepare_cp_prefill_attention(
            attn.attn,
            num_heads=attn.total_num_heads,
            num_kv_heads=attn.total_num_kv_heads,
            world=world,
            prologue_eps=attn.q_norm.variance_epsilon,
            o_proj=replica_layer.self_attn.o_proj,
            min_rows=min_rows,
        )


def _patched_causal_lm_forward(
    self: Gemma4ForCausalLM,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: Any = None,
    inputs_embeds: torch.Tensor | None = None,
    **kwargs: Any,
) -> Any:
    programs: dict[int, Gemma4ContextParallelProgram] | None = self.__dict__.get(
        "_cp_programs"
    )
    program = None if programs is None else programs.get(positions.shape[-1])
    if program is None or _profiling(self.model):
        return _orig_causal_lm_forward(
            self, input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
        )
    if input_ids is None or intermediate_tensors is not None or kwargs:
        raise NotImplementedError(
            "Context-parallel prefill supports token IDs only, got "
            f"inputs_embeds={inputs_embeds is not None}, "
            f"intermediate_tensors={intermediate_tensors is not None}, {list(kwargs)}"
        )
    _check_solo_prompt(positions.shape[-1])
    if not self._cp_attention_prepared:
        assert programs is not None
        _prepare_context_parallel_attention(self.model, programs.keys())
        self._cp_attention_prepared = True
    return program(input_ids, positions)


def _patched_mm_forward(
    self: Gemma4ForConditionalGeneration,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    intermediate_tensors: Any = None,
    inputs_embeds: torch.Tensor | None = None,
    **kwargs: Any,
) -> Any:
    lm = self.language_model
    programs: dict[int, Gemma4ContextParallelProgram] | None = lm.__dict__.get(
        "_cp_programs"
    )
    if (
        programs is not None
        and positions.shape[-1] in programs
        and not _profiling(lm.model)
    ):
        if inputs_embeds is not None or self.per_layer_embeddings is not None:
            raise NotImplementedError(
                "Context-parallel prefill supports text-only token IDs without "
                "multimodal or per-layer embeddings."
            )
        self._clear_mm_prefix_for_full_attn_layers()
        return lm(
            input_ids,
            positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=None,
            **kwargs,
        )
    return _orig_mm_forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=intermediate_tensors,
        inputs_embeds=inputs_embeds,
        **kwargs,
    )


def _patched_model_load_weights(
    self: Gemma4Model, weights: Iterable[tuple[str, torch.Tensor]]
) -> set[str]:
    if _REPLICA not in self._modules:
        return _orig_model_load_weights(self, weights)

    def with_replica() -> Iterator[tuple[str, torch.Tensor]]:
        for name, tensor in weights:
            yield name, tensor
            if _REPLICATED_TENSOR.match(name) or name.startswith("embed_tokens."):
                yield f"{_REPLICA}.{name}", tensor

    return _orig_model_load_weights(self, with_replica())


def apply_gemma4_context_parallel_patch(model_config: ModelConfig) -> None:
    """Install idempotent Gemma 4 context-parallel prefill patches."""
    cp_min_tokens = envs.TPU_GEMMA4_CP_MIN_TOKENS
    if cp_min_tokens <= 0:
        return
    architecture = getattr(model_config, "architecture", None)
    if architecture not in (
        "Gemma4ForCausalLM",
        "Gemma4ForConditionalGeneration",
    ):
        raise ValueError(
            "TPU_GEMMA4_CP_MIN_TOKENS requires Gemma4ForCausalLM or "
            f"Gemma4ForConditionalGeneration, got {architecture}"
        )
    if getattr(Gemma4ForCausalLM, "_TPU_CP_PATCHED", False):
        return
    Gemma4ForCausalLM.__init__ = _patched_causal_lm_init
    Gemma4ForCausalLM.forward = _patched_causal_lm_forward
    Gemma4ForConditionalGeneration.forward = _patched_mm_forward
    Gemma4Model.load_weights = _patched_model_load_weights
    Gemma4ForCausalLM._TPU_CP_PATCHED = True
    Gemma4ForConditionalGeneration._TPU_CP_PATCHED = True
    Gemma4Model._TPU_CP_PATCHED = True
    logger.info(
        "Applied TPU patch: Gemma 4 context-parallel prefill for buckets >= %d tokens.",
        cp_min_tokens,
    )
