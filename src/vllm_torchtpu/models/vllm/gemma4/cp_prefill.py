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
"""Opt-in zigzag context-parallel long-prompt prefill for Gemma 4 on TPU."""

from __future__ import annotations

import copy
import re
from collections.abc import Callable, Iterable, Iterator
from typing import Any

import torch
from torch import nn
from vllm.compilation.backends import set_model_tag
from vllm.compilation.decorators import support_torch_compile
from vllm.config import set_current_vllm_config
from vllm.config.utils import getattr_iter
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers import vocab_parallel_embedding
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.models.utils import maybe_prefix

from vllm_torchtpu import envs
from vllm_torchtpu.core.experimental.prefill_first_scheduler import (
    DEFAULT_SOLO_OVER_TOKENS,
    SCHEDULER_CLS,
    PrefillFirstScheduler,
)
from vllm_torchtpu.kernels.experimental.batched_rpa import cp_prefill as cp_kernel
from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8LinearMethodTPU
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.gemma4.layers import (
    FUSED_QUANT_DTYPE,
    all_gather_rows,
    apply_o_proj,
    configure_layer_fused_moe_head,
    configure_layer_fused_norm_quant,
    decoder_feedforward_tail,
    is_per_token_fp8,
    model_supports_multimodal,
    prequantized_linear,
    reduce_scatter_rows,
    rotary_cos_sin,
    split_norm_rotary_qkv,
    supports_qkv_prologue,
)
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    get_vllm_model_wrapper_context,
)

logger = init_logger(__name__)

CP_MIN_TOKENS = 16384
HBM_SAFETY_FLOOR_BYTES = 4 << 30
MAX_DEFAULT_EMBED_REPLICA_PROJ_BYTES = 26 << 30

_REPLICA = "cp_replica"
_REPLICATED_TENSOR = re.compile(
    r"layers\.\d+\.(self_attn\.(?:[qkvo]|qkv)_proj"
    r"|mlp\.(?:gate|up|gate_up|down)_proj|experts)\."
)

CPAttention = Callable[..., Any]
CPAttentionFromQKV = Callable[..., Any]


def uses_prefill_first_scheduler(scheduler_config: Any) -> bool:
    """Return True when ``scheduler_config`` selects ``PrefillFirstScheduler``."""
    sched_cls = scheduler_config.scheduler_cls
    if sched_cls is PrefillFirstScheduler or sched_cls == SCHEDULER_CLS:
        return True
    return isinstance(sched_cls, type) and issubclass(sched_cls, PrefillFirstScheduler)


def estimate_replica_bytes(model: Any, vllm_config: Any) -> tuple[int, int]:
    """Estimate per-rank ``(projection_bytes, embed_replica_bytes)`` of ``model``."""
    cfg = model.config
    quant_itemsize = 1 if vllm_config.quant_config is not None else 2
    proj_bytes = 0
    for layer in model.layers:
        attn = layer.self_attn
        qkv_out = (attn.total_num_heads + 2 * attn.total_num_kv_heads) * attn.head_dim
        o_in = attn.total_num_heads * attn.head_dim
        proj_bytes += (
            qkv_out * attn.hidden_size + o_in * attn.hidden_size
        ) * quant_itemsize
        inter = layer.mlp.down_proj.input_size
        proj_bytes += 3 * inter * layer.hidden_size * quant_itemsize
        if layer.enable_moe_block:
            moe_inter = getattr_iter(
                cfg, ("moe_intermediate_size", "expert_intermediate_size")
            )
            proj_bytes += (
                3 * cfg.num_experts * moe_inter * cfg.hidden_size * quant_itemsize
            )
    embed_dtype = model.embed_tokens.params_dtype
    embed_bytes = int(cfg.vocab_size) * int(cfg.hidden_size) * embed_dtype.itemsize
    return int(proj_bytes), int(embed_bytes)


def _query_free_hbm_bytes(vllm_config: Any) -> int | None:
    tpu_mod = getattr(torch, "tpu", None)
    mem_fn = getattr(tpu_mod, "mem_get_info", None)
    if callable(mem_fn):
        try:
            free_b, total_b = mem_fn()
        except RuntimeError:
            free_b = total_b = 0
        if total_b > 0:
            gmu = float(vllm_config.cache_config.gpu_memory_utilization)
            usable = int(total_b * gmu) - (total_b - free_b)
            return max(usable, 0)
    return None


def _layer_sliding_window(attn: Any) -> int | None:
    sw = getattr(attn, "sliding_window", None)
    if isinstance(sw, int) and sw > 0:
        return sw
    attn_mod = getattr(attn, "attn", None)
    impl = getattr(attn_mod, "impl", None)
    sw = getattr(impl, "sliding_window", None)
    return sw if isinstance(sw, int) and sw > 0 else None


def should_enable_context_parallel(
    model: Any, vllm_config: Any, *, free_hbm_bytes: int | None = None
) -> tuple[bool, bool, str]:
    """Return ``(enable_cp, include_embed_replica, skip_reason)``."""
    if vllm_config is None:
        return False, False, "missing vllm_config"
    if not uses_prefill_first_scheduler(vllm_config.scheduler_config):
        return False, False, "scheduler_cls is not PrefillFirstScheduler"
    parallel = vllm_config.parallel_config
    cache_cfg = vllm_config.cache_config
    checks = (
        ("multimodal enabled", model_supports_multimodal(vllm_config.model_config)),
        ("TP < 2", parallel.tensor_parallel_size < 2),
        ("PP > 1", parallel.pipeline_parallel_size > 1),
        ("DP > 1", parallel.data_parallel_size > 1),
        ("KV connector configured", vllm_config.kv_transfer_config is not None),
        ("LoRA configured", vllm_config.lora_config is not None),
        ("speculative decoding configured", vllm_config.speculative_config is not None),
        ("kv_sharing_fast_prefill enabled", bool(model.fast_prefill_enabled)),
        ("TPU_PRECOMPILE_PEER_REPLAY enabled", bool(envs.TPU_PRECOMPILE_PEER_REPLAY)),
        ("VLLM_TPU_MOST_MODEL_LEN set", envs.VLLM_TPU_MOST_MODEL_LEN is not None),
    )
    for reason, active in checks:
        if active:
            return False, False, reason

    has_moe = any(layer.enable_moe_block for layer in model.layers)
    if has_moe and (parallel.enable_ep_weight_filter or parallel.enable_eplb):
        return False, False, "MoE EP weight filter or EPLB enabled"
    for layer in model.layers:
        attn = layer.self_attn
        if attn.is_kv_shared_layer:
            return False, False, "KV-shared layers present"
        if layer.per_layer_input_gate is not None:
            return False, False, "per-layer input embeddings present"
        if attn.o_proj.bias is not None:
            return False, False, "attention output projection bias present"
        if not attn.rotary_emb.is_neox_style:
            return False, False, "non-neox rotary embedding"

    comp_sizes = vllm_config.compilation_config.compile_sizes or []
    sizes = sorted(int(s) for s in comp_sizes)
    above = [s for s in sizes if s >= DEFAULT_SOLO_OVER_TOKENS]
    if len(above) < 2 or not (above[0] < CP_MIN_TOKENS <= above[1]):
        return (
            False,
            False,
            f"compile_sizes {sizes} lack a clean split between "
            f"{DEFAULT_SOLO_OVER_TOKENS} and {CP_MIN_TOKENS}",
        )
    tp = parallel.tensor_parallel_size
    block_size = cache_cfg.block_size
    cp_sizes = [s for s in sizes if s >= CP_MIN_TOKENS]
    if any(s % (2 * tp * block_size) != 0 for s in cp_sizes):
        return (
            False,
            False,
            f"CP compile_sizes must be multiples of 2 * TP * block_size "
            f"({2 * tp * block_size})",
        )
    min_piece_pages = min(cp_sizes) // (2 * tp * block_size)
    for layer in model.layers:
        sw = _layer_sliding_window(layer.self_attn)
        if sw is not None:
            halo_pages = cp_kernel.num_halo_pages(sw, block_size)
            if halo_pages > min_piece_pages:
                return (
                    False,
                    False,
                    f"sliding halo ({halo_pages} pages) exceeds piece "
                    f"({min_piece_pages} pages)",
                )

    proj_bytes, embed_bytes = estimate_replica_bytes(model, vllm_config)
    avail = (
        _query_free_hbm_bytes(vllm_config) if free_hbm_bytes is None else free_hbm_bytes
    )
    if avail is not None:
        if avail < proj_bytes + HBM_SAFETY_FLOOR_BYTES:
            return (
                False,
                False,
                f"insufficient free HBM ({avail / (1 << 30):.1f} GiB < "
                f"{(proj_bytes + HBM_SAFETY_FLOOR_BYTES) / (1 << 30):.1f} GiB needed)",
            )
        with_embed = avail >= proj_bytes + embed_bytes + HBM_SAFETY_FLOOR_BYTES
        return True, with_embed, ""
    return True, proj_bytes <= MAX_DEFAULT_EMBED_REPLICA_PROJ_BYTES, ""


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
        inter = mlp.down_proj.input_size
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [inter, inter],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
            disable_tp=True,
        )
        self.down_proj = RowParallelLinear(
            inter,
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
            parallel = self.experts.moe_config.moe_parallel_config
            if parallel.ep_size == 1:
                parallel.use_ep = False


class Gemma4ProjectionReplica(nn.Module):
    """Unsharded per-rank projection copy exclusively for opt-in CP prefill."""

    def __init__(
        self, model: Any, quant_config: Any, prefix: str, *, include_embed: bool
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            _ReplicaLayer(layer, model.config, quant_config, f"{prefix}.layers.{i}")
            for i, layer in enumerate(model.layers)
        )
        self.embed_tokens: Any = (
            vocab_parallel_embedding.VocabParallelEmbedding(
                model.config.vocab_size,
                model.config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
                disable_tp=True,
            )
            if include_embed
            else None
        )


def _piece_rows(x: torch.Tensor, plan: Any, piece: int) -> torch.Tensor:
    size = plan.piece_tokens
    return x[piece * size : (piece + 1) * size]


def _rank_rows(x: torch.Tensor, plan: Any, rank: int) -> torch.Tensor:
    return torch.cat([_piece_rows(x, plan, p) for p in plan.rank_pieces(rank)])


def _rank_major(x: torch.Tensor, plan: Any) -> torch.Tensor:
    return torch.cat([_rank_rows(x, plan, r) for r in range(plan.world)])


def _token_order(rows: torch.Tensor, plan: Any) -> torch.Tensor:
    size = plan.piece_tokens
    blocks = []
    for piece in range(plan.num_pieces):
        rank, slot = plan.piece_owner(piece)
        start = (2 * rank + slot) * size
        blocks.append(rows[start : start + size])
    return torch.cat(blocks)


def _vocab_shard_embedding(embed: Any, input_ids: torch.Tensor) -> torch.Tensor:
    bounds = [
        getattr(embed, name, getattr(embed.shard_indices, name))
        for name in (
            "org_vocab_start_index",
            "org_vocab_end_index",
            "num_org_vocab_padding",
            "added_vocab_start_index",
            "added_vocab_end_index",
        )
    ]
    ids, outside = vocab_parallel_embedding.get_masked_input_and_mask(
        input_ids, *bounds
    )
    rows = embed.quant_method.embedding(embed, ids.long())
    return rows.masked_fill(outside.unsqueeze(-1), 0)


def _context_parallel_layer(
    layer: Any,
    replica: Any,
    positions: torch.Tensor,
    x: Any,
    plan: Any,
    rank: torch.Tensor,
    attention: CPAttention,
    attention_from_qkv: CPAttentionFromQKV | None,
    next_input_norm: Any,
    fused: bool,
) -> Any:
    attn = layer.self_attn
    if isinstance(x, tuple):
        residual, x_q, x_scale = x
        qkv = prequantized_linear(replica.self_attn.qkv_proj, x_q, x_scale)
    else:
        residual = x
        qkv, _ = replica.self_attn.qkv_proj(layer.input_layernorm(residual))

    if attention_from_qkv is not None and getattr(attn, "_tpu_cp_from_qkv", False):
        cos, sin = rotary_cos_sin(attn.rotary_emb, qkv, positions)
        attended = attention_from_qkv(
            attn.attn, qkv, cos, sin, attn.q_norm.weight, attn.k_norm.weight, plan, rank
        )
    else:
        q, k, v = split_norm_rotary_qkv(
            attn,
            positions,
            qkv,
            num_heads=attn.total_num_heads,
            num_kv_heads=attn.total_num_kv_heads,
        )
        attended = attention(attn.attn, q, k, v, plan, rank)

    out = apply_o_proj(replica.self_attn.o_proj, attended)
    return decoder_feedforward_tail(
        layer,
        out,
        residual,
        mlp_mod=replica.mlp,
        experts_mod=getattr(replica, "experts", None),
        project_prequant_gate_up=prequantized_linear,
        fused=fused,
        use_fused_moe_head=True,
        next_input_norm=next_input_norm,
    )


def _context_parallel_forward(
    model: Any,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    plan: Any,
    rank: torch.Tensor,
    rows: torch.Tensor,
    attention: CPAttention,
    attention_from_qkv: CPAttentionFromQKV | None,
) -> torch.Tensor:
    replica_mod = getattr(model, _REPLICA)
    embed = replica_mod.embed_tokens
    if embed is not None:
        x: Any = embed(input_ids.index_select(0, rows)) * model.normalizer
    else:
        shards = _vocab_shard_embedding(
            model.embed_tokens, _rank_major(input_ids, plan)
        )
        x = reduce_scatter_rows(shards) * model.normalizer
    pos_rows = positions.index_select(0, rows)
    pairs = list(zip(model.layers, replica_mod.layers))
    fused = bool(model._tpu_cp_fused_norms)
    for i, (layer, rep_layer) in enumerate(pairs):
        next_norm = (
            pairs[i + 1][0].input_layernorm if fused and i + 1 < len(pairs) else None
        )
        x = _context_parallel_layer(
            layer,
            rep_layer,
            pos_rows,
            x,
            plan,
            rank,
            attention,
            attention_from_qkv,
            next_norm,
            fused,
        )
    return _token_order(all_gather_rows(model.norm(x)), plan)


@support_torch_compile(dynamic_arg_dims={"input_ids": 0, "positions": 0})
class Gemma4ContextParallelProgram(nn.Module):
    """Compiled module executing one zigzag context-parallel token bucket."""

    compilation_config: Any

    def __init__(
        self, *, vllm_config: Any, prefix: str = "", model: Any, num_tokens: int
    ) -> None:
        super().__init__()
        del prefix
        from vllm_torchtpu.layers.adapter.cp_prefill_attention import (
            cp_prefill_attention,
            cp_prefill_attention_from_qkv,
        )

        self.__dict__["model"] = model
        self.plan = cp_kernel.make_zigzag_plan(
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
        self.attention: CPAttention = cp_prefill_attention
        self.attention_from_qkv: CPAttentionFromQKV = cp_prefill_attention_from_qkv

    def cleanup(self) -> None:
        return

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return _context_parallel_forward(
            self.model,
            input_ids,
            positions,
            self.plan,
            self.rank,
            self.rows,
            self.attention,
            self.attention_from_qkv,
        )


def _build_context_parallel_programs(
    model: Any, vllm_config: Any
) -> dict[int, Gemma4ContextParallelProgram]:
    config = vllm_config.compilation_config
    sizes = sorted(int(s) for s in config.compile_sizes)
    config.compile_sizes = [s for s in sizes if s < CP_MIN_TOKENS]
    programs: dict[int, Gemma4ContextParallelProgram] = {}
    for size in (s for s in sizes if s >= CP_MIN_TOKENS):
        program_config = copy.copy(vllm_config)
        program_config.compilation_config = copy.copy(config)
        program_config.compilation_config.compile_sizes = [size]
        program_config.compilation_config.traced_files = set()
        program_config.compilation_config.cache_dir = ""
        program_config.compilation_config.local_cache_dir = ""
        with set_current_vllm_config(program_config), set_model_tag(f"gemma4_cp{size}"):
            prog = Gemma4ContextParallelProgram(
                vllm_config=program_config, model=model, num_tokens=size
            )
        prog.compilation_config.cache_dir = ""
        prog.compilation_config.local_cache_dir = None
        programs[size] = prog
    return programs


def _configure_cp_fused_norms(model: Any, replica: Gemma4ProjectionReplica) -> bool:
    rep_layers: Any = replica.layers
    for layer, copied in zip(model.layers, rep_layers):
        if not (
            is_per_token_fp8(
                copied.self_attn.qkv_proj, expected_dtype=FUSED_QUANT_DTYPE
            )
            and is_per_token_fp8(
                copied.mlp.gate_up_proj, expected_dtype=FUSED_QUANT_DTYPE
            )
        ):
            return False
        if getattr(layer, "fused_norm_quant", None) is None and not (
            configure_layer_fused_norm_quant(layer)
        ):
            return False
        if not configure_layer_fused_moe_head(layer):
            return False
    return True


def _validate_replica_quant_methods(
    model: Any, replica: Gemma4ProjectionReplica
) -> None:
    rep_layers: Any = replica.layers
    for layer, copied in zip(model.layers, rep_layers):
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
    if replica.embed_tokens is not None and type(
        replica.embed_tokens.quant_method
    ) is not type(model.embed_tokens.quant_method):
        raise ValueError("Embedding replica quant_method does not match TP embedding.")


def _patched_causal_lm_init(self: Any, *, vllm_config: Any, prefix: str = "") -> None:
    type(self)._tpu_orig_init(self, vllm_config=vllm_config, prefix=prefix)
    self._cp_programs = {}
    self._cp_attention_prepared = False
    self._max_non_cp_tokens = DEFAULT_SOLO_OVER_TOKENS
    if prefix:
        return
    model = self.model
    enabled, include_embed, reason = should_enable_context_parallel(model, vllm_config)
    if not enabled:
        if uses_prefill_first_scheduler(vllm_config.scheduler_config):
            logger.info(
                "Skipping Gemma 4 context-parallel prefill replica (%s); "
                "long solo steps will use tensor/sequence parallelism.",
                reason,
            )
        return
    replica = Gemma4ProjectionReplica(
        model,
        vllm_config.quant_config,
        f"{maybe_prefix(prefix, 'model')}.{_REPLICA}",
        include_embed=include_embed,
    )
    _validate_replica_quant_methods(model, replica)
    model._tpu_cp_fused_norms = _configure_cp_fused_norms(model, replica)
    setattr(model, _REPLICA, replica)
    self._cp_programs = _build_context_parallel_programs(model, vllm_config)
    non_cp = vllm_config.compilation_config.compile_sizes
    self._max_non_cp_tokens = max(non_cp) if non_cp else DEFAULT_SOLO_OVER_TOKENS
    logger.info(
        "Gemma 4 context-parallel programs built for buckets %s "
        "(embed replica: %s, fused norms: %s); TP compiles %s.",
        sorted(self._cp_programs),
        include_embed,
        model._tpu_cp_fused_norms,
        non_cp,
    )


def _profiling(model: Any) -> bool:
    attn = model.layers[0].self_attn.attn
    return get_attention_context(attn.layer_name)[2].numel() == 0


def _check_solo_prompt(num_tokens: int) -> None:
    context = get_vllm_model_wrapper_context()
    query_start_loc = context.query_start_loc_cpu
    seq_lens = context.seq_lens_cpu
    if seq_lens is None and query_start_loc is None:
        return
    if seq_lens is None or query_start_loc is None:
        raise RuntimeError(
            f"Context-parallel step of {num_tokens} tokens is missing host "
            "query_start_loc / seq_lens metadata."
        )
    if len(seq_lens) != 1 or seq_lens[0] != query_start_loc[1]:
        raise RuntimeError(
            f"Context-parallel step of {num_tokens} tokens requires one uncached "
            f"prompt; got query_start_loc={query_start_loc.tolist()}, "
            f"seq_lens={seq_lens.tolist()}"
        )


def _prepare_context_parallel_attention(model: Any, min_rows: int) -> None:
    from vllm_torchtpu.layers.adapter.cp_prefill_attention import (
        prepare_cp_prefill_attention,
    )

    world = get_tensor_model_parallel_world_size()
    replica: Gemma4ProjectionReplica = getattr(model, _REPLICA)
    rep_layers: Any = replica.layers
    for layer, rep_layer in zip(model.layers, rep_layers):
        attn = layer.self_attn
        use_qkv = supports_qkv_prologue(attn, attn.attn.impl)
        o_proj = rep_layer.self_attn.o_proj
        o_method = o_proj.quant_method
        fp8_out = (
            use_qkv
            and is_per_token_fp8(o_proj, expected_dtype=torch.float8_e4m3fn)
            and isinstance(o_method, VllmFp8LinearMethodTPU)
            and o_method.supports_quantized_input(o_proj, min_rows)
        )
        attn._tpu_cp_from_qkv = use_qkv
        prepare_cp_prefill_attention(
            attn.attn,
            num_heads=attn.total_num_heads,
            num_kv_heads=attn.total_num_kv_heads,
            world=world,
            prologue_eps=attn.q_norm.variance_epsilon if use_qkv else None,
            fp8_out_rows=fp8_out,
        )


def _patched_causal_lm_forward(
    self: Any,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: Any = None,
    inputs_embeds: torch.Tensor | None = None,
    **kwargs: Any,
) -> Any:
    programs: dict[int, Any] = self.__dict__.get("_cp_programs") or {}
    num_tokens = positions.shape[-1]
    program = programs.get(num_tokens) if isinstance(num_tokens, int) else None
    if (
        len(programs) > 0
        and input_ids is not None
        and inputs_embeds is None
        and intermediate_tensors is None
        and not kwargs
        and _profiling(self.model)
    ):
        cap = int(getattr(self, "_max_non_cp_tokens", DEFAULT_SOLO_OVER_TOKENS))
        if num_tokens > cap:
            vllm_cfg = getattr(self, "vllm_config", None) or getattr(
                self.model, "vllm_config", None
            )
            cc = getattr(vllm_cfg, "compilation_config", None)
            saved_sizes = cc.compile_sizes if cc is not None else None
            if cc is not None:
                cc.compile_sizes = [cap]
            try:
                sliced = type(self)._tpu_orig_forward(
                    self, input_ids[:cap], positions[:cap], None, None
                )
            finally:
                if cc is not None and saved_sizes is not None:
                    cc.compile_sizes = saved_sizes
            out = sliced.new_zeros((num_tokens, *sliced.shape[1:]))
            out[:cap] = sliced
            return out
        return type(self)._tpu_orig_forward(
            self, input_ids, positions, intermediate_tensors, inputs_embeds
        )
    if (
        program is None
        or input_ids is None
        or inputs_embeds is not None
        or intermediate_tensors is not None
        or kwargs
        or _profiling(self.model)
    ):
        return type(self)._tpu_orig_forward(
            self, input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
        )
    _check_solo_prompt(num_tokens)
    if not self._cp_attention_prepared:
        min_rows = min(programs) // get_tensor_model_parallel_world_size()
        _prepare_context_parallel_attention(self.model, min_rows)
        self._cp_attention_prepared = True
    if program.rank.device != input_ids.device:
        program.to(input_ids.device)
    return program(input_ids, positions)


def _patched_model_load_weights(
    self: Any, weights: Iterable[tuple[str, torch.Tensor]]
) -> set[str]:
    replica = getattr(self, _REPLICA, None)
    if replica is None:
        loaded = type(self)._tpu_orig_load_weights(self, weights)
    else:
        has_embed = replica.embed_tokens is not None

        def with_replica() -> Iterator[tuple[str, torch.Tensor]]:
            for name, tensor in weights:
                yield name, tensor
                if _REPLICATED_TENSOR.match(name) or (
                    has_embed and name.startswith("embed_tokens.")
                ):
                    yield f"{_REPLICA}.{name}", tensor

        loaded = type(self)._tpu_orig_load_weights(self, with_replica())
    if isinstance(self, nn.Module):
        from vllm_torchtpu.runner import utils as runner_utils

        runner_utils.upcast_norm_weights(self)
    return loaded


def apply_gemma4_context_parallel_patch(gemma4_mod: Any) -> bool:
    """Install opt-in context-parallel prefill hooks idempotently."""
    causal_cls = gemma4_mod.Gemma4ForCausalLM
    model_cls = gemma4_mod.Gemma4Model
    if getattr(causal_cls, "_TPU_GEMMA4_CP_PATCHED", False):
        return False
    causal_cls._tpu_orig_init = causal_cls.__init__
    causal_cls._tpu_orig_forward = causal_cls.forward
    model_cls._tpu_orig_load_weights = model_cls.load_weights
    causal_cls.__init__ = _patched_causal_lm_init
    causal_cls.forward = _patched_causal_lm_forward
    model_cls.load_weights = _patched_model_load_weights
    causal_cls._TPU_GEMMA4_CP_PATCHED = True
    model_cls._TPU_GEMMA4_CP_PATCHED = True
    return True
