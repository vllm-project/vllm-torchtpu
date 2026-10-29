# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark draft model for DeepSeek-V4.1 on TPU."""

import re
from collections.abc import Iterable
from typing import cast

import torch
import torch.nn as nn
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.fused_moe import (
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen3_dspark import (
    DSparkConfidenceHead,
    DSparkMarkovHead,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.models.deepseek_v41.nvidia import model as upstream

from vllm_torchtpu.layers.adapter.quantization.deepseek_v41 import DeepseekV41MoE
from vllm_torchtpu.logger import init_logger

from ..common.mhc import (
    collapse,
    initialize_delayed_mhc,
    initialize_mhc_post,
    pack_mix_weights,
    post,
)
from .attention import (
    DeepseekV41Attention,
    _cache_slots,
    _insert,
    _token_requests,
    _uses_qa_rope,
    stage_attention_cache_scope,
)
from .attention_kernels import initialize_indexer, initialize_paged_attention
from .cache import initialize_cache_ops
from .engram import Engram, NgramHashState
from .model import DeepseekV41DecoderLayer
from .ops import compact_rope_caches, device_hooks

logger = init_logger(__name__)

_EXPERT_SCALE_RE = re.compile(r"\.experts\.\d+\.w[123]\.scale$")


def get_dspark_swa_index_width(window_size: int, dspark_block_size: int) -> int:
    """Padded 64-aligned SWA key width covering window_size + dspark_block_size."""
    return -(-(window_size + dspark_block_size) // 64) * 64


def _remap_dspark_name(name: str, *, enable_confidence_head: bool = True) -> str | None:
    """Map a checkpoint ``mtp.{i}.*`` name to this model's parameter path.

    Returns None for non-mtp weights (owned by the target model).
    """
    m = re.match(r"mtp\.(\d+)\.(.*)", name)
    if m is None:
        return None
    stage = int(m.group(1))
    rest = m.group(2)
    if rest.startswith("confidence_head.") and not enable_confidence_head:
        return None
    if rest.startswith("markov_head.embed."):
        return "model.markov_head.markov_w1." + rest.removeprefix("markov_head.embed.")
    if rest.startswith("markov_head.head."):
        return "model.markov_head.markov_w2." + rest.removeprefix("markov_head.head.")
    head_prefixes = (
        "norm.",
        "markov_head.",
        "confidence_head.",
    )
    if rest.startswith(("main_proj.", "main_norm.")) or rest.startswith(head_prefixes):
        return f"model.{rest}"
    if rest.endswith("ffn.gate.bias_vl"):
        rest = rest.removesuffix("ffn.gate.bias_vl") + "ffn.gate.bias"
    return f"model.layers.{stage}.{rest}"


def _insert_context_kv(
    attn: DeepseekV41Attention,
    kv: torch.Tensor,
    positions: torch.Tensor,
    metadata,
    *,
    rotated: bool = False,
) -> None:
    """Rotate (unless ``rotated``) and write normalized context KV rows into
    one draft layer's SWA cache."""
    swa_cache = attn.swa_cache_layer.kv_cache
    if metadata is None or not swa_cache.numel():
        return
    if not rotated:
        kv = attn._rotate(kv, positions)
    if isinstance(metadata, torch.Tensor):
        swa_slots = metadata
        swa_valid = swa_slots >= 0
    else:
        swa_slots = _cache_slots(swa_cache, metadata, positions)
        if metadata.seq_lens.numel() > 0:
            requests = _token_requests(metadata, positions).clamp(
                max=metadata.seq_lens.shape[0] - 1
            )
            offsets = positions - metadata.seq_lens[requests] + attn.window_size
            swa_valid = (swa_slots >= 0) & (offsets >= 0) & (offsets < attn.window_size)
        else:
            swa_valid = swa_slots >= 0
    _insert(swa_cache, swa_slots, kv, swa_valid)


class DSparkDeepseekV41Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        initialize_cache_ops()
        initialize_delayed_mhc()
        initialize_mhc_post()
        from .dspark_sampling import initialize_collapse

        self._boundary_collapse = initialize_collapse()

        initialize_paged_attention()
        initialize_indexer()

        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.hc_eps = config.hc_eps
        self.rms_norm_eps = config.rms_norm_eps
        self.num_hidden_layers = config.num_hidden_layers
        self.target_layer_ids = tuple(config.dspark_target_layer_ids)
        self.num_dspark_layers = (
            getattr(config, "n_mtp_layers", None)
            or getattr(config, "num_nextn_predict_layers", None)
            or 3
        )

        # Aliased to the target's embedding by the proposer's sharing utility
        # after load (has_own_embed_tokens=False).
        self.embed_tokens: nn.Module | None = None

        self.main_proj = ReplicatedLinear(
            config.hidden_size * len(self.target_layer_ids),
            config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "main_proj"),
        )
        self.main_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
        )

        current_vllm_config = get_current_vllm_config()
        with (
            device_hooks(
                upstream,
                _select_dsv4_attn_cls=lambda _: DeepseekV41Attention,
                Engram=Engram,
                NgramHashState=NgramHashState,
                DeepseekV4DecoderLayer=DeepseekV41DecoderLayer,
                DeepseekV4MoE=DeepseekV41MoE,
                _use_sequence_parallel=lambda _: False,
            ),
            device_hooks(torch.cuda, Stream=lambda: None),
        ):
            self.layers = nn.ModuleList(
                [
                    DeepseekV41DecoderLayer(
                        current_vllm_config,
                        prefix=maybe_prefix(
                            prefix, f"layers.{self.num_hidden_layers + i}"
                        ),
                        topk_indices_buffer=self.topk_indices_buffer,
                    )
                    for i in range(self.num_dspark_layers)
                ]
            )

        dspark_block_size = (
            getattr(config, "dspark_block_size", None)
            or getattr(config, "block_size", None)
            or vllm_config.speculative_config.num_speculative_tokens
        )
        max_block_rows = vllm_config.speculative_config.num_speculative_tokens + 1
        for layer in cast(Iterable[DeepseekV41DecoderLayer], self.layers):
            if max_block_rows > layer.attn.window_size:
                raise ValueError(
                    f"DSpark draft block of up to {max_block_rows} rows "
                    f"exceeds the DSV4.1 sliding window ({layer.attn.window_size})."
                )
            if layer.attn._csa2 and max_block_rows > 64:
                raise ValueError("CSA2 supports DSpark blocks of at most 64 queries")
            layer.attn.non_causal_multi_token_decode = True
            layer.attn.dspark_block_size = int(dspark_block_size)

        compact_rope_caches(
            (layer.attn for layer in self.layers),
            vllm_config.model_config.max_model_len,
        )

        # V4.1 collapses hyper-connections with the last layer's FFN pre-mix
        # instead of a learned hc_head.
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        draft_vocab_size = (
            getattr(config, "draft_vocab_size", None) or config.vocab_size
        )
        self.markov_head = DSparkMarkovHead(
            config.vocab_size,
            draft_vocab_size,
            config.dspark_markov_rank,
            prefix=maybe_prefix(prefix, "markov_head"),
        )
        self.confidence_head: DSparkConfidenceHead | None = None
        if getattr(config, "enable_confidence_head", True):
            self.confidence_head = DSparkConfidenceHead(
                config.hidden_size + config.dspark_markov_rank,
                prefix=maybe_prefix(prefix, "confidence_head"),
            )

    def finalize_mhc_broadcast_weights(self) -> None:
        layers = list(self.layers)
        if layers and layers[0].hc_attn_fn.dtype != torch.float32:
            return
        pack_mix_weights(layers)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        assert self.embed_tokens is not None
        return self.embed_tokens(input_ids)

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        """main_x = main_norm(main_proj(concat of target aux hidden states))."""
        return self.main_norm(self.main_proj(aux_hidden_states))

    def get_draft_attn_causal(self) -> list[bool]:
        return [False] * len(self.layers)

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [
            layer.attn.swa_cache_layer.prefix
            for layer in cast(Iterable[DeepseekV41DecoderLayer], self.layers)
        ]

    def precompute_and_store_context_kv(
        self,
        target_hidden_states: torch.Tensor,
        positions: torch.Tensor,
        metadata: tuple | list | None = None,
    ) -> torch.Tensor:
        """Insert the sliding-window context KV for every draft layer."""
        if target_hidden_states.shape[-1] != self.config.hidden_size:
            main_x = self.combine_hidden_states(target_hidden_states)
        else:
            main_x = target_hidden_states
        for i, layer in enumerate(cast(Iterable[DeepseekV41DecoderLayer], self.layers)):
            attn = layer.attn
            rotated = False
            if _uses_qa_rope(attn, main_x):
                # kv-only: skips the qr columns, norm and projection.
                _, kv, rotated = attn._qa_kv_norm_rope(main_x, positions, emit_qr=False)
            else:
                qr_kv = attn._fused_wqa_wkv_gemm(main_x)
                _, _, kv = attn._split_qkv_and_norm(qr_kv)
            layer_md = None if metadata is None else metadata[i]
            _insert_context_kv(attn, kv, positions, layer_md, rotated=rotated)
        return main_x

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        hidden_states = inputs_embeds.unsqueeze(-2).repeat(1, self.hc_mult, 1)

        residual = post_mix = res_mix = pre_mix = None
        with stage_attention_cache_scope():
            for layer in self.layers:
                hidden_states, residual, post_mix, res_mix, pre_mix, _ = layer(
                    hidden_states,
                    positions,
                    input_ids,
                    pre_mix,
                    post_mix,
                    res_mix,
                    residual,
                )
        hidden_states = post(hidden_states, residual, post_mix, res_mix)
        assert pre_mix is not None
        if (
            hidden_states.shape == (240, 4, 5120)
            and hidden_states.dtype == torch.bfloat16
            and pre_mix.shape == (240, 4)
            and pre_mix.dtype == torch.float32
        ):
            return self._boundary_collapse(hidden_states.flatten(1), pre_mix)
        return collapse(hidden_states, pre_mix)


class DSparkDeepseekV41ForCausalLM(nn.Module):
    """Registry entry for ``architectures=["DSparkV41DraftModel"]`` on TPU."""

    # Its MoE layers use EP even when the draft itself has TP=1.
    requires_dp_collective_pairing = True
    has_own_embed_tokens = False
    has_own_lm_head = False
    owns_context_kv = True
    draft_id_to_target_id = None

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        self.quant_config = vllm_config.quant_config
        # TPU FP8 linear adapters register weight_scale_inv.
        self.linear_scale_name = "weight_scale_inv"
        self.pad_shared_expert = (
            getattr(self.quant_config, "weight_block_size", None) is not None
        )
        self.model = DSparkDeepseekV41Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        # Aliased to the target's lm_head by the proposer's sharing utility.
        self.lm_head: nn.Module | None = None
        self.logits_processor = LogitsProcessor(self.config.vocab_size)

    # --- Hooks consumed by DFlashProposer / DSparkProposer -----------------

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.combine_hidden_states(aux_hidden_states)

    def get_draft_attn_causal(self) -> list[bool]:
        return self.model.get_draft_attn_causal()

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return self.model.get_draft_kv_cache_layer_names()

    def precompute_and_store_context_kv(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        metadata: tuple | list | None = None,
    ) -> torch.Tensor:
        return self.model.precompute_and_store_context_kv(
            hidden_states, positions, metadata
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    def prepare_tpu_draft_sampler(
        self, *, tp_size: int, block_size: int, sample_from_anchor: bool
    ):
        """Prepare fused sampling after the target LM weight is shared.

        None means the proposer retains its ordinary logits/Markov path.
        The returned sampler may also return None for an unsupported bucket.
        """
        from .dspark_sampling import initialize_samples

        self._decode_samples = None
        if not (tp_size == 1 and sample_from_anchor and block_size == 5):
            return None
        assert self.lm_head is not None
        weight = cast(torch.Tensor, self.lm_head.weight)
        markov = self.model.markov_head
        if not (
            weight.shape == (129280, 5120)
            and weight.dtype == torch.bfloat16
            and markov.markov_w2.weight.shape == (129280, 256)
            and markov.markov_w2.weight.dtype == torch.bfloat16
            and markov.markov_w1.weight.shape == (129280, 256)
            and markov.markov_w1.weight.dtype == torch.bfloat16
        ):
            return None
        self._decode_samples = initialize_samples()
        return self._sample_decode

    def _sample_decode(
        self, hidden: torch.Tensor, input_ids: torch.Tensor, block_size: int
    ) -> torch.Tensor | None:
        if (
            hidden.shape != (240, 5120)
            or hidden.dtype != torch.bfloat16
            or block_size != 5
        ):
            return None
        slot_hidden = hidden.view(48, 5, 5120).transpose(0, 1).reshape(240, 5120)
        logits = self.compute_logits(slot_hidden)
        if logits.shape != (240, 129280) or logits.dtype != torch.bfloat16:
            return None
        markov = self.model.markov_head
        anchors = input_ids[:240].view(48, 5)[:, 0]
        assert self._decode_samples is not None
        return self._decode_samples(
            logits,
            markov.markov_w1.weight,
            markov.markov_w2.weight,
            anchors,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Base logits U_k = lm_head(norm(head_hidden))."""
        assert self.lm_head is not None
        return self.logits_processor(self.lm_head, self.model.norm(hidden_states))

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.compute_logits(hidden_states)

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return draft_ids

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.bias(markov_embed, self.logits_processor)

    def compute_confidence(
        self, head_hidden: torch.Tensor, markov_embed: torch.Tensor
    ) -> torch.Tensor:
        assert self.model.confidence_head is not None
        return torch.sigmoid(self.model.confidence_head(head_hidden, markov_embed))

    # --- Weight loading -----------------------------------------------------

    def should_load_weight(self, name: str) -> bool:
        """Let the streaming loader reject unused tensors before reading them."""
        return self._remap_dspark_name(name) is not None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load the ``mtp.{0,1,2}.*`` draft weights from the target checkpoint."""
        first_layer = self.model.layers[0]
        use_native_mega_moe = first_layer.ffn.use_native_mega_moe
        n_draft_experts = (
            getattr(self.config, "dspark_n_routed_experts", 0)
            or self.config.n_routed_experts
        )
        if use_native_mega_moe:
            expert_mapping = upstream.make_deepseek_v4_expert_params_mapping(
                n_draft_experts
            )
        else:
            expert_mapping = fused_moe_make_expert_params_mapping(
                self,
                ckpt_gate_proj_name="w1",
                ckpt_down_proj_name="w2",
                ckpt_up_proj_name="w3",
                num_experts=n_draft_experts,
            )
        expert_scale_suffix = (
            ".weight_scale"
            if getattr(self.config, "expert_dtype", "fp4") == "fp4"
            else ".weight_scale_inv"
        )

        stacked_params_mapping = [
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
            ("attn.fused_wqa_wkv", "attn.wq_a", 0),
            ("attn.fused_wqa_wkv", "attn.wkv", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        loaded_confidence_head = False

        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        n_local_head = self.config.num_attention_heads // tp_size
        head_start = n_local_head * tp_rank
        head_end = n_local_head * (tp_rank + 1)

        for name, loaded_weight in weights:
            mapped = self._remap_dspark_name(name)
            if mapped is None:
                continue
            name = mapped
            if "confidence_head." in name:
                loaded_confidence_head = True

            if name.endswith(".scale"):
                suffix = (
                    expert_scale_suffix
                    if _EXPERT_SCALE_RE.search(name)
                    else f".{self.linear_scale_name}"
                )
                name = name.removesuffix(".scale") + suffix
            if ".shared_experts.w2" in name:
                name = name.replace(".shared_experts.w2", ".shared_experts.down_proj")
            if self.pad_shared_expert and ".shared_experts." in name:
                loaded_weight = upstream.DeepseekV4Model._pad_shared_expert_weight(
                    self.quant_config, name, loaded_weight
                )

            if ".experts." in name:
                if (
                    "weight_scale" in name
                    and loaded_weight.dtype == torch.float8_e8m0fnu
                ):
                    loaded_weight = loaded_weight.view(torch.uint8)
                for param_name, weight_name, expert_id, shard_id in expert_mapping:
                    if weight_name not in name:
                        continue
                    name_mapped = name.replace(weight_name, param_name)
                    param = params_dict[name_mapped]
                    success = param.weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        loaded_params.add(name_mapped)
                        break
                continue

            is_layer_param = name.startswith("model.layers.")
            for param_name, weight_name, stacked_shard_id in stacked_params_mapping:
                if not is_layer_param or weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, stacked_shard_id)
                loaded_params.add(name)
                break
            else:
                if "attn_sink" in name:
                    narrow = loaded_weight[head_start:head_end]
                    params_dict[name].data[: narrow.shape[0]].copy_(narrow)
                    loaded_params.add(name)
                    continue
                if name.endswith(".ffn.gate.bias"):
                    name = name.replace(
                        ".ffn.gate.bias", ".ffn.gate.e_score_correction_bias"
                    )
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        if self.model.confidence_head is not None and not loaded_confidence_head:
            self.model.confidence_head = None
        self.process_weights_after_loading()
        logger.info_once(
            "DSpark V4.1 draft model loaded: %d params", len(loaded_params)
        )
        return loaded_params

    def process_weights_after_loading(self) -> None:
        for layer in self.model.layers:
            layer.ffn.finalize_mega_moe_weights()
        self.model.finalize_mhc_broadcast_weights()

    def _remap_dspark_name(self, name: str) -> str | None:
        return _remap_dspark_name(
            name, enable_confidence_head=self.model.confidence_head is not None
        )
