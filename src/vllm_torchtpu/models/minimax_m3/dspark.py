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
"""MiniMax M3.1 DSpark drafter on TPU.

The checkpoint's ``dspark/`` folder holds a 5-layer dense block drafter: the
Qwen3 DSpark design (context K/V projected from five target layers, a
non-causal query block, Markov and confidence heads) built from MiniMax M3
layers. Against the Qwen3 drafter it changes the norms to Gemma RMSNorm
(``x * (1 + w)``), RoPE to partial NeoX (64 of 128 dims), and the MLP to the
dense SwiGLU-OAI one. Embedding and LM head are shared with the target.

Served by the TPU `DSparkProposer`; the context K/V write is model-owned
(`tpu_precompute_and_store_context_kv`) because the generic one assumes plain
RMSNorm and an unquantized fused K/V weight.
"""

import copy
from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import tensor_model_parallel_all_gather
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
)
from vllm.model_executor.models.qwen3_dflash import (
    DFlashQwen3Attention,
    DFlashQwen3DecoderLayer,
    _resolve_layer_attention,
)
from vllm.model_executor.models.qwen3_dspark import (
    Qwen3DSparkForCausalLM,
    Qwen3DSparkModel,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.transformers_utils.configs.minimax_m3 import MiniMaxM3TextConfig

from vllm_torchtpu.layers.core.quant_methods import get_tpu_quant_method
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.minimax_m3.moe import MiniMaxM3MLP
from vllm_torchtpu.models.minimax_m3.quant import QUANT_METHOD
from vllm_torchtpu.spec_decode.vocab_parallel_argmax import (
    MAX_VOCAB_FOR_FLOAT_IDS,
    reduce_shard_argmax_pairs,
    shard_argmax_pair,
)

if TYPE_CHECKING:
    from vllm.config import SpeculativeConfig

    from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata

logger = init_logger(__name__)

ARCHITECTURE = "DSparkMiniMaxDraftModel"
_MODEL_TYPE = "minimax_m3_dspark"
# Checkpoint tensor names: language_model.model.dspark.<module>.
CHECKPOINT_PREFIX = "language_model.model.dspark."


def configure_dspark_draft(speculative_config: "SpeculativeConfig") -> None:
    """Point the draft config at this drafter. Idempotent.

    vLLM 0.32 knows no MiniMax DSpark: it promotes the draft's ``text_config``
    as if it were a MiniMax MTP head, then (unknown DSpark architecture) tags
    it as the DeepSeek-V4 DSpark draft. Undo both, as later vLLM does: keep
    the text config, the MiniMax DSpark architecture, non-causal block
    attention, and the checkpoint's MXFP8 quantization on the TPU config.
    """
    if speculative_config.method != "dspark":
        raise NotImplementedError(
            "MiniMax M3 on TPU supports only the checkpoint's DSpark drafter "
            f"(speculative method 'dspark'), got {speculative_config.method!r}."
        )
    draft_model_config = speculative_config.draft_model_config
    hf_config = draft_model_config.hf_config
    if (text_config := getattr(hf_config, "text_config", None)) is not None:
        quantization_config = getattr(hf_config, "quantization_config", None)
        hf_config = text_config
        if getattr(hf_config, "quantization_config", None) is None:
            hf_config.quantization_config = quantization_config
    if not isinstance(hf_config, MiniMaxM3TextConfig) or not getattr(
        hf_config, "dspark_target_layer_ids", None
    ):
        raise NotImplementedError(
            "MiniMax M3 DSpark on TPU needs the checkpoint's dspark/ draft "
            f"(architecture {ARCHITECTURE}); got model "
            f"{draft_model_config.model!r}."
        )
    dflash_config = dict(getattr(hf_config, "dflash_config", None) or {})
    # The query block attends to itself bidirectionally.
    dflash_config.setdefault("causal", False)
    hf_config.model_type = _MODEL_TYPE
    hf_config.architectures = [ARCHITECTURE]
    hf_config.dflash_config = dflash_config
    hf_config.markov_rank = hf_config.dspark_markov_rank
    draft_model_config.hf_config = hf_config
    quantization_config = getattr(hf_config, "quantization_config", None) or {}
    if quantization_config.get("quant_method") == QUANT_METHOD:
        draft_model_config.quantization = get_tpu_quant_method(QUANT_METHOD)
    speculative_config.update_arch_()


def _per_head_norm(x: torch.Tensor, norm: nn.Module, head_dim: int) -> torch.Tensor:
    return norm(x.reshape(-1, head_dim)).reshape(x.shape)


class MiniMaxM3DSparkAttention(DFlashQwen3Attention):
    """Full GQA attention with Gemma q/k norms and partial NeoX RoPE.

    The inherited forward (qkv -> per-head norms -> RoPE -> attention ->
    o_proj) applies unchanged once the norms are Gemma ones.
    """

    def __init__(
        self,
        config: MiniMaxM3TextConfig,
        sliding_window: int | None,
        causal: bool,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        if cache_config is not None and cache_config.cache_dtype == "int4_per_token_head":
            # Only the target's MSA caches are 4-bit; the draft's KV stays fp8.
            cache_config = copy.copy(cache_config)
            cache_config.cache_dtype = "fp8"
        super().__init__(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_parameters={
                "rope_theta": config.rope_theta,
                "partial_rotary_factor": config.partial_rotary_factor,
            },
            max_position=config.max_position_embeddings,
            head_dim=config.head_dim,
            rms_norm_eps=config.rms_norm_eps,
            sliding_window=sliding_window,
            causal=causal,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)


class MiniMaxM3DSparkDecoderLayer(DFlashQwen3DecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        config: MiniMaxM3TextConfig,
        layer_idx: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        if not getattr(config, "dspark_use_dense_ffn", False):
            raise NotImplementedError("MiniMax DSpark supports dense FFN drafts only.")
        self.hidden_size = config.hidden_size
        sliding_window, causal = _resolve_layer_attention(config, layer_idx)
        self.self_attn = MiniMaxM3DSparkAttention(
            config,
            sliding_window=sliding_window,
            causal=causal,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = MiniMaxM3MLP(
            config,
            config.dspark_ffn_hidden_size,
            quant_config,
            prefix=f"{prefix}.mlp",
        )
        hidden_size, eps = config.hidden_size, config.rms_norm_eps
        self.input_layernorm = GemmaRMSNorm(hidden_size, eps=eps)
        self.post_attention_layernorm = GemmaRMSNorm(hidden_size, eps=eps)


class MiniMaxM3DSparkModel(Qwen3DSparkModel):
    decoder_layer_cls = MiniMaxM3DSparkDecoderLayer

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config, start_layer_id=start_layer_id, prefix=prefix
        )
        hidden_size, eps = self.config.hidden_size, self.config.rms_norm_eps
        self.hidden_norm = GemmaRMSNorm(hidden_size, eps=eps)
        self.norm = GemmaRMSNorm(hidden_size, eps=eps)

    def store_context_kv(
        self,
        target_hidden_states: torch.Tensor,
        positions: torch.Tensor,
        metadata: "tuple[AttentionMetadata, ...]",
    ) -> torch.Tensor:
        """Write each draft layer's K/V of the target context tokens.

        context = Gemma-norm(fc(concat(aux states))); per layer, K/V are that
        layer's qkv_proj rows of the context, with K Gemma-normed per head and
        roped at the context positions. Runs every projection through its
        layer, so the MXFP8 linears keep their checkpoint numerics.
        """
        context = self.hidden_norm(self.fc(target_hidden_states))
        num_tokens = context.shape[0]
        for layer, layer_metadata in zip(self.layers, metadata, strict=True):
            attn = layer.self_attn
            qkv, _ = attn.qkv_proj(context)
            k, v = qkv[:, attn.q_size :].split([attn.kv_size, attn.kv_size], dim=-1)
            k = _per_head_norm(k, attn.k_norm, attn.head_dim)
            k, _ = attn.rotary_emb(positions, k, None)
            impl = attn.attn.impl
            if impl.sharded_draft_dcp is not None:
                # DCP-sharded draft KV (`cp_draft_attention`): owner ranks write.
                impl.write_sharded_draft_kv(
                    attn.attn, k, v, attn.attn.kv_cache, layer_metadata
                )
                continue
            # The cache write rides the attention kernel; its output (for an
            # all-zero query) is discarded.
            query = torch.zeros(
                (num_tokens, attn.num_heads, attn.head_dim),
                dtype=context.dtype,
                device=context.device,
            )
            impl.forward(
                layer=attn.attn,
                query=query,
                key=k.reshape(num_tokens, attn.num_kv_heads, attn.head_dim),
                value=v.reshape(num_tokens, attn.num_kv_heads, attn.head_dim),
                kv_cache=attn.attn.kv_cache,
                attn_metadata=layer_metadata,
            )
        return context


class MiniMaxM3DSparkForCausalLM(Qwen3DSparkForCausalLM):
    """Registry entry for ``architectures=DSparkMiniMaxDraftModel``."""

    # Under DCP the draft's sliding-window KV is sharded over the group like
    # the target's (`layers/adapter/cp_draft_attention`), not whole per rank.
    shards_draft_kv_over_dcp = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        self.draft_model_config = speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        self.config.draft_vocab_size = self.config.vocab_size
        self.model = MiniMaxM3DSparkModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            start_layer_id=vllm_config.model_config.get_total_num_hidden_layers(),
        )
        # Replaced by the target's LM head once the draft loads.
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.target_vocab_size = vllm_config.model_config.get_vocab_size()
        self.draft_id_to_target_id = None
        # This rank's vocab shard of the Markov head, set by
        # enable_vocab_parallel_markov_argmax() once the LM head is shared.
        self._markov_cast_dtype: torch.dtype | None = None
        self.register_buffer("markov_w2_shard", None, persistent=False)
        self.register_buffer("vocab_shard_start", None, persistent=False)
        self.register_buffer("vocab_shard_pad_mask", None, persistent=False)

    def _vocab_parallel_markov_blocker(self) -> str | None:
        """Why the shard argmax cannot reproduce the gathered argmax, or None."""
        head, processor = self.lm_head, self.logits_processor
        markov_w2 = self.model.markov_head.markov_w2
        shard = getattr(head, "shard_indices", None)
        if head.tp_size <= 1 or shard is None:
            return "the LM head is not vocab-parallel"
        if "_gather_logits" in vars(processor):
            return "the draft is replicated"
        if head.weight.shape[0] != head.num_embeddings_per_partition:
            return "the LM head holds a replicated weight"
        if shard.num_added_elements_padded > 0:
            return "the LM head has added (LoRA) vocab"
        if processor.logits_as_input or processor.soft_cap is not None:
            return "the logits processor transforms the logits"
        if processor.scale != 1.0:
            return f"the logit scale is {processor.scale}"
        if not (
            processor.org_vocab_size == head.org_vocab_size == markov_w2.org_vocab_size
        ):
            return "the LM head, Markov head and logits vocab sizes differ"
        if head.org_vocab_size >= MAX_VOCAB_FOR_FLOAT_IDS:
            return "the vocab is too large for float32 ids"
        if not isinstance(markov_w2.quant_method, UnquantizedEmbeddingMethod):
            return "the Markov head is quantized"
        return None

    def _set_markov_vocab_shard(self, start: int, end: int, width: int) -> None:
        """Keep the Markov head rows of global ids ``[start, end)``, padded with
        zero rows to the LM head's ``width`` local rows, and a mask of the pad
        columns. Every rank gets tensors of the same shapes, so the traced
        program is rank-independent."""
        markov_w2 = self.model.markov_head.markov_w2.weight
        embed_dtype = self.model.markov_head.markov_w1.weight.dtype
        # Mirror LogitsProcessor._apply_head: a head_dtype different from the
        # input's casts both operands before the matmul.
        head_dtype = self.logits_processor.head_dtype
        cast = head_dtype is not None and head_dtype != embed_dtype
        self._markov_cast_dtype = head_dtype if cast else None
        rows = markov_w2[start:end]
        if cast:
            rows = rows.to(head_dtype)
        self.markov_w2_shard = F.pad(rows, (0, 0, 0, width - (end - start)))
        device = markov_w2.device
        self.vocab_shard_start = torch.tensor(start, dtype=torch.int64, device=device)
        if self.lm_head.org_vocab_size_padded == self.lm_head.org_vocab_size:
            self.vocab_shard_pad_mask = None
        else:
            self.vocab_shard_pad_mask = (
                torch.arange(width, device=device) >= end - start
            )

    def enable_vocab_parallel_markov_argmax(self) -> bool:
        """DSparkProposer opt-in: argmax the greedy Markov drafts per vocab
        shard. Call after the target's LM head is shared. Returns False (and
        the proposer keeps the gathered logits) when the shard path could not
        give the same ids."""
        blocker = self._vocab_parallel_markov_blocker()
        if blocker is not None:
            logger.info(
                "MiniMax DSpark: gathering full-vocab draft logits because %s.",
                blocker,
            )
            return False
        shard = self.lm_head.shard_indices
        self._set_markov_vocab_shard(
            shard.org_vocab_start_index,
            shard.org_vocab_end_index,
            self.lm_head.num_embeddings_per_partition,
        )
        return True

    def compute_draft_logits_shard(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """This rank's vocab shard of ``compute_draft_logits`` (pad columns
        included), without the all-gather."""
        return self.logits_processor(self.lm_head, hidden_states, skip_gather=True)

    def markov_shard_pair(
        self, logits_shard: torch.Tensor, markov_embed: torch.Tensor
    ) -> torch.Tensor:
        """``[rows, 2]`` (max, global id) of logits + Markov bias on this
        rank's shard. Element for element the same sums as the gathered path:
        the logits are the pre-gather values and the bias columns are the same
        matmul rows."""
        if self._markov_cast_dtype is not None:
            markov_embed = markov_embed.to(self._markov_cast_dtype)
        scores = logits_shard + F.linear(markov_embed, self.markov_w2_shard)
        if self.vocab_shard_pad_mask is not None:
            scores = scores.masked_fill(self.vocab_shard_pad_mask, float("-inf"))
        return shard_argmax_pair(scores, self.vocab_shard_start)

    def markov_argmax_shard(
        self, logits_shard: torch.Tensor, markov_embed: torch.Tensor
    ) -> torch.Tensor:
        """Draft-vocab ids equal to ``(compute_draft_logits + markov_bias)
        .argmax(-1)``, exchanging one (max, id) pair per rank and row."""
        pair = self.markov_shard_pair(logits_shard, markov_embed)
        gathered = tensor_model_parallel_all_gather(pair, dim=-1)
        return reduce_shard_argmax_pairs(gathered, self.lm_head.tp_size)

    def tpu_precompute_and_store_context_kv(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        metadata: "tuple[AttentionMetadata, ...]",
    ) -> torch.Tensor:
        return self.model.store_context_kv(hidden_states, positions, metadata)

    @staticmethod
    def _rename_checkpoint_weights(
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[tuple[str, torch.Tensor]]:
        for name, tensor in weights:
            name = name.removeprefix(CHECKPOINT_PREFIX)
            name = name.replace(".decoder_layer.", ".")
            if name.startswith("final_norm."):
                name = "norm." + name.removeprefix("final_norm.")
            # The E8M0 block scales multiply despite the name.
            name = name.replace(".weight_scale_inv", ".weight_scale")
            yield name, tensor

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        super().load_weights(self._rename_checkpoint_weights(weights))
