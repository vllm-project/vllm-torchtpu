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
import re
from collections.abc import Iterable
from typing import cast

import torch
import torch.nn as nn
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen3_dspark import DSparkMarkovHead
from vllm.model_executor.models.utils import WeightsMapper, maybe_prefix

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.deepseek_v4.layers import mhc_collapse_head
from vllm_torchtpu.models.vllm.deepseek_v4.model import (
    DeepseekV4DecoderLayer,
    DeepseekV4Model,
    _make_deepseek_v4_weights_mapper,
)

logger = init_logger(__name__)


def _remap_dspark_name(name: str) -> str | None:
    """Map a checkpoint ``mtp.{i}.*`` name to this model's parameter path.
    Returns None for non-mtp weights (owned by the target model).
    """
    m = re.match(r"mtp\.(\d+)\.(.*)", name)
    if m is None:
        return None
    stage = int(m.group(1))
    rest = m.group(2)
    if rest.startswith("confidence_head."):
        return None
    head_prefixes = (
        "norm.",
        "hc_head_fn",
        "hc_head_base",
        "hc_head_scale",
        "markov_head.",
    )
    if rest.startswith(("main_proj.", "main_norm.")) or rest.startswith(head_prefixes):
        return f"model.{rest}"
    return f"model.layers.{stage}.{rest}"


def _rewrite_dspark_name(name: str, mapper: WeightsMapper) -> str | None:
    """Checkpoint name -> ``DeepseekV4Model`` parameter path.

    The DSpark remap first, then the target's own naming conventions:
    scale suffixes, gate bias, shared-expert rename.
    """
    mapped = _remap_dspark_name(name)
    if mapped is None:
        return None
    mapped = mapper._map_name(mapped)
    if mapped is None:
        return None
    return mapped.removeprefix("model.")


def _insert_context_kv(
    attn, kv: torch.Tensor, positions: torch.Tensor, metadata
) -> None:
    """Insert the context KV of verified tokens into one draft layer's cache.

    ``attn`` is one draft layer's attention module, ``kv`` that layer's
    already-normalised latent K/V for the verified tokens, ``positions``
    their absolute positions, and ``metadata`` this layer's sliding-window
    attention metadata, built by the proposer from the target's verify batch.
    The DSV4 Pallas attention op is the only KV-write path on this backend,
    so the insert runs the op with a zero query and discards the attention
    output.
    """
    orig_sw_cache = attn._get_active_sw_cache()
    if orig_sw_cache is None:
        return
    # Draft layers are SWA-only on every DSpark checkpoint.
    assert attn.compress_ratio <= 1, (
        f"{attn.custom_prefix}: DSpark draft layers must be SWA-only, "
        f"got compress_ratio={attn.compress_ratio}"
    )

    kv = attn.kv_rope(kv, positions)
    num_tokens = kv.shape[0]
    dummy_q = torch.zeros(
        (num_tokens, attn.mla_attn.n_local_heads, attn.mla_attn.head_dim),
        dtype=kv.dtype,
        device=kv.device,
    )
    # only writes the SWA cache, and the clones keep the traced signature.
    operands = (
        dummy_q,
        kv,
        orig_sw_cache,
        metadata.seq_lens,
        metadata.block_tables,
        metadata.query_start_loc,
        metadata.request_distribution,
        orig_sw_cache.clone(),
        metadata.seq_lens.clone(),
        (positions + 1).to(torch.int32) // attn.compress_ratio,
        metadata.block_tables.clone(),
        metadata.query_start_loc.clone(),
        metadata.request_distribution.clone(),
        attn.attn_sink,
    )
    # TODO: this runs the full attention op to write the K/V rows into
    # the SWA cache and throws the attention output away. Replace with
    # a dedicated paged-cache insert kernel (like RPA's
    # ragged_kv_cache_update) once one exists for the DSV4 SWA layout.
    _, new_sw_cache = attn.attn_op(*operands)
    if new_sw_cache is not None and new_sw_cache.numel() > 0:
        orig_sw_cache.copy_(new_sw_cache)


class DSparkDeepseekV4Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        self.hc_mult = config.hc_mult
        self.hc_eps = config.hc_eps
        self.rms_norm_eps = config.rms_norm_eps
        self.num_hidden_layers = config.num_hidden_layers
        self.target_layer_ids = tuple(config.dspark_target_layer_ids)
        self.num_dspark_layers = getattr(config, "n_mtp_layers", None) or 3

        # Aliased to the target's embedding by the proposer's sharing
        # utility after load (has_own_embed_tokens=False).
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
        self.layers = nn.ModuleList(
            [
                DeepseekV4DecoderLayer(
                    current_vllm_config,
                    prefix=maybe_prefix(prefix, f"layers.{self.num_hidden_layers + i}"),
                    topk_indices_buffer=self.topk_indices_buffer,
                )
                for i in range(self.num_dspark_layers)
            ]
        )
        # The SWA kernel's non-causal block mode walks key blocks sized from
        # the sliding window, so a block longer than the window would
        # silently lose its earliest in-block keys. K + 1 rows covers the
        # DFlash 1 + K layout as well as the dense DSpark K layout.
        max_block_rows = vllm_config.speculative_config.num_speculative_tokens + 1
        for layer in cast(Iterable[DeepseekV4DecoderLayer], self.layers):
            if max_block_rows > layer.attn.window_size:
                raise ValueError(
                    f"DSpark draft block of up to {max_block_rows} rows "
                    f"exceeds the DSV4 sliding window "
                    f"({layer.attn.window_size})."
                )
            # The runtime counterpart of get_draft_attn_causal(): the SWA
            # kernel attends bidirectionally within the draft query block
            # (picked up by _build_attn_op as the `_ncb` op variant).
            layer.attn.non_causal_multi_token_decode = True

        # Head stack, loaded from the last mtp stage.
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        hc_dim = self.hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(
            torch.empty(self.hc_mult, hc_dim, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(self.hc_mult, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32),
            requires_grad=False,
        )
        draft_vocab_size = (
            getattr(config, "draft_vocab_size", None) or config.vocab_size
        )
        self.markov_head = DSparkMarkovHead(
            config.vocab_size,
            draft_vocab_size,
            config.dspark_markov_rank,
            prefix=maybe_prefix(prefix, "markov_head"),
        )

    # Borrowed from the target: the lazily built MHC post op, the expert
    # mapping, and the per-layer loader.
    # The loader expects names already rewritten by ``_rewrite_dspark_name``.
    mhc_post_op = DeepseekV4Model.mhc_post_op
    get_expert_mapping = DeepseekV4Model.get_expert_mapping
    load_weights = DeepseekV4Model.load_weights

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        assert self.embed_tokens is not None
        return self.embed_tokens(input_ids)

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        """main_x = main_norm(main_proj(concat of target aux states))."""
        return self.main_norm(self.main_proj(aux_hidden_states))

    def get_draft_attn_causal(self) -> list[bool]:
        # In-block attention is non-causal; consumed by the DSV4 attention
        # op's draft mode, not by the dense RPA kernel-init path.
        return [False] * len(self.layers)

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [
            layer.attn.swa_cache_layer.prefix
            for layer in cast(Iterable[DeepseekV4DecoderLayer], self.layers)
        ]

    def precompute_and_store_context_kv(
        self,
        target_hidden_states: torch.Tensor,
        positions: torch.Tensor,
        metadata: tuple,
    ) -> torch.Tensor:
        """Insert the sliding-window context KV for every draft layer.

        Each layer derives its context KV from the SAME combined target
        mirroring upstream; the paged insert goes through the TPU attention op.
        """
        main_x = self.combine_hidden_states(target_hidden_states)
        for i, layer in enumerate(cast(Iterable[DeepseekV4DecoderLayer], self.layers)):
            attn = layer.attn
            qr_kv, _ = attn.fused_wqa_wkv(main_x)
            kv = qr_kv[..., attn.mla_attn.q_lora_rank :]
            kv = attn.kv_norm(kv)
            _insert_context_kv(attn, kv, positions, metadata[i])
        return main_x

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        # Expand to hc_mult copies for hyper-connections ([T,H] -> [T,hc,H]).
        hidden_states = inputs_embeds.unsqueeze(-2).repeat(1, self.hc_mult, 1)

        residual = post_mix = res_mix = None
        for layer in self.layers:
            hidden_states, residual, post_mix, res_mix = layer(
                hidden_states,
                positions,
                input_ids,
                post_mix,
                res_mix,
                residual,
            )
        hidden_states = self.mhc_post_op(hidden_states, residual, post_mix, res_mix)
        return mhc_collapse_head(
            hidden_states,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.rms_norm_eps,
            self.hc_eps,
        )


class DSparkDeepseekV4ForCausalLM(nn.Module):
    """Registry entry for ``architectures=["DSparkDraftModel"]`` on TPU."""

    # Draft weights ship in the target checkpoint (mtp.*) without
    # embed/head; the proposer aliases the target's.
    has_own_embed_tokens = False
    has_own_lm_head = False
    # Opt into DFlashProposer's model-native context-KV path: this draft
    # owns its latent-KV projection, cache-layer names and paged insert.
    owns_context_kv = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.model = DSparkDeepseekV4Model(
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
        metadata: tuple,
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

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Base logits = lm_head(norm(pre-norm head hidden))."""
        assert self.lm_head is not None
        return self.logits_processor(self.lm_head, self.model.norm(hidden_states))

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Full-vocab draft: base logits, no d2t scatter.
        return self.compute_logits(hidden_states)

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return draft_ids

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.bias(markov_embed, self.logits_processor)

    # --- Weight loading -----------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load ``mtp.{i}.*`` draft weights from the target checkpoint.

        The per-stage tensor set is one full DSV4 decoder layer, so after
        the DSpark remap and the target's own name conventions the stream
        is handed to ``DeepseekV4Model.load_weights``
        """
        mapper = _make_deepseek_v4_weights_mapper(
            getattr(self.config, "expert_dtype", "fp4")
        )
        model_params = dict(self.model.named_parameters())
        loaded: set[str] = set()

        def _layer_stream():
            for name, weight in weights:
                mapped = _rewrite_dspark_name(name, mapper)
                if mapped is None:
                    continue
                if mapped.startswith("layers."):
                    yield mapped, weight
                    continue
                # Head-stack and combiner params load directly: through the
                # layer loader, ``markov_w1`` would collide with the ``w1``
                # stacking rule.
                param = model_params[mapped]
                getattr(param, "weight_loader", default_weight_loader)(param, weight)
                loaded.add(mapped)

        loaded |= self.model.load_weights(_layer_stream())
        logger.info("DSpark draft model loaded: %d params", len(loaded))
        return {f"model.{name}" for name in loaded}
