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
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
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

from vllm_torchtpu.layers.adapter import cp_draft_attention, cp_draft_head_split
from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR
from vllm_torchtpu.layers.adapter.quantization.unquantized import (
    VllmUnquantizedLinearMethod,
)
from vllm_torchtpu.layers.core.quant_methods import get_tpu_quant_method
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.minimax_m3.moe import MiniMaxM3MLP
from vllm_torchtpu.models.minimax_m3.quant import (
    QUANT_METHOD,
    MiniMaxM3Mxfp8LinearMethod,
)
from vllm_torchtpu.opt_in_config import option
from vllm_torchtpu.spec_decode.vocab_parallel_argmax import (
    MAX_VOCAB_FOR_FLOAT_IDS,
    masked_reduce,
    reduce_shard_argmax_pairs,
    set_masked_reduce,
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


def _rank_rows_blocker(model: nn.Module, dcp: int, kv_head_replicas: int) -> str | None:
    """Why `store_context_kv_rank_rows` cannot reproduce the sharded write, or
    None."""
    for layer in model.layers:
        attn = layer.self_attn
        sharded = getattr(attn.attn.impl, "sharded_draft_dcp", None)
        if sharded is None:
            return "a draft layer keeps its KV replicated"
        if sharded.get("head_split"):
            return "the draft KV is head-split (draft_kv_head_split; no K/V gather)"
        proj = attn.qkv_proj
        if not isinstance(
            proj.quant_method, (MiniMaxM3Mxfp8LinearMethod, VllmUnquantizedLinearMethod)
        ) or not getattr(proj, WEIGHT_FLIPPED_ATTR, False):
            return "qkv_proj is not a plain (k, n) matmul"
        if getattr(proj, "bias", None) is not None:
            return "qkv_proj has a bias"
        if dcp > kv_head_replicas and attn.num_kv_heads != 1:
            return f"{attn.num_kv_heads} local KV heads"
    return None


def prepare_rank_rows_context_kv(
    model: nn.Module, group, dcp: int, kv_head_replicas: int
) -> bool:
    """Give each draft layer the K/V weight of all the DCP group's KV heads,
    ``[hidden, 2 * G * D]`` (K heads, then V heads), for
    `store_context_kv_rank_rows`. The heads are in `write_with_dcp`'s order:
    every `kv_head_replicas`-th rank of the group supplies the next one, and a
    group no wider than the replicas shares the rank's own heads. Call once
    after the weights load and sharded draft DCP is enabled."""
    blocker = _rank_rows_blocker(model, dcp, kv_head_replicas)
    if blocker is not None:
        logger.info("MiniMax DSpark: context K/V written from all rows because %s.", blocker)
        model.rank_rows_context_kv = None
        return False
    for layer in model.layers:
        attn = layer.self_attn
        local = attn.qkv_proj.weight.data[:, attn.q_size :]  # [hidden, K | V]
        if dcp > kv_head_replicas:
            hidden = local.shape[0]
            gathered = group.all_gather(local.contiguous(), dim=1)
            heads = gathered.view(hidden, dcp, 2, attn.head_dim)[:, ::kv_head_replicas]
            local = heads.permute(0, 2, 1, 3).reshape(hidden, -1)
        attn.register_buffer("context_kv_weight", local.contiguous(), persistent=False)
    model.rank_rows_context_kv = dict(dcp=dcp)
    return True


def store_context_kv_rank_rows(
    model: nn.Module,
    aux_hidden_states: list[torch.Tensor],
    positions: torch.Tensor,
    metadata: "tuple[AttentionMetadata, ...]",
) -> torch.Tensor:
    """`MiniMaxM3DSparkModel.store_context_kv` for DCP-sharded draft KV, with
    the work split over the group by token rows instead of repeated per rank.

    DCP rank ``r`` takes rows ``[r * n, (r + 1) * n)``, ``n = ceil(T / dcp)``:
    the aux states' concat, fc and hidden norm on those rows only, then per
    layer one matmul for the K/V of every KV head of the group (no Q
    columns), the K norm and RoPE, and the cache bytes (`pack_rows`). One
    all-gather of every layer's packed rows gives each rank all T rows, of
    which it copies the tokens of its own blocks into its pages
    (`write_owned_rows`). Each row runs through the same per-row ops as in
    the replicated-rows path, so the cache bytes are the same.

    Returns this rank's rows of the context.
    """
    from vllm_torchtpu.distributed.dcp import get_dcp_group

    dcp = model.rank_rows_context_kv["dcp"]
    first = model.layers[0].self_attn.attn
    num_tokens = aux_hidden_states[0].shape[0]
    per_rank = -(-num_tokens // dcp)
    rows = first.draft_dcp_rank.to(torch.int64) * per_rank + torch.arange(
        per_rank, device=positions.device
    )
    rows = rows.clamp(max=num_tokens - 1)
    target = [h.index_select(0, rows) for h in aux_hidden_states]
    target = target[0] if len(target) == 1 else torch.cat(target, dim=-1)
    context = model.hidden_norm(model.fc(target))
    positions = positions.index_select(0, rows)
    packed = []
    for layer in model.layers:
        attn = layer.self_attn
        kv = torch.matmul(context, attn.context_kv_weight)
        k, v = kv.chunk(2, dim=-1)
        k = _per_head_norm(k, attn.k_norm, attn.head_dim)
        k, _ = attn.rotary_emb(positions, k, None)
        ops = attn.attn.impl.sharded_draft_dcp["ops"]
        packed.append(
            ops["pack"](
                attn.attn.kv_cache,
                k.reshape(per_rank, -1, attn.head_dim),
                v.reshape(per_rank, -1, attn.head_dim),
            )
        )
    gathered = cp_draft_attention.gather_rows(torch.stack(packed, dim=1), get_dcp_group())
    for i, (layer, layer_metadata) in enumerate(zip(model.layers, metadata, strict=True)):
        attn = layer.self_attn.attn
        ops = attn.impl.sharded_draft_dcp["ops"]
        new_cache = ops["write_rows"](
            attn.kv_cache,
            gathered[:num_tokens, i].contiguous(),
            layer_metadata.seq_lens,
            layer_metadata.block_tables,
            layer_metadata.query_start_loc,
            attn.draft_dcp_rank,
        )
        attn.kv_cache.copy_(new_cache)
    return context


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


# Set from additional_config minimax_dspark_split_fc by MiniMaxM3DSparkModel
# before the first trace (`_column_parallel`, `_context_kv`).
SPLIT_FC = False


def _flipped_weight(linear: nn.Module, dtype: torch.dtype) -> torch.Tensor | None:
    """The ``[n_in, n_out]`` weight of a bias-free linear whose apply is a
    plain ``matmul(x, weight)`` (unquantized, or MXFP8 dequantized to
    ``dtype`` at load, both flipped after loading), else None."""
    from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR

    if getattr(linear, "bias", None) is not None:
        return None
    if not getattr(linear, WEIGHT_FLIPPED_ATTR, False):
        return None
    # A quantized weight keeps a separate scale (e.g. the fp8 method).
    if hasattr(linear, "weight_scale") or linear.weight.dtype != dtype:
        return None
    return linear.weight


def _column_parallel(
    linear: nn.Module, x: torch.Tensor, force: bool = False
) -> torch.Tensor:
    """``linear(x)`` for a TP-replicated linear: with additional_config
    minimax_dspark_split_fc (r6) or ``force`` (the head-split draft KV, as
    r10), each rank computes 1/tp of the output columns, reading 1/tp of the
    weight, then the ranks all-gather."""
    if not (SPLIT_FC or force):
        return linear(x)
    weight = _flipped_weight(linear, x.dtype)
    tp = get_tensor_model_parallel_world_size()
    if weight is None or tp == 1 or weight.shape[1] % tp:
        return linear(x)
    cols = weight.shape[1] // tp
    start = get_tensor_model_parallel_rank() * cols
    part = torch.matmul(x, weight[:, start : start + cols])
    return tensor_model_parallel_all_gather(part, dim=-1)


def _context_kv(attn: nn.Module, context: torch.Tensor, force: bool = False):
    """A draft layer's K and V of the context tokens. Only the qkv_proj's K/V
    columns when its weight is a plain matrix: the Q columns would be
    discarded (1024 of this rank's 1280 for MiniMax-M3's draft at TP8).
    Only with additional_config minimax_dspark_split_fc (r6); else the whole
    qkv_proj. ``force``: as with the key (the head-split draft KV)."""
    split = SPLIT_FC or force
    weight = _flipped_weight(attn.qkv_proj, context.dtype) if split else None
    if weight is None:
        qkv, _ = attn.qkv_proj(context)
        kv = qkv[:, attn.q_size :]
    else:
        kv = torch.matmul(context, weight[:, attn.q_size :])
    return kv.split([attn.kv_size, attn.kv_size], dim=-1)


def set_split_fc(additional_config) -> None:
    global SPLIT_FC
    SPLIT_FC = bool(option(additional_config, "minimax_dspark_split_fc"))
    if SPLIT_FC:
        logger.info(
            "MiniMax-M3 DSpark split fc on: additional_config.minimax_dspark_split_fc"
        )


def set_markov_options(additional_config) -> None:
    """additional_config dspark_markov_kernel (`pallas_markov_argmax`) and
    spec_vocab_argmax_masked (`vocab_parallel_argmax`); both log when set."""
    from vllm_torchtpu.distributed import pallas_markov_argmax

    pallas_markov_argmax.set_enabled(option(additional_config, "dspark_markov_kernel"))
    masked = bool(option(additional_config, "spec_vocab_argmax_masked"))
    set_masked_reduce(masked)
    if masked:
        logger.info(
            "DSpark vocab-parallel argmax: float32 ids, masked cross-rank "
            "reduction (additional_config spec_vocab_argmax_masked=True)."
        )


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
        set_split_fc(vllm_config.additional_config)

    # Set by `prepare_rank_rows_context_kv` (sharded draft DCP only).
    rank_rows_context_kv: dict | None = None
    # One write op for every layer's head-split context K/V
    # (`cp_draft_head_split.write_layers_op`), set by `prepare_draft_kv_write`.
    _head_split_kv_write_op = None
    _head_split_context = False

    def prepare_draft_kv_write(self) -> None:
        """Build the all-layer head-split K/V write (outside any compiled
        forward) when every layer keeps its KV head-split with the same
        sharding (additional_config draft_kv_head_split)."""
        configs = set()
        for layer in self.layers:
            cfg = getattr(layer.self_attn.attn.impl, "sharded_draft_dcp", None)
            if cfg is None or not cfg.get("head_split"):
                configs.add(None)
                continue
            configs.add(
                (
                    cfg["dcp"],
                    min(cfg["dcp"], cfg["kv_head_replicas"]),
                    cfg["k_scale"],
                    cfg["v_scale"],
                )
            )
        if len(configs) == 1 and None not in configs:
            # As r10: fc split by columns over TP and K/V-only projections
            # (every rank projects all context rows for its own KV head).
            self._head_split_context = True
            self._head_split_kv_write_op = cp_draft_head_split.write_layers_op(
                *configs.pop(), len(self.layers)
            )
            logger.info(
                "MiniMax DSpark: one head-split context K/V write for %d draft "
                "layers.",
                len(self.layers),
            )

    def _write_head_split_context_kv(self, context, positions, metadata) -> None:
        """Every layer's own-head context K/V through the one all-layer write."""
        keys, values = [], []
        for layer in self.layers:
            attn = layer.self_attn
            k, v = _context_kv(attn, context, force=True)
            k = _per_head_norm(k, attn.k_norm, attn.head_dim)
            k, _ = attn.rotary_emb(positions, k, None)
            shape = (context.shape[0], attn.num_kv_heads, attn.head_dim)
            keys.append(k.reshape(shape))
            values.append(v.reshape(shape))
        attns = [layer.self_attn.attn for layer in self.layers]
        cp_draft_head_split.write_layers_with_dcp(
            keys=torch.stack(keys),
            values=torch.stack(values),
            kv_caches=[attn.kv_cache for attn in attns],
            attn_metadata=metadata,
            dcp_rank=attns[0].draft_dcp_rank,
            op=self._head_split_kv_write_op,
        )

    def store_context_kv(
        self,
        target_hidden_states: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
        positions: torch.Tensor,
        metadata: "tuple[AttentionMetadata, ...]",
    ) -> torch.Tensor:
        """Write each draft layer's K/V of the target context tokens.

        context = Gemma-norm(fc(concat(aux states))); per layer, K/V are that
        layer's qkv_proj rows of the context, with K Gemma-normed per head and
        roped at the context positions. Runs every projection through its
        layer, so the MXFP8 linears keep their checkpoint numerics.
        `target_hidden_states` is the aux states, or their concat.
        """
        aux = (
            list(target_hidden_states)
            if isinstance(target_hidden_states, (list, tuple))
            else [target_hidden_states]
        )
        if self.rank_rows_context_kv is not None:
            return store_context_kv_rank_rows(self, aux, positions, metadata)
        target_hidden_states = aux[0] if len(aux) == 1 else torch.cat(aux, dim=-1)
        context = self.hidden_norm(
            _column_parallel(
                self.fc, target_hidden_states, force=self._head_split_context
            )
        )
        num_tokens = context.shape[0]
        if self._head_split_kv_write_op is not None and all(
            m is metadata[0] for m in metadata
        ):
            self._write_head_split_context_kv(context, positions, metadata[0])
            return context
        for layer, layer_metadata in zip(self.layers, metadata, strict=True):
            attn = layer.self_attn
            k, v = _context_kv(attn, context, force=self._head_split_context)
            k = _per_head_norm(k, attn.k_norm, attn.head_dim)
            k, _ = attn.rotary_emb(positions, k, None)
            impl = attn.attn.impl
            if getattr(impl, "sharded_draft_dcp", None) is not None:
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
    # `tpu_precompute_and_store_context_kv` takes the aux states unconcatenated.
    tpu_context_kv_takes_aux_list = True

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
        # `pallas_markov_argmax.MarkovChainOp`: the whole greedy chain in one
        # kernel (additional_config dspark_markov_kernel), set with the vocab
        # shard when it builds.
        self.markov_chain_kernel = None
        self._max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        set_markov_options(vllm_config.additional_config)

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
        # float32 under spec_vocab_argmax_masked, as the (max, id) pairs carry
        # ids (`shard_argmax_pair`).
        start_dtype = torch.float32 if masked_reduce() else torch.int64
        self.vocab_shard_start = torch.tensor(start, dtype=start_dtype, device=device)
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
        start, end = shard.org_vocab_start_index, shard.org_vocab_end_index
        self._set_markov_vocab_shard(
            start, end, self.lm_head.num_embeddings_per_partition
        )
        from vllm_torchtpu.distributed import pallas_markov_argmax

        markov_w1 = self.model.markov_head.markov_w1.weight
        if (
            pallas_markov_argmax.enabled()
            and self.draft_id_to_target_id is None
            and self._markov_cast_dtype is None
            and markov_w1.dtype == self.markov_w2_shard.dtype == torch.bfloat16
        ):
            # Collective over the TP group (self-tested); None keeps the loop.
            self.markov_chain_kernel = pallas_markov_argmax.build(
                markov_w1, self.markov_w2_shard, start, end - start, self._max_num_seqs
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
        # int32 under spec_vocab_argmax_masked: the TPU emulates int64 ops on
        # the Markov loop's serial chain.
        dtype = torch.int32 if masked_reduce() else torch.int64
        return reduce_shard_argmax_pairs(gathered, self.lm_head.tp_size, dtype=dtype)

    def tpu_precompute_and_store_context_kv(
        self,
        hidden_states: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
        positions: torch.Tensor,
        metadata: "tuple[AttentionMetadata, ...]",
    ) -> torch.Tensor:
        return self.model.store_context_kv(hidden_states, positions, metadata)

    def prepare_sharded_context_kv(self, group, dcp: int, kv_head_replicas: int) -> bool:
        """DSparkProposer hook once sharded draft DCP is enabled: split the
        context K/V work over the DCP group by rows
        (`store_context_kv_rank_rows`). Returns whether it did."""
        return prepare_rank_rows_context_kv(self.model, group, dcp, kv_head_replicas)

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
