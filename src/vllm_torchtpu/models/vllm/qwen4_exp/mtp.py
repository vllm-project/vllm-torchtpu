# SPDX-License-Identifier: Apache-2.0
"""Qwen4Exp (Qwen3.8 Flash Next) MTP draft on the TPU-native backbone.

Follows upstream ``vllm.models.qwen4_exp.{nvidia,amd}.mtp`` exactly:

* the token embedding (shared with the target) passes ``pre_fc_norm_embedding``
  and ``fc_embedding`` to [T, H];
* the target's *pre-final-mixer* multi-stream hidden state [T, hc*H] passes one
  RMSNorm over all hc*H elements (``pre_fc_norm_hidden``), then the shared
  ``fc_hidden`` per HC stream; the embedding is added to every stream;
* one full-attention decoder layer (QSA with its indexer, routed + shared MoE
  and both hyper-connections; no PLE) runs on that multi-stream state;
* the MTP's own ``hyper_connection_mixer`` collapses it to [T, H] for the
  shared LM head, while the multi-stream state feeds the next draft step.

The QSA layer owns ordinary unified-pool pages (``mtp.layers.<L>.self_attn``
plus its compressed-key and raw-ring groups), planned by ``native_cache_specs``
and bound by ``Qwen4ExpPooledCache``. Its ragged geometry comes from its own
forward-context metadata, which the runner's MTP proposer builds for step 0
(the target's chunk) and for the one-token loop steps.
"""

import copy
import re
from collections.abc import Iterable
from typing import cast

import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata

from .config import mtp_layer_prefixes
from .hyperconnection import GatedResidual
from .model import (
    Qwen4ExpDecoderLayer,
    Qwen4ExpForCausalLM,
    _parameter_storage_key,
    hc_config,
    moe_weights_mapper,
    verify_moe_checkpoint,
)

from .qsa import QSAMetadata

_CHECKPOINT_LAYER = re.compile(r"^mtp\.layers\.(\d+)\.")


def remap_mtp_ignored_layers(names, first_layer):
    """Checkpoint ``mtp.layers.<i>`` -> draft module ``mtp.layers.<first+i>``."""
    return [
        _CHECKPOINT_LAYER.sub(
            lambda m: f"mtp.layers.{first_layer + int(m.group(1))}.", name
        )
        if name.startswith("mtp.layers.")
        else name
        for name in names
    ]


def draft_quant_config(quant_config, first_layer):
    """The target's FP8 config with the MTP layer's BF16 modules renamed.

    The draft shares the target's quant config object; mutating it would
    change the target's layer matching, so return a shallow copy with only
    the skip list replaced.
    """
    if quant_config is None:
        return None
    ignored = getattr(quant_config, "ignored_layers", None)
    if not ignored:
        return quant_config
    config = copy.copy(quant_config)
    config.ignored_layers = remap_mtp_ignored_layers(ignored, first_layer)
    return config


def draft_additional_config(additional_config):
    """The target's additional_config without the HC prefill tile options.

    The tiled HC kernels are chosen by `tokens >= tile and tokens % tile == 0`
    while tracing. The backbone is re-traced per token-bucket group, but the
    draft is traced once and its graph serves every bucket, so a draft traced
    at fewer than `tile` tokens cannot serve a larger bucket
    (ShapeSpecializationError) and vice versa. The draft's HC therefore keeps
    the untiled path at every token count.
    """
    qwen = {
        k: v
        for k, v in additional_config.get("qwen4_exp", {}).items()
        if not k.startswith("hc_prefill_")
    }
    return {**additional_config, "qwen4_exp": qwen}


def mtp_checkpoint_name(name):
    """Draft parameter name for a checkpoint tensor; None for target tensors.

    The draft shares the target's embedding and LM head, so only ``mtp.*``
    tensors belong to it. Checkpoint layer ``i`` is ModuleList entry ``i``.
    """
    if not name.startswith("mtp."):
        return None
    return "model." + name.removeprefix("mtp.")


class Qwen4ExpMultiTokenPredictor(nn.Module):
    def __init__(self, cfg, prefixes):
        super().__init__()
        t = cfg.model_config.hf_text_config
        self.hidden_size = t.hidden_size
        self.hc_count = t.hc_count
        self.embed_tokens = VocabParallelEmbedding(t.vocab_size, t.hidden_size)
        self.pre_fc_norm_embedding = GemmaRMSNorm(t.hidden_size, eps=t.rms_norm_eps)
        self.pre_fc_norm_hidden = GemmaRMSNorm(
            t.hidden_size * t.hc_count, eps=t.rms_norm_eps
        )
        # BF16 in the checkpoint (modules_to_not_convert); replicated so the
        # [T*hc, H] projection needs no collective.
        for name in ("fc_embedding", "fc_hidden"):
            setattr(
                self,
                name,
                ReplicatedLinear(
                    t.hidden_size,
                    t.hidden_size,
                    bias=False,
                    params_dtype=torch.bfloat16,
                    quant_config=None,
                    prefix=f"mtp.{name}",
                    return_bias=False,
                ),
            )
        self.layers = nn.ModuleList(
            Qwen4ExpDecoderLayer(
                cfg,
                t.num_hidden_layers + i,
                prefix,
                layer_type="full_attention",
            )
            for i, prefix in enumerate(prefixes)
        )
        tiles = cfg.additional_config.get("qwen4_exp", {})
        self.hyper_connection_mixer = GatedResidual(
            hc_config(t),
            use_combine=False,
            prefill_up_tile=int(tiles.get("hc_prefill_up_tile", 0)),
            prefill_norm_tile=int(tiles.get("hc_prefill_norm_tile", 0)),
            prefill_full_tile=int(tiles.get("hc_prefill_full_tile", 0)),
        )
        # Set by the Qwen3.8 megakernel driver before the first trace: the
        # layer's o_proj, both HCs and the MoE as one tail-kernel call.
        self.mk_tail: nn.Module | None = None

    def fuse_inputs(self, input_ids, hidden_states, inputs_embeds=None):
        """[T, hc*H] decoder input from the token and the previous hidden."""
        tokens = hidden_states.shape[0]
        embeds = (
            self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        )
        embeds = self.fc_embedding(self.pre_fc_norm_embedding(embeds))
        hidden = self.pre_fc_norm_hidden(hidden_states)
        hidden = self.fc_hidden(hidden.reshape(tokens * self.hc_count, -1))
        hidden = hidden.reshape(tokens, self.hc_count, self.hidden_size)
        return (hidden + embeds.unsqueeze(1)).reshape(tokens, -1)

    def forward(
        self, input_ids, positions, hidden_states, inputs_embeds, step, rows=None
    ):
        hidden = self.fuse_inputs(input_ids, hidden_states, inputs_embeds)
        layer = cast(Qwen4ExpDecoderLayer, self.layers[step % len(self.layers)])
        metadata = cast(
            dict[str, AttentionMetadata], get_forward_context().attn_metadata
        )[layer.self_attn.prefix]
        starts = cast(torch.Tensor, metadata.query_start_loc)
        seq_lens = cast(torch.Tensor, metadata.seq_lens)
        # The same ragged geometry the target adapter derives: context is the
        # length before this chunk. Loop steps advance seq_lens in place.
        context = (seq_lens - (starts[1:] - starts[:-1])).to(torch.int64)
        tables = cast(torch.Tensor, metadata.block_tables)
        tail = self.mk_tail
        # No branch on the token count: vLLM rejects a traced graph that does
        # (ShapeSpecializationError). The tail op resolves concrete rows itself;
        # the driver attaches it only when step 0 uses draft rows, so the
        # all-token path below is the fused loop's (<= request capacity rows).
        if tail is not None:
            # Stock attention (caches written for every token), then the kernel.
            block, _ = layer.attn_hyper_connection.mix(hidden)
            qsa_md = QSAMetadata(
                starts,
                context,
                torch.zeros_like(seq_lens, dtype=torch.int32),
                tables,
                tables,
            )
            pos = positions.to(torch.int64)
            if rows is not None:
                rows = rows.to(torch.int64)
                attention = layer.self_attn.attend_with_indices(
                    block, pos, qsa_md, rows=rows
                )[0]
                hidden = hidden[rows.clamp_min(0)]
            else:
                attention = layer.self_attn.attend(block, pos, qsa_md)
            hidden = tail(hidden, attention)
            return self.hyper_connection_mixer.mix(hidden)[0], hidden
        if rows is not None:
            hidden = layer.forward_rows(
                hidden,
                positions.to(torch.int64),
                rows.to(torch.int64),
                starts,
                torch.zeros_like(seq_lens, dtype=torch.int32),
                context,
                tables,
                tables,
            )
            return self.hyper_connection_mixer.mix(hidden)[0], hidden
        hidden = layer(
            hidden,
            positions.to(torch.int64),
            input_ids.to(torch.int64),
            starts,
            None,
            # Pooled QSA reads its ring slot and page tables from the
            # per-group forward-context metadata; these are placeholders.
            torch.zeros_like(seq_lens, dtype=torch.int32),
            context,
            torch.arange(hidden.shape[0], device=hidden.device) < starts[-1],
            tables,
            tables,
        )
        return self.hyper_connection_mixer.mix(hidden)[0], hidden


# Only the token dimension is dynamic. `draft_rows` (draft step 0's per-request
# rows) must stay static: vLLM compiles one trace per token bucket by setting
# every symbol of the trace to the bucket size (create_concrete_args), so a
# dynamic draft_rows dim was compiled at the bucket size (s64[256] at bucket
# 256) and failed against the request-sized rows a step passes. The proposer
# always passes it at the chunk's fixed request capacity instead.
QWEN4_EXP_MTP_DYNAMIC_ARG_DIMS = {
    "input_ids": 0,
    "positions": 0,
    "hidden_states": 0,
}


@support_torch_compile(dynamic_arg_dims=QWEN4_EXP_MTP_DYNAMIC_ARG_DIMS)
class Qwen4ExpMTP(nn.Module):
    """Draft returning (LM-head input [T, H], next-step hidden [T, hc*H])."""

    # Draft step 0 consumes the target's second output, the multi-stream
    # hidden state, rather than its LM-head input (see Eagle3Proposer).
    uses_target_aux_hidden_states = True
    has_own_embed_tokens = False
    has_own_lm_head = False
    # Draft step 0 may pass `draft_rows` (each request's last token): the
    # draft then writes its QSA caches for every token but evaluates the
    # layer, and returns both outputs, at those rows only.
    supports_draft_rows = True
    # Identical to the target's: vLLM writes this onto the shared quant config.
    packed_modules_mapping = Qwen4ExpForCausalLM.packed_modules_mapping

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        prefixes = mtp_layer_prefixes(vllm_config)
        if len(prefixes) != 1:
            raise ValueError("Qwen4Exp TPU MTP requires exactly one MTP layer")
        t = vllm_config.model_config.hf_text_config
        cfg = copy.copy(vllm_config)
        cfg.quant_config = draft_quant_config(
            vllm_config.quant_config, t.num_hidden_layers
        )
        cfg.additional_config = draft_additional_config(vllm_config.additional_config)
        self.config = t
        self.model = Qwen4ExpMultiTokenPredictor(cfg, prefixes)
        self.lm_head = ParallelLMHead(t.vocab_size, t.hidden_size, prefix="lm_head")
        self.logits_processor = LogitsProcessor(t.vocab_size)
        # Hidden width of draft step 0's target input and the loop carry.
        self.target_hidden_size = t.hidden_size * t.hc_count

    def embed_input_ids(self, input_ids):
        return self.model.embed_tokens(input_ids)

    def forward(
        self,
        input_ids,
        positions,
        hidden_states,
        intermediate_tensors=None,
        inputs_embeds=None,
        spec_step_idx: int = 0,
        draft_rows: torch.Tensor | None = None,
    ):
        if intermediate_tensors is not None:
            raise NotImplementedError("Qwen4Exp MTP runs whole on one stage")
        return self.model(
            input_ids,
            positions,
            hidden_states,
            inputs_embeds,
            spec_step_idx,
            draft_rows,
        )

    def compute_logits(self, hidden_states, spec_step_idx: int = 0):
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        parameters = dict(self.named_parameters())
        moe_mapper = moe_weights_mapper()
        layers = cast(nn.ModuleList, self.model.layers)
        moe_loaders = [AutoWeightsLoader(layer.mlp) for layer in layers]
        loaded: set[str] = set()
        source: set[str] = set()
        for original, weight in weights:
            name = mtp_checkpoint_name(original)
            if name is None:
                continue
            if name in source:
                raise ValueError(f"Duplicate checkpoint tensor {original}")
            source.add(name)
            if name.startswith("model.layers."):
                _, _, idx, family, relative = name.split(".", 4)
                if int(idx) >= len(layers):
                    raise ValueError(f"Unexpected Qwen4Exp MTP layer {original}")
                layer = cast(Qwen4ExpDecoderLayer, layers[int(idx)])
                if family == "mlp":
                    targets = moe_loaders[int(idx)].load_weights(
                        iter([(relative, weight)]), mapper=moe_mapper
                    )
                    loaded.update(
                        f"model.layers.{idx}.mlp.{target}" for target in targets
                    )
                    continue
                if family == "self_attn":
                    layer.self_attn.load_weights([(relative, weight)])
                    continue
            if name not in parameters:
                raise ValueError(f"Unknown Qwen4Exp MTP checkpoint tensor {original}")
            param = parameters[name]
            getattr(param, "weight_loader", default_weight_loader)(param, weight)
            loaded.add(name)
        for i, layer in enumerate(layers):
            verify_moe_checkpoint(
                layer.mlp,
                f"model.layers.{i}.mlp.",
                source,
                moe_mapper,
                f"MTP layer{i}",
            )
            layer.self_attn.finalize_weights()
        ordinary = {
            name
            for name in parameters
            if not name.startswith(("model.embed_tokens.", "lm_head."))
            and not any(marker in name for marker in (".mlp.experts.", ".self_attn."))
        }
        # Quantization methods may register names for the same storage.
        loaded_storage = {
            _parameter_storage_key(parameters[name])
            for name in loaded & parameters.keys()
        }
        missing = {
            name
            for name in ordinary - loaded
            if _parameter_storage_key(parameters[name]) not in loaded_storage
        }
        if missing:
            raise ValueError(f"Incomplete Qwen4Exp MTP checkpoint: {sorted(missing)}")
        return set(parameters)
