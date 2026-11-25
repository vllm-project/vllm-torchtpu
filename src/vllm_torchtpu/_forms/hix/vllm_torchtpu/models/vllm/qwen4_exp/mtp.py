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
from typing import ClassVar, Literal, cast

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
from vllm.v1.kv_cache_interface import CircularBufferSpec, FullAttentionSpec

from vllm_torchtpu import envs as vllm_torchtpu_envs
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.logger import init_logger

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

logger = init_logger(__name__)

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


# The target's embedding in the checkpoint (Qwen4ExpForCausalLM.hf_to_vllm_mapper maps
# "model.language_model." to "model."); the draft shares it.
TARGET_EMBED_CHECKPOINT_NAMES = (
    "model.language_model.embed_tokens.weight",
    "model.embed_tokens.weight",
)


def draft_needs_own_embedding(vllm_config) -> bool:
    """Whether this engine's draft must hold its own copy of the shared embedding.

    A pipeline builds the target's ``embed_tokens`` on its first stage only
    (``Qwen4ExpModel``: ``PPMissingLayer`` elsewhere), while the draft runs on the
    last stage (``tpu_runner._is_async_drafter``): q38 F2's prefill-only MTP
    producer. There the draft loads the checkpoint embedding itself (vocab x
    hidden bf16, 1.27 GB for Qwen3.8); everywhere else it borrows the target's.
    """
    parallel = getattr(vllm_config, "parallel_config", None)
    if int(getattr(parallel, "pipeline_parallel_size", 1) or 1) <= 1:
        return False
    from vllm.distributed import get_pp_group

    return not get_pp_group().is_first_rank


def draft_kv_producer(vllm_config) -> bool:
    """Whether this engine ships KV to another engine (kv_role kv_producer or kv_both).

    q38 X-D14-DRAFT-1: VLLM_TPU_QSA_DRAFT_NO_INDEX_STATE leaves the draft's compressed
    keys, raw ring and tags unwritten, which only an engine whose draft attends the
    shared selection may do; a producer would ship that stale index state to its peer.
    """
    transfer = getattr(vllm_config, "kv_transfer_config", None)
    return getattr(transfer, "kv_role", None) in ("kv_producer", "kv_both")


def replicated_lookup(table: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """q38 L4 X-D14-DRAFT-2 (c): the TP-summed vocab-parallel lookup, from one replicated table.

    The stock path (the TPU-patched ``VocabParallelEmbedding.forward``) gives each rank the rows of
    the ids in its vocab slice and exact zeros elsewhere (``get_masked_input_and_mask``), and the
    all-reduce adds them: the owner's row for an id in ``[0, rows)`` and a zero row for every other
    id (negative, or in the vocab padding). ``table`` holds the same bf16 rows (assembled by the same
    sum, ``gather_sharded_weight``), so this lookup is the same values with no collective.
    """
    ids = input_ids.long()
    valid = (ids >= 0) & (ids < table.shape[0])
    rows = torch.nn.functional.embedding(ids * valid, table)
    return rows.masked_fill(~valid.unsqueeze(-1), 0)


def agree_draft_embed_repl(on: bool, target: bool = False) -> list[tuple[int, int]]:
    """Every TP rank's (VLLM_TPU_DRAFT_EMBED_REPL, VLLM_TPU_TARGET_EMBED_REPL), which must be equal
    (fail-stop).

    The draft programs carry the embedding all-reduce only where the first flag is off, and the
    target's forwards only where the second is off, so a rank that disagrees would run different TP
    collectives than its peers. Every rank of a TP > 1 draft calls this at load, flags on or off,
    through the L4 agreement gather (runner/l4_agree.py ``gather``, which replaced DRAFT-ASM's own
    ``_gather`` in q38 L4 AGREE: one CPU-group all_gather of two int64; a failed gather raises).
    Unequal values raise on every rank, since every rank sees the same gathered list.
    """
    from vllm_torchtpu.runner.l4_agree import gather as _gather

    votes = [tuple(int(x) for x in v) for v in _gather([int(bool(on)), int(bool(target))])]
    if len(set(votes)) != 1:
        raise RuntimeError(
            "q38 X-D14-DRAFT-2 (c) / EMB-REP-TGT: VLLM_TPU_DRAFT_EMBED_REPL / VLLM_TPU_TARGET_EMBED_REPL "
            f"differ across the TP ranks (per-rank values {votes}); the draft / target programs would "
            "carry different collectives, so the engine stops. Set them the same on every worker."
        )
    return votes


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
        # q38 L4 X-D14-DRAFT-2 (a): read once at build time (static per graph).
        self.embed_h8 = bool(vllm_torchtpu_envs.VLLM_TPU_DRAFT_EMBED_H8)
        # q38 L4 X-D14-DRAFT-2 (c): read once at build time; the replicated table is bound at
        # load (Qwen4ExpMTP.bind_replicated_embedding) as the buffer ``embed_full``.
        self.embed_repl = bool(vllm_torchtpu_envs.VLLM_TPU_DRAFT_EMBED_REPL)
        # q38 L4 EMB-REP-TGT: the borrowed (target) embedding module also gets the table, as the
        # buffer ``_q38_repl_table`` the TPU-patched VocabParallelEmbedding.forward looks up.
        self.embed_repl_tgt = bool(vllm_torchtpu_envs.VLLM_TPU_TARGET_EMBED_REPL)
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
        # The draft is traced once for every token bucket (see
        # draft_additional_config): its attention must not choose its
        # structure by the token count either, and its HCs keep the unfused
        # decode mix (TPU_QWEN4_HC_DECODE_FUSED picks the fused path by the
        # token count while tracing).
        for layer in self.layers:
            layer.self_attn.single_graph = True
        for module in self.modules():
            if isinstance(module, GatedResidual):
                module.disable_decode_fused()
        # Set by the Qwen3.8 megakernel driver before the first trace: the
        # layer's o_proj, both HCs and the MoE as one tail-kernel call.
        self.mk_tail: nn.Module | None = None
        # q38 L4 DRAFT1-B (X-D14-DRAFT-3 (a)): keep the ragged metadata int32 (no int64 casts, so no X64Split
        # pairs on TPU). A trace-time switch: spec_decode/draft_one.py sets it around the DH-D program's calls
        # only, so it is baked into that program and every other draft program traces with it off.
        self.meta_i32 = False
        # q38 L1-H3 (VLLM_TPU_MK_DRAFT_ATTN): set by attach_draft_attn after the tail; the fused loop's
        # attention as one draft-attention kernel call per pass (megakernel_runtime/qwen38_draft_attn.py).
        self.mk_attn: nn.Module | None = None

    def embed(self, input_ids):
        """The token embedding. With VLLM_TPU_DRAFT_EMBED_H8 (q38 L4 X-D14-DRAFT-2
        (a)) and TP > 1, this rank's masked lookup is summed by the armed H8 decode
        all-reduce (layers/adapter/tp_ar_decode.py: one Pallas call; rows it does not
        take fall back to lax.psum inside the op) instead of XLA's all-reduce, as the
        draft's attention output already is. When H8 is not armed the partial takes
        the stock all-reduce, which is what the embedding's own forward does.
        With VLLM_TPU_DRAFT_EMBED_REPL (q38 L4 X-D14-DRAFT-2 (c)) and TP > 1, the rows come from
        the replicated table bound at load (``replicated_lookup``), with no all-reduce at all.
        """
        emb = self.embed_tokens
        if getattr(self, "embed_repl", False) and int(getattr(emb, "tp_size", 1)) > 1:
            # q38 L4 X-D14-DRAFT-2 (c), VLLM_TPU_DRAFT_EMBED_REPL: the same rows from the
            # replicated table, no collective (takes precedence over DRAFT-2 (a)).
            table = getattr(self, "embed_full", None)
            if table is None:
                raise RuntimeError(
                    "VLLM_TPU_DRAFT_EMBED_REPL is set but the draft's replicated embedding was "
                    "never bound (Qwen4ExpMTP.bind_replicated_embedding at load); refusing to trace "
                    "a draft whose collectives would differ from its peers'"
                )
            return replicated_lookup(table, input_ids)
        if not self.embed_h8 or int(getattr(emb, "tp_size", 1)) <= 1:
            return emb(input_ids)
        from vllm.distributed import tensor_model_parallel_all_reduce

        from vllm_torchtpu.layers.adapter import tp_ar_decode

        partial = emb._tpu_partial(input_ids)
        if tp_ar_decode.use_kernel(partial):
            return tp_ar_decode.kernel_all_reduce(partial)
        return tensor_model_parallel_all_reduce(partial)

    def fuse_inputs(self, input_ids, hidden_states, inputs_embeds=None):
        """[T, hc*H] decoder input from the token and the previous hidden."""
        tokens = hidden_states.shape[0]
        embeds = self.embed(input_ids) if inputs_embeds is None else inputs_embeds
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
        context = (seq_lens - (starts[1:] - starts[:-1])).to(
            torch.int32 if self.meta_i32 else torch.int64
        )
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
            elif self.mk_attn is not None:  # q38 L1-H3: the fused loop's pass
                attention = self.mk_attn(block, pos, qsa_md, layer.self_attn)
            else:
                attention = layer.self_attn.attend(block, pos, qsa_md)
            hidden = tail(hidden, attention)
            return self.hyper_connection_mixer.mix(hidden)[0], hidden
        if rows is not None:
            hidden = layer.forward_rows(
                hidden,
                positions.to(torch.int32 if self.meta_i32 else torch.int64),
                rows.to(torch.int32 if self.meta_i32 else torch.int64),
                starts,
                torch.zeros_like(seq_lens, dtype=torch.int32),
                context,
                tables,
                tables,
            )
            return self.hyper_connection_mixer.mix(hidden)[0], hidden
        hidden = layer(
            hidden,
            positions.to(torch.int32 if self.meta_i32 else torch.int64),
            input_ids.to(torch.int32 if self.meta_i32 else torch.int64),
            starts,
            None,
            # Pooled QSA reads its ring slot and page tables from the
            # per-group forward-context metadata; these are placeholders.
            torch.zeros_like(seq_lens, dtype=torch.int32),
            context,
            (
                torch.arange(hidden.shape[0], device=hidden.device, dtype=torch.int32)
                if self.meta_i32
                else torch.arange(hidden.shape[0], device=hidden.device)
            )
            < starts[-1],
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
    # Borrowed from the target, except on a pipeline stage without the target's
    # embedding (draft_needs_own_embedding, set per instance in __init__).
    has_own_embed_tokens = False
    has_own_lm_head = False
    # Draft cache specs a prefill-only producer may hold
    # (Eagle3Proposer._validate_prefill_only_draft_model): the QSA main KV and
    # compressed keys (FullAttentionSpec family) and the raw ring (request state).
    prefill_only_kv_spec_types = (FullAttentionSpec, CircularBufferSpec)
    # Draft step 0 may pass `draft_rows` (each request's last token): the
    # draft then writes its QSA caches for every token but evaluates the
    # layer, and returns both outputs, at those rows only.
    supports_draft_rows = True
    # Identical to the target's: vLLM writes this onto the shared quant config.
    packed_modules_mapping = Qwen4ExpForCausalLM.packed_modules_mapping
    # q38 F2 (config.pd_prefill_mtp_producer): a pipeline producer runs this draft
    # (prefill-only, K = 1) on its last stage. vLLM's config check requires every
    # model under a pipeline to satisfy `SupportsPP`, which `supports_pp` tests by
    # attribute and by `forward` taking `intermediate_tensors` (as upstream's
    # Qwen3_5MTP, db1494576). The pipeline never splits this draft: the last stage
    # builds and runs it whole, so no stage ever makes its intermediate tensors.
    supports_pp: ClassVar[Literal[True]] = True
    # q38 X-D14-DRAFT-1: set from the engine's kv_role in __init__ (draft_kv_producer).
    kv_producer = False

    def make_empty_intermediate_tensors(self, batch_size, dtype, device):
        raise NotImplementedError(
            "the Qwen4Exp MTP draft runs whole on the last pipeline stage; it has "
            "no intermediate tensors"
        )

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
        self.has_own_embed_tokens = draft_needs_own_embedding(vllm_config)
        self.kv_producer = draft_kv_producer(vllm_config)
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

    def bind_replicated_embedding(self) -> bool:
        """VLLM_TPU_DRAFT_EMBED_REPL (q38 L4 X-D14-DRAFT-2 (c)): replicate the shared embedding.

        Called once per rank at load from ``bind_shared_selection`` (eagle3 ``load_model``), after
        the draft borrowed the target's ``embed_tokens`` and before the first trace. With a vocab-parallel (TP > 1) embedding, every rank first agrees on the
        flag (``agree_draft_embed_repl``, fail-stop). With the flag on, the full table is assembled
        on the host by ``gather_sharded_weight`` (the TP CPU group sums each rank's rows into
        zeros, exactly) and bound as the non-persistent buffer ``model.embed_full``; the draft's
        three per-step embedding all-reduces are then gone. Returns whether it bound.
        """
        model = getattr(self, "model", None)
        emb = getattr(model, "embed_tokens", None)
        if emb is None or int(getattr(emb, "tp_size", 1) or 1) <= 1:
            # No vocab-parallel embedding on this draft: no embedding collective to remove.
            return False
        on = bool(getattr(model, "embed_repl", False))
        tgt = bool(getattr(model, "embed_repl_tgt", False))
        votes = agree_draft_embed_repl(on, tgt)
        if tgt and not on:
            raise ValueError(
                "VLLM_TPU_TARGET_EMBED_REPL=1 needs VLLM_TPU_DRAFT_EMBED_REPL=1 (it reuses DRAFT-2 (c)'s "
                "replicated table); set both or neither."
            )
        if not on:
            return False
        weight = getattr(emb, "weight", None)
        if (
            hasattr(emb, "weight_scale")
            or weight is None
            or weight.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        ):
            raise ValueError(
                "VLLM_TPU_DRAFT_EMBED_REPL needs an unquantized shared embedding; this one is "
                f"{getattr(weight, 'dtype', None)}"
                f"{' with a weight_scale' if hasattr(emb, 'weight_scale') else ''}. "
                "Unset VLLM_TPU_DRAFT_EMBED_REPL."
            )
        from vllm_torchtpu.spec_decode.utils import gather_sharded_weight

        full, org_vocab, dim, _ = gather_sharded_weight(emb)
        if tuple(full.shape) != (int(emb.org_vocab_size), int(emb.embedding_dim)):
            raise ValueError(
                f"VLLM_TPU_DRAFT_EMBED_REPL: assembled table {tuple(full.shape)} != "
                f"({emb.org_vocab_size}, {emb.embedding_dim})"
            )
        if full.device.type == "tpu":
            from vllm_torchtpu.utils import synchronize_tensors

            synchronize_tensors(full)
        model.register_buffer("embed_full", full, persistent=False)
        if tgt:
            # The same tensor, no copy: the module is the target's embedding, borrowed by the draft.
            emb.register_buffer("_q38_repl_table", full, persistent=False)
            logger.info(
                "q38 EMB-REP-TGT ENGAGED: the target's embedding looks up the replicated table "
                "(the draft's, no extra HBM); its per-forward embedding all-reduce is gone."
            )
        logger.info(
            "q38 X-D14-DRAFT-2 (c) ENGAGED: the MTP draft embeds from a replicated table "
            "[%d, %d] %s (%.3f GiB on this rank); its per-pass embedding all-reduce is gone "
            "(TP ranks agree: %s).",
            org_vocab,
            dim,
            str(full.dtype).removeprefix("torch."),
            full.numel() * full.element_size() / 1024**3,
            votes,
        )
        return True

    def bind_shared_selection(self, target_model: nn.Module, max_tokens: int) -> bool:
        """VLLM_TPU_QSA_DRAFT_SHARED_SELECTION: the draft's QSA layers attend
        the target's last QSA layer selection instead of their own indexer.

        The target layer writes its ``selected`` rows into a shared buffer;
        draft step 0 reads each request's last verified row from it and
        carries those rows to the later steps. Returns whether it bound.

        Also the load hook of q38 X-D14-DRAFT-2 (c): the proposer calls this once per rank after the
        draft borrowed the target's embedding and before anything traces the draft (eagle3
        ``load_model``), whatever the flags, so the replicated-embedding bind and its all-rank
        agreement run first (``bind_replicated_embedding``; a no-op for a TP-1 embedding).
        """
        self.bind_replicated_embedding()
        if vllm_torchtpu_envs.VLLM_TPU_QSA_DRAFT_NO_INDEX_STATE and self.kv_producer:
            # q38 X-D14-DRAFT-1 is decode-only: refused on any producer, bound or not.
            raise ValueError(
                "VLLM_TPU_QSA_DRAFT_NO_INDEX_STATE=1 on a KV producer (kv_role kv_producer / "
                "kv_both): the draft's compressed keys, raw ring and tags would go unwritten "
                "and be shipped stale to the consumer. Set it on the decode (kv_consumer) only."
            )
        if not vllm_torchtpu_envs.VLLM_TPU_QSA_DRAFT_SHARED_SELECTION:
            return False
        backbone = getattr(target_model, "model", None)
        layers = [
            layer
            for layer in getattr(backbone, "layers", ())
            if getattr(layer, "layer_type", None) == "full_attention"
        ]
        drafts = [
            cast(Qwen4ExpDecoderLayer, layer).self_attn for layer in self.model.layers
        ]
        # The shared ops are built when the pool is bound, after this load-time
        # call; the forward takes the shared path only once they exist.
        if not layers:
            return False
        target = layers[-1].self_attn
        width = target.max_selected_tokens
        from vllm_torchtpu import hixglue as _hxg  # q38 HIX-GLUE SL

        alloc = _hxg.sink_alloc_width(width)  # q38 HIX-GLUE SL: == width with VLLM_TPU_HXG_SINK_LAYOUT off
        sink = torch.full(
            (max_tokens, alloc), -1, dtype=torch.int32, device=target.rope_cache.device
        )
        target.register_buffer("_selection_sink", sink, persistent=False)
        for attention in drafts:
            assert attention.max_selected_tokens == width
            attention.register_buffer("_selection_source", sink, persistent=False)
            attention.register_buffer(
                "_selection_rows", torch.full_like(sink, -1), persistent=False
            )
        if alloc != width:  # q38 HIX-GLUE SL: the per-rank bind line
            _hxg.bind_line(
                "VLLM_TPU_HXG_SINK_LAYOUT", _hxg.rank_of(target),
                "shared-selection sink + %d draft carried-rows buffer(s) [%d, %d] int32 (logical width %d, %d pad "
                "columns of -1, never read)" % (len(drafts), max_tokens, alloc, width, alloc - width),
            )
        return True

    # compute_logits is exactly lm head + logits processor (no per-step head), so
    # LogitsProcessor.get_top_tokens reproduces its argmax (VLLM_TPU_DRAFT_TOP_TOKENS).
    plain_compute_logits = True

    def compute_logits(self, hidden_states, spec_step_idx: int = 0):
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        parameters = dict(self.named_parameters())
        moe_mapper = moe_weights_mapper()
        layers = cast(nn.ModuleList, self.model.layers)
        moe_loaders = [AutoWeightsLoader(layer.mlp) for layer in layers]
        loaded: set[str] = set()
        source: set[str] = set()
        own_embed = bool(self.has_own_embed_tokens)
        for original, weight in weights:
            if own_embed and original in TARGET_EMBED_CHECKPOINT_NAMES:
                if "model.embed_tokens.weight" in loaded:
                    raise ValueError(f"Duplicate checkpoint embedding {original}")
                param = parameters["model.embed_tokens.weight"]
                getattr(param, "weight_loader", default_weight_loader)(param, weight)
                loaded.add("model.embed_tokens.weight")
                continue
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
        if own_embed and "model.embed_tokens.weight" not in loaded:
            missing.add("model.embed_tokens.weight (the shared checkpoint embedding)")
        if missing:
            raise ValueError(f"Incomplete Qwen4Exp MTP checkpoint: {sorted(missing)}")
        return set(parameters)
