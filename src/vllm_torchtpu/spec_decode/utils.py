import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import Parameter
from vllm.distributed.parallel_state import get_tp_group
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

from vllm_torchtpu.layers.core.sequence_layout import (
    AllSequenceLayoutPlanner, SequenceLayoutPlan)
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.utils import synchronize_tensors

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm_torchtpu.layers.core.attention_metadata import \
        AttentionMetadataBuilderContext

_DEFAULT_DRAFT_SEQUENCE_LAYOUT_PLAN = (
    AllSequenceLayoutPlanner().prepare_dummy(
        num_tokens=0,
        num_reqs=0,
        kv_cache_initialized=False,
    ))


@dataclass
class DraftChunkInputs:
    # Full padded request-major token ids for this draft chunk, captured before a
    # partial sequence layout packs or slices the target input.
    input_ids: torch.Tensor
    # Device tensor, [padded_chunk_tokens].
    position_ids: torch.Tensor
    # Host copy of the chunk-local cumsum of scheduled tokens. [num_reqs + 1]
    query_start_loc_np: np.ndarray
    # The chunk's attention-metadata builder context (chunk-local seq_lens /
    # query_start_loc / request_distribution device tensors, plus the
    # chunk's start_index for block-table slicing).
    attn_ctx: "AttentionMetadataBuilderContext"
    # First request of the chunk in batch order.
    start_index: int
    # Real (unpadded) request count in the chunk.
    num_reqs: int
    # Per-layer aux hidden states from the target forward; each is
    # [padded_chunk_tokens, aux_hidden]. Only consumed when the draft
    # checkpoint wants them.
    aux_hidden_states: list[torch.Tensor]
    # Production supplies the exact chunk-local plan. Dummy/default consumers
    # share this immutable ALL-layout no-op plan and PCP callers override it.
    sequence_layout_plan: SequenceLayoutPlan = \
        _DEFAULT_DRAFT_SEQUENCE_LAYOUT_PLAN
    # Chunk-local per-request draft count (device tensor, padded), a snapshot
    # of the chunk's spec_decode_metadata.draft_lengths. Lets the async draft
    # path read num_draft on-device instead of re-scanning the scheduler dict
    # + H2D every step. None when the chunk has no spec metadata.
    draft_lengths: torch.Tensor | None = None
    # Plain target hidden state for the chunk,[padded_chunk_tokens, hidden_size].
    # Used directly -- bypassing the aux-hidden-state concatenation.
    hidden_states: torch.Tensor | None = None
    # Optional pre-built attention metadata from the target model's forward pass
    attn_metadata: Any = None


@contextlib.contextmanager
def _force_draft_tp1():
    """Collapse the TP group to world_size=1 / rank=0 for the duration of draft
    construction + weight load, so the eagle3/mtp draft loads fully replicated
    (tp=1) on every worker instead of TP-sharded.

    NOTE: mutates the singleton GroupCoordinator returned by get_tp_group() in
    place. This is safe only because model loading is single-threaded — no
    other code reads tp.world_size or tp.rank_in_group concurrently. Do not
    widen this context manager to cover parallel operations.

    Currently acceptable given the single-threaded load. The cleaner long-term
    fix is to construct/pass a dedicated tp=1 GroupCoordinator to the draft
    load path instead of mutating the shared singleton; deferred to future work.
    """
    from vllm.distributed.parallel_state import get_tp_group
    tp = get_tp_group()
    saved_ws, saved_rank = tp.world_size, tp.rank_in_group
    tp.world_size = 1
    tp.rank_in_group = 0
    try:
        yield
    finally:
        tp.world_size = saved_ws
        tp.rank_in_group = saved_rank


def gather_sharded_weight(
    sharded_module: torch.nn.Module
) -> tuple[torch.Tensor, int, int, torch.device]:
    """Gathers a sharded weight tensor across TP into a full replicated tensor."""
    tp_group = get_tp_group().cpu_group
    org_vocab = sharded_module.org_vocab_size
    dim = sharded_module.embedding_dim

    si = sharded_module.shard_indices
    num_added = si.added_vocab_end_index - si.added_vocab_start_index
    if num_added > 0:
        raise NotImplementedError(
            "Draft models on TPU do not support LoRA with vocabulary expansion."
        )

    num_org = si.org_vocab_end_index - si.org_vocab_start_index
    full = torch.zeros(org_vocab, dim, dtype=torch.float32)
    my_rows = sharded_module.weight.data[:num_org].to(torch.float32).cpu()
    full[si.org_vocab_start_index:si.org_vocab_end_index] = my_rows
    dist.all_reduce(full, group=tp_group)

    draft_dtype = sharded_module.weight.dtype
    draft_device = sharded_module.weight.device
    full_dev = full.to(draft_dtype).to(draft_device)
    return full_dev, org_vocab, dim, draft_device


def populate_draft_embed_from_target(draft_model: torch.nn.Module,
                                     target_embed: torch.nn.Module) -> None:
    """Fill the draft's own (tp=1, full-vocab) embed_tokens with the
    target's embedding, assembled on the host.
    """
    draft_embed = draft_model.model.embed_tokens
    assert draft_embed.org_vocab_size == target_embed.org_vocab_size, (
        f"draft embed org_vocab {draft_embed.org_vocab_size} != target "
        f"{target_embed.org_vocab_size}; cannot populate")

    full_dev, org_vocab, dim, draft_device = gather_sharded_weight(
        target_embed)

    new_embed = torch.nn.Embedding(org_vocab, dim, _weight=full_dev)
    new_embed.weight.requires_grad_(False)
    del draft_model.model.embed_tokens
    draft_model.model.embed_tokens = new_embed

    if draft_device.type == "tpu":
        synchronize_tensors(new_embed.weight)
    logger.info(
        "Draft embed_tokens replaced with full replicated nn.Embedding: "
        "%d x %d per worker.", org_vocab, dim)


def maybe_share_embeddings(draft_model: torch.nn.Module,
                           target_model: torch.nn.Module,
                           tp_layout_matches: bool,
                           force_share: bool = False) -> None:
    target_lm = target_model.get_language_model() if hasattr(
        target_model, "get_language_model") else target_model
    target_lm_model = getattr(target_lm, "model", None)
    target_embed = getattr(target_lm_model, "embed_tokens",
                           None) if target_lm_model is not None else None

    if force_share:
        share_embed = True
    elif hasattr(draft_model, "has_own_embed_tokens"):
        share_embed = not draft_model.has_own_embed_tokens
    else:
        logger.info("Draft model does not declare `has_own_embed_tokens`; "
                    "defaulting to share embed_tokens with the target.")
        share_embed = True

    if share_embed:
        if target_embed is None:
            raise RuntimeError(
                "Draft embedding sharing requires target_lm.model.embed_tokens"
            )
        if tp_layout_matches:
            logger.info(
                "Sharing the target's embed_tokens with the draft using the same TP layout."
            )
            draft_model.model.embed_tokens = target_embed
        else:
            logger.info(
                "Populating draft's own embed_tokens with a host-gathered copy of target embedding."
            )
            populate_draft_embed_from_target(draft_model, target_embed)


def populate_draft_lm_head_from_target(
        draft_model: torch.nn.Module, target_lm_head: torch.nn.Module) -> None:
    draft_lm_head = draft_model.model.lm_head if hasattr(
        draft_model, "model") and hasattr(draft_model.model,
                                          "lm_head") else getattr(
                                              draft_model, "lm_head", None)
    if draft_lm_head is None:
        draft_lm_head = ParallelLMHead(target_lm_head.org_vocab_size,
                                       target_lm_head.embedding_dim)
        draft_model.lm_head = draft_lm_head

    full_dev, org_vocab, dim, draft_device = gather_sharded_weight(
        target_lm_head)

    draft_lm_head.weight = Parameter(full_dev, requires_grad=False)

    if draft_device.type == "tpu":
        synchronize_tensors(draft_lm_head.weight)
    logger.info(
        "Draft lm_head replaced with full replicated nn.Linear: "
        "%d x %d per worker.", org_vocab, dim)


def maybe_share_lm_head(draft_model: torch.nn.Module,
                        target_model: torch.nn.Module,
                        draft_replicated: bool,
                        tp_layout_matches: bool,
                        force_share: bool = False,
                        materialize_if_mismatched: bool = True) -> None:
    target_lm = target_model.get_language_model() if hasattr(
        target_model, "get_language_model") else target_model
    target_lm_head = getattr(target_lm, "lm_head", None)

    if target_lm_head is None:
        raise RuntimeError(
            "Draft expects the target model to have an lm_head.")

    if force_share:
        share_lm_head = True
    elif hasattr(draft_model, "has_own_lm_head"):
        share_lm_head = not draft_model.has_own_lm_head
    else:
        logger.info("Draft model does not declare `has_own_lm_head`; "
                    "defaulting to share lm_head with the target.")
        share_lm_head = True

    if share_lm_head:
        if tp_layout_matches:
            logger.info(
                "Sharing the target's lm_head with the draft using the same TP layout."
            )
            if hasattr(draft_model, "lm_head"):
                draft_model.lm_head = target_lm_head
            elif hasattr(draft_model, "model") and hasattr(
                    draft_model.model, "lm_head"):
                draft_model.model.lm_head = target_lm_head
            else:
                draft_model.lm_head = target_lm_head
        elif materialize_if_mismatched:
            logger.info("Populating draft's own lm_head with a host-gathered, "
                        "per-worker replicated copy of the target lm_head.")
            populate_draft_lm_head_from_target(draft_model, target_lm_head)

    # We must override _gather_logits for replicated draft models even if we don't share the lm_head,
    # because the draft model's logits_processor will still try to gather logits across TP=1.
    if draft_replicated:
        lp = getattr(draft_model, "logits_processor", None)
        if lp is not None:
            assert hasattr(lp, "_gather_logits"), (
                "draft logits_processor has no _gather_logits to override; "
                "vLLM may have renamed it.")
            lp._gather_logits = lambda logits: logits


# Noise-token id resolution order, mirroring upstream
# vllm/v1/worker/gpu/spec_decode/utils.py.
_NOISE_TOKEN_KEYS = ("mask_token_id", "dspark_noise_token_id", "pard_token",
                     "ptd_token_id")


def _resolve_target_layer_ids(hf_dict: dict) -> list[int] | None:
    """Target layer indices whose hidden states feed the drafter.

    Mirrors upstream eagle3_utils.get_eagle3_aux_layers_from_config:
    ``eagle_aux_hidden_state_layer_ids`` stores aux ids (target + 1);
    ``dspark_target_layer_ids`` / ``target_layer_ids`` store target ids.
    """
    aux_ids = hf_dict.get("eagle_aux_hidden_state_layer_ids")
    if aux_ids:
        return [idx - 1 for idx in aux_ids]
    for key in ("dspark_target_layer_ids", "target_layer_ids"):
        ids = hf_dict.get(key)
        if ids:
            return list(ids)
    return None


def normalize_draft_config(hf_config) -> dict:
    """Resolve parallel draft config spellings into ``dflash_config``. Idempotent.

    DFlashProposer and DSparkProposer read dflash_config.mask_token_id and
    load_model reads dflash_config.target_layer_ids; feed both the
    resolved values so the whole base machinery applies unchanged.

    This must also run BEFORE GPUModelRunner.__init__: the upstream V1
    runner transiently constructs its own EagleProposer for any
    use_eagle() method (the TPU runner replaces it afterwards), and its
    _init_parallel_drafting_params only knows pard_token / ptd_token_id /
    dflash_config.mask_token_id — not dspark_noise_token_id.
    """
    hf_dict = hf_config.to_dict() if hasattr(hf_config, "to_dict") else {}

    noise_id = hf_dict.get("dflash_config", {}).get("mask_token_id")
    for key in _NOISE_TOKEN_KEYS:
        if noise_id is not None:
            break
        noise_id = hf_dict.get(key)
    if noise_id is None:
        raise ValueError(
            "Parallel drafter requires a noise/mask token id in the draft config "
            f"(one of dflash_config.mask_token_id, {_NOISE_TOKEN_KEYS}).")
    dflash_config = dict(hf_dict.get("dflash_config", {}))
    dflash_config["mask_token_id"] = noise_id
    if not dflash_config.get("target_layer_ids"):
        resolved = _resolve_target_layer_ids(hf_dict)
        if resolved is not None:
            dflash_config["target_layer_ids"] = resolved
    hf_config.dflash_config = dflash_config
    return dflash_config
