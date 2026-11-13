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
import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn import Parameter
from vllm.distributed.parallel_state import get_tp_group
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.utils import PPMissingLayer

from vllm_torchtpu.distributed.tp_collectives import tp_all_gather
from vllm_torchtpu.layers.core.sequence_layout import (
    AllSequenceLayoutPlanner,
    SequenceLayoutPlan,
)
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.utils import synchronize_tensors

logger = init_logger(__name__)


def _get_shard_meta_tensors(
    lm_head: torch.nn.Module, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    cached = getattr(lm_head, "_tp_shard_meta_dev", None)
    if cached is not None and cached[0].device == device:
        return cached
    start_t = torch.tensor(
        [int(lm_head.shard_indices.org_vocab_start_index)],
        dtype=torch.int32,
        device=device,
    )
    valid_t = torch.tensor(
        [
            int(
                lm_head.shard_indices.org_vocab_end_index
                - lm_head.shard_indices.org_vocab_start_index
            )
        ],
        dtype=torch.int32,
        device=device,
    )
    pair = (start_t, valid_t)
    object.__setattr__(lm_head, "_tp_shard_meta_dev", pair)
    return pair


def tp_local_argmax_logits(
    logits_processor,
    lm_head: torch.nn.Module,
    hidden_states: torch.Tensor,
    bias_local: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute global argmax across TP-sharded lm_head without all-gathering full vocab."""
    tp_size = getattr(lm_head, "tp_size", 1)
    if tp_size <= 1:
        logits = logits_processor(lm_head, hidden_states)
        if bias_local is not None:
            logits = logits + bias_local
        return logits.argmax(dim=-1)

    logits_local = logits_processor._get_logits(
        hidden_states, lm_head, None, skip_gather=True
    )
    if bias_local is not None:
        logits_local = logits_local + bias_local
    start_t, valid_t = _get_shard_meta_tensors(lm_head, logits_local.device)
    col_mask = (
        torch.arange(logits_local.shape[-1], device=logits_local.device) >= valid_t
    )
    logits_local = logits_local.masked_fill(col_mask, float("-inf"))
    local_val, local_idx = logits_local.max(dim=-1)
    global_idx = local_idx.to(torch.int32) + start_t
    packed = torch.stack(
        [local_val.to(torch.float32).view(torch.int32), global_idx], dim=-1
    )
    gathered = tp_all_gather(packed, dim=0).view(
        tp_size, packed.shape[0], 2
    )
    all_vals = gathered[..., 0].view(torch.float32)
    all_idxs = gathered[..., 1]
    best_rank = all_vals.argmax(dim=0, keepdim=True)
    return all_idxs.gather(0, best_rank).squeeze(0).to(torch.int64)


def _stateless_counter_uniform_u(
    row_indices: torch.Tensor,
    col_indices: torch.Tensor,
    seed_i32: torch.Tensor,
    min_val: float = 1e-4,
    span: float = 0.9998,
) -> torch.Tensor:
    """Zero-HBM 3-round bijective 32-bit counter hash in pure s32 VPU ALU."""
    h = (
        row_indices.to(torch.int32) * 0x5BD1E995
        ^ (col_indices.to(torch.int32) * 0x27D4EB2F)
        ^ seed_i32.to(torch.int32)
    )
    h = (h ^ ((h >> 16) & 0xFFFF)) * 0x7FEB352D
    h = (h ^ ((h >> 15) & 0x1FFFF)) * 0x165667B1
    h = h ^ ((h >> 16) & 0xFFFF)
    return ((h & 0x007FFFFF).to(torch.float32) + 0.5) * (span / 8388608.0) + min_val


def tp_local_random_spec_verify_logits(
    logits_processor,
    lm_head: torch.nn.Module,
    hidden_states: torch.Tensor,
    inv_temperatures_per_tok: torch.Tensor,
    row_draft_token_ids: torch.Tensor,
    seed_lo: torch.Tensor,
    u_local: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute exact global logsumexp, draft logit, and Gumbel-sampled token IDs across TP shards.

    Avoids materializing or AllGathering full-vocabulary [N, V] logits OR [N, V_shard] float32 uniform
    tensors in HBM by generating stateless counter-hash uniform variates in VPU registers when u_local
    is None and reducing each local shard to [N, 4] int32-bitcast scalars before tp_all_gather.
    """
    tp_size = getattr(lm_head, "tp_size", 1)
    if tp_size <= 1:
        logits_full = (
            logits_processor(lm_head, hidden_states).to(torch.float32)
            * inv_temperatures_per_tok
        )
        global_lse = torch.logsumexp(logits_full, dim=-1)
        safe_d = row_draft_token_ids.clamp(min=0, max=logits_full.shape[-1] - 1)
        draft_logit = torch.where(
            row_draft_token_ids.squeeze(-1) >= 0,
            logits_full.gather(-1, safe_d.to(torch.int64)).squeeze(-1),
            torch.full(
                (logits_full.shape[0],),
                -1e30,
                dtype=torch.float32,
                device=logits_full.device,
            ),
        )
        v_cols = torch.arange(
            logits_full.shape[-1], dtype=torch.int32, device=logits_full.device
        ).unsqueeze(0)
        if u_local is None:
            row_idx = torch.arange(
                logits_full.shape[0], dtype=torch.int32, device=logits_full.device
            ).unsqueeze(1)
            u_eff = _stateless_counter_uniform_u(row_idx, v_cols, seed_lo)
        else:
            u_eff = u_local
        gumbel_noise = -torch.log(-torch.log(u_eff) + 1e-30)
        scores = torch.where(
            v_cols == row_draft_token_ids,
            torch.full_like(logits_full, -1e30),
            logits_full + gumbel_noise,
        )
        sampled_ids = scores.argmax(dim=-1).to(torch.int32)
        return global_lse, draft_logit, sampled_ids

    logits_local = (
        logits_processor._get_logits(
            hidden_states, lm_head, None, skip_gather=True
        ).to(torch.float32)
        * inv_temperatures_per_tok
    )
    start_t, valid_t = _get_shard_meta_tensors(lm_head, logits_local.device)
    shard_cols = torch.arange(
        logits_local.shape[-1], dtype=torch.int32, device=logits_local.device
    )
    col_mask = shard_cols >= valid_t
    logits_local = logits_local.masked_fill(col_mask, float("-inf"))
    local_lse = torch.logsumexp(logits_local, dim=-1, keepdim=True)

    global_cols = shard_cols.unsqueeze(0) + start_t
    in_shard = (row_draft_token_ids >= start_t) & (
        row_draft_token_ids < (start_t + valid_t)
    )
    local_draft_idx = (row_draft_token_ids - start_t).clamp(
        min=0, max=logits_local.shape[-1] - 1
    )
    local_draft_logit = torch.where(
        in_shard,
        logits_local.gather(-1, local_draft_idx.to(torch.int64)),
        torch.full(
            (logits_local.shape[0], 1),
            -1e30,
            dtype=torch.float32,
            device=logits_local.device,
        ),
    )

    if u_local is None:
        row_idx = torch.arange(
            logits_local.shape[0], dtype=torch.int32, device=logits_local.device
        ).unsqueeze(1)
        u_eff = _stateless_counter_uniform_u(row_idx, global_cols, seed_lo)
    else:
        u_eff = u_local
    gumbel_noise = -torch.log(-torch.log(u_eff) + 1e-30)
    gumbel_scores = torch.where(
        global_cols == row_draft_token_ids,
        torch.full_like(logits_local, -1e30),
        logits_local + gumbel_noise,
    )
    local_best_score, local_best_idx = gumbel_scores.max(dim=-1, keepdim=True)
    global_idx = local_best_idx.to(torch.int32) + start_t

    packed = torch.cat(
        [
            local_lse.view(torch.int32),
            local_draft_logit.view(torch.int32),
            local_best_score.view(torch.int32),
            global_idx,
        ],
        dim=-1,
    )
    n_rows, raw_width = packed.shape[0], packed.shape[1]
    pad_width = ((raw_width + 127) // 128) * 128
    packed_aligned = (
        torch.cat((packed, torch.zeros((*packed.shape[:-1], pad_width - raw_width), dtype=packed.dtype, device=packed.device)), dim=-1)
        if pad_width > raw_width
        else packed
    )
    gathered = tp_all_gather(packed_aligned, dim=0).view(
        tp_size, n_rows, pad_width
    )[..., :raw_width]

    global_lse = torch.logsumexp(gathered[..., 0].view(torch.float32), dim=0)
    global_draft_logit = gathered[..., 1].view(torch.float32).max(dim=0).values
    winner_rank = gathered[..., 2].view(torch.float32).argmax(dim=0, keepdim=True)
    sampled_ids = gathered[..., 3].gather(0, winner_rank).squeeze(0).to(torch.int32)
    return global_lse, global_draft_logit, sampled_ids


def tp_local_topk_logits(
    logits_processor,
    lm_head: torch.nn.Module,
    hidden_states: torch.Tensor,
    k_top: int,
) -> torch.Tensor:
    """Compute exact global logsumexp + max tie rank + top-k (values, token_ids) with 128-lane aligned ICI gather."""
    k_eff = max(int(k_top), 1)
    tp_size = getattr(lm_head, "tp_size", 1)
    if tp_size <= 1:
        logits = logits_processor(lm_head, hidden_states)
        lse = torch.logsumexp(logits.to(torch.float32), dim=-1, keepdim=True)
        top_v, top_i = torch.topk(logits, k_eff, dim=-1)
        exact_rank = (logits >= top_v[:, 0:1]).sum(
            dim=-1, keepdim=True, dtype=torch.int32
        )
        return torch.cat(
            [
                lse.view(torch.int32),
                exact_rank,
                top_v.to(torch.float32).view(torch.int32),
                top_i.to(torch.int32),
            ],
            dim=-1,
        )

    logits_local = logits_processor._get_logits(
        hidden_states, lm_head, None, skip_gather=True
    )
    start_t, valid_t = _get_shard_meta_tensors(lm_head, logits_local.device)
    col_mask = (
        torch.arange(logits_local.shape[-1], device=logits_local.device) >= valid_t
    )
    logits_local = logits_local.masked_fill(col_mask, float("-inf"))
    local_lse = torch.logsumexp(logits_local.to(torch.float32), dim=-1, keepdim=True)
    local_v, local_i = torch.topk(logits_local, k_eff, dim=-1)
    local_ties = (logits_local >= local_v[:, 0:1]).sum(
        dim=-1, keepdim=True, dtype=torch.int32
    )
    global_i = local_i.to(torch.int32) + start_t
    packed = torch.cat(
        [
            local_lse.view(torch.int32),
            local_ties,
            local_v.to(torch.float32).view(torch.int32),
            global_i,
        ],
        dim=-1,
    )
    n_rows, raw_width = packed.shape[0], packed.shape[1]
    pad_width = ((raw_width + 127) // 128) * 128
    packed_aligned = (
        torch.cat((packed, torch.zeros((*packed.shape[:-1], pad_width - raw_width), dtype=packed.dtype, device=packed.device)), dim=-1)
        if pad_width > raw_width
        else packed
    )
    gathered = tp_all_gather(packed_aligned, dim=0).view(
        tp_size, n_rows, pad_width
    )[..., :raw_width]
    all_lse = gathered[..., 0:1].view(torch.float32)
    global_lse = torch.logsumexp(all_lse, dim=0)
    merged_v = (
        gathered[..., 2 : 2 + k_eff]
        .view(torch.float32)
        .permute(1, 0, 2)
        .reshape(n_rows, tp_size * k_eff)
    )
    merged_i = (
        gathered[..., 2 + k_eff :]
        .permute(1, 0, 2)
        .reshape(n_rows, tp_size * k_eff)
    )
    top_v, sel = torch.topk(merged_v, k_eff, dim=-1)
    top_i = merged_i.gather(-1, sel)
    shard_max = gathered[..., 2:3].view(torch.float32)
    shard_ties = gathered[..., 1:2]
    exact_rank = torch.where(
        shard_max == top_v[:, 0:1].unsqueeze(0), shard_ties, 0
    ).sum(dim=0, dtype=torch.int32)
    return torch.cat(
        [global_lse.view(torch.int32), exact_rank, top_v.view(torch.int32), top_i],
        dim=-1,
    )

if TYPE_CHECKING:
    from vllm.model_executor.models.interfaces import SupportsMultiModal

    from vllm_torchtpu.layers.core.attention_metadata import (
        AttentionMetadataBuilderContext,
    )

_DEFAULT_DRAFT_SEQUENCE_LAYOUT_PLAN = AllSequenceLayoutPlanner().prepare_dummy(
    num_tokens=0,
    num_reqs=0,
    kv_cache_initialized=False,
)


def fit_rows(t: torch.Tensor, rows: int, value: int = 0) -> torch.Tensor:
    """`t` with its leading dim sliced or padded (with `value`) to `rows`.

    `rows` is always a bucket, never the live request count: torch-tpu compiles
    a program per shape, so an exact-count slice costs a program per count.
    """
    n = t.shape[0]
    if n == rows:
        return t
    if n > rows:
        return t[:rows]
    pad = [0] * (2 * (t.dim() - 1)) + [0, rows - n]
    return F.pad(t, pad, value=value)


def dp_lockstep_num_reqs(dp_step_max_reqs: int, local_num_reqs: int) -> int:
    """Per-chunk request count all EP-DP lockstep ranks size the draft to: the
    step's all-reduced max (_dp_coordinated_step), not the static max_num_reqs."""
    num_reqs = max(1, dp_step_max_reqs)
    assert local_num_reqs <= num_reqs, (
        f"chunk has {local_num_reqs} requests but the DP-coordinated per-chunk "
        f"bound is {num_reqs}"
    )
    return num_reqs


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
    sequence_layout_plan: SequenceLayoutPlan = _DEFAULT_DRAFT_SEQUENCE_LAYOUT_PLAN
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


def handoff_seed_tokens(
    device_seed: torch.Tensor, req_ids: list[str], metadata: Any
) -> torch.Tensor:
    """Seeds a prefill-only draft's boundary row with the consumer's token.

    A Stage-3 producer drops the last prompt token for the decode peer to
    compute and samples a throwaway token in its place. The draft layer's KV
    row at the prompt boundary has to be built from the real continuation,
    which the send published in this step carries as
    ``SendMeta.next_prompt_token_id``. Requests without such a send keep the
    sampled seed.
    """
    reqs_to_send = getattr(metadata, "reqs_to_send", None)
    if not reqs_to_send:
        return device_seed
    indices: list[int] = []
    tokens: list[int] = []
    for index, req_id in enumerate(req_ids):
        send = reqs_to_send.get(req_id)
        token = None if send is None else getattr(send, "next_prompt_token_id", None)
        if token is not None:
            indices.append(index)
            tokens.append(int(token))
    if not indices:
        return device_seed
    seed = device_seed.clone()
    seed[torch.tensor(indices, dtype=torch.long, device=seed.device)] = torch.tensor(
        tokens, dtype=seed.dtype, device=seed.device
    )
    return seed


def gather_sharded_weight(
    sharded_module: VocabParallelEmbedding,
) -> tuple[torch.Tensor, int, int, torch.device]:
    """Gathers a sharded weight tensor across TP into a full replicated tensor."""
    tp_group = get_tp_group().cpu_group
    org_vocab = sharded_module.org_vocab_size
    dim = sharded_module.embedding_dim
    weight = cast(torch.Tensor, sharded_module.weight)

    si = sharded_module.shard_indices
    num_added = si.added_vocab_end_index - si.added_vocab_start_index
    if num_added > 0:
        raise NotImplementedError(
            "Draft models on TPU do not support LoRA with vocabulary expansion."
        )

    num_org = si.org_vocab_end_index - si.org_vocab_start_index
    full = torch.zeros(org_vocab, dim, dtype=torch.float32)
    my_rows = weight.data[:num_org].to(torch.float32).cpu()
    full[si.org_vocab_start_index : si.org_vocab_end_index] = my_rows
    dist.all_reduce(full, group=tp_group)

    draft_dtype = weight.dtype
    draft_device = weight.device
    full_dev = full.to(draft_dtype).to(draft_device)
    return full_dev, org_vocab, dim, draft_device


def populate_draft_embed_from_target(
    draft_model: torch.nn.Module, target_embed: VocabParallelEmbedding
) -> None:
    """Fill the draft's own (tp=1, full-vocab) embed_tokens with the
    target's embedding, assembled on the host.
    """
    draft_embed = draft_model.model.embed_tokens
    assert draft_embed.org_vocab_size == target_embed.org_vocab_size, (
        f"draft embed org_vocab {draft_embed.org_vocab_size} != target "
        f"{target_embed.org_vocab_size}; cannot populate"
    )

    full_dev, org_vocab, dim, draft_device = gather_sharded_weight(target_embed)

    new_embed = torch.nn.Embedding(org_vocab, dim, _weight=full_dev)
    new_embed.weight.requires_grad_(False)
    del draft_model.model.embed_tokens
    draft_model.model.embed_tokens = new_embed

    if draft_device.type == "tpu":
        synchronize_tensors(new_embed.weight)
    logger.info(
        "Draft embed_tokens replaced with full replicated nn.Embedding: "
        "%d x %d per worker.",
        org_vocab,
        dim,
    )


def maybe_share_embeddings(
    draft_model: torch.nn.Module,
    target_model: torch.nn.Module,
    tp_layout_matches: bool,
    force_share: bool = False,
) -> None:
    target_lm = (
        cast("SupportsMultiModal", target_model).get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    target_lm_model = getattr(target_lm, "model", None)
    target_embed = (
        getattr(target_lm_model, "embed_tokens", None)
        if target_lm_model is not None
        else None
    )

    if isinstance(target_embed, PPMissingLayer):
        # DeepseekV2-style targets build embed_tokens only on the first PP stage, so
        # the draft on the last stage keeps its own checkpoint-loaded embedding.
        draft_embed = getattr(getattr(draft_model, "model", None), "embed_tokens", None)
        needs_target = force_share or not getattr(
            draft_model, "has_own_embed_tokens", True
        )
        if draft_embed is None or needs_target:
            raise RuntimeError(
                "The target's embed_tokens is not on this pipeline stage and the "
                "draft has no loaded embed_tokens of its own"
            )
        logger.info(
            "Target embed_tokens is not on this pipeline stage; the draft "
            "keeps its own embed_tokens."
        )
        return

    if force_share:
        share_embed = True
    elif hasattr(draft_model, "has_own_embed_tokens"):
        share_embed = not draft_model.has_own_embed_tokens
    else:
        logger.info(
            "Draft model does not declare `has_own_embed_tokens`; "
            "defaulting to share embed_tokens with the target."
        )
        share_embed = True

    if share_embed:
        if target_embed is None:
            raise RuntimeError(
                "Draft embedding sharing requires target_lm.model.embed_tokens"
            )
        if tp_layout_matches:
            logger.info(
                "Sharing the target's embed_tokens with the draft using the "
                "same TP layout."
            )
            draft_model.model.embed_tokens = target_embed
        else:
            logger.info(
                "Populating draft's own embed_tokens with a host-gathered copy of "
                "target embedding."
            )
            populate_draft_embed_from_target(draft_model, target_embed)


def iter_mtp_shared_heads(draft_model: torch.nn.Module):
    """Yield the per-layer ``shared_head`` modules of a DeepSeek-style MTP draft.

    DeepSeek / GLM-5.2 MTP (`DeepSeekMTP`) has no top-level `lm_head`: each MTP
    layer owns the head at `model.layers.<N>.shared_head.head`, and
    `compute_logits` reaches it through there. Drafts that do carry a top-level
    head (eagle3, Qwen3.5 MTP) have no `shared_head` and yield nothing, so
    callers can walk this unconditionally.
    """
    inner = getattr(draft_model, "model", None)
    layers = getattr(inner, "layers", None)
    if layers is None:
        return
    items = layers.values() if isinstance(layers, torch.nn.ModuleDict) else layers
    for layer in items:
        shared_head = getattr(layer, "shared_head", None)
        if shared_head is not None and hasattr(shared_head, "head"):
            yield shared_head


def resolve_draft_logits_processor(draft_model: torch.nn.Module):
    """The draft's `LogitsProcessor`, wherever the draft class keeps it.

    Eagle3 / DFlash / Qwen3.5-MTP drafts expose it at the top level; a
    DeepSeek-style MTP draft keeps it on the inner `DeepSeekMultiTokenPredictor`
    (`draft_model.model.logits_processor`) and has none at the top level.
    """
    lp = getattr(draft_model, "logits_processor", None)
    if lp is None:
        inner = getattr(draft_model, "model", None)
        lp = getattr(inner, "logits_processor", None)
    return lp


def populate_draft_lm_head_from_target(
    draft_model: torch.nn.Module, target_lm_head: VocabParallelEmbedding
) -> None:
    shared_heads = list(iter_mtp_shared_heads(draft_model))
    if shared_heads:
        full_dev, org_vocab, dim, draft_device = gather_sharded_weight(target_lm_head)
        for shared_head in shared_heads:
            shared_head.head.weight = Parameter(full_dev, requires_grad=False)
            if draft_device.type == "tpu":
                synchronize_tensors(shared_head.head.weight)
        logger.info(
            "Populated %d MTP shared_head.head module(s) with a full "
            "replicated copy of the target lm_head: %d x %d per worker.",
            len(shared_heads),
            org_vocab,
            dim,
        )
        return

    draft_lm_head = (
        draft_model.model.lm_head
        if hasattr(draft_model, "model") and hasattr(draft_model.model, "lm_head")
        else getattr(draft_model, "lm_head", None)
    )
    if draft_lm_head is None:
        draft_lm_head = ParallelLMHead(
            target_lm_head.org_vocab_size, target_lm_head.embedding_dim
        )
        draft_model.lm_head = draft_lm_head

    full_dev, org_vocab, dim, draft_device = gather_sharded_weight(target_lm_head)

    draft_lm_head.weight = Parameter(full_dev, requires_grad=False)

    if draft_device.type == "tpu":
        synchronize_tensors(draft_lm_head.weight)
    logger.info(
        "Draft lm_head replaced with full replicated nn.Linear: %d x %d per worker.",
        org_vocab,
        dim,
    )


def maybe_share_lm_head(
    draft_model: torch.nn.Module,
    target_model: torch.nn.Module,
    draft_replicated: bool,
    tp_layout_matches: bool,
    force_share: bool = False,
    materialize_if_mismatched: bool = True,
) -> None:
    target_lm = (
        cast("SupportsMultiModal", target_model).get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    target_lm_head = getattr(target_lm, "lm_head", None)

    if target_lm_head is None:
        raise RuntimeError("Draft expects the target model to have an lm_head.")

    if force_share:
        share_lm_head = True
    elif hasattr(draft_model, "has_own_lm_head"):
        share_lm_head = not draft_model.has_own_lm_head
    else:
        logger.info(
            "Draft model does not declare `has_own_lm_head`; "
            "defaulting to share lm_head with the target."
        )
        share_lm_head = True

    if share_lm_head:
        if tp_layout_matches:
            logger.info(
                "Sharing the target's lm_head with the draft using the same TP layout."
            )
            # A DeepSeek-style MTP draft (DeepSeek / GLM-5.2) keeps its head at
            # model.layers.<N>.shared_head.head and has no top-level lm_head, so
            # the attribute walk below would bind a dead attribute and leave the
            # real head as loaded -- NaN logits when the checkpoint omits a copy
            # of the head at the MTP path. Bind the shared heads explicitly.
            shared_heads = list(iter_mtp_shared_heads(draft_model))
            for shared_head in shared_heads:
                del shared_head.head
                shared_head.head = target_lm_head
            if shared_heads:
                logger.info(
                    "Shared the target's lm_head with %d MTP "
                    "shared_head.head module(s).",
                    len(shared_heads),
                )

            if hasattr(draft_model, "lm_head"):
                draft_model.lm_head = target_lm_head
            elif hasattr(draft_model, "model") and hasattr(
                draft_model.model, "lm_head"
            ):
                draft_model.model.lm_head = target_lm_head
            elif not shared_heads:
                draft_model.lm_head = target_lm_head
        elif materialize_if_mismatched:
            logger.info(
                "Populating draft's own lm_head with a host-gathered, "
                "per-worker replicated copy of the target lm_head."
            )
            populate_draft_lm_head_from_target(draft_model, target_lm_head)

    # We must override _gather_logits for replicated draft models even if we don't share
    # the lm_head, because the draft model's logits_processor will still try to gather
    # logits across TP=1.
    if draft_replicated:
        lp = resolve_draft_logits_processor(draft_model)
        if lp is None:
            raise RuntimeError(
                "A replicated (draft_tp=1) draft needs its LogitsProcessor to "
                "skip the TP all-gather, but "
                f"{type(draft_model).__name__} exposes none at "
                "`logits_processor` or `model.logits_processor`. Without the "
                "override the draft would gather over a TP group it is not "
                "sharded across."
            )
        assert hasattr(lp, "_gather_logits"), (
            "draft logits_processor has no _gather_logits to override; "
            "vLLM may have renamed it."
        )
        lp._gather_logits = lambda logits: logits


# Noise-token id resolution order, mirroring upstream
# vllm/v1/worker/gpu/spec_decode/utils.py.
_NOISE_TOKEN_KEYS = (
    "mask_token_id",
    "dspark_noise_token_id",
    "pard_token",
    "ptd_token_id",
)


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
            f"(one of dflash_config.mask_token_id, {_NOISE_TOKEN_KEYS})."
        )
    dflash_config = dict(hf_dict.get("dflash_config", {}))
    dflash_config["mask_token_id"] = noise_id
    if not dflash_config.get("target_layer_ids"):
        resolved = _resolve_target_layer_ids(hf_dict)
        if resolved is not None:
            dflash_config["target_layer_ids"] = resolved
    hf_config.dflash_config = dflash_config
    return dflash_config
