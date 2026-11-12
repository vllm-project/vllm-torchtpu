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
"""Input embedding and lm_head sharded over every DP x TP rank.

vLLM-native data parallelism keeps each DP engine a full model replica outside
its MoE layers, so the input embedding and the lm_head are sharded only over
the engine's TP group. With a large vocabulary that replication is the largest
dense cost on the device: Qwen3.8-2.4T's untied 248,320 x 8,192 bf16 embedding
and lm_head take 3.79 GiB per device at DP16 x TP2, against 0.24 GiB sharded
over all 32 devices.

With TPU_DP_VOCAB_PARALLEL, both are built as vLLM VocabParallelEmbeddings over
the EP group, i.e. every DP x TP rank (EP rank = dp_rank * tp_size + tp_rank),
so each device holds 1 / (DP * TP) of each. Only the stock
VocabParallelEmbedding and ParallelLMHead over the model's text vocabulary are
re-sharded: this module replaces their forward and logits path, which a
subclass may override with its own reduction over the TP group, and other
tables (positions, per-layer or multimodal embedders) may run outside the
decoder's forward, where the engines are not in lockstep. Their collectives
span DP engines,
which is sound only because EP + DP runs the engines in lockstep: every step,
each engine runs the same number of forward chunks at the same DP-wide token
bucket (TPUModelRunner._dp_coordinated_step; idle engines run dummy chunks), and
every chunk, dummy or not, computes logits exactly once.

Embedding: all-gather the token ids over the DP group (an engine's TP ranks hold
the same ids), look up this rank's vocab shard, reduce-scatter over the DP group
(each engine gets its own tokens, summed over the shards of this TP index), then
all-reduce over the TP group (the other TP indices' shards). The lookup's
partial sums span every engine's tokens, DP x tokens x hidden per device; on a
v7x-8 at DP4 x TP2 that temporary stayed under the forward's other temporaries,
adding at most 0.15 GiB to peak HBM at 8,192 to 16,384 tokens per engine.

lm_head: every rank passes the same number of rows (TPUModelRunner pads each
logits batch to the step's DP-wide request bucket before the call and slices
the result back). All-gather the rows over the DP group (an engine's TP ranks
hold the same rows, so each row is projected once per TP index), project them
onto this rank's vocab shard, then all-to-all over the EP group: every TP rank
of an engine receives every shard's logits for that engine's rows, which
concatenate into its full-vocab logits.

Why the padding happens outside: TorchTPU pairs a collective across ranks by
program, so every rank must run the identical program around it. A row count
that differs between ranks, or a rank-dependent pad inside the program, hangs
(eager) or mismatches (compiled) the collectives.
"""

from typing import TYPE_CHECKING

import torch
from vllm.model_executor.layers.logits_processor import LogitsProcessor

from vllm_torchtpu import envs, utils
from vllm_torchtpu.distributed.collectives import all_to_all
from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.parallel_state import GroupCoordinator
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )

logger = init_logger(__name__)

# Set on every VocabParallelEmbedding (and ParallelLMHead) built over the EP
# group, and read by its forward and by TpuLogitsProcessor._get_logits.
DP_VOCAB_ATTR = "_tpu_dp_vocab"
# The shard bounds the TPU vocab patch registers as buffers on every
# VocabParallelEmbedding, so the compiled graph carries no rank-specific
# constants.
_SHARD_BOUND_BUFFERS = (
    "org_vocab_start_index",
    "org_vocab_end_index",
    "num_org_vocab_padding",
    "added_vocab_start_index",
    "added_vocab_end_index",
)


def _ineligible_reason(vllm_config: "VllmConfig") -> str | None:
    pc = vllm_config.parallel_config
    if not utils.dp_engines_in_lockstep(pc):
        return (
            "it needs data parallelism with expert parallelism, which steps the "
            "DP engines in lockstep"
        )
    if pc.pipeline_parallel_size > 1:
        return "pipeline parallelism is on"
    if pc.prefill_context_parallel_size > 1 or pc.decode_context_parallel_size > 1:
        return "context parallelism is on"
    if vllm_config.speculative_config is not None:
        return "speculative decoding is on (its logits batches exceed max_num_reqs)"
    if vllm_config.lora_config is not None:
        return "LoRA is on"
    if vllm_config.model_config.runner_type == "pooling":
        return (
            "pooling models compute no logits, so busy engines would skip the "
            "lm_head collectives that idle engines run"
        )
    if vllm_config.model_config.is_multimodal_model:
        # TODO: multimodal models embed outside the compiled forward, around
        # rank-local merges; enable once a multimodal DP + EP test covers it.
        return "multimodal models are not covered yet"
    return None


def group_for_new_layer(
    num_embeddings: int, org_num_embeddings: int | None, padding_size: int
) -> "GroupCoordinator | None":
    """The EP group when a VocabParallelEmbedding being built with these
    arguments should shard over every DP x TP rank, else None (keep vLLM's
    TP-group sharding)."""
    if not envs.TPU_DP_VOCAB_PARALLEL:
        return None
    from vllm.config import get_current_vllm_config
    from vllm.distributed.parallel_state import get_dp_group, get_ep_group, get_tp_group
    from vllm.model_executor.layers.vocab_parallel_embedding import pad_vocab_size

    vllm_config = get_current_vllm_config()
    reason = _ineligible_reason(vllm_config)
    if reason is not None:
        logger.warning_once(
            "Ignoring TPU_DP_VOCAB_PARALLEL: %s. The embedding and lm_head stay "
            "sharded over the TP group.",
            reason,
        )
        return None
    if num_embeddings != vllm_config.model_config.get_vocab_size():
        return None  # Not a table over the text vocabulary.
    ep, dp, tp = get_ep_group(), get_dp_group(), get_tp_group()
    if ep.world_size != dp.world_size * tp.world_size or ep.rank_in_group != (
        utils.ep_rank(
            dp_rank=dp.rank_in_group, tp_rank=tp.rank_in_group, tp_size=tp.world_size
        )
    ):
        raise RuntimeError(
            "TPU_DP_VOCAB_PARALLEL expects the EP group to be every DP x TP rank "
            "with ep_rank = dp_rank * tp_size + tp_rank; got "
            f"ep={ep.rank_in_group}/{ep.world_size}, "
            f"dp={dp.rank_in_group}/{dp.world_size}, "
            f"tp={tp.rank_in_group}/{tp.world_size}."
        )
    # Padded as in VocabParallelEmbedding.__init__, which splits the original
    # and the added vocab evenly over the group.
    org_vocab_size = org_num_embeddings or num_embeddings
    org_padded = pad_vocab_size(org_vocab_size, padding_size)
    padded = pad_vocab_size(org_padded + num_embeddings - org_vocab_size, padding_size)
    if org_padded % ep.world_size or padded % ep.world_size:
        logger.warning_once(
            "Ignoring TPU_DP_VOCAB_PARALLEL for a %d-row vocab: padded to %d rows "
            "(padding_size %d), it does not split over the %d DP x TP ranks. "
            "This layer stays sharded over the TP group.",
            num_embeddings,
            padded,
            padding_size,
            ep.world_size,
        )
        return None
    logger.info_once(
        "TPU_DP_VOCAB_PARALLEL: the embedding and lm_head are sharded over all "
        "%d DP x TP ranks (DP %d x TP %d).",
        ep.world_size,
        dp.world_size,
        tp.world_size,
    )
    return ep


def _groups() -> "tuple[GroupCoordinator, GroupCoordinator]":
    from vllm.distributed.parallel_state import get_dp_group, get_tp_group

    return get_dp_group(), get_tp_group()


def embedding_forward(
    layer: "VocabParallelEmbedding", input_: torch.Tensor
) -> torch.Tensor:
    """The embedding of this engine's `input_` from a vocab sharded over every
    DP x TP rank. Every engine passes the same number of ids."""
    import vllm.model_executor.layers.vocab_parallel_embedding as vpe

    dp, tp = _groups()
    ids = input_.reshape(-1)
    if dp.world_size > 1:
        ids = dp.all_gather(ids, dim=0)
    masked, mask = vpe.get_masked_input_and_mask(
        ids, *(getattr(layer, name) for name in _SHARD_BOUND_BUFFERS)
    )
    out = layer.quant_method.embedding(layer, masked.long())
    out = out.masked_fill(mask.unsqueeze(-1), 0)
    if dp.world_size > 1:
        out = dp.reduce_scatter(out, dim=0)
    if tp.world_size > 1:
        out = tp.all_reduce(out)
    return out.view(*input_.shape, out.shape[-1])


def get_logits(
    processor: "LogitsProcessor",
    hidden_states: torch.Tensor,
    lm_head: "VocabParallelEmbedding",
    embedding_bias: torch.Tensor | None,
) -> torch.Tensor:
    """Full-vocab logits for this rank's rows from an lm_head sharded over every
    DP x TP rank (the lm_head's parallel_group). Every rank passes the same
    number of rows."""
    group = lm_head.parallel_group
    assert group is not None
    dp, tp = _groups()
    rows = hidden_states.shape[0]
    gathered = dp.all_gather(hidden_states, dim=0)  # [dp * rows, hidden]
    local = processor._apply_head(lm_head, gathered, embedding_bias)
    # Block d * tp + t goes to EP rank d * tp + t: engine d's rows, the same
    # for every TP index t.
    send = local.view(dp.world_size, 1, rows, -1).expand(-1, tp.world_size, -1, -1)
    received = all_to_all(send.reshape(group.world_size, rows, -1), group)
    logits = received.transpose(0, 1).reshape(rows, -1)
    return logits[:, : processor.org_vocab_size]


def get_top_tokens(
    processor: "LogitsProcessor",
    hidden_states: torch.Tensor,
    lm_head: "VocabParallelEmbedding",
    embedding_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Greedy top token ids ``[rows]`` (int64) from an lm_head sharded over every
    DP x TP rank without exchanging full-vocab logits. Every rank passes the same
    number of rows."""
    if processor.scale <= 0.0 and processor.scale != 1.0:
        raise ValueError(
            "The local argmax reduction optimization is not supported for "
            "non-positive logit scaling factors."
        )
    group = lm_head.parallel_group
    assert group is not None
    dp, tp = _groups()
    rows = hidden_states.shape[0]
    gathered = (
        dp.all_gather(hidden_states, dim=0) if dp.world_size > 1 else hidden_states
    )
    local = processor._apply_head(lm_head, gathered, embedding_bias)
    if processor.soft_cap is not None:
        local = local / processor.soft_cap
        local = torch.tanh(local)
        local = local * processor.soft_cap
    if processor.scale != 1.0:
        local = local * processor.scale

    shard_width = local.shape[-1]
    vocab_start = getattr(
        lm_head,
        "org_vocab_start_index",
        lm_head.shard_indices.org_vocab_start_index,
    )
    vocab_end = getattr(
        lm_head,
        "org_vocab_end_index",
        lm_head.shard_indices.org_vocab_end_index,
    )
    global_vocab_ids = (
        torch.arange(shard_width, device=local.device, dtype=torch.int64) + vocab_start
    )
    valid_vocab_mask = (global_vocab_ids < vocab_end) & (
        global_vocab_ids < processor.org_vocab_size
    )
    local = local.masked_fill(~valid_vocab_mask, -float("inf"))

    local_max_vals, local_max_indices = local.max(dim=-1)
    global_indices = (local_max_indices + vocab_start).to(torch.int32)
    local_pair = torch.stack(
        [local_max_vals.float().view(torch.int32), global_indices], dim=-1
    )
    send = local_pair.view(dp.world_size, 1, rows, 2).expand(-1, tp.world_size, -1, -1)
    received = all_to_all(send.reshape(group.world_size, rows, 2), group)
    vals = received[:, :, 0].view(torch.float32)
    ids = received[:, :, 1]
    max_rank_idx = vals.argmax(dim=0, keepdim=True)
    return ids.gather(dim=0, index=max_rank_idx).squeeze(0).to(torch.int64)


class TpuLogitsProcessor(LogitsProcessor):
    """LogitsProcessor whose lm_head may be sharded over every DP x TP rank.

    register_logits_processor makes it the class every model's LogitsProcessor
    is built as, through vLLM's out-of-tree PluggableLayer replacement.
    """

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: "VocabParallelEmbedding",
        embedding_bias: torch.Tensor | None,
        skip_gather: bool = False,
    ) -> torch.Tensor | None:
        if not getattr(lm_head, DP_VOCAB_ATTR, False):
            return super()._get_logits(
                hidden_states, lm_head, embedding_bias, skip_gather
            )
        if skip_gather:
            raise NotImplementedError(
                "TPU_DP_VOCAB_PARALLEL: an lm_head sharded over DP has no "
                "TP-local logits to return (skip_gather)."
            )
        return get_logits(self, hidden_states, lm_head, embedding_bias)

    def get_top_tokens(
        self,
        lm_head: "VocabParallelEmbedding",
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not getattr(lm_head, DP_VOCAB_ATTR, False):
            return super().get_top_tokens(lm_head, hidden_states, embedding_bias)
        return get_top_tokens(self, hidden_states, lm_head, embedding_bias)


def register_logits_processor() -> None:
    """Under TPU_DP_VOCAB_PARALLEL, build every LogitsProcessor as a
    TpuLogitsProcessor."""
    from vllm.model_executor import custom_op

    if not envs.TPU_DP_VOCAB_PARALLEL:
        return
    if custom_op.op_registry_oot.get(LogitsProcessor.__name__) is TpuLogitsProcessor:
        return
    LogitsProcessor.register_oot(TpuLogitsProcessor)


def check_layers_sharded_over_dp(model: torch.nn.Module) -> bool:
    """Whether the model has an embedding or lm_head sharded over DP.

    Raises if the model cannot run them:
    - a layer sharded over DP shares a parameter with one sharded over the TP
      group: an lm_head tied to an input embedding that took the other sharding
      (a subclass, or a table TPU_DP_VOCAB_PARALLEL leaves alone);
    - a layer is sharded over DP but a LogitsProcessor of the model is not a
      TpuLogitsProcessor (a subclass, which the out-of-tree replacement does
      not reach; the model may pass its input embedding to it as the lm_head),
      or an lm_head is sharded over DP and the model has no LogitsProcessor.
    """
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        VocabParallelEmbedding,
    )

    owners: dict[int, tuple[str, bool]] = {}
    processors: dict[str, LogitsProcessor] = {}
    found = dp_lm_head = False
    for name, module in model.named_modules():
        if isinstance(module, LogitsProcessor):
            processors[name] = module
            continue
        if not isinstance(module, VocabParallelEmbedding):
            continue
        over_dp = bool(getattr(module, DP_VOCAB_ATTR, False))
        found |= over_dp
        dp_lm_head |= over_dp and isinstance(module, ParallelLMHead)
        for param in module.parameters(recurse=False):
            owner, owner_over_dp = owners.setdefault(id(param), (name, over_dp))
            if owner_over_dp != over_dp:
                dp_name, tp_name = (name, owner) if over_dp else (owner, name)
                raise NotImplementedError(
                    f"TPU_DP_VOCAB_PARALLEL: {dp_name} is sharded over all DP x "
                    f"TP ranks but shares a weight with {tp_name}, which stays "
                    "sharded over the TP group (tied embeddings). Unset "
                    "TPU_DP_VOCAB_PARALLEL for this model."
                )
    bypass = [
        f"{name} ({type(processor).__name__})"
        for name, processor in processors.items()
        if not isinstance(processor, TpuLogitsProcessor)
    ]
    if (found and bypass) or (dp_lm_head and not processors):
        raise NotImplementedError(
            "TPU_DP_VOCAB_PARALLEL: a layer is sharded over all DP x TP ranks, "
            "but the model computes logits without TpuLogitsProcessor "
            f"({', '.join(bypass) or 'no LogitsProcessor'}). Unset "
            "TPU_DP_VOCAB_PARALLEL for this model."
        )
    return found
