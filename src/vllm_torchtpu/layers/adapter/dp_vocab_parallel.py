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
so each device holds 1 / (DP * TP) of each. Their collectives span DP engines,
which is sound only because EP + DP runs the engines in lockstep: every step,
each engine runs the same number of forward chunks at the same DP-wide token
bucket (TPUModelRunner._dp_coordinated_step; idle engines run dummy chunks), and
every chunk, dummy or not, computes logits exactly once.

Embedding: all-gather the token ids over the DP group (an engine's TP ranks hold
the same ids), look up this rank's vocab shard, reduce-scatter over the DP group
(each engine gets its own tokens, summed over the shards of this TP index), then
all-reduce over the TP group (the other TP indices' shards).

lm_head: every rank passes the same fixed number of rows (the runner's
max_num_reqs; TPUModelRunner pads each logits batch to it before the call and
slices the result back). All-gather the rows over the EP group, project them
onto this rank's vocab shard, then all-to-all: each rank receives every shard's
logits for its own rows, which concatenate into its full-vocab logits.

Why the padding happens outside: TorchTPU pairs a collective across ranks by
program, so every rank must run the identical program around it. A row count
that differs between ranks, or a rank-dependent pad inside the program, hangs
(eager) or mismatches (compiled) the collectives.
"""

import functools
from typing import TYPE_CHECKING

import torch

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.parallel_state import GroupCoordinator
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )

logger = init_logger(__name__)

# Set on every VocabParallelEmbedding (and ParallelLMHead) built over the EP
# group, and read by its forward and by LogitsProcessor._get_logits.
DP_VOCAB_ATTR = "_tpu_dp_vocab"
# The lm_head's fixed logits row count; set_logits_rows writes it at model load.
ROWS_ATTR = "_tpu_dp_vocab_rows"


def _ineligible_reason(vllm_config: "VllmConfig") -> str | None:
    from vllm_torchtpu import utils

    pc = vllm_config.parallel_config
    if utils.get_dp_size(pc) <= 1 or not pc.enable_expert_parallel:
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
    if envs.TPU_GREEDY_TOP_TOKENS:
        return "TPU_GREEDY_TOP_TOKENS reads TP-sharded logits"
    return None


def group_for_new_layer() -> "GroupCoordinator | None":
    """The EP group when a VocabParallelEmbedding being built should shard over
    every DP x TP rank, else None (keep vLLM's TP-group sharding)."""
    if not envs.TPU_DP_VOCAB_PARALLEL:
        return None
    from vllm.config import get_current_vllm_config
    from vllm.distributed.parallel_state import get_dp_group, get_ep_group, get_tp_group

    reason = _ineligible_reason(get_current_vllm_config())
    if reason is not None:
        logger.warning_once(
            "Ignoring TPU_DP_VOCAB_PARALLEL: %s. The embedding and lm_head stay "
            "sharded over the TP group.",
            reason,
        )
        return None
    ep, dp, tp = get_ep_group(), get_dp_group(), get_tp_group()
    if ep.world_size != dp.world_size * tp.world_size or ep.rank_in_group != (
        dp.rank_in_group * tp.world_size + tp.rank_in_group
    ):
        raise RuntimeError(
            "TPU_DP_VOCAB_PARALLEL expects the EP group to be every DP x TP rank "
            "with ep_rank = dp_rank * tp_size + tp_rank; got "
            f"ep={ep.rank_in_group}/{ep.world_size}, "
            f"dp={dp.rank_in_group}/{dp.world_size}, "
            f"tp={tp.rank_in_group}/{tp.world_size}."
        )
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


def _all_to_all(x: torch.Tensor, group: "GroupCoordinator") -> torch.Tensor:
    """Block `d` of `x`'s leading axis goes to rank `d`; block `r` of the result
    came from rank `r`."""
    from torch.distributed import _functional_collectives as funcol

    out = funcol.all_to_all_single(x.contiguous(), None, None, group.device_group)
    if isinstance(out, funcol.AsyncCollectiveTensor):
        return funcol.wait_tensor(out)
    return out


def _shard_bound(layer: "VocabParallelEmbedding", name: str):
    # The TPU vocab patch registers the bounds as buffers so the compiled graph
    # carries no rank-specific constants; fall back to the plain ints.
    return getattr(layer, name, getattr(layer.shard_indices, name))


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
        ids,
        _shard_bound(layer, "org_vocab_start_index"),
        _shard_bound(layer, "org_vocab_end_index"),
        _shard_bound(layer, "num_org_vocab_padding"),
        _shard_bound(layer, "added_vocab_start_index"),
        _shard_bound(layer, "added_vocab_end_index"),
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
    DP x TP rank (the lm_head's parallel_group)."""
    rows = getattr(lm_head, ROWS_ATTR, None)
    if rows is None:
        raise RuntimeError(
            "TPU_DP_VOCAB_PARALLEL: the lm_head's logits row count was never set "
            "(dp_vocab_parallel.set_logits_rows at model load)."
        )
    if hidden_states.shape[0] != rows:
        raise ValueError(
            f"TPU_DP_VOCAB_PARALLEL: a logits batch of {hidden_states.shape[0]} "
            f"rows; every rank must pass exactly {rows} (pad before the call)."
        )
    group = lm_head.parallel_group
    assert group is not None
    gathered = group.all_gather(hidden_states, dim=0)  # [world * rows, hidden]
    local = processor._apply_head(lm_head, gathered, embedding_bias)
    received = _all_to_all(local.reshape(group.world_size, rows, -1), group)
    logits = received.transpose(0, 1).reshape(rows, -1)
    return logits[:, : processor.org_vocab_size]


def set_logits_rows(model: torch.nn.Module, rows: int) -> bool:
    """Fix the row count every rank pads its logits batches to. Returns whether
    the model has an embedding or lm_head sharded over DP."""
    found = False
    for module in model.modules():
        if getattr(module, DP_VOCAB_ATTR, False):
            setattr(module, ROWS_ATTR, int(rows))
            found = True
    return found


def install_logits_hook() -> None:
    """Route LogitsProcessor._get_logits for EP-sharded lm_heads to get_logits."""
    from vllm.model_executor.layers.logits_processor import LogitsProcessor

    if getattr(LogitsProcessor, "_tpu_dp_vocab_patch", False):
        return
    original = LogitsProcessor._get_logits

    @functools.wraps(original)
    def _get_logits(self, hidden_states, lm_head, embedding_bias, skip_gather=False):
        if not getattr(lm_head, DP_VOCAB_ATTR, False):
            return original(self, hidden_states, lm_head, embedding_bias, skip_gather)
        if skip_gather:
            raise NotImplementedError(
                "TPU_DP_VOCAB_PARALLEL: an lm_head sharded over DP has no "
                "TP-local logits to return (skip_gather)."
            )
        return get_logits(self, hidden_states, lm_head, embedding_bias)

    LogitsProcessor._get_logits = _get_logits
    LogitsProcessor._tpu_dp_vocab_patch = True
