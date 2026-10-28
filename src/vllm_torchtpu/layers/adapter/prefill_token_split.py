# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""TPU_PREFILL_SEQ_PARALLEL: the residual stream of a prefill step split by
tokens across the TP cores, for DeepSeek-V3.2-style models (GLM-5.2).

Today every TP core holds every token of the residual stream: o_proj ends in an
all-reduce, the fused MoE prefill kernel ends in a reduce-scatter and an
all-gather, and each core then runs the add + RMSNorm, the q / kv latent
projection (``fused_qkv_a_proj``) and the indexer key projection
(``wk_weights_proj``) on all tokens, though they work on each token alone and
their weights are replicated.

In a bucket of at least TPU_PREFILL_SEQ_PARALLEL_MIN_TOKENS tokens, from the
first MoE layer on, each core keeps only its rows: its block of each combine
chunk of the MoE kernel's reduce-scatter (``fused_moe_prefill_kernel_v2.
own_rows``), so the MoE output needs no reordering. Per layer:

- the add + RMSNorm and both projections run on the core's rows, and their
  outputs (2,784 columns on GLM-5.2) are all-gathered for attention and the
  indexer;
- o_proj runs without its reduction, and a reduce-scatter leaves the core its
  rows of the sum;
- the post-attention add + RMSNorm runs on the core's rows; the MoE op routes
  and quantizes them, all-gathers the fp8 rows, scales and routes, and drops
  the all-gather after its reduce-scatter (moe_tp_decode._split_body).

The final norm's output is all-gathered once for the LM head and the drafter.
The layers before the first MoE layer, smaller buckets and the drafter keep
today's path. Every sum adds the same terms; o_proj's reduce-scatter and the
norms' kernel (fewer rows per core) may round in another order.

``maybe_patch_deepseek_v2`` installs the model forward at platform activation;
it runs today's forward until ``arm`` (end of TPUModelRunner.load_model, after
moe_tp_decode and its prequant pairing) has checked the model and given it a
``TokenSplit``.
"""

from typing import Any

import jax
import torch
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from torch_tpu._internal import pallas

import vllm_torchtpu.envs as envs
from vllm_torchtpu.kernels.experimental.fused_moe_prefill.fused_moe_prefill_kernel_v2 import (  # noqa: E501
    gather_rows,
    own_rows,
    scatter_rows,
    token_chunks,
)
from vllm_torchtpu.layers.adapter import small_all_reduce_op
from vllm_torchtpu.layers.adapter.linear_common import PrequantSlot
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

TOKEN_SPLIT_ATTR = "_tpu_token_split"


def token_fn(mesh, kind: str):
    """The JAX function of `kind` ("own", "gather" or "scatter") over the MoE
    op's mesh, its operands' partition specs and its output row count, so the
    split follows the MoE kernel's reduce-scatter exactly. "own" takes the
    core's mesh index as data, an int32 [1, 1] per core (moe_tp_decode's rank
    operand): `lax.axis_index` lowers to a partition-id, which XLA's SPMD
    partitioner refuses in the model's whole-graph compile."""
    axis = mesh.axis_names[0]
    ranks = int(mesh.size)
    local, rows, specs = {
        "own": (
            lambda x, rank: own_rows(x, rank.reshape(()), ranks),
            lambda n: n // ranks,
            (P(axis), P(axis)),
        ),
        "gather": (lambda x: gather_rows(x, axis), lambda n: n * ranks, (P(axis),)),
        "scatter": (
            lambda x: scatter_rows(x, axis),
            lambda n: n // ranks,
            (P(axis),),
        ),
    }[kind]
    sharded = jax.shard_map(
        local, mesh=mesh, in_specs=specs, out_specs=P(axis), check_vma=False
    )

    def fn(*args: jax.Array) -> jax.Array:
        return sharded(
            *(jax.device_put(a, NamedSharding(mesh, s)) for a, s in zip(args, specs))
        )

    return fn, specs, rows


def _token_op(mesh, kind: str):
    """The torch op of `token_fn(mesh, kind)`."""
    run, specs, rows = token_fn(mesh, kind)
    axis = mesh.axis_names[0]

    # torch_tpu reads the operand types from these annotations.
    def fn_own(x: jax.Array, rank: jax.Array) -> jax.Array:
        return run(x, rank)

    def fn(x: jax.Array) -> jax.Array:
        return run(x)

    # pyrefly: ignore[no-matching-overload]
    op = pallas.jax_op(
        f"pallas::prefill_token_split_{kind}",
        fn_own if kind == "own" else fn,
        mesh=mesh,
        input_partition_specs=specs,
        output_partition_specs=P(axis),
    )
    op.register_fake(lambda x, *rest: x.new_empty((rows(x.shape[0]), *x.shape[1:])))
    return op


class TokenSplit:
    """The split of a model's residual stream; holds no module."""

    def __init__(
        self,
        mesh,
        first_layer: int,
        edge: int,
        moe_slot: PrequantSlot,
        rank_path: str,
    ):
        """`rank_path`: the attribute path from the first split layer to the
        MoE stage that holds this core's index in `mesh` as int32 [1, 1] data
        (the MoE op's rank operand). The split reads the buffer through that
        path, so the graph lifts it under the name the MoE layer gives it and
        buckets with and without the split share their compiled pieces."""
        self.ranks = int(mesh.size)
        self.rank_path = rank_path
        self.first_layer = first_layer
        self.edge = edge
        self.moe_slot = moe_slot
        self._own = _token_op(mesh, "own")
        self._gather = _token_op(mesh, "gather")
        self._scatter = _token_op(mesh, "scatter")

    def applies(self, tokens: int) -> bool:
        return tokens > self.edge and tokens % self.ranks == 0

    def rank_of(self, layer) -> torch.Tensor:
        stage = layer
        for name in self.rank_path.split("."):
            stage = getattr(stage, name)
        return stage._moe_tp_decode_rank

    def own_rows(self, x: torch.Tensor, rank: torch.Tensor) -> torch.Tensor:
        return self._own(x, rank)

    def gather(self, x: torch.Tensor, tokens: int) -> torch.Tensor:
        return self._gather(x).narrow(0, 0, tokens)

    def gather_columns(self, tokens: int, *parts: torch.Tensor):
        """Every core's rows of `parts`, side by side in one all-gather."""
        gathered = self.gather(torch.cat(parts, dim=-1), tokens)
        return gathered.split([part.shape[-1] for part in parts], dim=-1)

    def project_reduce_scatter(self, x: torch.Tensor, projection) -> torch.Tensor:
        """This core's rows of the row-parallel `projection` of x, summed
        over the cores. `retain` lifts the buffers today's all-reduce lifts."""
        partial = projection.quant_method.apply(projection, x, None)
        return self._scatter(small_all_reduce_op.retain(partial))

    def moe(self, mlp, x: torch.Tensor) -> torch.Tensor:
        self.moe_slot.value = True
        return mlp(x)


def _layer_forward(layer, positions, hidden_states, residual, split: TokenSplit):
    """DeepseekV2DecoderLayer.forward on this core's rows (bf16, no llama-4
    scaling, a MoE layer)."""
    if residual is None:
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)
    else:
        hidden_states, residual = layer.input_layernorm(hidden_states, residual)
    hidden_states = layer.self_attn.mla_attn(
        positions, hidden_states, None, token_split=split
    )
    hidden_states, residual = layer.post_attention_layernorm(hidden_states, residual)
    return split.moe(layer.mlp, hidden_states), residual


def split_model_forward(model, positions, hidden_states, split: TokenSplit):
    """DeepseekV2Model.forward's layers and final norm from the embedding,
    with the residual stream split from `split.first_layer` on."""
    tokens = positions.shape[0]
    torch._check(tokens % split.ranks == 0)
    residual = None
    for index, layer in enumerate(model.layers):
        if index < split.first_layer:
            hidden_states, residual = layer(positions, hidden_states, residual, None)
            continue
        if index == split.first_layer:
            rank = split.rank_of(layer)
            hidden_states = split.own_rows(hidden_states, rank)
            if residual is not None:
                residual = split.own_rows(residual, rank)
        hidden_states, residual = _layer_forward(
            layer, positions, hidden_states, residual, split
        )
    hidden_states, _ = model.norm(hidden_states, residual)
    return split.gather(hidden_states, tokens)


_UPSTREAM_MODEL_FORWARD: Any = None


def _model_forward(
    self,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors,
    inputs_embeds: torch.Tensor | None = None,
):
    split = getattr(self, TOKEN_SPLIT_ATTR, None)
    if (
        split is None
        or self.aux_hidden_state_layers
        or not split.applies(positions.shape[0])
    ):
        return _UPSTREAM_MODEL_FORWARD(
            self, input_ids, positions, intermediate_tensors, inputs_embeds
        )
    if inputs_embeds is None:
        assert input_ids is not None
        inputs_embeds = self.embed_input_ids(input_ids)
    return split_model_forward(self, positions, inputs_embeds, split)


def maybe_patch_deepseek_v2(*args: object, **kwargs: object) -> None:
    """Install the split-aware DeepseekV2Model.forward under
    TPU_PREFILL_SEQ_PARALLEL; it is today's forward until `arm` runs."""
    global _UPSTREAM_MODEL_FORWARD
    if not envs.TPU_PREFILL_SEQ_PARALLEL or _UPSTREAM_MODEL_FORWARD is not None:
        return None
    from vllm.model_executor.models import deepseek_v2

    _UPSTREAM_MODEL_FORWARD = deepseek_v2.DeepseekV2Model.forward
    deepseek_v2.DeepseekV2Model.forward = _model_forward
    logger.info("Applied TPU patch: DeepseekV2Model prefill token split forward.")
    return None


def _first_moe_layer(model) -> int | None:
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2MoE

    moe = [isinstance(layer.mlp, DeepseekV2MoE) for layer in model.layers]
    return moe.index(True) if True in moe else None


def _refusal(model, mesh_size: int | None = None) -> str | None:
    """Why `model` (a DeepseekV2Model) cannot run split, or None."""
    from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
    from vllm.model_executor.models.deepseek_v2 import (
        DeepseekV2MLAAttention,
        DeepseekV2MoE,
    )

    from vllm_torchtpu.layers.adapter.custom_ops.mla_attention_op import (
        VllmTPUMultiHeadLatentAttentionWrapper,
    )

    if _UPSTREAM_MODEL_FORWARD is None:
        return "the model forward is not patched"
    if get_pp_group().world_size != 1:
        return "pipeline parallelism"
    if get_tensor_model_parallel_world_size() < 2:
        return "no tensor parallelism"
    if getattr(model.config, "llama_4_scaling", None) is not None:
        return "llama-4 attention scaling"
    if (model.start_layer, model.end_layer) != (0, len(model.layers)):
        return "the model holds a slice of the layers"
    first = _first_moe_layer(model)
    if first is None or not all(
        isinstance(layer.mlp, DeepseekV2MoE) for layer in model.layers[first:]
    ):
        return "the MoE layers are not the last layers"
    if mesh_size is not None and mesh_size != get_tensor_model_parallel_world_size():
        return f"the MoE ops' mesh has {mesh_size} cores, not the TP group's"
    for layer in model.layers[first:]:
        attn = getattr(layer.self_attn, "mla_attn", None)
        if not isinstance(layer.self_attn, DeepseekV2MLAAttention) or not isinstance(
            attn, VllmTPUMultiHeadLatentAttentionWrapper
        ):
            return f"layer {layer.layer_idx} is not TPU MLA"
        if (
            attn.q_lora_rank is None
            or attn.kimi_preprocess is not None
            or attn.g_proj is not None
            or attn.fused_qkv_a_proj is None
        ):
            return f"layer {layer.layer_idx}'s MLA projections are not q-LoRA only"
        o_proj = attn.o_proj
        if getattr(o_proj, "bias", None) is not None or not o_proj.input_is_parallel:
            return f"layer {layer.layer_idx}'s o_proj has a bias or full input"
    return None


def arm(model: torch.nn.Module, token_buckets: list[int]) -> bool:
    """Give the target model's DeepseekV2Model a TokenSplit when every MoE
    layer runs moe_tp_decode's stage-A prefill kernel, the attention is the
    TPU MLA wrapper and every bucket the split serves divides into the MoE's
    chunk blocks; otherwise log why and keep today's path. Same decision on
    every rank (it reads only the model and the config)."""
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2Model

    from vllm_torchtpu.layers.adapter.moe_tp_decode import arm_token_split_op

    backbones = [m for m in model.modules() if isinstance(m, DeepseekV2Model)]
    if len(backbones) != 1:
        logger.warning("TPU_PREFILL_SEQ_PARALLEL: no single DeepseekV2Model; off")
        return False
    backbone = backbones[0]
    reason = _refusal(backbone)
    runners = []
    if reason is None:
        for layer in backbone.layers[_first_moe_layer(backbone) :]:
            found = [m for m in layer.mlp.modules() if isinstance(m, MoERunner)]
            if len(found) != 1:
                reason = f"layer {layer.layer_idx} has no single MoE runner"
                break
            runners.append(found[0])
    slot = PrequantSlot()
    armed = []
    if reason is None:
        for runner in runners:
            armed.append(arm_token_split_op(runner.routed_experts, slot))
            if armed[-1] is None:
                reason = "a MoE layer has no moe_tp_decode stage-A prefill kernel"
                break
    if reason is None and len({id(mesh) for mesh, _, _ in armed}) != 1:
        reason = "the MoE layers' ops use different meshes"
    if reason is None:
        reason = _refusal(backbone, int(armed[0][0].size))
    if reason is not None:
        logger.warning("TPU_PREFILL_SEQ_PARALLEL: %s; off", reason)
        return False
    mesh = armed[0][0]
    ranks = int(mesh.size)
    prefill_above = max(above for _, above, _ in armed)
    edge = max(
        envs.bucket_gate_edge(envs.TPU_PREFILL_SEQ_PARALLEL_MIN_TOKENS), prefill_above
    )
    uneven = [
        n
        for n in token_buckets
        if n > edge and n % ranks == 0 and n % (token_chunks(n) * ranks)
    ]
    if uneven:
        logger.warning(
            "TPU_PREFILL_SEQ_PARALLEL: buckets %s do not split into the MoE's "
            "chunk blocks over %d cores; off",
            uneven,
            ranks,
        )
        return False
    first = _first_moe_layer(backbone)
    assert first is not None
    stage = runners[0].routed_experts
    rank_path = next(n for n, m in backbone.layers[first].named_modules() if m is stage)
    setattr(backbone, TOKEN_SPLIT_ATTR, TokenSplit(mesh, first, edge, slot, rank_path))
    logger.info(
        "TPU_PREFILL_SEQ_PARALLEL: buckets above %d tokens split the residual "
        "stream over %d cores from layer %d of %d",
        edge,
        ranks,
        first,
        len(backbone.layers),
    )
    return True
