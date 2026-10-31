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
"""TPU_MOE_PREFILL_SEQUENCE_PARALLEL: a sequence-parallel residual stream
around the MoE, per compiled bucket, for DeepSeek-V2-family MLA models served
with TP = EP over one group (GLM-5.2).

On a backbone bucket `applies` takes (at least `MIN_TOKENS` tokens), every MoE
decoder layer runs with the residual sharded over the TP group, rank `r`
owning the contiguous rows `[r * T / tp, (r + 1) * T / tp)`:

  * the input norm runs on the rank's rows and its output is all-gathered for
    attention, which is unchanged (it sees all `T` rows, DCP included);
  * o_proj reduce-scatters its partial sums instead of all-reducing them;
  * the post-attention norm, the router, the routed experts and the shared
    expert run on the rank's rows only. The routed experts go through the
    fused EP kernel over the rank's own tokens with a bf16 result wire
    (`fused_moe_ep.build_sequence_parallel_op`), so the MoE output needs no
    all-reduce; the shared expert is re-sharded at load to TP within groups
    of `SHARED_GROUP` ranks (`member_slice`), which exchange their rows
    (`kernels/fused_moe/grouped_shared_expert.py`).

The dense layers in front of the first MoE layer keep the replicated stream,
and the model norm all-gathers the final rows. Buckets below `MIN_TOKENS`
(decode) trace today's forward unchanged.

The backbone warmup hands a bucket between traces only when they lift the same
buffers, so both branches of every MoE layer first route the layer's input
through every tensor either branch reads (`_retain_buffers`, no data read):
the first access names each buffer the same way in both traces.
"""

import inspect
from itertools import islice
from typing import Any

import jax
import torch
from jax.sharding import PartitionSpec
from torch_tpu._internal.pallas import jax_op
from vllm.compilation import backends as vllm_compilation_backends
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_reduce_scatter,
)
from vllm.model_executor.models import deepseek_v2

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.ep_mesh import EP_AXIS_NAME, build_ep_mesh, ep_mesh_index
from vllm_torchtpu.kernels.fused_moe.grouped_shared_expert import (
    grouped_shared_expert_sharded,
)
from vllm_torchtpu.layers.adapter import fused_moe_decode, fused_moe_ep
from vllm_torchtpu.layers.adapter.attention import _retain_buffers
from vllm_torchtpu.layers.adapter.linear_common import (
    KEEP_VLLM_LAYOUT_ATTR,
    WEIGHT_FLIPPED_ATTR,
)
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.utils import fresh_buffer

logger = init_logger(__name__)

# Smallest bucket that takes the split (a power of two: `bucket_gate_edge`).
# The same edge as TPU_DCP_PREFILL_TOKEN_PARALLEL, so the two split the same
# buckets.
MIN_TOKENS = 1024

# On an armed decoder layer: every tensor either branch reads, in a fixed order.
KEEP_ATTR = "_tpu_sp_keep"
# On an armed decoder layer's mlp: the replicated shared expert and the op.
SHARED_W13 = "_tpu_sp_shared_w13"
SHARED_S13 = "_tpu_sp_shared_s13"
SHARED_W2 = "_tpu_sp_shared_w2"
SHARED_S2 = "_tpu_sp_shared_s2"
SHARED_OP = "_tpu_sp_shared_op"
OP_ATTR = "_tpu_sp_moe_op"
SCALE_ATTR = "_tpu_sp_routed_scale"

UPSTREAM_DECODER_LAYER_FORWARD = deepseek_v2.DeepseekV2DecoderLayer.forward
UPSTREAM_MODEL_FORWARD = deepseek_v2.DeepseekV2Model.forward


def applies(num_tokens: int) -> bool:
    """Whether this bucket (static at trace time) takes the split. The
    backbone only: it is traced once per bucket group."""
    return (
        envs.TPU_MOE_PREFILL_SEQUENCE_PARALLEL
        and vllm_compilation_backends.model_tag == "backbone"
        and num_tokens > envs.bucket_gate_edge(MIN_TOKENS)
    )


# ---------------------------------------------------------------------------
# The shared expert, re-sharded from TP over all ranks to TP within groups
# ---------------------------------------------------------------------------

# Ranks per shared-expert group (consecutive EP mesh indices). Each rank holds
# 1/SHARED_GROUP of the expert: replicating it whole (+2.8 GB per core on
# GLM-5.2) does not fit next to the served KV budget.
SHARED_GROUP = 4


def merge_gate_up(
    weight: torch.Tensor, scale: torch.Tensor, shards: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """`shards` merged gate/up TP shards side by side (`[K, shards * 2i]`, each
    `[gate_r | up_r]`) as one `[gate | up]` projection, with their per-channel
    scales (`[shards * 2i]`). Each shard's scales already cover the whole
    contraction, so this is a column permutation: no value changes."""
    k, n = weight.shape
    i = n // (2 * shards)
    if n != 2 * shards * i or scale.shape != (n,):
        raise ValueError(f"gate_up shards {tuple(weight.shape)} / {tuple(scale.shape)}")
    order = torch.arange(n, device=weight.device).view(shards, 2, i).transpose(0, 1)
    order = order.reshape(n)
    return weight.index_select(1, order), scale.index_select(0, order)


def member_slice(
    w13: torch.Tensor,
    s13: torch.Tensor,
    w2: torch.Tensor,
    s2: torch.Tensor,
    *,
    tp: int,
    group: int,
    member: int,
) -> tuple[torch.Tensor, ...]:
    """Group member `member`'s share of a shared expert, from all `tp` TP
    shards gathered: `w13 [K, tp * 2i]` / `s13 [tp * 2i]` (shard r's columns
    `[gate_r | up_r]`), `w2 [tp * i, N]` / `s2 [tp, N]` (shard r's rows, scaled
    per shard). The member takes `tp / group` consecutive shards: its merged
    `[gate | up]` columns and the matching down rows, block-scaled with one
    block per shard, so every fp8 value and scale is kept as it was."""
    per = tp // group
    if per * group != tp or not 0 <= member < group:
        raise ValueError(f"tp={tp} does not split into {group} groups ({member})")
    cols = w13.shape[1] // tp
    lo, hi = member * per, (member + 1) * per
    gu, gu_s = merge_gate_up(
        w13[:, lo * cols : hi * cols], s13[lo * cols : hi * cols], per
    )
    rows = w2.shape[0] // tp
    return gu, gu_s, w2[lo * rows : hi * rows], s2[lo:hi]


def _gather(t: torch.Tensor, dim: int) -> torch.Tensor:
    """All-gather over the TP group at load. fp8 is gathered (and sliced by
    the caller) as bf16, which holds every e4m3 value exactly."""
    if t.dtype == torch.float8_e4m3fn:
        t = t.to(torch.bfloat16)
    return get_tp_group().all_gather(t, dim)


def _per_channel_fp8(linear: Any) -> tuple[torch.Tensor, torch.Tensor]:
    w, s = linear.weight, getattr(linear, "weight_scale", None)
    if (
        w.dtype != torch.float8_e4m3fn
        or s is None
        or s.dim() != 1
        or not getattr(linear, WEIGHT_FLIPPED_ATTR, False)
        or getattr(linear, KEEP_VLLM_LAYOUT_ATTR, False)
        or getattr(linear, "bias", None) is not None
    ):
        raise NotImplementedError(
            "TPU_MOE_PREFILL_SEQUENCE_PARALLEL re-shards a shared expert held as "
            "per-channel fp8 (k, n) linears without bias; got "
            f"{w.dtype} weight {tuple(w.shape)}, scale "
            f"{None if s is None else tuple(s.shape)}"
        )
    return w.data, s.data


def _group_shared_expert(mlp: Any, tp: int, rank: int) -> None:
    shared = mlp.shared_experts
    w13, s13 = _per_channel_fp8(shared.gate_up_proj)
    w2, s2 = _per_channel_fp8(shared.down_proj)
    fp8 = w13.dtype
    parts = member_slice(
        _gather(w13, 1),
        _gather(s13, 0),
        _gather(w2, 0),
        _gather(s2[None], 0),
        tp=tp,
        group=SHARED_GROUP,
        member=rank % SHARED_GROUP,
    )
    for name, t, dtype in zip(
        (SHARED_W13, SHARED_S13, SHARED_W2, SHARED_S2),
        parts,
        (fp8, torch.float32, fp8, torch.float32),
    ):
        # A fresh buffer, not a view of the gathered shards: torch_tpu hands the
        # executable the buffer behind a tensor, and peer-replayed programs
        # then disagree with this rank's argument shapes.
        mlp.register_buffer(name, fresh_buffer(t.to(dtype)), persistent=False)


_SHARED_OPS: dict[tuple, Any] = {}


def _build_shared_op(group: int) -> Any:
    """The grouped shared expert as one SPMD op over the EP mesh (built at
    load, outside Dynamo)."""
    key = (group,)
    if key in _SHARED_OPS:
        return _SHARED_OPS[key]
    mesh = build_ep_mesh()
    # jax_op hands fn the global arrays: the per-rank body goes under shard_map.
    sharded = grouped_shared_expert_sharded(mesh, EP_AXIS_NAME, group)

    def fn(
        x: jax.Array, w13: jax.Array, s13: jax.Array, w2: jax.Array, s2: jax.Array
    ) -> jax.Array:
        return sharded(x, w13, s13, w2, s2)

    spec = PartitionSpec(EP_AXIS_NAME)
    op = jax_op(
        f"pallas::moe_prefill_sp_shared_g{group}",
        fn,
        mesh=mesh,
        input_partition_specs=(spec,) * 5,
        output_partition_specs=spec,
    )
    op.register_fake(lambda x, *_args, **_kw: torch.empty_like(x))
    _SHARED_OPS[key] = op
    return op


def _shared_expert(mlp: Any, x: torch.Tensor) -> torch.Tensor:
    return getattr(mlp, SHARED_OP)(
        x,
        getattr(mlp, SHARED_W13),
        getattr(mlp, SHARED_S13),
        getattr(mlp, SHARED_W2),
        getattr(mlp, SHARED_S2),
    )


# ---------------------------------------------------------------------------
# Forwards
# ---------------------------------------------------------------------------


class _OProjReduceScatter:
    """The MLA op's `prefill_group`: o_proj's partial sums, reduce-scattered to
    each rank's rows instead of all-reduced."""

    @staticmethod
    def project_reduce_scatter(x: torch.Tensor, o_proj: Any) -> torch.Tensor:
        partial = o_proj.quant_method.apply(o_proj, x)
        return tensor_model_parallel_reduce_scatter(partial, 0)


_O_PROJ_RS = _OProjReduceScatter()


def _check_divides(num_tokens: int) -> int:
    tp = get_tensor_model_parallel_world_size()
    torch._check(num_tokens % tp == 0)
    return tp


def own_rows(x: torch.Tensor, rank: torch.Tensor, tp: int) -> torch.Tensor:
    """Rank `rank`'s contiguous `1/tp` of the rows of `x`. The rank is data
    (int32 `[1, 1]`), so every rank traces the same program."""
    rows = x.view(tp, x.shape[0] // tp, *x.shape[1:])
    return rows.index_select(0, rank.view(1)).view(x.shape[0] // tp, *x.shape[1:])


def _all_rows(x: torch.Tensor, num_tokens: int) -> torch.Tensor:
    # narrow, not [:n]: keeps the token dim the traced symbol itself.
    return tensor_model_parallel_all_gather(x, 0).narrow(0, 0, num_tokens)


def _moe(mlp: Any, x: torch.Tensor) -> torch.Tensor:
    """The MoE block on this rank's rows: routed (fused EP, own tokens) times
    the routed scale, plus the shared expert -- the runner's order."""
    experts = mlp.experts.routed_experts
    router_logits, _ = mlp.gate(x)
    routed = fused_moe_ep.run_sequence_parallel_op(
        getattr(mlp, OP_ATTR),
        x,
        experts.w13_weight,
        experts.w2_weight,
        experts.w13_weight_scale_inv,
        experts.w2_weight_scale_inv,
        router_logits,
        fused_moe_ep.rank_operand(),
        experts.e_score_correction_bias,
    )
    scale = getattr(mlp, SCALE_ATTR)
    if scale != 1.0:
        routed = routed * scale
    return _shared_expert(mlp, x) + routed


def _sequence_parallel_layer_forward(
    layer: Any,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    llama_4_scaling: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = positions.shape[0]
    tp = _check_divides(num_tokens)
    sharded_in = hidden_states.shape[0] != num_tokens
    if residual is None:
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)
    else:
        hidden_states, residual = layer.input_layernorm(hidden_states, residual)
    if sharded_in:
        hidden_states = _all_rows(hidden_states, num_tokens)
    hidden_states = layer.self_attn.mla_attn(
        positions, hidden_states, llama_4_scaling, prefill_group=_O_PROJ_RS
    )
    if not sharded_in:
        residual = own_rows(residual, fused_moe_ep.rank_operand(), tp)
    hidden_states, residual = layer.post_attention_layernorm(hidden_states, residual)
    return _moe(layer.mlp, hidden_states), residual


def patched_decoder_layer_forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    llama_4_scaling: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    keep = getattr(self, KEEP_ATTR, None)
    if keep is None:
        return UPSTREAM_DECODER_LAYER_FORWARD(
            self, positions, hidden_states, residual, llama_4_scaling
        )
    hidden_states = _retain_buffers(hidden_states, *keep)
    if not applies(positions.shape[0]):
        return UPSTREAM_DECODER_LAYER_FORWARD(
            self, positions, hidden_states, residual, llama_4_scaling
        )
    return _sequence_parallel_layer_forward(
        self, positions, hidden_states, residual, llama_4_scaling
    )


def patched_model_forward(
    self,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: Any = None,
    inputs_embeds: torch.Tensor | None = None,
) -> torch.Tensor:
    num_tokens = positions.shape[0]
    if not applies(num_tokens) or not getattr(self, KEEP_ATTR, False):
        return UPSTREAM_MODEL_FORWARD(
            self, input_ids, positions, intermediate_tensors, inputs_embeds
        )
    hidden_states = (
        inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
    )
    residual = None
    for layer in islice(self.layers, self.start_layer, self.end_layer):
        hidden_states, residual = layer(positions, hidden_states, residual, None)
    hidden_states, _ = self.norm(hidden_states, residual)
    if hidden_states.shape[0] != num_tokens:
        hidden_states = _all_rows(hidden_states, num_tokens)
    return hidden_states


def install_forwards() -> None:
    """Install the per-bucket forwards (`models/vllm/deepseek_v2_patch.py`).
    Before the model is built: the compile wrapper binds the model's forward
    at construction."""
    deepseek_v2.DeepseekV2DecoderLayer.forward = patched_decoder_layer_forward
    deepseek_v2.DeepseekV2Model.forward = patched_model_forward
    logger.info("Applied TPU patch: per-bucket sequence-parallel MoE forwards.")


# ---------------------------------------------------------------------------
# Arming, at the end of load_model
# ---------------------------------------------------------------------------


def check_config(vllm_config: Any) -> None:
    """Raise for a server config the split does not serve."""
    parallel = vllm_config.parallel_config
    refusals = {
        "needs --enable-expert-parallel": not parallel.enable_expert_parallel,
        "needs data_parallel_size 1": parallel.data_parallel_size != 1,
        "needs pipeline_parallel_size 1": parallel.pipeline_parallel_size != 1,
        "needs tensor_parallel_size > 1": parallel.tensor_parallel_size < 2,
        "does not take speculative decoding (the drafter is traced once for "
        "every bucket)": vllm_config.speculative_config is not None,
        "does not take TPU_MOE_SEQUENCE_PARALLEL": envs.TPU_MOE_SEQUENCE_PARALLEL,
        "does not take USE_MOE_FUSED_EP_KERNEL (it owns every bucket)": (
            envs.USE_MOE_FUSED_EP_KERNEL
        ),
        "does not take USE_MOE_TP_DECODE_KERNEL": envs.USE_MOE_TP_DECODE_KERNEL,
        "does not take TPU_FUSED_ADD_RMSNORM (its norm -> linear fp8 pairing "
        "would hand the rank's rows to a gathered linear)": envs.TPU_FUSED_ADD_RMSNORM,
        "does not take TPU_DCP_MLA_OPROJ_ALL_REDUCE": envs.TPU_DCP_MLA_OPROJ_ALL_REDUCE,
        "does not take TPU_MOE_SKIP_PADDED_TOKENS (its mask is over all rows)": (
            envs.TPU_MOE_SKIP_PADDED_TOKENS
        ),
    }
    bad = [why for why, hit in refusals.items() if hit]
    if bad:
        raise NotImplementedError(
            "TPU_MOE_PREFILL_SEQUENCE_PARALLEL " + "; ".join(bad) + "."
        )


def _backbone(model: torch.nn.Module) -> Any:
    inner = getattr(model, "model", None)
    if not isinstance(inner, deepseek_v2.DeepseekV2Model):
        raise NotImplementedError(
            "TPU_MOE_PREFILL_SEQUENCE_PARALLEL serves DeepseekV2Model backbones; "
            f"got {type(inner).__name__}"
        )
    if deepseek_v2.DeepseekV2Model.forward is not patched_model_forward:
        raise RuntimeError("install_forwards did not run before the model was built")
    if getattr(inner.config, "llama_4_scaling", None) is not None:
        raise NotImplementedError("llama_4_scaling is not threaded through the split")
    if inner.aux_hidden_state_layers:
        raise NotImplementedError("aux hidden states are not gathered by the split")
    return inner


def _keep_list(mlp: torch.nn.Module) -> list[torch.Tensor]:
    """The layer's tensors plus the module-global operands the other buckets'
    MoE ops read (the decode kernel's rank buffer, when it is armed)."""
    seen: dict[int, torch.Tensor] = {}
    extra = (fused_moe_ep.rank_operand(), fused_moe_decode.rank_operand())
    for t in (*mlp.parameters(), *mlp.buffers(), *extra):
        if t is not None:
            seen.setdefault(id(t), t)
    return list(seen.values())


def arm(model: torch.nn.Module, vllm_config: Any) -> int:
    """Prepare every MoE layer of the backbone; returns how many. Collective
    over the TP group (weight gathers, the EP mesh). Raises for anything the
    split cannot serve."""
    check_config(vllm_config)
    backbone = _backbone(model)
    tp = get_tensor_model_parallel_world_size()
    if torch.distributed.get_world_size() != tp:
        raise NotImplementedError("the TP group must be the whole world")
    layers = list(islice(backbone.layers, backbone.start_layer, backbone.end_layer))
    moe = [isinstance(layer.mlp, deepseek_v2.DeepseekV2MoE) for layer in layers]
    first = moe.index(True)
    if not all(moe[first:]):
        raise NotImplementedError(
            "the split needs every layer after the first MoE layer to be MoE"
        )
    for layer in layers[first:]:
        mla = getattr(layer.self_attn, "mla_attn", None)
        if "prefill_group" not in inspect.signature(type(mla).forward).parameters:
            raise NotImplementedError(
                f"the split needs the TPU MLA op's o_proj hook; got {type(mla)}"
            )
        mlp = layer.mlp
        experts = mlp.experts.routed_experts
        if mlp.shared_experts is None:
            raise NotImplementedError("the split expects a separate shared expert")
        if float(experts.routed_scaling_factor) != 1.0:
            raise NotImplementedError("the split applies the routed scale itself")
        op = fused_moe_ep.build_sequence_parallel_op(
            experts,
            topk=experts.moe_config.experts_per_token,
            renormalize=experts.renormalize,
            activation=experts.quant_method._tpu_activation_str,
        )
        if ep_mesh_index() != get_tp_group().rank_in_group:
            raise NotImplementedError(
                "the EP mesh index must be the TP rank (a rank-ordered EP mesh, "
                "as under DCP): the rank operand also picks the residual rows"
            )
        setattr(mlp, OP_ATTR, op)
        setattr(mlp, SCALE_ATTR, float(mlp.routed_scaling_factor))
        _group_shared_expert(mlp, tp, get_tp_group().rank_in_group)
        setattr(mlp, SHARED_OP, _build_shared_op(SHARED_GROUP))
        setattr(layer, KEEP_ATTR, _keep_list(mlp))
    setattr(backbone, KEEP_ATTR, True)
    n = len(layers) - first
    logger.info(
        "TPU_MOE_PREFILL_SEQUENCE_PARALLEL: %d MoE layers armed (layers %d..%d), "
        "buckets > %d tokens, tp=%d",
        n,
        backbone.start_layer + first,
        backbone.end_layer - 1,
        envs.bucket_gate_edge(MIN_TOKENS),
        tp,
    )
    return n
