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
"""DeepSeek-V4 mixture-of-experts layer."""

from typing import cast

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm_torchtpu.layers.adapter.custom_ops.moe_runner import VllmMoERunner
from vllm.model_executor.models.utils import extract_layer_index

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.tp_collectives import (
    tp_all_gather,
    tp_reduce_scatter,
)
from vllm_torchtpu.layers.adapter import moe_routing
from vllm_torchtpu.layers.adapter.fused_moe import (
    moe_fp8_dispatch_enabled,
    quantize_moe_dispatch,
)
from vllm_torchtpu.layers.adapter.fused_moe_ep import fused_moe_ep_supported
from vllm_torchtpu.layers.adapter.router_topk import (
    fused_gate_grouped_rowmax_select,
)
from vllm_torchtpu.models.vllm.deepseek_v4.layers import DeepseekV4MLP

_TOKEN_SHARD_MIN_TOKENS = envs.VLLM_DSV41_TOKEN_SHARD_MIN_TOKENS


class _DeepseekV4MoERunner(VllmMoERunner):
    """MoERunner that can skip shared_experts and/or gate when computed on the local TP shard.

    Subclasses the plugin's ``VllmMoERunner`` (not vLLM's ``MoERunner``): vLLM only swaps
    in ``VllmMoERunner`` for layers that pass no ``runner_cls``, and without it the layer
    takes the ``torch.ops.vllm.moe_forward*`` custom-op path, whose hoisted ``LayerName``
    opaque object torch_tpu's MLIR export cannot resolve.
    """

    def __init__(self, *args, **kwargs) -> None:
        self._skip_shared: list[bool] = [False]
        self._skip_gate: list[bool] = [False]
        super().__init__(*args, **kwargs)

    @property
    def gate(self):
        if self._skip_gate[0]:
            return None
        return self._modules.get("gate") or self._modules.get("_gate_mod")

    @gate.setter
    def gate(self, value) -> None:
        self._modules["gate"] = value

    @property
    def _shared_experts(self):
        if self._skip_shared[0]:
            return None
        return self._modules.get("_shared_experts") or self._modules.get(
            "_shared_experts_mod"
        )

    @_shared_experts.setter
    def _shared_experts(self, value) -> None:
        self._modules["_shared_experts"] = value


class DeepseekV4MoE(nn.Module):
    """DeepSeek-V4 MoE layer with dynamic hash routing and fused routed experts."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        reduce_results: bool = True,
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.prefix = prefix

        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.hidden_size = config.hidden_size
        self.n_routed_experts = config.n_routed_experts
        self.swiglu_limit = config.swiglu_limit
        self.renormalize = config.norm_topk_prob
        self.scoring_func = getattr(config, "scoring_func", "sqrtsoftplus")

        self.gate = GateLinear(
            input_size=config.hidden_size,
            output_size=config.n_routed_experts,
            bias=False,
            out_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )

        self.gate.e_score_correction_bias = None
        self.gate.tid2eid = None
        # DSpark draft layers sit past the target's mlp_layer_types.
        layer_index = extract_layer_index(prefix)
        is_hash_moe = (
            layer_index < len(config.mlp_layer_types)
            and config.mlp_layer_types[layer_index] == "hash_moe"
        )
        self.hash_indices_dtype = torch.int32
        if is_hash_moe:
            self.gate.tid2eid = nn.Parameter(
                torch.randint(
                    0,
                    config.n_routed_experts,
                    (config.vocab_size, config.num_experts_per_tok),
                    dtype=self.hash_indices_dtype,
                ),
                requires_grad=False,
            )
        elif getattr(config, "topk_method", None) == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float32),
                requires_grad=False,
            )

        if config.n_shared_experts is None:
            self.shared_experts = None
        else:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekV4MLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                swiglu_limit=self.swiglu_limit,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )

        self.tp_rank = get_tensor_model_parallel_rank()
        assert config.n_routed_experts % self.tp_size == 0

        self.experts = FusedMoEFactory(
            shared_experts=self.shared_experts,
            gate=self.gate,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            scoring_func=self.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            e_score_correction_bias=cast(
                torch.Tensor | None, self.gate.e_score_correction_bias
            ),
            hash_indices_table=cast(torch.Tensor | None, self.gate.tid2eid),
            swiglu_limit=self.swiglu_limit,
            router_logits_dtype=torch.float32,
            apply_routed_scale_to_output=True,
            reduce_results=reduce_results,
            runner_cls=_DeepseekV4MoERunner,
        )

        # The monolithic TPU MoE path never consults the router: MoERunner
        # calls `routed_experts.forward_monolithic`, which hands the
        # `RoutedExperts` layer straight to `quant_method.apply_monolithic`,
        # and `moe_routing.route` looks the table up as
        # `layer.hash_indices_table`. Passing `hash_indices_table` to
        # `FusedMoEFactory` above only reaches the router object, so mirror it
        # onto the experts layer.
        if self.gate.tid2eid is not None:
            object.__setattr__(
                self.experts.routed_experts, "hash_indices_table", self.gate.tid2eid
            )

    @property
    def routes_dp_gathered_tokens(self) -> bool:
        """Whether the experts see the DP-gathered batch or this rank's rows.

        Hash routing indexes its table by token id, so the ids handed down
        have to line up row-for-row with the hidden states the experts get.
        The runner gathers them only while the quant method does not own its
        own dispatch; an armed fused-EP layer does, and then the experts --
        and the ids -- stay local. Asked of the layer rather than assumed,
        because arming is decided per layer at weight load.
        """
        return self.experts.do_naive_dispatch_combine

    def process_weights_after_loading(self) -> None:
        if self.shared_experts is not None:
            self.shared_experts.process_weights_after_loading()

    def can_accept_prequant_hidden(self, *, token_sharded: bool = False) -> bool:
        if self.shared_experts is None:
            return False
        if token_sharded and not (
            self.tp_size > 1
            and getattr(
                getattr(self.experts, "moe_config", None),
                "skip_final_all_reduce",
                False,
            )
            and getattr(self.experts, "_skip_shared", None) is not None
        ):
            return False
        return self.shared_experts.can_accept_prequant_hidden(
            token_sharded=token_sharded
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        return_token_sharded: bool = False,
        hidden_states_fp8: torch.Tensor | None = None,
        hidden_states_scale: torch.Tensor | None = None,
        *,
        input_is_token_sharded: bool = False,
    ) -> torch.Tensor:
        """Forward pass executing router gate and fused expert computation."""
        if self.gate.tid2eid is not None and input_ids is None:
            raise ValueError("DeepSeek V4 hash MoE routing requires input_ids.")

        org_shape = hidden_states.shape
        skip_shared = getattr(self.experts, "_skip_shared", None)
        skip_gate = getattr(self.experts, "_skip_gate", None)
        if (
            _TOKEN_SHARD_MIN_TOKENS > 0
            and self.tp_size > 1
            and self.shared_experts is not None
            and "_tp_gate_up_w" in self.shared_experts._buffers
        ):
            self.shared_experts._lift_tp_shape_variant_tensors()

        run_local_shared = (
            input_is_token_sharded
            and return_token_sharded
            and self.tp_size > 1
            and getattr(
                getattr(self.experts, "moe_config", None),
                "skip_final_all_reduce",
                False,
            )
            and skip_shared is not None
            and self.shared_experts is not None
            and self.shared_experts.has_tp_local_weights()
        )
        x_local = hidden_states
        num_tokens_global = (
            x_local.shape[0] * self.tp_size
            if input_is_token_sharded
            else x_local.shape[0]
        )
        router_logits = hidden_states
        defer_shared_to_rs = run_local_shared
        routed = getattr(self.experts, "routed_experts", None)
        qm = getattr(routed, "quant_method", None)
        if (
            fused_moe_ep_supported(qm)
            and routed is not None
            and qm is not None
        ):
            if input_is_token_sharded:
                x_shard = x_local
                ids_shard = (
                    input_ids[
                        self.tp_rank * x_shard.shape[0] : (self.tp_rank + 1)
                        * x_shard.shape[0]
                    ]
                    if (
                        input_ids is not None
                        and input_ids.shape[0] != x_shard.shape[0]
                    )
                    else input_ids
                )
                gate_out = self.gate(x_shard)
                local_logits = (
                    gate_out[0] if isinstance(gate_out, tuple) else gate_out
                )
                routed_out = qm.apply_monolithic(
                    routed, x_shard, local_logits, input_ids=ids_shard
                )
                if self.shared_experts is not None:
                    if self.shared_experts.has_tp_local_weights():
                        shared_out = self.shared_experts.forward_tp_local(
                            x_shard,
                            hidden_states_fp8=hidden_states_fp8,
                            hidden_states_scale=hidden_states_scale,
                        )
                    else:
                        shared_out = self.shared_experts(
                            x_shard,
                            input_is_token_sharded=True,
                            return_token_sharded=True,
                        )
                    routed_out = routed_out + shared_out.view_as(routed_out)
                if return_token_sharded or self.tp_size <= 1:
                    return routed_out.view(-1, *org_shape[1:])
                return tp_all_gather(routed_out, dim=0).view(-1, *org_shape[1:])
            if self.tp_size > 1:
                t_orig = x_local.shape[0]
                t_pad = (t_orig + self.tp_size - 1) // self.tp_size * self.tp_size
                if t_pad > t_orig:
                    x_pad = torch.cat(
                        [
                            x_local,
                            x_local.new_zeros((t_pad - t_orig, x_local.shape[1])),
                        ],
                        dim=0,
                    )
                    ids_pad = (
                        torch.cat(
                            [
                                input_ids,
                                input_ids.new_full((t_pad - t_orig,), -1),
                            ],
                            dim=0,
                        )
                        if input_ids is not None
                        else None
                    )
                else:
                    x_pad = x_local
                    ids_pad = input_ids
                t_per_rank = t_pad // self.tp_size
                s_idx = self.tp_rank * t_per_rank
                e_idx = s_idx + t_per_rank
                x_shard = x_pad[s_idx:e_idx]
                ids_shard = ids_pad[s_idx:e_idx] if ids_pad is not None else None
                gate_out = self.gate(x_shard)
                local_logits = (
                    gate_out[0] if isinstance(gate_out, tuple) else gate_out
                )
                if t_pad > t_orig:
                    pad_rows = (
                        torch.arange(s_idx, e_idx, device=x_shard.device) >= t_orig
                    )
                    local_logits = torch.where(
                        pad_rows[:, None],
                        local_logits.new_full((), -float("inf")),
                        local_logits,
                    )
                routed_out = qm.apply_monolithic(
                    routed, x_shard, local_logits, input_ids=ids_shard
                )
                if self.shared_experts is not None:
                    if self.shared_experts.has_tp_local_weights():
                        shared_out = self.shared_experts.forward_tp_local(x_shard)
                    else:
                        shared_out = self.shared_experts(
                            x_shard,
                            input_is_token_sharded=True,
                            return_token_sharded=True,
                        )
                    routed_out = routed_out + shared_out.view_as(routed_out)
                if return_token_sharded:
                    return routed_out.view(-1, *org_shape[1:])
                return (
                    tp_all_gather(routed_out, dim=0)[:t_orig].view(-1, *org_shape[1:])
                )
        use_fp8_tp_dispatch = (
            input_is_token_sharded
            and self.tp_size > 1
            and run_local_shared
            and routed is not None
            and qm is not None
            and getattr(routed, "w13_weight", None) is not None
            and moe_fp8_dispatch_enabled(
                routed.w13_weight,
                getattr(routed, "w13_weight_scale", None),
                getattr(qm, "rhs_quant_dtype", None),
            )
        )
        if use_fp8_tp_dispatch:
            assert routed is not None
            if (
                self.gate.tid2eid is None
                and routed.use_grouped_topk
                and routed.num_expert_group > 1
                and routed.topk_group < routed.num_expert_group
                and routed.e_score_correction_bias is not None
                and getattr(self.gate, "is_unquantized", False)
            ):
                w_f32, local_topk_ids = fused_gate_grouped_rowmax_select(
                    x_local,
                    self.gate.weight,
                    routed.e_score_correction_bias,
                    topk=routed.moe_config.experts_per_token,
                    num_expert_group=routed.num_expert_group,
                    topk_group=routed.topk_group,
                    scoring_fn=routed.scoring_func,
                    renormalize=routed.renormalize,
                    routed_scaling_factor=float(routed.routed_scaling_factor),
                )
                local_topk_w = w_f32.to(x_local.dtype)
                local_topk_ids = moe_routing.maybe_modulo_remap_topk_ids(
                    local_topk_ids, routed
                )
            else:
                if self.gate.tid2eid is None:
                    local_gate_out = self.gate(x_local)
                    local_logits = (
                        local_gate_out[0]
                        if isinstance(local_gate_out, tuple)
                        else local_gate_out
                    )
                else:
                    local_logits = x_local[:0]
                local_input_ids = (
                    input_ids[
                        self.tp_rank * x_local.shape[0] : (self.tp_rank + 1)
                        * x_local.shape[0]
                    ]
                    if input_ids is not None
                    else None
                )
                local_topk_w, local_topk_ids = moe_routing.route(
                    routed, x_local, local_logits, input_ids=local_input_ids
                )
            packed_local = quantize_moe_dispatch(
                x_local, local_topk_w, local_topk_ids
            )
            hidden_states = tp_all_gather(packed_local, dim=0)
            router_logits = x_local[:0]
        elif input_is_token_sharded:
            hidden_states = tp_all_gather(x_local, dim=0)
            if (
                self.tp_size > 1
                and self.gate.tid2eid is None
                and routed is not None
                and skip_gate is not None
            ):
                if (
                    routed.use_grouped_topk
                    and routed.num_expert_group > 1
                    and routed.topk_group < routed.num_expert_group
                    and routed.e_score_correction_bias is not None
                    and getattr(self.gate, "is_unquantized", False)
                ):
                    w_f32, local_topk_ids = fused_gate_grouped_rowmax_select(
                        x_local,
                        self.gate.weight,
                        routed.e_score_correction_bias,
                        topk=routed.moe_config.experts_per_token,
                        num_expert_group=routed.num_expert_group,
                        topk_group=routed.topk_group,
                        scoring_fn=routed.scoring_func,
                        renormalize=routed.renormalize,
                        routed_scaling_factor=float(routed.routed_scaling_factor),
                    )
                    local_topk_w = w_f32.to(x_local.dtype)
                    local_topk_ids = moe_routing.maybe_modulo_remap_topk_ids(
                        local_topk_ids, routed
                    )
                else:
                    local_gate_out = self.gate(x_local)
                    local_logits = (
                        local_gate_out[0]
                        if isinstance(local_gate_out, tuple)
                        else local_gate_out
                    )
                    local_topk_w, local_topk_ids = moe_routing.route(
                        routed, x_local, local_logits, input_ids=None
                    )
                router_logits = moe_routing.all_gather_packed_topk_s32(
                    local_topk_w,
                    local_topk_ids,
                    all_gather_fn=tp_all_gather,
                    num_tokens_global=num_tokens_global,
                )
                skip_gate[0] = True
        shared_local_out: torch.Tensor | None = None
        if run_local_shared:
            assert self.shared_experts is not None and skip_shared is not None
            skip_shared[0] = True
            if not defer_shared_to_rs:
                shared_local_out = self.shared_experts.forward_tp_local(
                    x_local,
                    hidden_states_fp8=hidden_states_fp8,
                    hidden_states_scale=hidden_states_scale,
                )
        elif (
            self.shared_experts is not None
            and hidden_states_fp8 is not None
            and not input_is_token_sharded
        ):
            self.shared_experts.set_prequant_hidden(
                hidden_states_fp8, hidden_states_scale
            )
        try:
            if use_fp8_tp_dispatch:
                assert routed is not None and qm is not None
                final_hidden_states = qm.apply_monolithic(
                    routed, hidden_states, router_logits
                )
            else:
                # FusedMoEFactory gives the runner our gate, so it computes the logits.
                final_hidden_states = self.experts(
                    hidden_states=hidden_states,
                    router_logits=router_logits,
                    input_ids=input_ids,
                )
        finally:
            if skip_shared is not None:
                skip_shared[0] = False
            if skip_gate is not None:
                skip_gate[0] = False
            if self.shared_experts is not None and hidden_states_fp8 is not None:
                self.shared_experts.set_prequant_hidden(None, None)
        if self.experts.moe_config.skip_final_all_reduce and self.tp_size > 1:
            if return_token_sharded:
                if defer_shared_to_rs and self.shared_experts is not None:
                    xla_ops = getattr(torch.ops, "xla", None)
                    opt_barrier = getattr(xla_ops, "optimization_barrier_", None)
                    if opt_barrier is not None:
                        opt_barrier([final_hidden_states, x_local])
                    final_hidden_states = tp_reduce_scatter(
                        final_hidden_states, dim=0
                    )
                    shared_local_out = self.shared_experts.forward_tp_local(
                        x_local,
                        hidden_states_fp8=hidden_states_fp8,
                        hidden_states_scale=hidden_states_scale,
                    )
                else:
                    final_hidden_states = tp_reduce_scatter(
                        final_hidden_states, dim=0
                    )
                if shared_local_out is not None:
                    final_hidden_states = (
                        final_hidden_states
                        + shared_local_out.view_as(final_hidden_states)
                    )
                return final_hidden_states.view(-1, *org_shape[1:])
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states.view(-1, *org_shape[1:])
