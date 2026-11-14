# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""Kimi MoE routed through vLLM's TPU Pallas fused-MoE backend."""

from __future__ import annotations

import re
import zlib
from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING, cast

import torch
from torch import nn
from vllm.config import get_current_vllm_config_or_none
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.intra_chip import get_intra_chip_group
from vllm_torchtpu.layers.adapter import moe_routing
from vllm_torchtpu.layers.adapter.latent_proj_intra_chip import make_latent_projection
from vllm_torchtpu.layers.adapter.moe_hierarchical import (
    hierarchical_moe_parallel_config,
    hierarchical_split_or_none,
)

from .collective_ops import FusedPrefillCollectives
from .layers import KimiMLP

if TYPE_CHECKING:
    from vllm_torchtpu.layers.adapter.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4an_mxfp4 import (  # noqa: E501
        VllmCompressedTensorsW4ANMxfp4MoEMethod,
    )


# Non-uniform expert pruning (e.g. nota-ai/Kimi-K3-Nota-Global-Pruned-50) keeps a
# different number of routed experts in each layer, listed in the config's
# `n_routed_experts_per_layer`. Each MoE layer is built with its own count, rounded
# up to a multiple of the expert-parallel size so its experts split evenly across
# ranks. Padding experts have zero weights and a router correction bias of
# _PAD_BIAS, so routing never picks them.
_PAD_BIAS = -1.0e4
_ROUTER_RE = re.compile(
    r"layers\.(\d+)\.block_sparse_moe\.gate\.(weight|e_score_correction_bias)$"
)


def layer_expert_counts(config: KimiLinearConfig, layer_idx: int) -> tuple[int, int]:
    """Return a layer's routed expert count in the checkpoint and as built."""
    per_layer = getattr(config, "n_routed_experts_per_layer", None)
    if not per_layer:
        n = int(config.num_experts)
        return n, n  # a uniform checkpoint is built exactly as before
    n = int(per_layer[layer_idx])
    vllm_config = get_current_vllm_config_or_none()
    par = vllm_config.parallel_config if vllm_config is not None else None
    multiple = (
        par.tensor_parallel_size * par.data_parallel_size
        if par is not None and par.enable_expert_parallel
        else 1
    )
    return n, -(-n // multiple) * multiple


def per_layer_local_expert_ids(
    compute: Callable[..., set[int] | None],
    counts: Iterable[int],
    ep_size: int,
    ep_rank: int,
    placement: str = "linear",
) -> set[int] | None:
    """Return the experts a rank reads from a checkpoint with per-layer counts.

    `--enable-ep-weight-filter` gives each rank the experts that `compute` assigns
    it out of one count for every layer. Here each layer is split by its own padded
    count, so a rank owns a different range in each layer: it reads the union of
    those ranges (about twice its share, instead of every expert).
    """
    ids: set[int] = set()
    for n in {int(c) for c in counts if c}:
        padded = -(-n // ep_size) * ep_size
        local = compute(padded, ep_size, ep_rank, placement=placement)
        if local is None:
            return None
        ids |= {i for i in local if i < n}
    return ids


def install_per_layer_ep_weight_filter(counts: list[int]) -> None:
    """Make the loader's expert filter aware of per-layer counts (idempotent).

    The counts are captured when the model is built: the loader computes the
    filter later, outside the vLLM config context the layers are built in.
    """
    from vllm.model_executor.model_loader import default_loader

    compute = default_loader.compute_local_expert_ids
    if getattr(compute, "_kimi_per_layer", False):
        return

    def compute_local_expert_ids(
        num_experts: int, ep_size: int, ep_rank: int, placement: str = "linear"
    ) -> set[int] | None:
        if ep_size <= 1:
            return compute(num_experts, ep_size, ep_rank, placement=placement)
        return per_layer_local_expert_ids(compute, counts, ep_size, ep_rank, placement)

    compute_local_expert_ids._kimi_per_layer = True  # type: ignore[attr-defined]
    default_loader.compute_local_expert_ids = compute_local_expert_ids


def pad_router_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    padded: dict[int, tuple[int, int]],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Pad router rows (zeros) and correction biases (_PAD_BIAS) to built counts."""
    for name, tensor in weights:
        m = _ROUTER_RE.search(name)
        if m is not None and int(m.group(1)) in padded:
            n, n_pad = padded[int(m.group(1))]
            if n_pad > n and tensor.shape[0] == n:
                fill = 0.0 if m.group(2) == "weight" else _PAD_BIAS
                pad = torch.full(
                    (n_pad - n, *tensor.shape[1:]), fill, dtype=tensor.dtype
                )
                tensor = torch.cat([tensor, pad], dim=0)
        yield name, tensor


class KimiRoutedOutputTransform(nn.Module):
    def __init__(
        self,
        norm: RMSNorm | None,
        # Replicated or intra-chip sharded; both return ``(output, bias)``.
        up_proj: nn.Module,
    ) -> None:
        super().__init__()
        self.norm = norm
        self.up_proj = up_proj

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.norm is not None:
            hidden_states = self.norm(hidden_states)
        hidden_states, _ = self.up_proj(hidden_states)
        return hidden_states


class KimiMoE(nn.Module):
    """Standard or latent Kimi MoE backed by the fused TPU GMM custom op."""

    def __init__(
        self,
        config: KimiLinearConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        assert config.num_experts is not None
        assert config.num_experts_per_token is not None
        assert config.moe_intermediate_size is not None

        self.prefix = prefix
        layer_match = re.search(r"layers\.(\d+)\.", prefix)
        self.layer_idx = int(layer_match.group(1)) if layer_match else -1
        self.num_experts_ckpt, self.num_experts_padded = layer_expert_counts(
            config, self.layer_idx
        )
        if getattr(config, "n_routed_experts_per_layer", None):
            install_per_layer_ep_weight_filter(list(config.n_routed_experts_per_layer))
        num_experts = self.num_experts_padded
        hidden_size = config.hidden_size
        self.tp_size = get_tensor_model_parallel_world_size()
        vllm_config = get_current_vllm_config_or_none()
        self.use_ep = (
            vllm_config is not None
            and vllm_config.parallel_config.enable_expert_parallel
        )
        routed_expert_hidden_size = config.routed_expert_hidden_size
        expert_hidden_size = (
            routed_expert_hidden_size
            if routed_expert_hidden_size is not None
            else hidden_size
        )

        self.gate = GateLinear(
            input_size=hidden_size,
            output_size=num_experts,
            bias=False,
            out_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )
        self.gate.e_score_correction_bias = nn.Parameter(torch.zeros(num_experts))

        # Replicated by default; sharded across a chip's cores under
        # TPU_LATENT_PROJ_INTRA_CHIP_TP. These two are the model's largest
        # unquantized weights and are read in full by every rank each step.
        self.routed_expert_down_proj = (
            make_latent_projection(
                hidden_size,
                expert_hidden_size,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.routed_expert_down_proj",
            )
            if routed_expert_hidden_size is not None
            else None
        )
        self.routed_expert_norm = (
            RMSNorm(expert_hidden_size, config.rms_norm_eps)
            if self.routed_expert_down_proj is not None and config.latent_moe_use_norm
            else None
        )
        self.routed_expert_up_proj = (
            make_latent_projection(
                expert_hidden_size,
                hidden_size,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.routed_expert_up_proj",
            )
            if self.routed_expert_down_proj is not None
            else None
        )
        routed_output_transform = (
            KimiRoutedOutputTransform(
                self.routed_expert_norm, self.routed_expert_up_proj
            )
            if self.routed_expert_up_proj is not None
            else None
        )

        self.prefill_group = None
        self.shared_group = None
        if envs.TPU_K3_SP_PREFILL:
            self.prefill_group = FusedPrefillCollectives(prefix)
            self.shared_group = get_intra_chip_group()
            if self.shared_group is None or self.shared_group.world_size != 2:
                raise ValueError("K3 SP prefill needs a physical two-core chip group")

        self.shared_experts = (
            KimiMLP(
                hidden_size,
                config.moe_intermediate_size * config.num_shared_experts,
                config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.shared_experts",
                reduce_results=False,
                tp_group=self.shared_group,
                situ_beta=config.activation_situ_beta,
                situ_linear_beta=config.activation_situ_linear_beta,
            )
            if config.num_shared_experts > 0
            else None
        )

        padded_intermediate_size = config.moe_intermediate_size
        min_per_partition = getattr(config, "min_moe_intermediate_per_partition", 128)
        if self.use_ep:
            split = hierarchical_split_or_none()
            moe_tp_size = split[2] if split is not None else 1
        else:
            moe_tp_size = self.tp_size
        if (
            moe_tp_size > 1
            and padded_intermediate_size // moe_tp_size < min_per_partition
        ):
            padded_intermediate_size = min_per_partition * moe_tp_size

        situ_beta = config.activation_situ_beta if config.hidden_act == "situ" else None
        situ_linear_beta = (
            config.activation_situ_linear_beta if config.hidden_act == "situ" else None
        )
        # Under TPU_MOE_HIERARCHICAL_EP this builds the routed experts
        # with expert parallelism between chips and tensor parallelism
        # within one chip, instead of vLLM's flat all-experts-per-device EP.
        with hierarchical_moe_parallel_config():
            self.experts = FusedMoEFactory(
                shared_experts=self.shared_experts,
                num_experts=num_experts,
                top_k=config.num_experts_per_token,
                hidden_size=expert_hidden_size,
                intermediate_size=padded_intermediate_size,
                activation=config.hidden_act,
                activation_situ_beta=situ_beta,
                activation_situ_linear_beta=situ_linear_beta,
                renormalize=config.moe_renormalize,
                quant_config=quant_config,
                use_grouped_topk=config.use_grouped_topk,
                num_expert_group=config.num_expert_group,
                topk_group=config.topk_group,
                prefix=f"{prefix}.experts",
                scoring_func=config.moe_router_activation_func,
                e_score_correction_bias=cast(
                    torch.Tensor, self.gate.e_score_correction_bias
                ),
                routed_scaling_factor=config.routed_scaling_factor,
                routed_input_transform=self.routed_expert_down_proj,
                routed_output_transform=routed_output_transform,
            )
        if self.shared_group is not None and not hasattr(
            self.experts.routed_experts.quant_method, "apply_with_routing"
        ):
            raise ValueError("K3 SP prefill requires the native MXFP4 MoE backend")
        if self.num_experts_padded > self.num_experts_ckpt:
            # Padding experts are never routed to; zero them so their memory
            # holds no stray values.
            for param in self.experts.routed_experts.parameters():
                param.data.zero_()
        if padded_intermediate_size != config.moe_intermediate_size:
            routed_experts = self.experts.routed_experts
            w13_weight = getattr(routed_experts, "w13_weight", None)
            if w13_weight is None:
                w13_weight = cast(torch.Tensor, routed_experts.w13_weight_packed)
            w2_weight = getattr(routed_experts, "w2_weight", None)
            if w2_weight is None:
                w2_weight = cast(torch.Tensor, routed_experts.w2_weight_packed)
            w13_weight.data.zero_()
            w2_weight.data.zero_()
            self.experts.moe_config.intermediate_size_per_partition_unpadded = (
                config.moe_intermediate_size // moe_tp_size
            )

    def initialize_dummy_router(self) -> None:
        """Use fan-in random weights and fixed, distinct random layer biases.

        The generic dummy loader's +/-0.001 weights make the correction bias
        dominate sigmoid routing scores. Match the prototype's unit logit
        variance for normalized inputs without imposing any route counts.
        """
        config = get_current_vllm_config_or_none()
        if config is None or str(config.load_config.load_format).lower() != "dummy":
            return
        generator = torch.Generator(device="cpu").manual_seed(
            1234 + zlib.crc32(self.prefix.encode())
        )
        bias_param = cast(torch.Tensor, self.gate.e_score_correction_bias)
        bias = (
            torch.randn(
                bias_param.shape,
                generator=generator,
                dtype=torch.float32,
            )
            * 0.02
        )
        weight_param = cast(torch.Tensor, self.gate.weight)
        weight = torch.randn(
            weight_param.shape, generator=generator, dtype=torch.float32
        )
        # The TPU dense loader transposes weights to [input, output]. Use
        # the logical fan-in rather than depending on the stored layout.
        weight *= self.gate.input_size**-0.5
        with torch.no_grad():
            bias_param.copy_(bias)
            weight_param.copy_(weight)

    def forward(
        self, hidden_states: torch.Tensor, sequence_parallel: bool = False
    ) -> torch.Tensor:
        if sequence_parallel or self.shared_group is not None:
            return self._forward_expert_parallel(hidden_states, sequence_parallel)
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)
        router_logits, _ = self.gate(hidden_states)
        output = self.experts(hidden_states=hidden_states, router_logits=router_logits)
        return output.view(num_tokens, hidden_size)

    def _forward_expert_parallel(
        self, hidden_states: torch.Tensor, sequence_parallel: bool
    ) -> torch.Tensor:
        """Run EP experts with sharded or replicated tokens, then add shared experts."""
        group = self.prefill_group
        assert group is not None
        router_logits, _ = self.gate(hidden_states)
        latent = hidden_states
        if self.routed_expert_down_proj is not None:
            latent, _ = self.routed_expert_down_proj(latent)
        routed = self.experts.routed_experts
        if sequence_parallel:
            weights, ids = moe_routing.route(routed, latent, router_logits)
            latent, weights, ids = group.gather_moe_inputs(
                latent, weights, ids, num_experts=self.gate.output_size
            )
            output = cast(
                "VllmCompressedTensorsW4ANMxfp4MoEMethod", routed.quant_method
            ).apply_with_routing(routed, latent, weights, ids)
        else:
            output = routed.quant_method.apply_monolithic(routed, latent, router_logits)
        output = (
            group.reduce_scatter(output, dim=0)
            if sequence_parallel
            else group.all_reduce(output)
        )
        if self.routed_expert_norm is not None:
            output = self.routed_expert_norm(output)
        if self.routed_expert_up_proj is not None:
            output, _ = self.routed_expert_up_proj(output)
        if self.shared_experts is not None:
            output = output + self.shared_experts(
                hidden_states, sequence_parallel=sequence_parallel
            )
        return output
