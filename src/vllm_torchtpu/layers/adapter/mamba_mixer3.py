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
"""vLLM PluggableLayer parameter/weight-loader shell for Mamba-3.

``MambaMixer3`` defines the parameter shapes, TP weight loaders, state-cache
layout, and ``static_forward_context`` registration for Mamba-3, and requires
an out-of-tree backend subclass (``VllmMambaMixer3`` in
``vllm_torchtpu.layers.adapter.custom_ops.mamba_mixer3_op``) for execution.
"""

from __future__ import annotations

from typing import Any, NamedTuple, cast

import torch
from torch import nn
from vllm.config import (
    CacheConfig,
    ModelConfig,
    get_current_vllm_config,
)
from vllm.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.custom_op import CustomOp, PluggableLayer
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_mixer2 import Mixer2RMSNormGated
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateDtypeCalculator
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.parameter import BasevLLMParameter
from vllm.model_executor.utils import set_weight_attrs
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum


class Mamba3StateLayout(NamedTuple):
    """Per-TP-rank Mamba-3 boundary and recurrent state dimensions."""

    tped_dt_size: int
    mimo_rank: int
    num_rope_angles: int
    aux_dim: int
    aux_shape: tuple[int, int]
    ssm_shape: tuple[int, int, int]


def mamba3_state_layout(
    *,
    num_heads: int,
    head_dim: int,
    ssm_state_size: int,
    rope_fraction: float,
    is_mimo: bool,
    mimo_rank: int,
    tp_size: int,
) -> Mamba3StateLayout:
    """Compute per-rank Mamba-3 auxiliary and recurrent state shapes."""
    effective_rank = mimo_rank if is_mimo else 1
    tped_dt_size = divide(num_heads, tp_size)
    split_tensor_size = int(ssm_state_size * rope_fraction)
    if split_tensor_size % 2 != 0:
        split_tensor_size -= 1
    num_rope_angles = split_tensor_size // 2
    aux_dim = effective_rank * (head_dim + ssm_state_size) + num_rope_angles
    aux_shape = (tped_dt_size, aux_dim)
    ssm_shape = (tped_dt_size, head_dim, ssm_state_size)
    return Mamba3StateLayout(
        tped_dt_size=tped_dt_size,
        mimo_rank=effective_rank,
        num_rope_angles=num_rope_angles,
        aux_dim=aux_dim,
        aux_shape=aux_shape,
        ssm_shape=ssm_shape,
    )


def mamba3_state_dtype(
    model_config: ModelConfig | None,
    cache_config: CacheConfig | None,
) -> tuple[torch.dtype, torch.dtype]:
    """Resolve ``(aux_state_dtype, ssm_state_dtype)`` for Mamba-3."""
    if model_config is not None and cache_config is not None:
        _, ssm_state_dtype = MambaStateDtypeCalculator.mamba2_state_dtype(
            model_config.dtype,
            cache_config.mamba_cache_dtype,
            cache_config.mamba_ssm_cache_dtype,
        )
        return torch.float32, ssm_state_dtype
    return torch.float32, torch.float32


def set_param_weight_loader(param: nn.Parameter, loader: Any) -> None:
    """Bind a custom ``weight_loader`` onto a vLLM or plain PyTorch parameter."""
    if isinstance(param, BasevLLMParameter):
        param.weight_loader = loader
    else:
        if hasattr(param, "weight_loader"):
            delattr(param, "weight_loader")
        set_weight_attrs(param, {"weight_loader": loader})


@CustomOp.register("mamba3_gated_rms_norm")
class Mamba3RMSNormGated(Mixer2RMSNormGated):
    """Gate-optional RMSNorm for Mamba-3 (SISO passes gate, MIMO passes None)."""

    def forward_native(
        self,
        x: torch.Tensor,
        gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        if gate is not None:
            x = x * nn.functional.silu(gate.to(torch.float32))
        if not self.use_rms_norm or self.weight is None:
            return x.to(input_dtype)

        if self.n_groups == 1:
            if self.tp_size > 1:
                local_sums = x.pow(2).sum(dim=-1, keepdim=True)
                global_sums = tensor_model_parallel_all_reduce(local_sums)
                count = self.tp_size * x.shape[-1]
                variance = global_sums / count
            else:
                variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.variance_epsilon)
        else:
            redundant_tp = self.n_groups % self.tp_size != 0
            if redundant_tp:
                x = tensor_model_parallel_all_gather(x, -1)

            *prefix_dims, hidden_dim = x.shape
            group_count = hidden_dim // self.group_size
            x_grouped = x.reshape(*prefix_dims, group_count, self.group_size)
            variance = x_grouped.pow(2).mean(-1, keepdim=True)
            x_grouped = x_grouped * torch.rsqrt(variance + self.variance_epsilon)
            x = x_grouped.reshape(*prefix_dims, hidden_dim)

            if redundant_tp:
                start = self.per_rank_hidden_size * self.tp_rank
                end = start + self.per_rank_hidden_size
                x = x[..., start:end]

        weight = cast(torch.Tensor, self.weight)
        return (weight.to(torch.float32) * x).to(input_dtype)

    def forward_cuda(
        self,
        x: torch.Tensor,
        gate: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if gate is None or not self.use_rms_norm or self.weight is None:
            return self.forward_native(x, gate)
        return super().forward_cuda(x, gate)


@PluggableLayer.register("mamba_mixer3")
class MambaMixer3(MambaBase, PluggableLayer):
    """Mamba-3 Selective State Space Mixer shell requiring an OOT backend."""

    def __init__(
        self,
        hidden_size: int,
        ssm_state_size: int,
        intermediate_size: int,
        *,
        num_heads: int,
        head_dim: int,
        mimo_rank: int,
        use_bias: bool = False,
        n_groups: int = 1,
        rms_norm_eps: float = 1e-5,
        use_rms_norm: bool = False,
        is_mimo: bool = False,
        rope_fraction: float = 1.0,
        A_floor: float = 1e-4,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if not prefix:
            raise ValueError("MambaMixer3 requires a non-empty layer prefix.")

        self.tp_size = get_tensor_model_parallel_world_size()
        if num_heads % self.tp_size != 0:
            raise ValueError(
                f"MambaMixer3 requires num_heads ({num_heads}) to be divisible "
                f"by tp_size ({self.tp_size})."
            )
        if (n_groups % self.tp_size) != 0 and n_groups != 1:
            raise ValueError(
                f"MambaMixer3 requires n_groups ({n_groups}) to be divisible by "
                f"tp_size ({self.tp_size}) or equal to 1."
            )

        layout = mamba3_state_layout(
            num_heads=num_heads,
            head_dim=head_dim,
            ssm_state_size=ssm_state_size,
            rope_fraction=rope_fraction,
            is_mimo=is_mimo,
            mimo_rank=mimo_rank,
            tp_size=self.tp_size,
        )

        self.hidden_size = hidden_size
        self.ssm_state_size = ssm_state_size
        self.intermediate_size = intermediate_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.n_groups = n_groups
        self.is_mimo = is_mimo
        self.mimo_rank = layout.mimo_rank
        self.rope_fraction = rope_fraction
        self.num_rope_angles = layout.num_rope_angles
        self.A_floor = A_floor
        self.rms_norm_eps = rms_norm_eps
        self.use_rms_norm = use_rms_norm
        self.model_config = model_config
        self.cache_config = cache_config
        self.prefix = prefix

        self.tp_rank = get_tensor_model_parallel_rank()
        self.tped_intermediate_size = divide(intermediate_size, self.tp_size)
        self.tped_dt_size = layout.tped_dt_size
        self.tp_n_groups = (
            divide(n_groups, self.tp_size) if n_groups % self.tp_size == 0 else n_groups
        )
        self.groups_ssm_state_size = (
            self.mimo_rank * self.n_groups * self.ssm_state_size
        )
        self.tped_bc_size = self.mimo_rank * self.tp_n_groups * self.ssm_state_size
        self._state_shapes = (layout.aux_shape, layout.ssm_shape)

        # Global unsharded in_proj width.
        self.total_proj_size = (
            2 * intermediate_size
            + 2 * self.groups_ssm_state_size
            + 3 * num_heads
            + self.num_rope_angles
        )
        self.local_proj_size = (
            2 * self.tped_intermediate_size
            + 2 * self.tped_bc_size
            + 3 * self.tped_dt_size
            + self.num_rope_angles
        )

        self.in_proj = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=self.local_proj_size * self.tp_size,
            bias=use_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj",
        )
        if not isinstance(self.in_proj.quant_method, UnquantizedLinearMethod):
            raise NotImplementedError(
                "Quantized weights are not yet supported for MambaMixer3."
            )
        in_proj_loader = self._make_in_proj_weight_loader()
        set_param_weight_loader(cast(nn.Parameter, self.in_proj.weight), in_proj_loader)
        if self.in_proj.bias is not None:
            set_param_weight_loader(
                cast(nn.Parameter, self.in_proj.bias), in_proj_loader
            )

        self.B_norm_weight = nn.Parameter(torch.ones(ssm_state_size))
        self.C_norm_weight = nn.Parameter(torch.ones(ssm_state_size))

        head_loader = self._make_head_weight_loader(squeeze_siso_rank_dim=False)
        bc_bias_loader = self._make_head_weight_loader(squeeze_siso_rank_dim=True)

        if self.is_mimo:
            self.B_bias = nn.Parameter(
                torch.ones(self.tped_dt_size, self.mimo_rank, ssm_state_size)
            )
            self.C_bias = nn.Parameter(
                torch.ones(self.tped_dt_size, self.mimo_rank, ssm_state_size)
            )
            self.mimo_x = nn.Parameter(
                torch.ones(self.tped_dt_size, self.mimo_rank, head_dim) / self.mimo_rank
            )
            self.mimo_o = nn.Parameter(
                torch.ones(self.tped_dt_size, self.mimo_rank, head_dim) / self.mimo_rank
            )
            self.mimo_z = nn.Parameter(
                torch.ones(self.tped_dt_size, self.mimo_rank, head_dim)
            )
            set_weight_attrs(self.mimo_x, {"weight_loader": head_loader})
            set_weight_attrs(self.mimo_o, {"weight_loader": head_loader})
            set_weight_attrs(self.mimo_z, {"weight_loader": head_loader})
        else:
            self.B_bias = nn.Parameter(torch.ones(self.tped_dt_size, ssm_state_size))
            self.C_bias = nn.Parameter(torch.ones(self.tped_dt_size, ssm_state_size))
            self.register_parameter("mimo_x", None)
            self.register_parameter("mimo_o", None)
            self.register_parameter("mimo_z", None)

        self.D = nn.Parameter(torch.ones(self.tped_dt_size))
        self.dt_bias = nn.Parameter(torch.ones(self.tped_dt_size))

        set_weight_attrs(self.B_bias, {"weight_loader": bc_bias_loader})
        set_weight_attrs(self.C_bias, {"weight_loader": bc_bias_loader})
        set_weight_attrs(self.D, {"weight_loader": head_loader})
        set_weight_attrs(self.dt_bias, {"weight_loader": head_loader})

        self.norm = Mamba3RMSNormGated(
            intermediate_size, n_groups, self.use_rms_norm, eps=rms_norm_eps
        )

        self.out_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=use_bias,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )
        if not isinstance(self.out_proj.quant_method, UnquantizedLinearMethod):
            raise NotImplementedError(
                "Quantized weights are not yet supported for MambaMixer3."
            )

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        self.kv_cache = (torch.tensor([]), torch.tensor([]))

    def _make_in_proj_weight_loader(self):
        section_sizes = [
            self.intermediate_size,
            self.intermediate_size,
            self.groups_ssm_state_size,
            self.groups_ssm_state_size,
            self.num_heads,
            self.num_heads,
            self.num_heads,
            self.num_rope_angles,
        ]
        shard_modes = [
            "flat",
            "flat",
            "bc_group" if (self.n_groups % self.tp_size == 0) else "rep",
            "bc_group" if (self.n_groups % self.tp_size == 0) else "rep",
            "flat",
            "flat",
            "flat",
            "rep",
        ]
        tp_size = self.tp_size
        tp_rank = self.tp_rank
        mimo_rank = self.mimo_rank
        n_groups = self.n_groups
        tp_n_groups = self.tp_n_groups
        d_state = self.ssm_state_size

        def _loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
            if loaded_weight.shape[0] == param.shape[0]:
                param.data.copy_(loaded_weight)
                return
            expected_global = sum(section_sizes)
            if loaded_weight.shape[0] != expected_global:
                raise ValueError(
                    f"Expected in_proj loaded_weight dim 0 == {expected_global} "
                    f"or local {param.shape[0]}, got {loaded_weight.shape[0]}."
                )
            sections = torch.split(loaded_weight, section_sizes, dim=0)
            local_sections = []
            for sec, mode in zip(sections, shard_modes):
                if mode == "flat" and tp_size > 1:
                    shard_len = sec.shape[0] // tp_size
                    local_sections.append(sec.narrow(0, tp_rank * shard_len, shard_len))
                elif mode == "bc_group" and tp_size > 1:
                    # Upstream packs B/C as `(r, g, n)` -> shard over axis 1 (`g`).
                    tail = sec.shape[1:]
                    sec_rgn = sec.reshape(mimo_rank, n_groups, d_state, *tail)
                    sec_local = sec_rgn.narrow(1, tp_rank * tp_n_groups, tp_n_groups)
                    local_sections.append(
                        sec_local.reshape(mimo_rank * tp_n_groups * d_state, *tail)
                    )
                else:
                    local_sections.append(sec)
            param.data.copy_(torch.cat(local_sections, dim=0))

        return _loader

    def _make_head_weight_loader(self, *, squeeze_siso_rank_dim: bool = False):
        tp_size = self.tp_size
        tp_rank = self.tp_rank

        def _loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
            if squeeze_siso_rank_dim:
                if (
                    loaded_weight.ndim == 3
                    and param.ndim == 2
                    and loaded_weight.shape[1] == 1
                ):
                    loaded_weight = loaded_weight.squeeze(1)
                elif (
                    loaded_weight.ndim == 2 and param.ndim == 3 and param.shape[1] == 1
                ):
                    loaded_weight = loaded_weight.unsqueeze(1)
            if loaded_weight.shape == param.shape:
                param.data.copy_(loaded_weight)
                return
            shard_size = param.shape[0]
            if (
                loaded_weight.shape[0] != shard_size * tp_size
                or loaded_weight.shape[1:] != param.shape[1:]
            ):
                raise ValueError(
                    "Expected head parameter loaded_weight shape "
                    f"{(shard_size * tp_size, *param.shape[1:])}, "
                    f"got {tuple(loaded_weight.shape)}."
                )
            param.data.copy_(loaded_weight.narrow(0, tp_rank * shard_size, shard_size))

        return _loader

    def forward(
        self,
        hidden_states: torch.Tensor,
        mup_vector: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del hidden_states, mup_vector
        raise NotImplementedError(
            "MambaMixer3 requires an out-of-tree backend implementation "
            "(import vllm_torchtpu.layers.adapter.custom_ops.mamba_mixer3_op "
            "to register VllmMambaMixer3)."
        )

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        return mamba3_state_dtype(self.model_config, self.cache_config)

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return self._state_shapes

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.MAMBA2
