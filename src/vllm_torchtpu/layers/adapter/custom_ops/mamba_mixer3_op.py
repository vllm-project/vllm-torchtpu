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
"""Out-of-tree TPU custom operation and layer registration for MambaMixer3."""

import functools
import itertools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, cast

import jax
import torch
from torch import nn
from torch_tpu._internal import pallas
from vllm.config import CacheConfig, ModelConfig, get_current_vllm_config_or_none
from vllm.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.custom_op import CustomOp, PluggableLayer
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_mixer2 import Mixer2RMSNormGated
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateDtypeCalculator
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.parameter import BasevLLMParameter
from vllm.model_executor.utils import set_weight_attrs
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

from vllm_torchtpu.layers.core.mamba_mixer3 import run_jax_mamba_mixer3
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    get_vllm_model_wrapper_context,
)

if TYPE_CHECKING:

    class _Mamba3AttentionMetadata(Protocol):
        block_tables: torch.Tensor
        seq_lens: torch.Tensor
        query_start_loc: torch.Tensor
        request_distribution: torch.Tensor
        mamba_state_indices: torch.Tensor | None


def mamba_mixer3_core_tpu(
    projected_states: jax.Array,
    aux_state: jax.Array,
    recurrent_state: jax.Array,
    B_norm_weight: jax.Array,
    C_norm_weight: jax.Array,
    B_bias: jax.Array,
    C_bias: jax.Array,
    D: jax.Array,
    dt_bias: jax.Array,
    norm_weight: jax.Array | None,
    psi: jax.Array | None,
    phi: jax.Array | None,
    zeta: jax.Array | None,
    state_indices: jax.Array,
    query_start_loc: jax.Array,
    distribution: jax.Array,
    seq_lens: jax.Array,
    *,
    mesh: jax.sharding.Mesh,
    num_heads: int,
    head_dim: int,
    n_groups: int,
    d_state: int,
    num_rope_angles: int,
    chunk_size: int,
    is_mimo: bool,
    mimo_rank: int,
    A_floor: float,
    rms_norm_eps: float,
    apply_rms_norm: bool,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    r = mimo_rank if is_mimo else 1
    target_aux_dim = r * (head_dim + d_state) + num_rope_angles
    aux_dim = aux_state.shape[-1]
    if aux_dim > target_aux_dim:
        aux_state_in = aux_state[..., :target_aux_dim]
    else:
        aux_state_in = aux_state

    (new_aux_state, new_recurrent_state), output = run_jax_mamba_mixer3(
        projected_states,
        aux_state_in,
        recurrent_state,
        B_norm_weight,
        C_norm_weight,
        B_bias,
        C_bias,
        D,
        dt_bias,
        norm_weight,
        psi,
        phi,
        zeta,
        state_indices,
        query_start_loc,
        distribution,
        seq_lens,
        mesh=mesh,
        num_heads=num_heads,
        head_dim=head_dim,
        n_groups=n_groups,
        d_state=d_state,
        num_rope_angles=num_rope_angles,
        chunk_size=chunk_size,
        is_mimo=is_mimo,
        mimo_rank=mimo_rank,
        A_floor=A_floor,
        rms_norm_eps=rms_norm_eps,
        apply_rms_norm=apply_rms_norm,
    )
    if aux_dim > target_aux_dim:
        new_aux_state = aux_state.at[..., :target_aux_dim].set(new_aux_state)
    return new_aux_state, new_recurrent_state, output


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
        if gate is None:
            return self.forward_native(x, None)
        return super().forward_cuda(x, gate)


@PluggableLayer.register("mamba_mixer3")
class MambaMixer3(MambaBase, PluggableLayer):
    """Mamba-3 Selective State Space Mixer layer with exponential-trapezoidal
    discretization and continuous-time 2D RoPE."""

    def __init__(
        self,
        hidden_size: int,
        ssm_state_size: int,
        intermediate_size: int,
        use_bias: bool = False,
        n_groups: int = 1,
        num_heads: int = 80,
        head_dim: int = 64,
        rms_norm_eps: float = 1e-5,
        use_rms_norm: bool = False,
        is_mimo: bool = False,
        mimo_rank: int = 4,
        rope_fraction: float = 1.0,
        A_floor: float = 1e-4,
        chunk_size: int = 64,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % self.tp_size == 0, (
            "Tensor parallel world size must divide num_heads."
        )
        assert (n_groups % self.tp_size) == 0 or n_groups == 1, (
            "If tensor parallel world size does not divide n_groups, "
            "then n_groups must equal 1."
        )

        self.hidden_size = hidden_size
        self.ssm_state_size = ssm_state_size
        self.intermediate_size = intermediate_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.n_groups = n_groups
        self.is_mimo = is_mimo
        self.mimo_rank = mimo_rank if is_mimo else 1
        self.rope_fraction = rope_fraction
        self.num_rope_angles = int((ssm_state_size // 2) * rope_fraction)
        self.A_floor = A_floor
        self.chunk_size = chunk_size
        self.rms_norm_eps = rms_norm_eps
        self.use_rms_norm = use_rms_norm
        self.model_config = model_config
        self.cache_config = cache_config
        self.prefix = prefix

        self.tp_rank = get_tensor_model_parallel_rank()
        self.tped_intermediate_size = divide(intermediate_size, self.tp_size)
        self.tped_dt_size = divide(num_heads, self.tp_size)
        self.tp_n_groups = (
            divide(n_groups, self.tp_size) if n_groups % self.tp_size == 0 else n_groups
        )
        self.groups_ssm_state_size = (
            self.mimo_rank * self.n_groups * self.ssm_state_size
        )
        self.tped_bc_size = self.mimo_rank * self.tp_n_groups * self.ssm_state_size
        self.aux_dim = (
            self.mimo_rank * (self.head_dim + self.ssm_state_size)
            + self.num_rope_angles
        )
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
        in_proj_loader = self._make_in_proj_weight_loader()
        self._set_param_weight_loader(
            cast(nn.Parameter, self.in_proj.weight), in_proj_loader
        )
        if getattr(self.in_proj, "bias", None) is not None:
            self._set_param_weight_loader(
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
            self.Psi = nn.Parameter(
                torch.ones(self.tped_dt_size, self.mimo_rank, head_dim) / self.mimo_rank
            )
            self.Phi = nn.Parameter(
                torch.ones(self.tped_dt_size, self.mimo_rank, head_dim) / self.mimo_rank
            )
            self.Zeta = nn.Parameter(
                torch.ones(self.tped_dt_size, self.mimo_rank, head_dim)
            )
            set_weight_attrs(self.Psi, {"weight_loader": head_loader})
            set_weight_attrs(self.Phi, {"weight_loader": head_loader})
            set_weight_attrs(self.Zeta, {"weight_loader": head_loader})
        else:
            self.B_bias = nn.Parameter(torch.ones(self.tped_dt_size, ssm_state_size))
            self.C_bias = nn.Parameter(torch.ones(self.tped_dt_size, ssm_state_size))
            self.register_parameter("Psi", None)
            self.register_parameter("Phi", None)
            self.register_parameter("Zeta", None)

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

        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is not None:
            compilation_config = vllm_config.compilation_config
            if prefix and prefix not in compilation_config.static_forward_context:
                compilation_config.static_forward_context[prefix] = self

        self.kv_cache = (torch.tensor([]), torch.tensor([]))

    @staticmethod
    def _set_param_weight_loader(param: nn.Parameter, loader: Any) -> None:
        if isinstance(param, BasevLLMParameter):
            param.weight_loader = loader
        else:
            if hasattr(param, "weight_loader"):
                delattr(param, "weight_loader")
            set_weight_attrs(param, {"weight_loader": loader})

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
        shard_flags = [
            True,
            True,
            (self.n_groups % self.tp_size == 0),
            (self.n_groups % self.tp_size == 0),
            True,
            True,
            True,
            False,
        ]
        tp_size = self.tp_size
        tp_rank = self.tp_rank

        def _loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
            if loaded_weight.shape[0] == param.shape[0]:
                param.data.copy_(loaded_weight)
                return
            sections = torch.split(loaded_weight, section_sizes, dim=0)
            local_sections = []
            for sec, is_sharded in zip(sections, shard_flags):
                if is_sharded and tp_size > 1:
                    shard_len = sec.shape[0] // tp_size
                    local_sections.append(sec.narrow(0, tp_rank * shard_len, shard_len))
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
            assert loaded_weight.shape[0] == shard_size * tp_size
            param.data.copy_(loaded_weight.narrow(0, tp_rank * shard_size, shard_size))

        return _loader

    @property
    def mimo_x(self) -> nn.Parameter | None:
        return self.Psi

    @property
    def mimo_o(self) -> nn.Parameter | None:
        return self.Phi

    @property
    def mimo_z(self) -> nn.Parameter | None:
        return self.Zeta

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        if self.model_config is not None and self.cache_config is not None:
            _, ssm_state_dtype = MambaStateDtypeCalculator.mamba2_state_dtype(
                self.model_config.dtype,
                self.cache_config.mamba_cache_dtype,
                self.cache_config.mamba_ssm_cache_dtype,
            )
            return torch.float32, ssm_state_dtype
        return torch.float32, torch.float32

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        aux_shape = (self.tped_dt_size, self.aux_dim)
        ssm_shape = (self.tped_dt_size, self.head_dim, self.ssm_state_size)
        return aux_shape, ssm_shape

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.MAMBA2


# Mamba-3 custom ops shared across layers with identical static configs. A
# JaxCallable caches its jax.export per instance, so registering one op per
# layer would re-trace and re-lower the identical kernel for every layer.
_shared_mamba3_ops: dict[tuple[Any, ...], Any] = {}
_shared_mamba3_op_ids = itertools.count()


def _get_shared_mamba3_op(key: tuple[Any, ...], build: Callable[[str], Any]) -> Any:
    """Return the ``pallas::mamba_mixer3_<n>`` op for ``key``, building it once."""
    op = _shared_mamba3_ops.get(key)
    if op is None:
        op = build(f"pallas::mamba_mixer3_{next(_shared_mamba3_op_ids)}")
        _shared_mamba3_ops[key] = op
    return op


@MambaMixer3.register_oot
class VllmMambaMixer3(MambaMixer3):
    """Out-of-tree TPU implementation of MambaMixer3 backed by JAX."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        cache_config = getattr(self, "cache_config", None)
        if (
            cache_config is not None
            and (
                getattr(cache_config, "enable_prefix_caching", False) is True
                or getattr(cache_config, "mamba_cache_mode", "none") in ("all", "align")
            )
        ) or getattr(self, "mamba_cache_mode", "none") in ("all", "align"):
            raise NotImplementedError(
                "Prefix caching is not yet supported for MambaMixer3 on TPU."
            )
        self._apply_py_norm = False
        self.mamba_op = self._build_mamba_op()

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        _, ssm_state_dtype = super().get_state_dtype()
        return torch.float32, ssm_state_dtype

    def _use_fused_jax_rms_norm(self) -> bool:
        if self.norm.weight is None:
            return True
        return not (self.tp_size > 1 and self.n_groups == 1)

    def _build_mamba_op(self):
        vllm_context = get_vllm_model_wrapper_context()
        mesh = vllm_context.mesh

        apply_rms_norm = self._use_fused_jax_rms_norm()
        self._apply_py_norm = not apply_rms_norm
        rms_norm_eps = self.norm.variance_epsilon
        out_dim = self.tped_dt_size * self.head_dim

        op_key = (
            id(mesh),
            self.tped_dt_size,
            self.head_dim,
            self.tp_n_groups,
            self.ssm_state_size,
            self.num_rope_angles,
            self.chunk_size,
            self.is_mimo,
            self.mimo_rank,
            self.A_floor,
            rms_norm_eps,
            apply_rms_norm,
        )

        def _build_jax_op(op_name: str) -> Any:
            wrapped_fn = functools.partial(
                mamba_mixer3_core_tpu,
                mesh=mesh,
                num_heads=self.tped_dt_size,
                head_dim=self.head_dim,
                n_groups=self.tp_n_groups,
                d_state=self.ssm_state_size,
                num_rope_angles=self.num_rope_angles,
                chunk_size=self.chunk_size,
                is_mimo=self.is_mimo,
                mimo_rank=self.mimo_rank,
                A_floor=self.A_floor,
                rms_norm_eps=rms_norm_eps,
                apply_rms_norm=apply_rms_norm,
            )
            jax_op = pallas.jax_op(
                op_name,
                wrapped_fn,
                donate_argnums=(1, 2),
            )

            def _fake_mamba3(
                projected_states: torch.Tensor,
                aux_state: torch.Tensor,
                recurrent_state: torch.Tensor,
                *args: Any,
                **kwargs: Any,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                num_tokens = projected_states.size(0)
                return (
                    torch.empty_like(aux_state),
                    torch.empty_like(recurrent_state),
                    torch.empty(
                        (num_tokens, out_dim),
                        dtype=projected_states.dtype,
                        device=projected_states.device,
                    ),
                )

            jax_op.register_fake(_fake_mamba3)
            return jax_op

        mamba_jax_op = _get_shared_mamba3_op(op_key, _build_jax_op)

        def mamba3_impl(
            projected_states: torch.Tensor,
            aux_state: torch.Tensor,
            recurrent_state: torch.Tensor,
            B_norm_weight: torch.Tensor,
            C_norm_weight: torch.Tensor,
            B_bias: torch.Tensor,
            C_bias: torch.Tensor,
            D: torch.Tensor,
            dt_bias: torch.Tensor,
            norm_weight: torch.Tensor | None,
            psi: torch.Tensor | None,
            phi: torch.Tensor | None,
            zeta: torch.Tensor | None,
            state_indices: torch.Tensor,
            query_start_loc: torch.Tensor,
            request_distribution: torch.Tensor,
            seq_lens: torch.Tensor,
        ) -> torch.Tensor:
            new_aux, new_rec, outputs = mamba_jax_op(
                projected_states,
                aux_state,
                recurrent_state,
                B_norm_weight,
                C_norm_weight,
                B_bias,
                C_bias,
                D,
                dt_bias,
                norm_weight,
                psi,
                phi,
                zeta,
                state_indices,
                query_start_loc,
                request_distribution,
                seq_lens,
            )

            aux_state.copy_(new_aux)
            recurrent_state.copy_(new_rec)
            return outputs

        return mamba3_impl

    def forward(
        self,
        hidden_states: torch.Tensor,
        mup_vector: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens = hidden_states.size(0)
        out_dim = self.tped_dt_size * self.head_dim

        projected_states, _ = self.in_proj(hidden_states)
        if mup_vector is not None:
            projected_states = projected_states * mup_vector

        kv_cache = getattr(self, "kv_cache", None)
        if kv_cache is None or kv_cache[0].numel() == 0:
            ssm_output = torch.zeros(
                (num_tokens, out_dim),
                dtype=projected_states.dtype,
                device=projected_states.device,
            )
            apply_py_norm = True
        else:
            fc = get_forward_context()
            attn_metadata = cast(
                "dict[str, _Mamba3AttentionMetadata]", fc.attn_metadata
            )[self.prefix]
            aux_state, recurrent_state = kv_cache

            if attn_metadata.mamba_state_indices is not None:
                state_indices = attn_metadata.mamba_state_indices.to(torch.int32)
            else:
                max_reqs = attn_metadata.seq_lens.shape[0]
                max_blocks_per_req = attn_metadata.block_tables.shape[0] // max_reqs
                block_tables_2d = torch.reshape(
                    attn_metadata.block_tables,
                    (max_reqs, max_blocks_per_req),
                )
                state_indices = block_tables_2d[:, 0].to(torch.int32)

            request_distribution = attn_metadata.request_distribution
            mamba_request_distribution = getattr(
                attn_metadata, "mamba_request_distribution", None
            )
            if mamba_request_distribution is not None:
                request_distribution = mamba_request_distribution

            apply_py_norm = self._apply_py_norm
            norm_weight = cast(torch.Tensor | None, self.norm.weight)

            ssm_output = self.mamba_op(
                projected_states,
                aux_state,
                recurrent_state,
                self.B_norm_weight,
                self.C_norm_weight,
                self.B_bias,
                self.C_bias,
                self.D,
                self.dt_bias,
                norm_weight,
                self.Psi,
                self.Phi,
                self.Zeta,
                state_indices,
                attn_metadata.query_start_loc,
                request_distribution,
                attn_metadata.seq_lens,
            )

        if apply_py_norm:
            gate = (
                None
                if self.is_mimo
                else projected_states[..., : self.tped_intermediate_size]
            )
            if self.norm.weight is not None:
                hidden_states = self.norm(ssm_output, gate)
            elif gate is not None:
                hidden_states = (
                    ssm_output.to(torch.float32)
                    * nn.functional.silu(gate.to(torch.float32))
                ).to(projected_states.dtype)
            else:
                hidden_states = ssm_output
        else:
            hidden_states = ssm_output

        output, _ = self.out_proj(hidden_states)
        return output
