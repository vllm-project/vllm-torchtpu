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
"""Out-of-tree TPU custom operation and layer registration for MambaMixer2."""

from __future__ import annotations

import functools
from typing import Any, cast

import torch
from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

from vllm_torchtpu.layers.adapter.custom_ops.mamba_op_utils import (
    extract_mamba_step_metadata,
    finish_mamba_forward,
    get_shared_mamba_jax_op,
    resolve_tpu_mamba_mesh_and_validate,
    unpack_mamba_kv_cache,
    use_fused_jax_rms_norm,
)
from vllm_torchtpu.layers.core.mamba_mixer2 import run_jax_mamba_mixer
from vllm_torchtpu.layers.core.mamba_utils import TPU_SSD_BLOCK_LEN
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.utils import fresh_buffer

logger = init_logger(__name__)


@MambaMixer2.register_oot
class VllmMambaMixer2(MambaMixer2):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if is_conv_state_dim_first():
            raise NotImplementedError(
                "VLLM_SSM_CONV_STATE_LAYOUT=DS is not supported for MambaMixer2 on TPU."
            )
        if self.activation not in (None, "silu", "swish"):
            raise NotImplementedError(
                f"Unsupported MambaMixer2 activation {self.activation!r} on TPU; "
                "expected None, 'silu', or 'swish'."
            )
        self._apply_py_norm = False
        self.mamba_op = self._build_mamba_op()

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        # `ragged_causal_conv1d` keeps `conv_state` in fp32 on TPU to avoid
        # per-step bf16 state rounding drift across decode steps.
        # TODO: Validate and enable bf16 `conv_state` storage on TPU.
        conv_state_dtype, ssm_state_dtype = super().get_state_dtype()
        if conv_state_dtype != torch.float32:
            logger.info_once(
                "Overriding MambaMixer2 conv_state dtype from %s "
                "to torch.float32 on TPU.",
                conv_state_dtype,
            )
        return torch.float32, ssm_state_dtype

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        del act_dtype
        conv_weight = cast(torch.nn.Parameter, self.conv1d.weight)
        if conv_weight.ndim != 3:
            raise ValueError(
                f"Expected 3-D MambaMixer2 conv1d.weight (dim, 1, K), "
                f"got shape {tuple(conv_weight.shape)}."
            )
        conv_weight.data = fresh_buffer(conv_weight)

    def _use_fused_jax_rms_norm(self) -> bool:
        return use_fused_jax_rms_norm(self.norm, self.tp_size)

    def _build_mamba_op(self, *, interpret: bool = False):
        mesh = resolve_tpu_mamba_mesh_and_validate(
            "MambaMixer2",
            cache_config=self.cache_config,
            num_spec=self.num_spec,
            use_replayssm=self.use_replayssm,
        )

        local_n_groups = self.n_groups // self.tp_size
        rms_norm_eps = self.norm.variance_epsilon
        apply_rms_norm = self._use_fused_jax_rms_norm()
        self._apply_py_norm = not apply_rms_norm
        has_silu_activation = self.activation in ("silu", "swish")
        out_dim = (self.num_heads // self.tp_size) * self.head_dim

        op_key = (
            id(mesh),
            self.tped_intermediate_size,
            self.tped_conv_size,
            self.tped_dt_size,
            self.head_dim,
            local_n_groups,
            self.ssm_state_size,
            self.conv_kernel_size,
            TPU_SSD_BLOCK_LEN,
            rms_norm_eps,
            apply_rms_norm,
            has_silu_activation,
            interpret,
        )
        wrapped_fn = functools.partial(
            run_jax_mamba_mixer,
            mesh=mesh,
            tped_intermediate_size=self.tped_intermediate_size,
            tped_conv_size=self.tped_conv_size,
            tped_dt_size=self.tped_dt_size,
            num_heads=self.tped_dt_size,
            head_dim=self.head_dim,
            n_groups=local_n_groups,
            d_state=self.ssm_state_size,
            kernel_size=self.conv_kernel_size,
            chunk_size=TPU_SSD_BLOCK_LEN,
            rms_norm_eps=rms_norm_eps,
            apply_rms_norm=apply_rms_norm,
            has_silu_activation=has_silu_activation,
            interpret=interpret,
        )
        return get_shared_mamba_jax_op(
            "mamba_mixer2",
            op_key,
            wrapped_fn,
            out_dim,
            donate_argnums=(1, 2),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        mup_vector: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens = hidden_states.size(0)
        out_dim = (self.num_heads // self.tp_size) * self.head_dim

        projected_states, _ = self.in_proj(hidden_states)
        if mup_vector is not None:
            projected_states = projected_states * mup_vector

        unpacked_cache = unpack_mamba_kv_cache(self.kv_cache, "MambaMixer2")
        if unpacked_cache is None:
            ssm_output = torch.zeros(
                (num_tokens, out_dim),
                dtype=projected_states.dtype,
                device=projected_states.device,
            )
            apply_py_norm = True
        else:
            conv_state, recurrent_state = unpacked_cache
            state_indices, query_start_loc, request_distribution, seq_lens = (
                extract_mamba_step_metadata(self.prefix)
            )
            apply_py_norm = self._apply_py_norm
            norm_weight = cast(torch.Tensor | None, self.norm.weight)
            ssm_output = self.mamba_op(
                projected_states,
                conv_state,
                recurrent_state,
                cast(torch.Tensor, self.conv1d.weight),
                cast(torch.Tensor | None, self.conv1d.bias),
                self.A,
                self.D,
                self.dt_bias,
                norm_weight,
                state_indices,
                query_start_loc,
                request_distribution,
                seq_lens,
            )

        return finish_mamba_forward(
            ssm_output,
            projected_states,
            apply_py_norm=apply_py_norm,
            gate_size=self.tped_intermediate_size,
            is_mimo=False,
            norm=self.norm,
            out_proj=self.out_proj,
        )
