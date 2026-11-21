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

from __future__ import annotations

import functools
from typing import Any, cast

import torch

from vllm_torchtpu.layers.adapter.custom_ops.mamba_op_utils import (
    extract_mamba_step_metadata,
    finish_mamba_forward,
    get_shared_mamba_jax_op,
    resolve_tpu_mamba_mesh_and_validate,
    unpack_mamba_kv_cache,
    use_fused_jax_rms_norm,
)
from vllm_torchtpu.layers.adapter.mamba_mixer3 import MambaMixer3
from vllm_torchtpu.layers.core.mamba_mixer3 import run_jax_mamba_mixer3
from vllm_torchtpu.layers.core.mamba_utils import TPU_SSD_BLOCK_LEN


@MambaMixer3.register_oot
class VllmMambaMixer3(MambaMixer3):
    """Out-of-tree TPU implementation of MambaMixer3 backed by JAX/Pallas."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._apply_py_norm = False
        self.mamba_op = self._build_mamba_op()

    def _use_fused_jax_rms_norm(self) -> bool:
        return use_fused_jax_rms_norm(self.norm, self.tp_size)

    def _build_mamba_op(self, *, interpret: bool = False):
        mesh = resolve_tpu_mamba_mesh_and_validate(
            "MambaMixer3",
            cache_config=self.cache_config,
        )
        if TPU_SSD_BLOCK_LEN % self.mimo_rank != 0:
            raise ValueError(
                f"TPU_SSD_BLOCK_LEN ({TPU_SSD_BLOCK_LEN}) must be divisible by "
                f"mimo_rank ({self.mimo_rank})."
            )
        chunk_size = TPU_SSD_BLOCK_LEN // self.mimo_rank

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
            chunk_size,
            self.is_mimo,
            self.mimo_rank,
            self.A_floor,
            rms_norm_eps,
            apply_rms_norm,
            interpret,
        )
        wrapped_fn = functools.partial(
            run_jax_mamba_mixer3,
            mesh=mesh,
            num_heads=self.tped_dt_size,
            head_dim=self.head_dim,
            n_groups=self.tp_n_groups,
            d_state=self.ssm_state_size,
            num_rope_angles=self.num_rope_angles,
            chunk_size=chunk_size,
            is_mimo=self.is_mimo,
            mimo_rank=self.mimo_rank,
            A_floor=self.A_floor,
            rms_norm_eps=rms_norm_eps,
            apply_rms_norm=apply_rms_norm,
            interpret=interpret,
        )
        return get_shared_mamba_jax_op(
            "mamba_mixer3",
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
        out_dim = self.tped_dt_size * self.head_dim

        projected_states, _ = self.in_proj(hidden_states)
        if mup_vector is not None:
            projected_states = projected_states * mup_vector

        unpacked_cache = unpack_mamba_kv_cache(self.kv_cache, "MambaMixer3")
        if unpacked_cache is None:
            ssm_output = torch.zeros(
                (num_tokens, out_dim),
                dtype=projected_states.dtype,
                device=projected_states.device,
            )
            apply_py_norm = True
        else:
            aux_state, recurrent_state = unpacked_cache
            state_indices, query_start_loc, request_distribution, seq_lens = (
                extract_mamba_step_metadata(self.prefix)
            )
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
                self.mimo_x,
                self.mimo_o,
                self.mimo_z,
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
            is_mimo=self.is_mimo,
            norm=self.norm,
            out_proj=self.out_proj,
        )
