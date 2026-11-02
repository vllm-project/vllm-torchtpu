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

import functools
from typing import TYPE_CHECKING, Protocol, cast

import jax
import torch
from torch_tpu._internal import pallas
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

from vllm_torchtpu.layers.core.mamba_mixer2 import run_jax_mamba_mixer
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    get_vllm_model_wrapper_context,
)
from vllm_torchtpu.utils import fresh_buffer

_DEFAULT_CHUNK_SIZE = 64

if TYPE_CHECKING:

    class _Mamba2AttentionMetadata(Protocol):
        block_tables: torch.Tensor
        seq_lens: torch.Tensor
        query_start_loc: torch.Tensor
        request_distribution: torch.Tensor
        mamba_state_indices: torch.Tensor | None


def mamba_mixer2_core_tpu(
    projected_states: jax.Array,
    conv_state: jax.Array,
    recurrent_state: jax.Array,
    conv_weight: jax.Array,
    conv_bias: jax.Array | None,
    A: jax.Array,
    D: jax.Array,
    dt_bias: jax.Array,
    norm_weight: jax.Array | None,
    state_indices: jax.Array,
    query_start_loc: jax.Array,
    distribution: jax.Array,
    seq_lens: jax.Array,
    *,
    mesh: jax.sharding.Mesh,
    tped_intermediate_size: int,
    tped_conv_size: int,
    tped_dt_size: int,
    num_heads: int,
    head_dim: int,
    n_groups: int,
    d_state: int,
    kernel_size: int,
    chunk_size: int,
    rms_norm_eps: float,
    apply_rms_norm: bool,
    has_silu_activation: bool = True,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    (new_conv_state, new_recurrent_state), output = run_jax_mamba_mixer(
        projected_states,
        conv_state,
        recurrent_state,
        conv_weight,
        conv_bias,
        A,
        D,
        dt_bias,
        norm_weight,
        state_indices,
        query_start_loc,
        distribution,
        seq_lens,
        mesh=mesh,
        tped_intermediate_size=tped_intermediate_size,
        tped_conv_size=tped_conv_size,
        tped_dt_size=tped_dt_size,
        num_heads=num_heads,
        head_dim=head_dim,
        n_groups=n_groups,
        d_state=d_state,
        kernel_size=kernel_size,
        chunk_size=chunk_size,
        rms_norm_eps=rms_norm_eps,
        apply_rms_norm=apply_rms_norm,
        has_silu_activation=has_silu_activation,
        interpret=interpret,
    )
    return new_conv_state, new_recurrent_state, output


@MambaMixer2.register_oot
class VllmMambaMixer2(MambaMixer2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cache_config = self.cache_config
        if cache_config is not None and (
            cache_config.enable_prefix_caching
            or cache_config.mamba_cache_mode in ("all", "align")
        ):
            raise NotImplementedError(
                "Prefix caching is not yet supported for MambaMixer2 on TPU."
            )
        if self.num_spec > 0:
            raise NotImplementedError(
                "Speculative decoding is not yet supported for MambaMixer2 on TPU."
            )
        if self.use_replayssm:
            raise NotImplementedError(
                "--use-replayssm is not supported for MambaMixer2 on TPU."
            )
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
        _, ssm_state_dtype = super().get_state_dtype()
        # Per-layer conv_state is stored in float32 to match the Pallas causal
        # conv1d kernel's VMEM layout without per-step full-pool dtype casts.
        return torch.float32, ssm_state_dtype

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        del act_dtype
        conv_weight = cast(torch.nn.Parameter, self.conv1d.weight)
        w: torch.Tensor = conv_weight
        if w.ndim == 2:
            w = w.unsqueeze(1)
        conv_weight.data = fresh_buffer(w)

    def _use_fused_jax_rms_norm(self) -> bool:
        # When PyTorch runs multiprocess TP (self.tp_size > 1) and the
        # un-replicated group count is 1, Mixer2RMSNormGated requires a
        # cross-rank PyTorch all-reduce. In all other cases, fuse
        # GatedRMSNorm directly inside JAX.
        orig_single_group_replicated = (
            self.tp_size > 1
            and (self.n_groups // self.tp_size) == 1
            and getattr(self.norm, "n_groups", self.n_groups) == 1
        )
        return not orig_single_group_replicated

    def _build_mamba_op(self, *, interpret: bool = False):
        vllm_context = get_vllm_model_wrapper_context()
        mesh = vllm_context.mesh

        local_n_groups = self.n_groups // self.tp_size
        rms_norm_eps = getattr(self.norm, "variance_epsilon", 1e-5)
        apply_rms_norm = self._use_fused_jax_rms_norm()
        self._apply_py_norm = not apply_rms_norm
        has_silu_activation = self.activation in ("silu", "swish")

        op_name = f"pallas::mamba_mixer2_{self.prefix.replace('.', '_')}"
        wrapped_fn = functools.partial(
            mamba_mixer2_core_tpu,
            mesh=mesh,
            tped_intermediate_size=self.tped_intermediate_size,
            tped_conv_size=self.tped_conv_size,
            tped_dt_size=self.tped_dt_size,
            num_heads=self.tped_dt_size,
            head_dim=self.head_dim,
            n_groups=local_n_groups,
            d_state=self.ssm_state_size,
            kernel_size=self.conv_kernel_size,
            chunk_size=_DEFAULT_CHUNK_SIZE,
            rms_norm_eps=rms_norm_eps,
            apply_rms_norm=apply_rms_norm,
            has_silu_activation=has_silu_activation,
            interpret=interpret,
        )

        mamba_jax_op = pallas.jax_op(
            op_name,
            wrapped_fn,
            donate_argnums=(1, 2),
        )

        out_dim = (self.num_heads // self.tp_size) * self.head_dim

        def _fake_mamba(
            projected_states: torch.Tensor,
            conv_state: torch.Tensor,
            recurrent_state: torch.Tensor,
            *args,
            **kwargs,
        ):
            num_tokens = projected_states.size(0)
            return (
                torch.empty_like(conv_state),
                torch.empty_like(recurrent_state),
                torch.empty(
                    (num_tokens, out_dim),
                    dtype=projected_states.dtype,
                    device=projected_states.device,
                ),
            )

        mamba_jax_op.register_fake(_fake_mamba)

        def mamba_impl(
            projected_states: torch.Tensor,
            conv_state: torch.Tensor,
            recurrent_state: torch.Tensor,
            conv_weight: torch.Tensor,
            conv_bias: torch.Tensor | None,
            A: torch.Tensor,
            D: torch.Tensor,
            dt_bias: torch.Tensor,
            norm_weight: torch.Tensor | None,
            state_indices: torch.Tensor,
            query_start_loc: torch.Tensor,
            request_distribution: torch.Tensor,
            seq_lens: torch.Tensor,
        ) -> torch.Tensor:
            new_conv, new_rec, outputs = mamba_jax_op(
                projected_states,
                conv_state,
                recurrent_state,
                conv_weight,
                conv_bias,
                A,
                D,
                dt_bias,
                norm_weight,
                state_indices,
                query_start_loc,
                request_distribution,
                seq_lens,
            )

            conv_state.copy_(new_conv)
            recurrent_state.copy_(new_rec)

            return outputs

        return mamba_impl

    def forward(
        self,
        hidden_states: torch.Tensor,
        mup_vector: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens = hidden_states.size(0)
        out_dim = (self.num_heads // self.tp_size) * self.head_dim

        # 1. Input linear projection
        projected_states, _ = self.in_proj(hidden_states)
        if mup_vector is not None:
            projected_states = projected_states * mup_vector

        # 2. Core Conv1d + SSD Scan
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
                "dict[str, _Mamba2AttentionMetadata]", fc.attn_metadata
            )[self.prefix]
            conv_state, recurrent_state = kv_cache

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

            apply_py_norm = getattr(self, "_apply_py_norm", False)

            norm_weight = getattr(self.norm, "weight", None)
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
                attn_metadata.query_start_loc,
                request_distribution,
                attn_metadata.seq_lens,
            )

        # 3. Gated RMSNorm (applied in JAX unless cross-process TP all-reduce is needed)
        if apply_py_norm:
            gate = projected_states[..., : self.tped_intermediate_size]
            hidden_states = self.norm(ssm_output, gate)
        else:
            hidden_states = ssm_output

        # 4. Final linear projection
        output, _ = self.out_proj(hidden_states)
        return output
