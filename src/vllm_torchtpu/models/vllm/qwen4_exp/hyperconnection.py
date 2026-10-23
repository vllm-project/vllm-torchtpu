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
"""Checkpoint-native [token, branch, hidden] HyperConnections.

Reuse the common checkpoint structure with a TPU-specific SiLU precision
boundary; the NVIDIA model owns CUDA kernels and is not imported here.
"""

import torch
import torch.nn.functional as F
from vllm.models.qwen4_exp.common import hyperconnection as common_hc

from vllm_torchtpu.models.vllm.qwen4_exp.hc_prefill_policy import (
    HcPrefillMode,
    combine_auto_min_tokens,
    full_mix_auto_min_tokens,
    validate_hc_prefill_mode,
)

GroupedGemmaRMSNorm = common_hc.GroupedGemmaRMSNorm
HyperConnectionBase = common_hc.HyperConnectionBase
HyperConnectionConfig = common_hc.HyperConnectionConfig


class GatedResidual(common_hc.GatedResidual):
    """Match NVIDIA HC math with independent calibrated prefill choices.

    Auto captures hardware after worker initialization. A module constructed
    before initialization retains Torch fallback even after device transfer.
    Zero disables a component; positive tiles force only resource-safe routes.
    """

    def __init__(
        self,
        *args,
        prefill_full_tile: HcPrefillMode = "auto",
        prefill_combine_tile: HcPrefillMode | None = None,
        **kwargs,
    ):
        prefill_full_tile = validate_hc_prefill_mode(prefill_full_tile, full_mix=True)
        if prefill_combine_tile is None:
            prefill_combine_tile = prefill_full_tile
        prefill_combine_tile = validate_hc_prefill_mode(
            prefill_combine_tile, full_mix=False
        )
        super().__init__(*args, **kwargs)
        self.prefill_full_tile = prefill_full_tile
        self.prefill_combine_tile = prefill_combine_tile
        self._full_token_tile = (
            128 if prefill_full_tile == "auto" else prefill_full_tile
        )
        self._combine_token_tile = (
            128 if prefill_combine_tile == "auto" else prefill_combine_tile
        )
        self._auto_mix_min_tokens: int | None = None
        self._auto_combine_min_tokens: int | None = None
        self._prefill_full_op = None
        self._prefill_combine_op = None
        if not (prefill_full_tile or prefill_combine_tile):
            return
        from vllm_torchtpu.kernels.qwen4_exp.hc_resources import (
            supports_full_mix,
            supports_residual,
        )
        from vllm_torchtpu.kernels.qwen4_exp.vllm_adapter import (
            build_full_mix_op,
            build_gated_residual_op,
            current_hc_hardware,
        )

        # Worker initialization owns the device. Unknown layout/resources keep
        # the existing Torch path; CPU construction never initializes a TPU.
        hardware = current_hc_hardware()
        policy_args = (
            hardware,
            self.hc_count,
            self.hidden_size,
            self.config.hc_lowrank,
            self.config.hc_per_branch_norm,
            self.hc_norm.variance_epsilon,
        )
        self._auto_mix_min_tokens = full_mix_auto_min_tokens(*policy_args)
        self._auto_combine_min_tokens = combine_auto_min_tokens(*policy_args)
        if (
            hasattr(self, "block_inject_weight")
            and (
                prefill_combine_tile != "auto"
                or self._auto_combine_min_tokens is not None
            )
            and supports_residual(
                self.hc_count, self.hidden_size, self._combine_token_tile, hardware
            )
        ):
            self._prefill_combine_op = build_gated_residual_op(
                self.hc_count, self.hidden_size, self._combine_token_tile, hardware
            )
        if (
            self.config.hc_per_branch_norm
            and hasattr(self, "input_mix_weight_down")
            and (prefill_full_tile != "auto" or self._auto_mix_min_tokens is not None)
            and supports_full_mix(
                self.hc_count,
                self.hidden_size,
                self.input_mix_weight_down.weight.shape[0],
                self._full_token_tile,
                hardware,
            )
        ):
            self._prefill_full_op = build_full_mix_op(
                self.hc_count,
                self.hidden_size,
                self.hc_norm.variance_epsilon,
                self._full_token_tile,
                hardware,
            )

    def _normalize(self, hyper_input):
        if not self.config.hc_per_branch_norm:
            return super()._normalize(hyper_input)
        # At prefill sizes, reshaping FP32 [tokens, HC*H] to [tokens, HC, H]
        # changes the TPU layout. Reducing contiguous branches separately
        # keeps the projection-native flat layout and fuses their variance.
        flat = hyper_input.float()
        normalized = []
        for branch in range(self.hc_count):
            start = branch * self.hidden_size
            values = flat[..., start : start + self.hidden_size]
            variance = values.square().mean(dim=-1, keepdim=True)
            normalized.append(
                values * torch.rsqrt(variance + self.hc_norm.variance_epsilon)
            )
        output = torch.cat(normalized, dim=-1)
        return (output * (1.0 + self.hc_norm.weight.float())).to(hyper_input.dtype)

    def _use_prefill_mix(self, hyper_input):
        return (
            self._prefill_full_op is not None
            and hyper_input.ndim == 2
            and hyper_input.shape[0] >= self._full_token_tile
            and hyper_input.shape[0] % self._full_token_tile == 0
            and (
                self.prefill_full_tile != "auto"
                or (
                    self._auto_mix_min_tokens is not None
                    and hyper_input.shape[0] >= self._auto_mix_min_tokens
                    and hyper_input.device.type == "tpu"
                    and self.hc_norm.weight.device == hyper_input.device
                    and self.input_mix_weight_down.weight.device == hyper_input.device
                    and self.input_mix_weight_up.weight.device == hyper_input.device
                )
            )
            and hyper_input.dtype == torch.bfloat16
            and self.hc_norm.weight.dtype == torch.bfloat16
            and self.input_mix_weight_down.weight.dtype == torch.bfloat16
            and self.input_mix_weight_up.weight.dtype == torch.bfloat16
        )

    def mix(self, hyper_input):
        assert hyper_input.shape[-1] == self.hc_count * self.hidden_size
        if not hasattr(self, "input_mix_weight_down"):
            raise RuntimeError("mix was disabled for this hyper-connection")
        if self._use_prefill_mix(hyper_input):
            assert self._prefill_full_op is not None
            mixed, norm = self._prefill_full_op(
                hyper_input,
                self.hc_norm.weight,
                self.input_mix_weight_down.weight,
                self.input_mix_weight_up.weight,
            )
            return mixed, (hyper_input, norm)
        norm = self._normalize(hyper_input)
        gate = F.linear(norm, self.input_mix_weight_down.weight)
        # TorchTPU can lower BF16 SiLU through BF16 logistic/tanh, rounding
        # negative tails to zero. Compute the activation in FP32 and round
        # once, matching NVIDIA hc_silu's BF16 result boundary.
        gate = F.silu(gate.float() / self.hc_count).to(gate.dtype)
        gate = F.linear(gate, self.input_mix_weight_up.weight)
        gate = torch.sigmoid(gate.float())
        # Reduce the HC streams in the flat matmul-output layout. Unflattening
        # to [tokens, hc_count, hidden_size] forces an F32 layout conversion at
        # prefill sizes, while these fixed branch slices fuse into one reduction.
        weighted = gate * norm.float()
        mixed = weighted[..., : self.hidden_size]
        for branch in range(1, self.hc_count):
            start = branch * self.hidden_size
            mixed = mixed + weighted[..., start : start + self.hidden_size]
        mixed = mixed / self.hc_count
        return mixed.to(hyper_input.dtype), (hyper_input, norm)

    def _use_prefill_combine(self, block_output, hyper_input, hyper_input_normed):
        return (
            self._prefill_combine_op is not None
            and hyper_input.ndim == 2
            and block_output.ndim == 2
            and hyper_input_normed.ndim == 2
            and block_output.shape[0] == hyper_input.shape[0]
            and hyper_input_normed.shape[0] == hyper_input.shape[0]
            and hyper_input.shape[0] >= self._combine_token_tile
            and hyper_input.shape[0] % self._combine_token_tile == 0
            and (
                self.prefill_combine_tile != "auto"
                or (
                    self._auto_combine_min_tokens is not None
                    and hyper_input.shape[0] >= self._auto_combine_min_tokens
                    and hyper_input.device.type == "tpu"
                    and block_output.device == hyper_input.device
                    and hyper_input_normed.device == hyper_input.device
                    and self.hc_norm.weight.device == hyper_input.device
                    and self.block_inject_weight.weight.device == hyper_input.device
                    and hyper_input_normed.dtype == torch.bfloat16
                    and self.hc_norm.weight.dtype == torch.bfloat16
                    and self.block_inject_weight.weight.dtype == torch.bfloat16
                )
            )
            and hyper_input.dtype == torch.bfloat16
            and block_output.dtype == torch.bfloat16
        )

    def combine(self, block_output, residuals):
        if not hasattr(self, "block_inject_weight"):
            raise RuntimeError("combine was disabled for this hyper-connection")
        hyper_input, hyper_input_normed = residuals
        assert hyper_input.shape[-1] == self.hc_count * self.hidden_size
        assert block_output.shape[-1] == self.hidden_size

        injection = F.linear(hyper_input_normed, self.block_inject_weight.weight)
        injection = 2.0 * torch.sigmoid(injection.float() / self.hc_count)
        if self._use_prefill_combine(block_output, hyper_input, hyper_input_normed):
            assert self._prefill_combine_op is not None
            return self._prefill_combine_op(hyper_input, block_output, injection)
        # Keep each HC branch in the projection-native flat layout. The
        # three-dimensional FP32 broadcast otherwise requires a full-layout
        # copy before the next consumer at prefill sizes.
        residual = hyper_input.float()
        block = block_output.float()
        output = torch.cat(
            [
                residual[
                    ..., branch * self.hidden_size : (branch + 1) * self.hidden_size
                ]
                + block * injection[..., branch : branch + 1]
                for branch in range(self.hc_count)
            ],
            dim=-1,
        )
        return output.to(hyper_input.dtype)


__all__ = [
    "GatedResidual",
    "GroupedGemmaRMSNorm",
    "HyperConnectionBase",
    "HyperConnectionConfig",
]
