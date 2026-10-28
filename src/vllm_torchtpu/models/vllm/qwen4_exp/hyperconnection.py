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

from collections.abc import Callable

import torch
import torch.nn.functional as F
from vllm.models.qwen4_exp.common import hyperconnection as common_hc

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.qwen4_exp.hc_prefill_policy import (
    AUTO_COMBINE_TILE,
    AUTO_MIN_TOKENS,
    AUTO_MIX_TILES,
    HcPrefillMode,
    select_mix_tile,
    validate_hc_prefill_mode,
)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

GroupedGemmaRMSNorm = common_hc.GroupedGemmaRMSNorm
HyperConnectionBase = common_hc.HyperConnectionBase
HyperConnectionConfig = common_hc.HyperConnectionConfig


def _same_tpu_bf16(reference: torch.Tensor, *tensors: torch.Tensor) -> bool:
    return reference.device.type == "tpu" and all(
        tensor.device == reference.device and tensor.dtype == torch.bfloat16
        for tensor in (reference, *tensors)
    )


class GatedResidual(common_hc.GatedResidual):
    """Match NVIDIA HC math with independent prefill choices.

    Auto captures hardware after worker initialization. A module constructed
    before initialization retains Torch fallback even after device transfer.
    Zero disables a component; positive tiles force only resource-safe routes.
    """

    def __init__(
        self,
        config: HyperConnectionConfig,
        layer_idx: int | None = None,
        role: str | None = None,
        use_mix: bool = True,
        use_combine: bool = True,
        *,
        prefill_full_tile: HcPrefillMode | None = None,
        prefill_combine_tile: HcPrefillMode | None = None,
    ) -> None:
        if prefill_full_tile is None:
            tile = envs.VLLM_TPU_HC_PREFILL_TILE
            prefill_full_tile = "auto" if tile is None else tile
        prefill_full_tile = validate_hc_prefill_mode(prefill_full_tile, full_mix=True)
        if prefill_combine_tile is None:
            prefill_combine_tile = prefill_full_tile
        prefill_combine_tile = validate_hc_prefill_mode(
            prefill_combine_tile, full_mix=False
        )
        super().__init__(
            config,
            layer_idx=layer_idx,
            role=role,
            use_mix=use_mix,
            use_combine=use_combine,
        )
        self.use_mix = use_mix
        self.use_combine = use_combine
        self.prefill_full_tile = prefill_full_tile
        self.prefill_combine_tile = prefill_combine_tile
        full_tiles = (
            AUTO_MIX_TILES if prefill_full_tile == "auto" else (prefill_full_tile,)
        )
        self._combine_token_tile = (
            AUTO_COMBINE_TILE
            if prefill_combine_tile == "auto"
            else prefill_combine_tile
        )
        self._prefill_full_ops: dict[int, Callable] = {}
        self._prefill_combine_op = None
        if not (use_mix or use_combine) or not (
            prefill_full_tile or prefill_combine_tile
        ):
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
        if self.use_combine and self._combine_token_tile:
            if (
                prefill_combine_tile != "auto"
                and hardware is not None
                and hardware.supported
                and self._combine_token_tile % hardware.row_alignment
            ):
                raise ValueError(
                    "HC prefill combine tile must be a multiple of "
                    f"{hardware.row_alignment} for {hardware}"
                )
            if supports_residual(
                self.hc_count, self.hidden_size, self._combine_token_tile, hardware
            ):
                self._prefill_combine_op = build_gated_residual_op(
                    self.hc_count, self.hidden_size, self._combine_token_tile, hardware
                )
            else:
                logger.warning_once(
                    "HC combine uses Torch: unsupported resources/layout "
                    "(branches=%s, hidden=%s, tile=%s, hardware=%s)",
                    self.hc_count,
                    self.hidden_size,
                    self._combine_token_tile,
                    hardware,
                )
        if self.use_mix and prefill_full_tile:
            if self.config.hc_per_branch_norm:
                self._prefill_full_ops = {
                    tile: build_full_mix_op(
                        self.hc_count,
                        self.hidden_size,
                        self.hc_norm.variance_epsilon,
                        tile,
                        hardware,
                    )
                    for tile in full_tiles
                    if supports_full_mix(
                        self.hc_count,
                        self.hidden_size,
                        self.input_mix_weight_down.weight.shape[0],
                        tile,
                        hardware,
                    )
                }
                reason = "unsupported resources/layout"
            else:
                reason = "full mix requires per-branch normalization"
            if not self._prefill_full_ops:
                logger.warning_once(
                    "HC full mix uses Torch: %s "
                    "(branches=%s, hidden=%s, lowrank=%s, tiles=%s, hardware=%s)",
                    reason,
                    self.hc_count,
                    self.hidden_size,
                    self.input_mix_weight_down.weight.shape[0],
                    full_tiles,
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
            self.use_mix
            and bool(self._prefill_full_ops)
            and hyper_input.ndim == 2
            and (
                self.prefill_full_tile != "auto"
                or hyper_input.shape[0] >= AUTO_MIN_TOKENS
            )
            and _same_tpu_bf16(
                hyper_input,
                self.hc_norm.weight,
                self.input_mix_weight_down.weight,
                self.input_mix_weight_up.weight,
            )
            and select_mix_tile(hyper_input.shape[0], self._prefill_full_ops)
            is not None
        )

    def mix(self, hyper_input):
        assert hyper_input.shape[-1] == self.hc_count * self.hidden_size
        if not self.use_mix:
            raise RuntimeError("mix was disabled for this hyper-connection")
        if self._use_prefill_mix(hyper_input):
            tile = select_mix_tile(hyper_input.shape[0], self._prefill_full_ops)
            assert tile is not None
            mixed, norm = self._prefill_full_ops[tile](
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
            self.use_combine
            and self._prefill_combine_op is not None
            and hyper_input.ndim == 2
            and block_output.ndim == 2
            and hyper_input_normed.ndim == 2
            and block_output.shape[0] == hyper_input.shape[0]
            and hyper_input_normed.shape[0] == hyper_input.shape[0]
            and hyper_input.shape[0] >= self._combine_token_tile
            and hyper_input.shape[0] % self._combine_token_tile == 0
            and (
                self.prefill_combine_tile != "auto"
                or hyper_input.shape[0] >= AUTO_MIN_TOKENS
            )
            and _same_tpu_bf16(
                hyper_input,
                block_output,
                hyper_input_normed,
                self.block_inject_weight.weight,
            )
        )

    def combine(self, block_output, residuals):
        if not self.use_combine:
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
