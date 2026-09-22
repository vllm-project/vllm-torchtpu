# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-native [token, branch, hidden] HyperConnections.

Reuse the common checkpoint structure with a TPU-specific SiLU precision
boundary; the NVIDIA model owns CUDA kernels and is not imported here.
"""
import torch
import torch.nn.functional as F
from vllm.models.qwen4_exp.common import hyperconnection as common_hc

GroupedGemmaRMSNorm = common_hc.GroupedGemmaRMSNorm
HyperConnectionBase = common_hc.HyperConnectionBase
HyperConnectionConfig = common_hc.HyperConnectionConfig


class GatedResidual(common_hc.GatedResidual):
    """Match NVIDIA HC FP32 elementwise math between BF16 boundaries."""

    def mix(self, hyper_input):
        assert hyper_input.shape[-1] == self.hc_count * self.hidden_size
        if not hasattr(self, "input_mix_weight_down"):
            raise RuntimeError("mix was disabled for this hyper-connection")
        norm = self._normalize(hyper_input)
        gate = F.linear(norm,
                        self.input_mix_weight_down.weight) / self.hc_count
        # TorchTPU can lower BF16 SiLU through BF16 logistic/tanh, rounding
        # negative tails to zero. Compute the activation in FP32 and round
        # once, matching NVIDIA hc_silu's BF16 result boundary.
        gate = F.silu(gate.float()).to(gate.dtype)
        gate = F.linear(gate, self.input_mix_weight_up.weight)
        gate = torch.sigmoid(gate.float()).unflatten(
            -1, (self.hc_count, self.hidden_size))
        mixed = (
            gate *
            norm.float().unflatten(-1,
                                   (self.hc_count, self.hidden_size))).mean(-2)
        return mixed.to(hyper_input.dtype), (hyper_input, norm)

    def combine(self, block_output, residuals):
        if not hasattr(self, "block_inject_weight"):
            raise RuntimeError(
                "combine was disabled for this hyper-connection")
        hyper_input, hyper_input_normed = residuals
        assert hyper_input.shape[-1] == self.hc_count * self.hidden_size
        assert block_output.shape[-1] == self.hidden_size

        injection = F.linear(hyper_input_normed,
                             self.block_inject_weight.weight)
        injection = 2.0 * torch.sigmoid(injection.float() / self.hc_count)
        residual = hyper_input.float().unflatten(
            -1, (self.hc_count, self.hidden_size))
        output = (residual +
                  block_output.float().unsqueeze(-2) * injection.unsqueeze(-1))
        return output.flatten(-2).to(hyper_input.dtype)


__all__ = [
    "GatedResidual", "GroupedGemmaRMSNorm", "HyperConnectionBase",
    "HyperConnectionConfig"
]
