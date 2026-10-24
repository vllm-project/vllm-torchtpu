# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-native [token, branch, hidden] HyperConnections.

Reuse the common checkpoint structure with a TPU-specific SiLU precision
boundary; the NVIDIA model owns CUDA kernels and is not imported here.
"""

import torch
import torch.nn.functional as F
from vllm.models.qwen4_exp.common import hyperconnection as common_hc

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.qwen4_exp.hc_decode import decode_token_tile

GroupedGemmaRMSNorm = common_hc.GroupedGemmaRMSNorm
HyperConnectionBase = common_hc.HyperConnectionBase
HyperConnectionConfig = common_hc.HyperConnectionConfig


class GatedResidual(common_hc.GatedResidual):
    """Match NVIDIA HC FP32 elementwise math between BF16 boundaries."""

    def __init__(
        self,
        *args,
        prefill_up_tile: int = 0,
        prefill_norm_tile: int = 0,
        prefill_full_tile: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if prefill_up_tile and (prefill_up_tile < 128 or prefill_up_tile % 128):
            raise ValueError("HC prefill up tile must be a multiple of 128")
        if prefill_norm_tile and (prefill_norm_tile < 64 or prefill_norm_tile % 64):
            raise ValueError("HC prefill norm tile must be a multiple of 64")
        if prefill_full_tile and (prefill_full_tile < 128 or prefill_full_tile % 128):
            raise ValueError("HC prefill full mix tile must be a multiple of 128")
        self.prefill_up_tile = prefill_up_tile
        self.prefill_norm_tile = prefill_norm_tile
        self.prefill_full_tile = prefill_full_tile
        self._prefill_up_op = None
        self._prefill_norm_op = None
        self._prefill_full_op = None
        self._prefill_combine_op = None
        if (
            prefill_full_tile
            and self.hidden_size % 128 == 0
            and hasattr(self, "block_inject_weight")
        ):
            from vllm_torchtpu.kernels.qwen4_exp.vllm_adapter import (
                build_gated_residual_op,
            )

            self._prefill_combine_op = build_gated_residual_op(
                self.hc_count, self.hidden_size, prefill_full_tile
            )
        if (
            prefill_full_tile
            and self.config.hc_per_branch_norm
            and hasattr(self, "input_mix_weight_down")
        ):
            from vllm_torchtpu.kernels.qwen4_exp.vllm_adapter import build_full_mix_op

            self._prefill_full_op = build_full_mix_op(
                self.hc_count,
                self.hidden_size,
                self.hc_norm.variance_epsilon,
                prefill_full_tile,
            )
        if prefill_norm_tile and self.config.hc_per_branch_norm:
            from vllm_torchtpu.kernels.qwen4_exp.vllm_adapter import build_norm_op

            self._prefill_norm_op = build_norm_op(
                self.hc_count,
                self.hidden_size,
                self.hc_norm.variance_epsilon,
                prefill_norm_tile,
            )
        self._decode_mix_op = None
        self._decode_combine_mix_op = None
        if (
            envs.TPU_QWEN4_HC_DECODE_FUSED
            and self.config.hc_per_branch_norm
            and self.hidden_size % 128 == 0
            and hasattr(self, "input_mix_weight_down")
        ):
            from vllm_torchtpu.kernels.qwen4_exp.vllm_adapter import (
                build_decode_combine_mix_op,
                build_decode_mix_op,
            )

            inject = hasattr(self, "block_inject_weight")
            eps = self.hc_norm.variance_epsilon
            self._decode_mix_op = build_decode_mix_op(
                self.hc_count, self.hidden_size, eps, inject
            )
            if inject:
                # Used by the preceding HC's combine_mix (attn -> mlp).
                self._decode_combine_mix_op = build_decode_combine_mix_op(
                    self.hc_count, self.hidden_size, eps
                )
        if prefill_up_tile and hasattr(self, "input_mix_weight_up"):
            # Register before Dynamo traces forward; registration itself uses
            # Python inspect and cannot execute inside a compiled graph.
            from vllm_torchtpu.kernels.qwen4_exp.vllm_adapter import (
                build_up_epilogue_op,
            )

            self._prefill_up_op = build_up_epilogue_op(
                self.hc_count, self.hidden_size, prefill_up_tile
            )

    def _normalize(self, hyper_input):
        if not self.config.hc_per_branch_norm:
            return super()._normalize(hyper_input)
        if (
            self._prefill_norm_op is not None
            and hyper_input.shape[0] >= self.prefill_norm_tile
            and hyper_input.shape[0] % self.prefill_norm_tile == 0
            and hyper_input.dtype == torch.bfloat16
            and self.hc_norm.weight.dtype == torch.bfloat16
        ):
            return self._prefill_norm_op(hyper_input, self.hc_norm.weight)

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

    def _decode_weights(self):
        weights = (
            self.hc_norm.weight,
            self.input_mix_weight_down.weight,
            self.input_mix_weight_up.weight,
        )
        if hasattr(self, "block_inject_weight"):
            weights += (self.block_inject_weight.weight,)
        return weights

    def _decode_fused(self, hyper_input) -> bool:
        """TPU_QWEN4_HC_DECODE_FUSED: one Pallas call for <= 512 BF16 rows."""
        return (
            self._decode_mix_op is not None
            and hyper_input.dim() == 2
            and decode_token_tile(hyper_input.shape[0]) is not None
            and hyper_input.dtype == torch.bfloat16
            and all(w.dtype == torch.bfloat16 for w in self._decode_weights())
        )

    def _residual(self, hyper_input, norm):
        """``mix``'s residual for ``combine``, with the injection gate computed
        here as the fused decode mix computes it.

        This keeps ``block_inject_weight``'s first use inside ``mix`` on every
        path. Piecewise compilation orders a subgraph's inputs by first use, so
        a fused decode bucket and an unfused bucket of the same model then take
        the same inputs in the same order; the shape-variant hand-over refuses
        a bucket whose inputs moved (h21 j-b1210ac2).
        """
        if not hasattr(self, "block_inject_weight"):
            return hyper_input, norm
        injection = F.linear(norm, self.block_inject_weight.weight)
        return hyper_input, 2.0 * torch.sigmoid(injection.float() / self.hc_count)

    def mix(self, hyper_input):
        """Returns ``mixed, (hyper_input, injection)``: the combine's [T, hc] FP32
        injection gate, computed from the normalized input (``normed`` in its
        place for a hyper-connection without a combine); ``combine`` accepts
        either form."""
        assert hyper_input.shape[-1] == self.hc_count * self.hidden_size
        if not hasattr(self, "input_mix_weight_down"):
            raise RuntimeError("mix was disabled for this hyper-connection")
        if self._decode_fused(hyper_input):
            out = self._decode_mix_op(hyper_input, *self._decode_weights())
            if hasattr(self, "block_inject_weight"):
                mixed, injection = out
                return mixed, (hyper_input, injection)
            return out, (hyper_input, None)
        if (
            self._prefill_full_op is not None
            and hyper_input.shape[0] >= 2 * self.prefill_full_tile
            and hyper_input.shape[0] % (2 * self.prefill_full_tile) == 0
            and hyper_input.dtype == torch.bfloat16
            and self.hc_norm.weight.dtype == torch.bfloat16
            and self.input_mix_weight_down.weight.dtype == torch.bfloat16
            and self.input_mix_weight_up.weight.dtype == torch.bfloat16
        ):
            mixed, norm = self._prefill_full_op(
                hyper_input,
                self.hc_norm.weight,
                self.input_mix_weight_down.weight,
                self.input_mix_weight_up.weight,
            )
            return mixed, self._residual(hyper_input, norm)
        norm = self._normalize(hyper_input)
        gate = F.linear(norm, self.input_mix_weight_down.weight) / self.hc_count
        # TorchTPU can lower BF16 SiLU through BF16 logistic/tanh, rounding
        # negative tails to zero. Compute the activation in FP32 and round
        # once, matching NVIDIA hc_silu's BF16 result boundary.
        gate = F.silu(gate.float()).to(gate.dtype)
        if (
            self._prefill_up_op is not None
            and gate.shape[0] >= self.prefill_up_tile
            and gate.shape[0] % self.prefill_up_tile == 0
            and gate.dtype == torch.bfloat16
            and norm.dtype == torch.bfloat16
            and self.input_mix_weight_up.weight.dtype == torch.bfloat16
        ):
            mixed = self._prefill_up_op(gate, norm, self.input_mix_weight_up.weight)
            return mixed, self._residual(hyper_input, norm)
        gate = F.linear(gate, self.input_mix_weight_up.weight)
        gate = torch.sigmoid(gate.float()).unflatten(
            -1, (self.hc_count, self.hidden_size)
        )
        # Reduce the HC streams in the flat matmul-output layout. Unflattening
        # to [tokens, hc_count, hidden_size] forces an F32 layout conversion at
        # prefill sizes, while these fixed branch slices fuse into one reduction.
        weighted = gate.flatten(-2) * norm.float()
        mixed = weighted[..., : self.hidden_size]
        for branch in range(1, self.hc_count):
            start = branch * self.hidden_size
            mixed = mixed + weighted[..., start : start + self.hidden_size]
        mixed = mixed / self.hc_count
        return mixed.to(hyper_input.dtype), self._residual(hyper_input, norm)

    def combine(self, block_output, residuals):
        if not hasattr(self, "block_inject_weight"):
            raise RuntimeError("combine was disabled for this hyper-connection")
        hyper_input, hyper_input_normed = residuals
        assert hyper_input.shape[-1] == self.hc_count * self.hidden_size
        assert block_output.shape[-1] == self.hidden_size

        if hyper_input_normed.shape[-1] == self.hc_count:
            # The fused decode mix already produced the injection gate.
            injection = hyper_input_normed
        else:
            injection = F.linear(hyper_input_normed, self.block_inject_weight.weight)
            injection = 2.0 * torch.sigmoid(injection.float() / self.hc_count)
        if (
            self._prefill_combine_op is not None
            and hyper_input.shape[0] >= 2 * self.prefill_full_tile
            and hyper_input.shape[0] % (2 * self.prefill_full_tile) == 0
            and hyper_input.dtype == torch.bfloat16
            and block_output.dtype == torch.bfloat16
        ):
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

    def combine_mix(self, block_output, residuals, next_hc):
        """``next_hc.mix(self.combine(block_output, residuals))``.

        After a fused decode mix, this combine runs as the prologue of
        ``next_hc``'s fused mix (one Pallas call; the combined hyper input is
        written once and returned as the next residual).
        """
        hyper_input, gate = residuals
        if (
            next_hc._decode_combine_mix_op is not None
            and gate is not None
            and gate.shape[-1] == self.hc_count
            and gate.dtype == torch.float32
            and block_output.dtype == torch.bfloat16
            and next_hc._decode_fused(hyper_input)
        ):
            mixed, injection, combined = next_hc._decode_combine_mix_op(
                hyper_input, block_output, gate, *next_hc._decode_weights()
            )
            return mixed, (combined, injection)
        return next_hc.mix(self.combine(block_output, residuals))


__all__ = [
    "GatedResidual",
    "GroupedGemmaRMSNorm",
    "HyperConnectionBase",
    "HyperConnectionConfig",
]
