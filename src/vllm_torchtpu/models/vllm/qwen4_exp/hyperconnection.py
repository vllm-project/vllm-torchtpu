# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-native [token, branch, hidden] HyperConnections.

Reuse the common checkpoint structure with a TPU-specific SiLU precision
boundary; the NVIDIA model owns CUDA kernels and is not imported here.
"""

import math
import os
from collections.abc import Callable

import torch
import torch.nn.functional as F
from vllm.models.qwen4_exp.common import hyperconnection as common_hc

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.qwen4_exp.hc_decode import decode_token_tile
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


def sparse_core_all_gather_offload() -> bool:
    """Whether LIBTPU_INIT_ARGS turns on the SparseCore all-gather offload
    (the last setting of the flag wins, as in libtpu)."""
    enabled = False
    for arg in os.environ.get("LIBTPU_INIT_ARGS", "").split():
        name, _, value = arg.partition("=")
        if name == "--xla_tpu_enable_sparse_core_collective_offload_all_gather":
            enabled = value.lower() in ("true", "1")
    return enabled


def prefill_full_variant() -> tuple[bool, bool]:
    """``(up_t, inject)`` of the prefill full-mix ops.

    Both variants are off while the SparseCore all-gather offload is on. With
    the offload (c32, >= 8 MiB, i.e. the sequence-parallel hidden gathers),
    either one corrupts served steps: the first non-finite hidden state is an
    SP 4096 step and spreads to every later row of the step; c32 GSM8K reads
    0.0114-0.583. With the offload off the same flags read
    0.9712 / 0.9719 (UP_T) and 0.9712 (UP_T + INJECT), and the kernels are
    bitwise equal to the default variant on TPU in isolation.
    """
    up_t = envs.VLLM_TPU_HC_PREFILL_UP_T
    inject = envs.VLLM_TPU_HC_PREFILL_INJECT
    if (up_t or inject) and sparse_core_all_gather_offload():
        logger.warning_once(
            "VLLM_TPU_HC_PREFILL_UP_T / VLLM_TPU_HC_PREFILL_INJECT are ignored "
            "while the SparseCore all-gather offload is on (corrupts outputs)."
        )
        return False, False
    return up_t, inject


def _same_tpu_bf16(reference: torch.Tensor, *tensors: torch.Tensor) -> bool:
    return reference.device.type == "tpu" and all(
        tensor.device == reference.device and tensor.dtype == torch.bfloat16
        for tensor in (reference, *tensors)
    )


class GatedResidual(common_hc.GatedResidual):
    """Match NVIDIA HC math with independent prefill and decode fusion choices.

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
        prefill_up_tile: int = 0,
        prefill_norm_tile: int = 0,
        prefill_full_tile: HcPrefillMode | None = None,
        prefill_combine_tile: HcPrefillMode | None = None,
    ) -> None:
        if prefill_up_tile and (prefill_up_tile < 128 or prefill_up_tile % 128):
            raise ValueError("HC prefill up tile must be a multiple of 128")
        if prefill_norm_tile and (prefill_norm_tile < 64 or prefill_norm_tile % 64):
            raise ValueError("HC prefill norm tile must be a multiple of 64")
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
        self.prefill_up_tile = prefill_up_tile
        self.prefill_norm_tile = prefill_norm_tile
        self.prefill_full_tile = prefill_full_tile
        self.prefill_combine_tile = prefill_combine_tile
        wide = envs.VLLM_TPU_HC_PREFILL_WIDE_TILE
        if wide and isinstance(prefill_full_tile, int) and prefill_full_tile:
            if wide % prefill_full_tile or wide <= prefill_full_tile:
                raise ValueError(
                    "VLLM_TPU_HC_PREFILL_WIDE_TILE must be a larger multiple of "
                    "hc_prefill_full_tile"
                )
            full_tiles: tuple[int, ...] = (wide, prefill_full_tile)
        else:
            full_tiles = (
                AUTO_MIX_TILES if prefill_full_tile == "auto" else (prefill_full_tile,)
            )
        self._combine_token_tile = (
            AUTO_COMBINE_TILE
            if prefill_combine_tile == "auto"
            else prefill_combine_tile
        )
        self._prefill_up_op = None
        self._prefill_norm_op = None
        self._prefill_full_ops: dict[int, Callable] = {}
        self._prefill_combine_full_ops: dict[int, Callable] = {}
        self._prefill_combine_op = None
        self._prefill_full_up_t = False
        self._prefill_inject = False
        self._decode_mix_op = None
        self._tpu_decode_mix_op = None
        self._decode_combine_mix_op = None

        if (
            self.use_mix
            and envs.TPU_QWEN4_HC_DECODE_FUSED
            and self.config.hc_per_branch_norm
            and self.hidden_size % 128 == 0
            and hasattr(self, "input_mix_weight_down")
        ):
            from vllm_torchtpu.kernels.qwen4_exp.vllm_adapter import (
                build_decode_combine_mix_op,
                build_decode_mix_op,
            )

            inject = self.use_combine and hasattr(self, "block_inject_weight")
            eps = self.hc_norm.variance_epsilon
            self._decode_mix_op = build_decode_mix_op(
                self.hc_count, self.hidden_size, eps, inject
            )
            self._tpu_decode_mix_op = self._decode_mix_op
            if inject:
                # Used by the preceding HC's combine_mix (attn -> mlp).
                self._decode_combine_mix_op = build_decode_combine_mix_op(
                    self.hc_count, self.hidden_size, eps
                )

        if self.use_mix and prefill_norm_tile and self.config.hc_per_branch_norm:
            from vllm_torchtpu.kernels.qwen4_exp.vllm_adapter import build_norm_op

            self._prefill_norm_op = build_norm_op(
                self.hc_count,
                self.hidden_size,
                self.hc_norm.variance_epsilon,
                prefill_norm_tile,
            )

        if self.use_mix and prefill_up_tile and hasattr(self, "input_mix_weight_up"):
            # Register before Dynamo traces forward; registration itself uses
            # Python inspect and cannot execute inside a compiled graph.
            from vllm_torchtpu.kernels.qwen4_exp.vllm_adapter import (
                build_up_epilogue_op,
            )

            self._prefill_up_op = build_up_epilogue_op(
                self.hc_count, self.hidden_size, prefill_up_tile
            )

        if not (self.use_mix or self.use_combine) or not (
            (self.use_mix and prefill_full_tile)
            or (self.use_combine and prefill_combine_tile)
        ):
            return

        from vllm_torchtpu.kernels.qwen4_exp.hc_resources import (
            supports_full_mix,
            supports_residual,
        )
        from vllm_torchtpu.kernels.qwen4_exp.vllm_adapter import (
            build_combine_full_mix_op,
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
                up_t, inject = prefill_full_variant()
                self._prefill_full_up_t = up_t
                self._prefill_inject = (
                    inject and self.use_combine and hasattr(self, "block_inject_weight")
                )
                lowrank = self.input_mix_weight_down.weight.shape[0]
                eps = self.hc_norm.variance_epsilon
                has_variant = self._prefill_full_up_t or self._prefill_inject
                self._prefill_full_ops = {
                    tile: (
                        build_full_mix_op(
                            self.hc_count,
                            self.hidden_size,
                            eps,
                            tile,
                            hardware,
                            up_t=self._prefill_full_up_t,
                            inject=self._prefill_inject,
                        )
                        if has_variant
                        else build_full_mix_op(
                            self.hc_count, self.hidden_size, eps, tile, hardware
                        )
                    )
                    for tile in full_tiles
                    if (
                        supports_full_mix(
                            self.hc_count,
                            self.hidden_size,
                            lowrank,
                            tile,
                            hardware,
                            up_t=self._prefill_full_up_t,
                            inject=self._prefill_inject,
                        )
                        if has_variant
                        else supports_full_mix(
                            self.hc_count,
                            self.hidden_size,
                            lowrank,
                            tile,
                            hardware,
                        )
                    )
                }
                if envs.VLLM_TPU_HC_PREFILL_COMBINE_MIX and self._prefill_full_ops:
                    self._prefill_combine_full_ops = {
                        tile: build_combine_full_mix_op(
                            self.hc_count,
                            self.hidden_size,
                            eps,
                            tile,
                            hardware,
                            up_t=self._prefill_full_up_t,
                            inject=self._prefill_inject,
                        )
                        for tile in full_tiles
                        if supports_full_mix(
                            self.hc_count,
                            self.hidden_size,
                            lowrank,
                            tile,
                            hardware,
                            up_t=self._prefill_full_up_t,
                            combine=True,
                            inject=self._prefill_inject,
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

    @property
    def sp_tile_alignment(self) -> int:
        """Row multiple keeping active prefill Pallas kernels on one SP trace."""
        tiles: list[int] = [self.prefill_up_tile, self.prefill_norm_tile]
        if self._prefill_full_ops:
            tiles.append(max(self._prefill_full_ops))
        elif isinstance(self.prefill_full_tile, int) and self.prefill_full_tile:
            tiles.append(self.prefill_full_tile)
        if self._prefill_combine_op is not None:
            tiles.append(self._combine_token_tile)
        elif isinstance(self.prefill_combine_tile, int) and self.prefill_combine_tile:
            tiles.append(self.prefill_combine_tile)
        return math.lcm(1, *(tile for tile in tiles if tile))

    def _normalize(self, hyper_input, small_rows=False):
        if not self.config.hc_per_branch_norm:
            return super()._normalize(hyper_input)
        if (
            self._prefill_norm_op is not None
            and not small_rows
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

    def disable_decode_fused(self):
        """Keep the unfused mix at every token count.

        For a module traced once for every token bucket (the MTP draft): the
        fused path is chosen by the token count while tracing, so that one
        graph could not serve buckets on the other side of the choice.
        """
        self._decode_mix_op = None
        self._tpu_decode_mix_op = None
        self._decode_combine_mix_op = None

    def disable_prefill_fused(self):
        """Disable token-count-dependent prefill HC Pallas ops on single graphs."""
        self._prefill_full_ops = {}
        self._prefill_combine_full_ops = {}
        self._prefill_combine_op = None
        self._prefill_up_op = None
        self._prefill_norm_op = None

    def _decode_weights(self):
        weights = (
            self.hc_norm.weight,
            self.input_mix_weight_down.weight,
            self.input_mix_weight_up.weight,
        )
        if self.use_combine and hasattr(self, "block_inject_weight"):
            weights += (self.block_inject_weight.weight,)
        return weights

    def _decode_fused(self, hyper_input) -> bool:
        """TPU_QWEN4_HC_DECODE_FUSED: one Pallas call for <= 512 BF16 rows."""
        if self._decode_mix_op is None or hyper_input.dim() != 2:
            return False
        if decode_token_tile(hyper_input.shape[0]) is None:
            return False
        if self._decode_mix_op is self._tpu_decode_mix_op:
            if self.prefill_full_tile not in ("auto", 0) and self._use_prefill_mix(
                hyper_input
            ):
                return False
            return _same_tpu_bf16(hyper_input, *self._decode_weights())
        return hyper_input.dtype == torch.bfloat16 and all(
            w.dtype == torch.bfloat16 for w in self._decode_weights()
        )

    def _use_prefill_mix(self, hyper_input):
        extra = (
            (self.block_inject_weight.weight,)
            if getattr(self, "_prefill_inject", False)
            else ()
        )
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
                *extra,
            )
            and select_mix_tile(hyper_input.shape[0], self._prefill_full_ops)
            is not None
        )

    def _residual(self, hyper_input, norm):
        """``mix``'s residual for ``combine``, with the injection gate computed
        here as the fused decode mix computes it.

        This keeps ``block_inject_weight``'s first use inside ``mix`` on every
        path. Piecewise compilation orders a subgraph's inputs by first use, so
        a fused decode bucket and an unfused/prefill bucket of the same model
        then take the same inputs in the same order; the shape-variant hand-over
        refuses a bucket whose inputs moved (h21 j-b1210ac2).
        """
        if not self.use_combine or not hasattr(self, "block_inject_weight"):
            return hyper_input, norm
        injection = F.linear(norm, self.block_inject_weight.weight)
        return hyper_input, 2.0 * torch.sigmoid(injection.float() / self.hc_count)

    def _prefill_weights(self):
        weights = (
            self.hc_norm.weight,
            self.input_mix_weight_down.weight,
            # VLLM_TPU_HC_PREFILL_UP_T transposes inside the op.
            self.input_mix_weight_up.weight,
        )
        if getattr(self, "_prefill_inject", False):
            weights += (self.block_inject_weight.weight,)
        return weights

    def _prefill_residual(self, hyper_input, second):
        """``mix``'s residual pair from a full-mix kernel's second output."""
        if getattr(self, "_prefill_inject", False):
            # The kernel's BF16 gate projection; the gate as ``_residual``.
            return hyper_input, 2.0 * torch.sigmoid(second.float() / self.hc_count)
        return self._residual(hyper_input, second)

    def mix(self, hyper_input, small_rows=False):
        """Returns ``mixed, (hyper_input, injection)``: the combine's [T, hc] FP32
        injection gate, computed from the normalized input (``normed`` in its
        place for a hyper-connection without a combine); ``combine`` accepts
        either form.

        ``small_rows``: the rows are megakernel verify tokens (the segment and
        whole-step programs' stock mixes), so the prefill-tile Pallas ops are
        skipped even when the row count matches a tile (128 = 32 requests x 4
        verify rows); the result is the small-row path the kernel itself
        computes."""
        assert hyper_input.shape[-1] == self.hc_count * self.hidden_size
        if not self.use_mix:
            raise RuntimeError("mix was disabled for this hyper-connection")
        if self._decode_fused(hyper_input):
            assert self._decode_mix_op is not None
            out = self._decode_mix_op(hyper_input, *self._decode_weights())
            if self.use_combine and hasattr(self, "block_inject_weight"):
                mixed, injection = out
                return mixed, (hyper_input, injection)
            return out, (hyper_input, None)
        if not small_rows and self._use_prefill_mix(hyper_input):
            tile = select_mix_tile(hyper_input.shape[0], self._prefill_full_ops)
            assert tile is not None
            if getattr(self, "_prefill_inject", False):
                mixed, second = self._prefill_full_ops[tile](
                    hyper_input, *self._prefill_weights()
                )
                return mixed, self._prefill_residual(hyper_input, second)
            mixed, norm = self._prefill_full_ops[tile](
                hyper_input,
                self.hc_norm.weight,
                self.input_mix_weight_down.weight,
                self.input_mix_weight_up.weight,
            )
            return mixed, self._residual(hyper_input, norm)
        norm = self._normalize(hyper_input, small_rows)
        gate = F.linear(norm, self.input_mix_weight_down.weight)
        # TorchTPU can lower BF16 SiLU through BF16 logistic/tanh, rounding
        # negative tails to zero. Compute the activation in FP32 and round
        # once, matching NVIDIA hc_silu's BF16 result boundary.
        gate = F.silu(gate.float() / self.hc_count).to(gate.dtype)
        if (
            self._prefill_up_op is not None
            and not small_rows
            and gate.shape[0] >= self.prefill_up_tile
            and gate.shape[0] % self.prefill_up_tile == 0
            and gate.dtype == torch.bfloat16
            and norm.dtype == torch.bfloat16
            and self.input_mix_weight_up.weight.dtype == torch.bfloat16
        ):
            mixed = self._prefill_up_op(gate, norm, self.input_mix_weight_up.weight)
            return mixed, self._residual(hyper_input, norm)
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
        return mixed.to(hyper_input.dtype), self._residual(hyper_input, norm)

    def _injection(self, residuals):
        """The combine's [T, hc] FP32 gate from ``mix``'s residual pair."""
        _, hyper_input_normed = residuals
        if hyper_input_normed.shape[-1] == self.hc_count:
            # mix() already computed the [T, hc] FP32 injection gate.
            return hyper_input_normed
        injection = F.linear(hyper_input_normed, self.block_inject_weight.weight)
        return 2.0 * torch.sigmoid(injection.float() / self.hc_count)

    def _use_prefill_combine(self, block_output, hyper_input, hyper_input_normed):
        return (
            self.use_combine
            and self._prefill_combine_op is not None
            and hyper_input.ndim == 2
            and block_output.ndim == 2
            and hyper_input_normed is not None
            and hyper_input_normed.ndim == 2
            and block_output.shape[0] == hyper_input.shape[0]
            and hyper_input_normed.shape[0] == hyper_input.shape[0]
            and (
                self.prefill_combine_tile != "auto"
                or hyper_input.shape[0] >= AUTO_MIN_TOKENS
            )
            and hyper_input.shape[0] >= self._combine_token_tile
            and hyper_input.shape[0] % self._combine_token_tile == 0
            and _same_tpu_bf16(
                hyper_input,
                block_output,
                self.block_inject_weight.weight,
            )
            and hyper_input_normed.device == hyper_input.device
            and (
                (
                    hyper_input_normed.shape[1] == self.hc_count
                    and hyper_input_normed.dtype == torch.float32
                )
                or (
                    hyper_input_normed.shape[1] == hyper_input.shape[1]
                    and hyper_input_normed.dtype == torch.bfloat16
                )
            )
        )

    def combine(self, block_output, residuals):
        if not self.use_combine:
            raise RuntimeError("combine was disabled for this hyper-connection")
        hyper_input, hyper_input_normed = residuals
        assert hyper_input.shape[-1] == self.hc_count * self.hidden_size
        assert block_output.shape[-1] == self.hidden_size

        injection = self._injection(residuals)
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

    def combine_mix(self, block_output, residuals, next_hc):
        """``next_hc.mix(self.combine(block_output, residuals))``.

        After a fused decode mix, this combine runs as the prologue of
        ``next_hc``'s fused mix (one Pallas call; the combined hyper input is
        written once and returned as the next residual).
        """
        hyper_input, gate = residuals
        if (
            self.use_combine
            and next_hc._decode_combine_mix_op is not None
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
        if (
            self.use_combine
            and getattr(next_hc, "_prefill_combine_full_ops", None)
            and self._prefill_combine_op is not None
            and next_hc._use_prefill_mix(hyper_input)
            and self._use_prefill_combine(block_output, hyper_input, gate)
        ):
            tile = select_mix_tile(
                hyper_input.shape[0], next_hc._prefill_combine_full_ops
            )
            if tile is not None:
                mixed, second, combined = next_hc._prefill_combine_full_ops[tile](
                    hyper_input,
                    block_output,
                    self._injection(residuals),
                    *next_hc._prefill_weights(),
                )
                return mixed, next_hc._prefill_residual(combined, second)
        return next_hc.mix(self.combine(block_output, residuals))


__all__ = [
    "GatedResidual",
    "GroupedGemmaRMSNorm",
    "HyperConnectionBase",
    "HyperConnectionConfig",
    "prefill_full_variant",
    "sparse_core_all_gather_offload",
]
