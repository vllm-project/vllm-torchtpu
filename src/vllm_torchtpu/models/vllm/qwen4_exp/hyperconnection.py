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

import itertools
from collections.abc import Callable
from typing import NamedTuple

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

_HC_INSTANCE_IDS = itertools.count()


class HcPackedMixState(NamedTuple):
    """Carrier for a mix whose injection gates were already projected.

    ``owner_id`` binds the logits to the hyper-connection that produced them:
    every instance owns a distinct token, and ``combine`` refuses a state
    projected with another owner's injection weight. The public mix API keeps
    returning the ``(hyper_input, norm)`` tuple; only callers that opt into
    the packed prefill path receive this state.

    ``broadcast`` marks the chain's first mix, whose ``hyper_input`` is the
    un-expanded ``[tokens, hidden]`` residual: every branch slice of the
    expanded plane would repeat it, so the fused paths stage it once and
    expand it in kernel. Every other consumer — the public ``combine`` and any
    unfused boundary — expands it first and then follows exactly the same
    arithmetic over the genuine expanded residual.
    """

    hyper_input: torch.Tensor
    inject_logits: torch.Tensor
    owner_id: int
    broadcast: bool = False


def _same_tpu_bf16(reference: torch.Tensor, *tensors: torch.Tensor) -> bool:
    return reference.device.type == "tpu" and all(
        tensor.device == reference.device and tensor.dtype == torch.bfloat16
        for tensor in (reference, *tensors)
    )


def _mix_weights(owner) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        owner.hc_norm.weight,
        owner.input_mix_weight_down.weight,
        owner.input_mix_weight_up.weight,
    )


def _prior_combine_matches(prior, tile: int | None) -> bool:
    return (
        prior.use_combine
        and prior.prefill_combine_tile != 0
        and (prior.prefill_combine_tile == "auto" or prior._combine_token_tile == tile)
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
        self._hc_owner_id = next(_HC_INSTANCE_IDS)
        full_tiles = (
            AUTO_MIX_TILES if prefill_full_tile == "auto" else (prefill_full_tile,)
        )
        self._combine_token_tile = (
            AUTO_COMBINE_TILE
            if prefill_combine_tile == "auto"
            else prefill_combine_tile
        )
        self._prefill_full_ops: dict[int, Callable] = {}
        self._prefill_inject_ops: dict[int, Callable] = {}
        self._prefill_inject_broadcast_ops: dict[int, Callable] = {}
        self._prefill_boundary_ops: dict[int, Callable] = {}
        self._prefill_boundary_broadcast_ops: dict[int, Callable] = {}
        self._prefill_terminal_ops: dict[int, Callable] = {}
        self._prefill_combine_op = None
        if not (use_mix or use_combine) or not (
            prefill_full_tile or prefill_combine_tile
        ):
            return
        from vllm_torchtpu.kernels.qwen4_exp.hc_resources import (
            supports_combine_broadcast_mix,
            supports_combine_mix,
            supports_full_mix,
            supports_inject_broadcast_mix,
            supports_inject_mix,
            supports_residual,
            supports_terminal_mix,
        )
        from vllm_torchtpu.kernels.qwen4_exp.vllm_adapter import (
            build_combine_mix_broadcast_op,
            build_combine_mix_op,
            build_combine_mix_terminal_op,
            build_full_mix_inject_broadcast_op,
            build_full_mix_inject_op,
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

                def mix_ops(
                    builder: Callable, supports: Callable
                ) -> dict[int, Callable]:
                    return {
                        tile: builder(
                            self.hc_count,
                            self.hidden_size,
                            self.hc_norm.variance_epsilon,
                            tile,
                            hardware,
                        )
                        for tile in full_tiles
                        if supports(
                            self.hc_count,
                            self.hidden_size,
                            self.input_mix_weight_down.weight.shape[0],
                            tile,
                            hardware,
                        )
                    }

                self._prefill_full_ops = mix_ops(build_full_mix_op, supports_full_mix)
                if (
                    self.use_combine
                    and envs.VLLM_TPU_HC_PACKED_MIX
                    and self._prefill_full_ops
                ):
                    # Same tiles, resources, and epsilon as the full-mix ops;
                    # the inject projection shares the staged normalized value
                    # so the full-width norm tensor is never materialized.
                    self._prefill_inject_ops = mix_ops(
                        build_full_mix_inject_op, supports_inject_mix
                    )
                    # The broadcast variant serves the chain's first mix, whose
                    # residual arrives un-expanded; same tiles, weights, and
                    # epsilon, with its own resource facts for the staging
                    # planes that replace the full-width input ring.
                    self._prefill_inject_broadcast_ops = mix_ops(
                        build_full_mix_inject_broadcast_op,
                        supports_inject_broadcast_mix,
                    )
                    # Boundary ops fuse a preceding owner's combine into this
                    # owner's inject mix; the prior owner contributes only the
                    # gate logits at call time, so the same resource facts and
                    # this owner's weights define the admission here.
                    self._prefill_boundary_ops = mix_ops(
                        build_combine_mix_op, supports_combine_mix
                    )
                    # The broadcast boundary variant serves the chain's first
                    # boundary, whose prior owner's residual arrives
                    # un-expanded; this owner's weights and resource facts
                    # define the admission exactly like the boundary above.
                    self._prefill_boundary_broadcast_ops = mix_ops(
                        build_combine_mix_broadcast_op, supports_combine_broadcast_mix
                    )
                elif (
                    not self.use_combine
                    and envs.VLLM_TPU_HC_PACKED_MIX
                    and self._prefill_full_ops
                ):
                    # A mix-only owner is a terminal consumer: the preceding
                    # owner's combine produces the combined value and this
                    # owner's mix is that value's only reader, because no
                    # combine of its own exists to carry the residual further.
                    # The fused variant forms the combined value in vector
                    # memory and never streams it through HBM. The prior owner
                    # contributes only its gate logits at call time, so this
                    # owner's weights define the admission here.
                    self._prefill_terminal_ops = mix_ops(
                        build_combine_mix_terminal_op, supports_terminal_mix
                    )
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
            elif self.use_combine and not self._prefill_inject_ops:
                logger.warning_once(
                    "HC packed mix uses the norm-carrier path: %s "
                    "(branches=%s, hidden=%s, lowrank=%s, tiles=%s, hardware=%s)",
                    "disabled by configuration"
                    if not envs.VLLM_TPU_HC_PACKED_MIX
                    else "unsupported resources/layout",
                    self.hc_count,
                    self.hidden_size,
                    self.input_mix_weight_down.weight.shape[0],
                    full_tiles,
                    hardware,
                )
            elif not self.use_combine and not self._prefill_terminal_ops:
                logger.warning_once(
                    "HC terminal mix uses the unfused sequence: %s "
                    "(branches=%s, hidden=%s, lowrank=%s, tiles=%s, hardware=%s)",
                    "disabled by configuration"
                    if not envs.VLLM_TPU_HC_PACKED_MIX
                    else "unsupported resources/layout",
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
                *_mix_weights(self),
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
                *_mix_weights(self),
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

    def _use_packed_mix(self, hyper_input):
        # The staged tile is this kernel's unit of work, so the runtime token
        # count itself carries the distribution: an op admitted for the actual
        # geometry covers the call exactly when the count is whole tiles
        # (select_mix_tile below). No separate token floor applies here, unlike
        # the original full-mix/combine profitability heuristic.
        return (
            self.use_mix
            and self.use_combine
            and bool(self._prefill_inject_ops)
            and hyper_input.ndim == 2
            and _same_tpu_bf16(
                hyper_input,
                *_mix_weights(self),
                self.block_inject_weight.weight,
            )
            and select_mix_tile(hyper_input.shape[0], self._prefill_inject_ops)
            is not None
        )

    def mix_packed(self, hyper_input):
        """Mix while projecting this owner's injection gates in the same kernel.

        The normalized hyper input stays resident in the kernel: the injection
        logits are computed from it and returned in an ``HcPackedMixState``
        instead of materializing the full-width norm tensor. ``combine``
        accepts that state directly. Outside the admitted prefill layout this
        returns exactly the public ``mix`` result, so callers can use one
        entry point for every distribution.
        """
        assert hyper_input.shape[-1] == self.hc_count * self.hidden_size
        if not self.use_mix:
            raise RuntimeError("mix was disabled for this hyper-connection")
        if self._use_packed_mix(hyper_input):
            tile = select_mix_tile(hyper_input.shape[0], self._prefill_inject_ops)
            assert tile is not None
            mixed, logits = self._prefill_inject_ops[tile](
                hyper_input,
                *_mix_weights(self),
                self.block_inject_weight.weight,
            )
            return mixed, HcPackedMixState(hyper_input, logits, self._hc_owner_id)
        return self.mix(hyper_input)

    def _use_broadcast_mix(self, embedding):
        # The broadcast kernel reads one [tokens, hidden] tile where the
        # expanded plane would carry hc_count copies of it; the runtime token
        # count carries the distribution exactly like the packed mix, so the
        # same tile policy over the broadcast registry decides. The state this
        # returns feeds this owner's combine, so the owner must keep one.
        return (
            self.use_mix
            and self.use_combine
            and bool(self._prefill_inject_broadcast_ops)
            and embedding.ndim == 2
            and _same_tpu_bf16(
                embedding,
                *_mix_weights(self),
                self.block_inject_weight.weight,
            )
            and select_mix_tile(embedding.shape[0], self._prefill_inject_broadcast_ops)
            is not None
        )

    def mix_packed_broadcast(self, embedding):
        """First mix over an un-expanded residual broadcast across branches.

        Every branch slice of the expanded hyper input would repeat this
        ``[tokens, hidden]`` value, so the admitted kernel stages it once and
        expands it in kernel; the expanded plane never reaches HBM, and the
        returned state carries the un-expanded residual with ``broadcast``
        set so every consumer expands it exactly where its arithmetic needs
        the expanded words. Outside the admitted layout this materializes the
        expansion and returns exactly the public ``mix_packed`` result, so
        callers can use one entry point for every distribution.
        """
        assert embedding.shape[-1] == self.hidden_size
        if not self.use_mix:
            raise RuntimeError("mix was disabled for this hyper-connection")
        if self._use_broadcast_mix(embedding):
            tile = select_mix_tile(
                embedding.shape[0], self._prefill_inject_broadcast_ops
            )
            assert tile is not None
            mixed, logits = self._prefill_inject_broadcast_ops[tile](
                embedding,
                *_mix_weights(self),
                self.block_inject_weight.weight,
            )
            return mixed, HcPackedMixState(
                embedding, logits, self._hc_owner_id, broadcast=True
            )
        return self.mix_packed(embedding.repeat(1, self.hc_count))

    def _use_prefill_combine(self, block_output, hyper_input, gate_operand):
        # gate_operand is the normalized hyper input on the public path and
        # the packed injection logits on the packed path; both gate sources
        # share the row, rank, device, and dtype contract.
        return (
            self.use_combine
            and self._prefill_combine_op is not None
            and hyper_input.ndim == 2
            and block_output.ndim == 2
            and gate_operand.ndim == 2
            and block_output.shape[0] == hyper_input.shape[0]
            and gate_operand.shape[0] == hyper_input.shape[0]
            and hyper_input.shape[0] >= self._combine_token_tile
            and hyper_input.shape[0] % self._combine_token_tile == 0
            and (
                self.prefill_combine_tile != "auto"
                or hyper_input.shape[0] >= AUTO_MIN_TOKENS
            )
            and _same_tpu_bf16(
                hyper_input,
                block_output,
                gate_operand,
                self.block_inject_weight.weight,
            )
        )

    def combine(self, block_output, residuals):
        if not self.use_combine:
            raise RuntimeError("combine was disabled for this hyper-connection")
        if isinstance(residuals, HcPackedMixState):
            hyper_input = residuals.hyper_input
            if residuals.owner_id != self._hc_owner_id:
                raise RuntimeError(
                    "combine received a packed mix state projected by another "
                    "hyper-connection"
                )
            gate_operand = residuals.inject_logits
            if gate_operand.shape != hyper_input.shape[:-1] + (self.hc_count,):
                raise RuntimeError(
                    "combine received packed mix logits whose shape does not "
                    "match the hyper input rows and branch count"
                )
            if residuals.broadcast:
                # The state carries the un-expanded first residual: expand it
                # here so the combine runs over the genuine expanded words the
                # unfused caller would have passed.
                hyper_input = hyper_input.repeat(1, self.hc_count)
            # The logits already carry this owner's injection weight with the
            # BF16 projection boundary; only the FP32 gate remains.
            injection = 2.0 * torch.sigmoid(
                residuals.inject_logits.float() / self.hc_count
            )
        else:
            hyper_input, gate_operand = residuals
            injection = F.linear(gate_operand, self.block_inject_weight.weight)
            injection = 2.0 * torch.sigmoid(injection.float() / self.hc_count)
        assert hyper_input.shape[-1] == self.hc_count * self.hidden_size
        assert block_output.shape[-1] == self.hidden_size

        if self._use_prefill_combine(block_output, hyper_input, gate_operand):
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


def _boundary_ops(residuals, following):
    """The boundary registry a prior owner's packed state admits.

    A broadcast state carries the prior owner's un-expanded first residual, so
    only the broadcast kernel can stage it; every other state feeds the
    full-width boundary kernel.
    """
    if residuals.broadcast:
        return following._prefill_boundary_broadcast_ops
    return following._prefill_boundary_ops


def _use_boundary_mix(prior, following, block_output, residuals):
    hyper_input = residuals.hyper_input
    tile = select_mix_tile(hyper_input.shape[0], _boundary_ops(residuals, following))
    # The fused kernel embeds the prior owner's combine, so that owner's
    # explicit component override stays in force: a combine component disabled
    # with tile 0, or a forced positive combine tile different from the fused
    # tile, keeps the unfused public sequence. Auto mode leaves the combine
    # tile to the boundary admission itself.
    return (
        _prior_combine_matches(prior, tile)
        and following.use_mix
        and following.use_combine
        and bool(_boundary_ops(residuals, following))
        and prior.hc_count == following.hc_count
        and prior.hidden_size == following.hidden_size
        and hyper_input.ndim == 2
        and (not residuals.broadcast or hyper_input.shape[-1] == prior.hidden_size)
        and block_output.ndim == 2
        and block_output.shape[0] == hyper_input.shape[0]
        and residuals.inject_logits.shape
        == hyper_input.shape[:-1] + (following.hc_count,)
        and _same_tpu_bf16(
            hyper_input,
            block_output,
            residuals.inject_logits,
            *_mix_weights(following),
            following.block_inject_weight.weight,
        )
        and tile is not None
    )


def combine_then_mix_packed(prior, following, block_output, residuals):
    """Fuse ``prior``'s combine with ``following``'s packed mix.

    The boundary kernel stages the prior owner's residual tile once, forms the
    combined hyper input in vector memory, streams that single tile to HBM for
    the following owner's own ``combine``, and consumes the same value for the
    following mix's normalization and projections. The prior owner's injection
    gates are formed in kernel from its packed logits, so the eager sigmoid
    chain and the separate combine kernel of the unfused boundary disappear.
    A broadcast state stages the prior owner's un-expanded first residual
    instead of the expanded plane; every branch's combined value is formed from
    the same words the expanded plane would carry. Outside the admitted layout
    this returns exactly the unfused sequence, so every caller distribution
    keeps its public contract.

    Args:
        prior: The hyper-connection whose packed state ``residuals`` carries.
        following: The hyper-connection that mixes the combined value.
        block_output: The prior owner's block output.
        residuals: The prior owner's ``mix_packed`` result.

    Returns:
        (combined, mixed, packed state of ``following``); the tuple carrier
        appears unchanged in the fallback.
    """
    if (
        isinstance(residuals, HcPackedMixState)
        and residuals.owner_id == prior._hc_owner_id
        and _use_boundary_mix(prior, following, block_output, residuals)
    ):
        tile = select_mix_tile(
            residuals.hyper_input.shape[0], _boundary_ops(residuals, following)
        )
        assert tile is not None
        combined, mixed, logits = _boundary_ops(residuals, following)[tile](
            residuals.hyper_input,
            block_output,
            residuals.inject_logits,
            *_mix_weights(following),
            following.block_inject_weight.weight,
        )
        return (
            combined,
            mixed,
            HcPackedMixState(combined, logits, following._hc_owner_id),
        )
    combined = prior.combine(block_output, residuals)
    mixed, state = following.mix_packed(combined)
    return combined, mixed, state


def _use_terminal_mix(prior, terminal, block_output, residuals):
    hyper_input = residuals.hyper_input
    tile = select_mix_tile(hyper_input.shape[0], terminal._prefill_terminal_ops)
    # The fused kernel embeds the prior owner's combine, so that owner's
    # explicit component override stays in force exactly like the boundary
    # fusion: a combine component disabled with tile 0, or a forced positive
    # combine tile different from the fused tile, keeps the unfused public
    # sequence. The terminal owner must be mix-only, so no combine of its own
    # competes for the combined value. The terminal kernel stages the prior
    # owner's full-width residual, so a broadcast state keeps the public
    # sequence: the combine below expands the un-expanded residual first.
    return (
        _prior_combine_matches(prior, tile)
        and terminal.use_mix
        and not terminal.use_combine
        and bool(terminal._prefill_terminal_ops)
        and not residuals.broadcast
        and prior.hc_count == terminal.hc_count
        and prior.hidden_size == terminal.hidden_size
        and hyper_input.ndim == 2
        and block_output.ndim == 2
        and block_output.shape[0] == hyper_input.shape[0]
        and residuals.inject_logits.shape
        == hyper_input.shape[:-1] + (terminal.hc_count,)
        and _same_tpu_bf16(
            hyper_input,
            block_output,
            residuals.inject_logits,
            *_mix_weights(terminal),
        )
        and tile is not None
    )


def combine_then_mix_terminal(prior, terminal, block_output, residuals):
    """Fuse ``prior``'s final combine with ``terminal``'s mix-only mix.

    ``terminal`` must be a mix-only hyper-connection (``use_combine=False``)
    whose ``mix`` is the only consumer of the combined value: the caller owns
    that proof, because this entry never materializes the combined tensor.
    The kernel stages the prior owner's residual tile, forms the combined
    hyper input in place with the same one-BF16-rounding boundary the public
    ``combine`` stores, and consumes it directly for the terminal mix's
    normalization and projections, so no combined stream reaches HBM. Outside
    the admitted layout this returns exactly ``terminal.mix(prior.combine(
    block_output, residuals))[0]``, the unfused public sequence over the
    genuine combined value, so every caller distribution keeps the public
    contract.

    Args:
        prior: The combine-owning hyper-connection whose packed state
            ``residuals`` carries.
        terminal: The mix-only hyper-connection consuming the combined value.
        block_output: The prior owner's block output.
        residuals: The prior owner's ``mix_packed`` result.

    Returns:
        The terminal mix output (the ``mixed`` element of ``terminal.mix``'s
        public tuple).
    """
    if (
        isinstance(residuals, HcPackedMixState)
        and residuals.owner_id == prior._hc_owner_id
        and _use_terminal_mix(prior, terminal, block_output, residuals)
    ):
        tile = select_mix_tile(
            residuals.hyper_input.shape[0], terminal._prefill_terminal_ops
        )
        assert tile is not None
        return terminal._prefill_terminal_ops[tile](
            residuals.hyper_input,
            block_output,
            residuals.inject_logits,
            *_mix_weights(terminal),
        )
    combined = prior.combine(block_output, residuals)
    return terminal.mix(combined)[0]


__all__ = [
    "GatedResidual",
    "GroupedGemmaRMSNorm",
    "HcPackedMixState",
    "HyperConnectionBase",
    "HyperConnectionConfig",
    "combine_then_mix_packed",
    "combine_then_mix_terminal",
]
