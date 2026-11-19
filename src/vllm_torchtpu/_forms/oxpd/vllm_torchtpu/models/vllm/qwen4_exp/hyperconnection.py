# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-native [token, branch, hidden] HyperConnections.

Reuse the common checkpoint structure with a TPU-specific SiLU precision
boundary; the NVIDIA model owns CUDA kernels and is not imported here.
"""

import torch
import torch.nn.functional as F
from vllm.models.qwen4_exp.common import hyperconnection as common_hc

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.qwen4_exp.hc_decode import (
    decode_fused_allows,
    decode_fused_max_rows,
    decode_tile_rows,
    decode_token_tile,
    decode_xla_applies,
    decode_xla_max_rows,
)
from vllm_torchtpu.models.vllm.qwen4_exp import hc_engagement

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
        # Fused prefill mix and combine take T % tile == 0 (any count >= 2)
        # under TPU_QWEN4_HC_PREFILL_ODD_TILES or ..._FUSED_MIX, else even
        # tile counts only (the served default).
        self._prefill_tile_step = (
            prefill_full_tile
            if envs.TPU_QWEN4_HC_PREFILL_ODD_TILES
            or envs.TPU_QWEN4_HC_PREFILL_FUSED_MIX
            else 2 * prefill_full_tile
        )
        # TPU_QWEN4_HC_PREFILL_PARTIAL_TILE: any multiple of 32 rows (at least
        # two tiles) runs whole tiles plus one partial last tile.
        self._prefill_partial = bool(
            envs.TPU_QWEN4_HC_PREFILL_PARTIAL_TILE and prefill_full_tile
        )
        partial = self._prefill_partial
        norm_impl = envs.TPU_QWEN4_HC_PREFILL_NORM.lower()
        self._prefill_inj_mix_op = None
        self._prefill_combine_mix_op = None
        self._prefill_combine_mix_rr_op = None
        if (
            envs.TPU_QWEN4_HC_PREFILL_FUSED_MIX
            and prefill_full_tile
            and self.config.hc_per_branch_norm
            and self.hidden_size % 128 == 0
            and hasattr(self, "input_mix_weight_down")
            and hasattr(self, "block_inject_weight")
        ):
            from vllm_torchtpu.kernels.qwen4_exp.vllm_adapter import (
                build_prefill_combine_mix_op,
                build_prefill_inj_mix_op,
            )

            eps = self.hc_norm.variance_epsilon
            fold = bool(envs.TPU_QWEN4_HC_PREFILL_FOLD)
            self._prefill_inj_mix_op = build_prefill_inj_mix_op(
                self.hc_count, self.hidden_size, eps, prefill_full_tile, partial, norm_impl, fold
            )
            # Used by the preceding HC's combine_mix (attn -> mlp).
            self._prefill_combine_mix_op = build_prefill_combine_mix_op(
                self.hc_count, self.hidden_size, eps, prefill_full_tile, partial, norm_impl, fold
            )
            if envs.TPU_QWEN4_HC_PREFILL_DEFER_WHOLE and not fold:
                # The deferred boundary's prologue (mlp -> next attn) reads its
                # combined rows back (the fold kernel already does).
                self._prefill_combine_mix_rr_op = build_prefill_combine_mix_op(
                    self.hc_count, self.hidden_size, eps, prefill_full_tile, partial, norm_impl,
                    fold, reread=True,
                )
        if (
            prefill_full_tile
            and self.hidden_size % 128 == 0
            and hasattr(self, "block_inject_weight")
        ):
            from vllm_torchtpu.kernels.qwen4_exp.vllm_adapter import (
                build_gated_residual_op,
            )

            self._prefill_combine_op = build_gated_residual_op(
                self.hc_count, self.hidden_size, prefill_full_tile, partial
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
                partial,
                norm_impl,
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
        self._decode_up_t = False
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
            # TPU_QWEN4_HC_DECODE_UP_T: the kernel reads the up weight as
            # up.T, a bitcast of its served {0,1} layout, instead of paying a
            # 13 MB relayout copy per call. Same inputs, same graph order.
            up_t = bool(envs.TPU_QWEN4_HC_DECODE_UP_T)
            self._decode_up_t = up_t
            # q38 PORT-BL HC: TPU_QWEN4_HC_DECODE_TILE_ROWS (default 0) only changes the
            # kernel's row tile; passed as a keyword only when set, so an unset key builds
            # exactly today's ops (same arguments, same names).
            tile_kw = {}
            max_tile = decode_tile_rows(envs.TPU_QWEN4_HC_DECODE_TILE_ROWS)
            if max_tile:
                tile_kw["max_tile"] = max_tile
            self._decode_mix_op = build_decode_mix_op(
                self.hc_count, self.hidden_size, eps, inject, up_t, **tile_kw
            )
            # VLLM_TPU_HC_DECODE_COUNT (log-only): this module runs the fused mix.
            hc_engagement.record_built(up_t)
            if inject:
                # Used by the preceding HC's combine_mix (attn -> mlp); with
                # TPU_QWEN4_HC_DECODE_FOLD also mlp -> the next layer's attn HC.
                self._decode_combine_mix_op = build_decode_combine_mix_op(
                    self.hc_count, self.hidden_size, eps, up_t, **tile_kw
                )
            elif envs.TPU_QWEN4_HC_DECODE_FOLD:
                # q38 PORT-BL HC (TPU_QWEN4_HC_DECODE_FOLD, ported from Benliang Wang
                # 28419c4a5): the final mixer (no injection gate) takes the last layer's
                # mlp combine as the prologue of its fused decode mix. Same up_t as the
                # mix op above.
                self._decode_combine_mix_op = build_decode_combine_mix_op(
                    self.hc_count, self.hidden_size, eps, up_t, inject=False, **tile_kw
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
        # VLLM_TPU_HC_XLA_MAX_ROWS (HCX, default 0 = off), read once at build so a bad
        # value fails here and tracing never reads the environment (see _hcx_xla).
        self._hcx_max_rows = decode_xla_max_rows()
        # VLLM_TPU_HC_DECODE_FUSED_MAX_ROWS (HCGATE, default 0 = off): the fused decode
        # kernel only for <= N rows. Read once at build, like HCX, so a bad value fails
        # here and tracing never reads the environment (see _decode_fused).
        self._hcf_max_rows = decode_fused_max_rows()

    def _normalize(self, hyper_input, xla=False):
        """``xla``: HCX routed this mix to the XLA path; skip the norm tile op."""
        if not self.config.hc_per_branch_norm:
            return super()._normalize(hyper_input)
        if (
            not xla
            and self._prefill_norm_op is not None
            # q38 L3-HCGATE (T95u generalization): divisibility alone (a positive multiple of
            # the tile is >= the tile); a `>=` test would put a lower bound on the traced token
            # count that can meet another module's upper bound (O1s slots x (k + 1)) and pin it.
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
        if self._decode_mix_op is not None:
            hc_engagement.record_disabled(self._decode_up_t)
        self._decode_mix_op = None
        self._decode_combine_mix_op = None

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
        """TPU_QWEN4_HC_DECODE_FUSED: one Pallas call for <= 512 BF16 rows.

        HCGATE (VLLM_TPU_HC_DECODE_FUSED_MAX_ROWS = N): only for <= N rows; a larger
        input takes the unfused path exactly as with the fused ops unbuilt. The row
        limit is compared after the armed test (an unarmed module, e.g. the MTP draft
        after disable_decode_fused, never reads the row count) and, when unset, not
        at all (decode_fused_allows)."""
        return (
            self._decode_mix_op is not None
            and hyper_input.dim() == 2
            and decode_fused_allows(hyper_input.shape[0], self._hcf_max_rows)
            and decode_token_tile(hyper_input.shape[0]) is not None
            and hyper_input.dtype == torch.bfloat16
            and all(w.dtype == torch.bfloat16 for w in self._decode_weights())
        )

    def _decode_rows(self, tokens: int) -> bool:
        """Whether this module's fused decode path covers ``tokens`` rows
        (TPU_QWEN4_HC_DECODE_FUSED armed and a decode tile exists).

        ``combine`` asks this before the prefill-combine row test. Both read
        the token count while tracing: the decode tile guards ``tokens <=
        512`` and the prefill test ``tokens >= 2 * prefill_full_tile`` (512
        with the served tile of 256), so evaluating both at 512 refines the
        token symbol to the single value 512 and Dynamo specializes it. That
        trace's graph then drops the input_ids / valid_tokens SymInt inputs
        while every smaller bucket's graph keeps them, and the shape-variant
        hand-over refuses the 512 trace (SMOKE-S1, HCF Job 2). Unarmed, this
        returns False on an attribute test without reading the token count.

        HCGATE: rows above VLLM_TPU_HC_DECODE_FUSED_MAX_ROWS are not covered, so their
        combine is the unfused module's combine (same row tests, same ops). The limit
        is compared before the decode tile, so a row count above it never evaluates
        the tile's ``tokens <= 512`` guard (the 512 pin above).
        """
        return (
            self._decode_mix_op is not None
            and decode_fused_allows(tokens, self._hcf_max_rows)
            and decode_token_tile(tokens) is not None
        )

    def decode_fold_rows(self, tokens: int, dtype) -> bool:
        """TPU_QWEN4_HC_DECODE_FOLD (q38 PORT-BL HC): whether a previous HC's combine
        handed to this HC at ``tokens`` rows of ``dtype`` streams runs as the prologue of
        this HC's fused decode mix, i.e. ``combine_mix``'s decode branch (a combine-mix op
        is built, and ``_decode_fused`` holds for the combined streams).

        The backbone loop asks this before it defers a layer's mlp combine, so a row count
        this does not cover is never deferred: its trace is the unfolded one, op for op.
        Guard neutrality: the row count is compared only through ``_decode_rows``, i.e. the
        HCGATE limit and ``decode_token_tile``, the same tests ``_decode_fused`` already
        applies to every attn mix at that row count; an unarmed module (no combine-mix op,
        e.g. the MTP draft after ``disable_decode_fused``) returns before reading it."""
        return (
            self._decode_combine_mix_op is not None
            and self._decode_rows(tokens)
            and dtype == torch.bfloat16
            and all(w.dtype == torch.bfloat16 for w in self._decode_weights())
        )

    def _prefill_full_rows(self, tokens: int) -> bool:
        """Whole fused-prefill tiles: at least two, and an even count unless
        TPU_QWEN4_HC_PREFILL_ODD_TILES / ..._FUSED_MIX allow any count.

        q38 L3-HCGATE (T95u generalization, L2-LADDER j-4b5f2e6d): with the served even-count
        step (2 x tile), "a positive multiple of the step" already means ">= 2 tiles", so the
        test is divisibility alone and puts no bound on the traced token count. The old
        `tokens >= 512` met O1s' cap of 128 x 4 = 512 at 128 slots and pinned the symbol at the
        512 trace (n128 / Form A REFUSED). The prefill-only modes (partial / odd tiles) keep
        their `>=` test: they need it, and they are not on the decode path."""
        if self._prefill_partial:
            return tokens >= 2 * self.prefill_full_tile and tokens % 32 == 0
        if self._prefill_tile_step == 2 * self.prefill_full_tile:
            return tokens % self._prefill_tile_step == 0
        return (
            tokens >= 2 * self.prefill_full_tile
            and tokens % self._prefill_tile_step == 0
        )

    def _prefill_fused_mix(self, hyper_input) -> bool:
        """TPU_QWEN4_HC_PREFILL_FUSED_MIX: norm, gate, mix and the injection
        gate in one Pallas pass over whole prefill tiles."""
        return (
            self._prefill_inj_mix_op is not None
            and hyper_input.dim() == 2
            and self._prefill_full_rows(hyper_input.shape[0])
            and hyper_input.dtype == torch.bfloat16
            and all(w.dtype == torch.bfloat16 for w in self._decode_weights())
        )

    def _hcx_xla(self, hyper_input) -> bool:
        """VLLM_TPU_HC_XLA_MAX_ROWS (HCX, default off; port of q38mm 20e6fcf23):
        an HC of <= N rows skips every prefill-tile Pallas op (fused prefill mix,
        full mix, norm tile, up-epilogue tile, gated-residual combine, fused
        prefill combine_mix) and runs the XLA path that buckets which are not a
        multiple of the tiles (96, 224) already run. That drops the tile ops'
        per-call relayout of the 13 MB up weight (xprof ``copy.weight_ref``).

        Checked after the fused decode branch, which keeps precedence. Evaluated
        only on modules that have tile ops to skip, so a module traced once for
        every token bucket (the MTP draft, whose HC has no tile ops) gains no
        token-count guard; unset, it returns before reading the row count.
        """
        has_tiles = (
            self._prefill_norm_op is not None
            or self._prefill_up_op is not None
            or self._prefill_full_op is not None
            or self._prefill_inj_mix_op is not None
            or self._prefill_combine_op is not None
        )
        return (
            self._hcx_max_rows > 0
            and has_tiles
            and hyper_input.dim() == 2
            and decode_xla_applies(hyper_input.shape[0], self._hcx_max_rows)
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
        # HCX: a mix of <= N rows takes none of the prefill-tile ops below.
        xla = self._hcx_xla(hyper_input)
        if not xla and self._prefill_fused_mix(hyper_input):
            mixed, injection = self._prefill_inj_mix_op(
                hyper_input, *self._decode_weights()
            )
            return mixed, (hyper_input, injection)
        if (
            not xla
            and self._prefill_full_op is not None
            and self._prefill_full_rows(hyper_input.shape[0])
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
        norm = self._normalize(hyper_input, xla=xla)
        gate = F.linear(norm, self.input_mix_weight_down.weight) / self.hc_count
        # TorchTPU can lower BF16 SiLU through BF16 logistic/tanh, rounding
        # negative tails to zero. Compute the activation in FP32 and round
        # once, matching NVIDIA hc_silu's BF16 result boundary.
        gate = F.silu(gate.float()).to(gate.dtype)
        if (
            not xla
            and self._prefill_up_op is not None
            and gate.shape[0] % self.prefill_up_tile == 0   # divisibility alone (see _normalize)
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
            and not self._decode_rows(hyper_input.shape[0])
            # HCX: the combine of a <= N-row HC stays on XLA too, and is not
            # asked the prefill row test (>= 2 * prefill_full_tile), which with
            # a limit of 2 * tile would refine the token symbol to one value
            # (the _decode_rows note).
            and not self._hcx_xla(hyper_input)
            and self._prefill_full_rows(hyper_input.shape[0])
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

    def combine_mix(
        self, block_output, residuals, next_hc, reread=False, prefill_fused=True
    ):
        """``next_hc.mix(self.combine(block_output, residuals))``.

        After a fused decode mix, this combine runs as the prologue of
        ``next_hc``'s fused mix (one Pallas call; the combined hyper input is
        written once and returned as the next residual). ``reread``: the
        deferred boundary's prefill prologue variant that reads its combined
        rows back (TPU_QWEN4_HC_PREFILL_DEFER_WHOLE).

        q38 PORT-BL HC (TPU_QWEN4_HC_DECODE_FOLD, ported from Benliang Wang
        28419c4a5): ``next_hc`` may be the final mixer (no injection gate; its
        decode combine-mix op returns ``(mixed, combined)``, and the residual
        handed back is ``(combined, None)``). ``prefill_fused=False`` (the fold's
        hand-offs) never takes the fused prefill combine-mix: where the decode
        prologue does not apply, the separate combine and mix run, the same ops
        as unfolded.
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
            outs = next_hc._decode_combine_mix_op(
                hyper_input, block_output, gate, *next_hc._decode_weights()
            )
            if not hasattr(next_hc, "block_inject_weight"):
                # TPU_QWEN4_HC_DECODE_FOLD: the final mixer's op (built only with the fold).
                mixed, combined = outs
                return mixed, (combined, None)
            mixed, injection, combined = outs
            return mixed, (combined, injection)
        if (
            prefill_fused
            and next_hc._prefill_combine_mix_op is not None
            and gate is not None
            and gate.shape[-1] == self.hc_count
            and gate.dtype == torch.float32
            and block_output.dtype == torch.bfloat16
            and not next_hc._hcx_xla(hyper_input)
            and next_hc._prefill_fused_mix(hyper_input)
        ):
            op = next_hc._prefill_combine_mix_op
            if reread and getattr(next_hc, "_prefill_combine_mix_rr_op", None) is not None:
                op = next_hc._prefill_combine_mix_rr_op
            mixed, injection, combined = op(
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
