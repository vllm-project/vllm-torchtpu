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
"""Head-grouped BF16 and FP8 block-doubling chunked GDN prefill solver."""

import dataclasses
import enum
import math
from collections.abc import Callable
from typing import override

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.gdn.v3 import config, fp8_matmul, triangular_solve


class BlockDoublingPhase(enum.StrEnum):
    """Execution phase reported to a `phase_hook` callback during a chunk."""

    GROUP_START = "group_start"
    STATE_FIRST_PAIR = "state_first_pair"
    TAIL_HEAD = "tail_head"
    GROUP_END = "group_end"


_LoadKqScaledFn = Callable[
    [list[int], float, float],
    tuple[
        dict[int, jax.Array],
        dict[int, jax.Array],
        dict[int, jax.Array],
        dict[int, jax.Array],
    ],
]
_LoadRawKqFn = Callable[[list[int]], tuple[dict[int, jax.Array], dict[int, jax.Array]]]
_LoadVFn = Callable[[list[int]], dict[int, jax.Array]]
_LoadStateVFn = Callable[[int], jax.Array]
_StoreStateVFn = Callable[[int, jax.Array], None]
_StoreOutSliceFn = Callable[[int, list[jax.Array]], None]
_PhaseHookFn = Callable[[BlockDoublingPhase, int, int], None]

_LOG2_E = 1.0 / math.log(2.0)

# Fixed power-of-two operand scales shared with the triangular solver (`S_L`,
# `S_D`) and the FP8 subclass. Because all factors are powers of two, scaling
# only shifts exponent bits with zero rounding error:
#   _SCALE (256): lifts L2-normalized Q and K into FP8 normal range.
#   _S_Q (2048): lifts Q for Q @ K^T so decayed scores stay above underflow.
#   _INV_STATE_NEW / _INV_OUT_NEW: undo _SCALE / _S_Q and S_D on final updates.
_SCALE = 256.0
_LOG2_SCALE = 8.0
_INV_SCALE = 1.0 / _SCALE
_S_L = triangular_solve.S_L
_S_D = triangular_solve.S_D
_S_Q = 2048.0
_INV_STATE_NEW = _INV_SCALE / _S_D
_INV_OUT_NEW = 1.0 / (_S_Q * _S_D)
# Per-column upper bounds for power-of-two outlier scaling before casting to
# 8-bit FP8 (float8_e4m3fn max is 448). State uses 256 for [K; Q] @ state_prev;
# v_diff uses 16 to leave headroom for the 8x-scaled triangular solve.
_FP8_STATE_MAX_ABS = 256.0
_FP8_VD_MAX_ABS = 16.0


def out_stream_step(cfg: config.GDNConfig, out_itemsize: int) -> int:
    """Returns how many value heads each streamed output store covers."""
    # A store covers whole 8-sublane tiles along the head axis: 8 heads of 32-bit
    # output or 16 heads of 16-bit output. Returns 0 when the head count is not a
    # multiple of that, in which case the output is returned in one piece.
    out_step = 8 * (4 // out_itemsize)
    return 0 if cfg.num_v_heads % out_step else out_step


def _g_lanes(gating_log: jax.Array, chunk: int) -> jax.Array:
    """Computes log2 cumulative decay `[1, chunk, lanes]` via prefix sum."""
    # Hillis-Steele prefix sum in log2 so downstream decay uses native exp2.
    g_lanes = gating_log.astype(jnp.float32) * _LOG2_E
    step = 1
    while step < chunk:
        # Pad sub-8-row shifts to avoid unaligned slices; concat >= 8-row shifts.
        if step < 8:
            shifted = jnp.pad(g_lanes[:, :-step, :], ((0, 0), (step, 0), (0, 0)))
            g_lanes = g_lanes + shifted
        else:
            g_lanes = jnp.concat(
                [g_lanes[:, :step, :], g_lanes[:, step:, :] + g_lanes[:, :-step, :]],
                axis=1,
            )
        step *= 2
    return g_lanes


def _head_groups(cfg: config.GDNConfig) -> list[list[int]]:
    """Partitions V heads into pipeline groups of up to `max(32, v_per_kq_head)`."""
    num_heads, vpk = cfg.num_v_heads, cfg.v_per_kq_head
    group = min(num_heads, max(vpk, (32 // vpk) * vpk))
    return [
        list(range(g0, min(g0 + group, num_heads))) for g0 in range(0, num_heads, group)
    ]


@dataclasses.dataclass
class _HeadState:
    """Per-V-head intermediates passed from front stage to tail solve."""

    oqk_bf16: jax.Array
    k_t_bf16: jax.Array
    d_c: jax.Array | None
    v_diff_bf16: jax.Array | None
    out_updated_bf16: jax.Array | None
    d_inv: jax.Array | None = None
    lw: jax.Array | None = None
    cs_map: dict[int, jax.Array] = dataclasses.field(default_factory=dict)
    v_scale: jax.Array | None = None


class BlockDoublingChunk:
    """One chunk of the head-grouped BF16 block-doubling GDN solver."""

    def __init__(
        self,
        cfg: config.GDNConfig,
        g_lanes: jax.Array,  # [1, chunk, lanes] log2 cumulative decay
        beta: jax.Array,  # [1, chunk, lanes]
        load_kq_scaled: _LoadKqScaledFn | None = None,
        load_v: _LoadVFn | None = None,
        load_state_v: _LoadStateVFn | None = None,
        store_state_v: _StoreStateVFn | None = None,
        store_out_slice: _StoreOutSliceFn | None = None,
        out_step: int = 0,
        out_dtype: jnp.dtype | None = None,
        phase_hook: _PhaseHookFn | None = None,
    ):
        """Initializes decay tables, solver geometry, and streaming callbacks.

        Args:
            cfg: Chunked GDN configuration.
            g_lanes: Cumulative log2 decay `[1, chunk, lanes]`.
            beta: Sigmoid beta gates `[1, chunk, lanes]`.
            load_kq_scaled: Optional per-group loader returning scaled
                `(q_s, q_oqk, k_s, k_t_bf16)` maps.
            load_v: Optional per-pair V loader returning `{v: [chunk, v_dim]}`.
            load_state_v: Optional per-head state loader returning `[kq_dim, v_dim]`.
            store_state_v: Optional per-head state writer `(v, state) -> None`.
            store_out_slice: Optional streamed output writer `(v0, heads) -> None`.
            out_step: Number of value heads per `store_out_slice` call (must be > 0
                and divide `cfg.num_v_heads` when `store_out_slice` is provided).
            out_dtype: Output dtype cast before calling `store_out_slice`.
            phase_hook: Optional `(phase, index, total)` callback invoked with
                `BlockDoublingPhase` (`GROUP_START`, `STATE_FIRST_PAIR`,
                `TAIL_HEAD`, `GROUP_END`) so fused kernels can interleave
                independent matrix-unit work into vector-heavy GDN phases.
        """
        if store_out_slice is not None and (
            out_step <= 0 or cfg.num_v_heads % out_step != 0
        ):
            raise ValueError(
                "out_step must be positive and divide num_v_heads when "
                f"store_out_slice is set; got out_step={out_step}, "
                f"num_v_heads={cfg.num_v_heads}."
            )
        self.cfg = cfg
        self.fp8_dtype = self._resolve_fp8_dtype(cfg)
        self._load_kq_scaled = load_kq_scaled
        self._load_v = load_v
        self._load_state_v = load_state_v
        self._store_state_v = store_state_v
        self._store_out_slice = store_out_slice
        self._out_step = out_step
        self._out_dtype = out_dtype
        self._phase_hook = phase_hook
        self._out_flushed = 0
        self.num_heads = cfg.num_v_heads
        self.chunk = cfg.chunk_size
        self.v_dim = cfg.v_head_dim
        self.kq_dim = cfg.kq_head_dim
        self.solver = triangular_solve.BlockDoublingSolver(self.chunk, self.fp8_dtype)
        self.mm_dims = (((1,), (0,)), ((), ()))
        self.kkt_dims = (((1,), (1,)), ((), ()))
        self.pair_v_heads = cfg.v_per_kq_head % 2 == 0 and self.v_dim % 128 == 0
        self.half_v = cfg.v_per_kq_head // 2
        self._kkt_diag_map: dict[int, jax.Array] = {}

        # Precompute 2D [h_pad, chunk] decay tables so per-head slicing stays inside
        # lanes with no cross-lane broadcasts:
        #   _g_t8_2d: cumulative log2 decay shifted by -8 (undoes 256x K/Q scale)
        #   _gf_t8_2d: forward decay exp2(g - 8) applied to [K; Q] @ state_prev
        #   _gf_last_lanes: full-chunk decay exp2(g_last) applied to state_prev
        #   _g_row_beta_2d / _clamp_row_2d: pairwise decay exponents with beta_j
        #   _beta_gb_2d: backward decay beta_t * exp2(g_last - g_t) for state_new
        h_pad = min(g_lanes.shape[2], max(8, ((self.num_heads + 7) // 8) * 8))
        g_l8 = g_lanes - _LOG2_SCALE
        self._g_t8_2d = jnp.swapaxes(g_l8, 1, 2)[0, :h_pad, :]
        self._gf_t8_2d = jnp.exp2(self._g_t8_2d)
        self._gf_last_lanes = jnp.exp2(g_lanes[:, -1:, :])
        beta_f32 = beta.astype(jnp.float32)
        log2_beta = jnp.log2(beta_f32)
        self._g_row_beta_2d = jnp.swapaxes(g_lanes - log2_beta, 1, 2)[0, :h_pad, :]
        self._clamp_row_2d = jnp.swapaxes(log2_beta - _LOG2_SCALE, 1, 2)[0, :h_pad, :]
        self._beta_gb_2d = jnp.swapaxes(
            beta_f32 * jnp.exp2(g_lanes[:, -1:, :] - g_lanes), 1, 2
        )[0, :h_pad, :]

        iota_r = jax.lax.broadcasted_iota(jnp.int32, (self.chunk, self.chunk), 0)
        iota_c = jax.lax.broadcasted_iota(jnp.int32, (self.chunk, self.chunk), 1)
        self.strictly_lower_mask = iota_r > iota_c
        self.lower_mask = iota_r >= iota_c

    def _resolve_fp8_dtype(self, cfg: config.GDNConfig) -> jnp.dtype | None:
        del cfg
        return None

    def _split_high_low(
        self, x: jax.Array, *, truncate_e4m3: bool = False
    ) -> list[jax.Array]:
        del truncate_e4m3
        return [x if x.dtype == jnp.bfloat16 else x.astype(jnp.bfloat16)]

    def _dot(
        self,
        lhs: list[jax.Array],
        rhs: list[jax.Array],
        dims: jax.lax.DotDimensionNumbers,
    ) -> jax.Array:
        return jax.lax.dot_general(
            lhs[0], rhs[0], dims, preferred_element_type=jnp.float32
        )

    def _prep_state_pair(
        self, s_pair_bf16: jax.Array
    ) -> tuple[list[jax.Array], jax.Array | None]:
        return [s_pair_bf16], None

    def _scale_v_diff_pair(
        self, vd_parts: list[jax.Array]
    ) -> tuple[list[jax.Array], list[jax.Array | None]]:
        return vd_parts, [None] * len(vd_parts)

    def _output_multipliers(
        self, v_scale: jax.Array | None
    ) -> tuple[jax.Array | float, jax.Array]:
        del v_scale
        return _INV_STATE_NEW, jnp.bfloat16(_INV_OUT_NEW)

    def _t2d_col_slice(self, t2d: jax.Array, h0: int, count: int) -> jax.Array:
        """Slices transposed `[heads, chunk]` into `[count, chunk, 1]`."""
        # Align the row slice to 8-sublane boundaries before trimming to avoid
        # cross-sublane vector shuffles.
        b0 = (h0 // 8) * 8
        b1 = min(t2d.shape[0], ((h0 + count + 7) // 8) * 8)
        cols = t2d[b0:b1, :, None]
        if b0 == h0 and b1 - b0 == count:
            return cols
        return cols[h0 - b0 : h0 - b0 + count]

    def _gating_last(self, v: int) -> jax.Array:
        return self._gf_last_lanes[0, :, v : v + 1]

    def _load_state(self, v: int, state_prev: jax.Array | None) -> jax.Array:
        if self._load_state_v is not None:
            return self._load_state_v(v).astype(jnp.float32)
        assert state_prev is not None
        return state_prev[v].astype(jnp.float32)

    def _get_v(self, vs: list[int], v_large: jax.Array | None) -> dict[int, jax.Array]:
        if self._load_v is not None:
            return self._load_v(vs)
        assert v_large is not None
        return {v: v_large[v] for v in vs}

    def _head_pairs(self, heads: list[int]) -> list[tuple[int, ...]]:
        """Groups sibling V heads into pairs when V-head pairing is enabled."""
        if not self.pair_v_heads:
            return [(v,) for v in heads]
        # Pair (v, v + vpk // 2) within each K/Q group so two 128-column states
        # concatenate into one 256-column matrix multiply.
        vpk, half_v = self.cfg.v_per_kq_head, self.half_v
        in_set = set(heads)
        pairs: list[tuple[int, ...]] = []
        seen: set[int] = set()
        for v in heads:
            if v in seen:
                continue
            mate = v + half_v if (v % vpk) < half_v else -1
            if mate in in_set and mate not in seen:
                pairs.append((v, mate))
                seen.add(v)
                seen.add(mate)
            else:
                pairs.append((v,))
                seen.add(v)
        return pairs

    def _prep_kq(
        self,
        q_large: jax.Array | None,
        k_large: jax.Array | None,
        kqs: list[int],
    ) -> dict[int, tuple[jax.Array, jax.Array, list[jax.Array], jax.Array]]:
        """Computes per-KQ-head K K^T, Q K^T, stacked [K; Q], and 256 K^T."""
        if self._load_kq_scaled is not None:
            q_s_map, q_oqk_map, k_s_map, kt_s_map = self._load_kq_scaled(
                kqs, _SCALE, _S_Q
            )
        else:
            q_s_map = q_oqk_map = k_s_map = kt_s_map = None

        per_kq = {}
        for kq in kqs:
            if q_s_map is not None:
                assert k_s_map is not None and kt_s_map is not None
                assert q_oqk_map is not None
                q_s = q_s_map[kq]
                k_s = k_s_map[kq]
                k_t_bf16 = kt_s_map[kq]
                q_oqk = q_oqk_map[kq]
            else:
                assert q_large is not None and k_large is not None
                q_h = q_large[kq].astype(jnp.float32)
                k_h = k_large[kq].astype(jnp.float32)
                q_s = q_h * jnp.float32(_SCALE)
                k_s = k_h * jnp.float32(_SCALE)
                k_t_bf16 = jnp.swapaxes(k_s.astype(jnp.bfloat16), 0, 1)
                q_oqk = q_h * jnp.float32(_S_Q)

            k_parts = self._split_high_low(k_s)
            q_parts = self._split_high_low(q_s)
            q_parts_oqk = self._split_high_low(q_oqk)

            raw_kkt = jnp.where(
                self.strictly_lower_mask,
                self._dot(k_parts, k_parts, self.kkt_dims),
                0,
            )
            # raw_kkt = 256^2 K K^T; dividing by 256 pairs with map8 (= decay / 256)
            # so coef_v = L, l_s = S_L * L, and oqk_bf16 = S_Q * decay * Q K^T.
            self._kkt_diag_map[kq] = self.solver.select_block_diag(
                raw_kkt
            ) * jnp.float32(_INV_SCALE)
            k_k_t = (raw_kkt * jnp.float32(_S_L * _INV_SCALE)).astype(jnp.bfloat16)
            q_k_t = jnp.where(
                self.lower_mask,
                self._dot(q_parts_oqk, k_parts, self.kkt_dims),
                0,
            ).astype(jnp.bfloat16)
            merged = [
                jnp.concat([kp, qp], axis=0)
                for kp, qp in zip(k_parts, q_parts, strict=True)
            ]
            per_kq[kq] = (k_k_t, q_k_t, merged, k_t_bf16)
        return per_kq

    def _project_state_pairs(
        self,
        pairs: list[tuple[int, ...]],
        per_kq: dict[int, tuple[jax.Array, jax.Array, list[jax.Array], jax.Array]],
        state_prev: jax.Array | None,
    ) -> tuple[dict[tuple[int, ...], jax.Array], dict[tuple[int, ...], jax.Array]]:
        """Computes `[K; Q] @ state_prev` across sibling V-head pairs."""
        vpk = self.cfg.v_per_kq_head
        kq_states: dict[tuple[int, ...], jax.Array] = {}
        s_scales: dict[tuple[int, ...], jax.Array] = {}
        for pair in pairs:
            merged = per_kq[pair[0] // vpk][2]
            s_bf16_parts = [
                self._load_state(v, state_prev).astype(jnp.bfloat16) for v in pair
            ]
            s_pair_bf16 = (
                jnp.concat(s_bf16_parts, axis=1)
                if len(s_bf16_parts) > 1
                else s_bf16_parts[0]
            )
            s_limbs, sc_pair = self._prep_state_pair(s_pair_bf16)
            if sc_pair is not None:
                s_scales[pair] = sc_pair
            kq_states[pair] = self._dot(merged, s_limbs, self.mm_dims).astype(
                jnp.bfloat16
            )
        return kq_states, s_scales

    def _build_head(
        self,
        v: int,
        k_k_t: jax.Array,
        q_k_t: jax.Array,
        k_t_bf16: jax.Array,
        v_diff_bf16: jax.Array,
        out_updated_bf16: jax.Array,
        g_col_v: jax.Array,
        v_scale: jax.Array | None,
    ) -> _HeadState:
        """Builds per-head 8-row block inverses and pre-sliced doubling operands."""
        # Pairwise causal decay beta_j * exp2(g_i - g_j) / 256. Causal entries have
        # g_i <= g_j, so clamping at log2(beta_j) - 8 leaves i >= j exact while
        # preventing exp2 overflow (and 0 * inf = NaN) above the diagonal.
        map8 = jnp.exp2(
            jnp.minimum(
                g_col_v - self._g_row_beta_2d[v : v + 1, :],
                self._clamp_row_2d[v : v + 1, :],
            )
        )
        map8_bf16 = map8.astype(jnp.bfloat16)
        kq = v // self.cfg.v_per_kq_head
        coef_v = self._kkt_diag_map[kq] * self.solver.select_block_diag(map8)
        d_c = self.solver.invert_base_blocks(coef_v)
        l_s = k_k_t * map8_bf16
        oqk_bf16 = q_k_t * map8_bf16
        cs_map, lw = self.solver.slice_coupling(l_s)
        return _HeadState(
            oqk_bf16=oqk_bf16,
            k_t_bf16=k_t_bf16,
            d_c=d_c,
            v_diff_bf16=v_diff_bf16,
            out_updated_bf16=out_updated_bf16,
            lw=lw,
            cs_map=cs_map,
            v_scale=v_scale,
        )

    def front(
        self,
        q_large: jax.Array | None,
        k_large: jax.Array | None,
        v_large: jax.Array | None,
        heads: list[int],
        state_prev: jax.Array | None,
    ) -> dict[int, _HeadState]:
        """Runs K/Q and state matmuls, then builds and doubles block inverses."""
        # Sibling value heads sharing a key/query head are paired along the column
        # axis so [K; Q] @ state_prev runs in half as many matrix multiplies.
        vpk, chunk, v_dim = self.cfg.v_per_kq_head, self.chunk, self.v_dim
        pairs = self._head_pairs(heads)
        all_kqs = sorted({pair[0] // vpk for pair in pairs})
        per_kq = self._prep_kq(q_large, k_large, all_kqs)
        kq_states, s_scales = self._project_state_pairs(pairs, per_kq, state_prev)

        built: dict[int, _HeadState] = {}
        cur_kq = -1
        gf_kq: jax.Array | None = None
        g_kq: jax.Array | None = None
        for p_idx, pair in enumerate(pairs):
            if self._phase_hook is not None:
                self._phase_hook(BlockDoublingPhase.STATE_FIRST_PAIR, p_idx, len(pairs))
            kq_state = kq_states.pop(pair)
            sc_pair = s_scales.pop(pair, None)
            v_map = self._get_v(list(pair), v_large)
            p_kq = pair[0] // vpk
            if p_kq != cur_kq:
                # All value heads of a pair share one key/query head, so slice that
                # head's decay columns once and index into them below.
                cur_kq = p_kq
                h0_kq = p_kq * vpk
                cnt_kq = min(vpk, self.num_heads - h0_kq)
                gf_kq = self._t2d_col_slice(self._gf_t8_2d, h0_kq, cnt_kq)
                g_kq = self._t2d_col_slice(self._g_t8_2d, h0_kq, cnt_kq)
            assert gf_kq is not None and g_kq is not None

            vd_parts: list[jax.Array] = []
            outs_pair: dict[int, jax.Array] = {}
            g_cols_pair: dict[int, jax.Array] = {}
            for idx, v in enumerate(pair):
                lanes = slice(idx * v_dim, (idx + 1) * v_dim)
                gf_v = gf_kq[v - cur_kq * vpk]
                g_cols_pair[v] = g_kq[v - cur_kq * vpk]
                gf_b = gf_v.astype(jnp.bfloat16)
                if sc_pair is not None:
                    gf_b = gf_b * sc_pair[:, lanes]
                k_proj_b = kq_state[:chunk, lanes] * gf_b
                outs_pair[v] = kq_state[chunk:, lanes] * gf_b
                vd_parts.append(v_map.pop(v).astype(jnp.bfloat16) - k_proj_b)
            del kq_state
            vd_scaled, v_scales = self._scale_v_diff_pair(vd_parts)
            for idx, v in enumerate(pair):
                k_k_t, q_k_t, _, k_t_bf16 = per_kq[v // vpk]
                built[v] = self._build_head(
                    v,
                    k_k_t,
                    q_k_t,
                    k_t_bf16,
                    v_diff_bf16=vd_scaled[idx],
                    out_updated_bf16=outs_pair.pop(v),
                    g_col_v=g_cols_pair.pop(v),
                    v_scale=v_scales[idx],
                )

        self._kkt_diag_map.clear()
        del per_kq, gf_kq, g_kq
        ordered = {v: built[v] for v in heads}
        d_init: dict[int, jax.Array] = {}
        cs_map: dict[int, dict[int, jax.Array]] = {}
        for v in heads:
            h = ordered[v]
            assert h.d_c is not None
            d_init[v] = h.d_c
            h.d_c = None
            cs_map[v] = h.cs_map
            h.cs_map = {}
        d_inv_map = self.solver.double_inverses(d_init, cs_map, heads)
        for v in heads:
            ordered[v].d_inv = d_inv_map.pop(v)
        return ordered

    def tail(
        self,
        hs: dict[int, _HeadState],
        state_prev: jax.Array | None,
        outs: dict[int, jax.Array],
        states: dict[int, jax.Array],
    ) -> None:
        """Solves `(I + L) u = v_diff` and computes state / output updates."""
        heads = list(hs)
        d_inv_map: dict[int, jax.Array] = {}
        lw_map: dict[int, jax.Array | None] = {}
        rhs_bf16_map: dict[int, jax.Array] = {}
        for v in heads:
            h = hs[v]
            assert h.d_inv is not None and h.v_diff_bf16 is not None
            d_inv_map[v] = h.d_inv
            lw_map[v] = h.lw
            rhs_bf16_map[v] = h.v_diff_bf16
            h.d_inv = None
            h.lw = None
            h.v_diff_bf16 = None

        solved = self.solver.solve_blocks(
            d_inv_map,
            lw_map,
            rhs_bf16_map,
            heads,
            num_heads=self.num_heads,
            v_dim=self.v_dim,
        )

        # Fuse [state_new; out_new] = [256 K^T * beta_gb; 2048 Q K^T * beta *
        # decay] @ (8 u) into one dot per head so solved u is loaded once.
        for h_idx, v in enumerate(heads):
            if self._phase_hook is not None:
                self._phase_hook(BlockDoublingPhase.TAIL_HEAD, h_idx, len(heads))
            h = hs[v]
            assert h.out_updated_bf16 is not None
            u_limbs = solved.pop_u_limbs(v)

            beta_gb_bf16 = self._beta_gb_2d[v : v + 1, :].astype(jnp.bfloat16)
            kg_bf16 = h.k_t_bf16 * beta_gb_bf16
            ko_bf16 = jnp.concat([kg_bf16, h.oqk_bf16], axis=0)
            new = self._dot(self._split_high_low(ko_bf16), u_limbs, self.mm_dims)
            state_new = new[: self.kq_dim]
            out_new = new[self.kq_dim :]

            s_mult, o_mult_bf16 = self._output_multipliers(h.v_scale)
            h.v_scale = None

            s_prev_v = self._load_state(v, state_prev)
            s_new = s_prev_v * self._gating_last(v) + state_new * s_mult
            if self._store_state_v is not None:
                self._store_state_v(v, s_new)
            else:
                states[v] = s_new

            out_raw = h.out_updated_bf16
            h.out_updated_bf16 = None
            out_bf16 = out_raw + out_new.astype(jnp.bfloat16) * o_mult_bf16
            if self._store_out_slice is not None and self._out_dtype is not None:
                outs[v] = out_bf16.astype(self._out_dtype)
            else:
                outs[v] = out_bf16.astype(jnp.float32)

            # Hand completed heads to the caller in aligned `out_step` batches, as
            # one [chunk, v_head_dim] array per head, to keep register pressure
            # bounded across large head counts.
            if self._store_out_slice is not None:
                step = self._out_step
                while self._out_flushed + step <= self.num_heads and all(
                    (self._out_flushed + r) in outs for r in range(step)
                ):
                    v0 = self._out_flushed
                    self._store_out_slice(v0, [outs.pop(v0 + r) for r in range(step)])
                    self._out_flushed += step


class Fp8BlockDoublingChunk(BlockDoublingChunk):
    """FP8 block-doubling solver adding column scaling and high/low dots.

    Overrides the precision hooks of `BlockDoublingChunk` to:
      1. Scale `state_prev` columns below 256 before `[K; Q] @ state_prev`.
      2. Scale `v_diff` columns below 16 before solving `(I + L) u = v_diff`.
      3. Split operands into FP8 `[high, low]` pairs and unscale final updates.
    """

    @override
    def _resolve_fp8_dtype(self, cfg: config.GDNConfig) -> jnp.dtype:
        fp8_dtype = cfg.gdn_prefill_mode.mxu_operand_dtype
        assert fp8_dtype is not None, cfg.gdn_prefill_mode
        return fp8_dtype

    @override
    def _split_high_low(
        self, x: jax.Array, *, truncate_e4m3: bool = False
    ) -> list[jax.Array]:
        return fp8_matmul.split_high_low(x, self.fp8_dtype, truncate_e4m3=truncate_e4m3)

    @override
    def _dot(
        self,
        lhs: list[jax.Array],
        rhs: list[jax.Array],
        dims: jax.lax.DotDimensionNumbers,
    ) -> jax.Array:
        return fp8_matmul.dot_general(lhs, rhs, dims)

    @override
    def _prep_state_pair(
        self, s_pair_bf16: jax.Array
    ) -> tuple[list[jax.Array], jax.Array]:
        sc_pair, inv_pair = fp8_matmul.compute_column_pow2_scales(
            s_pair_bf16, _FP8_STATE_MAX_ABS
        )
        return (
            self._split_high_low(s_pair_bf16 * inv_pair, truncate_e4m3=True),
            sc_pair,
        )

    @override
    def _scale_v_diff_pair(
        self, vd_parts: list[jax.Array]
    ) -> tuple[list[jax.Array], list[jax.Array | None]]:
        # Scale both paired heads together so column max reduction stays one pass.
        vd_cat = jnp.concat(vd_parts, axis=1) if len(vd_parts) > 1 else vd_parts[0]
        v_sc_cat, v_inv_cat = fp8_matmul.compute_column_pow2_scales(
            vd_cat, _FP8_VD_MAX_ABS
        )
        vd_scaled_cat = vd_cat * v_inv_cat
        v_dim = self.v_dim
        vd_scaled = [
            vd_scaled_cat[:, idx * v_dim : (idx + 1) * v_dim]
            for idx in range(len(vd_parts))
        ]
        v_scales: list[jax.Array | None] = [
            v_sc_cat[:, idx * v_dim : (idx + 1) * v_dim] for idx in range(len(vd_parts))
        ]
        return vd_scaled, v_scales

    @override
    def _output_multipliers(
        self, v_scale: jax.Array | None
    ) -> tuple[jax.Array, jax.Array]:
        assert v_scale is not None
        s_mult = v_scale.astype(jnp.float32) * jnp.float32(_INV_STATE_NEW)
        o_mult_bf16 = v_scale * jnp.bfloat16(_INV_OUT_NEW)
        return s_mult, o_mult_bf16


def _chunk_cls_for_cfg(cfg: config.GDNConfig) -> type[BlockDoublingChunk]:
    if cfg.gdn_prefill_mode == config.GDNPrefillMode.BLOCK_DOUBLING_FP8:
        return Fp8BlockDoublingChunk
    return BlockDoublingChunk


def run_chunk_per_seq(
    q_large: jax.Array | None,
    k_large: jax.Array | None,
    v_large: jax.Array | None,
    gating_log: jax.Array,  # [1, chunk, lanes]
    beta: jax.Array,  # [1, chunk, lanes]
    state_prev: jax.Array | None,
    cfg: config.GDNConfig,
    *,
    load_kq_scaled: _LoadKqScaledFn | None = None,
    load_v: _LoadVFn | None = None,
    load_state_v: _LoadStateVFn | None = None,
    store_state_v: _StoreStateVFn | None = None,
    store_out_slice: _StoreOutSliceFn | None = None,
    out_step: int = 0,
    out_dtype: jnp.dtype | None = None,
    phase_hook: _PhaseHookFn | None = None,
) -> tuple[jax.Array | None, jax.Array | None]:
    """Runs one block-doubling chunked GDN step sequentially per head group.

    Args:
        q_large: Optional pre-loaded queries `[num_kq_heads, chunk, kq_dim]`.
        k_large: Optional pre-loaded keys `[num_kq_heads, chunk, kq_dim]`.
        v_large: Optional pre-loaded values `[num_v_heads, chunk, v_dim]`.
        gating_log: Log-decay gates `[1, chunk, lanes]`.
        beta: Sigmoid beta gates `[1, chunk, lanes]`.
        state_prev: Optional prior state `[num_v_heads, kq_dim, v_dim]`.
        cfg: Chunked GDN configuration.
        load_kq_scaled: Optional per-group scaled K/Q loader callback.
        load_v: Optional per-pair V loader callback.
        load_state_v: Optional per-head state loader callback.
        store_state_v: Optional per-head state store callback.
        store_out_slice: Optional streamed output slice store callback.
        out_step: Value heads per `store_out_slice` call.
        out_dtype: Output dtype when streaming via `store_out_slice`.
        phase_hook: Optional `(phase, index, total)` callback invoked with
            `BlockDoublingPhase` (`GROUP_START`, `STATE_FIRST_PAIR`,
            `TAIL_HEAD`, `GROUP_END`).

    Returns:
        `(out, state)` stacked across value heads; either element is `None` when
        the corresponding streaming callback writes directly to VMEM.
    """
    if not cfg.use_block_doubling:
        raise ValueError(
            "Expected active block-doubling mode; got"
            f" mode={cfg.gdn_prefill_mode.value!r}, chunk_size={cfg.chunk_size}."
        )
    if state_prev is None and load_state_v is None:
        raise ValueError("Either state_prev or load_state_v must be provided.")
    ctx = _chunk_cls_for_cfg(cfg)(
        cfg,
        _g_lanes(gating_log, cfg.chunk_size),
        beta,
        load_kq_scaled=load_kq_scaled,
        load_v=load_v,
        load_state_v=load_state_v,
        store_state_v=store_state_v,
        store_out_slice=store_out_slice,
        out_step=out_step,
        out_dtype=out_dtype,
        phase_hook=phase_hook,
    )
    outs: dict[int, jax.Array] = {}
    states: dict[int, jax.Array] = {}
    groups = _head_groups(cfg)
    for g_idx, grp in enumerate(groups):
        if phase_hook is not None:
            phase_hook(BlockDoublingPhase.GROUP_START, g_idx, len(groups))
        hs = ctx.front(q_large, k_large, v_large, grp, state_prev)
        ctx.tail(hs, state_prev, outs, states)
        if phase_hook is not None:
            phase_hook(BlockDoublingPhase.GROUP_END, g_idx, len(groups))
    out = (
        None
        if store_out_slice is not None
        else jnp.stack([outs[v] for v in range(cfg.num_v_heads)], axis=0)
    )
    state = (
        None
        if store_state_v is not None
        else jnp.stack([states[v] for v in range(cfg.num_v_heads)], axis=0)
    )
    return out, state


def run_chunk_streamed(
    *,
    load_raw_kq: _LoadRawKqFn,
    load_raw_v: _LoadVFn,
    cfg: config.GDNConfig,
    load_state_v: _LoadStateVFn,
    store_state_v: _StoreStateVFn,
    real_sizes: jax.Array | None = None,
    b_lanes: jax.Array | None = None,  # [1, chunk, lanes]
    a_lanes: jax.Array | None = None,  # [1, chunk, lanes]
    a_log: jax.Array | None = None,
    dt_bias: jax.Array | None = None,
    gating_log_lanes: jax.Array | None = None,  # [1, chunk, lanes]
    beta_lanes: jax.Array | None = None,  # [1, chunk, lanes]
    kq_prescaled_256: bool = False,
    store_out_slice: _StoreOutSliceFn | None = None,
    out_step: int = 0,
    out_dtype: jnp.dtype | None = None,
    phase_hook: _PhaseHookFn | None = None,
) -> tuple[jax.Array | None, jax.Array | None]:
    """Runs one block-doubling step, streaming Q, K, and V per head group.

    L2-normalizes and scales Q and K on demand unless `kq_prescaled_256=True`.
    Either pass precomputed `(gating_log_lanes, beta_lanes)` or raw gate inputs
    `(b_lanes, a_lanes, a_log, dt_bias)`.

    Args:
        load_raw_kq: Callback returning raw `(q_map, k_map)` for requested K/Q
            heads.
        load_raw_v: Callback returning raw `{v: [chunk, v_dim]}` for requested
            heads.
        cfg: Active block-doubling GDN configuration.
        load_state_v: Per-head state loader `(v) -> [kq_dim, v_dim]`.
        store_state_v: Per-head state writer `(v, state) -> None`.
        real_sizes: Optional valid token count `[1]` used to mask padded rows;
            when `None`, all rows in the chunk are treated as valid.
        b_lanes: Raw beta pre-activations `[1, chunk, lanes]` when not
            precomputed.
        a_lanes: Raw decay pre-activations `[1, chunk, lanes]` when not
            precomputed.
        a_log: Per-head log-decay parameter `[lanes]` when not precomputed.
        dt_bias: Per-head decay bias parameter `[lanes]` when not precomputed.
        gating_log_lanes: Optional precomputed log-decay gates
            `[1, chunk, lanes]`.
        beta_lanes: Optional precomputed sigmoid beta gates `[1, chunk, lanes]`.
        kq_prescaled_256: Whether `load_raw_kq` already returns L2-normalized
            Q/K scaled by 256 (with Q also scaled by `1 / sqrt(d_k)`).
        store_out_slice: Optional streamed output writer `(v0, heads) -> None`.
        out_step: Value heads per `store_out_slice` call.
        out_dtype: Output dtype when streaming via `store_out_slice`.
        phase_hook: Optional `(phase, index, total)` callback invoked with
            `BlockDoublingPhase` (`GROUP_START`, `STATE_FIRST_PAIR`,
            `TAIL_HEAD`, `GROUP_END`).

    Returns:
        `(out, None)` (recurrent state always streams through `store_state_v`),
        with `out` also `None` when `store_out_slice` streams outputs.
    """
    if not cfg.use_block_doubling:
        raise ValueError(
            "Expected active block-doubling mode; got"
            f" mode={cfg.gdn_prefill_mode.value!r}, chunk_size={cfg.chunk_size}."
        )
    has_precomputed = (gating_log_lanes is not None, beta_lanes is not None)
    raw_gate_args = (b_lanes, a_lanes, a_log, dt_bias)
    if any(has_precomputed) and not all(has_precomputed):
        raise ValueError(
            "gating_log_lanes and beta_lanes must both be set or both be None."
        )
    if all(has_precomputed) and any(arg is not None for arg in raw_gate_args):
        raise ValueError(
            "Precomputed (gating_log_lanes, beta_lanes) cannot be combined with"
            " raw gate inputs (b_lanes, a_lanes, a_log, dt_bias)."
        )
    if not all(has_precomputed) and any(arg is None for arg in raw_gate_args):
        raise ValueError(
            "b_lanes, a_lanes, a_log, and dt_bias are required unless"
            " gating_log_lanes and beta_lanes are provided."
        )

    compute = cfg.dtypes.compute
    q_scale = cfg.kq_head_dim**-0.5
    mask_2d: jax.Array | None = None
    mask_3d: jax.Array | None = None
    if real_sizes is not None:
        mask_dtype = jnp.int32 if jnp.dtype(compute).itemsize == 4 else jnp.int16
        iota = jax.lax.broadcasted_iota(mask_dtype, (cfg.chunk_size, 1), 0)
        mask_2d = iota < real_sizes[0].astype(mask_dtype)
        mask_3d = mask_2d[jnp.newaxis, :, :]

    def load_kq_scaled(
        kqs: list[int], scale: float, s_q: float
    ) -> tuple[
        dict[int, jax.Array],
        dict[int, jax.Array],
        dict[int, jax.Array],
        dict[int, jax.Array],
    ]:
        qs_raw, ks_raw = load_raw_kq(kqs)
        q_s_map: dict[int, jax.Array] = {}
        q_oqk_map: dict[int, jax.Array] = {}
        k_s_map: dict[int, jax.Array] = {}
        kt_s_map: dict[int, jax.Array] = {}
        # Derive s_q-scaled Q from scale-scaled Q so L2 normalization runs once.
        ratio_q = jnp.float32(s_q / scale)
        for kq in kqs:
            qh = qs_raw[kq]
            kh = ks_raw[kq]
            if mask_2d is not None:
                qh = jnp.where(mask_2d, qh.astype(compute), 0)
                kh = jnp.where(mask_2d, kh.astype(compute), 0)
            if kq_prescaled_256:
                assert scale == _SCALE, (
                    f"kq_prescaled_256 expects scale={_SCALE}; got {scale}."
                )
                q_s = qh.astype(compute)
                k_s = kh.astype(compute)
            else:
                qf = qh.astype(jnp.float32)
                kf = kh.astype(jnp.float32)
                inv_q = jax.lax.rsqrt(jnp.sum(qf * qf, axis=-1, keepdims=True) + 1e-6)
                inv_k = jax.lax.rsqrt(jnp.sum(kf * kf, axis=-1, keepdims=True) + 1e-6)
                q_s = (qf * (inv_q * jnp.float32(q_scale * scale))).astype(compute)
                k_s = (kf * (inv_k * jnp.float32(scale))).astype(compute)
            q_s_map[kq] = q_s
            q_oqk_map[kq] = q_s * ratio_q
            k_s_map[kq] = k_s
            kt_s_map[kq] = jnp.swapaxes(k_s.astype(jnp.bfloat16), 0, 1)
        return q_s_map, q_oqk_map, k_s_map, kt_s_map

    def load_v(vs: list[int]) -> dict[int, jax.Array]:
        vs_raw = load_raw_v(vs)
        if mask_2d is not None:
            return {v: jnp.where(mask_2d, vs_raw[v].astype(compute), 0) for v in vs}
        return {v: vs_raw[v].astype(compute) for v in vs}

    if gating_log_lanes is not None and beta_lanes is not None:
        gating_log = gating_log_lanes.astype(compute)
        beta = beta_lanes.astype(compute)
    else:
        assert b_lanes is not None and a_lanes is not None
        assert a_log is not None and dt_bias is not None
        b_c = b_lanes.astype(compute)
        a_c = a_lanes.astype(compute)
        al_c = a_log.reshape(1, 1, -1).astype(compute)
        dt_c = dt_bias.reshape(1, 1, -1).astype(compute)
        beta = jax.nn.sigmoid(b_c)
        gating_log = -jnp.exp(al_c) * jax.nn.softplus(a_c + dt_c)
    if mask_3d is not None:
        beta = jnp.where(mask_3d, beta, 0)
        gating_log = jnp.where(mask_3d, gating_log, 0)

    return run_chunk_per_seq(
        None,
        None,
        None,
        gating_log,
        beta,
        None,
        cfg,
        load_kq_scaled=load_kq_scaled,
        load_v=load_v,
        load_state_v=load_state_v,
        store_state_v=store_state_v,
        store_out_slice=store_out_slice,
        out_step=out_step,
        out_dtype=out_dtype,
        phase_hook=phase_hook,
    )
