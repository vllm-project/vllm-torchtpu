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
"""TPU-MoonEP: Dynamic Balanced Routing & Chip-Aware 2-Round Overlapped Weight RDMA.

Integrates directly into `fused_ep_moe_v2`:
  1. `build_pallas_moonep_planner`: Sparse SMEM greedy balancer + Chip-Aware
     2-Round Peer-First Tree multicast scheduler (zero host sync, ~50 us).
  2. `remap_topk_to_phys`: Zero-gather bit-exact MXU prefix-sum token remapper
     (handles masked/padded `topk_idx == e_total` rows -> `e_phys` sentinel).
  3. `moonep_local_first_visit_list`: Schedules active local experts before
     migrated experts so both RDMA rounds complete in the background behind MXU
     compute with zero global inter-round barriers.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

LANES = 128


def _align_lanes(n: int) -> int:
    return ((n + LANES - 1) // LANES) * LANES


@functools.lru_cache(maxsize=64)
def build_pallas_moonep_planner(
    *,
    ep: int,
    e_total: int,
    b_slots: int,
    h_hot: int,
    cap: int,
    min_mig_tokens: int = 0,
    r0_direct_max: int = 4,
    r1_fanout: int = 3,
):
    """Sparse SMEM MoonEP planner + Chip-Aware 2-Round Pipelined Tree scheduler."""
    epn = e_total // ep
    g_phys = epn + b_slots
    max_ops = ep * b_slots
    n_chips = max(1, (ep + 1) // 2)

    r_se = 0
    n_se = _align_lanes(h_hot) // LANES

    r_sae = r_se + n_se
    n_sae = _align_lanes(h_hot * ep) // LANES

    r_spid = r_sae + n_sae
    n_spid = _align_lanes(h_hot * ep) // LANES

    r_rge = r_spid + n_spid
    n_rge = _align_lanes(ep * b_slots) // LANES

    n_op_elems = ep * max_ops
    n_op_rows = _align_lanes(n_op_elems) // LANES
    r_r0_sle = r_rge + n_rge
    r_r0_dr = r_r0_sle + n_op_rows
    r_r0_ds = r_r0_dr + n_op_rows
    r_r1_ss = r_r0_ds + n_op_rows
    r_r1_dr = r_r1_ss + n_op_rows
    r_r1_ds = r_r1_dr + n_op_rows

    n_cnt_rows = _align_lanes(ep) // LANES
    r_r0_ns = r_r1_ds + n_op_rows
    r_r0_nr = r_r0_ns + n_cnt_rows
    r_r1_ns = r_r0_nr + n_cnt_rows
    r_r1_nr = r_r1_ns + n_cnt_rows

    r_meta = r_r1_nr + n_cnt_rows
    out_rows = ((r_meta + 1 + 7) // 8) * 8

    def write_flat(buf, base_row: int, idx, val):
        r = base_row + jnp.right_shift(idx, 7)
        c = idx & (LANES - 1)
        buf[r, c] = val

    def read_flat(buf, base_row: int, idx):
        r = base_row + jnp.right_shift(idx, 7)
        c = idx & (LANES - 1)
        return buf[r, c]

    def planner_kernel(
        expert_count_sm,  # [e_total] i32
        group_tokens_sm,  # [ep] i32
        out_hbm,  # [out_rows, 128] i32
        out_sm,  # [out_rows, 128] i32
        balance_sm,  # [ep] i32
        z_sm,  # [ep, ep] i32
        dest_slots_sm,  # [ep] i32
        rem_sm,  # [epn] i32
        quotas_sm,  # [ep] i32
        e_to_hot_sm,  # [e_total] i32
        hot_e_sm,  # [h_hot] i32
        hot_alloc_sm,  # [h_hot, ep] i32
        hot_phys_sm,  # [h_hot, ep] i32
        hot_n_need_sm,  # [h_hot] i32
        hot_need_d_sm,  # [h_hot, ep] i32
        hot_need_slot_sm,  # [h_hot, ep] i32
        home_need_cnt_sm,  # [ep] i32
        chip_used_sm,  # [n_chips] i32
        local_chip_sm,  # [n_chips] i32
        rank_used_sm,  # [ep] i32
        n_hot_sm,  # [1] i32
        cp_sem,
    ):
        zero = jnp.int32(0)
        one = jnp.int32(1)
        neg1 = jnp.int32(-1)
        ep_i = jnp.int32(ep)
        epn_i = jnp.int32(epn)
        g_phys_i = jnp.int32(g_phys)
        b_slots_i = jnp.int32(b_slots)
        h_hot_i = jnp.int32(h_hot)
        max_ops_i = jnp.int32(max_ops)
        min_mig_i = jnp.int32(min_mig_tokens)
        # Phase A / home-level gate uses `min_mig_i` (rank-overload threshold),
        # while Phase B per-destination shards are ~1/ep of home overload; cap
        # the per-dest floor at 5% of `cap` so raising `min_mig_frac` cleanly
        # skips mild-skew layers without starving cold destinations at high skew.
        dest_min_i = jnp.int32(
            min(min_mig_tokens, max(1, cap // 20)) if min_mig_tokens > 0 else 0
        )
        r0_dir_i = jnp.int32(r0_direct_max)
        r1_fan_i = jnp.int32(r1_fanout)
        denom_seed = jnp.int32(r1_fanout + 1)

        # Compute actual average tokens per rank (handles padded/masked steps where
        # total valid routed pairs < ep * cap cleanly!).
        def sum_gt(r, acc):
            return acc + group_tokens_sm[r]

        total_valid = lax.fori_loop(0, ep, sum_gt, zero)
        eff_cap = jnp.minimum(
            jnp.int32(cap),
            (total_valid + ep_i - one) // ep_i,
        )

        def init_ep(r, _):
            balance_sm[r] = group_tokens_sm[r] - eff_cap
            dest_slots_sm[r] = zero
            home_need_cnt_sm[r] = zero
            rank_used_sm[r] = zero
            write_flat(out_sm, r_r0_ns, r, zero)
            write_flat(out_sm, r_r0_nr, r, zero)
            write_flat(out_sm, r_r1_ns, r, zero)
            write_flat(out_sm, r_r1_nr, r, zero)

            def init_z(d, _c):
                z_sm[r, d] = zero
                return _c

            lax.fori_loop(0, ep, init_z, zero)

            def init_rge(b, _b):
                write_flat(out_sm, r_rge, r * b_slots_i + b, zero)
                return _b

            lax.fori_loop(0, b_slots, init_rge, zero)
            return zero

        lax.fori_loop(0, ep, init_ep, zero)

        def init_chip(c, _):
            chip_used_sm[c] = zero
            return zero

        lax.fori_loop(0, n_chips, init_chip, zero)
        out_sm[r_meta, 0] = zero

        def init_hot(i, _):
            hot_e_sm[i] = neg1
            hot_n_need_sm[i] = zero
            write_flat(out_sm, r_se, i, neg1)
            return zero

        lax.fori_loop(0, h_hot, init_hot, zero)
        n_hot_sm[0] = zero

        # 1. Phase A: Greedy overloaded home h -> roomiest destination u
        def phase_a_cond(step):
            return step < ep_i

        def phase_a_body(step):
            def find_extrema(r, st):
                max_i, max_v, min_i, min_v = st
                v = balance_sm[r]
                is_gt = v > max_v
                is_lt = v < min_v
                return (
                    jnp.where(is_gt, r, max_i),
                    jnp.where(is_gt, v, max_v),
                    jnp.where(is_lt, r, min_i),
                    jnp.where(is_lt, v, min_v),
                )

            h, max_v, u, min_v = lax.fori_loop(
                1,
                ep,
                find_extrema,
                (zero, balance_sm[0], zero, balance_sm[0]),
            )
            can_step = jnp.logical_and(max_v > min_mig_i, min_v < zero)

            @pl.when(can_step)
            def _do_move():
                move = jnp.minimum(max_v, -min_v)
                z_sm[h, u] = z_sm[h, u] + move
                balance_sm[h] = balance_sm[h] - move
                balance_sm[u] = balance_sm[u] + move

            return jnp.where(can_step, step + one, ep_i)

        lax.while_loop(phase_a_cond, phase_a_body, zero)

        # 2. Phase B: Visit overloaded homes in descending overload order
        def compute_home_overload(h, _):
            def sum_z(d, acc):
                return acc + z_sm[h, d]

            balance_sm[h] = lax.fori_loop(0, ep, sum_z, zero)
            return zero

        lax.fori_loop(0, ep, compute_home_overload, zero)

        def phase_b_step(_step, _):
            def find_heaviest_home(r, st):
                best_h, best_q = st
                q = balance_sm[r]
                ok = q > best_q
                return jnp.where(ok, r, best_h), jnp.where(ok, q, best_q)

            h, total_q = lax.fori_loop(0, ep, find_heaviest_home, (zero, zero))

            @pl.when(total_q > min_mig_i)
            def _work_home():
                balance_sm[h] = zero

                def copy_q(d, _c):
                    quotas_sm[d] = z_sm[h, d]
                    return _c

                lax.fori_loop(0, ep, copy_q, zero)
                base_e = h * epn_i

                def load_rem(le, _c):
                    g_e = base_e + le
                    rem_sm[le] = expert_count_sm[g_e]
                    e_to_hot_sm[g_e] = neg1
                    return _c

                lax.fori_loop(0, epn, load_rem, zero)

                def b_cond(done_flag):
                    return done_flag == zero

                def b_body(_done):
                    def find_d(r, st):
                        best_d, best_q = st
                        q = quotas_sm[r]
                        ok = jnp.logical_and(q > best_q, dest_slots_sm[r] < b_slots_i)
                        return jnp.where(ok, r, best_d), jnp.where(ok, q, best_q)

                    d, best_q = lax.fori_loop(0, ep, find_d, (neg1, zero))

                    def find_e(le, st):
                        best_e, best_r = st
                        rv = rem_sm[le]
                        ok = rv > best_r
                        return jnp.where(ok, le, best_e), jnp.where(ok, rv, best_r)

                    le, best_r = lax.fori_loop(0, epn, find_e, (neg1, zero))
                    take = jnp.minimum(best_r, best_q)
                    can_mig = jnp.logical_and(
                        jnp.logical_and(best_q > dest_min_i, best_r > dest_min_i),
                        take > dest_min_i,
                    )
                    g_e = base_e + jnp.maximum(le, zero)
                    cur_hi = jnp.where(can_mig, e_to_hot_sm[g_e], zero)
                    has_hi = jnp.logical_or(cur_hi >= zero, n_hot_sm[0] < h_hot_i)
                    can_mig = jnp.logical_and(can_mig, has_hi)

                    @pl.when(can_mig)
                    def _mig():
                        hi_existing = e_to_hot_sm[g_e]

                        @pl.when(hi_existing < zero)
                        def _alloc_hot():
                            nh = n_hot_sm[0]
                            n_hot_sm[0] = nh + one
                            e_to_hot_sm[g_e] = nh
                            hot_e_sm[nh] = g_e
                            hot_n_need_sm[nh] = zero
                            ec = expert_count_sm[g_e]

                            def init_hd(r, _c):
                                hot_alloc_sm[nh, r] = jnp.where(r == h, ec, zero)
                                hot_phys_sm[nh, r] = jnp.where(
                                    r == h, h * g_phys_i + le, neg1
                                )
                                return _c

                            lax.fori_loop(0, ep, init_hd, zero)

                        hi = e_to_hot_sm[g_e]
                        slot = dest_slots_sm[d]
                        dest_slots_sm[d] = slot + one
                        hot_alloc_sm[hi, d] = hot_alloc_sm[hi, d] + take
                        hot_alloc_sm[hi, h] = hot_alloc_sm[hi, h] - take
                        hot_phys_sm[hi, d] = d * g_phys_i + epn_i + slot
                        rem_sm[le] = rem_sm[le] - take
                        quotas_sm[d] = quotas_sm[d] - take

                        nn = hot_n_need_sm[hi]
                        hot_need_d_sm[hi, nn] = d
                        hot_need_slot_sm[hi, nn] = slot
                        hot_n_need_sm[hi] = nn + one
                        home_need_cnt_sm[h] = home_need_cnt_sm[h] + one

                        write_flat(out_sm, r_rge, d * b_slots_i + slot, g_e)
                        out_sm[r_meta, 0] = one

                    return jnp.where(can_mig, zero, one)

                lax.while_loop(b_cond, b_body, zero)

            return zero

        lax.fori_loop(0, ep, phase_b_step, zero)

        # 3. Emit split tables + Chip-Aware 2-Round Pipelined Tree Schedule
        n_hot = n_hot_sm[0]

        def emit_hot(hi, _):
            g_e = hot_e_sm[hi]
            write_flat(out_sm, r_se, hi, g_e)

            def write_d(d, running):
                nxt = running + hot_alloc_sm[hi, d]
                idx = hi * ep_i + d
                write_flat(out_sm, r_sae, idx, nxt)
                write_flat(out_sm, r_spid, idx, hot_phys_sm[hi, d])
                return nxt

            lax.fori_loop(0, ep, write_d, zero)

            h = g_e // epn_i
            le = g_e - h * epn_i
            n_need = hot_n_need_sm[hi]
            h_tot = home_need_cnt_sm[h]

            n_seed_tree = (n_need + r1_fan_i) // denom_seed
            n_seed = jnp.where(
                h_tot <= r0_dir_i,
                n_need,
                jnp.minimum(n_need, jnp.maximum(one, n_seed_tree)),
            )

            @pl.when(n_seed < n_need)
            def _pick_chip_diverse_seeds():
                def reset_local_chip(c, _c):
                    local_chip_sm[c] = zero
                    return _c

                lax.fori_loop(0, n_chips, reset_local_chip, zero)

                def pick_one(s, _):
                    def scan_j(j, st):
                        best_j, best_cost = st
                        dj = hot_need_d_sm[hi, j]
                        cj = jnp.right_shift(dj, 1)
                        # Prefer distinct physical chips within this expert first,
                        # then least-loaded relay rank, then least-loaded chip.
                        cost = (
                            jnp.left_shift(local_chip_sm[cj], 16)
                            + jnp.left_shift(rank_used_sm[dj], 8)
                            + chip_used_sm[cj]
                        )
                        better = cost < best_cost
                        return jnp.where(better, j, best_j), jnp.where(
                            better, cost, best_cost
                        )

                    bj, _ = lax.fori_loop(s, n_need, scan_j, (s, jnp.int32(0x3FFFFFFF)))
                    d_s = hot_need_d_sm[hi, s]
                    sl_s = hot_need_slot_sm[hi, s]
                    d_b = hot_need_d_sm[hi, bj]
                    sl_b = hot_need_slot_sm[hi, bj]
                    hot_need_d_sm[hi, s] = d_b
                    hot_need_slot_sm[hi, s] = sl_b
                    hot_need_d_sm[hi, bj] = d_s
                    hot_need_slot_sm[hi, bj] = sl_s
                    cb = jnp.right_shift(d_b, 1)
                    local_chip_sm[cb] = one
                    chip_used_sm[cb] = chip_used_sm[cb] + one
                    rank_used_sm[d_b] = rank_used_sm[d_b] + one
                    return zero

                lax.fori_loop(0, n_seed, pick_one, zero)

            def emit_r0(j, _):
                dr = hot_need_d_sm[hi, j]
                ds = hot_need_slot_sm[hi, j]
                op = read_flat(out_sm, r_r0_ns, h)
                flat_op = h * max_ops_i + op
                write_flat(out_sm, r_r0_sle, flat_op, le)
                write_flat(out_sm, r_r0_dr, flat_op, dr)
                write_flat(out_sm, r_r0_ds, flat_op, ds)
                write_flat(out_sm, r_r0_ns, h, op + one)
                prev_nr = read_flat(out_sm, r_r0_nr, dr)
                write_flat(out_sm, r_r0_nr, dr, prev_nr + one)
                return zero

            lax.fori_loop(0, n_seed, emit_r0, zero)

            def emit_r1(j, _):
                s = (j - n_seed) % n_seed
                sr = hot_need_d_sm[hi, s]
                ss = hot_need_slot_sm[hi, s]
                dr = hot_need_d_sm[hi, j]
                ds = hot_need_slot_sm[hi, j]
                op = read_flat(out_sm, r_r1_ns, sr)
                flat_op = sr * max_ops_i + op
                write_flat(out_sm, r_r1_ss, flat_op, ss)
                write_flat(out_sm, r_r1_dr, flat_op, dr)
                write_flat(out_sm, r_r1_ds, flat_op, ds)
                write_flat(out_sm, r_r1_ns, sr, op + one)
                prev_nr = read_flat(out_sm, r_r1_nr, dr)
                write_flat(out_sm, r_r1_nr, dr, prev_nr + one)
                return zero

            lax.fori_loop(n_seed, n_need, emit_r1, zero)
            return zero

        lax.fori_loop(0, n_hot, emit_hot, zero)

        cp = pltpu.make_async_copy(out_sm, out_hbm, cp_sem)
        cp.start()
        cp.wait()

    hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    call = pl.pallas_call(
        planner_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[],
            out_specs=hbm,
            scratch_shapes=(
                pltpu.SMEM((out_rows, LANES), jnp.int32),
                pltpu.SMEM((ep,), jnp.int32),
                pltpu.SMEM((ep, ep), jnp.int32),
                pltpu.SMEM((ep,), jnp.int32),
                pltpu.SMEM((epn,), jnp.int32),
                pltpu.SMEM((ep,), jnp.int32),
                pltpu.SMEM((e_total,), jnp.int32),
                pltpu.SMEM((h_hot,), jnp.int32),
                pltpu.SMEM((h_hot, ep), jnp.int32),
                pltpu.SMEM((h_hot, ep), jnp.int32),
                pltpu.SMEM((h_hot,), jnp.int32),
                pltpu.SMEM((h_hot, ep), jnp.int32),
                pltpu.SMEM((h_hot, ep), jnp.int32),
                pltpu.SMEM((ep,), jnp.int32),
                pltpu.SMEM((n_chips,), jnp.int32),
                pltpu.SMEM((n_chips,), jnp.int32),
                pltpu.SMEM((ep,), jnp.int32),
                pltpu.SMEM((1,), jnp.int32),
                pltpu.SemaphoreType.DMA,
            ),
            grid=(),
        ),
        out_shape=jax.ShapeDtypeStruct((out_rows, LANES), jnp.int32),
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
        name=f"fused_ep_moe_v2_moonep_plan_ep{ep}_e{e_total}_b{b_slots}",
    )

    def unpack(expert_count: jax.Array, group_tokens: jax.Array):
        packed = call(expert_count, group_tokens)

        def take(base_row: int, n_rows: int, length: int):
            return packed[base_row : base_row + n_rows].reshape(-1)[:length]

        split_e_ids = take(r_se, n_se, h_hot)
        split_alloc_end = take(r_sae, n_sae, h_hot * ep).reshape(h_hot, ep)
        split_phys_id = take(r_spid, n_spid, h_hot * ep).reshape(h_hot, ep)
        rdma_recv_ge = take(r_rge, n_rge, ep * b_slots).reshape(ep, b_slots)
        r0_sle = take(r_r0_sle, n_op_rows, n_op_elems).reshape(ep, max_ops)
        r0_dr = take(r_r0_dr, n_op_rows, n_op_elems).reshape(ep, max_ops)
        r0_ds = take(r_r0_ds, n_op_rows, n_op_elems).reshape(ep, max_ops)
        r1_ss = take(r_r1_ss, n_op_rows, n_op_elems).reshape(ep, max_ops)
        r1_dr = take(r_r1_dr, n_op_rows, n_op_elems).reshape(ep, max_ops)
        r1_ds = take(r_r1_ds, n_op_rows, n_op_elems).reshape(ep, max_ops)
        r0_ns = take(r_r0_ns, n_cnt_rows, ep)
        r0_nr = take(r_r0_nr, n_cnt_rows, ep)
        r1_ns = take(r_r1_ns, n_cnt_rows, ep)
        r1_nr = take(r_r1_nr, n_cnt_rows, ep)
        any_migrated = packed[r_meta, :1]

        return (
            split_e_ids,
            split_alloc_end,
            split_phys_id,
            rdma_recv_ge,
            r0_sle,
            r0_dr,
            r0_ds,
            r1_ss,
            r1_dr,
            r1_ds,
            r0_ns,
            r0_nr,
            r1_ns,
            r1_nr,
            any_migrated,
        )

    return unpack


def remap_topk_to_phys(
    topk_idx: jax.Array,
    tpe_prefix_me: jax.Array,
    split_e_ids: jax.Array,
    split_alloc_end: jax.Array,
    split_phys_id: jax.Array,
    any_migrated: jax.Array,
    *,
    epn: int,
    b_slots: int,
    e_total: int,
    e_phys: int,
) -> jax.Array:
    """Zero-gather bit-exact MXU remapper; preserves masked `e_total` as `e_phys`."""
    S, K = topk_idx.shape
    N = S * K
    h_hot = split_e_ids.shape[0]
    ep = split_phys_id.shape[1]
    flat_e = topk_idx.reshape(N).astype(jnp.int32)
    valid_tok = flat_e < jnp.int32(e_total)

    if epn > 0 and (epn & (epn - 1)) == 0:
        shift = epn.bit_length() - 1
        raw_base = flat_e + jnp.right_shift(flat_e, shift) * jnp.int32(b_slots)
    else:
        raw_base = flat_e + (flat_e // jnp.int32(epn)) * jnp.int32(b_slots)
    base_phys = jnp.where(valid_tok, raw_base, jnp.int32(e_phys))

    def do_remap(_):
        # Masked rows have flat_e == e_total and never match split_e_ids (< e_total).
        is_match = flat_e[:, None] == split_e_ids[None, :]
        is_split = jnp.any(is_match, axis=-1)

        BLK = 128
        pad = (-N) % BLK
        m_bf16 = is_match.astype(jnp.bfloat16)
        if pad:
            m_bf16 = jnp.pad(m_bf16, ((0, pad), (0, 0)))
        n_blk = m_bf16.shape[0] // BLK
        m_2d = m_bf16.reshape(n_blk, BLK, h_hot)

        lower_in = jnp.concatenate(
            [
                jnp.ones((1, BLK), jnp.bfloat16),
                jnp.tril(jnp.ones((BLK, BLK), jnp.bfloat16), -1),
            ],
            axis=0,
        )
        both = jnp.einsum(
            "ij,bje->bie", lower_in, m_2d, preferred_element_type=jnp.float32
        )
        blk_sum = both[:, 0, :]
        in_rank = both[:, 1:, :]

        if n_blk <= 512:
            lower_blk = jnp.tril(jnp.ones((n_blk, n_blk), jnp.bfloat16), -1)
            blk_off = jnp.einsum(
                "ab,be->ae",
                lower_blk,
                blk_sum.astype(jnp.bfloat16),
                preferred_element_type=jnp.float32,
            )
        else:
            bs_i = blk_sum.astype(jnp.int32)
            blk_off = (jnp.cumsum(bs_i, axis=0) - bs_i).astype(jnp.float32)

        local_rank = (
            (in_rank + blk_off[:, None, :]).astype(jnp.int32).reshape(-1, h_hot)[:N]
        )

        safe_se = jnp.where(split_e_ids >= 0, split_e_ids, 0)
        hot_tpe_prefix = tpe_prefix_me[safe_se]

        my_global_rank = jnp.sum(
            jnp.where(is_match, local_rank + hot_tpe_prefix[None, :], 0), axis=-1
        )
        is_match_f32 = is_match.astype(jnp.float32)
        my_alloc_end = jnp.einsum(
            "nh,hd->nd",
            is_match_f32,
            split_alloc_end.astype(jnp.float32),
            precision=lax.Precision.HIGHEST,
        ).astype(jnp.int32)
        dest_d = jnp.minimum(
            jnp.sum(my_global_rank[:, None] >= my_alloc_end, axis=-1).astype(jnp.int32),
            jnp.int32(ep - 1),
        )
        my_phys_row = jnp.einsum(
            "nh,hd->nd",
            is_match_f32,
            split_phys_id.astype(jnp.float32),
            precision=lax.Precision.HIGHEST,
        ).astype(jnp.int32)
        dest_oh = dest_d[:, None] == jnp.arange(ep, dtype=jnp.int32)[None, :]
        remapped = jnp.sum(jnp.where(dest_oh, my_phys_row, 0), axis=-1)
        return jnp.where(is_split, remapped, base_phys).reshape(S, K)

    return lax.cond(
        any_migrated[0] > jnp.int32(0),
        do_remap,
        lambda _: base_phys.reshape(S, K),
        operand=None,
    )


def moonep_local_first_visit_list(
    expert_rows: jax.Array, epn: int, g_local: int
) -> jax.Array:
    """Order active local experts (< epn, largest-first) before migrated slots."""
    order = jnp.arange(g_local, dtype=jnp.int32)
    rows_i = expert_rows.astype(jnp.int32)
    is_empty = rows_i <= 0
    is_mig = order >= epn
    tier = jnp.where(is_empty, 2, jnp.where(is_mig, 1, 0)).astype(jnp.int32)
    less = (
        (tier[None, :] < tier[:, None])
        | ((tier[None, :] == tier[:, None]) & (rows_i[None, :] > rows_i[:, None]))
        | (
            (tier[None, :] == tier[:, None])
            & (rows_i[None, :] == rows_i[:, None])
            & (order[None, :] < order[:, None])
        )
    )
    rank = jnp.sum(less.astype(jnp.int32), axis=1)
    return jnp.zeros((g_local,), dtype=jnp.int32).at[rank].set(order)
