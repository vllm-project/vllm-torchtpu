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
"""q38 G1BODY: issue-lean body of the BATCHED LOG-mode factored GDN tile.

VLLM_TPU_GDN_G1_BODY (default off). Design: scratch/g1body/g1body_design_v1.txt
(G1BODY, DSTEP lane). The tile's sequences sit in the sublanes ("L8"): every
per-(sequence, token) row of up to 128 lanes is one [seq_tile, 128] value, so
the conv, SiLU, l2norm, gates, the coefficient products, the window algebra
and the log rows run once per tile instead of once per sequence, on full
vregs instead of one row per vreg. The three per-sequence MXU products of the
served body (`compute_gdn.factored_products` / `factored_key_dots` /
`factored_update_pass`) are issued unchanged, per sequence, on the same
operand values. Per-sequence operand rows and results move between the two
layouts through VMEM scratch: aligned stores plus one sublane-strided load.

Exactness: every element goes through the served op sequence in the served
order (`compute_gdn.factored_gdn`, `compute_conv1d.causal_conv1d`,
`vmem_ldst.load_states_factored`), including the products with exact-zero
coefficients (they decide signed zeros), so the stored bytes are the served
kernel's. The decay products are the served masked folds without their exact
multiplications by 1.0. A sequence without an initial state gets its base
slot zeroed in VMEM before the common path, which is the served
`where(has_initial_state, base, 0)` without a select over the whole state.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.gdn.v3 import compute_gdn, config, vmem_ldst

LANES = 128


def _round8(n: int) -> int:
    return -(-n // 8) * 8


def _bank_stride(n: int) -> int:
    """n (a multiple of 8) bumped by 8 rows when it is a multiple of 32."""
    return n + 8 if n % 32 == 0 else n


class _Geom:
    """Static geometry of a G1 tile (all Python ints)."""

    def __init__(self, cfg: config.GDNConfig):
        plan = cfg.state_plan
        assert cfg.uses_factored_window and plan is not None
        assert cfg.kq_head_dim == cfg.v_head_dim == LANES, (cfg.kq_head_dim, cfg.v_head_dim)
        assert cfg.dim_size % LANES == 0 and cfg.seq_tile_size == 8, (cfg.dim_size, cfg.seq_tile_size)
        self.nb = cfg.seq_tile_size
        self.w = cfg.chunk_size
        self.k = cfg.log_positions
        self.nkq = cfg.num_kq_heads
        self.nv = cfg.num_v_heads
        self.vpk = cfg.v_per_kq_head
        self.nc = cfg.dim_size // LANES
        self.kprev = cfg.prev_kernel_size
        self.n_cols = compute_gdn._num_columns(cfg)
        self.base_col = compute_gdn._base_column(cfg)
        # Conv state / conv log in their bf16 typed views: [rows, view_lanes].
        self.conv_rows = plan.conv.rows_used
        self.view_lanes = self.kprev * cfg.dim_size // self.conv_rows
        assert self.view_lanes % LANES == 0 and cfg.dim_size % self.view_lanes == 0, self.view_lanes
        self.halves = self.view_lanes // LANES
        self.rows_per_conv_row = cfg.dim_size // self.view_lanes
        self.lconv_rows = plan.log_conv.rows_used
        assert self.lconv_rows == self.k * self.rows_per_conv_row, (self.lconv_rows, self.k)
        self.log_rows = plan.log.rows_used
        assert self.log_rows == self.k * (1 + self.nkq + self.nv), self.log_rows
        self.conv_pad = _round8(self.conv_rows)
        # Per-sequence row strides of the L8 staging scratches: a multiple of 8 (aligned per-sequence stores) that is not a
        # multiple of 32 -- a sublane-strided load at a 32-row stride splits into one load per sublane (dev1 dump: 400
        # single-sublane vld per tile at stride 32); at 8 / 24 / 40 it takes two 4-sublane loads.
        self.hist_rows = _bank_stride(self.conv_pad + _round8(self.lconv_rows))
        self.log_pad = _bank_stride(_round8(self.log_rows))
        self.lhs_rows = _round8(4 * (1 + self.k))  # compute_gdn._rows_to_8 of [k_hi, k_hi, k_lo, k_lo]

    def view_pos(self, j: int, cb: int) -> tuple[int, int]:
        """(typed view row, lane half) of conv row j, 128-lane column block cb."""
        return j * self.rows_per_conv_row + cb // self.halves, cb % self.halves


def g1_scratch_shapes(cfg: config.GDNConfig) -> dict:
    """VMEM scratch of a G1 tile (f32 everywhere)."""
    g = _Geom(cfg)
    f32 = jnp.float32
    v = pltpu.VMEM
    return dict(
        hist=v((g.halves, g.nb * g.hist_rows, LANES), f32),
        act=v((g.w * g.nc * 8, LANES), f32),
        conv_out=v((g.halves, (g.conv_pad + _round8(g.lconv_rows)) * 8, LANES), f32),
        vmask=v((g.nv * g.w * 8, LANES), f32),
        qk_l8=v((2 * g.nkq * g.w * 8, LANES), f32),
        qk_nrm=v((2 * g.nkq * 8 * 8, LANES), f32),
        gates=v((2 * g.w * 8, LANES), f32),
        log_in=v((g.nb * g.log_pad, LANES), f32),
        log_dec=v((max(g.k, 1) * 8, LANES), f32),
        log_upd=v((g.nv * max(g.k, 1) * 8, LANES), f32),
        rows=v((g.nkq, 2 * g.w * 8, LANES), f32),
        keys=v((g.nkq, g.n_cols * 8, LANES), f32),
        proj=v((g.nv, g.nb * 8, LANES), f32),
        kk=v((g.nkq, g.nb * 8, LANES), f32),
        c_pre=v((2 * g.w * g.n_cols * 8, LANES), f32),
        coef=v((2 * g.w * g.n_cols * 8, LANES), f32),
        lower=v((g.w * g.w * 8, LANES), f32),
        lhs=v((g.nkq, g.lhs_rows * 8, LANES), f32),
        rows_seq=v((g.nb, g.nkq, 2 * g.w, LANES), f32),
        keys_seq=v((g.nb, g.nkq, g.n_cols, LANES), f32),
        lhs_seq=v((g.nb, g.nkq, g.lhs_rows, LANES), f32),
        rhs_seq=v((g.nb, g.nv, g.lhs_rows, LANES), f32),
        dec_seq=v((g.nb, 8, LANES), f32),
        rhs=v((g.nv, g.lhs_rows * 8, LANES), f32),
        decay=v((8, LANES), f32),
        out=v((g.w * 8 * 8, LANES), f32),
        log_out=v((g.log_pad * 8, LANES), f32),
    )


def _l8(ref, start: int, nb: int, stride: int):
    """Rows start, start + stride, ... (nb of them) of a 2D [rows, 128] ref."""
    return ref[pl.ds(start, nb, stride=stride), :]


def _seq_rows(ref, s: int, n: int):
    """Sequence s's n rows of a slot-major scratch (slot i at rows i * 8 .. i * 8 + 7)."""
    return ref[pl.ds(s, n, stride=8), :]


def _sub_vec(vals, nb: int):
    """[nb, 128] int32 holding vals[i] in sublane i (built at the operand shape)."""
    io = jax.lax.broadcasted_iota(jnp.int32, (nb, LANES), 0)
    out = jnp.zeros((nb, LANES), jnp.int32)
    for i, x in enumerate(vals):
        out = jnp.where(io == i, jnp.asarray(x, jnp.int32), out)
    return out


def _lane_bcast(x, h: int):
    """[nb, 128] -> every lane = lane h."""
    return jnp.broadcast_to(x[:, h : h + 1], x.shape)


def coefficient_table(cfg: config.GDNConfig):
    """Static view of `compute_gdn.factored_masks`: per (row r, column j) of the
    coefficient matrix, (valid, t_j, p_r). The served masked fold multiplies the
    decays m with t_j < m <= p_r in chronological order (the other factors are
    exactly 1.0), starting from `valid`."""
    g = _Geom(cfg)
    w, k = g.w, g.k
    out = {}
    for r in range(2 * w):
        p = k + (r if r < w else r - w)
        for j in range(g.n_cols):
            if j < w:
                t = k + j
            elif j < w + k:
                t = j - w
            elif j == w + k:
                t = -1
            else:
                t = 1 << 20
            valid = t < p + (0 if r < w else 1)
            out[(r, j)] = (valid, t, p)
    return out


def g1_factored_tile(
    *,
    cfg: config.GDNConfig,
    metadata_ref,
    p_id,
    qkv_ref,  # [seq, W, 1, dim] f32 (compact window rows)
    b_ref,  # [seq, W, 1, lanes]
    a_ref,  # [seq, W, 1, lanes]
    conv_in_ref,
    rec_in_ref,
    log_in_refs,
    out_ref,  # [seq, W, n_v, d_v]
    conv_out_ref,
    rec_out_ref,
    log_out_refs,
    weights_ref,
    scr: dict,
) -> None:
    """One BATCHED LOG-mode factored tile: loads, conv, GDN window, stores.

    Stages, each leaving its results in VMEM scratch so few values stay live across stages (design P4):
    A1 conv per column block, A2 masks / l2norm / gates, A3 log rows, B per-sequence products, C coefficients and the window
    algebra per head, D per-sequence update pass and state store.
    """
    g = _Geom(cfg)
    plan = cfg.state_plan
    nb, w, k, nkq, nv, vpk, nc = g.nb, g.w, g.k, g.nkq, g.nv, g.vpk, g.nc
    f32 = jnp.float32
    seqs = range(nb)

    # ---- per-sequence scalars (SMEM), as load_states_factored reads them.
    s_idx = [metadata_ref.p_id_to_s_idx[p_id, i] for i in seqs]
    has_init = [metadata_ref.s_idx_has_initial_state[s_idx[i]] for i in seqs]
    read_off = [metadata_ref.s_idx_to_read_offset[s_idx[i]] for i in seqs]
    real = [metadata_ref.p_id_to_r_size[p_id, i] for i in seqs]
    counts = [jnp.where(has_init[i], read_off[i], 0) for i in seqs]
    ro_v = _sub_vec(read_off, nb)
    hi_v = _sub_vec([jnp.where(has_init[i], 1, 0) for i in seqs], nb)
    rs_v = _sub_vec(real, nb)
    cnt_v = _sub_vec(counts, nb)
    lane = jax.lax.broadcasted_iota(jnp.int32, (nb, LANES), 1)
    zeros = jnp.zeros((nb, LANES), f32)
    ones = jnp.ones((nb, LANES), f32)

    # ---- g_ld: a sequence without an initial state reads a zero base.
    for i in seqs:

        @pl.when(jnp.logical_not(jnp.asarray(has_init[i], jnp.bool_)))
        def _(i=i):
            vmem_ldst.store_state_region(
                rec_in_ref.at[i, 0],
                plan.recurrent,
                jnp.zeros((nv, cfg.kq_head_dim, cfg.v_head_dim), f32),
            )

    # Conv state + conv log: per sequence typed rows -> f32 staging (L8 rows are read back with one strided load).
    hist = scr["hist"]
    for i in seqs:
        conv_rows = vmem_ldst.load_state_region(
            conv_in_ref.at[i, 0], plan.conv, (g.conv_rows, g.view_lanes)
        ).astype(f32)
        lconv_rows = vmem_ldst.load_state_region(
            log_in_refs[1].at[i, 0], plan.log_conv, (g.lconv_rows, g.view_lanes)
        ).astype(f32)
        for hv in range(g.halves):
            lanes_h = slice(hv * LANES, (hv + 1) * LANES)
            hist.at[hv][pl.ds(i * g.hist_rows, g.conv_rows), :] = conv_rows[:, lanes_h]
            hist.at[hv][pl.ds(i * g.hist_rows + g.conv_pad, g.lconv_rows), :] = lconv_rows[:, lanes_h]

    def hist_row(j, cb):
        """History row j (0..kprev-1 conv state, kprev.. conv log) of block cb, L8."""
        if j < g.kprev:
            v, hv = g.view_pos(j, cb)
        else:
            v, hv = g.view_pos(j - g.kprev, cb)
            v += g.conv_pad
        return _l8(hist.at[hv], v, nb, g.hist_rows)

    # ---- stage A1, per column block: fold_conv_log + where(has_initial_state), causal conv, SiLU, conv out rows.
    qflat = qkv_ref.reshape(nb * w * nc, LANES)
    cw = weights_ref.conv.weight
    act, cso = scr["act"], scr["conv_out"]
    for cb in range(nc):
        hrows = [hist_row(j, cb) for j in range(g.kprev + k)]
        prev = []
        for j in range(g.kprev):
            x = hrows[j]
            for jj in range(k):
                x = jnp.where(jj < ro_v, hrows[j + jj + 1], x)
            prev.append(jnp.where(hi_v != 0, x, 0))
        xin = [_l8(qflat, t * nc + cb, nb, w * nc) for t in range(w)]
        lhs_rows = prev + xin  # kprev history rows, then the W window rows
        wks = [cw[kk, :, cb * LANES : (cb + 1) * LANES].astype(f32) for kk in range(cfg.kernel_size)]
        for t in range(w):
            y = jnp.zeros((nb, LANES), f32)
            for kk in range(cfg.kernel_size):
                y += lhs_rows[t + kk] * wks[kk]
            act_t = jax.nn.silu(y)
            act[pl.ds((t * nc + cb) * 8, 8), :] = act_t
            # The served body (vmem_ldst.load_activation_as_large) stores the SiLU output back into the compact qkv slot. A
            # masked row of a later tile that reuses the slot is not refilled, so served logs those bytes as its conv input
            # (qkv_rows[idx, 1:]): store the same bytes here (the xin rows above are already loaded).
            qflat[pl.ds(t * nc + cb, nb, stride=w * nc), :] = act_t
        # Conv checkpoint 0 = rows 1..kprev of [history, window]; conv log = window rows 1..K.
        for j in range(g.kprev):
            v, hv = g.view_pos(j, cb)
            cso.at[hv][pl.ds(v * 8, 8), :] = lhs_rows[1 + j]
        for j in range(k):
            v, hv = g.view_pos(j, cb)
            cso.at[hv][pl.ds((g.conv_pad + v) * 8, 8), :] = xin[1 + j]
    for i in seqs:
        cs = jnp.concatenate([_seq_rows(cso.at[hv], i, g.conv_rows) for hv in range(g.halves)], axis=1)
        vmem_ldst.store_state_region(conv_out_ref.at[i, 0], plan.conv, cs)
        lc = jnp.concatenate(
            [cso.at[hv][pl.ds(g.conv_pad * 8 + i, g.lconv_rows, stride=8), :] for hv in range(g.halves)],
            axis=1,
        )
        vmem_ldst.store_state_region(log_out_refs[1].at[i, 0], plan.log_conv, lc)

    def act_row(t, cb):
        return act[pl.ds((t * nc + cb) * 8, 8), :]

    # ---- stage A2 (L8): masks, l2norm, gates, as factored_gdn.
    keys_row, updates_row, _ = config.log_layout_rows(k, nkq, nv)
    trows, tkeys, tlog = scr["rows"], scr["keys"], scr["log_out"]
    vm, gates = scr["vmask"], scr["gates"]
    masks = [t < rs_v for t in range(w)]
    # l2norm runs on the served value shape [seq, n_kq, W, d] (the served reduction, U3 closed by construction): masked L8
    # rows -> per-sequence [W, d] slabs (aligned store + strided load) -> l2_norm -> slabs -> L8 rows (strided load).
    qkl8, qknrm = scr["qk_l8"], scr["qk_nrm"]
    for t in range(w):
        for h in range(nkq):
            qkl8[pl.ds((h * w + t) * 8, 8), :] = jnp.where(masks[t], act_row(t, h), 0)
            qkl8[pl.ds(((nkq + h) * w + t) * 8, 8), :] = jnp.where(masks[t], act_row(t, nkq + h), 0)

    def slabs(qk):
        return jnp.stack(
            [
                jnp.stack([qkl8[pl.ds((qk * nkq + h) * w * 8 + i, w, stride=8), :] for h in range(nkq)], axis=0)
                for i in seqs
            ],
            axis=0,
        )

    q_large = compute_gdn.l2_norm(slabs(0)) * cfg.kq_head_dim**-0.5
    k_large = compute_gdn.l2_norm(slabs(1))
    for i in seqs:
        for h in range(nkq):
            qknrm[pl.ds((h * 8 + i) * 8, w), :] = q_large[i, h]
            qknrm[pl.ds(((nkq + h) * 8 + i) * 8, w), :] = k_large[i, h]
    for t in range(w):
        for h in range(nkq):
            qn = qknrm[pl.ds(h * 64 + t, 8, stride=8), :]
            kn = qknrm[pl.ds((nkq + h) * 64 + t, 8, stride=8), :]
            trows.at[h][pl.ds((w + t) * 8, 8), :] = qn  # [k_c, q_c] rows of factored_products
            trows.at[h][pl.ds(t * 8, 8), :] = kn
            tkeys.at[h][pl.ds(t * 8, 8), :] = kn  # window keys of factored_products' keys
            if t >= 1:
                tlog[pl.ds((keys_row + (t - 1) * nkq + h) * 8, 8), :] = kn
        for h in range(nv):
            vm[pl.ds((h * w + t) * 8, 8), :] = jnp.where(masks[t], act_row(t, 2 * nkq + h), 0)
    pad = cfg.aligned_num_v_heads - nv
    a_log = jnp.pad(weights_ref.gdn.a_log[...], ((0, pad))).reshape(1, -1).astype(f32)
    dt_bias = jnp.pad(weights_ref.gdn.dt_bias[...], ((0, pad))).reshape(1, -1).astype(f32)
    bflat = b_ref.reshape(nb * w, LANES)
    aflat = a_ref.reshape(nb * w, LANES)
    for t in range(w):
        bt = _l8(bflat, t, nb, w).astype(f32)
        at = _l8(aflat, t, nb, w).astype(f32)
        gates[pl.ds(t * 8, 8), :] = jnp.where(masks[t], jax.nn.sigmoid(bt), 0)  # beta
        gating_log = -jnp.exp(a_log) * jax.nn.softplus(at + dt_bias)
        dec_t = jnp.exp(jnp.where(masks[t], gating_log, 0))
        gates[pl.ds((w + t) * 8, 8), :] = dec_t
        if t >= 1:
            tlog[pl.ds((t - 1) * 8, 8), :] = dec_t

    def beta_row(t):
        return gates[pl.ds(t * 8, 8), :]

    # ---- stage A3 (L8): split_log_rows on every sequence at once.
    lgs, ldec, lupd = scr["log_in"], scr["log_dec"], scr["log_upd"]
    for i in seqs:
        lgs[pl.ds(i * g.log_pad, g.log_rows), :] = vmem_ldst.load_state_region(
            log_in_refs[0].at[i, 0], plan.log, (g.log_rows, cfg.v_head_dim)
        )
    for j in range(k):
        live = j < cnt_v
        ldec[pl.ds(j * 8, 8), :] = jnp.where(live, _l8(lgs, j, nb, g.log_pad), 1.0)
        for h in range(nkq):
            tkeys.at[h][pl.ds((w + j) * 8, 8), :] = jnp.where(live, _l8(lgs, keys_row + j * nkq + h, nb, g.log_pad), 0.0)
        for h in range(nv):
            lupd[pl.ds((h * k + j) * 8, 8), :] = jnp.where(live, _l8(lgs, updates_row + j * nv + h, nb, g.log_pad), 0.0)
    for h in range(nkq):
        for j in range(w + k, g.n_cols):
            tkeys.at[h][pl.ds(j * 8, 8), :] = zeros

    # ---- stage B (per sequence): the served products and key dots.
    tproj, tkk = scr["proj"], scr["kk"]
    rows_seq, keys_seq = scr["rows_seq"], scr["keys_seq"]
    for i in seqs:  # per-sequence operand rows (strided loads, static), so the loop below indexes contiguous blocks
        rows_seq[i] = jnp.stack([_seq_rows(trows.at[h], i, 2 * w) for h in range(nkq)], axis=0)
        keys_seq[i] = jnp.stack([_seq_rows(tkeys.at[h], i, g.n_cols) for h in range(nkq)], axis=0)

    def stage_b(i, carry):  # one sequence's products (i static: the stage is unrolled over the tile's sequences)
        rows_i = rows_seq[i]
        keys_i = keys_seq[i]
        base_i = vmem_ldst._load_recurrent_state(rec_in_ref, cfg, i)
        proj_i = compute_gdn.factored_base_product(jnp.repeat(rows_i, vpk, axis=0), base_i)
        kk_i = compute_gdn.factored_key_dots(rows_i, keys_i)
        r0 = i * 8
        for h in range(nv):
            tproj.at[h][pl.ds(r0, 2 * w), :] = proj_i[h]
        for h in range(nkq):
            tkk.at[h][pl.ds(r0, 2 * w), pl.ds(0, g.n_cols)] = kk_i[h]
        return carry

    # Unrolled (G1-HELPER group 8, g1h-cen1): the scheduler overlaps the sequences' MXU streams (fori_loop: 124 bundles a
    # sequence with no overlap, M).
    for i in seqs:
        stage_b(i, 0)

    # ---- stage C1 (L8, head h in lane h): every coefficient, staged in scratch.
    def dcol(m):  # decays in chronological order (logged, then window)
        return ldec[pl.ds(m * 8, 8), :] if m < k else gates[pl.ds((w + m - k) * 8, 8), :]

    table = coefficient_table(cfg)
    chains: dict = {}

    def chain(t, p):
        """Product of the decays m, t < m <= p, left to right (the served fold)."""
        if (t, p) not in chains:
            chains[(t, p)] = dcol(p) if p - 1 == t else chain(t, p - 1) * dcol(p)
        return chains[(t, p)]

    def c_pre_val(r, j):
        valid, t, p = table[(r, j)]
        if not valid:
            return zeros
        return ones if t == p else chain(t, p)

    cpre_s, coef_s, low_s = scr["c_pre"], scr["coef"], scr["lower"]
    ncols = g.n_cols
    update_cols = compute_gdn._update_columns(cfg)
    # Pre-KK coefficients used: the base decay of every row, and query row 0's (coef_row) update columns.
    needed_cpre = {(r, g.base_col) for r in range(2 * w)} | {(w, c) for c in update_cols}
    for r in range(2 * w):
        kk_r = [_l8(tkk.at[h], r, nb, 8) for h in range(nkq)]
        for j in range(ncols):
            need_coef = j < w + k  # window and logged update columns
            if not need_coef and (r, j) not in needed_cpre:
                continue
            cp = c_pre_val(r, j)
            if (r, j) in needed_cpre:
                cpre_s[pl.ds((r * ncols + j) * 8, 8), :] = cp
            if need_coef:
                kkh = _lane_bcast(kk_r[0], j)
                for h in range(1, nkq):
                    kkh = jnp.where(lane >= h * vpk, _lane_bcast(kk_r[h], j), kkh)
                cf = cp * kkh  # coef * repeat(KK) of factored_window_vectors
                coef_s[pl.ds((r * ncols + j) * 8, 8), :] = cf
                if r < w and j < w:
                    low_s[pl.ds((r * w + j) * 8, 8), :] = beta_row(r) * cf  # lower = beta * coef

    def cpre(r, j, h):
        return _lane_bcast(cpre_s[pl.ds((r * ncols + j) * 8, 8), :], h)

    def coef(r, j, h):
        return _lane_bcast(coef_s[pl.ds((r * ncols + j) * 8, 8), :], h)

    def lower(r, c, h):
        return _lane_bcast(low_s[pl.ds((r * w + c) * 8, 8), :], h)

    # ---- stage C2 (L8, per head): window algebra, outputs, log rows, update terms.
    tout, tlhs, trhs, tdec = scr["out"], scr["lhs"], scr["rhs"], scr["decay"]
    for h in range(nv):
        partial = []
        for r in range(2 * w):
            x = cpre(r, g.base_col, h) * _l8(tproj.at[h], r, nb, 8)
            for j in range(k):
                x = x + coef(r, w + j, h) * lupd[pl.ds((h * k + j) * 8, 8), :]
            partial.append(x)
        vn = [_lane_bcast(beta_row(c), h) * (vm[pl.ds((h * w + c) * 8, 8), :] - partial[c]) for c in range(w)]
        for c in range(w - 1):
            tc = vn[c]
            vn = [vn[rr] - lower(rr, c, h) * tc for rr in range(w)]
        for r in range(w):
            o = partial[w + r]
            for c in range(w):
                o = o + coef(w + r, c, h) * vn[c]
            tout[pl.ds((r * 8 + h) * 8, 8), :] = o
        for j in range(k):
            tlog[pl.ds((updates_row + j * nv + h) * 8, 8), :] = vn[1 + j]
        # weighted terms of factored_update_pass, then the bf16 split.
        upd = [vn[0]] + [lupd[pl.ds((h * k + j) * 8, 8), :] for j in range(k)]
        wts = [cpre(w, c, h) * u for c, u in zip(update_cols, upd)]
        w_hi, w_lo = zip(*(compute_gdn._bf16_split(x) for x in wts))
        slot_vals = list(w_hi) + list(w_lo) + list(w_hi) + list(w_lo)
        for sl, x in enumerate(slot_vals):
            trhs.at[h][pl.ds(sl * 8, 8), :] = x
        for sl in range(len(slot_vals), g.lhs_rows):
            trhs.at[h][pl.ds(sl * 8, 8), :] = zeros
    for h in range(nkq):
        key_rows = [tkeys.at[h][pl.ds(0, 8), :]] + [tkeys.at[h][pl.ds((w + j) * 8, 8), :] for j in range(k)]
        k_hi, k_lo = zip(*(compute_gdn._bf16_split(x) for x in key_rows))
        slot_vals = list(k_hi) + list(k_hi) + list(k_lo) + list(k_lo)
        for sl, x in enumerate(slot_vals):
            tlhs.at[h][pl.ds(sl * 8, 8), :] = x
        for sl in range(len(slot_vals), g.lhs_rows):
            tlhs.at[h][pl.ds(sl * 8, 8), :] = zeros
    tdec[...] = cpre_s[pl.ds((w * ncols + g.base_col) * 8, 8), :]

    # ---- stores of the window outputs and the log rows (per sequence).
    for i in seqs:
        for r in range(w):
            out_ref[i, r] = tout[pl.ds(r * 64 + i, nv, stride=8), :].astype(out_ref.dtype)
        vmem_ldst.store_state_region(log_out_refs[0].at[i, 0], plan.log, _seq_rows(tlog, i, g.log_rows))

    # ---- stage D (per sequence, one at a time): the served update pass and the state store.
    lhs_seq, rhs_seq, dec_seq = scr["lhs_seq"], scr["rhs_seq"], scr["dec_seq"]
    for i in seqs:
        lhs_seq[i] = jnp.stack([_seq_rows(tlhs.at[h], i, g.lhs_rows) for h in range(nkq)], axis=0)
        rhs_seq[i] = jnp.stack([_seq_rows(trhs.at[h], i, g.lhs_rows) for h in range(nv)], axis=0)
        dec_seq[i] = jnp.broadcast_to(tdec[pl.ds(i, 1), :], (8, LANES))

    def stage_d(i, carry):
        lhs = jnp.repeat(lhs_seq[i], vpk, axis=0)
        rhs = rhs_seq[i]
        drow = dec_seq[i]
        decay_i = jnp.stack([drow[0:1, h : h + 1] for h in range(nv)], axis=0)  # [n_v, 1, 1]
        base_i = vmem_ldst._load_recurrent_state(rec_in_ref, cfg, i)
        state_i = base_i * decay_i + jax.lax.dot(
            lhs,
            rhs,
            dimension_numbers=(((1,), (1,)), ((0,), (0,))),
            preferred_element_type=jnp.float32,
        )
        vmem_ldst.store_state_region(rec_out_ref.at[i, 0], plan.recurrent, state_i)
        return carry

    for i in seqs:  # unrolled, as stage B
        stage_d(i, 0)
