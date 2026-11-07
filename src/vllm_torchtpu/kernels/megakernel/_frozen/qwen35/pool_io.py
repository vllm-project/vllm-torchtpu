"""Unified block pool (`raw[0..14]`) gather and scatter kernels for `qwen3_5_moe`.

In `vllm-torchtpu`, Qwen3.5-397B-A17B stores both full-attention FP8 KV pages and GatedDeltaNet
(GDN) recurrent (`S`) + causal-conv states in 15 per-attention-group raw pools (`raw[0..14]`), each
of per-rank shape `(num_pages, 2, 64, 4, 128) fp8_e4m3fn` (64 KiB per kernel page).

Because XLA TPU never places outputs of a communicating Pallas call (`collective_id` set) into
live-out HBM buffers, aliasing `raw[0..14]` inside the communicating decode megakernel would force
XLA to copy the entire ~36 GiB pool every step. Instead:
  1. `gather_gdn_states` (`collective_id=None`, pre-kernel) gathers the initial `(conv, rec)`
     states for the `NS` sequences from `ckpt_indices[s, read_offsets[s]]` in `raw[0..14]`.
  2. The communicating decode megakernel (`paged_kv=True`) reads paged FP8 KV directly from
     `raw[0..14]` and outputs `(hidden, conv, rec, knew, vnew)` with `knew`, `vnew` in `fp8_e4m3fn`.
  3. `scatter_pool_states` (`collective_id=None`, `input_output_aliases` on all `raw[0..14]`)
     scatters the `B` new FP8 K/V tokens into their destination KV pages/lanes and packs all `T`
     per-token GDN `(rec, conv)` states into checkpoint slots `ckpt_indices[s, 1..T]` in place.
"""

from __future__ import annotations

import dataclasses
import math

import jax
from jax import lax
import jax._src.state.discharge as _discharge
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np

from .megakernel import (
    BF16,
    F32,
    FP8,
    I32,
    MKConfig,
    O_NAMES,
    Tiles,
    U32,
    W_NAMES,
    build,
    default_tiles,
    plan,
    state_shapes,
)

# ---------------------------------------------------------------------------
# CPU Pallas interpret=True monkey-patch for chained Ref.bitcast(...).reshape(...) writes
# ---------------------------------------------------------------------------
if not getattr(_discharge.transform_swap_array, "_pool_io_patched", False):
    _orig_swap = _discharge.transform_swap_array

    def _patched_swap(x, transforms, val):
        if not transforms or all(
            isinstance(t, _discharge.indexing.NDIndexer) for t in transforms
        ):
            return _orig_swap(x, transforms, val)
        intermediates = [x]
        new_val = x
        for t in transforms:
            new_val = _discharge.transform_array(new_val, [t])
            intermediates.append(new_val)
        if val is None:
            return new_val, x
        new_x = val
        for inter, t in reversed(list(zip(intermediates[:-1], transforms))):
            if isinstance(t, _discharge.indexing.NDIndexer):
                _, new_x = _orig_swap(inter, [t], new_x)
            elif isinstance(t, _discharge.BitcastTransform):
                new_x = _discharge.bitcast(new_x, inter.dtype)
            elif isinstance(t, _discharge.ReshapeTransform):
                new_x = new_x.reshape(inter.shape)
            else:
                _, new_x = _orig_swap(inter, [t], new_x)
        return new_val, new_x

    _patched_swap._pool_io_patched = True
    _discharge.transform_swap_array = _patched_swap


# ---------------------------------------------------------------------------
# Kernel 1: Pre-Megakernel GDN State Gather (`gather_gdn_states`)
# ---------------------------------------------------------------------------
def gather_gdn_states(
    c: MKConfig,
    raw_pools,
    ckpt_indices,
    read_offsets,
    active_seqs=None,
    *,
    interpret=False,
):
    """Gather initial `(conv, rec)` GDN states for `c.NS` sequences from `raw_pools`.

    Args:
      c: `MKConfig` geometry.
      raw_pools: `c.NATT`-tuple of per-rank `(num_pages, 2, 64, 4, 128) fp8` pool arrays.
      ckpt_indices: `(c.NS, K) int32` checkpoint slot table (`K >= c.T + 1`).
      read_offsets: `(c.NS,) int32` index into `ckpt_indices[s, :]` (`0 <= read_offsets[s] < K`).
      active_seqs: optional `(c.NS,) int32` mask (non-zero for active sequences; defaults to 1s).
      interpret: run Pallas in CPU interpret mode.

    Returns:
      `(conv, rec)` with shapes matching `state_shapes(c)["conv"]` and `state_shapes(c)["rec"]`.
    """
    c.check()
    assert len(raw_pools) == c.NATT, (len(raw_pools), c.NATT)
    if active_seqs is None:
        act_mode = "all"
    elif isinstance(active_seqs, str) and active_seqs == "from_ckpt":
        act_mode = "from_ckpt"
    else:
        act_mode = "ref"
        active_seqs = active_seqs.astype(I32)

    sh = state_shapes(dataclasses.replace(c, paged_kv=False))
    conv_shape, conv_dt = sh["conv"]
    rec_shape, rec_dt = sh["rec"]
    ngdn = rec_shape[0]
    AE = c.att_every
    G_PER_ATT = AE - 1
    NS, B, T = c.NS, c.B, c.T
    vh, KD, VD, QKV = c.vh, c.KD, c.VD, c.QKV
    ssm_u32_rows = (vh * KD) // 2
    conv_rows = (3 * QKV) // 128
    conv_rows_pad = ((conv_rows + 15) // 16) * 16
    qkv_128 = QKV // 128
    direct_conv = T > 1 and ((c.HR * qkv_128) % 16 == 0 or bool(interpret))
    hr_rows_128 = c.HR * qkv_128
    conv_raw_shape = conv_shape if direct_conv else (ngdn, NS * conv_rows_pad, 128)
    conv_raw_dt = conv_dt if direct_conv else BF16
    ckpt_is_3d = ckpt_indices.ndim == 3
    roff_is_2d = read_offsets.ndim == 2
    n_pairs = NS // 2

    coal_rec = (G_PER_ATT > 0) and (NS * vh <= 128)
    rec_coal = max(1, G_PER_ATT) if coal_rec else 1

    scratch = {
        "page_buf": pltpu.VMEM((2, 5, 2, 64, 4, c.page_size), FP8),
        "rec_buf": pltpu.VMEM((2, rec_coal, *rec_shape[1:]), rec_dt),
        "conv_buf": pltpu.VMEM((2, max(1, G_PER_ATT), *conv_raw_shape[1:]), conv_raw_dt),
        "sem_in": pltpu.SemaphoreType.DMA((2,)),
        "sem_out": pltpu.SemaphoreType.DMA((2, 2)),
    }
    s_names = list(scratch)
    n_smem = 3 if act_mode == "ref" else 2

    def kernel(*refs):
        raw_refs = refs[:c.NATT]
        if act_mode == "ref":
            ckpt_ref, roff_ref, act_ref = refs[c.NATT:c.NATT + 3]
        else:
            ckpt_ref, roff_ref = refs[c.NATT:c.NATT + 2]
            act_ref = None
        conv_out_ref, rec_out_ref = refs[c.NATT + n_smem:c.NATT + n_smem + 2]
        scr = dict(zip(s_names, refs[c.NATT + n_smem + 2:]))
        page_buf = scr["page_buf"]
        rec_buf = scr["rec_buf"]
        conv_buf = scr["conv_buf"]
        sem_in = scr["sem_in"]
        sem_out = scr["sem_out"]
        conv_row_iota = (
            lax.broadcasted_iota(I32, (c.HR, QKV), 0) if direct_conv else None
        )

        for slot in range(2):
            conv_buf[slot] = jnp.zeros(conv_buf.shape[1:], conv_raw_dt)
            rec_buf[slot] = jnp.zeros(rec_buf.shape[1:], rec_dt)

        def is_active(s):
            if act_mode == "all":
                return jnp.bool_(True)
            if act_mode == "from_ckpt":
                return (ckpt_ref[0, s, 0] > 0) if ckpt_is_3d else (ckpt_ref[s, 0] > 0)
            return act_ref[s] != 0

        def page_base_for(sub, s):
            roff = roff_ref[sub, s] if roff_is_2d else roff_ref[s]
            if ckpt_is_3d:
                slot = ckpt_ref[sub, s, roff]
                return slot * 5
            else:
                slot = ckpt_ref[s, roff]
                return slot * 15 + sub * 5

        def start_page_read(raw_ref, sub, s, p_slot):
            @pl.when(is_active(s))
            def _():
                pb = page_base_for(sub, s)
                pltpu.make_async_copy(
                    raw_ref.at[pl.ds(pb, 5)], page_buf.at[p_slot], sem_in.at[p_slot]
                ).start()

        def finish_page_read(raw_ref, sub, s, p_slot, g_slot, r_slot, r_sub):
            @pl.when(is_active(s))
            def _():
                pb = page_base_for(sub, s)
                pltpu.make_async_copy(
                    raw_ref.at[pl.ds(pb, 5)], page_buf.at[p_slot], sem_in.at[p_slot]
                ).wait()

                # Unpack SSM state: pages 0..3 -> (ssm_u32_rows, 128) u32 -> (vh, KD, VD) bf16
                u32_val = (
                    page_buf.at[p_slot, pl.ds(0, 4)]
                    .bitcast(U32)
                    .reshape(512, 128)[:ssm_u32_rows, :]
                )
                s_km = pltpu.bitcast(u32_val, BF16).reshape(vh, KD, VD)
                h0 = s * vh if isinstance(s, int) else pl.multiple_of(s * vh, vh)
                rec_buf[r_slot, r_sub, pl.ds(h0, vh)] = s_km

                if direct_conv:
                    raw_hr = (
                        page_buf.at[p_slot, 4]
                        .bitcast(BF16)
                        .reshape(256, 128)[:hr_rows_128, :]
                        .reshape(c.HR, QKV)
                        .astype(conv_dt)
                    )
                    conv_buf[g_slot, sub, s] = jnp.where(conv_row_iota < 3, raw_hr, 0.0)
                else:
                    # Unpack Conv state: page 4 -> (conv_rows_pad, 128) bf16 (tile-aligned)
                    c0 = (
                        s * conv_rows_pad
                        if isinstance(s, int)
                        else pl.multiple_of(s * conv_rows_pad, conv_rows_pad)
                    )
                    conv_buf[g_slot, sub, pl.ds(c0, conv_rows_pad), :] = (
                        page_buf.at[p_slot, 4]
                        .bitcast(BF16)
                        .reshape(256, 128)[:conv_rows_pad, :]
                    )

        total_gdn = c.NATT * G_PER_ATT
        if total_gdn > 0 and NS >= 1:
            start_page_read(raw_refs[0], 0, 0, 0)

        for g in range(c.NATT):
            raw_ref = raw_refs[g]
            g_slot = g & 1
            if G_PER_ATT > 0 and g >= 2:
                o_prev = (g - 2) * G_PER_ATT
                pltpu.make_async_copy(
                    conv_buf.at[g_slot],
                    conv_out_ref.at[pl.ds(o_prev, G_PER_ATT)],
                    sem_out.at[g_slot, 0],
                ).wait()
                if coal_rec:
                    pltpu.make_async_copy(
                        rec_buf.at[g_slot],
                        rec_out_ref.at[pl.ds(o_prev, G_PER_ATT)],
                        sem_out.at[g_slot, 1],
                    ).wait()

            for sub in range(G_PER_ATT):
                o = g * G_PER_ATT + sub
                r_slot = g_slot if coal_rec else (o & 1)
                r_sub = sub if coal_rec else 0
                if not coal_rec and o >= 2:
                    pltpu.make_async_copy(
                        rec_buf.at[r_slot, 0],
                        rec_out_ref.at[o - 2],
                        sem_out.at[r_slot, 1],
                    ).wait()

                next_o = o + 1
                has_next_o = next_o < total_gdn
                next_raw_ref = raw_refs[next_o // G_PER_ATT] if has_next_o else None
                next_sub = (next_o % G_PER_ATT) if has_next_o else 0

                if NS == 1:
                    p_cur = o & 1
                    p_nxt = (o + 1) & 1
                    if has_next_o:
                        start_page_read(next_raw_ref, next_sub, 0, p_nxt)
                    finish_page_read(raw_ref, sub, 0, p_cur, g_slot, r_slot, r_sub)
                elif NS % 2 == 0:
                    if n_pairs == 1:
                        start_page_read(raw_ref, sub, 1, 1)
                        finish_page_read(raw_ref, sub, 0, 0, g_slot, r_slot, r_sub)
                        if has_next_o:
                            start_page_read(next_raw_ref, next_sub, 0, 0)
                        finish_page_read(raw_ref, sub, 1, 1, g_slot, r_slot, r_sub)
                    else:
                        def gather_pair_even(
                            pair,
                            _,
                            raw_ref=raw_ref,
                            sub=sub,
                            g_slot=g_slot,
                            r_slot=r_slot,
                            r_sub=r_sub,
                            has_next_o=has_next_o,
                            next_raw_ref=next_raw_ref,
                            next_sub=next_sub,
                        ):
                            s0 = 2 * pair
                            s1 = s0 + 1
                            start_page_read(raw_ref, sub, s1, 1)
                            finish_page_read(raw_ref, sub, s0, 0, g_slot, r_slot, r_sub)

                            @pl.when(s0 + 2 < NS)
                            def _():
                                start_page_read(raw_ref, sub, s0 + 2, 0)

                            if has_next_o:
                                @pl.when(s0 + 2 == NS)
                                def _():
                                    start_page_read(next_raw_ref, next_sub, 0, 0)

                            finish_page_read(raw_ref, sub, s1, 1, g_slot, r_slot, r_sub)
                            return 0

                        lax.fori_loop(0, n_pairs, gather_pair_even, 0)
                else:
                    p0 = o & 1
                    p1 = 1 - p0
                    if n_pairs == 1:
                        start_page_read(raw_ref, sub, 1, p1)
                        finish_page_read(raw_ref, sub, 0, p0, g_slot, r_slot, r_sub)
                        start_page_read(raw_ref, sub, 2, p0)
                        finish_page_read(raw_ref, sub, 1, p1, g_slot, r_slot, r_sub)
                    else:
                        def gather_pair_odd(
                            pair,
                            _,
                            raw_ref=raw_ref,
                            sub=sub,
                            g_slot=g_slot,
                            r_slot=r_slot,
                            r_sub=r_sub,
                            p0=p0,
                            p1=p1,
                        ):
                            s0 = 2 * pair
                            s1 = s0 + 1
                            start_page_read(raw_ref, sub, s1, p1)
                            finish_page_read(raw_ref, sub, s0, p0, g_slot, r_slot, r_sub)
                            start_page_read(raw_ref, sub, s0 + 2, p0)
                            finish_page_read(raw_ref, sub, s1, p1, g_slot, r_slot, r_sub)
                            return 0

                        lax.fori_loop(0, n_pairs, gather_pair_odd, 0)

                    if has_next_o:
                        start_page_read(next_raw_ref, next_sub, 0, p1)
                    finish_page_read(raw_ref, sub, NS - 1, p0, g_slot, r_slot, r_sub)

                if not coal_rec:
                    pltpu.make_async_copy(
                        rec_buf.at[r_slot, 0],
                        rec_out_ref.at[o],
                        sem_out.at[r_slot, 1],
                    ).start()

            if G_PER_ATT > 0:
                o_base = g * G_PER_ATT
                pltpu.make_async_copy(
                    conv_buf.at[g_slot],
                    conv_out_ref.at[pl.ds(o_base, G_PER_ATT)],
                    sem_out.at[g_slot, 0],
                ).start()
                if coal_rec:
                    pltpu.make_async_copy(
                        rec_buf.at[g_slot],
                        rec_out_ref.at[pl.ds(o_base, G_PER_ATT)],
                        sem_out.at[g_slot, 1],
                    ).start()

        if G_PER_ATT > 0:
            for g_tail in range(max(0, c.NATT - 2), c.NATT):
                tail_slot = g_tail & 1
                o_tail = g_tail * G_PER_ATT
                pltpu.make_async_copy(
                    conv_buf.at[tail_slot],
                    conv_out_ref.at[pl.ds(o_tail, G_PER_ATT)],
                    sem_out.at[tail_slot, 0],
                ).wait()
                if coal_rec:
                    pltpu.make_async_copy(
                        rec_buf.at[tail_slot],
                        rec_out_ref.at[pl.ds(o_tail, G_PER_ATT)],
                        sem_out.at[tail_slot, 1],
                    ).wait()
            if not coal_rec:
                for o_tail in range(max(0, total_gdn - 2), total_gdn):
                    r_tail = o_tail & 1
                    pltpu.make_async_copy(
                        rec_buf.at[r_tail, 0],
                        rec_out_ref.at[o_tail],
                        sem_out.at[r_tail, 1],
                    ).wait()
        else:
            cp_c = pltpu.make_async_copy(
                conv_buf.at[0, 0], conv_out_ref.at[0], sem_out.at[0, 0]
            )
            cp_r = pltpu.make_async_copy(
                rec_buf.at[0, 0], rec_out_ref.at[0], sem_out.at[0, 1]
            )
            cp_c.start()
            cp_r.start()
            cp_c.wait()
            cp_r.wait()

    in_specs = (
        [pl.BlockSpec(memory_space=pltpu.HBM)] * c.NATT
        + [pl.BlockSpec(memory_space=pltpu.SMEM)] * n_smem
    )
    out_shape = (
        [
            jax.ShapeDtypeStruct(conv_raw_shape, conv_raw_dt),
            jax.ShapeDtypeStruct(rec_shape, rec_dt),
        ]
        if interpret
        else [
            pltpu.HBM(conv_raw_shape, conv_raw_dt),
            pltpu.HBM(rec_shape, rec_dt),
        ]
    )
    out_specs = [
        pl.BlockSpec(memory_space=pltpu.HBM),
        pl.BlockSpec(memory_space=pltpu.HBM),
    ]
    call = pl.pallas_call(
        kernel,
        out_shape=out_shape,
        in_specs=in_specs,
        out_specs=out_specs,
        scratch_shapes=[scratch[k] for k in s_names],
        compiler_params=pltpu.CompilerParams(
            collective_id=None,
            disable_bounds_checks=True,
        ),
        interpret=bool(interpret),
        name="qwen3_5_gather_gdn_states",
    )
    smem_args = (
        (ckpt_indices, read_offsets, active_seqs)
        if act_mode == "ref"
        else (ckpt_indices, read_offsets)
    )
    conv_raw, rec_out = call(*raw_pools, *smem_args)
    if direct_conv:
        return conv_raw, rec_out
    taps_all = (
        conv_raw.reshape(ngdn, NS, conv_rows_pad, 128)[:, :, :conv_rows, :]
        .reshape(ngdn, NS, 3, QKV)
        .astype(conv_dt)
    )
    if T > 1:
        conv_out = jnp.pad(taps_all, ((0, 0), (0, 0), (0, c.HR - 3), (0, 0)))
    else:
        conv_3b = jnp.swapaxes(taps_all, 1, 2).reshape(ngdn, 3 * B, QKV)
        conv_out = jnp.pad(conv_3b, ((0, 0), (0, c.CR - 3 * B), (0, 0)))
    return conv_out, rec_out


# ---------------------------------------------------------------------------
# Kernel 2: Post-Megakernel Unified Pool Scatter (`scatter_pool_states`)
# ---------------------------------------------------------------------------
def scatter_pool_states(
    c: MKConfig,
    raw_pools,
    pos,
    page_table,
    knew,
    vnew,
    conv,
    rec,
    ckpt_indices,
    active_seqs=None,
    *,
    interpret=False,
):
    """In-place scatter of new FP8 KV tokens and per-token GDN states into `raw_pools`.

    Args:
      c: `MKConfig` geometry.
      raw_pools: `c.NATT`-tuple of per-rank `(num_pages, 2, 64, 4, 128) fp8` pool arrays.
      pos: `(c.B,) int32` token positions for rows `0..B-1` (`b = s * T + t`).
      page_table: `(c.NS, c.max_pages_per_seq) int32` page table.
      knew: `(c.NATT, c.MR, c.HD) fp8` (or `f32`/`bf16`, cast to `fp8`) new keys.
      vnew: `(c.NATT, c.MR, c.HD) fp8` (or `f32`/`bf16`, cast to `fp8`) new values.
      conv: GDN conv output from megakernel (`(ngdn, NS, HR, QKV)` for `T > 1` or
        `(ngdn, CR, QKV)` for `T == 1`, or `(ngdn, NS, 3, QKV)`); `None` when
        `c.att_every == 1` (attention-only stack: KV scatter only).
      rec: `(ngdn, B * vh, KD, VD) bf16` per-token GDN recurrent states from megakernel
        (`None` when `c.att_every == 1`).
      ckpt_indices: `(G_PER_ATT, c.NS, K)` or `(c.NS, K) int32` checkpoint slot table
        (`None` when `c.att_every == 1`).
      active_seqs: optional `(c.NS,) int32` mask (non-zero for active sequences; defaults to 1s).
      interpret: run Pallas in CPU interpret mode.

    Returns:
      Updated `raw_pools` (`c.NATT`-tuple, aliased in place to input `raw_pools`).
    """
    c.check()
    assert len(raw_pools) == c.NATT, (len(raw_pools), c.NATT)
    if active_seqs is None:
        act_mode = "all"
    elif isinstance(active_seqs, str) and active_seqs == "from_ckpt":
        act_mode = "from_ckpt"
    else:
        act_mode = "ref"
        active_seqs = active_seqs.astype(I32)
    if knew.dtype != FP8:
        knew = jnp.clip(knew, -448.0, 448.0).astype(FP8)
    if vnew.dtype != FP8:
        vnew = jnp.clip(vnew, -448.0, 448.0).astype(FP8)

    AE = c.att_every
    G_PER_ATT = AE - 1
    has_gdn = G_PER_ATT > 0
    assert has_gdn or act_mode != "from_ckpt", "attention-only stacks have no checkpoints"
    NS, B, T = c.NS, c.B, c.T
    vh, KD, VD, QKV, HD = c.vh, c.KD, c.VD, c.QKV, c.HD
    ssm_u32_rows = (vh * KD) // 2
    conv_rows = (3 * QKV) // 128
    conv_rows_pad = ((conv_rows + 15) // 16) * 16
    qkv_128 = QKV // 128
    n_pairs = NS // 2

    direct_conv = (
        has_gdn
        and T > 1
        and conv.ndim == 4
        and conv.shape[2] >= 3 + T
        and ((4 * qkv_128) % 16 == 0 or bool(interpret))
    )
    if not has_gdn:
        conv_in = None
    elif direct_conv:
        conv_in = conv
        conv_buf_shape = (2, 1, conv.shape[2], QKV)
        conv_buf_dt = conv.dtype
    else:
        if conv.ndim == 3:
            conv_4d = jnp.swapaxes(
                conv[:, :3 * B, :].reshape(conv.shape[0], 3, B, QKV), 1, 2
            )
        else:
            conv_4d = conv
        conv_has_hist = conv_4d.shape[2] >= 3 + T
        if conv_has_hist:
            conv_win = jnp.stack(
                [conv_4d[:, :, 1 + t : 4 + t, :] for t in range(T)], axis=2
            )
        else:
            conv_win = jnp.broadcast_to(
                conv_4d[:, :, None, 0:3, :], (conv_4d.shape[0], NS, T, 3, QKV)
            )
        conv_packed = conv_win.astype(BF16).reshape(
            conv_4d.shape[0], B, conv_rows, 128
        )
        if conv_rows_pad > conv_rows:
            conv_packed = jnp.pad(
                conv_packed,
                ((0, 0), (0, 0), (0, conv_rows_pad - conv_rows), (0, 0)),
            )
        conv_in = conv_packed.reshape(conv_4d.shape[0], B * conv_rows_pad, 128)
        conv_buf_shape = (2, T * conv_rows_pad, 128)
        conv_buf_dt = BF16

    ckpt_is_3d = has_gdn and ckpt_indices.ndim == 3
    ckpt_k = ckpt_indices.shape[-1] if has_gdn else T
    ckpt_t_off = 0 if ckpt_k == T else 1

    scratch = {
        "kv_page": pltpu.VMEM((2, 2, 64, 4, c.page_size), FP8),
        "knew_buf": pltpu.VMEM((c.NATT, c.MR, HD), FP8),
        "vnew_buf": pltpu.VMEM((c.NATT, c.MR, HD), FP8),
    }
    if has_gdn:
        scratch["page_buf"] = pltpu.VMEM((2, T * 5, 2, 64, 4, c.page_size), FP8)
        scratch["rec_buf"] = pltpu.VMEM((2, T * vh, KD, VD), rec.dtype)
        scratch["conv_buf"] = pltpu.VMEM(conv_buf_shape, conv_buf_dt)
    scratch["sem_kv_rd"] = pltpu.SemaphoreType.DMA((2,))
    scratch["sem_kv_wr"] = pltpu.SemaphoreType.DMA((2,))
    scratch["sem_kv_in"] = pltpu.SemaphoreType.DMA((2,))
    if has_gdn:
        scratch["sem_in"] = pltpu.SemaphoreType.DMA((2, 2))
        scratch["sem_p"] = pltpu.SemaphoreType.DMA((2, T))
    if NS > 1:
        scratch["k_cols_buf"] = pltpu.VMEM((c.NATT, 64, c.page_size), I32)
        scratch["v_cols_buf"] = pltpu.VMEM((c.NATT, 64, c.page_size), I32)
    s_names = list(scratch)
    n_smem = 2 + int(has_gdn) + int(act_mode == "ref")
    n_hbm = 4 if has_gdn else 2

    def batch_to_i32_cols(tok_mr_hd):
        rows_bf16 = tok_mr_hd.astype(BF16)
        if c.MR < c.page_size:
            rows_bf16 = jnp.pad(rows_bf16, ((0, c.page_size - c.MR), (0, 0)))
        cols_fp8 = jnp.swapaxes(rows_bf16, 0, 1).astype(FP8)
        return pltpu.bitcast(cols_fp8, I32)

    def kernel(*refs):
        # Inputs: *raw_pools (NATT), pos, page_table, [ckpt_indices], [active_seqs], knew, vnew,
        # [conv, rec] (the bracketed GDN operands only when has_gdn)
        smem_refs = list(refs[c.NATT:c.NATT + n_smem])
        pos_ref, pt_ref = smem_refs[:2]
        ckpt_ref = smem_refs[2] if has_gdn else None
        act_ref = smem_refs[-1] if act_mode == "ref" else None
        knew_ref, vnew_ref, conv_ref, rec_ref = (
            refs[c.NATT + n_smem : c.NATT + n_smem + n_hbm] + (None, None)
        )[:4]
        raw_out_refs = refs[c.NATT + n_smem + n_hbm : 2 * c.NATT + n_smem + n_hbm]
        scr = dict(zip(s_names, refs[2 * c.NATT + n_smem + n_hbm :]))
        kv_page = scr["kv_page"]
        knew_buf = scr["knew_buf"]
        vnew_buf = scr["vnew_buf"]
        page_buf = scr.get("page_buf")
        rec_buf = scr.get("rec_buf")
        conv_buf = scr.get("conv_buf")
        sem_kv_rd = scr["sem_kv_rd"]
        sem_kv_wr = scr["sem_kv_wr"]
        sem_kv_in = scr["sem_kv_in"]
        sem_in = scr.get("sem_in")
        sem_p = scr.get("sem_p")
        k_cols_buf = scr["k_cols_buf"] if NS > 1 else None
        v_cols_buf = scr["v_cols_buf"] if NS > 1 else None
        lane_iota = lax.broadcasted_iota(I32, (64, c.page_size), 1)
        w4_row_iota = (
            lax.broadcasted_iota(I32, (4, QKV), 0) if direct_conv else None
        )
        total_gdn = c.NATT * G_PER_ATT

        def is_active(s):
            if act_mode == "all":
                return jnp.bool_(True)
            if act_mode == "from_ckpt":
                return (ckpt_ref[0, s, 0] > 0) if ckpt_is_3d else (ckpt_ref[s, 0] > 0)
            return act_ref[s] != 0

        def pb_for(sub, s, t):
            if ckpt_is_3d:
                slot = ckpt_ref[sub, s, t + ckpt_t_off]
                return slot * 5
            else:
                slot = ckpt_ref[s, t + ckpt_t_off]
                return slot * 15 + sub * 5

        def is_contig_ckpt(sub, s):
            if not ckpt_is_3d or T <= 1:
                return jnp.bool_(T == 1)
            s0 = ckpt_ref[sub, s, ckpt_t_off]
            ok = jnp.bool_(True)
            for t in range(1, T):
                ok = jnp.logical_and(ok, ckpt_ref[sub, s, t + ckpt_t_off] == s0 + t)
            return ok

        def start_seq_in(o, s, in_slot):
            @pl.when(is_active(s))
            def _():
                r_base = (
                    s * T * vh
                    if isinstance(s, int)
                    else pl.multiple_of(s * T * vh, T * vh)
                )
                if direct_conv:
                    pltpu.make_async_copy(
                        conv_ref.at[o, pl.ds(s, 1)],
                        conv_buf.at[in_slot],
                        sem_in.at[in_slot, 0],
                    ).start()
                else:
                    c_base = (
                        s * T * conv_rows_pad
                        if isinstance(s, int)
                        else pl.multiple_of(s * T * conv_rows_pad, T * conv_rows_pad)
                    )
                    pltpu.make_async_copy(
                        conv_ref.at[o, pl.ds(c_base, T * conv_rows_pad)],
                        conv_buf.at[in_slot],
                        sem_in.at[in_slot, 0],
                    ).start()
                pltpu.make_async_copy(
                    rec_ref.at[o, pl.ds(r_base, T * vh)],
                    rec_buf.at[in_slot],
                    sem_in.at[in_slot, 1],
                ).start()

        def wait_seq_in(o, s, in_slot):
            @pl.when(is_active(s))
            def _():
                r_base = (
                    s * T * vh
                    if isinstance(s, int)
                    else pl.multiple_of(s * T * vh, T * vh)
                )
                if direct_conv:
                    pltpu.make_async_copy(
                        conv_ref.at[o, pl.ds(s, 1)],
                        conv_buf.at[in_slot],
                        sem_in.at[in_slot, 0],
                    ).wait()
                else:
                    c_base = (
                        s * T * conv_rows_pad
                        if isinstance(s, int)
                        else pl.multiple_of(s * T * conv_rows_pad, T * conv_rows_pad)
                    )
                    pltpu.make_async_copy(
                        conv_ref.at[o, pl.ds(c_base, T * conv_rows_pad)],
                        conv_buf.at[in_slot],
                        sem_in.at[in_slot, 0],
                    ).wait()
                pltpu.make_async_copy(
                    rec_ref.at[o, pl.ds(r_base, T * vh)],
                    rec_buf.at[in_slot],
                    sem_in.at[in_slot, 1],
                ).wait()

        def start_scatter_seq(raw_out_ref, sub, s, slot):
            @pl.when(is_active(s))
            def _():
                conv_bf16 = (
                    conv_buf[slot, 0].astype(BF16)
                    if (direct_conv and (4 * qkv_128) % 16 == 0)
                    else None
                )
                for t in range(T):
                    # Pack SSM state: (vh, KD, VD) bf16 -> (ssm_u32_rows, 128) u32 -> pages t*5 .. t*5+3
                    s_km = rec_buf[slot, pl.ds(t * vh, vh)]
                    u32_val = pltpu.bitcast(s_km.reshape(vh * KD, VD), U32)
                    page_buf.at[slot, pl.ds(t * 5, 4)].bitcast(U32).reshape(512, 128)[
                        :ssm_u32_rows, :
                    ] = u32_val

                    # Pack Conv state: (conv_rows_pad, 128) bf16 -> page t*5+4
                    if direct_conv:
                        if (4 * qkv_128) % 16 == 0:
                            w4 = jnp.where(
                                w4_row_iota < 3,
                                conv_bf16[1 + t : 5 + t, :],
                                jnp.bfloat16(0.0),
                            )
                            page_buf.at[slot, t * 5 + 4].bitcast(BF16).reshape(256, 128)[
                                :conv_rows_pad, :
                            ] = w4.reshape(conv_rows_pad, 128)
                        else:
                            taps_128 = (
                                conv_buf[slot, 0, 1 + t : 4 + t, :]
                                .astype(BF16)
                                .reshape(conv_rows, 128)
                            )
                            if conv_rows_pad > conv_rows:
                                taps_128 = jnp.pad(
                                    taps_128,
                                    ((0, conv_rows_pad - conv_rows), (0, 0)),
                                )
                            page_buf.at[slot, t * 5 + 4].bitcast(BF16).reshape(256, 128)[
                                :conv_rows_pad, :
                            ] = taps_128
                    else:
                        page_buf.at[slot, t * 5 + 4].bitcast(BF16).reshape(256, 128)[
                            :conv_rows_pad, :
                        ] = conv_buf[
                            slot, pl.ds(t * conv_rows_pad, conv_rows_pad), :
                        ]

                if T == 1:
                    pb0 = pb_for(sub, s, 0)
                    pltpu.make_async_copy(
                        page_buf.at[slot],
                        raw_out_ref.at[pl.ds(pb0, 5)],
                        sem_p.at[slot, 0],
                    ).start()
                else:
                    contig = is_contig_ckpt(sub, s)

                    @pl.when(contig)
                    def _():
                        pb0 = pb_for(sub, s, 0)
                        pltpu.make_async_copy(
                            page_buf.at[slot],
                            raw_out_ref.at[pl.ds(pb0, 5 * T)],
                            sem_p.at[slot, 0],
                        ).start()

                    @pl.when(jnp.logical_not(contig))
                    def _():
                        for t in range(T):
                            pb_t = pb_for(sub, s, t)
                            pltpu.make_async_copy(
                                page_buf.at[slot, pl.ds(t * 5, 5)],
                                raw_out_ref.at[pl.ds(pb_t, 5)],
                                sem_p.at[slot, t],
                            ).start()

        def wait_scatter_seq(raw_out_ref, sub, s, slot):
            @pl.when(is_active(s))
            def _():
                if T == 1:
                    pb0 = pb_for(sub, s, 0)
                    pltpu.make_async_copy(
                        page_buf.at[slot],
                        raw_out_ref.at[pl.ds(pb0, 5)],
                        sem_p.at[slot, 0],
                    ).wait()
                else:
                    contig = is_contig_ckpt(sub, s)

                    @pl.when(contig)
                    def _():
                        pb0 = pb_for(sub, s, 0)
                        pltpu.make_async_copy(
                            page_buf.at[slot],
                            raw_out_ref.at[pl.ds(pb0, 5 * T)],
                            sem_p.at[slot, 0],
                        ).wait()

                    @pl.when(jnp.logical_not(contig))
                    def _():
                        for t in range(T):
                            pb_t = pb_for(sub, s, t)
                            pltpu.make_async_copy(
                                page_buf.at[slot, pl.ds(t * 5, 5)],
                                raw_out_ref.at[pl.ds(pb_t, 5)],
                                sem_p.at[slot, t],
                            ).wait()

        # Start async DMA of 3D knew, vnew, and initial GDN (conv, rec) while zero-initializing page_buf
        cp_knew = pltpu.make_async_copy(knew_ref, knew_buf, sem_kv_in.at[0])
        cp_vnew = pltpu.make_async_copy(vnew_ref, vnew_buf, sem_kv_in.at[1])
        cp_knew.start()
        cp_vnew.start()

        if total_gdn > 0 and NS >= 1:
            start_seq_in(0, 0, 0)
            if NS >= 2:
                start_seq_in(0, 1, 1)
            elif total_gdn >= 2:
                start_seq_in(1, 0, 1)

        if G_PER_ATT > 0:
            for slot in range(2):
                if ssm_u32_rows < 512:
                    for t in range(T):
                        page_buf.at[slot, pl.ds(t * 5, 4)].bitcast(U32).reshape(512, 128)[
                            ...
                        ] = jnp.zeros((512, 128), U32)
                for t in range(T):
                    page_buf.at[slot, t * 5 + 4].bitcast(BF16).reshape(256, 128)[...] = jnp.zeros(
                        (256, 128), BF16
                    )

        cp_knew.wait()
        cp_vnew.wait()

        if NS > 1:
            for g in range(c.NATT):
                k_cols_buf[g] = batch_to_i32_cols(knew_buf[g])
                v_cols_buf[g] = batch_to_i32_cols(vnew_buf[g])

        # 1. Double-buffered scatter of B new KV tokens across all c.NATT attention layers
        def scatter_single_token_across_att(s_idx, b_idx):
            p = pos_ref[b_idx]
            lp = lax.div(p, c.page_size)
            ln = lax.rem(p, c.page_size)
            kp = pt_ref[s_idx, lp]
            mask_b = lane_iota == b_idx
            mask_ln = lane_iota == ln

            pltpu.make_async_copy(
                raw_out_refs[0].at[kp], kv_page.at[0], sem_kv_rd.at[0]
            ).start()
            if c.NATT >= 2:
                pltpu.make_async_copy(
                    raw_out_refs[1].at[kp], kv_page.at[1], sem_kv_rd.at[1]
                ).start()

            for g in range(c.NATT):
                k_slot = g & 1
                if g >= 2:
                    pltpu.make_async_copy(
                        kv_page.at[k_slot],
                        raw_out_refs[g - 2].at[kp],
                        sem_kv_wr.at[k_slot],
                    ).wait()
                pltpu.make_async_copy(
                    raw_out_refs[g].at[kp], kv_page.at[k_slot], sem_kv_rd.at[k_slot]
                ).wait()

                k_cols = k_cols_buf[g] if NS > 1 else batch_to_i32_cols(knew_buf[g])
                v_cols = v_cols_buf[g] if NS > 1 else batch_to_i32_cols(vnew_buf[g])
                k_col = jnp.sum(
                    jnp.where(mask_b, k_cols, jnp.int32(0)), axis=1, keepdims=True
                )
                v_col = jnp.sum(
                    jnp.where(mask_b, v_cols, jnp.int32(0)), axis=1, keepdims=True
                )
                k_view = kv_page.at[k_slot, 0].bitcast(I32).reshape(64, c.page_size)
                v_view = kv_page.at[k_slot, 1].bitcast(I32).reshape(64, c.page_size)
                k_view[...] = jnp.where(mask_ln, k_col, k_view[...])
                v_view[...] = jnp.where(mask_ln, v_col, v_view[...])

                pltpu.make_async_copy(
                    kv_page.at[k_slot],
                    raw_out_refs[g].at[kp],
                    sem_kv_wr.at[k_slot],
                ).start()
                if g + 2 < c.NATT:
                    pltpu.make_async_copy(
                        raw_out_refs[g + 2].at[kp],
                        kv_page.at[k_slot],
                        sem_kv_rd.at[k_slot],
                    ).start()

            for g_tail in range(max(0, c.NATT - 2), c.NATT):
                tail_slot = g_tail & 1
                pltpu.make_async_copy(
                    kv_page.at[tail_slot],
                    raw_out_refs[g_tail].at[kp],
                    sem_kv_wr.at[tail_slot],
                ).wait()

        if T == 1:
            def scatter_kv_row(b, _):
                @pl.when(is_active(b))
                def _():
                    scatter_single_token_across_att(b, b)

                return 0

            if B == 1:
                scatter_kv_row(0, 0)
            else:
                lax.fori_loop(0, B, scatter_kv_row, 0)
        else:
            def scatter_kv_seq(s, _):
                @pl.when(is_active(s))
                def _():
                    b0 = s * T
                    lp0 = lax.div(pos_ref[b0], c.page_size)
                    same_page = jnp.bool_(True)
                    for t in range(1, T):
                        same_page = jnp.logical_and(
                            same_page, lax.div(pos_ref[b0 + t], c.page_size) == lp0
                        )

                    @pl.when(same_page)
                    def _():
                        kp = pt_ref[s, lp0]
                        mask_b_list = [lane_iota == (b0 + t) for t in range(T)]
                        mask_ln_list = [
                            lane_iota == lax.rem(pos_ref[b0 + t], c.page_size)
                            for t in range(T)
                        ]

                        pltpu.make_async_copy(
                            raw_out_refs[0].at[kp], kv_page.at[0], sem_kv_rd.at[0]
                        ).start()
                        if c.NATT >= 2:
                            pltpu.make_async_copy(
                                raw_out_refs[1].at[kp], kv_page.at[1], sem_kv_rd.at[1]
                            ).start()

                        for g in range(c.NATT):
                            k_slot = g & 1
                            if g >= 2:
                                pltpu.make_async_copy(
                                    kv_page.at[k_slot],
                                    raw_out_refs[g - 2].at[kp],
                                    sem_kv_wr.at[k_slot],
                                ).wait()
                            pltpu.make_async_copy(
                                raw_out_refs[g].at[kp],
                                kv_page.at[k_slot],
                                sem_kv_rd.at[k_slot],
                            ).wait()

                            k_cols = (
                                k_cols_buf[g]
                                if NS > 1
                                else batch_to_i32_cols(knew_buf[g])
                            )
                            v_cols = (
                                v_cols_buf[g]
                                if NS > 1
                                else batch_to_i32_cols(vnew_buf[g])
                            )
                            k_view = (
                                kv_page.at[k_slot, 0]
                                .bitcast(I32)
                                .reshape(64, c.page_size)
                            )
                            v_view = (
                                kv_page.at[k_slot, 1]
                                .bitcast(I32)
                                .reshape(64, c.page_size)
                            )
                            k_cur = k_view[...]
                            v_cur = v_view[...]
                            for t in range(T):
                                k_col = jnp.sum(
                                    jnp.where(mask_b_list[t], k_cols, jnp.int32(0)),
                                    axis=1,
                                    keepdims=True,
                                )
                                v_col = jnp.sum(
                                    jnp.where(mask_b_list[t], v_cols, jnp.int32(0)),
                                    axis=1,
                                    keepdims=True,
                                )
                                k_cur = jnp.where(mask_ln_list[t], k_col, k_cur)
                                v_cur = jnp.where(mask_ln_list[t], v_col, v_cur)
                            k_view[...] = k_cur
                            v_view[...] = v_cur

                            pltpu.make_async_copy(
                                kv_page.at[k_slot],
                                raw_out_refs[g].at[kp],
                                sem_kv_wr.at[k_slot],
                            ).start()
                            if g + 2 < c.NATT:
                                pltpu.make_async_copy(
                                    raw_out_refs[g + 2].at[kp],
                                    kv_page.at[k_slot],
                                    sem_kv_rd.at[k_slot],
                                ).start()

                        for g_tail in range(max(0, c.NATT - 2), c.NATT):
                            tail_slot = g_tail & 1
                            pltpu.make_async_copy(
                                kv_page.at[tail_slot],
                                raw_out_refs[g_tail].at[kp],
                                sem_kv_wr.at[tail_slot],
                            ).wait()

                    @pl.when(jnp.logical_not(same_page))
                    def _():
                        for t in range(T):
                            scatter_single_token_across_att(s, b0 + t)

                return 0

            if NS == 1:
                scatter_kv_seq(0, 0)
            else:
                lax.fori_loop(0, NS, scatter_kv_seq, 0)

        # 2. Continuous double-buffered scatter of GDN (rec, conv) states across all total_gdn layers
        if total_gdn > 0:
            if NS == 1:
                for g in range(c.NATT):
                    raw_out_ref = raw_out_refs[g]
                    for sub in range(G_PER_ATT):
                        o = g * G_PER_ATT + sub
                        slot = o & 1
                        if o >= 2:
                            prev_o = o - 2
                            wait_scatter_seq(
                                raw_out_refs[prev_o // G_PER_ATT],
                                prev_o % G_PER_ATT,
                                0,
                                slot,
                            )
                        wait_seq_in(o, 0, slot)
                        start_scatter_seq(raw_out_ref, sub, 0, slot)
                        if o + 2 < total_gdn:
                            start_seq_in(o + 2, 0, slot)

                for o_tail in range(max(0, total_gdn - 2), total_gdn):
                    wait_scatter_seq(
                        raw_out_refs[o_tail // G_PER_ATT],
                        o_tail % G_PER_ATT,
                        0,
                        o_tail & 1,
                    )
            elif NS == 2:
                for g in range(c.NATT):
                    raw_out_ref = raw_out_refs[g]
                    for sub in range(G_PER_ATT):
                        o = g * G_PER_ATT + sub
                        if o >= 1:
                            prev_o = o - 1
                            wait_scatter_seq(
                                raw_out_refs[prev_o // G_PER_ATT],
                                prev_o % G_PER_ATT,
                                0,
                                0,
                            )
                        wait_seq_in(o, 0, 0)
                        start_scatter_seq(raw_out_ref, sub, 0, 0)
                        if o + 1 < total_gdn:
                            start_seq_in(o + 1, 0, 0)

                        if o >= 1:
                            prev_o = o - 1
                            wait_scatter_seq(
                                raw_out_refs[prev_o // G_PER_ATT],
                                prev_o % G_PER_ATT,
                                1,
                                1,
                            )
                        wait_seq_in(o, 1, 1)
                        start_scatter_seq(raw_out_ref, sub, 1, 1)
                        if o + 1 < total_gdn:
                            start_seq_in(o + 1, 1, 1)

                last_o = total_gdn - 1
                wait_scatter_seq(
                    raw_out_refs[last_o // G_PER_ATT], last_o % G_PER_ATT, 0, 0
                )
                wait_scatter_seq(
                    raw_out_refs[last_o // G_PER_ATT], last_o % G_PER_ATT, 1, 1
                )
            elif NS == 3:
                for g in range(c.NATT):
                    raw_out_ref = raw_out_refs[g]
                    for sub in range(G_PER_ATT):
                        o = g * G_PER_ATT + sub
                        if o >= 1:
                            prev_o = o - 1
                            wait_scatter_seq(
                                raw_out_refs[prev_o // G_PER_ATT],
                                prev_o % G_PER_ATT,
                                2,
                                0,
                            )
                        wait_seq_in(o, 0, 0)
                        start_scatter_seq(raw_out_ref, sub, 0, 0)
                        start_seq_in(o, 2, 0)

                        if o >= 1:
                            prev_o = o - 1
                            wait_scatter_seq(
                                raw_out_refs[prev_o // G_PER_ATT],
                                prev_o % G_PER_ATT,
                                1,
                                1,
                            )
                        wait_seq_in(o, 1, 1)
                        start_scatter_seq(raw_out_ref, sub, 1, 1)
                        if o + 1 < total_gdn:
                            start_seq_in(o + 1, 1, 1)

                        wait_scatter_seq(raw_out_ref, sub, 0, 0)
                        wait_seq_in(o, 2, 0)
                        start_scatter_seq(raw_out_ref, sub, 2, 0)
                        if o + 1 < total_gdn:
                            start_seq_in(o + 1, 0, 0)

                last_o = total_gdn - 1
                wait_scatter_seq(
                    raw_out_refs[last_o // G_PER_ATT], last_o % G_PER_ATT, 1, 1
                )
                wait_scatter_seq(
                    raw_out_refs[last_o // G_PER_ATT], last_o % G_PER_ATT, 2, 0
                )
            else:
                for g in range(c.NATT):
                    raw_out_ref = raw_out_refs[g]
                    for sub in range(G_PER_ATT):
                        o = g * G_PER_ATT + sub

                        def scatter_pair(
                            pair, _, raw_out_ref=raw_out_ref, sub=sub, o=o
                        ):
                            s0 = 2 * pair
                            s1 = s0 + 1

                            @pl.when(pair > 0)
                            def _():
                                wait_scatter_seq(raw_out_ref, sub, s0 - 2, 0)

                            wait_seq_in(o, s0, 0)
                            start_scatter_seq(raw_out_ref, sub, s0, 0)

                            @pl.when(s0 + 2 < NS)
                            def _():
                                start_seq_in(o, s0 + 2, 0)

                            @pl.when(pair > 0)
                            def _():
                                wait_scatter_seq(raw_out_ref, sub, s1 - 2, 1)

                            wait_seq_in(o, s1, 1)
                            start_scatter_seq(raw_out_ref, sub, s1, 1)

                            @pl.when(s1 + 2 < NS)
                            def _():
                                start_seq_in(o, s1 + 2, 1)

                            return 0

                        lax.fori_loop(0, n_pairs, scatter_pair, 0)

                        if NS % 2 == 1:
                            wait_scatter_seq(raw_out_ref, sub, NS - 3, 0)
                            wait_seq_in(o, NS - 1, 0)
                            start_scatter_seq(raw_out_ref, sub, NS - 1, 0)
                            if o + 1 < total_gdn:
                                start_seq_in(o + 1, 0, 0)
                                start_seq_in(o + 1, 1, 1)
                            wait_scatter_seq(raw_out_ref, sub, NS - 2, 1)
                            wait_scatter_seq(raw_out_ref, sub, NS - 1, 0)
                        else:
                            if o + 1 < total_gdn:
                                start_seq_in(o + 1, 0, 0)
                                start_seq_in(o + 1, 1, 1)
                            wait_scatter_seq(raw_out_ref, sub, NS - 2, 0)
                            wait_scatter_seq(raw_out_ref, sub, NS - 1, 1)

    in_specs = (
        [pl.BlockSpec(memory_space=pltpu.HBM)] * c.NATT
        + [pl.BlockSpec(memory_space=pltpu.SMEM)] * n_smem
        + [
            pl.BlockSpec(memory_space=pltpu.HBM),   # knew (3D HBM)
            pl.BlockSpec(memory_space=pltpu.HBM),   # vnew (3D HBM)
            pl.BlockSpec(memory_space=pltpu.HBM),   # conv_in
            pl.BlockSpec(memory_space=pltpu.HBM),   # rec
        ][:n_hbm]
    )
    out_shape = [jax.ShapeDtypeStruct(r.shape, r.dtype) for r in raw_pools]
    out_specs = [pl.BlockSpec(memory_space=pltpu.HBM)] * c.NATT
    call = pl.pallas_call(
        kernel,
        out_shape=out_shape,
        in_specs=in_specs,
        out_specs=out_specs,
        scratch_shapes=[scratch[k] for k in s_names],
        input_output_aliases={r: r for r in range(c.NATT)},
        compiler_params=pltpu.CompilerParams(
            collective_id=None,
            disable_bounds_checks=True,
        ),
        interpret=bool(interpret),
        name="qwen3_5_scatter_pool_states",
    )
    smem_args = (
        (pos, page_table)
        + ((ckpt_indices,) if has_gdn else ())
        + ((active_seqs,) if act_mode == "ref" else ())
    )
    if not interpret and isinstance(knew, jax.core.Tracer):
        knew = pltpu.with_memory_space_constraint(knew, pltpu.HBM)
        vnew = pltpu.with_memory_space_constraint(vnew, pltpu.HBM)
        if has_gdn:
            conv_in = pltpu.with_memory_space_constraint(conv_in, pltpu.HBM)
            rec = pltpu.with_memory_space_constraint(rec, pltpu.HBM)
    gdn_args = (conv_in, rec) if has_gdn else ()
    out = call(
        *raw_pools, *smem_args, knew, vnew, *gdn_args
    )
    return tuple(out)


# ---------------------------------------------------------------------------
# End-to-End Pooled Step Wrapper (`make_pooled_step` & `make_flat_pooled_step`)
# ---------------------------------------------------------------------------
class _PooledStepFn:
    """Callable step function that also unpacks as `(step_fn, info)`."""

    def __init__(self, fn, info):
        self._fn = fn
        self.info = info

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)

    def __iter__(self):
        yield self._fn
        yield self.info


def check_fused_pool_hlo(
    hlo_text: str, page_size: int = 128, num_pools: int | None = None
) -> tuple[bool, str]:
    """Verify that compiled HLO for fused pool I/O does not copy pool tensors into VMEM S(1)
    or duplicate full pool buffers in HBM.

    Returns:
      `(ok, reason)` where `ok` is `True` if safe for zero-copy in-place pool DMA writes.
    """
    if "tpu_custom_call" not in hlo_text and "qwen3_5_decode_megakernel" not in hlo_text:
        return False, "missing megakernel custom-call in HLO"
    if "gather_gdn_states" in hlo_text or "scatter_pool_states" in hlo_text:
        return False, "unexpected gather/scatter custom-call in fused HLO"
    pool_suffix = f",2,64,4,{page_size}]"
    for line in hlo_text.splitlines():
        if pool_suffix in line and (
            "copy-start(" in line or ("= f8e4m3fn[" in line and "copy(" in line)
        ):
            if "S(1)" in line:
                return False, f"pool operand copied to VMEM S(1): {line.strip()}"
            return False, f"unexpected full-pool copy in HLO: {line.strip()}"
    return True, "ok"


def make_flat_pooled_step(
    mesh,
    c: MKConfig,
    tiles: Tiles | None = None,
    *,
    interpret=False,
    skip=frozenset(),
    scopes=False,
    fused_pool_io: bool | None = None,
):
    """Build a flat positional `shard_map` pooled step for `sharded_jax_callable` in `vllm_torchtpu`.

    Every operand and output uses `PartitionSpec("tp")`, so `sharded_jax_callable` passes each
    rank's local PyTorch tensor (`pos [B]`, `hidden [M, H]`, `*w_flat`, `*raw_pools`,
    `page_table [NS, max_pages_per_seq]`, `ckpt_indices [3, NS, T]`, `read_offsets [3, NS]`)
    directly into `shard_map` with its unmodified per-rank shape.

    Returns:
      `(flat_step_fn, in_specs, out_specs, donate_argnums)`
    """
    use_fused = bool(c.fused_pool_io) if fused_pool_io is None else bool(fused_pool_io)
    if use_fused:
        c_paged = dataclasses.replace(
            c,
            paged_kv=True,
            fused_pool_io=True,
            act_mode="from_ckpt",
            ckpt_2d=False,
            roff_1d=False,
            ckpt_k=c.T,
        )
    else:
        c_paged = dataclasses.replace(c, paged_kv=True, fused_pool_io=False)
    call, info = build(
        c_paged,
        tiles,
        interpret=interpret,
        forced_routing=False,
        skip=skip,
        scopes=scopes,
    )
    tk = plan(c_paged, tiles or default_tiles(c_paged))["tk"]
    e13_shape_4d = (c_paged.EW, c_paged.H // tk["e13"], tk["e13"], 2 * c_paged.IR)
    e2_shape_4d = (c_paged.EW, c_paged.IR // tk["e2"], tk["e2"], c_paged.H)
    n_w_leaves = sum(
        c_paged.L if (not c_paged.stack_e and k in ("e13", "e2")) else 1 for k in W_NAMES
    )
    n_p = c_paged.NATT
    A = P(mesh.axis_names[0])
    # All inputs and outputs are sharded along axis 0 over the mesh so each rank's
    # local PyTorch tensor has the exact per-rank shape expected inside `shard_map`.
    in_specs = (A,) * (2 + n_w_leaves + n_p + 3)
    out_specs = (A,) * (1 + n_p)
    donate_argnums = tuple(range(2 + n_w_leaves, 2 + n_w_leaves + n_p))

    def _body(*args):
        pos, hidden = args[0], args[1]
        w_leaves = args[2:2 + n_w_leaves]
        raw_pools = args[2 + n_w_leaves:2 + n_w_leaves + n_p]
        page_table, ckpt_indices, read_offsets = args[2 + n_w_leaves + n_p:]

        w_grouped = []
        idx = 0
        for k in W_NAMES:
            if not c_paged.stack_e and k in ("e13", "e2"):
                target_4d = e13_shape_4d if k == "e13" else e2_shape_4d
                w_grouped.append(
                    tuple(
                        w.reshape(target_4d) if w.ndim == 3 else w
                        for w in w_leaves[idx:idx + c_paged.L]
                    )
                )
                idx += c_paged.L
            else:
                w_grouped.append(w_leaves[idx])
                idx += 1

        if use_fused:
            out = call(
                pos,
                hidden,
                *w_grouped,
                raw_pools,
                page_table,
                ckpt_indices,
                read_offsets,
            )
            if bool(interpret):
                hid, raw_out = out
                return (hid, *raw_out)
            hid = out[0] if isinstance(out, (tuple, list)) else out
            return (hid, *raw_pools)

        conv, rec = gather_gdn_states(
            c_paged,
            raw_pools,
            ckpt_indices,
            read_offsets,
            active_seqs="from_ckpt",
            interpret=interpret,
        )
        hid, conv_out, rec_out, knew, vnew = call(
            pos, hidden, *w_grouped, conv, rec, raw_pools, page_table
        )
        raw_out = scatter_pool_states(
            c_paged,
            raw_pools,
            pos,
            page_table,
            knew,
            vnew,
            conv_out,
            rec_out,
            ckpt_indices,
            active_seqs="from_ckpt",
            interpret=interpret,
        )
        return (hid, *raw_out)

    smap = jax.shard_map(
        _body, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False
    )
    return smap, in_specs, out_specs, donate_argnums


def make_flat_attn_step(
    mesh,
    c: MKConfig,
    tiles: Tiles | None = None,
    *,
    interpret=False,
    skip=frozenset(),
    scopes=False,
):
    """Flat `shard_map` pooled step of an attention-only stack (`c.att_every == 1`, the MTP
    draft layer) for `sharded_jax_callable`.

    Like `make_flat_pooled_step` without GatedDeltaNet states. Operands (all `PartitionSpec("tp")`,
    per-rank shapes): `pos [B]`, `hidden [M, H]`, `*w_flat`, `*raw_pools`,
    `page_table [NS, max_pages_per_seq]`, `active_seqs [NS]` (sequences whose new KV tokens are
    scattered) and `moe_rows [B]` (rows whose routed experts count, see `megakernel.build`).

    Returns:
      `(flat_step_fn, in_specs, out_specs, donate_argnums)`
    """
    assert c.att_every == 1, c.att_every
    c_paged = dataclasses.replace(c, paged_kv=True, fused_pool_io=False)
    call, info = build(
        c_paged,
        tiles,
        interpret=interpret,
        forced_routing=False,
        skip=skip,
        scopes=scopes,
        row_mask=True,
    )
    tk = plan(c_paged, tiles or default_tiles(c_paged))["tk"]
    e13_shape_4d = (c_paged.EW, c_paged.H // tk["e13"], tk["e13"], 2 * c_paged.IR)
    e2_shape_4d = (c_paged.EW, c_paged.IR // tk["e2"], tk["e2"], c_paged.H)
    n_w_leaves = sum(
        c_paged.L if (not c_paged.stack_e and k in ("e13", "e2")) else 1 for k in W_NAMES
    )
    n_p = c_paged.NATT
    A = P(mesh.axis_names[0])
    in_specs = (A,) * (2 + n_w_leaves + n_p + 3)
    out_specs = (A,) * (1 + n_p)
    donate_argnums = tuple(range(2 + n_w_leaves, 2 + n_w_leaves + n_p))

    def _body(*args):
        pos, hidden = args[0], args[1]
        w_leaves = args[2:2 + n_w_leaves]
        raw_pools = args[2 + n_w_leaves:2 + n_w_leaves + n_p]
        page_table, active_seqs, moe_rows = args[2 + n_w_leaves + n_p:]

        w_grouped = []
        idx = 0
        for k in W_NAMES:
            if not c_paged.stack_e and k in ("e13", "e2"):
                target_4d = e13_shape_4d if k == "e13" else e2_shape_4d
                w_grouped.append(
                    tuple(
                        w.reshape(target_4d) if w.ndim == 3 else w
                        for w in w_leaves[idx:idx + c_paged.L]
                    )
                )
                idx += c_paged.L
            else:
                w_grouped.append(w_leaves[idx])
                idx += 1

        # The kernel never touches the GatedDeltaNet state operands of an attention-only stack.
        no_gdn = jnp.zeros((8, 128), BF16)
        hid, _, _, knew, vnew = call(
            pos, hidden, *w_grouped, no_gdn, no_gdn, raw_pools, page_table, moe_rows
        )
        raw_out = scatter_pool_states(
            c_paged,
            raw_pools,
            pos,
            page_table,
            knew,
            vnew,
            None,
            None,
            None,
            active_seqs,
            interpret=interpret,
        )
        return (hid, *raw_out)

    smap = jax.shard_map(
        _body, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False
    )
    return smap, in_specs, out_specs, donate_argnums


def make_pooled_step(
    mesh,
    c: MKConfig,
    tiles: Tiles | None = None,
    *,
    interpret=False,
    forced_routing=False,
    skip=frozenset(),
    scopes=False,
    fused_pool_io: bool | None = None,
):
    """Build the `shard_map` decode/verify step over unified block pools `raw[0..14]`.

    Supports two calling conventions:
      1. Positional:
         `hid, raw_pools = step(pos, hidden, weights, raw_pools, page_table, ckpt_indices,
                                read_offsets, [rbias], [active_seqs])`
      2. States dict (matching `make_step`):
         `hid, states = step(pos, hidden, weights, states, [rbias])`
         where `states` contains `"raw"`, `"page_table"`, `"ckpt_indices"`, `"read_offsets"`
         (and optional `"active_seqs"`).

    Also supports both `step = make_pooled_step(...)` and `step, info = make_pooled_step(...)`.
    """
    use_fused = bool(c.fused_pool_io) if fused_pool_io is None else bool(fused_pool_io)
    c_paged = dataclasses.replace(c, paged_kv=True, fused_pool_io=use_fused)
    call, info = build(
        c_paged,
        tiles,
        interpret=interpret,
        forced_routing=forced_routing,
        skip=skip,
        scopes=scopes,
    )
    tk = plan(c_paged, tiles or default_tiles(c_paged))["tk"]
    e13_shape_4d = (c_paged.EW, c_paged.H // tk["e13"], tk["e13"], 2 * c_paged.IR)
    e2_shape_4d = (c_paged.EW, c_paged.IR // tk["e2"], tk["e2"], c_paged.H)
    _fused_cache = {
        (c_paged.act_mode, c_paged.ckpt_2d, c_paged.roff_1d, c_paged.ckpt_k or c_paged.T): call
    }

    def _get_fused_call(act_mode, ckpt_2d, roff_1d, ckpt_k):
        key = (act_mode, ckpt_2d, roff_1d, ckpt_k)
        if key not in _fused_cache:
            c_var = dataclasses.replace(
                c_paged,
                fused_pool_io=True,
                act_mode=act_mode,
                ckpt_2d=ckpt_2d,
                roff_1d=roff_1d,
                ckpt_k=ckpt_k,
            )
            _fused_cache[key], _ = build(
                c_var,
                tiles,
                interpret=interpret,
                forced_routing=forced_routing,
                skip=skip,
                scopes=scopes,
            )
        return _fused_cache[key]

    def _run_local(pos, hidden, weights, raw_pools, page_table, ckpt_indices, read_offsets,
                   rb_tuple, active_seqs):
        raw_6d = raw_pools[0].ndim == 6
        raw_in = tuple(r.squeeze(0) for r in raw_pools) if raw_6d else tuple(raw_pools)
        if page_table.ndim == 3:
            page_table = page_table.squeeze(0)
        if ckpt_indices.ndim == 4:
            ckpt_indices = ckpt_indices.squeeze(0)
        if read_offsets.ndim == 3:
            read_offsets = read_offsets.squeeze(0)
        if active_seqs is not None and active_seqs.ndim == 2:
            active_seqs = active_seqs.squeeze(0)

        w = []
        for k in W_NAMES:
            if isinstance(weights[k], tuple):
                target_4d = e13_shape_4d if k == "e13" else e2_shape_4d
                w.append(
                    tuple(
                        x[0].reshape(target_4d) if x[0].ndim == 3 else x[0]
                        for x in weights[k]
                    )
                )
            else:
                w.append(weights[k][0])

        if use_fused:
            act_mode = "ref" if active_seqs is not None else c_paged.act_mode
            ckpt_2d = ckpt_indices.ndim == 2
            roff_1d = read_offsets.ndim == 1
            ckpt_k = int(ckpt_indices.shape[-1])
            call_f = _get_fused_call(act_mode, ckpt_2d, roff_1d, ckpt_k)
            st_args = (
                (raw_in, page_table, ckpt_indices, read_offsets, active_seqs.astype(I32))
                if act_mode == "ref"
                else (raw_in, page_table, ckpt_indices, read_offsets)
            )
            out = call_f(pos, hidden, *w, *st_args, *rb_tuple)
            if bool(interpret):
                hid, raw_out = out
            else:
                hid = out[0] if isinstance(out, (tuple, list)) else out
                raw_out = raw_in
            if raw_6d:
                raw_out = tuple(jnp.expand_dims(r, 0) for r in raw_out)
            return hid, raw_out

        conv, rec = gather_gdn_states(
            c_paged, raw_in, ckpt_indices, read_offsets, active_seqs, interpret=interpret
        )
        hid, conv_out, rec_out, knew, vnew = call(
            pos, hidden, *w, conv, rec, raw_in, page_table, *rb_tuple
        )
        raw_out = scatter_pool_states(
            c_paged,
            raw_in,
            pos,
            page_table,
            knew,
            vnew,
            conv_out,
            rec_out,
            ckpt_indices,
            active_seqs,
            interpret=interpret,
        )
        if raw_6d:
            raw_out = tuple(jnp.expand_dims(r, 0) for r in raw_out)
        return hid, raw_out

    def _meta_spec(arr, ns_dim):
        return P("tp") if arr.shape[0] == c.R * ns_dim and c.R > 1 else P()

    def step(pos, hidden, weights, states_or_raw, *rest):
        ws = {
            k: tuple(P("tp") for _ in weights[k]) if isinstance(weights[k], tuple) else P("tp")
            for k in weights
        }
        if isinstance(states_or_raw, dict):
            states = states_or_raw
            raw_pools = tuple(states["raw"])
            page_table = states["page_table"]
            ckpt_indices = states["ckpt_indices"]
            read_offsets = states["read_offsets"]
            active_seqs = states.get("active_seqs", None)
            rb_tuple = rest

            raw_spec = tuple(P("tp") for _ in raw_pools)
            pt_spec = _meta_spec(page_table, c.NS)
            ck_spec = _meta_spec(
                ckpt_indices, c.NS if ckpt_indices.ndim == 2 else max(1, c.att_every - 1)
            )
            ro_spec = _meta_spec(
                read_offsets, c.NS if read_offsets.ndim == 1 else max(1, c.att_every - 1)
            )
            rb_specs = (P("tp"),) if forced_routing else ()

            if active_seqs is None:
                def local_dict(pos, hidden, weights, raw_pools, page_table, ckpt_indices,
                               read_offsets, *rb):
                    return _run_local(pos, hidden, weights, raw_pools, page_table, ckpt_indices,
                                      read_offsets, rb, None)

                in_specs = (P(), P("tp"), ws, raw_spec, pt_spec, ck_spec, ro_spec) + rb_specs
                hid, raw_out = jax.shard_map(
                    local_dict, mesh=mesh, in_specs=in_specs, out_specs=(P("tp"), raw_spec),
                    check_vma=False
                )(pos, hidden, weights, raw_pools, page_table, ckpt_indices, read_offsets,
                  *rb_tuple)
            else:
                ac_spec = _meta_spec(active_seqs, c.NS)

                def local_dict_act(pos, hidden, weights, raw_pools, page_table, ckpt_indices,
                                   read_offsets, active_seqs, *rb):
                    return _run_local(pos, hidden, weights, raw_pools, page_table, ckpt_indices,
                                      read_offsets, rb, active_seqs)

                in_specs = (
                    (P(), P("tp"), ws, raw_spec, pt_spec, ck_spec, ro_spec, ac_spec) + rb_specs
                )
                hid, raw_out = jax.shard_map(
                    local_dict_act, mesh=mesh, in_specs=in_specs, out_specs=(P("tp"), raw_spec),
                    check_vma=False
                )(pos, hidden, weights, raw_pools, page_table, ckpt_indices, read_offsets,
                  active_seqs, *rb_tuple)
            return hid, {**states, "raw": raw_out}

        # Positional convention:
        # (pos, hidden, weights, raw_pools, page_table, ckpt_indices, read_offsets, [rbias], [active_seqs])
        raw_pools = tuple(states_or_raw)
        page_table, ckpt_indices, read_offsets = rest[:3]
        extra = rest[3:]
        rb_tuple = ()
        active_seqs = None
        if forced_routing:
            rb_tuple = (extra[0],)
            if len(extra) > 1:
                active_seqs = extra[1]
        elif len(extra) > 0:
            active_seqs = extra[0]

        raw_spec = tuple(P("tp") for _ in raw_pools)
        pt_spec = _meta_spec(page_table, c.NS)
        ck_spec = _meta_spec(
            ckpt_indices, c.NS if ckpt_indices.ndim == 2 else max(1, c.att_every - 1)
        )
        ro_spec = _meta_spec(
            read_offsets, c.NS if read_offsets.ndim == 1 else max(1, c.att_every - 1)
        )
        rb_specs = (P("tp"),) if forced_routing else ()

        if active_seqs is None:
            def local_pos(pos, hidden, weights, raw_pools, page_table, ckpt_indices, read_offsets,
                          *rb):
                return _run_local(pos, hidden, weights, raw_pools, page_table, ckpt_indices,
                                  read_offsets, rb, None)

            in_specs = (P(), P("tp"), ws, raw_spec, pt_spec, ck_spec, ro_spec) + rb_specs
            return jax.shard_map(
                local_pos, mesh=mesh, in_specs=in_specs, out_specs=(P("tp"), raw_spec),
                check_vma=False
            )(pos, hidden, weights, raw_pools, page_table, ckpt_indices, read_offsets, *rb_tuple)
        else:
            ac_spec = _meta_spec(active_seqs, c.NS)

            def local_pos_act(pos, hidden, weights, raw_pools, page_table, ckpt_indices,
                              read_offsets, active_seqs, *rb):
                return _run_local(pos, hidden, weights, raw_pools, page_table, ckpt_indices,
                                  read_offsets, rb, active_seqs)

            in_specs = (P(), P("tp"), ws, raw_spec, pt_spec, ck_spec, ro_spec, ac_spec) + rb_specs
            return jax.shard_map(
                local_pos_act, mesh=mesh, in_specs=in_specs, out_specs=(P("tp"), raw_spec),
                check_vma=False
            )(pos, hidden, weights, raw_pools, page_table, ckpt_indices, read_offsets, active_seqs,
              *rb_tuple)

    return _PooledStepFn(step, info)


# ---------------------------------------------------------------------------
# Host / Test Packing & Unpacking Utilities
# ---------------------------------------------------------------------------
def kv_to_raw_page(k_page, v_page):
    """Pack `(page_size, 256)` K and V into `(2, 64, 4, page_size) fp8`."""
    ps = k_page.shape[0]
    k_fp8 = jnp.clip(k_page, -448.0, 448.0).astype(FP8) if k_page.dtype != FP8 else k_page
    v_fp8 = jnp.clip(v_page, -448.0, 448.0).astype(FP8) if v_page.dtype != FP8 else v_page
    k_pk = jnp.transpose(k_fp8.reshape(ps, 64, 4), (1, 2, 0))
    v_pk = jnp.transpose(v_fp8.reshape(ps, 64, 4), (1, 2, 0))
    return jnp.stack([k_pk, v_pk], axis=0)


def raw_page_to_kv(raw_page):
    """Unpack `(2, 64, 4, page_size) fp8` into `(k_page, v_page)` of shape `(page_size, 256) fp8`."""
    ps = raw_page.shape[-1]
    k_page = jnp.transpose(raw_page[0], (2, 0, 1)).reshape(ps, 256)
    v_page = jnp.transpose(raw_page[1], (2, 0, 1)).reshape(ps, 256)
    return k_page, v_page


def ssm_to_raw_pages(rec_slice):
    """Pack key-major `(vh, KD, VD) bf16` into 4 raw pages `(4, 2, 64, 4, 128) fp8`."""
    vh, KD, VD = rec_slice.shape
    ssm_u32_rows = (vh * KD) // 2
    s_t = np.asarray(rec_slice, dtype=jnp.dtype(BF16)).reshape(vh * KD, VD)
    # Bitcast pairs of bf16 along axis 0 into uint32 -> ((vh * KD) // 2, 128) u32
    u32_val = s_t.reshape(ssm_u32_rows, 2, 128).transpose(0, 2, 1).copy().view(np.uint32).reshape(
        ssm_u32_rows, 128
    )
    if ssm_u32_rows < 512:
        u32_val = np.pad(u32_val, ((0, 512 - ssm_u32_rows), (0, 0)))
    # View (512, 128) u32 as (4, 2, 64, 4, 128) fp8
    fp8_4p = u32_val.reshape(4, 2, 64, 128, 1).view(jnp.dtype(FP8)).transpose(0, 1, 2, 4, 3)
    return jnp.asarray(fp8_4p)


def conv_to_raw_page(taps_3_qkv):
    """Pack `(3, QKV)` conv taps into 1 raw page `(2, 64, 4, 128) fp8`."""
    QKV = taps_3_qkv.shape[-1]
    conv_rows = (3 * QKV) // 128
    taps_bf16 = np.asarray(taps_3_qkv, dtype=jnp.dtype(BF16)).reshape(conv_rows, 128)
    padded_256 = np.pad(taps_bf16, ((0, 256 - conv_rows), (0, 0)))
    # View (256, 128) bf16 as (2, 64, 4, 128) fp8
    fp8_1p = padded_256.reshape(2, 64, 2, 128, 1).view(jnp.dtype(FP8)).transpose(0, 1, 2, 4, 3)
    fp8_1p = fp8_1p.reshape(2, 64, 4, 128)
    return jnp.asarray(fp8_1p)


def default_pool_layout(c: MKConfig):
    """Allocate disjoint `page_table`, `ckpt_indices`, `read_offsets`, and minimum `num_pages`
    for `c.NS` sequences with context length `c.ctx` and `c.T` checkpoint slots per sequence
    per Mamba group (`ckpt_indices` of shape `(G_PER_ATT, NS, T)` and `read_offsets` of shape
    `(G_PER_ATT, NS)`).
    """
    G_PER_ATT = max(1, c.att_every - 1)
    pages_per_seq = c.ctx // c.page_size
    kv_pages_total = c.NS * pages_per_seq
    n_ckpt_slots = G_PER_ATT * c.NS * c.T
    base_slot = (kv_pages_total + 4) // 5
    min_num_pages = (base_slot + n_ckpt_slots) * 5

    pt = np.zeros((c.NS, max(c.max_pages_per_seq, pages_per_seq)), dtype=np.int32)
    for s in range(c.NS):
        pt[s, :pages_per_seq] = np.arange(s * pages_per_seq, (s + 1) * pages_per_seq, dtype=np.int32)

    ckpt = np.zeros((G_PER_ATT, c.NS, c.T), dtype=np.int32)
    for sub in range(G_PER_ATT):
        for s in range(c.NS):
            for k in range(c.T):
                ckpt[sub, s, k] = base_slot + (sub * c.NS + s) * c.T + k
    roff = np.zeros((G_PER_ATT, c.NS), dtype=np.int32)
    return jnp.asarray(pt), jnp.asarray(ckpt), jnp.asarray(roff), min_num_pages


def states_to_pool(c: MKConfig, St, page_table, ckpt_indices, read_offsets=None):
    """Pack reference states `St = {"conv", "rec", "kc", "vc"}` (with leading rank axis `R`)
    into `c.NATT` rank-folded raw pools `(R * num_pages, 2, 64, 4, 128) fp8`.
    """
    AE = c.att_every
    G_PER_ATT = AE - 1
    pt = np.asarray(page_table)
    ckpt = np.asarray(ckpt_indices)
    ckpt_is_3d = ckpt.ndim == 3
    if read_offsets is None:
        roff = np.zeros((max(1, G_PER_ATT), c.NS) if ckpt_is_3d else (c.NS,), dtype=np.int32)
    else:
        roff = np.asarray(read_offsets)
    roff_is_2d = roff.ndim == 2
    pages_per_seq = c.ctx // c.page_size

    raw_list = []
    for g in range(c.NATT):
        pool_g = np.zeros((c.R, c.num_pages, 2, 64, 4, c.page_size), dtype=jnp.dtype(FP8))
        # Pack KV pages for attention layer g
        kc_g = np.asarray(jnp.clip(St["kc"][:, g], -448.0, 448.0).astype(FP8))
        vc_g = np.asarray(jnp.clip(St["vc"][:, g], -448.0, 448.0).astype(FP8))
        for r in range(c.R):
            for s in range(c.NS):
                for p in range(pages_per_seq):
                    kp = int(pt[s, p])
                    t0 = p * c.page_size
                    k_sl = kc_g[r, s, t0:t0 + c.page_size]
                    v_sl = vc_g[r, s, t0:t0 + c.page_size]
                    pool_g[r, kp] = np.asarray(kv_to_raw_page(k_sl, v_sl))

        # Pack initial GDN states into ckpt_indices[..., read_offsets[...]]
        for sub in range(G_PER_ATT):
            o = g * G_PER_ATT + sub
            for r in range(c.R):
                for s in range(c.NS):
                    ro_val = int(roff[sub, s]) if roff_is_2d else int(roff[s])
                    if ckpt_is_3d:
                        slot = int(ckpt[sub, s, ro_val])
                        page_base = slot * 5
                    else:
                        slot = int(ckpt[s, ro_val])
                        page_base = slot * 15 + sub * 5
                    rec_sl = St["rec"][r, o, s * c.vh:(s + 1) * c.vh]
                    pool_g[r, page_base:page_base + 4] = np.asarray(ssm_to_raw_pages(rec_sl))
                    if c.T > 1:
                        taps = St["conv"][r, o, s, 0:3]
                    else:
                        taps = jnp.stack([St["conv"][r, o, j * c.B + s] for j in range(3)], axis=0)
                    pool_g[r, page_base + 4] = np.asarray(conv_to_raw_page(taps))

        raw_list.append(jnp.asarray(pool_g.reshape(c.R * c.num_pages, 2, 64, 4, c.page_size)))
    return tuple(raw_list)

