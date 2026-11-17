# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""glm52_decode_mk V16 (virtual DCP-R on the DCP-R/2 cache; stageB/v16/DESIGN-ADDENDUM.md): the replica page-parity
split's arithmetic, the half block tables, the pool guard and the configuration admission. Pure Python / host jnp, no
TPU. The kernel's on-device behaviour (oracle, faults, bit identity, timing) is the campaign's (stageB/v16).

The formulas below are a brute-force model written from the definitions (owner (t // 4) % D, li = (t // 4D)·4 + t % 4,
local page j = li // 256, request b's page j belongs to replica (j + b) % VR), not copied from dsa.py: the kernel's
expressions are checked against them by evaluating them on the same integers (`_kernel_*` mirror dsa.py's
`per_req_v16` / `l8_of_tile` / `gpos_of_l8` / `fix` lines expression for expression)."""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.glm52_decode_mk import config as C
from vllm_torchtpu.kernels.glm52_decode_mk import vllm_adapter as A
from vllm_torchtpu.kernels.glm52_decode_mk import wrapper as W

ILV, PG = 4, 256
R, D, VR = 16, 8, 2
V16 = dataclasses.replace(C.FULL, DCP=8)


# ------------------------------------------------------------------------------ brute-force model
def _owner(t):
    return (t // ILV) % D


def _local(t):
    return (t // (ILV * D)) * ILV + t % ILV


def _rank_positions(seq, r, b):
    """Global positions < seq that rank r scans / attends for request slot b under V16-p."""
    k, h = r % D, r // D
    t = np.arange(seq)
    keep = (_owner(t) == k) & (((_local(t) // PG) + b) % VR == h)
    return t[keep]


# ------------------------------------------------------------------------------ the kernel's expressions
def _kernel_req(seq, nq, r, b):
    """dsa.py per_req_v16 on Python ints: (tile count, nold in tile space, wave weight, hb)."""
    G_, mk, mh = ILV * D, r % D, r // D
    nloc = (seq // G_) * ILV + min(max(seq % G_ - ILV * mk, 0), ILV)
    nl0 = (seq // G_) * ILV + min(max(seq % G_, 0), ILV)
    p0 = seq - nq
    nold8 = (p0 // G_) * ILV + min(max(p0 % G_ - ILV * mk, 0), ILV)
    hb = (mh + b) % VR
    np8 = (nloc + PG - 1) >> 8
    j8n = nold8 >> 8
    nold_t = ((j8n + VR - 1 - hb) // VR) * PG + ((nold8 & (PG - 1)) if (j8n + VR - hb) % VR == 0 else 0)
    nph = (np8 + VR - 1 - hb) // VR
    w = (((nl0 + PG - 1) >> 8) + VR - 1) // VR
    return nph, nold_t, w, hb


def _kernel_gpos(m, hb, r):
    """l8_of_tile then gpos_of_l8."""
    G_, mk = ILV * D, r % D
    l8 = (VR * (m >> 8) + hb) * PG + (m & (PG - 1))
    return (l8 // ILV) * G_ + mk * ILV + l8 % ILV


def _kernel_fix_hit(pos, r, hb):
    """fix(): (hit, tile) of a new token at global position pos on rank r."""
    G_, mk = ILV * D, r % D
    lo2 = (pos // G_) * ILV + pos % ILV
    j8 = lo2 >> 8
    hit = ((pos % G_) // ILV == mk) and ((j8 + VR - hb) % VR == 0)
    return hit, j8 // VR, lo2 & (PG - 1)


SEQS = [1, 3, 4, 5, 31, 32, 33, 255, 256, 257, 2047, 2048, 2049, 2052, 4095, 4096, 4100, 6145, 8191, 8192, 10000,
        131072, 131075, 154626]


@pytest.mark.parametrize("seq", SEQS)
@pytest.mark.parametrize("b", [0, 1, 2, 7])
def test_replicas_partition_each_shard_and_ranks_cover_the_context(seq, b):
    seen = np.zeros(seq, np.int64)
    for r in range(R):
        pos = _rank_positions(seq, r, b)
        seen[pos] += 1
    assert (seen == 1).all()


@pytest.mark.parametrize("seq", SEQS)
@pytest.mark.parametrize("b", [0, 1, 3])
@pytest.mark.parametrize("r", [0, 1, 7, 8, 9, 15])
def test_tile_space_maps_onto_the_rank_positions_in_order(seq, b, r):
    nph, _, _, hb = _kernel_req(seq, 4, r, b)
    want = _rank_positions(seq, r, b)
    got = []
    for p in range(nph):
        for s in range(PG):
            g = _kernel_gpos(p * PG + s, hb, r)
            if g < seq:
                got.append(g)
    assert got == sorted(got)  # tile order = local (and global) order within a rank
    assert got == want.tolist()
    # the last tile holds at least one position (the tile count is exact, not padded)
    if nph:
        assert any(_kernel_gpos((nph - 1) * PG + s, hb, r) < seq for s in range(PG))


@pytest.mark.parametrize("seq", SEQS)
@pytest.mark.parametrize("nq", [1, 2, 4])
@pytest.mark.parametrize("b", [0, 1])
@pytest.mark.parametrize("r", [0, 3, 8, 12])
def test_nold_in_tile_space_is_exact(seq, nq, b, r):
    if seq < nq:
        return
    nph, nold_t, _, hb = _kernel_req(seq, nq, r, b)
    p0 = seq - nq
    for p in range(nph):
        for s in range(PG):
            m = p * PG + s
            g = _kernel_gpos(m, hb, r)
            if g < seq:
                assert (m < nold_t) == (g < p0), (m, g, p0, nold_t)


@pytest.mark.parametrize("seq", SEQS)
def test_wave_weight_is_the_max_tile_count_over_ranks_and_requests(seq):
    ws = {_kernel_req(seq, 4, r, b)[2] for r in range(R) for b in range(4)}
    assert len(ws) == 1  # rank-uniform
    assert ws.pop() == max(_kernel_req(seq, 4, r, b)[0] for r in range(R) for b in range(4))


@pytest.mark.parametrize("seq", [5, 257, 2049, 4100, 131075])
@pytest.mark.parametrize("b", [0, 1])
def test_fix_patches_each_new_token_on_exactly_its_two_owners_tile(seq, b):
    for pos in range(seq - 4, seq):
        owners = []
        for r in range(R):
            nph, _, _, hb = _kernel_req(seq, 4, r, b)
            hit, tile, slot = _kernel_fix_hit(pos, r, hb)
            if hit:
                owners.append(r)
                assert tile < nph
                assert _kernel_gpos(tile * PG + slot, hb, r) == pos
        assert len(owners) == 1  # exactly one rank scans / attends it
        assert owners[0] % D == _owner(pos)


def test_virtual_order_is_shard_major():
    v = [VR * (r % D) + r // D for r in range(R)]
    assert sorted(v) == list(range(R))
    assert v[0] < v[8] < v[1] < v[9]  # shard 0's replicas, then shard 1's


@pytest.mark.parametrize("rank", [0, 5, 8, 13])
def test_v16_tables_pick_the_replicas_pages(rank):
    rng = np.random.default_rng(rank)
    W8 = 17
    bt8 = jnp.asarray(rng.integers(0, 1000, (8, W8)), jnp.int32)
    bti8 = bt8 + 7
    bt, bti = W.v16_tables(V16, bt8, bti8, jnp.int32(rank))
    assert bt.shape == (8, 9) and bti.shape == (8, 9)
    for b in range(8):
        hb = (rank // D + b) % VR
        for p in range(9):
            j = min(VR * p + hb, W8 - 1)
            assert int(bt[b, p]) == int(bt8[b, j]) and int(bti[b, p]) == int(bti8[b, j])


def test_kernel_pages():
    assert W.kernel_pages(C.FULL, 256) == 256
    assert W.kernel_pages(V16, 512) == 256 and W.kernel_pages(V16, 513) == 257


def test_config_admission():
    assert V16.D == 8 and V16.VR == 2 and C.FULL.D == 16 and C.FULL.VR == 1
    V16.check()
    for bad in (4, 3, 32):
        with pytest.raises(AssertionError):
            dataclasses.replace(C.FULL, DCP=bad).check()
    # owner / local follow the cache's DCP
    assert [V16.owner(p) for p in (0, 4, 31, 32, 36)] == [0, 1, 7, 0, 1]
    assert V16.local(36) == 4 and C.FULL.local(36) == 0 and C.FULL.local(68) == 4


@pytest.mark.parametrize("seq", [1, 255, 2048, 2049, 131072, 655360, 1 << 20])
def test_rank0_pages_v16_is_the_max_replica_tiles(seq):
    got = A.rank0_pages(seq, page=PG, R=D, ilv=ILV, vr=VR)
    assert got == max(_kernel_req(seq, 1, r, b)[0] for r in range(R) for b in range(2))
    assert A.rank0_pages(seq, page=PG, R=16, ilv=ILV) == A.rank0_pages(seq, page=PG, R=16, ilv=ILV, vr=1)


def test_pool_guard_v16_headroom_equals_dcp16():
    assert A.check_pool(1 << 20, {16: 1344, 32: 1344}, page=PG, R=D, ilv=ILV, vr=VR) == 256
    assert A.check_pool(655360, {16: 1344}, page=PG, R=D, ilv=ILV, vr=VR) == 160


class _QB:
    def __init__(self, gs, rig):
        self.qrep_active, self.group_size, self.rank_in_group = True, gs, rig


@pytest.mark.parametrize("rank", [0, 7, 8, 15])
def test_qrep_layer_accepts_the_dcp8_group(rank):
    assert A.qrep_layer(_QB(8, rank % 8), R, rank, 0, 8)
    with pytest.raises(A.Unsupported):
        A.qrep_layer(_QB(16, rank), R, rank, 0, 8)
    with pytest.raises(A.Unsupported):
        A.qrep_layer(_QB(8, (rank + 1) % 8), R, rank, 0, 8)
    assert A.qrep_layer(_QB(16, rank), R, rank, 0)  # incumbent DCP16 form unchanged


@pytest.mark.parametrize("rank", [0, 3, 8, 15])
def test_q_b_local_v16_slices_the_group_weight_like_the_column_shard(rank):
    import torch

    cfg = V16
    nqb, DQ, QL = cfg.HR * cfg.DQ, cfg.DQ, cfg.QL
    full = torch.randint(-100, 100, (QL, cfg.NH * DQ), dtype=torch.int32).to(torch.float32)
    sfull = torch.rand(cfg.NH * DQ)
    lane = rank // D
    grp = full[:, lane * D * nqb : (lane + 1) * D * nqb]
    sgrp = sfull[lane * D * nqb : (lane + 1) * D * nqb]
    w, s = A.q_b_local(grp, sgrp, cfg, rank, qrep=True)
    assert torch.equal(w, full[:, rank * nqb : (rank + 1) * nqb])
    assert torch.equal(s, sfull[rank * nqb : (rank + 1) * nqb])
    # the [N, QL] orientation too
    w2, _ = A.q_b_local(grp.t().contiguous(), sgrp, cfg, rank, qrep=True)
    assert torch.equal(w2, w)
    with pytest.raises(A.Unsupported):  # a 64-head weight is not a DCP8 group weight
        A.q_b_local(full, sfull, cfg, rank, qrep=True)
