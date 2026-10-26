# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks of the glm52_decode_mk serving adapter's host-side helpers."""

import re

import pytest
import torch

from vllm_torchtpu.kernels.glm52_decode_mk import config as C
from vllm_torchtpu.kernels.glm52_decode_mk.vllm_adapter import (
    Unsupported,
    index_pattern,
    q_b_local,
    qrep_layer,
)


def _pat(fr, sk, L):
    cfg = C.MKConfig(
        R=16, L=L, first_dense=1, H=6144, NH=64, QL=2048, idx_freq=fr, idx_skip=sk
    )
    return [cfg.full_indexer(l) for l in range(L)]


def test_served_pattern_keeps_hf_values():
    served = _pat(4, 3, 78)
    assert index_pattern(served, 4, 3) == (4, 3)


def test_twin_fFs_maps_to_skip2():
    # the mirror twin's index_topk_pattern [F, F, S] (hf freq 4 / skip 3 would say F, F, F)
    fr, sk = index_pattern([True, True, False], 4, 3)
    assert _pat(fr, sk, 3) == [True, True, False]
    assert (fr, sk) == (4, 2)


def test_ten_layer_twin():
    pat = [True, True, True, False, False, False, True, False, False, False]
    assert index_pattern(pat, 4, 3) == (4, 3)


def test_unrepresentable_pattern_declines():
    with pytest.raises(Unsupported):
        index_pattern([False, True, False, True, True], 4, 3)


# ------------------------------------------------------------------ q_b_local: qrep 0 as is, qrep 1 sliced

CF = C.FULL
NQB, NALL = CF.HR * CF.DQ, CF.NH * CF.DQ


def _fp8(shape, seed):
    g = torch.Generator().manual_seed(seed)
    raw = torch.randint(0, 256, shape, dtype=torch.uint8, generator=g)
    raw[(raw & 0x7F) == 0x7F] = 0  # no NaN codes
    return raw.view(torch.float8_e4m3fn)


def _bytes(t):
    return t.contiguous().view(torch.uint8)


def test_q_b_local_qrep0_is_the_served_tensor():
    w = _fp8((CF.QL, NQB), 0)
    s = torch.rand(NQB)
    qw, qs = q_b_local(w, s, CF, rank=3)
    assert qw is w and qs.dtype == torch.float32 and torch.equal(qs, s)


@pytest.mark.parametrize("flipped", [True, False])
def test_q_b_local_qrep1_slice_is_the_column_parallel_shard(flipped):
    full = _fp8((CF.QL, NALL), 1)  # [K, N]: all 64 heads in head order
    scale = torch.rand(NALL)
    served = full if flipped else full.t().contiguous()
    for r in range(CF.R):
        cols = slice(r * NQB, (r + 1) * NQB)
        shard = full[
            :, cols
        ].contiguous()  # what ColumnParallelLinear at TP16 rank r holds
        qw, qs = q_b_local(served, scale, CF, rank=r, qrep=True)
        assert tuple(qw.shape) == (CF.QL, NQB)
        assert torch.equal(_bytes(qw), _bytes(shard)), r
        assert torch.equal(qs, scale[cols]), r
        # positive control: a neighbour's heads are a different operand
        nw, _ = q_b_local(served, scale, CF, rank=(r + 1) % CF.R, qrep=True)
        assert not torch.equal(_bytes(nw), _bytes(shard))


@pytest.mark.parametrize(
    "n, qrep, scale_n, msg",
    [
        (NALL, False, NALL, "want N 1024 (qrep 0"),
        (NQB, True, NQB, "want N 16384 (qrep 1"),
        (NALL, True, NQB, "is not per-channel [16384]"),
        (NQB, False, 8, "is not per-channel [1024]"),
    ],
)
def test_q_b_local_form_mismatch_declines(n, qrep, scale_n, msg):
    with pytest.raises(Unsupported, match=re.escape(msg)):
        q_b_local(_fp8((CF.QL, n), 2), torch.rand(scale_n), CF, rank=0, qrep=qrep)


def test_qrep_layer_accepts_only_the_world_group():
    class Q:
        pass

    q = Q()
    assert qrep_layer(q, 16, 5, 0) is False  # column-parallel
    q.qrep_active, q.group_size, q.rank_in_group = True, 16, 5
    assert qrep_layer(q, 16, 5, 0) is True
    for gs, rig in ((8, 5), (16, 4), (1, 0)):
        q.group_size, q.rank_in_group = gs, rig
        with pytest.raises(Unsupported, match="over a q group of"):
            qrep_layer(q, 16, 5, 7)
