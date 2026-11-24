# SPDX-License-Identifier: Apache-2.0
"""CPU unit + Pallas interpret parity tests for VLLM_TPU_MK_XLAYER_PREFETCH."""

from __future__ import annotations

import dataclasses

import jax
import numpy as np
import pytest
from jax.sharding import Mesh

from vllm_torchtpu import envs
from vllm_torchtpu.megakernel import kernel
from vllm_torchtpu.megakernel.blocks.hc import RESIDUAL_OUT
from vllm_torchtpu.megakernel.exec_spec import ExecSpec, Tiles
from vllm_torchtpu.megakernel.models import qwen38
from vllm_torchtpu.megakernel.models.qwen38_gdn_pool import gdn_slot_geometry
from vllm_torchtpu.megakernel.models.qwen38_pack import shard_step_inputs
from vllm_torchtpu.megakernel.wholestep import inputs as wi
from vllm_torchtpu.megakernel.wholestep.qsa_full_c import (
    CK_NEW,
    IR_NEW,
    KR_NEW,
    RING_NEW,
)
from vllm_torchtpu.megakernel.wholestep.spec import wholestep_spec
from vllm_torchtpu.megakernel_runtime import qwen38_segment_call as sc
from vllm_torchtpu.megakernel_runtime import qwen38_wholestep_call as wc

_TILES = Tiles(bank=1 << 20, nbd=10, nbe=8, ebank=512 << 10, vmem_mib=64)
G = qwen38.TINY


def test_xlayer_prefetch_env_wiring(monkeypatch):
    monkeypatch.delenv("VLLM_TPU_MK_XLAYER_PREFETCH", raising=False)
    assert not envs.VLLM_TPU_MK_XLAYER_PREFETCH
    assert sc.segment_tiles(16).xlayer_prefetch is False
    assert wc.wholestep_tiles(16).xlayer_prefetch is False
    monkeypatch.setenv("VLLM_TPU_MK_XLAYER_PREFETCH", "1")
    assert sc.segment_tiles(16).xlayer_prefetch is True
    assert wc.wholestep_tiles(16).xlayer_prefetch is True


def test_gdn_slot_geometry_tp8_and_tp4():
    # TP=8: vh=6, KD=128, VD=128, QKV=1280 -> (3, 16, 400, 512)
    assert gdn_slot_geometry(6, 128, 128, 1280) == (3, 16, 400, 512)
    # TP=4: vh=12, KD=128, VD=128, QKV=2560 -> (6, 32, 800, 896)
    assert gdn_slot_geometry(12, 128, 128, 2560) == (6, 32, 800, 896)


@pytest.mark.parametrize("table_prefetch", [False, True])
def test_xlayer_prefetch_wholestep_bit_identical(table_prefetch: bool):
    """Cross-layer prefetch is bit-identical across both HBM table paths
    (plain `load_tables` and TSLIM `table_prefetch=True`)."""
    if len(jax.devices()) < 2:
        pytest.skip("requires >= 2 host CPU devices")
    R, T, Lt = 2, 4, 8
    budget, pages, max_groups = 64, 8, 1024
    ctx_c = [301, 100]
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))

    def _run(xp: bool):
        tiles = dataclasses.replace(_TILES, xlayer_prefetch=xp)
        ex = ExecSpec(R=R, B=2 * T, T=T, ctx=128, interpret=True, tiles=tiles)
        d = wi.random_wholestep_inputs(
            23, G, ex, Lt, budget=budget, n_pages=pages, identical_experts=True, first="qsa"
        )
        d = wi.add_indexer_inputs(23, G, ex, d, Lt, max_groups=max_groups, ctx=ctx_c)
        d = wi.add_step_inputs(23, G, ex, d, Lt, ctx=ctx_c)
        d = wi.add_per_layer(23, G, ex, d, Lt)
        d = wi.add_carrier(23, d)
        d["hbm_tables"] = True
        idx = dict(
            max_groups=max_groups,
            pages_per_seq=d["mtab"].shape[-1],
            carrier="bf16",
            chunk_pages=8,
            n_layers=2,
            hbm_tables=True,
            per_request_scores=True,
            bounded_chunks=True,
            shard_index=True,
            union_attn=True,
            loop_requests=True,
        )
        if table_prefetch:
            idx["table_prefetch"] = True
        sp = wholestep_spec(
            G, R, L=Lt, qsa=dict(budget=budget), step=True, indexed=idx, first="qsa"
        )
        w_mk, states = wi.pack_wholestep(sp, G, ex, d, Lt)
        pos_mk, hid_mk, w_mk, st_mk = shard_step_inputs(
            mesh, ex, d["inp"].pos, d["inp"].x, w_mk, states
        )
        step, _ = kernel.make_step(mesh, sp, ex)
        _, outs = step(pos_mk, hid_mk, w_mk, st_mk, *wi.extras_c(d, mesh, R))
        return outs

    base = _run(False)
    fast = _run(True)
    for k in (RESIDUAL_OUT, CK_NEW, RING_NEW, KR_NEW, IR_NEW):
        np.testing.assert_array_equal(np.asarray(fast[k]), np.asarray(base[k]), err_msg=k)
