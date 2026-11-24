# SPDX-License-Identifier: Apache-2.0
"""CPU unit + Pallas interpret parity tests for VLLM_TPU_MK_UNION_SKIP_ROWCOMPACT."""

from __future__ import annotations

import jax
import numpy as np
import pytest
from jax.sharding import Mesh

from vllm_torchtpu.megakernel import kernel, vmem
from vllm_torchtpu.megakernel.blocks.hc import RESIDUAL_OUT
from vllm_torchtpu.megakernel.exec_spec import ExecSpec, Tiles
from vllm_torchtpu.megakernel.models import qwen38
from vllm_torchtpu.megakernel.models.qwen38_pack import shard_step_inputs
from vllm_torchtpu.megakernel.wholestep import inputs as wi
from vllm_torchtpu.megakernel.wholestep.qsa_full_c import (
    CK_NEW,
    IR_NEW,
    KR_NEW,
    RING_NEW,
)
from vllm_torchtpu.megakernel.wholestep.spec import wholestep_spec
from vllm_torchtpu.megakernel_runtime import qwen38_wholestep_call as wc

_TILES = Tiles(bank=1 << 20, nbd=10, nbe=8, ebank=512 << 10, vmem_mib=64)
G = qwen38.TINY


def test_skip_rowcompact_scratch_and_exchange_width(monkeypatch):
    """VLLM_TPU_MK_UNION_SKIP_ROWCOMPACT drops `qi_ids` SMEM and (when not debug)
    `qi_id` VMEM, shrinking the sharded selection exchange width 5x (640 -> 128)."""
    monkeypatch.setattr(wc, "UNION_ATTN", True)
    monkeypatch.delenv("VLLM_TPU_MK_UNION_SKIP_ROWCOMPACT", raising=False)
    assert "skip_rowcompact" not in wc.indexed_options()
    monkeypatch.setenv("VLLM_TPU_MK_UNION_SKIP_ROWCOMPACT", "1")
    assert wc.indexed_options().get("skip_rowcompact") is True

    ex = ExecSpec(R=2, B=16, T=4, ctx=128, tiles=_TILES)
    common = dict(
        max_groups=8192,
        pages_per_seq=256,
        carrier="fp8",
        n_layers=12,
        hbm_tables=True,
        per_request_scores=True,
        bounded_chunks=True,
        shard_index=True,
        union_attn=True,
        loop_requests=True,
    )

    def _build(kw):
        sp = wholestep_spec(
            G, 2, L=4, first="qsa", qsa=dict(budget=512), step=True, indexed=kw
        )
        blk = sp.layer_type.slots[0].branches[0].inner
        _, scr = vmem.build_scratch(sp, ex)
        return blk, scr

    blk_off, scr_off = _build(common)
    blk_on, scr_on = _build({**common, "skip_rowcompact": True})
    blk_dbg, scr_dbg = _build({**common, "skip_rowcompact": True, "debug": True})

    assert "qi_ids" in scr_off and "qi_id" in scr_off
    assert blk_off._shard_rows(ex)[2] == 640
    assert "qi_ids" not in scr_on and "qi_id" not in scr_on
    assert blk_on._shard_rows(ex)[2] == 128
    assert "qi_ids" not in scr_dbg and "qi_id" in scr_dbg
    assert blk_dbg._shard_rows(ex)[2] == 640


@pytest.mark.parametrize("union_bounded", [False, True])
def test_skip_rowcompact_bit_identical(union_bounded: bool):
    if len(jax.devices()) < 2:
        pytest.skip("requires >= 2 host CPU devices")
    R, T, Lt = 2, 4, 4
    budget, pages, max_groups = 64, 8, 1024
    ctx_c = [301, 100]
    ex = ExecSpec(R=R, B=2 * T, T=T, ctx=128, interpret=True, tiles=_TILES)
    d = wi.random_wholestep_inputs(
        31, G, ex, Lt, budget=budget, n_pages=pages, identical_experts=True, first="qsa"
    )
    d = wi.add_indexer_inputs(31, G, ex, d, Lt, max_groups=max_groups, ctx=ctx_c)
    d = wi.add_step_inputs(31, G, ex, d, Lt, ctx=ctx_c)
    d = wi.add_per_layer(31, G, ex, d, Lt)
    d = wi.add_carrier(31, d)
    d["hbm_tables"] = True
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))

    def _run(skip_rc: bool):
        idx = dict(
            max_groups=max_groups,
            pages_per_seq=d["mtab"].shape[-1],
            carrier="bf16",
            chunk_pages=8,
            n_layers=1,
            hbm_tables=True,
            per_request_scores=True,
            bounded_chunks=True,
            shard_index=True,
            union_attn=True,
            loop_requests=True,
            skip_rowcompact=skip_rc,
            debug=False,
        )
        if union_bounded:
            idx["union_bounded"] = True
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
