# SPDX-License-Identifier: Apache-2.0
"""CPU unit + Pallas interpret parity tests for VLLM_TPU_MK_GDN_VEC_REC."""

from __future__ import annotations

import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu import envs
from vllm_torchtpu.megakernel import kernel
from vllm_torchtpu.megakernel.blocks.gdn.params import GDNTunables
from vllm_torchtpu.megakernel.blocks.hc import RESIDUAL_OUT
from vllm_torchtpu.megakernel.exec_spec import ExecSpec, Tiles
from vllm_torchtpu.megakernel.models import qwen38
from vllm_torchtpu.megakernel.models.qwen38_inputs import (
    pack_segment,
    random_segment_inputs,
)
from vllm_torchtpu.megakernel.models.qwen38_pack import shard_step_inputs
from vllm_torchtpu.megakernel_runtime import qwen38_segment_call as sc
from vllm_torchtpu.megakernel_runtime import qwen38_wholestep_call as wc

TILES = Tiles(bank=1 << 20, nbd=10, nbe=8, ebank=512 << 10, vmem_mib=64)


def test_gdn_vec_rec_default_off_and_env_wiring(monkeypatch):
    monkeypatch.delenv("VLLM_TPU_MK_GDN_VEC_REC", raising=False)
    assert GDNTunables().vec_rec is False
    assert GDNTunables.NUMERICS["vec_rec"] == "approx"
    assert not envs.VLLM_TPU_MK_GDN_VEC_REC
    assert sc.segment_tiles(16).block("gdn").vec_rec is False
    assert wc.wholestep_tiles(32).block("gdn").vec_rec is False

    monkeypatch.setenv("VLLM_TPU_MK_GDN_VEC_REC", "1")
    assert sc.segment_tiles(16).block("gdn").vec_rec is True
    wt = wc.wholestep_tiles(32)
    assert wt.block("gdn").vec_rec is True
    assert wt.block("gdn").gdn_head_major is True


@pytest.mark.parametrize("B,hm", [(8, False), (24, False), (24, True)])
def test_gdn_vec_rec_matches_scalar_segment(B: int, hm: bool):
    """Vectorized conv_step + recurrence closely matches baseline on both
    whole-state (B=8) and WIDE chunked (B=24, head-major off/on) paths."""
    if len(jax.devices()) < 2:
        pytest.skip("requires >= 2 host CPU devices")
    R, L, T = 2, qwen38.SEGMENT_LAYERS, 4
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    spec = qwen38.segment_spec(qwen38.TINY, R, L=L, pool_gdn=False)

    def _run(vec_rec: bool):
        tiles = TILES.with_blocks({"gdn": {"gdn_head_major": hm, "vec_rec": vec_rec}})
        ex = ExecSpec(R=R, B=B, T=T, ctx=128, interpret=True, tiles=tiles)
        inp = random_segment_inputs(jax.random.PRNGKey(13), qwen38.TINY, ex, L, gain=0.25)
        w_mk, st_mk, attn = pack_segment(spec, qwen38.TINY, ex, inp)
        pos, hid, w_s, st_s = shard_step_inputs(mesh, ex, inp.pos, inp.x, w_mk, st_mk)
        tp = NamedSharding(mesh, P("tp"))
        step, _ = kernel.make_step(mesh, spec, ex)
        _, outs = step(pos, hid, w_s, st_s, jax.device_put(kernel.fold(attn), tp))
        return {k: np.asarray(v, np.float32) for k, v in outs.items()}

    ref = _run(False)
    got = _run(True)
    np.testing.assert_allclose(got[RESIDUAL_OUT], ref[RESIDUAL_OUT], atol=1e-2, rtol=1e-2)
    np.testing.assert_allclose(got["conv_ckpt"], ref["conv_ckpt"], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(got["rec_ckpt"], ref["rec_ckpt"], atol=2e-2, rtol=2e-2)
