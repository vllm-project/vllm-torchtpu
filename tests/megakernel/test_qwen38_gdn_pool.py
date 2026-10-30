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
"""Qwen3.8 segment kernel reading GDN states from a unified-pool byte layout (CPU)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.megakernel import kernel
from vllm_torchtpu.megakernel.blocks.hc import RESIDUAL_OUT
from vllm_torchtpu.megakernel.exec_spec import ExecSpec, Tiles
from vllm_torchtpu.megakernel.models import qwen38
from vllm_torchtpu.megakernel.models.qwen38_inputs import pack_segment, random_segment_inputs
from vllm_torchtpu.megakernel.models.qwen38_pack import shard_step_inputs
from vllm_torchtpu.megakernel.models.qwen38_reference import segment_reference

G = qwen38.TINY
TILES = Tiles(bank=1 << 20, nbd=10, nbe=8, ebank=512 << 10, vmem_mib=64)
LANES = 256


def _write_f32_rows(block: np.ndarray, rows: np.ndarray) -> None:
    """f32 typed rows -> one kernel block's bytes `[128, 1, 4, 256]` (4 pack rows per word,
    little-endian, row 0 in the low byte), starting at typed row 0."""
    kb, h2p, pack, lanes = block.shape
    words = np.zeros((kb * h2p * pack // 4, lanes), np.uint32)
    old = np.moveaxis(block.reshape(kb, h2p, pack // 4, 4, lanes), 3, -1).copy().view(np.uint32)
    words[...] = old.reshape(-1, lanes)
    words[: rows.shape[0]] = rows.view(np.uint32)
    b = words.reshape(kb, h2p, pack // 4, lanes, 1).view(np.uint8)
    block[...] = np.moveaxis(b, -1, 3).reshape(kb, h2p, pack, lanes)


def _encode(raw: np.ndarray, m: int, ssm: np.ndarray, conv: np.ndarray) -> None:
    flat = ssm.astype(np.float32).reshape(3, -1, LANES)
    for j in range(3):
        _write_f32_rows(raw[4 * m + j], flat[j])
    _write_f32_rows(raw[4 * m + 3], conv.astype(np.float32).reshape(-1, LANES))


@pytest.mark.parametrize("fresh", [False, True])
def test_pool_gdn_segment_matches_reference(fresh: bool):
    R, L, T, NS = 2, qwen38.SEGMENT_LAYERS, 4, 2
    B, NL = NS * T, L - 1
    ex = ExecSpec(R=R, B=B, T=T, ctx=128, interpret=True, tiles=TILES)
    spec = qwen38.segment_spec(G, R, L=L, pool_gdn=True)
    inp = random_segment_inputs(jax.random.PRNGKey(9), G, ex, L, gain=0.25)
    rng = np.random.default_rng(1)
    nmb = 1 + NL * NS
    rb = 1 + np.arange(NL * NS, dtype=np.int32).reshape(NL, NS)
    raws, conv0, rec0 = [], np.zeros((R, NL, NS, 3, 1280), np.float32), np.zeros((R, NL, NS, 6, 128, 128), np.float32)
    for r in range(R):
        raw = np.zeros((nmb * 4, 128, 1, 4, LANES), np.uint8)
        for l in range(NL):
            for s in range(NS):
                ssm = rng.standard_normal((6, 128, 128)).astype(np.float32) * 0.05
                conv = rng.standard_normal((3, 1280)).astype(np.float32) * 0.05
                _encode(raw, rb[l, s], ssm, conv)
                if not fresh:
                    rec0[r, l, s], conv0[r, l, s] = ssm, conv
        raws.append(raw)
    inp.st_l = {"conv": jnp.asarray(conv0), "rec": jnp.asarray(rec0).astype(jnp.bfloat16)}
    ref, ref_states = segment_reference(
        G, ex, inp.x, inp.hc, inp.w_l, inp.st_l, inp.attn, inp.o_proj, L=L
    )

    w_mk, _, attn = pack_segment(spec, G, ex, inp)
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    pos, hid, w_mk, st = shard_step_inputs(mesh, ex, inp.pos, inp.x, w_mk, {})
    tp, rep = NamedSharding(mesh, P("tp")), NamedSharding(mesh, P())
    pool = jnp.concatenate([jax.lax.bitcast_convert_type(jnp.asarray(r), jnp.float8_e4m3fn) for r in raws])
    extras = (
        jax.device_put(kernel.fold(attn), tp),
        jax.device_put(pool, tp),
        jax.device_put(jnp.asarray(rb), rep),
        jax.device_put(jnp.full((NL, NS), 0 if fresh else 1, jnp.int32), rep),
    )
    step, _ = kernel.make_step(mesh, spec, ex)
    _, outs = step(pos, hid, w_mk, st, *extras)
    res = np.asarray(outs[RESIDUAL_OUT].reshape(R, ex.M, -1)[:, :B], np.float32)
    refn = np.asarray(ref, np.float32)
    scale = float(np.abs(refn).mean())
    for r in range(R):
        assert float(np.abs(res[r] - refn).mean()) < 1e-2 * scale
    rec_mk = np.asarray(outs["rec_ckpt"], np.float32).reshape(R, NL, NS, T, 6, 128, 128)
    for l in range(NL):
        rec_ref = np.asarray(ref_states[l]["rec_ckpt"], np.float32).reshape(R, NS, T, 6, 128, 128)
        np.testing.assert_allclose(rec_mk[:, l], rec_ref, atol=2e-2, rtol=2e-2)


def _wide_pool_step(monkeypatch, chunk_gather: str, has_state: np.ndarray):
    """Residual and checkpoints of a WIDE (chunked-state) pool segment step."""
    from vllm_torchtpu.megakernel.models import qwen38_gdn_pool

    monkeypatch.setattr(qwen38_gdn_pool, "CHUNK_GATHER", chunk_gather)
    R, L, T, NS = 2, qwen38.SEGMENT_LAYERS, 4, 6
    B, NL = NS * T, L - 1
    ex = ExecSpec(R=R, B=B, T=T, ctx=128, interpret=True, tiles=TILES)
    assert ex.WIDE
    spec = qwen38.segment_spec(G, R, L=L, pool_gdn=True)
    inp = random_segment_inputs(jax.random.PRNGKey(5), G, ex, L, gain=0.25)
    rng = np.random.default_rng(3)
    nmb = 1 + NL * NS
    rb = 1 + rng.permutation(NL * NS).astype(np.int32).reshape(NL, NS)
    raws = []
    for _ in range(R):
        raw = np.zeros((nmb * 4, 128, 1, 4, LANES), np.uint8)
        for l in range(NL):
            for s in range(NS):
                ssm = rng.standard_normal((6, 128, 128)).astype(np.float32) * 0.05
                conv = rng.standard_normal((3, 1280)).astype(np.float32) * 0.05
                _encode(raw, rb[l, s], ssm, conv)
        raws.append(raw)
    w_mk, _, attn = pack_segment(spec, G, ex, inp)
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    pos, hid, w_mk, st = shard_step_inputs(mesh, ex, inp.pos, inp.x, w_mk, {})
    tp, rep = NamedSharding(mesh, P("tp")), NamedSharding(mesh, P())
    pool = jnp.concatenate([jax.lax.bitcast_convert_type(jnp.asarray(r), jnp.float8_e4m3fn) for r in raws])
    extras = (
        jax.device_put(kernel.fold(attn), tp),
        jax.device_put(pool, tp),
        jax.device_put(jnp.asarray(rb), rep),
        jax.device_put(jnp.asarray(has_state, jnp.int32), rep),
    )
    step, _ = kernel.make_step(mesh, spec, ex)
    _, outs = step(pos, hid, w_mk, st, *extras)
    return {k: np.asarray(outs[k]) for k in (RESIDUAL_OUT, "rec_ckpt", "conv_ckpt")}


def test_pool_gdn_chunk_gather_is_exact(monkeypatch):
    """Chunk gather (states read per `recb` slot) == states staged in `pg_rec`, bitwise,
    with a mix of requests with and without a prior state."""
    NL, NS = qwen38.SEGMENT_LAYERS - 1, 6
    has_state = (np.arange(NL * NS).reshape(NL, NS) % 3 != 1).astype(np.int32)
    staged = _wide_pool_step(monkeypatch, "0", has_state)
    chunked = _wide_pool_step(monkeypatch, "1", has_state)
    for k in staged:
        np.testing.assert_array_equal(chunked[k], staged[k], err_msg=k)
