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
"""The Qwen3.5 MTP draft layer on an attention-only megakernel step (`draft.py`).

A one-layer attention-only stack of the tiny geometry runs in Pallas interpret
mode on CPU: a draft step writes K/V only for its active sequences, a loop step
over the updated pool reproduces the verify step's last row, padded rows leave
the live rows and the pool unchanged, step 0 routes only the gathered rows
through the experts, and under decode context parallelism the new tokens land
only on the rank that owns their page.
"""

from __future__ import annotations

import dataclasses
import time
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.megakernel import core as mk
from vllm_torchtpu.models.qwen3_5.megakernel import draft, model
from vllm_torchtpu.models.qwen3_5.megakernel.profiles import baseline
from vllm_torchtpu.models.qwen3_5.megakernel.profiles.baseline import TINY_TILES
from vllm_torchtpu.models.qwen3_5.megakernel.tests.tiny import (
    make_synthetic_backbone,
    tiny_geom,
)

PS = 128
NUM_PAGES = 40
# The runner's token paddings of the AgentX decode leg.
PADDINGS = (4, 8, 16, 24, 32, 48, 64, 96, 128, 256)
LOOP = (4, 8, 12, 16, 20, 24, 28, 32)
VERIFY = (4, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64)


def _u8(x: Any) -> np.ndarray:
    return np.asarray(jax.lax.bitcast_convert_type(jnp.asarray(x), jnp.uint8))


def _rand_pool_u8(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    """Finite random fp8 bytes (exponent bits below 0x40, any sign)."""
    u8 = rng.integers(0, 0x40, shape, dtype=np.uint8)
    u8 |= (rng.integers(0, 2, shape, dtype=np.uint8) << 7).astype(np.uint8)
    return u8


def _fp8(u8: np.ndarray, sharding: Any = None) -> jax.Array:
    arr = jax.lax.bitcast_convert_type(jnp.asarray(u8), jnp.float8_e4m3fn)
    return jax.device_put(arr, sharding) if sharding is not None else arr


def _geom(R: int) -> Any:
    """One attention layer of the tiny geometry; at `R = 8` two KV heads (four
    ranks per KV head, as at TP8) and 128 shared-expert columns per rank."""
    if R == 1:
        return tiny_geom(L=1, att_every=1)
    return model.Qwen35Geometry(
        L=1,
        att_every=1,
        H=256,
        I=256,
        IS=1024,
        E=128,
        TOPK=2,
        NH=8,
        NKV=2,
        HD=256,
        ROT=64,
        KH=8,
        VH=16,
        KD=128,
        VD=128,
        moe_tp=False,
        e_slots=0,
    )


def _base_exec(R: int, skip: frozenset[str]) -> mk.ExecSpec:
    tiles = dataclasses.replace(TINY_TILES, bank=512 << 10).with_modules(
        {"gqa": {"kv_blk": 256, "nkv": 2, "kv_sub": 0, "split": True}}
    )
    return mk.ExecSpec(
        R=R,
        B=8,
        T=1,
        ctx=8 * PS,
        stack_e=False,
        interpret=True,
        tiles=tiles,
        debug=mk.DebugOptions(skip=skip),
        cache=mk.CacheExec(paged_kv=True, num_pages=NUM_PAGES),
    )


def _program(
    *,
    R: int = 1,
    loop: tuple[int, ...] = (4,),
    verify: tuple[int, ...] = (8, 16),
    dcp: int = 1,
    skip: frozenset[str] = frozenset(),
    signed_weights: bool = False,
) -> draft.Qwen35DraftProgram:
    """A bound draft program for a tiny one-layer stack on `R` CPU devices.

    `signed_weights`: router, shared-expert and expert weights of either sign
    and moderate size. The synthetic ones are all positive and large, so most
    rows route alike and their experts' outputs underflow to exactly zero."""
    devs = jax.devices("cpu")
    if len(devs) < R:
        pytest.skip(f"needs {R} CPU devices")
    mesh = Mesh(np.array(devs[:R]), ("tp",))
    g = _geom(R)
    ex = _base_exec(R, skip)
    torch.manual_seed(0)
    bb = make_synthetic_backbone(g, ex)
    layer = bb.layers[0]
    if signed_weights:
        rng = np.random.default_rng(11)
        w = layer.mlp._fused_gate_gu_fp8_weight
        u8 = torch.from_numpy(_rand_pool_u8(rng, tuple(w.shape)))
        layer.mlp._fused_gate_gu_fp8_weight = u8.view(torch.float8_e4m3fn)
        for p in (layer.mlp.experts.w13_weight, layer.mlp.experts.w2_weight):
            u8 = torch.from_numpy(_rand_pool_u8(rng, tuple(p.shape)))
            p.data.view(torch.uint8).copy_(u8)
    kvh = max(1, g.NKV // R)
    pool = torch.zeros((NUM_PAGES, 2 * kvh, g.HD // 4, 4, PS), dtype=torch.uint8)
    layer.self_attn.attn.kv_cache = [pool.view(torch.float8_e4m3fn)]
    prog = draft.Qwen35DraftProgram(
        layer,
        bb.norm,
        mesh,
        profile=baseline.PROFILE,
        tp_rank=0,
        dcp_size=dcp,
        loop_buckets=loop,
        verify_buckets=verify,
        geom=g,
        base_exec=ex,
    )
    prog.bind()
    return prog


def _rep(x: Any, R: int) -> jax.Array:
    """Rank-folded operand: each rank's (identical) copy stacked on dim 0."""
    x = jnp.asarray(x)
    return jnp.concatenate([x] * R, axis=0) if R > 1 else x


def _to_jax(t: torch.Tensor) -> jax.Array:
    t = t.detach().cpu().contiguous()
    if t.dtype == torch.float8_e4m3fn:
        return jnp.asarray(t.view(torch.uint8).numpy()).view(jnp.float8_e4m3fn)
    if t.dtype == torch.bfloat16:
        return jnp.asarray(t.view(torch.uint16).numpy()).view(jnp.bfloat16)
    return jnp.asarray(t.numpy())


def _leaves(prog: draft.Qwen35DraftProgram) -> list[jax.Array]:
    """The program's weight leaves, rank 0's on every rank."""
    return [_rep(_to_jax(t), prog.R) for t in prog._leaves(prog.layer)]


def _run(
    prog: draft.Qwen35DraftProgram,
    T: int,
    hidden: Any,
    block_tables: Any,
    seq_lens: Any,
    qsl: Any,
    pool_u8: np.ndarray,
    gather: Any = None,
) -> tuple[np.ndarray, np.ndarray]:
    """One draft step through `draft_fn(T)`; returns `(rows, pool bytes)`."""
    R = prog.R
    tp = NamedSharding(prog.mesh, P("tp"))
    meta = [_rep(hidden, R), _rep(block_tables, R), _rep(seq_lens, R), _rep(qsl, R)]
    if gather is not None:
        meta.append(_rep(gather, R))
    pool = _fp8(pool_u8.reshape(-1, *pool_u8.shape[-4:]), tp)
    out, new_pool = prog.draft_fn(T)(*meta, *_leaves(prog), pool)
    return np.asarray(out.astype(jnp.float32)), _u8(new_pool).reshape(pool_u8.shape)


# --- buckets ------------------------------------------------------------------


def test_parse_draft_buckets() -> None:
    assert draft.parse_draft_buckets(None, (4, 8)) == (4, 8)
    assert draft.parse_draft_buckets("16,4,8,8", ()) == (4, 8, 16)
    assert draft.parse_draft_buckets([12, 4], ()) == (4, 12)
    for off in ("0", "off", "none", 0, []):
        assert draft.parse_draft_buckets(off, (4,)) == ()
    for bad in ("6", "68", "0,4", [2]):
        with pytest.raises(ValueError, match="multiple of 4 in 4..64"):
            draft.parse_draft_buckets(bad, ())


def test_covering_buckets_keep_the_padded_step() -> None:
    assert draft.padded(28, PADDINGS) == 32
    assert draft.padded(112, PADDINGS) == 128
    assert draft.padded(300, PADDINGS) == 300
    # cap 28: 28 requests arrive as a 32-row loop step
    assert draft.covering_buckets(LOOP, 28, PADDINGS) == LOOP
    # cap 24 pads to 24: the list is what the running cap alone gives
    assert draft.covering_buckets(LOOP, 24, PADDINGS) == LOOP[:6]
    assert draft.covering_buckets(VERIFY, 96, PADDINGS) == VERIFY
    assert draft.covering_buckets(VERIFY, 12, PADDINGS) == (4, 8, 12, 16)
    # no paddings known: the cap itself
    assert draft.covering_buckets(LOOP, 28, ()) == LOOP[:7]
    # a cap between two buckets keeps the next one
    assert draft.covering_buckets((4, 8, 16), 10, ()) == (4, 8, 16)
    assert draft.covering_buckets((8, 16), 2, ()) == (8,)
    assert draft.covering_buckets((), 8, PADDINGS) == ()


def _bare(max_reqs: int, paddings: tuple[int, ...] = PADDINGS) -> Any:
    """A bound program without kernels: bucket selection and step admission."""
    prog = object.__new__(draft.Qwen35DraftProgram)
    prog.bound = True
    prog.layer_name = "mtp.layers.0.self_attn.attn"
    prog.tp_rank = 0
    prog._paddings = tuple(paddings)
    prog.loop_buckets = draft.covering_buckets(LOOP, max_reqs, prog._paddings)
    prog.verify_buckets = draft.covering_buckets(VERIFY, 4 * max_reqs, prog._paddings)
    prog.step_mix_tracker = draft.mk_stats.StepMixTracker(period_s=1e9)
    return prog


def _md(prog: Any, knr: int) -> dict[str, Any]:
    return {
        prog.layer_name: SimpleNamespace(
            seq_lens=torch.zeros(knr, dtype=torch.int32),
            query_start_loc=torch.zeros(knr + 1, dtype=torch.int32),
            block_tables=torch.zeros((knr, 8), dtype=torch.int32),
        )
    }


def test_use_for_step_takes_padded_steps() -> None:
    prog = _bare(16)
    md = _md(prog, 16)
    assert prog.use_for_step(md, 16, step0=False)  # 16 requests, one row each
    assert not prog.use_for_step(md, 24, step0=False)
    assert prog.use_for_step(md, 64, step0=True)  # 16 x 4 rows
    assert not prog.use_for_step(md, 65, step0=True)
    md3 = _md(prog, 3)
    assert prog.use_for_step(md3, 16, step0=True)  # 3 x 4 = 12 rows padded to 16
    assert not prog.use_for_step(md3, 24, step0=True)
    assert not prog.use_for_step({}, 4, step0=False)  # no metadata for the layer
    prog.bound = False
    assert not prog.use_for_step(md, 4, step0=False)
    assert prog.step_mix_tracker.counts == {
        "mk_loop_n16": 1,
        "xla_loop_n24": 1,
        "mk_s0_n64": 1,
        "xla_s0_n65": 1,
        "mk_s0_n16": 1,
        "xla_s0_n24": 1,
        "xla_loop_n4": 2,
    }


def test_note_step_logs_the_mix_each_period(caplog: pytest.LogCaptureFixture) -> None:
    prog = _bare(4)
    prog.step_mix_tracker = draft.mk_stats.StepMixTracker(period_s=0.0)
    prog.step_mix_tracker.t0 = time.monotonic() - 1.0
    with caplog.at_level("INFO"):
        assert prog.note_step("s0_n16", True) is True
    assert "MTP draft megakernel step mix" in caplog.text
    assert "'mk_s0_n16': 1" in caplog.text


# --- the draft step (interpret mode) --------------------------------------------


def test_step_operands_mask_inactive_sequences_and_ungathered_rows() -> None:
    prog = _program(verify=(8, 16), skip=frozenset({"experts"}))
    ex8 = prog.step_exec(4, 8, 8)
    ex16 = prog.step_exec(4, 16, 8)
    hid = jnp.ones((16, 256), jnp.bfloat16)
    bt = jnp.asarray([list(range(8, 16)), list(range(16, 24))], jnp.int32)
    lens = jnp.asarray([304, 704], jnp.int32)
    qsl = jnp.asarray([0, 4, 8], jnp.int32)
    g = jnp.asarray([3, 7], jnp.int32)
    pos8, h8, pt8, live8, m8 = draft.step_operands(ex8, hid[:8], bt, lens, qsl, g)
    pos16, h16, pt16, live16, m16 = draft.step_operands(ex16, hid, bt, lens, qsl, g)
    assert m8.tolist() == [0, 0, 0, 1, 0, 0, 0, 1]
    assert m16.tolist() == m8.tolist() + [0] * 8
    assert live8.tolist() == [1, 1] and live16.tolist() == [1, 1, 0, 0]
    assert pos8.tolist() == [300, 301, 302, 303, 700, 701, 702, 703]
    assert pos16.tolist()[8:] == [0, 1, 2, 3] * 2
    assert pt16[2:].tolist() == [[0] * 8] * 2
    # rows past the live sequences are zeroed
    assert float(jnp.abs(h16[8:].astype(jnp.float32)).sum()) == 0.0
    # a partial request (2 rows) is inactive: no K/V write, no experts
    qsl_p = jnp.asarray([0, 4, 6], jnp.int32)
    _, _, pt_p, live_p, m_p = draft.step_operands(ex8, hid[:8], bt, lens, qsl_p, g)
    assert live_p.tolist() == [1, 0] and pt_p[1].tolist() == [0] * 8
    assert m_p.tolist() == [0, 0, 0, 1, 0, 0, 0, 0]
    # the loop (no gather): every live row routes
    ex_l = prog.step_exec(1, 4, 8)
    _, _, _, live_l, m_l = draft.step_operands(
        ex_l,
        hid[:4],
        bt,
        lens,
        jnp.asarray([0, 1, 2], jnp.int32),
        None,
    )
    assert live_l.tolist() == [1, 1, 0, 0] and m_l.tolist() == [1, 1, 0, 0]


def test_verify_step_writes_only_active_kv_and_loop_step_reproduces_it() -> None:
    """A T=4 verify step writes only the active sequence's 4 new K/V columns,
    and a T=1 loop step over the updated pool reproduces the verify step's
    output for the last token (and rewrites the same K/V bytes)."""
    prog = _program(loop=(4,), verify=(8,), skip=frozenset({"experts"}))
    rng = np.random.default_rng(0)
    pool0 = _rand_pool_u8(rng, (NUM_PAGES, 2, 64, 4, PS))
    bt = np.asarray([list(range(8, 16)), list(range(16, 24))], np.int32)
    hid4 = jax.random.normal(jax.random.PRNGKey(1), (8, 256), jnp.float32).astype(
        jnp.bfloat16
    )
    # sequence 0 verifies 4 tokens at 300..303; sequence 1 is a partial request
    out4, pool4 = _run(
        prog,
        4,
        hid4,
        bt,
        np.asarray([304, 704], np.int32),
        np.asarray([0, 4, 6], np.int32),
        pool0,
        gather=np.asarray([3, 0], np.int32),
    )
    assert np.isfinite(out4[0]).all()
    changed = np.argwhere(np.any(pool4 != pool0, axis=(1, 2, 3)))
    assert changed.tolist() == [[10, 44], [10, 45], [10, 46], [10, 47]]

    out1, pool1 = _run(
        prog,
        1,
        jnp.zeros((4, 256), jnp.bfloat16).at[0].set(hid4[3]),
        bt[:1],
        np.asarray([304], np.int32),
        np.asarray([0, 1], np.int32),
        pool4,
    )
    assert np.array_equal(pool1, pool4)
    np.testing.assert_allclose(out1[0], out4[0], rtol=2e-2, atol=2e-2)


def test_padded_rows_leave_live_rows_and_pool_unchanged() -> None:
    """Draft step 0 of 2 requests in an 8-row bucket and in a 16-row bucket
    whose rows past the requests' 8 hold other values: same gathered rows, same
    pool."""
    prog = _program(verify=(8, 16), skip=frozenset({"experts"}))
    rng = np.random.default_rng(0)
    pool0 = _rand_pool_u8(rng, (NUM_PAGES, 2, 64, 4, PS))
    bt = np.asarray([list(range(8, 16)), list(range(16, 24))], np.int32)
    lens = np.asarray([304, 704], np.int32)
    qsl = np.asarray([0, 4, 8], np.int32)
    gather = np.asarray([3, 7], np.int32)
    hid = jax.random.normal(jax.random.PRNGKey(1), (16, 256), jnp.float32).astype(
        jnp.bfloat16
    )
    o8, p8 = _run(prog, 4, hid[:8], bt, lens, qsl, pool0, gather=gather)
    o16, p16 = _run(prog, 4, hid, bt, lens, qsl, pool0, gather=gather)
    assert sorted(k[:2] for k in prog._steps) == [(4, 8), (4, 16)]
    assert np.isfinite(o8).all()
    np.testing.assert_allclose(o16, o8, rtol=2e-2, atol=2e-2)
    assert np.array_equal(p16, p8)
    assert not np.array_equal(p8, pool0)


def test_step0_row_mask_routes_only_the_gathered_rows() -> None:
    """With the experts on: the program's step takes the `moe_rows` mask, the
    gathered rows match the same step with every live row routed, and an
    ungathered row loses its experts (the mask leaves them out of the routed
    list)."""
    prog = _program(verify=(8,), signed_weights=True)
    assert prog.base_exec.debug.modules["moe"].row_mask is True
    ex, step = prog.step(4, 8, 8)
    rng = np.random.default_rng(1)
    pool0 = _rand_pool_u8(rng, (NUM_PAGES, 2, 64, 4, PS))
    bt = jnp.asarray([list(range(8, 16)), list(range(16, 24))], jnp.int32)
    lens = jnp.asarray([304, 704], jnp.int32)
    qsl = jnp.asarray([0, 4, 8], jnp.int32)
    hid = jax.random.normal(jax.random.PRNGKey(2), (8, 256), jnp.float32).astype(
        jnp.bfloat16
    )
    gather = jnp.asarray([3, 7], jnp.int32)
    pos, h, pt, live, masked = draft.step_operands(ex, hid, bt, lens, qsl, gather)
    weights = prog._weights(_leaves(prog))

    def rows(moe_rows: jax.Array) -> np.ndarray:
        out, _ = step(pos, h, weights, (_fp8(pool0),), pt, moe_rows, live)
        return np.asarray(out.astype(jnp.float32))

    o_mask = rows(masked)
    o_all = rows(jnp.ones((8,), jnp.int32))
    assert np.isfinite(o_mask[:8]).all() and np.isfinite(o_all[:8]).all()
    np.testing.assert_allclose(o_mask[[3, 7]], o_all[[3, 7]], rtol=2e-2, atol=2e-2)
    assert float(np.abs(o_mask[0] - o_all[0]).max()) > 1e-2


def test_forward_layer_runs_the_jax_step_through_the_torch_op(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`forward_layer` (the drafter's entry point) runs the same step as the
    JAX function through the bridge's torch op: same gathered rows, and the
    draft layer's pool tensor updated in place to the same bytes."""
    import sys
    import types

    try:
        import vllm.forward_context as fc
    except ImportError:  # CPU without vLLM: a stand-in forward-context module
        fc = types.ModuleType("vllm.forward_context")
        monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
        monkeypatch.setitem(sys.modules, "vllm.forward_context", fc)

    prog = _program(loop=(4,), verify=(8,), skip=frozenset({"experts"}))
    rng = np.random.default_rng(3)
    pool0 = _rand_pool_u8(rng, (NUM_PAGES, 2, 64, 4, PS))
    bt = np.asarray([list(range(8, 16)), list(range(16, 24))], np.int32)
    lens = np.asarray([304, 704], np.int32)
    qsl = np.asarray([0, 4, 8], np.int32)
    gather = np.asarray([3, 7], np.int32)
    hid = jax.random.normal(jax.random.PRNGKey(4), (8, 256), jnp.float32).astype(
        jnp.bfloat16
    )
    want_rows, want_pool = _run(prog, 4, hid, bt, lens, qsl, pool0, gather=gather)

    pool_t = prog.layer.self_attn.attn.kv_cache[0]
    ptr = pool_t.data_ptr()
    pool_t.view(torch.uint8).copy_(torch.from_numpy(pool0))
    md = SimpleNamespace(
        block_tables=torch.from_numpy(bt.reshape(-1)),
        seq_lens=torch.from_numpy(lens),
        query_start_loc=torch.from_numpy(qsl),
    )
    ctx = SimpleNamespace(attn_metadata={prog.layer_name: md})
    monkeypatch.setattr(fc, "get_forward_context", lambda: ctx, raising=False)
    hid_t = torch.from_numpy(np.asarray(hid).view(np.uint16).copy()).view(
        torch.bfloat16
    )
    with torch.no_grad():
        out = prog.forward_layer(prog.layer, hid_t, torch.from_numpy(gather).long())
    assert tuple(out.shape) == (2, 256)
    np.testing.assert_array_equal(out.float().numpy(), want_rows)
    assert pool_t.data_ptr() == ptr
    assert np.array_equal(pool_t.view(torch.uint8).numpy(), want_pool)


# --- decode context parallelism (8 CPU devices) -------------------------------------

R8 = 8


def _group_pools(g: Any, seed: int) -> np.ndarray:
    """DCP1 pools `(R, num_pages, 2, 64, 4, 128)`: identical over a KV-head group."""
    grp = R8 // g.NKV
    rng = np.random.default_rng(seed)
    u8 = _rand_pool_u8(rng, (g.NKV, NUM_PAGES, 2, 64, 4, PS))
    return np.stack([u8[r // grp] for r in range(R8)])


def _shard_pools(pools1: np.ndarray, pt: np.ndarray, d: int) -> np.ndarray:
    """DCP pools: rank r's column j of sequence s holds global page `d * j + r % d`."""
    out = pools1.copy()
    width = pt.shape[1]
    for r in range(R8):
        for s in range(pt.shape[0]):
            for j in range(width):
                gp = d * j + (r % d)
                if gp < width:
                    out[r, int(pt[s, j])] = pools1[r, int(pt[s, gp])]
    return out


def _expected_dcp_pools(
    pools_in: np.ndarray,
    pools1_out: np.ndarray,
    pt: np.ndarray,
    starts: list[int],
    T: int,
    d: int,
) -> np.ndarray:
    """The DCP pools after the step: each new token's lanes copied from the
    DCP1 result into its owner rank's local page, nothing else changed."""
    out = pools_in.copy()
    for s, s0 in enumerate(starts):
        for t in range(T):
            p = s0 + t
            gp, ln = p // PS, p % PS
            for r in range(R8):
                if r % d != gp % d:
                    continue
                src = pools1_out[r, int(pt[s, gp])]
                out[r, int(pt[s, gp // d]), :, :, :, ln] = src[:, :, :, ln]
    return out


def _assert_in_band(got: np.ndarray, ref: np.ndarray) -> None:
    """DCP merges the attention partials over other KV subsets than DCP1, so
    bf16 roundings differ in order (a few 1e-4 of the elements outside
    atol = rtol = 2e-2)."""
    diff = np.abs(got - ref)
    over = diff > 2e-2 + 2e-2 * np.abs(ref)
    assert float(over.mean()) <= 1e-3, (float(over.mean()), float(diff.max()))
    assert float(diff.max()) <= 0.25, float(diff.max())


def test_dcp4_draft_step_matches_dcp1_and_writes_only_owner_pages() -> None:
    p1 = _program(R=R8, verify=(8,), skip=frozenset({"experts"}))
    p4 = _program(R=R8, verify=(8,), dcp=4, skip=frozenset({"experts"}))
    assert p1.step_exec(4, 8, 8).cache.dcp == 1
    assert p4.step_exec(4, 8, 8).cache.dcp == 4
    g = p1.geom
    # sequence 0: four tokens on page 2 (owner 2 at dcp 4); sequence 1: two on
    # page 4 (owner 0) and two on page 5 (owner 1)
    starts = [300, 638]
    pt = np.zeros((2, 8), np.int32)
    for s in range(2):
        pt[s] = 8 + 8 * s + np.arange(8)
    lens = np.asarray([s0 + 4 for s0 in starts], np.int32)
    qsl = np.asarray([0, 4, 8], np.int32)
    gather = np.asarray([3, 7], np.int32)
    hid = jax.random.normal(jax.random.PRNGKey(1), (8, 256), jnp.float32).astype(
        jnp.bfloat16
    )
    pools1 = _group_pools(g, 3)
    pools4 = _shard_pools(pools1, pt, 4)
    o1, q1 = _run(p1, 4, hid, pt, lens, qsl, pools1, gather=gather)
    o4, q4 = _run(p4, 4, hid, pt, lens, qsl, pools4, gather=gather)
    o1 = o1.reshape(R8, -1, 256)
    o4 = o4.reshape(R8, -1, 256)
    assert np.isfinite(o1).all() and np.isfinite(o4).all()
    for r in range(R8):
        _assert_in_band(o4[r], o1[r])
    assert np.array_equal(q4, _expected_dcp_pools(pools4, q1, pt, starts, 4, 4))
    assert not np.array_equal(q4, pools4)
