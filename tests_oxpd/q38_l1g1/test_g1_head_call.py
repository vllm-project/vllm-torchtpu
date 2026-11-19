# SPDX-License-Identifier: Apache-2.0
"""q38 L1-G1 GDN-COMMIT1: the head group call end to end (CPU interpret).

The real `build_segment_call(head=True)` closure with the real checkpoint scatter / commit kernels and the real pool
bytes; only the megakernel step is stubbed (it returns fixed random checkpoints of the served per-rank shapes, conv
[NL, NS, 8, 1280] f32 and rec [NL, NS*T*6, 128, 128] bf16, so the run takes seconds; the kernel's own reads under G1 are
covered by test_g1_readers_poison.py). Flag on vs off, same inputs:
  - the residual is bit-identical; the flag-on call writes no pool byte (its scatter is skipped);
  - the staging outputs are the kernel's checkpoints and the call's ids: per rank, the full flush (`full_scatter`)
    reproduces the flag-off pool byte for byte, and the runner's commit (`commit_from_offsets` on a post-verify read-offset
    table) equals today's scatter with only the accepted ids (and writes fewer blocks than the flush);
  - wb == gdn_blocks' write ids and st == the state ids on every rank's shard;
  - a body segment (head=False) keeps the flag-off program (scatter, 2 outputs, same bytes) with the flag on;
  - HEAD_CHUNK_ROWS (the head kind's chunk rows the precompile plans for) == the driver's KERNEL_ROWS.
Run: JAX_PLATFORMS=cpu PYTHONPATH=src:. pytest tests/q38_l1g1/test_g1_head_call.py
"""

from __future__ import annotations

import os
import sys
import types

try:
    import vllm.envs  # noqa: F401
except ImportError:
    _vllm = types.ModuleType("vllm")
    _envs = types.ModuleType("vllm.envs")
    _envs.VLLM_KV_CACHE_LAYOUT = "NHD"
    _vllm.envs = _envs
    sys.modules.setdefault("vllm", _vllm)
    sys.modules.setdefault("vllm.envs", _envs)

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from vllm_torchtpu.megakernel_runtime import qwen38_gdn_commit as gc  # noqa: E402
from vllm_torchtpu.megakernel_runtime import qwen38_gdn_state as gs  # noqa: E402
from vllm_torchtpu.megakernel_runtime import qwen38_segment_call as sc  # noqa: E402
from vllm_torchtpu.megakernel_runtime.qwen38_scatter import scatter_gdn_ckpts_fast  # noqa: E402

T, R, LANES, H = 4, 2, 256, 8


def _u8(x) -> np.ndarray:
    return np.asarray(jax.lax.bitcast_convert_type(jnp.asarray(x), jnp.uint8))


def _call(monkeypatch, *, on: bool, L: int, head: bool, seed: int = 7) -> dict:
    from vllm_torchtpu.megakernel_runtime.exported_bridge import make_typed_step_wrapper

    if on:
        monkeypatch.setenv(gc.ENV, "1")
    else:
        monkeypatch.delenv(gc.ENV, raising=False)
    NS, B = 2, 2 * T
    NL = L if head else L - 1
    nmb = 1 + NL * NS * T + 2
    rng = np.random.default_rng(seed)
    ck = 1 + rng.permutation(NL * NS * T).astype(np.int32).reshape(NL, NS, T)
    off = rng.integers(0, T, nmb).astype(np.int32)
    raws = [rng.integers(0, 256, (nmb * 4, 128, 1, 4, LANES), dtype=np.uint8) for _ in range(R)]
    conv_c = rng.standard_normal((R * NL, NS, 8, 1280)).astype(np.float32)
    rec_c = jnp.asarray(rng.standard_normal((R * NL, NS * T * 6, 128, 128)), jnp.float32).astype(jnp.bfloat16)

    def fn(pos, x, *rest):  # the kernel step: (residual, conv_ckpt, rec_ckpt), depending on what it reads
        xs, w = rest[:3], rest[3:]
        _, rb, hi = xs
        bump = jnp.sum(rb).astype(jnp.float32) + jnp.sum(hi).astype(jnp.float32) + jnp.sum(w[0])
        return x + bump, jnp.asarray(conv_c) + bump, rec_c

    fn.extra_names = ("gdn_pool", "gdn_rb", "gdn_hi")
    monkeypatch.setattr(sc.qwen38, "segment_spec", lambda *a, **k: None)
    monkeypatch.setattr(sc, "make_flat_segment_step", lambda mesh, spec, ex, extra_outs=(): (fn, None, None, 1))
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    tp, rep = NamedSharding(mesh, P("tp")), NamedSharding(mesh, P())
    layout = gs.Qwen38GdnPoolLayout(pool_shape=tuple(raws[0].shape), pool_dtype="float8_e4m3fn")
    call, in_specs, out_specs, donate = sc.build_segment_call(mesh, None, L, B, layout, interpret=True, head=head)
    pool = jax.device_put(jnp.concatenate(
        [jax.lax.bitcast_convert_type(jnp.asarray(r), jnp.float8_e4m3fn) for r in raws], axis=0), tp)
    hidden = jax.device_put(jnp.asarray(rng.standard_normal((R * B, H)), jnp.float32), tp)
    attn = jax.device_put(jnp.asarray(rng.standard_normal((R * B, H)), jnp.float32), tp)
    w0 = jax.device_put(jnp.asarray(rng.standard_normal((R * 4, H)), jnp.float32), tp)
    meta = []
    for ln in range(NL):
        meta += [jnp.asarray(ck[ln, :, 0]), jnp.asarray(off), jnp.asarray(ck[ln]),
                 jnp.arange(NS + 1, dtype=jnp.int32) * T, jnp.full((NS,), 100, jnp.int32)]
    meta = [jax.device_put(m, rep) for m in meta]
    lead = (hidden,) if head else (hidden, attn)
    typed = make_typed_step_wrapper("g1_seg_call_test", call, len(in_specs), len(out_specs))
    outs = jax.jit(typed)(*lead, w0, pool, *meta)
    _, wb, _ = (np.asarray(x) for x in sc.gdn_blocks(meta, NS, True))
    return dict(outs=[_u8(o) if i == 1 else np.asarray(o) for i, o in enumerate(outs)], out_specs=out_specs,
                raws=raws, layout=layout, ck=ck, wb=wb, NL=NL, NS=NS, nmb=nmb)


def test_head_call_stages_and_the_flush_reproduces_today(monkeypatch):
    off = _call(monkeypatch, on=False, L=3, head=True)
    on = _call(monkeypatch, on=True, L=3, head=True)
    assert len(off["outs"]) == 2 and len(on["outs"]) == 6 and len(on["out_specs"]) == 6
    res0, pool0 = off["outs"]
    res1, pool1, conv, rec, wb, st = on["outs"]
    np.testing.assert_array_equal(np.ascontiguousarray(res1).view(np.uint8), np.ascontiguousarray(res0).view(np.uint8))
    NL, NS, nmb, lay = on["NL"], on["NS"], on["nmb"], on["layout"]
    pin = np.concatenate(on["raws"]).reshape(-1)
    assert np.array_equal(pool1.reshape(-1), pin), "flag on: the head call must not write the pool"
    assert not np.array_equal(pool0.reshape(-1), pin), "control: the flag-off head call writes checkpoints"
    rng = np.random.default_rng(4)
    acc = rng.integers(0, T, (NL, NS)).astype(np.int32)
    offs = np.zeros(nmb, np.int32)
    offs[on["ck"][:, :, 0]] = acc
    p0 = pool0.reshape(R, -1)
    for r in range(R):
        sl = slice(r * NL, (r + 1) * NL)
        np.testing.assert_array_equal(wb[sl], on["wb"])
        np.testing.assert_array_equal(st[sl], on["ck"][:, :, 0])
        pr = jax.lax.bitcast_convert_type(jnp.asarray(on["raws"][r]), jnp.float8_e4m3fn)
        cr, rr, wr, sr = jnp.asarray(conv[sl]), jnp.asarray(rec[sl]), jnp.asarray(wb[sl]), jnp.asarray(st[sl])
        full = gc.full_scatter(pr, cr, rr, wr, layout=lay, interpret=True)
        assert np.array_equal(_u8(full).reshape(-1), p0[r]), f"rank {r}: full flush != the flag-off pool"
        com = gc.commit_from_offsets(pr, cr, rr, wr, sr, jnp.asarray(offs), layout=lay, interpret=True)
        ref = scatter_gdn_ckpts_fast(pr, cr, rr, gc.accepted_write_blocks(wr, jnp.asarray(acc)), layout=lay,
                                     interpret=True)
        assert np.array_equal(_u8(com), _u8(ref)), f"rank {r}: commit != the accepted-id scatter"
        assert not np.array_equal(_u8(com), _u8(full)), "control: the commit writes fewer blocks than the full flush"
    print(f"Q38G1H head call: residual identical, flag-on pool untouched, full flush == today, commit == accepted-id "
          f"scatter on {R} ranks ({NL} layers x {NS} requests)", flush=True)


def test_body_segment_keeps_the_scatter(monkeypatch):
    on = _call(monkeypatch, on=True, L=4, head=False)
    off = _call(monkeypatch, on=False, L=4, head=False)
    assert len(on["outs"]) == len(off["outs"]) == 2
    for a, b in zip(on["outs"], off["outs"]):
        np.testing.assert_array_equal(np.ascontiguousarray(a).view(np.uint8), np.ascontiguousarray(b).view(np.uint8))
    assert not np.array_equal(off["outs"][1].reshape(-1), np.concatenate(off["raws"]).reshape(-1))


def test_head_chunk_rows_match_the_driver():
    from vllm_torchtpu.megakernel_runtime.qwen38_driver import KERNEL_ROWS

    assert gc.HEAD_CHUNK_ROWS == KERNEL_ROWS
