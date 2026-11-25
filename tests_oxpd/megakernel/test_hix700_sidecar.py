# SPDX-License-Identifier: Apache-2.0
"""q38 HIX700 GDN-SIDECAR (VLLM_TPU_MK_GDN_SIDECAR=1): the whole step keeps its GDN verify checkpoints in a slot-indexed sidecar and the next
step reads the accepted one from it (megakernel/models/qwen38_gdn_sidecar.py). CPU interpret, real per-rank GDN shapes (TINY at R = 2: 2
k-heads, 6 v-heads, 128 x 128 states, QKV 1280), the real pool byte layout, the 16-row bucket (NS = 4, T = 4); the pool-GDN segment kernel
(qwen38.segment_spec, pool_gdn=True) with the block class swapped, so the block under test is exactly the whole step's.

Deciding checks:
  S1 side_on = 0 (admission / after a flush): the sidecar block gathers from the pool exactly as the pool block does: residual bit-identical,
     and the sidecar it leaves (its interpret outputs) equals the flag-off kernel's rec_ckpt / conv_ckpt bit for bit (so the next step can
     read it); the sidecar input bytes are random, so a read of them would move the residual.
  S2 the 2-step chain: step 1 = the flag-off kernel (checkpoints C1, scattered into the pool at write ids wb1 as today); step 2 reference =
     the flag-off kernel reading the accepted checkpoints from that pool (rb2 = wb1[l, s, off[l, s]], random off per layer and request);
     step 2 sidecar = the sidecar kernel on a STALE pool (no scatter: the accepted blocks hold random bytes) with sidecar = C1, side_on = 1,
     side_off = off: residual and the new checkpoints bit-identical to the reference.
  S3 positive control: the same sidecar step with side_on = 0 reads the stale pool, and its residual differs (the sidecar read is what makes
     S2 pass).
  S4 the conv window of every offset 0..3 (off forced to each value in turn), 2-step chain as S2.
Plus plumbing: the key's default, class selection only for the 16-row whole-step spec, refusals, operands / outputs.
Run: JAX_PLATFORMS=cpu PYTHONPATH=src pytest tests/megakernel/test_hix700_sidecar.py
"""

from __future__ import annotations

import collections
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
import pytest  # noqa: E402
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from vllm_torchtpu.megakernel import kernel  # noqa: E402
from vllm_torchtpu.megakernel.blocks.hc import RESIDUAL_OUT  # noqa: E402
from vllm_torchtpu.megakernel.exec_spec import ExecSpec, Tiles  # noqa: E402
from vllm_torchtpu.megakernel.models import qwen38  # noqa: E402
from vllm_torchtpu.megakernel.models import qwen38_gdn_ckpt_ink as ink  # noqa: E402
from vllm_torchtpu.megakernel.models import qwen38_gdn_sidecar as sc  # noqa: E402
from vllm_torchtpu.megakernel.models.qwen38_gdn_pool import Qwen38PoolGDN  # noqa: E402
from vllm_torchtpu.megakernel.models.qwen38_inputs import pack_segment, random_segment_inputs  # noqa: E402
from vllm_torchtpu.megakernel.models.qwen38_pack import shard_step_inputs  # noqa: E402
from vllm_torchtpu.megakernel.live_rows import LIVE_ROWS  # noqa: E402

T = 4
R = 2
B = 16
NS = B // T
G = qwen38.TINY
TILES = Tiles(bank=1 << 20, nbd=10, nbe=8, ebank=512 << 10, vmem_mib=64)
LANES = 256
KB = 128
VH, KD, VD, QKV = 6, 128, 128, 1280  # TINY per rank at R = 2 (the CKPT_INK test's shapes)


def _u8(x) -> np.ndarray:
    return np.asarray(jax.lax.bitcast_convert_type(jnp.asarray(x), jnp.uint8))


def _bits(a) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(a)).view(np.uint8)


def _encode(raw: np.ndarray, m: int, ssm: np.ndarray, conv: np.ndarray) -> None:
    from tests.megakernel.test_qwen38_gdn_pool import _encode as enc

    enc(raw, m, ssm, conv)


def _layout(raw: np.ndarray):
    from vllm_torchtpu.megakernel_runtime.qwen38_gdn_state import Qwen38GdnPoolLayout

    return Qwen38GdnPoolLayout(pool_shape=tuple(raw.shape), pool_dtype="float8_e4m3fn")


def _pools(seed: int):
    """Per-rank pools, every byte random; read blocks 1..NL*NS hold encoded states; write ids after them (all live)."""
    L = qwen38.SEGMENT_LAYERS
    NL = L - 1
    rng = np.random.default_rng(seed + 1)
    nmb = 1 + NL * NS + NL * NS * T
    rb = 1 + rng.permutation(NL * NS).astype(np.int32).reshape(NL, NS)
    hi = np.ones((NL, NS), np.int32)
    wb = (1 + NL * NS + rng.permutation(NL * NS * T)).astype(np.int32).reshape(NL, NS, T)
    raws = []
    for _ in range(R):
        raw = rng.integers(0, 256, (nmb * 4, 128, 1, 4, LANES), dtype=np.uint8)
        for l in range(NL):
            for s in range(NS):
                _encode(raw, int(rb[l, s]), rng.standard_normal((VH, KD, VD)).astype(np.float32) * 0.05,
                        rng.standard_normal((3, QKV)).astype(np.float32) * 0.05)
        raws.append(raw.reshape(nmb * 512 // KB, KB, 1, 4, LANES))
    return raws, rb, hi, wb, NL


def _spec(monkeypatch, side: bool):
    monkeypatch.delenv(ink.ENV, raising=False)
    if side:
        monkeypatch.setattr(ink, "pool_gdn_cls", lambda: sc.Qwen38SidecarGDN)
    else:
        monkeypatch.setattr(ink, "pool_gdn_cls", lambda: Qwen38PoolGDN)
    spec = qwen38.segment_spec(G, R, L=qwen38.SEGMENT_LAYERS, pool_gdn=True)
    blks = [getattr(b, "inner", b) for b in spec.all_blocks() if isinstance(getattr(b, "inner", b), sc.Qwen38SidecarGDN)]
    assert len(blks) == (1 if side else 0), blks
    for b in blks:
        NL = qwen38.SEGMENT_LAYERS - 1
        b.interpret_shapes = sc.side_shapes(NL, B, NS, T, VH, KD, VD, QKV)
    return spec


def _step(monkeypatch, *, side: bool, raws, rb, hi, x_seed: int, side_in=None, side_on: int = 0, side_off=None, live=None,
          trace_only: bool = False):
    """One 16-row pool-GDN segment step (R = 2) on the given per-rank pools. Flag-off: returns residual and rec_ckpt / conv_ckpt
    (per rank). Sidecar: returns residual and the sidecar the kernel leaves (per rank)."""
    L = qwen38.SEGMENT_LAYERS
    NL = L - 1
    ex = ExecSpec(R=R, B=B, T=T, ctx=128, interpret=True, tiles=TILES, live_rows=int(live is not None))  # X-K-LIVE mode 1 with `live`
    spec = _spec(monkeypatch, side)
    inp = random_segment_inputs(jax.random.PRNGKey(x_seed), G, ex, L, gain=0.25)
    w_mk, _, attn = pack_segment(spec, G, ex, inp)
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    pos, hid, w_mk, st = shard_step_inputs(mesh, ex, inp.pos, inp.x, w_mk, {})
    tp, rep = NamedSharding(mesh, P("tp")), NamedSharding(mesh, P())
    pool = jnp.concatenate([jax.lax.bitcast_convert_type(jnp.asarray(r), jnp.float8_e4m3fn) for r in raws])
    xs = {
        "qsa_attn": jax.device_put(kernel.fold(attn), tp),
        "gdn_pool": jax.device_put(pool, tp),
        "gdn_rb": jax.device_put(jnp.asarray(rb), rep),
        "gdn_hi": jax.device_put(jnp.asarray(hi), rep),
    }
    if side:
        rec, conv = side_in
        xs[sc.SIDE_REC] = jax.device_put(jnp.concatenate([jnp.asarray(r) for r in rec]), tp)
        xs[sc.SIDE_CONV] = jax.device_put(jnp.concatenate([jnp.asarray(c) for c in conv]), tp)
        xs[sc.SIDE_ON] = jax.device_put(jnp.asarray([side_on], jnp.int32), rep)
        xs[sc.SIDE_OFF] = jax.device_put(jnp.asarray(side_off, jnp.int32), rep)
    if live is not None:  # the planner's mask input: 1 per row of a live request (live_rows_from_qsl)
        xs[LIVE_ROWS] = jax.device_put(jnp.asarray(np.repeat(np.asarray(live, np.int32), T)), rep)
    names = [x.name for x in kernel.extra_inputs(spec, ex)]
    assert (sc.SIDE_REC in names) == side, names
    assert (LIVE_ROWS in names) == (live is not None), names
    step, info = kernel.make_step(mesh, spec, ex)
    if trace_only:
        return jax.make_jaxpr(step)(pos, hid, w_mk, st, *(xs[k] for k in names))
    o_names = info["plan"].o_names
    assert ("rec_ckpt" in o_names) == (not side) and (sc.SIDE_REC_OUT in o_names) == side, o_names
    _, outs = step(pos, hid, w_mk, st, *(xs[k] for k in names))
    res = np.asarray(outs[RESIDUAL_OUT]).reshape(R, ex.M, -1)[:, :B]
    if side:
        rec = np.asarray(outs[sc.SIDE_REC_OUT]).reshape(R, NL, B * VH, KD, VD)
        conv = np.asarray(outs[sc.SIDE_CONV_OUT]).reshape(R, NL, NS, 8, QKV)
    else:
        rec = np.asarray(outs["rec_ckpt"]).reshape(R, NL, B * VH, KD, VD)
        conv = np.asarray(outs["conv_ckpt"]).reshape(R, NL, NS, 8, QKV)
    return dict(res=res, rec=rec, conv=conv)


def _scatter(raws, rec, conv, wb):
    from vllm_torchtpu.megakernel_runtime.qwen38_scatter import scatter_gdn_ckpts_fast

    out = []
    for r in range(R):
        p = scatter_gdn_ckpts_fast(jax.lax.bitcast_convert_type(jnp.asarray(raws[r]), jnp.float8_e4m3fn),
                                   jnp.asarray(conv[r]), jnp.asarray(rec[r]), jnp.asarray(wb), layout=_layout(raws[r]), interpret=True)
        out.append(_u8(p).reshape(raws[r].shape))
    return out


def _junk_side(NL: int, seed: int):
    rng = np.random.default_rng(seed)
    rec = [(rng.standard_normal((NL, B * VH, KD, VD)) * 0.05).astype(jnp.bfloat16) for _ in range(R)]
    conv = [(rng.standard_normal((NL, NS, 8, QKV)) * 0.05).astype(np.float32) for _ in range(R)]
    return rec, conv


def _ndiff(a, b) -> int:
    return int((_bits(a) != _bits(b)).sum())


def test_s1_side_off_equals_pool_block(monkeypatch):
    raws, rb, hi, wb, NL = _pools(11)
    ref = _step(monkeypatch, side=False, raws=raws, rb=rb, hi=hi, x_seed=3)
    got = _step(monkeypatch, side=True, raws=raws, rb=rb, hi=hi, x_seed=3, side_in=_junk_side(NL, 5), side_on=0,
                side_off=np.zeros((NL, NS), np.int32))
    nres, nrec, nconv = _ndiff(got["res"], ref["res"]), _ndiff(got["rec"], ref["rec"]), _ndiff(got["conv"], ref["conv"])
    print(f"Q38SIDE S1: residual bytes differing {nres}; sidecar rec vs rec_ckpt {nrec}; sidecar conv vs conv_ckpt {nconv}", flush=True)
    assert nres == 0 and nrec == 0 and nconv == 0


def _chain(monkeypatch, off: np.ndarray, seed: int = 21) -> dict:
    raws, rb, hi, wb, NL = _pools(seed)
    s1 = _step(monkeypatch, side=False, raws=raws, rb=rb, hi=hi, x_seed=7)
    pool1 = _scatter(raws, s1["rec"], s1["conv"], wb)  # today: every checkpoint into its write block
    rb2 = np.take_along_axis(wb, off[:, :, None], axis=2)[:, :, 0].astype(np.int32)
    ref = _step(monkeypatch, side=False, raws=pool1, rb=rb2, hi=hi, x_seed=8)
    side_in = ([s1["rec"][r] for r in range(R)], [s1["conv"][r] for r in range(R)])
    got = _step(monkeypatch, side=True, raws=raws, rb=rb2, hi=hi, x_seed=8, side_in=side_in, side_on=1, side_off=off)  # stale pool
    return dict(ref=ref, got=got, raws=raws, rb2=rb2, hi=hi, side_in=side_in, off=off)


def test_s2_chain_side_read_equals_pool_read(monkeypatch):
    rng = np.random.default_rng(31)
    NL = qwen38.SEGMENT_LAYERS - 1
    off = rng.integers(0, T, (NL, NS)).astype(np.int32)
    c = _chain(monkeypatch, off)
    nres, nrec, nconv = _ndiff(c["got"]["res"], c["ref"]["res"]), _ndiff(c["got"]["rec"], c["ref"]["rec"]), _ndiff(c["got"]["conv"], c["ref"]["conv"])
    print(f"Q38SIDE S2 off {off.tolist()}: residual bytes differing {nres}; new checkpoints rec {nrec} conv {nconv}", flush=True)
    assert nres == 0 and nrec == 0 and nconv == 0
    # S3 positive control: side_on = 0 on the stale pool must differ
    ctl = _step(monkeypatch, side=True, raws=c["raws"], rb=c["rb2"], hi=c["hi"], x_seed=8, side_in=c["side_in"], side_on=0, side_off=off)
    nctl = _ndiff(ctl["res"], c["ref"]["res"])
    print(f"Q38SIDE S3 control (side_on = 0 on the stale pool): residual bytes differing {nctl} (want > 0)", flush=True)
    assert nctl > 0


@pytest.mark.parametrize("o", [0, 1, 2, 3])
def test_s4_every_offset(monkeypatch, o):
    NL = qwen38.SEGMENT_LAYERS - 1
    c = _chain(monkeypatch, np.full((NL, NS), o, np.int32), seed=40 + o)
    nres, nconv = _ndiff(c["got"]["res"], c["ref"]["res"]), _ndiff(c["got"]["conv"], c["ref"]["conv"])
    print(f"Q38SIDE S4 off={o}: residual bytes differing {nres}; conv {nconv}", flush=True)
    assert nres == 0 and nconv == 0


def _grow(pools, need_blocks: int, seed: int) -> list:
    """Pad each per-rank pool with random bytes up to `need_blocks` blocks (the same bytes in both worlds: same seed)."""
    out = []
    for p in pools:
        have = p.shape[0] // 4
        if have < need_blocks:
            pad = np.random.default_rng(seed).integers(0, 256, ((need_blocks - have) * 4,) + p.shape[1:], dtype=np.uint8)
            p = np.concatenate([p, pad])
        out.append(np.array(p))  # writable (the scatter's outputs are read-only views)
    return out


@pytest.mark.parametrize("live", [False, True])
def test_s5_six_step_chain_flush_and_membership(monkeypatch, live):
    """Six 16-row steps, the sidecar world against the reference world (today: every step scatters its checkpoints into the pool and
    every read is the pool's), random accepted offsets per step, every step's residual / rec / conv compared bit for bit:
      1 side_on 0 (admission: the pool gather)        2 side_on 1
      3 FLUSH (the sidecar into the pool at step 2's write ids: the runtime's flag-off scatter bytes), then side_on 0
      4 side_on 1
      5 row 1's request leaves and a new request takes the row; the new state lands (the connector, both worlds) in a block the old
        request wrote at step 4; FLUSH with row 1's write ids nulled (the runtime's keep_rows); side_on 0
      6 side_on 1
    Positive control: the step-5 flush WITHOUT the null overwrites the new request's state block, and step 5 then differs.
    live=True: X-K-LIVE mode 1 (the S2 / S3a form) at 3 live: the last request is a pad in every step (no state, null read / write blocks,
    as gdn_blocks gives the served call), the kernel's live mask marks it, and every row (pad rows included) must still match."""
    raws, rb, hi, wb, NL = _pools(51)
    lv = [1] * (NS - 1) + [0] if live else None
    if live:
        hi, rb, wb = hi.copy(), rb.copy(), wb.copy()
        hi[:, NS - 1], rb[:, NS - 1], wb[:, NS - 1, :] = 0, 0, 0
    rng = np.random.default_rng(52)
    pool_ref, pool_sc = _grow(raws, 0, 0), _grow(raws, 0, 0)
    side = _junk_side(NL, 53)
    rb_t, wb_t, wb_prev, off_prev = rb, wb, None, np.zeros((NL, NS), np.int32)
    bad, nctl = [], None
    for t in range(6):
        unnulled = None
        if t in (2, 4):
            wb_fl = wb_prev.copy()
            if t == 4:  # membership change in row 1
                nb = wb_prev[:, 1, 0].copy()  # a block the departed request wrote at step 4 (t = 3), reused for the new request
                st_rng = np.random.default_rng(54)
                for r in range(R):
                    for l in range(NL):
                        ssm = st_rng.standard_normal((VH, KD, VD)).astype(np.float32) * 0.05
                        cv = st_rng.standard_normal((3, QKV)).astype(np.float32) * 0.05
                        _encode(pool_ref[r], int(nb[l]), ssm, cv)
                        _encode(pool_sc[r], int(nb[l]), ssm, cv)
                rb_t = rb_t.copy()
                rb_t[:, 1] = nb
                wb_fl[:, 1, :] = 0  # the departed row's write ids nulled (block 0: never written)
                unnulled = _grow(_scatter(pool_sc, side[0], side[1], wb_prev), 0, 0)
            pool_sc = _grow(_scatter(pool_sc, side[0], side[1], wb_fl), 0, 0)
        on = 0 if t in (0, 2, 4) else 1
        ref = _step(monkeypatch, side=False, raws=pool_ref, rb=rb_t, hi=hi, x_seed=60 + t, live=lv)
        pool_ref = _grow(_scatter(pool_ref, ref["rec"], ref["conv"], wb_t), 0, 0)
        got = _step(monkeypatch, side=True, raws=pool_sc, rb=rb_t, hi=hi, x_seed=60 + t, side_in=side, side_on=on, side_off=off_prev, live=lv)
        n = (_ndiff(got["res"], ref["res"]), _ndiff(got["rec"], ref["rec"]), _ndiff(got["conv"], ref["conv"]))
        print(f"Q38SIDE S5 live={int(live)} step {t + 1} side_on {on}: residual / rec / conv bytes differing {n}", flush=True)
        if any(n):
            bad.append((t + 1, n))
        if unnulled is not None:
            ctl = _step(monkeypatch, side=True, raws=unnulled, rb=rb_t, hi=hi, x_seed=60 + t, side_in=side, side_on=0, side_off=off_prev, live=lv)
            nctl = _ndiff(ctl["res"], ref["res"])
            print(f"Q38SIDE S5 live={int(live)} control (step 5 flush without the null): residual bytes differing {nctl} (want > 0)", flush=True)
        side = ([got["rec"][r] for r in range(R)], [got["conv"][r] for r in range(R)])
        off = rng.integers(0, T, (NL, NS)).astype(np.int32)
        rb_t = np.take_along_axis(wb_t, off[:, :, None], axis=2)[:, :, 0].astype(np.int32)  # the next step reads the accepted checkpoint
        off_prev, wb_prev = off, wb_t
        wb_t = (wb_t + NL * NS * T).astype(np.int32)  # fresh write blocks for the next step (both worlds), just above this step's
        if live:
            wb_t[:, NS - 1, :] = 0  # the pad keeps null write blocks
        need = int(wb_t.max()) + 1
        pool_ref, pool_sc = _grow(pool_ref, need, 100 + t), _grow(pool_sc, need, 100 + t)
    assert not bad, bad
    assert nctl is not None and nctl > 0, nctl


@pytest.mark.parametrize("live", [False, True])
def test_s6_pad_request_masked(monkeypatch, live):
    """The served HIX shape at 3 live: the 16-row bucket's last request is a pad (has_state 0; its write ids null in the served call). Its
    sidecar slot holds the pad's previous-step checkpoints (junk, not zero); the side read must still give it the pool path's zero state /
    conv rows (`keep = HAS_STATE != 0`), so ALL 16 rows match the reference bit for bit. Control: the pad's sidecar plane is non-zero.
    live=True: with X-K-LIVE mode 1 (the S2 / S3a form) and the mask marking the pad, against the flag-off kernel in the same mode."""
    raws, rb, hi, wb, NL = _pools(61)
    hi, rb, wb = hi.copy(), rb.copy(), wb.copy()
    hi[:, NS - 1], rb[:, NS - 1], wb[:, NS - 1, :] = 0, 0, 0  # the pad request (no state, null blocks, as gdn_blocks gives it)
    lv = [1] * (NS - 1) + [0] if live else None
    rng = np.random.default_rng(62)
    off = rng.integers(0, T, (NL, NS)).astype(np.int32)
    s1 = _step(monkeypatch, side=False, raws=raws, rb=rb, hi=hi, x_seed=7, live=lv)
    pool1 = _scatter(raws, s1["rec"], s1["conv"], wb)
    rb2 = np.take_along_axis(wb, off[:, :, None], axis=2)[:, :, 0].astype(np.int32)
    ref = _step(monkeypatch, side=False, raws=pool1, rb=rb2, hi=hi, x_seed=8, live=lv)
    side_in = ([s1["rec"][r] for r in range(R)], [s1["conv"][r] for r in range(R)])
    got = _step(monkeypatch, side=True, raws=raws, rb=rb2, hi=hi, x_seed=8, side_in=side_in, side_on=1, side_off=off, live=lv)
    pad = slice((NS - 1) * T * VH, NS * T * VH)
    junk = int(sum(np.count_nonzero(np.asarray(side_in[0][r])[:, pad].astype(np.float32)) for r in range(R)))
    n = (_ndiff(got["res"], ref["res"]), _ndiff(got["rec"], ref["rec"]), _ndiff(got["conv"], ref["conv"]))
    print(f"Q38SIDE S6 live={int(live)} pad request (has_state 0): residual / rec / conv bytes differing {n}; non-zero pad sidecar values {junk} (want > 0)",
          flush=True)
    assert junk > 0
    assert n == (0, 0, 0), n


def _sub_jaxprs(v):
    if hasattr(v, "eqns"):
        yield v
    elif hasattr(v, "jaxpr") and hasattr(v.jaxpr, "eqns"):
        yield v.jaxpr
    elif isinstance(v, (tuple, list)):
        for x in v:
            yield from _sub_jaxprs(x)


def _side_dma_depths(closed, NL: int) -> collections.Counter:
    """The cond-nesting depth of every `dma_start` in the kernel jaxpr with a sidecar-plane operand (a ref of the sidecar rec / conv plane
    shape): the side-gather copies, the redirected checkpoint write-backs and (interpret) the init copies. Only the side-gather copies are
    live-gated, so a live-rows on / off difference isolates them."""
    rec_s, conv_s = sc.side_shapes(NL, B, NS, T, VH, KD, VD, QKV)
    out = collections.Counter()

    def aval_of(v):
        a = getattr(v, "aval", None)
        return getattr(a, "inner_aval", a)

    def walk(j, depth):
        for e in j.eqns:
            if e.primitive.name == "dma_start":
                avs = [aval_of(v) for v in e.invars]
                shapes = [tuple(getattr(a, "shape", ())) for a in avs]
                if tuple(rec_s) in shapes or tuple(conv_s) in shapes:
                    out[depth] += 1
            d = depth + (1 if e.primitive.name == "cond" else 0)
            for pv in e.params.values():
                for sj in _sub_jaxprs(pv):
                    walk(sj, d)

    walk(closed.jaxpr, 0)
    return out


def test_s7_live_pad_side_copies_gated(monkeypatch):
    """X-K-LIVE mode 1: a pad request's side copies are skipped (not started), as `_gather_live` skips its pool copies. Outputs cannot
    show it (the pad's rows are zeroed after the copies either way, S5 / S6 live=True), so this is the structural check, on the kernel's
    jaxpr: with live rows, every side-gather copy start (2 per request slot: rec rows + conv plane) sits one `cond` deeper (the request's
    live predicate) than without; nothing else on the sidecar's semaphores moves."""
    raws, rb, hi, wb, NL = _pools(71)
    side_in = _junk_side(NL, 72)
    off = np.zeros((NL, NS), np.int32)
    j0 = _step(monkeypatch, side=True, raws=raws, rb=rb, hi=hi, x_seed=3, side_in=side_in, side_on=1, side_off=off, trace_only=True)
    j1 = _step(monkeypatch, side=True, raws=raws, rb=rb, hi=hi, x_seed=3, side_in=side_in, side_on=1, side_off=off, live=[1] * NS,
               trace_only=True)
    d0, d1 = _side_dma_depths(j0, NL), _side_dma_depths(j1, NL)
    gone, came = dict(d0 - d1), dict(d1 - d0)
    print(f"Q38SIDE S7 side-plane copy starts by cond depth: live rows off {dict(d0)}, on {dict(d1)}; moved {gone} -> {came}", flush=True)
    assert sum(gone.values()) == 2 * NS and sum(came.values()) == 2 * NS, (d0, d1)
    assert len(gone) == 1 and len(came) == 1 and list(came)[0] == list(gone)[0] + 1, (gone, came)


# ------------------------------------------------------------------------------- plumbing (no kernel run)

def test_key_default_off_and_selection(monkeypatch):
    monkeypatch.delenv(sc.ENV, raising=False)
    from vllm_torchtpu import envs

    assert envs.VLLM_TPU_MK_GDN_SIDECAR is False
    assert sc.sidecar_gdn_cls(Qwen38PoolGDN) is Qwen38PoolGDN
    monkeypatch.setenv(sc.ENV, "1")
    assert sc.sidecar_gdn_cls(Qwen38PoolGDN) is Qwen38PoolGDN  # no bucket in context (segment specs)
    assert sc.building_rows(16, lambda: sc.sidecar_gdn_cls(Qwen38PoolGDN))() is sc.Qwen38SidecarGDN
    assert sc.building_rows(8, lambda: sc.sidecar_gdn_cls(Qwen38PoolGDN))() is Qwen38PoolGDN
    assert sc._CTX_ROWS[0] is None
    with pytest.raises(ValueError):
        sc.building_rows(16, lambda: sc.sidecar_gdn_cls(ink.Qwen38PoolGDNInk))()


def test_operands_and_outputs(monkeypatch):
    spec = _spec(monkeypatch, True)
    ex = ExecSpec(R=R, B=B, T=T, ctx=128, interpret=True, tiles=TILES)
    names = [x.name for x in kernel.extra_inputs(spec, ex)]
    assert names[-4:] == [sc.SIDE_REC, sc.SIDE_CONV, sc.SIDE_ON, sc.SIDE_OFF], names
    ex_tpu = ExecSpec(R=R, B=B, T=T, ctx=128, interpret=False, tiles=TILES)
    blk = [getattr(b, "inner", b) for b in spec.all_blocks() if isinstance(getattr(b, "inner", b), sc.Qwen38SidecarGDN)][0]
    assert blk.outputs(None, ex_tpu) == []
