# SPDX-License-Identifier: Apache-2.0
"""q38 X-GLUE-1-D14 (VLLM_TPU_MK_GDN_CKPT_INK=1): the pool GDN block writes its verify checkpoints into the pool in-kernel
(megakernel/models/qwen38_gdn_ckpt_ink.py). CPU interpret, real per-rank GDN shapes (TINY at R = 2: 2 k-heads, 6 v-heads,
128 x 128 states, QKV 1280) and the real pool byte layout.

The deciding check: on the same inputs, the pool after a flag-on kernel equals, byte for byte, the pool after today's path
(the flag-off kernel's `conv_ckpt` / `rec_ckpt` -> `scatter_gdn_ckpts_fast` with the same write ids), and the residual is
bit-identical. Every byte of the input pool is random, so a write to a wrong block, row or slot, a stale staging buffer or a
missing DMA moves bytes; the blocks that change must be exactly the non-null write ids (a positive control for the 0 %
case's zero count). Buckets: 8 / 16 rows (whole-state path), 20 (chunked, with a tail chunk), 32 (chunked; head-major
update as served, and the chunk gather); kernel blocks of 128 and 512 rows; 0 / 50 / 100 % live write ids and a pattern
where a request's read block is also its first checkpoint's write block. Plus X-K-LIVE mode 1 (pad requests' gather DMAs
skipped), the segment call end to end (head and body, both kernel block sizes; the wrapper drops the scatter and uses the
kernel's pool), and the plumbing (class selection, engagement line, operand / output lists).
Run: JAX_PLATFORMS=cpu PYTHONPATH=src pytest tests/megakernel/test_glue_ckpt_ink.py
"""

from __future__ import annotations

import logging
import os
import sys
import types
from types import SimpleNamespace

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
from vllm_torchtpu.megakernel.live_rows import LIVE_ROWS, live_rows_from_qsl  # noqa: E402
from vllm_torchtpu.megakernel.models import qwen38  # noqa: E402
from vllm_torchtpu.megakernel.models import qwen38_gdn_ckpt_ink as ink  # noqa: E402
from vllm_torchtpu.megakernel.models.qwen38_gdn_pool import Qwen38PoolGDN  # noqa: E402
from vllm_torchtpu.megakernel.models.qwen38_inputs import pack_segment, random_segment_inputs  # noqa: E402
from vllm_torchtpu.megakernel.models.qwen38_pack import shard_step_inputs  # noqa: E402

T = 4
R = 2
G = qwen38.TINY
TILES = Tiles(bank=1 << 20, nbd=10, nbe=8, ebank=512 << 10, vmem_mib=64)
LANES = 256
FLAG = ink.ENV


def _u8(x) -> np.ndarray:
    return np.asarray(jax.lax.bitcast_convert_type(jnp.asarray(x), jnp.uint8))


def _bits(a) -> np.ndarray:
    a = np.ascontiguousarray(np.asarray(a))
    return a.view(np.uint8)


def _encode(raw: np.ndarray, m: int, ssm: np.ndarray, conv: np.ndarray) -> None:
    from tests.megakernel.test_qwen38_gdn_pool import _encode as enc

    enc(raw, m, ssm, conv)


def _ids(pattern: str, NL: int, NS: int, rb: np.ndarray, rng) -> np.ndarray:
    """Write ids [NL, NS, T]: distinct blocks after the read blocks; `half` zeroes a random half, `none` all, `rw` makes
    every request's first checkpoint write its own read block (the served layout when the read offset is 0)."""
    base = 1 + NL * NS
    wb = (base + rng.permutation(NL * NS * T)).astype(np.int32).reshape(NL, NS, T)
    if pattern == "half":
        z = np.zeros(NL * NS * T, bool)
        z[rng.permutation(z.size)[: z.size // 2]] = True
        wb = np.where(z.reshape(wb.shape), 0, wb).astype(np.int32)
    elif pattern == "none":
        wb = np.zeros_like(wb)
    elif pattern == "rw":
        wb[:, :, 0] = np.where(rb > 0, rb, wb[:, :, 0])
    else:
        assert pattern == "all", pattern
    return wb


def _kernel_inputs(B: int, kb: int, pattern: str, seed: int, pads_like=None):
    """Segment inputs, per-rank pools (every byte random, then the read states encoded), read / has-state / write ids."""
    L = qwen38.SEGMENT_LAYERS
    NS, NL = B // T, L - 1
    rng = np.random.default_rng(seed + 1)
    nmb = 1 + NL * NS + NL * NS * T
    rb = 1 + rng.permutation(NL * NS).astype(np.int32).reshape(NL, NS)
    hi = (np.arange(NL * NS).reshape(NL, NS) % 3 != 1).astype(np.int32)
    if pads_like is not None:  # served pads: null read block, no prior state, null write ids (gdn_blocks)
        lv = np.asarray(pads_like, bool)[None]
        rb, hi = np.where(lv, rb, 0).astype(np.int32), np.where(lv, hi, 0).astype(np.int32)
    wb = _ids(pattern, NL, NS, rb, rng)
    if pads_like is not None:
        wb = np.where(np.asarray(pads_like, bool)[None, :, None], wb, 0).astype(np.int32)
    raws = []
    for _ in range(R):
        raw = rng.integers(0, 256, (nmb * 4, 128, 1, 4, LANES), dtype=np.uint8)
        for l in range(NL):
            for s in range(NS):
                ssm = rng.standard_normal((6, 128, 128)).astype(np.float32) * 0.05
                conv = rng.standard_normal((3, 1280)).astype(np.float32) * 0.05
                if rb[l, s]:
                    _encode(raw, int(rb[l, s]), ssm, conv)
        raws.append(raw.reshape(nmb * 512 // kb, kb, 1, 4, LANES))
    return raws, rb, hi, wb


def _layout(raw: np.ndarray):
    from vllm_torchtpu.megakernel_runtime.qwen38_gdn_state import Qwen38GdnPoolLayout

    return Qwen38GdnPoolLayout(pool_shape=tuple(raw.shape), pool_dtype="float8_e4m3fn")


def _run(monkeypatch, B: int, *, on: bool, kb: int = 128, pattern: str = "all", chunk_gather: str = "0",
         head_major: bool = False, seed: int = 5, live=None, pads_like=None) -> dict:
    """One pool-GDN segment step (R = 2). Flag off: the pool after today's scatter of the kernel's checkpoints; flag on:
    the kernel's own pool output. Inputs depend only on (B, kb, pattern, seed, pads_like)."""
    from vllm_torchtpu.megakernel.models import qwen38_gdn_pool
    from vllm_torchtpu.megakernel_runtime.qwen38_scatter import scatter_gdn_ckpts_fast

    monkeypatch.setattr(qwen38_gdn_pool, "CHUNK_GATHER", chunk_gather)
    if on:
        monkeypatch.setenv(FLAG, "1")
    else:
        monkeypatch.delenv(FLAG, raising=False)
    L = qwen38.SEGMENT_LAYERS
    NS, NL = B // T, L - 1
    tiles = TILES.with_blocks({"gdn": {"gdn_head_major": True}}) if head_major else TILES
    ex = ExecSpec(R=R, B=B, T=T, ctx=128, interpret=True, tiles=tiles, live_rows=1 if live is not None else False)
    assert ex.WIDE == (B > 16)
    spec = qwen38.segment_spec(G, R, L=L, pool_gdn=True)
    blks = ink.ink_blocks(spec)
    assert len(blks) == (1 if on else 0), blks
    raws, rb, hi, wb = _kernel_inputs(B, kb, pattern, seed, pads_like)
    if on:
        assert ink.ink_extra_outs(spec, True, _layout(raws[0])) == (ink.POOL_OUT,)
    inp = random_segment_inputs(jax.random.PRNGKey(seed), G, ex, L, gain=0.25)
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
        ink.WRITE_BLOCKS: jax.device_put(jnp.asarray(wb.reshape(NL, NS * T)), rep),
    }
    if live is not None:
        q = np.where(np.asarray(live, bool), T, 0).astype(np.int32)
        qsl = np.concatenate([[0], np.cumsum(q)]).astype(np.int32)
        xs[LIVE_ROWS] = jax.device_put(jnp.asarray(live_rows_from_qsl(jnp.asarray(qsl), NS, T)), rep)
    names = [x.name for x in kernel.extra_inputs(spec, ex)]
    assert (ink.WRITE_BLOCKS in names) == on, names
    step, info = kernel.make_step(mesh, spec, ex)
    o_names = info["plan"].o_names
    assert (("rec_ckpt" in o_names) and ("conv_ckpt" in o_names)) == (not on), o_names
    assert (ink.POOL_OUT in o_names) == on, o_names
    _, outs = step(pos, hid, w_mk, st, *(xs[k] for k in names))
    res = np.asarray(outs[RESIDUAL_OUT]).reshape(R, ex.M, -1)[:, :B]
    if on:
        new = _u8(outs[ink.POOL_OUT]).reshape(R, *raws[0].shape)
    else:
        conv = np.asarray(outs["conv_ckpt"]).reshape(R, NL, *np.asarray(outs["conv_ckpt"]).shape[1:])
        rec = np.asarray(outs["rec_ckpt"]).reshape(R, NL, *np.asarray(outs["rec_ckpt"]).shape[1:])
        new = []
        for r in range(R):
            p = scatter_gdn_ckpts_fast(jax.lax.bitcast_convert_type(jnp.asarray(raws[r]), jnp.float8_e4m3fn),
                                       jnp.asarray(conv[r]), jnp.asarray(rec[r]), jnp.asarray(wb),
                                       layout=_layout(raws[r]), interpret=True)
            new.append(_u8(p))
        new = np.stack(new)
    return dict(res=res, pool=new, pool_in=np.stack(raws), wb=wb, kb=kb)


def _changed_blocks(a: np.ndarray, b: np.ndarray, kb: int) -> set:
    """Manager blocks (per rank) whose bytes differ: {(rank, block)}."""
    per = 512 // kb
    d = (a != b).reshape(a.shape[0], a.shape[1] // per, -1).any(-1)
    return {(int(r), int(m)) for r, m in zip(*np.nonzero(d))}


def _check(off: dict, on: dict, tag: str) -> None:
    kb, wb = off["kb"], off["wb"]
    assert np.array_equal(off["pool_in"], on["pool_in"]), f"{tag}: inputs differ between the two runs"
    want = {(r, int(m)) for r in range(R) for m in np.unique(wb) if m != 0}
    ch_off = _changed_blocks(off["pool"], off["pool_in"], kb)
    ch_on = _changed_blocks(on["pool"], on["pool_in"], kb)
    nbad = int((on["pool"] != off["pool"]).sum())
    bad_blocks = sorted(_changed_blocks(on["pool"], off["pool"], kb))
    nres = int((_bits(on["res"]) != _bits(off["res"])).sum())
    print(f"Q38GLUE {tag}: live write ids {len(want) // R} per rank; blocks changed: today's scatter {len(ch_off)}, "
          f"in-kernel {len(ch_on)}; pool bytes differing (in-kernel vs today) {nbad} in blocks {bad_blocks[:8]}; "
          f"residual bytes differing {nres}", flush=True)
    assert ch_off == want, f"{tag}: today's scatter changed {sorted(ch_off ^ want)[:8]} vs the write ids (control)"
    assert nbad == 0, f"{tag}: in-kernel pool differs from today's scatter in {nbad} bytes, blocks {bad_blocks[:8]}"
    assert ch_on == want, f"{tag}: in-kernel write changed {sorted(ch_on ^ want)[:8]} vs the write ids"
    assert nres == 0, f"{tag}: residual not bit-identical ({nres} bytes)"


# (B, kb, pattern, chunk_gather, head_major): the bitwise cases (node ids are the mutant targets of the proof harness)
CASES = {
    "b8_all_k128": (8, 128, "all", "0", False),
    "b16_half_k512": (16, 512, "half", "0", False),
    "b16_rw_k128": (16, 128, "rw", "0", False),
    "b16_none_k128": (16, 128, "none", "0", False),
    "b20_all_k128_tail": (20, 128, "all", "0", True),
    "b32_half_k128_hm": (32, 128, "half", "0", True),
    "b32_rw_k512_cg": (32, 512, "rw", "1", True),
}


@pytest.mark.parametrize("case", list(CASES))
def test_ink_pool_bitwise(monkeypatch, case):
    B, kb, pattern, cg, hm = CASES[case]
    kw = dict(kb=kb, pattern=pattern, chunk_gather=cg, head_major=hm, seed=5 + B)
    off = _run(monkeypatch, B, on=False, **kw)
    on = _run(monkeypatch, B, on=True, **kw)
    _check(off, on, case)
    if pattern == "none":  # zero count; its positive control is every other case (want > 0 blocks there)
        assert not (on["pool"] != on["pool_in"]).any(), f"{case}: a null write id moved bytes"


LIVE1 = {"b16_live2": (16, [1, 1, 0, 0], 128, "0", False), "b32_live3_cg": (32, [1, 1, 1, 0, 0, 0, 0, 0], 512, "1", True)}


@pytest.mark.parametrize("case", list(LIVE1))
def test_ink_pool_bitwise_live_mode1(monkeypatch, case):
    """X-K-LIVE mode 1 (ExecSpec.live_rows = 1, served pads: null read block, no prior state, null write ids): in-kernel
    write vs today's scatter, both under mode 1."""
    B, lr, kb, cg, hm = LIVE1[case]
    kw = dict(kb=kb, pattern="all", chunk_gather=cg, head_major=hm, seed=31 + B, live=lr, pads_like=lr)
    off = _run(monkeypatch, B, on=False, **kw)
    on = _run(monkeypatch, B, on=True, **kw)
    _check(off, on, f"live1 {case}")


# ------------------------------------------------------------------------------- the segment call, end to end

def _segment_call(monkeypatch, *, on: bool, L: int, head: bool, kb: int, seed: int = 7):
    from vllm_torchtpu.megakernel_runtime.exported_bridge import make_typed_step_wrapper
    from vllm_torchtpu.megakernel_runtime.qwen38_segment_call import build_segment_call

    if on:
        monkeypatch.setenv(FLAG, "1")
    else:
        monkeypatch.delenv(FLAG, raising=False)
    NS, B = 2, 2 * T
    NL = L if head else L - 1
    nmb = 1 + NL * NS * T
    rng = np.random.default_rng(seed)
    ck = 1 + rng.permutation(NL * NS * T).astype(np.int32).reshape(NL, NS, T)
    ck[:, 1, 2] = 0  # one null checkpoint id per layer
    off = np.zeros((NL, nmb), np.int32)
    for ln in range(NL):
        for s in range(NS):
            off[ln, ck[ln, s, 0]] = 0  # read offset 0: the state is read from the first checkpoint's block
    raws = []
    for _ in range(R):
        raw = rng.integers(0, 256, (nmb * 4, 128, 1, 4, LANES), dtype=np.uint8)
        for m in range(1, nmb):
            _encode(raw, m, rng.standard_normal((6, 128, 128)).astype(np.float32) * 0.05,
                    rng.standard_normal((3, 1280)).astype(np.float32) * 0.05)
        raws.append(raw.reshape(nmb * 512 // kb, kb, 1, 4, LANES))
    ex = ExecSpec(R=R, B=B, T=T, ctx=128, interpret=True, tiles=TILES)
    spec0 = qwen38.segment_spec(G, R, L=L, head=head)
    inp = random_segment_inputs(jax.random.PRNGKey(seed), G, ex, L, gain=0.25, head=head)
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    tp, rep = NamedSharding(mesh, P("tp")), NamedSharding(mesh, P())
    w_mk, _, _ = pack_segment(spec0, G, ex, inp, head=head)
    spec = qwen38.segment_spec(G, R, L=L, head=head, pool_gdn=True)
    flat = []
    for name in kernel.build(spec, ex)[1]["plan"].w_names if not on else _w_names(spec, ex, raws[0]):
        v = w_mk[name]
        flat.extend(v if isinstance(v, tuple) else (v,))
    flat = [jax.device_put(kernel.fold(x), tp) for x in flat]
    hidden = jax.device_put(jnp.tile(inp.x, (R, 1)), tp)
    attn = jax.device_put(inp.attn.reshape(R * B, -1), tp)
    pool = jax.device_put(jnp.concatenate(
        [jax.lax.bitcast_convert_type(jnp.asarray(r), jnp.float8_e4m3fn) for r in raws], axis=0), tp)
    meta = []
    for ln in range(NL):
        meta += [jnp.asarray(ck[ln, :, 0]), jnp.asarray(off[ln]), jnp.asarray(ck[ln]),
                 jnp.arange(NS + 1, dtype=jnp.int32) * T, jnp.full((NS,), 100, jnp.int32)]
    meta = [jax.device_put(m, rep) for m in meta]
    fn, in_specs, out_specs, donate = build_segment_call(mesh, G, L, B, _layout(raws[0]), interpret=True, tiles=TILES,
                                                         head=head, pool_gdn=True)
    lead = (hidden,) if head else (hidden, attn)
    typed = make_typed_step_wrapper("glue_seg_call_test", fn, len(in_specs), len(out_specs))
    res, new_pool = jax.jit(typed)(*lead, *flat, pool, *meta)
    return dict(res=np.asarray(res), pool=_u8(new_pool), pool_in=np.stack(raws), ck=ck, kb=kb)


def _w_names(spec, ex, raw0):
    for b in ink.ink_blocks(spec):  # the flag-on spec's planner needs the interpret pool shape to build
        b.interpret_pool_shape = tuple(raw0.shape)
    return kernel.build(spec, ex)[1]["plan"].w_names


SEG = {"head_k128": (2, True, 128), "head_k512": (2, True, 512), "body_k128": (4, False, 128), "body_k512": (4, False, 512)}


@pytest.mark.parametrize("case", list(SEG))
def test_ink_segment_call_end_to_end(monkeypatch, case):
    """build_segment_call (the head call of D14 and the body segment): flag on == flag off, pool bytes and residual."""
    L, head, kb = SEG[case]
    off = _segment_call(monkeypatch, on=False, L=L, head=head, kb=kb)
    on = _segment_call(monkeypatch, on=True, L=L, head=head, kb=kb)
    ck = off["ck"]
    want = {(r, int(m)) for r in range(R) for m in np.unique(ck) if m != 0}
    new_off = off["pool"].reshape(off["pool_in"].shape)
    new_on = on["pool"].reshape(on["pool_in"].shape)
    ch_off = _changed_blocks(new_off, off["pool_in"], kb)
    nbad = int((new_on != new_off).sum())
    nres = int((_bits(on["res"]) != _bits(off["res"])).sum())
    print(f"Q38GLUE segcall {case}: blocks changed (today) {len(ch_off)} want {len(want)}; pool bytes differing {nbad}; "
          f"residual bytes differing {nres}", flush=True)
    assert ch_off == want, f"segcall {case}: control: today's call changed {sorted(ch_off ^ want)[:8]}"
    assert nbad == 0, f"segcall {case}: flag-on pool differs from flag-off in {nbad} bytes"
    assert nres == 0, f"segcall {case}: residual differs in {nres} bytes"


# ------------------------------------------------------------------------------- plumbing (no kernel run)

def test_flag_default_off_and_class_selection(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    from vllm_torchtpu import envs

    assert envs.VLLM_TPU_MK_GDN_CKPT_INK is False
    assert ink.pool_gdn_cls() is Qwen38PoolGDN
    assert ink.ink_blocks(qwen38.segment_spec(G, R, pool_gdn=True)) == []
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setattr(ink, "_LOGGED", False)
    recs: list = []
    h = logging.Handler()
    h.emit = recs.append  # type: ignore[method-assign]
    lg = logging.getLogger("vllm." + ink.__name__)
    old_level = lg.level
    lg.addHandler(h)
    lg.setLevel(logging.INFO)
    try:
        assert ink.pool_gdn_cls() is ink.Qwen38PoolGDNInk
        ink.pool_gdn_cls()
    finally:
        lg.removeHandler(h)
        lg.setLevel(old_level)
    lines = [r.getMessage() for r in recs if ink.ENGAGED in r.getMessage()]
    assert len(lines) == 1 and f"{FLAG}=1" in lines[0], lines
    from vllm_torchtpu.megakernel.wholestep.spec import wholestep_spec

    assert len(ink.ink_blocks(qwen38.segment_spec(G, R, pool_gdn=True))) == 1
    assert len(ink.ink_blocks(qwen38.segment_spec(G, R, L=2, head=True, pool_gdn=True))) == 1
    assert ink.ink_blocks(qwen38.segment_spec(G, R)) == []  # gather path (no pool GDN): never the in-kernel block
    assert len(ink.ink_blocks(wholestep_spec(G, R, L=8, pool_gdn=True))) == 1


def test_operands_outputs_and_scratch(monkeypatch):
    """Flag on: + gdn_wb (SMEM [NGDN, B]), no rec_ckpt / conv_ckpt, gdn_pool_out only in interpret mode, 2 x 4 DMA
    semaphores + 2 SMEM words of scratch and no new VMEM buffer (the staging ring is the gather's)."""
    B = 16
    monkeypatch.delenv(FLAG, raising=False)
    ex_t = ExecSpec(R=R, B=B, T=T, ctx=128, tiles=TILES)
    ex_i = ExecSpec(R=R, B=B, T=T, ctx=128, tiles=TILES, interpret=True)
    spec0 = qwen38.segment_spec(G, R, pool_gdn=True)
    _, info0 = kernel.build(spec0, ex_t)
    monkeypatch.setenv(FLAG, "1")
    spec1 = qwen38.segment_spec(G, R, pool_gdn=True)
    _, info1 = kernel.build(spec1, ex_t)
    n0 = [x.name for x in kernel.extra_inputs(spec0, ex_t)]
    n1 = [x.name for x in kernel.extra_inputs(spec1, ex_t)]
    assert n1 == n0 + [ink.WRITE_BLOCKS], (n0, n1)
    x = [e for e in kernel.extra_inputs(spec1, ex_t) if e.name == ink.WRITE_BLOCKS][0]
    assert x.memory_space == "smem" and x.shape == (qwen38.SEGMENT_LAYERS - 1, B), x
    assert {"rec_ckpt", "conv_ckpt"} <= set(info0["plan"].o_names)
    assert not ({"rec_ckpt", "conv_ckpt", ink.POOL_OUT} & set(info1["plan"].o_names)), info1["plan"].o_names
    v0 = {k: v for k, v in info0["vmem_shapes"].items()}
    v1 = {k: v for k, v in info1["vmem_shapes"].items()}
    assert v1 == v0, sorted(set(v1) ^ set(v0))
    assert info1["vmem_scratch_bytes"] == info0["vmem_scratch_bytes"]
    assert set(info1["s_names"]) - set(info0["s_names"]) == {"ck_sem", "ck_pend"}
    with pytest.raises(AssertionError, match="interpret_pool_shape"):
        kernel.build(qwen38.segment_spec(G, R, pool_gdn=True), ex_i)
    spec2 = qwen38.segment_spec(G, R, pool_gdn=True)
    for b in ink.ink_blocks(spec2):
        b.interpret_pool_shape = (36, 128, 1, 4, 256)
    _, info2 = kernel.build(spec2, ex_i)
    assert ink.POOL_OUT in info2["plan"].o_names


def test_refuses_decode_and_f32_state(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    spec = qwen38.segment_spec(G, R, pool_gdn=True)
    with pytest.raises(NotImplementedError, match="verify"):
        kernel.extra_inputs(spec, ExecSpec(R=R, B=8, T=1, ctx=128, tiles=TILES))
    monkeypatch.setattr(ink, "REC", jnp.float32)
    with pytest.raises(NotImplementedError, match="bf16"):
        kernel.extra_inputs(spec, ExecSpec(R=R, B=8, T=T, ctx=128, tiles=TILES))


def test_wrapper_helpers():
    fn_off = SimpleNamespace(extra_names=("gdn_pool", "gdn_rb", "gdn_hi"))
    fn_on = SimpleNamespace(extra_names=("gdn_pool", "gdn_rb", "gdn_hi", ink.WRITE_BLOCKS))
    wb = jnp.arange(3 * 2 * 4, dtype=jnp.int32).reshape(3, 2, 4)
    assert ink.ink_xs(fn_off, wb) == {}
    got = ink.ink_xs(fn_on, wb)[ink.WRITE_BLOCKS]
    assert got.shape == (3, 8) and np.array_equal(np.asarray(got), np.asarray(wb).reshape(3, 8))
    outs = ("res", "kv", "kr", "ir")
    assert ink.ink_outs(fn_off, outs, "pool", False) is outs
    assert ink.ink_outs(fn_on, outs, "pool", False) == ("res", "pool", "pool", "kv", "kr", "ir")
    assert ink.ink_outs(fn_on, outs + ("pout",), "pool", True) == ("res", "pout", "pout", "kv", "kr", "ir")
    assert ink.ink_extra_outs(None, True, None) == ()


# ------------------------------------------------------------------------------- whole-step call plumbing (stubbed kernel)

def test_wholestep_call_passes_the_pool_through(monkeypatch):
    """The real build_wholestep_call closures (`call` and D07-b1's `call_shared`) with the kernel, QSA inputs and writers
    stubbed (the D07-b1 test's stubs): flag on, the kernel gets gdn_wb = wb, the scatter is never called and the QSA
    writer gets the kernel's pool (interpret) unchanged; flag off, the scatter runs as before."""
    from tests.megakernel_runtime import test_l1d07b1_gdn_meta_shared as d7
    from vllm_torchtpu.megakernel_runtime import qwen38_wholestep_call as wsc
    from vllm_torchtpu.megakernel_runtime.qwen38_segment_call import gdn_blocks

    calls = {"scatter": 0}
    d7._stub(monkeypatch)
    stock_scatter = wsc.scatter_gdn_ckpts_fast

    def scatter(*a, **k):
        calls["scatter"] += 1
        return stock_scatter(*a, **k)

    monkeypatch.setattr(wsc, "scatter_gdn_ckpts_fast", scatter)
    seen = {}
    extra = d7.EXTRA + (ink.WRITE_BLOCKS,)

    def fn(pos, x, *rest):  # the kernel's pool output (interpret): 7 everywhere, its gdn_wb input in front per rank
        xs = dict(zip(extra, rest[: len(extra)]))
        wbf = xs[ink.WRITE_BLOCKS].reshape(-1)
        seen["n"] = int(wbf.shape[0])
        M = x.shape[0] // R
        z = jnp.zeros((R * d7.N_ATT, M, 4), jnp.float32)
        pout = jnp.full((R, d7.POOL), 7, jnp.int32).at[:, : wbf.shape[0]].set(wbf[None]).reshape(-1)
        return (x, z, z, z, pout)

    fn.extra_names = extra
    monkeypatch.setattr(wsc, "make_flat_segment_step",
                        lambda mesh, spec, ex, extra_outs=(): (fn, (P(), P("tp")) + (P(),) * len(extra), None, 1))
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    md, layers, gdn_ids = d7._metadata_objects()
    per_layer = [np.asarray(t.numpy()) for t in d7._per_layer_operands(md, layers, gdn_ids)]
    shared = [np.asarray(t.numpy()) for t in d7._module(layers, gdn_ids)._gdn_meta_shared(md)]
    rng = np.random.default_rng(3)
    hidden = rng.standard_normal((R * d7.B, d7.H)).astype(np.float32)
    w0 = rng.standard_normal((R * 4, d7.H)).astype(np.float32)
    pool = np.zeros((R * d7.POOL,), np.int32)
    tabs = [rng.integers(0, 50, (48 * n,)).astype(np.int32) for _ in range(d7.N_ATT) for n in (8, 32, 32)]
    tail = [*tabs, rng.integers(0, 300, (d7.B,)).astype(np.int32), np.arange(d7.B + 1, dtype=np.int32),
            rng.integers(0, 300, (d7.B,)).astype(np.int32), rng.standard_normal((16, 128)).astype(np.float32),
            *[rng.standard_normal((n,)).astype(np.float32) for _ in range(d7.N_ATT) for n in (256, 128)]]
    NS = d7.B // T
    _, wb_want, _ = (np.asarray(x) for x in jax.jit(lambda *m: gdn_blocks(m, NS, True))(*per_layer))
    for shared_flag, meta in ((False, per_layer), (True, shared)):
        if shared_flag:
            monkeypatch.setenv("VLLM_TPU_MK_GDN_META_DEDUP", "1")
        else:
            monkeypatch.delenv("VLLM_TPU_MK_GDN_META_DEDUP", raising=False)
        call, *_ = d7._build(mesh)
        calls["scatter"] = 0
        res, pool1 = jax.jit(call)(hidden, w0, pool, *meta, *tail)
        assert calls["scatter"] == 0, "scatter called with the in-kernel write"
        n = seen["n"]
        assert n == wb_want.size, (n, wb_want.shape)
        p = np.asarray(pool1).reshape(R, d7.POOL)
        assert (p[:, :n] == wb_want.reshape(1, -1)).all(), "the kernel's gdn_wb is not the call's wb"
        assert (p[:, n:-1] == 7).all(), "the QSA writer did not get the kernel's pool"
        print(f"Q38GLUE wholestep call ({call.__name__}): scatter calls 0, gdn_wb == wb {wb_want.shape}, pool from the "
              f"kernel", flush=True)
