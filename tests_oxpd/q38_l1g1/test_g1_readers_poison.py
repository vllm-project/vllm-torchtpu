# SPDX-License-Identifier: Apache-2.0
"""q38 L1-G1 GDN-COMMIT1: multi-step BITWISE equality of every downstream reader, with the non-accepted checkpoint blocks
poisoned (CPU interpret; the real pool-GDN segment kernel, TINY at R = 2: real per-rank GDN shapes, the real pool bytes).

Two worlds run the same steps from the same pools:
  base (flag off): every step's kernel checkpoints go through today's `scatter_gdn_ckpts_fast` (all four per request);
  G1:              the runner's commit (`commit_from_offsets`: accept = read_offsets[state ids] after the post-verify
                   update) writes only the accepted checkpoint, then EVERY other checkpoint block of the step is filled
                   with 0xFF bytes (an FP8 NaN in every byte, an f32 NaN in every word).
Per step the read / write / has-state ids come from the real `gdn_blocks` on runner-shaped metadata (state ids, the
block-indexed read-offset table, checkpoint ids, query starts, seq lens), accept counts 0..3 vary per request and step
(each value occurs), requests go dead for a step (pad rows: null ids) and come back, and one request crosses a block
boundary: the checkpoint group slides (new ckpt 0..2 = old ckpt 1..3, a fresh block last), the seed copy moves the block
the runner's `_spec_seed_rows_impl` picks (the checkpoint at the read offset; tests/q38_l1g1/test_g1_runner_hooks.py H3
pins that selection to the runner function) into new checkpoint 0 and the read offset resets to 0.

Readers checked every step, base vs G1, bit for bit: the kernel's in-kernel pool gather (whole-state path at 8 rows; the
chunked path with the per-slot chunk gather and the served head-major update at 32 rows) through its residual and its
conv / rec checkpoint outputs; the stock-side region gather `gather_gdn_states` on the step's read ids; the seed copy's
source block; and, at the end, the bytes of every request's block at its read offset. A NaN read anywhere would
propagate into these. Mutants (must be caught): a commit at accept + 1 (the reader then meets a poisoned block), and a
reader that resumes from read offset + 1 (a reader of a non-accepted block).
Run: JAX_PLATFORMS=cpu PYTHONPATH=src:. pytest tests/q38_l1g1/test_g1_readers_poison.py
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
import pytest  # noqa: E402
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from vllm_torchtpu.megakernel import kernel  # noqa: E402
from vllm_torchtpu.megakernel.blocks.hc import RESIDUAL_OUT  # noqa: E402
from vllm_torchtpu.megakernel.exec_spec import ExecSpec, Tiles  # noqa: E402
from vllm_torchtpu.megakernel.models import qwen38  # noqa: E402
from vllm_torchtpu.megakernel.models.qwen38_inputs import pack_segment, random_segment_inputs  # noqa: E402
from vllm_torchtpu.megakernel.models.qwen38_pack import shard_step_inputs  # noqa: E402
from vllm_torchtpu.megakernel_runtime import qwen38_gdn_commit as gc  # noqa: E402
from vllm_torchtpu.megakernel_runtime import qwen38_gdn_state as gs  # noqa: E402
from vllm_torchtpu.megakernel_runtime.qwen38_scatter import scatter_gdn_ckpts_fast  # noqa: E402
from vllm_torchtpu.megakernel_runtime.qwen38_segment_call import gdn_blocks  # noqa: E402

T, R, KB, LANES = 4, 2, 128, 256
G = qwen38.TINY
TILES = Tiles(bank=1 << 20, nbd=10, nbe=8, ebank=512 << 10, vmem_mib=64)
L = qwen38.SEGMENT_LAYERS
NL = L - 1  # GDN layers of a body segment
HEAD_L = 2  # head group layers (all GDN), as the X-GLUE-1 head-call case
POISON = 0xFF


def _encode(raw: np.ndarray, m: int, ssm: np.ndarray, conv: np.ndarray) -> None:
    from tests.megakernel.test_qwen38_gdn_pool import _encode as enc

    enc(raw, m, ssm, conv)


class World:
    """Per-rank pools [R] of kb = 128 blocks (uint8), checkpoint groups ck [NL, NS, T], a block-indexed read-offset
    table (shared by all layers' state blocks, as the runner's)."""

    def __init__(self, NS: int, seed: int, nl: int = NL):
        rng = np.random.default_rng(seed)
        self.NS, self.NL = NS, nl
        NL = nl
        self.nmb = 1 + NL * NS * T + NL * NS + 1  # null, checkpoint groups, fresh blocks for crossings, spare
        self.ck = (1 + np.arange(NL * NS * T, dtype=np.int32)).reshape(NL, NS, T)
        self.fresh = 1 + NL * NS * T + np.arange(NL * NS, dtype=np.int32).reshape(NL, NS)
        self.off = np.zeros(self.nmb, np.int32)
        self.raws = []
        for _ in range(R):
            raw = rng.integers(0, 256, (self.nmb * 4, KB, 1, 4, LANES), dtype=np.uint8)
            for ln in range(NL):
                for s in range(NS):
                    _encode(raw, int(self.ck[ln, s, 0]), rng.standard_normal((6, 128, 128)).astype(np.float32) * 0.05,
                            rng.standard_normal((3, 1280)).astype(np.float32) * 0.05)
            self.raws.append(raw)

    def copy(self) -> World:
        w = object.__new__(World)
        w.NS, w.NL, w.nmb, w.ck, w.fresh, w.off = self.NS, self.NL, self.nmb, self.ck.copy(), self.fresh.copy(), self.off.copy()
        w.raws = [r.copy() for r in self.raws]
        return w

    @property
    def st(self) -> np.ndarray:
        return self.ck[:, :, 0].copy()  # the state block = checkpoint 0 of the group

    def meta(self, live: np.ndarray, read_off: np.ndarray | None = None) -> list:
        q = np.where(live, T, 0).astype(np.int32)
        qsl = np.concatenate([[0], np.cumsum(q)]).astype(np.int32)
        seq = np.full(self.NS, 100, np.int32)
        off = self.off if read_off is None else read_off
        out = []
        for ln in range(self.NL):
            out += [jnp.asarray(self.st[ln]), jnp.asarray(off), jnp.asarray(self.ck[ln]), jnp.asarray(qsl),
                    jnp.asarray(seq)]
        return out

    def layout(self):
        return gs.Qwen38GdnPoolLayout(pool_shape=self.raws[0].shape, pool_dtype="float8_e4m3fn")

    def pool(self, r: int):
        return jax.lax.bitcast_convert_type(jnp.asarray(self.raws[r]), jnp.float8_e4m3fn)

    def set_pool(self, r: int, p) -> None:
        self.raws[r] = np.asarray(jax.lax.bitcast_convert_type(p, jnp.uint8)).copy()

    def mgr(self, r: int) -> np.ndarray:
        return self.raws[r].reshape(self.nmb, -1)


class Kernel:
    """The pool-GDN segment kernel at B rows (built once; the same weights and hidden every step): a body segment
    (L layers, L - 1 GDN) or, `head=True`, the head group kernel (HEAD_L layers, all GDN; no QSA attention input)."""

    def __init__(self, B: int, chunk_gather: str, head_major: bool, seed: int, monkeypatch, head: bool = False):
        from vllm_torchtpu.megakernel.models import qwen38_gdn_pool

        monkeypatch.setattr(qwen38_gdn_pool, "CHUNK_GATHER", chunk_gather)
        tiles = TILES.with_blocks({"gdn": {"gdn_head_major": True}}) if head_major else TILES
        self.ex = ex = ExecSpec(R=R, B=B, T=T, ctx=128, interpret=True, tiles=tiles)
        Lk = HEAD_L if head else L
        self.NL = Lk if head else Lk - 1
        spec = qwen38.segment_spec(G, R, L=Lk, head=head, pool_gdn=True)
        inp = random_segment_inputs(jax.random.PRNGKey(seed), G, ex, Lk, gain=0.25, head=head)
        spec0 = qwen38.segment_spec(G, R, L=Lk, head=True) if head else spec  # head: packed as the X-GLUE-1 head case
        w_mk, _, attn = pack_segment(spec0, G, ex, inp, head=head)
        self.mesh = mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
        self.pos, self.hid, self.w, self.st = shard_step_inputs(mesh, ex, inp.pos, inp.x, w_mk, {})
        self.tp, self.rep = NamedSharding(mesh, P("tp")), NamedSharding(mesh, P())
        self.attn = None if head else jax.device_put(kernel.fold(attn), self.tp)
        self.names = [x.name for x in kernel.extra_inputs(spec, ex)]
        self.step, _ = kernel.make_step(mesh, spec, ex)
        self.B = B

    def __call__(self, w: World, rb, hi) -> dict:
        pool = jnp.concatenate([w.pool(r) for r in range(R)])
        xs = {"qsa_attn": self.attn, "gdn_pool": jax.device_put(pool, self.tp),
              "gdn_rb": jax.device_put(jnp.asarray(rb), self.rep),
              "gdn_hi": jax.device_put(jnp.asarray(hi).astype(jnp.int32), self.rep)}
        _, outs = self.step(self.pos, self.hid, self.w, self.st, *(xs[k] for k in self.names))
        conv = np.asarray(outs["conv_ckpt"])
        rec = np.asarray(outs["rec_ckpt"])
        nl = self.NL
        return dict(res=np.asarray(outs[RESIDUAL_OUT]).reshape(R, self.ex.M, -1)[:, : self.B],
                    conv=conv.reshape((R, nl) + conv.shape[1:]), rec=rec.reshape((R, nl) + rec.shape[1:]))


def _bits(a) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(a)).view(np.uint8)


def _gathered(w: World, rb, hi) -> list:
    out = []
    for r in range(R):
        c, x = gs.gather_gdn_states(w.pool(r), jnp.asarray(rb), layout=w.layout(), T=T, has_initial_state=jnp.asarray(hi),
                                    interpret=True)
        out.append((_bits(c), _bits(x)))
    return out


def _cross(w: World, s: int) -> list:
    """Request `s` crosses a block boundary (every layer): the runner's seed selection (the checkpoint at the read
    offset; new_owner = 0) -> new checkpoint 0 (= old checkpoint 1), the group slides, read offset of the new state block
    = 0. Returns the seed source blocks [NL]."""
    srcs = []
    for ln in range(w.NL):
        old = w.ck[ln, s].copy()
        src = int(old[min(max(int(w.off[old[0]]), 0), T - 1)])
        new = np.array([old[1], old[2], old[3], w.fresh[ln, s]], np.int32)
        for r in range(R):
            m = w.mgr(r)
            m[new[0]] = m[src]
        w.ck[ln, s] = new
        w.off[new[0]] = 0
        srcs.append(src)
    return srcs


def _poison(w: World, wb: np.ndarray, acc: np.ndarray) -> int:
    n = 0
    for ln in range(w.NL):
        for s in range(w.NS):
            for t in range(T):
                m = int(wb[ln, s, t])
                if m and t != acc[s]:
                    for r in range(R):
                        w.mgr(r)[m] = POISON
                    n += 1
    return n


def _run(monkeypatch, B: int, plan: list, *, chunk_gather="0", head_major=False, mutant: str | None = None, seed=3,
         head: bool = False):
    """`plan`: per step (live mask [NS] of 0/1, accept [NS] in 0..3, crossing request or None); `head`: the head group
    kernel instead of a body segment."""
    NS = B // T
    k = Kernel(B, chunk_gather, head_major, seed, monkeypatch, head=head)
    base = World(NS, seed + 1, k.NL)
    NL = base.NL
    g1 = base.copy()
    lay = base.layout()
    seen_acc = set()
    for step, (live, acc, cross) in enumerate(plan):
        live = np.asarray(live, bool)
        acc = np.asarray(acc, np.int32)
        if cross is not None:
            sb, sg = _cross(base, cross), _cross(g1, cross)
            assert sb == sg
            for r in range(R):
                for src in sb:
                    np.testing.assert_array_equal(g1.mgr(r)[src][: 400 * 1024], base.mgr(r)[src][: 400 * 1024],
                                                  err_msg=f"step {step}: seed source block {src} differs")
        rb_b, wb_b, hi_b = (np.asarray(x) for x in gdn_blocks(base.meta(live), NS, True))
        rb_g, wb_g, hi_g = (np.asarray(x) for x in gdn_blocks(g1.meta(live), NS, True))
        assert np.array_equal(rb_b, rb_g) and np.array_equal(wb_b, wb_g) and np.array_equal(hi_b, hi_g)
        if mutant == "reader_plus1" and step == len(plan) - 1:
            rb_g, _, _ = (np.asarray(x) for x in gdn_blocks(g1.meta(live, (g1.off + 1) % T), NS, True))
        gb, gg = _gathered(base, rb_b, hi_b), _gathered(g1, rb_g, hi_g)
        same = all(np.array_equal(x, y) for pb, pg in zip(gb, gg) for x, y in zip(pb, pg))
        assert same or mutant, f"step {step}: region gather (stock-side reader) differs"
        ob, og = k(base, rb_b, hi_b), k(g1, rb_g, hi_g)
        nres = int((_bits(ob["res"]) != _bits(og["res"])).sum())
        ncp = int((_bits(ob["conv"]) != _bits(og["conv"])).sum() + (_bits(ob["rec"]) != _bits(og["rec"])).sum())
        nnan = int(np.isnan(og["res"].astype(np.float32)).sum())
        print(f"Q38G1P {'head ' if head else ''}B{B} step {step}: live {int(live.sum())}/{NS} accept {acc.tolist()} cross {cross}; residual bytes "
              f"differing {nres}, checkpoint bytes differing {ncp}, NaN in G1 residual {nnan}", flush=True)
        assert nres == 0 and ncp == 0, f"B{B} step {step}: G1 differs from base (res {nres}, ckpt {ncp}, NaN {nnan})"
        # writes: base scatters all four; G1 updates the read offsets, commits the accepted one, poisons the rest
        for s in range(NS):
            if live[s]:
                seen_acc.add(int(acc[s]))
                for w in (base, g1):
                    w.off[w.st[:, s]] = acc[s]
        a_commit = (acc + 1) % T if mutant == "commit_plus1" else acc
        off_c = g1.off.copy()
        for s in range(NS):
            if live[s]:
                off_c[g1.st[:, s]] = a_commit[s]
        for r in range(R):
            base.set_pool(r, scatter_gdn_ckpts_fast(base.pool(r), jnp.asarray(ob["conv"][r]), jnp.asarray(ob["rec"][r]),
                                                    jnp.asarray(wb_b), layout=lay, interpret=True))
            g1.set_pool(r, gc.commit_from_offsets(g1.pool(r), jnp.asarray(og["conv"][r]), jnp.asarray(og["rec"][r]),
                                                  jnp.asarray(wb_g), jnp.asarray(g1.st), jnp.asarray(off_c), layout=lay,
                                                  interpret=True))
        npois = _poison(g1, wb_g, acc)
        assert npois == NL * (int(live.sum()) * (T - 1)), npois
    assert seen_acc == set(range(T)) or mutant, f"accept counts seen {sorted(seen_acc)}"
    # the accepted state of every request (its block at the read offset) is the same bytes in both worlds
    for ln in range(NL):
        for s in range(NS):
            m = int(base.ck[ln, s, base.off[base.st[ln, s]]])
            for r in range(R):
                np.testing.assert_array_equal(g1.mgr(r)[m][: 400 * 1024], base.mgr(r)[m][: 400 * 1024],
                                              err_msg=f"read-offset block {m} (layer {ln}, request {s})")


# 8 rows (whole-state path): 2 requests; request 1 dead at step 2; request 0 crosses at step 3
PLAN8 = [([1, 1], [0, 3], None), ([1, 1], [2, 1], None), ([1, 0], [3, 0], None), ([1, 1], [1, 2], 0),
         ([1, 1], [0, 0], None), ([1, 1], [3, 3], None)]
# 32 rows (chunked path, chunk gather, head-major update): 8 requests, 2 dead at step 1, a crossing at step 2
PLAN32 = [([1] * 8, [0, 1, 2, 3, 3, 2, 1, 0], None), ([1, 1, 0, 1, 1, 1, 0, 1], [3, 0, 0, 1, 2, 2, 0, 3], None),
          ([1] * 8, [1, 3, 2, 0, 0, 1, 3, 2], 5), ([1] * 8, [2, 2, 1, 3, 0, 3, 1, 0], None)]


def test_readers_bitwise_with_poison_8_rows(monkeypatch):
    _run(monkeypatch, 8, PLAN8)


def test_readers_bitwise_with_poison_32_rows_chunk_gather(monkeypatch):
    _run(monkeypatch, 32, PLAN32, chunk_gather="1", head_major=True)


@pytest.mark.parametrize("mutant", ["commit_plus1", "reader_plus1"])
def test_poison_catches_a_wrong_commit_or_reader(monkeypatch, mutant):
    plan = [([1, 1], [0, 2], None), ([1, 1], [1, 3], None), ([1, 1], [0, 0], None)]
    with pytest.raises(AssertionError):
        _run(monkeypatch, 8, plan, mutant=mutant, seed=11)


# the head group kernel (head=True, 2 GDN layers) at 8 rows: request 1 dead at step 1, request 1 crosses at step 2
PLANH = [([1, 1], [3, 1], None), ([1, 0], [0, 2], None), ([1, 1], [2, 0], 1), ([1, 1], [1, 3], None)]


def test_readers_bitwise_with_poison_head_8_rows(monkeypatch):
    """The real head-group kernel reading checkpoints the commit wrote (the head kind's path under G1)."""
    _run(monkeypatch, 8, PLANH, head=True)


@pytest.mark.parametrize("mutant", ["commit_plus1", "reader_plus1"])
def test_poison_catches_a_wrong_commit_or_reader_head(monkeypatch, mutant):
    plan = [([1, 1], [0, 2], None), ([1, 1], [1, 3], None), ([1, 1], [0, 0], None)]
    with pytest.raises(AssertionError):
        _run(monkeypatch, 8, plan, mutant=mutant, seed=11, head=True)
