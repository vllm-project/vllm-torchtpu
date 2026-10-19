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
"""Step time of the Qwen3.5-397B decode megakernel's MTP verify pass at serving shape (p3 port).

Needs one tpu7x 2x2x1 host (8 TensorCores, the kernel's 8 ranks) or, with P3_MK_R=16, a 2x2x2
slice of two hosts run as one SPMD JAX program (every host runs the same pytest; JAX distributed
starts from JAX_COORDINATOR_ADDRESS / JAX_NUM_PROCESSES / JAX_PROCESS_ID); skips elsewhere. The step is
the fused pool I/O verify bucket the serving adapter builds (EP, 68 expert slots, T=4 rows per
sequence, 60 layers) on synthetic weights and a paged KV / GatedDeltaNet pool. Each sequence has
a ~92k-token context. Routing is forced from recorded AgentX decode routing (PLACEMENTS: one
routing file per expert placement table, built by make_bench_routing.py), so each rank streams
as many experts per
layer as it does in serving. `r28` runs the serving split of 112 rows: an 80-row call, then a
32-row call on the same pools.

Results go to `MK_BENCH_OUT` (default /workspace/tmp/agentx-p2-e1/bench/<job name>): one JSON
line per (case, variant) with the per-step time of N back-to-back steps, and an xprof trace of a
few steps per case for device timing. Variants drop kernel parts with the build's ablation
switches (`skip`) or turn on in-kernel trace scopes.

`test_parity` compares this tree's kernel with a reference copy of the package
(`MK_BENCH_REF_SRC`, default the frozen S0 tree) on identical inputs: hidden output and every
pool page the step writes.

Phase 3 harness port (p3) of agentx-p2-e1 tests/kernels/test_qwen35_mk_step_bench.py:

  * collected only with P3_BENCH=1 in the environment, so the tree's regular pytest runs
    skip it; needs the serving image (jax.profiler segfaults on the unittest default image);
  * P3_MK_MODE selects the expert layout the megakernel supports: ep (routed experts EP over
    the 8 ranks, moe_tp=False, the serving D leg) or tp (experts TP-sliced over the
    intermediate dim, moe_tp=True); expert-TP inside EP groups and SP variants are not in
    this kernel (see coverage.csv);
  * P3_MK_ROUTING_DIR holds the forced routing files (routing_<placement>.npz);
  * results go to one run directory (p3_bench_common.run_dir, root P3_OUT_ROOT) with env.json
    (environment id with devices), results.jsonl and trace/<label>_<case>_<variant>/ for
    scripts/vllm/benchmark/p3_xplane_device_time.py; MK_BENCH_OUT overrides the directory;
  * MK_BENCH_CASES / MK_BENCH_VARIANTS (comma lists) restrict the parametrization;
  * P3_MK_R is the rank count (8, or 16 on two hosts: TP16 = EP16, 32 routed experts a rank)
    and P3_MK_PLACEMENT the placement the session's world uses (c32 at R=8, ep16_c0 at R=16,
    whose forced routing make_bench_routing_r.py builds from the same recording and draw).
"""

from __future__ import annotations

import dataclasses
import functools
import gc
import importlib
import json
import os
import sys
import time
import types

import pytest

if not os.environ.get("P3_BENCH"):
    pytest.skip("phase 3 harness bench; set P3_BENCH=1 to run", allow_module_level=True)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

from vllm_torchtpu.kernels.qwen35_megakernel import config as C  # noqa: E402
from vllm_torchtpu.kernels.qwen35_megakernel import pool_io  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "scripts", "vllm", "benchmark"))
import p3_bench_common as common  # noqa: E402

ROUTING_DIR = os.environ.get("P3_MK_ROUTING_DIR", "/workspace/tmp/agentx-p2-e1")
# placement -> (expert slots per rank, forced routing built for it by make_bench_routing.py)
PLACEMENTS = {
    "c32": (68, os.path.join(ROUTING_DIR, "routing_c32.npz")),
    "c64_mk2p": (72, os.path.join(ROUTING_DIR, "routing_c64_mk2p.npz")),
    "c112_mk2p": (78, os.path.join(ROUTING_DIR, "routing_c112_mk2p.npz")),
    # EP16 table expert_placement_ep16_agentx_corr_c0.json: 32 primaries a rank, no copies
    "ep16_c0": (32, os.environ.get(
        "P3_MK_ROUTING_EP16", "/workspace/tmp/p3r-kern-routing/routing_ep16_c0.npz")),
}
PLACEMENT = os.environ.get("P3_MK_PLACEMENT", "c32")
MODE = os.environ.get("P3_MK_MODE", "ep")  # ep: routed experts EP over the ranks; tp: TP-sliced
# decode context parallelism inside the kernel (MKConfig.dcp): the G = R / 2 ranks of a KV-head
# group hold DCP page-interleaved shards; the bench's page tables are rank-local either way
DCP = int(os.environ.get("MK_BENCH_DCP", "1"))
OUT = os.environ.get("MK_BENCH_OUT") or common.run_dir(
    "mk_step", MODE if DCP == 1 else f"{MODE}-dcp{DCP}"
)
REF_SRC = os.environ.get(
    "MK_BENCH_REF_SRC",
    "/workspace/tmp/agentx-p1/s0-tree/src/vllm_torchtpu/kernels/qwen35_megakernel",
)
N_STEPS = int(os.environ.get("MK_BENCH_STEPS", "20"))
if os.path.isdir("/workspace/tmp"):  # in the pod: share compiled kernels across tests and runs
    jax.config.update("jax_compilation_cache_dir", "/workspace/tmp/p3-harness/jax-cache")
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


def _subset(name, default):
    sel = os.environ.get(name)
    return [x for x in sel.split(",") if x] if sel else list(default)


R = int(os.environ.get("P3_MK_R", "8"))
L, T = 60, 4
E_SLOTS = PLACEMENTS[PLACEMENT][0]
CTX = 92160
CTX_SPREAD = 8192
MAXP = 2048  # max_pages_per_seq at max_model_len 262144
PSEQ = 785  # pages reserved per sequence: (CTX + CTX_SPREAD + T) / 128 rounded up
NSEQ_MAX = 28
GDN_STRIDE = 5  # vLLM block 640 tokens = 5 kernel pages
CKPT0 = -(-(1 + NSEQ_MAX * PSEQ) // GDN_STRIDE)  # first GDN checkpoint slot after the KV pages
NUM_PAGES = 24576
FP8 = C.FP8

# case -> the bucket calls of one step, each (rows, first sequence)
CASES = {
    "b4": ((4, 0),),
    "b8": ((8, 0),),
    "b16": ((16, 0),),
    "b32": ((32, 0),),
    "b48": ((48, 0),),
    "b64": ((64, 0),),
    "b80": ((80, 0),),
    "r28": ((80, 0), (32, 20)),
}
# variant -> (ablation skips, trace scope level, Tiles overrides, module constant patches
# {(module, name): value}); tile overrides and patches apply to every bucket of the case
VARIANTS = {
    "base": (frozenset(), 0, {}, {}),
    "no_ar": (frozenset({"ar"}), 0, {}, {}),
    "no_experts": (frozenset({"experts"}), 0, {}, {}),
    "no_kv": (frozenset({"kv"}), 0, {}, {}),
    "no_attn": (frozenset({"attn"}), 0, {}, {}),
    "no_gdn": (frozenset({"gdn"}), 0, {}, {}),
    "no_dense": (frozenset({"dense"}), 0, {}, {}),
    "fixed": (frozenset({"experts", "kv"}), 0, {}, {}),
    "fixed_no_ar": (frozenset({"experts", "kv", "ar"}), 0, {}, {}),
    "scopes1": (frozenset(), 1, {}, {}),
    "scopes2": (frozenset(), 2, {}, {}),
    "no_ar_scopes1": (frozenset({"ar"}), 1, {}, {}),
    "arpipe_all": (frozenset(), 0, {}, {("context", "ARPIPE_OFF_B"): ()}),
    "arpipe_all_scopes1": (frozenset(), 1, {}, {("context", "ARPIPE_OFF_B"): ()}),
    "hier0": (frozenset(), 0, {"hier": 0}, {}),
    "hier0_no_experts": (frozenset({"experts"}), 0, {"hier": 0}, {}),
    "hier1": (frozenset(), 0, {"hier": 1}, {}),
    "hier1_arpipe": (frozenset(), 0, {"hier": 1}, {("context", "ARPIPE_OFF_B"): ()}),
}


def _init_distributed():
    """Joins the hosts of a multi-host slice into one JAX program (no-op on one host)."""
    if jax.distributed.is_initialized():
        return
    addr = os.environ.get("JAX_COORDINATOR_ADDRESS", "")
    nproc = int(os.environ.get("JAX_NUM_PROCESSES", "1"))
    if addr and nproc > 1:
        jax.distributed.initialize(
            coordinator_address=addr,
            num_processes=nproc,
            process_id=int(os.environ.get("JAX_PROCESS_ID", "0")),
        )


# position of a chip (x, y) in torch_tpu's tp-rank order on a tpu7x 2x2x1 host, ids
# [0, 1, 4, 5, 6, 7, 2, 3] = chips (0,0) (0,1) (1,1) (1,0), two TensorCores each
_CHIP_RING = {(0, 0): 0, (0, 1): 1, (1, 1): 2, (1, 0): 3}


def _tpu_mesh():
    _init_distributed()
    devs = [d for d in jax.devices() if d.platform == "tpu"]
    if len(devs) < R:
        pytest.skip(f"needs {R} TPU devices ({R // 8} tpu7x 2x2x1 host(s) of one slice)")
    by_id = {d.id: d for d in devs}
    # torch_tpu's tp-rank order on a tpu7x 2x2x1 host (vllm_adapter._build_tp_rank_mesh): rank
    # pairs (2k, 2k + 1) are the two TensorCores of one chip, as the hierarchical all-reduce assumes
    order = [0, 1, 4, 5, 6, 7, 2, 3]
    if R == 8 and jax.process_count() == 1 and set(order) <= set(by_id):
        devs = [by_id[i] for i in order]
    else:
        # host-major ranks, each host in the 2x2x1 order above (the ranks of a TP16 engine over
        # two hosts); rank pairs (2k, 2k + 1) stay the two TensorCores of one chip
        def key(d):
            c = tuple(getattr(d, "coords", (0, 0, 0)))
            return (d.process_index, c[2] if len(c) > 2 else 0,
                    _CHIP_RING.get((c[0], c[1]), 9), getattr(d, "core_on_chip", 0))
        devs = sorted(devs, key=key)
    return Mesh(np.array(devs[:R]), ("tp",))


def _cfg(b: int, e_slots: int = E_SLOTS, dcp: int = DCP) -> C.MKConfig:
    # tp mode: the routed experts are TP-sliced (moe_tp=True), every rank holds all experts,
    # and e_slots / forced routing counts do not apply (the placement table is EP specific).
    return dataclasses.replace(
        C.FULL_EP,
        R=R,
        moe_tp=MODE == "tp",
        stack_e=False,
        e_slots=e_slots if MODE == "ep" else 0,
        paged_kv=True,
        fused_pool_io=True,
        B=b,
        T=T,
        num_pages=NUM_PAGES,
        max_pages_per_seq=MAXP,
        gdn_stride_pages=GDN_STRIDE,
        ctx=MAXP * 128,
        dcp=dcp,
    )


@functools.lru_cache(maxsize=None)
def _filler(mesh, shape, dtype, kind):
    sh = NamedSharding(mesh, P("tp"))

    @functools.partial(jax.jit, out_shardings=sh)
    def f(key):  # a host numpy uint32[2] key: the same input on every host
        if kind == "bytes":  # |value| < 2 fp8 (no NaN / Inf), and bf16 / f32 with exponent <= 127
            b = jax.random.bits(key, shape, jnp.uint8)
            b = (b & 0x3F) | (b & 0x80)
            return jax.lax.bitcast_convert_type(b, dtype)
        if kind == "normal":
            return (jax.random.normal(key, shape, jnp.float32) * 0.05).astype(dtype)
        return jax.random.uniform(key, shape, jnp.float32, 0.005, 0.015).astype(dtype)

    return f


def _global(mesh, x):
    """Host array x, split on its first axis over the ranks (every host passes the same x)."""
    return jax.make_array_from_callback(x.shape, NamedSharding(mesh, P("tp")), lambda i: x[i])


def _replicated(mesh, x):
    x = np.asarray(x)
    return jax.make_array_from_callback(x.shape, NamedSharding(mesh, P()), lambda i: x[i])


def _key(*fold):
    k = jax.random.PRNGKey(fold[0])
    for f in fold[1:]:
        k = jax.random.fold_in(k, f)
    return np.asarray(k)


def _local_rows(arr, rows):
    """float32 numpy rows [:rows] of every addressable rank shard of a P("tp") array."""
    return [np.asarray(s.data[:rows].astype(jnp.float32)) for s in arr.addressable_shards]


class _World:
    """Weights, pools and per-case inputs shared by every test in the session."""

    def __init__(self, placement: str = PLACEMENT):
        self.mesh = _tpu_mesh()
        os.makedirs(OUT, exist_ok=True)
        self.placement = placement
        self.e_slots, routing = PLACEMENTS[placement]
        self.routing = np.load(routing)
        c = _cfg(32, self.e_slots)
        tiles = C.default_tiles(c)
        shapes = C.weight_shapes(c, tiles)
        lay, s_width = C.smalls_layout(c)
        w = {}
        for i, k in enumerate(C.W_NAMES):
            spec = shapes[k]
            if k in ("e13", "e2"):
                w[k] = tuple(
                    _filler(self.mesh, (R, *s.shape), s.dtype, "bytes")(_key(7, 1000 * i + l))
                    for l, s in enumerate(spec)
                )
                continue
            shape, dt = spec
            if k == "smalls":
                w[k] = _global(self.mesh, self._smalls(lay, s_width, shape, dt))
            elif k == "final_norm":
                w[k] = _global(self.mesh, np.zeros((R, *shape), np.float32))
            else:
                kind = "bytes" if jnp.dtype(dt) == jnp.dtype(FP8) else (
                    "normal" if jnp.dtype(dt) == jnp.dtype(C.BF16) else "scale")
                w[k] = _filler(self.mesh, (R, *shape), dt, kind)(_key(7, i))
        self.weights = w
        self.pool_shape = (R * NUM_PAGES, 2, 64, 4, 128)
        self.pools = self.fresh_pools()
        jax.block_until_ready((self.weights, self.pools))
        rng = np.random.default_rng(11)
        self.ctx = CTX + rng.integers(-CTX_SPREAD, CTX_SPREAD + 1, NSEQ_MAX)
        self.hidden_rows = rng.standard_normal((128, c.H)).astype(np.float32)

    def fresh_pools(self, seed: int = 99):
        f = _filler(self.mesh, self.pool_shape, FP8, "bytes")
        return tuple(f(_key(seed, a)) for a in range(C.FULL.NATT))

    def _smalls(self, lay, s_width, shape, dt):
        rng = np.random.default_rng(3)
        sm = rng.uniform(0.0, 0.1, (L, 1, s_width)).astype(np.float32)
        for name in ("in_s", "out_s", "sh13_s", "sh2_s"):
            o, n = lay[name]
            sm[:, :, o:o + n] = 0.01
        o, _ = lay["alog"]
        sm[:, :, o:o + 128] = -1.0
        if "ecopy" in lay:
            o, n = lay["ecopy"]
            sm[:, 0, o:o + n] = self.routing["ecopy"]
        assert sm.shape == tuple(shape), (sm.shape, shape)
        return np.broadcast_to(sm[None], (R, *shape)).astype(dt)

    def inputs(self, b: int, s0: int):
        """pos, hidden, page_table, ckpt_indices, read_offsets, rbias of a b-row call on
        sequences s0 .. s0 + b / T - 1."""
        c = _cfg(b)
        ns = b // T
        pos = np.zeros((b,), np.int32)
        pt = np.zeros((ns, MAXP), np.int32)
        ckpt = np.zeros((3, ns, T), np.int32)
        for s in range(ns):
            g = s0 + s
            n_ctx = int(self.ctx[g])
            pos[s * T:(s + 1) * T] = n_ctx + np.arange(T)
            npg = -(-(n_ctx + T) // 128)
            pt[s, :npg] = 1 + g * PSEQ + np.arange(npg)
            for sub in range(3):
                ckpt[sub, s] = CKPT0 + ((sub * NSEQ_MAX + g) * T) + np.arange(T)
        roff = np.zeros((3, ns), np.int32)
        hid = np.zeros((c.M, c.H), np.float32)
        hid[:b] = self.hidden_rows[:b]
        picks = self.routing[f"b{b}"]  # [L, b, 10] virtual expert ids
        rb = np.full((L, c.M, c.E), 1e-3, np.float32)
        rb += np.arange(c.E, dtype=np.float32)[None, None, :] * 1e-6
        for k in range(picks.shape[-1]):
            np.put_along_axis(
                rb[:, :b], picks[:, :, k:k + 1].astype(np.int64), 2.0 - 0.01 * k, axis=-1
            )
        return (
            _replicated(self.mesh, pos),
            _global(self.mesh, np.tile(hid, (R, 1)).astype(jnp.bfloat16)),
            _replicated(self.mesh, pt),
            _replicated(self.mesh, ckpt),
            _replicated(self.mesh, roff),
            _global(self.mesh, np.tile(rb, (R, 1, 1))),
        )

    def step_fn(self, mod, b: int, skip, scopes: int, tile_over=None, patch=None, dcp=DCP):
        c = _cfg(b, self.e_slots, dcp)
        tiles = mod.config.default_tiles(c)
        if tile_over:
            tiles = dataclasses.replace(tiles, **tile_over)
        saved = {}
        for (m, name), v in (patch or {}).items():
            obj = getattr(mod, m)
            saved[(m, name)] = getattr(obj, name)
            setattr(obj, name, v)
        try:
            fn = mod.pool_io.make_pooled_step(
                self.mesh,
                c,
                tiles,
                forced_routing=True,
                skip=skip,
                scopes=scopes,
                fused_pool_io=True,
            )
        finally:
            for (m, name), v in saved.items():
                setattr(getattr(mod, m), name, v)
        return jax.jit(fn, donate_argnums=(3,)), tiles


_WORLD = []


def _get_world(placement: str = PLACEMENT) -> _World:
    """The process's one _World, rebuilt when another placement is asked for (HBM holds one)."""
    if _WORLD and _WORLD[0].placement != placement:
        old = _WORLD.pop()
        old.weights = old.pools = None
        del old
        gc.collect()
    if not _WORLD:
        _WORLD.append(_World(placement))
    return _WORLD[0]


@pytest.fixture(scope="session")
def world():
    # one instance per process: the trace module re-exports this fixture under its own id
    return _get_world()


_THIS = types.SimpleNamespace(
    config=C, pool_io=pool_io, context=importlib.import_module(C.__name__.rsplit(".", 1)[0] + ".context")
)


def _ref_package():
    """The reference megakernel package loaded from REF_SRC under its own module name."""
    name = "qwen35_megakernel_ref"
    if name not in sys.modules:
        if not os.path.isdir(REF_SRC):
            pytest.skip(f"reference tree {REF_SRC} is not mounted")
        pkg = types.ModuleType(name)
        pkg.__path__ = [REF_SRC]
        sys.modules[name] = pkg
    return types.SimpleNamespace(
        config=importlib.import_module(f"{name}.config"),
        pool_io=importlib.import_module(f"{name}.pool_io"),
        context=importlib.import_module(f"{name}.context"),
    )


_ENV = {}


def _record(row):
    lead = jax.process_index() == 0  # one writer on a multi-host slice
    if not _ENV:
        os.makedirs(OUT, exist_ok=True)
        if lead:
            _ENV.update(common.write_env(OUT, {"bench": "mk_step", "mode": MODE, "R": R},
                                         devices=True))
        else:
            _ENV["id"] = f"process{jax.process_index()}"
    row = {"environment_id": _ENV["id"], "mode": MODE, "R": R, "dcp": DCP,
           "process": jax.process_index(), "process_count": jax.process_count(), **row}
    if lead:
        with open(os.path.join(OUT, "results.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")
    print("MK_BENCH " + json.dumps(row), flush=True)


def _run_case(world, mod, case, variant, label, n_steps=N_STEPS, trace=True):
    skip, scopes, tile_over, patch = VARIANTS[variant]
    calls = []
    t_build = time.time()
    for b, s0 in CASES[case]:
        fn, tiles = world.step_fn(mod, b, skip, scopes, tile_over, patch)
        calls.append((fn, world.inputs(b, s0), b, tiles))
    pools = world.pools

    def one(pools):
        hid = None
        for fn, (pos, hid_in, pt, ck, ro, rb), _, _ in calls:
            hid, pools = fn(pos, hid_in, world.weights, pools, pt, ck, ro, rb)
        return hid, pools

    hid, pools = one(pools)  # compile
    jax.block_until_ready(pools)
    t_compile = time.time() - t_build
    for _ in range(2):
        hid, pools = one(pools)
    jax.block_until_ready(pools)
    single = []
    for _ in range(5):
        t0 = time.perf_counter()
        hid, pools = one(pools)
        jax.block_until_ready(pools)
        single.append((time.perf_counter() - t0) * 1e3)
    t0 = time.perf_counter()
    for _ in range(n_steps):
        hid, pools = one(pools)
    jax.block_until_ready(pools)
    loop_ms = (time.perf_counter() - t0) * 1e3 / n_steps
    trace_dir = None
    if trace:
        proc = f"_p{jax.process_index()}" if jax.process_count() > 1 else ""
        trace_dir = os.path.join(OUT, "trace", f"{label}_{case}_{variant}{proc}")
        jax.profiler.start_trace(trace_dir)
        for _ in range(5):
            hid, pools = one(pools)
        jax.block_until_ready(pools)
        jax.profiler.stop_trace()
    world.pools = pools
    h = np.stack(_local_rows(hid, calls[-1][2]))
    row = {
        "label": label,
        "placement": world.placement,
        "case": case,
        "variant": variant,
        "rows": [b for _, _, b, _ in calls],
        "loop_ms": round(loop_ms, 4),  # host wall per step over N back-to-back steps (ms-scale steps)
        "single_ms": [round(x, 4) for x in sorted(single)],
        "n_steps": n_steps,
        "compile_s": round(t_compile, 1),
        "hidden_finite": bool(np.isfinite(h).all()),
        "tiles": [dataclasses.asdict(t) for _, _, _, t in calls],
        "expert_count_mean": [
            float(world.routing[f"count_b{b}"].mean()) for _, _, b, _ in calls
        ],
        "expert_count_max": [
            float(world.routing[f"count_b{b}"].max(axis=1).mean()) for _, _, b, _ in calls
        ],
        "trace": trace_dir,
    }
    _record(row)
    return row


def test_devices(world):
    devs = [
        {"id": d.id, "coords": list(getattr(d, "coords", ())),
         "core": getattr(d, "core_on_chip", None), "process": d.process_index}
        for d in world.mesh.devices.flat
    ]
    _record({"label": "devices", "mesh": devs, "jax": jax.__version__})


@pytest.mark.parametrize("case", _subset("MK_BENCH_CASES", CASES))
@pytest.mark.parametrize("variant", _subset("MK_BENCH_VARIANTS", VARIANTS))
def test_step(world, case, variant):
    _run_case(world, _THIS, case, variant, "new")


@pytest.mark.parametrize("case", ["b32", "r28"])
@pytest.mark.parametrize("placement", _subset("MK_BENCH_PLACEMENTS", PLACEMENTS))
def test_step_placement(placement, case):
    if MODE != "ep":
        pytest.skip("placement tables are EP specific")
    _run_case(_get_world(placement), _THIS, case, "base", "new")


@pytest.mark.parametrize("case", list(CASES))
def test_step_ref(world, case):
    _run_case(world, _ref_package(), case, "base", "ref")


def _written_pages(world, case):
    """Pool pages a step of `case` writes: (the new tokens' KV pages, the first page of every
    GatedDeltaNet checkpoint slot)."""
    kv, gdn = set(), []
    for b, s0 in CASES[case]:
        for s in range(b // T):
            g = s0 + s
            n_ctx = int(world.ctx[g])
            for p in range(n_ctx // 128, -(-(n_ctx + T) // 128)):
                kv.add(1 + g * PSEQ + p)
            for sub in range(3):
                for t in range(T):
                    gdn.append((CKPT0 + (sub * NSEQ_MAX + g) * T + t) * GDN_STRIDE)
    return np.array(sorted(kv)), np.array(gdn)


@jax.jit
def _shard_pages(shard, idx):
    return jax.lax.bitcast_convert_type(shard[idx], jnp.uint8)


def _take_pages(pool, idx):
    """uint8 bytes of pool pages idx (any index shape) on every rank this process addresses,
    [ranks, *idx.shape, 2, 64, 4, 128] in rank order, without a full-pool copy."""
    shards = sorted(pool.addressable_shards, key=lambda s: s.index[0].start or 0)
    return np.stack([np.asarray(_shard_pages(s.data, idx)) for s in shards])


def _unpack_rec(raw_u8, vh, kd=128, vd=128):
    """(n, rec_pages, 2, 64, 4, 128) uint8 checkpoint pages -> (n, vh, kd, vd) float32."""
    n, rec_pages = raw_u8.shape[:2]
    rows = (vh * kd) // 2
    u32 = (
        np.swapaxes(raw_u8, -2, -1).copy().view(np.uint32)
        .reshape(n, rec_pages * 128, 128)[:, :rows, :]
    )
    bf = np.swapaxes(u32[..., None].view(jnp.dtype(C.BF16)), -2, -1)
    return bf.reshape(n, vh, kd, vd).astype(np.float32)


def _unpack_conv(raw_u8, qkv):
    """(n, 2, 64, 4, 128) uint8 conv page -> (n, 3, qkv) float32."""
    rows = (3 * qkv) // 128
    bf = np.swapaxes(
        np.swapaxes(raw_u8, -2, -1).copy().view(jnp.dtype(C.BF16)), -2, -1
    ).reshape(-1, 256, 128)
    return bf[:, :rows, :].reshape(-1, 3, qkv).astype(np.float32)


def _parity_run(world, case, mod, tile_over=None):
    """Hidden output of every call of `case`, the KV pages of the new tokens and the
    GatedDeltaNet checkpoints after one step of `mod` from fresh pools: (hids, kv [NATT, ranks,
    n, 2, 64, 4, 128], gdn [NATT, ranks, n, gdn_pages, ...]). tile_over is a Tiles override dict,
    or a function of the bucket's row count returning one."""
    c = _cfg(32)
    kv_pages, gdn_pages = _written_pages(world, case)
    pools = world.fresh_pools(seed=5)
    hids = []
    for b, s0 in CASES[case]:
        over = tile_over(b) if callable(tile_over) else tile_over
        fn, _ = world.step_fn(mod, b, frozenset(), 0, over)
        pos, hid_in, pt, ck, ro, rb = world.inputs(b, s0)
        hid, pools = fn(pos, hid_in, world.weights, pools, pt, ck, ro, rb)
        hids.append(np.stack(_local_rows(hid, b)))
    g_idx = gdn_pages[:, None] + np.arange(c.gdn_pages)[None, :]
    kv = np.stack([_take_pages(p, kv_pages) for p in pools])
    gdn = np.stack([_take_pages(p, g_idx) for p in pools])
    del pools
    return hids, kv, gdn


def _kvf(a):
    return a.view(jnp.dtype(FP8)).astype(np.float32)


def _rec(a):
    c = _cfg(32)
    return _unpack_rec(a[..., : c.rec_pages, :, :, :, :].reshape(-1, c.rec_pages, 2, 64, 4, 128), c.vh)


def _conv(a):
    c = _cfg(32)
    return _unpack_conv(a[..., c.rec_pages, :, :, :, :].reshape(-1, 2, 64, 4, 128), c.QKV)


@pytest.mark.parametrize("case", ["b4", "b32", "r28"])
def test_parity(world, case):
    """This tree's kernel against the reference package on identical inputs: the hidden
    output of every call, the KV pages of the new tokens and the GatedDeltaNet checkpoints.
    The reference runs each bucket with this tree's all-reduce `hier`: the all-reduce algorithm
    fixes the order of the bf16 roundings of the partial sums, so two algorithms differ by
    rounding that compounds over 60 layers (test_parity_ab measures that against the f32
    one-shot all-reduce). On a multi-host slice every process checks the ranks it addresses."""
    ref = _ref_package()
    world.pools = None  # one pool set in HBM at a time (weights take 49 of 94 GiB)
    same_hier = lambda b: {"hier": C.default_tiles(_cfg(b, world.e_slots)).hier}
    outs = {"ref": _parity_run(world, case, ref, same_hier), "new": _parity_run(world, case, _THIS)}
    world.pools = world.fresh_pools()
    (h_ref, kv_ref, g_ref), (h_new, kv_new, g_new) = outs["ref"], outs["new"]
    kv_pages, _ = _written_pages(world, case)
    row = {
        "label": "parity",
        "case": case,
        "hidden_max_abs_diff": [float(np.abs(a - b).max()) for a, b in zip(h_new, h_ref)],
        "hidden_ref_max_abs": [float(np.abs(b).max()) for b in h_ref],
        "hidden_identical_frac": [float(np.mean(a == b)) for a, b in zip(h_new, h_ref)],
        "kv_bytes_identical_frac": float(np.mean(kv_new == kv_ref)),
        "gdn_bytes_identical_frac": float(np.mean(g_new == g_ref)),
        "kv_max_abs_diff": float(np.abs(_kvf(kv_new) - _kvf(kv_ref)).max()),
        "rec_max_abs_diff": float(np.abs(_rec(g_new) - _rec(g_ref)).max()),
        "conv_max_abs_diff": float(np.abs(_conv(g_new) - _conv(g_ref)).max()),
        "n_kv_pages": int(len(kv_pages)),
        "n_gdn_slots": int(g_ref.shape[2]),
    }
    _record(row)
    for a, b in zip(h_new, h_ref):
        assert np.isfinite(a).all() and np.isfinite(b).all()
        np.testing.assert_allclose(a, b, atol=2e-2, rtol=2e-2)
    np.testing.assert_allclose(_kvf(kv_new), _kvf(kv_ref), atol=0.25, rtol=0.13)
    np.testing.assert_allclose(_rec(g_new), _rec(g_ref), atol=2e-2, rtol=2e-2)
    np.testing.assert_allclose(_conv(g_new), _conv(g_ref), atol=2e-2, rtol=2e-2)


# All-reduce variants for test_parity_ab: (package, Tiles overrides). Every variant rounds each
# rank's partial to bf16 before sending; after that hier=3 (log2(R) butterfly) rounds the running
# sum to bf16 after each of its first log2(R) - 1 rounds, hier=0 with rsag once (the chunk sums),
# and hier=0 without rsag (one-shot push) not at all, so "f32" is the most exact all-reduce the
# kernel has. nbd only sizes the dense-weight ring (room in VMEM for the one-shot's R receive
# buffers at 32 rows).
AR_VARIANTS = {
    "def": ("new", {}),
    "h3": ("new", {"hier": 3}),
    "nbd6": ("new", {"nbd": 6}),
    "ref": ("ref", {}),
    "ref_h0": ("ref", {"hier": 0}),
    "f32": ("new", {"hier": 0, "rsag": False}),
    "f32_nbd6": ("new", {"hier": 0, "rsag": False, "nbd": 6}),
}
# "def" is this tree's default tiles (hier=0 at R=16, hier=2 at R=8), "ref" the reference
# package's (hier=3 up to 32 rows at R=16)
PARITY_AB = {
    "def_vs_ref_h0": ("def", "ref_h0"),  # same algorithm, two packages
    "h3_vs_ref": ("h3", "ref"),
    "def_vs_ref": ("def", "ref"),  # test_parity's comparison, without its bands
    "def_vs_nbd6": ("def", "nbd6"),
    "def_vs_f32": ("def", "f32"),
    "h3_vs_f32": ("h3", "f32"),
    "def_vs_f32_nbd6": ("def", "f32_nbd6"),
    "h3_vs_f32_nbd6": ("h3", "f32_nbd6"),
}


def _diff_stats(a, b):
    d = np.abs(a - b)
    return {
        "max_abs_diff": float(d.max()),
        "mean_abs_diff": float(d.mean()),
        "p999_abs_diff": float(np.quantile(d, 0.999)),
        "ref_max_abs": float(np.abs(b).max()),
        "ref_mean_abs": float(np.abs(b).mean()),
        "identical_frac": float(np.mean(a == b)),
        "over_2e-2_band_frac": float(np.mean(d > 2e-2 + 2e-2 * np.abs(b))),
    }


@pytest.mark.parametrize("case", ["b4", "b8", "b32"])
@pytest.mark.parametrize("pair", _subset("MK_BENCH_PARITY_AB", PARITY_AB))
def test_parity_ab(world, pair, case):
    """Numerics of two all-reduce variants on identical inputs, per pool (pool a holds the KV of
    full-attention layer 4a + 3 and the checkpoints of GatedDeltaNet layers 4a .. 4a + 2, so the
    per-pool lists are the error by depth) and for the hidden output. Records only; asserts that
    every output is finite."""
    mods = {"new": _THIS}
    out = {}
    world.pools = None
    for side in PARITY_AB[pair]:
        pkg, over = AR_VARIANTS[side]
        if pkg not in mods:
            mods[pkg] = _ref_package()
        try:
            out[side] = _parity_run(world, case, mods[pkg], over)
        except Exception as e:  # noqa: BLE001 - a variant that does not build is a result too
            world.pools = world.fresh_pools()
            _record({"label": "parity_ab", "pair": pair, "case": case, "error": f"{side}: {type(e).__name__}: {e}"[:2000]})
            pytest.skip(f"{side} does not build: {type(e).__name__}")
    world.pools = world.fresh_pools()
    (h_a, kv_a, g_a), (h_b, kv_b, g_b) = (out[s] for s in PARITY_AB[pair])
    natt = kv_a.shape[0]
    rec = [(_rec(g_a[i]), _rec(g_b[i])) for i in range(natt)]
    row = {
        "label": "parity_ab",
        "pair": pair,
        "sides": list(PARITY_AB[pair]),
        "tiles": {s: AR_VARIANTS[s][1] for s in PARITY_AB[pair]},
        "case": case,
        "hidden": [_diff_stats(a, b) for a, b in zip(h_a, h_b)],
        "kv_bytes_identical_frac_by_pool": [float(np.mean(kv_a[i] == kv_b[i])) for i in range(natt)],
        "kv_max_abs_diff_by_pool": [float(np.abs(_kvf(kv_a[i]) - _kvf(kv_b[i])).max()) for i in range(natt)],
        "kv_mean_abs_diff_by_pool": [float(np.abs(_kvf(kv_a[i]) - _kvf(kv_b[i])).mean()) for i in range(natt)],
        "kv_ref_mean_abs_by_pool": [float(np.abs(_kvf(kv_b[i])).mean()) for i in range(natt)],
        "rec_max_abs_diff_by_pool": [float(np.abs(ra - rb).max()) for ra, rb in rec],
        "rec_mean_abs_diff_by_pool": [float(np.abs(ra - rb).mean()) for ra, rb in rec],
        "rec_ref_mean_abs_by_pool": [float(np.abs(rb).mean()) for _, rb in rec],
        "conv_max_abs_diff_by_pool": [float(np.abs(_conv(g_a[i]) - _conv(g_b[i])).max()) for i in range(natt)],
        "gdn_bytes_identical_frac_by_pool": [float(np.mean(g_a[i] == g_b[i])) for i in range(natt)],
    }
    _record(row)
    for h in h_a + h_b:
        assert np.isfinite(h).all()


# ---------------------------------------------------------------------------------------------
# DCP parity: this tree at dcp = DCP against this tree at dcp 1 on the same logical KV
# ---------------------------------------------------------------------------------------------


def _kv_page_bytes(a: int, g: int, pages: np.ndarray) -> np.ndarray:
    """Deterministic fp8 bytes (|x| < 2, no NaN) of global KV pages `pages` of sequence g in
    pool a: [len(pages), 2, 64, 4, 128] uint8, the same on every rank of a KV-head group."""
    n = 2 * 64 * 4 * 128
    key = (np.asarray(pages, np.uint32) + np.uint32(g * PSEQ + a * NSEQ_MAX * PSEQ)).astype(np.uint32)
    x = (key[:, None] * np.uint32(2654435761)) + (np.arange(n, dtype=np.uint32) * np.uint32(40503))
    x = (x ^ (x >> np.uint32(13))) * np.uint32(2246822519)
    b = ((x >> np.uint32(24)) & np.uint32(0x3F)) | ((x >> np.uint32(8)) & np.uint32(0x80))
    return b.astype(np.uint8).reshape(len(pages), 2, 64, 4, 128)


def _set_pages_per_rank(pool, idx, vals):
    """pool[r * NUM_PAGES + idx[r]] = vals[r] on every rank, in place (donated)."""
    mesh = _WORLD[0].mesh
    fn = jax.jit(
        jax.shard_map(
            lambda p, i, v: p.at[i[0]].set(v[0]),
            mesh=mesh,
            in_specs=(P("tp"), P("tp"), P("tp")),
            out_specs=P("tp"),
            check_vma=False,
        ),
        donate_argnums=(0,),
    )
    return fn(pool, _global(mesh, idx[:, None]), _global(mesh, vals[:, None]))


def _dcp_parity_pools(world, case, dcp: int, seed: int = 5):
    """Fresh pools whose KV pages of `case`'s sequences are group-consistent at dcp 1 (every rank
    of a KV-head group holds the same page bytes) and, at dcp > 1, re-laid out so that rank r's
    local column j of sequence g holds global page dcp * j + r % dcp. GatedDeltaNet pages keep
    the per-rank seed content, the same in both layouts."""
    G = _cfg(32).G
    seqs = sorted({s0 + s for b, s0 in CASES[case] for s in range(b // T)})
    pools = list(world.fresh_pools(seed=seed))
    cols, vals = [], []
    for r in range(R):
        dr = r % dcp
        c_r, v_r = [], []
        for g in seqs:
            npg = -(-(int(world.ctx[g]) + T) // 128)
            if dcp == 1:
                gp = np.arange(npg)
                col = gp
            else:
                col = np.arange(-(-npg // dcp))
                gp = dcp * col + dr
                keep = gp < npg
                col, gp = col[keep], gp[keep]
            c_r.append(1 + g * PSEQ + col)
            v_r.append((g, gp))
        cols.append(np.concatenate(c_r))
        vals.append(v_r)
    n = min(len(c) for c in cols)
    idx = np.stack([c[:n] for c in cols]).astype(np.int32)
    for a in range(len(pools)):
        va = np.stack([
            np.concatenate([_kv_page_bytes(a, g, gp) for g, gp in vals[r]])[:n] for r in range(R)
        ])
        pools[a] = _set_pages_per_rank(
            pools[a], idx, jax.lax.bitcast_convert_type(jnp.asarray(va), FP8)
        )
    jax.block_until_ready(pools)
    return tuple(pools)


def _new_token_lanes(world, case, dcp: int, pools):
    """[(rank, page, lane)] -> bytes of every new token's KV column on the ranks holding it."""
    out = {}
    for b, s0 in CASES[case]:
        for s in range(b // T):
            g = s0 + s
            for t in range(T):
                p = int(world.ctx[g]) + t
                gp, ln = p // 128, p % 128
                for r in range(R):
                    if r % dcp != gp % dcp:
                        continue
                    out[(r, g, p)] = (1 + g * PSEQ + gp // dcp, ln)
    pages = np.array(sorted({v[0] for v in out.values()}))
    taken = [_take_pages(pool, pages) for pool in pools]  # [ranks, pages, 2, 64, 4, 128]
    res = {}
    for (r, g, p), (page, ln) in out.items():
        j = int(np.searchsorted(pages, page))
        res[(r, g, p)] = np.stack([tk[r, j, :, :, :, ln] for tk in taken])
    return res


def _bf16_ulp(x):
    """One bf16 ulp at |x| (7 fraction bits)."""
    a = max(float(x), 2.0 ** -100)
    return float(np.exp2(np.floor(np.log2(a)) - 7))


def _state_stats(a, b):
    """GatedDeltaNet state a against reference b: the relative mean difference and the largest
    difference in bf16 ulps of the reference's largest magnitude. The states are sums rounded
    to bf16, so two reduction or merge orders differ by absolute noise at the scale of the
    largest entries (one bf16 step at most, item1), not per element."""
    d = np.abs(a - b)
    ref_max = float(np.abs(b).max())
    return {
        "max_abs_diff": float(d.max()),
        "ref_max_abs": ref_max,
        "max_diff_ulps_of_ref_max": float(d.max() / _bf16_ulp(ref_max)),
        "rel_mean_diff": float(d.mean() / max(float(np.abs(b).mean()), 1e-30)),
    }


DCP_PARITY_TILES = {
    "def": {},
    "f32": {"hier": 0, "rsag": False},
}


@pytest.mark.parametrize("case", ["b4", "b32"])
@pytest.mark.parametrize("ar", _subset("MK_BENCH_DCP_PARITY_AR", DCP_PARITY_TILES))
def test_parity_dcp(world, case, ar):
    """dcp = DCP against dcp 1 (this tree, same tiles, same forced routing, same copy slots) on
    the same logical KV: the hidden output of every call, the GatedDeltaNet checkpoints and the
    new tokens' KV columns on the ranks that hold them. `ar` picks the all-reduce (def: the
    tree's tiles; f32: the one-shot push, the most exact all-reduce the kernel has)."""
    if DCP == 1:
        pytest.skip("MK_BENCH_DCP=1: nothing to compare")
    over = DCP_PARITY_TILES[ar]
    world.pools = None
    outs = {}
    for dcp in (1, DCP):
        c = _cfg(32, world.e_slots, dcp)
        _, gdn_pages = _written_pages(world, case)
        pools = _dcp_parity_pools(world, case, dcp)
        hids = []
        for b, s0 in CASES[case]:
            fn, _ = world.step_fn(_THIS, b, frozenset(), 0, over or None, dcp=dcp)
            pos, hid_in, pt, ck, ro, rb = world.inputs(b, s0)
            hid, pools = fn(pos, hid_in, world.weights, pools, pt, ck, ro, rb)
            hids.append(np.stack(_local_rows(hid, b)))
        g_idx = gdn_pages[:, None] + np.arange(c.gdn_pages)[None, :]
        gdn = np.stack([_take_pages(p, g_idx) for p in pools])
        lanes = _new_token_lanes(world, case, dcp, pools)
        outs[dcp] = (hids, gdn, lanes)
        del pools
    world.pools = world.fresh_pools()
    (h1, g1, l1), (hd, gd, ld) = outs[1], outs[DCP]
    kv1 = np.stack([l1[k] for k in sorted(ld)])
    kvd = np.stack([ld[k] for k in sorted(ld)])
    hid_stats = [_diff_stats(a, b) for a, b in zip(hd, h1)]
    rec_stats, conv_stats = _state_stats(_rec(gd), _rec(g1)), _state_stats(_conv(gd), _conv(g1))
    row = {
        "label": "parity_dcp",
        "case": case,
        "dcp": DCP,
        "ar": ar,
        "hidden": hid_stats,
        "kv_bytes_identical_frac": float(np.mean(kvd == kv1)),
        "kv_max_abs_diff": float(np.abs(_kvf(kvd) - _kvf(kv1)).max()),
        "gdn_bytes_identical_frac": float(np.mean(gd == g1)),
        "rec": rec_stats,
        "conv": conv_stats,
        "n_new_token_columns": int(len(ld)),
    }
    _record(row)
    for a, b in zip(hd, h1):
        assert np.isfinite(a).all() and np.isfinite(b).all()
    for st in hid_stats:
        assert st["over_2e-2_band_frac"] <= 2e-3, st
    np.testing.assert_allclose(_kvf(kvd), _kvf(kv1), atol=0.25, rtol=0.13)
    # the GatedDeltaNet states carry a recurring 0.2549 max abs difference in every comparison
    # of two reduction orders on record (item1 hier0 vs hier3, item2 parity_wide, this one),
    # with a relative mean difference of a few 1e-3: the mean is the gate, the max is recorded
    for st in (rec_stats, conv_stats):
        assert st["rel_mean_diff"] <= 1e-2, st
