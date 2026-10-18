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
"""Step time of the Qwen3.5-397B decode megakernel's MTP verify pass at serving shape (p3 port),
phase 4 A2 copy: wide split-plan cases (r48 .. r200), a wide forced routing file, KV page sharing
across sequences, the split independence test. test_p3_mk_step_bench.py is the shared p3 harness;
this file is item2_a1a2's own so that both can be edited in parallel.

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
  * parity legs run without redundant expert copies (see _World.weights_for); P3_MK_PARITY_COPIES=1
    keeps them (recorded, not gated) and P3_MK_REPLICATE_COPIES=1 gives the copy slots their
    primary's weights;
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

ROUTING_DIR = os.environ.get("P3_MK_ROUTING_DIR", "/workspace/tmp/p4-item2-routing")
# Wide cases (phase 4, A2): the serving split plan of a verify bucket of B rows (microbatches of
# at most 80 rows, `vllm_adapter._split_plan`), each call on its own sequences. Their forced
# routing (b<B>, count_b<B>, count_b<B>_m<i>, plan_b<B> in the routing file) is one independent
# draw per microbatch, so a microbatch hits as many experts as a served 80-row call and the union
# over the bucket is the recorded union of the whole batch (make_wide_routing.py).
WIDE_CASES = {"r48": 192, "r64": 256, "r128": 512, "r200": 800}
MK_SPLIT_BASE = 80
# A2: one wide call over the same rows (MKConfig.mb_rows microbatches inside the kernel, the
# MoE phase once over all rows); the w<n> case pairs with the r<n> split case on the same rows
WIDE_MB = int(os.environ.get("P3_MK_WIDE_MB", "64"))
# w16: one microbatch (n_mb 1); w8 (32 rows) pairs with the regular 32-row call for a
# two-microbatch step with P3_MK_WIDE_MB=16 (same rows, same routing union)
A2_CASES = {"w8": 32, "w16": 64, "w24": 96, "w48": 192, "w64": 256, "w128": 512}
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
OUT = os.environ.get("MK_BENCH_OUT") or common.run_dir("mk_step", f"{MODE}")
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
L, T = int(os.environ.get("P3_MK_L", "60")), 4  # P3_MK_L: a shorter stack (multiple of 4) for bisects
E_SLOTS = PLACEMENTS[PLACEMENT][0]
CTX = int(os.environ.get("P3_MK_CTX", "92160"))
CTX_SPREAD = min(8192, CTX // 2)
MAXP = 2048  # max_pages_per_seq at max_model_len 262144
PSEQ = -(-(CTX + CTX_SPREAD + T) // 128)  # pages reserved per sequence (785 at ctx 92160)
NSEQ_MAX = 200
GDN_STRIDE = 5  # vLLM block 640 tokens = 5 kernel pages
NUM_PAGES = 24576
# Pool layout: page 0 is the null page; GDN checkpoint slots (3 sub-layers x NSEQ_MAX sequences
# x T tokens, GDN_STRIDE pages each) follow, then the KV pages. The pool holds the KV pages of
# NSEQ_KV sequences; sequence g reads and writes the pages of g % NSEQ_KV, which keeps every
# sequence's bytes per step as served (KV DMAs do not care whose pages they are) while the
# checkpoints stay distinct. Parity tests need NSEQ_KV >= their sequence count (P3_MK_CTX).
CKPT0 = 1  # first GDN checkpoint slot
KV0 = (CKPT0 + 3 * NSEQ_MAX * T) * GDN_STRIDE  # first KV page
NSEQ_KV = (NUM_PAGES - KV0) // PSEQ
assert NSEQ_KV >= 1, (NUM_PAGES, KV0, PSEQ)
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


def _split_calls(B: int):
    """The split plan's calls of a B-row bucket: (rows, first sequence) per microbatch."""
    full, rem = divmod(B, MK_SPLIT_BASE)
    plan = [MK_SPLIT_BASE] * full + ([rem] if rem else [])
    calls, s0 = [], 0
    for rows in plan:
        calls.append((rows, s0))
        s0 += rows // T
    return tuple(calls)


CASES.update({k: _split_calls(B) for k, B in WIDE_CASES.items()})
CASES.update({k: ((B, 0),) for k, B in A2_CASES.items()})
# regular-kernel splits of the same rows in 64-row calls (and the 80-row call against 64 + 16):
# whether the 80-row and 64-row regular kernels agree on rows 0-15
CASES.update({
    "r20b": ((64, 0), (16, 16)),
    "r24b": ((32, 0), (32, 8), (32, 16)),
    "r48b": ((64, 0), (64, 16), (64, 32)),
    "r64b": ((64, 0), (64, 16), (64, 32), (64, 48)),
})
# w24 (96 rows) has no routing bucket of its own: both legs read rows 0..95 of the 192-row one
CASE_RKEY_EXTRA = {"r20b": None, "r24b": 192, "w24": 192, "r48b": 192, "r64b": 256}
# case -> routing bucket of the whole case (None: each call uses its own b<rows> routing)
CASE_RKEY = {k: WIDE_CASES.get(k) for k in CASES}
CASE_RKEY.update(CASE_RKEY_EXTRA)
_PAIRS = (("w8", "b32"), ("w16", "b64"), ("w48", "r48"), ("w64", "r64"),
          ("b80", "r20b"), ("w48", "r48b"), ("w64", "r64b"), ("w24", "r24b"),
          # split plan against split plan (both legs the regular kernel; they pick different copy
          # holders for the same rows): the leg-to-leg floor when redundant copies are in use
          ("r48b", "r48"), ("r64b", "r64"))
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
    # A2 wide-step MoE switches (bisecting the device parity of the wide kernel)
    "dynrow0": (frozenset(), 0, {}, {("context", "WIDE_DYN_ROW"): False}),
    "scatter_nn": (frozenset(), 0, {}, {("context", "WIDE_SCATTER_TN"): False}),
    "dynrow0_scatter_nn": (frozenset(), 0, {},
                           {("context", "WIDE_DYN_ROW"): False, ("context", "WIDE_SCATTER_TN"): False}),
    "ar2": (frozenset(), 0, {}, {("context", "WIDE_AR_SLOTS"): 2}),
    "ngi2": (frozenset(), 0, {}, {("context", "WIDE_NGI"): 2}),
    "cap16": (frozenset(), 0, {}, {("context", "WIDE_CAP"): 16}),
    "noreuse": (frozenset(), 0, {}, {("context", "WIDE_TILE_REUSE"): False}),
    # the expert loop runs but its output is dropped (wide leg); the split leg skips experts
    "xside": (frozenset(), 0, {}, {("context", "WIDE_SCATTER"): False}),
    # the KV stream in its own ring instead of the expert pool slots
    "nokve": (frozenset(), 0, {"kv_ering": False}, {}),
    "nopair": (frozenset(), 0, {"pair_kv": False}, {}),
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


def _cfg(b: int, e_slots: int = E_SLOTS, wide: bool | None = None) -> C.MKConfig:
    # tp mode: the routed experts are TP-sliced (moe_tp=True), every rank holds all experts,
    # and e_slots / forced routing counts do not apply (the placement table is EP specific).
    # wide: the A2 wide step (mb_rows microbatches in one call); default for b > 80.
    if wide is None:
        wide = b > MK_SPLIT_BASE
    return dataclasses.replace(
        C.FULL_EP,
        R=R,
        L=L,
        moe_tp=MODE == "tp",
        stack_e=False,
        e_slots=e_slots if MODE == "ep" else 0,
        paged_kv=True,
        fused_pool_io=True,
        B=b,
        T=T,
        mb_rows=WIDE_MB if wide else 0,
        num_pages=NUM_PAGES,
        max_pages_per_seq=MAXP,
        gdn_stride_pages=GDN_STRIDE,
        ctx=MAXP * 128,
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
                self._smalls_spec = (lay, s_width, shape, dt)
                w[k] = _global(self.mesh, self._smalls(lay, s_width, shape, dt))
            elif k == "final_norm":
                w[k] = _global(self.mesh, np.zeros((R, *shape), np.float32))
            else:
                kind = "bytes" if jnp.dtype(dt) == jnp.dtype(FP8) else (
                    "normal" if jnp.dtype(dt) == jnp.dtype(C.BF16) else "scale")
                w[k] = _filler(self.mesh, (R, *shape), dt, kind)(_key(7, i))
        if int(os.environ.get("P3_MK_REPLICATE_COPIES", "0")) and c.EW > c.ER:
            # redundant copy slots hold their primary's weights and scale row, as served: a
            # parity between legs that pick different copy holders then compares the same math
            self._replicate_copies(w, c)
        self.weights = w
        self._weights_leg = {}
        self.pool_shape = (R * NUM_PAGES, 2, 64, 4, 128)
        self.pools = self.fresh_pools()
        jax.block_until_ready((self.weights, self.pools))
        rng = np.random.default_rng(11)
        self.ctx = CTX + rng.integers(-CTX_SPREAD, CTX_SPREAD + 1, NSEQ_MAX)
        self.hidden_rows = rng.standard_normal((NSEQ_MAX * T, c.H)).astype(np.float32)

    def _replicate_copies(self, w, c):
        ER, EXTRA = c.ER, c.EW - c.ER
        codes = np.asarray(self.routing["ecopy"][:L]).astype(np.int64)  # [L, E]: 0 or 1 + r1 * EXTRA + k

        @functools.partial(jax.jit, donate_argnums=0)
        def put(arr, src_r, src_s, dst_r, dst_s):
            return arr.at[dst_r, dst_s].set(arr[src_r, src_s])

        @functools.partial(jax.jit, donate_argnums=0)
        def put_es(arr, l, src_r, src_s, dst_r, dst_s):
            return arr.at[dst_r, l, dst_s].set(arr[src_r, l, src_s])

        n_cp = 0
        for l in range(L):
            v = np.nonzero(codes[l])[0]
            if len(v) == 0:
                continue
            cc = codes[l][v] - 1
            src_r, src_s = v // ER, v % ER
            dst_r, dst_s = cc // EXTRA, ER + cc % EXTRA
            idx = tuple(jnp.asarray(a, jnp.int32) for a in (src_r, src_s, dst_r, dst_s))
            for k in ("e13", "e2"):
                sh = w[k][l].sharding
                w[k] = tuple(
                    jax.device_put(put(a, *idx), sh) if i == l else a for i, a in enumerate(w[k])
                )
            sh = w["es"].sharding
            w["es"] = jax.device_put(put_es(w["es"], l, *idx), sh)
            n_cp += len(v)
        jax.block_until_ready((w["e13"], w["e2"], w["es"]))
        print(f"replicated {n_cp} copy slots over {L} layers", flush=True)
        # check: every copy slot of a few layers holds its primary's bytes and scale row
        gap = {"e13": 0.0, "e2": 0.0, "es": 0.0}
        for l in sorted({0, 1, L // 2, L - 1}):
            v = np.nonzero(codes[l])[0]
            if len(v) == 0:
                continue
            cc = codes[l][v] - 1
            idx = tuple(jnp.asarray(a, jnp.int32) for a in (v // ER, v % ER, cc // EXTRA, ER + cc % EXTRA))
            for k in ("e13", "e2"):
                a = w[k][l]
                d = jnp.abs(a[idx[0], idx[1]].astype(jnp.float32) - a[idx[2], idx[3]].astype(jnp.float32))
                gap[k] = max(gap[k], float(jnp.max(d)))
            a = w["es"]
            d = jnp.abs(a[idx[0], l, idx[1]] - a[idx[2], l, idx[3]])
            gap["es"] = max(gap["es"], float(jnp.max(d)))
        print(f"replication check (layers 0, 1, {L // 2}, {L - 1}): max |copy - primary| "
              f"e13 {gap['e13']:g} e2 {gap['e2']:g} es {gap['es']:g}", flush=True)

    def weights_for(self, leg: str):
        """The weights a parity leg (wide or split) runs with: the world's, with the ecopy
        smalls segment zeroed (no redundant copies in use) unless P3_MK_PARITY_COPIES=1 keeps the
        copy slots, or P3_MK_LEG_NO_COPIES names the one leg that drops them.

        Parity legs run without redundant copies because the result of the production kernel
        depends on which rank computes an expert: every per-rank partial of the routed sum and
        every stage of the hierarchical all-reduce is rounded to bf16 (`collectives.py`), so two
        plans whose Po2C choices differ agree only to about 0.7-0.8 close at the 2e-2 tolerance
        after 60 layers (mean row diff 0.025 on |h| 0.8) with identical weights in primary and
        copy slots (split vs split r48b:r48 0.82, r64b:r64 0.81; copies off 0.99995). With the
        copies off, every expert runs on its primary rank in both plans and the comparison is
        the kernels'."""
        no_copies = (os.environ.get("P3_MK_LEG_NO_COPIES", "") == leg
                     or (not os.environ.get("P3_MK_LEG_NO_COPIES")
                         and not int(os.environ.get("P3_MK_PARITY_COPIES", "0"))))
        if not no_copies:
            return self.weights
        if "nocopies" not in self._weights_leg:
            lay, s_width, shape, dt = self._smalls_spec
            w = dict(self.weights)
            w["smalls"] = _global(self.mesh, self._smalls(lay, s_width, shape, dt, no_copies=True))
            self._weights_leg["nocopies"] = w
        return self._weights_leg["nocopies"]

    def fresh_pools(self, seed: int = 99):
        f = _filler(self.mesh, self.pool_shape, FP8, "bytes")
        return tuple(f(_key(seed, a)) for a in range(C.FULL.NATT))

    def _smalls(self, lay, s_width, shape, dt, no_copies: bool | None = None):
        if no_copies is None:
            no_copies = bool(int(os.environ.get("P3_MK_NO_COPIES", "0")))
        rng = np.random.default_rng(3)
        sm = rng.uniform(0.0, 0.1, (L, 1, s_width)).astype(np.float32)
        for name in ("in_s", "out_s", "sh13_s", "sh2_s"):
            o, n = lay[name]
            sm[:, :, o:o + n] = 0.01
        o, _ = lay["alog"]
        sm[:, :, o:o + 128] = -1.0
        if "ecopy" in lay:
            o, n = lay["ecopy"]
            # P3_MK_NO_COPIES=1: no redundant expert copies in use, so every expert runs on its
            # primary rank in both legs of a parity (the synthetic copy slots hold weights
            # unrelated to the primary's, so a Po2C choice that differs between legs is not a
            # kernel difference)
            if not no_copies:
                sm[:, 0, o:o + n] = self.routing["ecopy"][:L]
        assert sm.shape == tuple(shape), (sm.shape, shape)
        return np.broadcast_to(sm[None], (R, *shape)).astype(dt)

    def inputs(self, b: int, s0: int, rkey: int | None = None, wide: bool | None = None):
        """pos, hidden, page_table, ckpt_indices, read_offsets, rbias of a b-row call on
        sequences s0 .. s0 + b / T - 1 (hidden rows s0 * T ..). rkey: routing bucket holding the
        case's rows (the call's rows are s0 * T ..); None: the b<b> routing."""
        c = _cfg(b, wide=wide)
        ns = b // T
        pos = np.zeros((b,), np.int32)
        pt = np.zeros((ns, MAXP), np.int32)
        ckpt = np.zeros((3, ns, T), np.int32)
        for s in range(ns):
            g = s0 + s
            n_ctx = int(self.ctx[g])
            pos[s * T:(s + 1) * T] = n_ctx + np.arange(T)
            npg = -(-(n_ctx + T) // 128)
            pt[s, :npg] = KV0 + (g % NSEQ_KV) * PSEQ + np.arange(npg)
            for sub in range(3):
                ckpt[sub, s] = CKPT0 + ((sub * NSEQ_MAX + g) * T) + np.arange(T)
        roff = np.zeros((3, ns), np.int32)
        hid = np.zeros((c.M, c.H), np.float32)
        hid[:b] = self.hidden_rows[s0 * T:s0 * T + b]
        if rkey is None:
            picks = self.routing[f"b{b}"][:L]  # [L, b, 10] virtual expert ids
        else:
            picks = self.routing[f"b{rkey}"][:L, s0 * T:s0 * T + b]
        # the wide kernel reads the routing bias in 128-row chunks
        m_rb = -(-c.M // 128) * 128 if c.wide else c.M
        rb = np.full((L, m_rb, c.E), 1e-3, np.float32)
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

    def step_fn(self, mod, b: int, skip, scopes: int, tile_over=None, patch=None, wide=None):
        c = _cfg(b, self.e_slots, wide=wide)
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
    row = {"environment_id": _ENV["id"], "mode": MODE, "R": R,
           "process": jax.process_index(), "process_count": jax.process_count(), **row}
    if lead:
        with open(os.path.join(OUT, "results.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")
    print("MK_BENCH " + json.dumps(row), flush=True)


def _count_key(rkey, b, i):
    return f"count_b{rkey}_m{i}" if rkey else f"count_b{b}"


def _run_case(world, mod, case, variant, label, n_steps=N_STEPS, trace=True):
    skip, scopes, tile_over, patch = VARIANTS[variant]
    calls = []
    t_build = time.time()
    rkey = CASE_RKEY.get(case)
    wide = case in A2_CASES
    for b, s0 in CASES[case]:
        fn, tiles = world.step_fn(mod, b, skip, scopes, tile_over, patch, wide=wide)
        calls.append((fn, world.inputs(b, s0, rkey, wide=wide), b, tiles))
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
            float(world.routing[_count_key(rkey, b, i)].mean()) for i, (_, _, b, _) in enumerate(calls)
        ],
        "expert_count_max": [
            float(world.routing[_count_key(rkey, b, i)].max(axis=1).mean())
            for i, (_, _, b, _) in enumerate(calls)
        ],
        # the union over the case's rows (what an expert-major MoE phase streams)
        "expert_count_union_max": float(
            world.routing[f"count_b{rkey}"].max(axis=1).mean()) if rkey else None,
        "expert_count_union_mean": float(
            world.routing[f"count_b{rkey}"].mean()) if rkey else None,
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
                kv.add(KV0 + (g % NSEQ_KV) * PSEQ + p)
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


@pytest.mark.parametrize("case", ["b4", "b32", "r28"])
def test_parity(world, case):
    """This tree's kernel against the reference package on identical inputs: the hidden
    output of every call, the KV pages of the new tokens and the GatedDeltaNet checkpoints.
    On a multi-host slice every process checks the ranks it addresses."""
    ref = _ref_package()
    c = _cfg(32)
    kv_pages, gdn_pages = _written_pages(world, case)
    world.pools = None  # one pool set in HBM at a time (weights take 49 of 94 GiB)
    outs = {}
    for label, mod in (("ref", ref), ("new", _THIS)):
        pools = world.fresh_pools(seed=5)
        hids = []
        for b, s0 in CASES[case]:
            fn, _ = world.step_fn(mod, b, frozenset(), 0)
            pos, hid_in, pt, ck, ro, rb = world.inputs(b, s0, CASE_RKEY.get(case))
            hid, pools = fn(pos, hid_in, world.weights, pools, pt, ck, ro, rb)
            hids.append(np.stack(_local_rows(hid, b)))
        g_idx = gdn_pages[:, None] + np.arange(c.gdn_pages)[None, :]
        kv = np.stack([_take_pages(p, kv_pages) for p in pools])
        gdn = np.stack([_take_pages(p, g_idx) for p in pools])
        outs[label] = (hids, kv, gdn)  # kv [NATT, R, n, 2, 64, 4, 128], gdn [NATT, R, n, 5, ...]
        del pools
    world.pools = world.fresh_pools()
    (h_ref, kv_ref, g_ref), (h_new, kv_new, g_new) = outs["ref"], outs["new"]
    kvf = lambda a: a.view(jnp.dtype(FP8)).astype(np.float32)
    n = g_ref.shape[2]
    rec = lambda a: _unpack_rec(a[..., : c.rec_pages, :, :, :, :].reshape(-1, c.rec_pages, 2, 64, 4, 128), c.vh)
    conv = lambda a: _unpack_conv(a[..., c.rec_pages, :, :, :, :].reshape(-1, 2, 64, 4, 128), c.QKV)
    row = {
        "label": "parity",
        "case": case,
        "hidden_max_abs_diff": [float(np.abs(a - b).max()) for a, b in zip(h_new, h_ref)],
        "hidden_ref_max_abs": [float(np.abs(b).max()) for b in h_ref],
        "hidden_identical_frac": [float(np.mean(a == b)) for a, b in zip(h_new, h_ref)],
        "kv_bytes_identical_frac": float(np.mean(kv_new == kv_ref)),
        "gdn_bytes_identical_frac": float(np.mean(g_new == g_ref)),
        "kv_max_abs_diff": float(np.abs(kvf(kv_new) - kvf(kv_ref)).max()),
        "rec_max_abs_diff": float(np.abs(rec(g_new) - rec(g_ref)).max()),
        "conv_max_abs_diff": float(np.abs(conv(g_new) - conv(g_ref)).max()),
        "n_kv_pages": int(len(kv_pages)),
        "n_gdn_slots": int(n),
    }
    _record(row)
    for a, b in zip(h_new, h_ref):
        assert np.isfinite(a).all() and np.isfinite(b).all()
        np.testing.assert_allclose(a, b, atol=2e-2, rtol=2e-2)
    np.testing.assert_allclose(kvf(kv_new), kvf(kv_ref), atol=0.25, rtol=0.13)
    np.testing.assert_allclose(rec(g_new), rec(g_ref), atol=2e-2, rtol=2e-2)
    np.testing.assert_allclose(conv(g_new), conv(g_ref), atol=2e-2, rtol=2e-2)


def _case_seqs(case):
    return max(s0 + b // T for b, s0 in CASES[case])


@pytest.mark.parametrize("case", [k for k in ("r48", "r64") if k in CASES])
def test_split_independence(world, case):
    """The split plan's microbatches are independent: the last call of the case gives the same
    hidden rows and pool pages whether it runs after the other microbatches (on the pools they
    wrote) or alone on fresh pools. Needs pools without KV page sharing between the case's
    sequences (P3_MK_CTX small enough for NSEQ_KV >= its sequence count)."""
    if _case_seqs(case) > NSEQ_KV:
        pytest.skip(f"{case} needs NSEQ_KV >= {_case_seqs(case)} (have {NSEQ_KV}); lower P3_MK_CTX")
    c = _cfg(32)
    rkey = CASE_RKEY[case]
    b_last, s0_last = CASES[case][-1]
    last_seqs = range(s0_last, s0_last + b_last // T)
    kv_last = np.array(sorted(
        KV0 + (g % NSEQ_KV) * PSEQ + p for g in last_seqs
        for p in range(int(world.ctx[g]) // 128, -(-(int(world.ctx[g]) + T) // 128))))
    gdn_last = np.array([(CKPT0 + (sub * NSEQ_MAX + g) * T + t) * GDN_STRIDE
                         for g in last_seqs for sub in range(3) for t in range(T)])
    world.pools = None
    outs = {}
    for label, calls in (("seq", CASES[case]), ("alone", CASES[case][-1:])):
        pools = world.fresh_pools(seed=5)
        hid = None
        for b, s0 in calls:
            fn, _ = world.step_fn(_THIS, b, frozenset(), 0)
            pos, hid_in, pt, ck, ro, rb = world.inputs(b, s0, rkey)
            hid, pools = fn(pos, hid_in, world.weights, pools, pt, ck, ro, rb)
        h = np.stack(_local_rows(hid, b_last))
        g_idx = gdn_last[:, None] + np.arange(c.gdn_pages)[None, :]
        kv = np.stack([_take_pages(p, kv_last) for p in pools])
        gdn = np.stack([_take_pages(p, g_idx) for p in pools])
        outs[label] = (h, kv, gdn)
        del pools
    world.pools = world.fresh_pools()
    (h_a, kv_a, g_a), (h_b, kv_b, g_b) = outs["seq"], outs["alone"]
    kvf = lambda a: a.view(jnp.dtype(FP8)).astype(np.float32)
    rec = lambda a: _unpack_rec(a[..., : c.rec_pages, :, :, :, :].reshape(-1, c.rec_pages, 2, 64, 4, 128), c.vh)
    conv = lambda a: _unpack_conv(a[..., c.rec_pages, :, :, :, :].reshape(-1, 2, 64, 4, 128), c.QKV)
    _record({
        "label": "split_independence",
        "case": case,
        "rows": [b for b, _ in CASES[case]],
        "hidden_max_abs_diff": float(np.abs(h_a - h_b).max()),
        "hidden_ref_max_abs": float(np.abs(h_b).max()),
        "hidden_identical_frac": float(np.mean(h_a == h_b)),
        "kv_bytes_identical_frac": float(np.mean(kv_a == kv_b)),
        "gdn_bytes_identical_frac": float(np.mean(g_a == g_b)),
        "kv_max_abs_diff": float(np.abs(kvf(kv_a) - kvf(kv_b)).max()),
        "rec_max_abs_diff": float(np.abs(rec(g_a) - rec(g_b)).max()),
        "conv_max_abs_diff": float(np.abs(conv(g_a) - conv(g_b)).max()),
        "n_kv_pages": int(len(kv_last)),
        "n_gdn_slots": int(len(gdn_last)),
    })
    assert np.isfinite(h_a).all() and np.isfinite(h_b).all()
    np.testing.assert_allclose(h_a, h_b, atol=2e-2, rtol=2e-2)
    np.testing.assert_allclose(kvf(kv_a), kvf(kv_b), atol=0.25, rtol=0.13)
    np.testing.assert_allclose(rec(g_a), rec(g_b), atol=2e-2, rtol=2e-2)
    np.testing.assert_allclose(conv(g_a), conv(g_b), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("wide,split", [p for p in _PAIRS
                                        if p[0] in CASES and f"{p[0]}:{p[1]}" in _subset(
                                            "MK_PARITY_PAIRS", [f"{a}:{b}" for a, b in _PAIRS[1:4]])])
@pytest.mark.parametrize("variant", _subset("MK_PARITY_VARIANTS", ["base"]))
def test_parity_wide(world, wide, split, variant):
    """A2: the wide call over all rows against the split plan's microbatch calls on identical
    inputs and fresh pools: hidden rows of every sequence, the KV pages of the new tokens and the
    GatedDeltaNet checkpoints. Needs alias-free KV pages (P3_MK_CTX). The variant (ablation skips
    and module patches of VARIANTS) applies to both legs."""
    if _case_seqs(split) > NSEQ_KV:
        pytest.skip(f"{split} needs NSEQ_KV >= {_case_seqs(split)} (have {NSEQ_KV}); lower P3_MK_CTX")
    skip, scopes, tile_over, patch = VARIANTS[variant]
    c = _cfg(32)
    kv_pages, gdn_pages = _written_pages(world, split)
    world.pools = None
    outs = {}
    for label, case in (("split", split), ("wide", wide)):
        pools = world.fresh_pools(seed=5)
        hids = []
        is_wide = case in A2_CASES
        skip_leg = skip if (is_wide or variant != "xside") else frozenset({"experts"})
        for b, s0 in CASES[case]:
            fn, _ = world.step_fn(_THIS, b, skip_leg, scopes, tile_over, patch, wide=is_wide)
            pos, hid_in, pt, ck, ro, rb = world.inputs(b, s0, CASE_RKEY.get(case), wide=is_wide)
            hid, pools = fn(pos, hid_in, world.weights_for(label), pools, pt, ck, ro, rb)
            hids.append(np.stack(_local_rows(hid, b)))
        g_idx = gdn_pages[:, None] + np.arange(c.gdn_pages)[None, :]
        kv = np.stack([_take_pages(p, kv_pages) for p in pools])
        gdn = np.stack([_take_pages(p, g_idx) for p in pools])
        outs[label] = (np.concatenate(hids, axis=1), kv, gdn)
        del pools
    world.pools = world.fresh_pools()
    (h_s, kv_s, g_s), (h_w, kv_w, g_w) = outs["split"], outs["wide"]
    kvf = lambda a: a.view(jnp.dtype(FP8)).astype(np.float32)
    rec = lambda a: _unpack_rec(a[..., : c.rec_pages, :, :, :, :].reshape(-1, c.rec_pages, 2, 64, 4, 128), c.vh)
    conv = lambda a: _unpack_conv(a[..., c.rec_pages, :, :, :, :].reshape(-1, 2, 64, 4, 128), c.QKV)
    close = np.isclose(h_w, h_s, atol=2e-2, rtol=2e-2)
    # where the differences are: per 64-row block (the wide call's microbatch), per sequence
    nrow = h_w.shape[1]
    by_mb = [float(close[:, r0:r0 + WIDE_MB].mean()) for r0 in range(0, nrow, WIDE_MB)]
    diff_by_mb = [float(np.abs(h_w - h_s)[:, r0:r0 + WIDE_MB].max()) for r0 in range(0, nrow, WIDE_MB)]
    n_seq = _case_seqs(split)
    g_eq = (g_w == g_s).reshape(g_w.shape[0], g_w.shape[1], n_seq, 3, T, -1).mean(axis=(0, 1, 3, 4, 5))
    kv_eq = (kv_w == kv_s).reshape(kv_w.shape[0], kv_w.shape[1], kv_w.shape[2], -1).mean(axis=(0, 1, 3))
    _record({
        "label": "parity_wide",
        "case": wide,
        "variant": variant,
        "rows": [b for b, _ in CASES[split]],
        "hidden_close_by_mb": [round(x, 4) for x in by_mb],
        "hidden_close_by_16rows": [round(float(close[:, r0:r0 + 16].mean()), 4) for r0 in range(0, nrow, 16)],
        "hidden_max_abs_diff_by_16rows": [round(float(np.abs(h_w - h_s)[:, r0:r0 + 16].max()), 4) for r0 in range(0, nrow, 16)],
        "hidden_mean_abs_diff_by_row_first32": [round(float(np.abs(h_w - h_s)[:, r].mean()), 5) for r in range(min(32, nrow))],
        "hidden_ref_mean_abs_first32": [round(float(np.abs(h_s)[:, r].mean()), 4) for r in range(min(32, nrow))],
        "hidden_abs_diff_p999": float(np.quantile(np.abs(h_w - h_s), 0.999)),
        "hidden_max_abs_diff_by_mb": [round(x, 4) for x in diff_by_mb],
        "gdn_identical_by_seq": [round(float(x), 3) for x in g_eq],
        "kv_identical_by_page": [round(float(x), 3) for x in kv_eq],
        "hidden_max_abs_diff": float(np.abs(h_w - h_s).max()),
        "hidden_ref_max_abs": float(np.abs(h_s).max()),
        "hidden_identical_frac": float(np.mean(h_w == h_s)),
        "hidden_close_frac": float(close.mean()),
        "kv_bytes_identical_frac": float(np.mean(kv_w == kv_s)),
        "gdn_bytes_identical_frac": float(np.mean(g_w == g_s)),
        "kv_max_abs_diff": float(np.abs(kvf(kv_w) - kvf(kv_s)).max()),
        "rec_max_abs_diff": float(np.abs(rec(g_w) - rec(g_s)).max()),
        "conv_max_abs_diff": float(np.abs(conv(g_w) - conv(g_s)).max()),
        "n_kv_pages": int(len(kv_pages)),
        "n_gdn_slots": int(len(gdn_pages)),
    })
    assert np.isfinite(h_w).all() and np.isfinite(h_s).all()
    if int(os.environ.get("P3_MK_PARITY_COPIES", "0")) or os.environ.get("P3_MK_LEG_NO_COPIES"):
        # legs that assign experts to different ranks: the bf16 all-reduce payload floor
        # (see _World.weights_for) is recorded, not gated
        return
    # a reordered all-reduce or expert sum leaves a few 1e-2 .. 1e-1 differences after 60 layers
    # (item1 measured 0.02-0.03% of elements over 2e-2 for a different all-reduce order), so the
    # gate is statistical: at least 99.9% close overall and 99.5% in every 64-row block
    assert close.mean() >= 0.999 and min(by_mb) >= 0.995 and np.abs(h_w - h_s).max() < 1.0, (
        float(close.mean()), by_mb, float(np.abs(h_w - h_s).max()))
    assert np.mean(np.isclose(kvf(kv_w), kvf(kv_s), atol=0.25, rtol=0.13)) > 0.999
