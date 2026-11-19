#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""q38 L6 lever engalloc: CPU microbench of the engine core's per-request KV calls on the served decode plan.

Runs in a CDK CPU job only. Usage:
  python3 tests/q38_l6/bench_engine_alloc.py --config model_config.json [--n 3,6,15,25,52] [--steps 400]

Plan: built by the served code path (qwen4_exp pooled_cache_specs + get_kv_cache_groups) with Qwen3.8 Flash's
config.json and the served decode env (TP8, fp8 KV, MTP k=3, QSA main-KV DCP 4, packed compressed keys, main KV
unpaired behind the PP8 prefill, unified pool, eagle block drop off, prefix caching on, mamba cache mode align,
retention interval 0, watermark 0.05, 98,304 blocks). It must reproduce the served log's 65 'kv cache group sizes'
(37 x 512, 13 x 2048, 2 x 512, 13 x 4); if the builder fails, a reconstruction from that list is used and labelled.

Traffic (per N running requests): PD admissions of ~CTX-token prompts (external tokens, delayed caching, the
landing's cache_blocks), then decode steps in the async scheduler's order: allocate_slots(num_new = 1 + 3 spec,
lookahead 3) for every running request, then the previous step's outputs (accepted drafts ~ p=0.6 each, rejections
roll back num_computed_tokens and the placeholders) and cache_blocks(num_computed - placeholders) per request.
Arms: base (no twins), twins (served D1k), fast (twins + engalloc). Every call is timed; reported: median over
steps of the per-step mean us per call, and the mean over all calls (boundary steps included). Identity: the twins
and fast arms run the same seed and must end with identical block tables, free queue, hash map and unwritten set.
A profile pass attributes the twins arm's allocate_slots / cache_blocks time to the coordinator loops and to each
manager kind.
"""

import argparse
import gc
import hashlib
import json
import os
import random
import statistics as st
import sys
import time
from collections import Counter, defaultdict
from types import SimpleNamespace as NS

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--n", default="3,6,15,25,52")
ap.add_argument("--steps", type=int, default=400)
ap.add_argument("--warm", type=int, default=40)
ap.add_argument("--ctx", type=int, default=60000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--profile-n", type=int, default=6)
a = ap.parse_args()

SERVED_ENV = {
    "QWEN4_EXP_PACK_COMPRESSED_K": "1", "QWEN4_EXP_PACK_MAIN_KV": "0", "TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL": "1",
    "VLLM_TPU_QSA_MAIN_KV_DCP": "4", "VLLM_TPU_QSA_MTP_KV_DCP": "1", "VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP": "1",
    "TPU_RAIDEN_POOL_TAGS_PER_LAYER": "1", "TPU_KV_BYTES_EXPLICIT": "0",
}
os.environ.update(SERVED_ENV)
for k in list(os.environ):
    if k.startswith("VLLM_TPU_L6_"):
        del os.environ[k]
SERVED_SIZES = [512] * 37 + [2048] * 13 + [512] * 2 + [4] * 13

import torch  # noqa: E402

import vllm_torchtpu  # noqa: E402,F401
from vllm_torchtpu import patch_registry  # noqa: E402

for phase in ("platform_activation", "engine_core"):
    patch_registry.apply(phase)
from vllm.sampling_params import SamplingParams  # noqa: E402
from vllm.utils.hashing import get_hash_fn_by_name  # noqa: E402
from vllm.v1.core import kv_cache_utils as ku  # noqa: E402
from vllm.v1.core.kv_cache_manager import KVCacheManager  # noqa: E402
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash  # noqa: E402
from vllm.v1.kv_cache_interface import (  # noqa: E402
    CircularBufferSpec, FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, MambaSpec)
from vllm.v1.request import Request, RequestStatus  # noqa: E402

from vllm_torchtpu.core import kv_twin_groups, l6_ab  # noqa: E402
from vllm_torchtpu.core import l6_engine_alloc as ea_mod  # noqa: E402

NUM_BLOCKS = 98304
MML = 262144


def ns(d):
    if isinstance(d, dict):
        return NS(**{k: ns(v) for k, v in d.items()})
    if isinstance(d, list):
        return [ns(x) for x in d]
    return d


def served_groups():
    t = ns(json.load(open(a.config))["text_config"])
    from vllm.config.cache import _layout_from_name

    cfg = NS(
        model_config=NS(hf_text_config=t, hf_config=NS(model_type="qwen4_exp", text_config=t), dtype=torch.bfloat16,
                        max_model_len=MML, get_total_num_hidden_layers=lambda: 48, use_mla=False, is_hybrid=True),
        cache_config=NS(cache_dtype="fp8", block_size=128, enable_prefix_caching=True,
                        num_gpu_blocks_override=NUM_BLOCKS, prefix_cache_retention_interval=0,
                        mamba_cache_mode="align", mamba_cache_dtype="auto", mamba_ssm_cache_dtype="auto",
                        prefix_match_unit=None, kv_cache_memory_bytes=None, kv_cache_layout="BLHNC",
                        get_resolved_kv_cache_layout=lambda: _layout_from_name("BLHNC"),
                        enable_mamba_shared_prefix_checkpoint=False),
        parallel_config=NS(tensor_parallel_size=8, decode_context_parallel_size=1, pipeline_parallel_size=1,
                           prefill_context_parallel_size=1, data_parallel_size=1),
        scheduler_config=NS(disable_hybrid_kv_cache_manager=False, max_num_batched_tokens=2048, max_num_seqs=64),
        speculative_config=NS(method="mtp", num_speculative_tokens=3, use_eagle=lambda: True,
                              use_eagle_block_drop=lambda: False),
        kv_transfer_config=NS(kv_role="kv_consumer", is_kv_producer=False, is_kv_consumer=True),
        attention_config=NS(hisparse_config=None),
        device_config=NS(device_type="tpu"),
    )
    from vllm_torchtpu.models.vllm.qwen4_exp.config import ensure_kv_group_size_rule
    from vllm_torchtpu.models.vllm.qwen4_exp.pool_layout import pooled_cache_specs

    ensure_kv_group_size_rule()
    _, specs = pooled_cache_specs(cfg)
    groups = ku.get_kv_cache_groups(cfg, specs)
    kcfg = KVCacheConfig(num_blocks=NUM_BLOCKS, kv_cache_tensors=[], kv_cache_groups=groups)
    sched_bs, hash_bs = ku.resolve_kv_cache_block_sizes(kcfg, cfg)
    return groups, sched_bs, hash_bs


def reconstructed_groups():
    bf = torch.bfloat16
    specs = ([MambaSpec(block_size=512, shapes=((2, 2),), dtypes=(bf,), mamba_cache_mode="align",
                        num_speculative_blocks=3) for _ in range(37)]
             + [FullAttentionSpec(block_size=2048, num_kv_heads=1, head_size=256, dtype=bf) for _ in range(13)]
             + [FullAttentionSpec(block_size=512, num_kv_heads=1, head_size=128, dtype=bf) for _ in range(2)]
             + [CircularBufferSpec(block_size=4, num_kv_heads=1, head_size=132, head_size_v=0, dtype=bf,
                                   page_size_padded=8192) for _ in range(13)])
    return [KVCacheGroupSpec([f"layer.{i}"], s) for i, s in enumerate(specs)], 2048, 512


try:
    GROUPS, SCHED_BS, HASH_BS = served_groups()
    BASIS = "served builder (pooled_cache_specs + get_kv_cache_groups)"
except Exception as e:  # noqa: BLE001
    print(f"BENCH served builder failed: {type(e).__name__}: {e}; using the reconstruction from the served log")
    GROUPS, SCHED_BS, HASH_BS = reconstructed_groups()
    BASIS = "RECONSTRUCTION from the served log's group sizes"
sizes = [g.kv_cache_spec.block_size for g in GROUPS]
BUILT = (GROUPS, SCHED_BS, HASH_BS)
print(f"BENCH plan basis: {BASIS}")
print(f"BENCH groups {len(GROUPS)}: {dict(Counter((type(g.kv_cache_spec).__name__, g.kv_cache_spec.block_size) for g in GROUPS))}")
print(f"BENCH group sizes == served log list: {'MATCH' if sizes == SERVED_SIZES else 'DIFF ' + str(sizes)}")
print(f"BENCH scheduler_block_size {SCHED_BS} hash_block_size {HASH_BS}")
for i, g in enumerate(GROUPS):
    s = g.kv_cache_spec
    if i in (0, 37, 50, 52):
        extra = ""
        if isinstance(s, MambaSpec):
            extra = (f" mode={s.mamba_cache_mode} spec_blocks={s.num_speculative_blocks} "
                     f"ckpt_blocks={s.num_prefill_checkpoint_blocks} ckpt_align={s.prefill_checkpoint_alignment}")
        print(f"BENCH group {i}: {type(s).__name__} block {s.block_size} prefix_cacheable "
              f"{getattr(s, 'prefix_cacheable', '?')} eagle {getattr(g, 'is_eagle_group', '?')}{extra}")

fn = get_hash_fn_by_name("sha256")
init_none_hash(fn)
HASHER = get_request_block_hasher(HASH_BS, fn)


def mk512():
    """The builder's specs at the 512-token base page of T98a R4-K's served decode log (j-5d24df3b: 37 x 512,
    13 x 2048, 2 x 512, 13 x 4; the D-MK whole-step geometry): every block size doubled, scheduler 2048, hash 512."""
    import dataclasses

    groups = [KVCacheGroupSpec(g.layer_names, dataclasses.replace(
        g.kv_cache_spec, block_size=g.kv_cache_spec.block_size * (1 if g.kv_cache_spec.block_size == 4 else 2)))
        for g in BUILT[0]]
    return groups, BUILT[1] * 2, BUILT[2] * 2


GEOMS = [("builder", BUILT), ("mk512", None)]


def make(arm):
    kcfg = KVCacheConfig(num_blocks=NUM_BLOCKS, kv_cache_tensors=[], kv_cache_groups=GROUPS)
    m = KVCacheManager(kcfg, max_model_len=MML, scheduler_block_size=SCHED_BS, hash_block_size=HASH_BS,
                       max_in_flight_tokens=2 * 2048, enable_caching=True, use_eagle=False, num_prefill_lookahead=1,
                       dcp_world_size=1, pcp_world_size=1, watermark=0.05)
    tw = kv_twin_groups.install(m.coordinator) if arm in ("twins", "fast", "fastchk") else None
    ea = None
    if arm in ("fast", "fastchk"):
        os.environ["VLLM_TPU_L6_ENGINE_ALLOC_FAST"] = "1"
        l6_ab._resolve()
        ea = ea_mod.install(m, check=1 if arm == "fastchk" else 0)
        assert ea is not None, "engalloc refused the served plan"
    else:
        os.environ.pop("VLLM_TPU_L6_ENGINE_ALLOC_FAST", None)
        l6_ab._resolve()
    return m, tw, ea


class Drive:
    SPEC, LA, P = 3, 3, 0.6

    def __init__(self, arm, n, seed):
        self.arm = arm
        self.m, self.tw, self.ea = make(arm)
        self.rnd = random.Random(seed)
        self.reqs = []
        self.prev = []
        self.t_alloc, self.t_cache = [], []
        self.step_alloc, self.step_cache = [], []
        self.p_fast = {"alloc": [], "cache": []}
        self.p_full = {"alloc": [], "cache": []}
        for i in range(n):
            self.admit(i)

    def admit(self, i):
        m, rnd = self.m, self.rnd
        L = a.ctx + rnd.randrange(0, 4096)
        r = Request(f"r{i}", [rnd.randrange(1, 150000) for _ in range(L)], SamplingParams(max_tokens=10 ** 6),
                    None, block_hasher=HASHER)
        r.status = RequestStatus.WAITING_FOR_REMOTE_KVS
        assert m.allocate_slots(r, 0, num_external_computed_tokens=L - 1, delay_cache_blocks=True) is not None
        r.num_computed_tokens = L - 1
        m.cache_blocks(r, r.num_computed_tokens)
        r.status = RequestStatus.RUNNING
        r.append_output_token_ids([rnd.randrange(1, 150000)])
        self.reqs.append(r)

    def step(self, timed):
        m, rnd = self.m, self.rnd
        m.block_pool.__dict__.get("_tpu_unwritten_block_ids", set()).clear()
        m.new_step_starts()
        if self.ea is not None:
            ea_mod.flush()
            l6_ab.begin_step("engine_core")
        pc = time.perf_counter_ns
        ea = self.ea
        cur, ta = [], []
        for r in self.reqs:
            k = r.num_tokens + len(r.spec_token_ids) + r.num_output_placeholders - r.num_computed_tokens
            f0 = ea.n["fast"] if ea is not None else 0
            t0 = pc()
            res = m.allocate_slots(r, k, num_lookahead_tokens=self.LA)
            dt = pc() - t0
            ta.append(dt)
            if timed and ea is not None:
                (self.p_fast if ea.n["fast"] > f0 else self.p_full)["alloc"].append(dt)
            assert res is not None
            nspec = max(0, min(k + r.num_computed_tokens - r.num_tokens - r.num_output_placeholders,
                               len(r.spec_token_ids)))
            r.spec_token_ids = []
            cur.append((r, k, nspec))
        for r, k, nspec in cur:
            r.num_computed_tokens += k
            r.num_in_flight_tokens += k
            r.num_output_placeholders += 1 + nspec
            r.spec_token_ids = [-1] * self.SPEC
        tc = []
        for r, k, nspec in self.prev:
            r.num_in_flight_tokens -= k
            acc = 0
            while acc < nspec and rnd.random() < self.P:
                acc += 1
            rej = nspec - acc
            r.num_computed_tokens -= rej
            r.num_output_placeholders -= rej
            toks = [rnd.randrange(1, 150000) for _ in range(1 + acc)]
            r.append_output_token_ids(toks)
            r.num_output_placeholders -= len(toks)
            f0 = ea.n["fast"] if ea is not None else 0
            t0 = pc()
            m.cache_blocks(r, r.num_computed_tokens - r.num_output_placeholders)
            dt = pc() - t0
            tc.append(dt)
            if timed and ea is not None:
                (self.p_fast if ea.n["fast"] > f0 else self.p_full)["cache"].append(dt)
        self.prev = cur
        if timed:
            self.t_alloc += ta
            self.t_cache += tc
            self.step_alloc.append(sum(ta) / len(ta) / 1e3)
            if tc:
                self.step_cache.append(sum(tc) / len(tc) / 1e3)

    def fingerprint(self):
        m = self.m
        pool = m.block_pool
        q = pool.free_block_queue
        order, b = [], q.fake_free_list_head.next_free_block
        while b is not None and b is not q.fake_free_list_tail:
            order.append(b.block_id)
            b = b.next_free_block
        tables = [[[x.block_id for x in sm.req_to_blocks.get(r.request_id, ())] for sm in
                   m.coordinator.single_type_managers] for r in self.reqs]
        st_ = (order, tables,
               sorted((k, sorted([v.block_id] if hasattr(v, "block_id") else list(v)))
                      for k, v in pool.cached_block_hash_to_block._cache.items()),
               [(x.block_id, x.ref_cnt, x.block_hash, x.block_hash_num_tokens) for x in pool.blocks],
               sorted(pool.__dict__.get("_tpu_unwritten_block_ids", ())))
        return hashlib.sha256(repr(st_).encode()).hexdigest()[:16]


def run(arm, n, steps, warm, seed):
    gc.collect()
    gc.disable()
    try:
        d = Drive(arm, n, seed)
        for _ in range(warm):
            d.step(False)
        for _ in range(steps):
            d.step(True)
    finally:
        gc.enable()
    return d


def summ(xs):
    return (st.median(xs), st.mean(xs)) if xs else (float("nan"), float("nan"))


rows = []
print(f"BENCH ctx {a.ctx} steps {a.steps} warm {a.warm} seed {a.seed} python {sys.version.split()[0]} "
      f"torch {torch.__version__} nproc {os.cpu_count()}")


def med(xs):
    return st.median(xs) / 1e3 if xs else float("nan")


for geom, plan in GEOMS:
    GROUPS, SCHED_BS, HASH_BS = plan if plan is not None else mk512()
    HASHER = get_request_block_hasher(HASH_BS, fn)
    print(f"BENCH GEOMETRY {geom}: {len(GROUPS)} groups, sizes "
          f"{dict(Counter((type(g.kv_cache_spec).__name__, g.kv_cache_spec.block_size) for g in GROUPS))}, "
          f"scheduler {SCHED_BS} hash {HASH_BS}")
    for n in [int(x) for x in a.n.split(",")]:
        res = {}
        for arm in (("base", "twins", "fast") if geom == "builder" else ("twins", "fast")):
            d = run(arm, n, a.steps, a.warm, a.seed + n)
            sa, sc = summ(d.step_alloc), summ(d.step_cache)
            ma = st.mean(d.t_alloc) / 1e3
            mc = st.mean(d.t_cache) / 1e3 if d.t_cache else float("nan")
            eng = ""
            if d.ea is not None:
                nf, nu = len(d.p_fast["alloc"]) + len(d.p_fast["cache"]), len(d.p_full["alloc"]) + len(d.p_full["cache"])
                eng = (f" engalloc fast={d.ea.n['fast']} full={d.ea.n['full']} mismatch={d.ea.n['mismatch']}; timed calls "
                       f"fast share {nf / max(nf + nu, 1):.3f}; median us fast-path alloc {med(d.p_fast['alloc']):.1f} "
                       f"cache {med(d.p_fast['cache']):.1f}, stock-path alloc {med(d.p_full['alloc']):.1f} "
                       f"cache {med(d.p_full['cache']):.1f}")
            fp = d.fingerprint() if arm != "base" else "-"
            res[arm] = (sa, sc, ma, mc, fp)
            print(f"BENCH {geom} N={n:3d} {arm:5s} allocate_slots us/call median-of-steps {sa[0]:7.1f} mean {ma:7.1f} | "
                  f"cache_blocks us/call median-of-steps {sc[0]:7.1f} mean {mc:7.1f} | per request-step (mean) "
                  f"{ma + mc:7.1f} | fingerprint {fp}{eng}")
        same = res["twins"][4] == res["fast"][4]
        print(f"BENCH {geom} N={n:3d} IDENTITY twins vs fast (same seed, {a.steps + a.warm} steps): "
              f"{'IDENTICAL' if same else 'DIFFERENT'}")
        rows.append((geom, n, res, same))
GROUPS, SCHED_BS, HASH_BS = BUILT
HASHER = get_request_block_hasher(HASH_BS, fn)

# Check arm: every predicted-quiet call re-run on the stock path and compared (CHECK=1) at the profile N.
d = run("fastchk", a.profile_n, a.steps, a.warm, a.seed + a.profile_n)
twins_fp = {n: res["twins"][4] for g, n, res, _ in rows if g == "builder"}.get(a.profile_n, "-")
chk_ok = d.ea.n["mismatch"] == 0 and not d.ea.off and d.fingerprint() == twins_fp
print(f"BENCH N={a.profile_n} CHECK=1 arm: check={d.ea.n['check']} fast={d.ea.n['fast']} full={d.ea.n['full']} "
      f"mismatch={d.ea.n['mismatch']} off={d.ea.off} fingerprint {d.fingerprint()} vs twins {twins_fp}: "
      f"{'OK' if chk_ok else 'FAIL'}")

# Profile pass: where the twins arm's time goes (coordinator loops, then manager kinds).
acc = defaultdict(lambda: [0, 0])


def timed(key, f):
    def w(*x, **k):
        t0 = time.perf_counter_ns()
        try:
            return f(*x, **k)
        finally:
            v = acc[key]
            v[0] += time.perf_counter_ns() - t0
            v[1] += 1

    return w


d = Drive("twins", a.profile_n, a.seed + 1000)
for _ in range(a.warm):
    d.step(False)
m = d.m
coord = m.coordinator
for name in ("remove_skipped_blocks", "get_num_blocks_to_allocate", "allocate_new_blocks", "cache_blocks"):
    setattr(coord, name, timed(("coord", name), getattr(coord, name)))
kinds = {}
for sm in coord.single_type_managers:
    kind = type(sm).__name__ + f"/{sm.block_size}"
    kinds[kind] = kinds.get(kind, 0) + 1
    for name in ("remove_skipped_blocks", "get_num_blocks_to_allocate", "allocate_new_blocks", "cache_blocks"):
        setattr(sm, name, timed((kind, name), getattr(sm, name)))
m.allocate_slots = timed(("kvm", "allocate_slots"), m.allocate_slots)
m.cache_blocks = timed(("kvm", "cache_blocks"), m.cache_blocks)
for _ in range(a.steps):
    d.step(False)
na = acc[("kvm", "allocate_slots")][1]
nc = acc[("kvm", "cache_blocks")][1]
print(f"PROFILE twins arm N={a.profile_n}, {a.steps} steps (timers add overhead; read shares, not absolutes); "
      f"manager kinds {kinds}; twin classes {len(d.tw.classes)} {[len(c.twins) + 1 for c in d.tw.classes]}")
print(f"PROFILE KVCacheManager.allocate_slots {acc[('kvm', 'allocate_slots')][0] / na / 1e3:.1f} us/call ({na} calls); "
      f"cache_blocks {acc[('kvm', 'cache_blocks')][0] / max(nc, 1) / 1e3:.1f} us/call ({nc} calls)")
for key in sorted(acc, key=lambda k: -acc[k][0]):
    if key[0] == "kvm":
        continue
    tot, cnt = acc[key]
    per = na if key[0] == "coord" and key[1] != "cache_blocks" else (na + nc)
    print(f"PROFILE {key[0]:28s} {key[1]:28s} {tot / 1e3 / max(per, 1):8.2f} us per KVCacheManager call "
          f"({cnt} calls, {tot / max(cnt, 1) / 1e3:.2f} us each)")

import cProfile  # noqa: E402
import io  # noqa: E402
import pstats  # noqa: E402

d = Drive("twins", a.profile_n, a.seed + 2000)
for _ in range(a.warm):
    d.step(False)
pr = cProfile.Profile()
pr.enable()
for _ in range(min(a.steps, 200)):
    d.step(False)
pr.disable()
s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(30)
print("PROFILE cProfile (twins arm, tottime top 30):")
print("\n".join("PROFILE| " + ln for ln in s.getvalue().splitlines() if ln.strip()))

print("BENCH SUMMARY geometry N | base alloc | twins alloc | fast alloc | twins cache | fast cache | twins per req-step "
      "| fast per req-step (us, mean over all calls) | identity")
for g, n, res, same in rows:
    b = res.get("base", (0, 0, float("nan")))[2]
    print(f"BENCH SUMMARY {g:7s} {n:3d} | {b:7.1f} | {res['twins'][2]:7.1f} | {res['fast'][2]:6.1f} | "
          f"{res['twins'][3]:7.1f} | {res['fast'][3]:6.1f} | {res['twins'][2] + res['twins'][3]:7.1f} | "
          f"{res['fast'][2] + res['fast'][3]:6.1f} | {'IDENTICAL' if same else 'DIFFERENT'}")
ok = all(same for _, _, _, same in rows) and chk_ok
print(f"BENCH_DONE identity={'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
