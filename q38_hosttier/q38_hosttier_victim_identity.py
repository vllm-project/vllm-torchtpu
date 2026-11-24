#!/usr/bin/env python3
"""Victim host-tier bytes identity on the served PD PP8 prefill (q38 S13 L1; prereg q38_hosttier_victim_identity_prereg.md).

Runs in the infx-qwen38-flash-bench pod (BENCH=script, the served prefill env), from the tree root:

    python3 q38_hosttier/q38_hosttier_victim_identity.py --out-dir "$CDK_OUTPUT_DIR/q38f-bench/script"

One offline engine (child process), i3's served prefill flags (q38_hosttier_identity.engine_argv, arm `multi`:
TPUMultiConnector[Qwen4ExpRaidenConnector kv_producer + TPURaidenOffloadingConnector kv_both]) plus, set here:
  QWEN4_EXP_HOST_TIER_POLICY=victim   store on HBM eviction only, keep on load;
  QWEN4_EXP_PD_STAGING_LEASES=16      bounded PD staging, so both native registrations run as served;
and, from the launcher, a small pool so allocation evicts: QWEN4_EXP_STATIC_POOL_FILL=1 with
QWEN4_EXP_MAX_POOL_BYTES = 129 x the busiest stage's bytes per block (fill 0 would plan min(92, cap) pages).
Per prompt k in {0, 1, 2} (A_k = 32 x 3,200 + 2 random tokens, A'_k = A_k + 2,000):
  cold   A_k: computed; its 32 KV blocks of both token groups and the 4 state checkpoints are hashed on every
         stage (i3's capture); no store may happen (telemetry stores / store_keys / victim_store_keys deltas 0)
  age    one 16-chunk filler (51,201 tokens; then 128-token fillers if short) so >= 16 scheduler steps pass
  evict  102,401-token eviction prompts (at most 4) until every one of A_k's block hashes was victim-stored
         (drained) and none is left in the HBM prefix cache
  host   A'_k: local hit 0, host hit 102,400; the loaded pages are hashed and compared with cold's
k = 2 is the MUTANT: A_2's checkpoint victims are stored from another victim's block, so its state pages
must mismatch on the host load.
--selftest checks the verdict on synthetic records; --read FILE recomputes the verdict from a saved JSON.
"""

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import q38_hosttier_identity as I  # noqa: E402

BLOCK = I.BLOCK
# Addendum 1 (j-266f4814 read-out): tpu_sync bounds a storage only when leases x hint < its num_blocks
# (lane_xfer.md:865, C); at 16 leases x hint 86-87 (M) the 248-block pool kept the full mirror. 2 leases: 174 < 248.
LEASES = 2
HINT_MAX = 87               # M: j-266f4814 "host staging" lines, hint 86 (PP1-7) / 87 (PP0) at max-model-len 131072
POOL_BLOCKS = 248           # C: static fill 1 -> max(plan, cap pages) = max(1 + 6 x 41 + 1, 129); M: "blocks=248"
POLICY_ENV = {"QWEN4_EXP_HOST_TIER_POLICY": "victim", "QWEN4_EXP_PD_STAGING_LEASES": str(LEASES)}
POOL_IDS = (200, 300)       # usable IDs (247 expected): evictions within MAX_EVICT prompts, leases x hint below it
AGE_STEPS = 16              # >= the young-block guard (PP + 2 = 10)
MAX_EVICT = 6               # j-266f4814 needed 3 per prompt at 247 IDs
POLICY_LINE = "host-tier policy victim"
BOUNDED_LINE = "mode=bounded"
STAGING_LINE = "host staging"
BOUNDED_RANKS = 8           # one "host staging" line per pipeline stage
MARK_KEYS = (POLICY_LINE, STAGING_LINE, "drain timed out")


def scan_markers(lines):
    """The engagement lines the verdict reads. vLLM logs to stdout (workers inherit it), so the parent must feed
    the child's stdout as well as its stderr file (j-266f4814 scanned stderr only and found none)."""
    return [line.rstrip("\n") for line in lines if any(m in line for m in MARK_KEYS)]


def mutate_keep(keep, mutant_keys, n_blocks):
    """The k = 2 mutant: every kept mutant key is stored from a wrong source block -- another kept victim's block
    if the step has one, else a fixed other (never null) block. j-266f4814 swapped only within a step and the
    checkpoint victims, evicted last, came alone: mutant_swapped 0."""
    others = [b for k, b in keep if k not in mutant_keys]
    out, swapped = [], 0
    for k, b in keep:
        if k in mutant_keys:
            alt = others[0] if others else (b % (n_blocks - 1)) + 1
            if alt == b:
                alt = (alt % (n_blocks - 1)) + 1
            b, swapped = alt, swapped + 1
        out.append((k, b))
    return out, swapped


def child(a):
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["TPU_P2P_WAIT_PULL_TIMEOUT"] = str(I.TTL_S)
    os.environ.setdefault("RAIDEN_DISABLE_SINGLETON_WORKER", "1")
    os.environ.update(POLICY_ENV)
    from vllm_torchtpu.offload import victim as V
    if a.served_order:
        # The served (multi-process) engine core installs the victim hook at engine_core and only then runs
        # platform_activation, whose LIFO free patch replaces BlockPool.free_blocks (j-47fee8a1: every victim
        # unstamped). Installing the hook before the engine (and so before platform activation) reproduces that order.
        V.patch_block_pool_victims()
    import numpy as np
    from vllm import SamplingParams
    from vllm.engine.arg_utils import EngineArgs
    from vllm.inputs import TokensPrompt
    from vllm.usage.usage_lib import UsageContext
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.v1.core.kv_cache_utils import get_block_hash, get_group_id
    from vllm.v1.engine.llm_engine import LLMEngine
    from vllm.v1.kv_cache_interface import MambaSpec
    from vllm.v1.kv_offload.base import make_offload_key

    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen3.8-Flash-Next-FP8")
    model_dir = next((d for d in (f"/vllm-cache/{model_name}", "/vllm-cache", f"/root/.cache/huggingface/{model_name}")
                      if os.path.isfile(f"{d}/config.json")), None)
    rec = {"arm": "multi", "policy_env": dict(POLICY_ENV), "events": [], "prompts": [], "mem_before": I.mem(),
           "tier_bytes": a.cpu_bytes}
    emit = lambda **k: (rec["events"].append(k), print("Q38_VID_EVENT " + json.dumps(k), flush=True))
    if model_dir is None:
        rec["error"] = "model dir not found"
        return rec
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    ns, unknown = parser.parse_known_args(I.engine_argv(model_dir, "multi", a.cpu_bytes))
    if unknown:
        rec["error"] = f"unknown engine args {unknown}"
        return rec
    t0 = time.time()
    eng = LLMEngine.from_engine_args(EngineArgs.from_cli_args(ns), usage_context=UsageContext.LLM_CLASS)
    rec["engine_init_s"] = round(time.time() - t0, 1)
    sched = eng.engine_core.engine_core.scheduler
    connector = sched.connector
    children = list(getattr(connector, "_connectors", [connector]))
    rec["connector"] = type(connector).__name__
    rec["children"] = [type(c).__name__ for c in children]
    offload = next(c for c in children if type(c).__name__ == "TPURaidenOffloadingConnector")
    osched = offload.connector_scheduler
    kvc = sched.kv_cache_config
    kvm = sched.kv_cache_manager
    pool = kvm.block_pool
    mamba = {g for g, grp in enumerate(kvc.kv_cache_groups) if isinstance(grp.kv_cache_spec, MambaSpec)}
    cacheable = list(kvc.prefix_cacheable_group_ids)
    from vllm.v1.core.block_pool import BlockPool as _BP
    from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue as _FQ
    rec["free_blocks_owner"] = getattr(_BP.free_blocks, "__qualname__", "?")
    rec["queue_stamp"] = bool(_FQ.__dict__.get("_q38_victim_stamp", False))
    rec["served_order"] = bool(a.served_order)
    rec.update(pool_ids=int(kvc.num_blocks) - 1, cacheable_groups=cacheable, mamba_groups=sorted(mamba),
               victim_min_age=getattr(osched, "_victim_min_age", None), mem_ready=I.mem())
    emit(phase="ready", pool_ids=rec["pool_ids"], children=rec["children"], min_age=rec["victim_min_age"])

    tel = lambda: dict(osched.telemetry)
    # -- victim tap: what was kept (stored) and at what age; the k = 2 mutant ------------------------------
    stored = {}                 # offload key -> (block id, age)
    state = {"mutant_keys": set(), "swapped": 0}
    orig_select = V.select_victims

    def select(victims, step, min_age, groups):
        keep, counts = orig_select(victims, step, min_age, groups)
        age = {make_offload_key(get_block_hash(v.block_hash), get_group_id(v.block_hash)):
               (None if v.freed_step is None else step - v.freed_step) for v in victims}
        if state["mutant_keys"]:
            keep, n = mutate_keep(keep, state["mutant_keys"], int(kvc.num_blocks))
            state["swapped"] += n
        for k, b in keep:
            stored[k] = (b, age.get(k))
        return keep, counts

    V.select_victims = select
    keyof = lambda h: make_offload_key(get_block_hash(h), get_group_id(h))
    in_hbm = lambda h: pool.cached_block_hash_to_block.get_one_block(h) is not None

    matched, freed, seen_state = {}, {}, {}
    orig_matched = offload.get_num_new_matched_tokens

    def m(request, num_computed_tokens, *args, **kwargs):
        res = orig_matched(request, num_computed_tokens, *args, **kwargs)
        if res[0] is not None:
            matched[request.request_id] = (int(num_computed_tokens), int(res[0] or 0))
        return res

    offload.get_num_new_matched_tokens = m
    orig_free = kvm.free

    def free(request):
        ids = [list(x) for x in kvm.get_block_ids(request.request_id)]
        hashes = {}
        for g in cacheable:
            for b in ids[g]:
                h = pool.blocks[b].block_hash if b else None
                if h is not None:
                    hashes[f"{g}|{b}"] = h
        freed[request.request_id] = (ids, hashes)
        return orig_free(request)

    kvm.free = free
    sp = SamplingParams(max_tokens=1, temperature=0.0, detokenize=False, ignore_eos=True)

    def track(req_id):
        for g in mamba:
            r2b = kvm.coordinator.single_type_managers[g].req_to_blocks
            key = req_id if req_id in r2b else next((k for k in r2b if k.startswith(req_id + "-")), None)
            blocks = r2b.get(key, []) if key else []
            if len(blocks) >= nblocks and not blocks[nblocks - 1].is_null and blocks[nblocks - 1].block_id:
                seen_state.setdefault(req_id, {})[g] = int(blocks[nblocks - 1].block_id)

    def run(req_id, tokens):
        eng.add_request(req_id, TokensPrompt(prompt_token_ids=tokens), sp)
        done = False
        while not done:
            done = any(o.request_id == req_id and o.finished for o in eng.step())
            track(req_id)
        local, external = I.by_request(matched, req_id) or (None, None)
        r = {"req_id": req_id, "local_hit": local, "external_hit": external}
        emit(phase="request", **r)
        return r

    def wait_free(req_id, timeout=60):
        t_end = time.time() + timeout
        while I.by_request(freed, req_id) is None and time.time() < t_end:
            track(req_id)
            eng.step()
            time.sleep(0.05)
        return I.by_request(freed, req_id)

    def page_hashes(table, req_id):
        groups = {}
        for g in cacheable:
            if g in mamba:
                b, _ = I.state_checkpoint(table[g], seen_state.get(req_id, {}).get(g), nblocks)
                groups[str(g)] = [(nblocks - 1, b)] if b else []
            else:
                groups[str(g)] = [(i, table[g][i]) for i in range(nblocks)]
        out = {}
        for stage, part in enumerate(eng.collective_rpc(I._q38_page_hashes, args=(groups,))):
            out.update({f"s{stage}|{k}": v for k, v in part.items()})
        return out

    rng = np.random.default_rng(a.seed)
    vocab = min(150000, eng.vllm_config.model_config.get_vocab_size())
    nblocks = a.blocks
    for k in range(a.prompts + 1):
        mutant = k == a.prompts
        base = rng.integers(1000, vocab, nblocks * BLOCK + 2).tolist()
        ext = base + rng.integers(1000, vocab, a.suffix).tolist()
        p = {"k": k, "mutant": mutant}
        t_before = tel()
        p["cold"] = run(f"p{k}-cold", base)
        got = wait_free(f"p{k}-cold")
        t_after = tel()
        p["cold_store_delta"] = {x: t_after.get(x, 0) - t_before.get(x, 0) for x in ("stores", "store_keys", "victim_store_keys")}
        if got is None:
            p["error"] = "no block table for the cold request"
            rec["prompts"].append(p)
            break
        table, hashes = got
        p["cold_pages"] = page_hashes(table, f"p{k}-cold")
        own = {}
        for g in cacheable:
            cols = range(nblocks) if g not in mamba else [nblocks - 1]
            for i in cols:
                b = table[g][i] if i < len(table[g]) else 0
                if g in mamba and not b:
                    b = seen_state.get(f"p{k}-cold", {}).get(g, 0)
                h = hashes.get(f"{g}|{b}") or (pool.blocks[b].block_hash if b else None)
                if h is not None:
                    own[keyof(h)] = (g, h)
        p["own_keys"] = len(own)
        p["own_stored_at_cold"] = sum(1 for key in own if key in stored)
        if mutant:
            state["mutant_keys"] = {key for key, (g, _) in own.items() if g in mamba}
        # Age A_k: one filler of AGE_STEPS chunks (16 x 3,200 + 1 tokens, ~41 IDs, fits beside A_k's 68 without
        # evicting it; LRU evicts older leftovers first), then 128-token fillers only if the steps fell short.
        step0 = osched._victim_tap.step
        run(f"p{k}-age0", rng.integers(1000, vocab, AGE_STEPS * BLOCK + 1).tolist())
        wait_free(f"p{k}-age0")
        n = 1
        while osched._victim_tap.step - step0 < AGE_STEPS and n < 8:
            run(f"p{k}-age{n}", rng.integers(1000, vocab, 128).tolist())
            n += 1
        p["age_steps"] = osched._victim_tap.step - step0
        p["evict_prompts"] = 0
        while p["evict_prompts"] < MAX_EVICT:
            run(f"p{k}-evict{p['evict_prompts']}", rng.integers(1000, vocab, nblocks * BLOCK + 1).tolist())
            wait_free(f"p{k}-evict{p['evict_prompts']}")
            p["evict_prompts"] += 1
            if all(key in stored for key in own) and not any(in_hbm(h) for _, h in own.values()):
                break
        p["own_stored"] = sum(1 for key in own if key in stored)
        p["own_in_hbm"] = sum(1 for _, h in own.values() if in_hbm(h))
        ages = [stored[key][1] for key in own if key in stored]
        p["own_min_age"] = min((x for x in ages if x is not None), default=None)
        p["own_unstamped"] = sum(1 for x in ages if x is None)
        state["mutant_keys"] = set()
        p["mutant_swapped"] = state["swapped"] if mutant else 0
        p["host"] = run(f"p{k}-host", ext)
        got = wait_free(f"p{k}-host")
        if got is not None:
            hp = page_hashes(got[0], f"p{k}-host")
            common = sorted(set(p["cold_pages"]) & set(hp))
            mism = [x for x in common if p["cold_pages"][x] != hp[x]]
            p.update(pages_compared=len(common), pages_only_one_side=len(set(p["cold_pages"]) ^ set(hp)),
                     page_mismatches=len(mism),
                     page_mismatches_state=sum(1 for x in mism if int(x.split("|")[1]) in mamba),
                     state_pages_compared=sum(1 for x in common if int(x.split("|")[1]) in mamba),
                     page_mismatch_examples=mism[:5])
        del p["cold_pages"]
        rec["prompts"].append(p)
        emit(phase="prompt_done", **{x: p.get(x) for x in ("k", "own_keys", "own_stored", "own_in_hbm", "own_min_age",
                                                           "evict_prompts", "pages_compared", "page_mismatches",
                                                           "page_mismatches_state")})
    rec["telemetry"] = tel()
    rec["mem_end"] = I.mem()
    return rec


def verdict(rec, prompts=2, blocks=32, log_text=""):
    """The prereg's validity clauses and pass rule, on a child record and the child's log."""
    hit = blocks * BLOCK
    nkeys = blocks * 2 + 4
    ps = rec.get("prompts", [])
    ident = [p for p in ps if not p.get("mutant")]
    mut = next((p for p in ps if p.get("mutant")), None)
    tel = rec.get("telemetry", {})
    v = {"error": rec.get("error")}
    v["engine_up"] = (rec.get("connector") == "TPUMultiConnector"
                      and {"Qwen4ExpRaidenConnector", "TPURaidenOffloadingConnector"} <= set(rec.get("children", []))
                      and POLICY_LINE in log_text)
    staging = [line for line in log_text.splitlines() if STAGING_LINE in line]
    v["bounded_staging"] = (sum(1 for line in staging if BOUNDED_LINE in line) >= BOUNDED_RANKS
                            and not any("mode=full" in line for line in staging))
    v["pool_ok"] = isinstance(rec.get("pool_ids"), int) and POOL_IDS[0] <= rec["pool_ids"] <= POOL_IDS[1]
    # Write-through = the compute-time store counters. A cold step may evict OLDER blocks to make room, and those
    # victim stores are the policy working (j-266f4814: victim_store_keys +96 on cold k = 1, 2), so they are not counted.
    v["write_through_off"] = len(ps) == prompts + 1 and tel.get("stores", 1) == 0 and tel.get("store_keys", 1) == 0 \
        and all(p.get("cold_store_delta", {}).get("stores", 1) == 0 and p.get("cold_store_delta", {}).get("store_keys", 1) == 0
                and p.get("own_stored_at_cold", 1) == 0 for p in ps)
    v["evicted_to_host"] = len(ps) == prompts + 1 and all(
        p.get("own_keys") == nkeys and p.get("own_stored") == nkeys and p.get("own_in_hbm") == 0 for p in ps)
    v["aged"] = len(ps) == prompts + 1 and all(
        p.get("age_steps", 0) >= AGE_STEPS and p.get("own_unstamped", 1) == 0
        and (p.get("own_min_age") or 0) >= (rec.get("victim_min_age") or 10) for p in ps)
    v["host_hits_engaged"] = len(ident) == prompts and all(
        p.get("host", {}).get("local_hit") == 0 and p.get("host", {}).get("external_hit") == hit for p in ident)
    v["state_pages_ok"] = len(ident) == prompts and all(p.get("state_pages_compared", 0) > 0 for p in ident)
    v["drain_ok"] = tel.get("victim_store_failed", 0) == 0 and "drain timed out" not in log_text
    v["mutant_bites"] = bool(mut) and mut.get("mutant_swapped", 0) > 0 and mut.get("host", {}).get("external_hit") == hit \
        and mut.get("page_mismatches_state", 0) > 0
    v["pages_compared"] = sum(p.get("pages_compared", 0) for p in ident)
    v["page_mismatches"] = sum(p.get("page_mismatches", 0) + p.get("pages_only_one_side", 0) for p in ident)
    v["page_identity"] = v["pages_compared"] >= 1000 and v["page_mismatches"] == 0
    # Addendum 2: the served patch order must be in force -- LIFO's free_blocks on top of the hook -- and stamping must
    # come from the free queue (the fix); otherwise this run does not cover the path that failed in serving.
    v["served_order"] = bool(rec.get("served_order") and "_patch_vllm_block_pool_lifo_free" in str(rec.get("free_blocks_owner"))
                             and rec.get("queue_stamp"))
    v["valid"] = bool(v["served_order"] and v["engine_up"] and v["bounded_staging"] and v["pool_ok"] and v["write_through_off"]
                      and v["evicted_to_host"] and v["aged"] and v["host_hits_engaged"] and v["state_pages_ok"]
                      and v["drain_ok"] and v["mutant_bites"])
    v["verdict"] = ("PASS" if v["valid"] and v["page_identity"] else
                    "FAIL" if v["valid"] else "INVALID")
    v["failed_clauses"] = [c for c in ("served_order", "engine_up", "bounded_staging", "pool_ok", "write_through_off", "evicted_to_host",
                                       "aged", "host_hits_engaged", "state_pages_ok", "drain_ok", "mutant_bites")
                           if not v[c]]
    for x in ("victim_seen", "victim_young", "victim_unstamped", "victim_resident", "victim_stores",
              "victim_store_keys", "victim_drain_us", "victim_store_failed"):
        v[x] = tel.get(x)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(os.environ.get("CDK_OUTPUT_DIR", "/tmp/cdk_output"), "q38f-bench", "script"))
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--prompts", type=int, default=2)
    ap.add_argument("--blocks", type=int, default=32)
    ap.add_argument("--suffix", type=int, default=2000)
    ap.add_argument("--cpu-bytes", type=int, default=100 * 10**9)
    ap.add_argument("--seed", type=int, default=38)
    ap.add_argument("--timeout", type=int, default=4800)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--read", default="")
    ap.add_argument("--fake", action="store_true", help="CPU check of the parent: a stub child, no engine")
    ap.add_argument("--served-order", type=int, default=1, help="install the victim hook before the engine (served order)")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.read:
        out = json.load(open(a.read))
        v = verdict(out["record"], a.prompts, a.blocks, out.get("log_markers", ""))
        print("Q38_VID_REREAD " + json.dumps(v))
        return 0 if v["verdict"] == "PASS" else 3
    if a.child and a.fake:
        # vLLM-style engagement lines on stdout (as the engine prints them), noise on stderr, then the record
        for r in range(BOUNDED_RANKS):
            print(f"(Worker_PP{r} pid={100 + r}) INFO [pd_connector.py:583] Qwen PD producer {STAGING_LINE} rank0: "
                  f"{BOUNDED_LINE} leases={LEASES} hint=86 bounded_bytes_per_shard=1 full_mirror_bytes_per_shard=0", flush=True)
        print(f"(MainProcess pid=1) INFO [raiden_connector.py:296] TPURaidenOffloadingConnector: {POLICY_LINE} (store on "
              "HBM eviction, keep on load)", flush=True)
        print("stderr noise", file=sys.stderr, flush=True)
        print("Q38_VID_CHILD " + json.dumps(_good_record()), flush=True)
        return 0
    if a.child:
        try:
            rec = child(a)
        except Exception as e:  # noqa: BLE001
            import traceback
            rec = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-4000:]}
        print("Q38_VID_CHILD " + json.dumps(rec, default=str), flush=True)
        return 0
    os.makedirs(a.out_dir, exist_ok=True)
    logp = os.path.join(a.out_dir, "q38_hosttier_victim_identity_multi.log")
    argv = [sys.executable, os.path.abspath(__file__), "--child"] + [
        f"--{k.replace('_', '-')}={v}" for k, v in vars(a).items() if k in ("prompts", "blocks", "suffix", "cpu_bytes", "seed", "served_order")
    ] + (["--fake"] if a.fake else [])
    env, cache = I.arm_cache_env("victim")
    env.update(POLICY_ENV)
    rec, t, marks = None, time.time(), []
    with open(logp, "w") as log:
        p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=log, text=True, start_new_session=True, env=env)
        try:
            for line in p.stdout:
                print(line, end="", flush=True)
                marks += scan_markers([line])
                if line.startswith("Q38_VID_CHILD "):
                    rec = json.loads(line[len("Q38_VID_CHILD "):])
            p.wait(timeout=a.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, 9)
    rec = rec or {"error": "no child record (crash or timeout); see log"}
    rec["wall_s"] = round(time.time() - t, 1)
    markers = "\n".join(marks + scan_markers(open(logp, errors="replace").read().splitlines()))
    v = verdict(rec, a.prompts, a.blocks, markers)
    out = {"record": rec, "log_markers": markers, "verdict": v}
    json.dump(out, open(os.path.join(a.out_dir, "q38_hosttier_victim_identity.json"), "w"), indent=1, default=str)
    print("Q38_VID_VERDICT " + json.dumps(v), flush=True)
    if cache.startswith("/dev/shm/"):
        subprocess.run(["rm", "-rf", cache], check=False)
    return 0 if v["verdict"] == "PASS" else 3


_HIT, _NK = 32 * BLOCK, 68


def _prompt(k, mutant=False, **kw):
    p = {"k": k, "mutant": mutant, "cold_store_delta": {"stores": 0, "store_keys": 0, "victim_store_keys": 0},
         "own_keys": _NK, "own_stored": _NK, "own_in_hbm": 0, "own_min_age": 12, "own_unstamped": 0,
         "own_stored_at_cold": 0, "age_steps": 17, "evict_prompts": 3, "host": {"local_hit": 0, "external_hit": _HIT},
         "pages_compared": 805, "pages_only_one_side": 0, "page_mismatches": 0, "page_mismatches_state": 0,
         "state_pages_compared": 37, "mutant_swapped": 4 if mutant else 0}
    if mutant:
        p.update(page_mismatches=37, page_mismatches_state=37)
    p.update(kw)
    return p


def _good_record(ps=None, **kw):
    r = {"connector": "TPUMultiConnector", "children": ["Qwen4ExpRaidenConnector", "TPURaidenOffloadingConnector"],
         "pool_ids": POOL_BLOCKS - 1, "victim_min_age": 10,
         "prompts": ps if ps is not None else [_prompt(0), _prompt(1), _prompt(2, mutant=True)],
         "telemetry": {"victim_store_failed": 0, "stores": 0, "store_keys": 0},
         "served_order": True, "queue_stamp": True,
         "free_blocks_owner": "_patch_vllm_block_pool_lifo_free.<locals>.free_blocks"}
    r.update(kw)
    return r


def _log(bounded=BOUNDED_RANKS, full=0, policy=True):
    lines = [f"(Worker_PP{r} pid=1) INFO Qwen PD producer {STAGING_LINE} rank0: {BOUNDED_LINE} leases={LEASES}"
             for r in range(bounded)]
    lines += [f"(Worker_PP{r} pid=1) INFO Qwen PD producer {STAGING_LINE} rank0: mode=full leases=16" for r in range(full)]
    if policy:
        lines.append(f"(MainProcess pid=1) INFO TPURaidenOffloadingConnector: {POLICY_LINE} (store on HBM eviction)")
    return "\n".join(lines)


def selftest():
    prompt, rec, hit, nk, LOG = _prompt, _good_record, _HIT, _NK, _log()
    good = [prompt(0), prompt(1), prompt(2, mutant=True)]
    fails = []
    v = verdict(rec(good), log_text=LOG)
    if v["verdict"] != "PASS" or v["pages_compared"] != 1610:
        fails.append(("good", v))
    # victim stores of OLDER blocks during a cold step are the policy working (j-266f4814), not write-through
    vv = verdict(rec([prompt(0), prompt(1, cold_store_delta={"stores": 0, "store_keys": 0, "victim_store_keys": 96}),
                      good[2]]), log_text=LOG)
    if vv["verdict"] != "PASS":
        fails.append(("cold victim stores of older blocks must not invalidate", vv["failed_clauses"]))
    invalid = {   # the prereg's harness mutants, plus addendum 1's
        "cold step stored": (rec([prompt(0, cold_store_delta={"stores": 1, "store_keys": 4, "victim_store_keys": 0}),
                                  prompt(1), good[2]]), LOG),
        "write-through anywhere in the run": (rec(good, telemetry={"victim_store_failed": 0, "stores": 1, "store_keys": 4}), LOG),
        "own key stored at cold": (rec([prompt(0, own_stored_at_cold=1), prompt(1), good[2]]), LOG),
        "one key not resident": (rec([prompt(0, own_stored=nk - 1), prompt(1), good[2]]), LOG),
        "one key still in HBM": (rec([prompt(0, own_in_hbm=1), prompt(1), good[2]]), LOG),
        "victim stored young": (rec([prompt(0, own_min_age=9), prompt(1), good[2]]), LOG),
        "victim unstamped": (rec([prompt(0, own_unstamped=1), prompt(1), good[2]]), LOG),
        "not aged": (rec([prompt(0, age_steps=8), prompt(1), good[2]]), LOG),
        "mutant without state mismatch": (rec([prompt(0), prompt(1), prompt(2, True, page_mismatches_state=0)]), LOG),
        "mutant not applied": (rec([prompt(0), prompt(1), prompt(2, True, mutant_swapped=0)]), LOG),
        "pool too large": (rec(good, pool_ids=301), LOG),
        "pool too small (rev 2's 128)": (rec(good, pool_ids=128), LOG),
        "host hit not engaged": (rec([prompt(0, host={"local_hit": 0, "external_hit": 0}), prompt(1), good[2]]), LOG),
        "host hit had local blocks": (rec([prompt(0, host={"local_hit": hit, "external_hit": 0}), prompt(1), good[2]]), LOG),
        "no state pages compared": (rec([prompt(0, state_pages_compared=0), prompt(1), good[2]]), LOG),
        "store failed": (rec(good, telemetry={"victim_store_failed": 1, "stores": 0, "store_keys": 0}), LOG),
        "staging full mirror on a rank (j-266f4814)": (rec(good), _log(bounded=7, full=1)),
        "8 bounded lines plus a full-mirror line": (rec(good), _log(bounded=8, full=1)),
        "staging lines on only 7 ranks": (rec(good), _log(bounded=7)),
        "no staging lines": (rec(good), _log(bounded=0)),
        "policy line missing": (rec(good), _log(policy=False)),
        "PD child missing": (rec(good, children=["TPURaidenOffloadingConnector"]), LOG),
        "served order not reproduced (free_blocks is the hook's, not LIFO's)": (rec(good, free_blocks_owner="patch_block_pool_victims.<locals>.free_blocks"), LOG),
        "no free-queue stamping (pre-fix tree)": (rec(good, queue_stamp=False), LOG),
        "served order off": (rec(good, served_order=False), LOG),
        "too few prompts": (rec([prompt(0), good[2]]), LOG),
    }
    for name, (r, log) in invalid.items():
        if verdict(r, log_text=log)["verdict"] != "INVALID":
            fails.append(name)
    for name, ps in {"corrupted state page": [prompt(0, page_mismatches=1, page_mismatches_state=1), prompt(1), good[2]],
                     "page on one side only": [prompt(0, pages_only_one_side=1), prompt(1), good[2]]}.items():
        if verdict(rec(ps), log_text=LOG)["verdict"] != "FAIL":
            fails.append(name)
    if verdict(rec([prompt(0, pages_compared=400), prompt(1, pages_compared=400), good[2]]), log_text=LOG)["verdict"] == "PASS":
        fails.append("too few pages")
    # the mutant applies even when the checkpoint victims come alone in their step (j-266f4814: swapped 0)
    alone, n = mutate_keep([("m1", 40), ("m2", 41)], {"m1", "m2"}, 248)
    if n != 2 or any(b == own or not 0 < b < 248 for (_, b), own in zip(alone, (40, 41))):
        fails.append(("mutant alone in its step", alone, n))
    mixed, n = mutate_keep([("k", 7), ("m", 9)], {"m"}, 248)
    if n != 1 or mixed != [("k", 7), ("m", 7)]:
        fails.append(("mutant with another victim", mixed, n))
    if mutate_keep([("k", 7)], {"m"}, 248) != ([("k", 7)], 0):
        fails.append("no mutant key: unchanged")
    if mutate_keep([("m", 247)], {"m"}, 248)[0][0][1] in (247, 0) or mutate_keep([("m", 1)], {"m"}, 248)[0][0][1] in (1, 0):
        fails.append("mutant edge blocks")
    # bounded staging is feasible on the planned pool (lane_xfer.md:865) and the planned pool passes pool_ok
    if not (LEASES * HINT_MAX < POOL_BLOCKS and POOL_IDS[0] <= POOL_BLOCKS - 1 <= POOL_IDS[1]):
        fails.append("leases x hint must be below the planned pool, and the planned pool inside pool_ok")
    # the parent captures engagement lines from the child's STDOUT (where vLLM logs), end to end, with a stub child
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ, CACHE_DIR=tmp)
        subprocess.run([sys.executable, os.path.abspath(__file__), "--fake", "--out-dir", tmp, "--timeout", "120"],
                       capture_output=True, text=True, env=env, timeout=180)
        try:
            out = json.load(open(os.path.join(tmp, "q38_hosttier_victim_identity.json")))
            ok = (out["verdict"]["verdict"] == "PASS" and POLICY_LINE in out["log_markers"]
                  and out["log_markers"].count(BOUNDED_LINE) == BOUNDED_RANKS)
        except (OSError, KeyError, ValueError) as e:
            ok, out = False, {"error": repr(e)}
        if not ok:
            fails.append(("parent stdout capture with a stub child", out.get("verdict", out)))
    print("selftest", "PASS" if not fails else f"FAIL {fails}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
