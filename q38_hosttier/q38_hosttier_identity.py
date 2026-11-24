#!/usr/bin/env python3
"""Engine-level host-tier identity test on the served PD PP8 prefill (HOSTTIER CP2).

Runs inside the infx-qwen38-flash-bench pod (BENCH=script, ENGINE=disagg-prefill env), from the
vllm-torchtpu tree root:

    python3 q38_hosttier/q38_hosttier_identity.py --out-dir "$CDK_OUTPUT_DIR/q38f-bench/script"

One fresh offline engine per arm (child process), the served prefill leg's flags with:
  * --kv-transfer-config: arm `multi` = TPUMultiConnector[Qwen4ExpRaidenConnector (kv_producer),
    TPURaidenOffloadingConnector (kv_both)], the served shape with a host tier; arm `offload` = the
    offload connector alone on the same PP8 engine (no PD), as a fallback that isolates the tier;
  * --max-num-batched-tokens 3200 (= one 3,200-token pool page), --max-num-seqs 1, so a cold
    prefill and a cache-hit prefill compute the same 3,200-token chunks in single-request batches
    and recompute can be compared bit for bit, not only HBM against host;
  * --max-model-len 131072; TPU_P2P_WAIT_PULL_TIMEOUT=5 (no decode pulls; a producer's finished
    request releases its blocks after 5 s instead of 300 s).
The engine core runs in this process (VLLM_ENABLE_V1_MULTIPROCESSING=0), so the scheduler-side
Raiden offload manager can be timed and, for the mutant, intercepted.

Per prompt k (A_k = 32 x 3,200 + 1 random tokens, A'_k = A_k + 2,000 tokens; greedy, 8 tokens,
top-5 logprobs):
  1 cold      A_k on an empty cache; its pages (32 blocks of the two token groups, the four
              state groups' checkpoint at 102,400) are hashed on every stage
  2 hbm       A'_k: an HBM hit of 102,400 tokens
  3 evict     the HBM prefix cache is reset (reset_prefix_cache; host store untouched)
  4 host      A'_k again: local hit 0, host hit 102,400 tokens; the loaded pages are hashed and
              compared with step 1's (one page per layer per block)
  5 recompute A'_k with a cache_salt: no hit at all
Same-configuration floors (q38 CACHE, i2 addendum 2): step 2b hbm2 repeats the HBM hit before the
reset, and step 5b recompute2 repeats the recompute with another salt. hbm vs hbm2 is the engine's
own run-to-run floor on the hit path, recompute vs recompute2 on the no-hit path.
Identity: hbm == host (token ids and top-5 logprob floats, bitwise), judged only where hbm == hbm2;
recompute == hbm, judged against the recompute floor.
Prompts 0 and 1 are the identity set (>= 1,000 pages). Prompt 2 is the MUTANT: its host load drops
every state-group key (GDN and PLE checkpoints), so the hit resumes on whatever the fresh state
blocks hold; its output must differ from its HBM hit, and its state pages must mismatch.

--selftest checks the verdict logic on synthetic records (no TPU).
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

BLOCK = 3200
TTL_S = 5

# The served prefill leg's flags (infx-qwen38-flash-agentx-disagg, j-1b95e0b3 / the A/A render),
# minus frontend-only flags, with the identity overrides above.
ADDITIONAL_CONFIG = ('{"qwen4_exp":{"engram_config":{"cpu_offload":true,"dp_shared_memory":false,'
                     '"embedding_across_dp":false},"hc_prefill_full_tile":256,"hc_prefill_norm_tile":128,'
                     '"hc_prefill_up_tile":128}}')


# Arms that run another arm's connectors with extra environment (q38 CACHE): `multi4g` is `multi`
# with libtpu's premapped DMA buffer at 4 GiB per worker instead of 16 GiB (env_override.py only
# sets the 16 GiB default when unset), the check that frees ~103 GB of host memory for the tier.
ARM_ENV = {"multi4g": {"TPU_PREMAPPED_BUFFER_SIZE": str(4 << 30)}}
ARM_CONNECTORS = {"multi4g": "multi"}


def connector_arm(arm):
    return ARM_CONNECTORS.get(arm, arm)


def kv_transfer(arm, cpu_bytes):
    arm = connector_arm(arm)
    offload = {"kv_connector": "TPURaidenOffloadingConnector",
               "kv_connector_module_path": "vllm_torchtpu.offload.raiden_connector",
               "kv_role": "kv_both", "kv_connector_extra_config": {"cpu_bytes_to_use": cpu_bytes}}
    if arm == "offload":
        return offload
    pd = {"kv_connector": "Qwen4ExpRaidenConnector",
          "kv_connector_module_path": "vllm_torchtpu.models.vllm.qwen4_exp.pd_connector",
          "kv_role": "kv_producer", "kv_port": 14579}
    return {"kv_connector": "TPUMultiConnector",
            "kv_connector_module_path": "vllm_torchtpu.distributed.kv_transfer.tpu_multi_connector",
            "kv_role": "kv_producer", "kv_port": 14579,
            "kv_connector_extra_config": {"connectors": [pd, offload]}}


def engine_argv(model_dir, arm, cpu_bytes):
    return [
        "--model", model_dir, "--tensor-parallel-size", "1", "--data-parallel-size", "1",
        "--pipeline-parallel-size", "8", "--max-model-len", "131072", "--quantization", "fp8",
        "--max-num-batched-tokens", str(BLOCK), "--max-num-seqs", "1", "--no-async-scheduling",
        "--prefill-schedule-interval", "1", "--enable-prefix-caching",
        "--gpu-memory-utilization", "0.92", "--kv-cache-dtype", "fp8", "--language-model-only",
        "--attention-backend", "CUSTOM", "--block-size", "128",
        "--limit-mm-per-prompt", '{"image":0,"video":0}', "--mamba-cache-mode", "align",
        "--trust-remote-code", "--additional-config", ADDITIONAL_CONFIG,
        "--kv-transfer-config", json.dumps(kv_transfer(arm, cpu_bytes)),
    ]


# ----------------------------------------------------------------------------------------------
# worker-side page hashing (runs in every stage worker through collective_rpc)
# ----------------------------------------------------------------------------------------------
def _q38_page_hashes(worker, group_blocks):
    """blake2b-64 of this stage's pages: {gid|layer|position: hex} for its own layers."""
    import hashlib as _h

    import torch as _t

    from vllm_torchtpu.kv_cache_materializer import layer_to_pool_index

    runner = getattr(worker, "model_runner", None) or worker.worker.model_runner
    cfg = runner.kv_cache_config
    raws = runner.kv_cache_raw_tensors
    index = layer_to_pool_index(cfg)
    out = {}
    for gid, blocks in group_blocks.items():
        for layer in cfg.kv_cache_groups[int(gid)].layer_names:
            raw = raws[index[layer]]
            for pos, block in blocks:
                page = raw[int(block)].cpu().contiguous().view(_t.uint8).numpy().tobytes()
                out[f"{gid}|{layer}|{pos}"] = _h.blake2b(page, digest_size=8).hexdigest()
    return out


def by_request(d, req_id):
    """Value for an external request id. The engine keys its requests by the
    external id plus a random suffix ("p0-cold-a786e59d", j-46084fb7), so the
    scheduler-side hooks see internal ids; match the exact id or id + "-"."""
    if req_id in d:
        return d[req_id]
    hits = [v for k, v in d.items() if k.startswith(req_id + "-")]
    return hits[0] if len(hits) == 1 else None


def state_checkpoint(row, seen_block, nblocks):
    """The state group's checkpoint block at column ``nblocks - 1`` and where it was found.

    The checkpoint-retention patch (core/mamba_checkpoint_retention.py) moves a request's
    published checkpoint out of its block table (into ``_tpu_pinned_checkpoints``) once the
    request moves past it, so the table captured at free often holds the null block there
    (i2, j-f0998179). The harness therefore also records, after every engine step, the block
    each state group's table holds at that column while the request runs (``seen_block``).
    """
    if len(row) >= nblocks and row[nblocks - 1]:
        return int(row[nblocks - 1]), "table"
    if seen_block:
        return int(seen_block), "seen"
    return None, None


def mem():
    """Container memory (cgroup v2) and host MemAvailable, bytes: the tier sizing input."""
    def read(path):
        try:
            return int(open(path).read().strip())
        except (OSError, ValueError):
            return None

    avail = None
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                avail = int(line.split()[1]) * 1024
    except OSError:
        pass
    return {"cgroup_current": read("/sys/fs/cgroup/memory.current"),
            "cgroup_peak": read("/sys/fs/cgroup/memory.peak"),
            "cgroup_max": read("/sys/fs/cgroup/memory.max"), "mem_available": avail}


# ----------------------------------------------------------------------------------------------
# child: one engine, one arm
# ----------------------------------------------------------------------------------------------
def child(a):
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["TPU_P2P_WAIT_PULL_TIMEOUT"] = str(TTL_S)
    os.environ.setdefault("RAIDEN_DISABLE_SINGLETON_WORKER", "1")
    import numpy as np
    from vllm import SamplingParams
    from vllm.engine.arg_utils import EngineArgs
    from vllm.inputs import TokensPrompt
    from vllm.usage.usage_lib import UsageContext
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.v1.engine.llm_engine import LLMEngine
    from vllm.v1.kv_cache_interface import MambaSpec

    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen3.8-Flash-Next-FP8")
    model_dir = next((d for d in (f"/vllm-cache/{model_name}", "/vllm-cache",
                                  f"/root/.cache/huggingface/{model_name}")
                      if os.path.isfile(f"{d}/config.json")), None)
    rec = {"arm": a.arm, "events": [], "loads": [], "stores": 0, "prompts": [], "mem_before": mem(),
           "tier_bytes": a.cpu_bytes, "premapped_bytes": os.environ.get("TPU_PREMAPPED_BUFFER_SIZE")}
    emit = lambda **k: (rec["events"].append(k), print("Q38_ID_EVENT " + json.dumps(k), flush=True))
    if model_dir is None:
        rec["error"] = "model dir not found"
        return rec
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    ns, unknown = parser.parse_known_args(engine_argv(model_dir, a.arm, a.cpu_bytes))
    if unknown:
        rec["error"] = f"unknown engine args {unknown}"
        return rec
    t0 = time.time()
    eng = LLMEngine.from_engine_args(EngineArgs.from_cli_args(ns), usage_context=UsageContext.LLM_CLASS)
    rec["engine_init_s"] = round(time.time() - t0, 1)
    core = eng.engine_core.engine_core
    sched = core.scheduler
    connector = sched.connector
    children = list(getattr(connector, "_connectors", [connector]))
    rec["connector"] = type(connector).__name__
    rec["children"] = [type(c).__name__ for c in children]
    offload = next(c for c in children if type(c).__name__ == "TPURaidenOffloadingConnector")
    mgr = offload.connector_scheduler._raiden_manager
    kvc = sched.kv_cache_config
    mamba_groups = {g for g, grp in enumerate(kvc.kv_cache_groups) if isinstance(grp.kv_cache_spec, MambaSpec)}
    cacheable = list(kvc.prefix_cacheable_group_ids)
    from vllm_torchtpu.offload.raiden_store import all_ranks_kv_bytes_per_block

    rec["bytes_per_key"] = all_ranks_kv_bytes_per_block()
    rec["cacheable_groups"] = cacheable
    rec["mamba_groups"] = sorted(mamba_groups)
    rec["mem_ready"] = mem()
    emit(phase="ready", engine_init_s=rec["engine_init_s"], connector=rec["connector"],
         children=rec["children"], bytes_per_key=rec["bytes_per_key"], num_blocks=kvc.num_blocks,
         mem=rec["mem_ready"], tier_bytes=a.cpu_bytes)

    # -- scheduler-side instrumentation: host load timing, external hits, the mutant -----------
    state = {"mutant": False, "load_t0": {}, "matched": {}}
    orig_submit_load = mgr.submit_load_job

    def submit_load_job(job_id, keys, device_block_ids, req_id, pinned=True):
        keys, device_block_ids = list(keys), list(device_block_ids)
        dropped = 0
        if state["mutant"]:
            keep = [i for i, k in enumerate(keys) if int.from_bytes(bytes(k)[-4:], "big") not in mamba_groups]
            dropped = len(keys) - len(keep)
            keys = [keys[i] for i in keep]
            device_block_ids = [device_block_ids[i] for i in keep]
        state["load_t0"][job_id] = (time.time(), len(keys), req_id, dropped)
        return orig_submit_load(job_id, keys, device_block_ids, req_id, pinned=pinned)

    mgr.submit_load_job = submit_load_job
    orig_poll = mgr.poll_finished_jobs

    def poll_finished_jobs():
        done = orig_poll()
        for job in done:
            start = state["load_t0"].pop(job.job_id, None)
            if start is not None:
                t_submit, n_keys, req_id, dropped = start
                rec["loads"].append({"req_id": req_id, "keys": n_keys, "dropped_keys": dropped,
                                     "s": round(time.time() - t_submit, 6),
                                     "failed_blocks": len(getattr(job, "failed_device_block_ids", []) or [])})
        return done

    mgr.poll_finished_jobs = poll_finished_jobs
    orig_matched = offload.get_num_new_matched_tokens

    def matched(request, num_computed_tokens, *args, **kwargs):
        res = orig_matched(request, num_computed_tokens, *args, **kwargs)
        if res[0] is not None:
            state["matched"][request.request_id] = (int(num_computed_tokens), int(res[0] or 0))
        return res

    offload.get_num_new_matched_tokens = matched
    freed = {}
    kvm = sched.kv_cache_manager
    orig_free = kvm.free

    def free(request):
        try:
            freed[request.request_id] = [list(ids) for ids in kvm.get_block_ids(request.request_id)]
        except Exception as e:  # noqa: BLE001
            freed[request.request_id] = f"error {e!r}"
        return orig_free(request)

    kvm.free = free

    sp = SamplingParams(max_tokens=a.max_tokens, temperature=0.0, logprobs=5, detokenize=False,
                        ignore_eos=True)

    seen_state = {}  # external request id -> {state group: block at column nblocks - 1}

    def track(req_id):
        """Record each state group's block at column nblocks - 1 while the request runs."""
        for g in mamba_groups:
            r2b = kvm.coordinator.single_type_managers[g].req_to_blocks
            key = req_id if req_id in r2b else next((k for k in r2b if k.startswith(req_id + "-")), None)
            if key is None:
                continue
            blocks = r2b[key]
            if len(blocks) >= nblocks and not blocks[nblocks - 1].is_null and blocks[nblocks - 1].block_id:
                seen_state.setdefault(req_id, {})[g] = int(blocks[nblocks - 1].block_id)

    def run(req_id, tokens, salt=None):
        prompt = TokensPrompt(prompt_token_ids=tokens)
        if salt:
            prompt["cache_salt"] = salt
        t = time.time()
        eng.add_request(req_id, prompt, sp)
        out = None
        while out is None:
            for o in eng.step():
                if o.request_id == req_id and o.finished:
                    out = o
            track(req_id)
        c = out.outputs[0]
        lps = [[(int(t_), float(lp.logprob)) for t_, lp in sorted(d.items(), key=lambda kv: kv[1].rank)]
               for d in (c.logprobs or [])]
        local, external = by_request(state["matched"], req_id) or (None, None)
        r = {"req_id": req_id, "tokens": list(c.token_ids), "logprobs": lps, "s": round(time.time() - t, 3),
             "num_cached_tokens": out.num_cached_tokens, "local_hit": local, "external_hit": external}
        emit(phase="request", **{k: v for k, v in r.items() if k != "logprobs"})
        return r

    def settle():
        """Step until the producer's 5 s holds lapse, then reset the HBM prefix cache."""
        t_end = time.time() + 120
        while time.time() < t_end:
            for _ in range(20):
                eng.step()
            if eng.reset_prefix_cache():
                return True
            time.sleep(0.5)
        return False

    def wait_blocks(req_id, timeout=60):
        """Block tables captured when the scheduler frees the request (after
        the producer's 5 s hold); keyed by the engine's internal id."""
        t_end = time.time() + timeout
        while by_request(freed, req_id) is None and time.time() < t_end:
            track(req_id)
            eng.step()
            time.sleep(0.05)
        return by_request(freed, req_id)

    def hashes(table, nblocks, req_id):
        groups, sources = {}, {}
        for g in cacheable:
            row = table[g]
            if g in mamba_groups:
                block, src = state_checkpoint(row, seen_state.get(req_id, {}).get(g), nblocks)
                groups[str(g)] = [(nblocks - 1, block)] if block else []
                sources[str(g)] = src
            else:
                groups[str(g)] = [(i, row[i]) for i in range(nblocks)]
        merged = {}
        for stage, part in enumerate(eng.collective_rpc(_q38_page_hashes, args=(groups,))):
            for k, v in part.items():
                merged[f"s{stage}|{k}"] = v
        return groups, merged, sources

    rng = np.random.default_rng(a.seed)
    vocab = min(150000, eng.vllm_config.model_config.get_vocab_size())
    nblocks = a.blocks
    for k in range(a.prompts + 1):
        mutant = k == a.prompts
        # +2: a PD producer drops the last prompt token at admission (P-side
        # truncation, tpu_connector.py _maybe_truncate_prompt_for_handoff), so
        # it computes blocks*B + 1 tokens and checkpoints at blocks*B.
        base = rng.integers(1000, vocab, nblocks * BLOCK + 2).tolist()
        ext = base + rng.integers(1000, vocab, a.suffix).tolist()
        p = {"k": k, "mutant": mutant}
        p["cold"] = run(f"p{k}-cold", base)
        table_a = wait_blocks(f"p{k}-cold")
        if not isinstance(table_a, list):
            p["error"] = f"no block table for cold request: {table_a}"
            rec["prompts"].append(p)
            break
        groups_a, hash_a, src_a = hashes(table_a, nblocks, f"p{k}-cold")
        p["mamba_ckpt_present"] = all(groups_a[str(g)] for g in mamba_groups if g in cacheable)
        p["ckpt_source_cold"] = src_a
        p["hbm"] = run(f"p{k}-hbm", ext)
        p["hbm2"] = run(f"p{k}-hbm2", ext)  # same path, same configuration: the hit-path floor
        p["reset_ok"] = settle()
        state["mutant"] = mutant
        p["host"] = run(f"p{k}-host", ext)
        state["mutant"] = False
        table_h = wait_blocks(f"p{k}-host")
        if isinstance(table_h, list):
            _, hash_h, src_h = hashes(table_h, nblocks, f"p{k}-host")
            p["ckpt_source_host"] = src_h
            common = sorted(set(hash_a) & set(hash_h))
            mism = [key for key in common if hash_a[key] != hash_h[key]]
            p["pages_compared"] = len(common)
            p["pages_only_one_side"] = len(set(hash_a) ^ set(hash_h))
            p["page_mismatches"] = len(mism)
            p["page_mismatches_state"] = sum(1 for key in mism if int(key.split("|")[1]) in mamba_groups)
            p["state_pages_compared"] = sum(1 for key in common if int(key.split("|")[1]) in mamba_groups)
            p["page_mismatch_examples"] = mism[:5]
        else:
            p["error_host_table"] = str(table_h)
        if not mutant:
            p["recompute"] = run(f"p{k}-recompute", ext, salt=f"q38-salt-{k}")
            p["recompute2"] = run(f"p{k}-recompute2", ext, salt=f"q38-salt2-{k}")  # no-hit floor
        settle()
        rec["prompts"].append(p)
        p["mem"] = mem()
        emit(phase="prompt_done", k=k, mem=p["mem"], **{x: p.get(x) for x in (
            "pages_compared", "page_mismatches", "page_mismatches_state", "reset_ok", "mamba_ckpt_present")})
    rec["loads_pending"] = len(state["load_t0"])
    return rec


# ----------------------------------------------------------------------------------------------
# verdict
# ----------------------------------------------------------------------------------------------
def _same(x, y):
    return x is not None and y is not None and x["tokens"] == y["tokens"] and x["logprobs"] == y["logprobs"]


def verdict_arm(rec, prompts, blocks):
    v = {"arm": rec.get("arm"), "error": rec.get("error")}
    hit = blocks * BLOCK
    ps = rec.get("prompts", [])
    ident = [p for p in ps if not p.get("mutant")]
    mut = next((p for p in ps if p.get("mutant")), None)
    v["engine_up"] = "connector" in rec and "TPURaidenOffloadingConnector" in rec.get("children", [])
    if connector_arm(rec.get("arm")) == "multi":
        v["engine_up"] = v["engine_up"] and "Qwen4ExpRaidenConnector" in rec.get("children", [])
    v["host_hits_engaged"] = len(ident) == prompts and all(
        p.get("host", {}).get("local_hit") == 0 and p.get("host", {}).get("external_hit") == hit
        and p.get("hbm", {}).get("local_hit") == hit and p.get("reset_ok") for p in ident)
    v["checkpoints_present"] = all(p.get("mamba_ckpt_present") for p in ps)
    v["identity_hbm_host"] = len(ident) == prompts and all(_same(p.get("hbm"), p.get("host")) for p in ident)
    v["identity_recompute"] = len(ident) == prompts and all(_same(p.get("hbm"), p.get("recompute")) for p in ident)
    # Same-configuration floors: a path compared with itself.
    v["floor_hit_same"] = len(ident) == prompts and all(_same(p.get("hbm"), p.get("hbm2")) for p in ident)
    v["floor_recompute_same"] = len(ident) == prompts and all(
        _same(p.get("recompute"), p.get("recompute2")) for p in ident)
    v["hbm2_engaged"] = len(ident) == prompts and all(
        p.get("hbm2", {}).get("local_hit") == hit for p in ident)
    if v["identity_recompute"]:
        v["recompute_reading"] = "equal"
    elif not v["floor_recompute_same"]:
        v["recompute_reading"] = "differs; the no-hit path is not deterministic itself (floor > 0)"
    elif not v["floor_hit_same"]:
        v["recompute_reading"] = "differs; the hit path is not deterministic itself (floor > 0)"
    else:
        v["recompute_reading"] = "differs with both floors at 0: hit and recompute paths compute differently"
    v["pages_compared"] = sum(p.get("pages_compared", 0) for p in ident)
    v["page_mismatches"] = sum(p.get("page_mismatches", 0) for p in ident) + sum(
        p.get("pages_only_one_side", 0) for p in ident)
    v["page_identity"] = v["pages_compared"] >= 1000 and v["page_mismatches"] == 0
    # i3 (addendum 3): the GDN/PLE checkpoint pages must actually be compared on every identity
    # prompt (i2 compared none and could not notice).
    v["state_pages_compared"] = sum(p.get("state_pages_compared", 0) for p in ident)
    v["state_pages_ok"] = len(ident) == prompts and all(p.get("state_pages_compared", 0) > 0 for p in ident)
    v["mutant_bites"] = bool(mut) and mut.get("host", {}).get("external_hit") == hit and (
        not _same(mut.get("hbm"), mut.get("host"))) and mut.get("page_mismatches_state", 0) > 0
    loads = [l for l in rec.get("loads", []) if not l.get("dropped_keys")]
    if loads and rec.get("bytes_per_key"):
        rates = [l["keys"] * rec["bytes_per_key"] / l["s"] / 1e9 for l in loads if l["s"] > 0]
        v["host_load_gbps_median"] = round(sorted(rates)[len(rates) // 2], 3) if rates else None
        per_100k = [l["s"] * 1e5 / hit * 1e3 for l in loads]
        v["host_load_ms_per_100k_median"] = round(sorted(per_100k)[len(per_100k) // 2], 2)
    ready = (rec.get("mem_ready") or {}).get("cgroup_current")
    if ready and rec.get("tier_bytes"):
        # memory outside the tier at engine ready: the tier sizing input (P)
        v["nonpool_bytes_at_ready"] = ready - int(rec["tier_bytes"])
    v["valid"] = bool(v["engine_up"] and v["host_hits_engaged"] and v["checkpoints_present"] and v["mutant_bites"]
                      and v["hbm2_engaged"] and v["state_pages_ok"])
    # Tokens can be held to bitwise identity only where the hit path repeats bitwise (floor 0). If
    # it does not, the pass rests on the bytes (page identity), which no kernel noise can move.
    tokens_ok = v["identity_hbm_host"] if v["floor_hit_same"] else True
    v["pass"] = bool(v["valid"] and v["page_identity"] and tokens_ok)
    v["pass_kind"] = ("bytes+tokens" if v["floor_hit_same"] else
                      "bytes only (hit path not deterministic: hbm != hbm2, tokens not judged)")
    return v


# ----------------------------------------------------------------------------------------------
# parent
# ----------------------------------------------------------------------------------------------
CACHE_VARS = {
    "VLLM_CACHE_ROOT": "vllm", "VLLM_CONFIG_ROOT": "vllm_config",
    "VLLM_XLA_CACHE_PATH": "vllm/xla_cache", "TRITON_CACHE_DIR": "triton",
    "TORCH_TPU_CACHE_DIR": "torch_tpu_cache",
    "TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE": "torch_tpu_cache",
    "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT": "tier3_cache",
    "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT": "tier3_cache",
    "JAX_COMPILER_CACHE_DIR": "jax_cache", "JAX_COMPILATION_CACHE_DIR": "jax_cache",
}


def arm_cache_env(arm, environ=None):
    """The child's env with every compile cache under its own per-arm root.

    Arms differ in connector and pool setup, and a program compiled for one
    and reused by the other fails with RuntimeProgramInputMismatch (j-46084fb7,
    arm 'offload'). Returns (env, root)."""
    env = dict(os.environ if environ is None else environ)
    root = os.path.join(env.get("CACHE_DIR") or "/dev/shm/compile-cache", f"q38-arm-{arm}")
    for var, sub in CACHE_VARS.items():
        path = os.path.join(root, sub)
        env[var] = path
        os.makedirs(path, exist_ok=True)
    return env, root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(os.environ.get("CDK_OUTPUT_DIR", "/tmp/cdk_output"),
                                                      "q38f-bench", "script"))
    ap.add_argument("--arms", default="multi,offload")
    ap.add_argument("--arm", default="")
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--prompts", type=int, default=2)
    ap.add_argument("--blocks", type=int, default=32)
    ap.add_argument("--suffix", type=int, default=2000)
    # 1: the producer's served mode. With more, a producer finishes ahead of
    # its committed prefix and Stage-3 refuses to publish (j-46084fb7).
    ap.add_argument("--max-tokens", type=int, default=1)
    ap.add_argument("--cpu-bytes", type=int, default=100 * 10**9)
    ap.add_argument("--seed", type=int, default=38)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.child:
        try:
            rec = child(a)
        except Exception as e:  # noqa: BLE001
            import traceback

            rec = {"arm": a.arm, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-4000:]}
        print("Q38_ID_CHILD " + json.dumps(rec), flush=True)
        return 0
    os.makedirs(a.out_dir, exist_ok=True)
    out = {"t_start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "args": vars(a), "arms": {}}
    for arm in a.arms.split(","):
        for f in ("/tmp/libtpu_lockfile",):
            try:
                os.remove(f)
            except OSError:
                pass
        argv = [sys.executable, os.path.abspath(__file__), "--child", "--arm", arm] + [
            f"--{k.replace('_', '-')}={v}" for k, v in vars(a).items()
            if k in ("prompts", "blocks", "suffix", "max_tokens", "cpu_bytes", "seed")]
        log = open(os.path.join(a.out_dir, f"q38_hosttier_identity_{arm}.log"), "w")
        env, arm_cache = arm_cache_env(arm)
        env.update(ARM_ENV.get(arm, {}))
        t = time.time()
        rec = None
        try:
            p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=log, text=True,
                                 start_new_session=True, env=env)
            for line in p.stdout:
                print(line, end="", flush=True)
                if line.startswith("Q38_ID_CHILD "):
                    rec = json.loads(line[len("Q38_ID_CHILD "):])
            p.wait(timeout=a.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, 9)
        rec = rec or {"arm": arm, "error": "no child record (crash or timeout); see log"}
        rec["wall_s"] = round(time.time() - t, 1)
        out["arms"][arm] = {"record": rec, "verdict": verdict_arm(rec, a.prompts, a.blocks)}
        print("Q38_ID_ARM " + json.dumps(out["arms"][arm]["verdict"]), flush=True)
        subprocess.run(["pkill", "-9", "-f", "q38_hosttier_identity.py --child"], check=False)
        if arm_cache.startswith("/dev/shm/"):
            # tmpfs counts against the pod's memory: drop this arm's programs.
            subprocess.run(["rm", "-rf", arm_cache], check=False)
    out["t_end_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    primary_arm = a.arms.split(",")[0]  # multi for i2 (multi,offload); multi4g for the premapped check
    primary = out["arms"].get(primary_arm, {}).get("verdict", {})
    out["verdict"] = {"primary_arm": primary_arm, "pass": bool(primary.get("pass")), "valid": bool(primary.get("valid")),
                      "arms": {k: {x: v["verdict"].get(x) for x in ("valid", "pass", "pass_kind", "identity_hbm_host",
                                                                     "identity_recompute", "floor_hit_same",
                                                                     "floor_recompute_same", "recompute_reading",
                                                                     "page_identity", "state_pages_compared",
                                                                     "checkpoints_present",
                                                                     "pages_compared", "mutant_bites",
                                                                     "host_load_ms_per_100k_median",
                                                                     "host_load_gbps_median", "error")}
                               for k, v in out["arms"].items()}}
    with open(os.path.join(a.out_dir, "q38_hosttier_identity.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("Q38_ID_VERDICT " + json.dumps(out["verdict"]), flush=True)
    return 0 if out["verdict"]["pass"] else 3


def selftest():
    hit = 32 * BLOCK
    gen = lambda toks, lp=-0.5: {"tokens": toks, "logprobs": [[(t, lp)] for t in toks]}

    def prompt(k, mutant=False, host_lp=-0.5, rec_lp=-0.5, mism=0, mism_state=0, ext=hit, local_host=0,
               hbm2_lp=-0.5, rec2_lp=None, hbm2_local=hit, state_pages=37):
        p = {"k": k, "mutant": mutant, "reset_ok": True, "mamba_ckpt_present": True,
             "hbm": {**gen([1, 2, 3]), "local_hit": hit, "external_hit": 0},
             "hbm2": {**gen([1, 2, 3], hbm2_lp), "local_hit": hbm2_local, "external_hit": 0},
             "host": {**gen([1, 2, 3], host_lp), "local_hit": local_host, "external_hit": ext},
             "pages_compared": 805, "pages_only_one_side": 0, "page_mismatches": mism,
             "page_mismatches_state": mism_state, "state_pages_compared": state_pages}
        if not mutant:
            p["recompute"] = {**gen([1, 2, 3], rec_lp), "local_hit": 0, "external_hit": 0}
            p["recompute2"] = {**gen([1, 2, 3], rec_lp if rec2_lp is None else rec2_lp),
                               "local_hit": 0, "external_hit": 0}
        return p

    def rec(ps, children=("Qwen4ExpRaidenConnector", "TPURaidenOffloadingConnector")):
        return {"arm": "multi", "connector": "TPUMultiConnector", "children": list(children),
                "bytes_per_key": 46 * 10**6, "prompts": ps,
                "loads": [{"keys": 68, "s": 0.06, "dropped_keys": 0}]}

    good = [prompt(0), prompt(1), prompt(2, mutant=True, host_lp=-0.9, mism=37, mism_state=37)]
    fails = []
    v = verdict_arm(rec(good), 2, 32)
    if not (v["pass"] and v["valid"] and v["identity_recompute"] and v["pages_compared"] == 1610):
        fails.append(("good", v))
    cases = {
        "host logprob differs": [prompt(0, host_lp=-0.51), prompt(1), good[2]],
        "page mismatch": [prompt(0, mism=1), prompt(1), good[2]],
        "host hit not engaged": [prompt(0, ext=0), prompt(1), good[2]],
        "host hit had local blocks": [prompt(0, local_host=hit), prompt(1), good[2]],
        "too few prompts": [prompt(0), good[2]],
        "too few pages": [dict(prompt(0), pages_compared=400), dict(prompt(1), pages_compared=400), good[2]],
    }
    for name, ps in cases.items():
        if verdict_arm(rec(ps), 2, 32)["pass"]:
            fails.append(name)
    # a mutant that does not change the output (or leaves state pages equal) voids the run
    for name, m in {"mutant same output": prompt(2, mutant=True, mism=37, mism_state=37),
                    "mutant pages equal": prompt(2, mutant=True, host_lp=-0.9)}.items():
        vv = verdict_arm(rec([prompt(0), prompt(1), m]), 2, 32)
        if vv["valid"] or vv["pass"]:
            fails.append(name)
    if verdict_arm(rec(good, children=("TPURaidenOffloadingConnector",)), 2, 32)["valid"]:
        fails.append("multi arm without the PD child must not be valid")
    # recompute differing alone does not fail the primary, but is reported
    vr = verdict_arm(rec([prompt(0, rec_lp=-0.6), prompt(1), good[2]]), 2, 32)
    if not vr["pass"] or vr["identity_recompute"]:
        fails.append(("recompute reporting", vr))
    # floors (addendum 2): with hbm == hbm2 the tokens must match; with hbm != hbm2 the pass rests
    # on bytes; hbm2 must have hit HBM; the recompute reading follows the floors
    if v["pass_kind"] != "bytes+tokens" or not v["floor_hit_same"] or not v["floor_recompute_same"]:
        fails.append(("floors good", v))
    vf = verdict_arm(rec([prompt(0, hbm2_lp=-0.51, host_lp=-0.52), prompt(1), good[2]]), 2, 32)
    if not vf["pass"] or vf["floor_hit_same"] or not vf["pass_kind"].startswith("bytes only"):
        fails.append(("hit floor > 0 rests on bytes", vf))
    vf = verdict_arm(rec([prompt(0, hbm2_lp=-0.51, host_lp=-0.52, mism=1), prompt(1), good[2]]), 2, 32)
    if vf["pass"]:
        fails.append("hit floor > 0 with a page mismatch must fail")
    if verdict_arm(rec([prompt(0, hbm2_local=0), prompt(1), good[2]]), 2, 32)["valid"]:
        fails.append("hbm2 that missed HBM must not be valid")
    r1 = verdict_arm(rec([prompt(0, rec_lp=-0.6, rec2_lp=-0.7), prompt(1), good[2]]), 2, 32)["recompute_reading"]
    r2 = verdict_arm(rec([prompt(0, rec_lp=-0.6), prompt(1), good[2]]), 2, 32)["recompute_reading"]
    if not (r1.startswith("differs; the no-hit path") and r2.startswith("differs with both floors at 0")):
        fails.append(("recompute reading", r1, r2))
    # i3: state pages must be compared on every identity prompt, and a corrupted state page fails
    vs = verdict_arm(rec([prompt(0, state_pages=0), prompt(1), good[2]]), 2, 32)
    if vs["valid"] or vs["pass"]:
        fails.append("identity prompt without compared state pages must be invalid")
    vc = verdict_arm(rec([prompt(0, mism=1, mism_state=1), prompt(1), good[2]]), 2, 32)
    if vc["pass"] or vc["page_identity"]:
        fails.append("a corrupted state page must fail page identity")
    vc2 = verdict_arm(rec([prompt(0, mism=1, mism_state=1, hbm2_lp=-0.51, host_lp=-0.52), prompt(1), good[2]]), 2, 32)
    if vc2["pass"]:
        fails.append("a corrupted state page must fail the bytes-only route too")
    # the checkpoint is found in the table, else where the request held it while running
    if not (state_checkpoint([0] * 31 + [77], None, 32) == (77, "table")
            and state_checkpoint([0] * 32, 88, 32) == (88, "seen")
            and state_checkpoint([0] * 32, None, 32) == (None, None)
            and state_checkpoint([5] * 10, 88, 32) == (88, "seen")):
        fails.append("state_checkpoint lookup")
    if abs(v["host_load_ms_per_100k_median"] - 0.06 * 1e5 / hit * 1e3) > 0.01:
        fails.append(("load ms", v))
    # internal request ids carry a random suffix; resolution must be exact
    d = {"p0-cold-a786e59d": 1, "p0-host-0f00": 2, "p10-cold-1234": 3}
    if not (by_request(d, "p0-cold") == 1 and by_request(d, "p0-host") == 2
            and by_request(d, "p1-cold") is None and by_request({"x": 5}, "x") == 5
            and by_request({"p0-cold-aaaa": 1, "p0-coldx-bbbb": 2}, "p0-cold") == 1):
        fails.append("request id resolution")
    # the premapped arm runs the multi connectors with only its env changed
    if not (connector_arm("multi4g") == "multi" and kv_transfer("multi4g", 1) == kv_transfer("multi", 1)
            and ARM_ENV["multi4g"]["TPU_PREMAPPED_BUFFER_SIZE"] == str(4 * 1024**3) and "multi" not in ARM_ENV):
        fails.append("premapped arm")
    v4 = verdict_arm(dict(rec(good), arm="multi4g"), 2, 32)
    if not (v4["pass"] and v4["engine_up"]):
        fails.append(("multi4g verdict", v4))
    if verdict_arm(dict(rec(good, children=("TPURaidenOffloadingConnector",)), arm="multi4g"), 2, 32)["valid"]:
        fails.append("multi4g without the PD child must not be valid")
    # per-arm compile caches never share a directory
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        e1, r1 = arm_cache_env("multi", {"CACHE_DIR": tmp})
        e2, r2 = arm_cache_env("offload", {"CACHE_DIR": tmp})
        if r1 == r2 or any(e1[v] == e2[v] for v in CACHE_VARS) or not all(
                e1[v].startswith(r1) for v in CACHE_VARS):
            fails.append("per-arm caches")
    print("selftest", "PASS" if not fails else f"FAIL {fails}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
