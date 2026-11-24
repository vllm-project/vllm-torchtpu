"""q38 CACHE lane, q38-cache-tail-i1: token identity and computed-token / latency microbench of the 128-token
prefix tail (`--prefix-match-unit 128`) on the served PP8 prefill engine (bench recipe, BENCH=script).

Arms, each a FRESH engine in its own process, in this order:
  B  served prefill arguments (3,200-token prefix granularity)              -- the baseline
  T  B + --prefix-match-unit=128                                              -- the lever
  X  T + mutant: the TPU runner's copy-on-write block copies are dropped      -- must change tokens
  C  B with --no-enable-prefix-caching (cold reference, numerical floor)      -- descriptive
Protocol per arm (identical prompts in every arm: natural code text tokenised from this tree's src/):
  turn 1 (batch of 8): prefix_i, L_i in LENS (none a multiple of 128 or 3,200), 16 greedy tokens
  turn 2 (one request at a time): prefix_i + the next 3,200 corpus tokens, 16 greedy tokens; latency to the
         first output token per request
  turn 3 (batch of 8): prefix_i + 3,200 other tokens, 1 token; round wall time (the served warm-turn shape)
Each arm writes arm_<A>.json; the parent writes verdict.json and prints
  Q38_TAIL_I1 identity=<T==B on every turn-2/turn-3 token> engaged=<T hits at the tail> mutant_bites=<X!=B>
  computed_B=... computed_T=... (turn-2 computed tokens) ttft_ms_B=... ttft_ms_T=...
Exit 0 only if identity, engagement and the mutant all hold. Run from the tree root:
  PYTHONPATH=src python3 q38_cache/q38_cache_tail_i1.py            (parent; needs env BENCH_PY from the recipe)
"""
import glob
import json
import os
import signal
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.environ.get("Q38_TAIL_OUT") or os.path.join(os.environ.get("CDK_OUTPUT_DIR", "/tmp/cdk_output"), "q38_cache_tail_i1")
LENS = [9037, 20077, 33333, 50011, 64123, 81099, 104000, 129999]
EXT = 3200
GEN = 16
UNIT = 128
BLOCK = 3200
# The served prefill's `vllm serve` arguments (j-4e8d3ff8 prefill log, 06:24Z), less the frontend-only flags,
# the P/D connector and the profiler (this is an offline engine).
SERVED = [
    "/root/.cache/huggingface/Qwen/Qwen3.8-Flash-Next-FP8", "--tensor-parallel-size=1", "--data-parallel-size=1",
    "--max-model-len=262144", "--quantization=fp8", "--max-num-batched-tokens=8192", "--max-num-seqs=8",
    "--no-async-scheduling", "--prefill-schedule-interval=1", "--enable-prefix-caching",
    "--gpu-memory-utilization=0.92", "--kv-cache-dtype=fp8", "--language-model-only", "--attention-backend", "CUSTOM",
    "--block-size", "128", "--limit-mm-per-prompt", '{"image":0,"video":0}', "--mamba-cache-mode", "align",
    "--trust-remote-code",
    "--additional-config", '{"qwen4_exp":{"engram_config":{"cpu_offload":true,"dp_shared_memory":false,'
    '"embedding_across_dp":false},"hc_prefill_full_tile":256,"hc_prefill_norm_tile":128,"hc_prefill_up_tile":128}}',
    "--pipeline-parallel-size=8",
]
ARMS = {
    "B": ([], {}),
    "T": ([f"--prefix-match-unit={UNIT}"], {}),
    "X": ([f"--prefix-match-unit={UNIT}"], {"Q38_TAIL_MUTANT": "drop_cow"}),
    "C": (["--no-enable-prefix-caching"], {}),
}


def bench_module():
    """The bench recipe's q38f_leg_bench.py (env BENCH_PY), for its serve-args -> EngineArgs helpers."""
    src = os.environ.get("BENCH_PY")
    if not src:
        raise SystemExit("BENCH_PY is not set: run under the bench recipe")
    path = "/tmp/q38_cache_q38f_leg_bench.py"
    with open(path, "w") as f:
        f.write(src)
    import importlib.util

    spec = importlib.util.spec_from_file_location("q38f_leg_bench_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def corpus(tok, need):
    files = sorted(glob.glob(os.path.join(ROOT, "src", "**", "*.py"), recursive=True))
    ids, chars = [], 0
    for f in files:
        text = open(f, errors="replace").read()
        ids.extend(tok.encode(text, add_special_tokens=False))
        chars += len(text)
        if len(ids) >= need:
            break
    if len(ids) < need:
        raise SystemExit(f"corpus has {len(ids)} tokens, need {need}")
    return ids


def drain(eng, ids, t0=None):
    outs, first = {}, {}
    while set(ids) - set(outs):
        for o in eng.step():
            if o.request_id in ids:
                if o.request_id not in first and o.outputs and o.outputs[0].token_ids:
                    first[o.request_id] = time.time()
                if o.finished:
                    outs[o.request_id] = o
    return outs, first


def child(arm):
    extra, _ = ARMS[arm]
    mod = bench_module()
    from vllm import SamplingParams
    from vllm.engine.arg_utils import EngineArgs
    from vllm.usage.usage_lib import UsageContext
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.v1.engine.llm_engine import LLMEngine

    serve = list(SERVED)
    if arm == "C":  # cold reference: no prefix cache, so no align-mode state tables either
        serve.remove("--enable-prefix-caching")
        i = serve.index("--mamba-cache-mode")
        del serve[i:i + 2]
    serve += extra
    argv, dropped = mod.engine_argv(serve, None)
    print(f"[tail-i1 {arm}] engine args: {' '.join(argv)}", flush=True)
    ns, unknown = EngineArgs.add_cli_args(FlexibleArgumentParser()).parse_known_args(argv)
    if unknown:
        raise SystemExit(f"[tail-i1 {arm}] engine does not take {unknown}")
    t_init = time.time()
    eng = LLMEngine.from_engine_args(EngineArgs.from_cli_args(ns), usage_context=UsageContext.LLM_CLASS)
    cc = eng.vllm_config.cache_config
    res = {"arm": arm, "startup_s": round(time.time() - t_init, 1), "argv": argv,
           "prefix_match_unit": getattr(cc, "prefix_match_unit", "absent"),
           "enable_prefix_caching": cc.enable_prefix_caching, "block_size": cc.block_size,
           "mamba_block_size": getattr(cc, "mamba_block_size", None)}
    tok = eng.get_tokenizer()
    need = sum(L + 2 * EXT + 64 for L in LENS)
    ids = corpus(tok, need)
    convs, off = [], 0
    for L in LENS:
        convs.append((ids[off:off + L], ids[off + L:off + L + EXT], ids[off + L + EXT:off + L + 2 * EXT]))
        off += L + 2 * EXT + 64
    greedy = SamplingParams(max_tokens=GEN, temperature=0.0, ignore_eos=True, detokenize=False)
    one = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True, detokenize=False)

    def record(o):
        return {"tokens": list(o.outputs[0].token_ids), "cached": getattr(o, "num_cached_tokens", None),
                "prompt": len(o.prompt_token_ids)}

    # turn 1: batch
    t0 = time.time()
    rids = [f"t1-{i}" for i in range(len(convs))]
    for rid, (p, _, _) in zip(rids, convs):
        eng.add_request(rid, {"prompt_token_ids": p}, greedy)
    outs, _ = drain(eng, set(rids))
    res["turn1"] = [record(outs[r]) for r in rids]
    res["turn1_wall_s"] = round(time.time() - t0, 3)
    # turn 2: one at a time, latency to the first output token
    res["turn2"] = []
    for i, (p, e, _) in enumerate(convs):
        rid = f"t2-{i}"
        t0 = time.time()
        eng.add_request(rid, {"prompt_token_ids": p + e}, greedy)
        outs, first = drain(eng, {rid})
        r = record(outs[rid])
        r["ttft_ms"] = round((first.get(rid, time.time()) - t0) * 1e3, 1)
        res["turn2"].append(r)
    # turn 3: the served warm-turn shape as one batch
    t0 = time.time()
    rids = [f"t3-{i}" for i in range(len(convs))]
    for rid, (p, _, f) in zip(rids, convs):
        eng.add_request(rid, {"prompt_token_ids": p + f}, one)
    outs, _ = drain(eng, set(rids))
    res["turn3"] = [record(outs[r]) for r in rids]
    res["turn3_wall_s"] = round(time.time() - t0, 3)
    with open(os.path.join(OUT, f"arm_{arm}.json"), "w") as f:
        json.dump(res, f)
    print(f"[tail-i1 {arm}] done: turn2 cached {[r['cached'] for r in res['turn2']]}", flush=True)


def mutant_env(env):
    site = os.path.join(HERE, "mutant_site")
    env["PYTHONPATH"] = os.pathsep.join([site, env.get("PYTHONPATH", "")])
    return env


def verdict():
    arms = {}
    for a in ARMS:
        p = os.path.join(OUT, f"arm_{a}.json")
        if os.path.exists(p):
            arms[a] = json.load(open(p))
    B, T, X = arms.get("B"), arms.get("T"), arms.get("X")
    v = {"arms_present": sorted(arms)}
    if B and T:
        v["identity_turn1"] = [b["tokens"] == t["tokens"] for b, t in zip(B["turn1"], T["turn1"])]
        v["identity_turn2"] = [b["tokens"] == t["tokens"] for b, t in zip(B["turn2"], T["turn2"])]
        v["identity_turn3"] = [b["tokens"] == t["tokens"] for b, t in zip(B["turn3"], T["turn3"])]
        v["identity"] = all(v["identity_turn1"] + v["identity_turn2"] + v["identity_turn3"])
        exp_T = [L // UNIT * UNIT for L in LENS]
        exp_B = [(L - 1) // BLOCK * BLOCK for L in LENS]
        v["cached_B"] = [r["cached"] for r in B["turn2"]]
        v["cached_T"] = [r["cached"] for r in T["turn2"]]
        v["expected_B"], v["expected_T"] = exp_B, exp_T
        v["engaged"] = v["cached_T"] == exp_T
        v["baseline_as_expected"] = v["cached_B"] == exp_B
        v["computed_B"] = sum(r["prompt"] - (r["cached"] or 0) for r in B["turn2"])
        v["computed_T"] = sum(r["prompt"] - (r["cached"] or 0) for r in T["turn2"])
        v["ttft_ms_B"] = [r["ttft_ms"] for r in B["turn2"]]
        v["ttft_ms_T"] = [r["ttft_ms"] for r in T["turn2"]]
        v["turn3_wall_s"] = {a: arms[a].get("turn3_wall_s") for a in arms}
    if B and X:
        v["mutant_diff_turn2"] = [b["tokens"] != x["tokens"] for b, x in zip(B["turn2"], X["turn2"])]
        v["mutant_bites"] = any(v["mutant_diff_turn2"])
    if B and arms.get("C"):
        C = arms["C"]
        v["B_vs_cold_turn2"] = [b["tokens"] == c["tokens"] for b, c in zip(B["turn2"], C["turn2"])]
    ok = bool(v.get("identity") and v.get("engaged") and v.get("mutant_bites"))
    v["ok"] = ok
    with open(os.path.join(OUT, "verdict.json"), "w") as f:
        json.dump(v, f, indent=1)
    med = lambda xs: round(statistics.median(xs), 1) if xs else None
    print("Q38_TAIL_I1 ok=%s identity=%s engaged=%s mutant_bites=%s computed_B=%s computed_T=%s "
          "ttft_ms_B_median=%s ttft_ms_T_median=%s" % (
              ok, v.get("identity"), v.get("engaged"), v.get("mutant_bites"), v.get("computed_B"),
              v.get("computed_T"), med(v.get("ttft_ms_B")), med(v.get("ttft_ms_T"))), flush=True)
    return 0 if ok else 1


def parent():
    os.makedirs(OUT, exist_ok=True)
    for arm in ARMS:
        env = dict(os.environ)
        env.update(ARMS[arm][1])
        if arm == "X":
            env = mutant_env(env)
        print(f"==> Q38_TAIL_ARM_START {arm} {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}", flush=True)
        proc = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--child", arm], cwd=ROOT, env=env,
                                start_new_session=True)
        rc = proc.wait()
        try:  # the next arm needs the chips: no engine worker of this arm may outlive it
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        time.sleep(10)
        print(f"==> Q38_TAIL_ARM_END {arm} rc={rc} {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}", flush=True)
    return verdict()


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        child(sys.argv[2])
    elif len(sys.argv) == 2 and sys.argv[1] == "--verdict":
        sys.exit(verdict())
    else:
        sys.exit(parent())
