# SPDX-License-Identifier: Apache-2.0
"""Engine-level fold on/off twin under main-KV DCP-2, with page dumps (test-only).

One arm per process (env is read at import / trace time):
  python3 fold_twin.py on|off <plan.steps.json> <out_dir>

Boots the wmb twin's LLMEngine (dummy weights, the plan's c48 env, TP8,
VLLM_TPU_QSA_MAIN_KV_DCP=2) with the real scheduler (prefix caching, MTP k=3,
sync scheduling) and runs phases of fresh random prompts (no prefix hits):

  A700 / A900 / A1500 : one prompt alone, max_tokens=1
  B                   : R0..R3 (300..450 tokens, under one Mamba block, same
                        chunking in both arms) decoding, then B700 joins at
                        step 2 (max_tokens=1): the fold inside a mixed batch
  C800                : one prompt alone, 12 greedy tokens

After each dumped request finishes, every worker in FOLDTWIN_DUMP_RANKS
writes the pages of that request (every KV-cache group, every block-table
column it held during the phase, from the engine-core hook log) to
<out_dir>/pages_<arm>_<rid>_r<rank>.npz, trimmed after the last nonzero byte
(the pool starts zeroed and FOLDTWIN_BURN keeps warmup pages out). Group ->
layer names go to <out_dir>/groups.json. fold_twin_compare.py then names the
first group (in layer order) whose pages differ between the arms.
``on`` forces the fold on under DCP (FOLDCP_FORCE=1 bypasses
fold_blocked_by_context_parallel, as VLLM_TPU_MAMBA_TAIL_FOLD_DCP=1 did on
za6); ``off`` is the served good arm (fold guarded off by DCP).
"""

import json
import os
import sys
import time

CORE = os.environ.get("WMB_CORE", "/root/fab/wmb/core")
sys.path.insert(0, CORE)

import numpy as np  # noqa: E402

import forward_run as fr  # noqa: E402

ARM_ENV = {
    "on": {"VLLM_TPU_MAMBA_TAIL_FOLD": "1", "VLLM_TPU_MAMBA_TAIL_FOLD_UNIT": "256",
           "VLLM_TPU_MAMBA_TAIL_FOLD_DCP": "1", "FOLDCP_FORCE": "1"},
    "off": {"VLLM_TPU_MAMBA_TAIL_FOLD": "1", "VLLM_TPU_MAMBA_TAIL_FOLD_UNIT": "256",
            "VLLM_TPU_MAMBA_TAIL_FOLD_DCP": "0", "FOLDCP_FORCE": "0"},
    # Fold geometry on, the in-step checkpoint write off (foldtwin_hook FOLDTWIN_NOCKPT=1
    # makes the worker mirror prefill_checkpoints return no checkpoint).
    "nockpt": {"VLLM_TPU_MAMBA_TAIL_FOLD": "1", "VLLM_TPU_MAMBA_TAIL_FOLD_UNIT": "256",
               "VLLM_TPU_MAMBA_TAIL_FOLD_DCP": "1", "FOLDCP_FORCE": "1"},
}
# c48 (j-f37cd850) values the twin plan lacks.
C48_EXTRA = {"VLLM_TPU_FAST_METADATA_BUILD": "1", "VLLM_TPU_QSA_PREFILL_INDEXER_BKV": "1024",
             "TPU_QWEN4_HC_PREFILL_NORM": "bf16ep64", "VLLM_TPU_QSA_MAIN_KV_DCP": "2"}

# (name, [(rid, prompt_len, max_tokens, add_at_step)], dumped rids)
PHASES = [
    ("A900", [("A900", 900, 1, 0)], ["A900"]),
    ("A1100", [("A1100", 1100, 1, 0)], ["A1100"]),
    ("A1300", [("A1300", 1300, 1, 0)], ["A1300"]),
    ("A1500", [("A1500", 1500, 1, 0)], ["A1500"]),
    ("A1700", [("A1700", 1700, 1, 0)], ["A1700"]),
    ("A2100", [("A2100", 2100, 1, 0)], ["A2100"]),
    ("B", [("R0", 300, 10, 0), ("R1", 350, 10, 0), ("R2", 400, 10, 0), ("R3", 450, 10, 0),
           ("B1300", 1300, 1, 2)], ["B1300"]),
    ("C1300", [("C1300", 1300, 16, 0)], []),
]


def fingerprint(worker):
    r = worker.model_runner
    out = []
    for name, p in list(r.model.named_parameters())[:4]:
        out.append((name, float(p.detach().float().cpu().double().sum())))
    return out


def group_info(worker):
    r = worker.model_runner
    cfg = r.kv_cache_config
    groups = []
    for gid, g in enumerate(cfg.kv_cache_groups):
        spec = g.kv_cache_spec
        groups.append({"gid": gid, "spec": type(spec).__name__,
                       "block_size": int(getattr(spec, "block_size", 0)),
                       "dtype": str(getattr(spec, "dtype", "")),
                       "ckpt": int(getattr(spec, "num_prefill_checkpoint_blocks", 0) or 0),
                       "layers": list(g.layer_names)})
    raws = [{"shape": list(t.shape), "dtype": str(t.dtype)} for t in r.kv_cache_raw_tensors]
    return {"num_blocks": int(cfg.num_blocks), "groups": groups, "raws": raws,
            "rank": int(getattr(worker, "rank", -1))}


def dump_pages(worker, req_cols, path_prefix, ranks):
    """req_cols: {rid: {gid: {col: block_id}}}. Writes one npz per rank. Never raises: a
    worker exception here would take the engine down with it."""
    try:
        return _dump_pages(worker, req_cols, path_prefix, ranks)
    except Exception as e:  # noqa: BLE001
        return {"rank": int(getattr(worker, "rank", -1)), "error": repr(e)[:300]}


def _dump_pages(worker, req_cols, path_prefix, ranks):
    import torch

    rank = int(getattr(worker, "rank", -1))
    if rank not in ranks:
        return {"rank": rank, "skipped": True}
    r = worker.model_runner
    nb = int(r.kv_cache_config.num_blocks)
    ids = sorted({int(b) for g in req_cols.values() for cols in g.values()
                  for b in cols.values() if int(b) > 0})
    if not ids:
        return {"rank": rank, "pages": 0}
    out = {}
    for ti, raw in enumerate(r.kv_cache_raw_tensors):
        if raw.numel() % nb:
            out[f"meta_bad_t{ti}"] = np.array(list(raw.shape))
            continue
        # Gather whole pages straight from the pool tensor in fixed groups of 32 ids (one
        # compiled gather). No reshape of the pool: reshaping the 59 GB pool first made XLA
        # materialize a copy (CompileTimeHbmOom, j-79d653ac).
        split = raw.shape[0] // nb
        assert split == 1, f"pool rows per block {split}"
        parts = []
        for i in range(0, len(ids), 32):
            chunk = ids[i:i + 32]
            idx = torch.tensor(chunk + [chunk[0]] * (32 - len(chunk)), dtype=torch.long)
            got = raw.index_select(0, idx.to(raw.device)).cpu().contiguous()
            parts.append(got.view(32, -1).view(torch.uint8).numpy()[:len(chunk)].copy())
        rows = np.concatenate(parts)
        pos = {b: i for i, b in enumerate(ids)}
        for rid, g in req_cols.items():
            for gid, cols in g.items():
                for col, b in cols.items():
                    if int(b) <= 0:
                        continue
                    row = rows[pos[int(b)]]
                    nz = np.flatnonzero(row)
                    end = int(nz[-1]) + 1 if nz.size else 0
                    out[f"{rid}|g{gid}|c{col}|t{ti}|n{row.size}"] = row[:end].copy()
    np.savez(f"{path_prefix}_r{rank}.npz", **out)
    return {"rank": rank, "pages": len(ids), "keys": len(out)}


def read_log(path, start_line):
    recs = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= start_line:
                recs.append(json.loads(line))
    return recs


def columns_seen(recs, rid):
    """{gid: {col: last non-null block id}} over the phase's schedule records."""
    cols: dict = {}
    for rec in recs:
        t = next((v for k, v in rec.get("tables", {}).items()
                  if k == rid or k.startswith(rid + "-")), None)
        if not isinstance(t, list):
            continue
        for gid, blocks in enumerate(t):
            for col, b in enumerate(blocks):
                if b > 0:
                    cols.setdefault(gid, {})[col] = b
    return cols


def main():
    arm, steps, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)
    plan = json.load(open(steps))
    vw_path = os.path.join(os.path.dirname(CORE), plan.get("model_dir") or "", "vllm_workarounds.py")
    if plan.get("model_dir") and os.path.isfile(vw_path):
        import importlib.util

        spec = importlib.util.spec_from_file_location("vllm_workarounds", vw_path)
        vw = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(vw)
        if hasattr(vw, "pod_setup"):
            plan["twin"]["env"] = {**(plan["twin"].get("env") or {}), **(vw.pod_setup(plan) or {})}
    twin, buckets = plan["twin"], plan["compile_buckets"]
    real = os.environ.get("FOLDTWIN_MODEL_DIR")
    if real:
        # Real weights: dummy weights give near-uniform logits and near-zero KV/state.
        model_dir = real
        twin["server_args"].pop("load-format", None)
        twin.pop("router", None)
    else:
        model_dir = "/tmp/wmb_model"
        os.makedirs(model_dir, exist_ok=True)
        json.dump(plan["checkpoint_config"], open(os.path.join(model_dir, "config.json"), "w"))
    os.environ.update(fr._run_env(twin, fr.CACHE_DIR))
    os.environ.update(C48_EXTRA)
    os.environ.update(ARM_ENV[arm])
    # run_twin.sh exports FOLDTWIN_LOG / FOLDTWIN_BURN / FOLDCP_FORCE before Python starts, so
    # the .pth hooks are active in every process (driver, engine core, workers; fork or spawn).
    log_path = os.environ["FOLDTWIN_LOG"]
    assert os.environ.get("FOLDCP_FORCE", "0") == ARM_ENV[arm]["FOLDCP_FORCE"], "arm env mismatch"
    assert (os.environ.get("FOLDTWIN_NOCKPT", "0") == "1") == (arm == "nockpt"), "nockpt env"
    sys.pycache_prefix = os.environ["PYTHONPYCACHEPREFIX"]
    print("ARM", arm, {k: os.environ.get(k) for k in sorted({*ARM_ENV[arm], *C48_EXTRA})},
          flush=True)
    from vllm import SamplingParams
    from vllm.engine.arg_utils import EngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.v1.engine.llm_engine import LLMEngine

    argv = fr.engine_argv(twin, model_dir, buckets, "/tmp/wmb_xprof")
    sa = twin["server_args"]
    if "--enable-prefix-caching" not in argv:
        argv += ["--enable-prefix-caching"]
    if "--speculative-config" not in argv and sa.get("speculative-config"):
        argv += ["--speculative-config", str(sa["speculative-config"])]
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    parser.add_argument("model_tag")
    argv, compat = fr._compat(argv, parser)
    ns = parser.parse_args(argv)
    ns.model = ns.model_tag
    print("ENGINE ARGS:", " ".join(argv), flush=True)
    t = time.time()
    engine = fr._boot(LLMEngine, EngineArgs.from_cli_args(ns), {}, False)
    print(f"engine init {time.time() - t:.0f} s", flush=True)
    router = twin.get("router")
    if router:
        engine.collective_rpc(fr._randomize_router, args=(router,))
    fp = engine.collective_rpc(fingerprint)[0]
    print("FINGERPRINT", fp, flush=True)
    info = engine.collective_rpc(group_info)
    json.dump(info[0], open(os.path.join(out_dir, f"groups_{arm}.json"), "w"))
    print("GROUPS", len(info[0]["groups"], ), "raws", info[0]["raws"], "num_blocks",
          info[0]["num_blocks"], flush=True)
    ranks = [int(x) for x in os.environ.get("FOLDTWIN_DUMP_RANKS", "0,1").split(",")]

    vocab = int(plan["checkpoint_config"].get("vocab_size")
                or plan["checkpoint_config"].get("text_config", {}).get("vocab_size") or 150000)
    rng = np.random.default_rng(1234)
    result = {"arm": arm, "fingerprint": fp, "phases": {}}
    nlines = 0
    for pname, adds, dumped in PHASES:
        prompts = {rid: rng.integers(100, min(vocab, 150000) - 1, n).tolist()
                   for rid, n, _, _ in adds}
        rec = {rid: {"prompt_len": n, "tokens": [], "top1": [], "top5": [], "step": []}
               for rid, n, _, _ in adds}
        finished, step, done_dump = set(), 0, set()
        while len(finished) < len(adds) and step < 300:
            for rid, n, mt, at in adds:
                if at == step:
                    sp = SamplingParams(temperature=0.0, max_tokens=mt, logprobs=5,
                                        detokenize=False, ignore_eos=True)
                    engine.add_request(rid, {"prompt_token_ids": prompts[rid]}, sp)
            outs = engine.step()
            for o in outs:
                c = o.outputs[0]
                r = rec[o.request_id]
                toks = list(c.token_ids)
                for i in range(len(r["tokens"]), len(toks)):
                    r["tokens"].append(int(toks[i]))
                    r["step"].append(step)
                    lp = c.logprobs or []
                    if i < len(lp) and lp[i]:
                        d = lp[i]
                        best = d.get(toks[i])
                        r["top1"].append(float(best.logprob) if best is not None else None)
                        r["top5"].append(sorted(((int(k), float(v.logprob))
                                                 for k, v in d.items()),
                                                key=lambda kv: -kv[1]))
                    else:
                        r["top1"].append(None)
                        r["top5"].append(None)
                if o.finished:
                    finished.add(o.request_id)
            for rid in dumped:
                if rid in finished and rid not in done_dump:
                    recs = read_log(log_path, nlines)
                    cols = {rid: columns_seen(recs, rid)}
                    res = engine.collective_rpc(
                        dump_pages,
                        args=(cols, os.path.join(out_dir, f"pages_{arm}_{rid}"), ranks))
                    print(f"DUMP {pname} {rid}: {[x for x in res if not x.get('skipped')]}",
                          flush=True)
                    json.dump(cols, open(os.path.join(out_dir, f"cols_{arm}_{rid}.json"), "w"))
                    done_dump.add(rid)
            step += 1
        recs = read_log(log_path, nlines)
        nlines += len(recs)
        sched = [{"step": x["step"], "sched": x["sched"], "copies": x["copies"],
                  "computed_after": x.get("computed_after")} for x in recs]
        result["phases"][pname] = {"requests": rec, "steps": step, "sched": sched}
        print(f"PHASE {pname} ({arm}): {step} engine steps, finished {sorted(finished)}",
              flush=True)
        for x in sched:
            print(f"  sched step {x['step']}: {x['sched']} copies={x['copies']}", flush=True)
        for rid, r in rec.items():
            print(f"  {rid} P={r['prompt_len']} tokens={r['tokens'][:12]} "
                  f"top5[0]={(r['top5'] or [None])[0]}", flush=True)
    json.dump(result, open(os.path.join(out_dir, f"arm_{arm}.json"), "w"))
    try:
        engine.engine_core.shutdown()
    except Exception as e:  # noqa: BLE001
        print("shutdown:", e, flush=True)


if __name__ == "__main__":
    main()
