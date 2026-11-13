#!/usr/bin/env python3
from __future__ import annotations

import concurrent.futures
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.request

POD_NAME = "rsinghc-dp8-c512-scale"

# dsv41 addition (not in the handoff paste): DSV41_DP_RR=1 pins the KV-template warmup and the
# benchmark requests to DP ranks with the X-data-parallel-rank header, as the v2 driver and the
# patched `vllm bench serve` (VLLM_BENCH_DP_SIZE=8) did. Without it the DP load balancer spreads
# c=64 as e.g. {9,8,6,10,6,6,9,10}; the DP-coordinated bucket is the max over ranks (60 > 48
# tokens), so every rank runs the 384-token graph (~3x slower step). rank = (i // 8) % 8 keeps
# all 8 shared prefixes on every rank. Unset/0 = original behaviour.
DP_RR = int(os.environ.get("DSV41_DP_RR", "0") or 0)
DP_RR_SIZE = int(os.environ.get("DSV41_DP_RR_SIZE", "8") or 8)


def _dp_headers(i: int) -> dict:
    h = {"Content-Type": "application/json"}
    if DP_RR:
        h["X-data-parallel-rank"] = str((i // 8) % DP_RR_SIZE)
    return h


def atomic_write_json(path: str, obj: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)
    print(f"[SAVED-INPOD] {path} ({os.path.getsize(path)} bytes)", flush=True)


def boot_dp8_scale_server(topology: str = "dp8") -> None:
    if topology == "tp8":
        dp, tp, max_seqs, comp_cfg, moe_all39, skip_pad = 1, 8, 64, "[384]", "0", "0"
    else:
        dp, tp, max_seqs, comp_cfg, moe_all39, skip_pad = 8, 1, 64, "[48, 96, 192]", "0", "0"
    my_pgid = os.getpgid(os.getpid())
    for pid in os.listdir("/proc"):
        if pid.isdigit() and int(pid) != os.getpid():
            try:
                if os.getpgid(int(pid)) == my_pgid:
                    continue
                cmd = open(f"/proc/{pid}/cmdline").read().replace("\x00", " ")
                if any(k in cmd for k in ("/bin/vllm", "vllm serve", "VLLM::", "EngineCore", "launch_dp8")):
                    os.kill(int(pid), signal.SIGKILL)
            except Exception:
                pass
    time.sleep(2)

    launch_sh = f"""#!/bin/bash
ulimit -c 0
ulimit -n 65536 || true
rm -rf /dev/shm/psm_* /dev/shm/sem.* /tmp/libengram_gather_omp_*.so /tmp/override_cache_hit_ratio /tmp/disable_decode_bench_fill
mkdir -p /dev/shm/vllm_cache_27arr /dev/shm/profiles /dev/shm/xla_cache_27arr /workspace/perf_results
ln -sfn /dev/shm/vllm_cache_27arr /root/.cache/vllm
ln -sfn /dev/shm/xla_cache_27arr /tmp/xla_cache
grep -q 'metadata.google.internal' /etc/hosts || echo '127.0.0.1 metadata.google.internal' >> /etc/hosts
pgrep -f mock_metadata.py >/dev/null || (LD_PRELOAD="" python3 /workspace/mock_metadata.py > /workspace/mock_metadata.log 2>&1 &)
sleep 1

BASE_PORT=$(( 19400 + (RANDOM % 200) * 10 ))
unset TPU_PROCESS_PORT TPU_PROCESS_ADDRESSES TPU_WORKER_ID TPU_VISIBLE_CHIPS TPU_RUNTIME_METRICS_PORTS TPU_SKIP_MDS_QUERY VBAR_CONTROL_SERVICE_URL TPU_WORKER_HOSTNAMES TPU_TOPOLOGY_WRAP TPU_TOPOLOGY_ALT
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(( BASE_PORT - 1 ))
export TORCH_TPU_SLICEBUILDER_ADDRESSES="localhost:$((BASE_PORT+0)),localhost:$((BASE_PORT+1)),localhost:$((BASE_PORT+2)),localhost:$((BASE_PORT+3)),localhost:$((BASE_PORT+4)),localhost:$((BASE_PORT+5)),localhost:$((BASE_PORT+6)),localhost:$((BASE_PORT+7))"
export TPU_CHIPS_PER_HOST_BOUNDS=2,2,1
export CHIPS_PER_HOST_BOUNDS=2,2,1
export TPU_HOST_BOUNDS=1,1,1
export HOST_BOUNDS=1,1,1
export TPU_TOPOLOGY=2x2x1
export TOPOLOGY=2x2x1
export TPU_ACCELERATOR_TYPE=tpu7x-8
export TPU_VISIBLE_DEVICES=0,1,2,3
[ -S /var/run/tpu-plugin/vbar.sock ] && export VBAR_CONTROL_SERVICE_URL="unix:///var/run/tpu-plugin/vbar.sock"
export TPU_RUNTIME_METRICS_PORTS="8431,8432,8433,8434,8435,8436,8437,8438"
export GCE_METADATA_HOST=127.0.0.1:80
export GCE_METADATA_IP=127.0.0.1
export GOOGLE_CLOUD_PROJECT=tpu-prod-env-one-vm
export GCLOUD_PROJECT=tpu-prod-env-one-vm
export LD_PRELOAD=/workspace/metadata_hook.so

export VLLM_CACHE_ROOT=/dev/shm/vllm_cache_27arr
export VLLM_XLA_CACHE_PATH=/dev/shm/xla_cache_27arr
export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT=/dev/shm/xla_cache_27arr
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export OMP_NUM_THREADS=6
export GOMP_SPINCOUNT=1000
export USE_PHASED_PROFILER=false
export VLLM_LOG_POST_WARMUP_HBM=1
export VLLM_PLUGINS=torchtpu,torchtpu_layers
export PYTHONPATH="/workspace/sitecustomize_bench:/workspace/vllm:/workspace/vllm-torchtpu/src"
export TPU_ROPE_CACHE_TRUNCATE=1
export TPU_ROPE_CACHE_ROW_MAJOR=1
export TPU_MOE_COLLECTION_CHUNK_SIZE=8192
export VLLM_TPU_FAST_TOKEN_SUBSTITUTION=0
export ENABLE_CONNECTOR=1
export ENABLE_DECODE_BENCH_CONNECTOR=1

export USE_MOE_FUSED_EP_KERNEL=1
export MOE_FUSED_EP_KERNEL_IMPL=v2
export MOE_FUSED_EP_ENABLE_W4A8=1
export MOE_FUSED_EP_V2_SHARDED_PLAN=1
export MOE_REQUANTIZE_BLOCK_SIZE=512
export TPU_MOE_DENSE_EXPERT_THRESHOLD=0
export VLLM_TPU_MOE_FP8_DISPATCH=1
export VLLM_DSV41_MOE_EP_V2_ALL39={moe_all39}
export VLLM_DSV41_MOE_EP_V2_MAX_LOCAL_TOKENS=16384
export TPU_MOE_SKIP_PADDED_TOKENS={skip_pad}
export TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS=false
export VLLM_DSV41_MOE_FP4_PATH_C_MAX_TOKENS=0
export VLLM_DSV41_MOE_FP4_GMM1_FP8=0
export VLLM_DSV41_INDEXER_CAND_READ_16K=0
export VLLM_DSV41_DECODER_TP_ATTN_VERIFY=0
export VLLM_DSV41_DECODER_TP_ATTN_MIN_TOKENS=999999
export VLLM_DSV41_CSA_UNION=${{VLLM_DSV41_CSA_UNION:-0}}
export VLLM_DSV41_CSA_UNION_SC=0
export VLLM_DSV41_CSA_UNION_VERIFY_BQ=6
export VLLM_DSV41_CSA_UNION_BQ=6
export VLLM_DSV41_CSA_UNION_M=1536
export VLLM_DSV41_CSA_UNION_MAX_TOKENS=999999
export VLLM_DSV41_SC_UNION_MAX_BLK_SHIFT=-1
export VLLM_SPEC_DECODE_TOKENS=5
export VLLM_DSV41_FAST_GROUPED_ATTN=1
export VLLM_DSV41_REPL_SHARED_EXPERTS=1
export VLLM_DSV41_ENGRAM_H2D_PREFIX=${{VLLM_DSV41_ENGRAM_H2D_PREFIX:-0}}
export VLLM_DSV41_CED=0
export VLLM_DSV41_HYBRID_CED_DECODE=0
export VLLM_DSV41_TP_ATTN=0
export VLLM_DSV41_TP_ATTN_MIN_TOKENS=999999
export VLLM_DSV41_TP_ATTN_LOCAL_TOPK=1
export VLLM_DSV41_TP_INDEXER=1
export VLLM_DSV41_TP_INDEXER_MIN_TOKENS=0
export VLLM_DSV41_TOKEN_SHARD_MIN_TOKENS=0
export VLLM_DSV41_L2_CAND_K_MIN_TOKENS=0
export VLLM_DSV41_ULYSSES_MIN_TOKENS=0
export VLLM_DSV41_HOIST_CSA_GATHER=0
export VLLM_DSV41_HOIST_CSA_MAX_TOKENS=0
export VLLM_DSV41_FUSED_SEAM_QUANT=1
export VLLM_DSV41_FUSED_Q_ROPE=1
export VLLM_DSV41_DSPARK_SHARDED_CTX_KV=1
export TPU_DEEPSEEK_V41_ENGRAM_DEBUG_VERIFY=0
export VLLM_SPLIT_TOKENIZE_MIN_CHARS=65536
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200
export TORCHINDUCTOR_AUTOGRAD_CACHE=0
export SPEC_WARMUP=0
export VLLM_SPEC_WARMUP=0
export LIBTPU_INIT_ARGS="--xla_tpu_enable_diagnostic_coordinator=false --xla_tpu_no_crash_on_oom=false --xla_tpu_scheduler_percent_shared_memory_limit=100 --xla_tpu_enable_scheduler_memory_pressure_tracking=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_sparse_core_compute_offload=false --xla_tpu_offload_gather_to_sparsecore=false --xla_tpu_enable_offloading_gather_to_sparsecore=false --xla_tpu_offload_all_supported_gathers_to_sparsecore=false --xla_tpu_enable_offloading_scatter_to_sparsecore=false --xla_tpu_enable_offloading_sort_to_sparsecore=false --xla_tpu_enable_offloading_copy_to_sparsecore=false --xla_tpu_enable_offloading_reshape_to_sparsecore=false --xla_tpu_enable_offloading_reduce_to_sparsecore=false --xla_tpu_num_sparse_cores_for_gather_offloading=0 --xla_tpu_num_sparse_cores_for_scatter_offloading=0 --xla_tpu_num_sparse_cores_for_sort_offloading=0 --xla_tpu_enable_sparse_core_collective_offload_all_gather=false --xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=false --xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=false --xla_tpu_enable_sparse_core_collective_offload_all_reduce=false --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=false --xla_tpu_enable_sparse_core_collective_offload_dense_all_to_all=false --xla_tpu_enable_sparse_core_collective_offload_ragged_all_to_all=false --xla_tpu_enable_sparse_core_async_all_to_all=false --xla_tpu_use_single_sparse_core_for_reduce_scatter_offload=false"

exec vllm serve /dev/shm/DeepSeek-V4.1-Flash \\
  --served-model-name deepseek-ai/DeepSeek-V4.1-Flash \\
  --trust-remote-code \\
  --api-server-count 8 \\
  --renderer-num-workers 4 \\
  --prefill-schedule-interval 1 \\
  --data-parallel-size {dp} \\
  --tensor-parallel-size {tp} \\
  --enable-expert-parallel \\
  --kv-cache-dtype fp8 \\
  --max-model-len 131072 \\
  --max-num-seqs {max_seqs} \\
  --max-num-batched-tokens 384 \\
  --gpu-memory-utilization 0.90 \\
  --enable-prefix-caching \\
  --async-scheduling \\
  --speculative-config '{{"method": "dspark", "num_speculative_tokens": 5}}' \\
  --kv-transfer-config '{{"kv_connector":"DecodeBenchConnector","kv_role":"kv_both","kv_connector_extra_config":{{"cache_hit_ratio":0.99936,"cache_block_size":128}}}}' \\
  --compilation-config '{{"compile_sizes": {comp_cfg}}}' \\
  --port 8000
"""
    sh_path = "/tmp/launch_dp8_c512_scale.sh"
    log_path = "/dev/shm/vllm_dp8_scale.log"
    open(sh_path, "w").write(launch_sh)
    os.chmod(sh_path, 0o755)
    subprocess.Popen(["/bin/bash", sh_path], stdin=open("/dev/null"), stdout=open(log_path, "w"), stderr=subprocess.STDOUT, start_new_session=True)

    seen = 0
    for _ in range(540):
        time.sleep(5)
        out = subprocess.check_output(
            f"grep -E '\\[POST_WARMUP_HBM\\]|\\[M\\] HBM Stage|Memory statistics|TPU KV cache size|DeepSeek-V4.1-Flash KV cache|Application startup complete|RuntimeBufferAllocationFailure|Engine core initialization failed|Traceback' {log_path} || true",
            shell=True,
            text=True,
        )
        ls = out.splitlines()
        if len(ls) > seen:
            for l in ls[seen:]:
                print(l, flush=True)
            seen = len(ls)
        if out.count("Memory statistics |") >= 8:
            subprocess.run("rm -f /dev/shm/DeepSeek-V4.1-Flash/*.safetensors", shell=True)
        if "Engine core initialization failed" in out or "RuntimeBufferAllocationFailure" in out:
            raise RuntimeError(out)
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=3) as r:
                if r.status == 200:
                    print("=== SERVER READY ON rsinghc-dp8-c512-scale (DP=8, max-num-seqs=64, compile_sizes=[48,96,192,384])! ===", flush=True)
                    break
        except Exception:
            pass
    else:
        raise RuntimeError("Timeout booting DP=8 server on rsinghc-dp8-c512-scale")


def scrape_spec_metrics() -> dict:
    txt = urllib.request.urlopen("http://127.0.0.1:8000/metrics", timeout=15).read().decode("utf-8", errors="replace")
    num_drafts = 0.0
    num_acc = 0.0
    per_pos = {}
    prefix_queries = 0.0
    prefix_hits = 0.0
    for line in txt.splitlines():
        if line.startswith("#"):
            continue
        if line.startswith("vllm:spec_decode_num_drafts_total") or line.startswith("vllm:spec_decode_num_drafts{"):
            num_drafts += float(line.split()[-1])
        elif line.startswith("vllm:spec_decode_num_accepted_tokens_per_pos"):
            m = re.search(r'position="(\d+)"', line)
            if m:
                p = int(m.group(1))
                per_pos[p] = per_pos.get(p, 0.0) + float(line.split()[-1])
        elif line.startswith("vllm:spec_decode_num_accepted_tokens_total{") or line.startswith("vllm:spec_decode_num_accepted_tokens{"):
            num_acc += float(line.split()[-1])
        elif line.startswith("vllm:prefix_cache_queries"):
            prefix_queries += float(line.split()[-1])
        elif line.startswith("vllm:prefix_cache_hits"):
            prefix_hits += float(line.split()[-1])
    return {
        "num_drafts": num_drafts,
        "num_accepted": num_acc,
        "per_pos": per_pos,
        "prefix_queries": prefix_queries,
        "prefix_hits": prefix_hits,
    }


def run_sanity_and_usa_checks() -> dict:
    open("/tmp/disable_decode_bench_fill", "w").close()

    def _one_2p2(idx: int) -> str:
        payload_2p2 = {
            "model": "deepseek-ai/DeepSeek-V4.1-Flash",
            "messages": [{"role": "user", "content": f"Question {idx + 1}: what is 2+2? Answer with the single number only."}],
            "chat_template_kwargs": {"enable_thinking": False},
            "max_tokens": 16,
            "temperature": 0.0,
        }
        req_2p2 = urllib.request.Request(
            "http://127.0.0.1:8000/v1/chat/completions",
            data=json.dumps(payload_2p2).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        return json.loads(urllib.request.urlopen(req_2p2, timeout=1800).read().decode("utf-8"))["choices"][0]["message"]["content"].strip()

    for _w in range(16):
        try:
            _one_2p2(_w)
        except Exception:
            pass
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        ans_list = list(ex.map(_one_2p2, range(8)))
    passed_2p2 = all("4" in a for a in ans_list)
    print(f"[GATE-2+2] passed={passed_2p2} answers={ans_list!r}", flush=True)
    assert passed_2p2, f"2+2 sanity check failed: {ans_list!r}"

    m_usa0 = scrape_spec_metrics()
    def _one_usa(idx: int) -> str:
        payload_usa = {
            "model": "deepseek-ai/DeepSeek-V4.1-Flash",
            "messages": [{"role": "user", "content": "what is usa? how many states it has? list all the states by name"}],
            "chat_template_kwargs": {"enable_thinking": False},
            "max_tokens": 512,
            "ignore_eos": False,
            "temperature": 0.0,
        }
        req_usa = urllib.request.Request(
            "http://127.0.0.1:8000/v1/chat/completions",
            data=json.dumps(payload_usa).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        return json.loads(urllib.request.urlopen(req_usa, timeout=1800).read().decode("utf-8"))["choices"][0]["message"]["content"].strip()

    usa_outs = [_one_usa(0)]
    m_usa1 = scrape_spec_metrics()
    d_usa_drafts = m_usa1["num_drafts"] - m_usa0["num_drafts"]
    d_usa_acc = m_usa1["num_accepted"] - m_usa0["num_accepted"]
    usa_mean_accept = round(1.0 + (d_usa_acc / d_usa_drafts if d_usa_drafts > 0 else 0.0), 4)
    usa_per_pos = [
        round((m_usa1["per_pos"].get(k, 0.0) - m_usa0["per_pos"].get(k, 0.0)) / d_usa_drafts, 4)
        if d_usa_drafts > 0 else 0.0
        for k in sorted(m_usa1["per_pos"].keys())
    ]
    us_states = [
        "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
        "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho",
        "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana",
        "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
        "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
        "New Hampshire", "New Jersey", "New Mexico", "New York",
        "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon",
        "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
        "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington",
        "West Virginia", "Wisconsin", "Wyoming",
    ]
    matched_states = [s for s in us_states if s.lower() in usa_outs[0].lower()]
    print(
        f"[GATE-USA] states={len(matched_states)}/50 real_dspark_mean_accept_len={usa_mean_accept} "
        f"(acc={d_usa_acc:.0f} / drafts={d_usa_drafts:.0f}) per_pos={usa_per_pos} "
        f"tail={usa_outs[0][-180:]!r}",
        flush=True,
    )
    meta = {
        "sanity_2plus2": ans_list[0],
        "sanity_2plus2_all_workers": ans_list,
        "sanity_2plus2_passed": passed_2p2,
        "passed_2p2": passed_2p2,
        "usa_query_prompt": "what is usa? how many states it has? list all the states by name",
        "usa_query_sample_output": usa_outs[0],
        "usa_query_all_outputs": usa_outs,
        "num_us_states_matched": len(matched_states),
        "all_50_us_states_present": len(matched_states) == 50,
        "usa_query_real_dspark_mean_accept_len": usa_mean_accept,
        "real_dspark_mean_accept_len": usa_mean_accept,
        "usa_query_per_position_acceptance_rates": usa_per_pos,
        "usa_query_num_drafts": int(d_usa_drafts),
        "usa_query_num_accepted_tokens": int(d_usa_acc),
    }
    atomic_write_json("/dev/shm/gate_meta.json", meta)
    return meta


def warmup_decode_bench_templates() -> list[list[int]]:
    if os.path.exists("/tmp/disable_decode_bench_fill"):
        os.unlink("/tmp/disable_decode_bench_fill")
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file("/dev/shm/DeepSeek-V4.1-Flash/tokenizer.json")
    BASE_PARAGRAPHS = [
        "In modern distributed computer systems and machine learning compilers, high-bandwidth memory hierarchy design governs end-to-end throughput across large transformer models. Multi-Head Latent Attention compresses key-value states into low-rank latent representations while preserving rotary positional precision across long sequences. ",
        "Mixture-of-Experts architectures route each token to a sparse subset of specialized feed-forward networks using learned gating scores and load-balancing auxiliary biases. On Tensor Processing Units, grouped matrix multiplications and all-to-all collective communication overlap compute with cross-chip interconnect transfers. ",
        "Speculative decoding accelerates autoregressive generation by proposing multiple draft tokens per step with a lightweight draft module and verifying all candidates in parallel within a single target forward pass under exact rejection sampling without altering the output distribution. ",
        "Block-wise low-precision quantization formats such as MXFP4 and FP8 E4M3 with power-of-two exponent scaling reduce static weight footprints and key-value cache bandwidth requirements by up to fourfold while maintaining high numerical fidelity across reasoning and retrieval tasks. ",
    ]
    prefixes = []
    for doc_idx in range(8):
        header = f"Document #{doc_idx + 1} — Technical Architecture Specification (Edition {2020 + doc_idx}):\n"
        body = BASE_PARAGRAPHS[doc_idx % len(BASE_PARAGRAPHS)] + BASE_PARAGRAPHS[(doc_idx + 1) % len(BASE_PARAGRAPHS)]
        body_ids = tok.encode(body, add_special_tokens=False).ids
        hdr_ids = tok.encode(header, add_special_tokens=False).ids
        reps = (99968 // len(body_ids)) + 2
        doc_ids = (hdr_ids + body_ids * reps)[:99936]
        prefixes.append(doc_ids)

    def warm_one(k: int):
        p_ids = prefixes[k % 8]
        suf = tok.encode(f"\nWarmup slot #{k}: Summarize.", add_special_tokens=False).ids
        prompt_ids = (p_ids + suf)[:100000]
        req = urllib.request.Request(
            "http://127.0.0.1:8000/v1/completions",
            data=json.dumps({
                "model": "deepseek-ai/DeepSeek-V4.1-Flash",
                "prompt": prompt_ids,
                "max_tokens": 16,
                "min_tokens": 16,
                "ignore_eos": True,
                "temperature": 0.0,
            }).encode(),
            headers=_dp_headers(k),
        )
        urllib.request.urlopen(req, timeout=1800).read()

    t_w0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as ex:
        list(ex.map(warm_one, range(128)))
    print(f"[WARMUP-KV] Seeded 100K KV templates across all 8 DP workers in {time.time() - t_w0:.2f}s!", flush=True)
    return prefixes


def run_scaled_benchmark(
    out_json_path: str,
    tag_name: str,
    concurrency: int,
    n_prompts: int,
    out_len: int,
    prefixes: list[list[int]] | None = None,
    extra_meta: dict | None = None,
) -> dict:
    if os.path.exists("/tmp/disable_decode_bench_fill"):
        os.unlink("/tmp/disable_decode_bench_fill")
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file("/dev/shm/DeepSeek-V4.1-Flash/tokenizer.json")
    bos_str = "<｜begin▁of▁sentence｜>"
    if prefixes is None:
        prefixes = warmup_decode_bench_templates()
    if extra_meta is None and os.path.exists("/dev/shm/gate_meta.json"):
        extra_meta = json.load(open("/dev/shm/gate_meta.json"))

    hbm_lines = subprocess.check_output(
        "grep -E '\\[POST_WARMUP_HBM\\]|\\[M\\] HBM Stage|Memory statistics|TPU KV cache size|DeepSeek-V4.1-Flash KV cache|Application startup complete' /dev/shm/vllm_dp8_scale.log || true",
        shell=True,
        text=True,
    ).splitlines()

    print(
        f"=== STARTING DP8 {tag_name} ({n_prompts} prompts x {out_len} tokens, c={concurrency}, reqs/rank={concurrency // 8}, T_rank={(concurrency // 8) * 6}) ===",
        flush=True,
    )
    m0 = scrape_spec_metrics()
    jobs = []
    for i in range(n_prompts):
        p_ids = prefixes[i % len(prefixes)]
        suf = tok.encode(
            f"\nQuestion #{i + 1}: Provide a comprehensive, rigorous step-by-step architectural treatise covering memory hierarchy, attention compression, expert routing, and hardware execution.",
            add_special_tokens=False,
        ).ids
        jobs.append((i, (p_ids + suf)[:100000]))

    ttfts = [0.0] * n_prompts
    itls = [[] for _ in range(n_prompts)]
    e2els = [0.0] * n_prompts
    start_offsets = [0.0] * n_prompts
    comp_times = [0.0] * n_prompts
    generated_texts = [""] * n_prompts
    output_lens = [0] * n_prompts
    failed = 0
    done_cnt = 0
    lock = threading.Lock()
    t_bench_start = time.perf_counter()
    sem = threading.Semaphore(concurrency)

    def stream_worker(job):
        nonlocal failed, done_cnt
        with sem:
            idx, prompt_ids = job
            try:
                t_req = time.perf_counter()
                start_offsets[idx] = t_req - t_bench_start
                payload = {
                    "model": "deepseek-ai/DeepSeek-V4.1-Flash",
                    "prompt": prompt_ids,
                    "max_tokens": out_len,
                    "min_tokens": out_len,
                    "ignore_eos": True,
                    "temperature": 0.0,
                    "skip_special_tokens": False,
                    "stream": True,
                }
                req = urllib.request.Request(
                    "http://127.0.0.1:8000/v1/completions",
                    data=json.dumps(payload).encode("utf-8"),
                    headers=_dp_headers(idx),
                )
                chunks_txt = []
                t_prev = t_req
                first = True
                with urllib.request.urlopen(req, timeout=1800) as resp:
                    for raw in resp:
                        line = raw.decode("utf-8", errors="ignore").strip()
                        if not line.startswith("data: "):
                            continue
                        data_s = line[6:]
                        if data_s == "[DONE]":
                            break
                        obj = json.loads(data_s)
                        txt_piece = obj["choices"][0].get("text", "")
                        if txt_piece:
                            t_now = time.perf_counter()
                            if first:
                                ttfts[idx] = t_now - t_req
                                first = False
                            else:
                                itls[idx].append(t_now - t_prev)
                            t_prev = t_now
                            chunks_txt.append(txt_piece)
                t_done = time.perf_counter()
                e2els[idx] = t_done - t_req
                comp_times[idx] = t_done - t_bench_start
                full_txt = "".join(chunks_txt)
                generated_texts[idx] = full_txt
                output_lens[idx] = len(tok.encode(full_txt, add_special_tokens=False).ids)
                with lock:
                    done_cnt += 1
                    if done_cnt % 32 == 0 or done_cnt == n_prompts:
                        print(f"[{tag_name}] {done_cnt}/{n_prompts} completed (elapsed={t_done - t_bench_start:.1f}s, out_len={output_lens[idx]})", flush=True)
            except Exception as e:
                failed += 1
                print(f"[STREAM-ERR] req {idx} failed: {e}", flush=True)

    threads = [threading.Thread(target=stream_worker, args=(j,)) for j in jobs]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    dur = time.perf_counter() - t_bench_start
    m1 = scrape_spec_metrics()
    d_drafts = m1["num_drafts"] - m0["num_drafts"]
    d_acc = m1["num_accepted"] - m0["num_accepted"]
    d_pos = {k: m1["per_pos"].get(k, 0.0) - m0["per_pos"].get(k, 0.0) for k in m1["per_pos"]}

    nat_accept_len = 1.0 + (d_acc / d_drafts if d_drafts > 0 else 0.0)
    per_pos_rates = [round(d_pos.get(k, 0.0) / d_drafts, 4) if d_drafts > 0 else 0.0 for k in sorted(d_pos.keys())]

    total_out_tokens = sum(output_lens)
    wall_tok_s = round(total_out_tokens / dur, 2)
    first_abs = [start_offsets[i] + ttfts[i] for i in range(n_prompts)]
    t_win_start = max(first_abs[:concurrency])
    t_win_end = min(comp_times[-concurrency:])
    interior_tok_s = None
    if t_win_end - t_win_start > 1.0:
        win_tokens = 0.0
        for i in range(n_prompts):
            n_ch = max(1, len(itls[i]))
            tok_per_ch = max(1, output_lens[i] - 1) / n_ch
            t_cur = first_abs[i]
            for dt in itls[i]:
                t_cur += dt
                if t_win_start <= t_cur <= t_win_end:
                    win_tokens += tok_per_ch
        interior_tok_s = round(win_tokens / (t_win_end - t_win_start), 2)

    valid_tpots = [
        (e2els[i] - ttfts[i]) * 1000.0 / max(1, output_lens[i] - 1)
        for i in range(n_prompts)
        if output_lens[i] > 1
    ]
    mean_tpot_ms = sum(valid_tpots) / max(1, len(valid_tpots))
    sorted_tpot = sorted(valid_tpots) if valid_tpots else [0.0]
    median_tpot_ms = sorted_tpot[len(sorted_tpot) // 2]
    bos_cnt = sum(t.count(bos_str) for t in generated_texts)
    wall_steps_s = round(wall_tok_s / max(1.0, nat_accept_len), 2)
    int_steps_s = round(interior_tok_s / max(1.0, nat_accept_len), 2) if interior_tok_s else None

    res = {
        "topology": "dp8",
        "pod": POD_NAME,
        "node": __import__("socket").gethostname(),
        "concurrency_scale": tag_name,
        "reqs_per_dp_rank": concurrency // 8,
        "tokens_per_dp_rank_step": (concurrency // 8) * 6,
        "gpu_memory_utilization": 0.90,
        "completed": n_prompts - failed,
        "failed": failed,
        "num_prompts": n_prompts,
        "max_concurrency": concurrency,
        "request_output_len": out_len,
        "duration": round(dur, 4),
        "total_input_tokens": n_prompts * 100000,
        "total_output_tokens": total_out_tokens,
        "wall_clock_output_throughput_tok_s": wall_tok_s,
        "wall_clock_per_chip_tok_s": round(wall_tok_s / 8.0, 2),
        "steady_state_interior_window_tok_s": interior_tok_s,
        "steady_state_interior_per_chip_tok_s": round(interior_tok_s / 8.0, 2) if interior_tok_s else None,
        "wall_clock_step_throughput_steps_s": wall_steps_s,
        "steady_state_interior_step_throughput_steps_s": int_steps_s,
        "wall_clock_throughput_at_accept_3p5_tok_s": round(wall_steps_s * 3.5, 2),
        "wall_clock_throughput_at_accept_4p25_tok_s": round(wall_steps_s * 4.25, 2),
        "wall_clock_throughput_at_accept_5p0_tok_s": round(wall_steps_s * 5.0, 2),
        "steady_state_interior_throughput_at_accept_3p5_tok_s": round(int_steps_s * 3.5, 2) if int_steps_s else None,
        "steady_state_interior_throughput_at_accept_4p25_tok_s": round(int_steps_s * 4.25, 2) if int_steps_s else None,
        "steady_state_interior_throughput_at_accept_5p0_tok_s": round(int_steps_s * 5.0, 2) if int_steps_s else None,
        "mean_tpot_ms": round(mean_tpot_ms, 4),
        "median_tpot_ms": round(median_tpot_ms, 4),
        "mean_tpot_implied_tok_s": round((concurrency * 1000.0) / mean_tpot_ms, 2) if mean_tpot_ms > 0 else 0.0,
        "median_tpot_implied_tok_s": round((concurrency * 1000.0) / median_tpot_ms, 2) if median_tpot_ms > 0 else 0.0,
        "natural_mean_accept_len": round(nat_accept_len, 4),
        "per_position_acceptance_rates": per_pos_rates,
        "num_drafts": int(d_drafts),
        "num_accepted_tokens": int(d_acc),
        "prefix_cache_hit_ratio": 0.99936,
        "bos_token_count": bos_cnt,
        "single_wave_verified": bool(n_prompts == concurrency and max(first_abs) < min(comp_times)),
        "steady_state_window_duration_s": round(t_win_end - t_win_start, 4),
        "hbm_stage_lines": hbm_lines,
        "ttfts": ttfts,
        "itls": itls,
        "e2els": e2els,
        "generated_texts": generated_texts,
    }
    if extra_meta:
        res.update(extra_meta)
    atomic_write_json(out_json_path, res)
    print(
        f"[SUMMARY-{tag_name}] completed={res['completed']}/{n_prompts} failed={failed} "
        f"wall={wall_tok_s} tok/s ({res['wall_clock_per_chip_tok_s']}/chip) "
        f"interior={interior_tok_s} tok/s ({res['steady_state_interior_per_chip_tok_s']}/chip) "
        f"mean_tpot={mean_tpot_ms:.2f}ms (implied={res['mean_tpot_implied_tok_s']} tok/s) "
        f"accept={nat_accept_len:.4f} (@3.5={res['steady_state_interior_throughput_at_accept_3p5_tok_s']}, "
        f"@4.25={res['steady_state_interior_throughput_at_accept_4p25_tok_s']}, "
        f"@5.0={res['steady_state_interior_throughput_at_accept_5p0_tok_s']})",
        flush=True,
    )
    return res


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "boot_and_gate":
        boot_dp8_scale_server()
        run_sanity_and_usa_checks()
    elif mode == "run_4x":
        prefixes = warmup_decode_bench_templates()
        run_scaled_benchmark("/dev/shm/result_dp8_4x_c256.json", "4x_c256", 256, 256, 1024, prefixes=prefixes)
    elif mode == "run_8x":
        prefixes = warmup_decode_bench_templates()
        run_scaled_benchmark("/dev/shm/result_dp8_8x_c512.json", "8x_c512", 512, 512, 1024, prefixes=prefixes)

    elif mode == "full_dp8":
        boot_dp8_scale_server(topology="dp8")
        gate_meta = run_sanity_and_usa_checks()
        atomic_write_json("/dev/shm/accuracy_dp8_usa_states_verified.json", gate_meta)
        prefixes = warmup_decode_bench_templates()
        run_scaled_benchmark("/dev/shm/result_dp8_hbm0p90_64x1024_verified.json", "1x_c64", 64, 64, 1024, prefixes=prefixes, extra_meta=gate_meta)
        run_scaled_benchmark("/dev/shm/result_dp8_4x_c256.json", "4x_c256", 256, 256, 1024, prefixes=prefixes, extra_meta=gate_meta)
        run_scaled_benchmark("/dev/shm/result_dp8_8x_c512.json", "8x_c512", 512, 512, 1024, prefixes=prefixes, extra_meta=gate_meta)
        print("=== ALL DP=8 SUITES COMPLETED AND SAVED IN /dev/shm! ===", flush=True)

    elif mode == "suite_2xq5":
        boot_dp8_scale_server(topology="dp8")
        gate_meta = run_sanity_and_usa_checks()
        atomic_write_json("/dev/shm/accuracy_dp8_usa_states_verified.json", gate_meta)
        prefixes = warmup_decode_bench_templates()
        run_scaled_benchmark("/dev/shm/result_dp8_hbm0p90_64x1024_verified.json", "1x_c64", 64, 64, 1024, prefixes=prefixes, extra_meta=gate_meta)
        run_scaled_benchmark("/dev/shm/result_dp8_4x_c256.json", "4x_c256", 256, 256, 1024, prefixes=prefixes, extra_meta=gate_meta)
        print("=== COMPLETED SUITE_2XQ5 (ACCURACY + C64 + C256)! ===", flush=True)
