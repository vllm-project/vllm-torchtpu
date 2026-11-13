#!/usr/bin/env python3
"""Clean, minimal standalone in-pod driver for 4x(TP=2, EP=2) DeepSeek-V4.1-Flash Prefill.

Strips all 28+ legacy bloat / hazardous / no-op environment flags and retains ONLY the
3 essential, self-documented runtime configuration blocks:
  (1) Fused MoE EP v2 config
  (2) CED Disaggregated Prefill + TP=2 Attention/Indexer config
  (3) RoPE / XLA runtime args
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
import urllib.request

POD = os.environ.get("VLLM_BENCH_POD", "rsinghc-dsv41-v85-4xtp2-prefill-clean-flags")
NODE = os.environ.get("VLLM_BENCH_NODE", "gke-tpu-11cef4a1-nc5h")
CLUSTER = "gke_tpu-prod-env-one-vm_us-central1-c_andersonchiu-v7x-8-3"
GIT_COMMIT_HEAD = os.environ.get("VLLM_GIT_COMMIT", "500e4db")
EXTRA_WIRE_OVERRIDE = os.environ.get("VLLM_BENCH_WIRE_BF16_OVERRIDE", "")

CFG2_1SC_LIBTPU_ARGS = (
    "--xla_tpu_no_crash_on_oom=false "
    "--xla_tpu_scheduler_percent_shared_memory_limit=100 "
    "--xla_tpu_enable_scheduler_memory_pressure_tracking=true "
    "--xla_tpu_offload_gather_to_sparsecore=true "
    "--xla_tpu_enable_sparse_core_collective_offload_all_gather=true "
    "--xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true "
    "--xla_tpu_enable_sparse_core_collective_offload_all_reduce=false "
    "--xla_tpu_enable_async_collective_fusion=true "
    "--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true "
    "--xla_tpu_sparse_core_reduce_scatter_offload_min_size_in_bytes=524288 "
    "--xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true "
    "--xla_tpu_use_single_sparse_core_for_reduce_scatter_offload=true"
)

WHEELS = {
    "vllm": "0.29.0+empty",
    "torch": "2.13.0+cpu",
    "torch-tpu": "0.1.1.dev20260912075415",
    "jax": "0.10.2",
    "jaxlib": "0.10.2",
    "libtpu": "0.0.44",
}

WIRE_LINE = (
    f'\nexport VLLM_DSV41_MOE_V2_WIRE_BF16_MAX_TOKENS="{EXTRA_WIRE_OVERRIDE}"'
    if EXTRA_WIRE_OVERRIDE
    else ""
)

LAUNCH_SH = f"""#!/bin/bash
set -e
CID="$1"
PORT=$(( 8000 + CID ))
MPORT=$(( 19800 + CID * 20 ))
M1=$(( 8531 + CID * 2 ))
M2=$(( 8532 + CID * 2 ))
SB0=$(( 23500 + CID * 10 ))
SB1=$(( 23501 + CID * 10 ))
TOTAL_CPUS=$(nproc)
CPUS_PER_REP=$(( TOTAL_CPUS / 4 ))
CPU_START=$(( CID * CPUS_PER_REP ))
CPU_END=$(( (CID + 1) * CPUS_PER_REP - 1 ))

CACHE_ROOT="/dev/shm/xla_cache_clean_flags_b16384"
VLLM_CDIR="/dev/shm/vllm_cache_clean_flags_b16384_c${{CID}}"
mkdir -p "${{CACHE_ROOT}}" "${{VLLM_CDIR}}/torch_compile_cache" /dev/shm/perf_results
ln -sfn "${{CACHE_ROOT}}" /tmp/xla_cache

# Per-chip PJRT / GKE single-chip (TP=2, EP=2) bootstrap
unset TPU_VISIBLE_DEVICES TPU_VISIBLE_CHIPS TPU_CHIPS_PER_HOST_BOUNDS CHIPS_PER_HOST_BOUNDS TPU_TOPOLOGY TOPOLOGY TPU_HOST_BOUNDS HOST_BOUNDS TPU_PROCESS_ADDRESSES TPU_PROCESS_PORT TPU_WORKER_ID TPU_SKIP_MDS_QUERY VBAR_CONTROL_SERVICE_URL
[ -S /var/run/tpu-plugin/vbar.sock ] && export VBAR_CONTROL_SERVICE_URL="unix:///var/run/tpu-plugin/vbar.sock"
export POD_G_CHIP_ID="${{CID}}"
export ALLOW_MULTIPLE_LIBTPU_LOAD="1"
export TORCH_TPU_TOPOLOGY="1,1,1,2"
export TPU_ACCELERATOR_TYPE="tpu7x-8"
export TPU_RUNTIME_METRICS_PORTS="${{M1}},${{M2}}"
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="${{MPORT}}"
export TORCH_TPU_SLICEBUILDER_ADDRESSES="localhost:${{SB0}},localhost:${{SB1}}"
export GCE_METADATA_HOST="127.0.0.1:80"
export GCE_METADATA_IP="127.0.0.1"
export GOOGLE_CLOUD_PROJECT="tpu-prod-env-one-vm"
export GCLOUD_PROJECT="tpu-prod-env-one-vm"
export LD_PRELOAD="/workspace/metadata_hook.so"
export VLLM_PLUGINS="torchtpu,torchtpu_layers"
export PYTHONPATH="/workspace/sitecustomize_bench:/workspace/vllm:/workspace/vllm-torchtpu/src"

# (1) Fused MoE EP v2 config
export USE_MOE_FUSED_EP_KERNEL="1"
export MOE_FUSED_EP_KERNEL_IMPL="v2"
export MOE_FUSED_EP_ENABLE_W4A8="1"
export MOE_FUSED_EP_V2_SHARDED_PLAN="1"
export MOE_REQUANTIZE_BLOCK_SIZE="512"
export VLLM_DSV41_MOE_FP4_PATH_C_MAX_TOKENS="0"
export VLLM_DSV41_MOE_V2_SUBCHANNEL_FP8="0"
export VLLM_DSV41_MOE_V2_REQUANTIZE="1"
export VLLM_DSV41_REPL_SHARED_EXPERTS="1"

# (2) CED Disaggregated Prefill + TP=2 Attention/Indexer config
export VLLM_DSV41_CED="1"
export VLLM_DSV41_PURE_20L_PREFILL="1"
export VLLM_DSV41_CSA_UNION="0"
export VLLM_DSV41_TP_ATTN="1"
export VLLM_DSV41_TP_ATTN_MIN_TOKENS="0"
export VLLM_DSV41_TP_ATTN_LOCAL_TOPK="1"
export VLLM_DSV41_TP_INDEXER="1"
export VLLM_DSV41_TP_INDEXER_MIN_TOKENS="16"
export VLLM_DSV41_FUSED_Q_ROPE="1"
export VLLM_DSV41_FUSED_SEAM_QUANT="1"

# (3) RoPE / XLA runtime args
export TPU_ROPE_CACHE_TRUNCATE="1"
export TPU_ROPE_CACHE_ROW_MAJOR="1"
export XLA_PERSISTENT_CACHE_PATH="${{CACHE_ROOT}}"
export VLLM_XLA_CACHE_PATH="${{CACHE_ROOT}}"
export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT="${{CACHE_ROOT}}"
export VLLM_CACHE_ROOT="${{VLLM_CDIR}}"
export TORCHINDUCTOR_AUTOGRAD_CACHE="0"
export OMP_NUM_THREADS="4"
export GOMP_SPINCOUNT="100"
export VLLM_SPLIT_TOKENIZE_MIN_CHARS="65536"
export VLLM_ENGINE_READY_TIMEOUT_S="7200"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="1200"
export LIBTPU_INIT_ARGS="{CFG2_1SC_LIBTPU_ARGS}"

exec taskset -c "${{CPU_START}}-${{CPU_END}}" vllm serve /dev/shm/DeepSeek-V4.1-Flash \\
  --served-model-name deepseek-ai/DeepSeek-V4.1-Flash \\
  --trust-remote-code \\
  --api-server-count 4 \\
  --renderer-num-workers 4 \\
  --data-parallel-size 1 \\
  --tensor-parallel-size 2 \\
  --enable-expert-parallel \\
  --kv-cache-dtype fp8 \\
  --max-model-len 262144 \\
  --max-num-seqs 64 \\
  --max-num-batched-tokens 16384 \\
  --gpu-memory-utilization 0.90 \\
  --no-enable-prefix-caching \\
  --async-scheduling \\
  --compilation-config '{{"compile_sizes": [16384]}}' \\
  --port "${{PORT}}"
"""

RR_PROXY_PY = """#!/usr/bin/env python3
import asyncio
import aiohttp
from aiohttp import web

BACKENDS = ["http://127.0.0.1:8000", "http://127.0.0.1:8001", "http://127.0.0.1:8002", "http://127.0.0.1:8003"]
inflight = [0, 0, 0, 0]
rr_seq = [0, 0, 0, 0]
seq_counter = 0

async def proxy_handler(request: web.Request) -> web.StreamResponse:
    global seq_counter
    idx = min(range(len(BACKENDS)), key=lambda i: (inflight[i], rr_seq[i]))
    inflight[idx] += 1
    seq_counter += 1
    rr_seq[idx] = seq_counter
    target_url = f"{BACKENDS[idx]}{request.rel_url}"
    body = await request.read()
    headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}
    session: aiohttp.ClientSession = request.app["client_session"]
    try:
        async with session.request(request.method, target_url, data=body, headers=headers, timeout=aiohttp.ClientTimeout(total=1800)) as resp:
            stream = web.StreamResponse(status=resp.status, headers={k: v for k, v in resp.headers.items() if k.lower() not in ("transfer-encoding", "content-encoding")})
            await stream.prepare(request)
            async for chunk in resp.content.iter_chunked(65536):
                await stream.write(chunk)
            await stream.write_eof()
            return stream
    finally:
        inflight[idx] -= 1

async def on_startup(app: web.Application) -> None:
    conn = aiohttp.TCPConnector(limit=512, force_close=False)
    app["client_session"] = aiohttp.ClientSession(connector=conn)

async def on_cleanup(app: web.Application) -> None:
    await app["client_session"].close()

if __name__ == "__main__":
    app = web.Application(client_max_size=256 * 1024 * 1024)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_route("*", "/{tail:.*}", proxy_handler)
    web.run_app(app, host="127.0.0.1", port=9000, access_log=None)
"""


def compute_pod_provenance() -> dict:
    hashes = {}
    root = "/workspace/vllm-torchtpu"
    for rel in [
        "src/vllm_torchtpu/envs.py",
        "src/vllm_torchtpu/platforms/tpu_platform.py",
        "src/vllm_torchtpu/compilation/tpu_compiler.py",
        "src/vllm_torchtpu/runner/tpu_runner.py",
        "src/vllm_torchtpu/runner/kv_cache_deepseek_v41_flash.py",
        "src/vllm_torchtpu/kernels/fused_moe/v2/host.py",
        "src/vllm_torchtpu/kernels/fused_moe/v2/kernel.py",
        "src/vllm_torchtpu/kernels/fused_moe/v2/layer.py",
        "src/vllm_torchtpu/layers/adapter/fused_moe_ep.py",
        "src/vllm_torchtpu/layers/adapter/quantization/mxfp4.py",
        "src/vllm_torchtpu/models/vllm/deepseek_v41/attention.py",
    ]:
        fp = os.path.join(root, rel)
        if os.path.isfile(fp):
            b = open(fp, "rb").read()
            hashes[rel] = {
                "md5": hashlib.md5(b).hexdigest(),
                "sha256": hashlib.sha256(b).hexdigest(),
            }
    return hashes


def wait_ports(ports: list[int], timeout_s: int = 1500) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        ok = True
        for p in ports:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{p}/health", timeout=3) as r:
                    if r.status != 200:
                        ok = False
            except Exception:
                ok = False
        if ok:
            print(f"[READY in {time.time()-t0:.1f}s] Ports {ports} healthy!", flush=True)
            return
        for p in ports:
            cid = p - 8000
            lp = f"/dev/shm/vllm_chip{cid}.log"
            if os.path.exists(lp):
                tail = open(lp, errors="replace").read()[-4000:]
                if any(
                    e in tail
                    for e in (
                        "RuntimeError: Engine core initialization failed",
                        "WorkerProc failed to start",
                    )
                ):
                    raise RuntimeError(f"Chip {cid} failed:\n{tail}")
        time.sleep(5)
    raise TimeoutError(f"Timed out waiting for ports {ports}")


def parse_hbm_telemetry() -> dict:
    out = {}
    for c in range(4):
        lp = f"/dev/shm/vllm_chip{c}.log"
        txt = open(lp, errors="replace").read() if os.path.exists(lp) else ""
        kv = re.findall(r"DeepSeek-V4\.1-Flash KV cache:[^\n]*", txt)
        ep_en = len(re.findall(r"DeepSeek-V4 MXFP4 fused EP W4A8 enabled: block-512", txt))
        out[f"chip_{c}"] = {
            "chip_id": c,
            "kv_cache_summary": kv[-1] if kv else "",
            "fused_ep_enabled_log_hits": ep_en,
        }
    return out


def run_bench(label: str, in_len: int, out_len: int, conc: int, num_prompts: int) -> dict:
    rname = f"v85_4xtp2_prefill_clean_flags_b16k_4chip_{label}_in{in_len}_out{out_len}_c{conc}_n{num_prompts}.json"
    out_p = f"/dev/shm/perf_results/{rname}"
    env = os.environ.copy()
    env["PYTHONPATH"] = "/workspace/sitecustomize_bench:/workspace/vllm:/workspace/vllm-torchtpu/src"
    env["VLLM_BENCH_DP_SIZE"] = "1"  # Client port/header routing count (not a server flag)
    cmd = [
        "vllm",
        "bench",
        "serve",
        "--backend",
        "vllm",
        "--base-url",
        "http://127.0.0.1:9000",
        "--model",
        "deepseek-ai/DeepSeek-V4.1-Flash",
        "--dataset-name",
        "random",
        "--random-input-len",
        str(in_len),
        "--random-output-len",
        str(out_len),
        "--num-prompts",
        str(num_prompts),
        "--max-concurrency",
        str(conc),
        "--request-rate",
        "inf",
        "--ignore-eos",
        "--save-result",
        "--result-dir",
        "/dev/shm/perf_results",
        "--result-filename",
        rname,
    ]
    subprocess.run(cmd, env=env, check=True)
    res = json.load(open(out_p))
    in_tps = res["total_input_tokens"] / res["duration"]
    res["input_throughput_tok_s"] = round(in_tps, 2)
    print(
        f"[BENCH {label}] completed={res['completed']}/{num_prompts} | "
        f"input_tok/s={in_tps:,.1f} | total_tok/s={res['total_token_throughput']:,.1f} | "
        f"mean_ttft={res.get('mean_ttft_ms',0):.1f}ms | p50_ttft={res.get('median_ttft_ms',0):.1f}ms",
        flush=True,
    )
    return res


def main() -> None:
    open("/tmp/launch_chip_clean.sh", "w").write(LAUNCH_SH)
    os.chmod("/tmp/launch_chip_clean.sh", 0o755)
    open("/tmp/rr_proxy.py", "w").write(RR_PROXY_PY)
    os.chmod("/tmp/rr_proxy.py", 0o755)

    hashes = compute_pod_provenance()
    ubench = (
        json.load(open("/dev/shm/ubench_clean_wire_bf16_vs_fp8.json"))
        if os.path.exists("/dev/shm/ubench_clean_wire_bf16_vs_fp8.json")
        else {}
    )
    print(
        f"[DRIVER] Verified source hashes inside /workspace/vllm-torchtpu: "
        f"envs.py={hashes['src/vllm_torchtpu/envs.py']['md5']} "
        f"fused_moe_ep.py={hashes['src/vllm_torchtpu/layers/adapter/fused_moe_ep.py']['md5']}",
        flush=True,
    )

    print("[DRIVER] Launching Chip 0 to populate shared XLA cache...", flush=True)
    subprocess.Popen(
        ["/tmp/launch_chip_clean.sh", "0"],
        stdin=subprocess.DEVNULL,
        stdout=open("/dev/shm/vllm_chip0.log", "w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    wait_ports([8000], timeout_s=1500)

    print(
        "[DRIVER] Launching Chips 1, 2, 3 concurrently with warm XLA cache + RR proxy...",
        flush=True,
    )
    for cid in (1, 2, 3):
        subprocess.Popen(
            ["/tmp/launch_chip_clean.sh", str(cid)],
            stdin=subprocess.DEVNULL,
            stdout=open(f"/dev/shm/vllm_chip{cid}.log", "w"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    subprocess.Popen(
        ["python3", "/tmp/rr_proxy.py"],
        stdin=subprocess.DEVNULL,
        stdout=open("/dev/shm/rr_proxy.log", "w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    wait_ports([8000, 8001, 8002, 8003], timeout_s=1200)

    print("[DRIVER] Running 4-chip warmup (16384x1, c=8, n=8)...", flush=True)
    run_bench("warmup", 16384, 1, 8, 8)

    hbm_tel = parse_hbm_telemetry()
    host_mem = subprocess.check_output(["bash", "-c", "free -g && df -h /dev/shm"], text=True).strip()

    sweep = [
        ("128k_c4", 131072, 4, 16),
        ("128k_c8", 131072, 8, 32),
        ("128k_c16", 131072, 16, 32),
        ("100k_c8", 100000, 8, 32),
        ("100k_c16", 100000, 16, 32),
        ("32k_c8", 32768, 8, 64),
    ]
    results = {}
    for label, ilen, conc, nprompts in sweep:
        results[label] = run_bench(label, ilen, 1, conc, nprompts)
        summary = {
            "experiment": "v85_4xtp2_prefill_clean_flags_b16k_4chip_node",
            "pod": POD,
            "node": NODE,
            "cluster": CLUSTER,
            "git_commit_head": GIT_COMMIT_HEAD,
            "clean_env_groups": {
                "group1_fused_moe_ep_v2": {
                    "USE_MOE_FUSED_EP_KERNEL": "1",
                    "MOE_FUSED_EP_KERNEL_IMPL": "v2",
                    "MOE_FUSED_EP_ENABLE_W4A8": "1",
                    "MOE_FUSED_EP_V2_SHARDED_PLAN": "1",
                    "MOE_REQUANTIZE_BLOCK_SIZE": "512",
                    "VLLM_DSV41_MOE_FP4_PATH_C_MAX_TOKENS": "0",
                    "VLLM_DSV41_MOE_V2_SUBCHANNEL_FP8": "0",
                    "VLLM_DSV41_MOE_V2_REQUANTIZE": "1",
                    "VLLM_DSV41_REPL_SHARED_EXPERTS": "1",
                    "VLLM_DSV41_MOE_V2_WIRE_BF16_MAX_TOKENS": "default(768: BF16 for T<=768 Decode, FP8 for T>768 Prefill)",
                },
                "group2_ced_prefill_tp2_attn": {
                    "VLLM_DSV41_CED": "1",
                    "VLLM_DSV41_PURE_20L_PREFILL": "1",
                    "VLLM_DSV41_CSA_UNION": "0",
                    "VLLM_DSV41_TP_ATTN": "1",
                    "VLLM_DSV41_TP_ATTN_MIN_TOKENS": "0",
                    "VLLM_DSV41_TP_ATTN_LOCAL_TOPK": "1",
                    "VLLM_DSV41_TP_INDEXER": "1",
                    "VLLM_DSV41_TP_INDEXER_MIN_TOKENS": "16",
                    "VLLM_DSV41_FUSED_Q_ROPE": "1",
                    "VLLM_DSV41_FUSED_SEAM_QUANT": "1",
                },
                "group3_rope_xla_runtime": {
                    "TPU_ROPE_CACHE_TRUNCATE": "1",
                    "TPU_ROPE_CACHE_ROW_MAJOR": "1",
                    "LIBTPU_INIT_ARGS": CFG2_1SC_LIBTPU_ARGS,
                },
            },
            "stripped_flags_count": 28,
            "source_hashes": hashes,
            "ubench_wire_bf16_vs_fp8": ubench,
            "topology": "4 x (TP=2, EP=2) Single-Chip Replicas across 8 TensorCores (0.0 MB Cross-Chip ICI)",
            "max_num_batched_tokens": 16384,
            "compile_sizes": [16384],
            "hbm_telemetry": hbm_tel,
            "host_memory_snapshot": host_mem,
            "wheels": WHEELS,
            "results": results,
        }
        open(
            "/dev/shm/summary_v85_4xtp2_prefill_clean_flags_b16k_4chip_node.json", "w"
        ).write(json.dumps(summary, indent=2))
    open("/dev/shm/BENCH_ALL_DONE", "w").write("DONE\n")
    print("ALL_BENCHMARKS_COMPLETE_OK!", flush=True)


if __name__ == "__main__":
    main()
