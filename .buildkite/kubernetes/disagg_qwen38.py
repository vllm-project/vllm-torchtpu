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
"""Role-specific Qwen PD recipes for one eight-rank host per engine."""

import argparse
import json
import os
from pathlib import Path

MODEL = "Qwen/Qwen3.8-Flash-Next-FP8"
CONNECTOR = "vllm_torchtpu.models.vllm.qwen4_exp.pd_connector"


def role_profile(
    role: str,
    dp: int,
    model: str,
    host: str,
    instance: str = "qwen38-pd",
    *,
    apc: bool = False,
) -> dict:
    if role not in ("prefill", "decode") or dp not in (4, 8):
        raise ValueError("Expected prefill/decode and DP4 or DP8 on eight ranks")
    # Each 2x2x1 host supplies eight logical ranks. EP spans TP times DP.
    tp = 8 // dp
    prefill = role == "prefill"
    backend = "torch" if prefill else "native_dma"
    port = 27000 if prefill else 28000
    env = {
        "VLLM_HOST_IP": host,
        "PJRT_DEVICE": "TPU",
        "JAX_PLATFORMS": "tpu,cpu",
        "TPU_BACKEND_TYPE": "jax",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "TORCH_DEVICE_BACKEND_AUTOLOAD": "1",
        "TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS": "false",
        "TORCH_TPU_TIER2_COMPILATION_CACHE": "disabled",
        "MOE_REQUANTIZE_WEIGHT_DTYPE": "float8_e4m3fn",
        "TPU_KV_RESHARD_TRANSPORT": "raiden",
        "TPU_USE_RAIDEN_KV_CACHE_MANAGER": "1",
        "TPU_RAIDEN_RESHARD_IMPL": "store",
        "TPU_RAIDEN_TRANSFER_PARALLELISM": str(tp),
        "TPU_RAIDEN_DST_SHARDS": str(tp),
        "TPU_RAIDEN_ADVERTISE_HOST": host,
        "TPU_RAIDEN_ENGINE_ID": f"{instance}-{role}",
        "TPU_RAIDEN_JOB_NAME": f"{instance}-{role}",
        "TPU_KV_TRANSFER_NAMESPACE": f"{instance}-{role}",
        "TPU_KV_TRANSFER_PORT": "9100",
        "TPU_SIDE_CHANNEL_PORT": "9600",
        "TPU_RAIDEN_RESHARD_PORT_BASE": str(port),
        "TPU_RAIDEN_STORE_DISPATCH_PORT_BASE": str(port + 100),
        "TPU_P2P_WAIT_PULL_TIMEOUT": "600",
        "TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL": "1",
        "TPU_RAIDEN_QWEN35_ADMISSION": "0",
        "TPU_RAIDEN_KIMIK3_ADMISSION": "0",
        "TPU_RAIDEN_GLM_ADMISSION": "0",
        "TPU_RAIDEN_DSV4_ADMISSION": "0",
        "TPU_RAIDEN_PREFIX_AWARE_LOAD": "1" if apc else "0",
        # The model hook resolves this environment override before admission.
        "VLLM_TPU_QWEN38_FLASH_PLE_HOST_BACKEND": backend,
        "VLLM_TPU_HC_PREFILL_TILE": "256",
        "VLLM_TPU_HC_PACKED_MIX": "1",
        "TPU_MOE_DENSE_EXPERT_THRESHOLD": "0" if prefill else "128",
        "USE_MOE_FUSED_EP_KERNEL": "1" if prefill else "0",
        "MOE_FUSED_EP_V2_SHARDED_PLAN": "1" if prefill else "0",
        "FUSED_MOE_DECODE_KERNEL": "0",
        "USE_MOE_TP_DECODE_KERNEL": "0",
        "TPU_PREMAPPED_BUFFER_SIZE": str(4 << 30),
        "TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE": "disabled",
        "TPU_PARALLEL_PRECOMPILE": "1",
        "TPU_PRECOMPILE_PEER_REPLAY": "1",
        "VLLM_ENGINE_READY_TIMEOUT_S": "6800",
        "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
    }
    if prefill:
        env["MOE_FUSED_EP_KERNEL_IMPL"] = "v2"
    if apc:
        env["VLLM_SERVER_DEV_MODE"] = "1"
        env["TPU_RAIDEN_POOL_STAGING_LEASES"] = "1"
    qwen = dict(
        ple_host_backend=backend,
        engram_config=dict(
            cpu_offload=True, dp_shared_memory=prefill, embedding_across_dp=not prefill
        ),
    )
    compilation = dict(
        backend="vllm_torchtpu.compilation.tpu_compiler.TpuCompilerAdaptor",
        compile_sizes=[8192] if prefill else [8, 8192],
    )
    transfer = dict(
        kv_connector="Qwen4ExpRaidenConnector",
        kv_connector_module_path=CONNECTOR,
        kv_role="kv_producer" if prefill else "kv_consumer",
        kv_port=14579,
        kv_load_failure_policy="fail",
    )
    args = [
        "vllm",
        "serve",
        "--model",
        model,
        "--trust-remote-code",
        "--seed=42",
        "--dtype=bfloat16",
        "--kv-cache-dtype=fp8_e4m3",
        "--language-model-only",
        "--quantization=fp8",
        "--generation-config=vllm",
        "--distributed-executor-backend=mp",
        "--api-server-count=1",
        "--max-model-len=" + ("262144" if prefill else "33792"),
        "--max-num-batched-tokens=8192",
        "--max-num-seqs=" + ("8" if prefill else "16"),
        "--block-size=128",
        "--tensor-parallel-size=" + str(tp),
        "--data-parallel-size=" + str(dp),
        "--data-parallel-size-local=" + str(dp),
        "--enable-expert-parallel",
        "--disable-custom-all-reduce",
        "--gpu-memory-utilization=0.9",
        "--mamba-cache-dtype=float32",
        "--mamba-ssm-cache-dtype=bfloat16",
        "--mamba-cache-mode=" + ("align" if apc else "none"),
        "--no-disable-hybrid-kv-cache-manager",
        "--enable-prefix-caching" if apc else "--no-enable-prefix-caching",
        "--enable-chunked-prefill",
        "--async-scheduling",
        "--enable-prompt-tokens-details",
        "--no-enable-log-requests",
        "--hf-overrides",
        json.dumps(dict(text_config=dict(decoder_sparse_step=1, mlp_only_layers=[]))),
        "--additional-config",
        json.dumps(dict(qwen4_exp=qwen)),
        "--compilation-config",
        json.dumps(compilation),
        "--kv-transfer-config",
        json.dumps(transfer),
    ]
    return dict(
        role=role,
        dp=dp,
        tp=tp,
        ep=tp * dp,
        model=model,
        argv=args,
        env=env,
        unset_env=[] if prefill else ["MOE_FUSED_EP_KERNEL_IMPL"],
        max_request_tokens=33792,
        client_concurrency=dp * 8,
        prefix_caching=apc,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=["prefill", "decode", "check-result"])
    parser.add_argument("--result", type=Path)
    parser.add_argument("--expected-completed", type=int)
    parser.add_argument("--print", action="store_true", dest="print_only")
    args = parser.parse_args()
    if args.role == "check-result":
        if args.result is None or args.expected_completed is None:
            parser.error("check-result requires --result and --expected-completed")
        rows = [
            json.loads(line)
            for line in args.result.read_text().splitlines()
            if line.strip()
        ]
        if (
            len(rows) != 1
            or rows[0].get("failed") != 0
            or rows[0].get("completed") != args.expected_completed
        ):
            raise ValueError(
                "Benchmark result is missing, incomplete or has failed requests"
            )
        return
    mode = os.environ.get("DISAGG_TEST_MODE", "performance")
    if mode not in ("performance", "apc"):
        parser.error("DISAGG_TEST_MODE must be performance or apc")
    profile = role_profile(
        args.role,
        int(os.environ.get("DISAGG_DP_SIZE", "8")),
        os.environ.get("DISAGG_MODEL", MODEL),
        os.environ["POD_IP"],
        os.environ.get("WORKLOAD_NAME", "qwen38-pd"),
        apc=mode == "apc",
    )
    if args.print_only:
        print(json.dumps(profile, indent=2))
        return
    effective = os.environ.copy()
    for name in profile["unset_env"]:
        effective.pop(name, None)
    effective.update(profile["env"])
    print(json.dumps(profile), flush=True)
    os.execvpe(profile["argv"][0], profile["argv"], effective)


if __name__ == "__main__":
    main()
