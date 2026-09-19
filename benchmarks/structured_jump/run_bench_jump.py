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
"""Benchmark runner for Grammar-as-a-Draft (GaaD) Jump Decoding on TPU.

Evaluates 3 arms:
  B0: Baseline no-speculative decoding (alpha = 1.0)
  B1: Speculative decoding (ngram, K=7), Jump Decoding OFF
  B2: Speculative decoding (ngram, K=7), Jump Decoding ON (GaaD)

Supports workloads:
  - character: Harry Potter character info (decode-heavy)
  - telemetry: CloudServiceTelemetry schema (high-density enums)
  - city: Wikipedia city info extraction (prefill-heavy)
"""

from __future__ import annotations

import argparse
import json
import os

if os.environ.get("TPU_WORKER_HOSTNAMES", "").startswith("WARNING"):
    del os.environ["TPU_WORKER_HOSTNAMES"]
import pathlib
import time
from typing import Any

from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.sampling_params import StructuredOutputsParams

from benchmarks.structured_jump.schemas import (CHARACTER_REGEX, CITY_REGEX,
                                                TELEMETRY_SCHEMA)

CUR_DIR = pathlib.Path(__file__).resolve().parent


def load_workload_requests(
    workload: str,
    num_requests: int,
    data_path: str | None = None,
) -> tuple[list[str], SamplingParams]:
    """Prepares prompts and sampling parameters for the specified workload."""
    if workload == "character":
        dataset_file = data_path or str(CUR_DIR / "dataset.txt")
        names: list[str] = []
        with open(dataset_file, "r", encoding="utf-8") as f:
            for line in f:
                name = line.strip()
                if name:
                    names.append(name)
        if len(names) < num_requests:
            names = (names * ((num_requests // len(names)) + 1))[:num_requests]
        else:
            names = names[:num_requests]
        prompts = [
            f"{name} is a character in Harry Potter. Please fill in the following information about this character.\n"
            for name in names
        ]
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=256,
            structured_outputs=StructuredOutputsParams(
                regex=CHARACTER_REGEX,
                disable_any_whitespace=True,
            ),
        )
        return prompts, sampling_params

    elif workload == "telemetry":
        prompts = [
            "Generate a telemetry report for the inference gateway service in JSON format: "
        ] * num_requests
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=128,
            structured_outputs=StructuredOutputsParams(
                json=TELEMETRY_SCHEMA,
                disable_any_whitespace=True,
            ),
        )
        return prompts, sampling_params

    elif workload == "city":
        dataset_file = data_path or str(CUR_DIR / "questions.jsonl")
        if not os.path.exists(dataset_file):
            raise FileNotFoundError(
                f"City dataset not found at {dataset_file}. Run build_city_dataset.py first."
            )
        documents: list[str] = []
        with open(dataset_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                documents.append(item["document"])
        documents = documents[:num_requests]
        prompts = [
            "Please extract the information of a city from the following wikipedia page.\n"
            f"Page begin.\n{doc}Page end.\n"
            "Here is the name, country, and symbol of the city in JSON format.\n"
            for doc in documents
        ]
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=256,
            structured_outputs=StructuredOutputsParams(
                regex=CITY_REGEX,
                disable_any_whitespace=True,
            ),
        )
        return prompts, sampling_params

    else:
        raise ValueError(f"Unknown workload: {workload}")


def init_llm(
    model: str,
    arm: str,
    max_num_seqs: int,
    num_speculative_tokens: int = 7,
    max_model_len: int = 512,
    torch_profiler_dir: str | None = None,
    spec_method: str | None = None,
    enable_gaad: bool | None = None,
) -> LLM:
    """Instantiates the LLM engine according to the selected benchmark arm or spec method."""
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"  # keep stats accessible

    # Resolve spec_method and enable_gaad from arm if not explicitly passed
    arm_lower = arm.lower()
    if spec_method is None:
        if arm in ("B0", "baseline", "none"):
            spec_method = "none"
        elif arm in ("B1", "B2", "B3"):
            spec_method = "ngram"
        elif "ngram" in arm_lower:
            spec_method = "ngram"
        elif "eagle3" in arm_lower:
            spec_method = "eagle3"
        elif "dflash" in arm_lower:
            spec_method = "dflash"
        elif "mtp" in arm_lower:
            spec_method = "mtp"
        else:
            raise ValueError(f"Unknown arm or cannot infer spec_method: {arm}")

    if enable_gaad is None:
        if arm in ("B0", "B1"):
            enable_gaad = False
        elif arm in ("B2", "B3"):
            enable_gaad = True
        elif "gaad" in arm_lower:
            enable_gaad = True
        else:
            enable_gaad = False

    if enable_gaad:
        os.environ["VLLM_TPU_JUMP_DECODING"] = "1"
        from vllm_torchtpu import patch_registry
        patch_registry.apply("engine_core")
    else:
        os.environ["VLLM_TPU_JUMP_DECODING"] = "0"

    gpu_mem = 0.6
    extra_engine_kwargs: dict[str, Any] = {}

    if spec_method == "none":
        spec_config = None
    elif spec_method == "ngram":
        min_lookup = num_speculative_tokens if arm == "B3" else 3
        spec_config = {
            "method": "ngram",
            "prompt_lookup_max": num_speculative_tokens,
            "prompt_lookup_min": min_lookup,
            "num_speculative_tokens": num_speculative_tokens,
        }
    elif spec_method == "eagle3":
        spec_config = {
            "method": "eagle3",
            "model": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
            "num_speculative_tokens": num_speculative_tokens,
            "draft_tensor_parallel_size": 1,
        }
    elif spec_method == "dflash":
        spec_config = {
            "method": "dflash",
            "model": "z-lab/Qwen3-4B-DFlash-b16",
            "num_speculative_tokens": num_speculative_tokens,
            "draft_tensor_parallel_size": 1,
        }
    elif spec_method == "mtp":
        os.environ["VLLM_KV_CACHE_LAYOUT"] = "HND"
        spec_config = {
            "method": "mtp",
            "num_speculative_tokens": num_speculative_tokens,
        }
        extra_engine_kwargs.update({
            "language_model_only": True,
            "limit_mm_per_prompt": {
                "image": 0,
                "video": 0
            },
            "enable_prefix_caching": False,
        })
        gpu_mem = 0.3
    else:
        raise ValueError(f"Unsupported speculative method: {spec_method}")

    if spec_method in ("eagle3", "dflash", "mtp"):
        extra_engine_kwargs["async_scheduling"] = True

    engine_kwargs = {
        "model": model,
        "speculative_config": spec_config,
        "max_model_len": max_model_len,
        "max_num_seqs": max_num_seqs,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": gpu_mem,
        "disable_log_stats": False,
        **extra_engine_kwargs,
    }

    if torch_profiler_dir:
        import shutil
        if os.path.exists(torch_profiler_dir):
            shutil.rmtree(torch_profiler_dir, ignore_errors=True)
        os.makedirs(torch_profiler_dir, exist_ok=True)
        from vllm.config import ProfilerConfig
        engine_kwargs["profiler_config"] = ProfilerConfig(
            profiler="torch",
            torch_profiler_dir=torch_profiler_dir,
            ignore_frontend=True,
        )
        print(
            f"✅ Enabled ProfilerConfig with torch_profiler_dir={torch_profiler_dir}"
        )

    llm = LLM(**engine_kwargs)
    return llm


def warmup_engine(
    llm: LLM,
    prompt: str,
    sampling_params: SamplingParams,
    batch_size: int,
) -> None:
    """Explicit Warmup Run:

    Runs a warmup batch to compile runtime XLA graphs and warm FSM caches,
    completely excluded from benchmark timing.
    """
    print(f"🔥 Starting explicit warmup run (batch_size={batch_size})...")
    t0 = time.perf_counter()
    for i in range(batch_size):
        llm._add_request(prompt, sampling_params)

    while llm.llm_engine.has_unfinished_requests():
        llm.llm_engine.step()

    print(f"🔥 Warmup completed in {time.perf_counter() - t0:.3f} s.")


def run_workload(
    llm: LLM,
    prompts: list[str],
    sampling_params: SamplingParams,
    profile: bool = False,
) -> tuple[list[dict[str, Any]], int, float]:
    """Runs a batch of prompts through manual engine.step() loop to count decode steps exactly.

    Timing strictly measures steady-state generation after warmup has completed.
    """
    for prompt in prompts:
        llm._add_request(prompt, sampling_params)

    if profile:
        print("🔍 Starting hardware profiler session...")
        llm.start_profile()
        time.sleep(1.0)

    total_steps = 0
    final_outputs = []
    tic = time.perf_counter()

    try:
        while llm.llm_engine.has_unfinished_requests():
            step_outputs = llm.llm_engine.step()
            total_steps += 1
            for out in step_outputs:
                if out.finished:
                    final_outputs.append({
                        "request_id": out.request_id,
                        "text": out.outputs[0].text,
                        "token_ids": out.outputs[0].token_ids,
                    })
    finally:
        # Stop the clock before tearing the profiler down, not after.
        # stop_profile() collects the XSpace and merges the profile
        # directories; in run r-op-probe-0b0868 that took ~34 s against 22 s
        # of actual stepping. Measuring it made every profiled arm look ~34 s
        # slower than it was and compressed the difference between arms,
        # because the teardown cost is roughly constant.
        latency = time.perf_counter() - tic
        if profile:
            llm.stop_profile()
            print("✅ Hardware profiler session stopped and trace saved.")

    return final_outputs, total_steps, latency


def parse_args():
    parser = argparse.ArgumentParser(description="Run GaaD Benchmark on TPU")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument(
        "--arm",
        type=str,
        default="B0",
        help="Arm identifier (e.g. B0, B1, B2, or <method>_<base|gaad>)")
    parser.add_argument("--spec-method",
                        type=str,
                        default=None,
                        choices=["ngram", "eagle3", "dflash", "mtp", "none"],
                        help="Explicit speculative decoding method")
    parser.add_argument(
        "--enable-gaad",
        type=int,
        default=None,
        choices=[0, 1],
        help="1 to enable GaaD (Jump Decoding), 0 for baseline spec decoding")
    parser.add_argument("--gamma",
                        type=int,
                        default=None,
                        help="Verification window gamma (spec_k = gamma - 1)")
    parser.add_argument("--workload",
                        type=str,
                        default="character",
                        choices=["character", "telemetry", "city"])
    parser.add_argument("--num-requests", type=int, default=50)
    parser.add_argument("--parallel",
                        type=int,
                        default=1,
                        help="max_num_seqs batch size")
    parser.add_argument("--batch-size",
                        type=int,
                        default=None,
                        help="Alias for --parallel used by op-probe")
    parser.add_argument("--max-model-len",
                        type=int,
                        default=512,
                        help="max_model_len for LLM")
    parser.add_argument("--spec-k",
                        type=int,
                        default=7,
                        help="num_speculative_tokens")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--output-jsonl",
                        type=str,
                        default="benchmark_results.jsonl")
    parser.add_argument("--torch-profiler-dir",
                        type=str,
                        default=None,
                        help="Directory to save PyTorch profiler traces")
    parser.add_argument("--profile",
                        action="store_true",
                        help="Enable hardware profiling during decode")
    args, unknown = parser.parse_known_args()
    if unknown:
        print(
            f"Ignoring extra unrecognized arguments passed by template: {unknown}"
        )
    if args.batch_size is not None:
        args.parallel = args.batch_size
    if args.gamma is not None:
        args.spec_k = args.gamma - 1
    else:
        args.gamma = args.spec_k + 1

    # Infer spec_method and enable_gaad if not provided
    arm_lower = args.arm.lower()
    if args.spec_method is None:
        if args.arm in ("B0", "baseline", "none"):
            args.spec_method = "none"
        elif args.arm in ("B1", "B2", "B3") or "ngram" in arm_lower:
            args.spec_method = "ngram"
        elif "eagle3" in arm_lower:
            args.spec_method = "eagle3"
        elif "dflash" in arm_lower:
            args.spec_method = "dflash"
        elif "mtp" in arm_lower:
            args.spec_method = "mtp"
        else:
            args.spec_method = "ngram"

    if args.enable_gaad is None:
        if args.arm in ("B0", "B1") or "base" in arm_lower:
            args.enable_gaad = 0
        elif args.arm in ("B2", "B3") or "gaad" in arm_lower:
            args.enable_gaad = 1
        else:
            args.enable_gaad = 0

    # Auto-resolve default target model if user did not specify one.
    # NOTE: treat "" as unset too -- the llm_jobs op-probe template always
    # injects `--model ""` when no model is given via `lj job run -m`.
    if not args.model:
        if args.spec_method == "dflash":
            args.model = "Qwen/Qwen3-4B"
        elif args.spec_method == "mtp":
            args.model = "Qwen/Qwen3.5-4B"
        else:
            args.model = "NousResearch/Meta-Llama-3.1-8B-Instruct"

    if args.torch_profiler_dir and not args.profile:
        args.profile = True
    return args


def main():
    args = parse_args()
    print(
        f"=== Starting Benchmark Arm {args.arm} | Method: {args.spec_method} | GaaD: {args.enable_gaad} | Gamma: {args.gamma} (K={args.spec_k}) | Workload: {args.workload} | Model: {args.model} | Parallel: {args.parallel} ==="
    )

    num_reqs = max(args.num_requests, args.parallel)
    prompts, sampling_params = load_workload_requests(
        args.workload,
        num_reqs,
        args.data_path,
    )
    print(f"Loaded {len(prompts)} requests.")

    llm = init_llm(
        model=args.model,
        arm=args.arm,
        max_num_seqs=args.parallel,
        num_speculative_tokens=args.spec_k,
        max_model_len=args.max_model_len,
        torch_profiler_dir=args.torch_profiler_dir,
        spec_method=args.spec_method,
        enable_gaad=bool(args.enable_gaad),
    )

    # Explicit Warmup Run (excluded from benchmark timing)
    warmup_engine(llm, prompts[0], sampling_params, batch_size=args.parallel)

    try:
        from vllm_torchtpu.spec_decode.grammar_jump import get_stats
        jump_stats = get_stats()
        # Reset counters so warmup hits are not counted
        jump_stats.calls = 0
        jump_stats.hits = 0
        jump_stats.tokens_proposed = 0
        jump_stats.truncations = 0
        jump_stats.fsm_rejections = 0
    except Exception:
        jump_stats = None

    outputs, total_steps, latency = run_workload(llm,
                                                 prompts,
                                                 sampling_params,
                                                 profile=args.profile)

    total_output_tokens = sum(len(out["token_ids"]) for out in outputs)
    alpha = total_output_tokens / total_steps if total_steps else 0.0

    # Validation check: count valid JSON parses
    valid_json_count = 0
    for out in outputs:
        try:
            json.loads(out["text"], strict=False)
            valid_json_count += 1
        except Exception:
            pass

    stats_summary = jump_stats.summary() if jump_stats else "N/A"
    print(f"\n--- Results for {args.arm} ({args.workload}) ---")
    print(f"Latency: {latency:.3f} s")
    print(f"Total Steps: {total_steps}")
    print(f"Total Output Tokens: {total_output_tokens}")
    print(f"Alpha (Tokens/Step): {alpha:.3f}")
    print(f"Valid JSON: {valid_json_count}/{len(outputs)}")
    print(f"JumpStats: {stats_summary}")

    ms_per_tok = round((latency / total_output_tokens) *
                       1000, 3) if total_output_tokens else 0.0
    tps = round(total_output_tokens / latency, 2) if latency > 0 else 0.0

    result_entry = {
        "task": "structured_jump",
        "arm": args.arm,
        "spec_method": args.spec_method,
        "enable_gaad": bool(args.enable_gaad),
        "gamma": args.gamma,
        "spec_k": args.spec_k,
        "workload": args.workload,
        "model": args.model,
        "parallel": args.parallel,
        "num_requests": len(outputs),
        "latency": round(latency, 3),
        "decode_steps": total_steps,
        "output_tokens": total_output_tokens,
        "ms_per_token": ms_per_tok,
        "tokens_per_sec": tps,
        "alpha": round(alpha, 3),
        "valid_json_ratio":
        valid_json_count / len(outputs) if outputs else 0.0,
        "jump_stats": {
            "calls": jump_stats.calls if jump_stats else 0,
            "hits": jump_stats.hits if jump_stats else 0,
            "tokens_proposed": jump_stats.tokens_proposed if jump_stats else 0,
            "truncations": jump_stats.truncations if jump_stats else 0,
            "fsm_rejections": jump_stats.fsm_rejections if jump_stats else 0,
        } if jump_stats else {},
    }

    with open(args.output_jsonl, "a", encoding="utf-8") as f:
        f.write(json.dumps(result_entry) + "\n")

    # Save output texts for byte-comparison
    dump_path = f"output_{args.arm}_{args.workload}_{args.parallel}.json"
    with open(dump_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2)
    print(
        f"Outputs written to {dump_path} and summary appended to {args.output_jsonl}"
    )

    del llm
    cleanup_dist_env_and_memory()


if __name__ == "__main__":
    main()
