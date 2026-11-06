# SPDX-License-Identifier: Apache-2.0
"""Replay the real Engram.forward integration on one TPU with resident rows.

This is a single-layer integration benchmark. It excludes the embedding-table
lookup, host gathering, H2D transfer and full-model serving. The baseline uses
the original packed-row decoding and the real vLLM TPU FP8 linear method.

Run in the branch's pinned TPU environment, from the repository root:
  python scripts/dsv41_bench/bench_engram_packed_forward.py --output result.json
Add --profile trace-directory to capture each path after the formal timing.
"""

import argparse
import importlib.metadata
import json
import random
import statistics
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--profile-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--atol", type=float, default=0.02)
    parser.add_argument("--rtol", type=float, default=0.01)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 1:
        raise ValueError("iterations and warmup must be positive")

    import torch
    import torch_tpu  # noqa: F401
    from torch._dynamo.backends.registry import lookup_backend
    from torch_tpu._internal import sync
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    from vllm.model_executor.layers.linear import ColumnParallelLinear

    from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8Config
    from vllm_torchtpu.layers.core.quantization import quantize_tensor
    from vllm_torchtpu.models.vllm.deepseek_v41.common.engram import Engram

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    device = torch.device("tpu:0")
    torch.empty(1, device=device).cpu()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "benchmark": "single Engram layer, resident synthetic packed embeddings",
        "excluded": ["hash lookup", "host gather", "H2D", "whole-model evaluation"],
        "timing": "synchronized wall time including host dispatch",
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "torch-tpu", "jax", "libtpu", "vllm")
        },
        "shape": {"tokens": 288, "hash_heads": 24, "head_dim": 256},
        "eps": 1e-20,
        "clamp_value": 1e-6,
        "passed": False,
    }

    def save() -> None:
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    save()
    rendezvous = tempfile.TemporaryDirectory(prefix="engram-replay-tp1-")
    initialized = False
    try:
        # BasevLLMParameter reads the TP group even for a replicated layer.
        # Use the real single-rank groups, with no synthetic distributed state.
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"file://{rendezvous.name}/rendezvous",
            backend="tpu_dist",
        )
        initialized = True
        with set_current_vllm_config(VllmConfig()):
            ensure_model_parallel_initialized(1, 1)
        # Construct the actual projection and use its load-time FP8 processing.
        # Only the enormous table is replaced by already gathered resident rows.
        holder = Engram.__new__(Engram)
        torch.nn.Module.__init__(holder)
        holder.dim = 5120
        holder.hc_mult = 4
        holder.eps = result["eps"]
        holder.clamp_value = result["clamp_value"]
        holder.use_sequence_parallel = False
        holder.embed_tokens = SimpleNamespace(
            host_offload=True,
            tp_size=1,
            dp_size=1,
            n_hash_cols=24,
            part_n_hash_cols=24,
            block_size=32,
        )
        holder.wkv = ColumnParallelLinear(
            6144,
            25600,
            bias=False,
            gather_output=True,
            return_bias=False,
            params_dtype=torch.bfloat16,
            quant_config=VllmFp8Config(is_checkpoint_fp8_serialized=False),
            prefix="engram.wkv",
            disable_tp=True,
        )
        with torch.no_grad():
            holder.wkv.weight.copy_(
                (torch.randn(25600, 6144) * 0.025).to(torch.bfloat16)
            )
            holder.wkv.quant_method.process_weights_after_loading(holder.wkv)
        holder.q_weight = torch.nn.Parameter(
            torch.randn(4, 5120, dtype=torch.bfloat16), requires_grad=False
        )
        holder.k_weight = torch.nn.Parameter(
            torch.randn(4, 5120, dtype=torch.bfloat16), requires_grad=False
        )
        holder.to(device).eval()

        embedding = torch.randn(288, 24, 256)
        raw, scales = quantize_tensor(
            embedding, torch.float8_e4m3fn, block_size=32, use_ue8m0=True
        )
        packed = torch.cat((raw.view(torch.uint8).flatten(), scales.flatten()))
        holder._active_raw = packed.view(torch.int32).to(device).view(288, -1)
        holder._staged_raw = holder._active_raw
        holder._hbm_rows = None
        holder._host_raw = torch.empty(288, 24, 256, dtype=torch.uint8)
        holder._host_scale = torch.empty(288, 24, 8, dtype=torch.uint8)
        hidden = torch.randn(288, 4, 5120, dtype=torch.bfloat16).to(device)
        hash_ids = torch.zeros(288, 24, dtype=torch.int32, device=device)
        mask_cpu = torch.arange(288) % 7 != 0
        token_mask = mask_cpu.to(device)
        if not holder._uses_fused_offload(hidden, token_mask):
            raise RuntimeError("The fixture does not select the fused Engram path")

        # Record the graphs that really reach the TPU backend; testing the
        # predicate alone would miss a bridge or dispatch wiring failure.
        graphs: dict[str, list[list[str]]] = {"before": [], "after": []}
        tpu_backend = lookup_backend("tpu")

        def backend_for(name):
            def backend(graph, example_inputs):
                graphs[name].append([str(node.target) for node in graph.graph.nodes])
                return tpu_backend(graph, example_inputs)

            return backend

        paths = {
            "before": torch.compile(
                holder._forward_unfused, backend=backend_for("before"), fullgraph=True
            ),
            "after": torch.compile(
                holder, backend=backend_for("after"), fullgraph=True
            ),
        }

        def run(name):
            output = paths[name](hidden, hash_ids, token_mask)
            sync.synchronize(tensors=[output], wait=True)
            return output

        with torch.inference_mode():
            outputs = {name: run(name).cpu() for name in paths}
            result["graphs"] = graphs
            selected = any(
                "engram_packed_fused" in target
                for graph in graphs["after"]
                for target in graph
            )
            baseline_selected = any(
                "engram_packed_fused" in target
                for graph in graphs["before"]
                for target in graph
            )
            if not selected or baseline_selected:
                raise RuntimeError("Compiled A/B graphs do not select distinct paths")
            reference, actual = outputs["before"], outputs["after"]
            difference = (actual.float() - reference.float()).abs()
            hidden_cpu = hidden.cpu()
            masked_preserved = torch.equal(actual[~mask_cpu], hidden_cpu[~mask_cpu])
            finite = bool(torch.isfinite(actual).all())
            within_tolerance = torch.allclose(
                actual, reference, atol=args.atol, rtol=args.rtol
            )
            result["correctness"] = {
                "max_abs": difference.max().item(),
                "mean_abs": difference.mean().item(),
                "atol": args.atol,
                "rtol": args.rtol,
                "within_tolerance": within_tolerance,
                "finite": finite,
                "masked_rows_preserved": masked_preserved,
                "numeric_difference": "fused K statistics retain FP32 projection",
            }
            for name in paths:
                for _ in range(args.warmup):
                    run(name)
            samples = {name: [] for name in paths}
            rng = random.Random(args.seed)
            for _ in range(args.iterations):
                order = list(paths)
                rng.shuffle(order)
                for name in order:
                    start = time.perf_counter_ns()
                    run(name)
                    samples[name].append((time.perf_counter_ns() - start) / 1e6)
            result["timings"] = {
                name: {
                    "median_ms": statistics.median(values),
                    "samples_ms": values,
                }
                for name, values in samples.items()
            }
            result["speedup"] = statistics.median(samples["before"]) / (
                statistics.median(samples["after"])
            )
            save()
            if args.profile is not None:
                from torch_tpu._internal.profiler import TpuProfilerConfig

                for name in paths:
                    destination = str((args.profile / name).resolve())
                    Path(destination).mkdir(parents=True, exist_ok=True)
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.PrivateUse1,
                        ],
                        on_trace_ready=torch.profiler.tensorboard_trace_handler(
                            destination, use_gzip=True
                        ),
                        experimental_config=TpuProfilerConfig(
                            run_dir=destination,
                            host_tracer_level=2,
                            device_tracer_level=1,
                            python_tracer_level=1,
                        ),
                    ):
                        for _ in range(args.profile_steps):
                            run(name)
                result["profile"] = str(args.profile.resolve())
            if not (finite and masked_preserved and within_tolerance):
                raise RuntimeError("Single-layer numerical validation failed")
            result["passed"] = True
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save()
        if initialized:
            destroy_model_parallel()
            destroy_distributed_environment()
        rendezvous.cleanup()
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
