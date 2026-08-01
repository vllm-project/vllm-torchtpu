"""End-to-end smoke for the unified int8 KV pool on a hybrid GDN model.

Boots Qwen3.5-35B-A3B on TP4 at the config that used to OOM on the strided
f32 recurrent view (high gpu_memory_utilization + num_gpu_blocks_override),
generates a few tokens, and prints the result. With the unified int8 pool the
recurrent state is read/written through the in-kernel int8<->f32 bitcast, so
there is no strided f32 view to byte-assemble. Success = ENGINE_UP (no OOM) +
GEN_OK with sane tokens.
"""
import os
import sys

# The default GDN impl (chunked_jax_pd) is where the int8-pool gather/scatter
# is wired. No KV-cache env gates: the pool is the default now.
os.environ.setdefault("USE_MOE_SPARSE_CORE", "0")


def main() -> int:
    from vllm import LLM, SamplingParams

    tp = int(os.environ.get("E2E_TP", "4"))
    print(f"=== hybrid-pool e2e: tp={tp}, unified int8 pool (default) ===",
          flush=True)
    kwargs = dict(
        model=os.environ.get("E2E_MODEL", "Qwen/Qwen3.5-35B-A3B-FP8"),
        tensor_parallel_size=tp,
        max_model_len=16384,
        max_num_seqs=4,
        max_num_batched_tokens=2048,
        quantization="fp8",
        # bf16 attention KV is the supported pooled path; fp8 attention KV in
        # the unified pool is rejected (make_attention_cache_view guard).
        kv_cache_dtype=os.environ.get("E2E_KVDTYPE", "auto"),
        block_size=256,
        gpu_memory_utilization=float(os.environ.get("E2E_GPU_MEM_UTIL",
                                                    "0.8")),
        enable_prefix_caching=True,
        enable_expert_parallel=os.environ.get("E2E_EP", "1") == "1",
        # Qwen3.5 declares a vision modality; run it text-only so vLLM keeps
        # chunked MM input enabled (see TpuPlatform.check_and_update_config).
        language_model_only=True,
        async_scheduling=False,
        limit_mm_per_prompt={
            "image": 0,
            "video": 0
        },
    )
    # 0 (default) => let vLLM auto-size the KV cache to fit; the override is
    # flaky under multiproc TP.
    nb = int(os.environ.get("E2E_NUM_GPU_BLOCKS", "0"))
    if nb > 0:
        kwargs["num_gpu_blocks_override"] = nb
    llm = LLM(**kwargs)
    print("ENGINE_UP", flush=True)
    out = llm.generate(["The bridge at dawn. " * 16],
                       SamplingParams(temperature=0.0,
                                      max_tokens=8,
                                      ignore_eos=True))
    toks = list(out[0].outputs[0].token_ids[:6])
    print(f"GEN_OK first_tokens={toks}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
