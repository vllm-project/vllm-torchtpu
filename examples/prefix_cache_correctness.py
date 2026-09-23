"""End-to-end prefix-cache correctness check for the unified KV pool.

Prefix caching must be transparent: a request's output must not depend on
whether its prefix was served from cache. On a hybrid GDN model that property
hinges on the align-mode mamba state-block seeding — on a cache-hit resume the
recurrent state must be seeded from the cached boundary block. A wrong seed
reads stale state and corrupts the answer.

This check is deliberately built to flag real corruption without false alarms:

  * ``max_num_seqs=1`` — every request decodes with the same single-request
    batch geometry on both passes, so floating-point reduction order is
    identical and greedy decoding cannot flip on batch composition. (A batched
    cold-vs-warm token diff is dominated by this effect, not by caching.)
  * Each prompt is a long shared context (spans several KV blocks, so the warm
    pass resumes mamba state across a block boundary) followed by a distinct
    question with an unambiguous, dominant answer — so the small rounding
    difference between a cache-hit resume and a full recompute cannot flip the
    greedy token.
  * Two assertions per prompt: the warm (fully cached) generation matches the
    cold one token-for-token (reproducibility), and the answer is correct (the
    cache returns the right content, not merely a stable wrong one). The second
    is what catches a real state-bleed even if it were somehow reproducible.

Config is env-driven so the same file runs both the plain (default) and the
batched-RPA (``E2E_ATTN_BACKEND=CUSTOM``) pooled paths.

Success = PREFIX_CACHE_OK. Any mismatch prints the offending tokens for triage.
"""

import os
import sys

os.environ.setdefault("USE_MOE_SPARSE_CORE", "0")

# A long shared prefix, identical for every prompt and long enough to span
# several KV blocks so the warm pass exercises a block-boundary state resume.
_PARA = (
    "Photosynthesis is the process by which green plants, algae, and some "
    "bacteria convert light energy into chemical energy stored in sugars. It "
    "takes place mainly in the chloroplasts, using the green pigment "
    "chlorophyll to capture sunlight. The overall reaction combines carbon "
    "dioxide from the air with water drawn up from the roots, and releases the "
    "products back into the environment. "
)
SHARED_PREFIX = _PARA * 24  # ~1.5k tokens => several blocks at any block size

# (question, substring the correct answer must contain). Answers are single
# distinctive words so greedy is not on a knife's edge.
QA = [
    ("What is the capital of France? Answer with one word.", "Paris"),
    ("What is the capital of Japan? Answer with one word.", "Tokyo"),
    ("What is the opposite of hot? Answer with one word.", "cold"),
    ("What color is a clear daytime sky? Answer with one word.", "blue"),
    ("What gas do plants release during photosynthesis? One word.", "oxygen"),
    ("What is the largest planet in our solar system? One word.", "Jupiter"),
]


def main() -> int:
    from vllm import LLM, SamplingParams

    kwargs = dict(
        model=os.environ.get("E2E_MODEL", "Qwen/Qwen3.5-35B-A3B-FP8"),
        tensor_parallel_size=int(os.environ.get("E2E_TP", "4")),
        max_model_len=16384,
        # Batch-invariant: identical single-request geometry on both passes.
        max_num_seqs=1,
        max_num_batched_tokens=2048,
        quantization="fp8",
        kv_cache_dtype=os.environ.get("E2E_KVDTYPE", "auto"),
        gpu_memory_utilization=float(os.environ.get("E2E_GPU_MEM_UTIL", "0.8")),
        enable_prefix_caching=True,
        enable_expert_parallel=os.environ.get("E2E_EP", "1") == "1",
        language_model_only=True,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    backend = os.environ.get("E2E_ATTN_BACKEND")
    if backend:
        kwargs["attention_backend"] = backend
    llm = LLM(**kwargs)
    print("ENGINE_UP", flush=True)

    prompts = [SHARED_PREFIX + "\n\nQ: " + q + "\nA:" for q, _ in QA]
    sp = SamplingParams(temperature=0.0, max_tokens=8)

    def run(tag):
        outs = llm.generate(prompts, sp)
        toks = [tuple(o.outputs[0].token_ids) for o in outs]
        text = [o.outputs[0].text for o in outs]
        print(f"{tag}_DONE", flush=True)
        return toks, text

    cold_t, cold_x = run("COLD")  # shared prefix caches as this pass proceeds
    warm_t, warm_x = run("WARM")  # every prompt now fully cached

    unstable = [i for i in range(len(QA)) if cold_t[i] != warm_t[i]]
    wrong = [
        i
        for i, (_, a) in enumerate(QA)
        if a.lower() not in cold_x[i].lower() or a.lower() not in warm_x[i].lower()
    ]

    for i in unstable:
        print(
            f"  UNSTABLE req{i}: cold={list(cold_t[i])} warm={list(warm_t[i])}",
            flush=True,
        )
    for i in wrong:
        print(
            f"  WRONG req{i} want={QA[i][1]!r} cold={cold_x[i]!r} warm={warm_x[i]!r}",
            flush=True,
        )

    if not unstable and not wrong:
        print("PREFIX_CACHE_OK", flush=True)
        return 0
    print(f"PREFIX_CACHE_FAIL unstable={unstable} wrong={wrong}", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
