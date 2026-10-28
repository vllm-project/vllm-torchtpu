# Speculative Decoding

Speculative decoding accelerates token generation (reducing median TPOT and end-to-end latency) by proposing `K` draft tokens per step and verifying them in parallel in a single target-model forward pass.

`vllm-torchtpu` supports six speculative decoding methods on Cloud TPU (`tests/entrypoints/llm/test_speculative_decoding.py`):

| Method | `--speculative-config` `"method"` | Requires Draft Checkpoint? | Best For |
| :--- | :--- | :--- | :--- |
| **EAGLE-3** | `"eagle3"` | Yes (`yuhuili/EAGLE3-*`) | High acceptance rates across general chat, coding, and reasoning tasks. |
| **Multi-Token Prediction (MTP)** | `"mtp"` (or `"qwen3_next_mtp"`) | Built into checkpoint (`Qwen3.5`, `DeepSeek-V3/R1`, `GLM-5`) | Models trained with native MTP heads; zero external draft model management. |
| **DFlash** | `"dflash"` | Yes (`z-lab/*-DFlash-*`) | Block-diffusion draft heads proposing up to 16 tokens in a single forward pass (`Qwen3`, `Qwen3.5`). |
| **DSpark** | `"dspark"` | Yes, or built into the checkpoint (`DeepSeek-V4`) | A DFlash-style parallel block plus a sequential Markov head; optional adaptive verification. |
| **N-Gram** | `"ngram"` | No | Prompts with repetitive structure, code edits, citations, or RAG extraction. |
| **Suffix Decoding** | `"suffix"` | No | Speculating from a suffix tree built over prompt and prior responses. |

---

## 1. EAGLE-3 Speculative Decoding

Pass `--speculative-config` with `"method": "eagle3"`, the EAGLE-3 draft model checkpoint, and `num_speculative_tokens`:

```bash
vllm serve Qwen/Qwen3-32B \
  --tensor-parallel-size 8 \
  --max-model-len 8192 \
  --speculative-config '{
    "method": "eagle3",
    "model": "RedHatAI/Qwen3-32B-speculator.eagle3",
    "num_speculative_tokens": 3
  }'
```

---

## 2. Multi-Token Prediction (MTP)

Checkpoints with built-in Multi-Token Prediction heads (such as `Qwen/Qwen3.5-397B-A17B`, `Qwen/Qwen3.5-35B-A3B-FP8`, and `deepseek-ai/DeepSeek-R1`) require only `"method": "mtp"` and `"num_speculative_tokens"`:

```bash
vllm serve Qwen/Qwen3.5-35B-A3B-FP8 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --kv-cache-dtype fp8 \
  --speculative-config '{
    "method": "mtp",
    "num_speculative_tokens": 1
  }'
```

---

## 3. DFlash Speculative Decoding

DFlash uses a lightweight block-diffusion draft model to propose larger token blocks (e.g., `num_speculative_tokens=15`) verified in a single target pass:

```bash
vllm serve Qwen/Qwen3-8B \
  --tensor-parallel-size 2 \
  --max-model-len 4096 \
  --speculative-config '{
    "method": "dflash",
    "model": "z-lab/Qwen3-8B-DFlash-b16",
    "num_speculative_tokens": 15
  }'
```

---

## 4. DSpark Speculative Decoding

DSpark drafts a block in one parallel forward and then corrects each slot with a lightweight Markov head conditioned on the previously drafted token. `DeepSeek-V4` checkpoints that ship a DSpark draft carry it under `mtp.*`, so no separate model is needed:

```bash
vllm serve deepseek-ai/DeepSeek-V4-Pro-DSpark \
  --quantization deepseek_v4_fp8 \
  --data-parallel-size 16 --enable-expert-parallel \
  --async-scheduling \
  --speculative-config '{
    "method": "dspark",
    "num_speculative_tokens": 5
  }'
```

### Adaptive verification

With `"enable_adaptive_verification": true`, each decode step verifies only the drafts worth their cost. The checkpoint's confidence head scores every draft; a draft's survival probability is the running product of the confidences up to it, and the step verifies the globally most likely drafts, as many as maximize expected tokens per millisecond under a step-cost table the server profiles at startup. Outputs are unchanged: unverified drafts are treated as rejected.

The startup profile times decode-shaped verify steps at several request counts, at both ends of every token bucket they reach. A step's cost depends on its bucket and, when the MoE skips padding rows (`TPU_MOE_SKIP_PADDED_TOKENS=1`), on its real token count as well. Expect little difference at low per-engine concurrency, where a full verify step already fits the smallest bucket.

- Requires a checkpoint with a DSpark confidence head (`DeepSeek-V4` DSpark checkpoints have one). Without it the server logs a warning and verifies every draft.
- Applies to decode-only steps; steps that carry prefill chunks or structured-output requests verify every draft.
- The startup profile runs one context length, `VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN` (default 8192, capped by `--max-model-len`). Without tensor parallelism, the server corrects the profile online by timing back-to-back decode steps; with it, the profile stands as measured.
- Not supported with LoRA, pipeline parallelism, prefill context parallelism, or `--enforce-eager`.
- vLLM's acceptance-rate metric counts unverified drafts as rejected; the mean acceptance length stays comparable.

---

## 5. Draft-Model-Free Speculative Decoding (`ngram` & `suffix`)

When no separate draft model is available, use `"ngram"` or `"suffix"` matching directly on prompt tokens:

### N-Gram Speculative Decoding

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --tensor-parallel-size 2 \
  --speculative-config '{
    "method": "ngram",
    "num_speculative_tokens": 5,
    "prompt_lookup_max": 4
  }'
```

### Suffix Speculative Decoding

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --tensor-parallel-size 2 \
  --speculative-config '{
    "method": "suffix",
    "num_speculative_tokens": 5
  }'
```
