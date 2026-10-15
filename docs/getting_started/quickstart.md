# Quickstart

Once you have [provisioned a Cloud TPU VM](tpu_setup.md) and [installed `vllm-torchtpu`](installation.md), you can run online OpenAI-compatible serving or offline batch inference.

---

## 1. Online Serving (`vllm serve`)

Start an OpenAI-compatible HTTP server using `vllm serve`:

### Single-Chip (`TP=1`)

```bash
vllm serve meta-llama/Llama-3.2-1B-Instruct \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --port 8000
```

### 8-Device TPU Host (`TP=8` + Expert Parallelism)

```bash
vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --kv-cache-dtype fp8 \
  --max-model-len 16384 \
  --max-num-batched-tokens 8192 \
  --async-scheduling \
  --port 8000
```

Once the server logs `Application startup complete`, query the `/v1/chat/completions` endpoint from another terminal:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "messages": [
      {"role": "user", "content": "Explain how Tensor Processing Units accelerate matrix multiplication."}
    ],
    "max_tokens": 128,
    "temperature": 0.0
  }'
```

> [!TIP]
> **Persisting Compilation Caches Across Restarts**: The first startup compiles XLA/Pallas graphs for the configured token shapes. Set `VLLM_XLA_CACHE_PATH` to persist compiled artifacts across runs—see [Compilation & Warmup Cache](../user_guide/compilation_cache.md).

---

## 2. Offline Batch Inference (`vllm.LLM`)

Run offline batch generation directly from Python using `vllm.LLM`:

```python
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The capital of France is",
    "The future of AI hardware is",
]

sampling_params = SamplingParams(temperature=0.0, max_tokens=64)
llm = LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    tensor_parallel_size=1,
    max_model_len=1024,
)

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(f"Prompt: {output.prompt!r}")
    print(f"Generated: {output.outputs[0].text!r}\n")
```

Or run the repository's CLI example script directly:

```bash
python examples/offline_inference.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --tensor-parallel-size 1 \
  --max-model-len 1024
```

<details markdown="1">
<summary>Full Source: <code>examples/offline_inference.py</code></summary>

```python
---8<--- "examples/offline_inference.py"
```

</details>

---

## 3. Multimodal Vision-Language Inference

`vllm-torchtpu` supports vision-language models such as `Qwen/Qwen2.5-VL-3B-Instruct` and `Qwen/Qwen3.5-397B-A17B`:

```bash
python examples/multi_modal_inference.py \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --tensor-parallel-size 1 \
  --num-prompts 1
```

<details markdown="1">
<summary>Full Source: <code>examples/multi_modal_inference.py</code></summary>

```python
---8<--- "examples/multi_modal_inference.py"
```

</details>

---

## 4. Multi-LoRA Inference

Run offline inference with LoRA adapters enabled:

```bash
python examples/offline_lora_inference.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --enable-lora \
  --max-lora-rank 8
```

<details markdown="1">
<summary>Full Source: <code>examples/offline_lora_inference.py</code></summary>

```python
---8<--- "examples/offline_lora_inference.py"
```

</details>
