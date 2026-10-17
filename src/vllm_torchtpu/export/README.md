# StableHLO Export & Load Guide (`vllm-torchtpu`)

Export StableHLO (`.mlir`) graphs and `metadata.json` from `vllm-torchtpu`, then load them to run one-shot inference without re-tracing.

---

## 1. Environment Variables

| Variable | Description |
| :--- | :--- |
| `VLLM_TORCHTPU_EXPORT_HLO_DIR` | Directory where `forward_bucket*.mlir`, `compute_logits_req*.mlir`, `metadata.json`, and `weight.json` are written. |
| `VLLM_TORCHTPU_LOAD_HLO_DIR` | Directory of pre-exported `.mlir` and `metadata.json` to load directly, bypassing FX graph tracing. |

> **Note**: The two variables cannot be set simultaneously.

---

## 2. One-Shot Export

```bash
export VLLM_TORCHTPU_EXPORT_HLO_DIR="/workspace/qwen_export"
unset VLLM_TORCHTPU_LOAD_HLO_DIR

python examples/offline_inference.py \
    --model Qwen/Qwen3-0.6B \
    --tensor-parallel-size 1 \
    --max-model-len 64 \
    --max-num-batched-tokens 64 \
    --prompt "The capital of France is" \
    --max-tokens 5
```

---

## 3. One-Shot Load & Run

```bash
unset VLLM_TORCHTPU_EXPORT_HLO_DIR
export VLLM_TORCHTPU_LOAD_HLO_DIR="/workspace/qwen_export"

python examples/offline_inference.py \
    --model Qwen/Qwen3-0.6B \
    --tensor-parallel-size 1 \
    --max-model-len 64 \
    --max-num-batched-tokens 64 \
    --prompt "The capital of France is" \
    --max-tokens 10
```
