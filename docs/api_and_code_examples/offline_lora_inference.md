# Offline LoRA Inference

The following code example demonstrates how to run offline Multi-LoRA inference on TPU using vLLM (`--enable-lora`). Requests in the same batch can use different adapters (or the base model), and adapters load, swap, and evict without XLA recompilation.

> [!NOTE]
> **Current TPU Multi-LoRA scope**: Multi-LoRA runs on a single TPU device (`TP=1`, `DP=1`, `PP=1`, `PCP=1`, `DCP=1`) with dense generative models only, without speculative decoding, `fully_sharded_loras`, or `enable_tower_connector_lora`, and requires `lora_dtype` to match the model execution dtype (`bfloat16`). Unsupported settings are rejected at startup.

```python
--8<-- "examples/offline_lora_inference.py"
```

[View source on GitHub :material-open-in-new:](https://github.com/vllm-project/vllm-torchtpu/blob/main/examples/offline_lora_inference.py){ .md-button }
