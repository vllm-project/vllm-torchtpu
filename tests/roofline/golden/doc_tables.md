# Roofline Basis Reference Tables

Generated from `tests/roofline/fixtures/manifest.json` and `tests/roofline/golden/scenarios.json`.

## Supported Basis Models

| Fixture | Repo | Layers | Hidden | Heads (Q/KV) | Experts (Total/Top-K) | Vocab |
| :--- | :--- | ---: | ---: | :--- | :--- | ---: |
| `deepseek_v4_1_flash.json` | `deepseek-ai/DeepSeek-V4.1-Flash` | 40 | 5120 | 64/1 | 384/6 | 129280 |
| `glm_5_2_fp8.json` | `zai-org/GLM-5.2-FP8` | 78 | 6144 | 64/64 | 256/8 | 154880 |
| `glm_5_2_nvfp4.json` | `nvidia/GLM-5.2-NVFP4` | 78 | 6144 | 64/64 | 256/8 | 154880 |
| `minimax_m3_1_flash.json` | `MiniMaxAI/MiniMax-M3.1-Flash` | 60 | 6144 | 64/4 | 128/4 | 200064 |
| `qwen3_5_397b_fp8.json` | `Qwen/Qwen3.5-397B-A17B-FP8` | 60 | 4096 | 32/2 | 512/10 | 248320 |
| `qwen3_5_397b_nvfp4.json` | `nvidia/Qwen3.5-397B-A17B-NVFP4` | 60 | 4096 | 32/2 | 512/10 | 248320 |

## Representative Scenario Summary (`chat-8k1k`, mixed)

TPU serving cache policy: `--kv-dtype fp8` for all checkpoints and `--state-dtype bf16` for hybrid linear-attention models.

| Model | Slice | Legal Candidates | Top Layout | TPOT Floor (ms) | Est. Tput (tok/s/chip) | Ceiling (tok/s/chip) |
| :--- | :--- | ---: | :--- | ---: | ---: | ---: |
| `deepseek_v4_1_flash` | `2x2x1` | 174 | `TP2 x DP4 x EP8` | 1.40 | 7,031.5 | 98,325.2 |
| `deepseek_v4_1_flash` | `2x2x2` | 689 | `TP2 x DP4 x EP8` | 1.40 | 7,031.5 | 98,325.2 |
| `deepseek_v4_1_flash` | `2x2x4` | 1964 | `TP2 x DP4 x EP8` | 1.40 | 7,031.5 | 98,325.2 |
| `glm_5_2_fp8` | `2x2x1` | 102 | `TP2 x DP2 x EP4 x PP2 x DCP2` | 4.28 | 3,412.7 | 33,053.6 |
| `glm_5_2_fp8` | `2x2x2` | 497 | `TP2 x DP8 x EP16 x DCP2` | 3.52 | 4,047.4 | 19,611.6 |
| `glm_5_2_fp8` | `2x2x4` | 1556 | `TP2 x DP8 x EP16 x DCP2` | 3.52 | 4,047.4 | 19,611.6 |
| `glm_5_2_nvfp4` | `2x2x1` | 120 | `TP2 x DP4 x EP8 x DCP2` | 5.59 | 5,012.3 | 24,794.0 |
| `glm_5_2_nvfp4` | `2x2x2` | 545 | `TP2 x DP4 x EP8 x DCP2` | 5.59 | 5,012.3 | 24,794.0 |
| `glm_5_2_nvfp4` | `2x2x4` | 1658 | `TP2 x DP4 x EP8 x DCP2` | 5.59 | 5,012.3 | 24,794.0 |
| `minimax_m3_1_flash` | `2x2x1` | 174 | `TP2 x DP4 x EP8` | 2.36 | 5,643.2 | 58,812.0 |
| `minimax_m3_1_flash` | `2x2x2` | 689 | `TP2 x DP4 x EP8` | 2.36 | 5,643.2 | 58,812.0 |
| `minimax_m3_1_flash` | `2x2x4` | 1964 | `TP2 x DP4 x EP8` | 2.36 | 5,643.2 | 58,812.0 |
| `qwen3_5_397b_fp8` | `2x2x1` | 102 | `TP2 x DP4 x EP8` | 1.83 | 10,237.1 | 76,248.3 |
| `qwen3_5_397b_fp8` | `2x2x2` | 437 | `TP2 x DP4 x EP8` | 1.83 | 10,237.1 | 76,248.3 |
| `qwen3_5_397b_fp8` | `2x2x4` | 1306 | `TP2 x DP4 x EP8` | 1.83 | 10,237.1 | 76,248.3 |
| `qwen3_5_397b_nvfp4` | `2x2x1` | 120 | `TP2 x DP4 x EP8` | 2.68 | 11,226.1 | 52,101.6 |
| `qwen3_5_397b_nvfp4` | `2x2x2` | 485 | `TP2 x DP4 x EP8` | 2.68 | 11,226.1 | 52,101.6 |
| `qwen3_5_397b_nvfp4` | `2x2x4` | 1408 | `TP2 x DP4 x EP8` | 2.68 | 11,226.1 | 52,101.6 |
