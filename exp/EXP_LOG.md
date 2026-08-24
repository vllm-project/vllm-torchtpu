## 08-24-qwen35-server-refactor
- **Goal:** Simplify Qwen3.5 InferenceX server configuration while preserving every resolved sweep setting.
- **Setup:** Branch `wyzhang/exp/inferencex-v1-0824`. Compare all seven 8k1k sweep points with `exp/08-24-qwen35-server-refactor/run.sh` before and after refactoring.
- **Observations:** The first post-refactor capture failed because `run.sh` still read the removed `SHARDING_TABLE` variable. Updated the harness to call `server_config_for`, the shared lookup used by the refactored scripts.
- **Results:** Exact output matches for all 7 supported sweep points. See `settings-*.log`, `overrides-*.log`, and `custom-*.log`.
- **Learnings — what WORKED:** A four-field point table in `sweep.sh` replaced layered defaults and reference-point overrides. Exact resolved serve commands and MoE gates remained unchanged across every tested case. The final harness intercepts `vllm serve` directly after `PRINT_CONFIG` was removed.
- **Learnings — what DIDN’T work:** Directly reading the old `SHARDING_TABLE` in the verification harness failed with `unbound variable` after the table was replaced. Using the public `server_config_for` helper keeps the harness aligned with the scripts.
- **Next:** Use `server_config_for` when adding sweep points so server and sweep stay aligned.
- **Status:** done
