# TPU Megakernel Advisor

Offline CPU specification package for TPU megakernel analysis. Resolves sourced
hardware parameters (`resolve_chip_spec`), HuggingFace model configurations and
locked model presets (`resolve_model_profile`, `model_profile_from_hf_config`),
6-axis deployment layouts (`TP x DP x EP x PP x PCP x DCP`), and measurement
records (`load_measurements`) without requiring TPU hardware or serving imports.

## Supported Models

Three locked basis models are shipped in `data/presets/models.json`, each verified
against a pinned HuggingFace `config.json` snapshot and Hub parameter manifest:

- `qwen3.5-397b` (`Qwen/Qwen3.5-397B-A17B-FP8`)
- `kimi-k3` (`moonshotai/Kimi-K3`)
- `glm-5.2` (`zai-org/GLM-5.2-FP8`)

Any HuggingFace config whose architecture does not match a validated preset raises
`UnsupportedArchitectureError`.

## Adding a Model

1. Add a pinned HuggingFace `config.json` snapshot under
   `tests/megakernel/data/advisor/hf_configs/`.
2. Add the model geometry and dtype entry to `data/presets/models.json` and
   reference topologies to `data/presets/deployments.json`.
