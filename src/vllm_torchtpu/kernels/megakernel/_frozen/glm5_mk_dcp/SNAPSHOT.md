# Frozen Snapshot: `glm5_mk_dcp`

- **Source Repository**: `vllm-project/vllm-torchtpu`
- **Source Branch**: `origin/close-gaps/j-ebf3918f/1-mk-tp16dcp16`
- **Commit SHA**: `fc024ad91` (`fc024ad918ce6317dc01fa0b5c77019c7b67b14d`)
- **Commit Date**: 2026-10-09
- **Commit Title**: `glm5_mk_dcp adapter: skip weight kinds no layer has (dense-only parity bisect: torch.stack of an empty list)`
- **Purpose**: Frozen reference implementation for GLM-5.2 TP16xDCP16 target and MTP drafter megakernel (M0–M7 accuracy, HLO, VMEM, and latency gates). Do not modify `kernel.py`, `config.py`, `layout.py`, `pool_alias.py`, `replicas.py`, `topk2.py`, `vtopk.py`, or `wrapper.py`.

