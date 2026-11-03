# Qwen3.5-397B AgentX layer-run campaign on `kernels/quantized_matmul`

Campaign tooling, not serving code: nothing here is imported by vLLM.

The campaign scores `serve_e2e.py` here, a byte-for-byte copy of forge's
`scorers/serve_e2e.py` (a campaign from a repository runs the scorer its tree
holds), on `qwen3.5-397b-fp8-tp8-dp1-ep-agentx-prefill@layers`: the InferenceX
AgentX deployment of Qwen3.5-397B-A17B-FP8 (TP8, DP1, EP on one tpu7x 2x2x1 node,
Cloud DevKit) cut to its leading 4 of 60 layers (3 Gated DeltaNet + 1 full
attention, every expert kept), scored on the device's op time per step over the
steps that carry a 16k-token prefill chunk.

The scorer reads its serve configs, target and KV pin from the campaign's perfkit
corpus, which the campaign holds (not this tree):

    PERFKIT_CORPUS=<campaign corpus> \
    PERFKIT_PYTHON="$(perfkit env ensure --jax 0.11.2 --libtpu 0.0.50)" \
      perfkit scorer run --bench <pinned copy of serve_e2e.py> \
        --arm golden=<golden checkout>:control --arm cand=<candidate checkout> \
        --config qwen3.5-397b-fp8-tp8-dp1-ep-agentx-prefill@layers --candidate cand --job-id <id>

- `../../vllm/benchmarking/configs/qwen3.5-397b-fp8-tp8-dp1-ep-agentx.sh`: the
  benchmark config each arm is served with: the AgentX recipe's server flags and
  environment, and a 64 x 65536-token sweep.
- `shim/`: an empty `vllm` package so `run_benchmarks.sh --dry-run`, which perfkit
  reads the serve command from, runs on a host without vLLM.
- `validity.py`: the campaign's validity check (CPU only, about a minute): the
  arm's benchmark config launches the frozen cell, and the arm's
  `xla_quantized_matmul` compiles for tpu7x at every token bucket and dense-linear
  shape and is numerically sound at every bucket up to 3072 rows.

      python3 validity.py --arm <checkout> \
        --python "$(perfkit env ensure --jax 0.11.2 --libtpu 0.0.50)"
