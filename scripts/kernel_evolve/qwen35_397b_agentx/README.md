# Qwen3.5-397B AgentX layer-run campaign: what the scorer reads

The campaign is scored by `serve_e2e.py` here, a byte-for-byte copy of forge's
`scorers/serve_e2e.py` at the time the campaign was frozen (a campaign from a repository
runs the scorer its tree holds), on `qwen3.5-397b-fp8-tp8-dp1-ep-agentx@layers`: the
model cut to its leading 4 of 60 layers (3 Gated DeltaNet layers and 1 full-attention
layer, every expert kept), served on one tpu7x 2x2x1 node through Cloud DevKit with the
AgentX deployment's server flags, and scored on the device's op time per step: decode-only
steps and steps carrying a 16k-token prefill chunk, weighed by the AgentX replay's own
step mix (`decode_share`).

That scorer reads its serve config, its target and the cell's rules from a perfkit
corpus. They are here, so a run on any host reads them from its clone of this tree:

    PERFKIT_CORPUS=<clone>/scripts/kernel_evolve/qwen35_397b_agentx/corpus \
    PERFKIT_PYTHON=<a python with jax and numpy> \
      perfkit scorer run --bench <clone>/scripts/kernel_evolve/qwen35_397b_agentx/serve_e2e.py \
        --arm golden=<golden checkout>:control --arm cand=<candidate checkout> \
        --config qwen3.5-397b-fp8-tp8-dp1-ep-agentx@layers --candidate cand --job-id <id>

`PERFKIT_PYTHON` is the python the scorer reads the arms' profiles with: jax's
`ProfileData` reads the captures, so it needs jax (CPU is enough) and numpy.

- `corpus/scorers/serve_e2e.toml`: the serve config, its cell, `remote = "cdk"`, the
  gs:// checkpoint the cut is planned from, the replay's step mix, and the two layer runs,
  each streaming a checkpoint already cut to its layers (`cut_env`); the file says why.
- `corpus/targets/vllm-torchtpu-qwen35-agentx.toml`: the target. Its `repo` is `.`, the
  arm's own tree (each arm runs with its tree as the working directory). It names the
  serving image by digest and the checkpoint the pods stream (`MODEL_URI`, gs://).
- `corpus/ledger/cell-blocks.toml`: the cell's block rules, and its rows and census from
  golden's 4- and 8-layer runs (`perfkit layers budget`); `kv-blocks.json`: the full
  model's KV cache allocation, which every layer arm pins its cut to; `cell-map.json`,
  `experiments.jsonl`: the cell.
- `scripts/vllm/benchmarking/configs/qwen3.5-397b-fp8-tp8-dp1-ep-agentx.sh`: the
  benchmark config the arms are served with: the AgentX recipe's server flags and
  environment.
- `shim/`: lets `run_benchmarks.sh --dry-run`, which perfkit reads the serve command
  from, run on a host without vLLM installed.
- `validity.py`: the campaign's validity check. The arm's benchmark config is the one
  it is served with, so this fails a tree whose config launches another cell.

## Segment 2: the router top-k kernel

The segment on `src/vllm_torchtpu/kernels/router_topk` scores
`qwen3.5-397b-fp8-tp8-dp1-ep-agentx-prefill@layers`: the same cut, serve command and cell,
scored on the steps that carry a 16k-token prefill chunk alone (`score_phase = "prefill"`).
`corpus/scorers/serve_e2e.toml` says why the decode-only steps are left out.
`qwen3.5-397b-fp8-tp8-dp1-ep-agentx-profiled` is the full model with the phased profiler
armed, for a scoping job of golden alone.

`validity.py` adds two checks on the router kernel, run in a child under `--python`
(default `$PERFKIT_PYTHON`, which then needs jax 0.11.2 and libtpu 0.0.50 as well as
numpy; `pip install jax==0.11.2 libtpu==0.0.50 numpy`): an ahead-of-time compile for tpu7x
at every token bucket the server pads to, and an exact comparison with the documented
algorithm in TPU interpret mode at each of them.
