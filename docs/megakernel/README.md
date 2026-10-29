# Kimi-K3 on the megakernel backend

This backend serves Kimi-K3 as a prefill/decode (P/D) pair on TPU v7x. Each
role runs the tpu-megakernel Pallas kernels instead of vLLM's layer stack.
vLLM still provides scheduling, prefix caching, the Raiden KV transfer
between the roles, sampling parameters and the OpenAI API.

## Requirements

- Two 32-chip TPU v7x slices (four hosts each, 2x2x4): one for prefill and
  one for decode.
- vLLM 0.29 with this checkout installed editable.
- The Kimi-K3 checkpoint and, for speculative decoding, the DSpark
  speculator checkpoint.

## 1. Pre-shard the weights

The kernels load their weights from a pre-shard directory in a per-rank
layout. Create it once with `python -m tools.megakernel` (see
[Tools](#tools) for how it launches on an allocation). `preshard k3` and
`preshard prefill` use the host CPUs only, not the TPUs.

```bash
python -m tools.megakernel --jobid JOB preshard k3 \
    --checkpoint /path/to/Kimi-K3 --output /path/to/kimi-k3-tp32
python -m tools.megakernel --jobid JOB preshard prefill \
    --checkpoint /path/to/Kimi-K3 --output /path/to/kimi-k3-tp32
# Only for DSpark speculative decoding:
python -m tools.megakernel --jobid JOB preshard dspark \
    --draft /path/to/dspark --output /path/to/kimi-k3-tp32
```

Run `preshard prefill` after `preshard k3`, with the same output directory.
It refuses to overwrite an existing cache. To check an existing pre-shard,
`preshard verify --against DIR --checkpoint CKPT --draft CKPT` rebuilds a
sample of one rank and compares it byte for byte.

## 2. Start the servers

Each role is one `vllm serve` across its slice's four hosts. Both roles
take these flags:

```bash
COMMON=(
  --tensor-parallel-size 32 --enable-expert-parallel
  --decode-context-parallel-size 32 --cp-kv-cache-interleave-size 256
  --block-size 256
  --enable-prefix-caching --mamba-cache-mode align
  --trust-remote-code
  --load-format megakernel
)
CONNECTOR='"kv_connector": "TPURaidenConnector", "kv_connector_module_path": "vllm_torchtpu.distributed.kv_transfer.tpu_connector"'
```

Prefill (the KV producer):

```bash
vllm serve /path/to/Kimi-K3 "${COMMON[@]}" \
  --model-loader-extra-config '{"presharded": "/path/to/kimi-k3-tp32", "draft": "/path/to/dspark"}' \
  --kv-transfer-config "{${CONNECTOR}, \"kv_role\": \"kv_producer\"}" \
  --max-num-seqs 16 --max-num-batched-tokens 8192
```

Decode (the KV consumer):

```bash
vllm serve /path/to/Kimi-K3 "${COMMON[@]}" \
  --model-loader-extra-config '{"presharded": "/path/to/kimi-k3-tp32"}' \
  --kv-transfer-config "{${CONNECTOR}, \"kv_role\": \"kv_consumer\"}" \
  --max-num-seqs 8 \
  --speculative-config '{"method": "dspark", "model": "/path/to/dspark", "num_speculative_tokens": 7, "num_speculative_tokens_per_batch_size": [[1, 4, 7], [5, 8, 3]]}'
```

To run without speculative decoding, drop `"draft"` from the prefill's
loader config and `--speculative-config` from the decode. The two roles
must agree on DSpark: the KV transfer rejects a pair where only one side
uses it.

Then put the P/D proxy in front of the pair. It sends each request to
prefill, then to decode:

```bash
python examples/disagg/toy_proxy_server.py --port 8100 \
  --prefiller-hosts PREFILL_HOST --prefiller-ports 8000 \
  --decoder-hosts DECODE_HOST --decoder-ports 8000
```

The Raiden connector also needs its usual environment on both roles
(controller address, ports, premapped host buffer). See
[`raiden_pd_prefill.yaml`](../developers_guide/examples/raiden_pd_prefill.yaml)
for an example. A `--kv-cache-memory-bytes` flag on either role sets its
KV budget explicitly. The prefill's premapped host pool is twice that
budget, with an 8 GiB minimum. A `TPU_PREMAPPED_BUFFER_SIZE` setting
overrides it.

Setting `VLLM_XLA_CACHE_PATH` on both roles keeps the exported kernels
across restarts, so a warm restart skips most of the startup compilation.

## Options

| Option | How |
|---|---|
| DSpark draft schedule | `num_speculative_tokens_per_batch_size` in the decode's `--speculative-config`: each `[low, high, drafts]` entry sets how many drafts batch sizes `low`..`high` verify. Supported: 3 or 7 drafts at batches 1-8, 3 at batches 9-16, 0 anywhere. The decoder rejects any other schedule at startup. |
| FP8 KV cache | `--kv-cache-dtype fp8 --block-size 512 --cp-kv-cache-interleave-size 512` on both roles. It changes the arithmetic and may change generated tokens, so evaluate quality before using it by default. |
| Attention DP2 | `"attention_dp": 2` in the decode's loader config. It runs attention as two 16-rank groups, which allows larger speculative batches (3 drafts up to 32 requests). |
| BF16 prefill MoE reduction | `TPU_K3_MOE_REDUCE_DTYPE=bfloat16` on the prefill. It halves the MoE ring traffic but rounds at every hop. The default is `float32`. |
| Synthetic acceptance (benchmarking) | With `"rejection_sample_method": "synthetic"` in `--speculative-config`, add e.g. `"synthetic_acceptance_lengths": {"3": 2.5, "7": 3.5}` to the decode's loader config. This sets a mean acceptance length per draft width. The 7-draft value must match `synthetic_acceptance_length`. |

## Limits

- TP32, DCP 32 and block size 256 only (512 with the FP8 KV cache).
- Plain decode goes up to `--max-num-seqs` (kernel-checked to 64).
  Speculative verifies go up to 64 rows: 8 requests with seven drafts, or 16
  with three. Under attention DP2, three drafts reach 32 requests.
- The decode role rejects sampling parameters that the kernels do not
  implement: logprobs, penalties, `min_p`, logit bias, allowed and bad words,
  and `min_tokens`.
- Structured outputs work, but while a structured request is in the batch,
  the whole batch decodes without speculation.

## Tools

`python -m tools.megakernel` runs the serving programs without a vLLM
server. Run it from a Slurm login node with `--jobid JOB` (default
`$JOBID`), a four-host (32-chip) allocation. It relaunches itself with
`srun` as 32 processes, using the Python environment `--venv` (default
`$K3_VENV_DIR`, else the current one).

Every command except `preshard` takes `--presharded DIR`. Every `check`
also takes `--model`, the Kimi-K3 checkpoint, for the tokenizer and the
prompts. Other common flags:

- `--kv-cache-dtype fp8`: run with the FP8 KV cache.
- `--set FIELD=VALUE`: override a kernel setting (`KernelConfig` in
  `src/vllm_torchtpu/megakernel/kimi_k3/config.py`) to compare
  configurations. It can be repeated.
- `--json FILE`: append one record per result, with the config digest and
  git revision.

### Correctness checks

| Command | What it checks |
|---|---|
| `check handoff [--dspark CKPT]` | Prefill, then the KV pools, then decode, against a decode-only replay of each prompt. With `--dspark`, it also checks the draft ring that prefill hands over. |
| `check spec --dspark CKPT` | DSpark against plain greedy decode, batch-size transitions and seeded sampling. Add `--large-batches 16,32` for the larger three-draft batches (32 needs attention DP2). |
| `check packing` | Prompts packed into one prefill window against each prompt alone. |
| `check greedy --output FILE` | Real prompts replayed on the decoder, then greedy tokens saved to `FILE`. Compare two configurations' files. |
| `check masked [--dspark CKPT]` | Grammar-masked (structured output) decode against unmasked decode. |

For example:

```bash
python -m tools.megakernel --jobid JOB --presharded /path/to/kimi-k3-tp32 \
    check handoff --model /path/to/Kimi-K3 --dspark /path/to/dspark
```

To check attention DP2 with the tools, pass `--set attention_dp=2`.

### Benchmarks and profiles

```bash
# Decode step time per batch size and context length
python -m tools.megakernel --jobid JOB --presharded DIR \
    bench decode --batches 1,8,16 --contexts 1152,32768
# Speculative steps at a fixed synthetic acceptance
python -m tools.megakernel --jobid JOB --presharded DIR \
    bench decode --batches 1,4 --dspark /path/to/dspark --acceptance 0.9,0.8,0.7,0.6,0.5,0.4,0.3
# Prefill step time: fresh prompts, and 3000 new tokens on a 32K cached prefix
python -m tools.megakernel --jobid JOB --presharded DIR \
    bench prefill --prompts 8192x1,2048x4,3000x1@32768
```

- `bench decode` options: `--mix 1152x4,131072x4` adds one batch of mixed
  context lengths, and `--temperature` times the sampled ops.
- `bench prefill` option: `--token-file` takes prompts from real text
  instead of random token ids.
- `bench ... --trace DIR` saves an xprof trace of each case.
- `profile decode|prefill` runs the same cases with a trace. With
  `--llo DIR`, it also dumps the compiled LLO of every op.

Step times vary between allocations. Compare numbers only within one
allocation and with the same prompts.

### Coverage

`--coverage FILE` merges the kernel lines that a run executes into `FILE`.
It turns the export cache off, so every op is traced.
`python -m tools.megakernel.coverage FILE` then lists, for each kernel
file, the functions that no run entered and the branches that none took.

## Where the code is

- `src/vllm_torchtpu/megakernel/backend.py`: the `megakernel` load format and
  its loader keys.
- `src/vllm_torchtpu/megakernel/runner.py`: the model runner that hands each
  step to the program.
- `src/vllm_torchtpu/megakernel/kimi_k3/`: the prefill (`prefill.py`) and
  decode (`decode.py`, `decode_ops.py`) programs.
- `src/vllm_torchtpu/megakernel/kimi_k3/kernels/`: the Pallas kernels
  (`prefill_megakernel.py` and `decode_megakernel.py` are the entry points).
- `tools/megakernel/`: the tools above.
