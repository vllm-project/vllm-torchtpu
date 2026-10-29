# Qwen3.8 on the megakernel backend

This backend serves Qwen3.8 (`Qwen3_5MoeForCausalLM` with the
Qwen3.8 shape, NVFP4 checkpoint, text only) as a prefill/decode (P/D) pair
on TPU v7x:

- The prefill runs as four pipeline stages, one per host (vLLM PP4 x PCP8).
  Each stage is one megakernel over its 23 layers, with the checkpoint's
  weights.
- The decoder is a TP32 megakernel with DCP8 attention over vLLM's pools. It
  loads a per-rank weight cache (the rank cache).
- With MTP, the decoder speculates three tokens per step with the
  checkpoint's MTP layer. The prefill writes that layer's KV for the prompt.

The Raiden KV transfer reshards the prefill's KV and GDN states from PP x
PCP to the decoder's TP x DCP layout. vLLM provides scheduling, the KV
transfer, sampling parameters and the OpenAI API.

## Requirements

- Two 32-chip TPU v7x slices (four hosts each, 2x2x4): one for prefill and
  one for decode.
- The Qwen3.8 NVFP4 checkpoint.

## 1. Build the rank cache

The decoder loads its weights from a rank cache: each rank's arrays in the
decode layout (W4A8 experts), the MTP layer's (`rankNN/mtp/`) and the
checkpoint's config and tokenizer. Build it once on a four-host allocation
with `python -m tools.megakernel` (see the
[Kimi-K3 page](README.md#tools) for how it launches):

```bash
python -m tools.megakernel --jobid JOB qwen38 preshard \
    --checkpoint /path/to/Qwen3.8 --output /path/to/qwen38-tp32
```

It refuses to overwrite an existing directory. `--against DIR` instead of
`--output` rebuilds every rank and compares it with an existing cache,
writing nothing.

## 2. Start the servers

Each role is one `vllm serve` across its slice's four hosts. Both roles
take these flags:

```bash
COMMON=(
  --kv-cache-dtype fp8 --mamba-ssm-cache-dtype float32
  --mamba-cache-mode align
  --block-size 640 --cp-kv-cache-interleave-size 128
  --attention-backend CUSTOM --enable-expert-parallel
  --trust-remote-code --language-model-only
  --hf-overrides '{"quantization_config": null}'
  --load-format megakernel
)
CONNECTOR='"kv_connector": "TPURaidenConnector", "kv_connector_module_path": "vllm_torchtpu.distributed.kv_transfer.tpu_connector"'
```

The megakernels load their own weights, so vLLM's model needs no
quantization config. Both roles also need `VLLM_KV_CACHE_LAYOUT=HND`,
`TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1`, `TPU_USE_RAIDEN_KV_CACHE_MANAGER=1`
and `TPU_RAIDEN_QWEN35_ADMISSION=1`, plus the Raiden connector's usual
environment.

Prefill (the KV producer):

```bash
vllm serve /path/to/Qwen3.8 "${COMMON[@]}" \
  --pipeline-parallel-size 4 --prefill-context-parallel-size 8 --tensor-parallel-size 1 \
  --no-async-scheduling --enable-prefix-caching \
  --model-loader-extra-config '{"mtp": true}' \
  --kv-transfer-config "{${CONNECTOR}, \"kv_role\": \"kv_producer\", \"kv_connector_extra_config\": {\"destination_decode_context_parallel_size\": 8}}" \
  --max-num-seqs 64 --max-num-batched-tokens 32768
```

Decode (the KV consumer):

```bash
vllm serve /path/to/Qwen3.8 "${COMMON[@]}" \
  --tensor-parallel-size 32 --decode-context-parallel-size 8 \
  --no-enable-prefix-caching \
  --model-loader-extra-config '{"presharded": "/path/to/qwen38-tp32"}' \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3, "model": "/path/to/qwen38-tp32"}' \
  --kv-transfer-config "{${CONNECTOR}, \"kv_role\": \"kv_consumer\"}" \
  --max-num-seqs 16
```

The speculative config names the rank cache as the draft model: its
`config.json` is the checkpoint's without the quantization config. To run
without MTP, drop `"mtp"` from the prefill's loader config and
`--speculative-config` from the decode.

## Limits

- The decoder runs TP32 with DCP8 and at most 16 requests per step.
- Both roles need `--block-size 640 --cp-kv-cache-interleave-size 128`.
  The reshard moves GDN states in whole pool pages, and those need 128-token
  KV pages.
- MTP proposes exactly three tokens. The MTP decoder needs async scheduling
  (the default): it keeps each request's next token, position and drafts on
  the device.
- Prefix caching runs on the prefill role; the decoder runs without it.
- Sampling: greedy, temperature, top-p, top-k and seed. Other sampling
  parameters are rejected.

## Tools

`python -m tools.megakernel --jobid JOB qwen38 COMMAND` runs the programs
without a vLLM server:

| Command | What it does |
|---|---|
| `bench --prompts 8192x1,3000x3 [--checkpoint CKPT] [--mtp]` | Each prefill stage's step time. Synthetic weights unless `--checkpoint` is given. |
| `decode --presharded DIR [--mtp CKPT]` | The decoder's step time per batch size at `--context` tokens. |
| `probe --checkpoint CKPT` | Each layer's output range for one prompt, with real weights. |
| `preshard --checkpoint CKPT --output DIR` | Builds the rank cache (see above). |

## Where the code is

- `src/vllm_torchtpu/megakernel/qwen38/program.py`: the program, its
  configuration checks and loader keys.
- `src/vllm_torchtpu/megakernel/qwen38/stage.py`: a prefill stage.
- `src/vllm_torchtpu/megakernel/qwen38/decode.py`: the decoder, with MTP.
- `src/vllm_torchtpu/megakernel/qwen38/kernels/`: the decode and MTP Pallas
  kernels (`decode_megakernel.py` is the entry point; `paged_kv.py` reads
  and writes vLLM's pools).
- `tools/megakernel/qwen38.py`: the tools above.
