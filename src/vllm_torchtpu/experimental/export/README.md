# Experimental AOT export

`python -m vllm_torchtpu.experimental.export` exports a vLLM model to a StableHLO
serving directory (`metadata.json`, per-bucket MLIR modules and, with
`--export-mode checkpoint|all`, the weights in `serving_checkpoint/`).
It takes the `vllm serve` engine arguments plus a few export options:

```bash
python -m vllm_torchtpu.experimental.export \
  --model Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
  --output-dir /tmp/exported \
  --tensor-parallel-size 8 --enable-expert-parallel --quantization fp8 \
  --max-model-len 10240 --max-num-seqs 512 --max-num-batched-tokens 8192
```

The exporter loads dummy weights, so it reads only the checkpoint's config and
tokenizer files.

## Static parameters: pass them explicitly to match `vllm serve`

`vllm serve` settles several sizes at startup, some of them by measuring the
device. The exporter has to bake them into the artifact instead, and the
artifact keeps them no matter where it is served. When you want an exported
model to behave like a given `vllm serve` deployment, pass these values
explicitly rather than relying on the exporter's defaults:

- `--block-size`: defaults to the platform's choice after loading the model
  (`update_block_size_for_backend`, as in `vllm serve`), raised to fit the
  block table in SMEM. `vllm serve` logs `Using KV cache block size: N`.
- `--num-blocks`: defaults to the size computed from `--kv-cache-memory-bytes`,
  or else from `--gpu-memory-utilization` (see below). `vllm serve` logs
  `TPU KV cache size: T tokens`; `--num-blocks` = T / block size.
- `--kv-cache-memory-bytes`: unset by default. `vllm serve` logs
  `Memory statistics | ... total_hbm_avail_gb=G`; bytes = G x 2^30.
- `--max-num-seqs`, `--max-model-len`, `--max-num-batched-tokens`: default to
  vLLM's engine defaults. Pass what `vllm serve` was given (its
  `non-default args` log line lists them).
- `--token-buckets`: defaults to `compilation_config.compile_sizes`, else
  `16,32,64,128,256`. Cover the batch sizes you serve.
- `--req-buckets`: defaults to `8,16,32,64` up to `--max-num-seqs`, plus
  `--max-num-seqs`. Cover the request counts you serve.

### How `num_blocks` is sized

In order of precedence:

1. `--num-blocks`, or `num_gpu_blocks_override`, is used as given.
2. `--kv-cache-memory-bytes`: `num_blocks` = bytes / bytes per KV block. The
   device is not queried, so this also works on a host without the target TPU.
3. Otherwise, when the exporter loads the model itself: the device's HBM limit
   times `--gpu-memory-utilization` (default 0.9), minus the HBM in use after
   loading the dummy weights, divided by the bytes per KV block. Every rank
   computes this and all take the minimum, because rank 0's artifact serves
   every rank.

Bytes per KV block come from the KV state tensors the artifact declares, so
they include the attention backend's padding.

This follows `TPUWorker.determine_available_memory`, with two differences that
make the default less faithful than an explicit value:

- `vllm serve` runs a profile pass at `--max-num-batched-tokens` before
  measuring, so its KV budget excludes activation memory. The exporter does
  not, so activations must fit in the headroom `--gpu-memory-utilization`
  leaves. `--kv-cache-memory-bytes` skips the profile pass in `vllm serve`
  too, so it is the setting that gives both the same KV budget.
- The device is measured at export time. An artifact exported on one chip
  type, or at one utilization, keeps that `num_blocks` wherever it is served.

The exported KV states use `--kv-cache-dtype` (an fp8 KV cache is declared in
fp8, with fp8's packing), or the model's activation dtype when it is `auto`.

### Hybrid recurrent models

Models with recurrent layers, such as the Gated DeltaNet layers of Qwen3.5 and
Qwen3-Next, export each such layer's state as two `[num_blocks, ...]` tensors:
its conv state and its recurrent state, split over the model axis along the
conv channels and the value heads. The forward graphs take one more `user_io`
input after the states, `mamba_state_indices[max_num_seqs]`: the block holding
each request's recurrent state. These layers keep this per-layer layout rather
than vLLM's unified KV block pool, and the export runs with prefix caching off.
Bytes per KV block include the recurrent states.

### Example: match a `vllm serve` run

A `vllm serve` run of Qwen3-Coder-480B FP8 on a 2x2x1 v7x slice logged:

```text
Using KV cache block size: 128
Memory statistics | total_hbm_limit_gb=94.74GiB | ...
  | total_hbm_used_gb=56.65GiB | ... | total_hbm_avail_gb=33.35GiB
TPU KV cache size: 1,128,064 tokens, ...
```

To export the same KV capacity, pass `--block-size 128 --num-blocks 8813`
(1,128,064 / 128) together with the same `--max-model-len`, `--max-num-seqs` and
`--max-num-batched-tokens`.

## Real weights: `--export-mode`

The artifact's weights are the ones vLLM actually serves with, including what it
does to some weights while loading them: FP8 checkpoints are dequantized and
requantized to the runtime format, and checkpoint tensors are cast to the model
dtype. They are saved as an Orbax checkpoint (Zarr v2 arrays `array_{i}`, one per
`abstract_params` input, without OCDBT) in `serving_checkpoint/data/`:

| `--export-mode` | Runs on | Weights loaded | Writes |
|---|---|---|---|
| `model` | TPU | dummy | artifact, without weights |
| `checkpoint` | CPU | real | Orbax checkpoint into the existing artifact at `--output-dir` |
| `all` (default) | TPU | real, once | artifact + Orbax checkpoint |

```bash
python -m vllm_torchtpu.experimental.export --export-mode all \
  --model Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
  --output-dir /tmp/exported \
  --tensor-parallel-size 8 --enable-expert-parallel --quantization fp8 \
  --max-model-len 10240 --max-num-seqs 512 --max-num-batched-tokens 8192
```

`all` saves exactly the weights serving computes on TPU, from a single load.
`checkpoint` requantizes on CPU, where a few FP8 values that sit on a rounding
boundary round the other way (for Qwen3-30B-A3B-FP8, 0.002-0.004% of the FP8
weights differ by one FP8 ULP; scales and unquantized weights are identical).
`checkpoint` needs no TPU: pass the model arguments of the export that produced
`--output-dir`, and it loads every tensor-parallel rank's weights through vLLM on
CPU, one process per rank. Either way, one process per rank saves that rank's
shards: the processes join a JAX distributed runtime and Orbax writes each array's
chunks from the host memory of the rank that holds them, so the host needs
roughly the model size in free RAM.
