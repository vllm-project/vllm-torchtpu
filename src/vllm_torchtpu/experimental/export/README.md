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

[RESULTS.md](RESULTS.md) compares exported models served by rvllm with `vllm serve`:
nightly throughput and accuracy, plus Muse Spark Open.

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
  `Memory statistics | ... total_hbm_avail_gb=G | profile_run=ran`; bytes =
  G x 2^30. Use it only from a line that says `profile_run=ran`. With
  `profile_run=skipped`, the KV size was pinned before profiling (by
  `--num-gpu-blocks-override`, or the uniform hybrid KV layout's own count), G
  was measured without activation memory and over-sizes the KV cache: pass
  `--num-blocks` from the `TPU KV cache size` line instead.
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
2. `--kv-cache-memory-bytes`: `num_blocks` = (bytes - recurrent state bytes) /
   bytes per KV block. The device is not queried, so this also works on a host
   without the target TPU.
3. Otherwise, when the exporter loads the model itself: the device's HBM limit
   times `--gpu-memory-utilization` (default 0.9), minus the HBM in use after
   loading the dummy weights and the recurrent state bytes, divided by the
   bytes per KV block. Every rank computes this and all take the minimum,
   because rank 0's artifact serves every rank.

Bytes per KV block come from the KV state tensors the artifact declares, so
they include the attention backend's padding.

This follows `TPUWorker.determine_available_memory`, with two differences that
make the default less faithful than an explicit value:

- `vllm serve` runs a profile pass at `--max-num-batched-tokens` before
  measuring, so its KV budget excludes activation memory, unless the KV size is
  already pinned (`--num-gpu-blocks-override`, or the uniform hybrid KV
  layout's own count), in which case it skips the pass and logs
  `profile_run=skipped`. The exporter does not profile, so activations must fit in the headroom `--gpu-memory-utilization`
  leaves. `--kv-cache-memory-bytes` skips the profile pass in `vllm serve`
  too, so it is the setting that gives both the same KV budget.
- The device is measured at export time. An artifact exported on one chip
  type, or at one utilization, keeps that `num_blocks` wherever it is served.

The exported KV states use `--kv-cache-dtype` (an fp8 KV cache is declared in
fp8, with fp8's packing), or the model's activation dtype when it is `auto`.

### Hybrid recurrent models

Models with recurrent layers, such as the Gated DeltaNet layers of Qwen3.5 and
Qwen3-Next, export each such layer's state as two `[num_state_slots, ...]`
tensors: its conv state and its recurrent state, split over the model axis
along the conv channels and the value heads. A recurrent state is fixed-size
per request, so these pools hold one slot per request plus null slot 0
(`num_state_slots = max_num_seqs + 1`, recorded in `serving_config`) rather
than one per KV block. The forward graphs take one more `user_io` input after
the states, `mamba_state_indices[max_num_seqs]`: the slot holding each
request's recurrent state. With `--speculative-config` each request holds
`K + 1` consecutive slots, its state slot then one checkpoint per draft token
(`num_state_slots = max_num_seqs * (K + 1) + 1`), and the forward also takes
`mamba_slot_read_offsets[num_state_slots]`, the checkpoint each request resumes
from after a verify step, and `mamba_ckpt_indices[max_num_seqs, K + 1]`, its
`K + 1` slots. Without `--speculative-config`, `--max-verify-tokens K` gives
the same windows for tokens a step verifies without a drafter, such as tokens a
grammar forces: vLLM loads the model under an ngram config of `K` tokens, which
adds no drafter, and the artifact declares no speculative decoding. With
`--speculative-config`, `--max-verify-tokens` above its `num_speculative_tokens`
widens the window past the drafter's tokens (vLLM loads the config with `K`
tokens; the artifact keeps its own draft count). Each window slot costs one more
recurrent state per request. Whenever these windows span more than one token
(`--speculative-config`, or `--max-verify-tokens` above 1), `req_dist` grows
from 3 to 6 int32 elements: the first three are the attention kernel's request
distribution, as without windows, and the last three are the recurrent kernels'
distribution, in which a request's verify window counts as one windowed
segment rather than as a mixed sequence. Without windows `req_dist` keeps its 3
elements. These layers keep this per-layer layout rather than vLLM's unified KV
block pool, and the export runs with prefix caching off. The recurrent state
bytes come off the KV budget before it is divided among KV blocks.

### MTP drafting

With `--speculative-config '{"method": "mtp", "num_speculative_tokens": K}'`
the export loads the model together with the MTP draft vLLM builds for it (for
Qwen3.5, `Qwen3_5MTP` from the checkpoint's `mtp.*` weights) and adds the
drafter `draft_bucket{B}` (function `draft`) for every forward bucket. The draft projects with the
target's embedding and LM head, which keep their slots, so the artifact stores
them once; its own weights are exported under their checkpoint names (`mtp.`).
Its paged KV is a state numbered after the target's and counts against the KV
budget like the target's. `draft` runs the draft over every row of the step
and drafts the first token at each request's `last_token_indices` row; each
further token comes from one more pass over one row per request, as vLLM's MTP
proposer chains them.

An MTP draft from a separate checkpoint takes `"model": "<draft checkpoint>"`
(and optionally `draft_tensor_parallel_size`, which must equal the target's).
Gemma 4's assistant (`google/gemma-4-31B-it-assistant`, `Gemma4MTP`) is such a
draft: it is fed the target's last hidden state like an MTP layer, keeps its
own LM head, and its Q-only attention reads the target's KV (each draft layer
the last target layer of its attention type, as the TPU runner wires it). Its
`draft` therefore takes those target KV states, under their forward names and
indices, instead of KV of its own, writes none, and keeps every further pass at
the first pass's last position and KV length (`advance_draft_positions` is
false for it in the runner). The record stays `method: mtp`.

A model outside vLLM lists its MTP layer's weights among `abstract_params()`
under `mtp.` names and implements `mtp_abstract_states()` and `mtp_draft()`.
The export finds the target weights the draft shares by lowering it once, as
for the forward.

### DSpark and DFlash drafting

With `--speculative-config '{"method": "dspark", "num_speculative_tokens": K,
"model": "<draft checkpoint>"}'` the export loads the separate DSpark draft model
vLLM builds for it (e.g. `Qwen3DSparkForCausalLM`) and exports it as the same
drafter function `draft` / `draft_bucket{B}`. `draft_tensor_parallel_size`, if
given, must equal `--tensor-parallel-size`. The artifact's
`speculative_config` records only `method` and `num_speculative_tokens`; the
draft checkpoint is how vLLM loads it, not something the runtime reads.

- `forward` returns one more intermediate after `hidden_states`:
  `aux_hidden_states` `[B, n * hidden_size]`, the residual stream entering
  each of the draft's `n` target layers (after layer `target_layer_ids[i]`),
  concatenated per row, as vLLM's `set_aux_hidden_state_layers` returns it.
- `draft_bucket{B}` takes the drafter controls, the draft's weights (`draft.`
  for its `model.`; an embedding or LM head the checkpoint does not carry binds
  the target's slot), one paged KV state per draft layer (numbered after the
  target's, paged like it: `num_blocks` of `block_size`, through the step's
  `block_tables`) and `aux_hidden_states` in place of `hidden_states`.
- Every row of the step writes its context keys and values, projected from its
  `aux_hidden_states`, to the draft KV at its position. Then each request runs
  one non-causal block: its anchor `input_ids[last_token_indices[r]]` at
  `positions[last_token_indices[r]] + 1` followed by mask tokens (`K` tokens for
  dense DSpark checkpoints, `1 + K` with `sample_from_anchor: false`), and the
  Markov head picks the `K` drafts left to right. The block's own keys and
  values land past the accepted tokens, where the next step's context rows
  overwrite them.

Each `draft_bucket{B}` drafts only for the step's live requests. The drafter
(MTP or DSpark/DFlash) is lowered once per request count `N` in
`exporter.draft_slot_counts(B, max_num_seqs)` (powers of two below
`min(B, max_num_seqs)`, then that count), and `case_compose` merges the
variants into one module whose `@main` runs, through `stablehlo.case`, the
smallest `N` holding the step's request count (`req_dist[2]`). Every row still
writes its context KV; only the per-request work (a DSpark block, an MTP
draft's logits and further passes) covers `N` requests. The module keeps the
largest variant's signature, so the artifact contract does not change, and
`draft_tokens` rows past `N` are 0.

A DFlash draft (`"method": "dflash"` with its `model`) is driven exactly like
a DSpark one, so the artifact declares the runtime protocol, `dspark`: the same
`aux_hidden_states`, paged KV and block per step. Its block is the anchor and
`K` mask slots (`1 + K` tokens, `K` fixed by the checkpoint's block size), each
mask slot's draft is its argmax over the target vocabulary, and it always uses
the target's embedding and LM head.

These steps are copies of the TPU runner's DFlash / DSpark proposer
(`draft_model.py`), checked against it by `test_draft_model_parity.py`. The
checkpoint's confidence head (adaptive verification) is not exported.

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
`abstract_params` input, in a TensorStore OCDBT database) in
`serving_checkpoint/data/`, with `"format": "zarr2_ocdbt"` recorded in
`serving_config.checkpoint_config`:

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

## Debugging: `--debug-taps`

To find where an exported artifact and the vLLM runner diverge numerically,
`--debug-taps` exports the intermediates of chosen modules. It takes
comma-separated glob patterns of module names:

- the target's modules are named as in the wrapped model's `named_modules()`,
  e.g. `language_model.model.layers.3.self_attn`;
- the MTP draft's are named as in the checkpoint, e.g. `mtp.fc`.

A pattern that matches no module fails, and the error lists example names.

For the smallest token bucket, `forward` and `draft` are also exported with
the matching modules' outputs as extra results, after the variant's own:

```bash
python -m vllm_torchtpu.experimental.export ... \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}' \
  --debug-taps '*model.norm,mtp.fc,mtp.layers.0,mtp.pre_fc_norm_*'
```

```text
<output-dir>/assets.extra/debug/
  forward_debug_bucket16.mlir
  draft_debug_bucket16.mlir
  taps.json
```

`taps.json` lists, per function and bucket:

- the module's `file_path`;
- `num_outputs`, the serving variant's result count;
- for each tap, its `name`, `output_index`, `shape` and `dtype`.

Shapes are shard-local, as in the other modules. Tap names follow these rules:

- a module returning several tensors records `name.{i}` (a decoder layer's
  hidden states and residual are `.0` and `.1`);
- a module called more than once (the MTP layer in the draft chain) records
  `name#{call}` from its second call on.

Each debug module takes exactly the inputs of its serving variant
(`functions.<function>.<bucket>.inputs` in `metadata.json`), so a runtime can run
it in place of that variant for a step and read the taps back.

`assets.extra/` holds files outside the serving contract. `metadata.json` does
not name them, so a runtime reads nothing there unless a debug mode asks for it,
and the serving artifact is the same with or without `--debug-taps`.
