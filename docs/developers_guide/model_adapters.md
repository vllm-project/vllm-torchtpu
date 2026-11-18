# Model adapters

A model adapter lets one model directory own the per-step model inputs, the
dummy/profile forward and the KV cache of `TPUModelRunner`, without copying
the runner. The interface is in `src/vllm_torchtpu/runner/model_adapters.py`.
It is off by default: with no registration, `select_model_adapter` returns
`None` and every runner, KV cache and worker path is unchanged.

No adapter is in this tree yet. The CPU tests in
`tests/runner/test_model_adapters.py` use stubs to run the runner branches.

## Registration

A model directory calls `register_model_adapter(architectures, factory)` from
its registration hook. `TPUModelRunner.load_model` then calls
`factory(runner)` for `model_config.architecture`, before weight loading. The
factory returns a `ModelAdapter` subclass instance, or `None` to keep the
ordinary path for this configuration. A result of another type raises
`TypeError`. If a different factory is registered for the same architecture,
it replaces the earlier one with a warning.

## Contract

| Hook | When the runner calls it | What the adapter does |
| --- | --- | --- |
| `bind_model(model)` | Once, after the target model loads. | Binds the model (for example, builds its cache object). |
| `get_kv_cache_spec()` | From `KVCacheManager.get_kv_cache_spec`. | Returns the KV cache spec of each layer that needs a cache. |
| `initialize_kv_cache(config)` | From `KVCacheManager.initialize_kv_cache`, in place of the ordinary allocation. | Allocates the cache and sets `runner.kv_cache_config` (to `config`), `runner.attn_groups`, `runner.kv_caches`, the input batch block tables and any mamba slot state. The manager checks the first three right after the call. Returns the live named buffers; it may return `None` only when no KV connector is configured. The manager registers the returned buffers with the KV connector. |
| `begin_step()` | Once per `execute_model` step, after request reordering. | Returns the token store that every snapshot of the step carries. |
| `prepare_inputs(snapshot)` | Once per forward chunk, with a `ModelInputSnapshot`. | Returns the model kwargs of the chunk; `forward_model` passes them to the model. |
| `dummy_run(num_tokens, num_reqs, profile_prefix, dp_lockstep=...)` | From `_dummy_run` (profiling, warmup, DP lockstep chunks). | Runs the dummy forward on its own cache layout. Returns the wall time in ms when profiling. |
| `bind_draft(drafter)` | Optional. After the async drafter loads. | Binds the draft model layers. The default does nothing. |
| `precompile_fused_inputs(paddings, rows_options)` | Optional. In `capture_model`, after the backbone. | Precompiles the adapter input programs. The default does nothing. |
| `kv_cache_memory_budget()` | Optional. In `TPUWorker.determine_available_memory`. | Returns the KV cache bytes if the adapter sizes its cache itself, else `None` (the default). A number replaces the measured budget. |

## Supported configurations and limits

`TPUModelRunner._check_model_adapter_support` runs in `load_model` and raises
`NotImplementedError` before weight loading for a configuration that is not
supported. An adapter supports:

- one pipeline stage (no pipeline parallelism),
- no LoRA,
- text-only inputs (no multimodal inputs),
- no speculative decoding, or speculative decoding through the async
  hidden-state drafter only (`_is_async_drafter`: the `use_eagle()` methods
  or DFlash), without adaptive verification.

`_dummy_run` also raises `NotImplementedError` for a decode-shaped profile
(`profile_decode_query_lens`) or a pipeline hand-off (`through_handoff`).

## Testing plan

- CPU: `tests/runner/test_model_adapters.py` covers registration, selection in
  `load_model`, the configuration check, the KV cache delegation and its state
  check, the dummy run delegation and the precompile call.
  `tests/runner/test_kv_save_nan_check.py` runs `execute_model` through the
  adapter path over two chunks, and `tests/worker/test_tpu_worker.py` covers
  the adapter KV budget.
- TPU CI: the first in-tree adapter lands in its model directory
  (`src/vllm_torchtpu/models/<model>/experimental/`) together with an
  end-to-end TPU CI test of that model, which runs every hook above on a TPU
  device. Until then, only the CPU stubs run these branches.
