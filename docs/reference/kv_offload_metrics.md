# KV Cache Offloading Metrics

With `TPURaidenOffloadingConnector`, alone or as a child of `TPUMultiConnector`, the `/metrics` endpoint shows which tier served each prefix-cache hit: the engine's host DRAM pool, a host DRAM pool reached through the global registry (another engine or a store node), or the storage tier. For all other metrics, see [Production Metrics](https://docs.vllm.ai/en/latest/usage/metrics/) in the vLLM documentation.

## `vllm:prompt_tokens_cached_by_source`

vLLM's counter of cached prompt tokens, by the tier that supplied them. The connector reports the hits that it serves as `host`, `p2p`, or `disk`:

| `source` | Tier |
| :--- | :--- |
| `device` | The prefix cache in TPU memory (HBM). |
| `host` | The engine's host DRAM pool. |
| `p2p` | A block read through the global registry, from another engine or from a store node. |
| `disk` | The storage tier. |
| `external_unspecified` | No tier reported: tokens pulled from a prefill engine by `TPURaidenConnector` under `TPUMultiConnector` (which sets no hit source), or tokens of an offload hit that `TPURaidenOffloadingConnector` cannot assign to a tier (for example because the blocks were evicted before the load started). |

It is a counter, so a scrape shows it as `vllm:prompt_tokens_cached_by_source_total`. It counts a request's tokens when the scheduler admits the request, and does not count them again when the request is admitted again after a preemption.

A hit can come from more than one tier:

- **Blocks in different tiers.** Each tier counts the tokens of its own blocks. For example, the first 1024 tokens of a 2048-token hit are in host DRAM and the other 1024 are in storage: `host` and `disk` each grow by 1024.
- **Two kinds of cached data for the same tokens.** A hybrid model caches attention KV and linear-attention state, and a hit needs both. Such tokens are counted once, under the slowest tier that holds a part of them: `disk` is slower than `p2p`, and `p2p` is slower than `host`. For example, the attention KV of a 2048-token hit is in host DRAM and its linear-attention state is in storage: `disk` grows by 2048, and `host` does not grow.

## `tpu_raiden_*` Metrics

When `additional_config.raiden_telemetry_backends` is unset (defaulting to `["buffered"]`) or includes `"buffered"` without `"cloud_logging"`, C++ `BufferedMetricsExporter` samples from both the `EngineCore` scheduler (`KVCacheStore` recall, save, eviction, host-pool block, and storage lookup metrics) and TPU worker processes (`KVCacheManager` H2D/D2H DMA, host memory, and storage I/O metrics) are drained onto vLLM's `/metrics` endpoint alongside `vllm:kv_offload_*`. When `TPURaidenOffloadingConnector` runs alone, it drains both processes; under `TPUMultiConnector`, `TPURaidenConnector` drains the worker processes and `TPURaidenOffloadingConnector` drains the scheduler.

- **Lazy initialization.** Buffered gauges (`tpu_raiden_host_pool_blocks`, `tpu_raiden_storage_filesystem_bytes`, `tpu_raiden_host_memory_breakdown_bytes`, `tpu_raiden_buffer_allocated_bytes`) flush onto `/metrics` once the engine runs its first step. Worker-side gauges carry a `local_rank` label so TP ranks do not overwrite one another.
- **One drain per process.** The C++ sample buffer is process-wide and each read empties it. Under `TPUMultiConnector`, `TPURaidenConnector` reads the worker-process buffer (reporting `KVCacheManager` samples via `TpuKVConnectorStats`) and `TPURaidenOffloadingConnector` reads the scheduler-process buffer, so each sample reaches `/metrics` once.
- **Standalone Prometheus SHM mode.** Setting `additional_config={"raiden_telemetry_backends": ["prometheus"]}` disables buffered `/metrics` collector registration so metrics are served only from Raiden's standalone shared-memory exporter without duplicate scraping. Setting `additional_config={"raiden_telemetry_backends": []}` disables Raiden telemetry completely.
- **Cloud Logging.** `cloud_logging` and `buffered` share one C++ buffer, and a Cloud Logging flush empties it. When `cloud_logging` is in `raiden_telemetry_backends`, vLLM does not read the buffer, so `tpu_raiden_*` metrics go to Cloud Logging and not to `/metrics`.
