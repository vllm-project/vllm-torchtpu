# KV Cache Offloading Metrics

With `TPURaidenOffloadingConnector`, alone or as a child of `TPUMultiConnector`, the `/metrics` endpoint shows which tier served each prefix-cache hit: the engine's host DRAM pool, the pool of another engine, or the storage tier. For all other metrics, see [Production Metrics](https://docs.vllm.ai/en/latest/usage/metrics/) in the vLLM documentation.

## `vllm:prompt_tokens_cached_by_source`

vLLM's counter of cached prompt tokens, by the tier that supplied them. The connector reports the hits that it serves as `host`, `p2p`, or `disk`:

| `source` | Tier |
| :--- | :--- |
| `device` | The prefix cache in TPU memory (HBM). |
| `host` | The engine's host DRAM pool. |
| `p2p` | The host DRAM pool of another engine. |
| `disk` | The storage tier. |
| `external_unspecified` | No tier: tokens of a hit that the connector cannot assign to a tier, for example because the blocks were evicted before the load started. |

It is a counter, so a scrape shows it as `vllm:prompt_tokens_cached_by_source_total`. It counts a request's tokens when the scheduler admits the request, and does not count them again when the request is admitted again after a preemption.

A hit can come from more than one tier:

- **Blocks in different tiers.** Each tier counts the tokens of its own blocks. For example, the first 1024 tokens of a 2048-token hit are in host DRAM and the other 1024 are in storage: `host` and `disk` each grow by 1024.
- **Two kinds of cached data for the same tokens.** A hybrid model caches attention KV and linear-attention state, and a hit needs both. Such tokens are counted once, under the slowest tier that holds a part of them: `disk` is slower than `p2p`, and `p2p` is slower than `host`. For example, the attention KV of a 2048-token hit is in host DRAM and its linear-attention state is in storage: `disk` grows by 2048, and `host` does not grow.
