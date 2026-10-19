# Chunked prefill indexer

DeepSeek V4.1 v2 defaults to resident-query scoring and full-row SparseCore
streaming top-k. Production routing is derived from shapes and live metadata.
The explicit `prefill_mode` argument is only retained for standalone kernel A/B;
there is no server environment selector.

The new route handles a single live request in a 2048-query bucket with 32
heads, head dimension 128, BF16 queries, packed int32 index cache, and top-k
512. Page sizes must be multiples of 16 and divide the 2048-key tile.
Candidate publication/filtering, mixed requests and unsupported shapes retain
the existing route. Long-context tails with fewer than 128 live query rows
also retain it; contexts with at most 512 visible keys need no ranking.

Working widths follow effective compressed key lengths, rather than raw token
context or allocated cache capacity. Query tiles contain 128 rows; long score
rows use 32K-key VMEM output windows. The existing query pipeline retains
a complete score row in HBM until top-k finishes. Radix scan and compaction
parameters are both 8192. Returned logical indices remain ascending with
invalid entries padded by -1.

The implementation lives in `kernels/deepseek_v41/index_selection/`:
`scores.py`, `streaming_topk.py` and their local `tuned_block_sizes.py`.

Routing uses shapes and live-request metadata, not an explicit phase flag.
Ordinary decode/DSpark workloads retain existing score/top-k kernels. A single
request padded to 2048 queries with at most 512 effective keys can take the
all-indices shortcut even in decode; it does not execute the new scorer or
streaming top-k kernel. This is not a replacement for all prefill shapes or a
claim that every decode compilation graph is identical.

The benchmark launcher does not enable this route. It is enabled by the
runtime default. For A/B runs, use distinct output/cache directories per mode;
compare source revisions and compile each revision separately.

## Kernel A/B

Use the installed serving dependencies on a TPU host. The runner includes
record masking, causal lengths, logical-index sorting and padding.

```bash
python scripts/dsv41_bench/bench_prefill_indexer.py \
  --out /tmp/prefill-ab --capacity 524288 \
  --lengths 0 512 513 50176 65536 65537 262144 524288
```

Lengths are effective key counts: with compression ratio 2, 524288 keys
correspond to approximately 1M raw context tokens. In particular, 65537 is
an intentional boundary case immediately above 64K, not a typical workload.
Use `--shuffled` for noncontiguous pages or `--distribution ties` /
`--distribution ascending` for alternate score distributions.

The runner checks all output indices against baseline, checks four rows
against an independent FP32 reference, then reports the median of 25
synchronized host-latency samples after five warmups. `memory.json` records
compiler-estimated buffers. `--devices 8` uses replicated resident inputs and
checks every device; these synthetic measurements are not serving throughput.
