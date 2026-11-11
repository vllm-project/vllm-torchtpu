# Forward utilization and timing metrics

Target-model forwards automatically emit `vllm:tpu_forward_*` histograms through
the existing TPU KV connector metrics transport. No profiling flag is needed.
Each real runner chunk contributes one observation, including partial chunks.
Compilation warmup, zero-token KV coordination and draft-model forwards do not
contribute batch observations. Without a TPU connector metrics sink, observations
are drained but not exported.

Every histogram has the usual model/engine labels plus `rank` (worker global rank)
and `bucket` (the actual **local** padded forward shape). Neither request IDs nor
arbitrary lengths are labels. Histograms aggregate observations; they are not an
individual-forward trace.

## Batch composition

| Suffix | Observation |
| --- | --- |
| `valid_tokens` | Sum of scheduled query tokens in this chunk; excludes cached history and padding |
| `capacity_tokens` | Global padded capacity from the sequence layout plan |
| `utilization_ratio` | `valid_tokens / capacity_tokens` for this forward |
| `request_min_tokens` | Smallest request query span scheduled in this chunk |
| `request_max_tokens` | Largest request query span scheduled in this chunk |
| `requests` | Number of real requests in this chunk |

These are **global logical batch counts within a DP replica**, repeated on
PP/PCP/TP workers. For example, PCP8 local bucket 1024 has global capacity 8192;
a 6143-token chunk has
74.99% utilization. Select one worker when computing logical tokens per second;
do not add PP/PCP/TP copies. Sum one representative worker per DP replica to
measure total logical tokens per second. Inspect ranks separately to compare
pipeline stages.
Request lengths describe the current scheduled span, not the whole prompt or
remaining unscheduled input.

Capacity-weighted utilization, grouped by local bucket:

```promql
sum by (bucket) (rate(vllm:tpu_forward_valid_tokens_sum{rank="0"}[1m]))
/
sum by (bucket) (rate(vllm:tpu_forward_capacity_tokens_sum{rank="0"}[1m]))
```

The `valid_tokens_count` rate gives forward frequency and bucket selection
frequency. Divide any batch histogram's `_sum` rate by its `_count` rate for the
mean observation. Use the same engine/model/host filters in numerator and
denominator. Histogram quantiles expose distributions rather than exact extrema
over a scrape interval.

## Timing and gaps

| Suffix | Observation |
| --- | --- |
| `host_issue_seconds` | Instrumented host `forward_model` call duration |
| `host_gap_seconds` | Previous forward call return to the next call |
| `completion_span_seconds` | Background-observed start/end Event completion span |
| `completion_gap_seconds` | Previous forward end Event observation to the next start Event observation |
| `observer_lag_seconds` | Start Event submission to collector pickup |

`forward_timing.py` collects these observations independently of the existing
runner stage event timing. Forward completion metrics are **host-observed
approximations, not TPU hardware durations**.
No new synchronization is added to the serving thread. Event waits run on the
forward observer's background collector, with at most 1024 in-flight timings.
When full, it retains batch and host samples and skips completion sampling.
The first forward has no gap observation; gap labels refer to the following
forward's bucket. A skipped completion sample
breaks the completion-gap chain. Compare completion-span count with valid-token
count to check coverage, and inspect observer lag before interpreting spans.

Gaps include MTP, sampling, KV work, PP communication, scheduling and periods
without requests. Other work can overlap a host gap. Neither host nor completion
gap should be called device idle time or used as an exact TPU bubble percentage.
Use a short device profile to separate those components and calibrate the
completion estimates. Deferred observations may arrive in a later stats report;
an idle worker flushes them on its next runner stats collection.

Compare lengths during steady load with the same cache-hit length and concurrency.
Use token-weighted utilization and forward frequency together: smaller actual
batches can lower tokens/s without making each forward slower. Compare like
buckets before attributing a change to attention history cost, and exclude
warmup, the first cold requests and workload transitions from rate windows.

Some exporters normalize the `vllm:` prefix to `vllm.`. Verify the exported metric
names and units before adapting these queries to a hosted monitoring backend.
