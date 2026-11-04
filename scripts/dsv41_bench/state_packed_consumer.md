# State-packed actual consumer verification

This benchmark-only runner calls the unchanged `fused_compress_rope` through its
public `ring_tables` argument. It compares independent host-oracle metadata +
actual consumer against device packed metadata + the same actual consumer.
It does not exercise Torch dispatcher integration, collect timings, or claim a
speedup. Target: existing experiment `exp-0rfzvkskez`, TPU7 whole-tile configs,
T=96/192/240/256/288, B=48, hidden=5120, ratio=2.

## Fixtures and contracts

These are **35 adapted bounded consumer cases**, not execution of the original
35 adversarial metadata-only cases. Original `state_packed_shapes.py` is unchanged.
All seven case names retain original starts, distinct kv starts, and replay
minima. Every case replaces random pages with unique bounded request pages on
both banks and forced positions with contiguous request positions. Valid
request-first rows use odd positions to exercise saved-state reads; the invalid
page in `bounds` starts even, so a missing predecessor is never consumed.
Padding positions become zero. `bounds` retains null page -1, pages 0/1 and the
last valid page; original huge and out-of-capacity pages/forced positions remain
metadata-only. These adaptations avoid concurrent duplicate writes and invalid
consumer preconditions rather than silently treating invalid metadata as valid.

Two INT32 [4096,128] owners contain 1024 FP32 [1024] records total (4 MiB).
Both arms receive identical inputs, weights, gamma, rope and independently
allocated copies of the same nonconstant initial state. Both donate owner
arguments 4/5. Kernel read-before-write ordering is unchanged. Dense reference
reads are gathered from initial owners before applying projection writes.
CPU asserts all addresses fit or are null, writes are unique, and every valid
request-first row has a read. TPU checks packed metadata against the host oracle
before invoking the consumer.

## Comparisons and artifacts

- Actual-consumer A/B: exact latent, CSA2 words, scales, and both full owners.
- Existing dense JAX compress reference: BF16 latent (rtol/atol .02), FP32 scale
  (rtol .02, atol 2e-5), decoded FP8 values (rtol .15, atol .08).
  FP8 comparison accommodates quantization-bin changes; packed integer words
  are never treated as approximately equal integer values.
- Full projected owner reference: rtol 2e-4, atol 2e-5; untouched records exact.
- JSON source SHA256 and per-case results; StableHLO/HLO for both arms at each T.
  CPU results explicitly leave `consumer_cases` empty. Failure is not recorded
  as successful verification. No optional profiles are included in this runner.

Run from repository root:

```sh
python scripts/dsv41_bench/state_packed_consumer.py --cpu-test --out /tmp/consumer-cpu
python scripts/dsv41_bench/state_packed_consumer.py --out /tmp/consumer-tpu
```

Implementation handoff: CPU fixture/arithmetic checks pass all 35 adapted cases.
Actual TPU execution is intentionally not launched by this delegated task
(no jobs authorized); parent must run on the existing experiment and inspect
numerical/compile results before claiming consumer acceptance. Numerical
thresholds are declared before TPU execution, not relaxed after a failure.
