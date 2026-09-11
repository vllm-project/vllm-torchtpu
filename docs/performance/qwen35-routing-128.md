# Opt-in routing order and inverse for 128 routes

Set `TPU_MOE_JOINT_ROUTING_128=1` before starting the server to enable the
fixed-size MoE routing optimization. The default is `0`, which preserves the
original two `argsort` calls. Restart the server when changing this setting;
environment values and compiled graphs are cached.

The joint kernel is selected only for 128 flattened routing entries, signed
int32 expert IDs and sort keys, no expert parallelism, and JAX x64 disabled.
For example, a compiled block of 16 tokens with top-k 8 has 128 routes.
Routing values can vary arbitrarily, including duplicate keys and invalid-route
sentinels. This is a shape specialization, not a general sorting replacement.

Each entry's stable rank counts the entries preceding it in `(key, position)`
order. The ranks are the inverse permutation; a second exact integer reduction
reconstructs the sorted positions. One Pallas invocation produces both arrays.
Signed comparisons avoid overflow at int32 extrema and no floating-point
approximation or packed-key assumption is needed.

When the joint path is ineligible, the first sort remains. The same flag gates
an inverse-only helper for non-EP int32 IDs whose sorted permutation is
int32[128]. Remaining inputs use both original sorts. Masking, expert counts,
gathered hidden states, routing weights, and all six preparation outputs retain
their existing semantics.

## Recorded performance

These results were measured on frozen captured revisions before adding the
flag and porting to current upstream. They are not new benchmarks of this PR.

| Measurement | Baseline | Candidate | Latency reduction |
| --- | ---: | ---: | ---: |
| Qwen3.5-35B-A3B serving request | 1512.1672 ms | 1469.1359 ms | 2.8457% |
| Complete local routing preparation | 9.648066 us | 7.204613 us | 25.3258% |

The serving comparison uses the original two-sort implementation, eight TPUv6e
chips with TP8, 512 input tokens, 256 output tokens, concurrency one, and the
CUSTOM batched-RPA attention backend. Ten fresh confirmation pairs each used
30 timed requests per role after ten warmups. The bootstrap 95% interval for
the mean paired candidate/baseline ratio is [0.9577632, 0.9873624]. This does
not establish that the reduction exceeds 2% throughout the interval.

The local comparison instead uses the intermediate first-sort-plus-inverse-helper
implementation as its baseline. It covers all six outputs at hidden[16,2048]
and routing[16,8], with synthetic inputs, 256 experts and no EP or SparseCore.
Its percentage must not be treated as a full-model improvement or as a
comparison against the original two-sort preparation.

The changed model decode program replaces 80 sort calls with 40 joint calls.
Both outputs are consumed; the other 18 active programs retain identical LLO.
Instruction counts are scoped to the resolved native callees, not a model
cycle estimate. The PR links the full before/after LLO, extracted callee bodies,
per-request latency records, and the retained verification reports.

## Validation

Run the CPU regressions with:

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src python -m unittest discover \
    -s tests/kernels -p test_stable_order_and_inverse.py -v
JAX_PLATFORMS=cpu PYTHONPATH=src python -m unittest discover \
    -s tests/kernels -p test_inverse_permutation.py -v
```

On TPU, the first test file also exercises both native Pallas kernels without
interpretation. Fresh TPU performance validation of the flag-gated upstream
port remains separate from the captured measurements above.
