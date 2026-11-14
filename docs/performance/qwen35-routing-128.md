# Opt-in routing order and inverse for 128 routes

Set `TPU_MOE_JOINT_ROUTING_128=1` before starting the server to enable the
fixed-size MoE routing optimization. The default is `0`, which leaves routing
preparation exactly as it is without the flag. The flag is read when the MoE
module is imported, so restart the server after changing it.

## Where it applies

On the non-ragged paths, `prepare_routed_gmm_inputs` builds the MoE
permutation in one of three ways, in this order of precedence:

1. The counting sort (`TPU_MOE_COUNTING_SORT`, on by default), when the
   `local experts + 1` buckets fit one rank block (`min(512, routed rows)`).
2. Otherwise, the packed single sort (`TPU_MOE_SINGLE_SORT`, on by default):
   one packed `lax.sort` and one scatter for the inverse, up to 2,048 routed
   rows.
3. Otherwise, the three-sort form.

The flag only replaces step 2. It takes effect when the packed single sort
would run and, in addition, there are exactly 128 routed rows, no expert
parallelism, int32 sort keys and JAX x64 disabled. The counting sort keeps
precedence, and `TPU_MOE_SINGLE_SORT=0` also turns the kernel off.

For example, Qwen3.5-35B-A3B decode on TP8 routes 16 tokens with top-k 8 (128
routes) over 256 local experts. The counting sort declines (257 buckets > 128
rows), so the packed single sort runs, and the flag replaces it. With 127 or
fewer local experts at 128 routes, the counting sort runs and the flag has no
effect.

## How it works

Each entry's stable rank counts the entries preceding it in `(key, position)`
order. The ranks are the inverse permutation; a second exact integer reduction
reconstructs the sorted positions. One Pallas invocation produces both arrays,
and the sorted token ids are `order // topk`. Signed comparisons avoid overflow
at int32 extrema, and no floating-point approximation or packed-key assumption
is needed.

The three arrays are bit-identical to what the packed single sort returns, so
masking, expert counts, gathered hidden states, routing weights, and all six
preparation outputs keep their semantics.

## Recorded performance

These results were measured before this port, against a baseline that no
longer exists. At the time, this preparation ran two `argsort` calls (a stable
order, then its inverse). The current default for the same configuration is
one packed sort plus a scatter, so the figures below do not estimate the gain
over the current default. That gain has not been measured; it must be
measured again on TPU, and it may be small or nil.

| Measurement (superseded baseline) | Baseline | Candidate | Latency reduction |
| --- | ---: | ---: | ---: |
| Qwen3.5-35B-A3B serving request, vs two `argsort` calls | 1512.1672 ms | 1469.1359 ms | 2.8457% |
| Local routing preparation, vs first sort plus an inverse helper | 9.648066 us | 7.204613 us | 25.3258% |

The serving comparison used eight TPUv6e chips with TP8, 512 input tokens, 256
output tokens, concurrency one, and the CUSTOM batched-RPA attention backend.
Ten fresh confirmation pairs each used 30 timed requests per role after ten
warmups. The bootstrap 95% interval for the mean paired candidate/baseline
ratio is [0.9577632, 0.9873624]. This does not establish that the reduction
exceeds 2% throughout the interval.

The local comparison used an intermediate first-sort-plus-inverse-helper
implementation as its baseline. It covers all six outputs at hidden[16,2048]
and routing[16,8], with synthetic inputs, 256 experts and no EP or SparseCore.

In that measurement, the changed decode program replaced 80 sort calls with 40
joint calls. Against the current default, each MoE layer's routing preparation
instead goes from one packed `sort` and one inverse `scatter` to one joint
call.

## Validation

Run the CPU regressions with:

```bash
pytest -m "not tpu_test" tests/kernels/test_stable_order_and_inverse.py \
    tests/layers/core/test_fused_moe_gmm.py -k "joint or stable"
```

On TPU, `tests/kernels/test_stable_order_and_inverse.py` also exercises the
native Pallas kernel (marked `tpu_test`).
