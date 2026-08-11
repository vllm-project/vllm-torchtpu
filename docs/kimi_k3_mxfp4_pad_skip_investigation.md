# Kimi K3 MXFP4 batch-1 pad-skipping investigation

Date: 2026-08-11

## Conclusion

The roughly 240 us/layer result from the first full-model trace was real GMM
device time, but it was not a valid batch-1 routing workload. The MXFP4 packed
weights and E8M0 scales are `uint8` tensors allocated with `torch.empty`.
vLLM's dummy loader initializes floating-point tensors but, except on ROCm,
intentionally leaves integer tensors untouched. Garbage MXFP4 scales produced
invalid activations after the first MoE layer. Router top-k then collapsed onto
a fixed expert set, and rank 0 executed the equivalent of 16 active local
experts in nearly every remaining layer.

After initializing the integer MXFP4 dummy tensors to finite, deterministic
zeros, the compiled whole-model trace gives:

| Exact device event | Mean per MoE layer | Mean per decode token (92 layers) |
|---|---:|---:|
| GMM1 + fused SiLU | 46.078 us | 4.239 ms |
| GMM2 | 28.402 us | 2.613 ms |
| **Two GMMs** | **74.481 us** | **6.852 ms** |

This is the expected result: it is close to the controlled exact-shape
microbenchmark and to the original approximately 80 us expectation. Padded
routes are skipped. The model is not running with `enforce_eager=True`; every
decode token is one compiled whole-model XLA module.

## Configuration

- Model: `mgoin/Kimi-K3-pruned75`
- Load format: dummy, with deterministic finite MXFP4 integer tensors
- Device: TPU v7x-8
- Parallelism: TP8 with expert parallel enabled (EP8)
- Global/local routed experts: 224/28, with linear expert placement
- Experts selected per real token: top-16 globally
- Scheduler compile bucket: 16 tokens
- Actual decode batch: one token; the other 15 rows are padding
- Model dimensions used by the GMMs: hidden 3584, expert intermediate 3072
- MoE layers: 92
- XLA compilation: `CompilationMode.VLLM_COMPILE`, compile size `[16]`,
  `enforce_eager=False`

This report covers the requested TP8 run. Under EP8, a real token's 16 global
routes are distributed across the eight ranks; a rank commonly receives about
two local routes. The earlier EP16 setup has 14 local experts per rank and
about one local route per rank on average.

## Correct profiling procedure

1. Use finite dummy MXFP4 tensors. Dummy values need not resemble checkpoint
   values, but they must not contain uninitialized packed weights or scales.
2. Warm the model and begin capture only after prefill and first-use compilation
   have finished.
3. Profile a long, single-request decode window. Identify complete
   `tt_jit_layernorm_...` XLA module executions wholly contained in the trace.
4. Read exact raw `%gmm_v2...` XLA-op spans for the two GMM shapes. Do not use
   request wall time, Python dispatch time, or a fused-MoE annotation as kernel
   time.
5. Verify that each complete step contains 92 GMM1 and 92 GMM2 events. Report
   per-layer distributions, not just one selected event.
6. For EP-rank comparison on this single-host v7x topology, sample one worker
   profiler endpoint at a time. Starting all eight Torch profilers concurrently
   serialized device-profiler shutdown, hit the one-million-event limit, and
   truncated some device traces. Also, only worker endpoints 0, 2, 4, and 6
   expose TPU device planes; the odd endpoints return host-only/no-device
   traces. Eight logical worker files are therefore not eight independent
   device timelines on this system.

The accepted rank-0 trace starts 1.5 seconds into a warmed 200-token request
and captures 1.5 seconds. It contains 32 complete decode modules and no
prefill. Its raw path is:

```text
/mnt/scratch/kimi_k3_bs1_decode_skipfix_finite_dummy_20260811/plugins/profile/2026_08_11_08_16_05/127.0.0.1_19017.xplane.pb
```

## Correct rank-0 decode breakdown

The complete XLA-module envelope is 29.489 ms/token mean (29.490 ms median,
29.470--29.501 ms range). The exact GMM event distribution is:

| Kernel | Events/step | Mean/layer | Median | p90 | Min--max | Total/token |
|---|---:|---:|---:|---:|---:|---:|
| GMM1 + SiLU | 92 | 46.078 us | 46.086 us | 47.250 us | 44.484--48.029 us | 4.239 ms |
| GMM2 | 92 | 28.402 us | 28.386 us | 28.607 us | 28.055--29.124 us | 2.613 ms |
| **GMM total** | **184** | **74.481 us** | - | - | - | **6.852 ms** |

The GMM1 and GMM2 raw spans do not overlap: GMM2 begins immediately after
GMM1 (about 0.007 us gap in a checked layer). Therefore adding these two exact
event durations is valid. This is different from blindly summing broad XProf
categories, which may aggregate work across device timelines.

`framework_op_stats` independently gives 4.231 ms/token for GMM1 and
2.608 ms/token for GMM2, within 0.2% of the raw-event calculation. Its larger
categories are useful as an approximate whole-module breakdown:

| Category | Device self-time/token |
|---|---:|
| Dense BF16 matmuls (two grouped occurrences) | 11.306 ms |
| GMM1 + SiLU | 4.231 ms |
| All-reduce | 4.179 ms |
| Routing/dispatch major ops (`reshape`, `gather`, `sort`, `topk`) | 2.707 ms |
| GMM2 | 2.608 ms |
| KDA decode | 0.789 ms |
| MLA kernels | 0.126 ms |
| Other device ops | 3.817 ms |

These framework categories sum to 29.763 ms, 0.9% above the 29.489 ms module
envelope because the report aggregates self-time across device timelines. Use
the exact non-overlapping raw GMM spans for the MoE number, and the XLA-module
span for end-to-end device latency.

## Controlled exact-shape microbenchmark

The paired benchmark compiles GMM1 and GMM2 together, queues many invocations,
and synchronizes only after the queue. It therefore excludes per-call Python
and synchronization overhead. It uses the model's exact static shapes:

```text
local experts E=28; routed-row capacity M=256; hidden H=3584;
expert intermediate I=3072; native float4 weights; f32 block-32 scales;
GMM1 fused SiTU/SiLU-family activation; GMM2 output projection
```

| Distinct active local experts | Compiled GMM1+GMM2 pair |
|---:|---:|
| 1 | 58.42 us |
| 2 | 70.68 us |
| 4 | 95.84 us |
| 8 | 144.99 us |
| 16 | 244.14 us |

Dependent and independent queued chains gave essentially the same result.
When timed as separately compiled calls, GMM1 was about 48.8 us and GMM2 about
35.9 us at one active expert. The paired measurement is the appropriate
comparison to adjacent kernels inside the whole-model compiled program because
it preserves compilation context and avoids placing a synchronization boundary
between GMMs.

Two observations connect the microbenchmark to the model traces:

- The corrected full-model 74.48 us/layer lies near the two-active-local-expert
  point, which is the expected average occupancy for top-16 routing over EP8.
- The bad full-model 239.78 us/layer lies almost exactly on the 16-active-local-
  expert point. The original number was therefore an occupancy signal, not
  approximately 160 us of hidden CPU/profiler overhead.

## Root cause and fix

`initialize_single_dummy_weight` in vLLM returns immediately for non-floating
tensors on non-ROCm platforms. MXFP4 checkpoint containers are non-floating:
packed FP4 weights and E8M0 scales are stored as `uint8`. Before this fix they
retained whatever values happened to be in `torch.empty` storage.

The MXFP4 post-load path now detects `load_format=dummy` and zeros the packed
weights and scales before converting/moving them to the TPU. Zero is a finite,
deterministic encoding for these containers. This change is dummy-only and
cannot alter real checkpoint weights. A focused unit test fills all four
integer tensors with 255, invokes dummy initialization, and verifies that every
tensor is zero.

This is separate from pad skipping itself. The MXFP4 call path always passes
`skip_padded_tokens=True`; it does not enable an environment variable by
default. `zero_routing_weights_for_padding` clears routing weights for rows
1--15 in the compile-16/actual-1 batch. The JAX routing preparation uses

```text
valid_mask = (local_expert_id >= 0) & (routing_weight != 0)
```

and multiplies the one-hot expert counts by that mask before constructing
`group_sizes`. Thus padding contributes no GMM groups. The corrected timing is
also an end-to-end runtime check: if the 15 padded tokens still contributed
their top-16 routes, the pair would be at or beyond the high-occupancy end of
the microbenchmark rather than 74.48 us.

## Hypotheses, tests, and decisions

| ID | Hypothesis | Evidence | Decision |
|---|---|---|---|
| H1 | The model ran eager. | Launch log says `enforce_eager=False`, `VLLM_COMPILE`, compile size `[16]`; each token is one whole-model XLA module. | Rejected. |
| H2 | XProf added non-kernel overhead to the reported GMM time. | Exact raw GMM spans agree with `framework_op_stats`; events are device-side and non-overlapping. | Rejected for the GMM number. Broad category sums still need care. |
| H3 | Padded rows still entered `group_sizes`. | Runtime padding mask zeroes rows 1--15; compiled skip path counts only nonzero weights; corrected GMM time is low occupancy. | Rejected. |
| H4 | More real local experts were active than assumed. | Exact-shape active-expert sweep maps corrected time near 2 experts and bad time near 16; top-16/EP8 predicts about 2 routes/rank. | Confirmed for the bad trace; normal finite routing restores expected occupancy. |
| H5 | The microbenchmark used incomparable shapes or execution mode. | Repeated with E=28, M=256, H=3584, I=3072, native float4/f32 scale layouts, paired queued XLA execution. | Rejected after correction. |
| H6 | CPU dispatch/synchronization dominated the measurement. | Queued microbenchmark synchronizes after the loop; full trace reads device spans inside one compiled module. | Rejected. |
| H7 | A stale cache selected the old non-skipping op. | The post-change run compiled a new graph after cache removal; its graph has the padding mask and skip path. | Rejected. |
| H8 | Rank 0 alone hid a much slower EP device stream. | Sequential external captures of every profiler-capable device stream are recorded below; rank 0 has the largest GMM total. | Rejected for the observable device critical path. |
| H9 | Routing/SparseCore work was accidentally included as GMM. | The 239.78 us and 74.48 us values use only exact `%gmm_v2...` event names; routing is separately 2.707 ms/token in the corrected trace. | Rejected. |
| H10 | Dummy weights inherently invalidate pad skipping. | Padding validity comes from zero routing weights, independent of expert values. Finite dummy tensors retain skipping and produce the expected GMM latency. | Rejected. Uninitialized dummy tensors invalidated routing, not skipping. |

## Intermediate experiments

### Bad post-skip trace

```text
/mnt/scratch/kimi_k3_bs1_decode_skipfix_current_20260811/plugins/profile/2026_08_11_06_51_46/127.0.0.1_19015.xplane.pb
```

- XLA module: 44.658 ms/token.
- GMM1: 157.343 us/layer.
- GMM2: 82.433 us/layer.
- Pair: 239.776 us/layer, 22.059 ms/token.
- In the first complete step, 90/92 GMM1 events were around 160 us and only
  one was around 47 us; 91/92 GMM2 events were around 83 us and one was
  28.5 us. The first MoE layer saw finite input, then invalid dummy values
  corrupted later routing. This layerwise discontinuity was the key clue.

### Diagnostic forced-random-routing trace

`FORCE_MOE_RANDOM_ROUTING=1` was used only as a diagnostic, never as the final
configuration:

```text
/mnt/scratch/kimi_k3_bs1_decode_skipfix_random_20260811/plugins/profile/2026_08_11_07_54_24/127.0.0.1_19016.xplane.pb
```

- XLA module: 30.405 ms/token.
- GMM1: 36.630 us/layer.
- GMM2: 23.624 us/layer.
- Pair: 60.255 us/layer.

Replacing only routing removed the 16-expert signature while retaining the
same compiled full model and dummy expert tensors. That localized the problem
to invalid-activation-driven routing occupancy. The final accepted run does
not set this environment variable; it fixes dummy tensor initialization.

### Cross-device profiler experiment

The first attempt used the standard server profiler, which starts all worker
profilers together. Even a sub-second request caused several workers to log
`Hit maximum event limit of 1000000`: device-profiler shutdown serialized, so
the nominally short traces continued collecting while other workers flushed.
Those truncated traces are not suitable for cross-rank comparison.

For the accepted comparison, a temporary diagnostic hook exposed one external
profiler server per worker. Each endpoint was sampled separately for 500 ms
during a warmed batch-1 decode, with host and Python tracing disabled. The
backend produced TPU device planes on endpoints 0, 2, 4, and 6 and no device
events on the odd endpoints. The four device-plane results were:

| Profiler endpoint | Complete steps | GMM1/layer | GMM2/layer | Pair/layer |
|---:|---:|---:|---:|---:|
| 0 | 18 | 43.80 us | 26.98 us | **70.78 us** |
| 2 | 18 | 4.24 us | 11.31 us | 15.55 us |
| 4 | 18 | 26.63 us | 18.79 us | 45.42 us |
| 6 | 18 | 35.38 us | 22.96 us | 58.34 us |

The variation is expected from EP routing occupancy; endpoint 2 is frequently
at zero or very low local occupancy. Crucially, rank-0/device-stream 0 is the
slowest observed stream, not an accidentally optimistic rank. Its 70.78 us
external-profile result is close to the 74.48 us longer accepted trace and the
70.68 us two-active-expert paired microbenchmark. The temporary profiler-server
hook was removed after capture.

Trace root:

```text
/mnt/scratch/kimi_k3_bs1_decode_external_allranks_20260811
```

## Validation

Focused MXFP4 tests:

```text
8 passed, 15 warnings in 33.40 s
```

Pre-commit was run on the implementation, focused test, and this report. All
hooks passed, including isort, YAPF, Ruff, PyMarkdown, missing-`__init__`
detection, filename validation, and Pyrefly. `git diff --check` also passed.
