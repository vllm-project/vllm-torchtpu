<!-- SPDX-License-Identifier: Apache-2.0 -->

# DSv4.1 decode measurements, 2026-10-21 UTC

The 35 ms main-model forward and 14000 output tokens/chip/s targets remain
unmet. The serving measurements below cover an FP4 MoE height ladder for the
observed EP8 geometry and a joint request/scorer configuration for the
48-request full-context indexer, whole-batch router selection, WO_A row chunks
and the normalized mHC output placement. An optional calibrated expert
placement reaches 41.635 ms in its full-serving run, still above target. A later
communication-threshold trial shows no useful improvement. Selector/map fusion
has passed isolated EP8 checks but awaits a full-serving measurement.

## Configuration and provenance

The branch starts at `dsv41` revision `bdb38700dd25959c029c7d77f00cee91ccd3a0bc`.
The user-supplied reference is Falcon `exp-v35cbnk6j2`, source
`f12b6d7ac6430194ccf366033ee43c6ab208c641`. The base already contains other
optimizations; differences between that reference and this base are not
attributed to these changes.

All new serving runs use DSv4.1 Flash, 384 concurrent requests, a 98304-token
prefix, 512 output tokens, DSpark K5, TP1/DP8/EP8 and eight logical TPU7x
devices on four physical chips. Each run uses the same frozen input,
three unprofiled rounds and a separate six-step profile. Persistent compilation
caches are disabled. The runtime is torch 2.13.0+cpu, torch-tpu 0.2.0rc1,
JAX/JAXLIB 0.11.2 and libtpu 0.0.50, with vLLM revision
`e37e51dd246cf421c554e7d6f53e178e3fd29085`.

| Version | Source | Falcon experiment / result directory |
| --- | --- | --- |
| Base | `bdb38700` | `exp-qcbagjr3z8` / `results/baseline` |
| FP4 ladder | `c6ffd0fa` | `exp-qcbagjr3z8` / `results/rungs2-tuned` |
| FP4 ladder + indexer | `cd47b6c9` | `exp-1mv4ll6amh` / `results/indexer-tuned` |
| FP4 ladder + indexer + router | `7b971e97` | `exp-qg29pd6unq` / `results/router-tuned` |
| Previous + WO_A | `2881d570` | `exp-btjh485722` / `results/woa-tuned` |
| Previous + mHC VMEM output | `7bf96de9` | `exp-qfs10zulj7` / `results/mhc-vmem-tuned` |
| Previous + optional expert placement | `d1cecfb3` | `exp-gjb1967wya` / `results/expert-placement` |

The FP4 run's top-level runtime source and actual checkout are `c6ffd0fa`;
its copied nested source metadata still names the base. The first combined
trial in `exp-qcbagjr3z8` was interrupted by the experiment's two-hour limit
before formal results. Only the completed rerun above is reported.

## Serving results

Each cell with three values lists rounds 1, 2 and 3. Fixed-acceptance
throughput uses `384 * 3.47 / step_seconds / 4`, matching the reference
harness. Observed throughput uses accepted output tokens including bonus
tokens divided by the measured steady window; it has a different numerator.

| Metric | Base | FP4 ladder | FP4 ladder + indexer | FP4 ladder + indexer + router |
| --- | ---: | ---: | ---: | ---: |
| Host step median, ms | 55.011 / 55.019 / 54.959 | 54.090 / 54.278 / 54.170 | 53.086 / 52.586 / 52.530 | 53.495 / 53.188 / 53.376 |
| Fixed-acceptance output tokens/chip/s | 6056 / 6055 / 6061 | 6159 / 6137 / 6149 | 6275 / 6335 / 6341 | 6227 / 6263 / 6241 |
| Observed accepted+bonus tokens/chip/s | 6773 / 6840 / 6944 | 6945 / 6752 / 6891 | 7120 / 7048 / 7207 | 7021 / 7032 / 6819 |
| Main-model device forward, ms | 47.038 | 45.184 | 42.918 | 42.599 |
| Forward minus identified entry barriers, ms | 42.722 | 42.438 | 42.264 | 41.847 |
| Main-model FP4 MoE event time/step, ms | 15.442 | 15.163 | 15.119 | 15.127 |
| Full-context index score event time/step, ms | 2.836 | 2.835 | 2.670 | 2.669 |
| Startup HBM used per logical device, GiB range | 48.98–49.04 | 48.99–49.04 | 48.99–49.05 | 49.00–49.06 |
| Server launch to health ready, seconds | 1049.86 | 1059.46 | 1006.77 | 1015.20 |

Forward is the median of six per-step sums of the four sequential rank-0
main-model device modules, with 40 MoE layers verified in each step.
It excludes draft, logits/sampling and host gaps. It is not a sum of module
medians. Subtracting the identified `barrier-cores` intervals gives a separate
diagnostic metric, not a replacement for the actual forward measurement.
Most of the 47.04-to-42.60 ms change is entry-barrier variation; about
0.88 ms remains after this subtraction. Router selection's 43 inclusive events
(40 main and three draft) fall from 0.483 to 0.096 ms per step. The router
run's host step medians and throughput are worse than the preceding run,
despite the device gain; these runs do not establish a serving-throughput
improvement from the router change. One of its six profile steps contains
13.71 ms of entry barriers, retained in the reported samples and medians.
Operator durations are inclusive and
may overlap other work, so they must not be summed as a critical-path estimate.

HBM values are rounded startup server-log statistics, not sampled runtime
peaks. The joint indexer increases its isolated executable's temporary HBM by
12 MiB (42390016 to 54658304 bytes). Weight formats are unchanged; the optional
expert placement moves logical experts between ranks.
Startup includes loading, compilation and warmup; it is not a weight-load
benchmark. Prefill and mixed batches were not benchmarked for these changes.
The later GSM8K comparison below isolates optional expert placement; it does
not compare all accumulated changes against the original branch base.

At the reference fixed acceptance of 3.47, a 35 ms complete step would yield
9518 output tokens/chip/s; 14000 requires at most 23.794 ms. Counting all six
verification tokens, including rejected tokens, would be a different metric.
The target cannot be claimed by exchanging these definitions.

## Kernel validation and reproduction

The FP4 key includes TPU generation, hidden/intermediate dimensions, quantization
block, capacity, local expert count, EP size, allocated rows, activation, active
intermediate width, biases and scale padding. Only the measured key selects two
heights; misses keep four. Randomized paired EP8 runs at 288 local rows,
hidden 5120, active intermediate 2304 padded to 2560, 384 experts and top-k six
were bit-exact, with device kernel time approximately 0.308 to 0.292 ms.

```bash
python scripts/dsv41_bench/bench_fp4_moe_heights.py \
  --out /tmp/fp4-heights --seed 1020 --profile
```

Repeat with seed 1021. The harness checks outputs, retains individual timings,
dumps compiled HLO and optionally profiles the warmed executables.

The indexer was checked with seeds 1021–1024, lengths 49408 and 65536,
contiguous and shuffled pages, and a 17-request tail/ties case. Selected IDs
were exact. Complete score/top-k module time fell approximately
0.567 to 0.520 ms on contiguous pages and 0.741 to 0.680 ms on shuffled pages.
The larger request tile regressed the 17-request case, so the new configuration
is scoped to exactly 48 requests. The served full-context scorer count changes
from 15 to 12 per step; nine calls use the new 64K-capacity configuration and
three retain the 128K-capacity configuration.

Save these benchmark variants to `/tmp/indexer-variants.json`:

```json
{
  "reference": {
    "request_tile": 12,
    "score_config": {
      "key_tile": 99328,
      "compute_tile": 1024,
      "head_group": 32
    }
  },
  "automatic": {}
}
```

```bash
python scripts/dsv41_bench/bench_decode_overlap.py \
  --variants /tmp/indexer-variants.json --out /tmp/indexer-tuning \
  --seed 1024 --shuffled --profile --dump-hlo
python scripts/dsv41_bench/server_steps/run.py --scenario decode \
  --input /path/to/frozen-inputs.json --out /path/to/results \
  --rounds 3 --startup-timeout 7200 --profile
```

Use the runtime and checkpoint setup in `server_steps/README.md`. A smaller
benchmark does not establish whole-model throughput. Relevant CPU and TPU
regressions and pre-commit checks passed. Full-environment type checking
reported the same 57 pre-existing errors as the base, with no new errors;
it was not a clean full-environment pass.

## Whole-batch expert selection

At 288 local rows, the previous power-of-two divisor selected 32-row blocks,
executing nine grid iterations. The measured TPU7x entry processes 288 rows
in one block for 384 experts and top-k six. Other shapes and devices retain
the previous divisor. Both the one-table and separate-weight-table paths have
entries; hash-based routing still bypasses selection.

In Falcon `exp-83rwu9buu9`, EP8 whole-layer paired runs were bit-exact.
With softmax gating, selector time fell from 11.095 to 1.329 microseconds.
With `sqrtsoftplus`, correction bias, routed scale 1.5 and biased expert
logits, it fell from 11.696 to 1.743 microseconds. The latter whole-layer
device time excluding entry barriers fell from 0.52108 to 0.51082 ms.
These are synthetic whole-layer measurements, not an additional serving gain.
The regression test checks the whole-batch tile against the previous tile and
NumPy, including ties, non-finite rows and independent output weights.
The final automatic-dispatch run at seed 1022 was bit-exact and used the same
profiled executable as the explicit 288-row variant. Eight selector regression
cases passed on TPU; seven layer integration cases passed on CPU. The full
type-check error set remained identical to the base.

```bash
python scripts/dsv41_bench/bench_fp4_moe_heights.py \
  --out /tmp/router-tiles --seed 1022 --tune-router \
  --scoring-fn sqrtsoftplus --score-bias --gate-bias 0.9 \
  --routed-scaling-factor 1.5 --profile
```

This mode compares 32, 96, 144 and 288 rows with automatic table dispatch,
retaining compiled HLO and checking complete EP8 outputs against the 32-row
reference. Height tuning remains automatic throughout the router sweep.

## WO_A row chunks and serving validation

The TPU7x decode shape `[288,64,512]` with FP8 weight `[4096,8192]`,
FP32 scales, tile T=288/R=1024, quantized activations and no fused RoPE now
uses one 288-row contraction instead of three 96-row contractions. The
lookup includes device, shapes, dtypes and tile settings; other workloads
keep the previous divisor rule, and explicit `sub_t` values take precedence.

| Seed / experiment | 96-row reference (us) | 288-row candidate (us) |
| --- | ---: | ---: |
| 1023 / `exp-qg29pd6unq` | 25.439 | 23.842 |
| 1033 / `exp-1yxqyt6ap6` | 25.452 | 23.857 |
| 1035 / `exp-1yxqyt6ap6`, automatic dispatch | 25.458 | 23.813 |

These are device-module medians excluding entry barriers. Seeds 1033 and
1035 passed comparisons of all output bit patterns, including signed zero.
Seed 1023 used array equality. In the automatic-dispatch run, the explicit
288-row variant measured 23.825 us. Compiled generated code shrank from
307200 to 255488 bytes, with unchanged reported argument/output storage and
zero temporary HBM. The roughly 1.6 us gain per layer projects to only
0.064 ms over 40 layers and is not an end-to-end result.

Serving experiment `exp-btjh485722` completed three formal rounds and the
six-step profile at source `2881d570`. Its result is **not a serving speedup**:

| Metric | Router run `7b971e97` | WO_A run `2881d570` |
| --- | ---: | ---: |
| Host step median, ms | 53.495 / 53.188 / 53.376 | 55.604 / 55.747 / 55.667 |
| Fixed-acceptance output tokens/chip/s | 6227 / 6263 / 6241 | 5991 / 5976 / 5984 |
| Observed accepted+bonus tokens/chip/s | 7021 / 7032 / 6819 | 6876 / 6726 / 6641 |
| Main-model device forward, ms | 42.599 | 46.039 |
| Forward minus identified entry barriers, ms | 41.847 | 41.912 |
| Main-model FP4 MoE event time/step, ms | 15.127 | 15.160 |
| Full-context index score event time/step, ms | 2.669 | 2.672 |
| Startup HBM used per logical device, GiB range | 49.00–49.06 | 49.02–49.04 |
| Server launch to health ready, seconds | 1015.20 | 1018.22 |

The WO_A run's entry-barrier median is 4.076 ms; its six individual forward
samples are 52.396, 45.878, 45.796, 66.516, 45.982 and 46.097 ms. All six
contain the expected 40 main-model MoE layers. Removing entry barriers gives
41.912 ms versus 41.847 ms previously, so the raw forward regression mostly
coincides with increased barriers. These are separate jobs, not a same-node
A/B comparison; this does not isolate the cause of the host regression.
Inclusive WO_A-named event time decreased by about 0.060 ms per step, consistent
with its paired microbenchmarks, while main MoE, indexer and mHC event times
were similar. The exact-shape tuning remains enabled for this isolated gain
and smaller generated code, with **no demonstrated whole-model throughput gain**.
The 35 ms and 14000 output tokens/chip/s targets remain unmet.

Reproduce on the pinned TPU runtime from the repository root:

```bash
python scripts/dsv41_bench/bench_wo_a_tiles.py --out /tmp/woa-1033 --seed 1033 --profile
python scripts/dsv41_bench/bench_wo_a_tiles.py --out /tmp/woa-1035 --seed 1035 --profile
```

## mHC output placement and consumer fusion

Two additional prototypes in `exp-1yxqyt6ap6` preserve the five mHC outputs
and the FP8 activation/scale values used by a following 5120-by-1792 Q/KV
projection. All eight output checks, including the projection, were bit-exact
at seeds 1034 and 1036. Timing returns the five mHC outputs and projection;
quantized intermediates remain internal.

| Prototype | Seed 1034 reference / candidate (us) | Seed 1036 reference / candidate (us) |
| --- | ---: | ---: |
| Quantize normalized values inside the mHC kernel | 27.796 / 23.396 | 27.824 / 23.362 |
| Keep the normalized BF16 value in VMEM for its consumer | 27.747 / 26.634 | 27.646 / 26.607 |

These are complete microbenchmark device-module medians excluding entry
barriers. Explicit prequantization remains a prototype because it requires
passing additional values to the consumers. The VMEM-output implementation
retains the existing interface and has the serving validation below.

The mHC VMEM-output implementation retains the normalized BF16 activation in VMEM
for following consumers, without changing their interface or arithmetic.
At seeds 1034 and 1036, all mHC outputs, FP8 activation/scale and projected
values were bit-exact. Device medians for mHC plus the following 5120-by-1792
projection fell from 27.747 to 26.634 us and from 27.646 to 26.607 us.
The dispatch is limited to the kernel's TPU7x envelope with 288 tokens,
four streams, width 5120, BF16 norm weights and HBM gate inputs.
Other shapes retain the previous output placement. The isolated timings are
reported separately from the serving measurements below.

Source `7bf96de9` was rechecked in `exp-qfs10zulj7` with seeds 1045 and 1046;
the integrated candidate and baseline remained bit-exact across all eight
checked outputs. A 128-token fallback case at seed 1047 was also bit-exact.
For 288 tokens, compiled code decreased from 1264128 to 1251328 bytes;
arguments (29219840 bytes), outputs (15814688 bytes) and temporary storage
(zero) were unchanged. The fallback's compiled memory sizes were unchanged.
These pre-serving checks did not explicitly pin a DVFS state, so their timings
are not compared with the earlier fixed-state measurements. The formal serving
launcher retains its fixed P-state 7 configuration.

Pre-commit passed. After installing the missing `uv` tool in the evaluation
container, full-environment pyrefly reported the same 57 errors as a source copy
with the baseline mHC files restored: zero added or removed diagnostics.
The numerical/HLO/device-event archive is `results/mhc-vmem-tuned/micro-evidence.tgz`
in that experiment (SHA-256
`a59f7e3de9b4686927e58407cae04f34bdab913c030f9fae057c099024d0c78b`).
Its saved typecheck log is the initial missing-tool failure; the subsequent
paired check is stored separately as `typecheck-base.json` and
`typecheck-candidate.json` in the same result directory.

```bash
git show a04817ac:src/vllm_torchtpu/kernels/deepseek_v41/mhc/shifted_post_pre_rms.py \
  > /tmp/mhc-output-reference.py
python scripts/dsv41_bench/bench_mhc_output.py \
  --reference-source /tmp/mhc-output-reference.py \
  --out /tmp/mhc-output --seed 1045 --profile
```

Serving and profile checks in `exp-qfs10zulj7` completed successfully. All six
profile steps contain the expected four main-model modules with 1, 13, 6 and
20 MoE layers, respectively.

| Metric | WO_A run `2881d570` | mHC VMEM output `7bf96de9` |
| --- | ---: | ---: |
| Host step median, ms | 55.604 / 55.747 / 55.667 | 52.724 / 52.465 / 52.254 |
| Fixed-acceptance output tokens/chip/s | 5991 / 5976 / 5984 | 6318 / 6349 / 6375 |
| Observed accepted+bonus tokens/chip/s | 6876 / 6726 / 6641 | 7118 / 7200 / 7146 |
| Main-model device forward, ms | 46.039 | 42.608 |
| Forward minus identified entry barriers, ms | 41.912 | 41.728 |
| Main-model mHC event time/step, ms | 1.370 | 1.292 |
| Main-model FP4 MoE event time/step, ms | 15.160 | 15.152 |
| Main-model CSA event time/step, ms | 5.195 | 5.117 |
| Startup HBM used per logical device, GiB range | 49.02–49.04 | 48.98–49.04 |
| Server launch to health ready, seconds | 1018.22 | 1042.22 |

The new forward samples are 46.199, 42.728, 42.489, 60.111, 42.304 and
42.320 ms; the entry-barrier median is 1.114 ms. Most of the raw forward
reduction coincides with barrier variation. The per-step diagnostic excluding
those intervals improves by 0.184 ms. Inclusive mHC event time decreases by
0.078 ms, consistent with the paired isolated measurements; inclusive event
times may overlap and must not be added as critical-path savings. These
separate jobs do not isolate the cause of the larger host-step difference.
The change is retained for its exact outputs, smaller generated code and
measured mHC gain. Both requested performance targets remain unmet.

The verified serving analysis archive is
`results/mhc-vmem-tuned/serving-evidence.tgz` in that experiment (SHA-256
`42438353c39424c06db0fbf3ff87121ecc9328ad8b4adaccec79e4f3ca761798`).

## Experiments not adopted

For calibration captures of actual selected KV IDs, use:

```bash
python scripts/dsv41_bench/analyze_selected_kv.py /path/to/kv-selection \
  --out /path/to/kv-selection-summary.json
```

The NPZ/JSON pairs contain selected logical IDs, positions and request IDs
copied together after a real decode forward. Captures marked
`actual_stage_input_tensors` also freeze encoder and decoder input positions
before their respective calls; both must match every indexer's recorded
positions. Preserve speculative host metadata separately and use the
independent stage tensors to verify query positions. The analyzer verifies
checksums, query boundaries and compressed-context bounds before measuring per-request
unions. It compares sets rather than top-k column positions and never merges
logical IDs across requests or indexers. Reported row-traffic reductions are
ideal bounds for the captured selections, not kernel or serving speedups.
Synchronous capture copies invalidate performance measurements for that run.

A QK-only union preflight in `exp-rz9rsw9am2` used synthetic selections,
queries and BF16 keys, not real decode captures. Each of 48 requests had six
64-head queries selecting 512 rows apiece, with a union of 832 rows padded to
1024. Union construction and cache packing were outside timing. The reference
used the current CSA QK's two 256-feature accumulation steps on two MXUs,
with independent query accumulators; the candidate stacked all 384 query/head
rows and restored the original per-query column order outside Pallas.

| Seed | Independent QK (us) | Joint QK with external reorder (us) |
| --- | ---: | ---: |
| 1066 | 84.413 | 252.704 |
| 1068 | 84.374 | 254.377 |
| 1069 | 84.693 | 253.263 |

All complete QK output bits matched. Times cover all 48 requests and exclude
entry barriers, using P-state 7 and five ordered device repetitions. The
candidate computes a larger score matrix before selecting its columns;
compiled temporary storage increased from zero to 227901024 bytes. Even
with precomputed unions, external reorder was slower and is not adopted.
These results say nothing about full-attention or serving speedups.

The initial fused-reorder prototype had no timing result: after making gather indices
explicitly broadcast and removing a functional scatter, the pinned compiler
rejected its 512-column gather with `Multiple source vregs along gather dimension`.
This is an implementation/compiler constraint, not a hardware impossibility.
Source, HLO, numerical checks and device-event summaries are preserved at
`results/csa-union-qk-preflight-evidence.tgz` in `art-oksvnsji4r`, SHA-256
`0da662c10e3f9c6f62b816549a13e3f74150cee3063b21cf9f5ac4c6697227a5`.

The subsequent calibration in `exp-imdae3ecob` failed its sampling threshold:
host metadata triggered capture before every actual request had generated
44 tokens. This was a collector assertion, not an attention failure. Four
rejected diagnostics (ranks 0, 1, 5 and 6, forward 26) were recovered after
checking their original checksums, current encoder/decoder input equality,
six consecutive positions per request, all eight indexers' position/request
buffers and inactive buffer tails. Their shortest generated contexts were
40–42 tokens. This is one early full-shape forward on four ranks, not the
planned 16 steady captures. Unknown compressed-cache prefixes remain null.

Across 1536 request/indexer samples, the six-query union had median 1365.5,
90th percentile 1701 and maximum 2513 rows. There were no duplicate selected
IDs within a query. The ideal row-read reduction was 54.78%; this excludes
union formation, packing, padding, extra computation and score reordering.
Verified diagnostics are in `results/kv-rejected-evidence.tgz` in
`art-5gs7huyhse` (6992062 bytes, SHA-256
`24a53a5c62e0543a021f009a1c6101c708e09a098b8addd7521462b87f692a63`).

Two QK-only probes used these actual selected IDs with synthetic BF16 Q/K.
The fused variant replaced each unsupported 512-column source gather with
four supported 128-column gathers and restored the original column order
inside Pallas. Both candidates matched every reference output bit:

| Capture / indexer | Independent QK (us) | External reorder (us) | Fused reorder (us) |
| --- | ---: | ---: | ---: |
| Rank 0 / layer 2 / seed 1066 | 84.467 | 338.495 | 367.950 |
| Rank 6 / layer 36 / seed 1067 | 84.659 | 337.664 | 419.821 |

Each measurement covers 48 requests, six queries, 64 heads and 512 selected
keys, at P-state 7 with five ordered device repetitions and entry barriers
subtracted. Both cases padded the union to 2048 rows. Fused reorder reduced
compiled temporary storage from 454393440 to 786432 bytes, but remained
slower than independent QK even with union construction and cache packing
outside timing. Neither candidate is adopted. This is not a full-CSA or
serving measurement. Source, HLO, exactness checks and event summaries are
in `results/csa-union-qk-early-evidence.tgz` in the same artifact
(113826 bytes, SHA-256
`d8683c6f96491a55a30e519defd20fac40c5d06120bbb3872e0e243a791d8665`).

### Actual MoE routing calibration

`exp-c6elv5uzb0`, source `f3962941`, captured actual FP32 router logits and
correction biases from all 40 main-model MoE layers on all eight ranks.
The collector used fixed forwards 40 and 48, with independent encoder/decoder
position checks and actual EP ownership metadata. All 16 checksummed captures
passed validation; their shortest generated contexts were 62–91 tokens.
Startup completed in 1037.230 s and the request window completed. Synchronous
copies make this a calibration run, not a serving performance measurement.
The collector's alignment update is preserved in the evidence; its SHA-256 is
`60518fc392345dd1829712c6cd6d8f2fedb76f4d12d72ae1b8dd6366e2d65d44`.

Native TPU scoring and Pallas selection were replayed on these captured inputs.
The original in-graph selected expert IDs were not captured directly. Analyze
the replay with:

```bash
python scripts/dsv41_bench/analyze_moe_routes.py /path/to/replayed \
  --out /path/to/route-balance.json
```

The analyzer rejects incomplete or duplicate same-forward EP groups and
inconsistent biases or ownership. It retains 48 expert slots per rank, fits
greedy placements only on forward 40, and evaluates forward 48 without refitting.
The native plan rounds each source-rank/expert run to `ROWBLK=2` **before**
summing an expert's rows. The initial offline fit omitted that padding; its
archived row/tile estimates are raw-count proxies, not native work counts.
The analyzer now includes source padding (`--source-row-alignment`, default 2).
For the same initial placements tested below, corrected sums of per-layer
maximum 64-row-rounded work are 184960 versus 156480 rows. Refitting with
source padding yields 156288 rows; the timing sweep below tests that refit.
This correction does not change the recorded microbenchmark timings or outputs.
The complete KV
capture also confirmed a median six-query union of 1376 rows and an ideal
selected-row traffic reduction of 54.50% across 6144 request/indexer samples.

Paired complete EP8 MoE probes used actual forward-48 logits/biases and
synthetic BF16 activations/FP4 weights. Permutations fitted on forward 40
reordered weights and remapped expert IDs **after** native top-k selection,
preserving top-k slot order. Every complete output bit matched:

| Layer / seed | Map representation | Contiguous (ms) | Permuted (ms) |
| --- | --- | ---: | ---: |
| 8 / 1070 | Static prototype | 0.582439 | 0.499474 |
| 22 / 1071 | Static prototype | 0.592856 | 0.529180 |
| 17 / 1072 | Static prototype | 0.496447 | 0.501880 |
| 8 / 1073 | Replicated runtime operand | 0.582814 | 0.505419 |
| 17 / 1074 | Replicated runtime operand | 0.496446 | 0.509975 |

Times are five ordered complete-module device medians after entry-barrier
subtraction at P-state 7. The dynamic prototype also passes correction bias
as an operand. Load balancing benefits these skewed layers but regresses the
more balanced layer 17. These probes do not establish an end-to-end gain.

An all-layer dynamic-map sweep with seed 1075 reused just two compiled
executables across all 40 layers, using the same forward-40 placements and
forward-48 router inputs. All outputs matched bitwise. Thirty-three layers
improved and seven regressed (11, 12, 13, 17, 20, 21 and 36). The sum of
isolated module medians was 21.664918 ms versus 20.676277 ms, a 4.56% reduction.
This sum is not a main-model forward measurement. Twenty-nine layers saved
more than 5 us apiece; their isolated savings sum to 1.040508 ms. Selective
integration needs full-serving validation.

`exp-5a76c4qc86` (`art-162lb6bo44`, source `5d0c32f0`) repeats all 40 layers
with the source-aligned placements, seed 1076, and a four-way comparison:

| Variant | Sum of isolated MoE medians (ms) |
| --- | ---: |
| Original 64/128 heights, contiguous experts | 21.657505 |
| Source-aligned expert placement | 20.638479 |
| 32/128 heights, contiguous experts | 22.132250 |
| 32/128 heights and source-aligned placement | 21.278096 |

All 160 output checks were bit-exact. The smaller height regressed overall
and is not integrated. Thirty layers saved more than 5 us under the placement
change; their isolated savings total 1.075717 ms. Layer selection uses this
timing sweep; the serving run below measures the resulting policy. The verified
`results/actual-route-sweep-evidence.tgz` is 9814079 bytes with SHA-256
`210db553d1d8ec7b3ddab9cbbb243d80bc9846b43df2471ef018ce98330c965c`.

The optional `additional_config.deepseek_v41.expert_permutations` policy maps
logical expert IDs to physical IDs per main-model layer. Run:AI fetches the
original checkpoint byte ranges but filters and yields physical expert names.
The fused kernel applies the same replicated map after native top-k selection,
before EP-to-mesh ordering. Gate logits, bias, selected weights and slot order
are unchanged. Draft layers are excluded. This requires FP4 fused EP, linear
placement without EPLB and EP-filtered Run:AI loading; unsupported loading is
rejected rather than silently changing ownership. Defaults do not enable it.

The workload-specific 30-layer map is checked in as
`expert_permutations_decode_c384.json`. Its source bytes hash to
`8cf1d65275f04e2a142edf6f00b295604c6da5c9522a4f4a8f4f13ddfb7bcb7b`.
The launcher records the map and its hash in the benchmark output:

```bash
python scripts/dsv41_bench/server_steps/run.py --scenario decode \
  --input /path/to/inputs.json --out /tmp/decode-permuted --rounds 3 --profile \
  --expert-permutations scripts/dsv41_bench/expert_permutations_decode_c384.json
```

The runtime integration checks configuration, loader, bridge and model factory
behavior on the pinned runtime: 258 tests passed, including a rerun after fixing
an outdated mock's upstream signature. Full type checking adds no errors against
the same-source baseline (57 existing errors become 54). The microbenchmark
sum is not a main-model forward time.

The full run `exp-gjb1967wya`, source `d1cecfb3`, passed all three formal
windows and its separate profile. Six profile steps contain 180 mapped and
60 original main-model MoE calls, confirming the intended 30/10 layer split.

| Metric | Previous mHC run | Selective placement |
| --- | ---: | ---: |
| Main-model device forward, ms | 42.608259 | 41.635182 |
| Forward minus entry barriers, ms | 41.727823 | 40.492454 |
| Host step median, ms | 52.724 / 52.465 / 52.254 | 50.194 / 50.321 / 50.124 |
| Fixed-acceptance output tokens/chip/s | 6318 / 6349 / 6375 | 6637 / 6620 / 6646 |
| Observed accepted+bonus tokens/chip/s | 7118 / 7200 / 7146 | 7604 / 7429 / 7400 |
| Main-model MoE inclusive event time/step, ms | 15.152258 | 13.625933 |
| Startup HBM per logical device, GiB range | 48.98–49.04 | 49.01–49.04 |
| Server launch to health ready, seconds | 1042.219 | 1032.965 |

The six forward sums are 44.791704, 42.023947, 65.717492, 41.246417,
41.063629 and 41.063926 ms. The outlier remains in the measurement.
Entry barriers have a 1.138549 ms median. The 0.973077 ms forward reduction
is a measured improvement; neither the 35 ms nor 14000 output tokens/chip/s
target is met. Inclusive operator means are not additive critical-path costs.

Full-serving outputs are not bitwise-equivalent evidence. The previous run
itself reproduced only 40/384 complete greedy texts between rounds 1/2 and
1/3. The new run reproduces 31 and 34; same-round old/new comparisons match
32, 35 and 37. Of 21 requests identical across all three baseline rounds,
10, 14 and 16 match the candidate rounds. Median common-prefix lengths are
440.5–492 characters across versions, versus 450.5–460 within the baseline.
These data do not establish model-quality neutrality or its absence; quality
benchmarking remains outstanding. The placement stays explicit and defaults off.

Verified `art-6n8hvcpmag` serving evidence is
`results/expert-placement/serving-evidence.tgz`, 5393737 bytes, SHA-256
`a431a5793622cf5cac740be8c02453c6285e87b7d6f753ac0e9d34ff7a58f955`.

After serving stopped, the same job tested replacing only the FP8 arrival-row
gather in MoE combine with the existing SparseCore `ragged_gather_v2`, using
a same-width FP8/int8 bitcast and unchanged FP32 slot reduction. Both complete
EP8 outputs were bit-exact, but device medians regressed: biased seed 1077
0.496310 to 0.561177 ms; uniform seed 1078 0.459336 to 0.527027 ms.
This candidate is rejected. The verified `results/sc-combine-evidence.tgz`
is 687544 bytes, SHA-256
`a20a36cea86bb94f832d87d22e521bdf69841db82eae96acddc317cfc4e5c132`.

Verified archives in `art-hhm1zvix50`:

| Artifact path | Bytes | SHA-256 |
| --- | ---: | --- |
| `results/route-selection-evidence.tgz` | 138161865 | `8f232eba7bd3c6e66a0a74ebd81d35dc9cb942b83c744c0526d874dd7adf3c7c` |
| `results/route-replay-evidence.tgz` | 6097444 | `b28637e02f6767023f415a826b51b3c5f2d618653f747cb4a542b524c88dde6f` |
| `results/route-placement-evidence.tgz` | 990226 | `fbe6ed5b11650d7e02c46c57ff09b2cc2b3ca2b7b7e52023224c5c13259e03b6` |
| `results/dynamic-placement-evidence.tgz` | 672993 | `38b654cb0a171e57210658f8f683b45c4f10c63d6df611fbd8559d33274e689d` |
| `results/placement-sweep-evidence.tgz` | 4753934 | `8e06019db94f7c7401ca9c5eba885ec4977a4afc6afe4f4ba6bcf1185cc63b29` |

The following prototypes remain outside the production path. Unless labeled
otherwise, times are complete device-module medians after subtracting identified
entry barriers; operator-only measurements are explicitly identified.

| Prototype | Measurement | Decision |
| --- | --- | --- |
| Pack activation and routing metadata into one all-gather | Whole EP8 layer 0.469 to 0.918 ms, bit-exact | Reject |
| Compact FP4 weights with three slots and capacity 64 | Biased-gate MoE kernel 0.340 to 0.407 ms, or 0.367 to 0.448 ms, bit-exact | Reject |
| Smaller CSA interleave groups | Synchronized host latency 0.332 to 0.339–0.380 ms, bit-exact | Retain six-query interleave |
| Lossless FP8 candidate-key storage | Cache 576 to 297 MiB; packing plus four reindexes 1.482 to 3.584 ms after tile tuning | Reject despite smaller cache |
| Direct FP8 candidate QK with scales moved into the head reduction | Complete FP32 score bit-equality check failed | Do not promote or claim a speedup |
| Skip invalid candidate-tail DMA and compute | Dense synthetic unions of 3000–5400 blocks: 0.433 to 0.436 ms; full unions: 0.435 to 0.467 ms, bit-exact | Reject |
| One 49152-key DMA window with a larger VMEM budget | Full-union scorer/selection 0.436 to 0.450 ms, bit-exact | Retain 24576-key DMA windows |
| MoE routing rank windows of 32, 16 or 8 pairs | Whole EP8 layer 0.5125 to 0.5129–0.5139 ms, bit-exact | Retain existing rule |
| SparseCore gather writeback overlap and 1024 rows/subcore | Isolated gather 0.705 to 0.684 ms, bit-exact | Small synthetic gain; no serving validation or production change |
| mHC 32-token tiles or compiler-managed gate windows | With RMSNorm: 18.845 to 23.241 us (32-token/HBM), or 21.184 us (96-token/compiler), bit-exact | Retain 96-token/HBM configuration |
| mHC 288-token tile with the original 96-row projection chunks | With RMSNorm: 18.695 to 20.777 us, all five outputs bit-exact | Reject |
| Compact MoE gate/up staging with three gate/up slots and two down slots | Biased whole EP8 layer: 0.50020 to 0.50150 ms; uniform: 0.45972 to 0.45887 ms, bit-exact | No meaningful gain |
| Packed-word FP4-to-FP8 expansion | Biased whole EP8 layer: 0.50085 to 4.18015 ms; uniform: 0.45934 to 4.07370 ms, bit-exact | Reject |
| Compact gate/up weights in HBM, two staging slots | Biased whole EP8 layer: 0.53772 to 0.53714 ms; uniform: 0.45938 to 0.45773 ms, bit-exact | No meaningful performance gain; retain existing weight layout |

The compact-HBM experiment ran after serving in `exp-btjh485722`, with
biased `sqrtsoftplus` routing at seed 1037 and uniform softmax at seed 1038.
Compaction occurs outside the timed forward and removes the two unused
256-column gate/up tails. All output bit patterns, including signed zero,
match the original layout. Compiled argument storage decreases by exactly
62914560 bytes (60 MiB) per logical device and layer; temporary storage and
generated code sizes stay unchanged. Operator-only MoE medians are
0.369390 to 0.369348 ms (biased) and 0.292022 to 0.289723 ms (uniform).
The small time change does not justify changing the production weight loader.

A further compact-HBM/three-slot prototype initially halted before correctness
checks or profiling completed. Static inspection found a missing modulo-three
mapping in its DMA slot index. After that fix, `exp-bazj9i3rq6` completed both
seeds with bit-exact outputs. Whole-layer device time excluding entry barriers
was 0.53768 to 0.54127 ms for biased routing and 0.45957 to 0.45732 ms for uniform
routing. The third prefetch slot did not establish a useful gain. This was a
prototype indexing error; the failed execution is not evidence of a hardware
fault and supplies no performance result.

The same experiment tested consuming each 512-column gate/up pair through the
activation before computing the next pair, retaining the original 256-column
dots, K-block addition order and BF16 row-quantization reductions. All output
bits matched at biased seed 1039 and uniform seed 1040, but whole-layer device
time excluding entry barriers was 0.48145 to 0.48312 ms and 0.45931 to 0.45903 ms,
respectively. Compiled argument, output and temporary storage were unchanged;
generated code grew by 5120 bytes. Neither this fusion nor either compact-HBM
prototype is in production dispatch.

An exploratory G1024 requantization in `exp-bazj9i3rq6` reduced the number of
per-block scale operations but required padding the intermediate width from
2560 to 3072. The original VMEM accounting then selected one weight slot;
the profile's kernel name confirmed `nb1`. Compacting gate/up weights and
accounting for their actual staged width restored two slots. The conservative
VMEM estimate was 65748992 bytes against a 65766686-byte budget.

| Routing / seed | G512 reference (ms) | Padded G1024 (ms) | Compact G1024 (ms) |
| --- | ---: | ---: | ---: |
| Biased / 1041 | 0.55982 | 0.66395 | 0.54881 |
| Uniform / 1042 | 0.45970 | 0.62287 | 0.45388 |

These are complete EP8 device-module medians excluding entry barriers, with
the same two-height ladder. Quantization and compaction occur outside timing.
Compact and padded G1024 outputs match bit-for-bit, but both differ from G512:
relative L2 errors are 0.11498 and 0.11503; maximum absolute errors are 0.10229
and 0.10547, respectively. These inputs requantize synthetic G512 weights and
are **not a model-quality evaluation**. Compact G1024's compiled arguments use
964411392 bytes versus 970702848 for G512; temporary/output storage is unchanged.
The 1–2% isolated gain does not justify promoting the approximate quantization
and changed weight layout. G1024 remains outside production dispatch; no claim
of quality neutrality or serving improvement is made.

The corrected three-slot, gate/up activation and G1024 experiments have a
combined source/HLO/result archive at
`exp-bazj9i3rq6` / `results/final-micro-evidence.tgz`.

In `exp-c5bdngz3ad` (source `a04817ac`), explicit one- and two-block
FP4-to-FP8 lookahead retained the original K-block dot shapes, scales and
addition order. All complete EP8 output bits matched. At biased seed 1043,
whole-layer device medians excluding entry barriers were 0.571214 ms for the
reference, 0.571439 ms for one-block lookahead and 0.571498 ms for two blocks.
At uniform seed 1044 they were 0.459195, 0.459625 and 0.459485 ms. Generated
code sizes and compiled argument/output/temporary storage were unchanged.
Moving those reads earlier did not establish a speedup and is not adopted.

After serving completed in `exp-qfs10zulj7`, a two-rung MoE sweep retained
capacity 128 and changed only the lower height. Every EP8 output bit matched
at biased seed 1048 and uniform seed 1049, including signed zero.

| Routing / seed | 64/128 reference (ms) | 48/128 (ms) | 80/128 (ms) | 96/128 (ms) |
| --- | ---: | ---: | ---: | ---: |
| Biased / 1048 | 0.561563 | 0.564828 | 0.573694 | 0.580213 |
| Uniform / 1049 | 0.459417 | 0.465987 | 0.472470 | 0.483886 |

These are complete device-module medians excluding entry barriers, with
P-state 7 and five ordered profile repetitions. Argument/output/temporary
storage was unchanged. None of the alternative lower heights improved
performance, so the 64/128 dispatch remains. Source, HLO and results are at
`results/nonuniform-heights/evidence.tgz` (SHA-256
`ac1e646c094a42334aa7d1c13177a1e4018497d3b2612889f038334598d8701c`).

MoE combine experiments in `exp-rtlysk81mx` used source `ea0558b8`,
the same pinned runtime, EP8 with 288 local rows, P-state 7, 31 rotating
host repetitions and five ordered device repetitions. All complete EP8
output bits matched, including signed zero. Each biased case used
`sqrtsoftplus`, correction bias, gate bias 0.9 and routed scale 1.5;
uniform cases used softmax. Each row below has its own paired reference.

| Change | Biased seed: reference → candidate (ms) | Uniform seed: reference → candidate (ms) |
| --- | --- | --- |
| Fused scale lookup, HBM output | 1050: 0.504663 → 0.505881 | 1051: 0.459752 → 0.459507 |
| Fused scale lookup, VMEM output | 1050: 0.504663 → 0.506629 | 1051: 0.459752 → 0.461006 |
| External coefficients, VMEM output | 1050: 0.504663 → 0.512965 | 1051: 0.459752 → 0.467408 |
| Resident arrivals, fused scale lookup | 1052: 0.591569 → 0.592461 | 1053: 0.459744 → 0.459872 |
| Resident arrivals, external coefficients | 1054: 0.558883 → 0.557544 | 1055: 0.460206 → 0.458994 |

These are whole-device-module medians excluding entry barriers. The first
three variants used the existing three-buffer row-DMA combine. Resident
variants instead staged the entire 2112-by-40-by-128 FP8 arrival buffer in
VMEM, then indexed rows there. The scale-fused variants read the scale mirror
through scalar memory; external-coefficient variants retained the XLA scale
gather. All retained the original accumulation order and masked-row handling.

The post-expert tail, measured from the expert kernel's end to the enclosing
module's end, did not improve. Fused-scale row DMA increased it from about
26.3 us to 28.0–29.3 us; external coefficients with row DMA took 34.4–34.7 us.
Resident arrivals with fused scales took about 28.0 us. With external
coefficients, their paired tails were 26.836 → 26.939 us at seed 1054 and
26.262 → 26.443 us at seed 1055. Thus the small whole-module reductions in
the final table row do not establish a faster combine.

A further resident-arrival/external-coefficient sweep used reduction groups
16, 32, 48 and 96 at biased seed 1056 and uniform seed 1057. All were bit-exact.
Their whole-module medians were 0.505005–0.505284 ms against 0.505263 ms and
0.460385–0.460719 ms against 0.460986 ms, respectively. Their tails remained
26.738–26.870 us against 26.541 us and 26.328–26.558 us against 26.266 us.
No combine replacement is enabled.

Compiled argument/output storage stayed at 970702848/2949120 bytes.
For biased seed 1052, resident arrivals with fused scales reduced generated
code from 4645888 to 4333568 bytes and temporary storage from 99334336 to
98886112 bytes. External coefficients at seed 1054 used 4403200 bytes of code
and 99293408 bytes of temporary storage. These smaller compiled artifacts
did not produce a measured tail improvement; they are not server HBM peaks.

Source, HLO, numerical checks and device events are preserved under
`exp-rtlysk81mx` / `art-l6lqmlmybw`:

| Artifact path | Verified SHA-256 |
| --- | --- |
| `results/combine-scale-evidence.tgz` | `b73b33bc53c21e19cb4aee4a9f5902354b04b74860e55de802c426c4490e8a35` |
| `results/combine-resident-evidence.tgz` | `d223531c6d9d522c09e6f12884f3ccff550318db9f99eedf55b217c9573ecbf7` |
| `results/combine-resident-coef-evidence.tgz` | `1d52001cad5fedd1800afd61ae95fcb97f1c956e820374e83602c598d8ea422f` |
| `results/combine-resident-groups-evidence.tgz` | `3f3e3a827e769287ef3a1159d23303403481f0fdb97285436cd50c83ec20963e` |

Capacity-256 experiments in `exp-ruclqwp7hs` used source `275a56c1`,
the pinned runtime, EP8 with 288 local rows, P-state 7, 31 rotating host
repetitions and five ordered device repetitions. Increasing capacity reduces
the number of expert tiles for skewed routing, but also widens activation
staging and scale mirrors. The initial two-dimensional scale mirror failed
compilation because a dynamic row slice was not aligned to its eight-row tile.
Representing it as `[rows, 1, 256]` made the row dimension untiled and allowed
compilation; this layout change is included in all timings below.

| Routing / seed | Capacity 128, heights 64/128 (ms) | Capacity 256, one weight buffer (ms) | Capacity 256, two buffers, heights 64/128/256 (ms) | Capacity 256, two buffers, heights 64/128/192/256 (ms) |
| --- | ---: | ---: | ---: | ---: |
| Biased / 1060 | 0.502908 | 0.600451 | 0.512601 | 1.477304 |
| Uniform / 1061 | 0.460531 | 0.613309 | 0.478496 | 1.601546 |

Every complete EP8 output bit matched. Times are whole-device-module medians
excluding entry barriers. Two weight buffers required a prototype VMEM
estimate of 65363968 bytes against a 65766686-byte budget, accounting for
the actual 256-column widened weight blocks. At seed 1060, expert tile count
fell from 344 to 304, but expert kernel time increased from 335.154 to
342.307 us with three heights, and the post-expert tail rose from 26.733 to
28.894 us. At seed 1061 the tile count stayed at 384, and those times rose
from 292.655/26.331 to 309.102/28.612 us. The four-height regression has no
established compiler or hardware diagnosis.

A follow-up removed the intermediate 128-row height and added stronger skew:

| Routing / seed | Capacity 128 reference (ms) | Capacity 256, heights 64/256 (ms) | Capacity 256, heights 64/128/256 (ms) | Expert tiles, 128 → 256 |
| --- | ---: | ---: | ---: | ---: |
| Biased / 1062 | 0.562718 | 0.570758 | 0.562921 | 328 → 288 |
| Strongly biased / 1063 | 0.534970 | 0.536370 | 0.541608 | 227 → 179 |
| Uniform / 1064 | 0.459860 | 0.474558 | 0.479208 | 384 → 384 |

All outputs again matched bit-for-bit. Biased cases used `sqrtsoftplus`,
correction bias and routed scale 1.5, with gate bias 0.9 or 1.5 for the
strongly biased case; uniform cases used softmax. Even when the stronger skew
reduced expert tile count by 21%, whole-module time did not improve. The
64/256 expert kernel at seed 1063 was essentially unchanged at
366.291 → 366.011 us, while its tail increased from 27.125 to 28.790 us.
Capacity 128 and heights 64/128 remain the production dispatch.

Compiled argument/output storage remained 970702848/2949120 bytes. At seed
1060, the three-height/two-buffer candidate increased generated code from
4645888 to 6233600 bytes and temporary storage from 99334336 to 100959424
bytes. These are executable storage measurements, not serving HBM peaks.

The same job tested a separate load-time FP4 K permutation to avoid the
interleaving operation in a manual packed-word FP4-to-FP8 expansion. This was
a small dot probe with 16 independent RHS blocks per executable, FP8 LHS,
FP32 output and seed 1065; it was not a complete EP8 or serving benchmark.
All nine dot-output bit checks passed. Weight preparation was outside timing.

| M / K / N | Native conversion (us) | Manual interleaved (us) | Manual prepermuted (us) |
| --- | ---: | ---: | ---: |
| 64 / 512 / 256 | 9.845 | 11.190 | 10.407 |
| 128 / 512 / 256 | 10.179 | 11.484 | 10.889 |
| 64 / 256 / 256 | 9.602 | 10.019 | 9.789 |

These are medians for the entire 16-block device module excluding entry
barriers. Prepermutation improved the manual implementation but remained
slower than native conversion in every shape. No weight-format change is
adopted. A future implementation would also need to permute the active
256-row down-projection tail separately from the padded 512-row block.

The job succeeded. Source, HLO, checks and device events are preserved under
`exp-ruclqwp7hs` / `art-8vp4y0bles`:

| Artifact path | Verified SHA-256 |
| --- | --- |
| `results/capacity256-evidence.tgz` | `6a1b4baa0d6246c58455a488e489ac22c33b670a2cae92f7aa6c0f0fde63db20` |
| `results/capacity256-3d-evidence.tgz` | `f65e281f644074bd2bf876754dc8f1714d7e94e47c73c4a3eb52cb34c510c9be` |
| `results/capacity256-two-heights-evidence.tgz` | `06ceecfe8f147143f32051aacb2661d44651f7a87b89808eec309c4c5915676a` |
| `results/fp4-prepermuted-evidence.tgz` | `e531c88aed535da2f60d3583b5d5cd14beab651b70c6b66f1c1a1d0ba8af132c` |

A separate small MXU operand probe in `exp-c5bdngz3ad` tested FP8 LHS with a native
FP4 RHS at M64/K256/N256 and M128/K512/N256. The pinned compiler rejected
both with `Unsupported element type for matmul push rhs: 'f4E2M1FN'`.
Explicit FP4-to-FP8 conversion followed by asynchronous MXU operations was
bit-exact against the ordinary dot reference. This is a compiler/API result
for JAX/JAXLIB 0.11.2 and libtpu 0.0.50, not a claim about hardware capability;
there is no native-FP4-RHS timing result.

The same job tested putting CSA cache words and scale bits in a single
SparseCore gather record. For two 524288-record banks and 288-by-512 indices,
the 136- and 160-word formats were bit-exact but increased device module time
excluding entry barriers from 0.174463 ms to 1.752655 and 1.724891 ms.
Temporary storage grew from 1376064 bytes to 1360035360 and 1410367008 bytes.
Compiled HLO contains full-bank layout copies for these candidates. Packing
occurred outside timing; the copies are inside the measured executable.
The 129-word format failed numerical checks and was excluded from timing.
A flat-input variant failed compilation with `Reshape 1D ref is not supported.`
These results reject these layouts, not every possible packed representation.
The job completed successfully; its result/HLO/device-event files were retrieved
through a read-only CPU helper. The verified local archive SHA-256 is
`85ae39d7abf1685f2fe382cca6014a62e1d7463e37028adb7e6121417d210991`.

Candidate storage, dense-tail, DMA-window, routing-window and gather experiments
above ran in `exp-qg29pd6unq` after its serving process exited. The initial
1024–1800-block tail test was faster, but that density was too low to represent
the existing serving captures, whose maximum unions exceeded 5000 blocks.
The dense and full-union results determine the rejection.

Increasing SparseCore gather read concurrency from eight to sixteen did not
complete within 300 seconds after the baseline variant. Only that benchmark
process was terminated; this does not establish the cause of the stall.
The subsequent writeback-overlap comparison retained eight reads.

The mHC and asymmetric MoE experiments ran in `exp-1yxqyt6ap6`. A single
288-row mHC projection initially changed only the FP32 gates by at most
2.384e-7; retaining three 96-row projection chunks restored bit equality but
did not improve performance. The asymmetric MoE kept capacity 128 and the
automatic height ladder. Compact gate/up staging and accounting for the actual
256-column widened blocks required 62,144,512 bytes against a 65,766,686-byte
VMEM budget; compilation and both whole-layer output checks passed. The biased
case used `sqrtsoftplus`, correction bias and routed scale 1.5 at seed 1030;
the uniform softmax case used seed 1031. The packed-word FP4 expansion used the
same two input distributions and preserved signed zero, but regressed both
whole-layer device timings by more than eight times.

A further FP8 candidate-score diagnostic in that experiment applied recovered
query/key scales before the head reduction. It still changed the selected ID
arrays at compute tiles 2048, 4096 and 8192. Those candidates were excluded from
timing and remain outside the production path.

A separate 192-by-8192 dot diagnostic confirmed bit-exact query/key
round-trips through the FP8 cache and power-of-two scales. Both XLA and Pallas
FP8 dots nevertheless differed from their BF16-input references, while XLA
and Pallas agreed for each input type. `highest` precision did not remove the
difference, and this pinned TPU runtime rejected `ANY_F8_ANY_F8_F32` as
unsupported. This rules out cache loss as the cause in that diagnostic; it
does not establish the backend's internal arithmetic.

Dense FP8 GEMM fusion was measured in `exp-sh81ry5nm6` at commit `0e742279`,
with the same pinned TPU runtime and seed 1028. All 57 baseline/candidate cases
were bit-exact, including signed zero. The table compares the reference with the
fastest Pallas candidate for each observed shape at 288 rows. Times are device
module medians excluding entry barriers, from five ordered profile repetitions.

| K | N | Reference (us) | Best Pallas (us) | Candidate |
| --- | --- | --- | --- | --- |
| 5120 | 1792 | 9.113 | 12.473 | Fused quantization, M=288/N=256 |
| 1280 | 32768 | 26.748 | 30.552 | Fused quantization, M=288/N=2048 |
| 8192 | 5120 | 25.608 | 24.849 | Fused quantization, M=288/N=1024 |
| 5120 | 4608 | 17.719 | 19.126 | Fused quantization, M=288/N=512 |
| 2304 | 5120 | 11.107 | 12.235 | Fused quantization, M=288/N=1024 |

Four shapes regressed. The remaining 0.759 us isolated gain has no second-seed
or serving validation and does not justify a dispatch change. Some synchronized
host medians improved by 1–4 us despite slower device execution; those host
differences are not evidence of a faster kernel. No GEMM replacement is enabled.

```bash
python scripts/dsv41_bench/bench_dense_fp8.py \
  --out /tmp/dense-fp8 --seed 1028 --profile
```

Use a fresh output directory. The benchmark records every numerical failure and
compilation error, and times only bit-exact candidates. `--interpret-check` runs
the small CPU check instead of TPU timing.

## Grouped FP4 contraction experiments

`exp-lcj3uvjmz5` / `art-xte2ut35og` tested isolated alternatives to the FP4
expert FFN at source `db6dccb4`. The reference widens each 512-row FP4 block
to FP8, computes its dot, applies its FP32 column scale, then adds the partial
results. The candidates instead combine two or all contraction blocks before
the dot, either dequantizing to BF16 or rescaling to FP8 with a shared column
scale. Weight storage, capacity 128, heights 64/128 and expert prefetch remain
unchanged. These are experimental replacements used only by the benchmark.

The complete EP8 workload uses 288 local rows, 384 experts, top-k six, hidden
5120 and active intermediate 2304 padded to 2560. It uses synthetic weights
and inputs; these tests do not establish model quality. The table reports
device module medians excluding entry barriers, from five ordered profile
repetitions after 31 randomized synchronized host repetitions. The 256-column
variants are shown; narrowing to 128 columns regressed further in every case.

| Case / seed | Reference (ms) | Two-block candidate (ms) | All-block candidate (ms) | Candidate relative L2 error |
| --- | ---: | ---: | ---: | ---: |
| BF16, biased / 1079 | 0.467859 | 0.841221 | 0.841238 | 0.037283 / 0.037283 |
| BF16, uniform / 1080 | 0.459886 | 0.991852 | 0.992026 | 0.037287 / 0.037287 |
| FP8, biased / 1081 | 0.451959 | 1.531566 | 0.743145 | 0.053716 / 0.064201 |
| FP8, biased, power-of-two scales / 1082 | 0.484180 | 1.481079 | 0.727515 | 0 / 0 |

Biased cases use `sqrtsoftplus`, correction bias, gate bias 2 and routed scale
1.5. Uniform routing uses softmax. General scales are sampled from
`[0.005, 0.025)`; the final case rounds these scales to powers of two. Both
256-column FP8 variants are bit-exact in that final case, but remain slower.
The 128-column FP8 variant is not bit-exact even with power-of-two scales
(relative L2 0.039169). No assumption about real checkpoint scale distribution
is made from this synthetic test.

The BF16 regression is also present in the expert kernel event itself:
0.299987 → 0.672144 ms for the biased all-block variant and
0.291995 → 0.824101 ms for uniform routing. Its compiled temporary storage
is unchanged, while generated code grows from 4645888 to 6827008 bytes in
the biased case. These facts do not identify a specific compiler scheduling
cause. All candidates are rejected on measured latency; none enters serving,
and no quality benchmark was run for them.

Both evidence archives contain the isolated source, full-output error checks,
compiled HLO and device events. The original XPlane files remain in the same
Falcon artifact under each case's `profile` directory.

| Artifact path | Verified SHA-256 |
| --- | --- |
| `results/fp4-bf16-contract-evidence.tgz` | `7340748768193eacde81ba53ec783251d6b4203505e8e00ee0113beb1965e7cc` |
| `results/fp4-fp8-contract-evidence.tgz` | `6610479e7b181061ca77e75bc9ea28b0c033d40865c147ced2d1944291e163a1` |

After extracting the corresponding archive, put `source` on `PYTHONPATH`
alongside the checkout's `src` and use the pinned runtime described above:

```bash
python source/bench_fp4_bf16_contract.py --out /tmp/fp4-bf16-biased \
  --seed 1079 --profile --scoring-fn sqrtsoftplus --score-bias \
  --gate-bias 2 --routed-scaling-factor 1.5
python source/bench_fp4_bf16_contract.py --out /tmp/fp4-bf16-uniform \
  --seed 1080 --profile
python source/bench_fp4_fp8_contract.py --out /tmp/fp4-fp8-biased \
  --seed 1081 --profile --scoring-fn sqrtsoftplus --score-bias \
  --gate-bias 2 --routed-scaling-factor 1.5
python source/bench_fp4_fp8_contract.py --out /tmp/fp4-fp8-power2 \
  --seed 1082 --profile --scoring-fn sqrtsoftplus --score-bias \
  --gate-bias 2 --routed-scaling-factor 1.5 --power-two-scales
```

A separate reinspection of `exp-gjb1967wya` finds four full-context index
selection conditionals totaling 3.213629 ms per forward. They contain the
2.671015 ms of score kernels already reported, plus top-k and preparation;
these inclusive branch times must not be added to their contained operators.

## MoE communication compiler flags

`exp-padazu1xvk` (`art-4lqp0wb45u`, source `883fed02`) compared all-gather
SparseCore offload thresholds of zero (the serving default), 1 MiB and 64 MiB.
Each policy used a fresh process, the pinned runtime, EP8, 288 local tokens,
P-state 7, dynamic SMEM negotiation and enhanced launch barriers. All-reduce's
threshold stayed at 64 MiB. Two biased `sqrtsoftplus` inputs used seeds
1083/1084, correction bias, gate bias 2 and routed scale 1.5. Policy order was
reversed for the second seed; each process recorded 31 host samples and five
separate profile repetitions. Full FP32-cast BF16 output hashes matched across
all three policies for each seed; every output was finite.

| Seed | Threshold | Device module (ms) | Minus entry barriers (ms) | Host median (ms) |
| --- | --- | ---: | ---: | ---: |
| 1083 | Zero | 0.520761 | 0.439372 | 0.905439 |
| 1083 | 1 MiB | 0.517347 | 0.429744 | 0.907004 |
| 1083 | 64 MiB | 0.532950 | 0.457215 | 0.892510 |
| 1084 | Zero | 0.465749 | 0.407834 | 0.877564 |
| 1084 | 1 MiB | 0.461152 | 0.398057 | 0.876026 |
| 1084 | 64 MiB | 0.508395 | 0.433410 | 0.866740 |

Compiled HLO places the small routing-payload all-gather on TensorCore at
1 MiB while retaining the FP8 activation all-gather on SparseCore. The interval
before the expert kernel, excluding entry barriers, drops from 113.95 to
104.03 us and from 113.58 to 104.44 us. Expert-kernel and combine-tail times
are nearly unchanged. This supports a small dispatch improvement; host timing
does not establish a throughput gain. A full-serving test is still required.
The benchmark launcher accepts `--all-gather-offload-min-bytes 1048576` and
records that setting; zero remains its default.

A fourth policy enabled concurrent SparseCore offloading, offload queuing,
latency-hiding layer scheduling and dependent-collective scheduling, using flags
from the [MaxText tuning guide](https://maxtext.readthedocs.io/en/maxtext-v0.2.5/guides/optimization/benchmark_and_performance.html).
Both attempts failed compilation because the latency-hiding layer scheduler
requires the SparseCore collective aggregator. No timing exists for that policy.
The experiment therefore ended with failure status after preserving the six
completed comparisons; it is not a wholly successful sweep.

Source, HLO, runtime versions, output checks and device events are preserved in
`results/moe-compiler-flags-evidence.tgz`, 1002313 bytes, verified SHA-256
`7f468c3643c4dce755f838d8cfa385cefee34090cbf110919a4dae0df525191b`.

The full-serving follow-up, `exp-v4y0rtvuqj` / `art-mq2e6ibq2f`, uses source
`c5c1abe8` and the same optional placement map as `exp-gjb1967wya`. Its
1 MiB threshold gives 41.625856 ms main forward versus 41.635182 ms previously,
which does not establish a useful gain. Host step medians are 51.431451,
51.439920 and 51.254165 ms; fixed-acceptance rates are 6477, 6476 and
6499 tokens/physical-chip/s, below the previous 6637, 6620 and 6646.
Observed accepted-plus-bonus rates are 7361, 7099 and 7242. Zero remains the
default threshold. The separate profile, three formal rounds and startup
audits all completed; neither performance target was reached.
The profile covers all 40 main-model layers (30 mapped, ten original).
Paired subtraction of entry barriers gives 40.554884 ms, compared with
40.492454 ms previously; this diagnostic also shows no improvement. Resident
HBM is 48.99–49.03 GiB/logical device and startup is 1025.946 seconds.
The verified `results/hybrid-offload-serving-evidence.tgz` is 43997150 bytes,
SHA-256 `b4703c0b3272841a8e1947b22f61c0fe3b62f32c8e74360ee4e36a6359a3c3ea`.

The same job reran the scheduler sweep with its required sparse-core collective
aggregator enabled, resolving the earlier compilation failure. All eight cases
completed with bit-exact outputs. At seeds 1085/1086, adding scheduling flags
changes barrier-subtracted device time from 468.386/554.657 to
468.899/554.654 microseconds at the zero threshold, and from 458.370/544.667
to 458.438/544.961 microseconds at 1 MiB. Inclusive module time and synchronized
host latency do not show a consistent benefit. These scheduler flags are not
promoted. The verified `results/moe-scheduler-v2-evidence.tgz` is 1338665 bytes,
SHA-256 `ddca058f3490f817aa4b1ed4634fe080edcb095b27064d8b5e5abf1613e36e4a`.

## Expert placement quality comparison

`exp-cxneias4qy` / `art-ihfbeop646`, source `883fed02`, ran sequential
baseline and optional-placement servers on the same rig. Each completed one
formal C384 serving round, then two identical GSM8K repetitions before shutdown.
The frozen test dataset has SHA-256
`3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`.
The fixed sample contains 256 questions selected with seed zero; the request
hash is `d62c5beee42461673a41fa8f9914d8b3ce6c78b5d15b242f9d66a2d0130b55f6`.
Both modes use zero-shot chat, temperature/generation seed zero, disabled
thinking, concurrency 32 and a 2048-token output limit.

| Mode | Round 1 correct | Round 2 correct | Failures / truncations |
| --- | ---: | ---: | ---: |
| Baseline | 248/256 | 248/256 | 0 / 0 |
| Placement | 248/256 | 249/256 | 0 / 0 |

Round 1 has identical numeric answers for all 256 questions. Round 2 loses
no correct answers and gains question 93; that same question changes between
the placement server's own repetitions. Full response texts match across modes
for 238 and 237 questions; within-mode repeats match for 241 baseline and 238
placement questions. Thus greedy text differences alone do not establish a
quality regression. Exact paired McNemar p-values are 1.0 in both rounds.
This sample shows no accuracy decrease, but does not prove equivalence or an
accuracy improvement. The repeated generations are not 512 independent cases,
and this is not the few-shot leaderboard protocol.

The verified `results/placement-quality-evidence.tgz` is 39152627 bytes,
SHA-256 `56b1c072c1c816a045e74aa6d471bec1eeac594022fe02e531eeb27ded27b961`.
It includes the frozen data, per-question responses, manifests and evaluation
source. The job completed successfully and was released after artifact checks.

## Fusing the expert-ID map into selection

After both quality servers stopped, the same job compared the separate ID
lookup with two Pallas selector variants at biased seeds 1087/1088. The map
is a dynamic replicated uint32 operand. Both variants preserve logical expert
tie breaking and unbiased weights, and match every complete EP8 output bit.
One packs logical/physical IDs into the integer tie key; the other uses the
existing selection mask to reduce the physical ID. The latter is simpler and
is selected for the measured TPU7x/288-row/384-expert/top-six/two-table key.
Unmeasured shapes retain the separate lookup. Placement itself remains opt-in.

| Seed / variant | Device module (ms) | Minus entry barriers (ms) | Host median (ms) |
| --- | ---: | ---: | ---: |
| 1087 / separate | 0.540629 | 0.467578 | 0.934679 |
| 1087 / packed key | 0.547922 | 0.450633 | 0.925304 |
| 1087 / masked sum | 0.547745 | 0.450574 | 0.924990 |
| 1088 / separate | 0.576132 | 0.491581 | 0.948897 |
| 1088 / packed key | 0.571791 | 0.473061 | 0.934506 |
| 1088 / masked sum | 0.562066 | 0.473430 | 0.931703 |

Each process used 31 shuffled host repetitions and five ordered profile
repetitions with the serving compiler flags and zero all-gather threshold.
Inclusive device time does not improve at both seeds: entry barriers remain
in the table rather than being silently removed. Masked-sum temporary storage
falls from 101496768 to 101431296 bytes and generated code from 4552192 to
4528128 bytes; argument/output storage is unchanged.

The verified `results/fused-selector-map-evidence.tgz` is 1016083 bytes,
SHA-256 `337ec651c5cc968d0becfb2de7d68de017dff646e10a921c9d05f58175f85aa5`.
A subsequent small integer-input probe confirms that one FP32 MXU accumulator
can receive FP8 and BF16 updates in either order on the pinned compiler.
It establishes primitive support only, not attention speed or quantization
quality (`results/mixed-mxu-evidence.tgz`, 4575 bytes, SHA-256
`12948ed37da0c84a7c65d566585d6de80b79f6a9149fad82fc0be2b7603b990b`).

The production port was then checked in `exp-v4y0rtvuqj`, using base `c5c1abe8`
plus the archived patch with SHA-256
`5f2e508d93e71e6e2753dd72b0f09fbf302a5cda6e945c7f33ea5c942c4e5b7c`.
The benchmark disables only the fused-map dispatch for its reference; both
variants call the actual production layer with the same dynamic map and weights.
At two new seeds, all full EP8 output bits match and the isolated gain persists:

| Seed / variant | Device module (ms) | Minus entry barriers (ms) | Host median (ms) |
| --- | ---: | ---: | ---: |
| 1090 / separate | 0.523481 | 0.442158 | 0.869094 |
| 1090 / fused | 0.484947 | 0.426389 | 0.860511 |
| 1091 / separate | 0.597718 | 0.538179 | 0.974099 |
| 1091 / fused | 0.571186 | 0.516672 | 0.957667 |

The same 31-host/five-profile repetition protocol applies. Compiled HLO removes
the separate `u32[288,6]` ID-map gather. Production generated code is 4527616
bytes; temporary, argument and output storage agree with the
prototype figures above. Both local and pinned-runtime CPU router tests pass
28 cases, with four existing TPU-only tests skipped in CPU mode; the EP8
benchmark separately exercises compiled TPU execution. Full-environment pyrefly
reports the same 54 existing errors for base and candidate, with zero new errors.
The first validation attempt stopped because the image lacked pytest; after
installing pytest 8.4.2, the retry completed successfully. Both attempts' logs
are retained. Changed-file pre-commit checks pass.

The verified `results/selector-map-production-evidence.tgz` is 689830 bytes,
SHA-256 `d3e166c196e3732cf38739dbe7f5e33363b0302c0d67a8b4a0cb0306eca6aaf9`.
It includes the source patch, exact runner/compiler flags, checks, timings,
HLO and extracted device events; raw profiles remain in the Falcon artifact.
Full-serving validation of this fusion is still pending. Neither the isolated
speedup nor the placement-only quality comparison establishes that the complete
branch meets the two requested performance targets.
