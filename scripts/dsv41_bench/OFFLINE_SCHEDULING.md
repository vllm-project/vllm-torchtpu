# Offline bucket and scheduling checks

These tools screen co-located, offline DSv4.1 serving changes without starting
a server or allocating an accelerator. The target configuration is CED,
48 requests per rank and DSpark K5, with target buckets 288, 512 and 2048.

## Kernel admission

```sh
python scripts/dsv41_bench/audit_bucket_routes.py \
  --trace --output routes.json
```

The audit calls actual kernel eligibility functions and tile selectors with
abstract arrays and a mocked v7 descriptor. `--trace` also builds JAXPRs for
Q-A, Q-B, the ratio-2 compressor and fused output projection. It does not
compile TPU programs or validate numeric parity or speed.

At 512, those kernels have valid general configurations; Q-A, Q-B and the
compressor do not have measured 512-specific tuning entries. Sorted FP8 index
writes admit 512. Extra MoE routing, SWA and compressor metadata fusions do
not admit 512. Their fallback does not disable the entire MoE or attention
kernel. Data-dependent candidate selection, mixed-query SWA sharing, full
bridge dispatch, draft execution and Engram are outside this descriptor audit.

Do not add 512 to a set of *measured* buckets using a tracing result alone.

## Host scheduling model

```sh
python scripts/dsv41_bench/simulate_prefill.py \
  --input replay-input.json --dp-size 16 --prefill-model delta \
  --match-output-mean --calibrate-arrivals --draft-slots 4 \
  --output dp16-reference.json
python scripts/dsv41_bench/simulate_prefill.py \
  --input replay-input.json --dp-size 8 --prefill-model delta \
  --match-output-mean --calibrate-arrivals --draft-slots 4 \
  --cost-reference dp16-reference.json --output schedule.json
```

Input is a metadata export with `requests`, `summary` and `formal` objects.
Requests contain slot/rank/trajectory/turn, start and finish times, outcome,
prompt length and completion length. `summary` contains historical metrics;
`formal.categories.decode.median_step_ms` is the decode timing anchor.
No prompt or response text is required.

The model keeps each slot's successor dependent on its previous completion.
It compares immediate admission, the existing cadence/drain behavior,
strict one/two-step windows and two different 512 limits. CED retained rows
come from the unchanged production planner. Active slots and token budgets
are bounded; finite-cohort checks conserve prefill work and output tokens.

`rowcap512` limits scheduled target rows while keeping the input budget at
2048. `budget512` lowers both limits to 512. On the original vLLM pin, DSpark
K5 charges four additional drafting slots per scheduled request, so these
policies differ substantially. The backport of main PR #2208 removes that
charge for the TPU's separate draft forward; model it with `--draft-slots 0`.
Epic's independent request-derived draft query ladder is retained.
The report also includes `budget_fix_only` and `strict8_budget_fix` arms that
override the charge to zero on the same workload as the original baseline.

Right-censored requests have unknown output lengths. Missing slots are filled
from observed same-rank samples. `--match-output-mean` rescales those samples
using aggregate output/arrival counts; `--calibrate-arrivals` fits the scale
to the historical per-rank arrival rate per engine step. This is a surrogate
workload, not recovery of missing request histories. The calibration history
and all assumptions are saved in the result.

The model does not run the vLLM allocator, async execution queue or sampling.
The companion CPU tests exercise the real pinned AsyncScheduler separately,
including continued chunks, an open drain window, and the K5 budget backport.

The opt-in `TpuPrefillWindowScheduler` implements the strict one-step window
with the existing DP-synchronized throttle signal. On other steps it pauses
new admissions and temporarily fences in-progress prefills using the upstream
request eligibility field. Active decode requests continue; held prefills keep
their KV blocks and active request slots. The fence is restored after each
scheduling call, including exceptions.

Select it explicitly with
`--scheduler-cls vllm_torchtpu.core.tpu_scheduler.TpuPrefillWindowScheduler`
and `--prefill-schedule-interval 4`. Keep `DP_SCHED_ENABLED=0`: epic's platform
rejects that automatic selector combined with a custom scheduler. The current
default scheduler is unchanged. This candidate is host-tested at PP1 with K5;
the DP16 recipe below has also passed hardware serving qualification.

512 stage costs and the encoder/decoder cost split are unknown. Timing
sensitivity varies them explicitly and fits the old 2048 cost to aggregate
replay timing at its measured DP size. Without a matching cost reference,
cross-DP timing sensitivity is omitted. Historical DP16 costs applied to DP8
are hypothetical. Use
bucket counts, occupancy and sensitivity to select candidates; do not report
the modeled times as measured device throughput.

## Measured 8-chip serving qualification

On 2026-11-09, Falcon runs `exp-zmhsr0wir1` and `exp-yarpu1tn5r` compared
epic `483575e23` with candidate `2d96daef8` on 8 physical v7x chips
(two hosts, DP16/EP16/TP1/PP1). Both used `[288,512,2048]`, 48 requests per
rank, CED/SWA128, DSpark K5, FP8 KV/indexer and W4A8 push. Each replay ran
45 minutes with a 10-minute ramp: 768 closed-loop slots, 453 recorded turns
from 16 DeepSWE trajectories, fresh generations at temperature 1/top-p .95.

| Metric | Three-bucket epic | Budget fix + strict interval 4 |
| --- | ---: | ---: |
| Steady generation tokens/s | 13,346.7 | 20,043.6 |
| Full replay generation tokens/s, including ramp | 12,274.1 | 18,626.0 |
| Mean TTFT (s) | 0.920 | 1.048 |
| Formal C768 median step (ms) | 36.017 | 35.861 |
| Encoder 2048 forward share | 47.86% | 23.02% |
| Decoder 2048 forward share | 1.51% | 3.87% |

The combined candidate gained 50.2% steady throughput and 51.7% including
ramp, with about 128 ms extra mean TTFT. Both completed with zero HTTP or
transport errors, preemptions or replay compilation matches on either host.
Input hashes, replay code, core packages, model configuration and context
limit matched. Actual CED forward cadence agreed with engine metrics;
these are serving observations, not isolated device-stage timings.

512 compiled in encoder head/tail and decoder, and the log confirmed push
MoE's 512 program. General Q-A/Q-B/compressor configurations and metadata
fallbacks remain. At max_model_len 524288, the `[48,4096]` page table also
exceeds main_records' 772-column limit on both arms. No measured-bucket
fusion gate was broadened. Both arms already had three buckets, so this
comparison does not isolate the speedup from adding 512.

Interval 8 (`exp-rbsacceqke`) did not form the required consecutive full-DP
decode window in the fixed 512-output burst: early requests completed
before all requests entered decode. The formal gate failed and its replay
did not start. Interval 4 passed the unchanged gate. This does not establish
interval 8's unmeasured long-running throughput or an optimal interval.

This is one matched serving replay, without actual tool execution or SWE
task-success scoring. Repetition heuristics were 0.80%/0.90%, with sampling
and right censoring. No EPLB installation occurred during either replay;
actual layout switches and memory-pressure recovery remain unqualified.
Commands, identities and raw summaries are in
[the validation record](https://github.com/cjx0709/optimize-trace/pull/41).

## Serving launcher

The existing server launcher accepts an explicit ladder on either host:

```sh
python scripts/dsv41_bench/server_steps/run.py \
  --scenario decode --input INPUT --out OUTPUT \
  --compile-sizes 288 512 2048 \
  --scheduler-cls vllm_torchtpu.core.tpu_scheduler.TpuPrefillWindowScheduler \
  --prefill-schedule-interval 4
```

This command starts a server and is separate from the offline checks. Adding
the option does not change the launcher's default ladder or the CI recipe.
The explicit class disables the automatic DP scheduler selector in the child
environment. Both scheduler options are emitted for either host role.
