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

Right-censored requests have unknown output lengths. Missing slots are filled
from observed same-rank samples. `--match-output-mean` rescales those samples
using aggregate output/arrival counts; `--calibrate-arrivals` fits the scale
to the historical per-rank arrival rate per engine step. This is a surrogate
workload, not recovery of missing request histories. The calibration history
and all assumptions are saved in the result.

The model does not run the vLLM allocator, async execution queue or sampling.
The companion CPU tests exercise the real pinned AsyncScheduler separately,
including continued chunks, an open drain window, and the K5 budget backport.

512 stage costs and the encoder/decoder cost split are unknown. Timing
sensitivity varies them explicitly and fits the old 2048 cost to aggregate
replay timing at its measured DP size. Without a matching cost reference,
cross-DP timing sensitivity is omitted. Historical DP16 costs applied to DP8
are hypothetical. Use
bucket counts, occupancy and sensitivity to select candidates; do not report
the modeled times as measured device throughput.

## Future serving check

The existing server launcher accepts an explicit ladder on either host:

```sh
python scripts/dsv41_bench/server_steps/run.py \
  --scenario decode --input INPUT --out OUTPUT \
  --compile-sizes 288 512 2048
```

This command starts a server and is separate from the offline checks. Adding
the option does not change the launcher's default ladder or the CI recipe.
