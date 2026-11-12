# Workload manifests

Handed to the launcher with `--manifest`, for workloads the built-in one-pod
Job cannot express: several roles that talk to each other, or the hosts of one
multi-host slice.

Named for what they contain. `qwen3.5-397b-1p1d.yaml` serves that model in that
topology; `ray-multihost-slice.yaml` brings Ray up across a slice and runs
whatever command the step passes, so it is named for the bring-up rather than
any one benchmark.

`v7x/` predates this and is not referenced by any pipeline.

## What a manifest has to say

Only the hardware it wants, and what is genuinely its own:

- `nodeSelector` with the accelerator and topology, and a `google.com/tpu`
  count. Those three pick the queue, so they cannot come from anywhere else.
- Its containers, and `activeDeadlineSeconds` if the fleet's default is wrong
  for it.

Everything else comes from the fleet, through one annotation on the JobSet:

```yaml
metadata:
  annotations:
    tpu-ci.google.com/defaults: standard
```

That supplies the compilation and model caches with their mounts, the gcsfuse
sidecar settings, the TPU toleration, the service account, `restartPolicy`, the
TTL, the memory request for whichever host the pod lands on, and the retry rules
that let a run survive its node being repaired. It is defined once in ci-infra,
in `kueue/launcher/pod_defaults.yaml`.

The merge is additive, so a role can still declare a volume of its own — both
manifests here add a per-pod profile cache that shadows a subtree of the shared
one — or override anything it needs to differ on.

Roles that hold no chips inherit only the eviction policy: the caches are sized
from a TPU host's memory and a chipless role does not run on one, so it states
what it needs itself.

## Qwen3.8-Flash PD

`qwen3.8-flash-1p1d.yaml` uses the existing engine, proxy, benchmark and result
publisher. `DISAGG_DP_SIZE=8` is the default; `4` selects TP2/DP4 on the same
eight-rank host for manual validation. The nightly matrix enables only DP8.
TP2/DP4 remains a manual, native-unqualified recipe. Both engines must use the
same topology and `DISAGG_MODEL`.
The latter defaults to `Qwen/Qwen3.8-Flash-Next-FP8`; a resident checkpoint path
can be supplied for host tests. No model copy is part of the scripts.

| Setting | Prefill | Decode |
|---|---|---|
| PLE host backend | `torch` | `native_dma` |
| PLE offload / shared / across-DP | true / true / false | true / false / true |
| Maximum context / active slots | 262144 / 8 | 33792 / 16 |
| Compile sizes | `[8192]` | `[8, 8192]` |
| MoE | fused EP v2, sharded plan | dense threshold 128, fused decode off |

Both roles request block128 FP8 pages, FP32 convolution state and BF16 recurrent
state, CP1/PP1 and async/chunked scheduling. Native pool planning resolves the
actual manager-page and state-slot geometry, including optional MTP draft cache
groups and speculative recurrent scratch slots. The model-owned
`Qwen4ExpRaidenConnector` transfers native cache groups with parallelism equal
to TP, independently of DP or expert routing. Its physical layout checks remain
mandatory; the Qwen3.5 admission and block8192 recipe do not apply.
Operator bootstrap factors are preserved. Role-local environment overrides
also set the PLE backend so an inherited `native_dma` cannot change prefill.

PLE table weights stay local to each engine and are not part of the transfer
fingerprint. Admission compares measured carrier geometry/layout/dtype, logical
cache shapes, dtypes and encoding block sizes, GDN encoding/granules, loaded
FP8 KV scales and TP mapping. It validates the live cache owner bindings
independently. Local PLE
shards, backend/shared-memory choices and checkpoint paths may differ; incompatible
cache encodings, scales or storage bindings are rejected. Matching transfer
fingerprints do not establish model/checkpoint equivalence or numerical accuracy.
The model requires CP1/PP1; native MTP (`method="mtp"`) is supported on both
symmetric and asymmetric (`P k=1 -> D k=3`) roles under `mamba_cache_mode="none"`
or `"align"`. This whole-page mapper requires the actual local TP rank set and
one explicitly mapped committed snapshot for each recurrent owner. In aligned
mode the snapshot is selected at the published token position, excluding
historical checkpoints and trailing speculative scratch slots.
The Qwen adapter currently declares whole-page copies to matching transfer-unit
ordinals. Cross-TP splitting/combining of packed QSA/GDN pages needs a fragment
mapping that this adapter does not supply. Shared Raiden's support for explicit
resharding spans does not infer that model-specific mapping automatically.

`DISAGG_TEST_MODE=apc` selects the APC correctness lane, and
`DISAGG_TEST_MODE=mtp-apc` selects the asymmetric MTP (`P k=1 -> D k=3`) + APC
correctness lane with `TPU_DP_VOCAB_PARALLEL=1` and 1-token lookahead block
hashing. Both correctness modes enable APC with aligned Mamba state and Stage-3
prefix-aware loads; the performance lane keeps APC and MTP disabled. The correctness client uses the existing development RPC to read
the default DP engine's initialized TP-worker plans. Per-engine counters check
actual routing and hits on every DP rank. Qualification also requires native
supervisor receipts for loaded source and cache plans on all workers.
Development endpoints are enabled only in this test profile and must remain
on the isolated test network.

Local hits must align with the registered token-page spans. Only newly allocated
token suffix pages are transfer targets; committed GDN/PLE snapshots and private
raw rings transfer in full. Full local hits release the producer registration
without a receive. Hit lookup reconciles cacheable groups before making a
no-load decision.

The worker chooses native pool capacity from the actual HBM budget and the
upstream physical allocation plan, bounded by full-concurrency demand.
`max_num_seqs` remains a concurrency limit rather than a promise that every
request can hold `max_model_len` simultaneously. Existing scheduler admission
handles cache pressure. Explicit block-count overrides are honored or rejected
if they exceed the budget or cannot hold one maximum-length request. Once the
physical owners are allocated, later spec queries retain that exact capacity.

The client resets each role's cache separately and checks cold, both-warm,
prefill-only-warm, decode-only-warm and branched-prefix requests. Per-engine
local and external cache-hit counters establish routing and real reuse. Cold
and warm first greedy token IDs must match, with selected-token logprob absolute
difference at most 0.2, as in the Qwen3.5 PD divergence recipe. Interleaved prefixes
check state restoration. Repeated known-answer checks require exact normalized
answers and exercise multi-token continuation. Raw responses,
counter snapshots, resolved plans and failures are artifacts. These checks
require native execution; CPU contract tests alone do not qualify APC with PD.

The performance lane follows the 397B smoke (C8, 8192 input, 32 output, 8 requests),
then three 512-request measured rounds. It checks health, proxy, command status
and completed/failed accounting. It does not run a numeric-reference matrix.

The optional `check_pd_completions.py` diagnostic reports exact token IDs against
decode-local computation and repeats
each prompt on each requested DP rank. Numeric reference/repeat results remain
separate from performance checks; `--numeric-mode=strict` also fails on a numeric
difference. Report mode retains `numeric_reference_passed=false` or
`numeric_repeat_passed=false` and `diagnostic_failed=true` for differing outputs.
It does not qualify numerical accuracy. Lengths cover both sides of block128 and
8192 boundaries and a 32768-token input, always fitting the decode window. A
missing transfer handle is an error; so is an output without the expected
remote-computed prefix count. KV load failures fail the request rather than
recomputing. Native sender/receiver completion still requires engine receipts.
Raw request/response
and accounting failures are artifacts and fail the lane. Performance cells use
32768 input / 1024 output and concurrency DP times eight. The CPU benchmark
selects the upstream CPU platform explicitly, without uninstalling libtpu.

Print the exact role command/environment without starting an engine:

```bash
POD_IP=127.0.0.1 DISAGG_DP_SIZE=4 DISAGG_MODEL=/tmpfs/Qwen3.8-Flash-Next-FP8 \
  python3 .buildkite/kubernetes/disagg_qwen38.py prefill --print
```

Do not add a CPU request. It is a scheduling floor checked against the template
the autoscaler builds for the shape, and one large enough to matter stops the
pool building nodes at all.
