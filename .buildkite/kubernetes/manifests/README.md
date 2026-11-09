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
The TP2/DP4 decode startup failed during model profiling with incompatible
`(24, 60, 128)` and `(24, 62, 128)` broadcast shapes, before any paired client ran;
DP4 is not native-qualified. Both engines must use the same topology and
`DISAGG_MODEL`.
The latter defaults to `Qwen/Qwen3.8-Flash-Next-FP8`; a resident checkpoint path
can be supplied for host tests. No model copy is part of the scripts.

| Setting | Prefill | Decode |
|---|---|---|
| PLE host backend | `torch` | `native_dma` |
| PLE offload / shared / across-DP | true / true / false | true / false / true |
| Maximum context / active slots | 262144 / 8 | 33792 / 16 |
| Compile sizes | `[8192]` | `[8, 8192]` |
| MoE | fused EP v2, sharded plan | dense threshold 128, fused decode off |

Both roles use block128 FP8 pages, FP32 convolution state and BF16 recurrent
state, CP1/PP1, async/chunked scheduling, and APC/MTP off. The model-owned
`Qwen4ExpRaidenConnector` transfers native cache groups with parallelism equal
to TP, independently of DP. Its loaded checkpoint and physical layout checks
remain mandatory; the Qwen3.5 admission and block8192 recipe do not apply.
Operator bootstrap factors are preserved. Role-local environment overrides
also set the PLE backend so an inherited `native_dma` cannot change prefill.

PLE table weights stay local to each engine and are not part of the transfer
fingerprint. Admission compares measured carrier geometry/layout/dtype, logical
cache shapes and dtypes, GDN encoding/granules, loaded FP8 KV scales and TP
mapping. It validates the live cache owner bindings independently. Local PLE
shards, backend/shared-memory choices and checkpoint paths may differ; incompatible
cache encodings, scales or storage bindings are rejected. Matching transfer
fingerprints do not establish model/checkpoint equivalence or numerical accuracy.
PD does not reject CP/PP/APC/MTP configuration switches alone. This whole-page
mapper requires the actual local TP rank set and one explicitly mapped committed
snapshot for each recurrent owner. Spare or historical state pages are not
silently selected or dropped, regardless of cache-mode/speculative-capacity flags.
Model/backend support and prefix-completion checks remain separate requirements.
The Qwen adapter currently declares whole-page copies to matching transfer-unit
ordinals. Cross-TP splitting/combining of packed QSA/GDN pages needs a fragment
mapping that this adapter does not supply. Shared Raiden's support for explicit
resharding spans does not infer that model-specific mapping automatically.
The DP8 two-host smoke and three 512-request performance rounds completed with
zero request failures. The separate exact diagnostic found PD outputs differing
from decode-local outputs; numerical accuracy remains unqualified. These
results do not qualify TP2/DP4 or other checkpoint/layout combinations.

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
