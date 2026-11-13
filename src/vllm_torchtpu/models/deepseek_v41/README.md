# DeepSeek-V4.1 (`models/deepseek_v41/`)

One model directory, one skeleton, two tracks (see "Approach for Experimental
Iteration, Validation and Graduation"). Customers run `vllm serve` with no
extra flags and get the SUPPORTED default plan; benchmarks opt into a named
EXPERIMENTAL profile.

| File | Track (owner) | What it holds |
| --- | --- | --- |
| `variants.py` | Supported (Support) | SUPPORTED implementations of each slot and their parameters |
| `plans.py` | Supported (Support) | Slot -> ordered SUPPORTED variant names (first eligible wins) |
| `validation/plan_snapshots/*.json` | Supported (Support) | What `plans.py` selects for each standard config; CPU CI diffs it |
| `experimental/variants.py` | Experimental | EXPERIMENTAL implementations, reachable only through a profile |
| `experimental/profiles.py` | Experimental | Named benchmark recipes that pin every slot |
| `plan.py` | Supported (Support) | Resolution, publication into the config, and the `param()` reader |

## Slots

| Slot | Default (`plans.py`) | Other variants | Read by |
| --- | --- | --- | --- |
| `csa2_attention` | `tuned` | `base` | `v2/csa2_adapter.py`, `v2/preprocess_ops.py` |
| `indexer` | `tuned` (issue depth 8) | `base` | `v2/indexer_fp8.py` -> dense FP8 scorer |
| `mhc` | `tuned` | `base` | `deepseek_v41_mhc_op.py` |
| `engram` | `tuned` | `base` | `common/engram.py`, `v2/model.py`, `v2/cache.py` |
| `ced` | `tuned` | `base` | `v2/model.py` |
| `dspark_draft` | `tuned` | `base` | `v2/dspark.py` |
| `decode_step` | `tuned` | `base` | runner, `layers/core/`, `v2/`, libtpu flag |
| `host_gc` | `tuned` (freeze, every 500) | `base` | `gc_policy.py` (runner and engine core) |
| `moe_kernel` | `fused_ep_v2` | `push_fused_ep` (experimental) | `layers/adapter/fused_moe_ep.py`, `mxfp4.py` |
| `expert_layout` | `checkpoint` | `dp16_c768`, `dp32_c768`, `dp8_c384_r1`, `dp8_c384_perm` | `config.py` -> `expert_placement.py` |

`base` is each slot's code path before the DP16 decode hillclimb; `tuned` is the
path it landed (#2838, #2839, #2868, #2875, #2876). Rolling a slot back is a
one-line change in `plans.py` plus its snapshot.

## Profiles

| Profile | Config | What it pins |
| --- | --- | --- |
| `dsv41-dp16-c768` | TP1/DP16/EP16 on a 2x2x2 v7x slice, `moe=fused_ep_w4a8`, block size 256 | every `tuned` slot, `moe_kernel=push_fused_ep`, `expert_layout=dp16_c768`: the hillclimb's 28.24 ms landing (aa4360d22, j-f164e4eb) |

```
vllm serve ... --additional-config \
  '{"deepseek_v41":{"execution":"ced","moe":"fused_ep_w4a8"},"model_plan":{"profile":"dsv41-dp16-c768"}}'
```

The env vars these slots replaced (`MOE_FUSED_EP_PUSH_*`,
`TPU_DEEPSEEK_V41_{CSA2,ENGRAM,MHC,...}_*`, `TPU_GC_POLICY`, ...) are gone;
setting one stops configuration with an error naming its slot
(`plan.REMOVED_ENV`). The resolved plan is published under
`additional_config["model_plan"]["resolved"]`, so it enters the compile-cache
hash and reaches every engine core and worker with the config.

## Rules

- Customer-path code never imports `experimental/`
  (`tests/models/deepseek_v41/test_model_plan.py` enforces it); `plan.py` loads it
  lazily, only for a selected profile.
- Shared code reads a slot with `model_plan.param(slot, key, default)`, where
  `default` keeps its behavior for models without a plan (default-off hooks).
- Graduating an experimental variant: move it into `variants.py` (and its kernel
  from `kernels/experimental/` to `kernels/` with `git mv`), flip `plans.py`,
  refresh the snapshot, with bitwise proof or a paired eval within 1 pp and a
  same-host A/B.
