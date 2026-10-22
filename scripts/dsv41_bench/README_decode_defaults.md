<!-- SPDX-License-Identifier: Apache-2.0 -->

# Selected DSv4.1 decode defaults

This branch starts at `dsv41` revision
`bdb38700dd25959c029c7d77f00cee91ccd3a0bc`. It contains FP4 height and indexer
tuning, mHC output placement, and calibrated expert placement. Each selects its measured
configuration automatically. Other configurations keep their existing kernels
or native expert order.

| Component | Default selection | Original implementation |
| --- | --- | --- |
| FP4 MoE height ladder | Two heights for the measured TPU7x EP8 decode geometry; four-height fallback | `c6ffd0fa` |
| Full-context indexer | Joint 16-request groups / 32768 scorer window for the measured 48-request, six-query, 65536-capacity geometry | `cd47b6c9` |
| mHC normalized output | Keep the normalized BF16 value in VMEM for the supported TPU7x `(288, 4, 5120)` shape | `7bf96de9` |
| Expert placement | Built-in permutations for 30 main-model layers on the calibrated FP4 TP1/DP8 topology | `d1cecfb3`, plus automatic configuration resolution |

Router tile tuning, WO_A tuning, selector/map fusion, communication-threshold
changes and rejected kernel prototypes are excluded. The FP4, indexer and mHC
kernel files are identical to their original measured implementations.

## Automatic expert placement

The existing `fused_ep_w4a8` backend selects the built-in map when all of these
conditions hold:

- TPU v7, TP1, DP8, PCP1 and PP1, with expert parallelism enabled.
- 40 layers, 384 routed experts, hidden size 5120, MoE intermediate size 2304,
  and top-k six. DSV4.1 uses learned routing in all MoE layers.
- Linear expert placement, EPLB disabled, `runai_streamer` loading and
  `--enable-ep-weight-filter` enabled.

No external permutation JSON or extra enabling flag is required. The existing
benchmark launcher's typed fused-EP policy therefore enables placement on this
configuration. GMM and other loading/topology configurations retain native order.

Omitting `additional_config.deepseek_v41.expert_permutations` requests automatic
selection. Set it explicitly to `null` or `{}` to disable placement:

```json
{
  "deepseek_v41": {
    "execution": "ced",
    "moe": "fused_ep_w4a8",
    "expert_permutations": null
  }
}
```

A nonempty map remains an explicit custom placement and is validated before
loading. Resolution freezes the chosen map, including an empty result, into
the canonical configuration before worker copies and config hashing. A worker
whose DP config is later collapsed to one therefore retains the same placement.

The loader fetches original checkpoint byte ranges, filters by physical expert
ownership and yields the corresponding physical names. Gate scores, correction
biases and top-k selection keep logical expert order; the replicated map changes
only the expert IDs sent to dispatch. Draft layers retain their native placement.
The built-in table preserves every value of the earlier calibrated JSON
(SHA-256 `8cf1d65275f04e2a142edf6f00b295604c6da5c9522a4f4a8f4f13ddfb7bcb7b`).

The placement resolver does not read `num_hash_layers`: that field belongs to
DSV4, while upstream DSV4.1 explicitly constructs its MoE with zero hash layers.
The serving-policy tests use the upstream HF config class with nested
`text_config` and `quantization_config` to cover the actual checkpoint schema.

## Validation and limits

The extracted branch was checked in Falcon `exp-kcu0tqsle9` /
`art-cnup3hxakj`, using the base revision above plus source patch SHA-256
`8526ca1fc3ff92a6f0b69fd2fe3d619effb0c779b6964bcfc54bd5318d093a7a`.
All 315 configuration, loader, adapter, communicator and routing-table tests
passed in the pinned runtime. The tests cover automatic selection, explicit
disablement, incompatible layouts, worker serialization, all 11520 calibrated
destinations and the replicated map operand. Full-environment pyrefly reports
57 existing errors on the base and 54 on this branch, with no new errors.
Changed-file pre-commit checks pass.

Falcon `exp-wkw0p48cx4` additionally reproduced the missing-`num_hash_layers`
startup error before the correction. With the correction, all 315 tests pass
using the upstream HF config class, and `EngineArgs.create_engine_config()`
succeeds with the real checkpoint config and enables all 30 placement layers.
Full-environment pyrefly reports the same 54 existing errors and no new errors.
This checks configuration startup without loading model weights.

Fresh TPU checks also pass: FP4 automatic dispatch matches the four-height
reference at seed 1140, and all eight mHC consumer-chain outputs match at seed
1141. The final FP4 benchmark compares uint32 bit patterns, including signed
zero, and was rerun after tightening that assertion. The evidence retains both
the original source patch and the final benchmark script. Reproduce with:

```bash
python scripts/dsv41_bench/bench_fp4_moe_heights.py \
  --out /tmp/fp4-defaults --seed 1140 --profile
git show bdb38700:src/vllm_torchtpu/kernels/deepseek_v41/mhc/shifted_post_pre_rms.py \
  > /tmp/mhc-reference.py
python scripts/dsv41_bench/bench_mhc_output.py \
  --out /tmp/mhc-defaults --reference-source /tmp/mhc-reference.py \
  --seed 1141 --profile
```

Earlier paired kernel checks matched their reference outputs. Isolated results
were 0.308 to 0.292 ms for the FP4 kernel, 0.741 to 0.680 ms for shuffled-page
index score/selection, and 27.7 to 26.6 microseconds for the mHC consumer chain.
The indexer adds about 12 MiB of temporary storage to that isolated executable.

The earlier placement serving comparison (`exp-gjb1967wya`) measured forward
42.608 to 41.635 ms, fixed-acceptance throughput 6318–6375 to 6620–6646 output
tokens/physical-chip/s, and startup HBM 48.98–49.04 to 49.01–49.04 GiB/logical
device. That comparison included Router and WO_A changes which this branch
excludes; it is not an end-to-end measurement of this new combination.

Placement-only GSM8K checks (`exp-cxneias4qy`) measured 248/256 in both baseline
repetitions and 248/256, 249/256 with placement, with no failed or truncated
requests. They show no decrease on that sample, but do not establish equivalence.
The placement was calibrated with C384 / 96K prefix / DSpark K5 decode routes;
performance on other request distributions, prefill and mixed batches is unmeasured.

The selected combination still needs a full-serving benchmark. No 35 ms forward
or 14000 output tokens/chip/s claim is made for this branch.
