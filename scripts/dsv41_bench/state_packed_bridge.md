# State-packed production integration A/B

This is a compressor-region benchmark, **not E2E**: actual Torch custom op,
unchanged compressor, and both bank `copy_` operations. No attention/index timing.

## Production mode and source identity

`--production` compares the exact original preprocess/decode from commit
`f8ced378ae65af5a5c16c77695d0ad40795b80bf` against actual imported production modules.
The candidate is not AST-patched. The baseline is privately loaded with an isolated
op name and decode binding; no production globals are monkeypatched.

Frozen companions (byte hashes checked before either CPU or TPU execution):

- `state_packed_original_decode.py`: SHA256
  `45196d72d3b617228b7cb71709d6f692963564b7eb49e7b426eec1723e1840a0`
- `state_packed_original_preprocess.py`: SHA256
  `313c79a8f20eef7c9730c98dd316fe64c7c5ba90232dc78d4d82c2f03bd1e6be`

The original baseline uses JAX `_compress_native_metadata` for T != 288, not the
historical Torch-native metadata implementation. Without `--production`, the
old private two-edit candidate remains available against the frozen original.

Production dispatch adds only T96/192/240/256, runtime TPU7, hidden5120, ratio2,
and `select_config(...).token_block == T`. Metadata remains INT32, B48,
table48x8, starts49, kv_starts49, minimum48. The caller continues to pass the
intentional 288 support constant and enforces window128/capacity16/stride16/offset0.
T288's original body and combined-addresses path remain unchanged. Unsupported
new shapes/configurations retain the existing native metadata fallback.

The CPU audit executes actual guard expressions, checks 58 dispatch/metadata
cases plus runtime gates and 35 independent metadata fixtures. It also reverses
only the authorized guard/docstring edits and requires whole-module AST equality
with the original, proving the combined-addresses and T288 bodies unchanged.
CPU checks do not execute Pallas or establish accelerator parity/performance.

## Exact next-run overlay

Use base commit above. Overlay these paths from this worktree (no attention edits):

```text
src/vllm_torchtpu/kernels/experimental/decode_preparation.py
src/vllm_torchtpu/models/vllm/deepseek_v41/v2/preprocess_ops.py
scripts/dsv41_bench/state_packed_bridge.py
scripts/dsv41_bench/state_packed_guards.py
scripts/dsv41_bench/state_packed_original_decode.py
scripts/dsv41_bench/state_packed_original_preprocess.py
scripts/dsv41_bench/state_packed_consumer.py
scripts/dsv41_bench/state_packed_shapes.py
```

Existing tracked dependencies come from that base: `attention_integration.py`,
`attention_consumer.py`, `attention_state.py`, `compact_baseline.py`, and their
normal kernel imports. Standalone shape/consumer exploration now loads the
integrated packed kernel without guard rewriting; their CPU entrypoints remain
usable. Use bridge production mode for the actual Torch production A/B.

From repository root, in the existing provisioned TPU7 runtime:

```bash
python scripts/dsv41_bench/state_packed_bridge.py --production --cpu-test
TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONNOUSERSITE=1 \
/tmp/attention-venv/bin/python scripts/dsv41_bench/state_packed_bridge.py \
  --production --out /tmp/state-packed-production --samples 100 --profile
```

No jobs are launched by this handoff. Runtime pins remain torch2.13.0,
torch-tpu0.2.0rc1, jax/jaxlib0.11.2, libtpu0.0.50, numpy2.3.5.
The runner retains explicit backend loader invocation, fullgraph/static Torch
compile, donation (9,10), fresh materialized correctness owners, returned-owner
chaining and synchronized ABBA timing. Dynamo recompile_limit remains 16.
Seven fixture distributions times five T values compare complete outputs and
both full 512-record banks bit-for-bit, then check independent metadata/dense
oracles. Uniform timing asserts no cross-call read/write overlap. Ten profiles
contain sixteen synchronized region calls each. JSON records source hashes,
runtime versions, CPU audit, correctness and raw timings.

The earlier private candidate exp-f1ur4hx9go passed 35 actual-Torch A/B+oracle
cases; supplied device medians (original -> private, us): T96 28.8 -> 23.248,
T192 35.705 -> 29.9585, T240 37.333 -> 31.7275, T256 40.6535 -> 34.9385,
T288 35.191 -> 35.1875 with identical fingerprint 9694248204276475233.
These are prior private-candidate evidence.

## Accepted integrated production evidence

Production-mode experiment `exp-jutj1mr9zg`, analysis `an-jkfctywst8`, passed
all **35 actual-Torch A/B + independent-oracle cases**. Device analysis is archived
as `state-bridge-device-r8.json`. Results supplied by the parent experiment owner:

| T | Original device median (us) | Integrated device median (us) | Original synchronized median (us) | Integrated synchronized median (us) |
| --- | ---: | ---: | ---: | ---: |
| 96 | 28.972 | 23.4595 | 417.8485 | 409.8585 |
| 192 | 35.7605 | 30.141 | 417.3475 | 409.694 |
| 240 | 37.5395 | 31.928 | 434.744 | 429.4235 |
| 256 | 40.883 | 35.204 | 459.157 | 451.873 |
| 288 | 35.384 | 35.279 | 487.3625 | 484.729 |

T288 retains identical fingerprint **9694248204276475233**. Synchronized medians
have 200 samples per arm per T and are host-dominated. Device measurements cover
the compressor region, not E2E. No production sources changed after this run;
subsequent edits only repair historical benchmark loading and record evidence.

CI7610 has TPU parity/eval failures and logs returned 401; **not all CI green**.
