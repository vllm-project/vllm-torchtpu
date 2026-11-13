# Qwen3.5 candidate4 megakernel levers on the library layout

This branch carries, on top of `nanwei/mk-dev` (at `c297d9a45`) plus PD
support (see `PD_SUPPORT.md`), the megakernel changes our Qwen3.5 PD submission ran
between candidate30 and candidate4 (`5a1c82686`), in the library's form:
every lever is a default-off tile field, module tunable or plugin option,
and a new profile, `qwen35-candidate4`, turns on the ones the submission's
decode legs used. It is a hand-off branch, not a PR: take it apart into the
megakernel lanes as you see fit.

The code exactly as it ran (candidate30 plus candidate4's megakernel paths,
under the original authors) is the branch
`shriharim/qwen35-pd/mk-cand4-on-c30`. The non-megakernel parts of the
submission land on `main` through separate PRs and are not here.

## Why

Four of the five configurations of the submission (points #5, #8, #10, #12)
are TP8 + DCP4 decode legs that serve every MTP verify step on the
megakernel and every draft step on the MTP draft megakernel:

| Point | mode | running cap | verify rows reached | verify buckets | draft verify buckets | extra |
|---|---|---|---|---|---|---|
| #5 | ep | 16 | 64 | 4..64 step 4 | 4,8,12,16,20,24,32,40,48,56,64 | — |
| #8 | ep | 8 | 32 | 4..32 step 4 | same | — |
| #10 | tp | 3 | 12 | 4, 8, 12 | 4, 8, 12 | — |
| #12 | tp | 1 | 4 | 4 | 4,8,...,64 (11) | shared expert after the first expert DMAs |

No point reaches a verify bucket above 64 rows, which the template's
64-row verify limit already covers.

## Selecting it

No environment variables. On a decode leg:

```
--mamba-ssm-cache-dtype bfloat16
--additional-config '{"megakernel": {"plugin": "vllm_torchtpu.models.qwen3_5.megakernel",
  "options": {"profile": "qwen35-candidate4", "mode": "ep", "buckets": "0",
              "verify_buckets": "4,8,12,16,20,24,28,32",
              "mtp_draft": true, "mtp_draft_verify_buckets": "4,8,12,16,20,24,32,40,48,56,64"}}}'
```

(the #8 values). Per point:

| Point | `profile` | `mode` | `buckets` | `verify_buckets` | `mtp_draft_verify_buckets` |
|---|---|---|---|---|---|
| #5 | `qwen35-candidate4` | `ep` | `0` | `4,8,...,64` (16) | `4,8,12,16,20,24,32,40,48,56,64` |
| #8 | `qwen35-candidate4` | `ep` | `0` | `4,8,...,32` (8) | same |
| #10 | `qwen35-candidate4` | `tp` | `0` | `4,8,12` | `4,8,12` |
| #12 | `qwen35-candidate4-sh-at2` | `tp` | `0` | `4` | `4,8,12,16,20,24,32,40,48,56,64` |

The serving tree's environment keys map as follows: `VLLM_TPU_QWEN35_MEGAKERNEL`
→ the `megakernel` block, `..._MTP_MK` → `mtp_draft`, `..._MK_MODE` → `mode`,
`..._MK_BUCKETS` → `buckets`, `..._MK_VERIFY_BUCKETS` → `verify_buckets`
(capped at 64 rows), `..._MTP_MK_VERIFY_BUCKETS` → `mtp_draft_verify_buckets`,
`..._MK_SH_AT=2` → the `-sh-at2` profile, `..._MK_FUSED_POOL_IO_MAX_B=160` →
the profile's `fused_pool_io=True`. The wide-row keys (`SPLIT_WIDE`,
`MAX_SINGLE_B`, `RSAG_WIDE`, `WIDE_MB`, `WIDE_MIN_ROWS`) and `HRSAG` have no
effect at these points (see "Deferred"); `DCP_KV_HEADROOM` and `ZERO_COPY` are
dropped.

## The levers

Every option defaults off (`hier1_peers` defaults to the current `(2, 4, 6)`),
and the module or collective that owns it engages it only where candidate4's
own gate holds. "Class" is the option's `NUMERICS` entry: `exact` options
have a bit-identity flip in `EXACT_FLIPS`.

| Option | What it does | candidate4 source | Class | Where |
|---|---|---|---|---|
| `Tiles.hier1_peers` | `hier=1` cross-chip push to the other core parity `(3, 5, 7)` | `72ac041fa` (Haibo Huang) | approx (bit-identical in its 8-rank test) | `core/collectives` |
| `Tiles.hx` | `hier=2`: 4-chip RS + AG of the half-sum instead of the 4-chip push (from 48 rows) | `2f16fe452` | approx (bit-identical in its 8-rank test) | `core/collectives` |
| `Tiles.dwait_hoist` | one ring-bank computation per dense wait group | `961aa46c1` (Jordan Dotzel) | exact | `core/dma/stream.py` |
| `Tiles.arpipe`, `arpipe_h1` | defer the layer's last all-reduce round into the next layer's input projection; split input norm (fused verify buckets of the main kernel) | `961aa46c1` | approx (on TPU, off vs on: about 1–2% rms relative on the hidden rows over 8 layers, bf16 rounding level) | `core/kernel.py`, `core/collectives`, GQA / GDN call sites |
| `Tiles.ar0s` | start round-0 sends from inside the producing output projection | `961aa46c1` | approx in the registry; measured exact (bit-identical in its 8-rank test, and on TPU off vs on) | same |
| `moe.flush_in_ar1` | dense-ring flush inside the following all-reduce (TP / PIPE buckets) | `961aa46c1` | exact | `core/kernel.py` (`deferred_fill`), MoE |
| `gqa.wgroup` | grouped `att_in` ring waits | `961aa46c1` | exact | GQA |
| `gqa.gdefer`, `gqa.kvdefer` | `att_in` gate (and K / V) columns after the Q exchange starts | `961aa46c1`, guard from `f596f8587` | exact | GQA |
| `gqa.qswap` | RoPE partner lanes by lane rolls | `961aa46c1` | exact | GQA |
| `gdn.lcv` | conv window assembled in VMEM on the fused path | `961aa46c1` | approx (1 bf16 ULP on 2 recurrent-state bytes at T=4 on CPU, bitwise at T=1; on TPU, off vs on: about 1–2% rms relative on the hidden rows over 8 layers, bf16 rounding level) | `core/experimental/fused_pool_io.py` |
| `gdn.mxu_dr` | delta-rule update on the MXU (profile bucket table) | `961aa46c1`, table `84835ce1e` | approx (on TPU, off vs on: about 1–2% rms relative on the hidden rows over 8 layers, bf16 rounding level) | GDN fused path |
| `gdn.zdefer` | `lin_in` z-column chunks deferred into the recurrence | `961aa46c1` | approx (on TPU, off vs on: about 1–2% rms relative on the hidden rows over 8 layers, bf16 rounding level) | GDN fused path |
| `moe.route_shrink` | copy slot packed above the expert id (shorter scalar chain) | `961aa46c1` | exact | MoE |
| `moe.cp_interleave` | rank-minor copy-slot scan (Po2C is greedy, so it can change which rank serves a copy) | `961aa46c1` | approx | MoE |
| `moe.prim_now` | fixed-hit compaction stores no worklist entries | `961aa46c1` | exact | MoE |
| `moe.route_early` | routing weights under the hit-list copy; early expert refills | `961aa46c1` | exact | MoE |
| `moe.sh_pre`, `moe.sh2_pre_route` | shared expert before routing | `961aa46c1` | exact | MoE |
| `moe.flush_pre_drain` | EP: dense-ring flush before the last expert's down projection | `961aa46c1` | exact | MoE |
| `moe.sh_at` | shared expert inside / after routing (`-sh-at2` profile, ≤ 8 rows) | `0c1ced0f9` | exact | MoE |
| `moe.ep_fast_refill` | EP expert refills with static banks | `f596f8587` | exact | MoE |
| profile rule | fused-bucket dense ring depths 32..64 rows; 14-slot ring for attention-only stacks | `961aa46c1`, `f596f8587` | (`nbd` is exact) | `profiles/candidate4.py` |
| `mtp_draft*` options, `attach_draft` | the MTP draft layer on an attention-only pooled step | `2ec61abf8`, `8aea021d8` (Haibo Huang), `ad3ce556e` (Jordan Dotzel), `9ba9e6bb7` (Benliang Wang), `1d850bd67` | approx | `draft.py`, adapter, driver |

`exact` classes are backed by a bit-identity flip in `EXACT_FLIPS`, which runs
at R=2. `hier1_peers`, `hx` and `ar0s` only act at R=8: they are bit-identical
in their own 8-rank tests but stay `approx` until a flip harness runs at R=8.
On TPU (the port at `qwen35-candidate4` with one option off against all on),
`ar0s` off is bit-identical, so it could be reclassified `exact`; `lcv`,
`zdefer`, `mxu_dr` and `arpipe` are each engaged and each change the hidden
rows by about 1–2% rms relative over 8 layers, which is bf16 rounding, not an
error, but none of them is exact.

## The MTP draft megakernel

`draft.py` builds the MTP draft layer (one GQA + MoE layer) on an
attention-only pooled step of the library, and the Qwen3.5 program attaches
it to the drafter through the contract's `attach_draft` (the driver calls it
before the bind agreement; `on_warmup_done` checks it is warm). Options:
`mtp_draft` (off unless selected), `mtp_draft_buckets`,
`mtp_draft_verify_buckets`.

The drafter side (the seam in `spec_decode/eagle3.py`, `mtp_fused_loop.py`,
`mtp_loop_model.py`, the Qwen3.5 routing in `models/vllm/qwen3_next_patch.py`,
the step-0 eligibility memo, and the runner merge resolution that keeps
sequence parallelism off on megakernel steps) is not on this branch: the
draft path it hooks into lands on `main` through our spec-decode PRs, not
through `mk-dev`. It is a small patch series for our serving stack, delivered
next to this branch, under the original authors.

## Proposals for your review

**Shared (Layer 1) changes this branch makes**, all default-off and with the
traced default path unchanged: `Module.layer_schedule`, `CorePlan.schedule`
(and `info["schedule"]`), the `KernelContext` fields `layer_hook`,
`arp_scale`, `ar0` and `deferred_fill`, `TwoRing.claim` / `dtake_group` /
`dbanks_wait` / `arp_mm` / `dmatmul(hook=, split=, ws_pre=)` and deferred
segment flushes in `core/dma/stream.py`, the all-reduce changes in
`core/collectives/` (`hier1_peers`, `hx`, the deferred round, round-0 sends
from a column split, `fill=`; the schedule helpers sit in `ring_schedule.py`
and the matmul hand-off types in `hooks.py`, which `dma.stream` imports), and
`TileContext.fused_pool_io`.

- **Layer-loop schedule options (ARPIPE, AR0S, FLUSH_IN_AR1).** These three
  change the generic builder's per-layer schedule, so they touch your layer
  loop in `core/kernel.py`, not only a module:
  - `arpipe` defers the second round of the MoE all-reduce into the next
    layer's input projection: the input norm runs in two lane halves, the
    projection sums the own half's K tiles first, waits for the round in a
    hook between the halves, then sums the other half;
  - `ar0s` starts the round-0 sends of the mixer's all-reduce from inside its
    output projection, one column group at a time;
  - `flush_in_ar1` issues the MoE dense-ring refill inside the following
    all-reduce instead of at the end of the MoE segment.

  They are written as default-off tile / tunable options plus optional
  `KernelContext` fields that the builder sets only when they are on;
  modules that do not read those fields are unaffected, and with the
  options off the traced kernel is identical to the base's. The ring and
  collective pieces they use (`TwoRing.claim`, `dtake_group`,
  `dbanks_wait`, `arp_mm`, `dmatmul(hook=, split=)`, deferred segment
  flushes) are separate commits and usable on their own. Please review the
  layer-loop change as a proposal: if you would rather keep the generic
  loop as it is, the alternative is a Qwen3.5 outer loop in this folder.
- **New all-reduce modes.** `hx` and the `hier=1` peers `(3, 5, 7)` are
  collective modes: shared changes for the `collectives/` owners.
- **A draft decline declines the whole megakernel.** The Qwen3.5 program's
  draft attach votes `draft_declined` in the bind agreement, as your
  contract's code does, so a rank whose draft program cannot serve takes
  the main megakernel down on every rank. Candidate4 kept the main
  megakernel and dropped only the draft. If you prefer that, the draft needs
  its own agreement round.
- **The draft uses `MoEDebug.row_mask` in serving.** The draft step routes
  only the gathered rows through the experts, which is the debug row mask;
  it should become a serving feature if you keep this design.
- **The draft drops one bind check.** Candidate4 checked that the draft layer's
  raw KV pool is the tensor the runner allocated; `RunnerView` does not
  expose `kv_cache_raw_tensors`, so that identity check is not done.
- **Two facts the serving glue reads that the contract does not carry:** the
  runner keeps sequence parallelism off once a TP-mode program has loaded,
  and reads `program.mode == "tp"` for it; and the drafter tags its
  megakernel programs with a source hash of the library and the draft
  package (computed in `eagle3.py`, because the compile cache key does not
  cover megakernel source). Both would be cleaner as contract facts.
- **`TileContext.fused_pool_io`.** `bucket_exec_spec` passes the bucket's
  fused pool I/O decision to the tile rule, so a rule can tune fused buckets.
- **A note on `compiler_stats`.** Its normalizer masks `line=N` in the HLO
  location tables, but the `\bline=` pattern does not match inside
  `end_line=` (and `column=` / `end_column=` are not masked), so any edit
  above a traced frame shows up as a top-level program (TLP) difference. On
  this branch `kernel.py`'s `unpack_local_weight` moved from line 484 to 596,
  and that location entry is the only content difference in the compiled
  code against the base. Masking those fields too would make the gate
  independent of line shifts. Two more things showed up in the full
  matrix: it canonicalizes fusion names (`ge.17` and `ge.18` both become
  `ge`) and compares the DOT order graphs as text, so a swap in emission
  order reads as a difference; and it hardcodes `e_slots = E / R` for TP
  (`compiler_stats.py` lines 309 and 724), a value this mk-dev rejects, so
  TP configurations don't compile under it.

## Where the port deviates from candidate4

- ARPIPE_H1 and the `hier=1` AR0S engage only together with ARPIPE.
  Candidate4 engaged them on ARPIPE's common gate alone, which only differs
  in its `skip={"dense"}` ablation, where it sent half a payload; serving is
  unchanged.
- On a 2-rank butterfly all-reduce the deferred refill (`fill=`) is issued in
  round 0 (candidate4 has no such configuration).
- `DMM_UNROLL_OFF_B` (an empty table in candidate4, so a no-op) is not
  ported.
- The draft decline rule (see above).

## Deferred

| Lever | Why not here |
|---|---|
| 2-microbatch split verify buckets 68..128 rows | above 64 rows (no point reaches it) |
| single-launch B=80 and Split(80+r) | above 64 rows |
| 80-row microbatch split up to 800 rows | above 64 rows |
| wide megakernel step (MoE streams each hit expert once across microbatches), its all-reduce slot alternation and row floor | off at every point |
| single-call verify to 112 rows, flat RS+AG with deeper rings at 72..112 rows, ring depths above 64 rows | above 64 rows |
| hierarchical rsag (`hr`) | applies only to rsag + hier=0 buckets, which at R=8 are 72..112 rows |
| MXU delta-rule bucket entries 80..128 | above 64 rows |
| flat all-reduce off the 8-rank mesh | R=8 only |
| TP16 | all points TP8 |
| GDN checkpoint slot at the vLLM block stride | no effect at block 640; the template declines other strides by layout facts |
| persistent JAX compilation cache from `VLLM_CACHE_ROOT` | a deployment setting (an `os.environ` write in a kernel module), not a kernel option |
| DCP KV-budget headroom knob, zero-copy pool rebind | set to their defaults everywhere |

**Dead in candidate4:** `GDN_ZDEFER_ROLL`, `GDN_ZDEFER_ROLL_BUCKETS` and
`GDN_SEQ_ROLL_MIN_B` are assigned in its build context and read nowhere.

## Validation

On CPU in the CI image this tree pins (`vllm-torchtpu:c350823af…-v0.33.0`),
kernels in Pallas interpret mode, unless a bullet says TPU. The first TPU check
has run; the rest of the TPU checklist is at the end.

- **The default path is unchanged.** With no `qwen35-candidate4*` profile,
  the traced pooled step at R=8 is identical to `PD_SUPPORT.md`'s base in all
  50 configurations that trace: 8 on explicit tiles (split and fused pool
  I/O, T=1 and T=4, interpret and TPU lowering) and 42 on the resolved tiles
  of `qwen35-candidate30`, `-candidate30-fused` and `-baseline` (T=4 at 4, 8,
  16 and 32 rows and T=1 at 8 rows, split and fused, and DCP 4 at 8 and 32
  rows). Six more (64 rows) fail identically on both trees at the small test
  geometry. Source locations and object addresses are masked; nothing else.
  A second check, 24 configurations, is identical too: 8 with expert copy
  slots (vectorized routing, 4 copies per rank; split and fused; T=1 and
  T=4; interpret and TPU lowering), 8 on the tiny geometry, and 8 on the
  `qwen35-candidate30` and `-candidate30-fused` tiles at T=4 (8 and 32 rows,
  split and fused). It is sensitive: on a tree without the
  `cp_interleave`-off trace order in `moe/topk.py`, exactly the 8
  configurations with expert copy slots differ.
- **Compiled code at the default profiles.** AOT for tpu7x on CPU with
  mk-dev's `compiler_stats`. In one configuration (T=4, 4 rows, EP with 72
  expert slots, `qwen35-candidate30-fused`) the megakernel's full LLO
  (every pass) and its Mosaic are identical to the base's, VMEM, SMEM and
  bundle counts are unchanged, and the top-level program differs only in
  one source line number (see the `compiler_stats` note above); the same
  holds after merging mk-dev `c297d9a45` into both the base and this
  branch. Over the full matrix of 34 default configurations (this branch
  before the merge against the base), in all 32 that compile the
  megakernel compiles to identical code (full normalized LLO, final
  bundles and bundle counts, Mosaic, VMEM and SMEM), and the top-level
  program is identical too once the location table's line and column
  numbers are masked. The 17 that fail the raw gate differ only in
  emission order: the function-order debug graphs print identical edge
  sets in a different order, and two small XLA fusions whose canonical
  names collide (`ge.17` / `ge.18`, `reshape.89` / `.92`) are identical by
  raw id but emitted in swapped order. Raw dumps of both sides were kept
  for 9 of the 17; for the rest the evidence is the per-pass hash pattern.
  The 2 that don't compile are TP configurations, failing identically on
  both trees (see the `compiler_stats` note).
- **The profile reproduces candidate4's schedule.** Its tile rule was compared
  with candidate4's own `default_tiles` at the real geometry over the 4..64
  verify buckets, EP and TP (only the fused-bucket ring depths and `hx`
  differ from candidate30, and the rule reproduces both), and its bucket
  tables with candidate4's constants. Nine mutants of the rule (a ring depth,
  the `hx` threshold, the MXU and ARPIPE tables, the peers, a dropped flag,
  the fused gate, the `sh_at` row limit, the attention-only ring) each fail
  the test.
- **Each option:** its own unit tests; `exact` options flip bit-identically
  (`EXACT_FLIPS`); the approx ones state their band in their tests.
- **Mutation checks.** Each breaks the code on purpose and checks that the
  named tests then fail; the same tests pass on the unbroken tree first.
  - Module options, 16 mutants: 14 fail their tests. 2 survive,
    `prim_now_slot` and `sh_pre_tile`: the tests at those ids do not reach
    the mutated path. `sh_pre_tile` (every shared gate|up K tile taken from
    tile 0's weights) is a no-op at the tests' geometry, where the shared
    gate|up is one K tile (`[256, 256]` fp8, 64 KiB; `Tiles.bank` cannot go
    below a 128-row tile of the widest dense weight there). At the real
    geometry (K = 4096, 128 KiB bank) it spans at least two tiles, and the
    TPU comparison below covers that sum wherever `sh_pre` engaged (which
    buckets engaged it was not recorded). `prim_now_slot` (fixed hits reading
    their slot from the unwritten `wlist`) most likely survives because the
    tests' weights send almost every pick to experts that have copies, so
    there are no fixed hits (inferred, not measured); `prim_now` also has no
    on-chip check (see below).
  - The profile rule: 9 of 9 fail (above).
  - Collectives: the AR0S payload mutant (one column group's payload packed
    from the other group's half) fails
    `test_ar0s_bit_identical_to_plain_all_reduce` on this tree.
- **Test fakes brought up to date (test-only commits).** The module tests'
  `MiniCtx` and stream fakes now accept the layer-schedule inputs
  (`layer_hook`, `ar0`, `arp_scale`, `deferred_fill`; `hook=`, `split=`,
  `defer_flush=`) and assert they are off; without that, a module emit
  under them failed (test_gqa, test_gqa_pallas_core). `test_b8_knobs`'
  `test_b7_deferred_minors` fails on the PD-support base (`AttributeError`,
  measured; it passes here): its fake context has no `me`,
  which PD support's DCP change reads (`dcp_rank(ctx.me, …)` in
  `gqa_fused_init_buffers`); the fake now carries a TP rank.
- **Against candidate4's all-reduce: bit-identical, 34 of 34 cases.** The
  library's `AllReduceRing` and candidate4's `PallasAllReduce` were run on the
  same 8-rank inputs in interpret mode: `hier=1` with both peer sets, `hier=2`
  with and without `hx`, `hier=3`, and flat and reduce-scatter `hier=0`, on
  unfolded and folded payloads of 4, 3 and 16 rows. This is the one
  cross-tree CPU result, and it covers the all-reduce options only.
- **Against candidate4's whole kernel, on TPU: bit-identical.** One process
  on a v7x-8, interleaved, same weights and pools, on-chip output bytes
  compared. With `qwen35-candidate4` the port's output equals candidate4's
  kernel's in 10 of 10 main buckets (EP at 4, 8, 12, 16, 32, 48 and 64 rows;
  TP at 4 rows with `sh_at` 2, 8 and 12) and in 7 of 7 draft buckets (DCP 4;
  T=4 at 4, 8, 16 and 32 rows; T=1 at 4, 8 and 16). With
  `qwen35-candidate30` it equals candidate30's kernel in 2 of 2 (EP at 8 and
  32 rows). Step time, port over candidate4: 0.993 to 1.011. This ran on an
  earlier assembly of this branch, which differs from it in the
  `cp_interleave`-off trace order in `moe/topk.py` and in the mk-dev merge;
  at `qwen35-candidate4` its compiled code is identical to this tree's
  (`compiler_stats` EQUAL at T=4, 4 rows, EP with 72 expert slots, before
  and after the merge; the one configuration compared).
  (The same comparison on CPU did not finish compiling one interpret-mode
  kernel in 25 minutes, so there is no CPU cross-tree result for the whole
  kernel.)
- **Not checked on chip: the copy-routing options.** The TPU comparison ran
  with no expert copy slots (`e_slots` 0), so `route_shrink`, `prim_now` and
  `cp_interleave` have their unit tests and mutation checks but no on-chip
  equivalence check against candidate4.
- **mk-dev's suite.** On the merged tree: the adapter tests (26 files, 503
  passed; `test_jax_bridge_tpu.py` needs a TPU and fails the same way on the
  base), the Qwen3.5 template tests (the 17 files under
  `tests/models/qwen3_5/megakernel` and the template's `test_cpu_e2e.py`,
  232 passed), `test_pool_gather`, `test_pool_scatter`, `test_runtime_api`
  and `test_b7_deferred_minors` (63 passed), and the MoE tests `test_moe`,
  `test_moe_routing_ring` and `test_moe_cand4` (49 passed) all pass. Before
  the merge, with the same megakernel code apart from that trace order, the
  other test files passed
  (572 tests, none failing apart from the stale fake above): GQA,
  GDN, MoE, collectives and layer schedule, stream and kernel step, pool and
  DCP, the m1 attention + MoE and m2 GDN model tests, and the planner, VMEM,
  weights and model-definition files. The rest ran on the merged tree, the
  slowest files in node-id chunks because they exceed the 40-minute
  per-file limit on the shared CPU machine: the core chunk with
  `test_exec_spec`, the tunables and the profiles (19 files, 160 passed),
  `test_b1_tiles_profiles` (48 passed, 2 expected failures),
  `test_verify_model` (18 passed), `test_m3_cache_eplb` (53 tests: 46
  passed, 1 skipped, 6 expected failures) and `test_module_api` (12
  passed); the first 36 of `test_b1_tiles_profiles`' tests, 15 of
  `test_verify_model`'s and 42 of `test_m3_cache_eplb`'s ran before the
  merge. Every test file of the suite has run on this code, and none
  failed.

## TPU checklist

1. Done (see Validation): per-bucket output bytes and step time against
   candidate4's kernel, and the per-option on/off checks of `ar0s`, `lcv`,
   `zdefer`, `mxu_dr` and `arpipe`. Still open: the same with expert copy
   slots, for the copy-routing options.
2. Done on CPU for the default profiles (see Validation): the megakernel
   and the top-level program are identical in all 32 configurations that
   compile; the raw gate's 17 failures are emission order only. Still open:
   the TP configurations, and a rerun with a gate that compares the order
   graphs as sets and pairs small fusions by raw id.
3. Serving A/B at points #8 and #12 (megakernel on, off, and candidate4 as it
   ran), with GSM8K, MTP acceptance length, logprob drift and the megakernel
   step share as gates.
