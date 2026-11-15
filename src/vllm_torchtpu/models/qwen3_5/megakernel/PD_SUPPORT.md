# Qwen3.5 megakernel on prefill / decode disaggregated decode legs

This branch adds what the Qwen3.5 megakernel needs to serve the decode leg of a
prefill / decode disaggregated (PD) deployment. It is a hand-off branch on top
of `nanwei/mk-dev` (`c297d9a45`, merged in; measured below at `1b9f7a5fd`),
meant to be taken apart and landed in the megakernel lanes; it is not a PR.
(It was first built on `c69cc7c14` and rebased; the only change the rebase
needed is in the startup guard, see §2.)

## Why

Four of the five configurations of our Qwen3.5 PD submission (points #5, #8,
#10 and #12) run every decode step they can on the megakernel and on the MTP
draft megakernel. Each of them is a decode pod with:

- `--tensor-parallel-size 8 --decode-context-parallel-size 4`,
  `--cp-kv-cache-interleave-size 128`, `--block-size 640`;
- `TPURaidenConnector` as the KV connector, with `kv_role` `kv_consumer`
  (`destination_decode_context_parallel_size: 4`): the prefill leg sends the
  KV pages and the recurrent (GDN) state, and the decode pod only receives;
- MTP speculative decoding (three draft tokens) and prefix caching;
- `--mamba-ssm-cache-dtype bfloat16`.

On `nanwei/mk-dev` as it stands, those pods are declined at startup twice
over: first `parallelism_unsupported:dcp`, then
`kv_connector_unsupported:TPURaidenConnector`. Our fifth configuration (#1,
data parallel 4 by TP 2) runs with the megakernel off and is not affected.

## What was added

### 1. Decode context parallelism (DCP) as a declared parallelism

With DCP size `d`, the `d` ranks of a DCP group share a KV head and each holds
one shard of every sequence's KV pages. Global kernel page `p` of a sequence
sits on DCP rank `p % d`, at column `p // d` of that rank's page table. That
is vLLM's DCP layout when `cp_kv_cache_interleave_size` equals the 128-token
kernel page, and it is the rule the batched RPA kernel and the PD connector's
decode reshard use. vLLM builds DCP groups as contiguous runs of `d` TP ranks,
so a rank's DCP rank is `tp_rank % d` (mk-dev's `distributed/dcp.py` already
checks that alignment).

- **Ownership rule, in one place:** `megakernel/core/pool/dcp.py` has the
  traced helpers the kernels use (`dcp_rank`, `local_kv_len`,
  `page_owner_col`) and their host twins.
- **Execution setting:** `CacheExec.dcp` (default 1; more than 1 needs
  `paged_kv`). At 1 nothing changes: every DCP branch is a trace-time
  `if dcp > 1`.
- **GQA module:** the sequence-split attention runs over the `G / dcp` ranks
  that hold the same shard (`TS = G // dcp`; the split rank is `gi // dcp`),
  each row's worklist and tail mask use this rank's cached length
  (`local_kv_len`), and the merge over the `G` partials is unchanged. DCP
  needs the split attention (its merge combines the shards) and `dcp`
  dividing `G = R / NKV`; anything else is a `MegakernelConfigError`.
- **Pool writers:** a new token's K / V is written only on the rank that holds
  its page, at column `page // dcp`, in the fused pool I/O (the per-sequence
  write mode and both write masks) and in the split scatter kernel. In the
  scatter kernel two tokens of one sequence can share a physical page id yet
  sit on different ranks (pages `p` and `p + 1` share a column), so each
  token's write is masked by its own page's owner, and a rank that holds none
  of the tokens skips the read-modify-write. The scatter kernel reads its DCP
  rank with `lax.axis_index("tp")` inside the Pallas body, as the megakernel
  body reads `me`.
- **Recurrent state:** unchanged. DCP does not shard the GDN state; every
  rank keeps its full per-head slot.
- **Warmup sentinel:** `verify_sentinel_step(kv_owner=...)`. Only the DCP rank
  holding the sentinel token's page writes it; on the others that page must
  stay unchanged (so a rank writing pages it doesn't own fails warmup), and
  the recurrent slot must still change.
- **Runner facts:** `RunnerView.dcp_size`, `dcp_rank` (from vLLM's DCP
  group; it refuses to guess when DCP is on and there is no group) and
  `dcp_interleave`. `testing.fake_runner` takes
  `decode_context_parallel_size` and `cp_kv_cache_interleave_size`.
- **Layout facts:** `PoolLayoutFacts` gained `dcp_size`, `dcp_rank`,
  `dcp_interleave` and `dcp_page_owner`. The megakernel side says DCP size
  `d`, rank `tp_rank % d`, one page per run (`page_mod_dcp`); the framework
  side reports the runner's size, vLLM's DCP rank and
  `cp_kv_cache_interleave_size`. Any difference is `pool_layout_mismatch` at
  bind. At DCP 1 the fields are `(1, 0, 0, "replicated")` and stay out of the
  payload, so the fingerprint of every DCP 1 slot is what it was.
- **Host checks:** `check_step_pool_metadata` reads the rank-local page
  table under DCP (column `page // dcp`, duplicate writes keyed by owner).
- **Qwen3.5 template:** `SPEC.parallelism` adds `"dcp"`. `create` declines
  `dcp_geometry_unsupported` (in `SPEC.decline_codes`) when the DCP size
  doesn't divide the `tp_size / num_key_value_heads` ranks per KV head (TP8
  with two KV heads: DCP 1, 2 or 4). `bind` passes the DCP size into the
  cache settings and checks the DCP facts. The collective domain stays `tp`:
  DCP groups lie inside the TP group, and the kernel's all-reduces and merge
  are the TP group's.

### 2. Consumer-only KV transfer: `kv_transfer_consumer`

A new contract feature, narrower than `kv_transfer`. A plugin that declares it
serves a KV transfer connector when:

- this engine is a KV consumer (`kv_role == "kv_consumer"`, and no child of a
  multi-connector says otherwise), and
- every transfer connector's class declares `layerwise_transfer = False`.

Such a connector loads the KV pages and the recurrent state into the request's
own blocks before the scheduler runs the request (`start_load_kv`, completed
through `get_finished`), and it has no per-layer load or save hooks that a
megakernel bypassing the attention modules would skip. `TPUConnector` (and so
`TPURaidenConnector`) declares `layerwise_transfer = False`: its
`wait_for_layer_load` and `save_kv_layer` are no-ops. The guard finds the
class through vLLM's connector registry, where vllm-torchtpu registers its
connectors; a connector the registry doesn't know, or whose class declares
nothing, is declined (fail closed). The megakernel already writes the pools
back in place with `copy_`, so the connector's registered views stay valid.

`kv_transfer` keeps its meaning: the plugin serves any transfer connector and
honors its hooks itself. The consumer-only checks apply only to
`kv_transfer_consumer`. The Qwen3.5 SPEC declares `kv_transfer_consumer`.

On this base the guard reads the KV transfer config from `vllm_config`, as
the other startup guards do since #2806, and a config that names no
connector puts none in effect, so the consumer checks decline nothing for it
(whatever its `kv_role`). That is the one change the rebase from `c69cc7c14`
needed: the first version read the config through a helper #2806 removed.

**Why a separate new feature rather than narrowing `kv_transfer`:**
`kv_transfer_consumer` leaves the existing meaning of `kv_transfer` ("any
connector, any role", honored by the plugin itself) and its tests untouched;
two of mk-dev's guard tests pin that meaning, for example `NixlConnector`
with no role. It is easy to fold in or rename later: the guard code is one
function (`guards.kv_transfer_decline`), and the SPEC declares the feature in
one line.

| Connector | `kv_role` | Plugin declares | Result |
|---|---|---|---|
| no layer-wise hooks (`TPURaidenConnector`) | `kv_consumer` | `kv_transfer_consumer` | served |
| layer-wise hooks | `kv_consumer` | `kv_transfer_consumer` | `kv_connector_unsupported:<name>` |
| no layer-wise hooks | `kv_producer`, `kv_both` or none | `kv_transfer_consumer` | `kv_connector_unsupported:<name>` |
| unknown to vLLM's registry, or declares nothing | `kv_consumer` | `kv_transfer_consumer` | `kv_connector_unsupported:<name>` |
| offloading only | any | `kv_offload` | served (as before) |
| multi-connector: no-hooks child plus offloading child | `kv_consumer` | both | served |
| multi-connector whose transfer child says `kv_producer` | `kv_consumer` | both | `kv_connector_unsupported:<name>` |
| any transfer connector | any | neither | `kv_connector_unsupported:<name>` (as before) |
| any transfer connector | any | `kv_transfer` | served (as before) |

### 3. `mamba_ssm_cache_dtype`

Nothing changed. The SPEC still requires `--mamba-ssm-cache-dtype bfloat16`,
and our decode legs now pass it. A test pins that the leg is served with it
and declined `requires_unmet:mamba_ssm_cache_dtype` with `float32` or the
default.

## Startup on our decode legs

With all three in place, the guards run in this order on a TP8 + DCP4
consumer leg: chip, TP size, collective domain, parallelism (`dcp` declared),
features (`spec_decode`, `prefix_caching`, `kv_transfer_consumer`), then
`requires` (`mamba_ssm_cache_dtype`). `create` then checks the DCP size
against the KV heads, and `bind` compares the slot facts, DCP fields
included.

One more decline is outside this branch and may still apply. The guard
declines `multimodal_inputs` whenever the runner reports
`supports_mm_inputs`, which the runner takes from vLLM's
`ModelConfig.supports_multimodal_inputs`. vLLM 0.32 sets
`enable_mm_embeds = True` on every KV consumer, so that a decode leg accepts
pre-computed-embedding requests. Our pods also pass `--language-model-only`
and `--limit-mm-per-prompt '{"image":0,"video":0}'`; whether that keeps
`supports_mm_inputs` false on a KV consumer has to be checked on TPU (the
megakernel's bound or declined log line says which). Our serving tree
forwards token ids on KV consumer pods in the runner; that change lands with
our runner work, not here.

## What was ported, from where, by whom

- **`6edc0e95b`, Benliang Wang**, "Run the Qwen3.5 decode megakernel under
  decode context parallelism", in the old `kernels/qwen35_megakernel/` layout
  (`config.py` `MKConfig.dcp`, `attention_block.py`
  `PagedAttentionBlock.kv_len`, `context.py` `TS` / `dr` / `gs`,
  `pool_dma.py` `KVPoolDMA.kv_page_col` and `_rmw_token`, `pool_io.py`
  `scatter_pool_states(dcp_axis=...)`, the warmup sentinel on page 0's
  owner, and `tests/kernels/test_qwen35_megakernel_dcp.py`). It is the only
  DCP change to the megakernel in our serving tree. The first commit of this
  branch carries it into the library layout under Benliang's authorship, with
  a `Source-commit:` trailer and his sign-off.
- Not carried: the step bench's DCP option (`MK_BENCH_DCP`), the MTP draft
  megakernel's DCP line (this tree has no draft megakernel yet), and the
  platform validator hunk (`platforms/dcp_validation.py`, which mk-dev does
  not have; it lands with our platform work).
- Everything else here (runner facts, layout facts, host checks, the
  template's wiring and decline, the consumer-only feature and the
  connector's declaration) is new, by Shrihari Murali.

## What is verified on CPU

Everything here is measured (M) on this branch at `1b9f7a5fd`, in the CI
image whose dependencies match it (`vllm-torchtpu:c350823af…-v0.33.0`;
mk-dev's `pyproject.toml` and `docker/` are identical to `c350823af`), with
kernels in Pallas interpret mode on 8 CPU devices.

**New tests: 77 passed, 0 failed (M).**

- `tests/megakernel/core/test_gqa_dcp.py` (21): the ownership rule and the
  rank-local length against brute force (DCP 1, 2, 4); DCP 1 against DCP `d`
  page partitions; `CacheExec.dcp` validation; GQA's `TS` at DCP 1, 2, 4 and
  its declines; the sentinel's owner check; and an 8-rank Qwen3.5-shaped step
  (two KV heads, so four ranks per KV head as at TP8) at DCP 1 against DCP 4
  (split scatter, MTP verify rows), DCP 4 and DCP 2 (fused pool I/O), and
  DCP 4 (decode rows). Hidden outputs stay within the merge-order band (at
  most 1e-3 of the elements outside atol = rtol = 2e-2, none above 0.25), and
  every page is byte-exact against the expected ownership: new tokens on
  their owner rank's local page only, recurrent checkpoints as at DCP 1, all
  else untouched.
- `tests/megakernel/core/test_dcp_runtime.py` (9): DCP facts in the payload
  and the comparison, the DCP 1 fingerprint pinned to its old value, the host
  pool-metadata check on rank-local columns, `align_cache_exec(dcp=...)`.
- `tests/megakernel/adapter/test_pd_support.py` (3): the runner view's DCP
  facts and the framework's DCP slot facts against the megakernel's.
- `tests/megakernel/adapter/test_kv_transfer_consumer.py` (20): every row of
  the table above, the messages, an unset connector with the feature
  declared, and that `TPURaidenConnector` declares no layer-wise transfer and
  resolves through vLLM's registry.
- `tests/models/qwen3_5/megakernel/test_qwen35_dcp.py` (13): the SPEC admits
  DCP, `create` declines DCP 8 by name and builds DCP 1, 2 and 4 programs,
  `bind` checks the DCP facts, the runner's DCP facts reach the framework
  facts.
- `tests/models/qwen3_5/megakernel/test_qwen35_pd_leg.py` (11): the whole
  guard chain on a TP8 + DCP4 consumer leg (served), on producer and
  `kv_both` legs and on other SSM dtypes (declined), and on the
  configurations the template served before (still served).

**mk-dev's own test files for every changed module (M): 22 files, 817
passed, 0 failed** (16 skipped and 6 expected failures, all declared by the
files themselves). That is `test_guards` 45, `test_plugin_surface` 20,
`test_testkit` 15, `test_pool_layout_facts` 12, `test_runtime_contract` 19,
`test_pool_config` 2, `test_qwen35_pool_layout_facts` 2,
`test_qwen35_template` 9, `test_driver_bind` 12, `test_runtime_api` 19,
`test_folder_cpu_e2e` 2, `test_tpu_connector` 484, `test_gqa` 7,
`test_kernel_step` 7, `test_gqa_pallas_core` 9, `test_gqa_decls_ref` 10,
`test_pool_scatter` 30, `test_f32_conv_stock_interop` 9,
`test_vllm_qwen35_e2e` 25, `test_m1_qwen35_attn_moe` 19,
`test_pool_f32_conv_step` 14 and `test_m3_cache_eplb` 46 (run in four
parts; its 6 expected failures are the file's own CPU fused-path xfails).
None failed, so no base-against-head run was needed to separate environment
failures on this base.
Three pins in them changed because the surface they pin changed: the runner
view's property count (17 to 20), and the template's `SPEC.features` /
`SPEC.parallelism` and its list of decline codes.

**The default path is unchanged (M).** At DCP 1 the traced pooled step has
exactly the same jaxpr as on `1b9f7a5fd`, with zero differing lines in 8 of
8 configurations: megakernel body and pool scatter, split and fused pool
I/O, decode `T=1` and verify `T=4` buckets, TPU and interpret lowering
(source locations and object addresses masked; a two-rank tiny Qwen3.5
geometry). Two traces of the base were identical, so the comparison is
deterministic.

**Mutation checks (M).** Each mutant breaks one thing and runs the tests
that guard it; every one made them fail.

- On this base, for the guard and the consumer feature (where #2806
  interacts), 7 of 7 killed: no role check, no layer-wise check, no
  multi-connector flattening, no child-role check, `TPUConnector` without
  its declaration, the SPEC without `kv_transfer_consumer`, and an unset
  connector treated as a connector named `None` (the behaviour #2806
  removed).
- On the first base (`c69cc7c14`, where the files they target are the same
  apart from the guard's config read), 25 of 25 killed: the kernel ownership
  masks in the split scatter and the fused pool I/O, attention with the
  global length, the worklist's split rank, the scatter's page column, the
  sentinel's owner check, GQA's DCP checks, the facts payload and
  comparison, the host checks, `align_cache_exec`, the runner and program
  DCP ranks, the SPEC's `dcp` and `requires`, and `create`'s DCP decline.

## TPU checklist

None of this has run on a TPU yet. In order:

1. **Point #8 serving A/B, megakernel on against off.** One TP8 + DCP4 decode
   leg behind a prefill leg, both arms at the same time. Check token identity
   on fixed greedy prompts, GSM8K with real rejection sampling, MTP acceptance
   length, and the share of decode steps the megakernel served (`mk_t*`
   against `stock_*` in the step-mix counters). The bound log line must say
   `dcp=4`.
2. **DCP4 numerics on chip.** The CPU tests compare DCP 4 with DCP 1 in Pallas
   interpret mode. The Mosaic lowering of the ownership masks, of
   `lax.axis_index` in the scatter kernel and of the per-token masked writes
   in the fused pool I/O only runs on TPU. The warmup sentinel's owner check
   is the first on-chip signal: a rank that writes a page it doesn't hold
   fails warmup.
3. **DCP1 LLO diff.** The compiled-code diff at DCP 1 (Mosaic MLIR and LLO
   bundles, spills and VMEM per kernel), the library's usual evidence. The
   CPU jaxpr identity predicts zero diffs.
4. **Per-bucket step time.** The megakernel microbench at DCP 4 against DCP 1
   for every compiled bucket, interleaved on one allocation; the worklist per
   rank changes shape under DCP.
5. **The multimodal decline on KV consumers.** Confirm that a decode leg with
   `--language-model-only` is not declined `multimodal_inputs` (vLLM sets
   `enable_mm_embeds = True` on every KV consumer; see "Startup on our decode
   legs").
