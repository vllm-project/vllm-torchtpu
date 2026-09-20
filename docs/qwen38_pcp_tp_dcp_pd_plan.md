# Qwen38 c3: PCP16TP2 -> TP32DCP8 PD support assessment

2026-09-20. Source review only; no implementation or service changes.
Full target assumes32 devices on P and32 on D (four8-device hosts per engine).
Transport baseline: TPURaidenConnector with unified pools and HND.

## Existing implementation and gaps

- `distributed/kv_transfer/tpu_connector.py::_raiden_qwen35_admission_topology`
  admits PCP4/8 with TP1 producers and DP2/4/8/16 with TP1/2 consumers, plus
  pure PP shapes. It does not admit the requested P/D topology.
- `_local_raiden_transfer_rank` returns only PCP rank for Qwen producers.
  TP lanes would collide. `_validate_stage3_transfer_parallelism`, scheduler
  source metadata and completion counts assume producer parallelism=PCP.
- `_stage3_sharded_consumer` recognizes TP2 only. Destination unit enumeration
  already covers TP ranks, so extend the existing coordinated receiver path.
- `_raiden_interleave_tokens` treats consumers as contiguous. FA span lowering
  has no DCP destination ownership, and the transfer directory contains no DCP
  integration. Releasing admission restrictions alone cannot work.
- `_local_gdn_head_count` divides TP-local heads by PCP. Full model K16/TP2=8
  is not divisible by PCP16; quarter model K4/TP2=2 is not divisible by PCP4.
  Reuse `kernels/gdn/head_geometry.py` instead of duplicating division rules.
- `raiden/seq_on_lane.py::lower_fa_spans_sol` splits the producer's local KV
  heads evenly over dst_shards; dst_shards must divide local KV heads. It does
  not describe combined source TP, destination TP replication and DCP.
- HND span lowering currently requires producer interleave=kernel page size,
  and fingerprints require matching kernel pages. Manager block size is a
  separate quantity. Keep matching kernel page geometry for initial tests.

## Required mapping

Use one shared logical ownership description, lowered to the existing pool
byte-span IR. Distinguish worker identity, token ownership, FA KV-head ownership
and GDN state-head ownership. Do not use one scalar rank for every purpose.

P worker identity can follow the current PCP-major worker order p*TP+t.
However GDN state head order is TP-major: `layers/core/gdn_attention.py`
`_gdn_pcp_tp_state_axis` returns (tp,pcp). For the full shape, P(p,t)'s V/SSM
state corresponds to D TP rank `t*16+p`; for the quarter test, `t*4+p`.
Resolve actual communication groups/device assignment separately from these
logical head coordinates; never infer ownership from raw device IDs.

FA for the full model: P TP0 owns global KV heads0/1, TP1 owns2/3, and PCP
partitions their token ranges. D ranks0..7 own KV head0 with DCP8 token
stripes; ranks8..15 own head1,16..23 head2,24..31 head3 under the current
contiguous TP/DCP group construction. D rank/head ownership should ultimately
come from the group's declared geometry, rather than hardcoded rank ranges.

For D kernel page size B and request token index x, `schedule_cp.py` writes
global page j=floor(x/B) on DCP rank j%8 at local page floor(j/8), preserving
the in-page offset. Lower source PCP ownership through the existing
pcp_layout helpers, intersect it with destination head and token ownership,
and then translate through each side's actual block table and HND layout.
Destination FA offsets must be rank-local, not global token offsets.

GDN is different: DCP does not shard conv or SSM state over history tokens.
Transfer each request's final committed state by GDN head ownership. Both
full P and D have effective GDN degree32: each rank has K1/V4, Q/K replication
factor2. Preserve Q/K copies explicitly; do not concatenate copies into extra
logical heads. For this exact pair the corresponding source/destination local
state shapes match, enabling direct byte spans subject to measured layout.

Retain the existing N-1 prompt handoff: P's state is after the transferred
prefix; D executes the final prompt token once. FA tokens, conv state and SSM
state must describe the same boundary. Retain committed-block selection.

## Expected change scope

1. Connector request/work-unit metadata: separate PCP/TP/DCP geometry,
   unique source identities, complete source/destination unit lists, endpoint
   uniqueness and source lifetime through all receivers' completion.
2. Connector admission and coordination: generic valid geometry rather than
   TP2-only destination detection and enumerated topology names. Existing
   TP1/DCP1 cases use the same path with degree1.
3. `raiden/seq_on_lane.py`, `byte_spans.py`: combined head/token ownership
   intersections, replicated-head source selection and destination local offsets.
4. `raiden/pool_manifest.py`, fingerprint/metadata: describe the live HND and
   GDN state layout and replicated head identity; reuse GDN geometry helpers.
   Check SSM BF16, conv QK pair packing, pool offsets/padding and layer tags.
5. Native tpu_sync integration tests: existing ReshardClient supports spans,
   strides/counts and dst_unit_ordinal. Reuse the data transport. Whether C++
   planner coverage checks need extension remains to be verified; Python API
   inspection alone does not establish support for sharded destination lengths.

Attention/GDN/MoE computation kernels are not the expected primary edit sites.
The main risk is silent ownership/layout mismatch, not missing communication
primitives. This is a moderate resharding extension, not a flag-only change.

## Two-host functional test (one P, one D)

P and D MUST use the same logical model and weights. Previous independent
role-specific scaling configs cannot be paired for correctness testing.

| Setting | Full deployment | Primary scaled test |
|---|---|---|
|P|PCP16TP2|PCP4TP2|
|D|TP32DCP8|TP8DCP8|
|Layers|92|12 initially, then92|
|Hidden|8192|8192|
|FA Q/KV heads|64/4|16/1|
|FA head_dim|256|256|
|GDN K/V heads|16/128|4/32|
|GDN head dimensions|128/128|128/128|
|Routed experts|512|128|
|topk|10|10|
|MoE/shared expert intermediate|2048|2048|
|KV/SSM dtype|FP8/BF16|FP8/BF16|

This preserves D's per-rank FA Q2/KV1 and8-way DCP, plus P/D per-rank GDN
K1/V4 with replication2. With EP enabled, each rank owns16 routed experts.
It does NOT preserve all per-rank compute/HBM: P has FA Q8/KV1 instead of
Q32/KV2, and D's TP-sharded dense/shared-expert projections differ from TP32.
Do not infer full-cluster performance directly from this functional fixture.

Primary test adds FA KV replication on P: its single logical KV head appears
on both TP lanes. Choose an explicit non-overlapping source ownership for those
duplicate bytes. All8 P workers still participate for their distinct GDN state.

Supplementary test: same scaled GDN/MoE/layers, but FA heads64/4 on BOTH P/D,
P PCP4TP2 -> D TP8DCP2. It preserves P's two different KV heads per TP lane
and tests routing each head into the correct DCP group. It does not substitute
for the primary DCP8 test. Together they cover both source head splitting and
eight-way destination token sharding; neither alone exactly scales every axis.

Start with unified HND, matching128-token kernel pages/interleave if admitted;
derive manager blocks from actual state/pool requirements rather than assuming
128-token manager blocks. P global batch16384 gives4096 tokens per PCP rank;
TP2 does not further divide the token batch. Start with12 layers for iteration.

## Validation sequence

1. CPU ownership/span tests for full32->32 and scaled8->8 geometries: no holes,
   no duplicate writes, shuffled physical blocks, kernel/manager boundaries,
   partial tails, replicated Q/K heads and zero-owned-token ranks.
2. TPU marker/checksum transfer tests for actual HND, BF16 SSM and conv layout.
   Verify exact live bytes and head/token identifiers after transfer.
3. End-to-end PD requests, with proof that D loaded remote KV/state and did not
   silently recompute the prompt. Include boundary lengths and concurrent
   requests, plus N-1-state continuity.
4. Numerical validation needs identical logical weights across P/D and a
   non-PD reference. Same dummy seed under different TP partitions is not a
   sufficient guarantee. Use deterministic global-coordinate weights or a
   shared checkpoint for logits/state comparisons.

The two machines must run independent eight-device TPU runtimes for this test,
with separate engine IDs/ports and routable Raiden endpoints. Do not retain
the existing two-host model collective world. No slice or service changes have
been made as part of this assessment. Multi-host P/D engine behavior and full
32-worker runtime completion still need later hardware validation.
