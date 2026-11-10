# Qwen3.5 validation entries

One YAML file per serving configuration that Qwen3.5 changes are validated
against:

- `pNN-*.yaml`: five benchmark points of Qwen3.5-397B-A17B-FP8, each a
  prefill/decode disaggregated deployment.
  `pNN` is the point's number on the published throughput/interactivity curve,
  from p01 (lowest interactivity, highest throughput per dollar) to p12. The
  rest of the name is the decode layout, the
  prefill and decode replica counts (`4p3d`: four prefill and three decode
  replicas), the decode leg's `--max-num-seqs` (`capN`) and the benchmark
  concurrency (`cN`). The `t3` at the end of p12's name stands for the
  batch-1 host trims that its `decode_layout` lists.
- `gate-*.yaml`: the 1P1D accuracy gates, one per decode layout. Each runs
  its `arms` back to back and reports the `suite`.

## Reading a point entry

`record` holds the published result of the point: Mtok/$ and p90
interactivity (tokens per second per user). The record run used an earlier,
unreleased tree of this repository (the record tree), so `record` also
notes that tree's decode megakernel mode, which this tree does not have: run
the decode megakernel off here.

Each leg (`prefill`, `decode`) lists:

- `replicas` and `chips_per_replica` (one tpu7x host, 8 devices, per
  replica); `chips` is their total over both legs.
- `serve_args`: the `vllm serve` arguments the record run's leg used.
  `$MODEL` and `$SERVED_MODEL` are the model path and served name. The
  prefill legs name no `--scheduler-cls`: the record runs named
  PrefixAwareScheduler there, but their platform replaced it with the
  pipeline chunk scheduler for the PP > 1 leg, which is what ran. On this
  tree the platform installs that scheduler only when no class is named.
- `env`: the environment the leg set, limited to variables this package
  declares in `envs.py`. `$EXPERT_PLACEMENT_TABLE` is the path of the
  decode leg's expert placement table, which `not_yet_supported` describes.
  Per-pod wiring (ports, hosts, engine ids, cache directories) is
  deployment-specific and not recorded.
- `additional_config`: the `--additional-config` JSON for the leg. The
  `qwen3_5` section holds Qwen3.5 settings (`settings.py`); other sections
  belong to shared subsystems.

The decode legs run MTP with synthetic rejection sampling at a fixed
acceptance length of 3.39 (`--speculative-config`), the acceptance length the
benchmark fixes for this model, so the `record` numbers are at
that acceptance length. The gates use real rejection sampling and report the
real acceptance length.

The record tree's environment variables are being turned into settings,
rules or `additional_config` options change by change, and each change that
turns one that the points use into a setting or section updates the entries.
A value that this tree cannot run yet is kept as the record run used it and
listed under `not_yet_supported`, per leg, with what it needs: a capability
this tree lacks, or a file it does not ship (p01's permutation table); the
change that adds the capability removes it from that list. The entry tests
check that this tree's `envs.py` accepts every other environment value. A leg
run here without the listed values uses fused EP MoE v2 and main's
permutation placement, so the `record` numbers are not expected to
reproduce on this tree. The record run's TP8 + DCP4 decode legs also took
their KV cache size from a budget the record tree's config hook calibrated
with fused EP MoE v3 (their `serve_args` pass no `--kv-cache-memory-bytes`);
the change that adds v3 adds that budget, and until then those legs size
their KV cache by `--gpu-memory-utilization`.

Three serve arguments differ from the record jobs because `main`'s unified
GDN state pool differs from the record tree's (vllm-torchtpu #1620):

- `--mamba-ssm-cache-dtype bfloat16` on every leg. The record tree stored
  the Qwen GDN SSM state in bf16 by default; `main` stores the declared
  dtype (the checkpoint's float32) unless this flag asks for bf16. It is a
  precision trade-off, so the gates run it in `tradeoffs_on` only.
- `--block-size`: one block holds a request's whole GDN state on `main`
  (conv state in float32, SSM state in the declared dtype), so a smaller
  value is raised at startup. With the bf16 SSM state that is 2304 tokens on
  the TP1 prefill legs (2048 SSM + 256 conv) and 1280 on the DP4 x TP2 decode
  leg of #1 (1024 + 256); the record ran 2176 and 640. The TP8 decode legs
  need 320 and keep the record's 640.
- `TPU_GDN_CONV_QK_PAIR_LAYOUT` is not set: `main` moves the float32 conv
  state segment by segment and its Raiden pool manifest refuses the QK
  pair-blocked layout at startup.

## Reading a gate entry

`legs` derive from the point named in `base_point`, with one replica each and
the listed `serve_args_override`. `arms` differ in `additional_config` and
in the serve arguments `arm_serve_args_override` lists for each arm, applied
over the legs'. An arm's `additional_config` sections are deep-merged over
each leg's, key by key, so an arm turns a setting off by setting it to false.
The changes that add the trade-off settings set them true in `tradeoffs_on`
and false in `tradeoffs_off`, and the bf16 GDN SSM state
(`--mamba-ssm-cache-dtype bfloat16`, which every point leg passes) is on in
`tradeoffs_on` only.
`references` are earlier gates at the same layout (GSM8K strict-match, 1,319
samples, thinking on), with the tree each ran described in words.
