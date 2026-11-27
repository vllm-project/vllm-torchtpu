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

## Host memory of a prefill replica

Every point's prefill leg gives its offloading connector
`cpu_bytes_to_use: 720000000000`. `TPURaidenStoreOffloadingSpec` sizes the
Raiden host tier from that value alone (it divides it by the KV bytes of one
block over all of the replica's workers), so each prefill replica holds
720 GB (670.6 GiB) of host memory for the tier, and nothing checks the value
against the host's memory. The engine's other host memory comes on top: the
TPU runtime's premapped transfer buffer, Raiden's staging buffers and the
workers' heap, which grows while the programs compile. On record-tree
prefill engines in p01's layout (PP8 x TP1) with the same 720 GB tier, a
prefill pod held about 210 GiB of host memory before the tier registered and
885 GiB at steady state, which crept up by 4 to 5 GiB over two hours of
serving; two instances differed by 11 GiB. The tpu7x host allocates 917.9
GiB to the pod. A prefill replica of these entries therefore needs about
890 GiB of host memory. On a host with less, lower `cpu_bytes_to_use`, which
shrinks the host tier and with it the share of returning sessions that hit
it.

## Reading a gate entry

`legs` derive from the point named in `base_point`, with one replica each and
the listed `serve_args_override`. `arms` differ only in `additional_config`;
the changes that add the FP8 trade-off settings add them to `tradeoffs_on`.
`references` are earlier gates at the same layout (GSM8K strict-match, 1,319
samples, thinking on), with the tree each ran described in words.
