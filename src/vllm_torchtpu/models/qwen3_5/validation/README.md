# Qwen3.5 validation entries

One YAML file per serving configuration that Qwen3.5 changes are validated
against:

- `pNN-*.yaml`: the five InferenceX (IX) points of sheet
  `ix-tpu-pd-cand4-llmd-oct28`, each a prefill/decode disaggregated
  deployment. `record` names the job behind the published number and the tree
  it ran; `recipe` names the cloud-devkit-recipes file that launched it.
- `gate-*.yaml`: the 1P1D accuracy gates, one per decode layout. Each runs
  its `arms` back to back and reports the `suite`.

## Reading a point entry

Each leg (`prefill`, `decode`) lists:

- `replicas` and `chips_per_replica` (one tpu7x host, 8 devices, per
  replica); `chips` is their total over both legs.
- `serve_args`: the `vllm serve` arguments the record job's leg ran, from its
  rendered recipe. `$MODEL` and `$SERVED_MODEL` are the recipe's model path
  and served name. The prefill legs name no `--scheduler-cls`: the record
  jobs named PrefixAwareScheduler there, but their platform replaced it with
  the pipeline chunk scheduler for the PP > 1 leg, which is what ran. On this
  tree the platform installs that scheduler only when no class is named.
- `env`: the environment the leg set, limited to variables this package
  declares in `envs.py`. Per-pod wiring (ports, hosts, engine ids, cache
  directories) stays in the recipe.
- `additional_config`: the `--additional-config` JSON for the leg. The
  `qwen3_5` section holds Qwen3.5 settings (`settings.py`); other sections
  belong to shared subsystems.

The record jobs ran the submission tree named in `record.tree` (branch
`infx-qwen35-agentx-disagg-candidate4`), whose environment variables are being
turned into settings, rules or `additional_config` options change by change.
Each change that turns one the points use into a setting or section adds it to
the entries, so the entries describe how to run the points on the tree they
are in. Run the decode megakernel off on this tree: it is not on `main` yet.

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
each leg's, key by key, so an arm turns a setting off by setting it to false:
the changes that add the trade-off settings set them true in `tradeoffs_on`
and false in `tradeoffs_off`, and the bf16 GDN SSM state
(`--mamba-ssm-cache-dtype bfloat16`, which every point leg passes) is on in
`tradeoffs_on` only.
`references` are earlier gates at the same layout (GSM8K strict-match, 1,319
samples, thinking on).
