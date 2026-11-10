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
  and served name.
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

## Reading a gate entry

`legs` derive from the point named in `base_point`, with one replica each and
the listed `serve_args_override`. `arms` differ only in `additional_config`;
the changes that add the FP8 trade-off settings add them to `tradeoffs_on`.
`references` are earlier gates at the same layout (GSM8K strict-match, 1,319
samples, thinking on).
