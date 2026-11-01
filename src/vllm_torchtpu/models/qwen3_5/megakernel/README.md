# Qwen3.5 megakernel model folder

This folder serves Qwen3.5-397B-A17B with the TPU megakernel: one Pallas
kernel per decode (`T=1`) or MTP verify (`T=4`) step, plugged into the
vLLM-TorchTPU runner through the megakernel runtime. It is also the template
to copy for a new model.

| File | What it holds |
|---|---|
| `__init__.py` | `SPEC`: the `TpuMonoSpec` the runtime loads (stdlib only) |
| `adapter.py` | `create` and `Qwen35Program`: options, eligibility, weights, cache binding, per-step routing |
| `model.py` | `Qwen35`: the declarative `ModelDef` (geometry, schedule, weights); `register()` |
| `pool.py` | the raw KV+GDN pool layout |
| `hf_oracle.py` | the Hugging Face `transformers` reference used by verification |
| `profiles/` | `qwen35-baseline` (`baseline.py`), `qwen35-candidate30[-fused]` (`candidate30.py`); `PROFILES`, `DEFAULT_PROFILE` |
| `tests/` | the CPU end-to-end test (fake runner, no TPU) |
| `serving/` | `serve.sh`, `run_real_ckpt_smoke.py`, `mk-real-ckpt-smoke.yml`, `acceptance_benchmark.py`, `BENCH.md` |

Importing the folder registers nothing; `model.register()` adds `Qwen35` and
its profiles to the library registries for by-name lookups, and `create`
calls it.

## Quickstart

### 1. Copy (for a new model)

```bash
cp -r src/vllm_torchtpu/models/qwen3_5/megakernel \
  src/vllm_torchtpu/models/<model>/megakernel
```

The folder imports its own files relatively, so the copy runs as is under its
new package name. Then rename `SPEC.name`, the `ModelDef` (`model.py`:
`name`, `hf_architectures`, `hf_model_type`) and the profile names, and add
a shorthand to `megakernel_runtime.plugin.SHORTHANDS` if the model should be
picked from its checkpoint's architecture. See "Adapting this folder" below.

### 2. Test (CPU, no TPU)

```bash
JAX_PLATFORMS=cpu pytest src/vllm_torchtpu/models/qwen3_5/megakernel/tests
JAX_PLATFORMS=cpu pytest tests/models/qwen3_5/megakernel -n 8
```

The first runs the folder's own end-to-end test: `SPEC` loads, the runtime's
startup checks pass on a fake runner, `create` builds a program for a tiny
geometry, and the program binds, warms up and serves a decode step through
the CPU-interpret kernel. The second is the full suite (bit-exact golden
hashes, the reference checks, the vLLM integration tests).

### 3. Serve (one TPU v7x host)

```bash
MODEL=Qwen/Qwen3.5-397B-A17B-FP8 \
  src/vllm_torchtpu/models/qwen3_5/megakernel/serving/serve.sh
# MTP speculative decoding (the megakernel serves the T=4 verify steps):
MODE=mtp MODEL=... src/vllm_torchtpu/models/qwen3_5/megakernel/serving/serve.sh
```

`serve.sh` holds the candidate30 serving configuration. The flags the
megakernel needs:

- `VLLM_TPU_MEGAKERNEL=1` (on; the template is picked from the checkpoint's
  architecture), with the runner settings `serve.sh` exports
  (`TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1`, `VLLM_KV_CACHE_LAYOUT=HND`, ...);
- `--tensor-parallel-size 8 --enable-expert-parallel` (EP inside the TP group);
- `--mamba-ssm-cache-dtype bfloat16` (a `SPEC.requires` setting: without it
  the megakernel stays off with `requires_unmet:mamba_ssm_cache_dtype`),
  `--mamba-cache-mode align`, `--block-size 640`, `--kv-cache-dtype fp8`;
- `--language-model-only --limit-mm-per-prompt '{"image":0,"video":0}'`
  (text-only serving);
- for MTP: `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`.

Template settings go in `VLLM_TPU_MEGAKERNEL` (`k=v,k=v` or JSON) or
`--additional-config '{"megakernel": {"options": {...}}}'`: `profile`,
`mode` (`ep` / `tp`), `buckets`, `verify_buckets`, `fused_pool_io`,
`pool_gather`, `compile_hlo_check`. Runtime settings: `stats_period_s`, `ab`.

### 4. Smoke (real checkpoint, decode + MTP)

```bash
python3 src/vllm_torchtpu/models/qwen3_5/megakernel/serving/run_real_ckpt_smoke.py
```

It starts `serve.sh` in each mode, sends greedy completions, and checks the
server's megakernel log lines (plugin loaded, program bound, megakernel steps
in the step mix, no decline). It serves the checkout it runs from (its `src/`
first on `PYTHONPATH`), fails up front if the image's installed
`vllm_torchtpu` would shadow it, and writes nothing outside `/tmp`.

On the cloud-devkit cluster, the recipe `nanwei-mk-real-ckpt-smoke` runs it on
a v7x 2x2x1 host with the shared checkpoint cache: it runs
`cd /root/cloud-devkit/vllm-torchtpu && python3 -u ${TEST_PATH}`, where that
directory is an NFS export of a checkout of this repository and `TEST_PATH` is
any repo-relative path whose basename matches `run_real_ckpt_smoke*.py`:

```bash
# Stage a checkout on the shared NFS (example).
git -C /mnt/nfs_test/agentx-cache/<you>_mk_smoke checkout <sha>
cdk job create nanwei-mk-real-ckpt-smoke \
  NFS_SRC_PATH=/share/agentx-cache/<you>_mk_smoke \
  TEST_PATH=src/vllm_torchtpu/models/qwen3_5/megakernel/serving/run_real_ckpt_smoke.py \
  -t mk-smoke
cdk job log <job-id>
```

The recipe name comes from the cloud-devkit change under review;
`serving/mk-real-ckpt-smoke.yml` is the same JobSet, for a cloud-devkit
checkout that registers it locally. Optional recipe variables: `IMAGE`, `CPU`,
`MEMORY`, `DSHM_SIZE`, `NFS_SERVER_IP`.

`MK_SMOKE_MODE=decode|mtp|both`, `MK_SMOKE_CKPT`, and `MK_SMOKE_MEGAKERNEL`
(e.g. `profile=qwen35-baseline`) adjust a run.

### 5. Benchmark

See `serving/BENCH.md` (`serving/acceptance_benchmark.py`).

## Expected log lines

At startup (every TP rank; rank 0 shown):

```text
Megakernel plugin qwen35 1 loaded from vllm_torchtpu.models.qwen3_5.megakernel (spec numerics=approx (default profile), fingerprint 3f2a...)
Qwen3.5 megakernel bound to 15 unified pools (profile=qwen35-candidate30-fused, numerics=approx, num_pages=..., page_size=..., max_pages_per_seq=..., decode_buckets=(8, 16, 32), verify_buckets=(...), ...)
```

While serving, every `stats_period_s` (default 60 s), on rank 0:

```text
Qwen3.5 megakernel step mix over 60s: 950/1000 MK (95.0%) {'mk_t1_b8': 950, 'stock_prefill': 50}
```

`mk_t1_b<B>` / `mk_t4_b<B>` count megakernel decode / verify steps by bucket;
`stock_<reason>` counts steps the stock model served (prefill, mixed batches,
batches larger than the largest bucket). When the megakernel can't serve the
server, a warning names the reason and every step counts as
`stock_<code>`, e.g.:

```text
Megakernel model folder requires mamba_ssm_cache_dtype=bfloat16 (this server: auto); pass --mamba-ssm-cache-dtype bfloat16; skipping.
```

## Profiles

| Profile | Numerics | What it is |
|---|---|---|
| `qwen35-candidate30-fused` (default) | approx | the tuned candidate30 schedule plus 1-launch fused pool I/O (`B <= 48`, `B >= 56`) and in-kernel gather |
| `qwen35-candidate30` | approx | the tuned schedule with split pool I/O |
| `qwen35-baseline` | exact | the model's reference profile: conservative tiles, split pool I/O, all opt-in flags off |

`approx` profiles turn on optimizations that reorder floating-point work
(results differ from `qwen35-baseline` in the last bits); `exact` ones match
the reference bit for bit. Select one with `profile=<name>`. `SPEC.numerics`
is the default profile's; the bound line names the served profile's.

## Support matrix

| | Supported | Not supported (the megakernel stays off, named reason) |
|---|---|---|
| Chip | TPU v7x (`TPU7x`) | other chips (`unsupported_tpu_chip`) |
| Parallelism | TP=8, EP inside TP | other TP sizes (`unsupported_tp_size`); DP, PP, DCP / PCP, EP wider than TP (`parallelism_unsupported:<kind>`) |
| Steps | decode `T=1` (buckets 8, 16, 32), MTP verify `T=4` (up to 64 rows) | prefill and mixed steps (the stock model serves them) |
| Features | MTP speculative decoding, M-RoPE, prefix caching, KV offloading | LoRA, multimodal inputs, KV transfer connectors, dynamic EPLB, sleep mode, EAGLE-3 hidden-state capture |
| Server settings | `--mamba-ssm-cache-dtype bfloat16` | anything else (`requires_unmet:mamba_ssm_cache_dtype`) |
| Checkpoint | Qwen3.5-397B-A17B (FP8), its 60-layer hybrid GQA / GDN + MoE geometry | dense Qwen3.5 checkpoints, other geometries (`non_moe_checkpoint`, `geometry_mismatch`) |

The runtime checks the chip, TP size, parallelism, features and server
settings from `SPEC` before `create` runs; the checkpoint checks are
`create`'s.

## Adapting this folder

1. Copy the folder (above) and keep its imports relative.
2. `model.py`: describe the new architecture with the library's modules
   (`GQA`, `GDN`, `MoE`, `Norm`, ...) and its geometry from the HF config.
   Keep `register()`; register nothing at import time.
3. `profiles/`: start from `baseline.py` (exact numerics), name the
   profiles after the model, and set `DEFAULT_PROFILE`.
4. `adapter.py`: adjust `create`'s checkpoint checks and the program's weight
   loading / cache binding to the new model's vLLM modules.
5. `__init__.py`: set `SPEC` (`name`, `chips`, `tp_sizes`, `parallelism`,
   `features`, `requires`, `numerics`, `decline_codes`).
6. `tests/` and `tests/models/<model>/megakernel/`: adapt the CPU end-to-end
   test and the goldens; `serving/`: the serve flags and the smoke.

Copies made by hand or by the planned scaffold
(`python -m vllm_torchtpu.megakernel_runtime.new --from qwen3_5 ...`) carry
a provenance header atop every copied file:

```python
# Copied from src/vllm_torchtpu/models/qwen3_5/megakernel/<file> @ <commit>
```

so a change to this folder can list its copies. Files moved here (not copied)
carry none.
