---
name: cdk-g3-microbench
description: >-
  Universal drop-in `cdk-g3` CLI (`PERFKIT_CDK_BIN=cdk-g3`) and Evergreen Mode 2
  Google3 TPU microbenchmark & compiler-dump skill (`jfy-32478767326.us-central1.sourcemanager.dev/cloud-tpu-inference-test/cdk-g3`).
  Offloads pure-JAX/Pallas/TPU-C kernel, sublayer, collective (`jax.shard_map`),
  and XLA flag (`LIBTPU_INIT_ARGS`) inner-loop benchmarks from the contended GKE
  `cdk` cluster onto Google3 Backend A (Forge TPU `requires-ghostfish`, ~30-60s),
  Backend B (Minibench / MLCR warm pools ~5-15s & Percale `llo_orchestrate`),
  and Backend C (XManager / Borg TPU `group:msca-dynamic/tpu-perf-team-dynamic-xm`,
  1,422 Ghostfish chips, including 2-host `gf=2x2x2` ICI sweeps). Enforces Step 0
  `verify-scope` (`TRAP36F`/`TRAP36G`), `TRAP36H` MLCR Ghostfish fallback block,
  and `--collect-telemetry` / `verify-telemetry` (`TRAP36I` 4-category XProf,
  Trace JSON, HLO, and LLO/Mosaic artifacts).
---

# `cdk-g3` Universal Google3 TPU Microbenchmark & Compiler Telemetry Skill (`cdk-g3-microbench`)

> [!IMPORTANT]
> **Single Source of Truth Repository**: `https://jfy-32478767326.us-central1.sourcemanager.dev/cloud-tpu-inference-test/cdk-g3` (`git@jfy-32478767326-ssh.us-central1.sourcemanager.dev:cloud-tpu-inference-test/cdk-g3.git`).
> **Repo-Local Binary**: `./bin/cdk-g3` is committed directly in this repository. You can execute `./bin/cdk-g3` immediately without additional setup!
>
> To install system-wide on any GCE Cloud VM or Google3 Cloudtop:
> ```bash
> git clone git@jfy-32478767326-ssh.us-central1.sourcemanager.dev:cloud-tpu-inference-test/cdk-g3.git ~/cdk-g3
> ~/cdk-g3/install.sh
> export PERFKIT_CDK_BIN=cdk-g3
> export CDK_G3_STORAGE_URI=gs://forge-gcs-vm/forge/perfkit-cluster/_cdk_g3
> ```

---

## 1. When to Use `cdk-g3` vs. GKE `cdk` (6-Level Routing Matrix)

The GKE `cdk` cluster has only 2 `2x2x2` nodes and 5 `2x2x1` nodes (36 chips total). `cdk-g3` offloads **Levels 1–4** inner-loop iterations onto Google3's 1,422-chip `tpu_ghostfish` pools and pre-warmed MLCR pools with **zero Docker image rebuilds**:

| Level | Optimization Scope | Allowed File Paths | Target Tool & Backend |
| :--- | :--- | :--- | :--- |
| **Level 1** | Single Pallas / Tokamax / TPU-C Kernel | `vllm_torchtpu/kernels/**`, `probes/**`, `*_kernel.py`, `*_kernel.cc` | **`cdk-g3`** (`--backend=forge` or `--backend=minibench`) |
| **Level 2** | Fused Sublayer (MoE Block, MLA+DSA, GDN) | `vllm_torchtpu/kernels/**`, `examples/**` | **`cdk-g3`** (`--backend=minibench --minibench-mode=llo_orchestrate` or `--backend=forge`) |
| **Level 3** | Single-Host / Multi-Host `jax.shard_map` Collectives | `vllm_torchtpu/kernels/**`, `xm_launch.py` | **`cdk-g3`** (`--backend=xmanager --tpu-topology=gf=2x2x2`) |
| **Level 4** | XLA / Mosaic Compiler Flags (`LIBTPU_INIT_ARGS`, `XLA_FLAGS`) | `vllm_torchtpu/envs.py`, recipe env vars | **`cdk-g3`** (`--backend=forge` or `--backend=xmanager`) |
| **Level 5** | PyTorch Model Layer / `vllm` Worker / Attention Glue | `vllm_torchtpu/models/**`, `worker/**`, `attention/**`, `communicator/**` | **GKE `cdk`** (`cdk-hot-overlay-step-mix-twin` / `agentx-bench` mini) |
| **Level 6** | Full End-to-End `vllm serve` & `cdk dash` SSOT Certification | Any repo path + full Docker image | **GKE `cdk`** (`run-vllm-torchtpu-bench` + `cdk-dash`) |

---

## 2. Mandatory Step 0: Automatic Router & Pre-Flight Scope Verification (`cdk-g3 verify-scope` / `cdk-g3 route`)

Before dispatching any candidate, run `cdk-g3 verify-scope` (or its alias `cdk-g3 route`, supported in both flag orders). The human user never needs to specify which tool or backend to use—the router inspects the git worktree (`git diff` / `git status` when `--changed-files` is omitted) and AST import closure automatically:
1. **Automatic Git Diff Discovery & Routing Verdict (`routing_target`)**: Outputs `"routing_target": "CDK_G3"` (`rc=0`) vs. `"routing_target": "GKE_CDK"` (`rc=2`), along with `"recommended_cli"` (`"cdk-g3"` vs. `"cdk"`), `"recommended_backend"` (`"forge"`, `"minibench"`, `"xmanager"`, or `"gke_cdk"`), and `"recommended_recipe"` (`"forge-microbench"`, `"minibench-ab"`, `"xm-sweep"`, or `"run-vllm-torchtpu-bench"`).
2. **`[TRAP36F_OUT_OF_CDK_G3_KERNEL_SCOPE]` & `--allow-dependency-closure`**: Rejects (`rc=2`) modifications to `vllm/`, `Dockerfile`, or `csrc/` and routes them to GKE `cdk`. When `--allow-dependency-closure` is passed, pure-Python sublayer/module paths outside `vllm_torchtpu/kernels/**` are also admitted if their transitive AST import closure only imports pre-linked Google3 TPU runner packages (`jax`, `jax.experimental.pallas`, `tokamax`, `flax`, `optax`, `einops`, `numpy`, `scipy`, `absl`, Python stdlib).
3. **`[TRAP36G_FORBIDDEN_FRAMEWORK_IMPORT_IN_KERNEL_SCOPE]`**: Walks the transitive Python AST import closure across `--source-root` and routes any module that imports `torch`, `torch_xla`, `vllm`, or `triton` to GKE `cdk`.
4. **Auto-Staging (`--stage-bundle-dir`)**: Copies the verified transitive `.py` closure byte-for-byte into `--stage-bundle-dir`, writes empty 0-byte `__init__.py` stubs for parent package directories (preventing `vllm_torchtpu/__init__.py` from importing `torch`), and writes `MANIFEST.sha256`.

```bash
# Auto-detect modified files from git worktree and stage transitive closure:
cdk-g3 route \
  --source-root /path/to/vllm-torchtpu \
  --allow-dependency-closure \
  --topology 1x1x1 \
  --stage-bundle-dir /tmp/cdk_g3_staged_bundle \
  --output-receipt /tmp/cdk_g3_scope_receipt.json
```

---

## 3. The 3 Google3 TPU Backends (`--backend auto|forge|minibench|xmanager`)

### Backend A — Forge TPU (`--backend=forge`, ~30–60s turnaround, 0 XManager quota)
* Dispatches single-host Ghostfish (`gf=1x1x1`, `gf=2x2x1`) via `/google/bin/releases/arca9-local-blaze-cli/blaze-for-agents test -c opt --test_strategy=remote --nocache_test_results --test_output=errors`.
* Built-in recipe: `forge-microbench`.

```bash
cdk-g3 --storage-uri "$CDK_G3_STORAGE_URI" --backend=forge --collect-telemetry \
  job create forge-microbench \
  --tpu-topology gf=2x2x1 \
  --map-dir /tmp/cdk_g3_staged_bundle:scorer:/workspace/tree
```

### Backend B — Minibench / MLCR & Percale LLO Orchestrator (`--backend=minibench`)
* **Pre-Warm MLCR Pools (`--minibench-mode=mlcr`, ~5–15s turnaround)**: Supported on `vlp`, `vlp1t`, `vlp2t`, `glp`, `pf`, and `vf` (`v5e`, `v5p`, `v4`, `v6e` Trillium).
* **`[TRAP36H_MLCR_GHOSTFISH_SILENT_PATHWAYS_FALLBACK]` Guard**: MLCR has **no Ghostfish (`gf` / `tpu7x`) warm pools** (`mlcr_configs.py:24-64`); passing `--mode=mlcr` on `gf` silently falls back to spinning up a cold self-owned Pathways cluster (`usage_guide.md:518-532`). `cdk-g3` blocks `--minibench-mode=mlcr` on `gf` (`rc=2`) and auto-routes (`--minibench-mode=auto`):
  - Single-host Ghostfish (`gf=1x1x1`, `gf=2x2x1`) $\rightarrow$ `local_hermetic` (`//learning/performance/agents/tools/minibench:run_minibench_gf_1x1x1` / `run_minibench_gf_2x2x1`) or `llo_orchestrate` (`//learning/performance/agents/tools/minibench/llo_analysis:llo_orchestrate --enable_percale --generate_runtime_llo_report`).
  - Multi-host Ghostfish (`gf=2x2x2`) $\rightarrow$ `xm_pathways` (`//learning/performance/agents/tools/minibench:launch_minibench_xm`).
* Built-in recipe: `minibench-ab`.

```bash
# Single-host Ghostfish Percale + Runtime LLO Report:
cdk-g3 job create minibench-ab \
  --storage-uri "$CDK_G3_STORAGE_URI" \
  --backend minibench \
  --minibench-mode llo_orchestrate \
  --tpu-topology gf=1x1x1 \
  --reference-code examples/pallas_relu_ref.py \
  --custom-code examples/pallas_relu_cand.py \
  --input-generation-code examples/input_gen.py \
  --enable-percale --generate-runtime-llo-report \
  --collect-telemetry
```

### Backend C — XManager / Borg TPU (`--backend=xmanager`, 1,422 `tpu_ghostfish` chips)
* Required for multi-host `gf=2x2x2` (8 chips / 16 TensorCores) ICI collective sweeps (`[TRAP36E_MULTIHOST_REQUIRES_XMANAGER_BORG]`) and multi-WorkUnit tile sweeps.
* Uses `group:msca-dynamic/tpu-perf-team-dynamic-xm` (`[TRAP36_FORBIDDEN_ABC_LM_ALLOC]` blocks `abc-lm`) and always passes `--noxm_monitor_on_launch` (`[TRAP36C]`).
* Built-in recipe: `xm-sweep`.

```bash
cdk-g3 --storage-uri "$CDK_G3_STORAGE_URI" --backend=xmanager --collect-telemetry \
  job create xm-sweep \
  --tpu-topology gf=2x2x2 \
  -t agentx-ici-sweep
```

---

## 4. Guaranteed 4-Category Telemetry & Compiler Dumps (`--collect-telemetry` & `verify-telemetry`)

Passing `--collect-telemetry` injects `CDK_G3_TELEMETRY_DIR` and appends missing HLO (`--xla_dump_to=.../hlo --xla_dump_hlo_as_text=true --xla_dump_hlo_as_proto=true`) and LLO/Mosaic (`--xla_mosaic_dump_to=.../llo --xla_jf_dump_to=.../llo --xla_jf_dump_llo_text=true --xla_jf_dump_llo_static_gaps=true`) flags while preserving user `XLA_FLAGS` and `LIBTPU_INIT_ARGS`.

After syncing outputs (`cdk-g3 job sync-outputs <job_id>`), verify all 4 artifact categories (`xprof`, `trace_json`, `hlo`, `llo`) are non-empty (`[TRAP36I_MISSING_TELEMETRY_ARTIFACTS]`):

```bash
cdk-g3 job sync-outputs "$JOB_ID" --storage-uri "$CDK_G3_STORAGE_URI"
cdk-g3 verify-telemetry \
  --telemetry-dir "$HOME/cloud-devkit-data/job-outputs/$JOB_ID/telemetry" \
  --output-receipt /tmp/telemetry_receipt.json
```

---

## 5. Drop-In Compatibility across All 4 Agent Harnesses

1. **`tpu-perf-forge`**: Set `export PERFKIT_CDK_BIN=cdk-g3` and `export CDK_G3_STORAGE_URI=gs://forge-gcs-vm/forge/perfkit-cluster/_cdk_g3`. `perfkit/cluster.py` and `perfkit/cdk_arm.py` invoke the exact 8 `cdk` subcommands (`agent-letter`, `usage -o json`, `job create`, `job list`, `job desc`, `job log`, `job delete`, `job sync-outputs`) unchanged.
2. **`vllm-optimization-agent`**: Use `cdk-g3 verify-scope` + `cdk-g3 --backend=minibench` / `--backend=forge` for fast Stage-A/B kernel probes and Percale LLO reports before running `agentx-bench`.
3. **Native Jetski (`vllm-torchtpu`)**: Automatically loaded from `~/.gemini/skills/cdk-g3-microbench/SKILL.md` and `.gemini/skills/cdk-g3-microbench/SKILL.md`.
4. **`claude-tpu-opt` (`tpu_autonomous_loop`)**: Integrated with `sweep_runner/cdk_g3_bridge.py` and `skills/forge-xmanager-tpu-microbench/SKILL.md`.
