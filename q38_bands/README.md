# q38-best-per-band: one branch for every band

This branch is Benliang Wang's head (`5958dbfb7`) plus the Q3.8 PD team's two serving stacks. Every point on the dash sheet `q38-pd-ixc-frontier-1118` runs from it.

## Pick a form with one variable

The plugin ships in two complete forms. Each process picks one when `vllm_torchtpu` is first imported.

| `Q38_OXPD_FORM` | Form | What runs |
|---|---|---|
| unset or `0` (default) | Benliang's form | `src/vllm_torchtpu/` as on Benliang's head, plus our R-form stack (rfix + rw3). Every R-form addition is behind a default-off flag, except the IX #3079 prefill-only padding fix, which acts only when the prefill leg runs the MTP drafter. One line was added to Benliang's `compilation/tpu_compiler.py`: `"vllm_torchtpu/_forms"` in `_RUNTIME_CACHE_KEY_EXCLUDED_PATHS`. No hashed path contains that directory, so no compile hash changes. |
| `1` | oxpd form | `src/vllm_torchtpu/_forms/oxpd/vllm_torchtpu/`, a byte-for-byte copy of our oxpd tree's `src/vllm_torchtpu` (commit `2bc75911f5`, "w3"). |
| anything else | refused | `vllm_torchtpu` raises at import. |

- Set the variable in the environment of **every** process that imports `vllm_torchtpu`, on every leg: prefill, each decode replica, and the proxy if yours imports the plugin.
- In the oxpd form, each process prints `q38 form gate: Q38_OXPD_FORM=1, vllm_torchtpu runs the oxpd form from <path>` to stderr. Check for that line in every leg's log.
- **The forms cannot be mixed in one run.** A Benliang feature and an oxpd feature can be combined only after one of them is ported into the other form.

## Which form serves which band

| Band | Point | Mtok/$ | x (tok/s/user, p90) | TTFT p90 | Form | Recipe and flags | Run commit (provenance) |
|---|---|---|---|---|---|---|---|
| B1 | RK2144 | 199.89 | 82.24 | 9.200 s | Benliang's | `infx-qwen38-flash-agentx-disagg`, `B1_RK2144.flags.txt` | `d16384fe6` |
| B2 | RM4096 | 145.71 | 120.55 | 7.17 s | Benliang's | same recipe, `B2_RM4096.flags.txt` | `d16384fe6` |
| B3 | RB374d | 119.50 | 142.64 | 7.128 s | Benliang's | same recipe, `B3_RB374d.flags.txt` | `8247b68f7` |
| B4 | P2C128 | 139.93 | 169.82 | 7.952 s | oxpd | `exos-infx-qwen38-flash-agentx-disagg`, `B4_P2C128.flags.txt` | `2bc75911f5` |
| B4 | p2c112 | 131.37 | 176.45 | 4.677 s | oxpd | same, `B4_p2c112.flags.txt` | `2bc75911f5` |
| B4 | P2C96 | 112.64 | 188.34 | 2.669 s | oxpd | same, `B4_P2C96.flags.txt` | `2bc75911f5` |
| B5 | p2c96m12r64k48 | 109.61 | 240.61 | 7.194 s | oxpd | same, `B5_p2c96m12r64k48.flags.txt` | `2bc75911f5` |
| B6 | P3C48 | 46.95 | 372.00 | 1.330 s | oxpd | same, `B6_P3C48.flags.txt` | `2bc75911f5` |

- B1–B3 ran on the default cluster (8 chips, 1P1D). B4–B5 ran on Exos (12 chips, 1P2D), and B6 on Exos (16 chips, 1P3D).
- Each `*.flags.txt` lists every engine flag the dash job set explicitly, per leg, from its stored recipe. Anything not listed runs at the form's default.
- For B4–B6, add `Q38_OXPD_FORM=1` to every leg.

## Disclosure: VLLM_TPU_MTP_DRAFT_ROWS and IX admissibility

All six curve-of-record configurations run `VLLM_TPU_MTP_DRAFT_ROWS` on the decode leg; B4–B6 also run it on the prefill leg. Its IX admissibility requires a bit-identity proof (`reports/ix_draft_rows_audit_1119.md`). Measured so far: on the prefill leg it is **not** bit-identical; on the decode legs it is bitwise at op level. A strict TPU bitwise test is being built.

## Exos: required before running B4–B6 from this branch

- Every Exos pod runs the recipe's `_EXOS_PYPATCH` at start. It rewrites 7 plugin files it finds as `<dir>/vllm_torchtpu/<file>`.
- On this branch that patches the copies in `src/vllm_torchtpu/`, **not** the oxpd form's copies in `src/vllm_torchtpu/_forms/oxpd/vllm_torchtpu/`. An oxpd-form run on Exos would therefore miss the Exos fixes.
- **Option (a), proven on CPU and approved:** stage the tree pre-patched. Every Exos pod already runs the pypatch, and it is idempotent, so the recipe can stay unchanged.
  - The patched files come from a CPU-job artifact, `q38_oxpd_exos_prepatch_656402ca0a.tgz`. It holds 14 files: the 7 the pypatch rewrites, plus the same 7 under `_forms/oxpd/vllm_torchtpu/`, with a sha256 `MANIFEST.txt`.
  - Extract it over this branch's `src/`. Never run the pypatch on the VM, because it also writes into site-packages.
  - Measured on CPU: the oxpd copy then hashes to `031145cfb0`, exactly the post-bootstrap tree the B4–B6 dash pods ran. The whole tree hashes to `0ec8c4dd5a`, and an unmodified pod pypatch leaves it unchanged.
  - So the launcher's q38x guard pins are pre = post = `0ec8c4dd5a` for this branch's source (src hash `bdc8584bb2`). Any later code commit changes them.
- The launcher change itself is owned by the tuning/train-ops lanes. Until their variant exists, run B4–B6 on Exos from their run commit `2bc75911f5`.
- The proxy script on Exos comes from the image (`/root/torchtpu-vllm/examples/...`), so this branch's `examples/` does not affect Exos runs.

## Tests

- Benliang's form: `PYTHONPATH=src pytest tests/...`, as before.
- oxpd form: `Q38_OXPD_FORM=1 PYTHONPATH=src pytest tests_oxpd/...`. `tests_oxpd/` is w3's `tests/`, byte for byte. It is outside pytest's `testpaths`, so a plain `pytest` does not collect it.

## Caveat: some `tests_oxpd` files read sources by repo path

- These 43 files locate source files by a path relative to the repository, for example `Path(__file__).parents[2] / "src/vllm_torchtpu/..."`. So does `runner/test_qwen4_exp_pd_shared_wiring.py`, through its helper `runner/test_qwen4_exp_pd_route.py`.
- In this branch those paths resolve to `src/vllm_torchtpu/`, which is **Benliang's form**, not the oxpd copy.
- Where such a path only puts `src` on `sys.path` before `import vllm_torchtpu`, it is fine, because the import goes through the gate.
- Where it reads or loads a specific file by path, the test is **not a valid oxpd-form test** until that site is fixed.
- The fix is a tracked follow-up. These files were not part of the oxpd-form proof.

- `tests_oxpd/decfb2/q38_decfb2_vpsfb_mutrun.py`
- `tests_oxpd/decfb2/q38_decfb2_vpsfb_wiring_mutrun.py`
- `tests_oxpd/kernels/qwen4_exp/dcp2/test_o1s_hoist.py`
- `tests_oxpd/l4vfy/vfy_harness.py`
- `tests_oxpd/models/vllm/test_qwen4_exp_ple_host.py`
- `tests_oxpd/models/vllm/test_qwen4_exp_prefix_granularity.py`
- `tests_oxpd/models/vllm/test_qwen4_exp_stage_pages.py`
- `tests_oxpd/platforms/test_patch_registry.py`
- `tests_oxpd/q38_draft2/test_draft_embed_h8.py`
- `tests_oxpd/q38_draftemb/test_draft_embed_repl.py`
- `tests_oxpd/q38_draftemb/test_target_embed_repl.py`
- `tests_oxpd/q38_l1g1/test_g1_runner_hooks.py`
- `tests_oxpd/q38_l3_d60/test_compact_pipe_wiring.py`
- `tests_oxpd/q38_l4_foldfix/test_foldfix_flagoff.py`
- `tests_oxpd/q38_l4_samp2v/test_l4_samp2.py`
- `tests_oxpd/q38_l4_samp2v/test_samp2_vfy.py`
- `tests_oxpd/q38_l4_seedfix/test_seedfix.py`
- `tests_oxpd/q38_l4_v5c/test_v5c_flagoff.py`
- `tests_oxpd/q38_l4_vpshost/test_l4_vpshost.py`
- `tests_oxpd/q38_l4agree/test_l4agree_flagoff.py`
- `tests_oxpd/q38_l6/test_l6_fskip.py`
- `tests_oxpd/q38_l6/test_l6_nodefer.py`
- `tests_oxpd/q38_l6/test_l6_pmeter.py`
- `tests_oxpd/q38_l6instr/test_l6instr_submarks.py`
- `tests_oxpd/q38_moekv/test_compact_rows_wiring.py`
- `tests_oxpd/q38_samp1/test_vps_topk_keys.py`
- `tests_oxpd/q38_stallhunt/test_vfy_warm_alias_real.py`
- `tests_oxpd/q38_stallhunt/test_vfy_warm_alias.py`
- `tests_oxpd/q38mm_dstock/q38mm_vps_mutrun.py`
- `tests_oxpd/q38mm_dstock/q38mm_vps_wiring_mutrun.py`
- `tests_oxpd/runner/test_host_phases_subs.py`
- `tests_oxpd/runner/test_kv_group_size.py`
- `tests_oxpd/runner/test_postfwd_fused.py`
- `tests_oxpd/runner/test_qwen4_exp_pd_repack_l2.py`
- `tests_oxpd/runner/test_qwen4_exp_pd_repack_v2.py`
- `tests_oxpd/runner/test_qwen4_exp_pd_route.py`
- `tests_oxpd/runner/test_qwen4_exp_pd_shared_compressed.py`
- `tests_oxpd/runner/test_qwen4_exp_pd_staging.py`
- `tests_oxpd/runner/test_tpu_runner_step_seq_lens.py`
- `tests_oxpd/test_dp_affinity_shared.py`
- `tests_oxpd/test_dp_affinity.py`
- `tests_oxpd/test_logger_tree.py`
- `tests_oxpd/worker/test_tpu_worker.py`

## Changing code

- **Benliang's form:** edit `src/vllm_torchtpu/` as usual, and put new behaviour behind a default-off flag.
- **oxpd form:** edit `src/vllm_torchtpu/_forms/oxpd/vllm_torchtpu/`, whose paths mirror w3's `src/vllm_torchtpu/`. Keep that directory layout: w3's compile-cache key hashes kernel sources relative to the directory above `vllm_torchtpu/`, so this layout gives the oxpd form exactly w3's source hash.
- **Porting a branch built on w3 into the oxpd form:** apply its diff with a path prefix. Do **not** `git merge` it, because git would treat it as a merge of the whole 900-commit oxpd fork.

  ```
  git diff 2bc75911f5 <branch> -- src/vllm_torchtpu | git apply -p3 --directory=src/vllm_torchtpu/_forms/oxpd/vllm_torchtpu
  git diff 2bc75911f5 <branch> -- tests | git apply -p2 --directory=tests_oxpd
  ```

- **Moving an oxpd feature into Benliang's form:** port it into `src/vllm_torchtpu/` behind a default-off lever flag, with its own flag-off identity proof. When B4–B6 run without the gate, `src/vllm_torchtpu/_forms/` can be deleted.

## Why two forms instead of one merged tree

- The oxpd stack is about 900 commits whose only common base with Benliang's line is Jiaxin's `a3079238c0` (2026-11-05).
- A line merge leaves 43 files with 260 conflict hunks, even with the best base for each file. The hotspots are `tpu_runner.py`, `envs.py`, `fused_moe_decode_kernel.py` and `qsa_rpa.py`.
- Gating each hunk could not be checked by the available tests.
- The overlay is exact by construction: each form runs byte-identical code to the tree it came from.

## What was verified before this branch was pushed

All proofs ran as CPU jobs with the pass rules written down before launch. Base trees: Benliang's head `5958dbfb7`, rw3 `8247b68f7`, w3 `2bc75911f5`.

- **Bytes.** The oxpd copy equals w3's `src/vllm_torchtpu` file for file (650 files). Benliang's old `__init__.py` is byte-identical in `_bl_init.py`. Outside the oxpd copy, the source tree equals Benliang's except the 18 files of the rfix + rw3 merge.
- **Which code runs.** With the gate unset, every loaded `vllm_torchtpu` module comes from `src/vllm_torchtpu/`, byte-equal to Benliang's outside the merge. With `Q38_OXPD_FORM=1`, every module comes from the oxpd copy and matches a w3 checkout's own run. Five deliberate breaks of the gate were all caught.
- **Compile keys.** In the oxpd form, the compile-cache env factors under every B4–B6 point's env equal w3's, and the hashed kernel-source set equals w3's (270 files). In Benliang's form, the hashed source set gains only the 3 new R-form kernel files, and the env factors gain 6 default-off R-form keys. So a gate-unset run compiles cold once instead of reusing Benliang's caches.
- **Merge defect found and fixed.** The rw3 merge was textually clean but had broken `union_sc_bits` (a `NameError` on every call). After the fix, an undefined-name check over all source files matches the parent trees exactly. `union_sc_bits` traces to Benliang's exact program with `VLLM_TPU_QSA_PREFILL_UNION_SC_BITS_PLAN` off, and to rw3's exact program with it on.
- **Regression, Benliang's form.** Benliang's CPU suite (HC-FIX's 216-file set plus the R-form tests) shows 0 new failures against his head: 225 files in total.
  - `tests/kernels/test_qwen4_exp_union_walk_prefill.py` was compared in two parts. 101 tests ran in one job before its 7,200 s cap, and the last 5 ran in a second job. Head passed 106/106 (Benliang's 88 plus rw3's 18) and base passed 88/88, with 0 new failures.
  - The 8 `test_zero_prefix_fast_lifecycle` cases are TPU-only (`@pytest.mark.tpu_test`) and were not exercised on CPU.
- **Regression, oxpd form.** w3's core CPU suites: 120 files, run from `tests_oxpd/` with `Q38_OXPD_FORM=1` against a w3 checkout. 119 files show 0 new failures:
  - 87 + 27 files in the main runs;
  - the heavy files `test_portbl_moe_exact_width`, `test_lean_glue3` and `test_lean_pd_geometry`, each run alone;
  - `test_qwen4_exp_indexer_dense_tail_portbl`: head 27 passed / 10 skipped vs base, no new failure;
  - `megakernel/test_l1l47f.py`: base and head each passed 24/24 when run alone (about 62–64 min per tree).
  - Every loaded module came from the oxpd copy.
  - The 120th file, `runner/test_qwen4_exp_pd_shared_wiring.py`, was excluded because it loads sources by repo path (see the caveat above).
