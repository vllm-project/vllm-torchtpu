# q38-best-per-band: one branch for every band

This branch is Benliang Wang's head (`5958dbfb7`) plus the Q3.8 PD team's two serving stacks. Every point on the dash sheet `q38-pd-ixc-frontier-1118` runs from it.

## Pick a form with one variable

The plugin ships in two complete forms. Each process picks one when `vllm_torchtpu` is first imported.

| `Q38_OXPD_FORM` | Form | What runs |
|---|---|---|
| unset or `0` (default) | Benliang's form | `src/vllm_torchtpu/` as on Benliang's head, plus our R-form stack (rfix + rw3). Every R-form addition is behind a default-off flag, except the IX #3079 prefill-only padding fix, which acts only when the prefill leg runs the MTP drafter. |
| `1` | oxpd form | `src/vllm_torchtpu/_oxpd/`, a byte-for-byte copy of our oxpd tree (commit `2bc75911f5`, "w3"). |
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

## Exos: required before running B4–B6 from this branch

- Every Exos pod runs the recipe's `_EXOS_PYPATCH` at start. It rewrites 7 plugin files it finds as `<dir>/vllm_torchtpu/<file>`.
- On this branch that patches the copies in `src/vllm_torchtpu/`, **not** the oxpd form's copies in `_oxpd/`. An oxpd-form run on Exos would therefore miss the Exos fixes.
- A launcher that also applies the patch to `_oxpd/` is being built. Until it exists, run B4–B6 on Exos from their run commit `2bc75911f5`.
- The proxy script on Exos comes from the image (`/root/torchtpu-vllm/examples/...`), so this branch's `examples/` does not affect Exos runs.

## Tests

- Benliang's form: `PYTHONPATH=src pytest tests/...`, as before.
- oxpd form: `Q38_OXPD_FORM=1 PYTHONPATH=src pytest tests_oxpd/...`. `tests_oxpd/` is w3's `tests/`, byte for byte. It is outside pytest's `testpaths`, so a plain `pytest` does not collect it.

## Changing code

- **Benliang's form:** edit `src/vllm_torchtpu/` as usual, and put new behaviour behind a default-off flag.
- **oxpd form:** edit `src/vllm_torchtpu/_oxpd/`, whose paths mirror w3's `src/vllm_torchtpu/`.
- **Porting a branch built on w3 into the oxpd form:** apply its diff with a path prefix. Do **not** `git merge` it, because git would treat it as a merge of the whole 900-commit oxpd fork.

  ```
  git diff 2bc75911f5 <branch> -- src/vllm_torchtpu | git apply -p3 --directory=src/vllm_torchtpu/_oxpd
  git diff 2bc75911f5 <branch> -- tests | git apply -p2 --directory=tests_oxpd
  ```

- **Moving an oxpd feature into Benliang's form:** port it into `src/vllm_torchtpu/` behind a default-off lever flag, with its own flag-off identity proof. When B4–B6 run without the gate, `_oxpd/` can be deleted.

## Why two forms instead of one merged tree

- The oxpd stack is about 900 commits whose only common base with Benliang's line is Jiaxin's `a3079238c0` (2026-11-05).
- A line merge leaves 43 files with 260 conflict hunks, even with the best base for each file. The hotspots are `tpu_runner.py`, `envs.py`, `fused_moe_decode_kernel.py` and `qsa_rpa.py`.
- Gating each hunk could not be checked by the available tests.
- The overlay is exact by construction: each form runs byte-identical code to the tree it came from.
