# q38 dash points on this branch (MERGE-1124, 2026-11-24)

This branch is `caojx/qwen38-flash-1124-2` plus the code behind every q38 dash point that beats the sheet
`mbpf-q38pf1119-c1-stack-d3600-bestband` on dash d-5234ec7a (24 points, on sheets `q38-pd-ixc-frontier-1118` and
`q38-1p1d-curve-1122`). Each point reproduces **by env**: two selector keys pick the exact code the point ran.

## Your defaults do not change

- With `Q38_FORM` and `Q38_PROXY_FORM` unset, the package and the proxy run exactly as on `caojx/qwen38-flash-1124-2`.
- Two of your files are edited, and only to add the selectors: `src/vllm_torchtpu/__init__.py` (the form gate) and
  `examples/disagg/toy_proxy_server.py` (a gate right after its imports). Everything else is new files.
- `Q38_OXPD_FORM=1` still selects the w3 oxpd form, with the same log line.

## The two selector keys

- `Q38_FORM=<name>` runs `src/vllm_torchtpu/_forms/<name>/vllm_torchtpu/`, the point's own `vllm_torchtpu` package
  byte for byte, the same way `Q38_OXPD_FORM=1` runs the oxpd form. Log line:
  `q38 form gate: Q38_FORM=<name>, vllm_torchtpu runs the <name> form from <dir>`. It cannot be combined with
  `Q38_OXPD_FORM=1`.
- `Q38_PROXY_FORM=<name>` runs `examples/disagg/_q38_proxy_forms/toy_proxy_server_<name>.py`, the point's own toy
  proxy byte for byte, in place of `toy_proxy_server.py`. Log line: `q38 proxy form gate: Q38_PROXY_FORM=<name> runs
  <file>`.
- Set both on **every** container of the job: both legs, and the proxy container. On Exos the proxy runs inside the
  prefill container.

| Form | Source tree | Package tree (git) |
|---|---|---|
| `Q38_OXPD_FORM=1` (oxpd, unchanged) | w3 `2bc75911f5` | `69b1b47835` |
| `Q38_FORM=k_cut1` | CUT 1 `cf812d1e5b` | `884ce6aa3c` |
| `Q38_FORM=k_pkg` | B3 package `9f0c924d16` (= `139d15a14e`) | `28f4c88f40` |
| `Q38_FORM=k_sup` | B3 supersede `cc23d50903` | `15bc882819` |
| `Q38_FORM=oxk` | K3-ox.1 `1ea5a165a7` overlay (= `859e11f16e`) | `033a46e02a` |
| `Q38_FORM=d1121` | `d8f139fa` (= `caojx/qwen38-flash-1121`), default form | `47ed2c5504` (= `9cd151fab5` without `_forms/`, `__init__.py` = its `_bl_init.py`) |
| `Q38_FORM=k_sb11` | SB1-1 served head `17739744d1` (B1, posted 2026-11-25 03:08Z) | `577f761fba` |

| Proxy form | Blob | Source |
|---|---|---|
| `w3` | `7c1565a15e` | w3 `2bc75911f5` |
| `k` | `2a821e401b` | `9f0c924d16` (also CUT 1, K3-ox.1, `d8f139fa`) |
| `opener` | `0532c97577` | `139d15a14e` (`PROXY_AFFINITY_KEY=opener`, header-free sticky key) |

## The 24 points

SRC / TREE / N are the team pins of the tree each job received (its own `mapped-dirs/00-vllm-torchtpu.tar.gz`; every
one is file-for-file identical to the commit). Env delta = what to add to the point's recorded env.

| Band | Point | Mtok/$ @ x (TTFT p90) | Job | Commit | SRC / TREE / N | Env delta |
|---|---|---|---|---|---|---|
| B1 | T58x c212 | 297.96 @ 78.67 (6.53 s) | j-1070e35b | `cf812d1e5b` | 3ece92df2c / fa328adf05ce2821 / 1665 | `Q38_FORM=k_cut1 Q38_PROXY_FORM=k` |
| B3 | c216 supersede | 230.44 @ 156.03 (8.92 s) | j-d2fd8be6 | `cc23d50903` | e5c2427026 / 0d6b758d2520ca7a / 1658 | `Q38_FORM=k_sup Q38_PROXY_FORM=opener` |
| B3 | c208 cache -i2 | 228.72 @ 158.46 (7.46 s) | j-3e719845 | `139d15a14e` | 2f202ee68f / f98b2c6f6e48b0f2 / 1655 | `Q38_FORM=k_pkg Q38_PROXY_FORM=opener` |
| B3 | 1P1D r09i2 c92 | 146.71 @ 150.66 (7.62 s) | j-692bf921 | `d8f139fa` | 4bb42335e7 / 132fb2527f191fb1 / 3472 | `Q38_FORM=d1121 Q38_PROXY_FORM=k` |
| B4 | c192 cache | 219.00 @ 160.28 (4.74 s) | j-ae47c037 | `139d15a14e` | 2f202ee68f / f98b2c6f6e48b0f2 / 1655 | `Q38_FORM=k_pkg Q38_PROXY_FORM=opener` |
| B4 | c200spk (T55d) | 217.25 @ 160.32 (7.60 s) | j-57e28d43 | `9f0c924d16` | 2f202ee68f / bcc7c08022f5e092 / 1654 | `Q38_FORM=k_pkg Q38_PROXY_FORM=k` |
| B4 | T54f c192 | 215.40 @ 162.06 (5.40 s) | j-06309a47 | `9f0c924d16` | 2f202ee68f / bcc7c08022f5e092 / 1654 | `Q38_FORM=k_pkg Q38_PROXY_FORM=k` |
| B4 | Exos P2C128 | 139.93 @ 169.82 (7.95 s) | j-38139aef | `2bc75911f5` | f696bcf585 / b05f45626a164c20 / 2271 | `Q38_OXPD_FORM=1 Q38_PROXY_FORM=w3` |
| B4 | Exos p2c112 | 131.37 @ 176.45 (4.68 s) | j-4bf73187 | `2bc75911f5` | same | `Q38_OXPD_FORM=1 Q38_PROXY_FORM=w3` |
| B4 | 1P1D r07 c76 | 122.93 @ 164.40 (9.24 s) | j-e73ead5f | `d8f139fa` | 4bb42335e7 / 132fb2527f191fb1 / 3472 | `Q38_FORM=d1121 Q38_PROXY_FORM=k` |
| B5 | Exos p2c104m12r64k48 | 117.62 @ 238.97 (9.85 s) | j-8f66f71f | `2bc75911f5` | f696bcf585 / b05f45626a164c20 / 2271 | `Q38_OXPD_FORM=1 Q38_PROXY_FORM=w3` |
| B5 | Exos p2c96m12r64k48 | 109.61 @ 240.61 (7.19 s) | j-c325b23f | `2bc75911f5` | same | `Q38_OXPD_FORM=1 Q38_PROXY_FORM=w3` |
| B5 | 1P1D l04 c56 | 99.24 @ 205.72 (7.04 s) | j-0972cf08 | `2bc75911f5` | same | `Q38_OXPD_FORM=1 Q38_PROXY_FORM=w3` |
| B5 | 1P1D x3-l03 c48 | 84.68 @ 240.30 (5.54 s) | j-4cbd2b9c | `2bc75911f5` | same | `Q38_OXPD_FORM=1 Q38_PROXY_FORM=w3` |
| B6 | Exos p3c88 | 80.46 @ 276.22 (2.29 s) | j-ebb63496 | `2bc75911f5` | same | `Q38_OXPD_FORM=1 Q38_PROXY_FORM=w3` |
| B6 | Exos p3c80 | 75.05 @ 288.67 (2.21 s) | j-15ca9825 | `2bc75911f5` | same | `Q38_OXPD_FORM=1 Q38_PROXY_FORM=w3` |
| B6 | 1P1D l02 c36 | 67.36 @ 316.19 (3.54 s) | j-04b73c61 | `2bc75911f5` | same | `Q38_OXPD_FORM=1 Q38_PROXY_FORM=w3` |
| B6 | Exos p3c72 | 64.57 @ 321.27 (5.11 s) | j-a8bff01e | `2bc75911f5` | same | `Q38_OXPD_FORM=1 Q38_PROXY_FORM=w3` |
| B6 | HIX P1 p3c66m | 62.64 @ 408.48 (4.19 s) | j-6b77f816 | `1ea5a165a7` | 68d7bbd856 / de559b7c62e6fd52 / 3479 | drop `Q38_OXPD_FORM=1`; `Q38_FORM=oxk Q38_PROXY_FORM=k` |
| B6 | HIX-o p4c70m (KFO) | 52.81 @ 421.24 (4.35 s) | j-1e8b99d3 | `859e11f16e` | 68d7bbd856 / b30ec5e4bf4cb2d0 / 3480 | drop `Q38_OXPD_FORM=1`; `Q38_FORM=oxk Q38_PROXY_FORM=opener` |
| B6 | 1P1D l01 c24 | 51.70 @ 408.13 (5.83 s) | j-41e60c33 | `2bc75911f5` | f696bcf585 / b05f45626a164c20 / 2271 | `Q38_OXPD_FORM=1 Q38_PROXY_FORM=w3` |
| B6 | HIX-o p4c64m | 48.79 @ 423.91 (3.41 s) | j-74e61ed5 | `859e11f16e` | 68d7bbd856 / b30ec5e4bf4cb2d0 / 3480 | drop `Q38_OXPD_FORM=1`; `Q38_FORM=oxk Q38_PROXY_FORM=opener` |
| B6 | HIX-o p4c56m | 43.91 @ 426.27 (2.55 s) | j-64d5ef35 | `859e11f16e` | same | drop `Q38_OXPD_FORM=1`; `Q38_FORM=oxk Q38_PROXY_FORM=opener` |
| B6 | HIX-o p4c40m | 31.45 @ 437.66 (1.45 s) | j-52573012 | `859e11f16e` | same | drop `Q38_OXPD_FORM=1`; `Q38_FORM=oxk Q38_PROXY_FORM=opener` |

**Posted 2026-11-25 03:08Z on `q38-pd-ixc-frontier-1118`** (it untagged B1 T58x c212 `k_cut1`, j-1070e35b, as dominated; that row stays
here so its point remains reproducible).

| Band | Point | Mtok/$ @ x (TTFT p90) | Job | Commit | SRC / TREE / N | Env delta |
|---|---|---|---|---|---|---|
| B1 | SB1-1 c212 (posted 2026-11-25 03:08Z) | 307.30 @ 83.28 (5.75 s) | j-4be755be | `17739744d1` | 0d6c7d50a3 / e7753a66e8750121 / 1709 | `Q38_FORM=k_sb11 Q38_PROXY_FORM=k` |

Bands are by p90 interactivity: B1 ≤ 110, B2 110-140, B3 140-160, B4 160-205, B5 205-275, B6 ≥ 275 tok/s/user.
The 1P1D sheet is performance of record with accuracy uncertified. Routing disclosures (sticky decode routing on the
B3/B4 and HIX points, header-free `opener` key on the cache cars and HIX-o) are as recorded in each point's env.

## Living forms (where new q38 work lands)

The forms above are frozen: each is exactly what one dash point ran, and none is ever edited. New q38 work lands in a
living form, one per lineage, env-keyed and default-off, so the team builds on one branch:

| Form | Lineage | Package tree (git) | Lever keys (all default off) |
|---|---|---|---|
| `Q38_FORM=b1` | SB1-1 `17739744d1` + QGLUE-1 B1 `10f346d259` (clean merge) | `5140ebd562` | QGLUE-1: `VLLM_TPU_QSA_QCHAIN=1`, `VLLM_TPU_QSA_QCHAIN_EDITS=124`, `VLLM_TPU_QSA_PREFILL_RING_PRODUCER=1`, `VLLM_TPU_QSA_PREFILL_EPI_FOLD=1`; SB1-1's stack keys as in its row above |

- `b1` is a merge that has not yet run on a device: no served car stacks QGLUE-1 with SB1-1 on `b1` before one device
  bitwise check on `b1`.
- To work on a lineage: branch from `caojx/qwen38-flash-1124`, edit `src/vllm_torchtpu/_forms/<form>/vllm_torchtpu/`,
  and run with `Q38_FORM=<form>`. Register every new key in that form's `envs.py`, so it enters the compile key.
- Dash-bound full runs run from a pushed commit of `caojx/qwen38-flash-1124` only.
- A living form runs its own package: it does not get changes made to the main package (for example
  `VLLM_TPU_SHARED_BACKEND_CACHE`). Folding the living forms into the main package is the planned convergence step.

## Notes

- `src/sitecustomize.py` here adds the `CDK_MEMLOG` logger; it does nothing unless `CDK_MEMLOG=1`, which no point sets.
- `src/tpu_sync/` (vendored Python sources from the c1-stack-d3600 snapshot, no `__init__.py`) is outside every form;
  see the MERGE-1124 report for how it resolves on the serving image.
- Later step, not done here: fold the K forms (`k_cut1`, `k_pkg`, `k_sup`) into the main package behind flags, each with
  its own flag-off proof, and retire forms as they fold.
