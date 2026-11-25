# q38 dash points on this branch (MERGE-1124, 2026-11-24)

This branch is `caojx/qwen38-flash-1124-2` plus the code behind every q38 dash point that beats the sheet
`mbpf-q38pf1119-c1-stack-d3600-bestband` on dash d-5234ec7a (24 points, on sheets `q38-pd-ixc-frontier-1118` and
`q38-1p1d-curve-1122`). Each point reproduces **by env**: two selector keys pick the exact code the point ran.

## One change to your main package: the GDN LOG-conv correctness fix (no key)

One commit on this branch, "GDN v3: zero the LOG-mode conv log past a partial window; tile-pad the BATCHED metadata",
changes your main package's GDN v3 kernel with no env key, because it is a correctness fix:

- `kernels/gdn/v3/wrapper.py`: in the factored and the recurrent LOG branches, the conv inputs a tile logs keep window row
  `1 + t` only while `1 + t < real_size`; the rows the act-in DMAs never fill were logged from stale VMEM and now log zero.
- `kernels/gdn/v3/metadata.py` (`compute_batched_seq_metadata`): the per-slot arrays are padded to whole tiles, so a seat
  count that is not a multiple of the tile no longer reads past their end (padding slots are invalid: 0 rows, no state read
  or write).

q38 found it on device (GDN-SMOKE j-ab697a88) and fixed it on its own lineage (b35a4bc3c2 + d4f63efc18; GDN-SMOKE re-run
j-be5e3524 SMOKE-PASS, 12 cases). `git revert <that commit>` undoes it alone. The q38 forms are not touched: each still runs
exactly the code its dash point ran.

Known TPU finding on this branch, not caused by this fix: 13 test ids fail on TPU both with and without it (q38 TPU
coverage run tpucov3 j-bb072340-2900-4cca-81f2, verdict `COV-PASS new=0 fw=FW-COMMON`):

- 7 are `tests/kernels/gdn/v3/test_factored_window.py::TestFactoredBody::test_matches_recurrence` (normal-split,
  normal-vpu, tiny-split, tiny-vpu, underflow-split, zero-split, zero-vpu). On CPU the file passes on both trees (q38
  CPU job gfcovst4 j-c921db49-5e7b-419f-ab9c: `GFCOV FW FW-CLEAN file=tests/kernels/gdn/v3/test_factored_window.py
  fix=COMPLETE base=COMPLETE regression=0 ... -> no failure on either tree`).
- The other 6: 5 parametrizations of `tests/kernels/test_gdn_bf16_qkv_2d.py::test_q38_pooled_prefill_2d_rows_bitwise`
  and 1 of `tests/kernels/test_gdn_pooled_prefill_parity.py` (five-step:
  `TestProductionShapeDenseVsUnifiedPool::test_prefill_five_step_correctness_and_timing`).

## Your defaults do not change

- With `Q38_FORM` and `Q38_PROXY_FORM` unset, the package and the proxy run exactly as on `caojx/qwen38-flash-1124-2`,
  except the GDN v3 LOG-conv / BATCHED-metadata correctness fix (`aae3e349b3`), which changes the main-package kernel
  with no env key and changes the `kernels/` compile-cache key; reproduce Jiaxin's earlier points from
  `caojx/qwen38-flash-1124-2`.
- Two of your files are edited only to add the selectors: `src/vllm_torchtpu/__init__.py` (the form gate) and
  `examples/disagg/toy_proxy_server.py` (a gate right after its imports). The GDN fix above edits two more
  (`kernels/gdn/v3/wrapper.py`, `kernels/gdn/v3/metadata.py`). Everything else is new files.
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
| `Q38_FORM=b1` | SB1-1 `17739744d1` + QGLUE-1 B1 `10f346d259` + SB1-2 no-GDN `00f9aa3066` + SERVE HD rider `3d12a2ccb6` + R4XB walk ring `70664cdd56` (port `fec8a2526a`) + SERVE r0pin `37da395355` (clean merges) + the GDN LOG-conv fix (CORE `b35a4bc3c2` + F1 hand-port; no key) | `a5ef57de53` | QGLUE-1: `VLLM_TPU_QSA_QCHAIN=1`, `VLLM_TPU_QSA_QCHAIN_EDITS=124`, `VLLM_TPU_QSA_PREFILL_RING_PRODUCER=1`, `VLLM_TPU_QSA_PREFILL_EPI_FOLD=1`; SB1-1's stack keys as in its row above; SB1-2: `VLLM_TPU_QSA_DECODE_FUSED_UNION=1`, `VLLM_TPU_QSA_INDEXER_PADDED_ROWS=1`, `VLLM_TPU_PD_GATHER_BULK=1`, `VLLM_TPU_PD_STAGING_FOLD=1`, `VLLM_TPU_GC_WORKER=2600`, `VLLM_TPU_GC_WORKER_LOG=1`, `VLLM_TPU_KVL_INWAIT=1` (the PD / GC / KVL keys are deliberately outside the compile key: their runs prove a cold compile); HD rider: `VLLM_TPU_HD_STAGE_CPU=1`, `VLLM_TPU_HD_REQVEC=1` only (`VLLM_TPU_HD_MARKS` and `VLLM_TPU_HD_AB` unset; host-only, compile-ignored); R4XB: `VLLM_TPU_QSA_PREFILL_UNION_WALK_RING=3` or `4` (0 = off; registered and in the compile key, so `b1` compile caches miss once after this lands) ; r0pin arm c (decode only): `VLLM_TPU_CPU_PIN_RESERVE=24`, `VLLM_TPU_CPU_PIN_SKIP_CPUS=0-23` with `VLLM_TPU_CPU_PIN=slice` (RESERVE, SKIP_CPUS and `VLLM_TPU_CPU_IRQ_DUMP_S` are compile-ignored) |
| `Q38_FORM=hix` | HIX H `cfbd1689c1` (`q38-hix700-s3a-h`: S3a = S2 stack + PREGATE (flag off) + MKH + H2 + KV-SCALE-GUARD v2, then MESHGUARD, SIDE-FIX, T1 fix, D-b, PLE-WINMASK), its `_forms/oxpd` package byte for byte, + the HIX bundle `7eebc74eef` (7 commits, every key default off) | `42d3fdde6e` | H's served decode env (194 items, with `Q38_FORM=hix` in place of `Q38_OXPD_FORM=1`); proxy `Q38_PROXY_FORM=opener` (the served proxy map holds blob `0532c97577`) |
| `Q38_FORM=b3` | STKTREE L3S `7321c4e60e` = QGLUE-1 B3 `7fd4624a18` + SB3-1 / B3STACK `fe809505ac` (clean merge; no PLE) | `5793ea2980` | QGLUE-1 B3 and SB3-1 keys as in their lineages (all default off) |

- `hix` disclosures:
  - `VLLM_TPU_MK_GDN_SIDECAR` stays default off and is **NOT SERVABLE at MK buckets 8:16**: the one-trace bucket-8
    ShapeSpecializationError (`_side_active`, `qwen38_wholestep.py:503-513`) and the operand misalignment at buckets=16
    (IndexError, `qwen38_segment_call.py:114`). K1 is parked with it.
  - MEMO is inert across the whole S3a family, H included: no measured run has ever had memo_store > 0
    (memo_unsafe == fast on c70ns a1 decode-0, 458883 == 458883; 320 of 320 D98FAST lines on s16mkt3 j-14d3a112). The posted
    HIX points ran with memo off. A later MEMO fix lands as its own env-keyed commit.
  - Six members change behaviour with no key of their own: KV-SCALE-GUARD v2 (hard fail), MESHGUARD, SIDE-FIX, the T1
    fix, D-b (fail-closed routed-expert block check) and PLE-WINMASK. All but PLE-WINMASK sit in `megakernel_runtime/`,
    which runs only with the whole-step megakernel engaged (`VLLM_TPU_MK_*`); SIDE-FIX is in the SIDECAR path (off);
    PLE-WINMASK is in `kernels/qwen4_exp/ple_fused.py`. No posted point runs `hix`, so no posted behaviour changes.
  - Keys of the served env (186 vllm-family keys) and the compile key: 181 are registered in `envs.py`. 5 are not, and
    each is read directly and added to the compile factors only when set (`compilation/tpu_compiler.py:799-817`):
    `VLLM_TPU_QSA_FUSED_KV_WRITE_LEAN` (read in `kernels/qwen4_exp/fused_kv_write.py:40`),
    `VLLM_TPU_QSA_DECODE_KV_BLOCK_ONE_WINDOW` (`kernels/qwen4_exp/pooled_qsa.py:768`), `VLLM_TPU_HC_XLA_MAX_ROWS`
    (`kernels/qwen4_exp/hc_decode.py:118`), `VLLM_TPU_HC_DECODE_FUSED_MAX_ROWS` (`hc_decode.py:191`) and
    `VLLM_TPU_DECODE_TP_AR` (`kernels/collectives/tp8_ar_decode_fn.py:20`). 65 registered keys are in
    `_TPU_COMPILE_ENV_IGNORED` (host, logging and check keys, plus MKH's `VLLM_TPU_L6_MK_CTX` / `VLLM_TPU_L6_MK_THIN_TT`),
    so a warm compile-cache artifact cannot tell them apart: every dash-bound `hix` car carries a cold-compile row.
    Registering and un-ignoring the H-member keys is a later commit with its own record.
  - Evidence (DEV, 1200 s, post-data / not blind): the reader-corrected c70ns re-read of record (proof v3.1 PASS 121/121) reads
    KEEP-H: a1 (H, SIDECAR off) 465.82 x / 53.68 Mtok/$ against a0 422.60 / 52.10 (dx +10.2 %, dM +3.0 %); MEMO inert; MKH
    attributed by pid.
  - **Exos:** see Notes. `hix` runs on Exos from this branch only after the recipe patches the selected form.
  - DSIG for the hix form: U; required (a DSIG row or a counted 0-ShapeSpecializationError served log) before any
    dash-bound hix car.
  - HIX bundle (port of `q38-hix-bundle-t1` `7eebc74eef` = H + 7 commits by HIX-GLUE / HIX-KERN / HIX-STACK; every key
    default off): GLUE SL `VLLM_TPU_HXG_SINK_LAYOUT` (`59a1eabb05`), GLUE GM `VLLM_TPU_HXG_GDN_META` (`f6bc73c528`), GLUE
    PF0 `VLLM_TPU_HXG_PLE_ONE` (`aacc8156e5`), KERN K3 `VLLM_TPU_MK_KERN_ROUTE_VEC` with the K3-MOSAIC fix (`1eb39a0040`,
    `5584d10fd3`), KERN K8 `VLLM_TPU_MK_KERN_REC_SKIP` (`7eebc74eef`). KERN K1 (sidecar prefetch) and K6 (stock-order FP8
    KV) come in the same commits, default off and NOT part of the measured set. Served A/B T44f (DEV, read of record
    read2 j-e603e15b): KEEP-BUNDLE. Not proven on this branch yet: a CPU proof (keys-off identity, hash pair, engagement
    and a mutant per member, SERVED-COMPILE one-trace buckets 8:16 with DSIG and FORM-SELECT) and a confirmation A/B on
    this tree come before any dash car. The bundle tests under `tests_oxpd/` (`test_hxg_*`, `q38_hixk8`, `q38_hixkern`)
    need `Q38_FORM=hix`.
- `b3` disclosures:
  - Member evidence (not re-run here): QGLUE-1 B3 bitwise row bw-b3 j-9664d862 PASS; SB3-1 union rec3 j-51cfff2a PASS of
    record; first device proof of the form gate T46l (A `Q38_FORM=k_sup`, B `Q38_FORM=b3`, gate line on every container).
  - SB3-1 greedy-probe flag: no dash car carries SB3-1 before its exact-form GSM8K + LongBench gate pair passes of record.
  - The SB3-1 one-trace SERVED-COMPILE leg has not been run yet.
  - E1001 `CompileTimeScopedVmemOom` retry lines are form-level on `b3` cold compiles (descriptive).
  - Every lever key is default off, so `Q38_FORM` unset and every other form are byte-unchanged by this form.
  - **Exos:** see Notes. `b3` runs on Exos from this branch only after the recipe patches the selected form.
- QGLUE-1 in `b1`: a device STEP cut of record; no served gain shown at B1 c212 (within the measured A/A noise floor:
  span 2.34 ms, Mtok/$ ~1.5 %).
- HD rider disclosure (SERVE car 1, DEV, 1200 s): KEEP-FOR-BUILD on the host cycle only (max-over-ranks p50 non-wait cycle
  15.821 -> 14.563 ms, d -1.258 ms); a served step-time or Mtok/$ gain is NOT shown. "dp_loop +0.16 ms with ON at >= 48
  running; cause U; no HD code in dp_loop (C); relocated-wait not supported (M, natvar_g2 with shuffle / plant mutants)".
- R4XB disclosure (DEV; no served gain claimed): with `VLLM_TPU_QSA_PREFILL_UNION_WALK_RING` = 3 or 4 the gathered prefill
  walk streams its tiles through an R-slot VMEM ring that continues across grid blocks; same tiles, order and per-tile
  compute (bitwise the double-buffered walk). CPU proofs ran on the port before SB1-2 (`fec8a2526a`, `b1` package
  `9800ecd085`): r4xkh2 j-83579d68, r4xbb1f j-cd5289b0 and r4xrj j-de6b3d3e PASS; r4xsc j-111f15ef FAIL (a gate-design
  defect in its JID leg, re-judged by r4xrj) and r4xkh j-d8a0970a FAIL (superseded by r4xkh2) stand of record. Served:
  SERVED-CARRY NOT-SHOWN (T45o). It is re-applied here onto SB1-2 + HD by offsets only (SB1-2 touched `envs.py` and
  `qsa_rpa.py` in other hunks; `qsa_union_dense.py` is untouched by SB1-2).
- r0pin arm c disclosure (DEV, 1200 s; decode only): with `VLLM_TPU_CPU_PIN=slice`, `VLLM_TPU_CPU_PIN_RESERVE=24` and
  `VLLM_TPU_CPU_PIN_SKIP_CPUS=0-23` the decode slices skip node-0 CPUs 0-23 (the NIC IRQ cores) and their SMT siblings;
  `VLLM_TPU_CPU_PIN_ENGINE` must stay unset (node 0 keeps no reserve cores, so `reserved` raises SlicePinError) and
  `VLLM_TPU_CPU_IRQ_DUMP_S` (a log-only probe) stays unset. Served (r0sw t1 T45z, one window, arms a j-d6d3e68a,
  c j-b6b9524c, a2 j-2eb52d6e): decode period -1.322 ms vs the control mean (A/A spread 0.155), Mtok/$ 279.19 vs
  271.91 / 273.81 (+2.3 %, A/A 1.90).
  - Measured on the SB1-1 stack (r0pin tree `37da395355` = b1 with SB1-1 record keys, QGLUE-1 off), NOT on b1 + SB1-2 +
    HD rider. The interaction with SB1-2's GC_WORKER / KVL_INWAIT and the HD rider's host-side REQVEC is U. A 3600 s
    canonical confirmation A/B (A = pushed b1 form, B = A + arm c, A/A arm) is ruled and is built by SERVE; it runs
    after the SB1-2 FINAL reads.
  - 1200 s DEV only: Mtok/$ +2.3 % is not comparable to 3600 s points (DUR-COMPARE).
  - Residual: node-0 decode ranks still do about 0.60 ms more host work per step than node-1 ranks after arm c (X47
    0.600, NODE 0.608; descriptive; cause under a 0-chip SERVE lane).
  - Default GKE TPU host layout only (224 CPUs, 2 NUMA nodes x 56 cores, SMT sibling = cpu + 112); on any other layout
    the strict slice checks fail closed (SlicePinError), they do not pin wrongly. Exos hosts: U.
  - One-time `b1` compile-hash change: `VLLM_TPU_CPU_PIN_RESERVE` moves into `_TPU_COMPILE_ENV_IGNORED` (it is read
    only by `cpu_pinning.py`, after compile), so every `b1` car after this lands has a new TPU compile hash, keys set or
    not, and the first one compiles cold under a shared cache tag (on the pre-SB1-2 base: `d526f005fe` ->
    `d05d2b3d81`, CPU job ab2 j-cc9c93cd).
- GDN (a) disclosure (no key): the `b1` form carries the GDN v3 LOG-conv / BATCHED-metadata correctness fix (GDN-FIX CORE
  `b35a4bc3c2` in `compute_g1.py` + `wrapper.py`, F1 `d4f63efc18` hand-ported to `metadata.py`; the same fix as the main
  package's `aae3e349b3`, plus the factored-tile conv-log store in `compute_g1.py`). Every `b1` car after this runs it:
  the `b1` compile files hash changes, so `b1` compile caches miss once, and a dash-bound `b1` car on this package needs
  its own exact-form cold gates. The frozen forms keep the defect exactly as their points ran it, so no posted point
  changes. TPU coverage: tpucov-b1 COV-PASS (fix `1d0b2e5a06` vs base `2c7b97818b`; the `b1` `kernels/gdn/v3` files here
  are byte-identical to the ones it tested).
- `b1` is a merge that has not yet run on a device: no served car stacks QGLUE-1 with SB1-1 on `b1` before one device
  bitwise check on `b1`.
- To work on a lineage: branch from `shriharim/qwen38-flash-1125` (the q38 team branch; `caojx/qwen38-flash-1124`
  at `bbffe86acf` and `shriharim/q38-merge-1124` at `892c501154` hold the same history up to those commits and no
  longer move), edit `src/vllm_torchtpu/_forms/<form>/vllm_torchtpu/`, and run with `Q38_FORM=<form>`. Register
  every new key in that form's `envs.py`, so it enters the compile key.
- Dash-bound full runs run from a pushed commit of `shriharim/qwen38-flash-1125` only.
- Syncs with `caojx/qwen38-flash-1124-2` land on `shriharim/qwen38-flash-1125` by merge, not rebase, so the branch
  only moves by fast-forward.
- A living form runs its own package: it does not get changes made to the main package (for example
  `VLLM_TPU_SHARED_BACKEND_CACHE`). Folding the living forms into the main package is the planned convergence step.

## Notes

- **Exos runs under a form are not yet reproducible from this branch.** The Exos recipe (exos-infx-qwen38-flash-agentx-disagg
  v1.2) patches the mapped tree at start-up (its exos_vllm bootstrap edits 7 files of `vllm_torchtpu`: model_loader_patches,
  kv_block_override, the two ray executors, tpu_runner, connector_metadata, tpu_connector), but only the main package, never
  `src/vllm_torchtpu/_forms/<form>/`. The posted Exos points ran plain trees, whose package was the main package, so it was
  patched. Here their form would run unpatched. The record proofs of this branch show CPU-side byte identity of every form, not
  this runtime patch. Until the q38 recipe copy also patches the selected form (in progress), no Exos run under a form starts
  from this branch. Default-cluster points (B1, B3 / B4 cars, HIX P1 / HIX-o) are unaffected.
- `src/sitecustomize.py` here adds the `CDK_MEMLOG` logger; it does nothing unless `CDK_MEMLOG=1`, which no point sets.
- `src/tpu_sync/` (vendored Python sources from the c1-stack-d3600 snapshot, no `__init__.py`) is outside every form;
  see the MERGE-1124 report for how it resolves on the serving image.
- Later step, not done here: fold the K forms (`k_cut1`, `k_pkg`, `k_sup`) into the main package behind flags, each with
  its own flag-off proof, and retire forms as they fold.
