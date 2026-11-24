# Qwen3.8 Flash PD curve of 2026-11-23: every point on one branch

This folder lists every point on the dash sheet `q38-pd-ixc-frontier-1118` as of 2026-11-23 22:31 UTC (16 points), plus the
pending B3 package point T55d. All 17 run from this branch. Each `<band>_<point>.flags.txt` holds that job's exact per-leg
engine env and `vllm serve` arguments, parsed from the job's stored `recipe.yaml`. Recipe: `infx-qwen38-flash-agentx-disagg`.

## How to reproduce a point

1. Check out this branch and stage the tree exactly as the point's launcher stages its run commit (the job's received tree
   was identical, file for file, to the run commit listed in its flags file).
2. Set every env key and serve argument of the flags file on the matching leg.
3. **B4, B5, B6 (the 7 Exos points): also set `Q38_OXPD_FORM=1` on every leg.** They ran on w3 `2bc75911f5` directly; on this
   branch w3's code is the oxpd overlay (`src/vllm_torchtpu/_forms/oxpd/vllm_torchtpu/`). Check each leg's log for
   `q38 form gate: Q38_OXPD_FORM=1`. The two HIX points already set it.
4. Exos pods apply the recipe's `_EXOS_PYPATCH`, which does not reach the overlay copy (see `q38_bands/README.md`, Exos notes).
   Until the Exos launcher variant is re-pinned for this head, run B4-B6 on Exos from `2bc75911f5`, as before.

## Points

| Band | Point | Mtok/$ @ x (TTFT p90) | Job | Run commit | Form on this branch | Flags file |
|---|---|---|---|---|---|---|
| B1 | T58x Car I + VA, c212 | 297.96 @ 78.67 (6.531 s) | j-1070e35b-fd91-45fa-bb71 | `cf812d1e5b` CUT 1 | default (gate unset) | B1_T58x_c212.flags.txt |
| B2 | T61v c136-k3c | 208.46 @ 105.55 (7.403 s) | j-9b116a5c-2701-47d5-9511 | `271cf69f0e` K3 | default | B2_T61v_c136k3c.flags.txt |
| B3 | T56g c192sp | 206.35 @ 154.02 (7.746 s) | j-f1e903ce-0469-43b1-b842 | `5ef1f2a184` K3b | default | B3_T56g_c192sp.flags.txt |
| B3 | T57s c192s | 202.62 @ 154.34 (9.848 s) | j-0495f096-c6a4-4a14-bec9 | `5ef1f2a184` K3b | default | B3_T57s_c192s.flags.txt |
| B3 | T58y c184s | 200.82 @ 154.75 (6.725 s) | j-87d7b48d-d846-4d93-9408 | `5ef1f2a184` K3b | default | B3_T58y_c184s.flags.txt |
| B3 | T58n c176s | 196.49 @ 155.97 (5.042 s) | j-adc10c3f-4492-45d5-8994 | `5ef1f2a184` K3b | default | B3_T58n_c176s.flags.txt |
| B3 | T58f c168s | 189.46 @ 157.73 (3.927 s) | j-093560b4-3240-46ea-8ba3 | `5ef1f2a184` K3b | default | B3_T58f_c168s.flags.txt |
| B3 (pending) | T55d c200spk | 217.25 @ 160.32 | j-57e28d43-07cf-4f1e-870b | `9f0c924d16` B3 package | default | B3_T55d_c200spk_PENDING.flags.txt |
| B4 | P2C128 | 139.93 @ 169.82 (7.952 s) | j-38139aef-5cbe-4af7-a6de | `2bc75911f5` w3 | **oxpd: add Q38_OXPD_FORM=1** | B4_P2C128.flags.txt |
| B4 | p2c112 | 131.37 @ 176.45 (4.677 s) | j-4bf73187-9459-43a6-81db | `2bc75911f5` w3 | **oxpd: add Q38_OXPD_FORM=1** | B4_p2c112.flags.txt |
| B5 | p2c104m12r64k48 | 117.62 @ 238.97 (9.850 s) | j-8f66f71f-2e30-483f-b58c | `2bc75911f5` w3 | **oxpd: add Q38_OXPD_FORM=1** | B5_p2c104m12r64k48.flags.txt |
| B5 | p2c96m12r64k48 | 109.61 @ 240.61 (7.194 s) | j-c325b23f-5785-454b-bee8 | `2bc75911f5` w3 | **oxpd: add Q38_OXPD_FORM=1** | B5_p2c96m12r64k48.flags.txt |
| B6 | p3c88 | 80.46 @ 276.22 (2.294 s) | j-ebb63496-df01-4d7a-9e41 | `2bc75911f5` w3 | **oxpd: add Q38_OXPD_FORM=1** | B6_p3c88.flags.txt |
| B6 | p3c80 | 75.05 @ 288.67 (2.207 s) | j-15ca9825-0a7a-4943-b683 | `2bc75911f5` w3 | **oxpd: add Q38_OXPD_FORM=1** | B6_p3c80.flags.txt |
| B6 | p3c72 | 64.57 @ 321.27 (5.108 s) | j-a8bff01e-39c7-45d5-96a5 | `2bc75911f5` w3 | **oxpd: add Q38_OXPD_FORM=1** | B6_p3c72.flags.txt |
| HIX | P1 T58g p3c66m4k16s | 62.64 @ 408.48 (4.195 s) | j-6b77f816-02bd-410c-bf57 | `1ea5a165a7` K3-ox.1 | oxpd (flag already set) | HIX_P1_T58g_p3c66m4k16s.flags.txt |
| HIX | m3 T58u p4c70m3k16s | 52.62 @ 418.52 (4.624 s) | j-5d5b572a-1188-4bb0-9dcb | `1ea5a165a7` K3-ox.1 | oxpd (flag already set) | HIX_m3_T58u_p4c70m3k16s.flags.txt |

## What this branch is

`shriharim/q38-best-per-band` (7f9b39a8d5: Benliang's form + our R-form stack, with w3 as the oxpd overlay) plus the K-line
(rw3 -> K3 -> K3b -> CUT 1 / B3 package, and K3-ox.1's overlay hunks), cherry-picked with `-x` so every commit names its
source. Every K-line feature is behind its own env key, default off; no existing default changed. Two key pairs are refused
at startup (ValueError): `TPU_QWEN4_HC_PREFILL_COMPARE` with `TPU_QWEN4_HC_PREFILL_FOLD` or `TPU_QWEN4_HC_PREFILL_DOWN_ASM`, and
`VLLM_TPU_QSA_PREFILL_INDEXER_NO_COND` with `VLLM_TPU_QSA_PREFILL_COMPARE_CORE_STATS`.

The parity evidence per point (identical vs equivalent, with each difference's cover) is in the proof notes added with the
next commit.
