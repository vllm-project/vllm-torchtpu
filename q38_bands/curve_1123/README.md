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

## Read first (for the user and Jiaxin)

1. **Exos points: set `Q38_OXPD_FORM=1` on every leg.** Forgetting it fails loudly: their env sets `VLLM_TPU_HOST_GC`, which the
   default form refuses with a ValueError at import, so the engine does not start.
2. **One default-behaviour change for everyone on this branch: STAGE3FIX** (`4d87d69b27`, from `466c290598`). With no flag set, the
   grouped Stage-3 producer registration (every Qwen PD producer) applies the scheduler's releases and registers ahead sends with
   no TTL until the request finishes. It fixes a preemption crash (`Conflicting duplicate Stage-3 producer registration`). Every
   default-form point on the sheet ran with it. Everything else this integration adds is behind its own env key, default off.

3. **Decode routing disclosure.** These points run the proxy with `PROXY_DECODE_ROUTING=sticky`, a decode session affinity
   keyed on aiperf's per-conversation `X-Correlation-ID` header: every B3 point here (T56g c192sp, T57s c192s, T58y c184s,
   T58n c176s, T58f c168s, and the pending T55d c200spk) and HIX P1 (T58g) / m3 (T58u). Read from each job's stored recipe;
   B1 T58x, B2 T61v and B4-B6 do not set it (B1 is round-robin with an empty owner map). Prefill routing, eviction and the
   prefill cache are unaffected. The mechanism is disclosed with IX's llm-d configs as precedent (they use decode session
   affinity on `x-dynamo-session-id`). A prefix-hash affinity computed from the prompt is being built to replace it, and those
   points will be re-run with it. Reference: frontier_disclosures_1124.md (in the Q3.8 PD team's reports).
4. **Tuned values.** Some cache knobs (for example `VLLM_TPU_OFFLOAD_TIER_CLASS_RATIO=8`) were tuned on this workload. That is
   allowed: the server never reads the traffic.

## Parity evidence (CPU proof jobs, 0 TPU chips)

Per point and leg, the exact env from the job's stored recipe.yaml was applied to the run commit and to this branch, and every
function / method / class / module the selected form runs was compared body-only with env-decided branches resolved ("GBH").
IDENTICAL = no difference. EQUIVALENT (n) = n declared differences, each pinned by the sha256 of its diff and cleared by a cover
that ran and passed of record: program equality (HC ops built by the real GatedResidual; QSA attention, union decode, metadata
and prefill walk programs) or the K-line member's own proof of record.

| Band | Point | Verdict | Differences by class (prefill + decode) | Covers that ran of record | Row |
|---|---|---|---|---|---|
| B1 | T58x c212 | EQUIVALENT (12 prefill + 13 decode) | BASE-FORMS 4; BASE-SCBITS 2; MERGE-HC 4; REFUSAL-QSA 2; SIBLING-HCK 13 | GBH PASS, DYNHC PASS, QSA PASS, REF PASS, member proofs PASS (s7) | PASS |
| B2 | T61v c136-k3c | EQUIVALENT (32 prefill + 32 decode) | BASE-FORMS + SIBLING-CUT1 4; BASE-SCBITS 2; MERGE-HC 4; REFUSAL-QSA 2; SIBLING-CUT1 38; SIBLING-HCK 12; SIBLING-K3b 2 | GBH PASS, DYNHC PASS, QSA PASS, REF PASS, member proofs PASS (s7) | PASS |
| B3 | T56g c192sp | EQUIVALENT (31 prefill + 31 decode) | BASE-FORMS + SIBLING-CUT1 4; BASE-SCBITS 2; MERGE-HC 4; REFUSAL-QSA 2; SIBLING-CUT1 38; SIBLING-HCK 12 | GBH PASS, DYNHC PASS, QSA PASS, REF PASS, member proofs PASS (s7) | PASS |
| B3 | T57s c192s | EQUIVALENT (31 prefill + 31 decode) | BASE-FORMS + SIBLING-CUT1 4; BASE-SCBITS 2; MERGE-HC 4; REFUSAL-QSA 2; SIBLING-CUT1 38; SIBLING-HCK 12 | GBH PASS, DYNHC PASS, QSA PASS, REF PASS, member proofs PASS (s7) | PASS |
| B3 | T58y c184s | EQUIVALENT (31 prefill + 31 decode) | BASE-FORMS + SIBLING-CUT1 4; BASE-SCBITS 2; MERGE-HC 4; REFUSAL-QSA 2; SIBLING-CUT1 38; SIBLING-HCK 12 | GBH PASS, DYNHC PASS, QSA PASS, REF PASS, member proofs PASS (s7) | PASS |
| B3 | T58n c176s | EQUIVALENT (31 prefill + 31 decode) | BASE-FORMS + SIBLING-CUT1 4; BASE-SCBITS 2; MERGE-HC 4; REFUSAL-QSA 2; SIBLING-CUT1 38; SIBLING-HCK 12 | GBH PASS, DYNHC PASS, QSA PASS, REF PASS, member proofs PASS (s7) | PASS |
| B3 | T58f c168s | EQUIVALENT (31 prefill + 31 decode) | BASE-FORMS + SIBLING-CUT1 4; BASE-SCBITS 2; MERGE-HC 4; REFUSAL-QSA 2; SIBLING-CUT1 38; SIBLING-HCK 12 | GBH PASS, DYNHC PASS, QSA PASS, REF PASS, member proofs PASS (s7) | PASS |
| B3 (pending) | T55d c200spk | EQUIVALENT (25 prefill + 25 decode) | BASE-FORMS + SIBLING-CUT1 4; BASE-SCBITS 2; MERGE-HC 4; REFUSAL-QSA 2; SIBLING-CUT1 38 | GBH PASS, DYNHC PASS, QSA PASS, REF PASS, member proofs PASS (s7) | PASS |
| B4 | P2C128 | EQUIVALENT (5 prefill + 5 decode) | OX1-OVERLAY 10 | GBH PASS, overlay QSA PASS | PASS |
| B4 | p2c112 | EQUIVALENT (5 prefill + 5 decode) | OX1-OVERLAY 10 | GBH PASS, overlay QSA PASS | PASS |
| B5 | p2c104m12r64k48 | EQUIVALENT (5 prefill + 5 decode) | OX1-OVERLAY 10 | GBH PASS, overlay QSA PASS | PASS |
| B5 | p2c96m12r64k48 | EQUIVALENT (5 prefill + 5 decode) | OX1-OVERLAY 10 | GBH PASS, overlay QSA PASS | PASS |
| B6 | p3c88 | EQUIVALENT (5 prefill + 5 decode) | OX1-OVERLAY 10 | GBH PASS, overlay QSA PASS | PASS |
| B6 | p3c80 | EQUIVALENT (5 prefill + 5 decode) | OX1-OVERLAY 10 | GBH PASS, overlay QSA PASS | PASS |
| B6 | p3c72 | EQUIVALENT (5 prefill + 5 decode) | OX1-OVERLAY 10 | GBH PASS, overlay QSA PASS | PASS |
| HIX | P1 T58g | IDENTICAL | none | GBH PASS | PASS |
| HIX | m3 T58u | IDENTICAL | none | GBH PASS | PASS |

Record jobs: tbp_rec1 `j-8a5ff8de-940e-4cc5-93ef` (`TBP_SUMMARY PASS mode record head 4f52f0c40b src ffaa590ed5`: flag-off identity vs 7f9b39a8d5 under no form env and the base's own band envs, per-point parity for all 17 points, HC / QSA / MoE program identity, 3 mutants KILLED) and tbt_rec1 `j-370da087-88af-4572-a7ff` (`TBT_SUMMARY PASS mode record head 4f52f0c40b src ffaa590ed5`: 54 regression test files with 0 new failures, the refused-pairs test, IMAGE + ARGV for one B1, one B3 and one Exos point, 4 mutants KILLED). Proven code head `4f52f0c40b` (SRC ffaa590ed5); this README commit changes no file under src/.

