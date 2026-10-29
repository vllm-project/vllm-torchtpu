# DeepSeek-V4.1-Flash FP8 PD Disaggregation (`1P1D`, `DP=8`) — Empirical ISL (`50:20:12:9:9`) Benchmark Report & Manifest

> [!IMPORTANT]
> **Executive Summary (`1P1D` on `8` TPU7x Chips = `1P (2x2x1)` + `1D (2x2x1)`, `DP=8, TP=1, EP=8`, DSpark $K=5$, `80%` Target Prefix Cache Hit)**:
> - **Zero Failed Requests Across All Concurrency Sweeps**: `49/49` (`n=8`), `88/88` (`n=16`), `154/154` (`n=32`), `267/267` (`n=64`), `435/435` (`n=128`, `128` trunks), and `485/485` (`n=128`, `40` pinned trunks) with `53%–67%` structured multi-turn tool-calling completions (`finish_reason="tool_calls"`).
> - **Peak Steady-State Throughput (`n = 128`)**:
>   - **`5-or-N` Trunks (`128` distinct trunks, `61.1%` cache hit)**: **`6,363.3 tok/s` output** (`5,476.8 tok/s` full-drain) and **`150,784.0 tok/s` total input** (`58,999.4 tok/s` uncached prefill) at **`\$1.89/1M` output** and **`\$0.080/1M` input** (`\$0.076/1M` blended, 3-Yr CUD `\$43.20/hr`).
>   - **Pinned HBM Trunks (`40` distinct trunks, `77.0%` cache hit)**: **`7,008.7 tok/s` output** (`6,133.8 tok/s` full-drain) and **`167,537.9 tok/s` total input** (`38,530.9 tok/s` uncached prefill) at **`\$1.71/1M` output** and **`\$0.072/1M` input** (`\$0.069/1M` blended).
> - **Interactive Per-Stream Decode Speed & TTFT**:
>   - **`n = 8–16`**: **`100.6–102.2 tok/s` median per stream** (`P90_hi = 125.0–130.9 tok/s`, `Per-Token ITL = 9.78–9.94ms`, `StepChunk ITL = 27.0–28.8ms`, DSpark $\tau = 2.82\text{--}2.83$) with **`0.97s–1.37s` median TTFT** (`0.41s–1.02s` on `1–8K` and `1.97s–2.23s` on `64K+` at `~83K` ISL).
>   - **`n = 32–64`**: **`72.8–83.3 tok/s` median per stream** (`Per-Token ITL = 12.01–13.73ms`, DSpark $\tau = 2.74\text{--}2.82$) with **`1.62s–1.70s` median TTFT** (`2.68s–3.82s` P90) and **`2,383.7–4,087.2 tok/s` window output throughput**.
>   - **`n = 128`**: **`63.4–64.8 tok/s` median per stream** (`P95_low = 48.9–50.8 tok/s`, `P90_hi = 82.8–84.3 tok/s`, `Per-Token ITL = 15.44–15.76ms`, DSpark $\tau = 2.74\text{--}2.76$) with **`2.04s` median TTFT (`5.79s` P90)** at `40` trunks and **`3.01s` median TTFT (`10.79s` P90)** at `128` trunks.

---

## 1. End-to-End 1P1D Empirical ISL (`50:20:12:9:9`) Summary Table

- **Cluster & Cost Basis**: `8` TPU v7x (Ironwood) chips total = `1P` (`2x2x1` = `4` chips, `8` TensorCores, `DP=8, TP=1, EP=8`) + `1D` (`2x2x1` = `4` chips, `8` TensorCores, `DP=8, TP=1, EP=8`, `ENABLE_DSPARK=1` with $K=5$ speculative tokens/step).
- **3-Year CUD Rate**: `8 chips × \$5.40/chip-hr = \$43.20/hr` (`\$0.012/s`), matching the GLM-5.3 1P1D benchmark report methodology.
- **Workload**: Empirical `50:20:12:9:9` ISL distribution (`1–8K: 50%`, `8–16K: 20%`, `16–32K: 12%`, `32–64K: 9%`, `64K+: 9%`, weighted mean ISL `~21.5K` tokens), `max_tokens = 1,024` (`ignore_eos = True`), `60s` steady-state measurement window per `n` after Pass-1 XLA warmup and per-`n` Phase-0 trunk warmup, replaying exact multi-turn tool-calling schemas and exact sampling temperatures from `requests_mocked.jsonl.xz`.

| Sessions (`n`) [Trunks] | OK Reqs (Tools / In-Win) | Mean ISL (Cache Hit) | Mean OSL | Win Out (Drain tok/s) | Win In (Uncached tok/s) | Med TPS/Strm (`P95_low` / `P90_low` / `P90_hi`) | Per-Tok ITL (`StepChunk` Med/Mean) | DSpark $\tau$ (`Pos 0..4` Accept) | Med TTFT (`P90`) | Med E2E | Out Cost (\$/1M Win) | Out Cost (\$/1M Drain) | In Cost (\$/1M Win) | Blended (\$/1M Win) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`n = 8`** [`40` tr] | `49/49` (`33` / `41`) | `20,196` (`78.1%`) | `1,024.0` | **`771.5`** (`713.2`) | **`16,493.1`** (`3,607.8`) | **`102.21`** (`85.0` / `86.0` / `130.9`) | `9.78ms` (`27.01` / `36.02ms`) | **`2.83`** (`63/44/32/25/19%`) | **`0.97s`** (`1.66s`) | `10.60s` | `\$15.55` | `\$16.82` | `\$0.728` | `\$0.695` |
| **`n = 16`** [`40` tr] | `88/88` (`50` / `72`) | `21,942` (`78.9%`) | `1,024.0` | **`1,428.0`** (`1,232.8`) | **`32,181.3`** (`6,786.1`) | **`100.62`** (`76.1` / `81.9` / `125.0`) | `9.94ms` (`28.81` / `35.27ms`) | **`2.82`** (`62/44/32/24/18%`) | **`1.37s`** (`2.01s`) | `11.51s` | `\$8.40` | `\$9.73` | `\$0.373` | `\$0.357` |
| **`n = 32`** [`40` tr] | `154/154` (`84` / `122`) | `21,649` (`76.1%`) | `1,024.0` | **`2,383.7`** (`2,136.9`) | **`53,085.7`** (`12,740.1`) | **`83.27`** (`64.9` / `66.1` / `110.4`) | `12.01ms` (`33.78` / `42.53ms`) | **`2.82`** (`62/44/32/24/19%`) | **`1.62s`** (`2.68s`) | `13.79s` | `\$5.03` | `\$5.62` | `\$0.226` | `\$0.216` |
| **`n = 64`** [`64` tr] | `267/267` (`142` / `203`) | `21,528` (`78.7%`) | `1,024.0` | **`4,087.2`** (`3,623.6`) | **`93,299.4`** (`19,912.7`) | **`72.84`** (`50.0` / `53.4` / `103.8`) | `13.73ms` (`34.89` / `46.83ms`) | **`2.74`** (`61/43/31/22/17%`) | **`1.70s`** (`3.82s`) | `16.19s` | `\$2.94` | `\$3.31` | `\$0.129` | `\$0.123` |
| **`n = 128`** [`128` tr, `5-or-N`] | `435/435` (`240` / `307`) | `21,771` (`61.1%`) | `1,024.0` | **`6,363.3`** (`5,476.8`) | **`150,784.0`** (`58,999.4`) | **`63.44`** (`48.9` / `50.7` / `82.8`) | `15.76ms` (`42.82` / `55.60ms`) | **`2.76`** (`61/43/31/23/18%`) | **`3.01s`** (`10.79s`) | `20.32s` | `\$1.89` | `\$2.19` | `\$0.080` | `\$0.076` |
| **`n = 128`** [`40` tr, Pinned] | `485/485` (`272` / `357`) | `21,955` (`77.0%`) | `1,024.0` | **`7,008.7`** (`6,133.8`) | **`167,537.9`** (`38,530.9`) | **`64.75`** (`50.8` / `52.2` / `84.3`) | `15.44ms` (`42.30` / `53.56ms`) | **`2.74`** (`61/42/31/23/18%`) | **`2.04s`** (`5.79s`) | `18.55s` | **`\$1.71`** | **`\$1.96`** | **`\$0.072`** | **`\$0.069`** |

---

## 2. Per-Bucket Empirical ISL Breakdown (`50:20:12:9:9`)

### `n = 8` Sessions (`40` Trunks: `{'1-8K': 8, '8-16K': 8, '16-32K': 8, '32-64K': 8, '64K+': 8}`)
| ISL Bucket | Target Weight | Trunks | Completed Reqs | Mean ISL (tok) | Mean OSL (tok) | Cache Hit (%) | Med TTFT (s) | P90 TTFT (s) | Med E2E (s) | Med TPS/Strm (`P95_low` / `P90_low` / `P90_hi`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`1–8K`** | `50%` | `8` | `25` (`51.0%`) | `6,716` | `1,024.0` | `78.1%` | **`0.41s`** | `1.34s` | `10.30s` | **`106.5`** (`85.2` / `86.3` / `125.3`) |
| **`8–16K`** | `20%` | `8` | `10` (`20.4%`) | `14,082` | `1,024.0` | `78.5%` | **`0.91s`** | `1.17s` | `11.27s` | **`99.2`** (`81.2` / `85.1` / `114.4`) |
| **`16–32K`** | `12%` | `8` | `6` (`12.2%`) | `26,721` | `1,024.0` | `77.9%` | **`0.91s`** | `1.00s` | `9.54s` | **`119.5`** (`93.7` / `93.8` / `158.9`) |
| **`32–64K`** | `9%` | `8` | `4` (`8.2%`) | `48,932` | `1,024.0` | `77.4%` | **`1.56s`** | `1.71s` | `10.38s` | **`116.5`** (`99.9` / `102.7` / `149.6`) |
| **`64K+`** | `9%` | `8` | `4` (`8.2%`) | `81,206` | `1,024.0` | `78.5%` | **`1.97s`** | `2.52s` | `13.16s` | **`91.4`** (`88.5` / `89.0` / `119.1`) |

### `n = 16` Sessions (`40` Trunks: `{'1-8K': 8, '8-16K': 8, '16-32K': 8, '32-64K': 8, '64K+': 8}`)
| ISL Bucket | Target Weight | Trunks | Completed Reqs | Mean ISL (tok) | Mean OSL (tok) | Cache Hit (%) | Med TTFT (s) | P90 TTFT (s) | Med E2E (s) | Med TPS/Strm (`P95_low` / `P90_low` / `P90_hi`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`1–8K`** | `50%` | `8` | `44` (`50.0%`) | `6,847` | `1,024.0` | `78.9%` | **`1.02s`** | `1.75s` | `11.36s` | **`97.5`** (`70.3` / `79.3` / `122.4`) |
| **`8–16K`** | `20%` | `8` | `18` (`20.5%`) | `13,475` | `1,024.0` | `79.8%` | **`1.06s`** | `1.88s` | `12.53s` | **`89.5`** (`76.7` / `80.7` / `110.1`) |
| **`16–32K`** | `12%` | `8` | `10` (`11.4%`) | `25,188` | `1,024.0` | `77.6%` | **`0.98s`** | `1.79s` | `11.03s` | **`106.1`** (`87.4` / `88.8` / `135.0`) |
| **`32–64K`** | `9%` | `8` | `8` (`9.1%`) | `57,236` | `1,024.0` | `79.6%` | **`1.65s`** | `2.29s` | `10.35s` | **`122.0`** (`105.6` / `105.7` / `140.2`) |
| **`64K+`** | `9%` | `8` | `8` (`9.1%`) | `84,664` | `1,024.0` | `78.6%` | **`2.23s`** | `2.67s` | `11.70s` | **`103.6`** (`85.8` / `89.9` / `121.2`) |

### `n = 32` Sessions (`40` Trunks: `{'1-8K': 8, '8-16K': 8, '16-32K': 8, '32-64K': 8, '64K+': 8}`)
| ISL Bucket | Target Weight | Trunks | Completed Reqs | Mean ISL (tok) | Mean OSL (tok) | Cache Hit (%) | Med TTFT (s) | P90 TTFT (s) | Med E2E (s) | Med TPS/Strm (`P95_low` / `P90_low` / `P90_hi`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`1–8K`** | `50%` | `8` | `77` (`50.0%`) | `6,842` | `1,024.0` | `78.3%` | **`1.48s`** | `2.01s` | `13.76s` | **`81.5`** (`66.1` / `69.7` / `106.6`) |
| **`8–16K`** | `20%` | `8` | `31` (`20.1%`) | `12,720` | `1,024.0` | `78.2%` | **`1.50s`** | `2.63s` | `14.55s` | **`79.7`** (`64.0` / `65.5` / `94.6`) |
| **`16–32K`** | `12%` | `8` | `18` (`11.7%`) | `24,193` | `1,024.0` | `76.7%` | **`1.51s`** | `2.06s` | `12.40s` | **`93.6`** (`65.4` / `65.6` / `123.5`) |
| **`32–64K`** | `9%` | `8` | `14` (`9.1%`) | `56,653` | `1,024.0` | `79.1%` | **`2.05s`** | `2.45s` | `12.10s` | **`104.6`** (`72.4` / `79.9` / `121.3`) |
| **`64K+`** | `9%` | `8` | `14` (`9.1%`) | `84,584` | `1,024.0` | `72.3%` | **`2.76s`** | `3.67s` | `14.94s` | **`82.4`** (`68.6` / `72.4` / `104.4`) |

### `n = 64` Sessions (`64` Trunks: `{'1-8K': 32, '8-16K': 8, '16-32K': 8, '32-64K': 8, '64K+': 8}`)
| ISL Bucket | Target Weight | Trunks | Completed Reqs | Mean ISL (tok) | Mean OSL (tok) | Cache Hit (%) | Med TTFT (s) | P90 TTFT (s) | Med E2E (s) | Med TPS/Strm (`P95_low` / `P90_low` / `P90_hi`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`1–8K`** | `50%` | `32` | `134` (`50.2%`) | `6,957` | `1,024.0` | `78.9%` | **`1.56s`** | `2.52s` | `15.61s` | **`72.2`** (`50.5` / `54.0` / `95.2`) |
| **`8–16K`** | `20%` | `8` | `53` (`19.9%`) | `13,143` | `1,024.0` | `78.8%` | **`1.56s`** | `2.77s` | `16.36s` | **`69.2`** (`50.3` / `56.4` / `86.0`) |
| **`16–32K`** | `12%` | `8` | `32` (`12.0%`) | `26,190` | `1,024.0` | `77.6%` | **`1.69s`** | `2.03s` | `13.57s` | **`88.7`** (`51.9` / `53.7` / `122.9`) |
| **`32–64K`** | `9%` | `8` | `24` (`9.0%`) | `54,366` | `1,024.0` | `79.2%` | **`2.43s`** | `3.24s` | `16.18s` | **`74.8`** (`63.4` / `65.7` / `107.8`) |
| **`64K+`** | `9%` | `8` | `24` (`9.0%`) | `82,347` | `1,024.0` | `78.7%` | **`2.79s`** | `3.96s` | `17.42s` | **`70.5`** (`56.0` / `58.4` / `96.6`) |

### `n = 128` Sessions (`128` Trunks `5-or-N`: `{'1-8K': 64, '8-16K': 24, '16-32K': 16, '32-64K': 16, '64K+': 8}`)
| ISL Bucket | Target Weight | Trunks | Completed Reqs | Mean ISL (tok) | Mean OSL (tok) | Cache Hit (%) | Med TTFT (s) | P90 TTFT (s) | Med E2E (s) | Med TPS/Strm (`P95_low` / `P90_low` / `P90_hi`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`1–8K`** | `50%` | `64` | `218` (`50.1%`) | `6,875` | `1,024.0` | `68.5%` | **`2.06s`** | `10.10s` | `20.32s` | **`59.5`** (`48.7` / `50.2` / `77.8`) |
| **`8–16K`** | `20%` | `24` | `87` (`20.0%`) | `13,273` | `1,024.0` | `68.7%` | **`2.05s`** | `9.71s` | `20.93s` | **`59.0`** (`48.6` / `49.9` / `69.5`) |
| **`16–32K`** | `12%` | `16` | `52` (`12.0%`) | `26,418` | `1,024.0` | `63.5%` | **`2.30s`** | `8.19s` | `17.83s` | **`70.8`** (`50.7` / `52.5` / `88.5`) |
| **`32–64K`** | `9%` | `16` | `39` (`9.0%`) | `54,070` | `1,024.0` | `47.3%` | **`4.30s`** | `11.17s` | `19.47s` | **`71.1`** (`55.3` / `57.4` / `83.0`) |
| **`64K+`** | `9%` | `8` | `39` (`9.0%`) | `85,503` | `1,024.0` | `62.8%` | **`5.31s`** | `12.01s` | `21.69s` | **`63.8`** (`49.2` / `52.6` / `82.4`) |

### `n = 128` Sessions (`40` Pinned Trunks: `{'1-8K': 8, '8-16K': 8, '16-32K': 8, '32-64K': 8, '64K+': 8}`)
| ISL Bucket | Target Weight | Trunks | Completed Reqs | Mean ISL (tok) | Mean OSL (tok) | Cache Hit (%) | Med TTFT (s) | P90 TTFT (s) | Med E2E (s) | Med TPS/Strm (`P95_low` / `P90_low` / `P90_hi`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`1–8K`** | `50%` | `8` | `242` (`49.9%`) | `6,982` | `1,024.0` | `79.2%` | **`1.86s`** | `5.03s` | `18.86s` | **`61.5`** (`50.4` / `51.3` / `79.8`) |
| **`8–16K`** | `20%` | `8` | `97` (`20.0%`) | `13,088` | `1,024.0` | `78.2%` | **`1.91s`** | `5.82s` | `19.68s` | **`59.7`** (`51.1` / `52.0` / `77.7`) |
| **`16–32K`** | `12%` | `8` | `58` (`12.0%`) | `27,170` | `1,024.0` | `77.8%` | **`1.95s`** | `5.33s` | `16.64s` | **`71.3`** (`52.5` / `54.8` / `94.0`) |
| **`32–64K`** | `9%` | `8` | `44` (`9.1%`) | `54,349` | `1,024.0` | `77.1%` | **`2.35s`** | `5.31s` | `18.53s` | **`68.5`** (`51.3` / `53.7` / `87.5`) |
| **`64K+`** | `9%` | `8` | `44` (`9.1%`) | `84,593` | `1,024.0` | `75.3%` | **`3.02s`** | `7.00s` | `20.00s` | **`64.7`** (`48.6` / `50.2` / `85.0`) |

---

## 3. Deep-Dive Architectural & Systems Analysis

### 3.1 Hybrid KV Cache Coordinator & HBM Block Footprint (`n = 128 @ 128 Trunks` vs. `40 Trunks`)
In DeepSeek-V4.1-Flash (`5` KV cache groups: `3` SWA groups with `page_tokens=128, window=512`, `1` CSA group with `page_tokens=1024`, and `1` State group with `page_tokens=64, retention_interval=1024`), each DP rank on `1P` has a unified `BlockPool` of **`5,094` blocks**:
1. **Steady-State Retained Prefix Footprint (`remove_skipped_blocks`)**:
   - Once a trunk completes prefill on `1P`, `HybridKVCacheCoordinator.remove_skipped_blocks` (`scheduler.py:2996`) frees all SWA blocks outside the sliding window and all non-checkpointed State blocks (`93.75%` of State blocks), shrinking an `80K`-token (`64K+` bucket) trunk from `~3,292` blocks during prefill down to **`~380` retained blocks** (`~88%` reduction).
   - Consequently, even `128` distinct trunks (`16` trunks per DP rank, `2,730,163` total prompt tokens) occupy only **`~1,060` retained blocks per DP rank (`20.8%` of `5,094` blocks)** at rest, which is why Phase 0 Warmup populates `128/128` trunks in `43.47s` when warmed sequentially per DP rank in descending bucket order (`64K+ -> 32-64K -> 16-32K -> 8-16K -> 1-8K`).
2. **Active Prefill + Stage-3 D2D Transfer Footprint at `n = 128`**:
   - While a request is actively prefilling on `1P`—and until `1D` pulls the KV blocks over Stage-3 D2D and sends `_enqueue_stage3_release`—its temporary SWA/State blocks for the uncached suffix (`~20%` of ISL) remain pinned in `BlockPool`.
   - At `n = 8, 16, 32, 64` (`1–8` concurrent sessions per DP rank), active + retained blocks stay well under `5,094` blocks/rank, maintaining **`76.1%–78.9%` observed cache hit** across all buckets.
   - At `n = 128` (`16` concurrent sessions per DP rank), holding `128` distinct trunks (`16`/rank) alongside `16` active requests/rank pushes transient `BlockPool` utilization to the `5,094`-block ceiling on `32–64K` and `64K+` bursts, reducing the observed HBM cache hit rate to **`61.1%`** (`58,999.4 tok/s` uncached prefill, `6,363.3 tok/s` window output, `3.01s` Med TTFT).
   - Pinning `40` shared trunks (`5`/rank = `1` per ISL bucket per DP rank, `--trunks-above-n16 40`) eliminates active-request capacity eviction in HBM, restoring **`77.0%` observed cache hit** (`75.3%–79.2%` across all 5 buckets), cutting median TTFT to **`2.04s` (`P90 = 5.79s`)**, and boosting window output throughput to **`7,008.7 tok/s` (`167,537.9 tok/s` input)**!

### 3.2 DSpark ($K=5$) Speculative Decoding Dynamics in 1P1D (`τ = 2.74–2.83`)
Across all measured concurrency levels (`n = 8` through `n = 128`), scraping `vllm:spec_decode_*` counters on `1D` (`http://10.11.0.21:9400/metrics`) confirms a remarkably stable mean acceptance length of **$\tau = 2.74\text{--}2.83$ tokens/step** (`1.74–1.83` accepted draft tokens out of `5` drafted per step, with per-position acceptance rates `[61–63%, 42–44%, 31–32%, 22–25%, 17–19%]`):
- **Target Model Exactness vs. Draft KV Population**: In 1P1D Stage-3 disaggregation, `1P` runs only the target model (`DeepSeek-V4.1-Flash`) and transfers the `5` target KV cache groups (`3` SWA + `1` CSA + `1` State) for the first `N - 1` prompt tokens to `1D`. `1D` computes token `N` locally to produce the first target logits (`100%` token-for-token identical to local execution, verified by `[GATE-EQ]` across `64..20,000` tokens). Because `1P` does not run DSpark, `1D`'s `5` draft layers (`DraftLayer0..4`) populate their `draft_kv_caches` (`target_hidden_states`) starting from token `N` onward as `1D` decodes. Over `1,024`-token reasoning/tool-calling generations, DSpark rapidly builds local draft context on `1D` and sustains $\tau = 2.74\text{--}2.83$ across all batch sizes.
- **Per-Stream Decode Speed vs. Engine Step Time**:
  - At `n = 8`: `StepChunk Med = 27.01ms` $\div \tau (2.83) \implies \text{Per-Token ITL Med} = 9.78\text{ms}$ (**`102.21 tok/s`** per stream).
  - At `n = 64` (`8` active requests/DP rank): `StepChunk Med = 34.89ms` $\div \tau (2.74) \implies \text{Per-Token ITL Med} = 13.73\text{ms}$ (**`72.84 tok/s`** per stream).
  - At `n = 128` (`16` active requests/DP rank): `StepChunk Med = 42.30ms` $\div \tau (2.74) \implies \text{Per-Token ITL Med} = 15.44\text{ms}$ (**`64.75 tok/s`** per stream).

### 3.3 Exact Sampling Temperature & `top_p` Distribution Verification
Every benchmark request replayed the exact 30-entry `(temperature, top_p)` sampling schedule extracted from `requests_mocked.jsonl.xz`:
- `temperature = <omitted>` (model default `1.0`): **`73.3%`** (`319/435` at `n=128`)
- `temperature = 0.0` (greedy): **`13.3%`** (`58/435` at `n=128`)
- `temperature = 1.0`: **`6.9%`** (`30/435` at `n=128`)
- `temperature = 0.1`: **`3.2%`** (`14/435` at `n=128`)
- `temperature = 0.2`: **`3.2%`** (`14/435` at `n=128`)
- `top_p`: `79.8%` `<omitted>`, `10.3%` `0.95`, `9.9%` `1.0`.

---

## 4. Complete Reproduction Manifest & Commands

### 4.1 1P1D Server & Proxy Launch (`scripts/pd/run_pd.py`)
```bash
python3 scripts/pd/run_pd.py \
  --cluster bodaborg \
  --name emp_isl_1p1d \
  --pod-suffix emp \
  --keep-pod
```
This provisions:
- **Prefill Pod (`1P`)**: `2x2x1` TPU7x (`DP=8, TP=1, EP=8`), port `8400`, `--kv-transfer-config '{"kv_connector":"TPURaidenConnector","kv_role":"kv_producer"}'` with `--enable-prefix-caching --prefix-cache-retention-interval 1024 --max-num-batched-tokens 16384 --max-num-seqs 8 --reasoning-parser deepseek_v41 --tool-call-parser deepseek_v41 --enable-auto-tool-choice`.
- **Decode Pod (`1D`)**: `2x2x1` TPU7x (`DP=8, TP=1, EP=8`), port `9400`, `--kv-transfer-config '{"kv_connector":"TPURaidenConnector","kv_role":"kv_consumer"}'` with `ENABLE_DSPARK=1` (`method="dflash"`, `num_speculative_tokens=5`), `--async-scheduling --enable-prefix-caching --prefix-cache-retention-interval 1024 --max-num-batched-tokens 96 --max-num-seqs 16 --reasoning-parser deepseek_v41 --tool-call-parser deepseek_v41 --enable-auto-tool-choice`.
- **Disaggregated Proxy (`toy_proxy_server.py`)**: Port `8000` on the prefill pod (`--prefill-max-inflight 8` during trunk warmup / `--prefill-max-inflight 1` for single-request gates, forwarding `/reset_prefix_cache` to both `1P` and `1D`).

### 4.2 Empirical ISL (`50:20:12:9:9`) Benchmark Execution (`scripts/pd/bench_empirical_isl_dsv41.py`)
```bash
# 1. Standard 5-or-N Trunks Sweep (n = 8, 16, 32, 64, 128 -> 40, 40, 40, 64, 128 trunks)
python3 -u /tmp/bench_empirical_isl_dsv41.py \
  --base-url http://127.0.0.1:8000 \
  --prefill-url http://127.0.0.1:8400 \
  --decode-url http://<DECODE_POD_IP>:9400 \
  --dataset /data/requests_mocked.jsonl.xz \
  --sessions 8,16,32,64,128 \
  --duration-per-n 60 \
  --cache-hit-rate 0.80 \
  --max-tokens 1024 \
  --ignore-eos \
  --dp-size 8 \
  --trunks-above-n16 0 \
  --warmup-pass-s 8 \
  --num-chips 8 \
  --chip-hourly-usd 5.40 \
  --output-json /tmp/dsv41_pd_empirical_isl_summary.json

# 2. Pinned 40-Trunk High-Concurrency Run (n = 128 -> 40 trunks = 5 per DP rank)
python3 -u /tmp/bench_empirical_isl_dsv41.py \
  --base-url http://127.0.0.1:8000 \
  --prefill-url http://127.0.0.1:8400 \
  --decode-url http://<DECODE_POD_IP>:9400 \
  --dataset /data/requests_mocked.jsonl.xz \
  --sessions 128 \
  --duration-per-n 60 \
  --cache-hit-rate 0.80 \
  --max-tokens 1024 \
  --ignore-eos \
  --dp-size 8 \
  --trunks-above-n16 40 \
  --warmup-pass-s 0 \
  --num-chips 8 \
  --chip-hourly-usd 5.40 \
  --output-json /tmp/dsv41_pd_empirical_isl_n128_tr40_pinned.json
```

### 4.3 Harvested Artifacts & Provenance
- **Benchmark Harness**: [`bench_empirical_isl_dsv41.py`](file:///usr/local/google/home/chaitanyapk/projects/glm_torchtpu/dsv41/scripts/pd/bench_empirical_isl_dsv41.py)
- **Full Sweep JSON Summary (`n = 8, 16, 32, 64, 128`)**: [`dsv41_pd_empirical_isl_summary.json`](file:///usr/local/google/home/chaitanyapk/projects/glm_torchtpu/dsv41/runs/pd/20261029-074610_emp_isl_1p1d_emp/dsv41_pd_empirical_isl_summary.json)
- **Raw Sweep Logs**:
  - [`bench_empirical_isl_full.log`](file:///usr/local/google/home/chaitanyapk/projects/glm_torchtpu/dsv41/runs/pd/20261029-074610_emp_isl_1p1d_emp/bench_empirical_isl_full.log)
  - [`bench_empirical_isl_n128_tr40.log`](file:///usr/local/google/home/chaitanyapk/projects/glm_torchtpu/dsv41/runs/pd/20261029-074610_emp_isl_1p1d_emp/bench_empirical_isl_n128_tr40.log)
  - [`bench_empirical_isl_n128_tr40_pinned.log`](file:///usr/local/google/home/chaitanyapk/projects/glm_torchtpu/dsv41/runs/pd/20261029-074610_emp_isl_1p1d_emp/bench_empirical_isl_n128_tr40_pinned.log)
  - [`prefill__pd_gates.json`](file:///usr/local/google/home/chaitanyapk/projects/glm_torchtpu/dsv41/runs/pd/20261029-074610_emp_isl_1p1d_emp/prefill__pd_gates.json)
