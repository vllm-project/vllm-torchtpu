# SPDX-License-Identifier: Apache-2.0
"""q38 L4 X-D14-DRAFT-2 (c) 0-chip HBM fit: does the replicated draft embedding leave the decode
pool's 48 GiB cap (98,304 blocks, 8,929,245 tokens) unchanged, with the v4 member set and
X-GLUE-1-D14 composed?

Runs the real ``pooled_pool_blocks`` (models/vllm/qwen4_exp/config.py) on measured inputs:

* T98f j-abb1cbf2 decode pod.log, the 'Qwen4Exp unified pool sizing' lines (M; 8 / 8 ranks equal):
    base arm  hbm_limit=94.74GiB used=24.61GiB (scratch pool 1.12GiB) cap_available=63.67GiB -> 98304 blocks
    D14 arm   hbm_limit=94.74GiB used=26.68GiB (scratch pool 1.12GiB) cap_available=61.60GiB -> 98304 blocks
  (48.00GiB, 0 graph-time pool copies, 6.00GiB headroom). gpu_memory_utilization = (cap_available + used
  - scratch) / hbm_limit = 0.9200 on both arms.
* D14 whole-step operands packed after pool sizing: 2.17 GiB non-shared (M, L1 lane log, T98f pod.log).
* STACK-D v4 composed AOT (L1-V4 j-41971083, output.log md5 60b6bba0, M): whole-step memory_analysis at
  DCP 4 x 8 / 16 / 32 / 48 / 64 rows, temp 0.063 / 0.113 / 0.247 / 0.411 / 0.563 GiB, peak (argument +
  output - alias + temp) 63.176 / 63.228 / 63.365 / 63.532 / 63.688 GiB.
* X-GLUE-1-D14 AOT (L1-GLUE j-5f5ef759, M): flag-on whole-step temp 0.010 / 0.012 / 0.013 / 0.017 GiB at
  16 / 32 / 48 / 64 rows (flag off 0.111 / 0.246 / 0.411 / 0.564).
* The replicated table: Qwen3.8 embed_tokens [248320, 2560] bf16 (reports/model_facts.md:52) =
  1,271,398,400 B (C).

Values are the logged two-decimal GiB figures (each +-0.005 GiB). The sums are I (inferred from M parts):
no TPU run of the full v4 composition exists yet, so the STEP arm must confirm the pool line (M).
Last line: DRAFTEMB_FIT PASS|FAIL.
"""

import os
import sys

GiB = 1024**3
os.environ["QWEN4_EXP_MAX_POOL_BYTES"] = str(51539607552)  # FINAL's decode render cap (48 GiB)
os.environ.pop("QWEN4_EXP_POOL_HEADROOM_BYTES", None)
os.environ.pop("QWEN4_EXP_POOL_GRAPH_COPIES", None)

from vllm_torchtpu.models.vllm.qwen4_exp import config as qcfg  # noqa: E402

LIMIT = 94.74 * GiB
SCRATCH = 1.12 * GiB
BLOCK_BYTES = 48 * GiB // 98304  # 524,288 B per unified-pool block
WANT_BLOCKS = 98304
TABLE = 248320 * 2560 * 2
WHOLESTEP_AFTER = 2.17 * GiB
V4_TEMP = [0.063, 0.113, 0.247, 0.411, 0.563]
V4_PEAK = [63.176, 63.228, 63.365, 63.532, 63.688]
GLUE_TEMP_ON = [0.010, 0.012, 0.013, 0.017]
ARMS = {"base": (24.61, 63.67), "d14": (26.68, 61.60)}


def blocks_for(used, cap_available, extra):
    physical = LIMIT - used * GiB + SCRATCH
    return qcfg.pooled_pool_blocks(
        BLOCK_BYTES,
        cap_available=int(cap_available * GiB - extra),
        physical_available=int(physical - extra),
        min_blocks=1,
    )


def main():
    fail = []
    headroom = qcfg.pool_headroom_bytes()
    print(f"FIT headroom {headroom / GiB:.2f} GiB, cap {int(os.environ['QWEN4_EXP_MAX_POOL_BYTES']) / GiB:.2f} GiB, "
          f"block {BLOCK_BYTES} B, table {TABLE} B = {TABLE / GiB:.3f} GiB (C)")
    for arm, (used, cap) in ARMS.items():
        util = (cap + used - SCRATCH / GiB) / (LIMIT / GiB)
        b0 = blocks_for(used, cap, 0)
        b1 = blocks_for(used, cap, TABLE)
        margin = (cap * GiB - TABLE - headroom - WANT_BLOCKS * BLOCK_BYTES) / GiB
        ok = b0 == WANT_BLOCKS and b1 == WANT_BLOCKS and margin > 0
        print(f"FIT pool {arm}: util {util:.4f}; blocks measured-input {b0} (log said 98304, M), "
              f"with the table {b1}; margin left at pool sizing {margin:.2f} GiB (I) -> {'OK' if ok else 'FAIL'}")
        if not ok:
            fail.append(f"pool-{arm}")
    # Positive control: an 8 GiB addition must shrink the pool, else this check cannot bite.
    bc = blocks_for(*ARMS["d14"], 8 * GiB)
    bites = bc < WANT_BLOCKS
    print(f"FIT control: +8 GiB before sizing gives {bc} blocks ({'SHRINKS, check bites' if bites else 'NO SHRINK: check is blind'})")
    if not bites:
        fail.append("control")
    # Post-pool residency on the D14 arm with the v4 member set's whole-step temp (and with X-GLUE-1-D14's).
    used, _ = ARMS["d14"]
    resident = used * GiB - SCRATCH + WANT_BLOCKS * BLOCK_BYTES + WHOLESTEP_AFTER + TABLE
    cap_total = 0.92 * LIMIT
    for label, temps in (("v4", V4_TEMP), ("v4+GLUE", GLUE_TEMP_ON)):
        peak = resident + max(temps) * GiB
        ok = peak <= cap_total and peak <= LIMIT
        print(f"FIT resident {label}: (used - scratch) {used - SCRATCH / GiB:.2f} + pool 48.00 + whole-step "
              f"after-sizing 2.17 + table {TABLE / GiB:.3f} + max temp {max(temps):.3f} = {peak / GiB:.2f} GiB "
              f"vs 0.92 cap {cap_total / GiB:.2f} (margin {(cap_total - peak) / GiB:.2f}) and hbm_limit "
              f"{LIMIT / GiB:.2f} (margin {(LIMIT - peak) / GiB:.2f}) (I) -> {'OK' if ok else 'FAIL'}")
        if not ok:
            fail.append(f"resident-{label}")
    # The v4 composed AOT whole-step footprint plus the table stays under the device limit.
    worst = max(V4_PEAK) * GiB + TABLE
    ok = worst <= LIMIT
    print(f"FIT v4 AOT: whole-step peak {max(V4_PEAK):.3f} GiB (M, j-41971083) + table = {worst / GiB:.3f} GiB "
          f"<= {LIMIT / GiB:.2f} -> {'OK' if ok else 'FAIL'}")
    if not ok:
        fail.append("v4-aot")
    print("DRAFTEMB_FIT " + ("PASS" if not fail else "FAIL " + " ".join(fail)))
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
