# SPDX-License-Identifier: Apache-2.0
"""q38 L4 v5 composition 0-chip HBM fit for the full v5 set (CDK CPU pod). V5COMP-BUILD, after DRAFTEMB-BUILD's
tests/q38_draftemb/q38_draftemb_fit.py (credited; its measured inputs and its call of the real pooled_pool_blocks).

Does the decode pool still size to 98,304 blocks (48 GiB) with every v5 HBM addition allocated before pool sizing?
  * DRAFT-2 (c)'s replicated draft embedding: 248320 x 2560 bf16 = 1,271,398,400 B (C; bound at load);
  * LMH-STREAM stage 0's draft hidden-major LM-head copy: 159,006,720 B per rank (M, its census at Qwen3.8 TP8;
    armed in load_model after the drafter, before the KV pool is sized);
  * L47: +0.35 GiB (L1's figure, taken as given; counted before sizing, the conservative side);
  * v6 (q38-l4-v6c2, V6COMP-BUILD): H3's packed draft-attention operands, DraftAttn's buffers beside the draft's own
    attention weights: qkvg [H, NH/R * 2 * HD + 2 * HD] bf16 at Qwen3.8 FULL geometry, R = 8 (C; the tile padding and the
    smalls / final-norm rows, a few KiB, not counted); T1-K1FUSE adds no persistent tensor (programs only).
    UNMEASURED (stated, not in the sum): the compiled-program HBM of K1's programs, of H3's kernel and of DH-D /
    DRAFT-CAP's programs.
Measured pool inputs (M, T98f j-abb1cbf2 decode pod.log, as draftemb's fit): base arm used 24.61 GiB, cap_available
63.67 GiB; D14 arm used 26.68 GiB, cap_available 61.60 GiB; hbm_limit 94.74 GiB; scratch pool 1.12 GiB.
Prints V5C_FIT lines; last line V5C_FIT PASS|FAIL. The sums are I (from M and C parts)."""

import os
import sys

GiB = 1024**3
os.environ["QWEN4_EXP_MAX_POOL_BYTES"] = str(51539607552)  # FINAL's decode render cap (48 GiB)
os.environ.pop("QWEN4_EXP_POOL_HEADROOM_BYTES", None)
os.environ.pop("QWEN4_EXP_POOL_GRAPH_COPIES", None)

from vllm_torchtpu.models.vllm.qwen4_exp import config as qcfg  # noqa: E402

LIMIT = 94.74 * GiB
SCRATCH = 1.12 * GiB
BLOCK_BYTES = 48 * GiB // 98304
WANT_BLOCKS = 98304
ADDS = {"draft2c_table": 248320 * 2560 * 2, "lmh_draft_copy": 159006720, "l47": int(0.35 * GiB)}


def _h3_operands(R=8):
    from vllm_torchtpu.megakernel.models import qwen38

    g = qwen38.FULL
    return int(g.H) * (int(g.NH) // R * 2 * int(g.HD) + 2 * int(g.HD)) * 2


ADDS["h3_qkvg_operands"] = _h3_operands()
ARMS = {"base": (24.61, 63.67), "d14": (26.68, 61.60)}


def blocks_for(used, cap_available, extra):
    physical = LIMIT - used * GiB + SCRATCH
    return qcfg.pooled_pool_blocks(BLOCK_BYTES, cap_available=int(cap_available * GiB - extra),
                                   physical_available=int(physical - extra), min_blocks=1)


def main():
    fail = []
    headroom = qcfg.pool_headroom_bytes()
    total = sum(ADDS.values())
    print("V5C_FIT additions " + " ".join(f"{k}={v} B ({v / GiB:.3f} GiB)" for k, v in ADDS.items())
          + f" total={total} B ({total / GiB:.3f} GiB); headroom {headroom / GiB:.2f} GiB; block {BLOCK_BYTES} B")
    for arm, (used, cap) in ARMS.items():
        b0, b1 = blocks_for(used, cap, 0), blocks_for(used, cap, total)
        margin = (cap * GiB - total - headroom - WANT_BLOCKS * BLOCK_BYTES) / GiB
        ok = b0 == WANT_BLOCKS and b1 == WANT_BLOCKS and margin > 0
        print(f"V5C_FIT pool {arm}: blocks measured-input {b0}, with the v5 set {b1} (want {WANT_BLOCKS}); margin left at "
              f"pool sizing {margin:.2f} GiB (I) -> {'OK' if ok else 'FAIL'}")
        print(f"V5C_FIT unmeasured {arm}: compiled-program HBM of K1FUSE's programs, H3's kernel and DH-D / DRAFT-CAP's "
              f"programs (not in the sum)")
        if not ok:
            fail.append(f"pool-{arm}")
    bc = blocks_for(*ARMS["d14"], 8 * GiB)
    print(f"V5C_FIT control: +8 GiB before sizing gives {bc} blocks ({'SHRINKS, check bites' if bc < WANT_BLOCKS else 'NO SHRINK'})")
    if bc >= WANT_BLOCKS:
        fail.append("control")
    print("V5C_FIT " + ("PASS" if not fail else "FAIL " + " ".join(fail)))
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
