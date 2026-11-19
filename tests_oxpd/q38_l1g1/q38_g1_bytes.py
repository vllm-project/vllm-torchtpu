# SPDX-License-Identifier: Apache-2.0
"""q38 L1-G1 op-count: HBM bytes and DMAs of today's GDN checkpoint scatter vs the G1 commit, per served decode step.

  python3 tests/q38_l1g1/q38_g1_bytes.py

Pure arithmetic from the kernels' loop structure (label C; no device, no import of the tree), at 16 / 32 / 48 / 64 rows
(NS = rows / 4 requests, all live), for the whole step's 33 GDN layers and the head call's 3:
  today  (megakernel_runtime/qwen38_scatter.py + the XLA conv-window prep of qwen38_gdn_state._conv_windows):
         prep reads conv_ckpt rows 1..3+T (6 of 8) and writes [NL, NS, T, 3, QKV] f32 windows; the kernel loads every
         checkpoint, pads included (rec [vh, KD, VD] bf16 + the 3-row f32 window: `in_copies`, loop over n = NL*NS*T),
         and writes 400 pool rows (3 x 128 state + 16 conv; 1024 B per row) per live checkpoint;
  G1     (megakernel_runtime/qwen38_gdn_commit.py): an int gather of NL*NS offsets (no float prep); the kernel loads, per
         live request, its accepted rec checkpoint and its 8 conv-history rows, and writes 400 pool rows.
The time column scales T98h's measured scatter (j-02dc361c: 0.323 / 0.495 / 0.698 ms at 32 / 48 / 64 rows, M) by the
byte ratio (I: same effective bandwidth; the commit's per-item DMA pattern is the scatter's).
"""

from __future__ import annotations

VH, KD, VD, QKV, HR, T = 6, 128, 128, 1280, 8, 4
ROW = 1024  # pool bytes per token row (1 x 4 x 256 fp8, or 2 x 256 bf16)
REC_B = VH * KD * VD * 2  # one bf16 checkpoint state
WIN_B = 3 * QKV * 4  # one f32 conv window
HIST_B = HR * QKV * 4  # one request's conv history rows
WRITE_B = (3 * 128 + 16) * ROW  # pool rows written per checkpoint
T98H = {32: 0.323, 48: 0.495, 64: 0.698}


def today(NL: int, rows: int) -> dict:
    NS = rows // T
    n = NL * NS * T
    prep = NL * NS * (3 + T - 1) * QKV * 4 + NL * NS * T * WIN_B
    load = n * (REC_B + WIN_B)
    write = n * WRITE_B
    return dict(prep=prep, load=load, write=write, total=prep + load + write, dmas=n * 2 + n * 4)


def g1(NL: int, rows: int) -> dict:
    NS = rows // T
    k = NL * NS
    prep = k * 4 * 4  # offsets gather + src / dst / accept int32 vectors
    load = k * (REC_B + HIST_B)
    write = k * WRITE_B
    return dict(prep=prep, load=load, write=write, total=prep + load + write, dmas=k * 2 + k * 4)


def staging(NL: int, rows: int) -> int:
    NS = rows // T
    return NL * NS * T * REC_B + NL * NS * HR * QKV * 4 + NL * NS * T * 4 + NL * NS * 4


def main() -> None:
    mib = 2 ** 20
    print("Q38G1B layers rows | today: prep load write total MiB, DMAs | G1: load write total MiB, DMAs | ratio | "
          "T98h ms -> G1 ms (I), saving ms (I) | staging MiB per rank")
    for NL, tag in ((33, "whole"), (3, "head")):
        for rows in (16, 32, 48, 64):
            a, b = today(NL, rows), g1(NL, rows)
            r = b["total"] / a["total"]
            t = T98H.get(rows) if NL == 33 else None
            tcol = f"{t:.3f} -> {t * r:.3f}, saving {t * (1 - r):.3f}" if t else "-"
            print(f"Q38G1B {tag:5s} NL {NL:2d} rows {rows:2d} | {a['prep'] / mib:6.1f} {a['load'] / mib:7.1f} "
                  f"{a['write'] / mib:7.1f} {a['total'] / mib:7.1f}, {a['dmas']:5d} | {b['load'] / mib:6.1f} "
                  f"{b['write'] / mib:6.1f} {b['total'] / mib:6.1f}, {b['dmas']:4d} | {r:.3f} | {tcol} | "
                  f"{staging(NL, rows) / mib:6.1f}")
    print(f"Q38G1B staging at b64, whole + head: {(staging(33, 64) + staging(3, 64)) / 2 ** 30:.3f} GiB per rank "
          f"(whole only {staging(33, 64) / 2 ** 30:.3f} GiB)")


if __name__ == "__main__":
    main()
