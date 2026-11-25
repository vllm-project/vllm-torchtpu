# SPDX-License-Identifier: Apache-2.0
"""q38 HIX-KERN K8 REC-SKIP (lane HIX-KERN under HIX-LEAD-10, 2026-11-25): VLLM_TPU_MK_KERN_REC_SKIP, default off, registered in envs
(hashed: it changes the traced program, so a warm cache never hands one arm the other arm's program).

Under X-K-LIVE mode 1 (VLLM_TPU_MK_LIVE_ROWS=1, LIVE_GDN_COMPUTE unset: the served 16-row whole step) the whole-state GDN recurrence
`gdn_rec` still runs every request slot of the bucket, so a pad request's T rows are computed on its state: 12 / 8 / 4 of the 16 rows
at 1 / 2 / 3 live. With the key on, `gdn_rec_kskip` (blocks/gdn/recurrence.py) runs gdn_rec's row body UNROLLED with the same static
row offsets, each request's T rows under `pl.when(req_live(ctx, s))`; a pad request writes zero `mix` rows and leaves its `recb` rows.

This module holds the strict getter, the refusals and the per-rank bind line, which `Qwen38WholeStep.__init__` calls on every rank at
every engine start (warm or cold) through logger.info (never info_once):

  'q38 HIX-KERN K8 bind rank=<r>: rec_skip=<0|1> live_rows_mode=<0|1|2> gdn_ink=<v>'
"""

from __future__ import annotations

import logging

from vllm_torchtpu import envs

K8_BIND_PREFIX = "q38 HIX-KERN K8 bind rank="


def rec_skip_on() -> bool:
    """VLLM_TPU_MK_KERN_REC_SKIP (strict bool getter)."""
    return bool(getattr(envs, "VLLM_TPU_MK_KERN_REC_SKIP", False))


def k8_fields() -> dict:
    from vllm_torchtpu.megakernel.live_rows import live_rows_mode

    return {"rec_skip": int(rec_skip_on()), "live_rows_mode": int(live_rows_mode()),
            "gdn_ink": int(str(getattr(envs, "VLLM_TPU_MK_GDN_INK", "0") or "0"))}


def k8_check() -> None:
    """Refusals at engine start (raised, not logged): the key needs X-K-LIVE mode 1 and is not composed with GDN-REC-WSHM."""
    f = k8_fields()
    if f["rec_skip"] and f["live_rows_mode"] != 1:
        raise ValueError("q38 HIX-KERN K8: VLLM_TPU_MK_KERN_REC_SKIP=1 needs X-K-LIVE mode 1 (VLLM_TPU_MK_LIVE_ROWS=1, LIVE_GDN_COMPUTE unset);"
                         " live_rows_mode=%d" % f["live_rows_mode"])
    if f["rec_skip"] and (f["gdn_ink"] & 1):
        raise ValueError("q38 HIX-KERN K8: VLLM_TPU_MK_KERN_REC_SKIP=1 is not composed with GDN-REC-WSHM (VLLM_TPU_MK_GDN_INK bit 1)")


def k8_bind_line(rank: int) -> str:
    """Log (logger.info, every call, every rank) and return this rank's K8 bind line; refuses invalid compositions first."""
    k8_check()
    f = k8_fields()
    msg = "%s%d: %s" % (K8_BIND_PREFIX, int(rank), " ".join("%s=%d" % (k, v) for k, v in f.items()))
    logging.getLogger("vllm." + __name__).info(msg)
    return msg
