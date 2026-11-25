# SPDX-License-Identifier: Apache-2.0
"""q38 HIX-KERN: the in-kernel bundle keys of the 16-row whole step (lane HIX-KERN, 2026-11-24; scratch/hix/kern/step1_plan_v1.txt).

Every key is registered in envs (so it is in the compile key: a warm cache never loads the other arm's program) and defaults off. Each
item is bitwise against its flag-off twin by construction (stated per item in its own module); this module holds only the strict
getters and the per-rank bind line, which `Qwen38WholeStep.__init__` calls on every rank at every engine start (warm or cold), through
logger.info (never info_once):

  'q38 HIX-KERN bind rank=<r>: side_prefetch=<0|1> sidecar=<0|1> route_vec=<0|1> stock_kv=<0|1>'

side_prefetch (VLLM_TPU_MK_KERN_SIDE_PREFETCH, K1 GS-ALL in its sidecar form): the GDN-SIDECAR block starts GDN ordinal o+1's sidecar
state copies at the end of ordinal o's GDN body instead of at the start of ordinal o+1 (models/qwen38_gdn_sidecar.py). It needs
VLLM_TPU_MK_GDN_SIDECAR=1: with the sidecar off the key would be inert, so the bind refuses it (fail closed, no silent no-op arm).
route_vec (VLLM_TPU_MK_KERN_ROUTE_VEC, K3 RC-VEC): the MoE hit list is built by an order-preserving vector pass instead of the scalar
dedup loop (blocks/moe/kern_route.py).
stock_kv (VLLM_TPU_MK_KERN_STOCK_KV, K6): the union walk feeds FP8 K / V to the dots and applies the KV scales after them
(wholestep/union_attn.py, attend_request only: refused with VLLM_TPU_MK_UNION_PIPE, whose tile body it does not edit).
"""

from __future__ import annotations

import logging

from vllm_torchtpu import envs

KERN_BIND_PREFIX = "q38 HIX-KERN bind rank="


def side_prefetch_on() -> bool:
    """VLLM_TPU_MK_KERN_SIDE_PREFETCH (strict bool getter)."""
    return bool(getattr(envs, "VLLM_TPU_MK_KERN_SIDE_PREFETCH", False))


def route_vec_on() -> bool:
    """VLLM_TPU_MK_KERN_ROUTE_VEC (strict bool getter)."""
    return bool(getattr(envs, "VLLM_TPU_MK_KERN_ROUTE_VEC", False))


def stock_kv_on() -> bool:
    """VLLM_TPU_MK_KERN_STOCK_KV (strict bool getter)."""
    return bool(getattr(envs, "VLLM_TPU_MK_KERN_STOCK_KV", False))


def kern_fields() -> dict:
    from vllm_torchtpu.megakernel.models.qwen38_gdn_sidecar import sidecar_on

    return {"side_prefetch": int(side_prefetch_on()), "sidecar": int(sidecar_on()), "route_vec": int(route_vec_on()),
            "stock_kv": int(stock_kv_on())}


def kern_check() -> None:
    """Refusals at engine start (raised, not logged): side_prefetch without the sidecar."""
    f = kern_fields()
    if f["side_prefetch"] and not f["sidecar"]:
        raise ValueError("q38 HIX-KERN: VLLM_TPU_MK_KERN_SIDE_PREFETCH=1 needs VLLM_TPU_MK_GDN_SIDECAR=1 (it prefetches sidecar planes)")
    if f["stock_kv"] and int(getattr(envs, "VLLM_TPU_MK_UNION_PIPE", 0) or 0):
        raise ValueError("q38 HIX-KERN: VLLM_TPU_MK_KERN_STOCK_KV=1 edits attend_request only; unset VLLM_TPU_MK_UNION_PIPE")


def kern_bind_line(rank: int) -> str:
    """Log (logger.info, every call, every rank) and return this rank's HIX-KERN bind line; refuses invalid compositions first."""
    kern_check()
    f = kern_fields()
    msg = "%s%d: %s" % (KERN_BIND_PREFIX, int(rank), " ".join("%s=%d" % (k, v) for k, v in f.items()))
    logging.getLogger("vllm." + __name__).info(msg)
    return msg
