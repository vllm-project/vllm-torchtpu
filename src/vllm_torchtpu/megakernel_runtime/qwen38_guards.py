# SPDX-License-Identifier: Apache-2.0
"""Configuration guards of the Qwen3.8 megakernel driver (framework-free, CPU-testable)."""

from __future__ import annotations


def whole_step_allowed(whole: bool, main_kv_dcp: int, dcp_capable: bool = False) -> tuple[bool, str | None]:
    """Whether ``whole=1`` may run, and the reason when it may not.

    The whole-step program writes main KV rows and attends every page of a
    request's context itself (megakernel/wholestep, qwen38_wholestep_cache).
    Under main-KV DCP (VLLM_TPU_QSA_MAIN_KV_DCP > 1) each rank holds only 1/dcp
    of a request's pages, and a whole-step program without a DCP path would
    write whole rows into a DCP-sharded pool and attend a quarter of the
    context as if it were all of it. Until the whole-step program carries a DCP
    path (``dcp_capable``), it is refused there and the segment program, whose
    QSA core is stock and DCP-correct, serves instead."""
    if not whole:
        return False, None
    if int(main_kv_dcp or 0) > 1 and not dcp_capable:
        return False, (
            f"whole=1 refused: VLLM_TPU_QSA_MAIN_KV_DCP={int(main_kv_dcp)} and the whole-step "
            "program has no DCP path (it would write whole rows into a DCP-sharded pool); "
            "the segment program serves"
        )
    return True, None
