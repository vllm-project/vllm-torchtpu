# SPDX-License-Identifier: Apache-2.0
"""Configuration guards of the Qwen3.8 megakernel driver (framework-free, CPU-testable)."""

from __future__ import annotations


def whole_step_allowed(
    whole: bool, main_kv_dcp: int, dcp_capable: bool = False, pair_only: bool = False, union: bool = False,
    group_dcp: bool = False, compressed: bool = False,
) -> tuple[bool, str | None]:
    """Whether ``whole=1`` may run, and the reason when it may not.

    The whole-step program writes main KV rows and attends every page of a
    request's context itself (megakernel/wholestep, qwen38_wholestep_cache).
    Under main-KV DCP (VLLM_TPU_QSA_MAIN_KV_DCP > 1) each rank holds only 1/dcp
    of a request's pages, and a whole-step program without a DCP path would
    write whole rows into a DCP-sharded pool and attend a quarter of the
    context as if it were all of it. Until the whole-step program carries a DCP
    path (``dcp_capable``), it is refused there and the segment program, whose
    QSA core is stock and DCP-correct, serves instead.

    ``pair_only`` (the whole step imported from dongruizi / MM 37b960ea9, c32 DCP-2 path): its DCP path is
    main-KV DCP 2 only (pair rank ``me ^ 1``, d640fddff; qsa_full_b asserts dcp in (1, 2)) and needs union
    attention (``union``, ``wholeunion=1``; qsa_full_b: dcp 2 needs union_attn), so DCP > 2, or DCP 2
    without union, is refused there too and the segment program serves.

    ``group_dcp`` (q38 L1-D14, D14): that whole step with its DCP-4 group path (qsa_full_c ``_group_*``: the
    stock DCP group of 4 TP ranks sharing a KV head, peers ``me ^ off``) also runs at main-KV DCP 4, again only
    with union attention. Its compressed-key DCP (``compressed``, VLLM_TPU_QSA_COMPRESSED_DCP) is still pair-only
    (qsa_full_b asserts dcp == 2 for it), so DCP 4 with compressed keys split is refused."""
    if not whole:
        return False, None
    if int(main_kv_dcp or 0) > 1 and not dcp_capable:
        return False, (
            f"whole=1 refused: VLLM_TPU_QSA_MAIN_KV_DCP={int(main_kv_dcp)} and the whole-step "
            "program has no DCP path (it would write whole rows into a DCP-sharded pool); "
            "the segment program serves"
        )
    if pair_only and group_dcp and int(main_kv_dcp or 0) == 4:
        if not union:
            return False, (
                "whole=1 refused: VLLM_TPU_QSA_MAIN_KV_DCP=4 needs wholeunion=1 (the DCP-4 whole step attends "
                "through union attention); the segment program serves"
            )
        if compressed:
            return False, (
                "whole=1 refused: VLLM_TPU_QSA_MAIN_KV_DCP=4 with VLLM_TPU_QSA_COMPRESSED_DCP: the whole step's "
                "compressed-key DCP is main-KV DCP 2 only (pair ranks); the segment program serves"
            )
        return True, None
    if pair_only and int(main_kv_dcp or 0) > 2:
        return False, (
            f"whole=1 refused: VLLM_TPU_QSA_MAIN_KV_DCP={int(main_kv_dcp)} and the whole-step DCP path is "
            "main-KV DCP 2 only (pair ranks); the segment program serves"
        )
    if pair_only and int(main_kv_dcp or 0) == 2 and not union:
        return False, (
            "whole=1 refused: VLLM_TPU_QSA_MAIN_KV_DCP=2 needs wholeunion=1 (the DCP-2 whole step attends "
            "through union attention); the segment program serves"
        )
    return True, None
