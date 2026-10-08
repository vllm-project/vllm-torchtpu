# SPDX-License-Identifier: Apache-2.0
"""Speculative decoding implementations for vLLM on TorchTPU."""

from vllm_torchtpu.spec_decode.dflash import DFlashProposer
from vllm_torchtpu.spec_decode.dspark import (
    DEEPSEEK_V4_DEFAULT_K_MTP,
    MAX_SAFE_TREE_BRANCHING_FACTOR,
    MLA_COMPRESSED_KV_DIM,
    MLA_ROPE_DIM,
    MLA_TOTAL_KV_DIM,
    SPARSECORE_SRAM_CEILING_BYTES,
    DecodeTPOTModelResult,
    DSparkHostDispatchOverlapManager,
    DSparkProposer,
    DSparkTreeArbiter,
    DSparkTreeArbiterResult,
    SparseCoreBufferFootprint,
    SparseCoreSRAMSpillError,
    Stage3CFP8KVCacheEngine,
    run_stage3c_gate7b_verification,
)

__all__ = [
    "DFlashProposer",
    "DSparkProposer",
    "DSparkTreeArbiter",
    "DSparkHostDispatchOverlapManager",
    "Stage3CFP8KVCacheEngine",
    "SparseCoreSRAMSpillError",
    "SparseCoreBufferFootprint",
    "DSparkTreeArbiterResult",
    "DecodeTPOTModelResult",
    "SPARSECORE_SRAM_CEILING_BYTES",
    "MAX_SAFE_TREE_BRANCHING_FACTOR",
    "DEEPSEEK_V4_DEFAULT_K_MTP",
    "MLA_COMPRESSED_KV_DIM",
    "MLA_ROPE_DIM",
    "MLA_TOTAL_KV_DIM",
    "run_stage3c_gate7b_verification",
]
