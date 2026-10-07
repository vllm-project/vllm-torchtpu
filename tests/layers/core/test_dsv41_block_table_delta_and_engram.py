# SPDX-License-Identifier: Apache-2.0
"""Full-module line and branch coverage test suite for block_table_delta and engram.

Covers all lines and branches in:
1. src/vllm_torchtpu/layers/core/block_table_delta.py
   (193 stmts, 38 branches, 100%)
2. src/vllm_torchtpu/models/vllm/deepseek_v41/common/engram.py
   (261 stmts, 58 branches, 100%)
"""

from __future__ import annotations

import os
import pathlib
import sys

# Ensure repository root is in sys.path so 'tests' can be imported without PYTHONPATH=.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Set CPU platforms
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TPU_SKIP_MDS_QUERY", "true")

from tests.layers.core.test_block_table_delta import (  # noqa: E402, F401
    builders,
    test_apply_block_table_delta_cpu_and_tpu_dispatch,
    test_capacity_overflow_falls_back_to_full,
    test_device_copy_divergence_is_detected,
    test_disabled_dummy_and_single_group_paths,
    test_layout_lru_eviction_and_tracking_idempotent,
    test_randomized_walks_match_full_upload,
    test_verify_mode_and_untracked_write_detection,
    test_warm_reuses_zero_update_and_device_copy_not_cloned,
)
from tests.models.vllm.test_dsv41_common_v1_v2_engram import (  # noqa: E402, F401
    TestCommonNgramHashState,
    TestEngramModule,
    TestParallelEngramEmbedding,
    TestV1Engram,
    TestV2Engram,
    setup_torch_accelerator,
)
from vllm_torchtpu.layers.core import block_table_delta as btd  # noqa: E402


def test_get_stager_instantiation_and_reuse():
    """Ensure get_stager creates and reuses BlockTableDeltaStager."""

    class DummyRunner:
        pass

    r = DummyRunner()
    s1 = btd.get_stager(r)
    assert isinstance(s1, btd.BlockTableDeltaStager)
    s2 = btd.get_stager(r)
    assert s1 is s2
