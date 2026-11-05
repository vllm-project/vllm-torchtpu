# SPDX-License-Identifier: Apache-2.0
"""Real-kernel DCP-2 gate on TPU (DCP2_HARNESS_DEVICE=tpu; skipped elsewhere).

Mosaic batched RPA (served decode env) and the paired-page block-sparse
prefill (kv_tile 1024 = 2 x 512, FP8, D256, 6 q heads per KV head) run per
rank on pools where every unallocated page, the null page 0 and every row
past the sequence end hold the FP8 NaN pattern 0x7F, so a kernel that masks
by multiplication (0 x NaN) shows up as a NaN leak. Each rank's partial is
merged with dcp_combine and compared with the replicated served route on the
same TPU: gated <= 2 BF16 ulps, pre-gate within TPU_TOL, no NaN.
"""

import numpy as np
import pytest
from dcp2_harness import ON_TPU, decode_case, prefill_case
from test_dcp2_decode import SEATS, check_case, production_rpa_attend
from test_dcp2_prefill import check_production_prefill

pytestmark = pytest.mark.skipif(not ON_TPU, reason="DCP2_HARNESS_DEVICE=tpu only")

B = 512
SERVED_DECODE_ENV = {
    "VLLM_TPU_QSA_DECODE_SLOT_READ": "1",
    "VLLM_TPU_QSA_DECODE_LEAN_FETCH": "1",
    "VLLM_TPU_QSA_DECODE_FAST_FETCH": "1",
    "VLLM_TPU_QSA_DECODE_COMPACT_SCHEDULE": "1",
    "VLLM_TPU_QSA_DECODE_RPA_MAX_QUERIES": "128",
}


@pytest.mark.parametrize("served_env", [True, False], ids=["served_env", "default_env"])
@pytest.mark.parametrize("slot", [0, 1])
@pytest.mark.parametrize("rows", [32, 96])
def test_tpu_decode_rpa(monkeypatch, rows, slot, served_env):
    if served_env:
        for k, v in SERVED_DECODE_ENV.items():
            monkeypatch.setenv(k, v)
    case = decode_case(np.random.default_rng(rows + slot), SEATS[rows], pad_rows=4,
                       kv_head_offset=slot)
    print(rows, slot, served_env, check_case(case, production_rpa_attend))


PREFILL = {
    "c2048_p1234": [(1234, 2048)],
    "c2048_p5000": [(5000, 2048)],
    "c8192_p1234": [(1234, 8192)],
    "c8192_p9003": [(9003, 8192)],
    # Ragged multi-request mix: fresh, unaligned prefixes, tails on both owners.
    "ragged_mix": [(5000, 1000), (0, 700), (9003, 347), (2047, 1)],
}


@pytest.mark.parametrize("slot", [0, 1])
@pytest.mark.parametrize("name", sorted(PREFILL))
def test_tpu_prefill_block_sparse(name, slot):
    case = prefill_case(np.random.default_rng(len(name) + slot), PREFILL[name], budget=512,
                        kv_head_offset=slot)
    print(name, slot, check_production_prefill(case))
