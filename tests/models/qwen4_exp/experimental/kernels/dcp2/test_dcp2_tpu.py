# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Real-kernel DCP-2 gate on TPU (started by test_dcp2_tpu_lane.py).

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

from tests.models.qwen4_exp.experimental.kernels.dcp2.dcp2_harness import (
    ON_TPU,
    decode_case,
    prefill_case,
)
from tests.models.qwen4_exp.experimental.kernels.dcp2.test_dcp2_decode import (
    SEATS,
    check_case,
    production_rpa_attend,
)
from tests.models.qwen4_exp.experimental.kernels.dcp2.test_dcp2_prefill import (
    check_production_prefill,
)
from vllm_torchtpu.models.qwen4_exp.experimental.kernels import qsa_rpa

# Run on the TPU lane through test_dcp2_tpu_lane.py, which starts this module
# with the harness in TPU mode (it picks its mode once, at import).
pytestmark = [
    pytest.mark.tpu_test,
    pytest.mark.skipif(not ON_TPU, reason="run through test_dcp2_tpu_lane.py"),
]

B = 512
# The large-batch decode options (decode_batch=large in variants.py).
SERVED_DECODE_OPTIONS = {
    "VLLM_TPU_QSA_DECODE_COMPACT_SCHEDULE": True,
    "VLLM_TPU_QSA_DECODE_RPA_MAX_QUERIES": 128,
}


@pytest.mark.parametrize("served_env", [True, False], ids=["served_env", "default_env"])
@pytest.mark.parametrize("slot", [0, 1])
@pytest.mark.parametrize("rows", [32, 96])
def test_tpu_decode_rpa(monkeypatch, rows, slot, served_env):
    if served_env:
        for k, v in SERVED_DECODE_OPTIONS.items():
            monkeypatch.setitem(qsa_rpa._OPTIONS, k, v)
    case = decode_case(
        np.random.default_rng(rows + slot), SEATS[rows], pad_rows=4, kv_head_offset=slot
    )
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
    case = prefill_case(
        np.random.default_rng(len(name) + slot),
        PREFILL[name],
        budget=512,
        kv_head_offset=slot,
    )
    print(name, slot, check_production_prefill(case))
