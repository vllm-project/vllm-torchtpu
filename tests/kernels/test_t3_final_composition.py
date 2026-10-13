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
"""Trial-3 final stack: the t4-combined attention composition with
TPU_MLA_DECODE_ROW_WIDTHS=128,256,384 added to the all-on side.

Runs `test_t4_combined_composition.test_attention_all_flags_on_vs_off`
(DCP-8; mixed-256, decode-16, decode-32 and a 2048-token mixed step) with the
row widths in its FLAGS_ON: every rank's caches, partial outputs and lse must
still equal the all-off run bit for bit, and the threaded a2a merge must too.
Wiring: the kernel receives the widths on the all-on run only. The other
trial-3 flags are not attention-kernel flags: TPU_DCP_MERGE_RS_BF16 and
TPU_MTP_DCP_MERGE_SIZED are merge-side (test_t3_final_merge_multichip, which
reads `T3_FINAL_DUMP`), TPU_SAMPLER_VERIFY_TRIM is the runner's sampler and
VLLM_TPU_SEGMENT_TOKENIZE the API server (their own suites).
"""

import os
import sys
from unittest import mock

import pytest

sys.path.insert(0, os.path.dirname(__file__))

import test_t4_combined_composition as t4  # noqa: E402

pytestmark = pytest.mark.tpu_test

ROW_WIDTHS = "128,256,384"
T3_FLAGS_ON = dict(t4.FLAGS_ON, TPU_MLA_DECODE_ROW_WIDTHS=ROW_WIDTHS)


def test_t3_attention_all_flags_on_vs_off():
    from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_kernel

    seen = []
    real = sparse_kernel.sparse_ragged_paged_attention

    def spy(*args, **kwargs):
        seen.append(tuple(kwargs.get("decode_row_widths", ())))
        return real(*args, **kwargs)

    env = {}
    if os.environ.get("T3_FINAL_DUMP"):
        env["T4_COMB_DUMP"] = os.environ["T3_FINAL_DUMP"]
    with (
        mock.patch.object(t4, "FLAGS_ON", T3_FLAGS_ON),
        mock.patch.object(sparse_kernel, "sparse_ragged_paged_attention", spy),
        mock.patch.dict(os.environ, env),
    ):
        os.environ.pop("TPU_MLA_DECODE_ROW_WIDTHS", None)
        t4.test_attention_all_flags_on_vs_off()
    want = tuple(int(w) for w in ROW_WIDTHS.split(","))
    print("decode_row_widths per kernel call:", seen)
    # Off run first (no widths), then the on run (widths on every capped call).
    assert () in seen and want in seen, seen
    assert set(seen) <= {(), want}, seen
