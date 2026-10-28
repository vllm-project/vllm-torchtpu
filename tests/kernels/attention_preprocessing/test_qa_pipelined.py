# SPDX-License-Identifier: Apache-2.0
"""Double buffering changes only data movement, never the Q-A/KV output."""

import numpy as np
import pytest
from test_serial import make_inputs

from vllm_torchtpu.kernels.attention_preprocessing import qa_pipelined, serial


@pytest.mark.cpu_test
@pytest.mark.parametrize("tokens", [256, 768])
def test_matches_serial_bitwise(tokens):
    args, _ = make_inputs(tokens=tokens, head_dim=128)
    qr, kv = qa_pipelined.fused_qa_kv_norm_swa_pipelined(*args, interpret=True)
    qr_ref, kv_ref = serial.fused_qa_kv_norm_swa_serial(*args, interpret=True)
    for actual, expected in ((qr, qr_ref), (kv, kv_ref)):
        np.testing.assert_array_equal(
            np.asarray(actual, np.float32), np.asarray(expected, np.float32)
        )
    np.testing.assert_array_equal(np.asarray(qr[0]), 0)
