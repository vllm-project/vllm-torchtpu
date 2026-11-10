# SPDX-License-Identifier: Apache-2.0
"""fill_step_seq_lens zeroes the padded slots only under the flag."""

import numpy as np
import pytest

tpu_runner = pytest.importorskip("vllm_torchtpu.runner.tpu_runner")


def test_padded_slots_zeroed_when_enabled():
    seq = np.full(8, 99, np.int32)
    tpu_runner.fill_step_seq_lens(
        seq, np.array([5, 7], np.int32), np.array([1, 2], np.int32), zero_padded=True
    )
    assert seq.tolist() == [6, 9, 0, 0, 0, 0, 0, 0]


def test_padded_slots_kept_by_default():
    seq = np.full(4, 99, np.int32)
    tpu_runner.fill_step_seq_lens(seq, np.array([5], np.int32), np.array([1], np.int32))
    assert seq.tolist() == [6, 99, 99, 99]
