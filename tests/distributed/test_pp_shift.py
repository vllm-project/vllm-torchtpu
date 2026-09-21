# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm_torchtpu.distributed.pp_shift import pipeline_permutation

pytestmark = pytest.mark.cpu_test


def test_permutation_forms_one_cycle_per_column():
    rows = ((4, 5), (0, 1), (6, 7), (2, 3))  # stage-major ranks
    perm = pipeline_permutation(rows)
    # Column 0: ranks 4 -> 0 -> 6 -> 2 -> 4; column 1: 5 -> 1 -> 7 -> 3 -> 5.
    assert perm == ((4, 0), (0, 6), (6, 2), (2, 4), (5, 1), (1, 7), (7, 3),
                    (3, 5))
    assert sorted(s for s, _ in perm) == sorted(t for _, t in perm)


def test_one_device_per_stage_is_one_cycle():
    assert pipeline_permutation(
        ((0, ), (1, ), (2, ))) == ((0, 1), (1, 2), (2, 0))
