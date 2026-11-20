# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""The two-level TP all-reduce's chip table, and its data movement replayed
on the host: every rank must end with every piece reduced exactly once."""

import numpy as np
import pytest

from vllm_torchtpu.kernels.collectives.tp_all_reduce import two_level_table


def _rows(table, n):
    width = len(table) // n
    return [table[i * width : (i + 1) * width] for i in range(n)]


def _replay(table, n, x):
    """The kernel's moves on [n, cols] partials (pieces of cols / n columns)."""
    chips = n // 2
    rows = _rows(table, n)
    w = x.shape[1] // n
    half_w = chips * w
    out = np.full_like(x, np.nan)
    owned = {}
    for me, (partner, pos, half, *group) in enumerate(rows):
        assert group[pos] == me
        for q in range(chips):
            target = group[q]
            start = half * half_w + q * w
            # The reducer of piece q of this half gets every chip's pair sum.
            pair = x[me, start : start + w] + x[partner, start : start + w]
            owned.setdefault((target, start), []).append(pair)
    for (target, start), parts in owned.items():
        assert len(parts) == chips
        assert rows[target][2] * half_w + rows[target][1] * w == start
        total = np.sum(parts, axis=0)
        # Gathered across chips to the same half, then to the partner.
        for me in range(n):
            out[me, start : start + w] = total
    return out


@pytest.mark.parametrize("n", [4, 8, 16])
def test_two_level_replay_reduces_every_piece_once(n):
    rng = np.random.default_rng(n)
    order = rng.permutation(n)
    chips = [sorted(order[i : i + 2].tolist()) for i in range(0, n, 2)]
    table = two_level_table(chips)
    rows = _rows(table, n)
    for me, (partner, pos, half, *group) in enumerate(rows):
        assert rows[partner][0] == me and rows[partner][1] == pos
        assert rows[partner][2] == 1 - half
        assert sorted([*group, *rows[partner][3:]]) == list(range(n))
    x = rng.integers(-5, 6, size=(n, 16 * n)).astype(np.float64)
    out = _replay(table, n, x)
    np.testing.assert_array_equal(out, np.broadcast_to(x.sum(0), out.shape))


def test_two_level_table_refuses_other_shapes():
    assert two_level_table([[0, 1]]) is None
    assert two_level_table([[0, 1, 2, 3], [4, 5, 6, 7]]) is None
    assert two_level_table([[0], [1]]) is None
