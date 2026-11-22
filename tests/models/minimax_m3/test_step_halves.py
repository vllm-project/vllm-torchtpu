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
"""models/minimax_m3/step_halves.split_metadata: the halves partition every
request's queries, and the whole step's metadata is the step's own."""

import dataclasses

import pytest
import torch

from vllm_torchtpu.models.minimax_m3.step_halves import split_metadata


@dataclasses.dataclass
class _Metadata:
    seq_lens: torch.Tensor
    query_start_loc: torch.Tensor
    query_start_loc_cpu: object = dataclasses.field(init=False)
    seq_lens_cpu: object = dataclasses.field(init=False)


def _positions(md: _Metadata) -> list[list[int]]:
    """Per request, the positions of its queries in the step."""
    qsl, lens = md.query_start_loc.tolist(), md.seq_lens.tolist()
    return [
        list(range(lens[i] - (qsl[i + 1] - qsl[i]), lens[i]))
        if qsl[i + 1] > qsl[i]
        else []
        for i in range(len(lens))
    ]


@pytest.mark.parametrize(
    "qsl, seq_lens, split",
    [
        # 3 decodes x 8 rows, an 8168-token chunk straddling the split, unused slots
        ([0, 8, 16, 24, 8192, 8192], [125787, 125787, 125787, 107905, 0], 4096),
        # padded bucket: the real rows end before the split
        ([0, 8, 3000, 3000, 3000], [90000, 40000, 0, 0], 4096),
        # a request wholly in the second half, one with no queries
        ([0, 8, 16, 16, 40], [100, 200, 300, 64], 16),
    ],
)
def test_split_metadata(qsl, seq_lens, split):
    md = _Metadata(
        torch.tensor(seq_lens, dtype=torch.int32),
        torch.tensor(qsl, dtype=torch.int32),
    )
    first, second, whole = split_metadata(md, split)
    assert whole is md
    for part in (first, second):
        assert part.seq_lens.dtype == md.seq_lens.dtype
        assert part.query_start_loc.dtype == md.query_start_loc.dtype
    want, a, b = _positions(md), _positions(first), _positions(second)
    assert all(want[i] == a[i] + b[i] for i in range(len(want)))
    assert first.query_start_loc.tolist()[-1] <= split
    assert second.query_start_loc.tolist()[0] == 0
