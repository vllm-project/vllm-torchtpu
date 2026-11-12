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
"""The token-half split's query layout (`minimax_m3/split.py`), on CPU."""

import importlib.util
import pathlib

import numpy as np
import pytest

_SPLIT = (
    pathlib.Path(__file__).parents[3]
    / "src/vllm_torchtpu/models/minimax_m3/split.py"
)
_spec = importlib.util.spec_from_file_location("minimax_m3_split", _SPLIT)
split = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(split)


def _positions(qsl, seq_lens, num_reqs, num_tokens):
    """Absolute position of each real token, as `msa.token_positions`."""
    pos = {}
    for r in range(num_reqs):
        q_len = qsl[r + 1] - qsl[r]
        for i in range(q_len):
            pos[qsl[r] + i] = (r, seq_lens[r] - q_len + i)
    return pos


def _batches():
    rng = np.random.default_rng(0)
    for _ in range(300):
        max_reqs = 8
        num_reqs = int(rng.integers(1, max_reqs + 1))
        q = rng.integers(0, 40, size=num_reqs)
        q[rng.random(num_reqs) < 0.3] = 1
        qsl = np.zeros(max_reqs + 1, np.int32)
        qsl[1 : num_reqs + 1] = np.cumsum(q)
        qsl[num_reqs + 1 :] = qsl[num_reqs]
        seq = np.zeros(max_reqs, np.int32)
        seq[:num_reqs] = q + rng.integers(0, 500, size=num_reqs)
        # Padded rows keep stale lengths (the runner does not clear them).
        seq[num_reqs:] = rng.integers(0, 500, size=max_reqs - num_reqs)
        total = int(qsl[num_reqs])
        bucket = max(16, 1 << int(np.ceil(np.log2(max(total, 1)))))
        yield qsl, seq, num_reqs, bucket


@pytest.mark.parametrize("batch", list(_batches()))
def test_halves_cover_the_step(batch):
    qsl, seq, num_reqs, num_tokens = batch
    split_at = num_tokens // 2
    fq, fs, sq, ss = split.split_query_layout(qsl, seq, split_at)
    whole = _positions(qsl, seq, num_reqs, num_tokens)
    first = _positions(fq, fs, num_reqs, split_at)
    second = {
        t + split_at: v
        for t, v in _positions(sq, ss, num_reqs, num_tokens - split_at).items()
    }
    assert set(first) | set(second) == set(whole)
    assert not set(first) & set(second)
    for t, v in {**first, **second}.items():
        assert whole[t] == v
    # Each half sees the context its queries need, never beyond.
    for r in range(num_reqs):
        if fq[r + 1] > fq[r]:
            assert fs[r] == seq[r] - (sq[r + 1] - sq[r])
        if sq[r + 1] > sq[r]:
            assert ss[r] == seq[r]


@pytest.mark.parametrize("batch", list(_batches()))
def test_join_restores_real_rows(batch):
    qsl, seq, num_reqs, num_tokens = batch
    fq, fs, sq, ss = split.split_query_layout(qsl, seq, num_tokens // 2)
    np.testing.assert_array_equal(fq + sq, qsl)
    joined = split.join_seq_lens(fs, sq, ss)
    real = qsl[1:] - qsl[:-1] > 0
    np.testing.assert_array_equal(joined[real], seq[real])


@pytest.mark.parametrize("batch", list(_batches())[:50])
def test_tensor_split_point_matches_int(batch):
    """The model passes the split point as a 0-d int32 tensor computed in the
    graph (a python int is frozen at the traced bucket)."""
    torch = pytest.importorskip("torch")
    qsl, seq, _, num_tokens = batch
    split_at = num_tokens // 2
    want = split.split_query_layout(qsl, seq, split_at)
    got = split.split_query_layout(
        torch.from_numpy(qsl),
        torch.from_numpy(seq),
        torch.ones(split_at, dtype=torch.int32).sum(),
    )
    for w, g in zip(want, got):
        assert g.dtype == torch.int32
        np.testing.assert_array_equal(g.numpy(), w)
