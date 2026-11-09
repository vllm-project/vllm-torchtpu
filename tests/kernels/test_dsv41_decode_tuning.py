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
"""Config misses must preserve request boundaries without padding the batch."""

from unittest.mock import Mock

import pytest

from vllm_torchtpu.kernels.deepseek_v41.index_selection import (
    tuned_block_sizes as tuning,
)


def key(device_kind="test-device"):
    return tuning.DecodeTuningKey(device_kind, 6, 32, 131072, 128, 512, 2)


@pytest.mark.cpu_test
@pytest.mark.parametrize("requests, expected", [(48, 16), (24, 16), (7, 7), (17, 16)])
def test_decode_tuning_miss_preserves_whole_requests(monkeypatch, requests, expected):
    log = Mock()
    monkeypatch.setattr(tuning, "logger", log)
    lookup = key()
    assert tuning.get_decode_request_tile(lookup, requests) == expected
    log.warning_once.assert_called_once_with(
        "[decode_indexer] No tuned request_tile for %s; using %d", lookup, expected
    )


@pytest.mark.cpu_test
def test_decode_tuning_matches_exact_hardware_and_shape(monkeypatch):
    log = Mock()
    monkeypatch.setattr(tuning, "logger", log)
    monkeypatch.setitem(tuning.DECODE_REQUEST_TILES, key(), 24)
    assert tuning.get_decode_request_tile(key(), 48) == 24
    log.warning_once.assert_not_called()
    assert tuning.get_decode_request_tile(key(device_kind="other-device"), 48) == 16
    assert tuning.get_decode_request_tile(key(), 96) == 24


@pytest.mark.cpu_test
@pytest.mark.parametrize("tile", [0, -1, True])
def test_decode_tuning_rejects_invalid_table_entries(monkeypatch, tile):
    monkeypatch.setitem(tuning.DECODE_REQUEST_TILES, key(), tile)
    with pytest.raises(ValueError, match="Invalid decode request_tile"):
        tuning.get_decode_request_tile(key(), 48)


@pytest.mark.cpu_test
def test_decode_tuning_rejects_empty_batch():
    with pytest.raises(ValueError, match="requests must be positive"):
        tuning.get_decode_request_tile(key(), 0)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "requests,queries,capacity",
    [
        (24, 6, 49152),
        (64, 6, 49152),
        (96, 6, 49152),
        (48, 6, 98304),
        (48, 3, 98304),
        (7, 6, 49152),
    ],
)
def test_candidate_default_groups_bound_scores_without_splitting_requests(
    requests, queries, capacity
):
    lookup = tuning.CandidateTuningKey("TPU7x", queries, 32, capacity, 8192, 512)
    tile = tuning.get_candidate_block_sizes(lookup, requests).request_tile
    assert tile <= requests
    assert tile * queries * capacity <= tuning.CANDIDATE_SCORE_ELEMENTS


@pytest.mark.cpu_test
def test_candidate_exact_tuning_precedes_default(monkeypatch):
    lookup = tuning.CandidateTuningKey("TPU7x", 6, 32, 49152, None, 512)
    config = tuning.CandidateBlockSizes(4096, 12)
    monkeypatch.setitem(tuning.CANDIDATE_BLOCK_SIZES, lookup, config)
    assert tuning.get_candidate_block_sizes(lookup, 24) == config
