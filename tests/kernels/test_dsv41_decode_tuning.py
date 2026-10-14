# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Config misses must preserve request boundaries without padding the batch."""

from unittest.mock import Mock

import pytest

from vllm_torchtpu.kernels.deepseek_v41.index_selection import (
    tuned_block_sizes as tuning,
)


def key(requests=48, device_kind="test-device"):
    return tuning.DecodeTuningKey(
        device_kind, requests, 6, 32, 131072, 128, 4096, 512, 2
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize("requests, expected", [(48, 16), (24, 12), (7, 7), (17, 1)])
def test_decode_tuning_miss_preserves_whole_requests(monkeypatch, requests, expected):
    log = Mock()
    monkeypatch.setattr(tuning, "logger", log)
    lookup = key(requests)
    assert tuning.get_decode_request_tile(lookup) == expected
    log.warning_once.assert_called_once_with(
        "[decode_indexer] No tuned request_tile for %s; using %d", lookup, expected
    )


@pytest.mark.cpu_test
def test_decode_tuning_matches_exact_hardware_and_shape(monkeypatch):
    log = Mock()
    monkeypatch.setattr(tuning, "logger", log)
    monkeypatch.setitem(tuning.DECODE_REQUEST_TILES, key(), 24)
    assert tuning.get_decode_request_tile(key()) == 24
    log.warning_once.assert_not_called()
    assert tuning.get_decode_request_tile(key(device_kind="other-device")) == 16
    assert tuning.get_decode_request_tile(key(requests=96)) == 16


@pytest.mark.cpu_test
@pytest.mark.parametrize("tile", [0, 5, True])
def test_decode_tuning_rejects_invalid_table_entries(monkeypatch, tile):
    monkeypatch.setitem(tuning.DECODE_REQUEST_TILES, key(), tile)
    with pytest.raises(ValueError, match="Invalid decode request_tile"):
        tuning.get_decode_request_tile(key())


@pytest.mark.cpu_test
def test_decode_tuning_rejects_empty_batch():
    with pytest.raises(ValueError, match="requests must be positive"):
        tuning.get_decode_request_tile(key(requests=0))
