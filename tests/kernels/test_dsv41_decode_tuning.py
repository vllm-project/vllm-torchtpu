# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Config misses must preserve request boundaries without padding the batch."""

from unittest.mock import Mock

import jax
import jax.numpy as jnp
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


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "device,shape,capacity,k,packed,expected",
    [
        ("TPU7x", (48, 6, 32, 128), 49152, 512, False, 8192),
        ("TPU7x", (48, 6, 32, 128), 49152, 512, True, 4096),
        ("other", (48, 6, 32, 128), 49152, 512, False, 4096),
        ("TPU7x", (48, 5, 32, 128), 49152, 512, False, 8192),
        ("TPU7x", (24, 6, 32, 128), 49152, 512, False, 8192),
        ("TPU7x", (48, 6, 32, 128), 40960, 512, False, 8192),
        ("TPU7x", (24, 6, 32, 128), 12288, 512, False, 4096),
        ("TPU7x", (24, 6, 32, 128), 6144, 512, False, 2048),
        ("TPU7x", (48, 12, 32, 128), 49152, 512, False, 4096),
        ("TPU7x", (48, 6, 32, 128), 49152, 256, False, 4096),
    ],
)
def test_candidate_key_tile_defaults_respect_alignment_and_workload(
    device, shape, capacity, k, packed, expected
):
    assert tuning.get_candidate_key_tile(device, shape, capacity, k, packed) == expected


@pytest.mark.cpu_test
def test_candidate_c384_key_tile_retains_full_request_group():
    tile = tuning.get_candidate_key_tile("TPU7x", (48, 6, 32, 128), 49152, 512, False)
    lookup = tuning.CandidateTuningKey("TPU7x", 48, 6, 32, 49152, tile, 512)
    assert tuning.get_candidate_request_tile(lookup) == 48


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "requests,queries,capacity,expected",
    [
        (24, 6, 49152, 24),
        (64, 6, 49152, 32),
        (96, 6, 49152, 48),
        (48, 6, 98304, 24),
        (48, 3, 98304, 48),
        (7, 6, 49152, 7),
    ],
)
def test_candidate_default_groups_bound_scores_without_splitting_requests(
    requests, queries, capacity, expected
):
    lookup = tuning.CandidateTuningKey(
        "TPU7x", requests, queries, 32, capacity, 8192, 512
    )
    tile = tuning.get_candidate_request_tile(lookup)
    assert tile == expected
    assert requests % tile == 0
    assert tile * queries * capacity <= tuning.CANDIDATE_SCORE_ELEMENTS


@pytest.mark.cpu_test
def test_candidate_exact_tuning_precedes_default(monkeypatch):
    lookup = ("TPU7x", (24, 6, 32, 128), 49152, 512, False)
    monkeypatch.setitem(tuning.CANDIDATE_KEY_TILES, lookup, 4096)
    assert tuning.get_candidate_key_tile(*lookup) == 4096
    request_key = tuning.CandidateTuningKey("TPU7x", 24, 6, 32, 49152, 4096, 512)
    monkeypatch.setitem(tuning.CANDIDATE_REQUEST_TILES, request_key, 12)
    assert tuning.get_candidate_request_tile(request_key) == 12


@pytest.mark.cpu_test
@pytest.mark.parametrize("key_tile,expected", [(None, 8192), (4096, 4096)])
@pytest.mark.parametrize("requests,request_tile", [(48, 48), (24, 24), (64, 32)])
@pytest.mark.parametrize("halfplane", [False, True])
@pytest.mark.parametrize("flat", [0, 8])
def test_candidate_indexer_default_dispatch_uses_measured_tiles(
    monkeypatch, key_tile, expected, requests, request_tile, halfplane, flat
):
    from vllm_torchtpu.kernels.deepseek_v41.index_selection import candidate

    calls = []
    expected_halfplane = halfplane

    def score(
        q,
        weights,
        keys,
        ids,
        membership,
        lengths,
        *,
        block_size,
        key_tile,
        halfplane,
        skip,
    ):
        assert halfplane is expected_halfplane and skip
        calls.append((q.shape[0], key_tile))
        return jnp.zeros((*q.shape[:2], ids.shape[1] * block_size), jnp.float32)

    def score_flat(*args, group_tile, **kwargs):
        assert group_tile == flat
        return score(*args, **kwargs).reshape(-1, args[3].shape[1] * 8)

    # The switch is read at trace time: drop traces of the other setting.
    monkeypatch.setenv("TPU_DEEPSEEK_V41_CANDIDATE_FLAT", str(flat))
    candidate.candidate_indexer.clear_cache()
    monkeypatch.setattr(jax, "devices", lambda: [Mock(device_kind="TPU7x")])
    monkeypatch.setattr(candidate, "candidate_scores", score)
    monkeypatch.setattr(
        candidate,
        "candidate_scores_flat",
        score_flat if flat else Mock(side_effect=AssertionError("flat")),
    )
    monkeypatch.setattr(
        candidate.decode,
        "_select",
        lambda scores, lengths, k, *args: jnp.zeros((scores.shape[0], k), jnp.int32),
    )
    inputs = (
        jax.ShapeDtypeStruct((requests, 6, 32, 128), jnp.bfloat16),
        jax.ShapeDtypeStruct((requests, 6, 32), jnp.float32),
        jax.ShapeDtypeStruct((requests, 49152, 128), jnp.bfloat16),
        jax.ShapeDtypeStruct((requests, 6144), jnp.int32),
        jax.ShapeDtypeStruct((requests, 6144), jnp.int32),
        jax.ShapeDtypeStruct((requests, 6), jnp.int32),
    )
    result = jax.eval_shape(
        lambda *args: candidate.candidate_indexer(
            *args, block_size=8, key_tile=key_tile, halfplane=halfplane
        ),
        *inputs,
    )
    assert result.shape == (requests * 6, 512)
    assert calls and all(call == (request_tile, expected) for call in calls)
