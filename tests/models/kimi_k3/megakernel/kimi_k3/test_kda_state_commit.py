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

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl

from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.kernels import paged

_SENTINEL_CONV = 77.0
_SENTINEL_REC = 999.0


@pytest.fixture
def interpret_pallas(monkeypatch: pytest.MonkeyPatch) -> None:
    original = pl.pallas_call

    def _run_interpreted(*args, **kwargs):
        return original(*args, **kwargs, interpret=True)

    monkeypatch.setattr(pl, "pallas_call", _run_interpreted)


def _reference_commit(
    conv_slots: np.ndarray,
    rec_slots: np.ndarray,
    new_conv: np.ndarray,
    new_rec: np.ndarray,
    slots: list[int],
    *,
    source_rows: list[int] | None = None,
    null_period: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    expected_conv = conv_slots.copy()
    expected_rec = rec_slots.copy()
    for i, target in enumerate(slots):
        if null_period is not None and target % null_period == 0:
            continue
        src = source_rows[i] if source_rows is not None else i
        expected_conv[target] = new_conv[src]
        expected_rec[target] = new_rec[src]
    return expected_conv, expected_rec


def _make_state_fixtures(
    num_slots: int, num_src_rows: int
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    conv_slots = jnp.full((num_slots, 2, 1, 1, 1, 128), _SENTINEL_CONV, jnp.bfloat16)
    rec_slots = jnp.full((num_slots, 2, 1, 8, 128), _SENTINEL_REC, jnp.float32)
    row_ids = jnp.arange(num_src_rows, dtype=jnp.float32) + 1.0
    lane_ramp = jnp.arange(128, dtype=jnp.float32) * 0.01
    new_conv = (row_ids[:, None, None, None, None, None] * 10.0 + lane_ramp).astype(
        jnp.bfloat16
    )
    new_conv = jnp.broadcast_to(new_conv, (num_src_rows, 2, 1, 1, 1, 128))
    new_rec = row_ids[:, None, None, None, None] * 100.0 + lane_ramp
    new_rec = jnp.broadcast_to(new_rec, (num_src_rows, 2, 1, 8, 128))
    return conv_slots, rec_slots, new_conv, new_rec


@pytest.mark.parametrize("batch_size", [2, 4, 8, 16, 32])
def test_should_select_two_buffers_for_f32_multi_row_batches(
    batch_size: int,
) -> None:
    buffers = paged.kda_commit_buffer_count(
        batch_size=batch_size,
        conv_row_shape=(69, 3, 3, 4, 128),
        conv_dtype=jnp.bfloat16,
        recurrent_row_shape=(69, 3, 128, 128),
        recurrent_dtype=jnp.float32,
    )
    assert buffers == 2


def test_should_select_single_buffer_for_batch_one_or_oversized_rows() -> None:
    assert (
        paged.kda_commit_buffer_count(
            batch_size=1,
            conv_row_shape=(69, 3, 3, 4, 128),
            conv_dtype=jnp.bfloat16,
            recurrent_row_shape=(69, 3, 128, 128),
            recurrent_dtype=jnp.float32,
        )
        == 1
    )
    assert (
        paged.kda_commit_buffer_count(
            batch_size=4,
            conv_row_shape=(0,),
            conv_dtype=jnp.uint8,
            recurrent_row_shape=(20 * 1024 * 1024,),
            recurrent_dtype=jnp.uint8,
        )
        == 2
    )
    assert (
        paged.kda_commit_buffer_count(
            batch_size=4,
            conv_row_shape=(1,),
            conv_dtype=jnp.uint8,
            recurrent_row_shape=(20 * 1024 * 1024,),
            recurrent_dtype=jnp.uint8,
        )
        == 1
    )


@pytest.mark.parametrize(
    ("slots", "null_period"),
    [
        ([3, 1, 0, 5, 2, 4, 7, 6], 8),
        ([2, 2, 5, 5], 8),
        ([2, 0, 2, 5], 8),
        ([0, 0, 0, 0], 8),
        ([1, 0, 3, 4, 5, 0, 7, 6], 4),
        ([0, 2, 0, 3], None),
    ],
)
def test_should_commit_ping_pong_kda_states_in_input_order_and_skip_null_slots(
    interpret_pallas: None,
    slots: list[int],
    null_period: int | None,
) -> None:
    del interpret_pallas
    conv_slots, rec_slots, new_conv, new_rec = _make_state_fixtures(8, len(slots))
    expected_conv, expected_rec = _reference_commit(
        np.asarray(conv_slots),
        np.asarray(rec_slots),
        np.asarray(new_conv),
        np.asarray(new_rec),
        slots,
        null_period=null_period,
    )

    out_conv, out_rec = paged.commit_kda_states(
        conv_slots,
        rec_slots,
        new_conv,
        new_rec,
        jnp.asarray(slots, jnp.int32),
        null_period=null_period,
    )

    np.testing.assert_array_equal(np.asarray(out_conv), expected_conv)
    np.testing.assert_array_equal(np.asarray(out_rec), expected_rec)


def test_should_commit_selected_source_rows_with_ping_pong_and_null_skip(
    interpret_pallas: None,
) -> None:
    del interpret_pallas
    conv_slots, rec_slots, new_conv, new_rec = _make_state_fixtures(8, 10)
    slots = [3, 0, 3, 6]
    source_rows = [7, 9, 2, 5]
    expected_conv, expected_rec = _reference_commit(
        np.asarray(conv_slots),
        np.asarray(rec_slots),
        np.asarray(new_conv),
        np.asarray(new_rec),
        slots,
        source_rows=source_rows,
        null_period=8,
    )

    out_conv, out_rec = paged.commit_kda_states(
        conv_slots,
        rec_slots,
        new_conv,
        new_rec,
        jnp.asarray(slots, jnp.int32),
        source_rows=jnp.asarray(source_rows, jnp.int32),
        null_period=8,
    )

    np.testing.assert_array_equal(np.asarray(out_conv), expected_conv)
    np.testing.assert_array_equal(np.asarray(out_rec), expected_rec)


@pytest.mark.parametrize(("slot_val", "null_period"), [(3, 8), (0, 8), (0, None)])
def test_should_commit_single_row_via_single_buffer_path(
    interpret_pallas: None,
    slot_val: int,
    null_period: int | None,
) -> None:
    del interpret_pallas
    conv_slots, rec_slots, new_conv, new_rec = _make_state_fixtures(8, 3)
    slots = [slot_val]
    source_rows = [2]
    expected_conv, expected_rec = _reference_commit(
        np.asarray(conv_slots),
        np.asarray(rec_slots),
        np.asarray(new_conv),
        np.asarray(new_rec),
        slots,
        source_rows=source_rows,
        null_period=null_period,
    )

    out_conv, out_rec = paged.commit_kda_states(
        conv_slots,
        rec_slots,
        new_conv,
        new_rec,
        jnp.asarray(slots, jnp.int32),
        source_rows=jnp.asarray(source_rows, jnp.int32),
        null_period=null_period,
    )

    np.testing.assert_array_equal(np.asarray(out_conv), expected_conv)
    np.testing.assert_array_equal(np.asarray(out_rec), expected_rec)


@pytest.mark.parametrize(
    ("batch_size", "attention_dp", "expect_pallas_commit"),
    [
        (1, 1, True),
        (2, 1, True),
        (4, 1, True),
        (8, 1, True),
        (16, 1, True),
        (8, 2, True),
        (16, 2, True),
        (8, 4, False),
    ],
)
def test_should_dispatch_decode_paged_kda_commit_for_dp1_and_dp2_all_batches(
    monkeypatch: pytest.MonkeyPatch,
    batch_size: int,
    attention_dp: int,
    expect_pallas_commit: bool,
) -> None:
    committed: list[tuple[int | None, tuple[int, ...]]] = []

    def fake_decode_local(weights, states, tokens, positions, *, batch_size, **kwargs):
        del weights, states, tokens, positions, kwargs
        logits = jnp.zeros((1, batch_size, 16), jnp.float32)
        conv = jnp.zeros((batch_size, 2, 1, 1, 1, 128), jnp.bfloat16)
        rec = jnp.zeros((batch_size, 2, 1, 8, 128), jnp.float32)
        new_keys = jnp.zeros((batch_size, 1, 3, 256), jnp.bfloat16)
        new_vals = jnp.zeros((batch_size, 1, 1, 128), jnp.bfloat16)
        return logits, (conv, rec, new_keys, new_vals)

    def spy_commit(
        conv_slots, rec_slots, conv, rec, slots, source_rows=None, null_period=None
    ):
        del conv, rec, source_rows
        committed.append((null_period, slots.shape))
        return conv_slots, rec_slots

    monkeypatch.setattr(paged, "decode_local", fake_decode_local)
    monkeypatch.setattr(paged, "commit_kda_states", spy_commit)
    monkeypatch.setattr(paged, "commit_latent_rows", lambda pools, *a, **k: pools)
    monkeypatch.setattr(jax.lax, "axis_index", lambda _axis: jnp.int32(0))

    num_slots = 16
    conv_slots = jnp.zeros((num_slots, 2, 1, 1, 1, 128), jnp.bfloat16)
    rec_slots = jnp.zeros((num_slots, 2, 1, 8, 128), jnp.float32)
    pool = (jnp.zeros((2, 128, 2, 640), jnp.bfloat16),)
    tokens = jnp.zeros((batch_size,), jnp.int32)
    positions = jnp.zeros((batch_size,), jnp.int32)
    block_tables = jnp.zeros((batch_size, 4), jnp.int32)
    slots = jnp.arange(1, batch_size + 1, dtype=jnp.int32)

    paged.decode_paged(
        weights={},
        conv_slots=conv_slots,
        recurrent_slots=rec_slots,
        pool=pool,
        tokens=tokens,
        positions=positions,
        block_tables=block_tables,
        slots=slots,
        batch_size=batch_size,
        dcp_pool_map=(0,),
        attention_dp=attention_dp,
        sequence_parallel_size=1,
    )

    if expect_pallas_commit:
        assert committed == [(num_slots // attention_dp, (batch_size,))]
    else:
        assert committed == []
