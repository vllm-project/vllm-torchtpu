# SPDX-License-Identifier: Apache-2.0
"""Token-major PLE checkpoints write exactly the request-major (target, value) pairs.

Pure JAX (no Pallas), so it runs on CPU: the writer kernel is shared, so equal
non-null pairs mean equal pool bytes.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.qwen4_exp.ple import short_conv
from vllm_torchtpu.kernels.qwen4_exp.pool import (
    request_checkpoints,
    token_checkpoints,
)

STATE_LEN, WINDOW, CHANNELS = 9, 4, 128


def _pairs(values, targets):
    values, targets = np.asarray(values), np.asarray(targets)
    out = {}
    for row, target in enumerate(targets):
        if target > 0:
            assert target not in out, f"duplicate target {target}"
            out[int(target)] = values[row].view(np.uint16)
    return out


def _case(rng, lengths, windowed, ids_on, initial, tokens):
    count = len(lengths)
    starts = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
    assert starts[-1] <= tokens
    ids = np.where(ids_on, 1 + np.arange(count), 0).astype(np.int32)
    ckpt = (1 + np.arange(count * WINDOW, dtype=np.int32)).reshape(count, WINDOW)
    x = jnp.asarray(rng.normal(size=(tokens, CHANNELS)), jnp.bfloat16)
    history = jnp.asarray(rng.normal(size=(count, STATE_LEN, CHANNELS)), jnp.bfloat16)
    starts, initial = jnp.asarray(starts), jnp.asarray(initial)
    lengths = starts[1:] - starts[:-1]
    active = (jnp.asarray(ids) > 0) & (lengths > 0)
    slots = jnp.where(active, jnp.arange(count, dtype=jnp.int32), -1)
    weight = jnp.asarray(rng.normal(size=(CHANNELS, 4)), jnp.bfloat16)
    _, final = short_conv(
        x,
        weight,
        history,
        starts,
        slots,
        initial,
        dilation=3,
        request_owned_history=True,
    )
    args = (
        x,
        history,
        starts,
        initial,
        active,
        jnp.asarray(ckpt),
        jnp.asarray(windowed),
    )
    old = _pairs(*request_checkpoints(x, history, final, *args[2:]))
    new = _pairs(*token_checkpoints(*args))
    return old, new


@pytest.mark.parametrize(
    "lengths,windowed,ids_on,initial,tokens",
    [
        # verify windows, one idle row, one short row, padded tail
        ([4, 4, 0, 2, 3], [1, 1, 1, 1, 1], [1, 1, 0, 1, 1], [1, 0, 1, 1, 0], 16),
        # decode rows (one token each)
        ([1, 1, 1, 1], [1] * 4, [1, 0, 1, 1], [1, 1, 0, 1], 4),
        # leading windows then a prefill row with and without history
        ([4, 3, 6, 5], [1, 1, 0, 0], [1, 1, 1, 1], [1, 1, 0, 1], 18),
        # a prefill row shorter than the state, idle prefill, zero-length windowed
        ([2, 0, 3, 7], [0, 1, 0, 0], [1, 1, 0, 1], [0, 1, 1, 1], 12),
    ],
)
def test_token_checkpoints_match_request_checkpoints(
    lengths, windowed, ids_on, initial, tokens
):
    old, new = _case(
        np.random.default_rng(len(lengths) * 7 + tokens),
        lengths,
        np.asarray(windowed, bool),
        np.asarray(ids_on, bool),
        np.asarray(initial, bool),
        tokens,
    )
    assert old, "case writes nothing"
    assert old.keys() == new.keys()
    for target in old:
        np.testing.assert_array_equal(
            new[target], old[target], err_msg=f"target {target}"
        )


def test_token_checkpoints_catch_an_off_by_one_window():
    """Positive control: shifting the token windows by one must be detected."""
    rng = np.random.default_rng(3)
    old, new = _case(
        rng,
        [4, 4],
        np.array([1, 1], bool),
        np.array([1, 1], bool),
        np.array([1, 1], bool),
        8,
    )
    shifted = {k: new[k + 1] for k in new if k + 1 in new}
    assert any(not np.array_equal(shifted[k], old[k]) for k in shifted)


def _runner_starts(lengths, count):
    """query_start_loc as TPUModelRunner._prepare_inputs builds it: the
    scheduled requests' cumulative lengths, padded rows repeating the last."""
    starts = np.zeros(count + 1, np.int32)
    starts[1 : len(lengths) + 1] = np.cumsum(lengths)
    starts[len(lengths) + 1 :] = starts[len(lengths)]
    return starts


def _bucket(tokens):
    return max(16, 1 << (int(tokens) - 1).bit_length())


@pytest.mark.parametrize("count", [64, 120, 128, 256])
def test_step_rows_hold_every_request_with_tokens(count):
    """starts[min(R, T)] == starts[-1] for the layouts the runner and the
    warm-up produce, so `step_request_rows` drops only empty rows."""
    from vllm_torchtpu.models.vllm.qwen4_exp.inputs import dummy_query_lengths

    layouts = []
    for tokens in (16, 32, 64, 128, 256, 512, 1024, 2048):
        # Warm-up and profile forwards (Qwen4ExpAdapter.dummy_run), and the
        # DP lock-step idle forward, whose rows are all empty.
        for lengths in (dummy_query_lengths(tokens, count, count), [0] * count):
            starts = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
            layouts.append((starts, tokens))
    for running in range(1, count + 1):
        for window in (1, 2, 3, 4):
            # Decode and verify windows, alone and with a prefill chunk last.
            for lengths in ([window] * running, [window] * (running - 1) + [97]):
                layouts.append((_runner_starts(lengths, count), _bucket(sum(lengths))))
    cut = 0
    for starts, tokens in layouts:
        assert len(starts) == count + 1 and starts[-1] <= tokens
        n = min(count, tokens)
        assert starts[n] == starts[-1], (starts, tokens)
        cut += n < count
    assert cut, "no layout has fewer tokens than rows"


def _speculative_run(
    monkeypatch, lengths, windowed, ids_on, initial, tokens, sliced, skip_idle=False
):
    """pooled_short_conv_speculative with the pool reads and writes in plain
    JAX; returns (output, written pairs, final pool).

    ``skip_idle`` sets VLLM_TPU_PLE_SKIP_IDLE_ROWS; the selected-row read then
    returns NaN in every row it does not select, so any use of such a row
    shows up in the output or the written values.
    """
    import vllm_torchtpu.kernels.qwen4_exp.pool as pool_module

    count = len(lengths)
    pages = 1 + count * (WINDOW + 1)
    rng = np.random.default_rng(count * 31 + tokens)
    elements = STATE_LEN * CHANNELS
    pool = jnp.asarray(rng.normal(size=(pages, elements)), jnp.bfloat16)
    starts = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
    assert starts[-1] <= tokens
    ids = np.where(ids_on, 1 + np.arange(count), 0).astype(np.int32)
    ckpt = (1 + count + np.arange(count * WINDOW, dtype=np.int32)).reshape(
        count, WINDOW
    )
    ckpt = np.where(np.asarray(ids_on)[:, None], ckpt, 0).astype(np.int32)
    read_ids = np.where(ids_on, ckpt[:, 1], 0).astype(np.int32)
    x = jnp.asarray(rng.normal(size=(tokens, CHANNELS)), jnp.bfloat16)
    weight = jnp.asarray(rng.normal(size=(CHANNELS, 4)), jnp.bfloat16)
    writes = []

    def read_region(pool, ids, *, elements, dtype, tok0=0):
        return pool[jnp.maximum(ids, 0), :elements].astype(dtype)

    def read_selected_region(pool, ids, selected, *, elements, dtype):
        rows = read_region(pool, ids, elements=elements, dtype=dtype)
        return jnp.where(selected[:, None], rows, jnp.nan)

    def write_region(pool, values, ids, *, tok0=0, skip_idle_source=False):
        writes.append((values, ids))
        rows = jnp.where(ids > 0, ids, pool.shape[0])
        return pool.at[rows].set(values.astype(pool.dtype), mode="drop")

    monkeypatch.setenv("VLLM_TPU_PLE_SKIP_IDLE_ROWS", "1" if skip_idle else "0")
    monkeypatch.setattr(pool_module, "read_region", read_region)
    monkeypatch.setattr(pool_module, "read_selected_region", read_selected_region)
    monkeypatch.setattr(pool_module, "write_region", write_region)
    if not sliced:
        monkeypatch.setattr(pool_module, "step_request_rows", lambda _, *rows: rows)
    pool, output = pool_module.pooled_short_conv_speculative(
        x,
        weight,
        pool,
        jnp.asarray(starts),
        jnp.asarray(ids),
        jnp.asarray(initial, bool),
        jnp.asarray(read_ids),
        jnp.asarray(ckpt),
        jnp.asarray(windowed, bool),
        dilation=3,
        state_len=STATE_LEN,
    )
    monkeypatch.undo()
    (pair,) = writes
    return (
        np.asarray(output).view(np.uint16),
        _pairs(*pair),
        np.asarray(pool).view(np.uint16),
    )


def _rows(head, count, fill=0):
    return list(head) + [fill] * (count - len(head))


@pytest.mark.parametrize(
    "lengths,windowed,ids_on,initial,tokens",
    [
        # verify windows with an idle row and a short window, padded tail
        (
            _rows([4, 4, 0, 3], 16),
            _rows([1] * 4, 16),
            _rows([1, 1, 0, 1], 16),
            _rows([1, 0, 1, 1], 16),
            12,
        ),
        # one-token decodes filling the bucket exactly
        (
            _rows([1] * 8, 32),
            _rows([1] * 8, 32),
            _rows([1] * 8, 32, 1),
            _rows([1, 1, 0, 1, 1, 1, 0, 1], 32),
            8,
        ),
        # leading windows then prefill rows with and without history
        (
            _rows([4, 4, 7], 32),
            _rows([1, 1], 32),
            _rows([1, 1, 1], 32, 1),
            _rows([1, 1, 0], 32, 1),
            16,
        ),
        # warm-up layout: one-token rows, the last absorbing the rest
        (
            _rows([1, 1, 1, 13], 64),
            _rows([], 64),
            _rows([1] * 4, 64),
            _rows([0] * 4, 64),
            16,
        ),
    ],
)
def test_step_request_rows_keep_outputs_and_writes(
    monkeypatch, lengths, windowed, ids_on, initial, tokens
):
    """T < R: the sliced function writes the same (target, value) pairs and
    returns the same output bytes as the function over all R rows."""
    assert tokens < len(lengths)
    args = (lengths, windowed, ids_on, initial, tokens)
    out_full, pairs_full, pool_full = _speculative_run(monkeypatch, *args, sliced=False)
    out_cut, pairs_cut, pool_cut = _speculative_run(monkeypatch, *args, sliced=True)
    assert pairs_full, "case writes nothing"
    np.testing.assert_array_equal(out_cut, out_full)
    assert pairs_cut.keys() == pairs_full.keys()
    for target in pairs_full:
        np.testing.assert_array_equal(pairs_cut[target], pairs_full[target])
    np.testing.assert_array_equal(pool_cut, pool_full)


@pytest.mark.parametrize(
    "lengths,windowed,ids_on,initial,tokens",
    [
        # verify windows, an idle row in between, a padded tail (T < R)
        (
            _rows([4, 4, 0, 3], 16),
            _rows([1] * 4, 16),
            _rows([1, 1, 0, 1], 16),
            _rows([1, 0, 1, 1], 16),
            12,
        ),
        # windows filling a bucket larger than the rows (T >= R)
        (
            _rows([4] * 6, 8),
            _rows([1] * 6, 8),
            _rows([1] * 6, 8),
            _rows([1] * 6, 8),
            32,
        ),
        # a windowed row with tokens but no state slot
        (
            _rows([4, 4, 4, 2], 8),
            _rows([1] * 4, 8),
            _rows([1, 0, 1, 1], 8),
            _rows([1] * 4, 8),
            16,
        ),
        # windows then prefill rows with and without history (token-major path)
        (
            _rows([4, 4, 7], 32),
            _rows([1, 1], 32),
            _rows([1, 1, 1], 32, 1),
            _rows([1, 1, 0], 32, 1),
            16,
        ),
        # prefill-sized bucket (request-major path: T > R * window)
        (
            _rows([4, 3, 20, 5], 4),
            _rows([1, 1], 4),
            _rows([1, 1, 1, 1], 4),
            _rows([1, 0, 1, 0], 4),
            40,
        ),
    ],
)
def test_skip_idle_rows_keep_outputs_and_writes(
    monkeypatch, lengths, windowed, ids_on, initial, tokens
):
    """VLLM_TPU_PLE_SKIP_IDLE_ROWS: rows the history read skips hold NaN here,
    and the output bytes and the written (target, value) pairs still match."""
    args = (lengths, windowed, ids_on, initial, tokens)
    out_off, pairs_off, pool_off = _speculative_run(monkeypatch, *args, sliced=True)
    out_on, pairs_on, pool_on = _speculative_run(
        monkeypatch, *args, sliced=True, skip_idle=True
    )
    assert pairs_off, "case writes nothing"
    np.testing.assert_array_equal(out_on, out_off)
    assert pairs_on.keys() == pairs_off.keys()
    for target in pairs_off:
        np.testing.assert_array_equal(pairs_on[target], pairs_off[target])
    np.testing.assert_array_equal(pool_on, pool_off)


def test_step_request_rows_catch_a_request_past_the_bucket(monkeypatch):
    """Positive control: a request with tokens at row >= T (a layout the
    runner never builds) loses its writes when the rows are cut."""
    lengths = _rows([4, 0, 0, 0, 0, 0, 0, 0, 0, 4], 16)
    args = (lengths, _rows([1] * 10, 16), _rows([1] * 10, 16), _rows([1] * 10, 16), 8)
    _, pairs_full, _ = _speculative_run(monkeypatch, *args, sliced=False)
    _, pairs_cut, _ = _speculative_run(monkeypatch, *args, sliced=True)
    assert pairs_cut.keys() != pairs_full.keys()


def _scheduled_steps(*indices):
    """Grid steps whose block indices differ from the step before (step 0 always
    moves): the steps the Pallas pipeline issues a fetch or writeback for."""
    indices = np.stack([np.asarray(i) for i in indices], axis=1)
    changed = np.any(indices[1:] != indices[:-1], axis=1)
    return {0} | {int(i) + 1 for i in np.flatnonzero(changed)}


def test_skip_idle_rows_schedule_nothing_for_padded_rows():
    """VLLM_TPU_PLE_SKIP_IDLE_ROWS: padded and idle rows repeat the previous
    step's block indices, so neither kernel moves bytes for them."""
    from vllm_torchtpu.kernels.qwen4_exp import pool as pool_module

    count, tokens = 16, 32
    # Rows 0-4 and 6-7 verify windows; row 5 has tokens but no state slot;
    # rows 8-15 are padding (no tokens, no slot). 30 real tokens, 2 padded.
    lengths = np.array(_rows([4, 4, 4, 4, 4, 4, 4, 2], count))
    ids_on = np.array(_rows([1, 1, 1, 1, 1, 0, 1, 1], count))
    starts = jnp.asarray(np.concatenate([[0], np.cumsum(lengths)]), jnp.int32)
    ids = jnp.asarray(np.where(ids_on, 1 + np.arange(count), 0), jnp.int32)
    ckpt = jnp.asarray(
        np.where(
            ids_on[:, None],
            (100 + np.arange(count * WINDOW)).reshape(count, WINDOW),
            0,
        ),
        jnp.int32,
    )
    read_ids = ckpt[:, 2]
    active = (ids > 0) & (starts[1:] > starts[:-1])
    selected_rows = set(np.flatnonzero(np.asarray(active)).tolist())

    # History read: input block (read id of the owning row) and output row.
    owner = pool_module._last_selected_row(active)
    blocks = jnp.maximum(read_ids, 0)[owner]
    assert _scheduled_steps(blocks, owner) == selected_rows | {0}

    # Checkpoint write: the value row fetched for each token row.
    x = jnp.zeros((tokens, CHANNELS), jnp.bfloat16)
    history = jnp.zeros((count, STATE_LEN, CHANNELS), jnp.bfloat16)
    windowed = jnp.ones((count,), bool)
    _, targets = token_checkpoints(
        x, history, starts, jnp.ones((count,), bool), active, ckpt, windowed
    )
    written = set(np.flatnonzero(np.asarray(targets) > 0).tolist())
    assert len(written) == 26  # 6 rows x 4 + 1 row x 2; row 5 and padding write nothing
    source = pool_module._last_selected_row(targets > 0)
    assert _scheduled_steps(source) == written | {0}
