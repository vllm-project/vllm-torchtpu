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
"""VLLM_TPU_MK_QSA_THR_FAST (on-demand tie passes, early-exit value passes, static row
tiers) selects exactly what the plain full-scan selection selects: the same compacted ids
in the same order, on plain, tie-heavy, all-zero and select-all rows (plain jnp at the
served row width S = 512, plus the interpret kernel against the flag off)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.megakernel.wholestep import indexer as ix

I32, F32 = jnp.int32, jnp.float32
S, G, K = 512, 65536, 512


def _bits(key: int, n: int, nvs: list, kind: str):
    s = jax.random.uniform(jax.random.PRNGKey(key), (n, S, 128), F32)
    if kind == "ties":  # few distinct values: truncated ties at the threshold
        s = jnp.floor(s * 7) / 7
    elif kind == "zero":  # the bench's zero keys: every score 0
        s = jnp.zeros_like(s)
    ids = jnp.broadcast_to(jnp.arange(S)[:, None] * 128 + jnp.arange(128)[None], (n, S, 128))
    nv = jnp.asarray(nvs, I32).reshape(n, 1, 1)
    return jnp.where(ids < nv, jax.lax.bitcast_convert_type(s, I32), -1), ids, nv


def _compact(bits, thr, tmax, nv, Sx=S):
    """Compacted ids `[R, 1, K]` and the selection mask `[R, Sx, 128]` (its count is the
    kernel's `kept`)."""
    consts = ix.compact_consts(Sx, K)
    lid, b = consts[-1], bits[:, :Sx]
    ov = nv > K
    sel = (ov & ((b > thr) | ((b == thr) & (lid <= tmax)))) | (~ov & (b >= 0))
    ids = np.stack([
        np.asarray(ix.compact_row(b[r], thr[r], tmax[r], nv[r], K, consts))
        for r in range(bits.shape[0])
    ])
    return np.concatenate([ids.reshape(-1), np.asarray(sel).reshape(-1)])


CASES = [  # (nv per row, kind)
    ([25000, 25001, 25001, 25002], "plain"),
    ([50000, 49000, 50001, 50002], "plain"),
    ([25000, 25001, 25001, 25002], "ties"),
    ([25000, 25001, 30000, 25002], "zero"),
    ([32768, 32769, 49152, 49153], "plain"),  # tier boundaries (256 / 384 rows) +- 1
    ([100, 512, 513, 65536], "plain"),  # select-all, exactly K, K + 1, full
    ([100, 512, 513, 65536], "ties"),
    ([0, 0, 1, 2], "plain"),  # empty rows, nothing over K
]
FAST = [dict(tie_skip=True), dict(early_exit=(19, 2)), dict(early_exit=(30, 1)),
        dict(early_exit=(3, 4)), dict(tie_skip=True, early_exit=(19, 2))]


@pytest.mark.parametrize("nvs,kind", CASES)
def test_fast_thresholds_select_the_same_ids(nvs, kind):
    bits, ids, nv = _bits(sum(nvs) % 997, len(nvs), nvs, kind)
    thr, tmax = ix.thresholds(bits, ids, nv, K, G)
    want = _compact(bits, thr, tmax, nv)
    for kw in FAST:
        thr2, tmax2 = ix.thresholds(bits, ids, nv, K, G, **kw)
        np.testing.assert_array_equal(_compact(bits, thr2, tmax2, nv), want, err_msg=str(kw))
        if not kw.get("early_exit"):  # tie skip alone: the same (thr, tmax)
            np.testing.assert_array_equal(thr2, thr)
            np.testing.assert_array_equal(tmax2, tmax)
    # Row tiers: a tier holding every visible group compacts to the same ids.
    for Sx in (256, 384):
        if max(nvs) <= Sx * 128:
            thr3, tmax3 = ix.thresholds(bits[:, :Sx], ids[:, :Sx], nv, K, G,
                                        tie_skip=True, early_exit=(19, 2))
            got = _compact(bits, thr3, tmax3, nv, Sx)
            n_ids = len(nvs) * K  # ids, then the mask over the tier's rows only
            np.testing.assert_array_equal(got[:n_ids], want[:n_ids])
            np.testing.assert_array_equal(got[n_ids:].reshape(len(nvs), -1),
                                          want[n_ids:].reshape(len(nvs), -1)[:, :Sx * 128])


def test_early_exit_fires_on_gapped_scores():
    """Distinct random scores stop before the last value pass with `lo` below the K-th
    value (so the selection test covers the exit path)."""
    bits, ids, nv = _bits(5, 4, [30000] * 4, "plain")
    thr, _ = ix.thresholds(bits, ids, nv, K, G)
    thr2, tmax2 = ix.thresholds(bits, ids, nv, K, G, early_exit=(19, 2))
    assert (np.asarray(thr2) < np.asarray(thr)).any()
    assert (np.asarray(tmax2) == G).all()


def test_ge_exit_condition_is_caught(monkeypatch):
    """Mutation: stopping once `cnt(bits >= lo) >= K` (instead of `== K`) changes ids."""
    bits, ids, nv = _bits(7, 4, [25000, 25001, 25001, 25002], "plain")
    want = _compact(bits, *ix.thresholds(bits, ids, nv, K, G), nv)
    monkeypatch.setattr(ix, "_exact_top", lambda n_ge, K: n_ge >= K)
    got = _compact(bits, *ix.thresholds(bits, ids, nv, K, G, early_exit=(19, 2)), nv)
    assert not np.array_equal(got, want)


def test_tie_skip_without_truncation_check_is_caught():
    """Mutation: skipping the id passes on a truncated tie (tmax = G) changes the
    selection."""
    bits, ids, nv = _bits(9, 4, [25000, 25001, 25001, 25002], "ties")
    thr, tmax = ix.thresholds(bits, ids, nv, K, G)
    assert (np.asarray(tmax) < G).any()
    want = _compact(bits, thr, tmax, nv)
    assert not np.array_equal(_compact(bits, thr, jnp.full_like(tmax, G), nv), want)


# ------------------------------------------------------------- interpret kernel, flag off/on
@pytest.mark.parametrize("ctx,zero", [([3000, 6000], False), ([7000, 100], False),
                                      ([6000, 100], True)])
def test_kernel_thr_fast_bit_identical(monkeypatch, ctx, zero):
    """The step kernel (long mode, union attention, 16 score rows: tiers 8 / 12 / 16) with
    VLLM_TPU_MK_QSA_THR_FAST (1 = ter2, ter, tec2) gives bit-identical ids, residuals, keys
    and rings: requests in each tier, a select-all request, and all-zero keys (every row a
    truncated tie)."""
    from test_wholestep import _run_c

    ck_fn = (lambda ck: jnp.zeros_like(ck)) if zero else None
    run = lambda: _run_c(long=True, rank_ids=True, union=True, ctx=ctx,  # noqa: E731
                         max_groups=2048, ck_fn=ck_fn)
    monkeypatch.setenv("VLLM_TPU_MK_QSA_THR_FAST", "0")
    base = run()
    for fast, exit_ in (("1", "19,2"), ("ter", "30,1"), ("tec2", "19,2")):
        monkeypatch.setenv("VLLM_TPU_MK_QSA_THR_FAST", fast)
        monkeypatch.setenv("VLLM_TPU_MK_QSA_THR_EXIT", exit_)
        got = run()
        for i, name in ((0, "residual"), (3, "ck_new"), (4, "ring_new"), (7, "ids")):
            np.testing.assert_array_equal(got[i], base[i], err_msg=f"{name} {fast} {exit_}")
