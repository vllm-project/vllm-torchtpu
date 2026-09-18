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
"""KV-cache writeback correctness for the batched_rpa kernel."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, wrapper

NUM_KV_HEADS = 2
NUM_Q_HEADS = 4
HEAD_DIM = 128
PAGE_SIZE = 128
KV_PACKING = 2
KV_LENS = (1, 129, 70, 300, 128, 600)

_DECODE_BLOCKS = configs.BlockSizes(
    bq_sz=1, bq_c_sz=1, bkv_sz=256, batch_size=8, n_buffer=3
)

_LAYOUTS = [configs.KVLayout.HEAD_ALONG_SUBLANE]


def _cdiv(a, b):
    return -(-a // b)


def _build(kv_lens, kv_layout, seed=0):
    rng = np.random.default_rng(seed)
    num_seqs = len(kv_lens)
    pages_per_seq = _cdiv(max(kv_lens), PAGE_SIZE)
    kv_dtype = jnp.bfloat16

    q = jnp.asarray(
        rng.standard_normal((num_seqs, NUM_Q_HEADS, HEAD_DIM)) * 0.5, jnp.bfloat16
    )
    k = jnp.asarray(
        rng.standard_normal((num_seqs, NUM_KV_HEADS, HEAD_DIM)) * 0.5, kv_dtype
    )
    v = jnp.asarray(
        rng.standard_normal((num_seqs, NUM_KV_HEADS, HEAD_DIM)) * 0.5, kv_dtype
    )

    shape = wrapper.get_kv_cache_shape(
        pages_per_seq * num_seqs + 4,
        PAGE_SIZE,
        NUM_KV_HEADS,
        HEAD_DIM,
        kv_dtype,
        kv_layout=kv_layout,
    )
    kv_cache = jnp.asarray(rng.standard_normal(shape) * 0.3, kv_dtype)
    page_indices = rng.permutation(pages_per_seq * num_seqs + 4)[
        : pages_per_seq * num_seqs
    ]
    return dict(
        q=q,
        k=k,
        v=v,
        kv_cache=kv_cache,
        kv_lens=jnp.asarray(np.asarray(kv_lens, np.int32)),
        page_indices=jnp.asarray(page_indices.astype(np.int32)),
        cu_q_lens=jnp.asarray(np.arange(num_seqs + 1, dtype=np.int32)),
        distribution=jnp.asarray(np.array([num_seqs] * 3, np.int32)),
        pages_per_seq=pages_per_seq,
        kv_layout=kv_layout,
    )


def _run(b):
    return wrapper.ragged_paged_attention(
        b["q"],
        b["k"],
        b["v"],
        b["kv_cache"],
        b["kv_lens"],
        b["page_indices"],
        b["cu_q_lens"],
        b["distribution"],
        sm_scale=HEAD_DIM**-0.5,
        decode_block_sizes=_DECODE_BLOCKS,
        kv_layout=b["kv_layout"],
    )


def _slot(b, s):
    pos = int(b["kv_lens"][s]) - 1
    page = int(b["page_indices"][s * b["pages_per_seq"] + pos // PAGE_SIZE])
    return page, pos % PAGE_SIZE


def _extract(cache_np, page, lane, head_idx, kv_layout):
    if kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        sub = cache_np[page, head_idx, :, :, lane]
        return sub.reshape(-1)[:HEAD_DIM]
    group, p = divmod(head_idx, KV_PACKING)
    return cache_np[page, lane, group, p, :HEAD_DIM]


@pytest.mark.parametrize("kv_layout", _LAYOUTS, ids=lambda layout: layout.value)
@pytest.mark.parametrize("kv_lens", [KV_LENS])
def test_new_kv_lands_on_the_right_page_and_lane(kv_layout, kv_lens):
    b = _build(kv_lens, kv_layout)
    _, out_cache = jax.block_until_ready(_run(b))
    out = np.asarray(out_cache.astype(jnp.float32))
    tol = 1e-2

    for s in range(len(kv_lens)):
        page, lane = _slot(b, s)
        for h in range(NUM_KV_HEADS):
            k_exp = np.asarray(b["k"][s, h].astype(jnp.float32))
            v_exp = np.asarray(b["v"][s, h].astype(jnp.float32))
            np.testing.assert_allclose(
                _extract(out, page, lane, 2 * h, kv_layout),
                k_exp,
                atol=tol,
                rtol=0,
                err_msg=f"seq {s}: K not written at page {page} lane {lane} head {h}",
            )
            np.testing.assert_allclose(
                _extract(out, page, lane, 2 * h + 1, kv_layout),
                v_exp,
                atol=tol,
                rtol=0,
                err_msg=f"seq {s}: V not written at page {page} lane {lane} head {h}",
            )


@pytest.mark.parametrize("kv_layout", _LAYOUTS, ids=lambda layout: layout.value)
@pytest.mark.parametrize("kv_lens", [KV_LENS])
def test_live_tokens_are_preserved(kv_layout, kv_lens):
    b = _build(kv_lens, kv_layout)
    before = np.array(b["kv_cache"].astype(jnp.float32))
    live = [(s, _slot(b, s)) for s in range(len(kv_lens))]

    _, out_cache = jax.block_until_ready(_run(b))
    after = np.asarray(out_cache.astype(jnp.float32))

    for s, (page, lane) in live:
        for pos in range(int(b["kv_lens"][s]) - 1):
            p = int(b["page_indices"][s * b["pages_per_seq"] + pos // PAGE_SIZE])
            ln = pos % PAGE_SIZE
            for h in range(2 * NUM_KV_HEADS):
                np.testing.assert_array_equal(
                    _extract(after, p, ln, h, kv_layout),
                    _extract(before, p, ln, h, kv_layout),
                    err_msg=f"seq {s}: live token {pos} (page {p} lane {ln} "
                    f"head {h}) was clobbered by the writeback",
                )


@pytest.mark.parametrize("kv_layout", _LAYOUTS, ids=lambda layout: layout.value)
@pytest.mark.parametrize("kv_lens", [KV_LENS])
def test_writeback_touches_only_the_new_token_lane(kv_layout, kv_lens):
    b = _build(kv_lens, kv_layout)
    before = np.array(b["kv_cache"].astype(jnp.float32))
    slots = [_slot(b, s) for s in range(len(kv_lens))]

    _, out_cache = jax.block_until_ready(_run(b))
    after = np.asarray(out_cache.astype(jnp.float32))

    allowed = np.zeros(
        before.shape[0::4]
        if kv_layout == configs.KVLayout.SEQ_ALONG_LANE
        else before.shape[0:2],
        dtype=bool,
    )
    for page, lane in slots:
        allowed[page, lane] = True

    if kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        changed = np.any(after != before, axis=(1, 2, 3))
    else:
        changed = np.any(after != before, axis=(2, 3, 4))

    stray = np.argwhere(changed & ~allowed)
    assert stray.size == 0, (
        f"{len(stray)} (page, lane) positions changed outside any new "
        f"token's own slot; first at {tuple(stray[0])}"
    )


@pytest.mark.parametrize("kv_layout", _LAYOUTS, ids=lambda layout: layout.value)
def test_appended_token_is_visible_to_the_next_step(kv_layout):
    kv_lens = KV_LENS
    b = _build(kv_lens, kv_layout)
    _, cache1 = jax.block_until_ready(_run(b))

    kv_lens2 = tuple(n + 1 for n in kv_lens)
    b2 = _build(kv_lens2, kv_layout, seed=1)
    b2["kv_cache"] = cache1
    b2["page_indices"] = b["page_indices"]
    b2["pages_per_seq"] = b["pages_per_seq"]
    b2["q"] = jnp.asarray(
        np.repeat(
            np.asarray(b["k"].astype(jnp.float32)), NUM_Q_HEADS // NUM_KV_HEADS, axis=1
        ),
        jnp.bfloat16,
    )

    out2, _ = jax.block_until_ready(_run(b2))
    out2 = np.asarray(out2.astype(jnp.float32))
    assert np.isfinite(out2).all()

    for s in range(len(kv_lens)):
        for h in range(NUM_Q_HEADS):
            kv_h = h // (NUM_Q_HEADS // NUM_KV_HEADS)
            v_new = np.asarray(b["v"][s, kv_h].astype(jnp.float32))
            got = out2[s, h]
            assert np.dot(got, v_new) > 0, (
                f"seq {s} head {h}: output does not correlate with the "
                f"value appended by the previous step -- writeback likely "
                f"landed on the wrong lane"
            )
