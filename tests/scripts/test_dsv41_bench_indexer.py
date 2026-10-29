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
"""CPU unit tests for scripts/dsv41_bench/bench_indexer.py."""

import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

BENCH_DIR = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "dsv41_bench"
sys.path.insert(0, str(BENCH_DIR))

import bench_indexer as bi  # noqa: E402
import common  # noqa: E402

from vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels import (  # noqa: E402
    _pad_compacted_records,
    compact_records,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v2.indexer_impl import (  # noqa: E402  # noqa: E402
    _compact_selected,
    _visible_top_k,
    paged_indexer_jax,
    write_index_keys,
)


def test_build_indexer_inputs_and_write_index_keys_cpu():
    inp = bi._build_indexer_inputs(
        rng_seed=42,
        tokens=64,
        tp_size=8,
        tp_rank=0,
        num_seqs=2,
        kv_len=512,
        compress_ratio=2,
        heads=8,
        head_dim=128,
        topk=32,
        candidate_blocks=8,
        block_size=16,
    )
    assert inp["width"] == 256
    assert inp["part"] == 16
    assert inp["q_part_t"].shape == (16, 8, 128)
    assert inp["lengths_part_t"][0].item() == 8  # underfull first row when num_seqs > 1

    # Check each written key and every untouched sibling halfplane independently.
    pool = jnp.asarray(inp["pool_t"].numpy())
    wrec = jnp.asarray(inp["write_records_t"].numpy())
    wkeys = jnp.asarray(inp["write_keys_t"].float().numpy(), jnp.bfloat16)
    updated1 = write_index_keys(pool, wrec, wkeys)
    actual = np.asarray(updated1).view(np.uint32)
    expected = np.asarray(pool).view(np.uint32).copy()
    bits = np.asarray(wkeys.view(jnp.uint16)).astype(np.uint32)
    split = np.concatenate((bits[:, ::2], bits[:, 1::2]), axis=1)
    for record, key in zip(np.asarray(wrec), split):
        if record < 0:
            continue
        shift = np.uint32(16 * (record % 2))
        mask = np.uint32(65535) << shift
        expected[record // 2] = (expected[record // 2] & ~mask) | (key << shift)
    np.testing.assert_array_equal(actual, expected)


def test_visible_top_k_signed_zero_and_lengths_mask_cpu():
    inp = bi._build_indexer_inputs(
        rng_seed=99,
        tokens=64,
        tp_size=8,
        tp_rank=0,
        num_seqs=2,
        kv_len=512,
        compress_ratio=2,
        heads=8,
        head_dim=128,
        topk=32,
        candidate_blocks=8,
        block_size=16,
    )
    scores = jnp.asarray(inp["scores_t"].numpy())
    lengths = jnp.asarray(inp["lengths_part_t"].numpy())
    batch_max = jnp.asarray([inp["batch_max"]], dtype=jnp.int32)
    topk = 32

    idx_ord = _visible_top_k(scores, topk, batch_max.max(), lengths, unordered=False)
    vals_ord = jnp.take_along_axis(scores, jnp.maximum(idx_ord, 0), axis=1)
    sel_ord = jnp.where((idx_ord >= 0) & (vals_ord > -jnp.inf), idx_ord, -1).astype(
        jnp.int32
    )

    idx_unord = _visible_top_k(scores, topk, batch_max.max(), lengths, unordered=True)
    vals_unord = jnp.take_along_axis(scores, jnp.maximum(idx_unord, 0), axis=1)
    sel_unord = _compact_selected(
        jnp.where((idx_unord >= 0) & (vals_unord > -jnp.inf), idx_unord, -1).astype(
            jnp.int32
        )
    )

    _, ref_idx = jax.lax.top_k(scores, topk)
    ref_idx = jnp.sort(ref_idx, axis=-1)
    ref_vals = jnp.take_along_axis(scores, ref_idx, axis=1)
    ref_sel = jnp.where(
        (ref_idx >= 0) & (ref_vals > -jnp.inf) & (lengths[:, None] > 0), ref_idx, -1
    ).astype(jnp.int32)

    exp_comp = torch.from_numpy(np.asarray(compact_records(ref_sel)).copy())
    got_ord_comp = torch.from_numpy(np.asarray(compact_records(sel_ord)).copy())
    got_unord_comp = torch.from_numpy(
        np.asarray(_pad_compacted_records(sel_unord)).copy()
    )

    rep_ord = common.compare_tensors(got_ord_comp, exp_comp, common.EXACT)
    rep_unord = common.compare_tensors(got_unord_comp, exp_comp, common.EXACT)
    assert rep_ord["ok"] and rep_ord["bitwise_equal"], rep_ord
    assert rep_unord["ok"] and rep_unord["bitwise_equal"], rep_unord


@pytest.mark.parametrize(
    ("cand_write", "cand_filter"),
    [(False, False), (True, False), (False, True)],
)
def test_paged_indexer_jax_subops_precompact_matches_baseline_cpu(
    cand_write: bool, cand_filter: bool
):
    inp = bi._build_indexer_inputs(
        rng_seed=77,
        tokens=64,
        tp_size=8,
        tp_rank=0,
        num_seqs=2,
        kv_len=1024,
        compress_ratio=2,
        heads=8,
        head_dim=128,
        topk=32,
        candidate_blocks=8,
        block_size=16,
    )
    q = jnp.asarray(inp["q_part_t"].float().numpy(), jnp.bfloat16)
    pool = jnp.asarray(inp["pool_t"].numpy())
    records = jnp.asarray(inp["records_t"].numpy())
    weights = jnp.asarray(inp["weights_part_t"].numpy())
    lengths = jnp.asarray(inp["lengths_part_t"].numpy())
    candidates = jnp.asarray(inp["candidates_part_t"].numpy())
    requests = jnp.asarray(inp["requests_part_t"].numpy())
    batch_max = jnp.asarray([inp["batch_max"]], dtype=jnp.int32)

    sel_base, pub_base = paged_indexer_jax(
        q,
        pool,
        records,
        weights,
        lengths,
        candidates,
        requests,
        topk=32,
        block_size=16,
        candidate_write=cand_write,
        candidate_filter=cand_filter,
        unordered=False,
        batch_max=batch_max,
    )
    sel_pre, pub_pre = paged_indexer_jax(
        q,
        pool,
        records,
        weights,
        lengths,
        candidates,
        requests,
        topk=32,
        block_size=16,
        candidate_write=cand_write,
        candidate_filter=cand_filter,
        unordered=True,
        batch_max=batch_max,
    )

    comp_base = torch.from_numpy(np.asarray(compact_records(sel_base)).copy())
    comp_pre = torch.from_numpy(np.asarray(_pad_compacted_records(sel_pre)).copy())
    rep_sel = common.compare_tensors(comp_pre, comp_base, common.EXACT)
    rep_pub = common.compare_tensors(
        torch.from_numpy(np.asarray(pub_pre).copy()),
        torch.from_numpy(np.asarray(pub_base).copy()),
        common.EXACT,
    )
    assert rep_sel["ok"] and rep_sel["bitwise_equal"], rep_sel
    assert rep_pub["ok"] and rep_pub["bitwise_equal"], rep_pub


@pytest.mark.parametrize("subop", bi.SUBOPS)
@pytest.mark.parametrize("mode", bi.MODES)
def test_compute_roofline_indexer(subop: str, mode: str):
    hw = common.Tpu7xRoofline()
    rep = bi._compute_roofline(
        hw,
        subop=subop,
        mode=mode,
        tokens=2048,
        part=272,
        tp_size=8,
        width=45056,
        heads=64,
        head_dim=128,
        topk=512,
        candidate_blocks=128,
        block_size=16,
        measured_ms=1.0,
    )
    assert rep["lower_bound_ms"] > 0.0
    assert rep["roofline_efficiency"] > 0.0
