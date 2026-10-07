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
"""CPU unit tests for `scripts/dsv41_bench/bench_sparse_mla_swa.py`."""

import pathlib
import sys

import jax.numpy as jnp
import numpy as np
import pytest
import torch

sys.path.insert(
    0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts" / "dsv41_bench")
)

import bench_sparse_mla_swa as b_mla  # noqa: E402
import common  # noqa: E402

from vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels import (  # noqa: E402
    _attend,
    _attend_window,
    _attention_state,
    compact_records,
)


@pytest.mark.parametrize("num_seqs", [1, 4])
def test_swa_inputs_match_dense_attend_on_cpu(num_seqs: int):
    tokens = 256
    window_size = 128
    rng = torch.Generator(device="cpu").manual_seed(42)
    q_loc = b_mla._build_query_start_loc(torch, tokens, num_seqs)
    assert q_loc.shape == (num_seqs + 1,)
    assert int(q_loc[0].item()) == 0
    assert int(q_loc[-1].item()) == tokens - 2

    swa_pool, swa_records, swa_slots = b_mla._build_swa_inputs(
        torch, rng, tokens, window_size, q_loc
    )
    assert swa_records.shape == (tokens, window_size)
    assert int((swa_slots[-2:] == -1).all().item()) == 1

    q = (torch.randn((tokens, 8, 512), generator=rng) * 0.25).to(torch.bfloat16)
    q_jax = jnp.asarray(q.float().numpy(), jnp.bfloat16)
    pool_jax = jnp.asarray(swa_pool.numpy(), jnp.int32)
    rec_jax = jnp.asarray(swa_records.numpy(), jnp.int32)
    scale = float(512**-0.5)

    win_max, win_den, win_num = _attend_window(
        q_jax,
        pool_jax,
        rec_jax,
        _attention_state(q_jax),
        scale,
        max_requests=num_seqs,
    )
    ref_max, ref_den, ref_num = _attend(
        q_jax,
        pool_jax,
        rec_jax,
        _attention_state(q_jax),
        scale,
    )
    for name, act, exp in (
        ("swa_max", win_max, ref_max),
        ("swa_den", win_den, ref_den),
        ("swa_num", win_num, ref_num),
    ):
        res = common.compare_tensors(
            torch.from_numpy(np.asarray(act).copy()),
            torch.from_numpy(np.asarray(exp).copy()),
            common.LOSSLESS,
        )
        assert res["ok"], f"{name}: {res}"


def test_compact_records_torch_ref_is_bitwise_identical_to_jax():
    rng = torch.Generator(device="cpu").manual_seed(123)
    tokens = 256
    topk = 512
    q_loc = b_mla._build_query_start_loc(torch, tokens, num_seqs=2)
    _, _, _, main_records = b_mla._build_main_kv_inputs(
        torch,
        rng,
        tokens=tokens,
        kv_len=8192,
        compress_ratio=16,
        page_size=4,
        topk=topk,
        query_start_loc=q_loc,
        unique_per_chunk=512,
    )
    ref = b_mla._compact_records_torch_ref(torch, main_records)
    jax_out = compact_records(jnp.asarray(main_records.numpy(), jnp.int32))
    act = torch.from_numpy(np.asarray(jax_out, dtype=np.int32).copy())
    res = common.compare_tensors(act, ref, common.EXACT)
    assert res["ok"], res


def test_roofline_covers_all_subops_and_modes():
    hw = common.Tpu7xRoofline()
    for subop in b_mla.SUBOPS:
        for mode in b_mla.MODES:
            rep = b_mla._compute_roofline(
                hw=hw,
                subop=subop,
                mode=mode,
                tokens=2048,
                tp_size=8,
                heads_total=64,
                head_dim=512,
                window_size=128,
                topk=512,
                measured_ms=0.5,
            )
            assert rep["lower_bound_ms"] > 0.0
            assert rep["x38_compressed_layers_lower_bound_ms"] == pytest.approx(
                rep["lower_bound_ms"] * b_mla.COMPRESSED_LAYERS
            )
    sc_base = b_mla._compute_roofline(
        hw=hw,
        subop="sc_gather_only",
        mode="baseline",
        tokens=2048,
        tp_size=8,
        heads_total=64,
        head_dim=512,
        window_size=128,
        topk=512,
        measured_ms=0.5,
    )
    sc_union = b_mla._compute_roofline(
        hw=hw,
        subop="sc_gather_only",
        mode="union_gather",
        tokens=2048,
        tp_size=8,
        heads_total=64,
        head_dim=512,
        window_size=128,
        topk=512,
        measured_ms=0.5,
        union_cap=4096,
    )
    # 128 * 512 = 65536 rows/chunk vs 4096 unique rows/chunk -> 16x reduction.
    assert sc_base["hbm_s"] == pytest.approx(16.0 * sc_union["hbm_s"])


@pytest.mark.parametrize("num_seqs", [1, 4])
def test_union_gather_indices_match_bench_kv_inputs_on_cpu(num_seqs: int):
    from vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels import (
        compute_chunk_union_gather_indices,
    )

    rng = torch.Generator(device="cpu").manual_seed(314 + num_seqs)
    tokens = 256
    topk = 512
    page_size = 16
    unique_per_chunk = 2048
    q_loc = b_mla._build_query_start_loc(torch, tokens, num_seqs=num_seqs)
    main_pool, _, pages, main_records = b_mla._build_main_kv_inputs(
        torch,
        rng,
        tokens=tokens,
        kv_len=16384,
        compress_ratio=2,
        page_size=page_size,
        topk=topk,
        query_start_loc=q_loc,
        unique_per_chunk=unique_per_chunk,
    )
    compacted = b_mla._compact_records_torch_ref(torch, main_records)
    total_num_pages = main_pool.shape[0] // page_size
    union_phys, inv_idx, num_unique = compute_chunk_union_gather_indices(
        jnp.asarray(compacted.numpy(), jnp.int32),
        jnp.asarray(pages.reshape(-1).numpy(), jnp.int32),
        jnp.asarray(q_loc.numpy(), jnp.int32),
        page_size=page_size,
        total_num_pages=total_num_pages,
        chunk_size=128,
        union_cap=unique_per_chunk,
    )
    assert int(jnp.max(num_unique)) <= unique_per_chunk
    valid_count = int(q_loc[-1].item())
    tok_per_seq = q_loc[1:] - q_loc[:-1]
    seq_seg = torch.repeat_interleave(
        torch.arange(num_seqs, dtype=torch.int64), tok_per_seq
    )
    rec_valid = compacted[:valid_count].long()
    valid_mask = rec_valid >= 0
    exp_phys = pages[
        seq_seg[:, None], rec_valid.clamp(min=0) // page_size
    ].long() * page_size + (rec_valid.clamp(min=0) % page_size)
    got_phys = torch.from_numpy(
        np.asarray(
            jnp.take_along_axis(union_phys[:, None, :], inv_idx, axis=2).reshape(
                tokens, topk
            )[:valid_count]
        ).copy()
    ).long()
    res = common.compare_tensors(
        got_phys[valid_mask], exp_phys[valid_mask], common.EXACT
    )
    assert res["ok"], res
