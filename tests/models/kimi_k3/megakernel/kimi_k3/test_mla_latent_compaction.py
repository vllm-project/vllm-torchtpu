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

from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3 import pools as k3_pools
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3 import prefill as k3_prefill
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.kernels import (
    prefill_megakernel as pm,
)
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.kernels.mla_a2a import (
    mla_local_normed_a2a_attention,
    mla_qkv_up_unstriped_and_prefix_gather_pallas,
    mla_qkv_up_unstriped_pallas,
)
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.layout import FP8_LAYOUT, LAYOUT
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.pools import KDA_LAYERS, MLA_LAYERS
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.prefill import (
    WINDOWS,
    KimiPrefiller,
    PoolWriter,
    max_owned_pages,
    pack_owned_page_writes,
)
from vllm_torchtpu.models.kimi_k3.megakernel.program import RequestInput, StepInput


def _reference_pool_write_mla_only(
    pools_in: list[np.ndarray],
    mla_raw: tuple[int, ...],
    latents_576_bf16: jax.Array,
    table: list[int],
    *,
    block_shape: tuple[int, int, int],
    pool_dtype: Any,
    writes: int,
) -> list[np.ndarray]:
    out = [arr.copy() for arr in pools_in]
    rows_per_block, tokens_per_row, block_lanes = block_shape
    tokens_per_page = rows_per_block * tokens_per_row
    pages = len(table)
    table_arr = jnp.asarray(table, jnp.int32)
    if writes < pages:
        order = jnp.argsort((table_arr == 0).astype(jnp.int32), stable=True)[:writes]
        targets = np.asarray(table_arr[order])
    else:
        order = None
        targets = np.asarray(table_arr)
    for m in range(MLA_LAYERS):
        idx = mla_raw[m]
        page_rows = latents_576_bf16[m].reshape(pages, tokens_per_page, 576)
        if order is not None:
            page_rows = jnp.take(page_rows, order, axis=0)
        rows = (
            jnp.pad(page_rows, ((0, 0), (0, 0), (0, block_lanes - 576)))
            .astype(pool_dtype)
            .reshape(writes, *block_shape)
        )
        rows_bits = np.asarray(
            jax.lax.bitcast_convert_type(
                rows, jnp.uint8 if pool_dtype == jnp.float8_e4m3fn else jnp.uint16
            )
        )
        for p in range(writes):
            out[idx][int(targets[p])] = rows_bits[p]
    return out


def _torch_to_jax_cpu(t: torch.Tensor) -> jax.Array:
    if t.dtype == torch.bfloat16:
        bits = np.ascontiguousarray(t.view(torch.uint16).cpu().numpy())
        return jax.lax.bitcast_convert_type(jnp.asarray(bits, jnp.uint16), jnp.bfloat16)
    if t.dtype == torch.float8_e4m3fn:
        bits = np.ascontiguousarray(t.view(torch.uint8).cpu().numpy())
        return jax.lax.bitcast_convert_type(
            jnp.asarray(bits, jnp.uint8), jnp.float8_e4m3fn
        )
    return jnp.asarray(np.ascontiguousarray(t.cpu().numpy()))


def _jax_to_torch_cpu(a: jax.Array, orig_dtype: torch.dtype) -> torch.Tensor:
    if orig_dtype == torch.bfloat16:
        bits = np.asarray(jax.lax.bitcast_convert_type(a, jnp.uint16))
        return torch.from_numpy(bits.copy()).view(torch.bfloat16)
    if orig_dtype == torch.float8_e4m3fn:
        bits = np.asarray(jax.lax.bitcast_convert_type(a, jnp.uint8))
        return torch.from_numpy(bits.copy()).view(torch.float8_e4m3fn)
    return torch.from_numpy(np.asarray(a).copy()).to(orig_dtype)


def _scatter_kda_state_cpu_ref(pool, ssm_vals, conv_vals, state_indices):
    n_items = state_indices.shape[0]
    packing = pool.shape[2]
    ssm_u = jax.lax.bitcast_convert_type(ssm_vals.reshape(n_items, 77, 640), pool.dtype)
    ssm_rows = ssm_u.transpose(0, 1, 3, 2)
    if jnp.dtype(pool.dtype) == jnp.dtype(jnp.bfloat16):
        conv_rows = conv_vals.reshape(n_items, 4, 2, 640)
    else:
        conv_u = jax.lax.bitcast_convert_type(
            conv_vals.reshape(n_items, 4, 2, 640), pool.dtype
        )
        conv_rows = conv_u.transpose(0, 1, 2, 4, 3).reshape(n_items, 4, packing, 640)
    state_81 = jnp.concatenate((ssm_rows, conv_rows), axis=1)
    return pool.at[state_indices, :81].set(state_81)


def _cpu_register_op(
    name: str,
    mesh: Mesh,
    body: Any,
    in_specs: tuple,
    out_specs: tuple,
    donate_argnums: tuple[int, ...] = (),
    export_key: str | None = None,
) -> Any:
    del name, donate_argnums, export_key
    mapped = jax.jit(
        jax.shard_map(
            body, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False
        )
    )

    def runner(*torch_args: torch.Tensor):
        jax_args = tuple(_torch_to_jax_cpu(arg) for arg in torch_args)
        jax_outs = mapped(*jax_args)
        pool_dtypes = [arg.dtype for arg in torch_args[5 : 5 + len(out_specs)]]
        return tuple(
            _jax_to_torch_cpu(out, dt)
            for out, dt in zip(jax_outs, pool_dtypes, strict=True)
        )

    return runner


@pytest.mark.parametrize(
    ("pages", "slots", "expected"),
    [
        (1, 1, 1),
        (2, 1, 1),
        (2, 2, 2),
        (32, 1, 1),
        (32, 2, 2),
        (32, 16, 16),
        (64, 1, 2),
        (64, 2, 4),
        (40, 2, 4),
        (20, 16, 16),
    ],
)
def test_should_compute_exact_dcp_pigeonhole_bound_in_max_owned_pages(
    pages: int, slots: int, expected: int
) -> None:
    assert max_owned_pages(pages, slots) == expected


def test_should_reject_invalid_bounds_in_owned_page_helpers() -> None:
    with pytest.raises(ValueError, match="pages >= 1"):
        max_owned_pages(0, 2)
    with pytest.raises(ValueError, match="slots >= 1"):
        max_owned_pages(4, 0)
    with pytest.raises(ValueError, match="max_writes >= 1"):
        pack_owned_page_writes([0, 5], 0)
    with pytest.raises(ValueError, match="> max_writes=1"):
        pack_owned_page_writes([0, 11, 0, 22], 1)


def test_should_pack_owned_pages_and_targets_across_all_dcp_ranks_and_windows() -> None:
    for page_tokens in (512, 256):
        for window in WINDOWS:
            pages = window // page_tokens
            for slots in (1, 2, 16):
                eff_slots = max(1, min(slots, pages))
                bound = max_owned_pages(pages, eff_slots)
                base_len = pages // eff_slots
                rem = pages % eff_slots
                pieces: list[tuple[int, int, int]] = []
                offset_page = 0
                for s in range(eff_slots):
                    plen = base_len + (1 if s < rem else 0)
                    pieces.append((offset_page, plen, 7 + 13 * s))
                    offset_page += plen
                for rank in range(32):
                    table = [0] * pages
                    expected_pairs: list[tuple[int, int]] = []
                    for off_p, plen, first_global in pieces:
                        for p in range(plen):
                            if (first_global + p) % 32 == rank:
                                blk = 100 + (off_p + p) * 3 + s
                                table[off_p + p] = blk
                                expected_pairs.append((off_p + p, blk))
                    owned_p, owned_t = pack_owned_page_writes(table, bound)
                    assert len(owned_p) == bound
                    assert len(owned_t) == bound
                    assert (
                        list(
                            zip(
                                owned_p[: len(expected_pairs)],
                                owned_t[: len(expected_pairs)],
                                strict=True,
                            )
                        )
                        == expected_pairs
                    )
                    assert owned_p[len(expected_pairs) :] == [0] * (
                        bound - len(expected_pairs)
                    )
                    assert owned_t[len(expected_pairs) :] == [0] * (
                        bound - len(expected_pairs)
                    )


def test_should_preserve_owned_pages_when_splitting_cp_prefix() -> None:
    owned = jnp.asarray([[3, 0]], jnp.int32)
    arena = pm.CpPrefix(
        pools=("p0", "p1"),
        ids="ids",
        local="loc",
        counts="cnt",
        order="ord",
        owned_pages=owned,
    )

    first_arena, second_arena = arena.split()

    assert first_arena.pools == ("p0",)
    assert second_arena.pools == ("p1",)
    assert first_arena.owned_pages is owned
    assert second_arena.owned_pages is owned


def test_should_reject_invalid_shapes_in_compact_owned_mla_pages() -> None:
    good_lat = jnp.zeros((1024, 640), jnp.bfloat16)
    good_owned = jnp.asarray([1, 0], jnp.int32)
    with pytest.raises(ValueError, match="Incompatible latent"):
        pm.compact_owned_mla_pages(
            jnp.zeros((1, 1024, 640), jnp.bfloat16),
            good_owned,
            block_shape=(128, 4, 640),
            pool_dtype=jnp.float8_e4m3fn,
        )
    with pytest.raises(ValueError, match="Incompatible latent"):
        pm.compact_owned_mla_pages(
            jnp.zeros((768, 640), jnp.bfloat16),
            good_owned,
            block_shape=(128, 4, 640),
            pool_dtype=jnp.float8_e4m3fn,
        )
    with pytest.raises(ValueError, match="Expected 1D owned_pages"):
        pm.compact_owned_mla_pages(
            good_lat,
            jnp.zeros((1, 2), jnp.int32),
            block_shape=(128, 4, 640),
            pool_dtype=jnp.float8_e4m3fn,
        )
    with pytest.raises(ValueError, match="exceeds block_lanes"):
        pm.compact_owned_mla_pages(
            jnp.zeros((1024, 768), jnp.bfloat16),
            good_owned,
            block_shape=(128, 4, 640),
            pool_dtype=jnp.float8_e4m3fn,
        )


@pytest.mark.parametrize(
    ("layout_spec", "jax_dtype", "torch_dtype", "window", "slots", "live", "table"),
    [
        (
            FP8_LAYOUT,
            jnp.float8_e4m3fn,
            torch.float8_e4m3fn,
            1024,
            2,
            2,
            [0, 5],
        ),
        (
            FP8_LAYOUT,
            jnp.float8_e4m3fn,
            torch.float8_e4m3fn,
            2048,
            2,
            1,
            [0, 0, 6, 0],
        ),
        (
            LAYOUT,
            jnp.bfloat16,
            torch.bfloat16,
            1024,
            2,
            2,
            [0, 4, 0, 7],
        ),
        (
            FP8_LAYOUT,
            jnp.float8_e4m3fn,
            torch.float8_e4m3fn,
            1024,
            2,
            1,
            [0, 0],
        ),
    ],
)
def test_should_match_pool_bytes_bitwise_across_compaction_and_pool_writer(
    monkeypatch: pytest.MonkeyPatch,
    layout_spec: Any,
    jax_dtype: Any,
    torch_dtype: torch.dtype,
    window: int,
    slots: int,
    live: int,
    table: list[int],
) -> None:
    one_mesh = Mesh(np.asarray(jax.devices()[:1]), ("k3_sp",))
    monkeypatch.setattr(k3_pools, "scatter_kda_state_fused", _scatter_kda_state_cpu_ref)
    monkeypatch.setattr(k3_prefill, "register_op", _cpu_register_op)
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *a, **kw: self)

    pages = window // layout_spec.tokens_per_block
    max_writes = max_owned_pages(pages, slots)
    writes = max_owned_pages(pages, live)
    owned_p, owned_t = pack_owned_page_writes(table, max_writes)

    rng = np.random.default_rng(1234 + window + live)
    lat_576_f32 = rng.standard_normal((MLA_LAYERS, window, 576), dtype=np.float32) * 2.5
    lat_576 = jnp.asarray(lat_576_f32, jnp.bfloat16)
    lat_640 = jnp.pad(lat_576, ((0, 0), (0, 0), (0, 64)))

    owned_jax = jnp.asarray(owned_p, jnp.int32)
    compacted_from_640 = jnp.stack(
        [
            pm.compact_owned_mla_pages(
                lat_640[m],
                owned_jax,
                block_shape=layout_spec.block_shape,
                pool_dtype=jax_dtype,
            )
            for m in range(MLA_LAYERS)
        ],
        axis=0,
    )
    compacted_from_576 = jnp.stack(
        [
            pm.compact_owned_mla_pages(
                lat_576[m],
                owned_jax,
                block_shape=layout_spec.block_shape,
                pool_dtype=jax_dtype,
            )
            for m in range(MLA_LAYERS)
        ],
        axis=0,
    )
    uint_dtype = jnp.uint8 if jax_dtype == jnp.float8_e4m3fn else jnp.uint16
    np.testing.assert_array_equal(
        np.asarray(jax.lax.bitcast_convert_type(compacted_from_640, uint_dtype)),
        np.asarray(jax.lax.bitcast_convert_type(compacted_from_576, uint_dtype)),
    )

    num_blocks = 10
    np_uint = np.uint8 if jax_dtype == jnp.float8_e4m3fn else np.uint16
    init_bits = [
        rng.integers(1, 120, size=(num_blocks, *layout_spec.block_shape), dtype=np_uint)
        for _ in range(MLA_LAYERS)
    ]
    mla_raw = tuple(range(MLA_LAYERS))
    kda_raw = tuple(i % MLA_LAYERS for i in range(KDA_LAYERS))

    ref_bits = _reference_pool_write_mla_only(
        init_bits,
        mla_raw,
        lat_576,
        table,
        block_shape=layout_spec.block_shape,
        pool_dtype=jax_dtype,
        writes=writes,
    )

    raw_pools_torch = [
        torch.from_numpy(arr.copy()).view(torch_dtype) for arr in init_bits
    ]
    writer = PoolWriter(
        one_mesh, 0, raw_pools_torch, list(mla_raw), list(kda_raw), None
    )
    lat_torch = _jax_to_torch_cpu(compacted_from_640[None], torch_dtype)
    rec = torch.zeros((1, KDA_LAYERS, slots, 3, 128, 128), dtype=torch.float32)
    conv = torch.zeros((1, KDA_LAYERS, slots, 3, 3, 3, 128), dtype=torch.bfloat16)
    zero_kda = torch.zeros(live * KDA_LAYERS, dtype=torch.int32)
    writer.write_packed(
        lat_torch,
        rec,
        conv,
        rec,
        conv,
        torch.tensor([owned_t], dtype=torch.int32),
        zero_kda,
        zero_kda,
        window=window,
        has_ckpt=True,
    )

    actual_bits = [
        t.view(torch.uint8 if torch_dtype == torch.float8_e4m3fn else torch.uint16)
        .cpu()
        .numpy()
        for t in raw_pools_torch
    ]
    for m in range(MLA_LAYERS):
        np.testing.assert_array_equal(actual_bits[m][1:], ref_bits[m][1:])

    if any(blk != 0 for blk in table):
        wrong_targets = [8 if t != 0 else 0 for t in owned_t]
        bad_pools = [
            torch.from_numpy(arr.copy()).view(torch_dtype) for arr in init_bits
        ]
        bad_writer = PoolWriter(
            one_mesh, 0, bad_pools, list(mla_raw), list(kda_raw), None
        )
        bad_writer.write_packed(
            lat_torch,
            rec,
            conv,
            rec,
            conv,
            torch.tensor([wrong_targets], dtype=torch.int32),
            zero_kda,
            zero_kda,
            window=window,
            has_ckpt=False,
        )
        bad_bits = bad_pools[0].view(
            torch.uint8 if torch_dtype == torch.float8_e4m3fn else torch.uint16
        )
        assert not np.array_equal(bad_bits.numpy()[1:], ref_bits[0][1:])


def test_should_reject_mismatched_owned_targets_shape_in_write_packed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    one_mesh = Mesh(np.asarray(jax.devices()[:1]), ("k3_sp",))
    monkeypatch.setattr(k3_prefill, "register_op", _cpu_register_op)
    raw_pools = [
        torch.zeros((4, *FP8_LAYOUT.block_shape), dtype=torch.uint8).view(
            torch.float8_e4m3fn
        )
        for _ in range(MLA_LAYERS)
    ]
    writer = PoolWriter(
        one_mesh,
        0,
        raw_pools,
        list(range(MLA_LAYERS)),
        [i % MLA_LAYERS for i in range(KDA_LAYERS)],
        None,
    )
    lat = torch.zeros(
        (1, MLA_LAYERS, 2, *FP8_LAYOUT.block_shape), dtype=torch.uint8
    ).view(torch.float8_e4m3fn)
    rec = torch.zeros((1, KDA_LAYERS, 2, 3, 128, 128), dtype=torch.float32)
    conv = torch.zeros((1, KDA_LAYERS, 2, 3, 3, 3, 128), dtype=torch.bfloat16)
    kda_blk = torch.zeros(KDA_LAYERS, dtype=torch.int32)

    with pytest.raises(ValueError, match="Incompatible compacted write shapes"):
        writer.write_packed(
            lat,
            rec,
            conv,
            rec,
            conv,
            torch.zeros((1, 4), dtype=torch.int32),
            kda_blk,
            kda_blk,
            window=2048,
        )


@pytest.mark.parametrize("tokens", [16, 32])
def test_should_emit_zero_padded_640_latent_across_mla_qkv_up_unstriped_branches(
    tokens: int,
) -> None:
    num_ranks, stripe, q_rank, tile = 2, 8, 128, 16
    rng = np.random.default_rng(99 + tokens)
    q_kv_rope = jnp.asarray(
        rng.standard_normal((tokens, q_rank + 576), dtype=np.float32), jnp.bfloat16
    )
    both_raw = jnp.pad(q_kv_rope, ((0, 0), (0, 64)))
    q_up = jnp.asarray(
        rng.standard_normal((q_rank, 3 * 256), dtype=np.float32) * 0.1, jnp.bfloat16
    )
    kv_up = jnp.asarray(
        rng.standard_normal((512, 3 * 256), dtype=np.float32) * 0.1, jnp.bfloat16
    )

    q576, k576, v576, lat576 = mla_qkv_up_unstriped_pallas(
        both_raw,
        q_up,
        kv_up,
        num_ranks=num_ranks,
        stripe=stripe,
        q_rank=q_rank,
        tile=tile,
        query_scale=pm.SCALE_LOG2,
        length=tokens - 3,
        return_padded_latent_640=False,
        interpret=True,
    )
    q640, k640, v640, lat640 = mla_qkv_up_unstriped_pallas(
        both_raw,
        q_up,
        kv_up,
        num_ranks=num_ranks,
        stripe=stripe,
        q_rank=q_rank,
        tile=tile,
        query_scale=pm.SCALE_LOG2,
        length=tokens - 3,
        return_padded_latent_640=True,
        interpret=True,
    )

    assert lat576.shape == (tokens, 576)
    assert lat640.shape == (tokens, 640)
    np.testing.assert_array_equal(
        np.asarray(jax.lax.bitcast_convert_type(q640, jnp.uint16)),
        np.asarray(jax.lax.bitcast_convert_type(q576, jnp.uint16)),
    )
    np.testing.assert_array_equal(
        np.asarray(jax.lax.bitcast_convert_type(k640, jnp.uint16)),
        np.asarray(jax.lax.bitcast_convert_type(k576, jnp.uint16)),
    )
    np.testing.assert_array_equal(
        np.asarray(jax.lax.bitcast_convert_type(v640, jnp.uint16)),
        np.asarray(jax.lax.bitcast_convert_type(v576, jnp.uint16)),
    )
    np.testing.assert_array_equal(
        np.asarray(jax.lax.bitcast_convert_type(lat640[:, :576], jnp.uint16)),
        np.asarray(jax.lax.bitcast_convert_type(lat576, jnp.uint16)),
    )
    assert float(jnp.max(jnp.abs(lat640[:, 576:].astype(jnp.float32)))) == 0.0


def test_should_emit_zero_padded_640_latent_in_fused_pg_and_mla_attention() -> None:
    tokens, num_ranks, stripe, q_rank, tile = 16, 1, 16, 128, 16
    rng = np.random.default_rng(777)
    q_kv_rope = jnp.asarray(
        rng.standard_normal((tokens, q_rank + 576), dtype=np.float32), jnp.bfloat16
    )
    both_raw = jnp.pad(q_kv_rope, ((0, 0), (0, 64)))
    q_up = jnp.asarray(
        rng.standard_normal((q_rank, 3 * 256), dtype=np.float32) * 0.1, jnp.bfloat16
    )
    kv_up = jnp.asarray(
        rng.standard_normal((512, 3 * 256), dtype=np.float32) * 0.1, jnp.bfloat16
    )
    pool = jnp.zeros((4, 16, 2, 128), jnp.bfloat16)
    tables = jnp.zeros((3, 3, 1), jnp.int32)
    counts = jnp.asarray([1], jnp.int32)
    ids = jnp.asarray([1, 2], jnp.int32)
    order = jnp.asarray([1, 1], jnp.int32)

    _, _, _, lat576, _ = mla_qkv_up_unstriped_and_prefix_gather_pallas(
        both_raw,
        q_up,
        kv_up,
        pool,
        tables,
        counts,
        ids,
        order,
        num_ranks=num_ranks,
        stripe=stripe,
        q_rank=q_rank,
        tile=tile,
        query_scale=pm.SCALE_LOG2,
        length=13,
        return_padded_latent_640=False,
        interpret=True,
    )
    _, _, _, lat640, _ = mla_qkv_up_unstriped_and_prefix_gather_pallas(
        both_raw,
        q_up,
        kv_up,
        pool,
        tables,
        counts,
        ids,
        order,
        num_ranks=num_ranks,
        stripe=stripe,
        q_rank=q_rank,
        tile=tile,
        query_scale=pm.SCALE_LOG2,
        length=13,
        return_padded_latent_640=True,
        interpret=True,
    )
    assert lat640.shape == (tokens, 640)
    np.testing.assert_array_equal(
        np.asarray(jax.lax.bitcast_convert_type(lat640[:, :576], jnp.uint16)),
        np.asarray(jax.lax.bitcast_convert_type(lat576, jnp.uint16)),
    )
    assert float(jnp.max(jnp.abs(lat640[:, 576:].astype(jnp.float32)))) == 0.0

    if len(jax.devices()) >= 4:
        mesh4 = Mesh(np.asarray(jax.devices()[:4]), ("k3_sp",))
        hidden = jnp.asarray(
            rng.standard_normal((4, 16, 256), dtype=np.float32) * 0.2, jnp.bfloat16
        )
        prefix = jnp.asarray(
            rng.standard_normal((4, 16, 256), dtype=np.float32) * 0.2, jnp.bfloat16
        )
        m_qa = jnp.asarray(
            rng.standard_normal((256, 128), dtype=np.float32) * 0.1, jnp.bfloat16
        )
        m_qb = jnp.asarray(
            rng.standard_normal((128, 3 * 256), dtype=np.float32) * 0.1, jnp.bfloat16
        )
        m_ka = jnp.asarray(
            rng.standard_normal((256, 576), dtype=np.float32) * 0.1, jnp.bfloat16
        )
        m_kb = jnp.asarray(
            rng.standard_normal((512, 3 * 256), dtype=np.float32) * 0.1, jnp.bfloat16
        )
        qnorm = jnp.ones((128,), jnp.bfloat16)
        knorm = jnp.ones((512,), jnp.bfloat16)
        m_gate = jnp.asarray(
            rng.standard_normal((1, 256, 4 * 384), dtype=np.float32) * 0.05,
            jnp.bfloat16,
        )
        m_o = jnp.asarray(
            rng.standard_normal((1, 4 * 384, 256), dtype=np.float32) * 0.05,
            jnp.bfloat16,
        )
        seg_desc = jnp.asarray([0, 64, 0, 0, 0], jnp.int32)

        def _call(pad640: bool):
            return jax.shard_map(
                lambda h, p: mla_local_normed_a2a_attention(
                    h[0],
                    p[0],
                    query_down_projection=m_qa,
                    query_up_projection=m_qb,
                    key_value_down_projection=m_ka,
                    key_value_up_projection=m_kb,
                    query_normalization=qnorm,
                    key_value_normalization=knorm,
                    m_gate_family=m_gate,
                    m_o_family=m_o,
                    slot=0,
                    length=64,
                    segment_desc=seg_desc,
                    num_ranks=4,
                    sp_axis="k3_sp",
                    segments=1,
                    flash_tile=64,
                    token_ranks=(0, 1, 2, 3),
                    flash_fn=lambda _q, _k, v, *a, **kw: jnp.zeros_like(v),
                    use_pallas_a2a=False,
                    use_pallas_owner=False,
                    return_padded_latent_640=pad640,
                    interpret=pltpu.InterpretParams(detect_races=False),
                )[1][None],
                mesh=mesh4,
                in_specs=(P("k3_sp"), P("k3_sp")),
                out_specs=P("k3_sp"),
                check_vma=False,
            )(hidden, prefix)

        lat_a2a_576 = _call(False)
        lat_a2a_640 = _call(True)
        assert lat_a2a_576.shape == (4, 64, 576)
        assert lat_a2a_640.shape == (4, 64, 640)
        np.testing.assert_array_equal(
            np.asarray(
                jax.lax.bitcast_convert_type(lat_a2a_640[:, :, :576], jnp.uint16)
            ),
            np.asarray(jax.lax.bitcast_convert_type(lat_a2a_576, jnp.uint16)),
        )
        assert (
            float(jnp.max(jnp.abs(lat_a2a_640[:, :, 576:].astype(jnp.float32)))) == 0.0
        )


def test_should_wire_owned_pages_and_targets_through_prefiller_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *a, **kw: self)
    seen_run_packed: list[tuple[tuple[int, ...], list[int]]] = []
    seen_write_packed: list[tuple[tuple[int, ...], list[int], int]] = []

    class _FakeProgram:
        windows = (512, 1024, 2048)
        capacity = 2048
        page_tokens = 512

        def window_slots(self, window: int) -> int:
            return max(1, min(2, window // 512))

        def sp_order(self, window: int) -> np.ndarray:
            return np.arange(window, dtype=np.int32)

        def stack_states(self, states: list):
            n = len(states)
            return (
                torch.zeros((1, KDA_LAYERS, n, 3, 128, 128), dtype=torch.float32),
                torch.zeros((1, KDA_LAYERS, n, 3, 3, 3, 128), dtype=torch.bfloat16),
            )

        def split_states(self, rec: torch.Tensor, conv: torch.Tensor):
            return [(rec[:, :, i], conv[:, :, i]) for i in range(int(rec.shape[2]))]

        def run_packed(
            self,
            shard: torch.Tensor,
            extent: torch.Tensor,
            segments: torch.Tensor,
            recurrent: torch.Tensor,
            conv: torch.Tensor,
            pools: list[torch.Tensor],
            ids: torch.Tensor,
            local: torch.Tensor,
            owned_pages: torch.Tensor,
            ring: Any = None,
            *,
            need_logits: bool = True,
        ):
            del extent, segments, pools, ids, local, ring, need_logits
            window = int(shard.shape[-1]) * 32
            max_w = max_owned_pages(window // 512, self.window_slots(window))
            seen_run_packed.append((tuple(owned_pages.shape), owned_pages[0].tolist()))
            latents = torch.ones(
                (1, MLA_LAYERS, max_w, *FP8_LAYOUT.block_shape), dtype=torch.uint8
            ).view(torch.float8_e4m3fn)
            logits = torch.zeros((self.window_slots(window), 16), dtype=torch.float32)
            return logits, latents, recurrent, conv, recurrent, conv

    class _FakeWriter:
        layout = FP8_LAYOUT

        def mla_pools(self) -> list[torch.Tensor]:
            return [torch.zeros((4, *FP8_LAYOUT.block_shape), dtype=torch.uint8)] * 24

        def copy_blocks(self, pairs: list[tuple[int, int]]) -> None:
            del pairs

        def write_packed(
            self,
            latents: torch.Tensor,
            recurrent: torch.Tensor,
            conv: torch.Tensor,
            recurrent_at: torch.Tensor,
            conv_at: torch.Tensor,
            owned_targets: torch.Tensor,
            kda_blocks: torch.Tensor,
            blocks_at: torch.Tensor,
            rings: Any = None,
            *,
            window: int,
            has_ckpt: bool = True,
        ) -> None:
            del (
                recurrent,
                conv,
                recurrent_at,
                conv_at,
                kda_blocks,
                blocks_at,
                rings,
                has_ckpt,
            )
            seen_write_packed.append(
                (tuple(latents.shape), owned_targets[0].tolist(), window)
            )

    class _CpuPrefillerHarness(KimiPrefiller):
        def __init__(self) -> None:
            self.config = SimpleNamespace(
                max_num_seqs=2,
                block_size=512,
                max_num_batched_tokens=2048,
                max_model_len=2048,
            )
            self.page_tokens = 512
            self.cp_rank = 1
            self.cp_capacity = 1024
            self.slots = 2
            self.rank = 1
            self.compute_logits = False
            self.ring = None
            self.draft_latents = None
            self.program = _FakeProgram()
            self._writer = _FakeWriter()
            self._requests = {}
            self._zero_pair = (
                torch.zeros((1, KDA_LAYERS, 3, 128, 128), dtype=torch.float32),
                torch.zeros((1, KDA_LAYERS, 3, 3, 3, 128), dtype=torch.bfloat16),
            )
            self._mla_group = 0
            self._kda_groups = [(1, 512)]

    prefiller = _CpuPrefillerHarness()

    mla_blocks = np.asarray([42, 99], dtype=np.int32)
    kda_blocks = np.arange(10, 10 + 4 * KDA_LAYERS, dtype=np.int32)
    prefiller.step(
        StepInput(
            requests=[
                RequestInput(
                    req_id="req-1024",
                    computed=0,
                    scheduled=1024,
                    token_ids=np.arange(1024, dtype=np.int32),
                    num_tokens=1024,
                    num_prompt_tokens=1024,
                    block_tables=(mla_blocks, kda_blocks),
                )
            ]
        )
    )

    assert seen_run_packed == [((1, 2), [1, 0])]
    assert seen_write_packed == [
        ((1, MLA_LAYERS, 2, *FP8_LAYOUT.block_shape), [42, 0], 1024)
    ]


def test_should_build_compacted_program_and_deduplicate_write_ops_in_compile_and_warmup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *a, **kw: self)
    monkeypatch.setattr(jax.lax, "with_sharding_constraint", lambda x, _s: x)
    one_mesh = Mesh(np.asarray(jax.devices()[:1]), ("k3_sp",))
    monkeypatch.setattr(k3_prefill, "kernel_mesh", lambda axis, size: (one_mesh, 0))

    registered_global: dict[str, tuple[Any, tuple, tuple]] = {}
    registered_local: dict[str, tuple[Any, tuple, tuple]] = {}

    def _stub_global_op(
        name: str,
        mesh: Mesh,
        fn: Any,
        in_specs: tuple,
        out_specs: tuple,
        donate_argnums: tuple[int, ...] = (),
        export_key: str | None = None,
    ) -> Any:
        del mesh, donate_argnums, export_key
        assert len(in_specs) <= 64, f"{name} exceeded 64 Torch custom op args"
        registered_global[name] = (fn, in_specs, out_specs)
        if "_meta_" in name:
            return lambda pool, ids, loc: (
                torch.zeros((32,), dtype=torch.int32),
                torch.zeros((5,), dtype=torch.int32),
            )
        if "_first_" in name:

            def _run_first_stub(*args: Any):
                shard, _extent, _segments, rec, conv = args[:5]
                window = int(shard.shape[-1]) * 32
                max_w = max_owned_pages(window // 512, int(rec.shape[2]))
                fn(*(_torch_to_jax_cpu(a) for a in args))
                return (
                    torch.zeros((1, window // 32, 16), dtype=torch.bfloat16),
                    torch.zeros((1, 8, window // 32, 16), dtype=torch.bfloat16),
                    rec,
                    conv,
                    rec,
                    conv,
                    torch.zeros(
                        (1, MLA_LAYERS, max_w, *FP8_LAYOUT.block_shape),
                        dtype=torch.uint8,
                    ).view(torch.float8_e4m3fn),
                )

            return _run_first_stub
        if "_group8_" in name:
            return lambda *args: args[:7]
        if "_final_nologits_" in name:
            return lambda _p, _b, lat, *_rest: (lat,)
        return lambda _p, _b, lat, *_rest: (
            torch.zeros((2, 16), dtype=torch.float32),
            lat,
        )

    def _stub_local_op(
        name: str,
        mesh: Mesh,
        body: Any,
        in_specs: tuple,
        out_specs: tuple,
        donate_argnums: tuple[int, ...] = (),
        export_key: str | None = None,
    ) -> Any:
        del mesh, body, donate_argnums, export_key
        assert len(in_specs) <= 64, f"{name} exceeded 64 Torch custom op args"
        registered_local[name] = (in_specs, out_specs, ())
        if "stack_states" in name:
            return lambda *args: (
                torch.stack(args[: len(args) // 2], dim=2),
                torch.stack(args[len(args) // 2 :], dim=2),
            )
        if "split_states" in name:
            return lambda rec, conv: (
                *[rec[:, :, i] for i in range(int(rec.shape[2]))],
                *[conv[:, :, i] for i in range(int(conv.shape[2]))],
            )
        if "read_states" in name:
            return lambda *_a: (
                torch.zeros((1, KDA_LAYERS, 2, 3, 128, 128), dtype=torch.float32),
                torch.zeros((1, KDA_LAYERS, 2, 3, 3, 3, 128), dtype=torch.bfloat16),
            )
        if "first_token" in name:
            return lambda logits, _p: (
                torch.zeros((int(logits.shape[0]),), dtype=torch.int32),
            )
        if "copy_blocks" in name:
            return lambda *args: args[: len(out_specs)]
        return lambda *args: (
            tuple(args[5 : 5 + len(out_specs)]) if len(out_specs) > 1 else args[:1]
        )

    monkeypatch.setattr(k3_prefill, "register_global_op", _stub_global_op)
    monkeypatch.setattr(k3_prefill, "register_op", _stub_local_op)

    seen_arenas: list[tuple[Any, Any]] = []
    real_rank_loader = k3_prefill.kernel_modules("rank_loader")[0]

    class _StubKernels:
        MOE_FAMILIES = pm.MOE_FAMILIES
        CpPrefix = pm.CpPrefix
        cp_prefix_meta = staticmethod(lambda *a, **kw: ("c", "o"))

        @staticmethod
        def make_embedding_sp(mesh: Any, *, seq_len: int):
            del mesh
            return lambda tok, _emb, _len: jnp.zeros(
                (32, seq_len // 32, 16), jnp.bfloat16
            )

        @staticmethod
        def make_prefill_stack_sp_indexed(mesh: Any, *, seq_len: int, **kw: Any):
            del mesh, seq_len, kw

            def _first(
                hidden: Any,
                _c: Any,
                _k: Any,
                _m: Any,
                _moe: Any,
                kda_st: Any,
                mla_st: Any,
                _len: Any,
                *,
                segment_desc: Any,
                arena: Any,
                route: Any = None,
            ):
                del segment_desc, route
                seen_arenas.append((mla_st[0].shape, arena.owned_pages))
                return hidden, hidden[:, None], kda_st, mla_st

            return _first

        @staticmethod
        def make_regular_group8_sp_indexed(mesh: Any, **kw: Any):
            del mesh, kw
            return lambda *a, **k: None

        @staticmethod
        def make_final_mla_sp_indexed(mesh: Any, **kw: Any):
            del mesh, kw
            return lambda *a, **k: None

        @staticmethod
        def make_prefill_logits_sp(mesh: Any, **kw: Any):
            del mesh, kw
            return lambda *a, **k: None

    monkeypatch.setattr(
        k3_prefill,
        "kernel_modules",
        lambda *_names: (_StubKernels(), real_rank_loader),
    )
    prog = k3_prefill.PrefillProgram(
        windows=(512, 1024, 2048),
        capacity=2048,
        slots=2,
        page_tokens=512,
        compute_logits=True,
    )
    monkeypatch.setattr(
        prog,
        "zero_state",
        lambda: (
            torch.zeros((1, KDA_LAYERS, 3, 128, 128), dtype=torch.float32),
            torch.zeros((1, KDA_LAYERS, 3, 3, 3, 128), dtype=torch.bfloat16),
        ),
    )
    for grp, names in prog.names.items():
        prog.weights[grp] = {
            n: torch.zeros((1, 1), dtype=torch.bfloat16) for n in names
        }

    raw_pools = [
        torch.zeros((4, *FP8_LAYOUT.block_shape), dtype=torch.uint8).view(
            torch.float8_e4m3fn
        )
        for _ in range(MLA_LAYERS)
    ]
    writer = PoolWriter(
        one_mesh,
        0,
        raw_pools,
        list(range(MLA_LAYERS)),
        [i % MLA_LAYERS for i in range(KDA_LAYERS)],
        None,
    )

    class _WarmupPrefiller(KimiPrefiller):
        def __init__(self) -> None:
            self.config = SimpleNamespace(
                max_num_seqs=2,
                block_size=512,
                max_num_batched_tokens=2048,
                max_model_len=2048,
            )
            self.page_tokens = 512
            self.cp_rank = 0
            self.cp_capacity = 1024
            self.slots = 2
            self.rank = 0
            self.compute_logits = True
            self.ring = None
            self.draft_latents = None
            self.program = prog
            self._writer = writer
            self._sampler_ops = {}

        def _precompile(self) -> None:
            return None

    prefiller = _WarmupPrefiller()
    calls = prefiller.compilation_calls()
    write_call_ops = [
        c.op
        for c in calls
        if c.op
        in [writer.write_op(1, 1, 1, 1, has_ckpt=ck) for ck in (False, True)]
        + [
            writer.write_op(2, wr, 2, lv, has_ckpt=ck)
            for lv, wr in ((1, 1), (2, 2))
            for ck in (False, True)
        ]
    ]
    assert len(write_call_ops) == 6

    prefiller.warmup()
    assert [sh for sh, _ in seen_arenas] == [
        (32, MLA_LAYERS, 1, *FP8_LAYOUT.block_shape),
        (32, MLA_LAYERS, 1, *FP8_LAYOUT.block_shape),
        (32, MLA_LAYERS, 2, *FP8_LAYOUT.block_shape),
        (32, MLA_LAYERS, 2, *FP8_LAYOUT.block_shape),
        (32, MLA_LAYERS, 2, *FP8_LAYOUT.block_shape),
        (32, MLA_LAYERS, 2, *FP8_LAYOUT.block_shape),
    ]
    assert all(owned is not None for _, owned in seen_arenas)
