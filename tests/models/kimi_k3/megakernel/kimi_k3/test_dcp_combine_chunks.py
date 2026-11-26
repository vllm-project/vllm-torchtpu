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
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.extend import core as jex_core

from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3 import config as k3_config
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.kernels import mla_dcp


def test_should_send_plain_b8_packet_in_8_row_tiles_after_their_rows_attend() -> None:
    sends = mla_dcp.combine_groups(8, mla_dcp.HEADS_PER_RANK, 1, chunks=True)

    assert sends == [(0, 3, 0, 8), (3, 6, 8, 16), (6, 8, 16, 24)]


def test_should_cover_every_row_and_packet_row_exactly_once_when_chunked() -> None:
    sends = mla_dcp.combine_groups(8, mla_dcp.HEADS_PER_RANK, 1, chunks=True)

    rows = [r for first, end, _, _ in sends for r in range(first, end)]
    packet = [p for _, _, first, end in sends for p in range(first, end)]
    assert rows == list(range(8))
    assert packet == list(range(3 * 8))
    assert all(
        (end - first) % mla_dcp.PACKET_TILE_ROWS == 0 for _, _, first, end in sends
    )


def test_should_attend_every_packet_row_of_a_tile_before_sending_it() -> None:
    slots = mla_dcp.packet_layout(8, mla_dcp.HEADS_PER_RANK)[0]
    sends = mla_dcp.combine_groups(8, mla_dcp.HEADS_PER_RANK, 1, chunks=True)

    for _, end_row, first_packet, end_packet in sends:
        assert all(p // slots < end_row for p in range(first_packet, end_packet))


def test_should_send_whole_packet_after_all_rows_when_not_chunked() -> None:
    assert mla_dcp.combine_groups(8, mla_dcp.HEADS_PER_RANK, 1, chunks=False) == [
        (0, 8, 0, 24)
    ]


def test_should_send_plain_b4_packet_in_8_row_prefix_plus_tail_tile() -> None:
    sends = mla_dcp.combine_groups(4, mla_dcp.HEADS_PER_RANK, 1, chunks=True)

    assert sends == [(0, 3, 0, 8), (3, 4, 8, 12)]
    assert mla_dcp.combine_groups(4, mla_dcp.HEADS_PER_RANK, 1, chunks=False) == [
        (0, 4, 0, 12)
    ]


@pytest.mark.parametrize(
    ("batch", "heads", "sequence_rows", "expected"),
    [
        (1, mla_dcp.HEADS_PER_RANK, 1, [(0, 1, 0, 3)]),
        (
            2,
            mla_dcp.HEADS_PER_RANK,
            1,
            [(0, 1, 0, 4), (1, 2, 4, 8)],
        ),
        (4, mla_dcp.HEADS_PER_RANK, 2, [(0, 4, 0, 12)]),
        (4, 6, 1, [(0, 4, 0, 24)]),
        (8, mla_dcp.HEADS_PER_RANK, 2, [(0, 8, 0, 24)]),
        (8, 6, 1, [(0, 8, 0, 48)]),
    ],
)
def test_should_keep_packet_layout_groups_outside_plain_b4_and_b8_even_when_chunked(
    batch: int,
    heads: int,
    sequence_rows: int,
    expected: list[tuple[int, int, int, int]],
) -> None:
    assert mla_dcp.combine_groups(batch, heads, sequence_rows, chunks=True) == expected


def test_should_size_combine_semaphores_for_every_chunked_tile() -> None:
    sends = mla_dcp.combine_groups(8, mla_dcp.HEADS_PER_RANK, 1, chunks=True)
    persistent = mla_dcp.persistent_scratch(8, group=32, heads=mla_dcp.HEADS_PER_RANK)
    combine_semaphores = persistent[
        mla_dcp.PERSISTENT_NAMES.index("dcp combine semaphores")
    ]

    assert combine_semaphores.shape[-1] >= len(sends)


def test_should_reject_chunked_tiles_beyond_semaphore_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mla_dcp, "MAX_GROUPS", 2)

    with pytest.raises(ValueError, match="semaphore groups"):
        mla_dcp.combine_groups(8, mla_dcp.HEADS_PER_RANK, 1, chunks=True)


def test_should_default_combine_chunks_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        k3_config,
        "DEFAULT_DCP_COMBINE_CHUNKS",
        k3_config.PRODUCTION_DEFAULTS["DEFAULT_DCP_COMBINE_CHUNKS"],
    )

    assert k3_config.KernelConfig().dcp_combine_chunks


def test_should_disable_combine_chunks_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(k3_config, "DEFAULT_DCP_COMBINE_CHUNKS", False)

    assert not k3_config.KernelConfig().dcp_combine_chunks


def _trace_plain_decode_attend(
    batch: int, chunks: bool, monkeypatch: pytest.MonkeyPatch
) -> jex_core.ClosedJaxpr:
    group, heads, fp8 = 32, mla_dcp.HEADS_PER_RANK, jnp.float8_e4m3fn
    monkeypatch.setattr(
        k3_config, "_ACTIVE", k3_config.KernelConfig(dcp_combine_chunks=chunks)
    )
    plan = mla_dcp.page_plan(batch, heads, 1, fp8)
    persistent = mla_dcp.persistent_scratch(
        batch, group=group, heads=heads, cache_dtype=fp8
    )
    scoped = mla_dcp.scoped_scratch(
        batch, group=group, heads=heads, cache_dtype=fp8, page_tokens=512, **plan
    )
    pages = 4

    def kernel(
        pos_ref, table_ref, layer_ref, pool_ref, q_ref, nl_ref, out_ref, *scratch
    ):
        cache = mla_dcp.DcpCache(
            (pool_ref,), table_ref, pages, layer=layer_ref[0], group=group
        )
        workspace = mla_dcp.DcpWorkspace(*scratch)
        attention = mla_dcp.DcpAttention(cache, workspace, pos_ref, batch)
        attention.prime()
        out_ref[...] = attention.attend(q_ref[...], nl_ref[...], 0)

    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((batch, heads, mla_dcp.LATENT), jnp.float32),
        in_specs=[smem, smem, smem, pl.BlockSpec(memory_space=pltpu.HBM), vmem, vmem],
        out_specs=vmem,
        scratch_shapes=(*persistent, *scoped),
    )
    operands = (
        jax.ShapeDtypeStruct((batch,), jnp.int32),
        jax.ShapeDtypeStruct((batch * pages,), jnp.int32),
        jax.ShapeDtypeStruct((1,), jnp.int32),
        jax.ShapeDtypeStruct((batch * pages, 128, 4, mla_dcp.LANES), fp8),
        jax.ShapeDtypeStruct((batch, heads, mla_dcp.LANES), jnp.float32),
        jax.ShapeDtypeStruct((batch, mla_dcp.LANES), jnp.float32),
    )
    return jax.make_jaxpr(call, axis_env=[("tp", group)])(*operands)


def _count_primitive(jaxpr, name: str) -> int:
    count = 0
    for eqn in jaxpr.eqns:
        count += eqn.primitive.name == name
        for value in eqn.params.values():
            for sub in value if isinstance(value, (list, tuple)) else (value,):
                inner = getattr(sub, "jaxpr", sub)
                if hasattr(inner, "eqns"):
                    count += _count_primitive(inner, name)
    return count


@pytest.mark.parametrize("chunks", [False, True])
@pytest.mark.parametrize("batch", [4, 8, 16, 64])
def test_should_trace_plain_fp8_decode_attend_when_combine_chunks_toggle(
    batch: int, chunks: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    _trace_plain_decode_attend(batch, chunks, monkeypatch)


@pytest.mark.parametrize("batch", [4, 8])
def test_should_trace_attention_once_when_combine_chunks_split_the_packet(
    batch: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    single_send = _trace_plain_decode_attend(batch, False, monkeypatch).jaxpr
    tiled_sends = _trace_plain_decode_attend(batch, True, monkeypatch).jaxpr

    assert _count_primitive(single_send, "dot_general") > 0
    assert _count_primitive(tiled_sends, "dot_general") == _count_primitive(
        single_send, "dot_general"
    )
