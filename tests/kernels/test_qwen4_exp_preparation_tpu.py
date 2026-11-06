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
"""Native regressions for compressed preparation and virtual page metadata."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch


@pytest.mark.tpu_test
@torch.inference_mode()
def test_compressed_preparation_keeps_partial_and_aliased_pairs(
    dma=False, rope_rows=67584
):
    from torch_tpu._internal import pallas

    from vllm_torchtpu.kernels.qwen4_exp.pooled_qsa import write_compressed
    from vllm_torchtpu.kernels.qwen4_exp.qsa import _token_metadata, index_rope

    def build(fused, fp8):
        def core(
            pool: jax.Array,
            values: jax.Array,
            positions: jax.Array,
            trig: jax.Array,
            starts: jax.Array,
            context: jax.Array,
            table: jax.Array,
        ) -> jax.Array:
            slots = jnp.where(starts[1:] > starts[:-1], jnp.arange(32), -1)
            req, logical, valid = _token_metadata(16, starts, context, slots)
            complete = valid & (logical % 4 == 3)
            if fused:
                return write_compressed(
                    pool,
                    values,
                    complete,
                    req,
                    logical,
                    starts,
                    context,
                    table,
                    4,
                    logical_block_size=1664,
                    batch_pairs=True,
                    rotary_dim=64,
                    trig_rows=None
                    if dma
                    else jax.lax.optimization_barrier(trig[positions]),
                    trig_cache=trig if dma else None,
                    positions=positions if dma else None,
                )
            keys = index_rope(values, positions, trig, 64)
            return write_compressed(
                pool,
                keys,
                complete,
                req,
                logical,
                starts,
                context,
                table,
                4,
                logical_block_size=1664,
                batch_pairs=False,
            )

        op = pallas.jax_op(
            f"qsa_prepare_test::compressed_{fused}_{fp8}_{dma}_{rope_rows}",
            core,
            donate_argnums=(0,),
        )

        @op.register_fake
        def fake(pool, values, positions, trig, starts, context, table):
            return torch.empty_like(pool)

        return torch.compile(op, backend="tpu", fullgraph=True, dynamic=False)

    generator = torch.Generator().manual_seed(18627)
    angles = torch.outer(
        torch.arange(rope_rows).float(), 1e7 ** (-torch.arange(32).float() / 32)
    )
    trig = torch.nn.functional.pad(
        torch.cat((angles.cos(), angles.sin()), 1).bfloat16(), (0, 64)
    ).to("tpu")
    # Repeat the final DMA block, including its last row, to exercise deduplication.
    positions = (rope_rows - 8 + torch.arange(16, dtype=torch.int32) % 8).to("tpu")
    values = torch.randn((16, 128), generator=generator).bfloat16().to("tpu")
    for fp8 in (False, True):
        dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
        packing = 4 if fp8 else 2
        raw = torch.randint(
            0, 256, (8, 208, 1, 4, 128), generator=generator, dtype=torch.uint8
        )
        initial = raw.view(dtype).reshape(8, 208, 1, packing, 128)
        funcs = [build(f, fp8) for f in (False, True)]
        for case in (
            "partial_even",
            "partial_odd",
            "holes",
            "idle",
            "chunk16",
            "alias",
        ):
            counts = torch.tensor([1] * 8 + [0] * 24, dtype=torch.int32)
            if case == "holes":
                counts = torch.tensor([0, 1] * 8 + [0] * 16, dtype=torch.int32)
            if case == "idle":
                counts.zero_()
            if case == "chunk16":
                counts.zero_()
                counts[0] = 16
            context = torch.full(
                (32,), 4099 if case != "partial_odd" else 4103, dtype=torch.int32
            )
            if case == "chunk16":
                context.zero_()
            starts = torch.cat(
                (torch.zeros(1, dtype=torch.int32), counts.cumsum(0).int())
            )
            table = (
                (2 + torch.arange(32, dtype=torch.int32) % 4)[:, None]
                .expand(32, 41)
                .clone()
            )
            if case == "alias":
                table[:] = 2
            args = (
                values,
                positions,
                trig,
                starts.to("tpu"),
                context.to("tpu"),
                table.to("tpu"),
            )
            outputs = [
                fn(initial.to("tpu"), *args).cpu().view(torch.uint8).numpy()
                for fn in funcs
            ]
            np.testing.assert_array_equal(
                outputs[0], outputs[1], err_msg=f"{fp8}/{case}"
            )


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    "batch,budget,columns,ratio,page",
    [
        (1, 512, 41, 4, 1664),
        (1, 1, 1, 4, 128),
        (16, 512, 41, 4, 1664),
        (8, 256, 2, 4, 128),
        (32, 256, 42, 8, 1664),
        (16, 512, 129, 4, 128),
    ],
)
@torch.inference_mode()
def test_virtual_metadata_matches_integer_oracle(batch, budget, columns, ratio, page):
    from torch_tpu._internal import pallas

    from vllm_torchtpu.kernels.sparse_gqa_decode.metadata import virtual_page_metadata

    def core(
        blocks: jax.Array, lengths: jax.Array, table: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        return virtual_page_metadata(
            blocks, lengths, table, ratio=ratio, page_size=page
        )

    op = pallas.jax_op(
        f"qsa_prepare_test::pages_{batch}_{budget}_{columns}_{ratio}_{page}", core
    )

    @op.register_fake
    def fake(blocks, lengths, table):
        return table.new_empty((batch, budget + 1)), lengths.new_empty((batch,))

    run = torch.compile(op, backend="tpu", fullgraph=True, dynamic=False)
    generator = torch.Generator().manual_seed(9281)
    for case in ("boundaries", "active8", "holes", "idle", "full", "alias"):
        lengths = torch.full((batch,), min(65537, page * columns), dtype=torch.int32)
        if case == "boundaries":
            boundary_lengths = [
                0,
                1,
                2,
                3,
                4,
                7,
                8,
                1663,
                1664,
                1665,
                2047,
                2048,
                2049,
                65535,
                65536,
                65537,
            ]
            lengths = torch.tensor(
                [
                    min(boundary_lengths[row % len(boundary_lengths)], page * columns)
                    for row in range(batch)
                ],
                dtype=torch.int32,
            )
        if case == "active8":
            lengths[batch // 2 :] = 0
        if case == "holes":
            lengths[::2] = 0
        if case == "idle":
            lengths[:] = 0
        blocks = torch.full((batch, budget), -711, dtype=torch.int32)
        table = torch.randint(
            0, 17313, (batch, columns), generator=generator, dtype=torch.int32
        )
        if case == "alias":
            table[:] = 17
        expected = torch.zeros((batch, budget + 1), dtype=torch.int32)
        for row in range(batch):
            total = int(lengths[row]) // ratio
            count = min(total, budget)
            if count:
                blocks[row, :count] = torch.randperm(total, generator=generator)[
                    :count
                ].int()
            for col in range(count + int(lengths[row] % ratio > 0)):
                group = int(blocks[row, col]) if col < count else total
                per_page = page // ratio
                expected[row, col] = (
                    table[row, group // per_page] * per_page + group % per_page
                )
        pages, kv = run(blocks.to("tpu"), lengths.to("tpu"), table.to("tpu"))
        assert torch.equal(pages.cpu(), expected), case
        assert torch.equal(
            kv.cpu(),
            torch.minimum(lengths // ratio, torch.tensor(budget)) * ratio
            + lengths % ratio,
        ), case


@pytest.mark.parametrize(
    "batch,budget,columns,ratio,page,expected",
    [
        (1, 512, 41, 4, 1664, False),
        (1, 1, 1, 4, 128, False),
        (16, 512, 41, 4, 1664, True),
        (8, 512, 41, 4, 1664, True),
        (16, 256, 41, 4, 1664, True),
        (16, 512, 42, 4, 1664, True),
        (16, 512, 41, 8, 1664, True),
        (16, 512, 41, 4, 128, True),
        (16, 512, 129, 4, 1664, False),
        (16, 512, 41, 3, 1664, False),
        (16, 1 << 20, 41, 4, 1664, False),
    ],
)
def test_page_metadata_policy_follows_layout_and_resource_capacity(
    batch, budget, columns, ratio, page, expected, monkeypatch
):
    from jax.experimental.pallas import tpu as pltpu

    from vllm_torchtpu.kernels.sparse_gqa_decode.tuned_block_sizes import (
        use_fused_page_metadata,
    )

    monkeypatch.setattr(
        pltpu,
        "get_tpu_info",
        lambda: pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1),
    )
    mesh = jax.sharding.AbstractMesh(
        (1,),
        ("device",),
        abstract_device=jax.sharding.AbstractDevice(
            device_kind="TPU7x", num_cores=1, platform="tpu"
        ),
    )
    with jax.sharding.use_abstract_mesh(mesh):
        assert (
            use_fused_page_metadata(
                jax.ShapeDtypeStruct((batch, budget), jnp.int32),
                jax.ShapeDtypeStruct((batch,), jnp.int32),
                jax.ShapeDtypeStruct((batch, columns), jnp.int32),
                ratio,
                page,
            )
            == expected
        )


def test_page_metadata_cpu_keeps_generic_gather_without_hardware(monkeypatch):
    from jax.experimental.pallas import tpu as pltpu

    from vllm_torchtpu.kernels.sparse_gqa_decode.metadata import virtual_page_metadata

    def forbidden():
        raise AssertionError("CPU page mapping must not query TPU hardware")

    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    monkeypatch.setattr(pltpu, "get_tpu_info", forbidden)
    pages, lengths = virtual_page_metadata(
        jnp.array([[0, 1], [-1, -1]], jnp.int32),
        jnp.array([9, 0], jnp.int32),
        jnp.array([[3], [0]], jnp.int32),
        ratio=4,
        page_size=128,
    )
    np.testing.assert_array_equal(pages, [[96, 97, 98], [0, 0, 0]])
    np.testing.assert_array_equal(lengths, [9, 0])


@pytest.mark.parametrize(
    "rows,count,dtype,kind,expected",
    [
        (67584, 16, jnp.bfloat16, "TPU_7X", True),
        (65536, 16, jnp.bfloat16, "TPU_7X", True),
        (128, 16, jnp.bfloat16, "TPU_7X", True),
        (65535, 16, jnp.bfloat16, "TPU_7X", False),
        (0, 16, jnp.bfloat16, "TPU_7X", False),
        (67584, 8, jnp.bfloat16, "TPU_7X", False),
        (67584, 32, jnp.bfloat16, "TPU_7X", False),
        (67584, 16, jnp.float32, "TPU_7X", False),
        (67584, 16, jnp.bfloat16, "TPU_V6E", False),
    ],
)
def test_completed_rope_dma_shape_policy(
    rows, count, dtype, kind, expected, monkeypatch
):
    from types import SimpleNamespace

    from jax.experimental.pallas import tpu as pltpu

    from vllm_torchtpu.kernels.qwen4_exp import tuned_block_sizes as policy

    monkeypatch.setattr(
        policy,
        "_hardware",
        lambda: SimpleNamespace(chip_version=getattr(pltpu.ChipVersion, kind)),
    )
    assert (
        policy.use_completed_rope_dma(
            jax.ShapeDtypeStruct((rows, 128), dtype),
            jax.ShapeDtypeStruct((count,), jnp.int32),
        )
        == expected
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize("rope_rows", [128, 65536, 67584])
def test_completed_rope_dma_keeps_partial_and_aliased_pairs(rope_rows):
    test_compressed_preparation_keeps_partial_and_aliased_pairs(
        dma=True, rope_rows=rope_rows
    )
