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
"""Admission and dtype boundary regression tests for retained decode preparation."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.qwen4_exp import retained_preparation as prep


@pytest.mark.parametrize("rope_rows", [128, 65536, 67584])
def test_preparation_depends_on_compute_layout_not_serving_capacity(
    monkeypatch, rope_rows
):
    arr = jax.ShapeDtypeStruct
    args = [
        arr((16, 12288), jnp.bfloat16),
        arr((16, 512), jnp.bfloat16),
        arr((16, 640), jnp.bfloat16),
        arr((256,), jnp.bfloat16),
        arr((256,), jnp.bfloat16),
        arr((128,), jnp.bfloat16),
        arr((256,), jnp.float32),
        arr((rope_rows, 128), jnp.bfloat16),
    ]
    options = dict(rotary_dim=64)
    monkeypatch.setattr(
        prep,
        "_hardware",
        lambda: SimpleNamespace(chip_version=pltpu.ChipVersion.TPU_7X),
    )
    assert prep.use_retained_preparation(*args, **options)
    for index, value in [
        (0, arr((32, 12288), jnp.bfloat16)),
        (6, None),
        (6, arr((256,), jnp.bfloat16)),
        (7, arr((0, 128), jnp.bfloat16)),
        (7, arr((rope_rows, 64), jnp.bfloat16)),
    ]:
        changed = args.copy()
        changed[index] = value
        assert not prep.use_retained_preparation(*changed, **options)
    monkeypatch.setattr(prep, "_hardware", lambda: None)
    assert not prep.use_retained_preparation(*args, **options)


@pytest.mark.tpu_test
@pytest.mark.parametrize("rope_rows", [128, 65536])
@pytest.mark.parametrize("axial", [False, True])
def test_prepared_producer_matches_original_values_for_other_rope_capacities(
    rope_rows, axial
):
    import torch

    # Each static operator pair is an independent compilation experiment.
    torch._dynamo.reset()
    from torch_tpu._internal import pallas

    from vllm_torchtpu.kernels.qwen4_exp.qsa import produce

    rope_axes = tuple([0, 1, 2] * 10 + [0, 1]) if axial else None

    def build(prepared):
        def core(
            q: jax.Array,
            k: jax.Array,
            v: jax.Array,
            iqk: jax.Array,
            nq: jax.Array,
            nk: jax.Array,
            niq: jax.Array,
            positions: jax.Array,
            trig: jax.Array,
        ) -> tuple[jax.Array, ...]:
            if prepared:
                return prep.produce_prepared(
                    q,
                    k,
                    v,
                    iqk,
                    nq,
                    1 + nk.astype(jnp.float32),
                    niq,
                    positions,
                    trig,
                    1e-6,
                    64,
                    rope_axes,
                )
            return produce(
                q, k, v, iqk, nq, nk, niq, positions, trig, 1e-6, 64, rope_axes
            )

        op = pallas.jax_op(
            f"qsa_prepare_test::producer_{rope_rows}_{prepared}_{axial}", core
        )

        @op.register_fake
        def fake(q, k, v, iqk, nq, nk, niq, positions, trig):
            shapes = [
                (16, 24, 256),
                (16, 2, 256),
                (16, 2, 256),
                (16, 24, 256),
                (16, 4, 128),
                (16, 128),
            ]
            return tuple(
                torch.empty(
                    s,
                    device=q.device,
                    dtype=torch.float32 if prepared and i == 0 else q.dtype,
                )
                for i, s in enumerate(shapes)
            )

        return torch.compile(op, backend="tpu", fullgraph=True, dynamic=False)

    generator = torch.Generator().manual_seed(9451)
    inputs = [
        torch.randn(s, generator=generator).bfloat16().to("tpu")
        for s in [(16, 12288), (16, 512), (16, 512), (16, 640), (256,), (256,), (128,)]
    ]
    positions = torch.linspace(0, rope_rows - 1, 16).int()
    if axial:
        positions = torch.stack([positions, positions.roll(1), positions.roll(2)])
    positions = positions.to("tpu")
    trig = torch.randn((rope_rows, 128), generator=generator).bfloat16().to("tpu")
    inputs += [positions, trig]
    with torch.inference_mode():
        expected, actual = [build(p)(*inputs) for p in (False, True)]
        for ref, got in zip(expected, actual):
            torch.testing.assert_close(
                got.cpu().float(), ref.cpu().float(), rtol=0, atol=0
            )
