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
"""Admission rules of the fused MLA DSpark draft kernel."""

from __future__ import annotations

import importlib
from typing import Any

import jax.numpy as jnp

from vllm_torchtpu.models.kimi_k3.megakernel.kernels import dspark, dspark_mla_kernel
from vllm_torchtpu.models.kimi_k3.megakernel.kernels.dspark_mla import MlaDraftConfig

pytest: Any = importlib.import_module("pytest")
PAGES = 16


@pytest.mark.parametrize(
    ("geometry", "error"),
    [
        ({"requests": 1}, None),
        ({"requests": 8}, None),
        ({"requests": 16, "rows": 4, "block": 4}, None),
        ({"requests": 1, "rows": 6}, "4 or 8 rows"),
        ({"requests": 1, "block": 1}, "block must be"),
        ({"requests": 9}, "up to 8 requests"),
        ({"requests": 3, "attention_dp": 2}, "attention DP2"),
    ],
)
def test_fused_draft_admission(geometry: dict[str, int], error: str | None) -> None:
    """Supported geometries build a kernel; others raise naming the limit."""
    if error is None:
        kernel = dspark_mla_kernel.fused_draft_local(
            MlaDraftConfig(), pages=PAGES, **geometry
        )
        assert callable(kernel)
    else:
        with pytest.raises(ValueError, match=error):
            dspark_mla_kernel.fused_draft_local(
                MlaDraftConfig(), pages=PAGES, **geometry
            )


def test_dspark_typed_contract_dataclasses() -> None:
    """Verify frozen contract dataclasses round-trip through nucleus_accept."""
    values = jnp.array([[2.0, 1.0], [3.0, 0.5]], dtype=jnp.float32)
    ids = jnp.array([[10, 11], [12, 13]], dtype=jnp.int32)
    lse = jnp.log(jnp.sum(jnp.exp(values), axis=1))
    drafts = jnp.array([10, 12], dtype=jnp.int32)
    uniforms = jnp.array([[0.1, 0.2], [0.1, 0.2]], dtype=jnp.float32)
    inputs = dspark.NucleusAcceptInputs(
        values=values,
        ids=ids,
        lse=lse,
        drafts=drafts,
        forced_count=-1,
        top_p=0.9,
        uniforms=uniforms,
    )
    c1, b1 = inputs.accept()
    c2, b2 = dspark.nucleus_accept(values, ids, lse, drafts, -1, 0.9, uniforms)
    assert int(c1) == int(c2)
    assert int(b1) == int(b2)
