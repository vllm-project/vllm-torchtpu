# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
# Source provenance: inferact/k3-megakernel@30c9ba966b
#   tests/megakernel/kimi_k3/test_prefill_expert_banks.py
"""Checkpoint partitioning must not change the expert selected by a model layer."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.models.kimi_k3.megakernel.common.constants import (
    EXPERT_FAMILIES,
    NUM_LAYERS,
)
from vllm_torchtpu.models.kimi_k3.megakernel.kernels.prefill_megakernel import (
    expert_family_slot,
)
from vllm_torchtpu.models.kimi_k3.megakernel.prefill import _bank_expert_weights

# Layer 0 uses a dense FFN; layers 1..92 (92 layers) carry routed MoE banks.
_NUM_MOE_LAYERS = NUM_LAYERS - 1
_FIRST_BANK_LAYERS = 3
_FINAL_BANK_SLOT = _NUM_MOE_LAYERS - 1
_FINAL_BANK_INDEX = 12
_NUM_BANKS = 13
_REGULAR_BANK_STRIDE = 8
_ELEMENTS_PER_LAYER = 6
_FAMILY_TAG_SCALE = 1000
_DENSE_FIXTURE_LEN = 7


def _expected_bank_index(slot: int) -> int:
    """Return the prefill execution bank index owning MoE layer ``slot``."""
    if slot < _FIRST_BANK_LAYERS:
        return 0
    if slot == _FINAL_BANK_SLOT:
        return _FINAL_BANK_INDEX
    return 1 + (slot - _FIRST_BANK_LAYERS) // _REGULAR_BANK_STRIDE


@pytest.mark.parametrize(
    "boundaries",
    [
        [],
        [1, 2, 17, 90],
        [3, 11, 91],
        list(range(1, _NUM_MOE_LAYERS)),
    ],
)
def test_expert_banks_preserve_every_model_layer(boundaries: list[int]) -> None:
    """Verify streamed expert banks preserve every layer's weights across splits."""
    weights = {
        family: (
            np.arange(_NUM_MOE_LAYERS * _ELEMENTS_PER_LAYER, dtype=np.int32).reshape(
                _NUM_MOE_LAYERS, 2, 3
            )
            + index * _FAMILY_TAG_SCALE
        )
        for index, family in enumerate(EXPERT_FAMILIES)
    }
    parts = {family: np.split(value, boundaries) for family, value in weights.items()}
    dense = np.arange(_DENSE_FIXTURE_LEN)
    num_parts = len(boundaries) + 1
    stream = [
        ("moe", "shared_gu", dense),
        *[
            ("moe", f"{family}@{part}/{num_parts}", parts[family][part])
            for part in range(num_parts)
            for family in EXPERT_FAMILIES
        ],
    ]
    output = list(_bank_expert_weights(iter(stream)))
    assert output[0][0:2] == ("moe", "shared_gu")
    assert output[0][2] is dense
    banks = {name: value for _group, name, value in output[1:]}
    assert len(banks) == _NUM_BANKS * len(EXPERT_FAMILIES)
    for family in EXPERT_FAMILIES:
        selected = []
        for slot in range(_NUM_MOE_LAYERS):
            bank = _expected_bank_index(slot)
            value = banks[f"{family}#bank{bank}"]
            local_slot = int(expert_family_slot(value.shape[0], jnp.int32(slot)))
            selected.append(value[local_slot])
        np.testing.assert_array_equal(np.stack(selected), weights[family])
        total_rows = sum(
            banks[f"{family}#bank{bank_idx}"].shape[0] for bank_idx in range(_NUM_BANKS)
        )
        assert total_rows == _NUM_MOE_LAYERS
    np.testing.assert_array_equal(
        expert_family_slot(_NUM_MOE_LAYERS, jnp.arange(_NUM_MOE_LAYERS)),
        np.arange(_NUM_MOE_LAYERS),
    )


@pytest.mark.parametrize("kind", ["missing_family", "short", "long", "out_of_order"])
def test_expert_banks_reject_incomplete_or_invalid_checkpoints(kind: str) -> None:
    """Verify invalid or incomplete expert checkpoint streams raise ValueError."""
    stream = [
        ("moe", family, np.zeros((_NUM_MOE_LAYERS, 1), np.int32))
        for family in EXPERT_FAMILIES
    ]
    if kind == "missing_family":
        stream.pop()
    elif kind == "out_of_order":
        stream[0] = ("moe", "expert_gu@1/2", stream[0][2])
    else:
        bad_layers = _NUM_MOE_LAYERS - 1 if kind == "short" else _NUM_MOE_LAYERS + 1
        stream[0] = ("moe", "expert_gu", np.zeros((bad_layers, 1), np.int32))
    with pytest.raises(ValueError, match=r"Missing|Incomplete|Too many|Out-of-order"):
        list(_bank_expert_weights(iter(stream)))
