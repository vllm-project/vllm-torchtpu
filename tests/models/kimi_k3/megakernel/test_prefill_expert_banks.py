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
"""Checkpoint partitioning must not change the expert selected by a model layer."""

import unittest

import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.models.kimi_k3.megakernel.kernels.prefill_megakernel import (
    expert_family_slot,
)
from vllm_torchtpu.models.kimi_k3.megakernel.prefill import _bank_expert_weights

FAMILIES = ("expert_gu", "expert_gus", "expert_down", "expert_ds")
_NUM_MOE_LAYERS = 92
_FIRST_BANK_LAYERS = 3
_FINAL_BANK_SLOT = 91
_FINAL_BANK_INDEX = 12
_NUM_BANKS = 13
_BOUNDARY_CASES: tuple[list[int], ...] = (
    [],
    [1, 2, 17, 90],
    [3, 11, 91],
    list(range(1, _NUM_MOE_LAYERS)),
)


def test_expert_banks_preserve_every_model_layer() -> None:
    """Verify streamed expert banks preserve every layer's weights across splits."""
    for boundaries in _BOUNDARY_CASES:
        weights = {
            family: (
                np.arange(_NUM_MOE_LAYERS * 6, dtype=np.int32).reshape(
                    _NUM_MOE_LAYERS, 2, 3
                )
                + index * 1000
            )
            for index, family in enumerate(FAMILIES)
        }
        parts = {
            family: np.split(value, boundaries) for family, value in weights.items()
        }
        dense = np.arange(7)
        num_parts = len(boundaries) + 1
        stream = [
            ("moe", "shared_gu", dense),
            *[
                ("moe", f"{family}@{part}/{num_parts}", parts[family][part])
                for part in range(num_parts)
                for family in FAMILIES
            ],
        ]
        output = list(_bank_expert_weights(iter(stream)))
        assert output[0][0:2] == ("moe", "shared_gu")
        assert output[0][2] is dense
        banks = {name: value for _group, name, value in output[1:]}
        assert len(banks) == _NUM_BANKS * len(FAMILIES)
        for family in FAMILIES:
            selected = []
            for slot in range(_NUM_MOE_LAYERS):
                bank = (
                    0
                    if slot < _FIRST_BANK_LAYERS
                    else _FINAL_BANK_INDEX
                    if slot == _FINAL_BANK_SLOT
                    else 1 + (slot - _FIRST_BANK_LAYERS) // 8
                )
                value = banks[f"{family}#bank{bank}"]
                local_slot = int(expert_family_slot(value.shape[0], jnp.int32(slot)))
                selected.append(value[local_slot])
            np.testing.assert_array_equal(np.stack(selected), weights[family])
            total_rows = sum(
                banks[f"{family}#bank{bank_idx}"].shape[0]
                for bank_idx in range(_NUM_BANKS)
            )
            assert total_rows == _NUM_MOE_LAYERS
        np.testing.assert_array_equal(
            expert_family_slot(_NUM_MOE_LAYERS, jnp.arange(_NUM_MOE_LAYERS)),
            np.arange(_NUM_MOE_LAYERS),
        )


def test_expert_banks_reject_incomplete_or_invalid_checkpoints() -> None:
    """Verify invalid or incomplete expert checkpoint streams raise ValueError."""
    case = unittest.TestCase()
    for kind in ("missing_family", "short", "long", "out_of_order"):
        stream = [
            ("moe", family, np.zeros((_NUM_MOE_LAYERS, 1), np.int32))
            for family in FAMILIES
        ]
        if kind == "missing_family":
            stream.pop()
        elif kind == "out_of_order":
            stream[0] = ("moe", "expert_gu@1/2", stream[0][2])
        else:
            bad_layers = _NUM_MOE_LAYERS - 1 if kind == "short" else _NUM_MOE_LAYERS + 1
            stream[0] = ("moe", "expert_gu", np.zeros((bad_layers, 1), np.int32))
        with case.assertRaisesRegex(
            ValueError,
            r"Missing|Incomplete|Too many|Out-of-order",
        ):
            list(_bank_expert_weights(iter(stream)))
