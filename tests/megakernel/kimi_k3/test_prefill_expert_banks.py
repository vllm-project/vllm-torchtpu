# SPDX-License-Identifier: Apache-2.0
"""Checkpoint partitioning must not change the expert selected by a model layer."""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.megakernel.kimi_k3.kernels.prefill_megakernel import (
    expert_family_slot,
)
from vllm_torchtpu.megakernel.kimi_k3.prefill import _bank_expert_weights

pytestmark = pytest.mark.cpu_test
FAMILIES = ("expert_gu", "expert_gus", "expert_down", "expert_ds")


@pytest.mark.parametrize(
    "boundaries", [[], [1, 2, 17, 90], [3, 11, 91], list(range(1, 92))]
)
def test_expert_banks_preserve_every_model_layer(boundaries):
    # Interleaved checkpoint streams with boundaries both inside and between
    # execution groups. Values identify both their family and global layer.
    weights = {
        family: np.arange(92 * 6, dtype=np.int32).reshape(92, 2, 3) + i * 1000
        for i, family in enumerate(FAMILIES)
    }
    parts = {family: np.split(value, boundaries) for family, value in weights.items()}
    dense = np.arange(7)
    stream = [("moe", "shared_gu", dense)]
    for part in range(len(boundaries) + 1):
        for family in FAMILIES:
            stream.append(
                ("moe", f"{family}@{part}/{len(boundaries) + 1}", parts[family][part])
            )
    output = list(_bank_expert_weights(iter(stream)))
    assert output[0][0:2] == ("moe", "shared_gu")
    assert output[0][2] is dense
    banks = {name: value for group, name, value in output[1:]}
    assert len(banks) == 13 * len(FAMILIES)
    for family in FAMILIES:
        selected = []
        for slot in range(92):
            bank = 0 if slot < 3 else 12 if slot == 91 else 1 + (slot - 3) // 8
            value = banks[f"{family}#bank{bank}"]
            local_slot = int(expert_family_slot(value.shape[0], jnp.int32(slot)))
            selected.append(value[local_slot])
        np.testing.assert_array_equal(np.stack(selected), weights[family])
        assert sum(banks[f"{family}#bank{i}"].shape[0] for i in range(13)) == 92
    np.testing.assert_array_equal(expert_family_slot(92, jnp.arange(92)), np.arange(92))


@pytest.mark.parametrize("kind", ["missing_family", "short", "long", "out_of_order"])
def test_expert_banks_reject_incomplete_or_invalid_checkpoints(kind):
    stream = [("moe", family, np.zeros((92, 1), np.int32)) for family in FAMILIES]
    if kind == "missing_family":
        stream.pop()
    elif kind == "out_of_order":
        stream[0] = ("moe", "expert_gu@1/2", stream[0][2])
    else:
        stream[0] = (
            "moe",
            "expert_gu",
            np.zeros((91 if kind == "short" else 93, 1), np.int32),
        )
    with pytest.raises(ValueError):
        list(_bank_expert_weights(iter(stream)))
