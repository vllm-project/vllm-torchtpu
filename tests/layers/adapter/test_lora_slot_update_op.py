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
"""The LoRA slot write: it replaces one slot of every stacked tensor, leaves
the other slots alone, and updates the stacked tensors in place."""

import jax.numpy as jnp
import numpy as np
import pytest
import torch
import torch._dynamo.utils

from tests.distributed.tpu_test_utils import run_in_isolated_process
from vllm_torchtpu.layers.adapter.custom_ops import lora_slot_update_op

NUM_SLOTS = 4
# Shaped like the stacked LoRA tensors of a linear layer and an embedding:
# [slots, 1, rank, in], [slots, 1, out, rank] and [slots, vocab, rank].
SHAPES = [(NUM_SLOTS, 1, 8, 16), (NUM_SLOTS, 1, 32, 8), (NUM_SLOTS, 40, 8)]
TPU = "tpu"


@pytest.fixture(autouse=True)
def _seed():
    # The CPU generator only: torch.manual_seed also seeds the TPU, which opens
    # the chip in this process and locks the spawned TPU cases out of it.
    torch.default_generator.manual_seed(0)


def _stacked_weights(shapes=SHAPES):
    return [torch.randn(shape) for shape in shapes]


def _new_slots(stacked_weights):
    """One new slot per stacked tensor, and the same values packed in
    stacked_weights order as the slot write takes them."""
    slots = [torch.randn(weights.shape[1:]) for weights in stacked_weights]
    return slots, torch.cat([slot.flatten() for slot in slots])


def _expected(stacked_weights, slots, slot):
    expected = [weights.clone() for weights in stacked_weights]
    for weights, new_slot in zip(expected, slots):
        weights[slot] = new_slot
    return expected


def _compute_on_host(stacked_weights, slot_weights, slot):
    updated = lora_slot_update_op._compute_slot_update(
        [jnp.asarray(weights.numpy()) for weights in stacked_weights],
        jnp.asarray(slot_weights.numpy()),
        jnp.asarray([slot], dtype=jnp.int32),
    )
    return [torch.from_numpy(np.array(weights)) for weights in updated]


@pytest.mark.parametrize("slot", range(NUM_SLOTS))
def test_only_the_chosen_slot_is_replaced(slot):
    stacked_weights = _stacked_weights()
    slots, slot_weights = _new_slots(stacked_weights)

    updated = _compute_on_host(stacked_weights, slot_weights, slot)

    for weights, expected in zip(updated, _expected(stacked_weights, slots, slot)):
        assert torch.equal(weights, expected)


@pytest.mark.parametrize("extra", [1, -1])
def test_slot_weights_of_the_wrong_size_are_rejected(extra):
    stacked_weights = _stacked_weights()
    size = sum(weights[0].numel() for weights in stacked_weights) + extra

    with pytest.raises(AssertionError, match="slot_weights holds"):
        _compute_on_host(stacked_weights, torch.zeros(size), 0)


# The tests below run on the chip, each in a spawned child, and
# spawns_tpu_workers runs them before tests that hold the chip in-process.


def _write_on_tpu(stacked_weights, slot_weights, slot):
    lora_slot_update_op.write_adapter_to_slot(
        stacked_weights,
        slot_weights,
        torch.tensor([slot], dtype=torch.int32, device=TPU),
    )


def _xla_compiles():
    return torch.tpu._get_cache_stats().num_compilations


def _writes_match_the_host_result_on_tpu():
    host_weights = _stacked_weights()
    stacked_weights = [weights.to(TPU) for weights in host_weights]
    for slot in (1, 3, 0):
        slots, slot_weights = _new_slots(host_weights)
        _write_on_tpu(stacked_weights, slot_weights.to(TPU), slot)
        host_weights = _expected(host_weights, slots, slot)
        for weights, expected in zip(stacked_weights, host_weights):
            assert torch.equal(weights.cpu(), expected)


@pytest.mark.tpu_test
@pytest.mark.spawns_tpu_workers
def test_writes_match_the_host_result_on_tpu():
    """The compiled program writes what the JAX function computes, and the
    caller's tensors see the result."""
    run_in_isolated_process(_writes_match_the_host_result_on_tpu)


def _one_compile_serves_every_slot():
    host_weights = _stacked_weights()
    stacked_weights = [weights.to(TPU) for weights in host_weights]
    _, slot_weights = _new_slots(host_weights)
    slot_weights = slot_weights.to(TPU)
    with torch.no_grad():
        _write_on_tpu(stacked_weights, slot_weights, 0)
    torch.accelerator.synchronize()
    graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
    compiles = _xla_compiles()
    # With grad enabled, unlike the first write.
    for slot in range(1, NUM_SLOTS):
        _write_on_tpu(stacked_weights, slot_weights, slot)
    torch.accelerator.synchronize()
    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == graphs
    assert _xla_compiles() == compiles


@pytest.mark.tpu_test
@pytest.mark.spawns_tpu_workers
def test_one_compile_serves_every_slot():
    """The slot is a program input and the op enters no_grad itself, so
    writing another slot, with grad enabled or not, neither traces again nor
    compiles another program."""
    run_in_isolated_process(_one_compile_serves_every_slot)


def _write_holds_no_second_copy_of_the_stacked_tensors():
    # 128 MiB, so a second copy stands out from allocator noise.
    host_weights = _stacked_weights([(NUM_SLOTS, 1 << 23)])
    stacked_weights = [weights.to(TPU) for weights in host_weights]
    _, slot_weights = _new_slots(host_weights)
    slot_weights = slot_weights.to(TPU)
    _write_on_tpu(stacked_weights, slot_weights, 0)
    torch.accelerator.synchronize()
    before = torch.accelerator.memory_allocated(TPU)
    torch.accelerator.reset_peak_memory_stats(TPU)
    _write_on_tpu(stacked_weights, slot_weights, 1)
    torch.accelerator.synchronize()
    extra = torch.accelerator.max_memory_allocated(TPU) - before
    # A temporary the size of one slot (32 MiB) is allowed; a second copy of
    # the stacked tensor (128 MiB) is not.
    assert extra < stacked_weights[0].nbytes // 2, extra


@pytest.mark.tpu_test
@pytest.mark.spawns_tpu_workers
def test_write_holds_no_second_copy_of_the_stacked_tensors():
    """The stacked tensors are donated, so a write updates them in place
    instead of allocating new ones next to them."""
    run_in_isolated_process(_write_holds_no_second_copy_of_the_stacked_tensors)
