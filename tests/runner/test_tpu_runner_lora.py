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
"""LoRA in the TPU runner: the mapping that _prepare_inputs hands to the LoRA
manager, and the HBM an adapter swap reserves."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
from runner_input_test_utils import _make_runner, _scheduler_output
from vllm.lora.request import LoRARequest

from vllm_torchtpu.lora.lora_manager import TPUWorkerLoRAManager
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


def test_each_chunk_maps_its_own_requests_padded_to_what_the_graph_reads():
    """A step split into chunks of 3 requests. Each chunk maps only its own
    requests, padded with id 0 (no adapter) to the token bucket and to the
    logits rows, which are the lengths the compiled graph reads."""
    scheduled = [1, 1, 1, 1, 3]
    runner = _make_runner(
        num_computed_tokens=[8, 8, 8, 8, 0],
        prompt_tokens=[8, 8, 8, 8, 3],
        scheduled_tokens=scheduled,
        token_paddings=[16, 64],
        pcp_size=1,
    )
    runner.num_reqs_max_model_len = 3
    runner.lora_config = SimpleNamespace()
    runner.input_batch.request_lora_mapping = np.array(
        [7, 0, 9, 9, 9, 0, 0, 0], dtype=np.int64
    )
    adapters = {
        7: LoRARequest("seven", 7, "/adapters/seven"),
        9: LoRARequest("nine", 9, "/adapters/nine"),
    }
    runner.input_batch.lora_id_to_lora_request = adapters
    calls = []
    runner._set_active_loras = lambda *args: calls.append(args)

    graph_lengths = []
    start_index = 0
    while start_index < len(scheduled):
        prepared = TPUModelRunner._prepare_inputs(
            runner, _scheduler_output(scheduled), start_index, num_decode_reqs=4
        )
        logits_indices, end_index = prepared[1], prepared[4]
        graph_lengths.append((runner.input_ids.shape[0], logits_indices.shape[0]))
        start_index = end_index

    assert [(prompt, token) for prompt, token, _ in calls] == [
        ((7, 0, 9) + (0,) * 5, (7, 0, 9) + (0,) * 13),
        ((9, 9) + (0,) * 6, (9, 9, 9, 9) + (0,) * 12),
    ]
    assert [(len(token), len(prompt)) for prompt, token, _ in calls] == graph_lengths
    # Each chunk activates at least the adapters its own mapping uses.
    for prompt, _, requests in calls:
        assert set(prompt) - {0} <= {r.lora_int_id for r in requests}


def test_lora_slot_write_hbm_bytes_comes_from_the_lora_manager():
    """Without LoRA nothing is reserved; with it, the LoRA manager sizes the
    reserve."""
    no_lora = SimpleNamespace(lora_config=None)
    assert TPUModelRunner.lora_slot_write_hbm_bytes(no_lora) == 0

    lora_manager = MagicMock(spec=TPUWorkerLoRAManager)
    lora_manager.slot_write_hbm_bytes.return_value = 123
    with_lora = SimpleNamespace(
        lora_config=SimpleNamespace(), lora_manager=lora_manager
    )
    assert TPUModelRunner.lora_slot_write_hbm_bytes(with_lora) == 123
