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
"""Multi-LoRA end to end with tensor parallelism over every local chip.

These tests are in a separate file from test_lora.py for the reason
test_generate_tp.py gives: a TP engine needs all TPU devices. The adapters and
the merged checkpoint are test_lora.py's.

With tensor parallelism a greedy batch repeated on one engine may pick a
different token where the top tokens are close, with or without LoRA. So runs
are compared by the distribution of the first token, and a difference counts
only against how far another adapter moves that distribution.

A test here runs on pull requests unless it is marked nightly.

Run with: pytest tests/entrypoints/llm/test_lora_tp.py -v
"""

import pytest
from vllm.lora.request import LoRARequest

from tests.entrypoints.llm.test_lora import (
    LORA_ARGS,
    ORACLE_PROMPTS,
    _allow_callable_rpcs,  # noqa: F401
    _engine,
    _first_token_distributions,
    _greedy,
    _mean_total_variation,
    adapters,  # noqa: F401
    merged_model,  # noqa: F401
)

pytestmark = [pytest.mark.tpu_test, pytest.mark.multichip]


@pytest.fixture(scope="module")
def tp_size() -> int:
    try:
        from tpu_info import device as tpu_device

        _, num_chips = tpu_device.get_local_chips()
    except Exception as e:
        pytest.skip(f"Unable to detect local TPU chip count: {e}")
    if num_chips < 4:
        pytest.skip(f"LoRA TP tests need at least 4 chips, found {num_chips}")
    return num_chips


def _lora_slots(worker):
    """The adapter id in each LoRA slot of this rank, and the class names of
    its LoRA layers."""
    manager = worker.model_runner.lora_manager._adapter_manager
    layer_classes = sorted(type(module).__name__ for module in manager.modules.values())
    return list(manager.lora_index_to_id), layer_classes


@pytest.fixture(scope="class")
def tp_lora_llm(tp_size):
    with _engine(**LORA_ARGS, tensor_parallel_size=tp_size) as llm:
        yield llm


class TestLoRATensorParallelEngine:
    @pytest.mark.nightly
    def test_adapters_change_the_output(self, tp_lora_llm, adapters):  # noqa: F811
        base = _greedy(tp_lora_llm, None)
        a = _greedy(tp_lora_llm, adapters["a"])
        b = _greedy(tp_lora_llm, adapters["b"])

        assert a != base and b != base and a != b

    def test_every_rank_loads_the_adapter(self, tp_lora_llm, adapters, tp_size):  # noqa: F811
        _greedy(tp_lora_llm, adapters["a"])

        ranks = tp_lora_llm.collective_rpc(_lora_slots)

        assert len(ranks) == tp_size
        for slots, lora_layers in ranks:
            assert adapters["a"].lora_int_id in slots
            # The manager swapped in the TPU layer for every o_proj and
            # down_proj.
            assert "TpuRowParallelLinearWithLoRA" in lora_layers
            assert "RowParallelLinearWithLoRA" not in lora_layers
        assert all(rank == ranks[0] for rank in ranks)

    @pytest.mark.nightly
    def test_mixed_batch_matches_uniform_batches(self, tp_lora_llm, adapters):  # noqa: F811
        """Rows of one batch that use different adapters, or none, come out as
        they do in batches that use a single one, and not as another row's
        adapter would make them."""
        cycle = [adapters["a"], adapters["b"], None]
        rows = range(len(ORACLE_PROMPTS))
        uniform = [
            _first_token_distributions(tp_lora_llm, request) for request in cycle
        ]
        mixed = _first_token_distributions(tp_lora_llm, [cycle[i % 3] for i in rows])

        own = _mean_total_variation(mixed, [uniform[i % 3][i] for i in rows])
        other = _mean_total_variation(mixed, [uniform[(i + 1) % 3][i] for i in rows])
        assert own < 0.25 * other, (own, other)

    @pytest.mark.nightly
    def test_evicted_adapter_reloads_with_the_same_output(
        self,
        tp_lora_llm,
        adapters,  # noqa: F811
    ):
        before = _first_token_distributions(tp_lora_llm, adapters["a"])
        other = _first_token_distributions(tp_lora_llm, adapters["b"])
        _greedy(tp_lora_llm, adapters["c"])
        # The CPU cache holds max_loras adapters, so "a" is now gone from it.
        assert adapters["a"].lora_int_id not in tp_lora_llm.llm_engine.list_loras()

        after = _first_token_distributions(tp_lora_llm, adapters["a"])

        reload_gap = _mean_total_variation(after, before)
        adapter_effect = _mean_total_variation(other, before)
        assert reload_gap < 0.25 * adapter_effect, (reload_gap, adapter_effect)

    @pytest.mark.nightly
    def test_adapter_rpcs_reach_every_rank(self, tp_lora_llm, adapters):  # noqa: F811
        engine = tp_lora_llm.llm_engine
        request = LoRARequest("rpc", 50, adapters["a"].lora_path)

        assert engine.add_lora(request)
        assert request.lora_int_id in engine.list_loras()
        for slots, _ in tp_lora_llm.collective_rpc(_lora_slots):
            assert request.lora_int_id in slots
        assert engine.remove_lora(request.lora_int_id)
        assert request.lora_int_id not in engine.list_loras()
        for slots, _ in tp_lora_llm.collective_rpc(_lora_slots):
            assert request.lora_int_id not in slots


@pytest.mark.nightly
def test_lora_matches_the_adapter_merged_into_the_weights(
    tp_size,
    adapters,  # noqa: F811
    merged_model,  # noqa: F811
):
    """As in test_lora.py, with all three engines at the same TP size: the
    LoRA output sits much closer to the merged model than the adapter moves it
    from the base model. A rank that drops the delta, or applies another
    rank's slice of it, puts the ratio near 1 or above."""
    with _engine(**LORA_ARGS, tensor_parallel_size=tp_size) as llm:
        lora = _first_token_distributions(llm, adapters["a"])
    with _engine(tensor_parallel_size=tp_size) as llm:
        base = _first_token_distributions(llm)
    with _engine(model=merged_model, tensor_parallel_size=tp_size) as llm:
        merged = _first_token_distributions(llm)

    lora_gap = _mean_total_variation(lora, merged)
    adapter_effect = _mean_total_variation(merged, base)
    assert lora_gap < 0.5 * adapter_effect, (lora_gap, adapter_effect)


@pytest.mark.nightly
def test_lm_head_adapter_matches_one_chip(tp_size, adapters):  # noqa: F811
    """An adapter on lm_head alone moves the output as it does on one chip."""
    with _engine(**LORA_ARGS, tensor_parallel_size=tp_size) as llm:
        lm_head = _first_token_distributions(llm, adapters["lm_head"])
    with _engine(**LORA_ARGS) as llm:
        one_chip = _first_token_distributions(llm, adapters["lm_head"])
        one_chip_base = _first_token_distributions(llm)

    tp_gap = _mean_total_variation(lm_head, one_chip)
    adapter_effect = _mean_total_variation(one_chip, one_chip_base)
    assert tp_gap < 0.5 * adapter_effect, (tp_gap, adapter_effect)
