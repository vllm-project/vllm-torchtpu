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
"""The expert placement tables' effect on the Qwen3.5 KV budget."""

import pathlib

import pytest

from tests.models.qwen3_5.test_qwen3_5_hooks import (
    CREDIT_EP8,
    LAYERS,
    SLOT_LAYER_BYTES,
    _run_hook,
    _vllm_config,
)
from vllm_torchtpu.layers.adapter import expert_placement
from vllm_torchtpu.models import qwen3_5

pytestmark = pytest.mark.cpu_test

VALIDATION = pathlib.Path(qwen3_5.__file__).parent / "validation"
TP8_TABLE = VALIDATION / "expert_placement_tp8_ep8_agentx_corr_c32.json"
DP8_TABLE = VALIDATION / "expert_placement_dp8_ep8.json"


@pytest.fixture
def table(monkeypatch):
    def use(path):
        monkeypatch.setenv("TPU_EXPERT_PLACEMENT", str(path))
        expert_placement.clear_cache()

    yield use
    expert_placement.clear_cache()


def test_tp8_decode_legs_get_the_budget_they_ran_with(table):
    """68 slots per rank (64 primaries, 4 copies) over 60 layers, with the
    drafter's layer budgeted for its 4 copy slots: (12 x 60 - 4) slot-layers of
    3 x 4096 x 1024 B. Every TP8 decode leg of the record jobs logged
    42,509,364,992 B."""
    table(TP8_TABLE)
    cfg = _run_hook(_vllm_config(tp=8, dp=1, dcp=4, max_num_seqs=8))
    assert cfg.cache_config.kv_cache_memory_bytes == 42_509_364_992
    assert 33_500_000_000 + ((80 - 68) * LAYERS - 4) * SLOT_LAYER_BYTES == (
        42_509_364_992
    )


def test_a_permutation_table_budgets_like_no_table(table):
    table(DP8_TABLE)
    cfg = _run_hook(_vllm_config(tp=2, dp=4, dcp=1, max_num_seqs=28))
    assert cfg.cache_config.kv_cache_memory_bytes == 32_500_000_000 + CREDIT_EP8
