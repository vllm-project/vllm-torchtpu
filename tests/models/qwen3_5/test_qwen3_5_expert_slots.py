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
"""The expert placement tables' effect on the Qwen3.5 KV budget, and the
validation entries that use them."""

import json
import pathlib

import pytest
import yaml

from tests.models.qwen3_5.test_qwen3_5_hooks import (
    CREDIT_EP8,
    LAYERS,
    SLOT_LAYER_BYTES,
    _run_hook,
    _vllm_config,
)
from vllm_torchtpu.distributed import collectives_options
from vllm_torchtpu.layers.adapter import expert_placement
from vllm_torchtpu.models import qwen3_5
from vllm_torchtpu.models.qwen3_5 import hooks

pytestmark = pytest.mark.cpu_test

VALIDATION = pathlib.Path(qwen3_5.__file__).parent / "validation"
TP8_TABLE = VALIDATION / "expert_placement_tp8_ep8_agentx_corr_c32.json"
DP8_TABLE = VALIDATION / "expert_placement_dp8_ep8.json"
POINTS = sorted(VALIDATION.glob("p*.yaml"))


@pytest.fixture(autouse=True)
def _tpu7x(monkeypatch):
    """The measured runs were on TPU7x, and the hook applies the calibrated KV
    budget only on a TPU7x device."""
    monkeypatch.setattr(hooks, "_chip_family", lambda: "7x")


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
    # 33,500,000,000 + (12 x 60 - 4) x 12,582,912 B.
    assert cfg.cache_config.kv_cache_memory_bytes == 42_509_364_992


def _write(tmp_path, source, layers):
    """`source`'s table, placing only `layers` (indices past the target's 60
    take layer 0's placement)."""
    data = json.loads(source.read_text())
    data["layers"] = {str(n): data["layers"][str(n % LAYERS)] for n in layers}
    if "copy_order" in data:
        data["copy_order"] = {
            str(n): data["copy_order"][str(n % LAYERS)] for n in layers
        }
    path = tmp_path / f"partial_{len(layers)}.json"
    path.write_text(json.dumps(data))
    return path


@pytest.mark.parametrize(
    "source, layers, credit",
    [
        # A permutation table that places half the layers: every layer still
        # holds 64 slots per rank, as without a table.
        (DP8_TABLE, range(30), CREDIT_EP8),
        # A copies table that places half the layers: 68 slots in those, 64
        # in the rest, and the drafter's 4 copy slots.
        (TP8_TABLE, range(30), (12 * 30 + 16 * 30 - 4) * SLOT_LAYER_BYTES),
        # A copies table that also lists the drafter's layer (index 60): that
        # is not a 61st target layer.
        (TP8_TABLE, range(61), (12 * LAYERS - 4) * SLOT_LAYER_BYTES),
    ],
    ids=["permutation_half", "copies_half", "copies_and_drafter"],
)
def test_layers_a_table_does_not_place_keep_their_credit(
    table, tmp_path, source, layers, credit
):
    table(_write(tmp_path, source, list(layers)))
    cfg = _run_hook(_vllm_config(tp=8, dp=1, dcp=4, max_num_seqs=8))
    assert cfg.cache_config.kv_cache_memory_bytes == 33_500_000_000 + credit


def test_a_permutation_table_budgets_like_no_table(table):
    """A table that only permutes the 512 experts over 8 ranks holds 64 slots
    per rank in every layer, as no table does. The budget applies only at the
    measured TP8 + DCP4 decode layout, so the table is checked there."""
    table(DP8_TABLE)
    cfg = _run_hook(_vllm_config(tp=8, dp=1, dcp=4, max_num_seqs=8))
    assert cfg.cache_config.kv_cache_memory_bytes == 33_500_000_000 + CREDIT_EP8


@pytest.mark.parametrize("path", POINTS, ids=lambda p: p.stem)
def test_decode_legs_place_their_experts_with_a_table_here(path):
    """Each decode leg names its table through the $EXPERT_PLACEMENT_TABLE
    placeholder; README.md says which table of this directory it stands for."""
    entry = yaml.safe_load(path.read_text())
    decode, prefill = entry["legs"]["decode"], entry["legs"]["prefill"]
    assert decode["env"]["TPU_EXPERT_PLACEMENT"] == "$EXPERT_PLACEMENT_TABLE"
    table = DP8_TABLE if "dp4tp2" in path.stem else TP8_TABLE
    placement = expert_placement.parse(json.loads(table.read_text()))
    # TP8 decode carries the copies table, DP4 x TP2 decode the permutation.
    want = 64 if "dp4tp2" in path.stem else 68
    assert placement.slots_per_rank == want
    assert "TPU_EXPERT_PLACEMENT" not in prefill["env"]


@pytest.mark.parametrize("path", POINTS, ids=lambda p: p.stem)
def test_decode_legs_arm_the_small_collectives(path):
    entry = yaml.safe_load(path.read_text())
    decode = entry["legs"]["decode"]["additional_config"]
    prefill = entry["legs"]["prefill"]["additional_config"]
    assert collectives_options.get(_Config(decode)).small_all_reduce is True
    assert collectives_options.get(_Config(prefill)).small_all_reduce is False


class _Config:
    def __init__(self, additional_config):
        self.additional_config = additional_config
