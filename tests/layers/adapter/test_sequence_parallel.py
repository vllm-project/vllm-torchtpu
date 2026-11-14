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
"""Row gating of sequence-parallel MoE: which steps slice their rows."""

from types import SimpleNamespace

import pytest
from vllm.distributed import parallel_state

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter import sequence_parallel as sp

pytestmark = pytest.mark.cpu_test


def _parallel(dp, tp):
    return SimpleNamespace(data_parallel_size=dp, tensor_parallel_size=tp)


@pytest.fixture
def groups(monkeypatch):
    """Initialized model-parallel groups of a chosen (dp, tp) layout."""

    def set_layout(dp, tp):
        monkeypatch.setattr(
            parallel_state, "model_parallel_is_initialized", lambda: True
        )
        monkeypatch.setattr(
            parallel_state, "get_dp_group", lambda: SimpleNamespace(world_size=dp)
        )
        monkeypatch.setattr(
            parallel_state, "get_tp_group", lambda: SimpleNamespace(world_size=tp)
        )

    return set_layout


@pytest.fixture
def threshold_512(monkeypatch):
    """The row threshold a model that follows each step's form would use."""
    monkeypatch.setattr(sp, "DEFAULT_MIN_ROWS", 512)


def test_min_rows_defaults_before_the_groups_exist(monkeypatch):
    monkeypatch.setattr(parallel_state, "model_parallel_is_initialized", lambda: False)
    assert sp.min_rows() == sp.DEFAULT_MIN_ROWS == 0


def test_a_small_step_stays_sliced_until_a_model_follows_the_form(groups, monkeypatch):
    """No model reads a step's form yet, so a 16-row sequence-parallel step
    must stay sliced: the model slices it, and the communicator must not treat
    it as replicated."""
    groups(1, 8)
    monkeypatch.setattr(envs, "TPU_MOE_SEQUENCE_PARALLEL", True)
    monkeypatch.setattr(sp, "_ACTIVE", False)
    assert sp.step_is_sequence_parallel(16) is True
    sp.set_active(sp.step_is_sequence_parallel(16))
    assert sp.rows_are_replicated(True) is False


@pytest.mark.parametrize("layout", [(1, 8), (4, 2), (2, 4)])
def test_unlisted_layouts_use_the_default(groups, layout):
    groups(*layout)
    assert sp.min_rows() == sp.DEFAULT_MIN_ROWS
    assert sp.min_rows(_parallel(*layout)) == sp.DEFAULT_MIN_ROWS


def test_a_layout_entry_applies_to_that_layout_only(groups, monkeypatch):
    monkeypatch.setattr(sp, "_FOLLOWS_STEP_FORM", True)
    monkeypatch.setitem(sp._MIN_ROWS_BY_LAYOUT, (2, 4), 96)
    groups(2, 4)
    assert sp.min_rows() == 96
    groups(2, 2)
    assert sp.min_rows() == sp.STEP_FORM_MIN_ROWS
    # An explicit parallel config wins over the process's groups.
    assert sp.min_rows(_parallel(2, 4)) == 96


@pytest.mark.parametrize(
    "layout,threshold", [((4, 2), 64), ((1, 8), 512), ((2, 4), 512), ((1, 2), 512)]
)
def test_a_model_that_follows_the_form_raises_the_threshold(
    groups, monkeypatch, layout, threshold
):
    """The Qwen3.5 decode crossovers: 64 rows at DP4 x TP2, 512 at TP8 and on
    unlisted layouts. Before a model follows the form, every layout keeps 0."""
    groups(*layout)
    monkeypatch.setattr(sp, "_FOLLOWS_STEP_FORM", False)
    assert sp.min_rows() == 0
    sp.follow_step_form()
    assert sp.follows_step_form()
    assert sp.min_rows() == threshold
    assert sp.min_rows(_parallel(*layout)) == threshold


def test_a_step_is_sliced_only_for_a_model_that_follows_the_form(groups, monkeypatch):
    groups(1, 8)
    monkeypatch.setattr(envs, "TPU_MOE_SEQUENCE_PARALLEL", True)
    monkeypatch.setattr(sp, "_FOLLOWS_STEP_FORM", False)
    assert sp.step_is_sequence_parallel(512) is True
    assert sp.step_rows_are_sliced(512) is False
    monkeypatch.setattr(sp, "_FOLLOWS_STEP_FORM", True)
    assert sp.step_rows_are_sliced(512) is True
    assert sp.step_rows_are_sliced(511) is False


@pytest.mark.parametrize(
    "enabled,rows,expected",
    [(True, 511, False), (True, 512, True), (True, 4096, True), (False, 4096, False)],
)
@pytest.mark.usefixtures("threshold_512")
def test_step_is_sequence_parallel_from_the_row_count(
    groups, monkeypatch, enabled, rows, expected
):
    groups(1, 8)
    monkeypatch.setattr(envs, "TPU_MOE_SEQUENCE_PARALLEL", enabled)
    assert sp.step_is_sequence_parallel(rows) is expected


@pytest.mark.parametrize("active", [True, False])
def test_set_active_decides_sliced_versus_replicated(monkeypatch, active):
    monkeypatch.setattr(sp, "_ACTIVE", not active)
    sp.set_active(active)
    assert sp.is_active() is active
    assert sp.rows_are_sliced(True) is active
    assert sp.rows_are_replicated(True) is not active
    # A layer that is not sequence-parallel is neither.
    assert not sp.rows_are_sliced(False)
    assert not sp.rows_are_replicated(False)


def test_fp8_gather_is_always_on():
    assert sp.fp8_gather() is True


@pytest.mark.usefixtures("threshold_512")
def test_check_buckets_refuses_an_uneven_sliced_bucket(groups):
    groups(1, 8)
    sp.check_buckets([8, 256, 512, 1024], tp_size=8)
    with pytest.raises(NotImplementedError, match="do not divide"):
        sp.check_buckets([516], tp_size=8)
    # Below the threshold the bucket is replicated, so it need not divide.
    sp.check_buckets([12], tp_size=8)
