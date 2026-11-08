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
"""Op wiring of the ragged all-to-all row copy paths and the row scatter's
send index."""

from types import SimpleNamespace

import pytest
import vllm.config

from vllm_torchtpu.layers.adapter import ragged_a2a_moe as ragged_a2a


def _build_op(monkeypatch, env, owner_combine=False):
    """Build the exchange op under these environment variables; returns the
    kwargs the op passes to ragged_a2a_moe."""
    monkeypatch.setattr(vllm.config, "get_current_vllm_config_or_none", lambda: None)
    monkeypatch.setattr(
        ragged_a2a, "get_dp_tp_mesh", lambda: SimpleNamespace(shape={"dp": 8, "tp": 4})
    )
    monkeypatch.setattr(ragged_a2a, "_OPS", {})
    monkeypatch.setattr(ragged_a2a, "_routed_rows_per_decode_step", lambda topk: 0)
    seen = {}

    def _ragged_a2a_moe(*args, **kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(ragged_a2a, "ragged_a2a_moe", _ragged_a2a_moe)

    class _Op:
        def register_fake(self, fn):
            pass

    def _jax_op(name, fn, **kwargs):
        fn(*([None] * 8))
        return _Op()

    monkeypatch.setattr(ragged_a2a.pallas, "jax_op", _jax_op)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    ragged_a2a._get_op(
        topk=10,
        activation="silu",
        rhs_quant_dtype=None,
        skip_padded_tokens=False,
        owner_combine=owner_combine,
    )
    return seen


@pytest.mark.cpu_test
@pytest.mark.parametrize("row_gather", [False, True])
@pytest.mark.parametrize("row_scatter", [False, True])
def test_row_copy_paths_reach_the_exchange(monkeypatch, row_gather, row_scatter):
    off = [
        name
        for name, on in (
            ("a2a_row_gather", row_gather),
            ("a2a_row_scatter", row_scatter),
        )
        if not on
    ]
    seen = _build_op(
        monkeypatch,
        {"TPU_MOE_RAGGED_A2A": "1", "TPU_DECODE_FAST_PATHS_OFF": ",".join(off)},
    )
    assert seen["sorted_plan"] is True
    assert seen["row_gather"] is row_gather
    assert seen["row_scatter"] is row_scatter


@pytest.mark.cpu_test
@pytest.mark.parametrize("path", ["a2a_row_gather", "a2a_row_scatter"])
def test_row_copy_paths_need_the_sorted_plan(monkeypatch, path):
    other = {"a2a_row_gather": "a2a_row_scatter", "a2a_row_scatter": "a2a_row_gather"}
    env = {
        "TPU_MOE_RAGGED_A2A": "1",
        "TPU_DECODE_FAST_PATHS_OFF": f"a2a_sorted_plan,{other[path]}",
    }
    with pytest.raises(ValueError, match=f"{path} needs a2a_sorted_plan"):
        _build_op(monkeypatch, env)


@pytest.mark.cpu_test
@pytest.mark.parametrize("rows_minor", [False, True])
def test_send_index_layout_reaches_the_exchange(monkeypatch, rows_minor):
    env = {"TPU_MOE_RAGGED_A2A": "1"}
    if not rows_minor:
        env["TPU_DECODE_FAST_PATHS_OFF"] = "a2a_send_rows_minor"
    seen = _build_op(monkeypatch, env)
    assert seen["row_scatter"] is True
    assert seen["send_index_rows_minor"] is rows_minor


@pytest.mark.cpu_test
def test_send_index_layout_follows_the_row_scatter(monkeypatch):
    env = {"TPU_MOE_RAGGED_A2A": "1", "TPU_DECODE_FAST_PATHS_OFF": "a2a_row_scatter"}
    seen = _build_op(monkeypatch, env)
    assert seen["row_scatter"] is False
    assert seen["send_index_rows_minor"] is False
