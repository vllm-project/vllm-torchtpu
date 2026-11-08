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
"""Op wiring of the owner tail (the path a2a_owner_tail)."""

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
@pytest.mark.parametrize("owner_tail", [False, True])
def test_owner_tail_reaches_the_exchange(monkeypatch, owner_tail):
    env = {"TPU_MOE_RAGGED_A2A": "1"}
    if not owner_tail:
        env["TPU_DECODE_FAST_PATHS_OFF"] = "a2a_owner_tail"
    seen = _build_op(monkeypatch, env, owner_combine=True)
    assert seen["owner_combine"] is True
    assert seen["owner_tail"] is owner_tail


@pytest.mark.cpu_test
def test_owner_tail_follows_the_layer_owner_combine(monkeypatch):
    """A layer whose owner combine is not armed gets no owner tail, whatever
    the switches say."""
    env = {"TPU_MOE_RAGGED_A2A": "1"}
    seen = _build_op(monkeypatch, env, owner_combine=False)
    assert seen["owner_combine"] is False
    assert seen["owner_tail"] is False
