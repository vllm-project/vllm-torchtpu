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
"""Op wiring of TPU_MOE_A2A_OWNER_TAIL."""

from types import SimpleNamespace
from unittest import mock

import pytest
import vllm.config

import vllm_torchtpu.envs as envs
from vllm_torchtpu.layers.adapter import ragged_a2a_moe as ragged_a2a


def _build_op(monkeypatch, **knobs):
    """Build the exchange op with these envs knobs; returns the kwargs the op
    passes to ragged_a2a_moe."""
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
    # Patched with create=True and removed afterwards, so later tests read the
    # environment again (see envs.__getattr__).
    patches = [
        mock.patch.object(envs, name, value, create=True)
        for name, value in knobs.items()
    ]
    for p in patches:
        p.start()
    try:
        ragged_a2a._get_op(
            topk=10, activation="silu", rhs_quant_dtype=None, skip_padded_tokens=False
        )
    finally:
        for p in patches:
            p.stop()
    return seen


@pytest.mark.cpu_test
@pytest.mark.parametrize("owner_tail", [False, True])
def test_owner_tail_reaches_the_exchange(monkeypatch, owner_tail):
    seen = _build_op(
        monkeypatch,
        TPU_MOE_A2A_OWNER_COMBINE=True,
        TPU_MOE_A2A_OWNER_TAIL=owner_tail,
    )
    assert seen["owner_combine"] is True
    assert seen["owner_tail"] is owner_tail


@pytest.mark.cpu_test
def test_owner_tail_needs_the_owner_combine(monkeypatch):
    with pytest.raises(ValueError, match="TPU_MOE_A2A_OWNER_TAIL needs"):
        _build_op(
            monkeypatch,
            TPU_MOE_A2A_OWNER_COMBINE=False,
            TPU_MOE_A2A_OWNER_TAIL=True,
        )
