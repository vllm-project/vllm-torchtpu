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
"""ops/tp_all_reduce: the ModelAllReduce wrapper and the unarmed fallback."""

import pytest
import torch

from vllm_torchtpu.distributed import tpu_communicator
from vllm_torchtpu.models.minimax_m3.experimental.ops import tp_all_reduce


def test_zero_cap_does_not_arm():
    impl = tp_all_reduce.PallasTpAllReduce(0)
    assert not impl.setup(None, None, torch.device("cpu"))


def test_setup_arms_with_the_model_hidden_size(monkeypatch):
    calls = []
    monkeypatch.setattr(tp_all_reduce, "arm", lambda *args: calls.append(args) or True)
    config = type("C", (), {})()
    config.model_config = type("M", (), {"get_hidden_size": lambda self: 6144})()
    impl = tp_all_reduce.PallasTpAllReduce(524288)
    assert impl.setup(None, config, "dev")
    assert calls == [(6144, "dev", 524288)]


def test_installs_through_the_communicator_hook(monkeypatch):
    monkeypatch.setattr(tp_all_reduce, "arm", lambda *args: True)
    monkeypatch.setattr(tpu_communicator, "_model_all_reduce", None)
    config = type("C", (), {})()
    config.model_config = type("M", (), {"get_hidden_size": lambda self: 6144})()
    model = torch.nn.Module()
    impl = tp_all_reduce.PallasTpAllReduce(524288)
    setattr(model, tpu_communicator.MODEL_ALL_REDUCE_ATTR, impl)
    assert tpu_communicator.install_model_all_reduce(model, config, "dev")
    assert tpu_communicator._model_all_reduce is impl


@pytest.mark.parametrize("bad", [-1, "512", 1.5, True])
def test_rejects_bad_caps(bad):
    with pytest.raises(ValueError):
        tp_all_reduce.PallasTpAllReduce(bad)


def test_unarmed_all_reduce_defers_to_caller(monkeypatch):
    monkeypatch.setattr(tp_all_reduce, "_ARMED", None)
    x = torch.zeros(8, 1024, dtype=torch.bfloat16)
    assert tp_all_reduce.PallasTpAllReduce(524288).all_reduce((0, 1), x) is None


def test_armed_all_reduce_routes_only_its_group_and_bf16_2d(monkeypatch):
    calls = []

    def op(x, rank):
        calls.append(tuple(x.shape))
        return x + 1

    armed = tp_all_reduce._Armed(ranks=(0, 1), op=op, rank=torch.zeros(1, 1))
    monkeypatch.setattr(tp_all_reduce, "_ARMED", armed)
    impl = tp_all_reduce.PallasTpAllReduce(524288)
    x = torch.zeros(8, 1024, dtype=torch.bfloat16)
    assert impl.all_reduce((0, 2), x) is None  # another group
    assert impl.all_reduce((0, 1), x.float()) is None  # not bf16
    assert impl.all_reduce((0, 1), x[0]) is None  # not 2-D
    out = impl.all_reduce((0, 1), x)
    assert out is not None and calls == [(8, 1024)]
