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
"""The pipeline hand-off program is named after its devices.

Every pipeline column traces the same permute program; only the device
assignment differs, and torch_tpu's compile cache does not key on it. A shared
cache then hands one column another column's executable. The two stages of a
column trace it for different local devices, and a restart can swap which host
holds which stage, so the name also carries the device it runs on.
"""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch_tpu")

from vllm_torchtpu.distributed import pp_shift  # noqa: E402


def _built_names(monkeypatch, device_ids, local=None):
    calls = []
    monkeypatch.setattr(pp_shift, "_OPS", {})
    local = device_ids[0] if local is None else local
    monkeypatch.setattr(pp_shift, "_get_tpu_global_device_id", lambda: local)
    monkeypatch.setattr(pp_shift, "pp_positions", lambda: (0, 1))
    monkeypatch.setattr(
        pp_shift,
        "jax_op",
        lambda name, fn, **kwargs: calls.append((name, fn.__name__)) or object(),
    )
    mesh = SimpleNamespace(
        devices=np.asarray([SimpleNamespace(id=d) for d in device_ids])
    )
    pp_shift.pp_permute_op(mesh)
    pp_shift.pp_permute_op(mesh, num_tensors=2)
    return calls


def test_columns_get_distinct_program_names(monkeypatch):
    first = _built_names(monkeypatch, (2, 10))
    second = _built_names(monkeypatch, (3, 11))

    assert first == [
        ("vllm_torchtpu::pp_permute_dev2_10_on2", "pp_permute1_dev2_10_on2"),
        ("vllm_torchtpu::pp_permute2_dev2_10_on2", "pp_permute2_dev2_10_on2"),
    ]
    assert not {name for pair in first for name in pair} & {
        name for pair in second for name in pair
    }


def test_stages_of_a_column_get_distinct_program_names(monkeypatch):
    # j-883084e3: restarts warm from the first attempt's per-host cache died
    # in the hand-off warmup; this name was one cache key on both hosts.
    first = _built_names(monkeypatch, (6, 14), local=6)
    last = _built_names(monkeypatch, (6, 14), local=14)

    assert not {name for pair in first for name in pair} & {
        name for pair in last for name in pair
    }
