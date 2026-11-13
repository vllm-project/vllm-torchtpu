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
from types import SimpleNamespace as NS

from vllm_torchtpu.offload import model_owned_pool, raiden_store


def test_registered_pool_geometry_wins(monkeypatch):
    monkeypatch.setattr(model_owned_pool, "_POOLS", {})
    cfg = NS(model_config=NS(architecture="FooForCausalLM"))
    other = NS(model_config=NS(architecture="BarForCausalLM"))
    assert raiden_store.resolve_model_owned_pool_geometry(cfg, None) is None
    model_owned_pool.register_model_owned_pool(
        ["FooForCausalLM"], lambda v, k: (512, (2, 128), "bf16", 512), lambda k: 2
    )
    assert raiden_store.resolve_model_owned_pool_geometry(cfg, None) == (
        512,
        (2, 128),
        "bf16",
        512,
    )
    assert raiden_store.resolve_kernel_geometry(cfg, None) == (
        512,
        (2, 128),
        "bf16",
        512,
    )
    assert model_owned_pool.model_owned_pool(other) is None
    assert model_owned_pool.model_owned_pool(cfg).regions(None) == 2
    assert model_owned_pool.model_owned_pool(None) is None
