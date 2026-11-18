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
"""Tests for the runner helpers' persistent compile cache."""

import os
from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.compilation import helper_cache, tpu_compiler


@pytest.fixture
def helpers(tmp_path, monkeypatch):
    """A helper backend over a fake TpuBackend that records what it compiled
    and what it replayed; `run(rank=..)` stands for one process."""
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path / "vllm"))
    monkeypatch.setenv("TPU_PRECOMPILE_PEER_REPLAY", "0")
    monkeypatch.setattr(
        tpu_compiler, "compute_tpu_compilation_hash", lambda config: config.tpu_hash
    )
    monkeypatch.setattr(
        "vllm.compilation.caching.aot_compile_hash_factors",
        lambda config: [config.compute_hash()],
    )
    compiled, replayed, plain = [], [], []

    class Compiled:
        def __call__(self, *args):
            return ("compiled",)

        def serialize(self):
            return f"entry-{len(compiled)}"

    def fake_backend(graph, example_inputs):
        compiled.append(graph)
        return Compiled()

    def fake_deserialize(entry):
        replayed.append(entry)
        return lambda *args: (entry,)

    monkeypatch.setattr(tpu_compiler, "_tpu_backend", fake_backend)
    monkeypatch.setattr(tpu_compiler, "_deserialize_entry", fake_deserialize)
    monkeypatch.setattr(
        torch._dynamo,
        "lookup_backend",
        lambda name: lambda graph, inputs: plain.append(graph) or (lambda *a: "plain"),
    )

    def run(
        rank=0,
        config_hash="cfg1",
        tpu_hash="tpu1",
        graph=None,
        inputs=None,
        name="sample",
        configure=True,
    ):
        helper_cache.reset()
        if configure:
            helper_cache.configure(
                SimpleNamespace(
                    tpu_hash=tpu_hash,
                    compute_hash=lambda: config_hash,
                    parallel_config=SimpleNamespace(rank=rank, data_parallel_rank=0),
                )
            )
        if graph is None:
            graph = torch.fx.symbolic_trace(lambda x: x * 2 + 1)
        if inputs is None:
            inputs = [torch.zeros(4, 8)]
        return helper_cache.backend(name)(graph, inputs)

    yield run, compiled, replayed, plain, tmp_path / "vllm" / "tpu_helper_cache"
    helper_cache.reset()


def test_unconfigured_uses_the_plain_tpu_backend(helpers):
    run, compiled, replayed, plain, root = helpers
    assert run(configure=False)() == "plain"
    assert len(plain) == 1
    assert compiled == []
    assert not root.exists()


def test_a_later_start_replays_the_saved_helper_graph(helpers):
    run, compiled, replayed, plain, root = helpers
    first = run()
    # A later process, same configuration: replayed, not compiled.
    second = run()

    assert len(compiled) == 1
    assert second() == replayed[-1]
    (digest,) = os.listdir(root)
    assert len(os.listdir(root / digest / "rank_0_0" / "sample")) == 1
    assert first() is not None
    assert plain == []


@pytest.mark.parametrize("change", ["config_hash", "tpu_hash", "inputs", "graph"])
def test_anything_that_decides_the_graph_compiles_again(helpers, change):
    run, compiled, replayed, plain, root = helpers
    run()
    changed = {
        "config_hash": {"config_hash": "cfg2"},
        "tpu_hash": {"tpu_hash": "tpu2"},
        "inputs": {"inputs": [torch.zeros(8, 8)]},
        "graph": {"graph": torch.fx.symbolic_trace(lambda x: x * 3 + 1)},
    }[change]
    run(**changed)

    assert len(compiled) == 2


def test_helpers_keep_separate_directories(helpers):
    run, compiled, replayed, plain, root = helpers
    run(name="sample")
    run(name="gather")

    assert len(compiled) == 2
    (digest,) = os.listdir(root)
    assert sorted(os.listdir(root / digest / "rank_0_0")) == ["gather", "sample"]


def test_a_rank_replays_a_helper_graph_a_peer_saved(helpers, monkeypatch):
    run, compiled, replayed, plain, root = helpers
    monkeypatch.setenv("TPU_PRECOMPILE_PEER_REPLAY", "1")
    run(rank=0)
    compiled_before = len(compiled)
    run(rank=1)

    assert len(compiled) == compiled_before
    (digest,) = os.listdir(root)
    assert sorted(os.listdir(root / digest)) == ["rank_0_0", "rank_1_0"]


def test_a_graph_without_a_fingerprint_uses_the_plain_backend(helpers, monkeypatch):
    run, compiled, replayed, plain, root = helpers
    monkeypatch.setattr(
        "vllm_torchtpu.compilation.graph_fingerprint._graph_fingerprint",
        lambda graph: None,
    )
    assert run()() == "plain"
    assert compiled == []
