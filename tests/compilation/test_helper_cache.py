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

from tests.compilation.test_tpu_compiler import _install_fake_backend
from vllm_torchtpu.compilation import helper_cache, tpu_compiler


def _make_config(
    *,
    rank: int = 0,
    dp_index: int = 0,
    dp_rank: int = 0,
    config_hash: str = "cfg1",
    tpu_hash: str = "tpu1",
    cache_dir: str | None = None,
    num_gpu_blocks: int | None = None,
    gpu_memory_utilization: float = 0.9,
):
    return SimpleNamespace(
        tpu_hash=tpu_hash,
        compute_hash=lambda: f"{config_hash}-{tpu_hash}",
        model_config=SimpleNamespace(compute_hash=lambda: config_hash),
        parallel_config=SimpleNamespace(
            rank=rank,
            data_parallel_rank=dp_rank,
            data_parallel_index=dp_index,
            compute_hash=lambda: "p1",
        ),
        compilation_config=SimpleNamespace(
            cache_dir=cache_dir,
            compute_hash=lambda: "cc1",
        ),
        cache_config=SimpleNamespace(
            num_gpu_blocks=num_gpu_blocks,
            gpu_memory_utilization=gpu_memory_utilization,
        ),
        quant_config=None,
    )


@pytest.fixture
def helpers(tmp_path, monkeypatch):
    """Helper backend over `_install_fake_backend` recording compiles and replays."""
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path / "vllm"))
    monkeypatch.setenv("TPU_PRECOMPILE_PEER_REPLAY", "0")
    monkeypatch.setattr(
        tpu_compiler,
        "compute_tpu_compilation_hash",
        lambda config, *, include_kv=True: (
            f"{config.tpu_hash}-kv{config.cache_config.num_gpu_blocks}"
            if include_kv
            else config.tpu_hash
        ),
    )
    fake = _install_fake_backend(monkeypatch)
    plain = []
    monkeypatch.setattr(
        torch._dynamo,
        "lookup_backend",
        lambda name: lambda graph, inputs: plain.append(graph) or (lambda *a: "plain"),
    )

    def run(
        rank=0,
        dp_index=0,
        dp_rank=0,
        config_hash="cfg1",
        tpu_hash="tpu1",
        cache_dir=None,
        num_gpu_blocks=None,
        gpu_memory_utilization=0.9,
        graph=None,
        inputs=None,
        name="sample",
        configure=True,
    ):
        helper_cache.reset()
        fake.set_tag(rank)
        if configure:
            helper_cache.configure(
                _make_config(
                    rank=rank,
                    dp_index=dp_index,
                    dp_rank=dp_rank,
                    config_hash=config_hash,
                    tpu_hash=tpu_hash,
                    cache_dir=cache_dir,
                    num_gpu_blocks=num_gpu_blocks,
                    gpu_memory_utilization=gpu_memory_utilization,
                )
            )
        if graph is None:
            graph = torch.fx.symbolic_trace(lambda x: x * 2 + 1)
        if inputs is None:
            inputs = [torch.zeros(4, 8)]
        return helper_cache.backend(name)(graph, inputs)

    yield (
        run,
        fake.compiled,
        fake.replayed,
        plain,
        tmp_path / "vllm" / "tpu_helper_cache",
    )
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


def test_kv_cache_size_does_not_change_helper_digest(helpers):
    run, compiled, replayed, plain, root = helpers
    run(num_gpu_blocks=None, gpu_memory_utilization=0.9)
    run(num_gpu_blocks=128, gpu_memory_utilization=0.95)

    assert len(compiled) == 1
    assert len(replayed) == 1


def test_dense_dp_replicas_use_data_parallel_index(helpers):
    """Dense DP workers have data_parallel_rank=0 on every replica; using
    data_parallel_index keeps their helper caches separate."""
    run, compiled, replayed, plain, root = helpers
    run(rank=0, dp_rank=0, dp_index=0)
    run(rank=0, dp_rank=0, dp_index=1)

    assert len(compiled) == 2
    hashes = os.listdir(root)
    assert len(hashes) == 1
    assert sorted(os.listdir(os.path.join(root, hashes[0]))) == [
        "rank_0_0",
        "rank_0_1",
    ]


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


def test_helper_compile_decorator_replays_bound_method_across_starts(
    tmp_path, monkeypatch
):
    """Decorating a module method with helper_compile produces the same key
    across two simulated starts when compilation_config.cache_dir is set."""
    monkeypatch.setenv("TPU_PRECOMPILE_PEER_REPLAY", "0")
    monkeypatch.setattr(
        tpu_compiler,
        "compute_tpu_compilation_hash",
        lambda config, *, include_kv=True: "tpu1",
    )
    fake = _install_fake_backend(monkeypatch)
    custom_cache = str(tmp_path / "custom_cache")

    class HelperModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.ones(4))

        @helper_cache.helper_compile("bound_add")
        def add_one(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.bias

    x = torch.zeros(2, 4)

    helper_cache.reset()
    helper_cache.configure(_make_config(cache_dir=custom_cache))
    torch._dynamo.reset()
    HelperModule().add_one(x)
    assert len(fake.compiled) == 1
    assert len(fake.replayed) == 0

    helper_cache.reset()
    helper_cache.configure(_make_config(cache_dir=custom_cache))
    torch._dynamo.reset()
    mod2 = HelperModule()
    mod2.add_one(x)
    assert len(fake.compiled) == 1
    assert len(fake.replayed) == 1
    # A grad-mode change (like logits sweep -> execute_model) re-traces in Dynamo
    # but reuses the in-process runnable without calling _replay a second time.
    with torch.no_grad():
        mod2.add_one(x)
    assert len(fake.compiled) == 1
    assert len(fake.replayed) == 1
    assert (tmp_path / "custom_cache" / "tpu_helper_cache").exists()
    helper_cache.reset()


def test_helper_compile_cache_switch_and_worker_gate(monkeypatch):
    """additional_config.helper_compile_cache defaults to False, rejects non-bools,
    and gates helper_cache.configure alongside VLLM_DISABLE_COMPILE_CACHE."""
    import inspect

    import vllm.envs as vllm_envs

    from vllm_torchtpu.worker.tpu_worker import TPUWorker

    assert (
        helper_cache.helper_compile_cache_enabled(SimpleNamespace(additional_config={}))
        is False
    )
    assert (
        helper_cache.helper_compile_cache_enabled(
            SimpleNamespace(additional_config={"helper_compile_cache": True})
        )
        is True
    )
    with pytest.raises(ValueError, match="helper_compile_cache"):
        helper_cache.helper_compile_cache_enabled(
            SimpleNamespace(additional_config={"helper_compile_cache": "yes"})
        )

    configured = []
    monkeypatch.setattr(helper_cache, "configure", lambda cfg: configured.append(cfg))

    for enabled, disable_cache, expected in (
        (False, False, False),
        (False, True, False),
        (True, True, False),
        (True, False, True),
    ):
        configured.clear()
        monkeypatch.setattr(vllm_envs, "VLLM_DISABLE_COMPILE_CACHE", disable_cache)
        cfg = SimpleNamespace(additional_config={"helper_compile_cache": enabled})
        assert helper_cache.configure_if_enabled(cfg) is expected
        assert len(configured) == (1 if expected else 0)

    assert "configure_if_enabled" in inspect.getsource(
        TPUWorker.compile_or_warm_up_model
    )
