# Copyright 2025 Google LLC
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
"""Shared pytest configuration for all tests."""

import os
import sys

# Keep TP collectives compiled in-graph (no graph break) for fullgraph TP.
os.environ.setdefault("TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS",
                      "false")
os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "0")

# Forking after TorchTPU/JAX/PyTorch background threads have started can
# deadlock vLLM engine or worker subprocess startup. Set this before test
# modules import vLLM so every LLM test uses a clean spawned process.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

# Disable vLLM background usage telemetry in tests to prevent background threads
# and IPC sockets from hanging multi-process worker teardown.
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
os.environ.setdefault("VLLM_DO_NOT_TRACK", "1")

import pytest
import torch


def pytest_collection_modifyitems(items):
    """Run whole-slice worker-spawning tests before in-process TPU tests.

    Once the pytest process itself initializes the TPU runtime (any test that
    runs jax/torch_tpu in-process), a test that later spawns a fresh
    ``torch.distributed.run`` worker group can no longer bootstrap the slice:
    the workers fail with "PjRtClient is not initialized" and libtpu may tear
    down the whole session with SLICE_FAILURE_SW_INJECT_ERROR. libtpu exposes
    no release API, so the only safe ordering is spawn-first. The sort is
    stable, so relative order within each group is unchanged.
    """
    items.sort(key=lambda item: 0
               if item.get_closest_marker("spawns_tpu_workers") else 1)


def pytest_addoption(parser):
    """Add --use-tpu command line option."""
    parser.addoption(
        "--use-tpu",
        action="store_true",
        default=True,
        help="Run tests on TPU device instead of CPU",
    )


@pytest.fixture(autouse=True)
def _clear_kv_cache_layout_cache():
    """Stop `get_kv_cache_layout()`'s `lru_cache` leaking between tests.

    Memoized process-wide, so the first caller would pin the answer and a
    later `monkeypatch.setenv` would be silently ignored.
    """
    from vllm.v1.attention.backends.utils import set_kv_cache_layout
    set_kv_cache_layout(None)
    yield
    set_kv_cache_layout(None)


@pytest.fixture
def vllm_config_context():
    """Run a test body inside an active vLLM config.

    With `VLLM_KV_CACHE_LAYOUT` unset, `get_kv_cache_layout()` asks the KV
    connector, which needs a current config. Production always has one; tests
    calling impls or platform helpers directly do not.
    """
    from vllm.config import VllmConfig, set_current_vllm_config
    with set_current_vllm_config(VllmConfig()):
        yield


@pytest.fixture
def device(request):
    """Get the device to run tests on (CPU or TPU).

    Usage:
        pytest tests/ -v              # Run on CPU
        pytest tests/ -v --use-tpu    # Run on TPU
    """
    use_tpu = request.config.getoption("--use-tpu")
    if use_tpu:
        try:
            import torch_tpu  # noqa: F401
            return torch.device("tpu")
        except ImportError:
            pytest.skip("torch_tpu not available")
        except Exception as e:
            pytest.fail(f"TPU requested but failed to initialize: {e}")
    return torch.device("cpu")


def pytest_collection_finish(session):
    """Fail fast if collecting the suite claimed the accelerator.

    libtpu is process-exclusive. A PJRT client built in the pytest PARENT locks
    out every child, and vLLM's engine and workers run in spawned children (see
    VLLM_WORKER_MULTIPROC_METHOD above), so every engine-starting test in the
    run then dies with "pjrt client not initialized" -- far from the module
    that actually claimed the device.

    Collection is import-time, so anything evaluated at module scope counts:
    the usual culprit is a `pytest.mark.skipif(jax.device_count() < N)`, which
    builds a client to answer and does so even when the marker goes on to
    deselect every test in the file. Put that check in a fixture instead; a
    fixture runs at setup, after deselection.

    Checked here rather than left to CI because the failure is invisible to any
    run that does not collect the offending file and an engine test together.
    """
    if "jax" not in sys.modules:
        return
    from jax._src import xla_bridge
    if not xla_bridge.backends_are_initialized():
        return
    raise pytest.UsageError(
        "A JAX backend was initialized during test collection, which claims "
        "the TPU for this process and makes every spawned engine/worker child "
        "fail with 'pjrt client not initialized'. Some test module touches the "
        "accelerator at import time -- most often a module-level "
        "`jax.device_count()` in a `skipif`. Move it into a fixture. To find "
        "the module: bisect with `pytest --collect-only <subset>` and check "
        "`jax._src.xla_bridge.backends_are_initialized()` after each.")
