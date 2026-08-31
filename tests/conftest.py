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


def pytest_addoption(parser):
    """Add --use-tpu command line option."""
    parser.addoption(
        "--use-tpu",
        action="store_true",
        default=True,
        help="Run tests on TPU device instead of CPU",
    )


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
