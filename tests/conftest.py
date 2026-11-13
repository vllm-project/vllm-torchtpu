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
os.environ.setdefault("TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS", "false")
os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "0")

# Forking after TorchTPU/JAX/PyTorch background threads have started can
# deadlock vLLM engine or worker subprocess startup. Set this before test
# modules import vLLM so every LLM test uses a clean spawned process.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

# Disable vLLM background usage telemetry in tests to prevent background threads
# and IPC sockets from hanging multi-process worker teardown.
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
os.environ.setdefault("VLLM_DO_NOT_TRACK", "1")

from typing import NamedTuple

import pytest
import torch
import vllm.v1.outputs

if not hasattr(vllm.v1.outputs, "RoutedExpertsLists"):

    class RoutedExpertsLists(NamedTuple):
        pass

    vllm.v1.outputs.RoutedExpertsLists = RoutedExpertsLists  # type: ignore[attr-defined]


def pytest_collection_modifyitems(items):
    """Run whole-slice worker-spawning tests before in-process TPU tests.

    Once the pytest process itself initializes the TPU runtime (any test that
    runs jax/torch_tpu in-process), a test that later spawns a fresh
    ``torch.distributed.run`` worker group can no longer bootstrap the slice:
    the workers fail with "PjRtClient is not initialized" and libtpu may tear
    down the whole session with SLICE_FAILURE_SW_INJECT_ERROR. libtpu exposes
    no release API, so the only safe ordering is spawn-first.

    Within each group, files run in path order, whatever order they were
    passed in. Tests that start an engine in a subprocess (tests/entrypoints)
    need the TPU free, so they must run before the in-process tests under
    tests/kernels and tests/layers, and a sharded run receives its files in
    the planner's order. The sort is stable, so tests within a file keep
    their order.
    """
    items.sort(
        key=lambda item: (
            0 if item.get_closest_marker("spawns_tpu_workers") else 1,
            item.nodeid.split("::")[0],
        )
    )


def pytest_addoption(parser):
    """Add --use-tpu command line option."""
    parser.addoption(
        "--use-tpu",
        action="store_true",
        default=True,
        help="Run tests on TPU device instead of CPU",
    )


@pytest.fixture(autouse=True)
def _reset_warning_once():
    """`warning_once` dedups by message for the life of the process, so without
    this a test passes or fails depending on what ran before it."""
    try:
        import vllm.logger
    except ImportError:
        yield
        return

    vllm.logger._print_warning_once.cache_clear()
    yield
    vllm.logger._print_warning_once.cache_clear()


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: multi-rank or large-batch CPU interpret tests"
    )
    config.addinivalue_line("markers", "golden: bit-exact golden hash regression tests")


_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
_CPU_JAX_GROUP_RSS_LIMIT_MIB = int(
    os.environ.get("VLLM_TPU_CPU_TEST_GROUP_RSS_MIB", "2400")
)
_CPU_JAX_HARD_RSS_LIMIT_MIB = int(
    os.environ.get("VLLM_TPU_CPU_TEST_HARD_RSS_MIB", "3600")
)
_MALLOC_TRIM_FN = None
_MALLOC_TRIM_RESOLVED = False


def _glibc_malloc_trim() -> None:
    global _MALLOC_TRIM_FN, _MALLOC_TRIM_RESOLVED
    if not _MALLOC_TRIM_RESOLVED:
        _MALLOC_TRIM_RESOLVED = True
        try:
            import ctypes

            libc = ctypes.CDLL("libc.so.6")
            fn = getattr(libc, "malloc_trim", None)
            if fn is not None:
                fn.argtypes = [ctypes.c_size_t]
                fn.restype = ctypes.c_int
                _MALLOC_TRIM_FN = fn
        except OSError:
            _MALLOC_TRIM_FN = None
    if _MALLOC_TRIM_FN is not None:
        _MALLOC_TRIM_FN(0)


def _current_rss_mib() -> float:
    try:
        with open("/proc/self/statm", encoding="ascii") as statm:
            parts = statm.read().split()
        return int(parts[1]) * _PAGE_SIZE / (1024 * 1024)
    except (OSError, IndexError, ValueError):
        return 0.0


def _reclaim_cpu_jax_memory() -> None:
    import gc

    from tests.distributed.tpu_test_utils import has_tpu_vfio_devices

    is_cpu = not has_tpu_vfio_devices()
    if is_cpu and "jax" in sys.modules:
        from jax._src import xla_bridge

        if xla_bridge.backends_are_initialized():
            sys.modules["jax"].clear_caches()
    gc.collect()
    if is_cpu:
        _glibc_malloc_trim()


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item, nextitem):
    """Bound XLA:CPU / LLVM MCJIT arena growth inside large parametrized modules."""
    from tests.distributed.tpu_test_utils import has_tpu_vfio_devices

    if has_tpu_vfio_devices() or "jax" not in sys.modules:
        return
    rss_mib = _current_rss_mib()
    cur_fn = getattr(item, "originalname", item.name)
    nxt_fn = (
        getattr(nextitem, "originalname", nextitem.name)
        if nextitem is not None
        else None
    )
    group_done = cur_fn != nxt_fn
    if (
        group_done and rss_mib >= _CPU_JAX_GROUP_RSS_LIMIT_MIB
    ) or rss_mib >= _CPU_JAX_HARD_RSS_LIMIT_MIB:
        _reclaim_cpu_jax_memory()


@pytest.fixture(scope="module", autouse=True)
def _module_host_and_cpu_jax_gc():
    """Reclaim host allocations (and CPU JAX executables) at module boundaries."""
    yield
    _reclaim_cpu_jax_memory()


@pytest.fixture
def vllm_config_context():
    """Run backend helpers with the same resolved layout as worker startup."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

    config = VllmConfig()
    resolve_kv_cache_layout(config, [["LBNHC", "LBHNC"]])
    with set_current_vllm_config(config):
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
        "`jax._src.xla_bridge.backends_are_initialized()` after each."
    )
