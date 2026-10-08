# SPDX-License-Identifier: Apache-2.0
"""Shared pytest configuration for the Omni tests.

tests_omni/ holds tests that need the real vllm-omni package, which only the
``-omni`` CI image installs. It sits outside tests/ so the general CI lanes,
which glob ``tests/**`` in the dev image, never collect it; only the Omni lanes
in .buildkite/pipeline_tests.yml run it.

tests/conftest.py does not apply here, so the environment defaults that
engine-starting tests rely on are repeated below. Keep them in sync with it.
"""

from __future__ import annotations

import os

os.environ.setdefault("TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS", "false")
os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
os.environ.setdefault("VLLM_DO_NOT_TRACK", "1")

from typing import NamedTuple

import vllm.v1.outputs

if not hasattr(vllm.v1.outputs, "RoutedExpertsLists"):

    class RoutedExpertsLists(NamedTuple):
        pass

    vllm.v1.outputs.RoutedExpertsLists = RoutedExpertsLists  # type: ignore[attr-defined]
# isort: split

import contextlib
import copy
import gc
import json
import threading
from collections.abc import Generator
from typing import Any

import psutil
import pytest

from tests.tpu_release import wait_for_tpu_release

omni_fixture_lock = threading.Lock()


def cleanup_test_environment() -> None:
    """Tear down distributed state and wait for TPU device nodes to be released."""
    from vllm.distributed import cleanup_dist_env_and_memory

    cleanup_dist_env_and_memory()
    gc.collect()
    wait_for_tpu_release()


def _parse_omni_runner_param(param: Any) -> tuple[str, str | None, dict[str, Any]]:
    if isinstance(param, str):
        model, stage_cfg, extra = param, None, {}
    elif isinstance(param, (tuple, list)) and 1 <= len(param) <= 3:
        model = param[0]
        stage_cfg = param[1] if len(param) >= 2 else None
        extra = dict(param[2] or {}) if len(param) == 3 else {}
    else:
        raise ValueError("Invalid omni_runner param; expected model or tuple")
    prefix = os.environ.get("MODEL_PREFIX", "").rstrip("/")
    return f"{prefix}/{model}" if prefix else str(model), stage_cfg, extra


def _omni_runner_cache_key(param: Any) -> tuple[str, str | None, str]:
    model, stage_cfg, extra = _parse_omni_runner_param(param)
    return (model, stage_cfg, json.dumps(extra, sort_keys=True, default=repr))


class OmniRunner:
    """Context-managed wrapper around :class:`vllm_omni.entrypoints.omni.Omni`. Adapted
    from :class:`vllm_omni.tests.helpers.runtime.OmniRunner`
    (https://github.com/vllm-project/vllm-omni/blob/main/tests/helpers/runtime.py)
    and vLLM's :class:`tests.conftest.VllmRunner`
    (https://github.com/vllm-project/vllm/blob/main/tests/conftest.py).
    """

    def __init__(self, model_name: str, seed: int = 42, **kwargs: Any) -> None:
        cleanup_test_environment()
        self.model_name = model_name
        self.seed = seed
        self.omni: Omni | None = None
        kwargs.setdefault("stage_init_timeout", 600)
        kwargs.setdefault("init_timeout", 1800)
        kwargs.setdefault("log_stats", False)
        kwargs.setdefault("seed", seed)

        try:
            from vllm_omni.entrypoints.omni import Omni

            self.omni = Omni(model=model_name, **kwargs)
        except BaseException:
            self.close()
            raise

    def get_default_sampling_params_list(self) -> list[Any]:
        assert self.omni is not None, "Omni was not initialized"
        return copy.deepcopy(list(self.omni.default_sampling_params_list))

    def generate(
        self, prompts: Any, sampling_params_list: Any = None, *args: Any, **kwargs: Any
    ) -> Generator[Any, None, None]:
        assert self.omni is not None, "Omni cannot generate before initializing"
        if sampling_params_list is None:
            sampling_params_list = self.get_default_sampling_params_list()
        return self.omni.generate(prompts, sampling_params_list, *args, **kwargs)

    def _cleanup_process(self) -> None:
        with contextlib.suppress(Exception):
            children = psutil.Process().children(recursive=True)
            for p in children:
                with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                    p.terminate()
            _, alive = psutil.wait_procs(children, timeout=5)
            for p in alive:
                with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                    p.kill()

    def __getattr__(self, name: str) -> Any:
        omni = self.__dict__.get("omni")
        if omni is not None:
            return getattr(omni, name)
        raise AttributeError(name)

    def __enter__(self) -> OmniRunner:
        return self

    def close(self, *_: Any) -> None:
        if self.omni is not None:
            self.omni.close()
            self.omni = None

        self._cleanup_process()
        cleanup_test_environment()

    shutdown = __exit__ = close


class _OmniRunnerManager:
    """Module-scoped cache ensuring each model's OmniRunner is spun up once."""

    def __init__(self, lock: threading.Lock) -> None:
        self._lock = lock
        self._active_key: tuple[str, str | None, str] | None = None
        self._active_runner: OmniRunner | None = None

    def get_or_create(self, param: Any) -> OmniRunner:
        key = _omni_runner_cache_key(param)
        model, stage_cfg, extra = _parse_omni_runner_param(param)
        with self._lock:
            omni = getattr(self._active_runner, "omni", None)
            if (
                self._active_key == key
                and omni is not None
                and not getattr(omni, "_shutdown_called", False)
                and bool(getattr(omni, "is_running", True))
            ):
                assert self._active_runner is not None
                return self._active_runner
            self.close()
            self._active_runner = OmniRunner(
                model, seed=42, deploy_config=stage_cfg, **extra
            )
            self._active_key = key
            return self._active_runner

    def close(self) -> None:
        if self._active_runner is not None:
            runner = self._active_runner
            self._active_runner = self._active_key = None
            runner.close()


@pytest.fixture(scope="module")
def _omni_runner_manager() -> Generator[_OmniRunnerManager, None, None]:
    manager = _OmniRunnerManager(omni_fixture_lock)
    try:
        yield manager
    finally:
        with omni_fixture_lock:
            manager.close()


@pytest.fixture(scope="module")
def omni_runner(
    request: pytest.FixtureRequest, _omni_runner_manager: _OmniRunnerManager
) -> Generator[OmniRunner, None, None]:
    """Module-scoped :class:`OmniRunner` that spins up each model only once."""
    yield _omni_runner_manager.get_or_create(request.param)


@pytest.fixture(scope="function")
def omni_runner_function(
    request: pytest.FixtureRequest,
) -> Generator[OmniRunner, None, None]:
    """Function-scoped :class:`OmniRunner` torn down after each test."""
    model, stage_cfg, extra = _parse_omni_runner_param(request.param)
    with (
        omni_fixture_lock,
        OmniRunner(model, seed=42, deploy_config=stage_cfg, **extra) as runner,
    ):
        yield runner


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Group ``omni_runner`` tests by model within each module to spin up once."""
    first_seen: dict[tuple[str, tuple[str, str | None, str]], int] = {}

    def _sort_key(idx_and_item: tuple[int, pytest.Item]) -> tuple[int, str, int]:
        idx, item = idx_and_item
        group = 0 if item.get_closest_marker("spawns_tpu_workers") else 1
        file_path = item.nodeid.split("::")[0]
        callspec = getattr(item, "callspec", None)
        if callspec is None or "omni_runner" not in callspec.params:
            return (group, file_path, idx)
        key = (file_path, _omni_runner_cache_key(callspec.params["omni_runner"]))
        return (group, file_path, first_seen.setdefault(key, idx))

    indexed = sorted(enumerate(items), key=_sort_key)
    items[:] = [item for _, item in indexed]
