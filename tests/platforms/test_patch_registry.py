# SPDX-License-Identifier: Apache-2.0
"""Checks staged patch execution without importing the TPU runtime."""

import contextlib
import importlib.util
import logging
import multiprocessing
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

_REGISTRY_PATH = (
    Path(__file__).resolve().parents[2] / "src/vllm_torchtpu/patch_registry.py"
)


def _load_registry():
    name = "_test_tpu_patch_registry"
    spec = importlib.util.spec_from_file_location(name, _REGISTRY_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def registry(monkeypatch):
    module = _load_registry()
    callbacks = ModuleType("_test_patch_callbacks")
    monkeypatch.setitem(sys.modules, callbacks.__name__, callbacks)
    module.PATCHES = ()
    yield module, callbacks
    sys.modules.pop(module.__name__, None)


def _patch(registry, callbacks, name, stages, **kwargs):
    patch = registry.Patch(f"{callbacks.__name__}:{name}", stages, **kwargs)
    registry.PATCHES += (patch,)
    return patch


@contextlib.contextmanager
def captured_records(logger):
    """Collect the INFO records emitted on `logger`.

    Deliberately not pytest's `caplog`: vLLM's dictConfig sets
    `propagate=False` on the `vllm` logger, so records from its children never
    reach the root logger where `caplog` installs its handler. Any test that
    imports vLLM first leaves this one capturing nothing.
    """
    records: list[logging.LogRecord] = []
    handler = logging.Handler(logging.INFO)
    handler.emit = records.append
    old_level = logger.level
    # Another test may have raised the level or called logging.disable();
    # either drops the record before any handler runs.
    old_disable = logging.root.manager.disable
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logging.disable(logging.NOTSET)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
        logging.disable(old_disable)


def test_order_and_idempotence_across_stages(registry):
    reg, callbacks = registry
    seen = []
    callbacks.first = lambda: seen.append("first")
    callbacks.second = lambda: seen.append("second")
    callbacks.worker = lambda: seen.append("worker")
    _patch(reg, callbacks, "first", ("platform_activation", "worker_init"))
    _patch(reg, callbacks, "second", ("platform_activation", "engine_core"))
    _patch(reg, callbacks, "worker", ("worker_init",))
    with captured_records(reg._logger) as records:
        reg.apply("platform_activation")
        reg.apply("engine_core")
        reg.apply("worker_init")
        reg.apply("worker_init")
    assert seen == ["first", "second", "worker"]
    assert len(records) == 2


def test_failed_patch_retries_without_repeating_completed_patches(registry):
    reg, callbacks = registry
    callbacks.first = Mock()
    callbacks.failing = Mock(side_effect=[RuntimeError("unavailable"), None])
    callbacks.last = Mock()
    for name in ("first", "failing", "last"):
        _patch(reg, callbacks, name, ("worker_init",))
    with pytest.raises(RuntimeError, match="unavailable"):
        reg.apply("worker_init")
    callbacks.last.assert_not_called()
    reg.apply("worker_init")
    callbacks.first.assert_called_once_with()
    assert callbacks.failing.call_count == 2
    callbacks.last.assert_called_once_with()


def test_nonmatching_model_does_not_consume_patch(registry):
    reg, callbacks = registry
    callbacks.model = Mock(side_effect=lambda model_config: model_config.enabled)
    patch = _patch(
        reg,
        callbacks,
        "model",
        ("platform_activation", "model_load"),
        model_config=True,
    )
    reg.apply("platform_activation", model_config=SimpleNamespace(enabled=False))
    assert patch.target not in reg._applied
    config = SimpleNamespace(enabled=True)
    reg.apply("model_load", model_config=config)
    reg.apply("model_load", model_config=config)
    assert callbacks.model.call_count == 2
    callbacks.model.assert_called_with(model_config=config)


def test_refreshes_late_bindings_without_reinstalling(registry):
    reg, callbacks = registry
    callbacks.install = Mock()
    bindings = {}
    callbacks.refresh = lambda: bindings.update(late="patched")
    _patch(
        reg,
        callbacks,
        "install",
        ("platform_activation", "engine_core"),
        refresh=f"{callbacks.__name__}:refresh",
    )
    reg.apply("platform_activation")
    bindings["late"] = "original"
    reg.apply("engine_core")
    callbacks.install.assert_called_once_with()
    assert bindings["late"] == "patched"


def test_refresh_failure_does_not_reinstall_wrapper(registry):
    reg, callbacks = registry
    callbacks.install = Mock()
    callbacks.refresh = Mock(side_effect=[RuntimeError("refresh"), None])
    _patch(
        reg,
        callbacks,
        "install",
        ("engine_core",),
        refresh=f"{callbacks.__name__}:refresh",
    )
    reg.apply("engine_core")
    with pytest.raises(RuntimeError, match="refresh"):
        reg.apply("engine_core")
    reg.apply("engine_core")
    callbacks.install.assert_called_once_with()
    assert callbacks.refresh.call_count == 2


def test_import_reentry_preserves_order(registry):
    reg, callbacks = registry
    seen = []

    def first():
        seen.append("start")
        reg.apply("platform_activation")
        seen.append("end")

    callbacks.first = first
    callbacks.second = lambda: seen.append("second")
    _patch(reg, callbacks, "first", ("platform_activation",))
    _patch(reg, callbacks, "second", ("platform_activation",))
    reg.apply("platform_activation")
    assert seen == ["start", "end", "second"]


def test_concurrent_application_installs_once(registry):
    reg, callbacks = registry
    callbacks.install = Mock()
    _patch(reg, callbacks, "install", ("worker_init",))
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(reg.apply, ["worker_init"] * 16))
    callbacks.install.assert_called_once_with()


def test_invalid_stage_is_rejected(registry):
    reg, _ = registry
    with pytest.raises(ValueError, match="Unknown TPU patch stage"):
        reg.apply("worker")


def test_manifest_preserves_lifecycle_dependencies():
    reg = _load_registry()
    targets = [p.target for p in reg.PATCHES]
    assert len(targets) == len(set(targets))
    stages = {
        stage: [p.target.split(":")[-1] for p in reg.PATCHES if stage in p.stages]
        for stage in (
            "import",
            "platform_activation",
            "engine_core",
            "worker_init",
            "model_load",
        )
    }
    assert stages["import"] == ["_patch_jax_pallas_fori_lowering"]
    assert stages["engine_core"] == [
        "_patch_vllm_hybrid_pcp_block_sizes",
        "_patch_vllm_mamba_split_scheduler_block_size",
        "_patch_vllm_offloading_config_build",
        "_patch_vllm_hybrid_kv_load_failure_recovery",
        "_patch_vllm_same_step_prefix_hits",
        "_patch_vllm_merge_multimodal_embeddings",
    ]
    assert stages["platform_activation"] == (
        stages["worker_init"] + stages["model_load"]
    )
    assert stages["model_load"] == ["maybe_patch_qwen2_5_vl", "maybe_patch_qwen3_vl"]
    assert "patch_moe_expert_write_staging" in stages["worker_init"]
    for names in (stages["platform_activation"], stages["worker_init"]):
        assert names.index("_patch_vllm_force_v1_runner_tpu") < names.index(
            "_patch_vllm_config_triton_tpu"
        )


def test_plugin_import_sets_environment_before_jax():
    import os
    import subprocess

    source = _REGISTRY_PATH.parent.parent
    env = dict(os.environ, PYTHONPATH=str(source), JAX_PLATFORMS="tpu")
    env.pop("TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS", None)
    code = """
import builtins
import os
import sys
original_import = builtins.__import__
seen = []
def checked_import(name, *args, **kwargs):
    if name == "jax._src.pallas.mosaic":
        assert os.environ["TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS"] == "false"
        assert os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] == "0"
        assert "torch_tpu" not in sys.modules
        seen.append(name)
        raise ImportError("JAX unavailable")
    return original_import(name, *args, **kwargs)
builtins.__import__ = checked_import
import vllm_torchtpu
assert seen == ["jax._src.pallas.mosaic"]
assert "vllm" not in sys.modules
assert "torch_tpu" not in sys.modules
"""  # noqa: E501
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_fork_during_application_retains_completed_patches(registry):
    import os
    from threading import Event, Thread

    reg, callbacks = registry
    started = Event()
    release = Event()
    parent_pid = os.getpid()
    seen = []
    callbacks.completed = lambda: seen.append("completed")

    def pending():
        if os.getpid() == parent_pid:
            started.set()
            assert release.wait(10)
        seen.append("pending")

    callbacks.pending = pending
    _patch(reg, callbacks, "completed", ("worker_init",))
    _patch(reg, callbacks, "pending", ("worker_init",))
    context = multiprocessing.get_context("fork")
    parent, child = context.Pipe(duplex=False)

    def in_child():
        reg.apply("worker_init")
        reg.apply("worker_init")
        child.send(seen)
        child.close()

    thread = Thread(target=reg.apply, args=("worker_init",))
    process = context.Process(target=in_child)
    thread.start()
    try:
        assert started.wait(5)
        process.start()
        child.close()
        release.set()
        assert parent.poll(5), "Fork child is blocked on registry state"
        assert parent.recv() == ["completed", "pending"]
        process.join(5)
        assert process.exitcode == 0
    finally:
        release.set()
        thread.join(5)
        if process.pid is not None and process.is_alive():
            process.terminate()
            process.join(5)
        parent.close()
        child.close()
    assert seen == ["completed", "pending"]
