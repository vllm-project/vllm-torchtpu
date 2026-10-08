# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the OmniRunner test engine fixture."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from . import conftest as omni_conftest
from .conftest import OmniRunner, _OmniRunnerManager, omni_fixture_lock

pytestmark = pytest.mark.omni
_EVENTS: list[tuple[str, str]] = []


class _FakeOmni:
    def __init__(self, model: str, **kwargs: Any) -> None:
        self.model, self.kwargs = model, kwargs
        self.is_running, self._shutdown_called = True, False
        self.default_sampling_params_list = [{"steps": 2}]
        _EVENTS.append(("init", model))

    def generate(self, prompts: Any, sampling_params_list: Any = None) -> list[dict]:
        return [{"model": self.model, "prompts": prompts, "sp": sampling_params_list}]

    def close(self) -> None:
        if not self._shutdown_called:
            self._shutdown_called, self.is_running = True, False
            _EVENTS.append(("close", self.model))


@pytest.fixture(scope="module", autouse=True)
def _patch_omni_env():
    mp = pytest.MonkeyPatch()
    for mod in ("vllm_omni", "vllm_omni.entrypoints", "vllm_omni.entrypoints.omni"):
        mp.setitem(sys.modules, mod, types.ModuleType(mod))
    sys_omni = sys.modules["vllm_omni.entrypoints.omni"]
    mp.setattr(sys_omni, "Omni", _FakeOmni, raising=False)
    mp.setattr(omni_conftest, "cleanup_test_environment", lambda: None)
    _EVENTS.clear()
    yield
    mp.undo()


@pytest.mark.parametrize(
    "omni_runner,task",
    [
        ("model-a", "t2v"),
        ("model-b", "t2v"),
        ("model-a", "t2v_euler"),
        ("model-b", "i2v"),
        ("model-a", "t2v_dmd2"),
    ],
    indirect=["omni_runner"],
)
def test_omni_runner_spins_up_once_per_model(
    omni_runner: OmniRunner, task: str
) -> None:
    out = omni_runner.generate(f"prompt-{task}")
    assert out[0] == {
        "model": omni_runner.model_name,
        "prompts": f"prompt-{task}",
        "sp": [{"steps": 2}],
    }
    assert [m for act, m in _EVENTS if act == "init"].count(omni_runner.model_name) == 1


def test_omni_runner_lifecycle_and_error_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _OmniRunnerManager(omni_fixture_lock)
    r1 = manager.get_or_create(("model-c", None, {"init_timeout": 600}))
    assert manager.get_or_create(("model-c", None, {"init_timeout": 600})) is r1
    r1.omni.close()
    r2 = manager.get_or_create(("model-c", None, {"init_timeout": 600}))
    assert r2 is not r1
    manager.close()
    assert r2.omni is None

    cleaned: list[bool] = []
    monkeypatch.setattr(
        omni_conftest, "cleanup_test_environment", lambda: cleaned.append(True)
    )
    monkeypatch.setattr(
        sys.modules["vllm_omni.entrypoints.omni"],
        "Omni",
        lambda **_: (_ for _ in ()).throw(RuntimeError("init failed")),
    )
    with pytest.raises(RuntimeError, match="init failed"):
        OmniRunner("broken-model")
    assert len(cleaned) == 2


def test_omni_runner_seed_configuration() -> None:
    # Default to no seed
    r_default = OmniRunner("model-no-seed")
    assert "seed" not in r_default.omni.kwargs

    # Configured seed directly on OmniRunner
    r_seeded = OmniRunner("model-seeded", seed=42)
    assert r_seeded.omni.kwargs["seed"] == 42

    # Configured via extra kwargs without TypeError
    manager = _OmniRunnerManager(omni_fixture_lock)
    r_extra = manager.get_or_create(("model-extra-seed", None, {"seed": 100}))
    assert r_extra.omni.kwargs["seed"] == 100
    manager.close()
