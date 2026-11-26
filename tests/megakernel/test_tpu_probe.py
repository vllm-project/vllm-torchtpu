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
"""The `tpu_test` probe in `tests/megakernel/conftest.py`: it skips only when
JAX has no TPU backend, and waits for a busy TPU before failing."""

from __future__ import annotations

from typing import Any

import jax
import pytest
from jax.extend import backend as jax_backend

from tests.megakernel import conftest as mk_conftest

_BUSY = RuntimeError("Unable to initialize backend 'tpu': TPU is already in use")


def _patch_probe(
    monkeypatch: pytest.MonkeyPatch, outcomes: list[Any]
) -> dict[str, list[Any]]:
    """`jax.devices("tpu")` yields `outcomes` in turn (an exception is raised);
    the sleeps and backend clears are recorded, and the clock follows the
    sleeps."""
    seen: dict[str, list[Any]] = {"sleep": [], "clear": []}
    clock = [0.0]

    def devices(platform: str | None = None) -> list[Any]:
        out = outcomes.pop(0) if len(outcomes) > 1 else outcomes[0]
        if isinstance(out, BaseException):
            raise out
        return out

    def sleep(s: float) -> None:
        seen["sleep"].append(s)
        clock[0] += s

    monkeypatch.setattr(jax, "devices", devices)
    monkeypatch.setattr(mk_conftest.time, "sleep", sleep)
    monkeypatch.setattr(mk_conftest.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(jax_backend, "clear_backends", lambda: seen["clear"].append(1))
    return seen


def test_busy_tpu_is_retried_with_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    cpu = jax.local_devices(backend="cpu")[:1]
    seen = _patch_probe(monkeypatch, [_BUSY, _BUSY, cpu])
    assert mk_conftest._probe_tpu_devices() == cpu
    assert seen["sleep"] == [2.0, 4.0] and len(seen["clear"]) == 2


def test_tpu_busy_past_the_wait_fails_with_the_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _patch_probe(monkeypatch, [_BUSY])
    with pytest.raises(pytest.fail.Exception, match="still busy after 120s.*in use"):
        mk_conftest._probe_tpu_devices()
    assert sum(seen["sleep"]) == pytest.approx(mk_conftest._TPU_BUSY_WAIT_S)


def test_no_tpu_backend_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_probe(monkeypatch, [RuntimeError("Unknown backend: 'tpu' requested")])
    with pytest.raises(pytest.skip.Exception, match="no TPU backend"):
        mk_conftest._probe_tpu_devices()
    _patch_probe(monkeypatch, [[]])
    with pytest.raises(pytest.skip.Exception, match="no TPU devices"):
        mk_conftest._probe_tpu_devices()
