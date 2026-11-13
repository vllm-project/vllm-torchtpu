# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from vllm_torchtpu.worker import kv_budget_pin as pin

GIB = 1 << 30


@pytest.mark.parametrize(
    ("measured", "recorded", "tol", "expected"),
    [
        (100 * GIB, None, 0.01, 100 * GIB),  # nothing recorded yet
        (100 * GIB, 99.5 * GIB, 0.01, 99.5 * GIB),  # fits, within tolerance
        (100 * GIB, 100 * GIB, 0.01, 100 * GIB),  # identical
        (100 * GIB, 100.2 * GIB, 0.01, 100 * GIB),  # does not fit: measured
        (100 * GIB, 98 * GIB, 0.01, 100 * GIB),  # too far below: measured
        (100 * GIB, 99.5 * GIB, 0.0, 100 * GIB),  # disabled
    ],
)
def test_choose_budget(measured, recorded, tol, expected):
    rec = None if recorded is None else int(recorded)
    assert pin.choose_budget(int(measured), rec, tol) == int(expected)


def test_pin_roundtrip_stabilizes_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(pin, "_pin_dir", lambda: tmp_path)
    monkeypatch.setattr(pin, "pin_key", lambda cfg: "k1")
    monkeypatch.setattr(pin.envs, "VLLM_TPU_KV_BUDGET_PIN_TOLERANCE", 0.01, raising=False)
    cfg = SimpleNamespace()
    first = pin.pin_kv_budget(cfg, 100 * GIB, write=True)
    assert first == 100 * GIB
    # Next start measures a bit more: the recorded budget is reused.
    assert pin.pin_kv_budget(cfg, 100 * GIB + 170 * (1 << 20), write=True) == first
    # A rank that does not write still reads the same pin.
    assert pin.pin_kv_budget(cfg, 100 * GIB + 1, write=False) == first
    # Less HBM than recorded: the measured value wins and is recorded.
    lower = 100 * GIB - 1
    assert pin.pin_kv_budget(cfg, lower, write=True) == lower
    assert pin.pin_kv_budget(cfg, 100 * GIB, write=False) == lower


def test_pin_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(pin, "_pin_dir", lambda: tmp_path)
    monkeypatch.setattr(pin, "pin_key", lambda cfg: "k1")
    monkeypatch.setattr(pin.envs, "VLLM_TPU_KV_BUDGET_PIN_TOLERANCE", 0.0, raising=False)
    assert pin.pin_kv_budget(SimpleNamespace(), 5 * GIB, write=True) == 5 * GIB
    assert not list(tmp_path.iterdir())
