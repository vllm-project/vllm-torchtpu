# SPDX-License-Identifier: Apache-2.0
"""R5 (D-mid) on top of TPU_RUNNER_STAGE_TIMING_MODE: the default (auto) is off
under P/D; TPU_RUNNER_STAGE_TIMING=1 restores the KV-connector-driven default;
on / off still force it."""

import pytest

from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


@pytest.mark.parametrize(
    "mode,legacy,kv_uses,want",
    [
        (None, None, True, False),  # R5: off by default under P/D
        (None, None, False, False),
        ("auto", None, True, False),
        (None, "1", True, True),  # opt back in to the old auto behaviour
        (None, "true", True, True),
        (None, "1", False, False),  # no stats sink, nothing to feed
        ("on", None, False, True),
        ("1", None, False, True),
        ("off", "1", True, False),
        ("0", None, True, False),
    ],
)
def test_stage_timing_enabled(monkeypatch, mode, legacy, kv_uses, want):
    for k, v in (("TPU_RUNNER_STAGE_TIMING_MODE", mode), ("TPU_RUNNER_STAGE_TIMING", legacy)):
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)
    enabled, _ = TPUModelRunner._stage_timing_enabled(kv_uses)
    assert enabled is want
