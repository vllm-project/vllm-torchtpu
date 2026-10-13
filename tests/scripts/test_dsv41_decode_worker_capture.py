# SPDX-License-Identifier: Apache-2.0
"""Capture decode draft shapes without consuming device output."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts/dsv41_bench/server_steps"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from server_step_worker import DECODE_SCENARIO, Recorder  # noqa: E402

pytestmark = pytest.mark.cpu_test


def test_decode_captures_scheduled_drafts_and_native_prepare_shapes():
    output = (SimpleNamespace(shape=(128, 32)), SimpleNamespace(shape=(128,)))
    calls = []

    def prepare(chunk):
        calls.append(chunk.num_reqs)
        return output

    runner = SimpleNamespace(
        execute_model=lambda scheduler: None,
        sample_tokens=lambda: None,
        drafter=SimpleNamespace(_prepare_dflash_inputs=prepare, block_size=5),
    )
    recorder = Recorder(runner, DECODE_SCENARIO, 0)
    recorder.enabled = True
    recorder.execute_model(
        SimpleNamespace(scheduled_spec_decode_tokens={"r": [1, 2, 3, 4, 5]})
    )
    result = runner.drafter._prepare_dflash_inputs(SimpleNamespace(num_reqs=16))
    assert result is output
    assert calls == [16]
    assert recorder.pending["scheduled_draft_lengths"] == {"r": 5}
    stage = recorder.pending["stages"][0]
    assert (stage["requests"], stage["rows"], stage["padded_rows"]) == (16, 80, 128)
    recorder.enabled = False
    runner.drafter._prepare_dflash_inputs(SimpleNamespace(num_reqs=16))
    assert len(recorder.stages) == 1
