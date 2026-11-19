# SPDX-License-Identifier: Apache-2.0
"""q38 L1-V10INT: the Mamba checkpoint retention patch line does not claim decode checkpoints are on under MTP.

`patch_mamba_checkpoint_retention` reads VLLM_TPU_MAMBA_DECODE_CHECKPOINT once, but `cache_blocks` publishes a decode
checkpoint only for a manager with ``num_speculative_blocks == 0`` (known per manager at run time; MTP k=3 serving has
3). The patch line therefore names the knob and the condition instead of saying "on". Log only: the decode-checkpoint
behaviour with and without spec blocks is pinned by tests/platforms/test_mamba_checkpoint_retention.py.
"""

import inspect
import logging

import pytest
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
from vllm.v1.core.single_type_kv_cache_manager import MambaManager

from vllm_torchtpu.core import mamba_checkpoint_retention

_PATCHED = (
    (MambaManager, "cache_blocks"),
    (MambaManager, "_remove_blocks_in_range"),
    (MambaManager, "remove_skipped_blocks"),
    (MambaManager, "pop_blocks_for_free"),
    (KVCacheCoordinator, "free"),
    (KVCacheCoordinator, "pop_blocks_for_free"),
    (BlockPool, "get_cached_block"),
    (BlockPool, "_insert_block_hash"),
)


def _patch_line(monkeypatch, knob: str) -> list[str]:
    for owner, name in _PATCHED:
        monkeypatch.setattr(owner, name, inspect.unwrap(getattr(owner, name)))
    monkeypatch.setenv("VLLM_TPU_MAMBA_CHECKPOINT_RETENTION", "1")
    monkeypatch.setenv("VLLM_TPU_MAMBA_DECODE_CHECKPOINT", knob)
    records = []

    class _Keep(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    log = mamba_checkpoint_retention.logger
    handler = _Keep(level=logging.INFO)
    old = log.level
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    try:
        mamba_checkpoint_retention.patch_mamba_checkpoint_retention()
    finally:
        log.removeHandler(handler)
        log.setLevel(old)
    lines = [m for m in records if m.startswith("Applied TPU patch: Mamba align checkpoints")]
    print("Q38 MCR patch line:", lines)
    return lines


def test_knob_on_names_the_spec_decode_condition(monkeypatch):
    lines = _patch_line(monkeypatch, "1")
    assert lines == [
        "Applied TPU patch: Mamba align checkpoints pinned until request free, freed after their KV "
        "(decode checkpoints: knob on, effective only without spec decode (num_speculative_blocks == 0; off under MTP))."
    ], lines
    assert "on (no spec decode)" not in lines[0]


@pytest.mark.parametrize("knob", ["0"])
def test_knob_off_says_off(monkeypatch, knob):
    lines = _patch_line(monkeypatch, knob)
    assert lines == [
        "Applied TPU patch: Mamba align checkpoints pinned until request free, freed after their KV (decode checkpoints: off)."
    ], lines
