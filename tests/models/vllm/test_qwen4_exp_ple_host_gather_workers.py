# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_PLE_HOST_GATHER_WORKERS: the threaded host gather is exact."""

import types

import numpy as np
import pytest

from vllm_torchtpu.models.vllm.qwen4_exp.ple_storage import HostPLETable


@pytest.mark.parametrize("workers,rows", [("0", 40000), ("4", 40000), ("4", 100)])
def test_threaded_take_matches_serial(monkeypatch, workers, rows):
    monkeypatch.setenv("VLLM_TPU_PLE_HOST_GATHER_WORKERS", workers)
    rng = np.random.default_rng(0)
    table = rng.integers(0, 255, size=(5000, 48), dtype=np.uint8)
    ids = rng.integers(-10, 5010, size=rows)
    owner = types.SimpleNamespace()
    out = np.empty((rows, 48), np.uint8)
    HostPLETable._take_rows(owner, table, ids, out)
    np.testing.assert_array_equal(out, np.take(table, ids, axis=0, mode="clip"))
    pool = owner.__dict__.get("_gather_pool")
    assert (pool is not None) == (workers == "4" and rows >= 2 << 14)
    if pool is not None:
        pool.shutdown()
