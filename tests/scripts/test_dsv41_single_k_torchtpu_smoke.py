# SPDX-License-Identifier: Apache-2.0
"""Host fixtures preserve the actual int64 metadata failure boundary."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_test
SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts/dsv41_bench/single_k_torchtpu_smoke.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("writer_gate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_import_does_not_initialize_torch_or_jax():
    code = """
import importlib.util
import sys
class Block:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('torch', 'torch_tpu', 'jax'):
            raise RuntimeError('runtime imported during collection: ' + fullname)
sys.meta_path.insert(0, Block())
spec = importlib.util.spec_from_file_location('gate', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert len(module.case_specs()) == 16
"""
    subprocess.run([sys.executable, "-c", code, str(SCRIPT)], check=True)


@pytest.mark.parametrize("n", [288, 2048])
@pytest.mark.parametrize("banks", [1, 2])
def test_fixture_preserves_int64_boundary_and_unique_last_record(n, banks):
    gate = _module()
    ids, values = gate.fixture(n, 4096, banks)
    assert ids.dtype == np.int64  # Actual failing Torch slots are long, not int32.
    assert len(ids) == n and values.shape == (n, 128)
    assert ids[-2] == banks * 4096 - 1 and ids[-1] == -1
    assert len(np.unique(ids[ids >= 0])) == n - 1
    assert {0, 1, 5, 6}.issubset(ids)
    if banks == 2:
        assert np.any(ids >= 4096)


@pytest.mark.parametrize("layout", ["ordinary", "halfplane"])
def test_sparse_oracle_preserves_sibling_and_invalid_slot(layout):
    gate = _module()
    ids = np.array([0, 3, -1], dtype=np.int64)
    bits = np.arange(3 * 128, dtype=np.uint32).reshape(3, 128)
    sample, expected = gate.expected_rows(ids, bits, layout, 8, 1)[0]
    expected = expected.view(np.uint32)
    for record, value in zip(ids[:2], bits[:2]):
        row = expected[np.flatnonzero(sample == record // 2)[0]]
        if layout == "ordinary":
            offset = (record % 2) * 64
            np.testing.assert_array_equal(
                row[offset : offset + 64], value[::2] | (value[1::2] << 16)
            )
            np.testing.assert_array_equal(
                row[64 - offset : 128 - offset], gate.SENTINEL
            )
        else:
            shift = (record % 2) * 16
            np.testing.assert_array_equal(
                (row >> shift) & 65535, np.r_[value[::2], value[1::2]]
            )
            np.testing.assert_array_equal(
                (row >> (16 - shift)) & 65535, gate.SENTINEL if shift else 0
            )
    np.testing.assert_array_equal(expected[sample == 2], gate.SENTINEL)


def test_receipt_validation_rejects_incomplete_or_forged_execution():
    import copy

    gate = _module()
    calls = [
        {"mode": mode, "iteration": iteration, "status": "passed"}
        for mode in ("eager", "compiled")
        for iteration in range(2)
    ]
    results = [
        {
            "rank": rank,
            "device_id": rank,
            "status": "passed",
            "cases": [
                {**case, "status": "passed", "executions": copy.deepcopy(calls)}
                for case in gate.case_specs()
            ],
        }
        for rank in range(8)
    ]
    assert gate.validate_results(results)
    mutations = [
        lambda data: data[0].update(device_id=1),
        lambda data: data[0].update(rank=1),
        lambda data: data[0]["cases"].pop(),
        lambda data: data[0]["cases"].__setitem__(
            0, copy.deepcopy(data[0]["cases"][1])
        ),
        lambda data: data[0]["cases"][0]["executions"].pop(),
        lambda data: data[0]["cases"][0]["executions"][0].update(mode="compiled"),
        lambda data: data[0]["cases"][0]["executions"][0].update(status="failed"),
        lambda data: data[0]["cases"][0].update(n=1),
    ]
    for mutate in mutations:
        invalid = copy.deepcopy(results)
        mutate(invalid)
        assert not gate.validate_results(invalid)
