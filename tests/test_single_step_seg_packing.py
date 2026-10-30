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
"""Single-step prefill seg metadata: one packed tensor, fixed slices.

Regression for the v17ce gate garbage: the builders used to pass one cached
zero tensor as both `mamba_seg_bounds` and every group's
`mamba_seg_state_indices` on cut-free steps. torch-tpu's lazy tracing CSEs
equal device tensors and dynamo folds them into a single graph input, so on
steps with cuts the traced GDN program read the seg_bounds query offsets as
checkpoint block ids and overwrote other requests' KV pages. The invariant:
every seg field is a distinct fixed slice of ONE packed tensor, on both the
zero path and the cut path, so the traced structure never depends on values.
"""

import importlib.abc
import importlib.machinery
import importlib.util
import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """CPU hosts without torch_tpu (same stub as the kernel tests)."""

    def find_spec(self, name, path=None, target=None):
        if name == "torch_tpu" or name.startswith("torch_tpu."):
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
        return None

    def create_module(self, spec):
        m = mock.MagicMock(name=spec.name)
        m.__path__ = []
        m.__spec__ = spec
        return m

    def exec_module(self, module):
        pass


if importlib.util.find_spec("torch_tpu") is None:
    sys.meta_path.insert(0, _StubFinder())
    import torch._dynamo

    if "tpu" not in torch._dynamo.list_backends(exclude_tags=()):
        torch._dynamo.register_backend(name="tpu")(lambda gm, _: gm.forward)

from vllm_torchtpu.layers.core import attention_metadata as am  # noqa: E402

MAMBA_GIDS = [0, 1, 2]


class _Runner:
    device = torch.device("cpu")

    def _mamba_seg_gids(self):
        return MAMBA_GIDS

    def _mamba_seg_slot(self, gid):
        return MAMBA_GIDS.index(gid)


def _fields(runner, packed, n):
    ctx = SimpleNamespace(mamba_seg_packed=packed)
    return {gid: am._seg_fields(runner, ctx, gid, n) for gid in MAMBA_GIDS}


def _assert_distinct_slices(fields, base):
    seen = set()
    for gid, (bounds, idx) in fields.items():
        for t in (bounds, idx):
            assert t.untyped_storage().data_ptr() == base.untyped_storage().data_ptr()
        assert bounds.storage_offset() == 0
        assert idx.storage_offset() == (1 + MAMBA_GIDS.index(gid)) * base[0].numel()
        assert idx is not bounds
        seen.add(idx.storage_offset())
    # Every group reads its own row.
    assert len(seen) == len(MAMBA_GIDS)


def test_zero_path_is_packed_slices():
    runner = _Runner()
    fields = _fields(runner, None, 8)
    base = am._seg_zeros(runner, 8)
    assert base.shape == (1 + len(MAMBA_GIDS), 8, 2)
    assert base.dtype == torch.int32
    _assert_distinct_slices(fields, base)
    # Cached per num_reqs: the zero path never allocates per step.
    assert am._seg_zeros(runner, 8) is base


def test_cut_path_same_structure_and_values():
    runner = _Runner()
    n = 8
    packed = torch.zeros((1 + len(MAMBA_GIDS), n, 2), dtype=torch.int32)
    packed[0, 3] = torch.tensor([640, 1024])
    for k in range(len(MAMBA_GIDS)):
        packed[1 + k, 3] = torch.tensor([100 + k, 200 + k])
    fields = _fields(runner, packed, n)
    _assert_distinct_slices(fields, packed)
    for k, gid in enumerate(MAMBA_GIDS):
        bounds, idx = fields[gid]
        assert bounds[3].tolist() == [640, 1024]
        assert idx[3].tolist() == [100 + k, 200 + k]


def test_runner_packs_bounds_and_group_rows():
    """TPUModelRunner._build_mamba_seg_arrays: one upload, row 0 bounds, row
    1 + slot each group's ids (a group absent from the copy plan stays 0)."""
    try:
        from vllm_torchtpu.runner import tpu_runner
    except Exception:  # pragma: no cover - runner needs the full TPU stack
        import pytest

        pytest.skip("tpu_runner not importable on this host")
    BS = 640
    tables = {
        0: np.array([[0, 11, 12, 13, 14]], np.int32),
        1: np.array([[0, 21, 22, 23, 24]], np.int32),
    }
    runner = SimpleNamespace(
        device=torch.device("cpu"),
        _mamba_copy_plan=[(0, []), (1, [])],
        _mamba_state_block_size=BS,
        input_batch=SimpleNamespace(
            req_ids=["r"],
            num_computed_tokens_cpu=np.array([640]),
            block_table={
                g: SimpleNamespace(get_cpu_tensor=lambda t=t: torch.from_numpy(t))
                for g, t in tables.items()
            },
        ),
    )
    runner._mamba_seg_enabled = lambda: True
    runner._mamba_seg_gids = lambda: MAMBA_GIDS
    sched = SimpleNamespace(
        tpu_mamba_seg_cuts={"r": (1280, 1408)}, num_scheduled_tokens={"r": 900}
    )
    packed = tpu_runner.TPUModelRunner._build_mamba_seg_arrays(runner, sched, 0, 1, 2)
    assert tuple(packed.shape) == (4, 2, 2)
    # computed 640, end 1540: full cut 1280 -> col 1, partial 1408 -> col 3.
    assert packed[0, 0].tolist() == [640, 768]
    assert packed[1, 0].tolist() == [11, 13]
    assert packed[2, 0].tolist() == [21, 23]
    assert packed[3, 0].tolist() == [0, 0]
    assert packed[:, 1].abs().sum().item() == 0
