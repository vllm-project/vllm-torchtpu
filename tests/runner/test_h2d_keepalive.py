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
"""Host-only tests of the runner's uploads from temporaries: none is uploaded
non_blocking outside the staged path (the spec-decode metadata's uploads,
whose index tensors later steps reuse, are staged too), and the runner's
other cached uploads are synchronous, so no cached tensor can read a freed
temporary. The staged path's keep-alive until the input-staging fence is
tested in tests/layers/core/test_attention_metadata.py.

The runner helpers under test are extracted from the source with ``ast`` and
run against a fake torch; the spec-decode metadata test runs the real
manager on CPU.
"""

import ast
import pathlib
import re
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_torchtpu.runner.speculative_decoding_manager import (
    SpeculativeDecodingManager,
)

SRC = (
    pathlib.Path(__file__).resolve().parents[2]
    / "src/vllm_torchtpu/runner/tpu_runner.py"
)


def test_no_non_blocking_upload_of_a_temporary_remains():
    root = SRC.parents[1]
    src = "".join(
        (root / f).read_text()
        for f in (
            "runner/tpu_runner.py",
            "runner/speculative_decoding_manager.py",
            "layers/core/attention_metadata.py",
        )
    )
    # Any device expression: self.device, device, self.runner.device, ...
    pat = re.compile(
        r"torch\.(?:tensor|from_numpy)\((?:[^()]|\((?:[^()]|\([^()]*\))*\))*\)"
        r"\s*\.to\(\s*[^,()]+,\s*non_blocking=True",
        re.S,
    )
    assert not pat.findall(src)
    assert "- 1\n                ).to(self.device, non_blocking=True)" not in src
    assert (
        "reshape(-1).to(\n                runner.device, non_blocking=True" not in src
    )
    assert "unsqueeze(1).to(logits.device, non_blocking=True)" not in src


def test_cached_subtract_indices_upload_synchronously():
    src = SRC.read_text()
    i = src.index("            sub_cache = self._async_subtract_idx_cache")
    end_marker = "sub_cache[sub_key] = (seq_idx_dev, pos_idx_dev, qx_sub_mask)"
    end = src.index(end_marker, i) + len(end_marker)
    block = src[i:end]
    assert "non_blocking=" not in block
    assert "stage_h2d_upload(" not in block
    assert "sub_cache.clear()" not in block
    assert "while len(sub_cache) >= 256:" in block
    assert "sub_cache.pop(next(iter(sub_cache)))" in block
    # Two base tensors, each uploaded synchronously on a miss.
    assert "seq_idx_dev = torch.from_numpy(seq_idx_np).to(self.device)" in block
    assert "pos_idx_dev = torch.from_numpy(pos_idx_np).to(self.device)" in block
    assert block.count(".to(self.device)") == 2

    uploads = []

    class _FakeDevTensor:
        def __init__(self, arr):
            self.arr = np.asarray(arr).copy()

        def __getitem__(self, slc):
            return _FakeDevTensor(self.arr[slc])

    class _FakeHostTensor:
        def __init__(self, arr):
            self.arr = np.asarray(arr)

        def to(self, device, non_blocking=False):
            uploads.append((device, non_blocking, self.arr.shape))
            return _FakeDevTensor(self.arr)

    code = textwrap.dedent(block)
    runner = SimpleNamespace(device="tpu", _async_subtract_idx_cache={})
    for k in range(260):
        sub_key = (4, 8, ((k, 2), (None, 3), (1, 1)))
        ns = {
            "self": runner,
            "sub_key": sub_key,
            "sub_pairs": sub_key[2],
            "num_reqs": len(sub_key[2]),
            "target_num_reqs": 4,
            "padded_total_num_scheduled_tokens": 8,
            "np": np,
            "torch": SimpleNamespace(from_numpy=_FakeHostTensor),
        }
        exec(code, ns)  # noqa: S102
        if k == 0:
            assert ns["seq_idx_dev"].arr.tolist() == [0, -1, 1, -1]
            assert ns["pos_idx_dev"].arr.tolist() == [0, 0, -1, -1, -1, 1, -1, -1]

    # 260 distinct keys: FIFO at 256 keeps the 256 most recent (never .clear()),
    # and each miss did 2 synchronous uploads.
    assert len(runner._async_subtract_idx_cache) == 256
    assert (4, 8, ((0, 2), (None, 3), (1, 1))) not in runner._async_subtract_idx_cache
    assert (4, 8, ((3, 2), (None, 3), (1, 1))) not in runner._async_subtract_idx_cache
    assert (4, 8, ((4, 2), (None, 3), (1, 1))) in runner._async_subtract_idx_cache
    assert (4, 8, ((259, 2), (None, 3), (1, 1))) in runner._async_subtract_idx_cache
    assert len(uploads) == 2 * 260 and all(not nb for _, nb, _ in uploads)


def test_cached_subst_indices_fifo_256_and_synchronous():
    src = SRC.read_text()
    i = src.index("subst_cache = self._async_subst_dev_cache")
    block = src[i : src.index("subst_cache[subst_key] = substitution_indices", i)]
    assert "non_blocking=" not in block
    assert "stage_h2d_upload(" not in block
    assert "while len(subst_cache) >= 256:" in block
    assert block.count(".to(self.device)") == 1


def test_uniform_and_heterogeneous_sampling_params_cache():
    """_build_padded_sampling_params reuses one uniform cache entry across
    varying num_active_reqs and uploads synchronously (never stage_h2d_upload)."""
    tree = ast.parse(SRC.read_text())
    fn_node = next(
        item
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TPUModelRunner"
        for item in node.body
        if isinstance(item, ast.FunctionDef)
        and item.name == "_build_padded_sampling_params"
    )
    fn_src = textwrap.dedent(ast.get_source_segment(SRC.read_text(), fn_node))
    assert "stage_h2d_upload(" not in fn_src
    assert "non_blocking=" not in fn_src

    uploads = []

    class _FakeTensor:
        def __init__(self, arr, dtype="bf16", device_type="cpu"):
            self._arr = np.asarray(arr)
            self.dtype = dtype
            self.device = SimpleNamespace(type=device_type)
            self.shape = self._arr.shape

        def __getitem__(self, idx):
            return _FakeTensor(
                self._arr[idx], dtype=self.dtype, device_type=self.device.type
            )

        def numpy(self):
            return self._arr

        def copy_(self, other):
            self._arr[:] = other._arr
            return self

        def unsqueeze(self, dim):
            return _FakeTensor(
                np.expand_dims(self._arr, dim),
                dtype=self.dtype,
                device_type=self.device.type,
            )

        def to(self, device, non_blocking=False):
            uploads.append((self.dtype, non_blocking))
            dev_type = getattr(device, "type", str(device))
            return _FakeTensor(self._arr.copy(), dtype=self.dtype, device_type=dev_type)

        def __ge__(self, val):
            return self._arr >= val

    fake_torch = SimpleNamespace(
        Tensor=_FakeTensor,
        int32="int32",
        float32="float32",
        ones=lambda n, dtype="float32": _FakeTensor(np.ones(n), dtype=dtype),
        zeros=lambda n, dtype="int32": _FakeTensor(
            np.zeros(n, dtype=np.int32), dtype=dtype
        ),
        full=lambda shape, val, dtype="float32": _FakeTensor(
            np.full(shape, val), dtype=dtype
        ),
        zeros_like=lambda t: _FakeTensor(np.zeros_like(t._arr), dtype=t.dtype),
        where=lambda cond, a, b: _FakeTensor(
            np.where(cond, a._arr, b._arr), dtype=b.dtype
        ),
    )
    ns = {"torch": fake_torch, "np": np}
    exec(fn_src, ns)  # noqa: S102
    build = ns["_build_padded_sampling_params"]

    runner = SimpleNamespace(
        vocab_size=151936,
        _sampling_params_dev_cache={},
        input_batch=SimpleNamespace(
            temperature_cpu_tensor=_FakeTensor(np.full(32, 0.6, dtype=np.float32)),
            top_k_cpu_tensor=_FakeTensor(
                np.full(32, 20, dtype=np.int32), dtype="int32"
            ),
            top_p_cpu_tensor=_FakeTensor(np.full(32, 0.95, dtype=np.float32)),
        ),
    )
    logits = _FakeTensor(np.zeros((32, 8)), dtype="bf16", device_type="tpu")

    # Vary num_active_reqs (32 -> 17 -> 1) and start/end offsets: all hit the
    # same single uniform cache entry after the first call.
    out_32 = build(runner, 0, 32, logits, padded_num_reqs=32)
    assert len(uploads) == 3 and all(not nb for _, nb in uploads)
    out_17 = build(runner, 0, 17, logits, padded_num_reqs=32)
    out_slice = build(runner, 5, 13, logits, padded_num_reqs=32)
    out_1 = build(runner, 0, 1, logits, padded_num_reqs=32)
    assert out_17 is out_32 and out_slice is out_32 and out_1 is out_32
    assert len(uploads) == 3
    assert len(runner._sampling_params_dev_cache) == 1
    for n in (1, 8, 17, 32):
        assert np.allclose(out_32[0]._arr[:n, 0], 0.6)
        assert (out_32[1]._arr[:n, 0] == 20).all()
        assert np.allclose(out_32[2]._arr[:n, 0], 0.95)

    # Heterogeneous batch falls back to per-slice key with neutral tail padding.
    runner.input_batch.temperature_cpu_tensor._arr[1] = 0.8
    out_het = build(runner, 0, 4, logits, padded_num_reqs=32)
    assert out_het is not out_32
    assert len(uploads) == 6
    assert np.allclose(out_het[0]._arr[:4, 0], [0.6, 0.8, 0.6, 0.6])
    assert np.allclose(out_het[0]._arr[4:, 0], 1.0)


def _staging_runner(input_ids):
    return SimpleNamespace(
        device=torch.device("cpu"),
        arange_np=np.arange(100, dtype=np.int32),
        num_tokens_paddings=[8, 16, 32, 64],
        input_ids_cpu=input_ids,
        _unfenced_h2d_sources=[],
        _registered_h2d_sources=set(),
    )


@pytest.mark.parametrize("defer", [False, True])
def test_spec_decode_metadata_uploads_are_staged(defer):
    """Every host array the spec-decode metadata uploads is staged until the
    next input-staging fence: the 6 index tensors (cached for later steps) and
    the drafts, on the uncached and the cached path and when a deferred
    upload is made."""
    input_ids = torch.arange(1, 11, dtype=torch.int32)
    runner = _staging_runner(input_ids)
    manager = SpeculativeDecodingManager(runner)
    layout = (np.array([3, 2], dtype=np.int32), np.array([4, 7], dtype=np.int32), 4)

    fresh = manager.get_spec_decode_metadata(*layout, defer_draft_token_ids=defer)
    staged = [host for host, _ in runner._unfenced_h2d_sources]
    assert len(staged) == (6 if defer else 7)
    device_tensors = [
        fresh.draft_lengths,
        fresh.target_logits_indices,
        fresh.bonus_logits_indices,
        fresh.final_logits_indices,
        fresh.segment_ids,
        fresh.group_indices,
    ]
    if not defer:
        device_tensors.insert(0, fresh.draft_token_ids)
    for host, dev in zip(staged, device_tensors, strict=True):
        assert torch.equal(host, dev)

    runner._unfenced_h2d_sources.clear()
    manager.get_spec_decode_metadata(*layout, defer_draft_token_ids=defer)
    assert len(runner._unfenced_h2d_sources) == (0 if defer else 1)
    if defer:
        uploaded = fresh.upload_deferred_draft_token_ids(runner)
        assert len(runner._unfenced_h2d_sources) == 1
        assert torch.equal(runner._unfenced_h2d_sources[0][0], uploaded)
