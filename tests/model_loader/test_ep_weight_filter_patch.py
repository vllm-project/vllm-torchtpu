# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""Unit tests for the DefaultModelLoader EP weight-filter patch.

vLLM 0.27.0's ``should_skip_weight`` only matches names ending in
``.weight``, so a compressed-tensors MXFP4 checkpoint (``.weight_packed`` /
``.weight_scale``) reads every expert's bytes on every rank. The TPU patch
widens the suffix allowlist. No TPU, no network.
"""

import torch

from vllm_torchtpu import model_loader_patches
from vllm_torchtpu.model_loader_patches import (
    _local_tensor_ranges,
    _should_skip_weight_tpu,
    _start_ep_range_prefetch,
    patch_default_loader_ep_weight_filter,
)

_LOCAL = {0, 1, 27}


def _expert_name(expert_id: int, suffix: str) -> str:
    return (
        f"language_model.model.layers.3.block_sparse_moe.experts.{expert_id}.w1{suffix}"
    )


# ---------------------------------------------------------------------------
# _should_skip_weight_tpu
# ---------------------------------------------------------------------------


def test_remote_expert_packed_and_scale_skipped():
    assert _should_skip_weight_tpu(_expert_name(500, ".weight_packed"), _LOCAL)
    assert _should_skip_weight_tpu(_expert_name(500, ".weight_scale"), _LOCAL)
    assert _should_skip_weight_tpu(_expert_name(500, ".weight"), _LOCAL)


def test_remote_expert_dsv4_scale_skipped():
    # DeepSeek-V4 stores the per-expert MXFP4 block scales as ``.scale``.
    name = "layers.3.ffn.experts.{}.w1.scale"
    assert _should_skip_weight_tpu(name.format(500), _LOCAL)
    assert not _should_skip_weight_tpu(name.format(27), _LOCAL)


def test_local_expert_packed_and_scale_kept():
    assert not _should_skip_weight_tpu(_expert_name(27, ".weight_packed"), _LOCAL)
    assert not _should_skip_weight_tpu(_expert_name(27, ".weight_scale"), _LOCAL)


def test_remote_expert_input_scale_kept():
    # NVFP4 per-expert activation scales feed a global-max reduction in
    # FlashInfer backends; they must stay unfiltered on every rank.
    assert not _should_skip_weight_tpu(_expert_name(500, ".input_scale"), _LOCAL)


def test_no_local_ids_keeps_everything():
    # Filter off (EP weight filter disabled): the upstream contract is that
    # local_expert_ids=None keeps every tensor.
    assert not _should_skip_weight_tpu(_expert_name(500, ".weight_packed"), None)


def test_dense_and_fused_names_kept():
    for name in (
        "language_model.model.layers.3.self_attn.o_proj.weight",
        "language_model.model.layers.3.block_sparse_moe.gate.weight",
        "language_model.model.embed_tokens.weight",
        # 3D fused-expert layout: no numeric id, full tensor must load.
        "model.layers.3.mlp.experts.gate_proj.weight",
    ):
        assert not _should_skip_weight_tpu(name, _LOCAL), name


# ---------------------------------------------------------------------------
# Patch installation
# ---------------------------------------------------------------------------


def _restore_iterators(monkeypatch):
    from vllm.model_executor.model_loader import default_loader, weight_utils

    monkeypatch.setattr(
        weight_utils,
        "safetensors_weights_iterator",
        weight_utils.safetensors_weights_iterator,
    )
    monkeypatch.setattr(
        default_loader,
        "safetensors_weights_iterator",
        default_loader.safetensors_weights_iterator,
    )


def test_patch_installs_predicate_and_is_idempotent(monkeypatch):
    from vllm.model_executor.model_loader import (
        default_loader,
        ep_weight_filter,
        weight_utils,
    )

    # Register teardown restoration of the attributes the patch overwrites.
    monkeypatch.setattr(
        weight_utils, "should_skip_weight", weight_utils.should_skip_weight
    )
    monkeypatch.setattr(
        ep_weight_filter, "should_skip_weight", ep_weight_filter.should_skip_weight
    )
    # delattr(raising=False) records nothing when the flag is absent, so it
    # would not undo the stamp the patch sets. setattr registers a teardown
    # that deletes (or restores) it.
    monkeypatch.setattr(weight_utils, "_tpu_ep_filter_patch", False, raising=False)
    _restore_iterators(monkeypatch)

    patch_default_loader_ep_weight_filter()
    assert weight_utils.should_skip_weight is _should_skip_weight_tpu
    assert ep_weight_filter.should_skip_weight is _should_skip_weight_tpu
    assert weight_utils._tpu_ep_filter_patch is True
    assert default_loader.safetensors_weights_iterator is (
        weight_utils.safetensors_weights_iterator
    )

    # A second call must not re-apply over a foreign value.
    foreign = object()
    weight_utils.should_skip_weight = foreign
    patch_default_loader_ep_weight_filter()
    assert weight_utils.should_skip_weight is foreign


# ---------------------------------------------------------------------------
# Iterator integration: real safetensors keys, lazy load path
# ---------------------------------------------------------------------------


def test_lazy_iterator_filters_remote_packed_and_scale(tmp_path, monkeypatch):
    from safetensors.torch import save_file
    from vllm.model_executor.model_loader import weight_utils

    e = "model.layers.0.mlp.experts"
    tensors = {
        f"{e}.0.w1.weight_packed": torch.zeros(8, dtype=torch.uint8),
        f"{e}.0.w1.weight_scale": torch.zeros(1, dtype=torch.uint8),
        f"{e}.7.w1.weight_packed": torch.zeros(8, dtype=torch.uint8),
        f"{e}.7.w1.weight_scale": torch.zeros(1, dtype=torch.uint8),
        f"{e}.7.w1.input_scale": torch.zeros(1, dtype=torch.float32),
        "model.layers.0.self_attn.o_proj.weight": torch.zeros(2, 2),
    }
    shard = tmp_path / "model-00001.safetensors"
    save_file(tensors, str(shard))

    # Track which tensors are actually read from disk.
    read_names = []
    real_safe_open = weight_utils.safe_open

    class _TrackingSafeOpen:
        def __init__(self, *args, **kwargs):
            self._inner = real_safe_open(*args, **kwargs)

        def __enter__(self):
            self._inner.__enter__()
            return self

        def __exit__(self, *args):
            return self._inner.__exit__(*args)

        def keys(self):
            return self._inner.keys()

        def get_tensor(self, name):
            read_names.append(name)
            return self._inner.get_tensor(name)

    monkeypatch.setattr(weight_utils, "should_skip_weight", _should_skip_weight_tpu)
    monkeypatch.setattr(weight_utils, "safe_open", _TrackingSafeOpen)

    yielded = {
        name
        for name, _ in weight_utils.safetensors_weights_iterator(
            [str(shard)], use_tqdm_on_load=False, local_expert_ids={0}
        )
    }

    assert f"{e}.0.w1.weight_packed" in yielded
    assert f"{e}.0.w1.weight_scale" in yielded
    assert f"{e}.7.w1.input_scale" in yielded
    assert "model.layers.0.self_attn.o_proj.weight" in yielded
    assert f"{e}.7.w1.weight_packed" not in yielded
    assert f"{e}.7.w1.weight_scale" not in yielded
    # Skipped tensors must never be read from disk, not just not yielded.
    assert f"{e}.7.w1.weight_packed" not in read_names
    assert f"{e}.7.w1.weight_scale" not in read_names


# ---------------------------------------------------------------------------
# EP-local range prefetch
# ---------------------------------------------------------------------------

_E = "model.layers.0.mlp.experts"


def _save_shard(tmp_path):
    from safetensors.torch import save_file

    # Two remote experts sit between the local ones and the dense tensor, so
    # the kept tensors form more than one byte range.
    tensors = {
        f"{_E}.0.w1.weight_packed": torch.ones(64, dtype=torch.uint8),
        f"{_E}.0.w1.scale": torch.ones(4, dtype=torch.uint8),
        f"{_E}.5.w1.weight_packed": torch.ones(64, dtype=torch.uint8),
        f"{_E}.6.w1.weight_packed": torch.ones(64, dtype=torch.uint8),
        f"{_E}.7.w1.scale": torch.ones(4, dtype=torch.uint8),
        "model.layers.0.self_attn.o_proj.weight": torch.ones(4, 4),
    }
    shard = tmp_path / "model-00001.safetensors"
    save_file(tensors, str(shard))
    return str(shard)


def _header(path):
    import json
    import struct

    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return 8 + n, header


def test_local_tensor_ranges_cover_exactly_the_kept_tensors(tmp_path):
    shard = _save_shard(tmp_path)
    base, header = _header(shard)

    ranges = _local_tensor_ranges(shard, {0})

    covered = set()
    for lo, hi in ranges:
        covered.update(range(lo, hi))
    for name, info in header.items():
        start, end = (base + o for o in info["data_offsets"])
        span = set(range(start, end))
        if _should_skip_weight_tpu(name, {0}):
            assert not span & covered, name
        else:
            assert span <= covered, name
    # Ranges are sorted, disjoint and non-adjacent (adjacent spans merge).
    assert all(a[1] < b[0] for a, b in zip(ranges, ranges[1:]))


def test_local_tensor_ranges_all_local_is_one_range(tmp_path):
    shard = _save_shard(tmp_path)
    base, header = _header(shard)
    end = base + max(info["data_offsets"][1] for info in header.values())

    assert _local_tensor_ranges(shard, {0, 5, 6, 7}) == [(base, end)]


def test_range_prefetch_reads_only_local_ranges(tmp_path, monkeypatch):
    import os

    shard = _save_shard(tmp_path)

    import threading

    started = []

    class _RecordedThread(threading.Thread):
        def start(self):
            started.append(self)
            super().start()

    reads = []
    real_pread = os.pread

    def tracking_pread(fd, n, off):
        reads.append((off, off + n))
        return real_pread(fd, n, off)

    monkeypatch.setattr(model_loader_patches.threading, "Thread", _RecordedThread)
    monkeypatch.setattr(model_loader_patches.os, "pread", tracking_pread)

    _start_ep_range_prefetch([shard], {0}, num_threads=2, block_size=16)
    for thread in started:
        thread.join(timeout=30)

    expected = set()
    for lo, hi in _local_tensor_ranges(shard, {0}):
        expected.update(range(lo, hi))
    read = set()
    for lo, hi in reads:
        assert hi - lo <= 16
        read.update(range(lo, hi))
    assert read == expected


def test_prefetch_strategy_under_filter_switches_to_lazy(tmp_path, monkeypatch):
    from vllm.model_executor.model_loader import ep_weight_filter, weight_utils

    shard = _save_shard(tmp_path)
    monkeypatch.setattr(
        weight_utils, "should_skip_weight", weight_utils.should_skip_weight
    )
    monkeypatch.setattr(
        ep_weight_filter, "should_skip_weight", ep_weight_filter.should_skip_weight
    )
    monkeypatch.setattr(weight_utils, "_tpu_ep_filter_patch", False, raising=False)
    _restore_iterators(monkeypatch)

    original = weight_utils.safetensors_weights_iterator
    strategies = []

    def recording_iterator(files, use_tqdm, strategy=None, local_ids=None, **kw):
        strategies.append(strategy)
        return original(files, use_tqdm, strategy, local_ids, **kw)

    prefetches = []
    monkeypatch.setattr(
        weight_utils, "safetensors_weights_iterator", recording_iterator
    )
    monkeypatch.setattr(
        model_loader_patches,
        "_start_ep_range_prefetch",
        lambda files, ids, threads, block: prefetches.append((files, ids)),
    )
    patch_default_loader_ep_weight_filter()
    iterator = weight_utils.safetensors_weights_iterator

    names = {
        name for name, _ in iterator([shard], False, "prefetch", local_expert_ids={0})
    }
    assert strategies == ["lazy"]
    assert prefetches == [([shard], {0})]
    assert f"{_E}.0.w1.scale" in names
    assert f"{_E}.7.w1.scale" not in names

    # Filter off: upstream prefetch runs unchanged.
    list(iterator([shard], False, "prefetch", local_expert_ids=None))
    assert strategies == ["lazy", "prefetch"]
    assert len(prefetches) == 1
