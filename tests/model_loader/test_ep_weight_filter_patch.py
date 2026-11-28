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
widens the suffix allowlist, and skips tensors the loading model marks
unused (``is_unused_checkpoint_weight``) before they are read. No TPU, no
network.
"""

from types import SimpleNamespace

import pytest
import torch
import vllm.config
import vllm.distributed
from safetensors.torch import save_file
from vllm.config.load import LoadConfig
from vllm.model_executor.layers.fused_moe.expert_map_manager import (
    determine_expert_map,
)
from vllm.model_executor.model_loader import (
    default_loader,
    ep_weight_filter,
    runai_streamer_loader,
    weight_utils,
)

from vllm_torchtpu import model_loader_patches
from vllm_torchtpu.model_loader_patches import (
    _compute_local_expert_ids,
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
    assert _should_skip_weight_tpu(_expert_name(500, ".weight_scale_inv"), _LOCAL)
    assert _should_skip_weight_tpu(_expert_name(500, ".weight_scale_2"), _LOCAL)
    assert _should_skip_weight_tpu(_expert_name(500, ".weight"), _LOCAL)


def test_remote_expert_dsv4_scale_skipped():
    # DeepSeek-V4 stores the per-expert MXFP4 block scales as ``.scale``.
    name = "layers.3.ffn.experts.{}.w1.scale"
    assert _should_skip_weight_tpu(name.format(500), _LOCAL)
    assert not _should_skip_weight_tpu(name.format(27), _LOCAL)


def test_local_expert_packed_and_scale_kept():
    assert not _should_skip_weight_tpu(_expert_name(27, ".weight_packed"), _LOCAL)
    assert not _should_skip_weight_tpu(_expert_name(27, ".weight_scale"), _LOCAL)
    assert not _should_skip_weight_tpu(_expert_name(27, ".weight_scale_inv"), _LOCAL)
    assert not _should_skip_weight_tpu(_expert_name(27, ".weight_scale_2"), _LOCAL)


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
    monkeypatch.setattr(
        default_loader.DefaultModelLoader,
        "_get_weights_iterator",
        default_loader.DefaultModelLoader._get_weights_iterator,
    )
    loader_cls = default_loader.DefaultModelLoader
    monkeypatch.setattr(
        loader_cls,
        "_init_ep_weight_filter",
        loader_cls._init_ep_weight_filter,  # noqa: SLF001 -- restore loader hook
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
        f"{e}.0.w1.weight_scale_inv": torch.zeros(1, dtype=torch.float32),
        f"{e}.0.w1.weight_scale_2": torch.zeros(1, dtype=torch.float32),
        f"{e}.7.w1.weight_packed": torch.zeros(8, dtype=torch.uint8),
        f"{e}.7.w1.weight_scale": torch.zeros(1, dtype=torch.uint8),
        f"{e}.7.w1.weight_scale_inv": torch.zeros(1, dtype=torch.float32),
        f"{e}.7.w1.weight_scale_2": torch.zeros(1, dtype=torch.float32),
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
    assert f"{e}.0.w1.weight_scale_inv" in yielded
    assert f"{e}.0.w1.weight_scale_2" in yielded
    assert f"{e}.7.w1.input_scale" in yielded
    assert "model.layers.0.self_attn.o_proj.weight" in yielded
    assert f"{e}.7.w1.weight_packed" not in yielded
    assert f"{e}.7.w1.weight_scale" not in yielded
    assert f"{e}.7.w1.weight_scale_inv" not in yielded
    assert f"{e}.7.w1.weight_scale_2" not in yielded
    # Skipped tensors must never be read from disk, not just not yielded.
    assert f"{e}.7.w1.weight_packed" not in read_names
    assert f"{e}.7.w1.weight_scale" not in read_names
    assert f"{e}.7.w1.weight_scale_inv" not in read_names
    assert f"{e}.7.w1.weight_scale_2" not in read_names


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


# ---------------------------------------------------------------------------
# Unused checkpoint weight filtering (MTP shards + co-located tensors + EP)
# ---------------------------------------------------------------------------


def _write_indexed_mtp_checkpoint(tmp_path):
    import json

    from safetensors.torch import save_file

    shards = {
        "main.safetensors": {
            "model.layers.0.self_attn.q_proj.weight": torch.ones(2, 2),
        },
        "vocab.safetensors": {
            "model.embed_tokens.weight": torch.full((4, 2), 2.0),
        },
        "head.safetensors": {
            "lm_head.weight": torch.full((4, 2), 3.0),
        },
        "mtp_remote_expert.safetensors": {
            "mtp.layers.0.mlp.experts.7.w1.weight_packed": torch.ones(
                8, dtype=torch.uint8
            ),
            "mtp.layers.0.mlp.experts.7.w1.weight_scale": torch.ones(
                1, dtype=torch.uint8
            ),
        },
        "mtp_local_and_main.safetensors": {
            "model.layers.1.self_attn.o_proj.weight": torch.full((2, 2), 9.0),
            "mtp.fc.weight": torch.full((2, 4), 4.0),
            "mtp.layers.0.mlp.experts.0.w1.weight_packed": torch.full(
                (8,), 5, dtype=torch.uint8
            ),
            "mtp.layers.0.mlp.experts.7.w2.weight_packed": torch.full(
                (8,), 6, dtype=torch.uint8
            ),
        },
    }
    weight_map = {}
    for filename, tensors in shards.items():
        save_file(tensors, str(tmp_path / filename))
        weight_map.update(dict.fromkeys(tensors, filename))
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))


def test_default_loader_skips_unused_shards_and_colocated_tensors(
    tmp_path, monkeypatch
):
    import os
    from types import SimpleNamespace

    import pytest
    from vllm.config.load import LoadConfig
    from vllm.model_executor.model_loader import (
        default_loader,
        ep_weight_filter,
        weight_utils,
    )
    from vllm.model_executor.models.qwen3_next_mtp import Qwen3NextMTP

    _write_indexed_mtp_checkpoint(tmp_path)

    monkeypatch.setattr(
        weight_utils, "should_skip_weight", weight_utils.should_skip_weight
    )
    monkeypatch.setattr(
        ep_weight_filter, "should_skip_weight", ep_weight_filter.should_skip_weight
    )
    monkeypatch.setattr(weight_utils, "_tpu_ep_filter_patch", False, raising=False)
    _restore_iterators(monkeypatch)
    patch_default_loader_ep_weight_filter()

    read_files = set()
    read_tensors = []
    real_safe_open = weight_utils.safe_open

    class _TrackingSafeOpen:
        def __init__(self, filename, *args, **kwargs):
            self._name = os.path.basename(filename)
            self._inner = real_safe_open(filename, *args, **kwargs)

        def __enter__(self):
            self._inner.__enter__()
            return self

        def __exit__(self, *args):
            return self._inner.__exit__(*args)

        def keys(self):
            return self._inner.keys()

        def get_tensor(self, name):
            read_files.add(self._name)
            read_tensors.append(name)
            return self._inner.get_tensor(name)

    monkeypatch.setattr(weight_utils, "safe_open", _TrackingSafeOpen)

    loader = object.__new__(default_loader.DefaultModelLoader)
    loader.load_config = LoadConfig(load_format="safetensors")
    loader.local_expert_ids = {0}
    loader._encoder_only_lm_prefixes = None

    # Emulate _load_mtp_draft_sharing_head wrapping is_unused_checkpoint_weight
    # when embed_tokens and lm_head are borrowed from the target model.
    base_is_unused = Qwen3NextMTP.is_unused_checkpoint_weight
    shared_keys = ("embed_tokens", "lm_head")
    dummy_model = SimpleNamespace(
        is_unused_checkpoint_weight=lambda name: (
            base_is_unused(dummy_model, name) or any(key in name for key in shared_keys)
        )
    )
    model_config = SimpleNamespace(
        model=str(tmp_path),
        revision=None,
        quantization=None,
    )

    yielded = dict(loader.get_all_weights(model_config, dummy_model))

    # Main, borrowed-head and remote-expert-only shards are never read.
    assert read_files == {"mtp_local_and_main.safetensors"}
    # Co-located main-model and remote expert tensors are skipped before get_tensor.
    assert set(read_tensors) == {
        "mtp.fc.weight",
        "mtp.layers.0.mlp.experts.0.w1.weight_packed",
    }
    assert set(yielded) == set(read_tensors)

    # A missing shard still fails, even one the draft does not read.
    (tmp_path / "main.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="main.safetensors"):
        list(loader.get_all_weights(model_config, dummy_model))


def test_should_partition_draft_experts_by_dspark_count_across_loaders(
    tmp_path, monkeypatch
):
    hf_cfg = SimpleNamespace(n_routed_experts=256, dspark_n_routed_experts=128)
    target_cfg = SimpleNamespace(
        is_moe=True,
        runner_type="generate",
        get_num_experts=lambda: 256,
        hf_config=hf_cfg,
    )
    draft_cfg = SimpleNamespace(
        model=str(tmp_path),
        revision=None,
        quantization=None,
        multimodal_config=None,
        is_moe=True,
        runner_type="draft",
        get_num_experts=lambda: 256,
        hf_config=hf_cfg,
    )

    state = {"ep_size": 4, "ep_rank": 1}
    vllm_cfg = SimpleNamespace(
        model_config=target_cfg,
        parallel_config=SimpleNamespace(
            enable_expert_parallel=True,
            enable_ep_weight_filter=True,
            enable_eplb=False,
            data_parallel_size=1,
            tensor_parallel_size=4,
            prefill_context_parallel_size=1,
            expert_placement_strategy="linear",
        ),
    )

    monkeypatch.setattr(vllm.config, "get_current_vllm_config", lambda: vllm_cfg)
    monkeypatch.setattr(
        vllm.distributed,
        "get_tensor_model_parallel_rank",
        lambda: state["ep_rank"],
    )
    # Isolate from earlier hierarchical-EP tests that permanently wrap
    # default_loader.compute_local_expert_ids in the same pytest worker.
    monkeypatch.setattr(
        default_loader,
        "compute_local_expert_ids",
        ep_weight_filter.compute_local_expert_ids,
    )

    # At ep_size=1 all experts are local (returns None); at EP in {2, 4, 8} draft
    # partitions 128 experts work-proportionally (disjoint union == 0..127).
    vllm_cfg.parallel_config.tensor_parallel_size = 1
    state["ep_rank"] = 0
    assert _compute_local_expert_ids(draft_cfg) is None
    for ep_size in (2, 4, 8):
        vllm_cfg.parallel_config.tensor_parallel_size = ep_size
        per_rank_sets = []
        for rank in range(ep_size):
            state["ep_rank"] = rank
            ids = _compute_local_expert_ids(draft_cfg)
            assert ids is not None
            assert len(ids) == 128 // ep_size
            _, expert_map, _ = determine_expert_map(
                ep_size=ep_size,
                ep_rank=rank,
                global_num_experts=draft_cfg.hf_config.dspark_n_routed_experts,
                expert_placement_strategy="linear",
            )
            assert expert_map is not None
            expected_map_ids = {
                gid for gid, lid in enumerate(expert_map.tolist()) if lid >= 0
            }
            assert ids == expected_map_ids
            per_rank_sets.append(ids)
        assert set().union(*per_rank_sets) == set(range(128))
        assert sum(len(s) for s in per_rank_sets) == 128

    # At ep_size=4, ep_rank=1: target owns {64..127} whereas draft owns {32..63}.
    vllm_cfg.parallel_config.tensor_parallel_size = 4
    state["ep_rank"] = 1
    assert _compute_local_expert_ids(target_cfg) == set(range(64, 128))
    assert _compute_local_expert_ids(draft_cfg) == set(range(32, 64))

    # DefaultModelLoader filters safetensors using draft partition {32..63}.
    shard = tmp_path / "model-00001.safetensors"
    packed = torch.ones(8, dtype=torch.uint8)
    save_file(
        {
            "layers.0.ffn.experts.32.w1.weight_packed": packed,
            "layers.0.ffn.experts.64.w1.weight_packed": packed.clone(),
            "layers.0.self_attn.o_proj.weight": torch.ones(2, 2),
        },
        str(shard),
    )
    read_names = []
    real_safe_open = weight_utils.safe_open

    class _TrackingSafeOpen:
        def __init__(self, filename, *args, **kwargs):
            self._inner = real_safe_open(filename, *args, **kwargs)

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

    monkeypatch.setattr(weight_utils, "safe_open", _TrackingSafeOpen)
    monkeypatch.setattr(
        weight_utils, "should_skip_weight", weight_utils.should_skip_weight
    )
    monkeypatch.setattr(
        ep_weight_filter, "should_skip_weight", ep_weight_filter.should_skip_weight
    )
    monkeypatch.setattr(weight_utils, "_tpu_ep_filter_patch", False, raising=False)
    _restore_iterators(monkeypatch)
    patch_default_loader_ep_weight_filter()

    loader = default_loader.DefaultModelLoader(LoadConfig(load_format="safetensors"))
    loaded = []
    dummy_model = SimpleNamespace(
        load_weights=lambda weights: loaded.extend(name for name, _ in weights)
    )
    loader.load_weights(dummy_model, draft_cfg)
    assert loader.local_expert_ids == set(range(32, 64))
    assert set(read_names) == {
        "layers.0.ffn.experts.32.w1.weight_packed",
        "layers.0.self_attn.o_proj.weight",
    }
    assert set(loaded) == set(read_names)

    # RunaiModelStreamerLoader threads model_config and resets it in finally.
    rsl_cls = runai_streamer_loader.RunaiModelStreamerLoader
    monkeypatch.setattr(rsl_cls, "load_weights", rsl_cls.load_weights)
    orig_rsl_iter = rsl_cls._get_weights_iterator  # noqa: SLF001 -- restore hook
    monkeypatch.setattr(rsl_cls, "_get_weights_iterator", orig_rsl_iter)
    monkeypatch.setattr(rsl_cls, "_tpu_sharded_ep_patch", False, raising=False)
    monkeypatch.setattr(rsl_cls, "_loading_model", None, raising=False)
    monkeypatch.setattr(rsl_cls, "_loading_model_config", None, raising=False)

    captured_stream_ids = []
    fail_on_call = {"flag": False}

    def fake_orig_load(self, model, model_config):
        list(self._get_weights_iterator(model_config.model, model_config.revision))
        if fail_on_call["flag"]:
            raise RuntimeError("injected stream failure")

    monkeypatch.setattr(rsl_cls, "load_weights", fake_orig_load)
    monkeypatch.setattr(
        model_loader_patches, "_patch_runai_local_rank_discovery", lambda: None
    )
    monkeypatch.setattr(
        model_loader_patches,
        "_sharded_runai_weights_iterator",
        lambda files, local_ids, use_tqdm, model, **_kwargs: (
            captured_stream_ids.append(set(local_ids)) or iter(())
        ),
    )
    model_loader_patches.patch_runai_sharded_expert_streaming()

    streamer = object.__new__(rsl_cls)
    streamer.load_config = LoadConfig(load_format="runai_streamer")
    monkeypatch.setattr(
        streamer, "_prepare_weights", lambda path, rev: ["shard-0.safetensors"]
    )
    streamer.load_weights(dummy_model, draft_cfg)
    assert captured_stream_ids == [set(range(32, 64))]
    assert streamer._loading_model_config is None  # noqa: SLF001 -- verify reset

    fail_on_call["flag"] = True
    with pytest.raises(RuntimeError, match="injected stream failure"):
        streamer.load_weights(dummy_model, draft_cfg)
    assert streamer._loading_model_config is None  # noqa: SLF001 -- verify reset


def test_should_fallback_to_model_num_experts_when_draft_dspark_count_absent_or_zero(
    monkeypatch,
):
    vllm_cfg = SimpleNamespace(
        model_config=SimpleNamespace(
            is_moe=True,
            runner_type="generate",
            get_num_experts=lambda: 256,
            hf_config=SimpleNamespace(n_routed_experts=256),
        ),
        parallel_config=SimpleNamespace(
            enable_expert_parallel=True,
            enable_ep_weight_filter=True,
            enable_eplb=False,
            data_parallel_size=1,
            tensor_parallel_size=4,
            prefill_context_parallel_size=1,
            expert_placement_strategy="linear",
        ),
    )
    monkeypatch.setattr(vllm.config, "get_current_vllm_config", lambda: vllm_cfg)
    monkeypatch.setattr(vllm.distributed, "get_tensor_model_parallel_rank", lambda: 1)
    monkeypatch.setattr(
        default_loader,
        "compute_local_expert_ids",
        ep_weight_filter.compute_local_expert_ids,
    )

    for hf_cfg in (SimpleNamespace(), SimpleNamespace(dspark_n_routed_experts=0)):
        draft_cfg = SimpleNamespace(
            is_moe=True,
            runner_type="draft",
            get_num_experts=lambda: 64,
            hf_config=hf_cfg,
        )
        assert _compute_local_expert_ids(draft_cfg) == set(range(16, 32))


def test_should_support_both_default_loader_init_ep_filter_signatures_and_delegate(
    monkeypatch,
):
    vllm_cfg = SimpleNamespace(
        model_config=SimpleNamespace(
            is_moe=True,
            runner_type="generate",
            get_num_experts=lambda: 256,
            hf_config=SimpleNamespace(n_routed_experts=256),
        ),
        parallel_config=SimpleNamespace(
            enable_expert_parallel=True,
            enable_ep_weight_filter=True,
            enable_eplb=False,
            data_parallel_size=1,
            tensor_parallel_size=4,
            prefill_context_parallel_size=1,
            expert_placement_strategy="linear",
        ),
    )
    monkeypatch.setattr(vllm.config, "get_current_vllm_config", lambda: vllm_cfg)
    monkeypatch.setattr(vllm.distributed, "get_tensor_model_parallel_rank", lambda: 1)
    monkeypatch.setattr(
        default_loader,
        "compute_local_expert_ids",
        ep_weight_filter.compute_local_expert_ids,
    )

    draft_cfg = SimpleNamespace(
        is_moe=True,
        runner_type="draft",
        get_num_experts=lambda: 256,
        hf_config=SimpleNamespace(n_routed_experts=256, dspark_n_routed_experts=128),
    )
    target_cfg = vllm_cfg.model_config
    dummy_model = SimpleNamespace(name="dummy")

    # 1. vLLM >=0.34 signature: _init_ep_weight_filter(self, model, model_config).
    delegated_v34 = []

    def fake_v34_init_ep_filter(self, model, model_config) -> None:
        delegated_v34.append((model, model_config))
        if model_config.runner_type == "generate":
            self.local_expert_ids = {99}

    monkeypatch.setattr(
        weight_utils, "should_skip_weight", weight_utils.should_skip_weight
    )
    monkeypatch.setattr(
        ep_weight_filter, "should_skip_weight", ep_weight_filter.should_skip_weight
    )
    monkeypatch.setattr(weight_utils, "_tpu_ep_filter_patch", False, raising=False)
    _restore_iterators(monkeypatch)
    monkeypatch.setattr(
        default_loader.DefaultModelLoader,
        "_init_ep_weight_filter",
        fake_v34_init_ep_filter,
    )
    patch_default_loader_ep_weight_filter()

    loader = default_loader.DefaultModelLoader(LoadConfig(load_format="safetensors"))
    # DSpark draft overrides local_expert_ids without delegating to upstream.
    loader._init_ep_weight_filter(dummy_model, draft_cfg)  # noqa: SLF001
    assert loader.local_expert_ids == set(range(32, 64))
    assert delegated_v34 == []

    # Target model delegates with both (model, model_config) args intact.
    loader._init_ep_weight_filter(dummy_model, target_cfg)  # noqa: SLF001
    assert delegated_v34 == [(dummy_model, target_cfg)]
    assert loader.local_expert_ids == {99}

    # Reusing the same loader for a non-MoE model clears stale local_expert_ids.
    non_moe_cfg = SimpleNamespace(
        is_moe=False,
        runner_type="draft",
        get_num_experts=lambda: 0,
        hf_config=SimpleNamespace(),
    )
    loader._init_ep_weight_filter(dummy_model, non_moe_cfg)  # noqa: SLF001
    assert loader.local_expert_ids is None
    assert delegated_v34[-1] == (dummy_model, non_moe_cfg)

    # 2. vLLM 0.33 signature: _init_ep_weight_filter(self, model_config).
    delegated_v33 = []

    def fake_v33_init_ep_filter(self, model_config) -> None:
        delegated_v33.append(model_config)
        self.local_expert_ids = {77}

    monkeypatch.setattr(weight_utils, "_tpu_ep_filter_patch", False, raising=False)
    _restore_iterators(monkeypatch)
    monkeypatch.setattr(
        default_loader.DefaultModelLoader,
        "_init_ep_weight_filter",
        fake_v33_init_ep_filter,
    )
    patch_default_loader_ep_weight_filter()

    loader._init_ep_weight_filter(draft_cfg)  # noqa: SLF001
    assert loader.local_expert_ids == set(range(32, 64))
    assert delegated_v33 == []

    loader._init_ep_weight_filter(model_config=target_cfg)  # noqa: SLF001
    assert delegated_v33 == [target_cfg]
    assert loader.local_expert_ids == {77}
