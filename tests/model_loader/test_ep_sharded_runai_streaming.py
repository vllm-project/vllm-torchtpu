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
"""Unit tests for EP-sharded Run:AI weight streaming (model_loader_patches).

Covers the pieces that decide WHAT gets fetched: the per-tensor skip
predicate, and the construction of the byte-range request plan from
safetensors metadata. No TPU, no network: the Run:AI streamer is faked.
"""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm_torchtpu.model_loader_patches import (
    _compute_local_expert_ids,
    _sharded_runai_weights_iterator,
    _should_skip,
)

# ---------------------------------------------------------------------------
# _should_skip
# ---------------------------------------------------------------------------

_LOCAL = {0, 1, 27}


def test_local_expert_ids_use_loader_ownership_mapping():
    """Run:AI filtering must honor hierarchical EP's loader patch."""
    model_config = SimpleNamespace(
        is_moe=True,
        get_num_experts=lambda: 896,
    )
    parallel_config = SimpleNamespace(
        enable_expert_parallel=True,
        enable_ep_weight_filter=True,
        enable_eplb=False,
        data_parallel_size=1,
        tensor_parallel_size=32,
        prefill_context_parallel_size=1,
        expert_placement_strategy="linear",
    )
    config = SimpleNamespace(
        model_config=model_config,
        parallel_config=parallel_config,
    )
    chip_aware_ids = set(range(56))

    compute_ids = MagicMock(return_value=chip_aware_ids)
    vllm = ModuleType("vllm")
    vllm.__path__ = []
    config_module = ModuleType("vllm.config")
    config_module.get_current_vllm_config = MagicMock(return_value=config)
    model_executor = ModuleType("vllm.model_executor")
    model_executor.__path__ = []
    loader_module = ModuleType("vllm.model_executor.model_loader")
    loader_module.default_loader = SimpleNamespace(compute_local_expert_ids=compute_ids)
    distributed_module = ModuleType("vllm.distributed")
    distributed_module.get_dp_group = MagicMock()
    distributed_module.get_pcp_group = MagicMock()
    distributed_module.get_tensor_model_parallel_rank = MagicMock(return_value=11)
    logger_module = ModuleType("vllm.logger")
    logger_module.init_logger = MagicMock(return_value=MagicMock())

    fake_modules = {
        "vllm": vllm,
        "vllm.config": config_module,
        "vllm.model_executor": model_executor,
        "vllm.model_executor.model_loader": loader_module,
        "vllm.distributed": distributed_module,
        "vllm.logger": logger_module,
    }
    with patch.dict(sys.modules, fake_modules):
        assert _compute_local_expert_ids() == chip_aware_ids

    compute_ids.assert_called_once_with(896, 32, 11, placement="linear")


def test_skips_nonlocal_expert_packed_and_scale():
    base = "language_model.model.layers.3.block_sparse_moe.experts.500"
    assert _should_skip(f"{base}.w1.weight_packed", _LOCAL)
    assert _should_skip(f"{base}.w2.weight_scale", _LOCAL)
    assert _should_skip(f"{base}.w3.weight", _LOCAL)


def test_keeps_local_expert_weights():
    base = "language_model.model.layers.3.block_sparse_moe.experts.27"
    assert not _should_skip(f"{base}.w1.weight_packed", _LOCAL)
    assert not _should_skip(f"{base}.w2.weight_scale", _LOCAL)


def test_keeps_dense_shared_and_gate_weights():
    prefix = "language_model.model.layers.3"
    for name in (
        f"{prefix}.self_attn.o_proj.weight",
        f"{prefix}.block_sparse_moe.shared_experts.up_proj.weight",
        f"{prefix}.block_sparse_moe.gate.weight",
        f"{prefix}.block_sparse_moe.gate.e_score_correction_bias",
        "language_model.model.embed_tokens.weight",
    ):
        assert not _should_skip(name, _LOCAL), name


def test_keeps_expert_tensors_with_unknown_suffix():
    # Only known heavy suffixes are skippable; anything else is kept even
    # for a non-local expert, mirroring upstream's conservatism.
    name = "language_model.model.layers.3.block_sparse_moe.experts.500.w1.some_metadata"
    assert not _should_skip(name, _LOCAL)


def test_keeps_fused_expert_tensors_without_numeric_id():
    # 3D fused-expert layouts have no numeric id; the full tensor must load
    # so the weight_loader can slice it.
    name = "model.layers.3.mlp.experts.gate_proj.weight"
    assert not _should_skip(name, _LOCAL)


# ---------------------------------------------------------------------------
# Request-plan construction
# ---------------------------------------------------------------------------


def _tensor_meta(name):
    return SimpleNamespace(name=name)


def _fake_file_meta(offset, names_and_sizes):
    return SimpleNamespace(
        offset=offset,
        tensors_metadata=[_tensor_meta(n) for n, _ in names_and_sizes],
        read_sizes=[s for _, s in names_and_sizes],
    )


def test_request_plan_groups_contiguous_kept_runs():
    """Kept tensors separated by a skipped one become separate byte-range
    requests, offsets account for the skipped bytes, and skipped tensors are
    never part of any request."""
    # The Run:AI streamer ships with PR #331; skip until it is in the image.
    pytest.importorskip("runai_model_streamer")
    e = "m.layers.0.experts"
    file_meta = _fake_file_meta(
        offset=1000,
        names_and_sizes=[
            (f"{e}.0.w1.weight_packed", 10),  # kept   @1000
            (f"{e}.0.w1.weight_scale", 2),  # kept   @1010
            (f"{e}.5.w1.weight_packed", 10),  # skip   @1012
            (f"{e}.5.w1.weight_scale", 2),  # skip   @1022
            ("m.layers.0.dense.weight", 7),  # kept   @1024
        ],
    )

    captured = {}

    class FakeStreamer:
        def __init__(self):
            self.file_streamer = self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def stream_files(self, requests, credentials, device, is_distributed):
            captured["requests"] = requests

        def get_chunks(self):
            return iter(())  # plan construction is the unit under test

    # The iterator imports the Run:AI modules lazily and reads these names
    # as attributes at call time, so patching the real modules' attributes
    # intercepts them deterministically.
    import runai_model_streamer.safetensors_streamer.safetensors_pytorch as sp
    from runai_model_streamer.safetensors_streamer import safetensors_streamer

    fake_metadata = MagicMock()
    fake_metadata.from_files.return_value = [file_meta]

    with (
        patch.object(safetensors_streamer, "SafetensorsStreamer", FakeStreamer),
        patch.object(sp, "SafetensorsMetadata", fake_metadata),
    ):
        list(
            _sharded_runai_weights_iterator(
                ["gs://bucket/file0.safetensors"],
                local_expert_ids={0},
                use_tqdm_on_load=False,
            )
        )

    requests = captured["requests"]
    assert [(r.offsets, r.sizes) for r in requests] == [
        ([1000, 1010], [10, 2]),  # expert 0's contiguous run
        ([1024], [7]),  # dense weight after the skipped gap
    ]
    total_requested = sum(sum(r.sizes) for r in requests)
    assert total_requested == 19  # 31 bytes in file, 12 skipped


def test_zero_size_kept_tensor_bypasses_fetch_plan():
    """A kept zero-byte tensor must never enter the fetch plan (the
    streamer's request iterator silently drops a zero-byte chunk group, and
    one at the head of its queue aborts the remaining stream). It is
    yielded as an empty tensor instead, and it must not break the
    contiguity of the surrounding run."""
    # The Run:AI streamer ships with PR #331; skip until it is in the image.
    pytest.importorskip("runai_model_streamer")
    file_meta = _fake_file_meta(
        offset=100,
        names_and_sizes=[
            ("m.layers.0.dense_a.weight", 8),  # kept @100
            ("m.layers.0.empty.weight", 0),  # kept, zero bytes
            ("m.layers.0.dense_b.weight", 4),  # kept @108 -- same run
        ],
    )
    # Give the zero-size tensor real metadata so create_torch_tensor takes
    # its empty-tensor path.
    file_meta.tensors_metadata[1].get_item_count = lambda: 0
    file_meta.tensors_metadata[1].shape = [0]
    file_meta.tensors_metadata[1].get_torch_dtype = lambda: None

    captured = {}

    class FakeStreamer:
        def __init__(self):
            self.file_streamer = self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def stream_files(self, requests, credentials, device, is_distributed):
            captured["requests"] = requests

        def get_chunks(self):
            return iter(())

    import runai_model_streamer.safetensors_streamer.safetensors_pytorch as sp
    from runai_model_streamer.safetensors_streamer import safetensors_streamer

    fake_metadata = MagicMock()
    fake_metadata.from_files.return_value = [file_meta]
    fake_create = MagicMock(return_value=MagicMock())

    with (
        patch.object(safetensors_streamer, "SafetensorsStreamer", FakeStreamer),
        patch.object(sp, "SafetensorsMetadata", fake_metadata),
        patch.object(sp, "create_torch_tensor", fake_create),
    ):
        yielded = [
            name
            for name, _ in _sharded_runai_weights_iterator(
                ["gs://bucket/file0.safetensors"],
                local_expert_ids=set(),
                use_tqdm_on_load=False,
            )
        ]

    requests = captured["requests"]
    # One contiguous run through the zero-byte tensor; no 0-length chunk.
    assert [(r.offsets, r.sizes) for r in requests] == [([100, 108], [8, 4])]
    # The zero-size tensor is still yielded (as an empty tensor).
    assert "m.layers.0.empty.weight" in yielded


def test_streamer_split_preserves_chunk_index_mapping():
    """Pin the library contract this loader depends on: when
    RUNAI_STREAMER_MEMORY_LIMIT forces FilesRequestsIterator to split one
    FileChunks across several sub-requests, get_global_file_and_range must
    keep reporting (our request id, index within OUR range list)."""
    # The Run:AI streamer ships with PR #331; skip until it is in the image.
    pytest.importorskip("runai_model_streamer")
    from runai_model_streamer.file_streamer.requests_iterator import (
        FileChunks,
        FilesRequestsIterator,
    )

    # One request of 5 chunks, memory limit 6 -> split into [3,3],[3,3],[3]
    chunks = [3, 3, 3, 3, 3]
    it = FilesRequestsIterator(6, [FileChunks.contiguous(7, "/f", 100, list(chunks))])

    seen = []
    while True:
        req = it.next_request()
        if req is None:
            break
        for local_file_idx, fc in enumerate(req.files):
            for local_range_idx in range(len(fc.sizes)):
                file_id, global_idx = it.get_global_file_and_range(
                    req, local_file_idx, local_range_idx
                )
                seen.append((file_id, global_idx))
    assert seen == [(7, i) for i in range(len(chunks))]


def test_patch_falls_back_when_filter_inactive():
    """With no local ids (flag off / old vLLM), the patched method must
    delegate to the original iterator; the patch must not double-apply."""
    from vllm.model_executor.model_loader import runai_streamer_loader as rsl

    from vllm_torchtpu import model_loader_patches as mlp

    original = MagicMock(name="original_get_weights_iterator")

    class FakeLoader:
        _get_weights_iterator = original
        load_weights = MagicMock(name="original_load_weights")

    with patch.object(rsl, "RunaiModelStreamerLoader", FakeLoader):
        mlp.patch_runai_sharded_expert_streaming()
        patched_once = FakeLoader._get_weights_iterator
        mlp.patch_runai_sharded_expert_streaming()  # idempotent
        assert FakeLoader._get_weights_iterator is patched_once

        with patch.object(mlp, "_compute_local_expert_ids", return_value=None):
            loader = FakeLoader()
            FakeLoader._get_weights_iterator(loader, "gs://b/m", None)
        original.assert_called_once_with(loader, "gs://b/m", None, None)


def test_patch_forwards_is_unused_weight_when_filter_active():
    """With EP sharding active, _get_weights_iterator forwards every shard and
    is_unused_weight to _sharded_runai_weights_iterator, which skips unused
    tensors from the shard headers without a separate file pre-filter."""
    from vllm.model_executor.model_loader import runai_streamer_loader as rsl

    from vllm_torchtpu import model_loader_patches as mlp

    class FakeLoader:
        _get_weights_iterator = MagicMock(name="original_get_weights_iterator")
        load_weights = MagicMock(name="original_load_weights")
        load_config = SimpleNamespace(use_tqdm_on_load=False)

        def _prepare_weights(self, model_or_path, revision):
            return ["gs://b/m/shard0.safetensors", "gs://b/m/shard1.safetensors"]

    unused_fn = lambda name: name.startswith("unused.")  # noqa: E731
    with (
        patch.object(rsl, "RunaiModelStreamerLoader", FakeLoader),
        patch.object(rsl, "is_runai_obj_uri", return_value=True),
        patch.object(rsl, "_filter_remote_weights") as mock_filter,
        patch.object(mlp, "_compute_local_expert_ids", return_value={0}),
        patch.object(mlp, "_sharded_runai_weights_iterator") as mock_sharded_iter,
    ):
        mlp.patch_runai_sharded_expert_streaming()
        loader = FakeLoader()
        FakeLoader._get_weights_iterator(loader, "gs://b/m", "main", unused_fn)
        mock_filter.assert_not_called()
        mock_sharded_iter.assert_called_once_with(
            ["gs://b/m/shard0.safetensors", "gs://b/m/shard1.safetensors"],
            {0},
            False,
            None,
            is_unused_weight=unused_fn,
        )


def test_layer_finisher_runs_after_consumer_receives_tensor():
    pytest.importorskip("runai_model_streamer")
    import runai_model_streamer.safetensors_streamer.safetensors_pytorch as sp
    import torch
    from runai_model_streamer.safetensors_streamer import safetensors_streamer

    from vllm_torchtpu import model_loader_patches as mlp

    names = ["model.layers.0.q.weight", "model.layers.0.v.weight"]
    metadata = _fake_file_meta(100, [(name, 4) for name in names])
    received, finalized = [], []
    model = torch.nn.Module()

    class FakeFinisher:
        def __init__(self, actual_model, actual_names):
            assert actual_model is model
            assert actual_names == names

        def consumed(self, name):
            assert received[-1] == name
            finalized.append(name)

    class FakeStreamer:
        def __init__(self):
            self.file_streamer = self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def stream_files(self, *args, **kwargs):
            pass

        def get_chunks(self):
            return iter([(0, 0, b"1234"), (0, 1, b"5678")])

    with (
        patch.object(safetensors_streamer, "SafetensorsStreamer", FakeStreamer),
        patch.object(sp.SafetensorsMetadata, "from_files", return_value=[metadata]),
        patch.object(sp, "create_torch_tensor", return_value=torch.ones(1)),
        patch.object(mlp, "_OnlineFp8LayerFinisher", FakeFinisher),
        patch.object(mlp.envs, "TPU_SHARDED_LOAD_SYNC_EVERY", 0),
    ):
        for name, _ in _sharded_runai_weights_iterator(
            ["gs://b/m"], {0}, False, model=model
        ):
            assert name not in finalized
            received.append(name)
    assert finalized == names


@pytest.mark.parametrize("fail", [False, True])
def test_loading_model_is_scoped_to_load_weights(fail):
    import torch
    from vllm.model_executor.model_loader import runai_streamer_loader as rsl

    from vllm_torchtpu import model_loader_patches as mlp

    model = torch.nn.Module()

    class FakeLoader:
        _get_weights_iterator = MagicMock()

        def load_weights(self, actual_model, model_config):
            assert self._loading_model is actual_model
            if fail:
                raise RuntimeError("load failed")

    with patch.object(rsl, "RunaiModelStreamerLoader", FakeLoader):
        mlp.patch_runai_sharded_expert_streaming()
        loader = FakeLoader()
        assert loader._loading_model is None
        if fail:
            with pytest.raises(RuntimeError, match="load failed"):
                loader.load_weights(model, None)
        else:
            loader.load_weights(model, None)
    assert loader._loading_model is None


def test_local_rank_discovery_reuses_exported_group_size(monkeypatch):
    pytest.importorskip("runai_model_streamer")
    from runai_model_streamer.distributed_streamer import (
        distributed_streamer as rds,
    )

    from vllm_torchtpu import model_loader_patches as mlp

    mlp._patch_runai_local_rank_discovery()
    mlp._patch_runai_local_rank_discovery()  # idempotent
    params = rds._distributedStreamerParams()
    monkeypatch.setenv("RUNAI_STREAMER_PROCESS_GROUP_SIZE", "8")
    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_rank", return_value=5),
        patch("torch.distributed.new_group", side_effect=AssertionError("collective")),
    ):
        assert params.find_local_ranks() == (8, 5, [])
    monkeypatch.delenv("RUNAI_STREAMER_PROCESS_GROUP_SIZE")
    with patch("torch.distributed.is_initialized", return_value=False):
        assert params.find_local_ranks() == (1, 0, [[0]])


def test_ep_rank_is_dp_major_over_dp_pcp_tp():
    """The EP rank the loader and TPU_DP_VOCAB_PARALLEL both assume."""
    from vllm_torchtpu import utils

    ranks = [
        utils.ep_rank(dp_rank=d, pcp_rank=p, pcp_size=2, tp_rank=t, tp_size=3)
        for d in range(2)
        for p in range(2)
        for t in range(3)
    ]
    assert ranks == list(range(12))


def _config_with_auto_runai_limit(limit: int | None) -> SimpleNamespace:
    from vllm.config import LoadConfig, ParallelConfig

    from vllm_torchtpu.model_loader_patches import TPU_AUTO_RUNAI_MEMORY_LIMIT_KEY

    return SimpleNamespace(
        load_config=LoadConfig(load_format="runai_streamer"),
        parallel_config=ParallelConfig(tensor_parallel_size=8),
        additional_config=(
            {} if limit is None else {TPU_AUTO_RUNAI_MEMORY_LIMIT_KEY: limit}
        ),
    )


@pytest.mark.parametrize(
    "auto_limit, caller_limit, expected",
    [
        (4096, None, 4096),  # the largest tensor fits: the cap is kept
        (None, None, None),  # no automatic cap: the streamer's own default applies
        (4096, 2048, 2048),  # a caller's own limit is never replaced
    ],
)
def test_file_streamer_patch_fills_in_only_a_missing_memory_limit(
    monkeypatch, auto_limit, caller_limit, expected
):
    """The branches that the real streamer test below does not reach."""
    rfs = pytest.importorskip("runai_model_streamer.file_streamer.file_streamer")

    from vllm_torchtpu import model_loader_patches as mlp

    forwarded = []

    def record(self, file_stream_requests, credentials, device, memory_limit):
        forwarded.append(memory_limit)

    monkeypatch.delenv("RUNAI_STREAMER_MEMORY_LIMIT", raising=False)
    monkeypatch.setattr(rfs.FileStreamer, "stream_files", record)
    monkeypatch.setattr(
        rfs.FileStreamer, "_tpu_memory_limit_patch", False, raising=False
    )
    mlp.patch_runai_file_streamer_memory_limit()

    request = SimpleNamespace(sizes=[1024], max_chunk_size=lambda: 1024)
    kwargs = {} if caller_limit is None else {"memory_limit": caller_limit}
    with patch(
        "vllm.config.get_current_vllm_config_or_none",
        return_value=_config_with_auto_runai_limit(auto_limit),
    ):
        rfs.FileStreamer.stream_files(None, [request], None, "cpu", **kwargs)
    assert forwarded == [expected]


@pytest.mark.parametrize(
    "tp, dp, dp_size_env, uses_ray, extra_config, expected",
    [
        (1, 16, None, True, {}, 5 << 30),  # 2-host DP=16: min(8, 16) ranks here
        (1, 1, "16", True, {}, 5 << 30),  # per-engine config: DP size from the env
        (2, 1, None, True, {}, 20 << 30),  # 2-rank job on an 8-chip host
        (1, 1, None, True, {}, None),  # single-rank job on an 8-chip host
        (8, 1, None, False, {}, None),  # mp workers write LOCAL_WORLD_SIZE themselves
        (1, 16, None, True, {"memory_limit": 2 << 30}, None),  # operator limit wins
    ],
)
def test_ray_worker_derives_runai_memory_limit_from_local_world_size(
    monkeypatch, tp, dp, dp_size_env, uses_ray, extra_config, expected
):
    """Under Ray, ParallelConfig.nnodes stays 1 and config time records no cap,
    so each worker reads LOCAL_WORLD_SIZE clamped to the job's rank count
    (world_size * dp_size)."""
    from vllm.config import LoadConfig, ParallelConfig
    from vllm.v1.executor.abstract import Executor

    from vllm_torchtpu import model_loader_patches as mlp

    class FakeRayExecutor(Executor):
        uses_ray = True

    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    monkeypatch.delenv("RUNAI_STREAMER_MEMORY_LIMIT", raising=False)
    monkeypatch.delenv("TORCH_TPU_DP_SIZE", raising=False)
    if dp_size_env is not None:
        monkeypatch.setenv("TORCH_TPU_DP_SIZE", dp_size_env)
    with patch("vllm.v1.executor.ray_utils.assert_ray_available"):
        parallel_config = ParallelConfig(
            tensor_parallel_size=tp,
            data_parallel_size=dp,
            data_parallel_size_local=dp,
            distributed_executor_backend=FakeRayExecutor if uses_ray else "mp",
        )
    vllm_config = SimpleNamespace(
        load_config=LoadConfig(
            load_format="runai_streamer", model_loader_extra_config=extra_config
        ),
        parallel_config=parallel_config,
        additional_config={},
    )

    assert mlp._resolve_auto_runai_memory_limit(vllm_config) == expected


def test_real_safetensors_streamer_uses_floored_auto_memory_limit(
    monkeypatch, tmp_path
):
    """Stream a real local safetensors file through SafetensorsStreamer with
    the FileStreamer patch installed and an automatic cap smaller than the
    largest tensor, verifying both call compatibility and real ring sizing."""
    pytest.importorskip("runai_model_streamer")
    import torch
    from runai_model_streamer import SafetensorsStreamer
    from runai_model_streamer.file_streamer import file_streamer as rfs
    from safetensors.torch import save_file

    from vllm_torchtpu import model_loader_patches as mlp

    # Two 16 KiB tensors (32 KiB total); largest tensor is 16 KiB.
    tensor_a = torch.arange(4096, dtype=torch.float32)
    tensor_b = torch.arange(4096, 8192, dtype=torch.float32)
    path = tmp_path / "weights.safetensors"
    save_file({"a": tensor_a, "b": tensor_b}, str(path))
    largest_bytes = tensor_a.numel() * tensor_a.element_size()
    total_bytes = largest_bytes * 2

    monkeypatch.delenv("RUNAI_STREAMER_MEMORY_LIMIT", raising=False)
    monkeypatch.setattr(rfs.FileStreamer, "stream_files", rfs.FileStreamer.stream_files)
    monkeypatch.setattr(
        rfs.FileStreamer, "_tpu_memory_limit_patch", False, raising=False
    )
    mlp.patch_runai_file_streamer_memory_limit()

    # Auto limit (8 KiB) is below the 16 KiB tensor: without flooring this
    # raises RunaiStreamerMemoryLimitException; with flooring it caps the ring
    # to 1 x 16 KiB (< 32 KiB total, unlike the 40 GB default).
    with (
        patch(
            "vllm.config.get_current_vllm_config_or_none",
            return_value=_config_with_auto_runai_limit(largest_bytes // 2),
        ),
        SafetensorsStreamer() as streamer,
    ):
        streamer.stream_files([str(path)], device="cpu", is_distributed=False)
        loaded = dict(streamer.get_tensors())
        ring = streamer.file_streamer.ring_info()

    assert set(loaded) == {"a", "b"}
    assert torch.equal(loaded["a"], tensor_a)
    assert torch.equal(loaded["b"], tensor_b)
    assert ring is not None
    num_buffers, buffer_size = ring[1], ring[2]
    assert buffer_size == largest_bytes
    assert num_buffers == 1
    assert num_buffers * buffer_size < total_bytes
