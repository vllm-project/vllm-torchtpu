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
    _runai_weight_prefix,
    _sharded_runai_weights_iterator,
    _should_skip,
)

# ---------------------------------------------------------------------------
# _should_skip
# ---------------------------------------------------------------------------

_LOCAL = {0, 1, 27}


@pytest.mark.parametrize(
    "runner_type,architectures,expected",
    [
        ("draft", ["DSparkV41DraftModel"], "mtp."),
        ("generate", ["DeepseekV41ForCausalLM"], None),
        ("generate", ["DSparkV41DraftModel"], None),
        ("draft", ["OtherDraftModel"], None),
        ("draft", None, None),
    ],
)
def test_only_v41_dspark_filters_the_checkpoint_namespace(
    runner_type, architectures, expected
):
    config = SimpleNamespace(
        runner_type=runner_type,
        hf_config=SimpleNamespace(architectures=architectures),
    )
    assert _runai_weight_prefix(config) == expected


@pytest.mark.parametrize(
    "runner_type,expected_experts", [("generate", 896), ("draft", 128)]
)
def test_local_expert_ids_use_loader_ownership_mapping(runner_type, expected_experts):
    """Run:AI filtering must honor hierarchical EP's loader patch."""
    model_config = SimpleNamespace(
        is_moe=True,
        get_num_experts=lambda: 896,
        runner_type=runner_type,
        hf_config=SimpleNamespace(dspark_n_routed_experts=128),
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

    compute_ids.assert_called_once_with(expected_experts, 32, 11, placement="linear")


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


@pytest.mark.parametrize("weight_prefix", [None, "mtp."])
def test_request_plan_groups_contiguous_kept_runs(weight_prefix):
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
            ("mtp.0.ffn.experts.0.w1.weight", 10),  # kept @1031
            ("mtp.0.ffn.experts.5.w1.weight", 10),  # skip @1041
            ("mtp.0.main_proj.weight", 4),  # kept @1051
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
                weight_prefix=weight_prefix,
            )
        )

    requests = captured["requests"]
    if weight_prefix is None:
        expected = [
            ([1000, 1010], [10, 2]),
            ([1024, 1031], [7, 10]),
            ([1051], [4]),
        ]
    else:
        expected = [([1031], [10]), ([1051], [4])]
    assert [(r.offsets, r.sizes) for r in requests] == expected
    total_requested = sum(sum(r.sizes) for r in requests)
    assert total_requested == (33 if weight_prefix is None else 14)


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


def test_row_sliced_tensor_fetches_only_its_rows():
    """A tensor with a registered row slice becomes its own request over just
    those rows' bytes, breaks the surrounding run, and is yielded with the
    sliced row count."""
    # The Run:AI streamer ships with PR #331; skip until it is in the image.
    pytest.importorskip("runai_model_streamer")
    import torch

    from vllm_torchtpu import model_loader_patches as mlp

    file_meta = _fake_file_meta(
        offset=100,
        names_and_sizes=[
            ("layers.1.engram.q_weight", 8),  # kept @100
            ("layers.1.engram.embed.weight", 40),  # 10 rows x 4 B @108
            ("layers.1.engram.k_weight", 8),  # kept @148
        ],
    )
    file_meta.tensors_metadata[1].shape = [10, 4]

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
            for request in captured["requests"]:
                for index, size in enumerate(request.sizes):
                    yield request.id, index, torch.empty(size, dtype=torch.uint8)

    import runai_model_streamer.safetensors_streamer.safetensors_pytorch as sp
    from runai_model_streamer.safetensors_streamer import safetensors_streamer

    fake_metadata = MagicMock()
    fake_metadata.from_files.return_value = [file_meta]

    def fake_create(buffer, tensor_meta):
        shape = getattr(tensor_meta, "shape", [buffer.numel()])
        return buffer.view(shape)

    with (
        patch.dict(mlp._ROW_SLICES, clear=True),
        patch.object(safetensors_streamer, "SafetensorsStreamer", FakeStreamer),
        patch.object(sp, "SafetensorsMetadata", fake_metadata),
        patch.object(sp, "create_torch_tensor", fake_create),
    ):
        mlp.register_row_slice("layers.1.engram.embed.weight", 3, 5)
        yielded = {
            name: tuple(tensor.shape)
            for name, tensor in _sharded_runai_weights_iterator(
                ["gs://bucket/file0.safetensors"],
                local_expert_ids=set(),
                use_tqdm_on_load=False,
            )
        }

    assert [(r.offsets, r.sizes) for r in captured["requests"]] == [
        ([100], [8]),
        ([108 + 3 * 4], [5 * 4]),  # rows 3..7 only
        ([148], [8]),
    ]
    assert yielded["layers.1.engram.embed.weight"] == (5, 4)
    # The file's metadata keeps the full shape for any other consumer.
    assert file_meta.tensors_metadata[1].shape == [10, 4]


def test_row_slice_matches_dotted_suffix_only():
    from vllm_torchtpu import model_loader_patches as mlp

    with patch.dict(mlp._ROW_SLICES, clear=True):
        mlp.register_row_slice("layers.1.engram.embed.weight", 0, 2)
        assert mlp._row_slice_for("layers.1.engram.embed.weight") == (0, 2)
        assert mlp._row_slice_for("model.layers.1.engram.embed.weight") == (0, 2)
        assert mlp._row_slice_for("layers.11.engram.embed.weight") is None
        assert mlp._row_slice_for("layers.1.engram.embed.scale") is None


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

    with patch.object(rsl, "RunaiModelStreamerLoader", FakeLoader):
        mlp.patch_runai_sharded_expert_streaming()
        patched_once = FakeLoader._get_weights_iterator
        mlp.patch_runai_sharded_expert_streaming()  # idempotent
        assert FakeLoader._get_weights_iterator is patched_once

        with patch.object(mlp, "_compute_local_expert_ids", return_value=None):
            loader = FakeLoader()
            FakeLoader._get_weights_iterator(loader, "gs://b/m", None)
        original.assert_called_once_with(loader, "gs://b/m", None)
