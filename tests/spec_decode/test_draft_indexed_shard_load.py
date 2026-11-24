# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for VLLM_TPU_DRAFT_LOAD_INDEXED_SHARDS."""

from __future__ import annotations

import json
import os
import types
from pathlib import Path
from unittest import mock

import pytest
import torch
from safetensors.torch import save_file
from vllm.config import LoadConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.model_loader import default_loader as dl
from vllm.model_executor.model_loader import runai_streamer_loader as rsl

from vllm_torchtpu import envs
from vllm_torchtpu.model_loader_patches import (
    _filter_draft_safetensors_files,
    draft_weight_load_context,
    patch_draft_indexed_shard_loading,
    patch_runai_sharded_expert_streaming,
)


class _DummyMTPDraft(torch.nn.Module):
    has_own_embed_tokens = False
    has_own_lm_head = False

    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.fc_embedding = torch.nn.Linear(4, 4, bias=False)
        self.model.fc_hidden = torch.nn.Linear(4, 4, bias=False)
        self.model.pre_fc_norm_embedding = torch.nn.Module()
        self.model.pre_fc_norm_embedding.weight = torch.nn.Parameter(torch.zeros(4))
        self.model.layers = torch.nn.ModuleList([torch.nn.Module()])
        self.model.layers[0].self_attn_q = torch.nn.Parameter(torch.zeros(4, 4))

    def load_weights(self, weights):
        loaded = set()
        params = dict(self.named_parameters())
        for name, tensor in weights:
            if not name.startswith("mtp."):
                continue
            mapped = "model." + name.removeprefix("mtp.")
            if mapped in params:
                params[mapped].data.copy_(tensor)
                loaded.add(mapped)
        return loaded


class _DummyMTPWithOwnHead(torch.nn.Module):
    has_own_embed_tokens = True
    has_own_lm_head = True

    def load_weights(self, weights):
        return {name for name, _ in weights}


def _make_sharded_checkpoint(tmp_path: Path) -> tuple[list[str], dict[str, torch.Tensor]]:
    torch.manual_seed(42)
    shards: dict[str, dict[str, torch.Tensor]] = {}
    for i in range(6):
        shards[f"model-{i + 1:05d}-of-00010.safetensors"] = {
            f"model.layers.{i}.self_attn.q_proj.weight": torch.randn(4, 4),
            f"model.layers.{i}.mlp.gate_proj.weight": torch.randn(4, 4),
        }
    shards["model-00007-of-00010.safetensors"] = {
        "model.embed_tokens.weight": torch.randn(8, 4),
    }
    shards["model-00008-of-00010.safetensors"] = {
        "lm_head.weight": torch.randn(8, 4),
    }
    expected_mtp = {
        "mtp.fc_embedding.weight": torch.randn(4, 4),
        "mtp.pre_fc_norm_embedding.weight": torch.randn(4),
        "mtp.fc_hidden.weight": torch.randn(4, 4),
        "mtp.layers.0.self_attn_q": torch.randn(4, 4),
    }
    shards["model-00009-of-00010.safetensors"] = {
        "mtp.fc_embedding.weight": expected_mtp["mtp.fc_embedding.weight"],
        "mtp.pre_fc_norm_embedding.weight": expected_mtp[
            "mtp.pre_fc_norm_embedding.weight"
        ],
    }
    shards["model-00010-of-00010.safetensors"] = {
        "mtp.fc_hidden.weight": expected_mtp["mtp.fc_hidden.weight"],
        "mtp.layers.0.self_attn_q": expected_mtp["mtp.layers.0.self_attn_q"],
    }

    weight_map: dict[str, str] = {}
    shard_paths: list[str] = []
    for fname, tensors in sorted(shards.items()):
        fpath = tmp_path / fname
        save_file(tensors, str(fpath))
        shard_paths.append(str(fpath))
        for key in tensors:
            weight_map[key] = fname

    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}),
        encoding="utf-8",
    )
    return shard_paths, expected_mtp


def _make_model_config(model_dir: str, runner_type: str):
    return types.SimpleNamespace(
        model=model_dir,
        revision=None,
        quantization=None,
        multimodal_config=None,
        model_weights=None,
        runner_type=runner_type,
        is_moe=False,
    )


def test_default_flag_off(monkeypatch):
    monkeypatch.delenv("VLLM_TPU_DRAFT_LOAD_INDEXED_SHARDS", raising=False)
    assert envs.VLLM_TPU_DRAFT_LOAD_INDEXED_SHARDS is False


def test_default_loader_filters_target_shards_and_matches_unfiltered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    patch_draft_indexed_shard_loading()
    all_shards, expected_mtp = _make_sharded_checkpoint(tmp_path)
    expected_draft_shards = [all_shards[8], all_shards[9]]
    model_cfg = _make_model_config(str(tmp_path), runner_type="draft")

    seen_off: list[list[str]] = []
    seen_on: list[list[str]] = []
    opened_files_on: list[str] = []

    orig_st_iter = dl.safetensors_weights_iterator

    def _spy_iter(files, *args, sink=None, track_files=False, **kwargs):
        sink.append(list(files))
        if track_files:
            opened_files_on.extend(files)
        return orig_st_iter(files, *args, **kwargs)

    with set_current_vllm_config(VllmConfig()):
        # 1. Baseline: VLLM_TPU_DRAFT_LOAD_INDEXED_SHARDS=0 loads all 10 shards.
        monkeypatch.setenv("VLLM_TPU_DRAFT_LOAD_INDEXED_SHARDS", "0")
        draft_off = _DummyMTPDraft()
        loader_off = dl.DefaultModelLoader(LoadConfig())
        with (
            mock.patch.object(
                dl,
                "safetensors_weights_iterator",
                side_effect=lambda files, *a, **kw: _spy_iter(
                    files, *a, sink=seen_off, **kw
                ),
            ),
            draft_weight_load_context(
                shared_keys=("embed_tokens", "lm_head"), speculative_method="mtp"
            ),
        ):
            loader_off.load_weights(draft_off, model_cfg)

        assert len(seen_off) == 1
        assert sorted(seen_off[0]) == sorted(all_shards)

        # 2. Enabled: VLLM_TPU_DRAFT_LOAD_INDEXED_SHARDS=1 reads only the 2 MTP shards.
        monkeypatch.setenv("VLLM_TPU_DRAFT_LOAD_INDEXED_SHARDS", "1")
        draft_on = _DummyMTPDraft()
        loader_on = dl.DefaultModelLoader(LoadConfig())
        with (
            mock.patch.object(
                dl,
                "safetensors_weights_iterator",
                side_effect=lambda files, *a, **kw: _spy_iter(
                    files, *a, sink=seen_on, track_files=True, **kw
                ),
            ),
            draft_weight_load_context(
                shared_keys=("embed_tokens", "lm_head"), speculative_method="mtp"
            ),
        ):
            loader_on.load_weights(draft_on, model_cfg)

        assert len(seen_on) == 1
        assert sorted(seen_on[0]) == sorted(expected_draft_shards)
        assert sorted(opened_files_on) == sorted(expected_draft_shards)

        # Bit-identical weights loaded into the draft module.
        params_off = dict(draft_off.named_parameters())
        params_on = dict(draft_on.named_parameters())
        assert set(params_off) == set(params_on)
        for name in params_off:
            assert torch.equal(params_off[name], params_on[name]), name
        assert torch.equal(
            params_on["model.fc_embedding.weight"], expected_mtp["mtp.fc_embedding.weight"]
        )

        # 3. Target model (runner_type='generate') still reads all 10 shards.
        seen_target: list[list[str]] = []
        target_cfg = _make_model_config(str(tmp_path), runner_type="generate")
        target_dummy = _DummyMTPWithOwnHead()
        loader_target = dl.DefaultModelLoader(LoadConfig())
        with mock.patch.object(
            dl,
            "safetensors_weights_iterator",
            side_effect=lambda files, *a, **kw: _spy_iter(
                files, *a, sink=seen_target, **kw
            ),
        ):
            loader_target.load_weights(target_dummy, target_cfg)
        assert len(seen_target) == 1
        assert sorted(seen_target[0]) == sorted(all_shards)


@pytest.mark.parametrize("local_expert_ids", [None, {0}])
def test_runai_loader_filters_draft_shards_both_ep_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, local_expert_ids
):
    patch_runai_sharded_expert_streaming()
    patch_draft_indexed_shard_loading()
    all_shards, expected_mtp = _make_sharded_checkpoint(tmp_path)
    expected_draft_shards = [all_shards[8], all_shards[9]]

    loader = object.__new__(rsl.RunaiModelStreamerLoader)
    loader.load_config = LoadConfig()
    loader._is_distributed = False

    monkeypatch.setenv("VLLM_TPU_DRAFT_LOAD_INDEXED_SHARDS", "1")
    draft_model = _DummyMTPDraft()
    draft_cfg = _make_model_config(str(tmp_path), runner_type="draft")

    seen_files: list[list[str]] = []
    from vllm_torchtpu import model_loader_patches as mlp

    def _fake_sharded_iter(files, expert_ids, use_tqdm, model):
        seen_files.append(list(files))
        for p in files:
            from safetensors.torch import load_file
            for k, v in load_file(p).items():
                yield k, v

    def _fake_rsl_iter(files, use_tqdm=True, *a, **kw):
        seen_files.append(list(files))
        for p in files:
            from safetensors.torch import load_file
            for k, v in load_file(p).items():
                yield k, v

    with (
        mock.patch.object(
            mlp, "_compute_local_expert_ids", return_value=local_expert_ids
        ),
        mock.patch.object(
            mlp, "_sharded_runai_weights_iterator", side_effect=_fake_sharded_iter
        ),
        mock.patch.object(
            rsl, "list_safetensors", side_effect=lambda path: sorted(all_shards)
        ),
        mock.patch.object(
            rsl, "runai_safetensors_weights_iterator", side_effect=_fake_rsl_iter
        ),
        draft_weight_load_context(
            shared_keys=("embed_tokens", "lm_head"), speculative_method="mtp"
        ),
    ):
        # Check _prepare_weights path (used by BOTH local_expert_ids=None and ={0}).
        loader._draft_shard_filter_state = (draft_model, draft_cfg)
        prep = loader._prepare_weights(str(tmp_path), None)
        assert sorted(prep) == sorted(expected_draft_shards)

        # Also exercise RunaiModelStreamerLoader.load_weights -> _get_weights_iterator
        # on local_expert_ids={0} (_sharded_runai_weights_iterator) and on None.
        if local_expert_ids is not None:
            loader.load_weights(draft_model, draft_cfg)
            assert len(seen_files) == 1
            assert sorted(seen_files[0]) == sorted(expected_draft_shards)
            assert torch.equal(
                draft_model.model.fc_embedding.weight, expected_mtp["mtp.fc_embedding.weight"]
            )
        else:
            loader.load_weights(draft_model, draft_cfg)
            assert len(seen_files) == 1
            assert sorted(seen_files[0]) == sorted(expected_draft_shards)
            assert torch.equal(
                draft_model.model.fc_embedding.weight, expected_mtp["mtp.fc_embedding.weight"]
            )

        # Target model keeps all 10 shards through _prepare_weights.
        loader._draft_shard_filter_state = (
            _DummyMTPWithOwnHead(),
            _make_model_config(str(tmp_path), runner_type="generate"),
        )
        prep_target = loader._prepare_weights(str(tmp_path), None)
        assert sorted(prep_target) == sorted(all_shards)

def test_shared_keys_and_fallbacks(tmp_path: Path):
    all_shards, _ = _make_sharded_checkpoint(tmp_path)
    own_head_model = _DummyMTPWithOwnHead()

    # Unshared embed/lm_head -> shards 7 & 8 kept alongside shards 9 & 10.
    with draft_weight_load_context(shared_keys=(), speculative_method="mtp"):
        kept_unshared = _filter_draft_safetensors_files(
            all_shards,
            hf_folder=str(tmp_path),
            index_file="model.safetensors.index.json",
            model_or_path=str(tmp_path),
            model=own_head_model,
        )
    assert sorted(kept_unshared) == sorted(all_shards[6:10])

    # Shared embed/lm_head -> only shards 9 & 10 kept.
    with draft_weight_load_context(
        shared_keys=("embed_tokens", "lm_head"), speculative_method="mtp"
    ):
        kept_shared = _filter_draft_safetensors_files(
            all_shards,
            hf_folder=str(tmp_path),
            index_file="model.safetensors.index.json",
            model_or_path=str(tmp_path),
            model=own_head_model,
        )
    assert sorted(kept_shared) == sorted(all_shards[8:10])

    # Fallback when index has no mtp.* keys (e.g. standalone EAGLE3 repo).
    standalone_index = tmp_path / "standalone.index.json"
    standalone_index.write_text(
        json.dumps(
            {
                "weight_map": {
                    "midlayer.q_proj.weight": os.path.basename(all_shards[0]),
                    "fc.weight": os.path.basename(all_shards[1]),
                }
            }
        ),
        encoding="utf-8",
    )
    assert (
        _filter_draft_safetensors_files(
            all_shards,
            hf_folder=str(tmp_path),
            index_file="standalone.index.json",
            model_or_path=str(tmp_path),
            model=_DummyMTPDraft(),
        )
        == all_shards
    )

    # Fallback when index file is missing.
    assert (
        _filter_draft_safetensors_files(
            all_shards,
            hf_folder=str(tmp_path),
            index_file="missing.index.json",
            model_or_path=str(tmp_path),
            model=_DummyMTPDraft(),
        )
        == all_shards
    )

