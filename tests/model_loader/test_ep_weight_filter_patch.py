# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the DefaultModelLoader EP weight-filter patch.

vLLM 0.27.0's ``should_skip_weight`` only matches names ending in
``.weight``, so a compressed-tensors MXFP4 checkpoint (``.weight_packed`` /
``.weight_scale``) reads every expert's bytes on every rank. The TPU patch
widens the suffix allowlist. No TPU, no network.
"""

import torch

from vllm_torchtpu.model_loader_patches import (
    _should_skip_weight_tpu, patch_default_loader_ep_weight_filter)

_LOCAL = {0, 1, 27}


def _expert_name(expert_id: int, suffix: str) -> str:
    return ("language_model.model.layers.3.block_sparse_moe"
            f".experts.{expert_id}.w1{suffix}")


# ---------------------------------------------------------------------------
# _should_skip_weight_tpu
# ---------------------------------------------------------------------------


def test_remote_expert_packed_and_scale_skipped():
    assert _should_skip_weight_tpu(_expert_name(500, ".weight_packed"), _LOCAL)
    assert _should_skip_weight_tpu(_expert_name(500, ".weight_scale"), _LOCAL)
    assert _should_skip_weight_tpu(_expert_name(500, ".weight"), _LOCAL)


def test_local_expert_packed_and_scale_kept():
    assert not _should_skip_weight_tpu(_expert_name(27, ".weight_packed"),
                                       _LOCAL)
    assert not _should_skip_weight_tpu(_expert_name(27, ".weight_scale"),
                                       _LOCAL)


def test_remote_expert_input_scale_kept():
    # NVFP4 per-expert activation scales feed a global-max reduction in
    # FlashInfer backends; they must stay unfiltered on every rank.
    assert not _should_skip_weight_tpu(_expert_name(500, ".input_scale"),
                                       _LOCAL)


def test_no_local_ids_keeps_everything():
    # Filter off (EP weight filter disabled): the upstream contract is that
    # local_expert_ids=None keeps every tensor.
    assert not _should_skip_weight_tpu(_expert_name(500, ".weight_packed"),
                                       None)


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


def test_patch_installs_predicate_and_is_idempotent(monkeypatch):
    from vllm.model_executor.model_loader import ep_weight_filter, weight_utils

    # Register teardown restoration of the attributes the patch overwrites.
    monkeypatch.setattr(weight_utils, "should_skip_weight",
                        weight_utils.should_skip_weight)
    monkeypatch.setattr(ep_weight_filter, "should_skip_weight",
                        ep_weight_filter.should_skip_weight)
    # delattr(raising=False) records nothing when the flag is absent, so it
    # would not undo the stamp the patch sets. setattr registers a teardown
    # that deletes (or restores) it.
    monkeypatch.setattr(weight_utils,
                        "_tpu_ep_filter_patch",
                        False,
                        raising=False)

    patch_default_loader_ep_weight_filter()
    assert weight_utils.should_skip_weight is _should_skip_weight_tpu
    assert ep_weight_filter.should_skip_weight is _should_skip_weight_tpu
    assert weight_utils._tpu_ep_filter_patch is True

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

    monkeypatch.setattr(weight_utils, "should_skip_weight",
                        _should_skip_weight_tpu)
    monkeypatch.setattr(weight_utils, "safe_open", _TrackingSafeOpen)

    yielded = {
        name
        for name, _ in weight_utils.safetensors_weights_iterator(
            [str(shard)], use_tqdm_on_load=False, local_expert_ids={0})
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
