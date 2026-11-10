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
"""Opt-in DiT torch.compile (VLLM_TORCHTPU_OMNI_COMPILE) and its cache key."""

import os
import re
import sys
import types
from importlib import metadata
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.compiler.config as compiler_config

from vllm_torchtpu.omni import attention as omni_attention
from vllm_torchtpu.omni import patches
from vllm_torchtpu.omni import platform as omni_platform
from vllm_torchtpu.omni.platform import OmniTpuPlatform

pytestmark = pytest.mark.omni

_RUNNER = "vllm_omni.diffusion.worker.diffusion_model_runner"
_SPMD_UTIL = "torch_tpu._internal.distributed.spmd_util"
_DEVICE_MESH = "torch_tpu._internal.distributed.device_mesh"
_FLAGS = (
    "LIBTPU_INIT_ARGS",
    "XLA_FLAGS",
    "TORCH_TPU_INTERNAL_XLA_OPTIONS",
    "TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS",
)


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_offsets = []

    def forward(self, hidden_states, temb):
        self.seen_offsets.append(hidden_states.storage_offset())
        return hidden_states + temb


def _dit_cls(name, repeated=("_Block",)):
    def __init__(self, n_blocks=2):
        torch.nn.Module.__init__(self)
        self.blocks = torch.nn.ModuleList(_Block() for _ in range(n_blocks))

    return type(
        name,
        (torch.nn.Module,),
        {"__init__": __init__, "_repeated_blocks": list(repeated)},
    )


def _fake_regionally_compile(compiled):
    """Mimics vLLM-Omni's regionally_compile: block.forward = compile(forward)."""

    def regionally_compile(model, *args, **kwargs):
        for _, block in model.named_modules():
            if type(block).__name__ in model._repeated_blocks:
                inner = block.forward
                compiled.append((inner, kwargs))
                block.forward = lambda *a, _f=inner, **k: _f(*a, **k)
        return model

    return MagicMock(side_effect=regionally_compile)


@pytest.fixture
def runner_module(monkeypatch):
    """A fresh fake diffusion_model_runner module mirroring upstream's gate."""
    module = types.ModuleType(_RUNNER)
    compiled = []
    module.compiled = compiled
    module.regionally_compile = _fake_regionally_compile(compiled)

    class DiffusionModelRunner:
        def __init__(self, pipeline, enforce_eager=False):
            self.pipeline = None
            self._pipeline = pipeline
            self.od_config = SimpleNamespace(
                enforce_eager=enforce_eager,
                diffusion_compile_granularity="regional",
                diffusion_compile_dynamic=True,
            )
            self.upstream_compiled = []

        def _compile_transformer(self, attr_name):
            self.upstream_compiled.append(attr_name)

        def load_model(self):
            # diffusion_model_runner.py:400-421 (vLLM-Omni 0.32.0rc1).
            self.pipeline = self._pipeline
            self.gate_seen = OmniTpuPlatform.supports_torch_inductor()
            if not self.od_config.enforce_eager and self.gate_seen:
                if hasattr(self.pipeline, "setup_compile"):
                    self.pipeline.setup_compile()
                else:
                    for attr_name in self.pipeline._dit_modules:
                        self._compile_transformer(attr_name)

    module.DiffusionModelRunner = DiffusionModelRunner
    monkeypatch.setitem(sys.modules, _RUNNER, module)
    monkeypatch.setattr(patches, "_register_tpu_dynamo_backend", lambda: None)
    monkeypatch.setattr(patches, "_build_dit_ops_eagerly", lambda: None)
    return module


@pytest.fixture
def spmd_util(monkeypatch):
    module = types.ModuleType(_SPMD_UTIL)

    def spmd_safe(fn):
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        wrapper.spmd_safe_inner = fn
        return wrapper

    module.spmd_safe = MagicMock(side_effect=spmd_safe)
    monkeypatch.setitem(sys.modules, _SPMD_UTIL, module)
    return module


@pytest.fixture
def cache_tag():
    with compiler_config.patch(cache_key_tag=""):
        yield


def _pipeline(model, dit_modules=("transformer",)):
    return SimpleNamespace(transformer=model, _dit_modules=list(dit_modules))


# --------------------------------------------------------------------------
# Flag and gate
# --------------------------------------------------------------------------


def test_compile_mode_parsing(monkeypatch):
    monkeypatch.delenv(patches.OMNI_COMPILE_ENV, raising=False)
    assert patches.omni_compile_mode() == "0"
    for mode in ("0", "regional", "full", "Regional", "FULL"):
        monkeypatch.setenv(patches.OMNI_COMPILE_ENV, mode)
        assert patches.omni_compile_mode() == mode.lower()
    monkeypatch.setenv(patches.OMNI_COMPILE_ENV, "inductor")
    with pytest.raises(ValueError, match="VLLM_TORCHTPU_OMNI_COMPILE"):
        patches.omni_compile_mode()


@pytest.mark.parametrize("mode", ["regional", "full"])
def test_gate_open_only_inside_diffusion_load_model(monkeypatch, runner_module, mode):
    monkeypatch.setenv(patches.OMNI_COMPILE_ENV, mode)
    runner = runner_module.DiffusionModelRunner(
        _pipeline(torch.nn.Linear(2, 2), dit_modules=())
    )
    # Outside DiffusionModelRunner.load_model (e.g. VoxCPM2, Qwen3-TTS code
    # predictor in AR stages) the gate stays closed, flag or not.
    assert OmniTpuPlatform.supports_torch_inductor() is False
    runner.load_model()
    assert runner.gate_seen is False  # load_model not wrapped yet
    patches._patch_dit_compile()
    runner.load_model()
    assert runner.gate_seen is True
    assert OmniTpuPlatform.supports_torch_inductor() is False
    # enforce_eager is still honoured by upstream's own check.
    eager_runner = runner_module.DiffusionModelRunner(
        _pipeline(_dit_cls("WanTransformer3DModel")()), enforce_eager=True
    )
    eager_runner.load_model()
    runner_module.regionally_compile.assert_not_called()


def test_gate_closed_for_pipeline_setup_compile(monkeypatch, runner_module):
    monkeypatch.setenv(patches.OMNI_COMPILE_ENV, "regional")
    patches._patch_dit_compile()
    pipeline = _pipeline(torch.nn.Linear(2, 2))
    pipeline.setup_compile = MagicMock()
    runner = runner_module.DiffusionModelRunner(pipeline)
    runner.load_model()
    pipeline.setup_compile.assert_not_called()
    runner_module.regionally_compile.assert_not_called()


def test_flag_off_is_inert(monkeypatch, runner_module):
    monkeypatch.delenv(patches.OMNI_COMPILE_ENV, raising=False)
    model = _dit_cls("WanTransformer3DModel")()
    runner = runner_module.DiffusionModelRunner(_pipeline(model))
    # Unpatched upstream path: supports_torch_inductor() is False.
    runner.load_model()
    assert runner.upstream_compiled == []
    # Even with the wrapper installed, flag "0" never compiles.
    patches._patch_dit_compile()
    runner.load_model()
    patches.tpu_compile_transformer(runner, "transformer")
    runner_module.regionally_compile.assert_not_called()
    assert OmniTpuPlatform.supports_torch_inductor() is False


def test_set_device_installs_compile_only_with_flag(monkeypatch):
    calls = []
    monkeypatch.setattr(
        omni_platform, "_patch_dit_compile", lambda: calls.append("patch")
    )
    monkeypatch.setattr(
        OmniTpuPlatform,
        "configure_diffusion_compile_cache",
        classmethod(lambda cls, world_size=None: calls.append(("cache", world_size))),
    )
    monkeypatch.setattr(
        OmniTpuPlatform,
        "_prepare_singlehost_tpu_env",
        staticmethod(lambda world_size: None),
    )
    monkeypatch.setattr(torch, "accelerator", MagicMock(), raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv(patches.OMNI_COMPILE_ENV, raising=False)
    with patch("torch.empty"), patch("torch.device"):
        OmniTpuPlatform.set_device(None)
    assert calls == []
    monkeypatch.setenv(patches.OMNI_COMPILE_ENV, "regional")
    with patch("torch.empty"), patch("torch.device"):
        OmniTpuPlatform.set_device(None)
    assert calls == ["patch", ("cache", 1)]


def test_patch_is_idempotent(monkeypatch, runner_module):
    monkeypatch.setenv(patches.OMNI_COMPILE_ENV, "regional")
    runner_cls = runner_module.DiffusionModelRunner
    orig_compile = runner_cls._compile_transformer
    patches._patch_dit_compile()
    patched = (runner_cls.load_model, runner_cls._compile_transformer)
    patches._patch_dit_compile()
    assert (runner_cls.load_model, runner_cls._compile_transformer) == patched
    assert runner_cls._tpu_orig_compile_transformer is orig_compile


# --------------------------------------------------------------------------
# _compile_transformer wrapper
# --------------------------------------------------------------------------


def test_regional_compile_uses_tpu_backend(monkeypatch, runner_module):
    monkeypatch.setenv(patches.OMNI_COMPILE_ENV, "regional")
    monkeypatch.delenv(patches.OMNI_COMPILE_SPMD_SAFE_ENV, raising=False)
    patches._patch_dit_compile()
    model = _dit_cls("FluxTransformer2DModel")()
    model.compile = MagicMock()
    runner = runner_module.DiffusionModelRunner(_pipeline(model))
    runner.load_model()
    runner_module.regionally_compile.assert_called_once_with(
        model, backend="tpu", dynamic=False
    )
    model.compile.assert_not_called()
    assert runner.pipeline.transformer is model
    assert runner.upstream_compiled == []


def test_full_compile_uses_tpu_backend(monkeypatch, runner_module):
    monkeypatch.setenv(patches.OMNI_COMPILE_ENV, "full")
    patches._patch_dit_compile()
    model = _dit_cls("WanTransformer3DModel")()
    model.compile = MagicMock()
    runner = runner_module.DiffusionModelRunner(_pipeline(model))
    runner.load_model()
    model.compile.assert_called_once_with(backend="tpu", dynamic=False)
    runner_module.regionally_compile.assert_not_called()


def test_regional_compile_with_real_torch_compile(monkeypatch, runner_module):
    # The kwargs reach torch.compile (mocked) through regionally_compile.
    monkeypatch.setenv(patches.OMNI_COMPILE_ENV, "regional")
    compile_calls = []

    def fake_compile(fn, *args, **kwargs):
        compile_calls.append(kwargs)
        return fn

    def regionally_compile(model, *args, **kwargs):
        for _, block in model.named_modules():
            if type(block).__name__ in model._repeated_blocks:
                block.forward = torch.compile(block.forward, *args, **kwargs)
        return model

    monkeypatch.setattr(torch, "compile", fake_compile)
    runner_module.regionally_compile = regionally_compile
    model = _dit_cls("QwenImageTransformer2DModel")(n_blocks=3)
    runner = runner_module.DiffusionModelRunner(_pipeline(model))
    runner.pipeline = runner._pipeline
    patches.tpu_compile_transformer(runner, "transformer")
    assert compile_calls == [{"backend": "tpu", "dynamic": False}] * 3


def test_compile_failure_keeps_eager_model(
    monkeypatch, runner_module, spmd_util, cache_tag
):
    monkeypatch.setenv(patches.OMNI_COMPILE_ENV, "regional")
    monkeypatch.setenv(patches.OMNI_COMPILE_SPMD_SAFE_ENV, "1")
    runner_module.regionally_compile = MagicMock(side_effect=RuntimeError("boom"))
    model = _dit_cls("WanTransformer3DModel")()
    pipeline = _pipeline(model)
    runner = runner_module.DiffusionModelRunner(pipeline)
    runner.pipeline = pipeline
    warnings = []
    monkeypatch.setattr(patches.logger, "warning", lambda *a: warnings.append(a))
    patches.tpu_compile_transformer(runner, "transformer")
    assert len(warnings) == 1
    # spmd_safe wrappers are removed again.
    assert all("forward" not in block.__dict__ for block in model.blocks)


def test_missing_transformer_attr_is_skipped(monkeypatch, runner_module):
    monkeypatch.setenv(patches.OMNI_COMPILE_ENV, "regional")
    runner = runner_module.DiffusionModelRunner(SimpleNamespace(transformer=None))
    runner.pipeline = runner._pipeline
    patches.tpu_compile_transformer(runner, "transformer_2")
    runner_module.regionally_compile.assert_not_called()


# --------------------------------------------------------------------------
# spmd_safe regions
# --------------------------------------------------------------------------


def _compile_regional(runner_module, model):
    pipeline = _pipeline(model)
    runner = runner_module.DiffusionModelRunner(pipeline)
    runner.pipeline = pipeline
    patches.tpu_compile_transformer(runner, "transformer")
    return runner


def test_spmd_safe_wraps_allow_listed_wan_blocks(
    monkeypatch, runner_module, spmd_util, cache_tag
):
    monkeypatch.setenv(patches.OMNI_COMPILE_ENV, "regional")
    monkeypatch.setenv(patches.OMNI_COMPILE_SPMD_SAFE_ENV, "1")
    model = _dit_cls("WanTransformer3DModel")(n_blocks=3)
    _compile_regional(runner_module, model)
    assert spmd_util.spmd_safe.call_count == 3
    # Wrapped before compile: regionally_compile saw the spmd_safe wrappers.
    for inner, kwargs in runner_module.compiled:
        assert hasattr(inner, "spmd_safe_inner")
        assert kwargs == {"backend": "tpu", "dynamic": False}
    assert patches.SPMD_SAFE_CACHE_TAG in compiler_config.cache_key_tag.split("|")

    # Blocks get offset-0 inputs (Omni SP shards are chunk views).
    full = torch.arange(24.0).view(1, 4, 6)
    shard = full.chunk(2, dim=1)[1]
    assert shard.storage_offset() == 12
    block = model.blocks[0]
    out = block(shard, torch.zeros(1, 2, 6))
    assert block.seen_offsets == [0]
    torch.testing.assert_close(out, shard)


def test_spmd_safe_wraps_hook_registry_original_forward(
    monkeypatch, runner_module, spmd_util, cache_tag
):
    monkeypatch.setenv(patches.OMNI_COMPILE_ENV, "regional")
    monkeypatch.setenv(patches.OMNI_COMPILE_SPMD_SAFE_ENV, "1")
    model = _dit_cls("WanTransformer3DModel")(n_blocks=1)
    block = model.blocks[0]
    original = block.forward
    block._omni_original_forward = original
    fn_ref = SimpleNamespace(original_forward=original)
    block._hook_registry = SimpleNamespace(
        _hooks={"magcache": SimpleNamespace(fn_ref=fn_ref)}
    )
    regions = patches._wrap_spmd_safe_regions(model)
    assert block._omni_original_forward.spmd_safe_inner is original
    # Hooks holding the original callable follow it (regionally_compile then
    # retargets them to the compiled callable by identity).
    assert fn_ref.original_forward is block._omni_original_forward
    assert "forward" not in block.__dict__
    patches._unwrap_regions(regions)
    assert block._omni_original_forward is original
    assert fn_ref.original_forward is original


@pytest.mark.parametrize(
    "model_cls",
    [
        "WanTransformer3DModel",
        "FluxTransformer2DModel",
        "Flux2Transformer2DModel",
        "QwenImageTransformer2DModel",
    ],
)
def test_spmd_safe_wraps_allow_listed_models(
    monkeypatch, runner_module, spmd_util, cache_tag, model_cls
):
    monkeypatch.setenv(patches.OMNI_COMPILE_ENV, "regional")
    monkeypatch.setenv(patches.OMNI_COMPILE_SPMD_SAFE_ENV, "1")
    model = _dit_cls(model_cls)(n_blocks=2)
    _compile_regional(runner_module, model)
    assert spmd_util.spmd_safe.call_count == 2
    assert patches.SPMD_SAFE_CACHE_TAG in compiler_config.cache_key_tag.split("|")
    shard = torch.arange(24.0).view(1, 4, 6).chunk(2, dim=1)[1]
    model.blocks[0](shard, torch.zeros(1, 2, 6))
    assert model.blocks[0].seen_offsets == [0]


def test_flux_regional_compile_includes_single_transformer_blocks(
    monkeypatch, runner_module, spmd_util, cache_tag
):
    monkeypatch.setenv(patches.OMNI_COMPILE_ENV, "regional")
    monkeypatch.setenv(patches.OMNI_COMPILE_SPMD_SAFE_ENV, "1")

    class FluxTransformerBlock(_Block):
        pass

    class FluxSingleTransformerBlock(_Block):
        pass

    class FluxTransformer2DModel(torch.nn.Module):
        _repeated_blocks = ["FluxTransformerBlock"]

        def __init__(self):
            super().__init__()
            self.transformer_blocks = torch.nn.ModuleList(
                FluxTransformerBlock() for _ in range(2)
            )
            self.single_transformer_blocks = torch.nn.ModuleList(
                FluxSingleTransformerBlock() for _ in range(3)
            )

    model = FluxTransformer2DModel()
    _compile_regional(runner_module, model)
    assert model._repeated_blocks == [
        "FluxTransformerBlock",
        "FluxSingleTransformerBlock",
    ]
    assert spmd_util.spmd_safe.call_count == 5
    assert len(runner_module.compiled) == 5
    # Slice view from block 0 (like hidden_states[:, text_seq_len:]) enters
    # block 1 at storage offset 0.
    cat_out = torch.arange(36.0).view(1, 6, 6)
    img_slice = cat_out[:, 2:]
    assert img_slice.storage_offset() == 12
    model.single_transformer_blocks[1](img_slice, torch.zeros(1, 4, 6))
    assert model.single_transformer_blocks[1].seen_offsets == [0]


def test_real_omni_pipeline_setup_compile_gate(monkeypatch):
    from vllm_omni.diffusion.models.flux.pipeline_flux import FluxPipeline
    from vllm_omni.diffusion.models.flux2_klein.pipeline_flux2_klein import (
        Flux2KleinPipeline,
    )
    from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2Pipeline
    from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_audio import (
        LTX2TextToAudioPipeline,
    )
    from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image_edit import (
        QwenImageEditPipeline,
    )

    monkeypatch.setenv(patches.OMNI_COMPILE_ENV, "regional")
    for cls in (
        FluxPipeline,
        Flux2KleinPipeline,
        QwenImageEditPipeline,
        LTX2Pipeline,
    ):
        assert not hasattr(cls, "setup_compile"), cls.__name__
        inst = cls.__new__(cls)
        token = patches._dit_compile_runner.set(SimpleNamespace(pipeline=inst))
        try:
            assert patches.dit_compile_gate_open() is True
        finally:
            patches._dit_compile_runner.reset(token)

    audio_inst = LTX2TextToAudioPipeline.__new__(LTX2TextToAudioPipeline)
    token = patches._dit_compile_runner.set(SimpleNamespace(pipeline=audio_inst))
    try:
        assert patches.dit_compile_gate_open() is False
    finally:
        patches._dit_compile_runner.reset(token)


@pytest.mark.parametrize(
    "model_cls",
    [
        "LTX2VideoTransformer3DModel",
        "UnlistedTransformerModel",
    ],
)
def test_spmd_safe_only_for_allow_listed_models(
    monkeypatch, runner_module, spmd_util, cache_tag, model_cls
):
    monkeypatch.setenv(patches.OMNI_COMPILE_ENV, "regional")
    monkeypatch.setenv(patches.OMNI_COMPILE_SPMD_SAFE_ENV, "1")
    model = _dit_cls(model_cls)()
    _compile_regional(runner_module, model)
    spmd_util.spmd_safe.assert_not_called()
    assert patches.SPMD_SAFE_CACHE_TAG not in compiler_config.cache_key_tag.split("|")
    # No offset-0 shim either.
    shard = torch.arange(24.0).view(1, 4, 6).chunk(2, dim=1)[1]
    model.blocks[0](shard, torch.zeros(1, 2, 6))
    assert model.blocks[0].seen_offsets == [12]


@pytest.mark.parametrize("mode, spmd", [("regional", "0"), ("full", "1")])
def test_spmd_safe_off_without_flag_or_for_full(
    monkeypatch, runner_module, spmd_util, cache_tag, mode, spmd
):
    monkeypatch.setenv(patches.OMNI_COMPILE_ENV, mode)
    monkeypatch.setenv(patches.OMNI_COMPILE_SPMD_SAFE_ENV, spmd)
    model = _dit_cls("WanTransformer3DModel")()
    model.compile = MagicMock()
    _compile_regional(runner_module, model)
    spmd_util.spmd_safe.assert_not_called()
    assert compiler_config.cache_key_tag == ""


# --------------------------------------------------------------------------
# Compile cache key
# --------------------------------------------------------------------------


def _tags(monkeypatch, world_size=1, existing="user-tag", assignment=None, **env):
    device_mesh = types.ModuleType(_DEVICE_MESH)
    device_mesh.get_device_assignment = lambda: (
        list(range(world_size)) if assignment is None else assignment
    )
    monkeypatch.setitem(sys.modules, _DEVICE_MESH, device_mesh)
    monkeypatch.setenv(patches.OMNI_COMPILE_ENV, "regional")
    env_patch = {name: env.get(name, "") for name in _FLAGS}
    with (
        patch.dict(os.environ, env_patch),
        compiler_config.patch(cache_key_tag=existing),
    ):
        OmniTpuPlatform.configure_diffusion_compile_cache(world_size)
        return compiler_config.cache_key_tag.split("|")


def test_cache_key_keeps_user_tag_and_adds_sources_and_flags(monkeypatch):
    user, sources, flags = _tags(monkeypatch)
    assert user == "user-tag"
    assert re.fullmatch(r"tpu-pallas-sources-v1:[0-9a-f]{64}", sources)
    assert re.fullmatch(r"tpu-compiler-v2:[0-9a-f]{16}", flags)
    assert _tags(monkeypatch, existing="")[0].startswith("tpu-pallas-sources-v1:")


def test_cache_key_untouched_with_flag_off(monkeypatch):
    monkeypatch.delenv(patches.OMNI_COMPILE_ENV, raising=False)
    with compiler_config.patch(cache_key_tag="user-tag"):
        OmniTpuPlatform.configure_diffusion_compile_cache(4)
        assert compiler_config.cache_key_tag == "user-tag"


def test_cache_key_rank_assignment_only_for_multi_rank(monkeypatch):
    def rank_tags(tags):
        return [t for t in tags if t.startswith("tpu-rank-assignment-v1:")]

    assert rank_tags(_tags(monkeypatch)) == []
    four = rank_tags(_tags(monkeypatch, world_size=4))
    assert len(four) == 1 and four[0].startswith("tpu-rank-assignment-v1:4:")
    other = rank_tags(_tags(monkeypatch, world_size=4, assignment=[3, 2, 1, 0]))
    assert other != four
    monkeypatch.setenv("RANK", "1")
    assert rank_tags(_tags(monkeypatch, world_size=4)) != four


def test_cache_key_tracks_every_flag(monkeypatch):
    flags_tag = _tags(monkeypatch)[-1]
    for name in _FLAGS:
        assert _tags(monkeypatch, **{name: "--x=1"})[-1] != flags_tag, name


def test_cache_key_tracks_versions_chip_and_topology(monkeypatch):
    flags_tag = _tags(monkeypatch)[-1]
    real_version = metadata.version
    for dist in ("torch_tpu", "libtpu", "jax", "jaxlib"):

        def bumped(name, dist=dist):
            if name == dist:
                return "999.0"
            try:
                return real_version(name)
            except metadata.PackageNotFoundError:
                return ""

        with patch("importlib.metadata.version", side_effect=bumped):
            assert _tags(monkeypatch)[-1] != flags_tag, dist
    with patch.object(
        omni_platform.hardware, "get_tpu_device_name", return_value="TPU v0"
    ):
        assert _tags(monkeypatch)[-1] != flags_tag
    monkeypatch.setenv("TORCH_TPU_TOPOLOGY", "4x4x4")
    assert _tags(monkeypatch)[-1] != flags_tag


def test_cache_key_is_idempotent_and_replaces_stale_entries(monkeypatch):
    once = _tags(monkeypatch, world_size=4)
    twice = _tags(monkeypatch, world_size=4, existing="|".join(once))
    assert twice == once
    # A later call with a new flag value replaces, rather than appends.
    updated = _tags(monkeypatch, world_size=4, existing="|".join(once), XLA_FLAGS="--y")
    assert len(updated) == len(once)
    assert updated[:-1] == once[:-1] and updated[-1] != once[-1]


def test_cache_key_pallas_sources(monkeypatch, tmp_path):
    files = [
        os.path.relpath(p, os.path.dirname(os.path.dirname(omni_platform.__file__)))
        for p in omni_platform._pallas_source_files()
    ]
    assert "omni/rope.py" in files
    assert "omni/attention.py" in files
    assert "kernels/diffusion_attention/kernel.py" in files
    assert "kernels/diffusion_attention/qk_norm_rope.py" in files
    assert "layers/adapter/custom_ops/optimization_barrier_op.py" in files
    source = tmp_path / "kernel.py"
    source.write_text("a = 1\n")
    monkeypatch.setattr(omni_platform, "_pallas_source_files", lambda: [str(source)])
    before = omni_platform._pallas_sources_tag()
    source.write_text("a = 2\n")
    assert omni_platform._pallas_sources_tag() != before


def test_set_compile_cache_tag(cache_tag):
    patches.set_compile_cache_tag("a")
    patches.set_compile_cache_tag("a")
    patches.set_compile_cache_tag("p:1", prefix="p:")
    patches.set_compile_cache_tag("p:2", prefix="p:")
    assert compiler_config.cache_key_tag == "a|p:2"


# --------------------------------------------------------------------------
# Graph-break hygiene
# --------------------------------------------------------------------------


def test_sdpa_fallback_warning_skipped_while_compiling(monkeypatch):
    monkeypatch.setattr(omni_attention, "_flash_attention_op", lambda: object())
    impl = omni_attention.TpuFlashAttentionImpl(
        num_heads=2, head_size=8, causal=True, softmax_scale=8**-0.5
    )
    q = torch.zeros(1, 4, 2, 8, dtype=torch.bfloat16)
    warned = []
    monkeypatch.setattr(
        omni_attention.logger, "warning_once", lambda *a: warned.append(a)
    )
    with patch("torch.compiler.is_compiling", return_value=True):
        assert impl._needs_sdpa_fallback(q, q, None) is True
    assert warned == []
    assert impl._needs_sdpa_fallback(q, q, None) is True
    assert len(warned) == 1


def test_check_env_reads_scopes_compile_cache_vars_to_omni_platform():
    import importlib.util
    import pathlib

    tool_path = (
        pathlib.Path(omni_platform.__file__).resolve().parents[3]
        / "tools"
        / "check_env_reads.py"
    )
    spec = importlib.util.spec_from_file_location("check_env_reads", tool_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    scoped_only = {
        "LIBTPU_INIT_ARGS",
        "TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS",
        "TORCH_TPU_INTERNAL_XLA_OPTIONS",
        "TPU_CHIPS_PER_PROCESS_BOUNDS",
        "TPU_PROCESS_BOUNDS",
        "XLA_FLAGS",
    }
    assert scoped_only.isdisjoint(mod.BOOTSTRAP_VARS)
    assert {
        mod.PACKAGE / "omni" / "platform.py": scoped_only,
    } == mod.SCOPED_BOOTSTRAP_VARS
    assert mod.find_reads() == mod.load_baseline()
