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
from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest import mock

import numpy as np
import pytest
import torch

from vllm_torchtpu.megakernel.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.runtime.contract import (
    DOMAIN_KERNEL_MESH,
    DOMAIN_TP,
    DOMAIN_TP_X_DCP,
    TpuMonoSpec,
)
from vllm_torchtpu.megakernel_runtime import (
    agreement as agreement_mod,
)
from vllm_torchtpu.megakernel_runtime import (
    config as config_mod,
)
from vllm_torchtpu.megakernel_runtime import (
    driver as driver_mod,
)
from vllm_torchtpu.megakernel_runtime import (
    guards as guards_mod,
)
from vllm_torchtpu.megakernel_runtime import (
    hooks as hooks_mod,
)
from vllm_torchtpu.megakernel_runtime.plugin import RunnerView, Shorthand
from vllm_torchtpu.megakernel_runtime.testing import fake_runner

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


@pytest.mark.parametrize(
    ("domain", "tp", "dp", "expected_ranks"),
    [
        (DOMAIN_TP, 32, 1, 32),
        (DOMAIN_TP, 8, 4, 8),
        (DOMAIN_TP_X_DCP, 32, 1, 32),
        (DOMAIN_TP_X_DCP, 8, 4, 8),
        (DOMAIN_TP_X_DCP, 16, 2, 16),
        (DOMAIN_KERNEL_MESH, 32, 1, 32),
        (DOMAIN_KERNEL_MESH, 8, 4, 32),
        (DOMAIN_KERNEL_MESH, 16, 2, 32),
    ],
)
def test_collective_domain_world_sizes_and_agreements(
    domain: str, tp: int, dp: int, expected_ranks: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify domain world size derivations, agreement routing, and mesh checks."""
    runner = fake_runner(tp_size=tp)
    runner.parallel_config.data_parallel_size = dp
    runner.vllm_config.parallel_config = runner.parallel_config
    view = RunnerView(runner, tp_rank=0)
    assert guards_mod.domain_world_size(domain, runner, view) == expected_ranks

    parallelism = frozenset({"tp", "dp"}) if dp > 1 else frozenset({"tp"})
    spec = TpuMonoSpec(
        "s_test", "1", create="m:c", collective_domain=domain, parallelism=parallelism
    )
    assert guards_mod.check_spec(runner, spec, view=view) is None

    fake_tp = object()
    fake_world = object()
    monkeypatch.setattr(agreement_mod, "_tp_cpu_group", lambda: fake_tp)
    monkeypatch.setattr(agreement_mod, "_world_cpu_group", lambda: fake_world)
    expected_group = fake_world if domain == DOMAIN_KERNEL_MESH else fake_tp
    assert agreement_mod.agreement_group(domain, runner) is expected_group

    monkeypatch.setattr(driver_mod, "_get_pallas_mesh_fn", lambda: None)
    import jax

    devs = jax.devices()
    mesh = jax.sharding.Mesh(np.asarray([devs[0]]), ("tp",))
    mesh_built = driver_mod._build_tp_rank_mesh(1, fallback_mesh=mesh)
    assert driver_mod.check_mesh_size(1, int(mesh_built.devices.size)) is None


class _DummyObsModel(torch.nn.Module):
    """Dummy model recording loaded weights."""

    def __init__(self) -> None:
        super().__init__()
        self.loaded_weights: dict[str, torch.Tensor] = {}

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        for name, tensor in weights:
            self.loaded_weights[name] = tensor


class _DummyObsLoader:
    """Dummy loader returning raw checkpoint weights."""

    def __init__(self, raw: list[tuple[str, torch.Tensor]]) -> None:
        self.raw = raw

    def get_all_weights(
        self, model_config: object, model: object
    ) -> Iterator[tuple[str, torch.Tensor]]:
        del model_config, model
        yield from self.raw

    def load_model(self, vllm_config: object, model_config: object) -> _DummyObsModel:
        del vllm_config
        model = _DummyObsModel()
        model.load_weights(self.get_all_weights(model_config, model))
        return model


def _runner_load_model() -> Any:
    """`TPUModelRunner.load_model`; compiled from source without torch_tpu."""
    import ast
    import contextlib
    import logging
    import textwrap
    from collections.abc import Callable, Iterator
    from typing import cast

    import vllm_torchtpu
    from vllm_torchtpu import megakernel_runtime
    from vllm_torchtpu.megakernel_runtime.hooks import load_tensor_stream

    path = Path(vllm_torchtpu.__file__).parent / "runner" / "tpu_runner.py"
    src = path.read_text(encoding="utf-8")
    fn = next(
        n
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.FunctionDef) and n.name == "load_model"
    )
    fn_src = textwrap.dedent(ast.get_source_segment(src, fn) or "")

    @contextlib.contextmanager
    def _noop_context(*_args: object, **_kwargs: object) -> Iterator[None]:
        yield

    orig_device = torch.device

    def fake_device(d: object) -> torch.device:
        return orig_device("cpu") if str(d) == "tpu" else orig_device(str(d))

    class _FakeTorch:
        def __getattr__(self, name: str) -> Any:
            if name == "device":
                return fake_device
            return getattr(torch, name)

    class _FallbackNamespace:
        def __getattr__(self, name: str) -> Any:
            return False

    ns: dict[str, Any] = {
        "torch": _FakeTorch(),
        "logger": logging.getLogger("tpu_runner"),
        "get_tpu_quantization_config": lambda *_: None,
        "get_model_loader": lambda cfg: getattr(cfg, "_loader", None),
        "load_tensor_stream": load_tensor_stream,
        "set_vllm_model_wrapper_context": _noop_context,
        "set_current_vllm_config": _noop_context,
        "cast": cast,
        "Callable": Callable,
        "Iterator": Iterator,
        "validate_online_fp8": lambda *_: None,
        "envs": _FallbackNamespace(),
        "runner_utils": _FallbackNamespace(),
        "evict_loaded_weights_page_cache": lambda *_: None,
        "_find_topk_indices_buffer": lambda *_: None,
        "greedy_top_tokens": _FallbackNamespace(),
        "arm_runner_models": lambda *_: None,
        "megakernel_runtime": megakernel_runtime,
        "token_padding": _FallbackNamespace(),
        "VllmCompressedTensorsW4ANMxfp4MoEMethod": object(),
        "DFlashProposer": object(),
    }
    exec(compile(fn_src, str(path), "exec"), ns)
    return ns["load_model"]


def _verify_tpu_runner_observe_model_loader() -> None:
    """Verify TPUModelRunner model loading wrapper with load_tensor hook."""
    load_model = _runner_load_model()

    observed_in_runner: list[str] = []

    def runner_load_tensor(name: str, tensor: torch.Tensor, _options: object) -> object:
        observed_in_runner.append(name)
        if name == "layer.0.weight":
            return name, tensor * 2
        return None

    mock_runner_plugin = SimpleNamespace(
        load_tensor=runner_load_tensor, layout_request=None
    )
    raw_checkpoint_weights = [
        ("layer.0.w13_weight_scale_2", torch.tensor([0.5, 1.5])),
        ("layer.0.weight", torch.tensor([2.0, 3.0])),
    ]
    loader = _DummyObsLoader(raw_checkpoint_weights)
    runner_vllm_cfg = SimpleNamespace(
        additional_config={"megakernel": {"enabled": True}},
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["ArchTest"]),
            quantization=None,
        ),
        load_config=SimpleNamespace(_loader=loader),
        mesh=None,
    )

    with (
        mock.patch(
            "vllm_torchtpu.megakernel_runtime.hooks.select_plugin",
            return_value=(
                SimpleNamespace(enabled=True),
                (config_mod.PluginRef("m"), {}),
            ),
        ),
        mock.patch(
            "vllm_torchtpu.megakernel_runtime.hooks.import_plugin",
            return_value=(mock_runner_plugin, ModuleType("dummy")),
        ),
    ):
        runner_instance = SimpleNamespace(
            vllm_config=runner_vllm_cfg,
            model_config=runner_vllm_cfg.model_config,
            load_config=runner_vllm_cfg.load_config,
            device_config=SimpleNamespace(device=SimpleNamespace(type="cpu")),
            mesh=None,
            is_pooling_model=False,
            _is_async_drafter=False,
            _initialize_pallas_kernels=lambda: None,
            parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        )
        load_model(runner_instance)

    assert observed_in_runner == [
        "layer.0.w13_weight_scale_2",
        "layer.0.weight",
    ]
    loaded = runner_instance.model.loaded_weights
    assert "layer.0.w13_weight_scale_2" in loaded
    assert torch.equal(loaded["layer.0.weight"], torch.tensor([4.0, 6.0]))


def test_load_tensor_hook_streams_raw_tensors_and_check_spec() -> None:
    """Verify weight observation hook streaming and replacement."""
    observed: list[str] = []

    def hook_fn(name: str, tensor: torch.Tensor, _options: object) -> object:
        observed.append(name)
        if name == "layer.0.weight":
            return name, tensor * 2
        return None

    mock_plugin = SimpleNamespace(load_tensor=hook_fn, layout_request=None)
    vllm_config = SimpleNamespace(
        additional_config={"megakernel": {"enabled": True}},
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["ArchTest"])
        ),
    )
    weights = [
        ("layer.0.w13_weight_scale_2", torch.tensor([1.0, 2.0])),
        ("layer.0.weight", torch.tensor([3.0, 4.0])),
    ]

    with mock.patch("vllm_torchtpu.megakernel_runtime.hooks.select_plugin") as mock_sel:
        mock_sel.return_value = (
            SimpleNamespace(enabled=True),
            (config_mod.PluginRef("m"), {}),
        )
        with mock.patch(
            "vllm_torchtpu.megakernel_runtime.hooks.import_plugin"
        ) as mock_imp:
            mock_imp.return_value = (mock_plugin, ModuleType("dummy"))
            streamed = list(hooks_mod.load_tensor_stream(vllm_config, weights))

    assert observed == ["layer.0.w13_weight_scale_2", "layer.0.weight"]
    assert len(streamed) == 2
    assert streamed[0][0] == "layer.0.w13_weight_scale_2"
    assert torch.equal(streamed[0][1], torch.tensor([1.0, 2.0]))
    assert streamed[1][0] == "layer.0.weight"
    assert torch.equal(streamed[1][1], torch.tensor([6.0, 8.0]))

    disabled_cfg = SimpleNamespace(
        additional_config={"megakernel": {"enabled": False}},
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["ArchTest"])
        ),
    )
    assert list(hooks_mod.load_tensor_stream(disabled_cfg, weights)) == weights

    _verify_tpu_runner_observe_model_loader()


def test_layout_request_presizing_and_validation() -> None:
    """Verify layout request presizing, caching, and input validation."""
    # Validation errors
    with pytest.raises(ValueError, match="unknown keys"):
        hooks_mod._validate_layout_request({"unknown_key": 16})
    with pytest.raises(ValueError, match="must be a positive int"):
        hooks_mod._validate_layout_request({"block_size": 0})
    with pytest.raises(ValueError, match="must be a positive int"):
        hooks_mod._validate_layout_request({"block_size": -1})
    with pytest.raises(ValueError, match="must be a positive int"):
        hooks_mod._validate_layout_request({"block_size": True})
    with pytest.raises(ValueError, match="must be a positive int"):
        hooks_mod._validate_layout_request({"block_size": "16"})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="page_size and block_size disagree"):
        hooks_mod._validate_layout_request({"page_size": 16, "block_size": 32})

    def fake_layout_request(_vllm_config: object, _options: object) -> dict[str, int]:
        return {"block_size": 128, "mamba_slot_tokens": 640}

    mock_plugin = SimpleNamespace(layout_request=fake_layout_request, load_tensor=None)
    vllm_config = SimpleNamespace(
        additional_config={"megakernel": {"enabled": True}},
        cache_config=SimpleNamespace(
            block_size=16,
            user_specified_block_size=False,
            cache_dtype=torch.bfloat16,
        ),
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            get_num_kv_heads=lambda *_: 8,
            get_head_size=lambda: 128,
            is_hybrid=False,
            hf_config=SimpleNamespace(architectures=["ArchTest"]),
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )

    with mock.patch("vllm_torchtpu.megakernel_runtime.hooks.select_plugin") as mock_sel:
        mock_sel.return_value = (
            SimpleNamespace(enabled=True),
            (config_mod.PluginRef("m"), {}),
        )
        with mock.patch(
            "vllm_torchtpu.megakernel_runtime.hooks.import_plugin"
        ) as mock_imp:
            mock_imp.return_value = (mock_plugin, ModuleType("dummy"))
            req = hooks_mod.apply_layout_request(vllm_config)

    assert req == {"block_size": 128, "mamba_slot_tokens": 640}
    assert vllm_config.cache_config.block_size == 128

    # user_specified_block_size wins
    user_cfg = SimpleNamespace(
        additional_config={"megakernel": {"enabled": True}},
        cache_config=SimpleNamespace(block_size=16, user_specified_block_size=True),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["ArchTest"])
        ),
    )
    with mock.patch("vllm_torchtpu.megakernel_runtime.hooks.select_plugin") as mock_sel:
        mock_sel.return_value = (
            SimpleNamespace(enabled=True),
            (config_mod.PluginRef("m"), {}),
        )
        with mock.patch(
            "vllm_torchtpu.megakernel_runtime.hooks.import_plugin"
        ) as mock_imp:
            mock_imp.return_value = (mock_plugin, ModuleType("dummy"))
            hooks_mod.apply_layout_request(user_cfg)
    assert user_cfg.cache_config.block_size == 16

    # disabled gives None
    disabled_cfg = SimpleNamespace(
        additional_config={"megakernel": {"enabled": False}},
        cache_config=SimpleNamespace(block_size=16),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["ArchTest"])
        ),
    )
    assert hooks_mod.apply_layout_request(disabled_cfg) is None

    # dcp_interleave_size validation
    bad_dcp_cfg = SimpleNamespace(
        additional_config={"megakernel": {"enabled": True}},
        cache_config=SimpleNamespace(block_size=16, user_specified_block_size=False),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["ArchTest"])
        ),
    )
    with mock.patch("vllm_torchtpu.megakernel_runtime.hooks.select_plugin") as mock_sel:
        mock_sel.return_value = (
            SimpleNamespace(enabled=True),
            (config_mod.PluginRef("m"), {}),
        )
        bad_dcp_plugin = SimpleNamespace(
            layout_request=lambda *_: {"block_size": 16, "dcp_interleave_size": 12},
            load_tensor=None,
        )
        with mock.patch(
            "vllm_torchtpu.megakernel_runtime.hooks.import_plugin"
        ) as mock_imp:
            mock_imp.return_value = (bad_dcp_plugin, ModuleType("dummy"))
            with pytest.raises(ValueError, match="divisible by dcp_interleave_size"):
                hooks_mod.apply_layout_request(bad_dcp_cfg)

    # Block slot derivation with floor
    block_utils_mod = importlib.import_module(
        "vllm_torchtpu.platforms.tpu_block_size_utils"
    )
    derive_fn = block_utils_mod._derive_tpu_block_slot_config

    class FakeBackend:
        @staticmethod
        def get_name() -> str:
            return "TEST"

        @staticmethod
        def get_min_page_size(_vllm_config: object) -> int:
            return 16

        @staticmethod
        def get_supported_kernel_block_sizes() -> list[int]:
            return [16]

        @staticmethod
        def get_kv_cache_page_size_bytes(
            block_size: int, *_args: object, **_kwargs: object
        ) -> int:
            return block_size * 1024

    d_640 = derive_fn(vllm_config, FakeBackend, input_block_size=128)
    assert d_640["final_block_size"] == 640


def test_shorthand_hf_fields_disambiguation() -> None:
    """Verify shorthand disambiguation based on Hugging Face config fields."""
    sh_base = Shorthand(
        module="m.base",
        architectures=("ArchA",),
        model_types=(),
        legacy_keys=frozenset(),
        hf_fields=(),
    )
    sh_spec = Shorthand(
        module="m.spec",
        architectures=("ArchA",),
        model_types=(),
        legacy_keys=frozenset(),
        hf_fields=(("hidden_size", 8192), ("num_hidden_layers", 92)),
    )

    with mock.patch.dict(
        config_mod.SHORTHANDS, {"base": sh_base, "special": sh_spec}, clear=True
    ):
        hf_match = SimpleNamespace(
            architectures=["ArchA"], hidden_size=8192, num_hidden_layers=92
        )
        assert config_mod.shorthand_for_arch(hf_match) == "special"

        hf_nested = SimpleNamespace(
            architectures=["ArchA"],
            text_config=SimpleNamespace(hidden_size=8192, num_hidden_layers=92),
        )
        assert config_mod.shorthand_for_arch(hf_nested) == "special"

        hf_non_match = SimpleNamespace(
            architectures=["ArchA"], hidden_size=4096, num_hidden_layers=60
        )
        assert config_mod.shorthand_for_arch(hf_non_match) == "base"


def test_plugin_resolution_and_disambiguation(tmp_path: Path) -> None:
    """Verify bare plugin convention, unknown errors, and nested import errors."""
    bad_cfg = config_mod.parse_config("plugin=nonexistent_bare_plugin_12345")
    res_bad = config_mod.resolve_plugin_ref(bad_cfg)
    assert res_bad is not None
    with pytest.raises(MegakernelConfigError, match="failed to import"):
        config_mod.import_plugin(res_bad[0])

    # Bare name resolving via models package path
    models_pkg = importlib.import_module("vllm_torchtpu.models")
    fake_pkg_dir = tmp_path / "fake_bare_model"
    fake_pkg_dir.mkdir(parents=True)
    (fake_pkg_dir / "__init__.py").write_text("", encoding="utf-8")
    (fake_pkg_dir / "megakernel.py").write_text("SPEC = object()\n", encoding="utf-8")

    models_pkg.__path__.append(str(tmp_path))
    try:
        bare_cfg = config_mod.parse_config("plugin=fake_bare_model")
        res_bare = config_mod.resolve_plugin_ref(bare_cfg)
        assert res_bare is not None
        ref_bare, _ = res_bare
        assert ref_bare.module == "vllm_torchtpu.models.fake_bare_model.megakernel"

        # Nested ModuleNotFoundError inside plugin module must re-raise
        broken_pkg_dir = tmp_path / "broken_bare_model"
        broken_pkg_dir.mkdir(parents=True)
        (broken_pkg_dir / "__init__.py").write_text(
            "import non_existent_submod_999\n", encoding="utf-8"
        )
        (broken_pkg_dir / "megakernel.py").write_text(
            "SPEC = object()\n", encoding="utf-8"
        )

        broken_cfg = config_mod.parse_config("plugin=broken_bare_model")
        with pytest.raises(ModuleNotFoundError, match="non_existent_submod_999"):
            config_mod.resolve_plugin_ref(broken_cfg)
    finally:
        models_pkg.__path__.remove(str(tmp_path))
