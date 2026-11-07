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
"""Unit tests for multi-model megakernel hooks and weight streaming."""

from __future__ import annotations

import ast
import builtins
import contextlib
import importlib
import logging
import textwrap
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import cast
from unittest import mock

import jax
import numpy as np
import torch

import vllm_torchtpu
from vllm_torchtpu import megakernel_runtime
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
from vllm_torchtpu.megakernel_runtime.hooks import load_tensor_stream
from vllm_torchtpu.megakernel_runtime.plugin import RunnerView, Shorthand
from vllm_torchtpu.megakernel_runtime.testing import fake_runner

EXPECTED_STREAMED_COUNT = 2
EXPECTED_DEFAULT_BLOCK_SIZE = 16
EXPECTED_REQUESTED_BLOCK_SIZE = 128
EXPECTED_DERIVED_BLOCK_SIZE = 640
_BUILD_TP_MESH_ATTR = "_build_tp_rank_mesh"
_EXEC_ATTR = "exec"
_VALIDATE_LAYOUT_ATTR = "_validate_layout_request"
_DERIVE_BLOCK_SLOT_ATTR = "_derive_tpu_block_slot_config"


@contextlib.contextmanager
def _assert_raises(exc_type: type[BaseException], match: str) -> Iterator[None]:
    """Assert that the block raises `exc_type` containing `match`."""
    caught: BaseException | None = None
    try:
        yield
    except exc_type as exc:
        caught = exc
    if caught is None:
        msg = f"Expected {exc_type.__name__} matching {match!r}"
        raise AssertionError(msg)
    assert match in str(caught), f"Expected {match!r} in {caught!r}"


def test_collective_domain_world_sizes_and_agreements() -> None:
    """Verify domain world size derivations, agreement routing, and mesh checks."""
    cases = (
        (DOMAIN_TP, 32, 1, 32),
        (DOMAIN_TP, 8, 4, 8),
        (DOMAIN_TP_X_DCP, 32, 1, 32),
        (DOMAIN_TP_X_DCP, 8, 4, 8),
        (DOMAIN_TP_X_DCP, 16, 2, 16),
        (DOMAIN_KERNEL_MESH, 32, 1, 32),
        (DOMAIN_KERNEL_MESH, 8, 4, 32),
        (DOMAIN_KERNEL_MESH, 16, 2, 32),
    )
    build_tp_rank_mesh = getattr(driver_mod, _BUILD_TP_MESH_ATTR)
    for domain, tp, dp, expected_ranks in cases:
        runner = fake_runner(tp_size=tp)
        runner.parallel_config.data_parallel_size = dp
        runner.vllm_config.parallel_config = runner.parallel_config
        view = RunnerView(runner, tp_rank=0)
        assert guards_mod.domain_world_size(domain, runner, view) == expected_ranks

        parallelism = frozenset({"tp", "dp"}) if dp > 1 else frozenset({"tp"})
        spec = TpuMonoSpec(
            "s_test",
            "1",
            create="m:c",
            collective_domain=domain,
            parallelism=parallelism,
        )
        assert guards_mod.check_spec(runner, spec, view=view) is None

        fake_tp = object()
        fake_world = object()
        with (
            mock.patch.object(agreement_mod, "_tp_cpu_group", return_value=fake_tp),
            mock.patch.object(
                agreement_mod, "_world_cpu_group", return_value=fake_world
            ),
            mock.patch.object(driver_mod, "_get_pallas_mesh_fn", return_value=None),
        ):
            expected_group = fake_world if domain == DOMAIN_KERNEL_MESH else fake_tp
            assert agreement_mod.agreement_group(domain, runner) is expected_group

            devs = jax.devices()
            mesh = jax.sharding.Mesh(np.asarray([devs[0]]), ("tp",))
            mesh_built = build_tp_rank_mesh(1, fallback_mesh=mesh)
            assert driver_mod.check_mesh_size(1, int(mesh_built.devices.size)) is None


class _DummyObsModel(torch.nn.Module):
    """Dummy model recording loaded weights."""

    def __init__(self) -> None:
        """Initialize empty loaded-weights dictionary."""
        super().__init__()
        self.loaded_weights: dict[str, torch.Tensor] = {}

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        """Record every `(name, tensor)` pair in `self.loaded_weights`."""
        for name, tensor in weights:
            self.loaded_weights[name] = tensor


class _DummyObsLoader:
    """Dummy loader returning raw checkpoint weights."""

    def __init__(self, raw: list[tuple[str, torch.Tensor]]) -> None:
        """Store raw `(name, tensor)` pairs."""
        self.raw = raw

    def get_all_weights(
        self, _model_config: object, _model: object
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield raw checkpoint weights."""
        yield from self.raw

    def load_model(self, vllm_config: object, model_config: object) -> _DummyObsModel:
        """Instantiate `_DummyObsModel` and stream `get_all_weights` into it."""
        if vllm_config is None:
            msg = "vllm_config must not be None"
            raise ValueError(msg)
        model = _DummyObsModel()
        model.load_weights(self.get_all_weights(model_config, model))
        return model


def _runner_load_model() -> Callable[[object], object]:
    """Compile `TPUModelRunner.load_model` from source without `torch_tpu`."""
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
        """Provide a no-op context manager stub."""
        yield

    orig_device = torch.device

    def fake_device(dev_spec: object) -> torch.device:
        """Map `'tpu'` device requests to `'cpu'` in deviceless tests."""
        return (
            orig_device("cpu") if str(dev_spec) == "tpu" else orig_device(str(dev_spec))
        )

    class _FakeTorch:
        """Proxy `torch` module overriding `device('tpu')` with CPU."""

        def __getattr__(self, name: str) -> object:
            """Return `fake_device` for `device` or forward to `torch`."""
            if name == "device":
                return fake_device
            return getattr(torch, name)

    class _FallbackNamespace:
        """Stub namespace returning `False` for any attribute read."""

        def __getattr__(self, _name: str) -> bool:
            """Return `False` for any queried flag."""
            return False

    ns: dict[str, object] = {
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
    exec_fn = getattr(builtins, _EXEC_ATTR)
    exec_fn(compile(fn_src, str(path), "exec"), ns)
    return cast("Callable[[object], object]", ns["load_model"])


def _verify_tpu_runner_observe_model_loader() -> None:
    """Verify `TPUModelRunner` model loading wrapper with `load_tensor` hook."""
    load_model = _runner_load_model()

    observed_in_runner: list[str] = []

    def runner_load_tensor(name: str, tensor: torch.Tensor, _options: object) -> object:
        """Record tensor names and double `layer.0.weight`."""
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
            model=_DummyObsModel(),
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
        """Record `name` and double `layer.0.weight`."""
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
    assert len(streamed) == EXPECTED_STREAMED_COUNT
    assert streamed[0][0] == "layer.0.w13_weight_scale_2"
    t0 = cast("torch.Tensor", streamed[0][1])
    t1 = cast("torch.Tensor", streamed[1][1])
    assert torch.equal(t0, torch.tensor([1.0, 2.0]))
    assert streamed[1][0] == "layer.0.weight"
    assert torch.equal(t1, torch.tensor([6.0, 8.0]))

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
    validate_fn = getattr(hooks_mod, _VALIDATE_LAYOUT_ATTR)
    with _assert_raises(ValueError, "unknown keys"):
        validate_fn({"unknown_key": 16})
    with _assert_raises(ValueError, "must be a positive int"):
        validate_fn({"block_size": 0})
    with _assert_raises(ValueError, "must be a positive int"):
        validate_fn({"block_size": -1})
    with _assert_raises(ValueError, "must be a positive int"):
        validate_fn({"block_size": True})
    with _assert_raises(ValueError, "must be a positive int"):
        validate_fn({"block_size": "16"})
    with _assert_raises(ValueError, "page_size and block_size disagree"):
        validate_fn({"page_size": 16, "block_size": 32})

    def fake_layout_request(_vllm_config: object, _options: object) -> dict[str, int]:
        """Return synthetic layout request with `block_size=128`."""
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
    assert vllm_config.cache_config.block_size == EXPECTED_REQUESTED_BLOCK_SIZE

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
    assert user_cfg.cache_config.block_size == EXPECTED_DEFAULT_BLOCK_SIZE

    disabled_cfg = SimpleNamespace(
        additional_config={"megakernel": {"enabled": False}},
        cache_config=SimpleNamespace(block_size=16),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["ArchTest"])
        ),
    )
    assert hooks_mod.apply_layout_request(disabled_cfg) is None

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
            layout_request=lambda *_: {
                "block_size": 16,
                "dcp_interleave_size": 12,
            },
            load_tensor=None,
        )
        with mock.patch(
            "vllm_torchtpu.megakernel_runtime.hooks.import_plugin"
        ) as mock_imp:
            mock_imp.return_value = (bad_dcp_plugin, ModuleType("dummy"))
            with _assert_raises(ValueError, "divisible by dcp_interleave_size"):
                hooks_mod.apply_layout_request(bad_dcp_cfg)

        bad_type_plugin = SimpleNamespace(
            layout_request=lambda *_: ["invalid_layout_request"],
            load_tensor=None,
        )
        with (
            mock.patch(
                "vllm_torchtpu.megakernel_runtime.hooks.import_plugin",
                return_value=(bad_type_plugin, ModuleType("dummy")),
            ),
            _assert_raises(TypeError, "layout_request must return a Mapping"),
        ):
            hooks_mod.resolve_layout_request(bad_dcp_cfg)

    block_utils_mod = importlib.import_module(
        "vllm_torchtpu.platforms.tpu_block_size_utils"
    )
    derive_fn = getattr(block_utils_mod, _DERIVE_BLOCK_SLOT_ATTR)

    class FakeBackend:
        """Stub attention backend for block-slot derivation tests."""

        @staticmethod
        def get_name() -> str:
            """Return stub backend identifier."""
            return "TEST"

        @staticmethod
        def get_min_page_size(_vllm_config: object) -> int:
            """Return minimum page size in tokens."""
            return 16

        @staticmethod
        def get_supported_kernel_block_sizes() -> list[int]:
            """Return supported kernel block sizes."""
            return [16]

        @staticmethod
        def get_kv_cache_page_size_bytes(
            block_size: int, *_args: object, **_kwargs: object
        ) -> int:
            """Return synthetic page byte size for `block_size`."""
            return block_size * 1024

    d_640 = derive_fn(vllm_config, FakeBackend, input_block_size=128)
    assert d_640["final_block_size"] == EXPECTED_DERIVED_BLOCK_SIZE


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
    with _assert_raises(MegakernelConfigError, "failed to import"):
        config_mod.import_plugin(res_bad[0])

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

        broken_pkg_dir = tmp_path / "broken_bare_model"
        broken_pkg_dir.mkdir(parents=True)
        (broken_pkg_dir / "__init__.py").write_text(
            "import non_existent_submod_999\n", encoding="utf-8"
        )
        (broken_pkg_dir / "megakernel.py").write_text(
            "SPEC = object()\n", encoding="utf-8"
        )

        broken_cfg = config_mod.parse_config("plugin=broken_bare_model")
        with _assert_raises(ModuleNotFoundError, "non_existent_submod_999"):
            config_mod.resolve_plugin_ref(broken_cfg)
    finally:
        models_pkg.__path__.remove(str(tmp_path))


def test_load_tensor_stream_weight_consumed_sentinel_and_option() -> None:
    """Verify `WEIGHT_CONSUMED` and `consume_weights` drop consumed weights."""
    sentinel = hooks_mod.WEIGHT_CONSUMED
    t_drop = torch.tensor([1.0, 2.0])
    t_keep = torch.tensor([3.0, 4.0])
    t_swap = torch.tensor([5.0, 6.0])

    def _sentinel_hook(name: str, _tensor: object, _opts: object) -> object:
        """Drop `drop.w`, transform `swap.w`, and pass `keep.w` through."""
        if name == "drop.w":
            return sentinel
        if name == "swap.w":
            return ("swapped.w", t_swap * 2)
        return None

    with mock.patch.object(
        hooks_mod, "_plugin_hook", return_value=(_sentinel_hook, {})
    ):
        out_default = list(
            hooks_mod.load_tensor_stream(
                SimpleNamespace(),
                [("drop.w", t_drop), ("keep.w", t_keep), ("swap.w", t_swap)],
            )
        )
    assert len(out_default) == EXPECTED_STREAMED_COUNT
    assert out_default[0][0] == "keep.w"
    assert out_default[0][1] is t_keep
    assert out_default[1][0] == "swapped.w"
    assert torch.equal(cast("torch.Tensor", out_default[1][1]), t_swap * 2)

    def _consume_opt_hook(name: str, _tensor: object, _opts: object) -> object:
        """Return `sentinel` when consumed under `consume_weights=True`."""
        if name == "swap.w":
            return ("swapped.w", t_swap + 1)
        if name == "drop.w":
            return sentinel
        return None

    with mock.patch.object(
        hooks_mod,
        "_plugin_hook",
        return_value=(_consume_opt_hook, {"consume_weights": True}),
    ):
        out_consumed = list(
            hooks_mod.load_tensor_stream(
                SimpleNamespace(),
                [("drop.w", t_drop), ("keep.w", t_keep), ("swap.w", t_swap)],
            )
        )
    assert len(out_consumed) == EXPECTED_STREAMED_COUNT
    assert out_consumed[0][0] == "keep.w"
    assert out_consumed[0][1] is t_keep
    assert out_consumed[1][0] == "swapped.w"
    assert torch.equal(cast("torch.Tensor", out_consumed[1][1]), t_swap + 1)
