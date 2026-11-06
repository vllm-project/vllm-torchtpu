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
"""Multi-model registration hooks and runtime extensions."""

from __future__ import annotations

import importlib
import importlib.util
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING
from unittest import mock

import jax
import numpy as np
import pytest  # pyrefly: ignore[missing-import]
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
    fingerprint as fp_mod,
)
from vllm_torchtpu.megakernel_runtime import (
    guards as guards_mod,
)
from vllm_torchtpu.megakernel_runtime import (
    hooks as hooks_mod,
)
from vllm_torchtpu.megakernel_runtime import (
    testkit,
)
from vllm_torchtpu.megakernel_runtime.plugin import RunnerView, Shorthand
from vllm_torchtpu.megakernel_runtime.testing import fake_runner

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from pathlib import Path


def _check_domain_world_sizes() -> None:
    """Verify domain world size calculations for various runner topologies."""
    exp_world_32 = 32
    exp_world_8 = 8

    r_32_8 = fake_runner(
        tp_size=32, parallel_config=SimpleNamespace(decode_context_parallel_size=8)
    )
    v_32_8 = RunnerView(r_32_8, tp_rank=0)
    assert guards_mod.domain_world_size(DOMAIN_TP_X_DCP, r_32_8, v_32_8) == exp_world_32

    r_1_8 = fake_runner(
        tp_size=1, parallel_config=SimpleNamespace(decode_context_parallel_size=8)
    )
    v_1_8 = RunnerView(r_1_8, tp_rank=0)
    assert guards_mod.domain_world_size(DOMAIN_TP_X_DCP, r_1_8, v_1_8) == exp_world_8

    r_8_8 = fake_runner(
        tp_size=8, parallel_config=SimpleNamespace(decode_context_parallel_size=8)
    )
    v_8_8 = RunnerView(r_8_8, tp_rank=0)
    assert guards_mod.domain_world_size(DOMAIN_TP_X_DCP, r_8_8, v_8_8) == exp_world_8

    r_bad_dcp = fake_runner(
        tp_size=6, parallel_config=SimpleNamespace(decode_context_parallel_size=4)
    )
    v_bad_dcp = RunnerView(r_bad_dcp, tp_rank=0)
    with pytest.raises(MegakernelConfigError):
        guards_mod.domain_world_size(DOMAIN_TP_X_DCP, r_bad_dcp, v_bad_dcp)

    r_dp2_tp16 = fake_runner(
        tp_size=16, parallel_config=SimpleNamespace(data_parallel_size=2)
    )
    v_dp2_tp16 = RunnerView(r_dp2_tp16, tp_rank=0)
    assert (
        guards_mod.domain_world_size(DOMAIN_KERNEL_MESH, r_dp2_tp16, v_dp2_tp16)
        == exp_world_32
    )

    r_dp4_tp8 = fake_runner(
        tp_size=8, parallel_config=SimpleNamespace(data_parallel_size=4)
    )
    v_dp4_tp8 = RunnerView(r_dp4_tp8, tp_rank=0)
    assert (
        guards_mod.domain_world_size(DOMAIN_KERNEL_MESH, r_dp4_tp8, v_dp4_tp8)
        == exp_world_32
    )

    r_dp1_tp32 = fake_runner(
        tp_size=32, parallel_config=SimpleNamespace(data_parallel_size=1)
    )
    v_dp1_tp32 = RunnerView(r_dp1_tp32, tp_rank=0)
    assert (
        guards_mod.domain_world_size(DOMAIN_KERNEL_MESH, r_dp1_tp32, v_dp1_tp32)
        == exp_world_32
    )


def test_collective_domain_tp_x_dcp_and_kernel_mesh_world_size() -> None:
    """Verify collective domain world sizes and mesh size checking."""
    runner = fake_runner(tp_size=1)
    view = RunnerView(runner, tp_rank=0)

    spec_tp = TpuMonoSpec("s_tp", "1", create="m:c", collective_domain=DOMAIN_TP)
    spec_dcp = TpuMonoSpec(
        "s_dcp", "1", create="m:c", collective_domain=DOMAIN_TP_X_DCP
    )
    spec_km = TpuMonoSpec(
        "s_km", "1", create="m:c", collective_domain=DOMAIN_KERNEL_MESH
    )

    assert guards_mod.check_spec(runner, spec_tp, view=view) is None
    assert guards_mod.check_spec(runner, spec_dcp, view=view) is None
    assert guards_mod.check_spec(runner, spec_km, view=view) is None

    bogus_spec = SimpleNamespace(
        chips=(),
        tp_sizes=(),
        collective_domain="bogus_domain",
        parallelism=frozenset({"tp"}),
        features=frozenset(),
        requires={},
    )
    res_bogus = guards_mod.check_spec(runner, bogus_spec, view=view)
    assert res_bogus is not None
    assert res_bogus.code == "parallelism_unsupported:bogus_domain"

    _check_domain_world_sizes()

    devs = jax.devices()
    devs_2 = [devs[0], devs[1]] if len(devs) > 1 else [devs[0], devs[0]]
    mesh_2 = jax.sharding.Mesh(np.asarray(devs_2), ("tp",))
    r_km = fake_runner(
        tp_size=1, parallel_config=SimpleNamespace(data_parallel_size=2), mesh=mesh_2
    )
    v_km = RunnerView(r_km, tp_rank=0)
    mesh_ranks = guards_mod.domain_world_size(DOMAIN_KERNEL_MESH, r_km, v_km)
    exp_mesh_ranks_2 = 2
    assert mesh_ranks == exp_mesh_ranks_2
    build_mesh_fn = vars(driver_mod)["_build_tp_rank_mesh"]
    mesh_built = build_mesh_fn(mesh_ranks, fallback_mesh=mesh_2)
    assert driver_mod.check_mesh_size(mesh_ranks, int(mesh_built.devices.size)) is None

    r_32_8 = fake_runner(
        tp_size=32, parallel_config=SimpleNamespace(decode_context_parallel_size=8)
    )
    r_1_8 = fake_runner(
        tp_size=1, parallel_config=SimpleNamespace(decode_context_parallel_size=8)
    )
    sentinel_tp_cpu = object()
    sentinel_dcp_cpu = object()
    sentinel_world_cpu = object()
    with (
        mock.patch.object(torch.distributed, "is_available", return_value=True),
        mock.patch.object(torch.distributed, "is_initialized", return_value=True),
        mock.patch(
            "vllm.distributed.get_tp_group",
            return_value=SimpleNamespace(world_size=32, cpu_group=sentinel_tp_cpu),
        ),
        mock.patch(
            "vllm.distributed.get_dcp_group",
            return_value=SimpleNamespace(world_size=8, cpu_group=sentinel_dcp_cpu),
        ),
        mock.patch(
            "vllm.distributed.get_world_group",
            return_value=SimpleNamespace(world_size=2, cpu_group=sentinel_world_cpu),
        ),
    ):
        assert agreement_mod.agreement_group(DOMAIN_TP, r_32_8) is sentinel_tp_cpu
        assert agreement_mod.agreement_group(DOMAIN_TP_X_DCP, r_32_8) is sentinel_tp_cpu
        assert agreement_mod.agreement_group(DOMAIN_TP_X_DCP, r_1_8) is sentinel_dcp_cpu
        assert (
            agreement_mod.agreement_group(DOMAIN_KERNEL_MESH, r_km)
            is sentinel_world_cpu
        )


class _DummyObsModel(torch.nn.Module):
    """Dummy model recording loaded weights."""

    def __init__(self) -> None:
        """Initialize empty weight dictionary."""
        super().__init__()
        self.loaded_weights: dict[str, torch.Tensor] = {}

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        """Record weights into dictionary."""
        for name, tensor in weights:
            self.loaded_weights[name] = tensor


class _DummyObsLoader:
    """Dummy loader returning raw checkpoint weights."""

    def __init__(self, raw: list[tuple[str, torch.Tensor]]) -> None:
        """Store raw checkpoint weights."""
        self.raw = raw

    def get_all_weights(
        self, model_config: object, model: object
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield raw checkpoint weights."""
        del model_config, model
        yield from self.raw

    def load_model(self, vllm_config: object, model_config: object) -> _DummyObsModel:
        """Load weights into dummy model."""
        del vllm_config
        model = _DummyObsModel()
        model.load_weights(self.get_all_weights(model_config, model))
        model.loaded_weights.pop("layer.0.w13_weight_scale_2", None)
        return model


def _verify_tpu_runner_observe_model_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify TPUModelRunner model loading wrapper with load_tensor hook."""
    if importlib.util.find_spec("vllm") is None:
        return

    import sys

    if "vllm_torchtpu.runner.mm_encoder_manager" not in sys.modules:
        monkeypatch.setitem(
            sys.modules,
            "vllm_torchtpu.runner.mm_encoder_manager",
            mock.MagicMock(),
        )
    backends_utils = importlib.import_module("vllm.v1.attention.backends.utils")
    with mock.patch.object(
        backends_utils,
        "get_kv_cache_layout",
        create=True,
        return_value=None,
    ):
        tpu_runner_mod = importlib.import_module("vllm_torchtpu.runner.tpu_runner")
        tpu_model_runner_cls = vars(tpu_runner_mod)["TPUModelRunner"]

        observed_in_runner: list[str] = []

        def runner_observe(
            name: str, _tensor: torch.Tensor, _options: object = None
        ) -> None:
            """Record weight tensor names in runner."""
            observed_in_runner.append(name)

        mock_runner_plugin = SimpleNamespace(
            load_tensor=runner_observe, layout_request=None
        )
        runner_vllm_cfg = SimpleNamespace(
            additional_config={"megakernel": {"enabled": True}},
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(architectures=["ArchTest"]),
                quantization=None,
            ),
            load_config=SimpleNamespace(),
            mesh=None,
        )
        raw_checkpoint_weights = [
            ("layer.0.w13_weight_scale_2", torch.tensor([0.5, 1.5])),
            ("layer.0.weight", torch.tensor([2.0, 3.0])),
        ]
        loader = _DummyObsLoader(raw_checkpoint_weights)
        orig_device = torch.device

        def fake_device(d: object) -> torch.device:
            """Map tpu device to cpu device."""
            return orig_device("cpu") if str(d) == "tpu" else orig_device(str(d))

        with (
            mock.patch("torch.device", side_effect=fake_device),
            mock.patch(
                "vllm_torchtpu.runner.tpu_runner.get_model_loader",
                return_value=loader,
            ),
            mock.patch(
                "vllm_torchtpu.runner.tpu_runner.get_tpu_quantization_config",
                return_value=None,
            ),
            mock.patch(
                "vllm_torchtpu.runner.tpu_runner.validate_online_fp8",
                return_value=None,
            ),
            mock.patch(
                "vllm_torchtpu.megakernel_runtime.hooks.select_plugin",
                return_value=(
                    SimpleNamespace(enabled=True),
                    (config_mod.PluginRef("dummy"), {}),
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
            tpu_model_runner_cls.load_model(runner_instance)

        assert "layer.0.w13_weight_scale_2" in observed_in_runner
        assert "layer.0.weight" in observed_in_runner
        assert "layer.0.w13_weight_scale_2" not in runner_instance.model.loaded_weights


def test_load_tensor_hook_streams_raw_tensors_and_check_spec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify weight observation hook streaming and specification checks."""
    mod_path = tmp_path / "obs_plugin.py"
    mod_path.write_text(
        "from vllm_torchtpu.megakernel_runtime.plugin import TpuMonoSpec\n"
        "def create(view, options): return None\n"
        "def load_tensor(name, tensor, options): pass\n"
        'SPEC = TpuMonoSpec("obs_test", "1", create=create, '
        'load_tensor=".obs_plugin:load_tensor")\n',
        encoding="utf-8",
    )

    spec_with_obs = TpuMonoSpec(
        "obs_test",
        "1",
        create="obs_mod:create",
        load_tensor=".obs_plugin:load_tensor",
    )
    spec_no_obs = TpuMonoSpec("obs_test", "1", create="obs_mod:create")

    testkit.check_spec(spec_with_obs, lazy_create=True)
    bad_spec = TpuMonoSpec(
        "obs_test",
        "1",
        create="obs_mod:create",
        load_tensor=lambda *_args, **_kwargs: None,
    )
    with pytest.raises(AssertionError):
        testkit.check_spec(bad_spec, lazy_create=True)

    fp_none = fp_mod.plugin_fingerprint(
        module="m",
        attr="SPEC",
        plugin=spec_no_obs,
        source_hash="h",
        config={},
        raw_env="",
    )
    fp_obs = fp_mod.plugin_fingerprint(
        module="m",
        attr="SPEC",
        plugin=spec_with_obs,
        source_hash="h",
        config={},
        raw_env="",
    )
    assert fp_none != fp_obs

    observed: list[str] = []

    def hook_fn(name: str, _tensor: torch.Tensor, _options: object = None) -> None:
        """Record weight tensor names observed during model loading."""
        observed.append(name)

    mock_plugin = SimpleNamespace(
        load_tensor=hook_fn,
        layout_request=None,
    )
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
            (config_mod.PluginRef("dummy"), {}),
        )
        imp_target = "vllm_torchtpu.megakernel_runtime.hooks.import_plugin"
        with mock.patch(imp_target) as mock_imp:
            mock_imp.return_value = (mock_plugin, ModuleType("dummy"))
            streamed = list(hooks_mod.load_tensor_stream(vllm_config, weights))

    exp_streamed_count_2 = 2
    assert observed == ["layer.0.w13_weight_scale_2", "layer.0.weight"]
    assert len(streamed) == exp_streamed_count_2
    assert streamed[0][0] == "layer.0.w13_weight_scale_2"

    disabled_cfg = SimpleNamespace(
        additional_config={"megakernel": {"enabled": False}},
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["ArchTest"])
        ),
    )
    passthrough = list(hooks_mod.load_tensor_stream(disabled_cfg, weights))
    assert passthrough == weights

    _verify_tpu_runner_observe_model_loader(monkeypatch)


def _verify_block_slot_config_presizing() -> None:
    """Verify block and slot derivation under layout requests."""
    if importlib.util.find_spec("vllm") is None:
        return

    block_utils_mod = importlib.import_module(
        "vllm_torchtpu.platforms.tpu_block_size_utils"
    )
    derive_fn = vars(block_utils_mod)["_derive_tpu_block_slot_config"]

    class FakeBackend:
        """Fake attention backend with fixed page sizing."""

        @staticmethod
        def get_name() -> str:
            """Return backend name."""
            return "TEST"

        @staticmethod
        def get_min_page_size(_vllm_config: object) -> int:
            """Return minimum page size."""
            return 16

        @staticmethod
        def get_supported_kernel_block_sizes() -> list[int]:
            """Return supported kernel block sizes."""
            return [16]

        @staticmethod
        def get_kv_cache_page_size_bytes(
            block_size: int, *_args: object, **_kwargs: object
        ) -> int:
            """Return page size in bytes."""
            bytes_per_kib = 1024
            return block_size * bytes_per_kib

    vllm_cfg_640 = SimpleNamespace(
        _megakernel_layout_applied={
            "block_size": 128,
            "mamba_slot_tokens": 256,
            "mamba_block_size": 640,
        },
        cache_config=SimpleNamespace(
            block_size=128,
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
    d_640 = derive_fn(vllm_cfg_640, FakeBackend, input_block_size=128)
    exp_final_block_640 = 640
    exp_input_block_128 = 128
    assert d_640["final_block_size"] == exp_final_block_640
    assert vllm_cfg_640.cache_config.block_size == exp_input_block_128

    vllm_cfg_256 = SimpleNamespace(
        _megakernel_layout_applied={"slot_stride_tokens": 256},
        cache_config=SimpleNamespace(
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
    d_256 = derive_fn(vllm_cfg_256, FakeBackend, input_block_size=16)
    exp_final_block_256 = 256
    assert d_256["final_block_size"] == exp_final_block_256

    vllm_cfg_user_32 = SimpleNamespace(
        _megakernel_layout_applied={"block_size": 128, "mamba_block_size": 640},
        cache_config=SimpleNamespace(
            block_size=32,
            user_specified_block_size=True,
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
    d_user_32 = derive_fn(vllm_cfg_user_32, FakeBackend, input_block_size=32)
    exp_user_block_32 = 32
    assert d_user_32["final_block_size"] == exp_user_block_32

    vllm_cfg_zero = SimpleNamespace(
        _megakernel_layout_applied={"mamba_slot_tokens": 0},
        cache_config=SimpleNamespace(
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
    with pytest.raises(ValueError, match="must be > 0"):
        derive_fn(vllm_cfg_zero, FakeBackend, input_block_size=16)


def test_layout_request_presizing_overrides_slot_and_block_config() -> None:
    """Verify layout request presizing overrides slot and block configurations."""

    def fake_layout_request(_vllm_config: object, _options: object) -> dict[str, int]:
        """Return fake layout request sizing parameters."""
        return {
            "block_size": 128,
            "cp_kv_cache_interleave_size": 128,
            "mamba_block_size": 640,
            "slot_stride_tokens": 256,
        }

    mock_plugin = SimpleNamespace(
        layout_request=fake_layout_request,
        load_tensor=None,
    )
    vllm_config = SimpleNamespace(
        additional_config={"megakernel": {"enabled": True}},
        cache_config=SimpleNamespace(block_size=16),
        parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=16),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["ArchTest"])
        ),
    )

    with mock.patch("vllm_torchtpu.megakernel_runtime.hooks.select_plugin") as mock_sel:
        mock_sel.return_value = (
            SimpleNamespace(enabled=True),
            (config_mod.PluginRef("dummy"), {}),
        )
        imp_target = "vllm_torchtpu.megakernel_runtime.hooks.import_plugin"
        with mock.patch(imp_target) as mock_imp:
            mock_imp.return_value = (mock_plugin, ModuleType("dummy"))
            res = hooks_mod.apply_layout_request(vllm_config)

    exp_block_128 = 128
    exp_mamba_tokens_640 = 640
    exp_slot_stride_256 = 256
    exp_user_block_32 = 32
    exp_disabled_block_16 = 16

    assert res is not None
    assert vllm_config.cache_config.block_size == exp_block_128
    assert vllm_config.parallel_config.cp_kv_cache_interleave_size == exp_block_128
    assert res["mamba_block_size"] == exp_mamba_tokens_640
    assert res["slot_stride_tokens"] == exp_slot_stride_256
    assert "mamba_block_size" not in vllm_config.additional_config

    res_idempotent = hooks_mod.apply_layout_request(vllm_config)
    assert res_idempotent == res

    user_spec_cfg = SimpleNamespace(
        additional_config={"megakernel": {"enabled": True}},
        cache_config=SimpleNamespace(block_size=32, user_specified_block_size=True),
        parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=1),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["ArchTest"])
        ),
    )
    with mock.patch(
        "vllm_torchtpu.megakernel_runtime.hooks.select_plugin"
    ) as mock_sel2:
        mock_sel2.return_value = (
            SimpleNamespace(enabled=True),
            (config_mod.PluginRef("dummy"), {}),
        )
        with mock.patch(
            "vllm_torchtpu.megakernel_runtime.hooks.import_plugin"
        ) as mock_imp2:
            mock_imp2.return_value = (mock_plugin, ModuleType("dummy"))
            hooks_mod.apply_layout_request(user_spec_cfg)
    assert user_spec_cfg.cache_config.block_size == exp_user_block_32

    conflict_cfg = SimpleNamespace(
        additional_config={"megakernel": {"enabled": True}},
        cache_config=SimpleNamespace(block_size=16),
        parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=32),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["ArchTest"])
        ),
    )
    with (
        mock.patch(
            "vllm_torchtpu.megakernel_runtime.hooks.select_plugin",
            return_value=(
                SimpleNamespace(enabled=True),
                (config_mod.PluginRef("dummy"), {}),
            ),
        ),
        mock.patch(
            "vllm_torchtpu.megakernel_runtime.hooks.import_plugin",
            return_value=(mock_plugin, ModuleType("dummy")),
        ),
        pytest.raises(MegakernelConfigError),
    ):
        hooks_mod.apply_layout_request(conflict_cfg)

    bad_type_plugin = SimpleNamespace(
        layout_request=lambda *_a: ["not_a_mapping"],
        load_tensor=None,
    )
    bad_type_cfg = SimpleNamespace(
        additional_config={"megakernel": {"enabled": True}},
        cache_config=SimpleNamespace(block_size=16),
        parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=16),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["ArchTest"])
        ),
    )
    with (
        mock.patch(
            "vllm_torchtpu.megakernel_runtime.hooks.select_plugin",
            return_value=(
                SimpleNamespace(enabled=True),
                (config_mod.PluginRef("dummy"), {}),
            ),
        ),
        mock.patch(
            "vllm_torchtpu.megakernel_runtime.hooks.import_plugin",
            return_value=(bad_type_plugin, ModuleType("dummy")),
        ),
        pytest.raises(MegakernelConfigError, match="Mapping or dataclass"),
    ):
        hooks_mod.apply_layout_request(bad_type_cfg)

    disabled_cfg = SimpleNamespace(
        additional_config={"megakernel": {"enabled": False}},
        cache_config=SimpleNamespace(block_size=16),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["ArchTest"])
        ),
    )
    assert hooks_mod.apply_layout_request(disabled_cfg) is None
    assert disabled_cfg.cache_config.block_size == exp_disabled_block_16

    _verify_block_slot_config_presizing()


def test_shorthand_hf_fields_disambiguation() -> None:
    """Verify shorthand disambiguation by HuggingFace config fields."""
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
