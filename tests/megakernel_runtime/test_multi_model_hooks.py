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

import os
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import jax
import numpy as np
import pytest
import torch

from vllm_torchtpu.megakernel.runtime.contract import (
    DOMAIN_KERNEL_MESH,
    DOMAIN_TP,
    DOMAIN_TP_X_DCP,
    TpuMonoSpec,
)
from vllm_torchtpu.megakernel.runtime.eligibility import check_step_eligibility
from vllm_torchtpu.megakernel.runtime.policy import (
    BucketPolicy,
    parse_mk_verify_buckets,
)
from vllm_torchtpu.megakernel.runtime.stats import BatchShape
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


def test_collective_domain_tp_x_dcp_and_kernel_mesh_world_size() -> None:
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

    r_32_8 = fake_runner(
        tp_size=32, parallel_config=SimpleNamespace(decode_context_parallel_size=8)
    )
    v_32_8 = RunnerView(r_32_8, tp_rank=0)
    assert guards_mod.domain_world_size(DOMAIN_TP_X_DCP, r_32_8, v_32_8) == 32

    r_1_8 = fake_runner(
        tp_size=1, parallel_config=SimpleNamespace(decode_context_parallel_size=8)
    )
    v_1_8 = RunnerView(r_1_8, tp_rank=0)
    assert guards_mod.domain_world_size(DOMAIN_TP_X_DCP, r_1_8, v_1_8) == 8

    r_8_8 = fake_runner(
        tp_size=8, parallel_config=SimpleNamespace(decode_context_parallel_size=8)
    )
    v_8_8 = RunnerView(r_8_8, tp_rank=0)
    assert guards_mod.domain_world_size(DOMAIN_TP_X_DCP, r_8_8, v_8_8) == 8

    r_dp2_tp16 = fake_runner(
        tp_size=16, parallel_config=SimpleNamespace(data_parallel_size=2)
    )
    v_dp2_tp16 = RunnerView(r_dp2_tp16, tp_rank=0)
    assert (
        guards_mod.domain_world_size(DOMAIN_KERNEL_MESH, r_dp2_tp16, v_dp2_tp16) == 32
    )

    r_dp4_tp8 = fake_runner(
        tp_size=8, parallel_config=SimpleNamespace(data_parallel_size=4)
    )
    v_dp4_tp8 = RunnerView(r_dp4_tp8, tp_rank=0)
    assert guards_mod.domain_world_size(DOMAIN_KERNEL_MESH, r_dp4_tp8, v_dp4_tp8) == 32

    r_dp1_tp32 = fake_runner(
        tp_size=32, parallel_config=SimpleNamespace(data_parallel_size=1)
    )
    v_dp1_tp32 = RunnerView(r_dp1_tp32, tp_rank=0)
    assert (
        guards_mod.domain_world_size(DOMAIN_KERNEL_MESH, r_dp1_tp32, v_dp1_tp32) == 32
    )

    devs = jax.devices()[:2]
    mesh_2 = jax.sharding.Mesh(np.asarray(devs), ("tp",))
    r_km = fake_runner(
        tp_size=1, parallel_config=SimpleNamespace(data_parallel_size=2), mesh=mesh_2
    )
    v_km = RunnerView(r_km, tp_rank=0)
    mesh_ranks = guards_mod.domain_world_size(DOMAIN_KERNEL_MESH, r_km, v_km)
    assert mesh_ranks == 2
    mesh_built = driver_mod._build_tp_rank_mesh(mesh_ranks, fallback_mesh=mesh_2)
    assert driver_mod.check_mesh_size(mesh_ranks, int(mesh_built.devices.size)) is None

    g_dcp = agreement_mod.agreement_group(DOMAIN_TP_X_DCP, r_32_8)
    g_km = agreement_mod.agreement_group(DOMAIN_KERNEL_MESH, r_km)
    assert g_dcp is None or hasattr(g_dcp, "size")
    assert g_km is None or hasattr(g_km, "size")


def test_observe_weight_hook_streams_raw_tensors_and_check_spec(tmp_path: Path) -> None:
    mod_path = tmp_path / "obs_plugin.py"
    mod_path.write_text(
        'from vllm_torchtpu.megakernel_runtime.plugin import TpuMonoSpec\n'
        'def create(view, options): return None\n'
        'def observe_weight(name, tensor, options): pass\n'
        'SPEC = TpuMonoSpec("obs_test", "1", create=create, '
        'observe_weight=".obs_plugin:observe_weight")\n',
        encoding="utf-8",
    )

    spec_with_obs = TpuMonoSpec(
        "obs_test",
        "1",
        create="obs_mod:create",
        observe_weight=".obs_plugin:observe_weight",
    )
    spec_no_obs = TpuMonoSpec("obs_test", "1", create="obs_mod:create")

    testkit.check_spec(spec_with_obs, lazy_create=True)
    bad_spec = TpuMonoSpec(
        "obs_test",
        "1",
        create="obs_mod:create",
        observe_weight=lambda n, t, o: None,
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

    def hook_fn(name: str, tensor: torch.Tensor, options: Any = None) -> None:
        observed.append(name)

    mock_plugin = SimpleNamespace(
        observe_weight=hook_fn,
        layout_request=None,
    )
    vllm_config = SimpleNamespace(
        additional_config={"megakernel": {"enabled": True}},
        model_config=SimpleNamespace(hf_config=SimpleNamespace(architectures=["ArchTest"])),
    )

    weights = [
        ("layer.0.w13_weight_scale_2", torch.tensor([1.0, 2.0])),
        ("layer.0.weight", torch.tensor([3.0, 4.0])),
    ]

    with mock.patch("vllm_torchtpu.megakernel_runtime.hooks.select_plugin") as mock_sel:
        mock_sel.return_value = (
            SimpleNamespace(enabled=True),
            ("plugin_ref", {}),
        )
        imp_target = "vllm_torchtpu.megakernel_runtime.hooks.import_plugin"
        with mock.patch(imp_target) as mock_imp:
            mock_imp.return_value = (mock_plugin, types.ModuleType("dummy"))
            streamed = list(hooks_mod.observe_weight_stream(vllm_config, weights))

    assert observed == ["layer.0.w13_weight_scale_2", "layer.0.weight"]
    assert len(streamed) == 2
    assert streamed[0][0] == "layer.0.w13_weight_scale_2"

    disabled_cfg = SimpleNamespace(
        additional_config={"megakernel": {"enabled": False}},
        model_config=SimpleNamespace(hf_config=SimpleNamespace(architectures=["ArchTest"])),
    )
    passthrough = list(hooks_mod.observe_weight_stream(disabled_cfg, weights))
    assert passthrough == weights

    # Verify TPUModelRunner.load_model model_loader wrapper path via subprocess
    runner_check_code = (
        "import sys\n"
        "from unittest import mock\n"
        "import vllm.v1.attention.backends.utils as utils\n"
        'utils.get_kv_cache_layout = getattr(utils, "get_kv_cache_layout", lambda *a, **k: None)\n'
        'sys.modules["vllm_torchtpu.runner.mm_encoder_manager"] = mock.MagicMock()\n'
        "import types\n"
        "from types import SimpleNamespace\n"
        "from typing import Any\n"
        "import torch\n"
        "from vllm_torchtpu.runner.tpu_runner import TPUModelRunner\n"
        "class DummyModel(torch.nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.loaded_weights = {}\n"
        "    def load_weights(self, weights):\n"
        "        for name, tensor in weights:\n"
        "            self.loaded_weights[name] = tensor\n"
        "class DummyLoader:\n"
        "    def __init__(self, raw):\n"
        "        self.raw = raw\n"
        "    def get_all_weights(self, model_config, model):\n"
        "        yield from self.raw\n"
        "    def load_model(self, vllm_config, model_config):\n"
        "        model = DummyModel()\n"
        "        model.load_weights(self.get_all_weights(model_config, model))\n"
        '        model.loaded_weights.pop("layer.0.w13_weight_scale_2", None)\n'
        "        return model\n"
        "observed_in_runner = []\n"
        "def runner_observe(name, tensor, options=None):\n"
        "    observed_in_runner.append(name)\n"
        "mock_runner_plugin = SimpleNamespace(observe_weight=runner_observe, layout_request=None)\n"
        "runner_vllm_cfg = SimpleNamespace(\n"
        '    additional_config={"megakernel": {"enabled": True}},\n'
        '    model_config=SimpleNamespace(hf_config=SimpleNamespace(architectures=["ArchTest"]), quantization=None),\n'
        "    load_config=SimpleNamespace(),\n"
        "    mesh=None,\n"
        ")\n"
        'raw_checkpoint_weights = [("layer.0.w13_weight_scale_2", torch.tensor([0.5, 1.5])), ("layer.0.weight", torch.tensor([2.0, 3.0]))]\n'
        "loader = DummyLoader(raw_checkpoint_weights)\n"
        "orig_device = torch.device\n"
        'def fake_device(d):\n    return orig_device("cpu") if d == "tpu" else orig_device(d)\n'
        "with (\n"
        '    mock.patch("torch.device", side_effect=fake_device),\n'
        '    mock.patch("vllm_torchtpu.runner.tpu_runner.get_model_loader", return_value=loader),\n'
        '    mock.patch("vllm_torchtpu.runner.tpu_runner.get_tpu_quantization_config", return_value=None),\n'
        '    mock.patch("vllm_torchtpu.runner.tpu_runner.validate_online_fp8", return_value=None),\n'
        '    mock.patch("vllm_torchtpu.megakernel_runtime.hooks.select_plugin", return_value=(SimpleNamespace(enabled=True), ("plugin_ref", {}))),\n'
        '    mock.patch("vllm_torchtpu.megakernel_runtime.hooks.import_plugin", return_value=(mock_runner_plugin, types.ModuleType("dummy"))),\n'
        "):\n"
        "    runner_instance = SimpleNamespace(\n"
        "        vllm_config=runner_vllm_cfg,\n"
        "        model_config=runner_vllm_cfg.model_config,\n"
        "        load_config=runner_vllm_cfg.load_config,\n"
        '        device_config=SimpleNamespace(device=SimpleNamespace(type="cpu")),\n'
        "        mesh=None,\n"
        "        is_pooling_model=False,\n"
        "        _is_async_drafter=False,\n"
        "        _initialize_pallas_kernels=lambda: None,\n"
        "        parallel_config=SimpleNamespace(pipeline_parallel_size=1),\n"
        "    )\n"
        "    TPUModelRunner.load_model(runner_instance)\n"
        'assert "layer.0.w13_weight_scale_2" in observed_in_runner, observed_in_runner\n'
        'assert "layer.0.weight" in observed_in_runner, observed_in_runner\n'
        'assert "layer.0.w13_weight_scale_2" not in runner_instance.model.loaded_weights\n'
        'print("OBSERVE_RUNNER_OK")\n'
    )
    cmd_env = {k: v for k, v in os.environ.items() if k != "VLLM_PLUGINS"}
    res_runner = subprocess.run(
        [sys.executable, "-c", runner_check_code],
        env=cmd_env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "OBSERVE_RUNNER_OK" in res_runner.stdout


def test_layout_request_presizing_overrides_slot_and_block_config() -> None:
    def fake_layout_request(vllm_config: Any, options: Any) -> dict[str, Any]:
        return {
            "block_size": 128,
            "cp_kv_cache_interleave_size": 128,
            "mamba_block_size": 640,
            "mamba_slot_stride": 256,
        }

    mock_plugin = SimpleNamespace(
        layout_request=fake_layout_request,
        observe_weight=None,
    )
    vllm_config = SimpleNamespace(
        additional_config={"megakernel": {"enabled": True}},
        cache_config=SimpleNamespace(block_size=16, cp_kv_cache_interleave_size=16),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(architectures=["ArchTest"])),
    )

    with mock.patch("vllm_torchtpu.megakernel_runtime.hooks.select_plugin") as mock_sel:
        mock_sel.return_value = (
            SimpleNamespace(enabled=True),
            ("plugin_ref", {}),
        )
        imp_target = "vllm_torchtpu.megakernel_runtime.hooks.import_plugin"
        with mock.patch(imp_target) as mock_imp:
            mock_imp.return_value = (mock_plugin, types.ModuleType("dummy"))
            res = hooks_mod.apply_layout_request(vllm_config)

    assert res is not None
    assert vllm_config.cache_config.block_size == 128
    assert vllm_config.cache_config.cp_kv_cache_interleave_size == 128
    assert vllm_config.additional_config["mamba_block_size"] == 640
    assert vllm_config.additional_config["mamba_slot_stride"] == 256

    res_idempotent = hooks_mod.apply_layout_request(vllm_config)
    assert res_idempotent is not None

    disabled_cfg = SimpleNamespace(
        additional_config={"megakernel": {"enabled": False}},
        cache_config=SimpleNamespace(block_size=16),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(architectures=["ArchTest"])),
    )
    assert hooks_mod.apply_layout_request(disabled_cfg) is None
    assert disabled_cfg.cache_config.block_size == 16

    presize_check_code = (
        "from types import SimpleNamespace\n"
        "import torch\n"
        "from vllm_torchtpu.platforms.tpu_block_size_utils import _derive_tpu_block_slot_config\n"
        "class FakeBackend:\n"
        "    @staticmethod\n"
        '    def get_name():\n        return "TEST"\n'
        "    @staticmethod\n"
        "    def get_min_page_size(_vllm_config):\n        return 16\n"
        "    @staticmethod\n"
        "    def get_supported_kernel_block_sizes():\n        return [16]\n"
        "    @staticmethod\n"
        "    def get_kv_cache_page_size_bytes(block_size, *_args, **_kwargs):\n        return block_size * 1024\n"
        "vllm_cfg_640 = SimpleNamespace(\n"
        '    additional_config={"mamba_slot_tokens": 640},\n'
        "    cache_config=SimpleNamespace(user_specified_block_size=False, cache_dtype=torch.bfloat16),\n"
        "    model_config=SimpleNamespace(\n"
        "        dtype=torch.bfloat16, get_num_kv_heads=lambda *_: 8, get_head_size=lambda: 128, is_hybrid=False,\n"
        '        hf_config=SimpleNamespace(architectures=["ArchTest"]),\n'
        "    ),\n"
        "    parallel_config=SimpleNamespace(tensor_parallel_size=1),\n"
        ")\n"
        'd_640 = _derive_tpu_block_slot_config(vllm_cfg_640, FakeBackend, input_block_size=16)\n'
        'assert d_640["final_block_size"] == 640, d_640\n'
        "vllm_cfg_256 = SimpleNamespace(\n"
        '    additional_config={"slot_stride_tokens": 256},\n'
        "    cache_config=SimpleNamespace(user_specified_block_size=False, cache_dtype=torch.bfloat16),\n"
        "    model_config=SimpleNamespace(\n"
        "        dtype=torch.bfloat16, get_num_kv_heads=lambda *_: 8, get_head_size=lambda: 128, is_hybrid=False,\n"
        '        hf_config=SimpleNamespace(architectures=["ArchTest"]),\n'
        "    ),\n"
        "    parallel_config=SimpleNamespace(tensor_parallel_size=1),\n"
        ")\n"
        'd_256 = _derive_tpu_block_slot_config(vllm_cfg_256, FakeBackend, input_block_size=16)\n'
        'assert d_256["final_block_size"] == 256, d_256\n'
        'print("PRESIZE_OK")\n'
    )
    cmd_env = {k: v for k, v in os.environ.items() if k != "VLLM_PLUGINS"}
    res_presize = subprocess.run(
        [sys.executable, "-c", presize_check_code],
        env=cmd_env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "PRESIZE_OK" in res_presize.stdout


def test_shorthand_hf_fields_disambiguation_and_verify_tokens_eight() -> None:
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

    prof_8 = SimpleNamespace(buckets=(2,), verify_buckets=(8, 16))
    pol_8 = BucketPolicy.resolve(prof_8, verify_tokens=(8,))
    assert pol_8.verify_tokens == (8,)
    assert (
        BucketPolicy.resolve(
            prof_8, speculative_tokens=7, verify_tokens=(8,)
        ).verify_buckets
        == (8, 16)
    )

    pol_4_8 = BucketPolicy.resolve(
        SimpleNamespace(buckets=(2,), verify_buckets=(4, 8, 16)),
        verify_tokens=(4, 8),
    )
    assert pol_4_8.verify_tokens == (4, 8)
    assert list(pol_4_8.iter_buckets()) == [
        (1, 2),
        (4, 4),
        (4, 8),
        (4, 16),
        (8, 8),
        (8, 16),
    ]

    v_bkts = parse_mk_verify_buckets("8,16,32", verify_tokens=(8,))
    assert v_bkts == (8, 16, 32)

    b_shape_8 = BatchShape(eligible=True, reason="ok", num_reqs=1, tokens_per_seq=8)
    dec_8 = check_step_eligibility(
        b_shape_8,
        pol_8,
        compiled_buckets={(8, 8)},
        bucket_out_rows={(8, 8): 8},
        input_rows=8,
    )
    assert dec_8.eligible is True
    assert dec_8.tokens_per_seq == 8

    pol_def = BucketPolicy()
    dec_def = check_step_eligibility(
        b_shape_8,
        pol_def,
        compiled_buckets={(1, 8)},
        bucket_out_rows={},
        input_rows=8,
    )
    assert dec_def.eligible is False
    assert dec_def.reason == "stock_q8"
