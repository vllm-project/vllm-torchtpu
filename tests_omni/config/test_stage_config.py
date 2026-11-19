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
"""TPU L0 tests for vllm_omni.config.stage_config and config resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from transformers import Qwen3OmniMoeConfig
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm_omni.config.resolver import resolve_omni_config
from vllm_omni.config.stage_config import (
    DeployConfig,
    StageDeployConfig,
    _apply_platform_overrides,
    load_deploy_config,
    merge_pipeline_deploy,
    resolve_deploy_yaml,
)
from vllm_omni.engine.arg_utils import OmniEngineArgs
from vllm_omni.engine.stage_init_utils import (
    build_engine_args_dict_from_omni_stage_config,
    build_vllm_config,
    filter_dataclass_kwargs,
)
from vllm_omni.model_executor.models.qwen3_omni.pipeline import (
    QWEN3_OMNI_THINKER_ONLY_PIPELINE,
)
from vllm_omni.platforms import current_omni_platform
from vllm_omni.worker.mixins import OmniWorkerMixin

from tests_omni.omni_test_utils import PR_TIMEOUT_S
from vllm_torchtpu.omni.platform import OmniTpuPlatform

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]

BASE_STAGE_YAML = (
    "stages:\n"
    "  - stage_id: 0\n"
    '    devices: "0"\n'
    "    env:\n"
    "      BASE_VAR: base\n"
    "      PLATFORM_VAR: default\n"
    "    default_sampling_params:\n"
    "      temperature: 0.6\n"
    "      top_k: 20\n"
)
PLATFORMS_YAML = (
    "platforms:\n"
    "  tpu:\n"
    "    stages:\n"
    "      - stage_id: 0\n"
    '        devices: "0,1"\n'
    "        tensor_parallel_size: 2\n"
    "        env:\n"
    "          PLATFORM_VAR: tpu\n"
    "          TPU_ONLY_VAR: active\n"
    "        default_sampling_params:\n"
    "          temperature: 0.8\n"
    "  npu:\n"
    "    stages:\n"
    "      - stage_id: 0\n"
    '        devices: "7"\n'
    "        env:\n"
    "          PLATFORM_VAR: npu\n"
    "  rocm:\n"
    "    stages:\n"
    "      - stage_id: 0\n"
    "        default_sampling_params:\n"
    "          temperature: 0.1\n"
    "  xpu:\n"
    "    stages:\n"
    "      - stage_id: 0\n"
    "        tensor_parallel_size: 8\n"
)


def _save_qwen3_omni_config(
    tmp_path: Path,
    *,
    enable_audio_output: bool = True,
) -> Path:
    model_dir = tmp_path / "qwen3_omni_tiny"
    Qwen3OmniMoeConfig(enable_audio_output=enable_audio_output).save_pretrained(
        model_dir
    )
    return model_dir


@pytest.mark.parametrize("model_runner", ["v1", "v2"])
def test_stage_model_runner_selection(
    tmp_path: Path,
    model_runner: str,
) -> None:
    """Stage Model Runner Selection (v1 vs v2).

    - Resolve QWEN3_OMNI_THINKER_ONLY_PIPELINE with _apply_platform_overrides
      and merge_pipeline_deploy on TPU for model_runner v1 vs v2, then build
      VllmConfig on a local Qwen3OmniMoeConfig.
    - Setup check: current_omni_platform.device_name == "tpu" and merged
      yaml_engine_args["use_v2_model_runner"] matches v1/v2.
    - Outcome check: vllm_config.model_config.use_v2_model_runner matches v1/v2
      and parallel_config.worker_cls subclasses OmniWorkerMixin.
    """
    assert current_omni_platform.device_name == "tpu", (
        f"Expected TPU omni platform, got {current_omni_platform.device_name!r}"
    )
    model_dir = _save_qwen3_omni_config(tmp_path, enable_audio_output=False)

    deploy = DeployConfig(
        stages=[StageDeployConfig(stage_id=0, model_runner=model_runner)],
    )
    deploy = _apply_platform_overrides(deploy)
    merged = merge_pipeline_deploy(QWEN3_OMNI_THINKER_ONLY_PIPELINE, deploy)
    expected_v2 = model_runner == "v2"
    assert merged[0].yaml_engine_args["use_v2_model_runner"] is expected_v2, (
        f"Expected yaml_engine_args['use_v2_model_runner'] is {expected_v2}, "
        f"got {merged[0].yaml_engine_args['use_v2_model_runner']!r}"
    )

    engine_args_dict = {
        **merged[0].yaml_engine_args,
        "model": str(model_dir),
        "stage_id": merged[0].stage_id,
        "model_stage": merged[0].model_stage,
        "worker_type": merged[0].worker_type,
        "skip_tokenizer_init": True,
    }
    vllm_config, _ = build_vllm_config(
        merged[0], str(model_dir), engine_args_dict=engine_args_dict
    )
    assert vllm_config.model_config.use_v2_model_runner is expected_v2, (
        f"Expected use_v2_model_runner is {expected_v2}, "
        f"got {vllm_config.model_config.use_v2_model_runner!r}"
    )
    worker_cls = resolve_obj_by_qualname(vllm_config.parallel_config.worker_cls)
    assert issubclass(worker_cls, OmniWorkerMixin), (
        f"Expected worker_cls {worker_cls!r} to subclass OmniWorkerMixin"
    )


@pytest.mark.parametrize(
    ("moe_backend", "linear_backend"),
    [("auto", "auto"), ("triton", "torch")],
)
def test_ar_stage_attention_moe_and_linear_backends(
    tmp_path: Path,
    moe_backend: str,
    linear_backend: str,
) -> None:
    """AR Stage Attention, MoE, and Linear Backends.

    - Load a qwen3_omni deploy YAML setting attention_backend, moe_backend,
      and linear_backend on AR stage 0 (plus ignored top-level keys), project
      stage 0, and build OmniEngineArgs.create_engine_config() on TPU.
    - Setup check: top-level backend keys are ignored on other stages while
      stage 0 projects the backends and get_omni_ar_worker_cls().
    - Outcome check: VllmConfig preserves worker_cls, FLASH_ATTN backend, and
      kernel_config moe_backend/linear_backend.
    """
    assert current_omni_platform.device_name == "tpu", (
        f"Expected TPU omni platform, got {current_omni_platform.device_name!r}"
    )
    model_dir = _save_qwen3_omni_config(tmp_path, enable_audio_output=True)
    deploy_path = tmp_path / "ar_backends.yaml"
    deploy_path.write_text(
        "attention_backend: TRITON_ATTN\n"
        "moe_backend: torch\n"
        "linear_backend: torch\n"
        "stages:\n"
        "  - stage_id: 0\n"
        "    attention_backend: FLASH_ATTN\n"
        f"    moe_backend: {moe_backend}\n"
        f"    linear_backend: {linear_backend}\n"
        "  - stage_id: 1\n"
        "    max_num_seqs: 4\n",
        encoding="utf-8",
    )

    resolution = resolve_omni_config(
        str(model_dir),
        trust_remote_code=False,
        deploy_config_path=str(deploy_path),
        cli_overrides=None,
        stage_overrides=None,
        strategy_config_path=None,
    )
    stage0_args = build_engine_args_dict_from_omni_stage_config(
        resolution.stage_by_id(0),
        str(model_dir),
    )
    stage1_args = build_engine_args_dict_from_omni_stage_config(
        resolution.stage_by_id(1),
        str(model_dir),
    )

    assert "attention_backend" not in stage1_args, (
        "Top-level attention_backend leaked into stage 1"
    )
    assert stage1_args.get("moe_backend") != "torch", (
        "Top-level moe_backend leaked into stage 1"
    )
    assert stage1_args.get("linear_backend") != "torch", (
        "Top-level linear_backend leaked into stage 1"
    )
    assert stage0_args["attention_backend"] == "FLASH_ATTN", (
        f"Expected stage 0 attention_backend 'FLASH_ATTN', "
        f"got {stage0_args.get('attention_backend')!r}"
    )
    assert stage0_args["moe_backend"] == moe_backend, (
        f"Expected stage 0 moe_backend {moe_backend!r}, "
        f"got {stage0_args.get('moe_backend')!r}"
    )
    assert stage0_args["linear_backend"] == linear_backend, (
        f"Expected stage 0 linear_backend {linear_backend!r}, "
        f"got {stage0_args.get('linear_backend')!r}"
    )
    expected_worker = current_omni_platform.get_omni_ar_worker_cls()
    assert stage0_args["worker_cls"] == expected_worker, (
        f"Expected worker_cls {expected_worker!r}, "
        f"got {stage0_args.get('worker_cls')!r}"
    )

    stage0_args["skip_tokenizer_init"] = True
    filtered = filter_dataclass_kwargs(OmniEngineArgs, stage0_args)
    vllm_config = OmniEngineArgs(**filtered).create_engine_config()
    assert vllm_config.parallel_config.worker_cls == expected_worker, (
        f"Expected VllmConfig worker_cls {expected_worker!r}, "
        f"got {vllm_config.parallel_config.worker_cls!r}"
    )
    assert vllm_config.attention_config.backend.name == "FLASH_ATTN", (
        f"Expected attention backend 'FLASH_ATTN', "
        f"got {vllm_config.attention_config.backend.name!r}"
    )
    assert vllm_config.kernel_config.moe_backend == moe_backend, (
        f"Expected kernel_config.moe_backend {moe_backend!r}, "
        f"got {vllm_config.kernel_config.moe_backend!r}"
    )
    assert vllm_config.kernel_config.linear_backend == linear_backend, (
        f"Expected kernel_config.linear_backend {linear_backend!r}, "
        f"got {vllm_config.kernel_config.linear_backend!r}"
    )


@pytest.mark.parametrize(
    (
        "yaml_text",
        "cli_overrides",
        "stage_overrides",
        "expected_s0",
        "expected_s1",
    ),
    [
        pytest.param(
            "stages:\n"
            "  - stage_id: 0\n"
            "    gpu_memory_utilization: 0.55\n"
            "  - stage_id: 1\n"
            "    gpu_memory_utilization: 0.65\n",
            None,
            None,
            0.55,
            0.65,
            id="yaml_per_stage",
        ),
        pytest.param(
            "stages:\n"
            "  - stage_id: 0\n"
            "    gpu_memory_utilization: 0.55\n"
            "  - stage_id: 1\n"
            "    gpu_memory_utilization: 0.65\n"
            "platforms:\n"
            "  tpu:\n"
            "    stages:\n"
            "      - stage_id: 0\n"
            "        gpu_memory_utilization: 0.45\n",
            None,
            None,
            0.45,
            0.65,
            id="platforms_tpu_override",
        ),
        pytest.param(
            "stages:\n"
            "  - stage_id: 0\n"
            "    gpu_memory_utilization: 0.55\n"
            "  - stage_id: 1\n"
            "    gpu_memory_utilization: 0.65\n"
            "platforms:\n"
            "  tpu:\n"
            "    stages:\n"
            "      - stage_id: 0\n"
            "        gpu_memory_utilization: 0.45\n",
            {"gpu_memory_utilization": 0.72},
            None,
            0.72,
            0.72,
            id="cli_overrides_precedence",
        ),
        pytest.param(
            "stages:\n"
            "  - stage_id: 0\n"
            "    gpu_memory_utilization: 0.55\n"
            "  - stage_id: 1\n"
            "    gpu_memory_utilization: 0.65\n"
            "platforms:\n"
            "  tpu:\n"
            "    stages:\n"
            "      - stage_id: 0\n"
            "        gpu_memory_utilization: 0.45\n",
            {"gpu_memory_utilization": 0.72},
            {
                "0": {"gpu_memory_utilization": 0.35},
                "1": {"gpu_memory_utilization": 0.85},
            },
            0.35,
            0.85,
            id="stage_overrides_highest_precedence",
        ),
    ],
)
def test_per_stage_gpu_memory_utilization(
    tmp_path: Path,
    yaml_text: str,
    cli_overrides: dict[str, Any] | None,
    stage_overrides: dict[str, dict[str, Any]] | None,
    expected_s0: float,
    expected_s1: float,
) -> None:
    """Per-stage GPU memory budget (gpu_memory_utilization).

    - Resolve multi-stage pipeline configs via load_deploy_config and
      resolve_omni_config with OmniTpuPlatform active across YAML,
      platforms.tpu, cli_overrides, and stage_overrides precedence.
    - Setup check: current_omni_platform is OmniTpuPlatform (device_name "tpu")
      and resolution yields distinct multi-stage configs.
    - Outcome check: each stage's cache_config.gpu_memory_utilization equals
      its expected fraction, and fractions outside (0.0, 1.0] raise ValueError.
    """
    assert current_omni_platform.device_name == "tpu", (
        f"Expected TPU omni platform, got {current_omni_platform.device_name!r}"
    )
    model_dir = _save_qwen3_omni_config(tmp_path, enable_audio_output=True)
    deploy_path = tmp_path / "gpu_mem.yaml"
    deploy_path.write_text(yaml_text, encoding="utf-8")

    deploy = load_deploy_config(deploy_path)
    assert len(deploy.stages) >= 2, (
        f"Expected >= 2 deploy stages, got {len(deploy.stages)}"
    )

    resolution = resolve_omni_config(
        str(model_dir),
        trust_remote_code=False,
        deploy_config_path=str(deploy_path),
        cli_overrides=cli_overrides,
        stage_overrides=stage_overrides,
        strategy_config_path=None,
    )
    assert len(resolution.stage_configs) >= 2, (
        f"Expected >= 2 resolved stages, got {len(resolution.stage_configs)}"
    )
    actual_s0 = resolution.stage_by_id(0).cache_config.gpu_memory_utilization
    actual_s1 = resolution.stage_by_id(1).cache_config.gpu_memory_utilization
    assert actual_s0 == expected_s0, (
        f"Stage 0 gpu_memory_utilization {actual_s0} != expected {expected_s0}"
    )
    assert actual_s1 == expected_s1, (
        f"Stage 1 gpu_memory_utilization {actual_s1} != expected {expected_s1}"
    )

    bad_deploy_path = tmp_path / "bad_gpu_mem.yaml"
    bad_deploy_path.write_text(
        "stages:\n  - stage_id: 0\n    gpu_memory_utilization: 1.2\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        resolve_omni_config(
            str(model_dir),
            trust_remote_code=False,
            deploy_config_path=str(bad_deploy_path),
            cli_overrides=None,
            stage_overrides=None,
            strategy_config_path=None,
        )


def test_deploy_config_and_base_config_overlays(tmp_path: Path) -> None:
    """Deploy Configuration & Base Config Overlays.

    - Load base + overlay deploy YAMLs using base_config inheritance,
      connectors, deep-merged default_sampling_params/engine_extras, and
      platforms.tpu stage blocks through resolve_omni_config on TPU.
    - Setup check: current_omni_platform.device_name == "tpu",
      OmniTpuPlatform.get_default_stage_config_path() == "vllm_omni/deploy",
      and resolve_deploy_yaml strips base_config during recursive resolution.
    - Outcome check: resolved stage configs match exact expected inherited vs
      overridden scalars, deep-merged sampling/connector dicts, and TPU
      platform-section stage overrides.
    """
    assert current_omni_platform.device_name == "tpu", (
        f"Expected TPU omni platform, got {current_omni_platform.device_name!r}"
    )
    assert OmniTpuPlatform.get_default_stage_config_path() == "vllm_omni/deploy", (
        "Expected OmniTpuPlatform default stage config path 'vllm_omni/deploy', "
        f"got {OmniTpuPlatform.get_default_stage_config_path()!r}"
    )

    model_dir = _save_qwen3_omni_config(tmp_path, enable_audio_output=True)
    base_path = tmp_path / "base.yaml"
    base_path.write_text(
        "connectors:\n"
        "  shared:\n"
        "    name: SharedMemoryConnector\n"
        "    extra:\n"
        "      shm_threshold_bytes: 4096\n"
        "stages:\n"
        "  - stage_id: 0\n"
        "    max_num_seqs: 8\n"
        "    gpu_memory_utilization: 0.75\n"
        "    default_sampling_params:\n"
        "      temperature: 0.7\n"
        "      top_p: 0.9\n"
        "    engine_extras:\n"
        "      additional_config:\n"
        "        base_flag: true\n"
        "        shared_flag: 1\n"
        "  - stage_id: 1\n"
        "    max_num_seqs: 4\n",
        encoding="utf-8",
    )
    overlay_path = tmp_path / "overlay.yaml"
    overlay_path.write_text(
        "base_config: base.yaml\n"
        "connectors:\n"
        "  shared:\n"
        "    extra:\n"
        "      shm_threshold_bytes: 8192\n"
        "stages:\n"
        "  - stage_id: 0\n"
        "    max_num_seqs: 16\n"
        "    default_sampling_params:\n"
        "      top_p: 0.95\n"
        "    engine_extras:\n"
        "      additional_config:\n"
        "        shared_flag: 2\n"
        "platforms:\n"
        "  tpu:\n"
        "    stages:\n"
        "      - stage_id: 1\n"
        "        max_num_seqs: 12\n",
        encoding="utf-8",
    )

    raw_merged = resolve_deploy_yaml(overlay_path)
    assert "base_config" not in raw_merged, (
        "resolve_deploy_yaml did not strip base_config"
    )
    deploy = load_deploy_config(overlay_path)
    assert deploy.connectors["shared"]["name"] == "SharedMemoryConnector", (
        f"Expected inherited connector name, got {deploy.connectors!r}"
    )
    assert deploy.connectors["shared"]["extra"]["shm_threshold_bytes"] == 8192, (
        f"Expected overridden connector threshold 8192, got {deploy.connectors!r}"
    )

    resolution = resolve_omni_config(
        str(model_dir),
        trust_remote_code=False,
        deploy_config_path=str(overlay_path),
        cli_overrides=None,
        stage_overrides=None,
        strategy_config_path=None,
    )
    s0 = resolution.stage_by_id(0)
    s1 = resolution.stage_by_id(1)
    assert s0.scheduler_config.max_num_seqs == 16, (
        f"Expected stage 0 max_num_seqs 16, got {s0.scheduler_config.max_num_seqs}"
    )
    assert s0.cache_config.gpu_memory_utilization == 0.75, (
        "Expected inherited stage 0 gpu_memory_utilization 0.75, "
        f"got {s0.cache_config.gpu_memory_utilization}"
    )
    assert s0.model_config.default_sampling_params["temperature"] == 0.7, (
        "Expected inherited sampling temperature 0.7, "
        f"got {s0.model_config.default_sampling_params!r}"
    )
    assert s0.model_config.default_sampling_params["top_p"] == 0.95, (
        "Expected overridden sampling top_p 0.95, "
        f"got {s0.model_config.default_sampling_params!r}"
    )
    assert s0.runtime_config.additional_config["base_flag"] is True, (
        "Expected inherited additional_config base_flag, "
        f"got {s0.runtime_config.additional_config!r}"
    )
    assert s0.runtime_config.additional_config["shared_flag"] == 2, (
        "Expected overridden additional_config shared_flag=2, "
        f"got {s0.runtime_config.additional_config!r}"
    )
    assert s1.scheduler_config.max_num_seqs == 12, (
        "Expected TPU platform override stage 1 max_num_seqs 12, "
        f"got {s1.scheduler_config.max_num_seqs}"
    )


@pytest.mark.parametrize(
    "deploy_yaml",
    [
        pytest.param(f"{BASE_STAGE_YAML}{PLATFORMS_YAML}", id="direct"),
        pytest.param(
            f"base_config: platform_base.yaml\n{PLATFORMS_YAML}", id="base_config"
        ),
    ],
)
def test_per_platform_overlays_in_deploy_yaml(
    tmp_path: Path,
    deploy_yaml: str,
) -> None:
    """Per-Platform Overlays in Deploy YAML.

    - TPU path: with OmniTpuPlatform active, resolve deploy YAMLs containing
      platforms: {tpu, npu, rocm, xpu} (directly or via base_config) through
      resolve_omni_config and _apply_platform_overrides.
    - Setup check: current_omni_platform.device_name == "tpu" and
      load_deploy_config populates deploy.platforms with tpu and non-TPU blocks.
    - Outcome check: resolved stage_configs[0] applies the tpu overlay (devices,
      env, default_sampling_params, diffusion_config) and equals an equivalent
      flat deploy YAML while ignoring npu/rocm/xpu sections.
    """
    assert current_omni_platform.device_name == "tpu", (
        f"Expected TPU omni platform, got {current_omni_platform.device_name!r}"
    )
    model_dir = _save_qwen3_omni_config(tmp_path, enable_audio_output=False)

    (tmp_path / "platform_base.yaml").write_text(BASE_STAGE_YAML, encoding="utf-8")
    deploy_path = tmp_path / "platform_deploy.yaml"
    deploy_path.write_text(deploy_yaml, encoding="utf-8")

    flat_path = tmp_path / "platform_flat_tpu.yaml"
    flat_path.write_text(
        "stages:\n"
        "  - stage_id: 0\n"
        '    devices: "0,1"\n'
        "    tensor_parallel_size: 2\n"
        "    env:\n"
        "      BASE_VAR: base\n"
        "      PLATFORM_VAR: tpu\n"
        "      TPU_ONLY_VAR: active\n"
        "    default_sampling_params:\n"
        "      temperature: 0.8\n"
        "      top_k: 20\n",
        encoding="utf-8",
    )

    deploy = load_deploy_config(deploy_path)
    assert deploy.platforms is not None, (
        f"Expected deploy.platforms mapping from {deploy_path}, got None"
    )
    assert "tpu" in deploy.platforms, (
        f"Expected 'tpu' in deploy.platforms, got {list(deploy.platforms)!r}"
    )
    assert "npu" in deploy.platforms, (
        f"Expected 'npu' in deploy.platforms, got {deploy.platforms!r}"
    )

    applied = _apply_platform_overrides(load_deploy_config(deploy_path))
    assert applied.stages[0].devices == "0,1", (
        f"Expected _apply_platform_overrides devices '0,1', "
        f"got {applied.stages[0].devices!r}"
    )

    resolved = resolve_omni_config(
        str(model_dir),
        trust_remote_code=False,
        deploy_config_path=str(deploy_path),
        cli_overrides=None,
        stage_overrides=None,
        strategy_config_path=None,
    )
    flat_resolved = resolve_omni_config(
        str(model_dir),
        trust_remote_code=False,
        deploy_config_path=str(flat_path),
        cli_overrides=None,
        stage_overrides=None,
        strategy_config_path=None,
    )

    s0 = resolved.stage_configs[0]
    flat_s0 = flat_resolved.stage_configs[0]
    assert s0.runtime_config.devices == flat_s0.runtime_config.devices == "0,1", (
        f"Expected devices '0,1', got {s0.runtime_config.devices!r}"
    )
    assert s0.runtime_config.env == flat_s0.runtime_config.env, (
        f"Expected env {flat_s0.runtime_config.env!r}, got {s0.runtime_config.env!r}"
    )
    assert (
        s0.model_config.default_sampling_params
        == flat_s0.model_config.default_sampling_params
    ), (
        f"Expected default_sampling_params "
        f"{flat_s0.model_config.default_sampling_params!r}, "
        f"got {s0.model_config.default_sampling_params!r}"
    )
    assert s0.parallel_config == flat_s0.parallel_config, (
        f"Expected parallel_config {flat_s0.parallel_config!r}, "
        f"got {s0.parallel_config!r}"
    )
