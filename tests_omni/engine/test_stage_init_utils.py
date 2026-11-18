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
"""TPU tests for vLLM-Omni stage initialization and environment isolation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
import yaml
from transformers import Qwen3OmniMoeConfig
from vllm_omni.config.resolver import resolve_omni_config
from vllm_omni.engine import stage_runtime as stage_runtime_mod
from vllm_omni.engine.stage_client import StageClientBase
from vllm_omni.engine.stage_engine_startup import StageReplicaResources
from vllm_omni.engine.stage_init_utils import (
    ReplicaInitPlan,
    build_vllm_config,
    extract_stage_metadata_from_omni_stage_config,
    stage_runtime_env,
)
from vllm_omni.engine.stage_runtime import StageRuntime
from vllm_omni.platforms import current_omni_platform
from vllm_omni.worker.omni_connector_validation import validate_worker_omni_connector

from tests_omni.omni_test_utils import PR_TIMEOUT_S
from vllm_torchtpu.executors.tpu_multiproc_executor import TpuMultiprocExecutor
from vllm_torchtpu.platforms.tpu_platform import TpuPlatform

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]

PARALLEL_CASES = [
    (0, (2, 1, 2, True)),
    (1, (1, 2, 1, False)),
    (2, (2, 1, 1, False)),
]


def _write_tiny_qwen3_omni_config(model_dir: Path) -> Path:
    cfg = Qwen3OmniMoeConfig()
    cfg.architectures = ["Qwen3OmniMoeForConditionalGeneration"]
    cfg.thinker_config.text_config.num_hidden_layers = 2
    cfg.talker_config.text_config.num_hidden_layers = 2
    cfg.save_pretrained(model_dir)
    return model_dir


def _resolve_parallel_stages(model_dir: Path, deploy_path: Path) -> tuple[Any, ...]:
    common = {
        "skip_tokenizer_init": True,
        "enforce_eager": True,
        "max_model_len": 128,
        "max_num_seqs": 2,
        "async_scheduling": False,
    }
    deploy_doc = {
        "pipeline": "qwen3_omni_moe",
        "async_chunk": False,
        "connectors": {"shm_conn": {"name": "SharedMemoryConnector"}},
        "stages": [
            {
                **common,
                "stage_id": 0,
                "devices": "0,1,2,3",
                "tensor_parallel_size": 2,
                "enable_expert_parallel": True,
                "data_parallel_size": 2,
                "pipeline_parallel_size": 1,
                "output_connectors": {"to_stage_1": "shm_conn"},
            },
            {
                **common,
                "stage_id": 1,
                "devices": "4,5",
                "tensor_parallel_size": 1,
                "enable_expert_parallel": False,
                "data_parallel_size": 1,
                "pipeline_parallel_size": 2,
                "input_connectors": {"from_stage_0": "shm_conn"},
                "output_connectors": {"to_stage_2": "shm_conn"},
            },
            {
                **common,
                "stage_id": 2,
                "devices": "6,7",
                "tensor_parallel_size": 2,
                "enable_expert_parallel": False,
                "data_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "input_connectors": {"from_stage_1": "shm_conn"},
            },
        ],
    }
    deploy_path.write_text(yaml.safe_dump(deploy_doc), encoding="utf-8")
    resolution = resolve_omni_config(
        str(model_dir),
        trust_remote_code=False,
        deploy_config_path=str(deploy_path),
        cli_overrides=None,
        stage_overrides=None,
        strategy_config_path=None,
    )
    return resolution.stage_configs


class _DummyStageClient(StageClientBase):
    stage_type = "diffusion"
    replica_id = 0
    is_comprehension = False

    def __init__(self, stage_id: int) -> None:
        self.stage_id = stage_id
        self.final_output = stage_id == 1
        self.final_output_type = "video"

    def shutdown(self) -> None:
        return None


def _assert_env_snapshot(
    snapshot: dict[str, str | None],
    expected: dict[str, str | None],
    *,
    label: str,
) -> None:
    for key, want in expected.items():
        got = snapshot.get(key)
        assert got == want, f"{label}: env[{key!r}]={got!r}, expected {want!r}"


@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason=(
        "TPUWorker does not provide an Omni connector model runner "
        "(tracked in vllm-torchtpu PR #2065)"
    ),
)
@pytest.mark.parametrize(
    "stage_id,expected_degrees",
    PARALLEL_CASES,
    ids=["ar_tp2_dp2_ep", "ar_pp2", "gen_tp2"],
)
def test_per_stage_tp_pp_dp_ep_ar_and_gen_stages(
    stage_id: int,
    expected_degrees: tuple[int, int, int, bool],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-stage TP / PP / DP / EP (AR and GEN stages).

    - TPU path: run ``merge_pipeline_deploy`` and ``build_vllm_config`` on AR and
      GEN stages of ``QWEN3_OMNI_PIPELINE`` under ``TpuPlatform`` and
      ``OmniTpuPlatform``.
    - Setup check: assert ``current_omni_platform.device_name == "tpu"`` and
      merged stage engine args carry the configured ``(tp, pp, dp, ep)`` tuple.
    - Outcome check: assert ``VllmConfig.parallel_config`` preserves per-stage
      degrees, selects ``TpuMultiprocExecutor``, and keeps an
      ``OmniConnectorModelRunnerMixin`` worker class.
    """
    monkeypatch.setattr(
        TpuPlatform, "device_control_env_var", TpuPlatform.device_control_env_var
    )
    for env_key in (
        "TORCH_TPU_DP_SIZE",
        "TORCH_TPU_DP_MASTER_ADDR",
        "TORCH_TPU_DP_MASTER_PORT",
        "TORCH_TPU_SLICEBUILDER_ADDRESSES",
        "TORCH_TPU_TOPOLOGY",
    ):
        monkeypatch.delenv(env_key, raising=False)

    assert current_omni_platform.device_name == "tpu", (
        "Expected active omni platform 'tpu', "
        f"got {current_omni_platform.device_name!r}"
    )

    model_dir = _write_tiny_qwen3_omni_config(tmp_path / "qwen3_omni_cfg")
    stages = _resolve_parallel_stages(model_dir, tmp_path / "qwen3_parallel.yaml")
    stage_cfg = stages[stage_id]
    stage_pc = stage_cfg.parallel_config
    actual_stage_degrees = (
        stage_pc.tensor_parallel_size,
        stage_pc.pipeline_parallel_size,
        stage_pc.data_parallel_size,
        stage_pc.enable_expert_parallel,
    )
    assert actual_stage_degrees == expected_degrees, (
        f"stage {stage_id}: resolved stage degrees {actual_stage_degrees} "
        f"!= expected {expected_degrees}"
    )

    connector_spec = {
        "name": "SharedMemoryConnector",
        "extra": {"role": "sender"},
    }
    vllm_config, executor_cls = build_vllm_config(
        stage_cfg,
        str(model_dir),
        stage_connector_spec=connector_spec,
    )

    pc = vllm_config.parallel_config
    actual_pc_degrees = (
        pc.tensor_parallel_size,
        pc.pipeline_parallel_size,
        pc.data_parallel_size,
        pc.enable_expert_parallel,
    )
    assert actual_pc_degrees == expected_degrees, (
        f"stage {stage_id}: ParallelConfig degrees {actual_pc_degrees} "
        f"!= expected {expected_degrees}"
    )
    assert executor_cls is TpuMultiprocExecutor, (
        f"stage {stage_id}: executor_cls {executor_cls} is not TpuMultiprocExecutor"
    )
    validate_worker_omni_connector(pc.worker_cls, required=True)


def test_per_stage_env_isolation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-Stage Environment Variable Isolation.

    - TPU path: resolve a two-stage deploy YAML with per-stage and
      ``platforms.tpu`` env overrides, then run ``stage_runtime_env`` and
      ``StageRuntime._initialize_local_diffusion_replica``.
    - Setup check: assert ``current_omni_platform.device_name == "tpu"``,
      ``device_control_env_var == "TPU_VISIBLE_CHIPS"``, and distinct merged
      per-stage ``runtime_config.env`` dicts.
    - Outcome check: assert stage 0 sees base and TPU-overridden env vars,
      stage 1 sees only its own env vars, and parent ``os.environ`` is restored
      on both normal exit and launch failure.
    """
    tracked_keys = (
        "TPU_VISIBLE_CHIPS",
        "OMNI_STAGE_SHARED",
        "OMNI_STAGE0_ONLY",
        "OMNI_STAGE1_ONLY",
        "OMNI_TPU_STAGE0_FLAG",
    )
    monkeypatch.setenv("TPU_VISIBLE_CHIPS", "baseline_chips")
    monkeypatch.setenv("OMNI_STAGE_SHARED", "parent_value")
    monkeypatch.delenv("OMNI_STAGE0_ONLY", raising=False)
    monkeypatch.delenv("OMNI_STAGE1_ONLY", raising=False)
    monkeypatch.delenv("OMNI_TPU_STAGE0_FLAG", raising=False)
    baseline_snapshot = {k: os.environ.get(k) for k in tracked_keys}

    assert current_omni_platform.device_name == "tpu", (
        f"Expected omni platform 'tpu', got {current_omni_platform.device_name!r}"
    )
    assert current_omni_platform.device_control_env_var == "TPU_VISIBLE_CHIPS", (
        "Expected device_control_env_var 'TPU_VISIBLE_CHIPS', "
        f"got {current_omni_platform.device_control_env_var!r}"
    )

    model_dir = tmp_path / "wan_stub"
    model_dir.mkdir()
    (model_dir / "model_index.json").write_text(
        json.dumps({"_class_name": "WanPipeline"}),
        encoding="utf-8",
    )
    deploy_path = tmp_path / "wan_env_isolation.yaml"
    deploy_doc = {
        "pipeline": "wan2_2_eg",
        "async_chunk": False,
        "connectors": {"wan_encode_connector": {"name": "SharedMemoryConnector"}},
        "stages": [
            {
                "stage_id": 0,
                "devices": "0",
                "output_connectors": {"to_stage_1": "wan_encode_connector"},
                "env": {
                    "OMNI_STAGE0_ONLY": "s0_base",
                    "OMNI_STAGE_SHARED": "stage0_base",
                },
            },
            {
                "stage_id": 1,
                "devices": "1",
                "input_connectors": {"from_stage_0": "wan_encode_connector"},
                "env": {
                    "OMNI_STAGE1_ONLY": "s1_base",
                    "OMNI_STAGE_SHARED": "stage1_base",
                },
            },
        ],
        "platforms": {
            "tpu": {
                "stages": [
                    {
                        "stage_id": 0,
                        "env": {
                            "OMNI_STAGE_SHARED": "stage0_tpu",
                            "OMNI_TPU_STAGE0_FLAG": "1",
                            "TPU_VISIBLE_CHIPS": "0",
                        },
                    },
                    {
                        "stage_id": 1,
                        "env": {"TPU_VISIBLE_CHIPS": "1"},
                    },
                ]
            }
        },
    }
    deploy_path.write_text(yaml.safe_dump(deploy_doc), encoding="utf-8")

    resolution = resolve_omni_config(
        str(model_dir),
        trust_remote_code=False,
        deploy_config_path=str(deploy_path),
        cli_overrides=None,
        stage_overrides=None,
        strategy_config_path=None,
    )
    stage0_cfg, stage1_cfg = resolution.stage_configs
    expected_stage0_env = {
        "OMNI_STAGE0_ONLY": "s0_base",
        "OMNI_STAGE_SHARED": "stage0_tpu",
        "OMNI_TPU_STAGE0_FLAG": "1",
        "TPU_VISIBLE_CHIPS": "0",
    }
    expected_stage1_env = {
        "OMNI_STAGE1_ONLY": "s1_base",
        "OMNI_STAGE_SHARED": "stage1_base",
        "TPU_VISIBLE_CHIPS": "1",
    }
    assert stage0_cfg.runtime_config.env == expected_stage0_env, (
        f"stage 0 merged env {stage0_cfg.runtime_config.env} != {expected_stage0_env}"
    )
    assert stage1_cfg.runtime_config.env == expected_stage1_env, (
        f"stage 1 merged env {stage1_cfg.runtime_config.env} != {expected_stage1_env}"
    )

    with stage_runtime_env(0, stage0_cfg.runtime_config):
        direct_snapshot = {k: os.environ.get(k) for k in tracked_keys}
    _assert_env_snapshot(
        direct_snapshot,
        {
            "TPU_VISIBLE_CHIPS": "0",
            "OMNI_STAGE_SHARED": "stage0_tpu",
            "OMNI_STAGE0_ONLY": "s0_base",
            "OMNI_STAGE1_ONLY": None,
            "OMNI_TPU_STAGE0_FLAG": "1",
        },
        label="stage_runtime_env(stage 0)",
    )
    _assert_env_snapshot(
        {k: os.environ.get(k) for k in tracked_keys},
        baseline_snapshot,
        label="post stage_runtime_env baseline",
    )

    runtime = StageRuntime(
        stage_configs=list(resolution.stage_configs),
        model=str(model_dir),
        config_path=str(deploy_path),
        stage_init_timeout=5,
        async_chunk=False,
    )
    runtime._init_visible_devices_baseline = None
    observed_launches: dict[int, dict[str, str | None]] = {}
    should_fail = False

    def _record_launch(
        **kwargs: Any,
    ) -> tuple[_DummyStageClient, StageReplicaResources]:
        sid = kwargs["metadata"].stage_id
        observed_launches[sid] = {k: os.environ.get(k) for k in tracked_keys}
        if should_fail:
            raise RuntimeError("simulated diffusion replica launch failure")
        return _DummyStageClient(sid), StageReplicaResources()

    monkeypatch.setattr(
        stage_runtime_mod, "launch_diffusion_stage_replica", _record_launch
    )

    plan0 = ReplicaInitPlan(
        replica_id=0,
        num_replicas=1,
        launch_mode="local",
        stage_cfg=stage0_cfg,
        metadata=extract_stage_metadata_from_omni_stage_config(stage0_cfg),
        stage_connector_spec={},
        omni_kv_connector=({}, None, None),
    )
    plan1 = ReplicaInitPlan(
        replica_id=0,
        num_replicas=1,
        launch_mode="local",
        stage_cfg=stage1_cfg,
        metadata=extract_stage_metadata_from_omni_stage_config(stage1_cfg),
        stage_connector_spec={},
        omni_kv_connector=({}, None, None),
    )

    runtime._initialize_local_diffusion_replica(plan0, stage_init_timeout=5)
    _assert_env_snapshot(
        {k: os.environ.get(k) for k in tracked_keys},
        baseline_snapshot,
        label="parent after stage 0 normal exit",
    )

    runtime._initialize_local_diffusion_replica(plan1, stage_init_timeout=5)
    _assert_env_snapshot(
        observed_launches[0],
        {
            "TPU_VISIBLE_CHIPS": "0",
            "OMNI_STAGE_SHARED": "stage0_tpu",
            "OMNI_STAGE0_ONLY": "s0_base",
            "OMNI_STAGE1_ONLY": None,
            "OMNI_TPU_STAGE0_FLAG": "1",
        },
        label="stage 0 launch env",
    )
    _assert_env_snapshot(
        observed_launches[1],
        {
            "TPU_VISIBLE_CHIPS": "1",
            "OMNI_STAGE_SHARED": "stage1_base",
            "OMNI_STAGE0_ONLY": None,
            "OMNI_STAGE1_ONLY": "s1_base",
            "OMNI_TPU_STAGE0_FLAG": None,
        },
        label="stage 1 launch env",
    )
    _assert_env_snapshot(
        {k: os.environ.get(k) for k in tracked_keys},
        baseline_snapshot,
        label="parent after stage 1 normal exit",
    )

    should_fail = True
    with pytest.raises(
        RuntimeError, match="simulated diffusion replica launch failure"
    ):
        runtime._initialize_local_diffusion_replica(plan0, stage_init_timeout=5)
    _assert_env_snapshot(
        {k: os.environ.get(k) for k in tracked_keys},
        baseline_snapshot,
        label="parent after stage 0 launch failure",
    )
