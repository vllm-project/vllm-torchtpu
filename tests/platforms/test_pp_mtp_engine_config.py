# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 MTP under pipeline parallelism through create_engine_config.

The engine args path the API server takes: SpeculativeConfig verifies the
draft (`deepseek_mtp`, not SupportsPP) against the target's pipeline size.
TPU_PP_MTP_PREFILL_KV makes it verify as one stage; without it the upstream
refusal stands. Uses the real GLM-5.2-FP8 config.json; no weights are read.
"""

import json
import os
import shutil

import pytest
from vllm.engine.arg_utils import EngineArgs

import vllm_torchtpu  # noqa: F401  (applies the import-stage patches)
from vllm_torchtpu import envs

CONFIG = os.path.join(os.path.dirname(__file__), "data", "glm52_fp8_config.json")


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    shutil.copy(CONFIG, tmp_path / "config.json")
    # Stage 1 must start on a layer with its own indexer (the served split).
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "42,36")
    monkeypatch.setenv("VLLM_USE_RAY_V2_EXECUTOR_BACKEND", "1")
    return str(tmp_path)


def _engine_args(model_dir):
    return EngineArgs(
        model=model_dir,
        skip_tokenizer_init=True,
        pipeline_parallel_size=2,
        tensor_parallel_size=1,
        async_scheduling=False,
        max_model_len=4096,
        max_num_batched_tokens=4096,
        block_size=256,
        speculative_config={"method": "mtp", "num_speculative_tokens": 1},
        kv_transfer_config=json.dumps(
            {
                "kv_connector": "TPURaidenOffloadingConnector",
                "kv_connector_module_path": "vllm_torchtpu.offload.raiden_connector",
                "kv_role": "kv_both",
                "kv_connector_extra_config": {"cpu_bytes_to_use": 1 << 30},
            }
        ),
    )


def _gate(monkeypatch, on):
    monkeypatch.setattr(envs, "TPU_PP_MTP_PREFILL_KV", on)


def test_engine_config_admits_mtp_under_pp_with_the_gate(model_dir, monkeypatch):
    _gate(monkeypatch, True)
    config = _engine_args(model_dir).create_engine_config()
    assert config.parallel_config.pipeline_parallel_size == 2
    spec = config.speculative_config
    assert spec is not None and spec.method == "mtp"
    assert spec.draft_model_config.hf_config.model_type == "deepseek_mtp"
    # The draft keeps the target's pipeline size; only its check sees one stage.
    assert spec.draft_parallel_config.pipeline_parallel_size == 2


def test_engine_config_refuses_mtp_under_pp_without_the_gate(model_dir, monkeypatch):
    _gate(monkeypatch, False)
    with pytest.raises(
        NotImplementedError, match="Pipeline parallelism is not supported"
    ):
        _engine_args(model_dir).create_engine_config()
