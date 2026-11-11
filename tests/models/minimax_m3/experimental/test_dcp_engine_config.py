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
"""DCP beyond a KV head's replicas passes `create_engine_config` for MiniMax M3.

vLLM caps GQA DCP at tp // num_kv_heads (2 for M3 at TP 8). The API server
builds and validates its VllmConfig before platform activation, so the patch
lifting the cap must already be installed by the general plugin (model
registration). Runs in a fresh process, like `vllm serve`, so no earlier test
has activated the platform.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = """
import sys
from vllm.plugins import load_general_plugins

load_general_plugins()  # what the CLI does before building the config
from vllm.config.model import ModelConfig

assert getattr(
    ModelConfig.verify_with_parallel_config, "_tpu_gqa_dcp_limit_patch", False
), "GQA DCP cap patch not installed by the general plugin"
from vllm.platforms import current_platform

current_platform.device_type = "cpu"  # no TPU in CI
from vllm.engine.arg_utils import EngineArgs

config = EngineArgs(
    model=sys.argv[1],
    tensor_parallel_size=8,
    decode_context_parallel_size=int(sys.argv[2]),
    skip_tokenizer_init=True,
    max_model_len=4096,
    block_size=128,
    kv_cache_dtype="fp8",
    enforce_eager=True,
    language_model_only=True,
).create_engine_config()
parallel = config.parallel_config
print(
    "RESULT",
    parallel.decode_context_parallel_size,
    parallel.cp_kv_cache_interleave_size,
    config.cache_config.prefix_match_unit,
)
"""


def _config() -> dict:
    """The released checkpoint's architecture and head counts, tiny otherwise
    (no weights are read)."""
    return {
        "architectures": ["MiniMaxM3SparseForConditionalGeneration"],
        "model_type": "minimax_m3_vl",
        "text_config": {
            "hidden_size": 512,
            "intermediate_size": 256,
            "num_hidden_layers": 2,
            "num_attention_heads": 64,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "vocab_size": 1024,
            "max_position_embeddings": 8192,
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
            "sparse_attention_config": {
                "use_sparse_attention": True,
                "sparse_index_dim": 128,
                "sparse_num_index_heads": 4,
                "sparse_topk_blocks": 16,
                "sparse_block_size": 128,
                "sparse_score_type": "max",
                "sparse_init_block": 0,
                "sparse_local_block": 1,
                "sparse_attention_freq": [1, 1],
                "sparse_disable_index_value": [1, 1],
            },
        },
    }


@pytest.mark.parametrize("dcp", [8, 2])
def test_create_engine_config_accepts_dcp_up_to_tp(tmp_path, dcp):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(_config()))
    script = tmp_path / "build_config.py"
    script.write_text(_SCRIPT)
    src = Path(__file__).resolve().parents[3] / "src"
    env = dict(
        os.environ,
        PYTHONPATH=f"{src}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
        JAX_PLATFORMS="cpu",
        # The TPU platform reads the slice topology from the hardware.
        TORCH_TPU_TOPOLOGY="2x2x1",
    )
    result = subprocess.run(
        [sys.executable, str(script), str(model_dir), str(dcp)],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        timeout=600,
    )
    lines = [line for line in result.stdout.splitlines() if line.startswith("RESULT")]
    assert result.returncode == 0 and lines, result.stderr[-3000:]
    assert lines[-1].split()[1:] == [str(dcp), "128", "128"]
