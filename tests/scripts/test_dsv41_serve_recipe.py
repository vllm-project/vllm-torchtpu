# SPDX-License-Identifier: Apache-2.0
"""Exercise emitted serve argv without loading weights or initializing TPU."""

import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.cpu_test


def test_cached_recipe_selects_policy_without_kernel_env(tmp_path):
    vllm = tmp_path / "vllm"
    vllm.write_text(
        "#!/usr/bin/env python3\nimport json,os,sys\n"
        "print(json.dumps({'argv':sys.argv[1:], 'env':dict(os.environ)}))\n"
    )
    vllm.chmod(0o755)
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("TPU_DEEPSEEK_V41_", "MOE_", "USE_MOE_"))
    }
    env["PATH"] = str(tmp_path) + os.pathsep + env["PATH"]
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/dsv41_bench/serve_cached_prefill.sh"),
            "--max-num-seqs",
            "4",
            "--compilation-config",
            '{"compile_sizes":[24,128,256,2048]}',
        ],
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    output = json.loads(result.stdout)
    argv = output["argv"]
    assert json.loads(argv[argv.index("--additional-config") + 1]) == {
        "deepseek_v41": {"execution": "ced", "moe": "fused_ep_w4a8"}
    }
    assert json.loads(argv[argv.index("--engram-config") + 1]) == {"cpu_offload": True}
    assert not any(
        k.startswith(("TPU_DEEPSEEK_V41_", "MOE_", "USE_MOE_")) for k in output["env"]
    )
    assert json.loads(argv[-1])["compile_sizes"] == [24, 128, 256, 2048]


def test_quality_recipe_uses_dp8_and_full_forward():
    recipe = ROOT / "scripts/vllm/benchmarking/configs/deepseek-v41-flash-dp8-ep.sh"
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; printf "%s\\n" "$TENSOR_PARALLELISM" '
            '"$DATA_PARALLELISM" "$EXTRA_SERVE_ARGS"',
            "bash",
            str(recipe),
        ],
        env={**os.environ, "EXTRA_SERVE_ARGS": ""},
        check=True,
        text=True,
        capture_output=True,
    )
    tp, dp, args = result.stdout.splitlines()
    assert (tp, dp) == ("1", "8")
    # The benchmark runner splits EXTRA_SERVE_ARGS without shell eval.
    additional = next(
        x.split("=", 1)[1] for x in args.split() if x.startswith("--additional-config=")
    )
    assert json.loads(additional)["deepseek_v41"]["execution"] == "full"
    assert "--block-size=16" in shlex.split(args)
