# SPDX-License-Identifier: Apache-2.0
"""Exercise emitted serve argv without loading weights or initializing TPU."""

import json
import os
import shlex
import shutil
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


@pytest.mark.parametrize("model_uri", ["", "gs://test-models/DeepSeek-V4.1-Flash"])
def test_quality_recipe_uses_c384_dspark_decode(model_uri):
    recipe = ROOT / "scripts/vllm/benchmarking/configs/deepseek-v41-flash-dp8-ep.sh"
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; printf "%s\\n" "$TENSOR_PARALLELISM" '
            '"$DATA_PARALLELISM" "$MAX_NUM_SEQS" "$MAX_MODEL_LEN" '
            '"$MAX_NUM_BATCHED_TOKENS" "$EXTRA_LM_EVAL_MODEL_ARGS" '
            '"$MOE_FUSED_EP_PREPROCESS_BACKEND" "$EXTRA_SERVE_ARGS"',
            "bash",
            str(recipe),
        ],
        env={**os.environ, "EXTRA_SERVE_ARGS": "", "MODEL_URI": model_uri},
        check=True,
        text=True,
        capture_output=True,
    )
    tp, dp, seqs, context, budget, client, preprocess, args = result.stdout.splitlines()
    assert (tp, dp, seqs, context, budget) == ("1", "8", "48", "98816", "2048")
    assert "num_concurrent=384" in client.split(",")
    assert preprocess == "topology"
    # The benchmark runner splits EXTRA_SERVE_ARGS without shell eval.
    options = dict(x.split("=", 1) for x in args.split() if "=" in x)
    assert json.loads(options["--additional-config"])["deepseek_v41"] == {
        "execution": "ced",
        "moe": "fused_ep_w4a8",
    }
    assert json.loads(options["--compilation-config"])["compile_sizes"] == [288, 2048]
    assert json.loads(options["--speculative-config"]) == {
        "method": "dspark",
        "num_speculative_tokens": 5,
        "model": model_uri or "deepseek-ai/DeepSeek-V4.1-Flash",
    }
    assert "--block-size=16" in shlex.split(args)


@pytest.mark.parametrize("limit", [None, "none", "384"])
def test_eval_flow_forwards_sample_limit(tmp_path, limit):
    scripts = tmp_path / "scripts/vllm/benchmarking"
    scripts.mkdir(parents=True)
    shutil.copy(ROOT / "scripts/vllm/benchmarking/run_eval_flow.sh", scripts)
    results = tmp_path / "results"
    results.mkdir()
    (results / "config.json").write_text(
        json.dumps(
            {
                "model": "test-model",
                "lm_eval_tasks": "gsm8k",
                "extra_lm_eval_model_args": "num_concurrent=384",
                "eval_tolerance": 0.03,
            }
        )
    )
    for name, body in {
        "run_benchmarks.sh": "#!/bin/sh\nexit 0\n",
        "upload_results.py": "",
    }.items():
        script = scripts / name
        script.write_text(body)
        script.chmod(0o755)
    binary = tmp_path / "bin"
    binary.mkdir()
    for name, body in {
        "lm_eval": '#!/bin/sh\nprintf "%s\\n" "$@" > "$EVAL_ARGV"\n',
        "stdbuf": '#!/bin/sh\nshift\nexec "$@"\n',
    }.items():
        script = binary / name
        script.write_text(body)
        script.chmod(0o755)
    argv_path = tmp_path / "eval-argv.txt"
    command = [
        "bash",
        str(scripts / "run_eval_flow.sh"),
        "--config",
        "test-model",
        "--host",
        "localhost",
        "--results-dir",
        str(results),
        "--run-lm-eval",
        "--skip-benchmark",
        "--skip-db-upload",
    ]
    if limit is not None:
        command += ["--lm-eval-limit", limit]
    subprocess.run(
        command,
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": str(binary) + os.pathsep + os.environ["PATH"],
            "EVAL_ARGV": str(argv_path),
        },
        check=True,
        capture_output=True,
        text=True,
    )
    argv = argv_path.read_text().splitlines()
    if limit == "none":
        assert "--limit" not in argv
    else:
        assert argv[argv.index("--limit") + 1] == (limit or "100")
    assert "num_concurrent=384" in argv[argv.index("--model_args") + 1]
