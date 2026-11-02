# SPDX-License-Identifier: Apache-2.0
"""Exercise emitted serve argv without loading weights or initializing TPU."""

import ast
import importlib.metadata
import json
import os
import runpy
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace
from typing import Optional

import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("atomic", [False, True])
@pytest.mark.parametrize("interrupted", [False, True])
def test_runai_metadata_read_during_download(
    tmp_path, monkeypatch, atomic, interrupted
):
    """Exercise the installed downloader with a transport paused during writing."""
    dist = importlib.metadata.distribution("runai-model-streamer-gcs")
    source = Path(
        dist.locate_file("runai_model_streamer_gcs/files/files.py")
    ).read_text()
    candidate = tmp_path / "files.py"
    candidate.write_text(source)
    if atomic:
        # Apply the same installer as CI, to an isolated copy of the dependency.
        monkeypatch.setattr(
            importlib.metadata,
            "distribution",
            lambda _: SimpleNamespace(
                version=dist.version, locate_file=lambda _: candidate
            ),
        )
        runpy.run_path(
            str(ROOT / ".buildkite/scripts/patch_runai_gcs_metadata.py"),
            run_name="__main__",
        )
    functions = [
        node
        for node in ast.parse(candidate.read_text()).body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("pull_files", "_safe_destination_path")
    ]
    assert len(functions) == 2
    started, release = Event(), Event()
    errors = []
    payload = b'{"model_type":"deepseek_v41"}'

    class Blob:
        def download_to_filename(self, path):
            with open(path, "wb") as stream:
                started.set()
                assert release.wait(10), "reader did not release download"
                stream.write(payload[:10] if interrupted else payload)
                if interrupted:
                    raise OSError("injected download interruption")

    namespace = {
        "Optional": Optional,
        "List": list,
        "os": os,
        "Path": Path,
        "tempfile": tempfile,
        "_create_client": lambda: SimpleNamespace(
            get_bucket=lambda _: SimpleNamespace(blob=lambda _: Blob())
        ),
        "list_files": lambda *a, **kw: ("bucket", "model", ["model/config.json"]),
    }
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(candidate), "exec"),
        namespace,
    )
    config = tmp_path / "config.json"
    config.write_bytes(payload)

    def download():
        try:
            namespace["pull_files"]("gs://test-bucket/model", str(tmp_path), ["*.json"])
        except Exception as exc:
            errors.append(exc)

    worker = Thread(target=download, daemon=True)
    worker.start()
    try:
        assert started.wait(10), "download did not start"
        if atomic:
            assert json.loads(config.read_text()) == json.loads(payload)
        else:
            with pytest.raises(json.JSONDecodeError):
                json.loads(config.read_text())
    finally:
        release.set()
        worker.join(10)
    assert not worker.is_alive()
    if interrupted:
        assert len(errors) == 1
        assert isinstance(errors[0], OSError)
        assert str(errors[0]) == "injected download interruption"
    else:
        assert errors == []
    assert (config.read_bytes() == payload) is (atomic or not interrupted)
    assert not list(tmp_path.glob(".runai-*"))


def test_runai_metadata_patch_rejects_changed_source():
    patch = runpy.run_path(str(ROOT / ".buildkite/scripts/patch_runai_gcs_metadata.py"))
    with pytest.raises(RuntimeError, match="source changed"):
        patch["patch_source"]("different dependency implementation")


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
    assert argv.count("--compilation-config") == 1


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
