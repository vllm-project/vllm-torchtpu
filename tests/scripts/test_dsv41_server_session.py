# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for CI launches and profiling on an already warmed server."""

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts/dsv41_bench/server_steps"
sys.path.insert(0, str(_SCRIPTS))

import run as launcher  # noqa: E402

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    "scenario,rounds,buckets",
    [("encoder_extend", 5, [128, 2048]), ("decode", 1, [288, 2048])],
)
@pytest.mark.parametrize("profile", ["disabled", "passed", "failed"])
@pytest.mark.parametrize("model_uri", [None, "gs://test-bucket/checkpoint"])
def test_ci_launch_and_optional_profile_share_one_server(
    tmp_path, monkeypatch, scenario, rounds, buckets, profile, model_uri
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.json").write_text("{}")
    monkeypatch.setenv("MODEL_PATH", str(tmp_path))
    monkeypatch.delenv("MODEL_URI", raising=False)
    if model_uri is not None:
        monkeypatch.setenv("MODEL_URI", model_uri)
    # Artifact collection alone must not enable a profiler on a formal CI run.
    monkeypatch.setenv("ARTIFACT_LOCAL_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("USE_PHASED_PROFILER", "1")
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(__file__=__file__))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--scenario",
            scenario,
            "--input",
            "inputs.json",
            "--out",
            "results",
            "--startup-timeout",
            "7200",
            *(["--profile"] if profile != "disabled" else []),
        ],
    )
    monkeypatch.setattr(launcher, "load_frozen_input", lambda _: {})
    monkeypatch.setattr(launcher, "source_identity", lambda *a, **kw: {})
    monkeypatch.setattr(launcher.importlib.metadata, "version", lambda _: "1")
    monkeypatch.setattr(launcher, "bootstrap_ports", lambda *a: None)
    health = iter([SimpleNamespace(ok=False), SimpleNamespace(ok=True)])
    monkeypatch.setattr(launcher.requests, "get", lambda *a, **kw: next(health))
    inherited_roots = (
        "JAX_COMPILATION_CACHE_DIR",
        "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT",
        "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT",
    )
    for key in inherited_roots:
        monkeypatch.setenv(key, str(tmp_path / "persistent-cache"))
    compiler_roots = []
    starts, clients, stopped, audits = [], [], [], []
    server = SimpleNamespace(poll=lambda: None)

    def start(command, *, env, **kwargs):
        starts.append(command)
        compiler_roots.append(Path(env["VLLM_CACHE_ROOT"]))
        assert compiler_roots[-1].is_dir()
        assert env["USE_PHASED_PROFILER"] == "0"
        assert env["TP"] == "1" and env["DP"] == "8"
        assert env["CHUNK"] == "2048"
        assert env["VLLM_DISABLE_COMPILE_CACHE"] == "1"
        assert env["JAX_ENABLE_COMPILATION_CACHE"] == "false"
        assert env["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] == "1"
        assert all(key not in env for key in inherited_roots)
        return server

    def client(command, **kwargs):
        clients.append(command)
        if "--profile-only" in command:
            if profile == "failed":
                raise subprocess.CalledProcessError(1, command)
            return
        out = Path(command[command.index("--out") + 1])
        out.mkdir()
        for i in range(1, rounds + 1):
            (out / f"round-{i}-summary.json").write_text(
                json.dumps({"audit_wall_interval_ns": [2 * i, 2 * i + 1]})
            )

    monkeypatch.setattr(launcher.subprocess, "Popen", start)
    monkeypatch.setattr(launcher.subprocess, "run", client)
    monkeypatch.setattr(launcher, "stop", stopped.append)
    monkeypatch.setattr(
        launcher, "assert_no_compilation", lambda log, x: audits.append(x)
    )
    if profile == "failed":
        with pytest.raises(subprocess.CalledProcessError):
            launcher.main()
    else:
        launcher.main()
    assert len(starts) == 1 and stopped == [server]
    assert not (tmp_path / "persistent-cache").exists()
    assert all(not root.exists() for root in compiler_roots)
    manifest = json.loads((tmp_path / "results/manifest.json").read_text())
    assert manifest["cache_mode"] == "disabled"
    assert manifest["serialization"] == "enabled"
    assert manifest["identity"]["model_weights"] == (model_uri or str(tmp_path))
    command = starts[0]
    assert json.loads(command[command.index("--compilation-config") + 1]) == {
        "compile_sizes": buckets
    }
    assert clients[0][clients[0].index("--rounds") + 1] == str(rounds)
    assert len(audits) == 1 and len(audits[0]) == rounds
    assert len(clients) == (1 if profile == "disabled" else 2)
    assert ("--profiler-config" in command) == (profile != "disabled")
    if profile != "disabled":
        config = json.loads(command[command.index("--profiler-config") + 1])
        assert config["torch_profiler_dir"] == str(tmp_path / "results/profile")
        assert "--profile-only" in clients[1] and "--rounds" not in clients[1]
        assert ("--reuse-prefix" in clients[1]) == (scenario == "decode")
    status = json.loads((tmp_path / "results/startup.json").read_text())
    assert status["formal_status"] == "passed"
    assert status["profile_status"] == profile
    assert status["completed"] == (profile != "failed")


@pytest.mark.parametrize("profile_only,waves", [(False, 6), (True, 1)])
def test_prefill_keeps_formal_warmup_but_skips_duplicate_profile_warmup(
    tmp_path, monkeypatch, profile_only, waves
):
    import bench_cached_extend as client

    scenario = replace(client.SCENARIO, prefix_tokens=16, extend_tokens=16)
    monkeypatch.setattr(client, "SCENARIO", scenario)
    monkeypatch.setattr(client.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        client.hashlib,
        "sha256",
        lambda _: SimpleNamespace(
            hexdigest=lambda: (
                "9ca4086c68c610b9205e1e46e1a51d3d18bb10495ed88d341c63b725ccca4064"
            )
        ),
    )
    monkeypatch.setattr(client, "summarize_extend", lambda _: {"categories": {}})
    inputs = tmp_path / "inputs.json"
    inputs.write_text(
        json.dumps(
            {
                "prompts": [[i] * 32 for i in range(8)],
                "prime_suffix": [[99, 100]] * 8,
                "manifest": {},
            }
        )
    )
    requests, controls = [], []

    def response(value):
        return SimpleNamespace(
            content=b"json", json=lambda: value, raise_for_status=lambda: None
        )

    def post(url, json, **kwargs):
        if url.endswith("/v1/completions"):
            requests.append(json)
            return response(
                {
                    "usage": {
                        "prompt_tokens": len(json["prompt"]),
                        "completion_tokens": 1,
                    }
                }
            )
        controls.append(url)
        return response({"results": [{"rank": rank} for rank in range(8)]})

    def metrics(*args, **kwargs):
        hits = sum(16 for r in requests if len(r["prompt"]) == 32)
        return SimpleNamespace(
            raise_for_status=lambda: None,
            text=(
                f"vllm:request_prefill_time_seconds_count {len(requests)}\n"
                f"vllm:prefix_cache_hits_total {hits}\n"
                "vllm:num_preemptions_total 0\n"
            ),
        )

    monkeypatch.setattr(client.requests, "post", post)
    monkeypatch.setattr(client.requests, "get", metrics)
    out = tmp_path / "profile"
    client.client_main(
        [
            "--input",
            str(inputs),
            "--out",
            str(out),
            *(["--profile-only"] if profile_only else []),
        ]
    )
    assert len(requests) == 16 * waves
    assert sum(url.endswith("/reset_prefix_cache") for url in controls) == waves
    assert sum(url.endswith("/start_profile") for url in controls) == int(profile_only)
    assert sum(url.endswith("/stop_profile") for url in controls) == int(profile_only)
    assert (out / "warmup-summary.json").exists() is not profile_only
    summaries = list(out.glob("round-*-summary.json"))
    assert len(summaries) == (1 if profile_only else 5)
    for path in summaries:
        report = json.loads(path.read_text())
        assert report["profiled"] is profile_only and not report["warmup"]


@pytest.mark.parametrize("model_uri", [None, "gs://test-bucket/checkpoint"])
def test_shell_model_source_and_disabled_compile_cache(tmp_path, model_uri):
    executable = tmp_path / "vllm"
    executable.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$VLLM_DISABLE_COMPILE_CACHE" '
        '"$LIBTPU_INIT_ARGS" "$@"\n'
    )
    executable.chmod(0o755)
    env = {
        **os.environ,
        "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
        "VLLM_DISABLE_COMPILE_CACHE": "1",
        "MODEL_PATH": str(tmp_path),
    }
    env.pop("MODEL_URI", None)
    if model_uri is not None:
        env["MODEL_URI"] = model_uri
    result = subprocess.run(
        ["bash", str(_SCRIPTS.parent / "serve_cached_prefill.sh")],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    cache, compiler_flags, *command = result.stdout.splitlines()
    assert cache == "1"
    assert "--xla_tpu_dvfs_p_state=7" in compiler_flags
    assert command[:2] == ["serve", model_uri or str(tmp_path)]
    assert command[command.index("--tokenizer") + 1] == str(tmp_path)
    draft = json.loads(command[command.index("--speculative-config") + 1])
    assert draft["model"] == command[1]
    assert "--profiler-config" not in command
