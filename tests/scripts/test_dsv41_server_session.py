# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for CI launches and profiling on an already warmed server."""

import json
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
    [("encoder_extend", 5, [128, 256, 2048]), ("decode", 1, [288, 2048])],
)
@pytest.mark.parametrize("profile", ["disabled", "passed", "failed"])
def test_ci_launch_and_optional_profile_share_one_server(
    tmp_path, monkeypatch, scenario, rounds, buckets, profile
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.json").write_text("{}")
    monkeypatch.setenv("MODEL_PATH", str(tmp_path))
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
            "--cache-dir",
            "cache",
            "--cache-mode",
            "cold",
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
    starts, clients, stopped, audits = [], [], [], []
    server = SimpleNamespace(poll=lambda: None)

    def start(command, *, env, **kwargs):
        starts.append(command)
        assert env["USE_PHASED_PROFILER"] == "0"
        assert env["TP"] == "1" and env["DP"] == "8"
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
