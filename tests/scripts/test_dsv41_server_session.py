# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for CI launches and profiling on an already warmed server."""

import json
import os
import signal
import subprocess
import sys
import threading
from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts/dsv41_bench/server_steps"
sys.path.insert(0, str(_SCRIPTS))

import run as launcher  # noqa: E402

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    "prefix,extend,alignment,expected",
    [
        (98304, 0, 256, 98048),
        (98304, 0, 512, 97792),
        (98304, 16384, 512, 98304),
        (98305, 0, 512, 98304),
        (98305, 16384, 512, 98304),
    ],
)
def test_decode_prefix_hits_follow_finalized_cache_alignment(
    prefix, extend, alignment, expected
):
    from bench_decode import expected_prefix_hit

    assert expected_prefix_hit(prefix, extend, alignment) == expected


def test_decode_prefix_hit_rejects_invalid_alignment():
    from bench_decode import expected_prefix_hit

    with pytest.raises(ValueError, match="alignment must be positive"):
        expected_prefix_hit(98304, 0, 0)


def test_decode_admission_waits_for_all_current_requests(tmp_path, monkeypatch):
    import bench_decode as client

    scenario = replace(
        client.SCENARIOS["decode_dp16"],
        prefix_tokens=16,
        dp_size=2,
        concurrency=4,
    )
    monkeypatch.setitem(client.SCENARIOS, "decode_dp16", scenario)
    inputs = tmp_path / "inputs.json"
    inputs.write_text(
        json.dumps(
            {
                "prompts": [[i] * 17 for i in range(4)],
                "prime_suffix": [[99, 100]] * 4,
                "manifest": {},
            }
        )
    )
    log = tmp_path / "server.log"
    log.write_text("Added request cmpl-old-0-01234567.\n" * 4)
    arrived, resume = threading.Event(), threading.Event()
    lock = threading.Lock()
    requests = []
    acknowledged = set()
    polls = 0

    def response(value):
        return SimpleNamespace(
            content=b"json", json=lambda: value, raise_for_status=lambda: None
        )

    def post(url, json, **kwargs):
        if url.endswith("/v1/completions"):
            with lock:
                requests.append((json, kwargs["headers"]))
                if len(requests) == 4:
                    arrived.set()
            assert resume.wait(5)
            return response({"usage": {"prompt_tokens": 16, "completion_tokens": 512}})
        if url.endswith("/resume"):
            # HTTP threads starting is not enough: all post-send log records
            # must have been observed before any request is allowed to execute.
            assert len(acknowledged) == 4
            resume.set()
        return response({})

    def sleep(_):
        nonlocal polls
        assert arrived.wait(5)
        polls += 1
        with log.open("a") as stream:
            for payload, _ in requests[: 2 if polls == 1 else 4]:
                request_id = payload["request_id"]
                acknowledged.add(request_id)
                # Duplicate acknowledgements must not satisfy missing IDs.
                stream.write(f"Added request cmpl-{request_id}-0-01234567.\n" * 2)

    monkeypatch.setattr(client.requests, "post", post)
    monkeypatch.setattr(client.time, "sleep", sleep)
    client.client_main(
        [
            "--scenario",
            "decode_dp16",
            "--input",
            str(inputs),
            "--out",
            str(tmp_path / "profile"),
            "--server-log",
            str(log),
            "--profile-only",
        ]
    )
    assert polls == 2
    assert len({payload["request_id"] for payload, _ in requests}) == 4
    assert sorted(headers["X-data-parallel-rank"] for _, headers in requests) == [
        "0",
        "0",
        "1",
        "1",
    ]


def test_decode_admission_timeout_rejects_old_and_incomplete_logs(tmp_path):
    from bench_decode import wait_for_admission

    log = tmp_path / "server.log"
    log.write_text("Added request cmpl-wave-1-0-01234567.\n")
    offset = log.stat().st_size
    with log.open("a") as stream:
        stream.write("Received request cmpl-wave-1-0: params: ...\n")
        stream.write("Added request cmpl-wave-0-0-01234567.\n" * 2)
        stream.write("Added request cmpl-wave-1-0-01234567.")
    with pytest.raises(TimeoutError, match="1/2 requests missing"):
        wait_for_admission(
            log, offset, {"cmpl-wave-0-0", "cmpl-wave-1-0"}, [], timeout=0
        )
    with log.open("a") as stream:
        stream.write("\n")
    wait_for_admission(log, offset, {"cmpl-wave-0-0", "cmpl-wave-1-0"}, [])


@pytest.mark.parametrize("failed", [False, True])
def test_decode_admission_rejects_early_completion(tmp_path, failed):
    from bench_decode import wait_for_admission

    log = tmp_path / "server.log"
    log.touch()
    future = Future()
    if failed:
        future.set_exception(ValueError("request failed"))
    else:
        future.set_result({})
    with pytest.raises(
        ValueError if failed else RuntimeError,
        match="request failed" if failed else "scheduler was paused",
    ):
        wait_for_admission(log, 0, {"cmpl-wave-0-0"}, [future])


@pytest.mark.parametrize(
    "scenario,rounds,buckets",
    [("encoder_extend", 5, [128, 2048]), ("decode", 1, [288, 2048])],
)
@pytest.mark.parametrize("profile", ["disabled", "passed", "failed"])
@pytest.mark.parametrize("model_uri", [None, "gs://test-bucket/checkpoint"])
def test_ci_launch_and_optional_profile_share_one_server(
    tmp_path,
    monkeypatch,
    scenario,
    rounds,
    buckets,
    profile,
    model_uri,
    checker_failure=None,
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
            *(
                ["--head-address", "127.0.0.1", "--repetition-request", "fixture.json"]
                if scenario == "decode_dp16"
                else []
            ),
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
    if scenario == "decode_dp16":
        monkeypatch.setenv("HOST", "0.0.0.0")
        monkeypatch.setenv("MOE_FUSED_EP_KERNEL_IMPL", "v2")
        monkeypatch.setenv("MOE_FUSED_EP_PUSH_SC_PREPUSH", "1")
        monkeypatch.setenv("MOE_FUSED_EP_PUSH_TC_EXCHANGE", "0")
        monkeypatch.setenv("TPU_MOE_EPLB_DYNAMIC", "0")
    compiler_roots = []
    starts, clients, stopped, audits = [], [], [], []
    server = SimpleNamespace(poll=lambda: None)

    def check_wait(**kwargs):
        assert kwargs["timeout"] == 2100
        if checker_failure == "timeout":
            raise subprocess.TimeoutExpired("checker", 2100)
        return 1 if checker_failure else 0

    checker = SimpleNamespace(wait=check_wait)

    def start(command, *, env, **kwargs):
        if "check" in command:
            assert len(clients) == 1 and len(starts) == 1
            assert not (tmp_path / "results/formal-complete").exists()
            assert "--request" in command and "check_repetition.py" in command[1]
            return checker
        starts.append(command)
        compiler_roots.append(Path(env["VLLM_CACHE_ROOT"]))
        assert compiler_roots[-1].is_dir()
        assert env["USE_PHASED_PROFILER"] == "0"
        assert env["TP"] == "1"
        assert env["DP"] == ("16" if scenario == "decode_dp16" else "8")
        if scenario in ("decode", "decode_dp16"):
            assert "--enable-log-requests" in command
            assert command[command.index("--max-log-len") + 1] == "0"
        if scenario == "decode_dp16":
            assert env["HEADLESS"] == "0"
            assert env["HOST"] == "127.0.0.1"
            assert env["TPU_DEEPSEEK_V41_ENGRAM_HBM"] == "first"
            assert env["MOE_FUSED_EP_KERNEL_IMPL"] == "push"
            assert env["MOE_FUSED_EP_PUSH_SC_PREPUSH"] == "2"
            assert env["MOE_FUSED_EP_PUSH_TC_EXCHANGE"] == "1"
            assert env["TPU_MOE_EPLB_DYNAMIC"] == "1"
            assert env["MOE_FUSED_EP_PREPROCESS_BACKEND"] == "topology"
            assert env["MAX_SEQS"] == "48"
            assert env["MAX_MODEL_LEN"] == "524288"
            assert env["LD_PRELOAD"] == ""
            assert env["TPU_RECLAIM_COMPILE_MEMORY"] == "1"
            assert json.loads(env["ENGRAM_CONFIG"]) == {
                "cpu_offload": True,
                "dp_shared_memory": False,
            }
            assert command[command.index("--data-parallel-size-local") + 1] == "8"
            # An explicit start rank would make vLLM infer hybrid LB, which
            # rejects the headless worker used by this single-API deployment.
            assert "--data-parallel-start-rank" not in command
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
        if scenario == "decode_dp16":
            assert command[command.index("--scenario") + 1] == "decode_dp16"
            (out.parent / "shutdown-permitted").touch()
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
    if checker_failure:
        with pytest.raises((RuntimeError, subprocess.TimeoutExpired)):
            launcher.main()
        assert not (tmp_path / "results/formal-complete").exists()
    elif profile == "failed":
        with pytest.raises(subprocess.CalledProcessError):
            launcher.main()
    else:
        launcher.main()
    assert len(starts) == 1
    assert stopped == ([checker, server] if scenario == "decode_dp16" else [server])
    assert not (tmp_path / "persistent-cache").exists()
    assert all(not root.exists() for root in compiler_roots)
    manifest = json.loads((tmp_path / "results/manifest.json").read_text())
    assert manifest["cache_mode"] == "disabled"
    if scenario == "decode_dp16":
        assert manifest["moe_kernel"] == "push"
        assert manifest["moe_push_sc_prepush"] == 2
        assert manifest["moe_push_tc_exchange"] is True
        assert manifest["moe_eplb_dynamic"] is True
        assert manifest["packages"]["tpu-sync-torch"] == "1"
    assert manifest["serialization"] == "enabled"
    assert manifest["engram_transfer"] == "raw_dma"
    assert manifest["identity"]["engram_transfer"] == "raw_dma"
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
    assert status["completed"] == (profile != "failed" and not checker_failure)
    if scenario == "decode_dp16":
        assert status["repetition_status"] == (
            "failed" if checker_failure else "passed"
        )


def test_dp16_ci_launch_keeps_local_shapes_and_selects_hbm(tmp_path, monkeypatch):
    test_ci_launch_and_optional_profile_share_one_server(
        tmp_path,
        monkeypatch,
        "decode_dp16",
        1,
        [288, 2048],
        "disabled",
        "gs://test-bucket/checkpoint",
    )


@pytest.mark.parametrize("failure", ["loop", "timeout"])
def test_dp16_repetition_failure_closes_server_without_success_marker(
    tmp_path, monkeypatch, failure
):
    test_ci_launch_and_optional_profile_share_one_server(
        tmp_path,
        monkeypatch,
        "decode_dp16",
        1,
        [288, 2048],
        "disabled",
        "gs://test-bucket/checkpoint",
        checker_failure=failure,
    )


@pytest.mark.parametrize("early_exit", [False, True])
def test_dp16_worker_waits_for_shutdown_and_never_runs_client(
    tmp_path, monkeypatch, early_exit
):
    monkeypatch.setenv("TPU_MOE_EPLB_DYNAMIC", "0")
    monkeypatch.setenv("MODEL_PATH", str(tmp_path))
    (tmp_path / "config.json").write_text("{}")
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(__file__=__file__))
    monkeypatch.setattr(launcher, "source_identity", lambda *a, **kw: {})
    monkeypatch.setattr(launcher.importlib.metadata, "version", lambda _: "1")
    monkeypatch.setattr(
        launcher.requests, "get", lambda *a, **kw: SimpleNamespace(ok=False)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--scenario",
            "decode_dp16",
            "--input",
            "unused.json",
            "--out",
            str(tmp_path / "worker"),
            "--host-rank",
            "1",
            "--head-address",
            "127.0.0.1",
        ],
    )
    handlers, commands, stopped = {}, [], []
    monkeypatch.setattr(launcher.signal, "signal", lambda s, f: handlers.update({s: f}))
    monkeypatch.setattr(
        launcher.time, "sleep", lambda _: handlers[signal.SIGTERM](signal.SIGTERM, None)
    )
    server = SimpleNamespace(poll=lambda: 17 if early_exit else None, returncode=17)

    def start(command, *, env, **kwargs):
        commands.append(command)
        assert env["HEADLESS"] == "1" and env["DP"] == "16"
        assert env["HOST"] == "127.0.0.1"
        assert env["MOE_FUSED_EP_KERNEL_IMPL"] == "push"
        assert env["MOE_FUSED_EP_PUSH_SC_PREPUSH"] == "2"
        assert env["MOE_FUSED_EP_PUSH_TC_EXCHANGE"] == "1"
        assert env["TPU_MOE_EPLB_DYNAMIC"] == "1"
        assert command[command.index("--data-parallel-start-rank") + 1] == "8"
        return server

    monkeypatch.setattr(launcher.subprocess, "Popen", start)
    monkeypatch.setattr(
        launcher.subprocess, "run", lambda *a, **kw: pytest.fail("worker ran client")
    )
    monkeypatch.setattr(launcher, "stop", stopped.append)
    if early_exit:
        with pytest.raises(RuntimeError, match="Worker server exited"):
            launcher.main()
    else:
        launcher.main()
    timing = json.loads((tmp_path / "worker/startup.json").read_text())
    assert timing["completed"] == (not early_exit)
    assert timing["formal_status"] == "pending"
    assert len(commands) == 1 and stopped == [server]


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
        return response(
            {
                "results": [
                    {"rank": rank, "prefix_hit_alignment": 16} for rank in range(8)
                ]
            }
        )

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
    assert command[command.index("--api-server-count") + 1] == "1"
    draft = json.loads(command[command.index("--speculative-config") + 1])
    assert draft["model"] == command[1]
    assert "--profiler-config" not in command
