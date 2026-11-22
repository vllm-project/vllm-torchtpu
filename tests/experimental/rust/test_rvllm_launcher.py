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
"""Tests for `python -m vllm_torchtpu.experimental.rust` (the rvllm launcher).

The argument split uses vLLM's real `vllm serve` parser. The launcher runs in
process with its exec and export steps recorded; one test runs it as
`python -m` to check that it really replaces itself with the binary.
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[3]

try:
    from vllm_torchtpu.experimental.export import cli as export_cli
    from vllm_torchtpu.experimental.rust import launcher
    from vllm_torchtpu.platforms.tpu_platform import TpuPlatform

    # Off a TPU, vLLM's parser construction builds a DeviceConfig, which fails
    # on `torch.device("tpu")`. Leave the device unset for this one call.
    with patch.object(TpuPlatform, "uses_host_device_handling", return_value=True):
        PARSER = export_cli.build_arg_parser()
    # Without a usable platform device (the CPU CI runner), build_arg_parser()
    # falls back to a minimal parser that also defines --max-model-len, so test
    # for the `model_tag` positional that only the real `vllm serve` parser has.
    if "model_tag" not in {a.dest for a in PARSER._actions}:
        raise RuntimeError("the vllm serve parser is not available")
except Exception as e:  # noqa: BLE001
    pytest.skip(f"vLLM's serve parser is unavailable: {e}", allow_module_level=True)


def split(*argv):
    return launcher.split_args(list(argv), PARSER)


def test_engine_arguments_go_to_the_exporter_and_deployment_flags_to_rvllm():
    s = split(
        "Qwen/Qwen3-0.6B",
        "--max_model_len",
        "2048",
        "-tp=4",
        "--enforce-eager",
        "--speculative-config",
        '{"method": "ngram"}',
        "--port",
        "8001",
        "--served-model-name",
        "qwen",
        "--no-async-scheduling",
        "--load-format",
        "dummy",
        "--role=decode",
        "--kv-producers",
        "tcp:prefill:7100",
    )
    assert s.model == "Qwen/Qwen3-0.6B"
    assert s.export_args == [
        "--max_model_len",
        "2048",
        "-tp=4",
        "--enforce-eager",
        "--speculative-config",
        '{"method": "ngram"}',
        "--load-format",
        "dummy",
    ]
    assert s.serve_args == [
        "--port",
        "8001",
        "--served-model-name",
        "qwen",
        "--no-async-scheduling",
        "--load-format",
        "dummy",
        "--role=decode",
        "--kv-producers",
        "tcp:prefill:7100",
    ]
    assert s.warnings == []


def test_kv_transfer_config_goes_to_rvllm_only():
    config = '{"kv_connector":"TPUConnector","kv_role":"kv_consumer"}'
    s = split("m", "--max-model-len", "8", "--kv-transfer-config", config)
    assert s.export_args == ["--max-model-len", "8"]
    assert s.serve_args == ["--kv-transfer-config", config]
    # Prefill and decode servers of one model share the cached export.
    producer = config.replace("consumer", "producer")
    prefill = split("m", "--kv-transfer-config", producer)
    decode = split("m", "--kv-transfer-config", config)
    assert prefill.export_args == decode.export_args == []


def test_the_model_can_be_given_with_model():
    assert split("--model=m", "--max-model-len", "8").model == "m"
    assert split("--model", "m").model == "m"


@pytest.mark.parametrize(
    "argv, message",
    [
        (["m", "--api-key", "k"], "--api-key is a `vllm serve` frontend option"),
        (["m", "--chat-template", "t.jinja"], "--chat-template is a `vllm serve`"),
        (["m", "--enable-auto-tool-choice"], "frontend option that rvllm does not"),
        (["m", "--no-such-option"], "neither a `vllm serve` option nor"),
        (["m", "--output-dir", "/tmp/x"], "chosen by the launcher"),
        (["m", "--served-model-name", "a", "b"], "one --served-model-name"),
        (["m", "extra"], "unexpected argument 'extra'"),
        (["m", "--role"], "--role needs a value"),
    ],
)
def test_options_rvllm_cannot_honor_fail(argv, message):
    with pytest.raises(launcher.LauncherError) as e:
        split(*argv)
    assert message in str(e.value)


def test_log_only_frontend_options_warn():
    s = split("m", "--uvicorn-log-level", "debug", "--disable-fastapi-docs")
    assert s.export_args == [] and s.serve_args == []
    assert len(s.warnings) == 2
    assert all("only configures vLLM's own API server" in w for w in s.warnings)


FAKE_RVLLM = """#!/bin/bash
printf '%s\\n' "$@" > "${FAKE_LOG:-/dev/null}"
"""


@pytest.mark.parametrize(
    "args",
    [
        ["m", "--rvllm-binary=gs://b", "--port", "1"],
        ["m", "--port", "1", "--rvllm-binary", "gs://b"],
    ],
)
def test_the_binary_option_is_taken_out_anywhere(args):
    assert launcher.take_binary_option(args) == ("gs://b", ["m", "--port", "1"])


@pytest.mark.parametrize(
    "args, message",
    [
        (["--rvllm-binay", "x"], "unknown option --rvllm-binay"),
        (["m", "--rvllm-binary"], "--rvllm-binary needs a value"),
    ],
)
def test_bad_launcher_options_fail(args, message):
    with pytest.raises(launcher.LauncherError) as e:
        launcher.take_binary_option(args)
    assert message in str(e.value)


@pytest.fixture
def run(tmp_path, monkeypatch):
    """Runs the launcher in process, recording what it would exec and export.

    HOME points at a temporary directory, which holds the cache. After the
    subcommand it passes the fake binary, unless the call names its own `binary`
    (None passes none). `run.record` holds what was exec'd and exported.
    """
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    empty_bin = tmp_path / "empty_bin"
    empty_bin.mkdir(exist_ok=True)
    monkeypatch.setenv("PATH", str(empty_bin))
    monkeypatch.delenv("VLLM_TPU_EXPERIMENTAL_USE_RUST", raising=False)
    monkeypatch.delenv("VLLM_TPU_EXPERIMENTAL_RVLLM_BINARY", raising=False)
    rvllm = tmp_path / "rvllm"
    rvllm.write_text(FAKE_RVLLM)
    rvllm.chmod(0o755)
    monkeypatch.setattr(launcher.export_cli, "build_arg_parser", lambda: PARSER)
    record = SimpleNamespace(execs=[], exports=[], rvllm=rvllm)

    def fake_export(command):
        record.exports.append(list(command))
        out = Path(command[command.index("--output-dir") + 1])
        out.mkdir(parents=True)
        (out / "metadata.json").write_text("{}")

    monkeypatch.setattr(launcher, "run_export", fake_export)
    monkeypatch.setattr(launcher, "_exec", lambda argv: record.execs.append(list(argv)))

    def call(command, *args, binary=str(rvllm)):
        options = [] if binary is None else ["--rvllm-binary", binary]
        launcher.main([command, *options, *args])
        return record

    call.record = record
    return call


def test_serve_exports_then_execs_rvllm_and_prints_both_commands(run, capsys):
    r = run("serve", "m", "--max-model-len", "2048", "--port", "8001")
    (export,) = r.exports
    partial = export[export.index("--output-dir") + 1]
    artifact = partial.removesuffix(".partial")
    assert export == [
        sys.executable,
        "-m",
        "vllm_torchtpu.experimental.export",
        "m",
        "--output-dir",
        partial,
        "--export-mode",
        "all",
        "--max-model-len",
        "2048",
    ]
    assert r.execs == [[str(r.rvllm), "serve", artifact, "--port", "8001"]]
    assert (Path(artifact) / ".complete").exists()
    err = capsys.readouterr().err
    assert "export command:" in err
    assert f"serve command: {r.rvllm} serve {artifact} --port 8001" in err


def test_serve_reuses_the_export_for_the_same_model_arguments(run):
    run("serve", "m", "--max-model-len", "2048", "--port", "1")
    r = run("serve", "--model=m", "--max-model-len", "2048", "--port", "2")
    assert len(r.exports) == 1, "deployment flags do not change the export"
    first = r.execs[0][2]
    assert r.execs[1][2] == first
    run("serve", "m", "--max-model-len", "4096")
    assert len(r.exports) == 2
    assert r.execs[2][2] != first


def test_serve_fails_on_an_unsupported_option_before_anything_runs(run, capsys):
    with pytest.raises(SystemExit) as e:
        run("serve", "m", "--api-key", "secret")
    assert e.value.code == 2
    assert "--api-key" in capsys.readouterr().err


def test_serve_warns_about_log_only_options(run, capsys):
    r = run("serve", "m", "--uvicorn-log-level", "debug")
    assert "warning: ignoring --uvicorn-log-level" in capsys.readouterr().err
    assert r.execs[0][3:] == []


def test_bench_execs_vllm_bench_and_needs_no_binary(run):
    launcher.main(["bench", "serve", "--max-concurrency", "8"])
    assert run.record.execs == [
        [sys.executable, "-m", "vllm.entrypoints.cli.main", "bench", "serve"]
        + ["--max-concurrency", "8"]
    ]


def test_other_subcommands_exec_the_binary(run):
    r = run("proxy", "--port", "8000")
    assert r.execs == [[str(r.rvllm), "proxy", "--port", "8000"]]


@pytest.mark.parametrize(
    "binary, message",
    [(None, "pass --rvllm-binary"), ("/no/such/path", "neither a file, a directory")],
)
def test_the_binary_option_is_checked(run, capsys, binary, message):
    with pytest.raises(SystemExit):
        run("proxy", binary=binary)
    assert message in capsys.readouterr().err


def test_the_binary_can_be_provided_via_vllm_tpu_rvllm_binary(run, monkeypatch):
    monkeypatch.setenv("VLLM_TPU_EXPERIMENTAL_RVLLM_BINARY", str(run.record.rvllm))
    r = run("proxy", "--port", "8000", binary=None)
    assert r.execs == [[str(r.rvllm), "proxy", "--port", "8000"]]


def _fake_binary(directory):
    """An executable fake `rvllm` in `directory`."""
    directory.mkdir()
    binary = directory / "rvllm"
    binary.write_text(FAKE_RVLLM)
    binary.chmod(0o755)
    return binary


def test_the_binary_is_found_on_path(run, tmp_path, monkeypatch):
    on_path = _fake_binary(tmp_path / "on_path")
    monkeypatch.setenv("PATH", str(on_path.parent))
    r = run("proxy", "--port", "8000", binary=None)
    assert r.execs == [[str(on_path), "proxy", "--port", "8000"]]


def test_the_env_var_wins_over_path(run, tmp_path, monkeypatch):
    on_path = _fake_binary(tmp_path / "on_path")
    from_env = _fake_binary(tmp_path / "from_env")
    monkeypatch.setenv("PATH", str(on_path.parent))
    monkeypatch.setenv("VLLM_TPU_EXPERIMENTAL_RVLLM_BINARY", str(from_env))
    r = run("proxy", binary=None)
    assert r.execs == [[str(from_env), "proxy"]]


def test_the_option_wins_over_the_env_var_and_path(run, tmp_path, monkeypatch):
    on_path = _fake_binary(tmp_path / "on_path")
    from_env = _fake_binary(tmp_path / "from_env")
    from_option = _fake_binary(tmp_path / "from_option")
    monkeypatch.setenv("PATH", str(on_path.parent))
    monkeypatch.setenv("VLLM_TPU_EXPERIMENTAL_RVLLM_BINARY", str(from_env))
    r = run("proxy", binary=str(from_option))
    assert r.execs == [[str(from_option), "proxy"]]


def test_the_binary_error_mentions_vllm_tpu_rvllm_binary(run, capsys):
    with pytest.raises(SystemExit):
        run("proxy", binary=None)
    err = capsys.readouterr().err
    assert "VLLM_TPU_EXPERIMENTAL_RVLLM_BINARY" in err


@pytest.mark.parametrize(
    "env_val, argv, should_intercept",
    [
        (None, ["vllm", "serve", "m", "--port", "8000"], False),
        ("1", ["vllm", "bench", "serve"], False),
        ("1", ["vllm-omni", "serve", "m"], False),
        ("1", ["vllm", "serve", "m", "--port", "8000"], True),
        ("true", ["vllm", "serve", "m", "--port", "8000"], True),
        ("on", ["vllm", "serve", "m", "--port", "8000"], True),
        ("0", ["vllm", "serve", "m", "--port", "8000"], False),
        ("false", ["vllm", "serve", "m", "--port", "8000"], False),
    ],
)
def test_vllm_tpu_use_rust_env_triggers_launcher(
    monkeypatch, env_val, argv, should_intercept
):
    from sitecustomize import maybe_run_experimental_rust

    recorded = []

    def fake_main(passed_argv):
        recorded.append(list(passed_argv))

    monkeypatch.setattr("vllm_torchtpu.experimental.rust.launcher.main", fake_main)

    if env_val is None:
        monkeypatch.delenv("VLLM_TPU_EXPERIMENTAL_USE_RUST", raising=False)
    else:
        monkeypatch.setenv("VLLM_TPU_EXPERIMENTAL_USE_RUST", env_val)

    monkeypatch.setattr(sys, "argv", argv)
    assert maybe_run_experimental_rust() is should_intercept
    if should_intercept:
        assert recorded == [argv[1:]]
    else:
        assert recorded == []


def test_vllm_tpu_use_rust_invalid_bool_raises(monkeypatch):
    from sitecustomize import maybe_run_experimental_rust

    monkeypatch.setenv("VLLM_TPU_EXPERIMENTAL_USE_RUST", "ture")
    monkeypatch.setattr(sys, "argv", ["vllm", "serve", "m"])
    with pytest.raises(ValueError, match="Invalid boolean value 'ture'"):
        maybe_run_experimental_rust()


def test_use_rust_with_vllm_rust_frontend_is_an_error(monkeypatch, capsys):
    from sitecustomize import maybe_run_experimental_rust

    monkeypatch.setattr(
        "vllm_torchtpu.experimental.rust.launcher.main",
        lambda argv: pytest.fail("the launcher must not run"),
    )
    monkeypatch.setenv("VLLM_TPU_EXPERIMENTAL_USE_RUST", "1")
    monkeypatch.setenv("VLLM_USE_RUST_FRONTEND", "1")
    monkeypatch.setattr(sys, "argv", ["vllm", "serve", "m"])
    with pytest.raises(SystemExit) as e:
        maybe_run_experimental_rust()
    assert e.value.code == 2
    assert "set only one" in capsys.readouterr().err


@pytest.mark.parametrize("use_rust", [True, False])
def test_vllm_serve_execs_rvllm_only_with_the_env_var(tmp_path, use_rust):
    """`vllm serve --help` reaches the fake rvllm only when the env var is set."""
    rvllm = _fake_binary(tmp_path / "bin")
    log = tmp_path / "args"
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [str(ROOT / "src"), os.environ.get("PYTHONPATH", "")]
        ),
        "VLLM_TPU_EXPERIMENTAL_RVLLM_BINARY": str(rvllm),
        "FAKE_LOG": str(log),
    }
    env.pop("VLLM_USE_RUST_FRONTEND", None)
    env.pop("VLLM_TPU_EXPERIMENTAL_USE_RUST", None)
    if use_rust:
        env["VLLM_TPU_EXPERIMENTAL_USE_RUST"] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", "--help"],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if use_rust:
        assert result.returncode == 0, result.stderr
        assert log.read_text().splitlines() == ["serve", "--help"]
    else:
        assert not log.exists(), result.stderr


def _publish(tmp_path, binary, sha=None):
    """Lays out a download location: the binary and its metadata.json."""
    site = tmp_path / "site"
    site.mkdir(exist_ok=True)
    data = binary.read_bytes()
    (site / "rvllm").write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    (site / "metadata.json").write_text(json.dumps({"sha256": sha or digest}))
    return site, digest


def test_downloaded_binary_is_verified_and_cached(run, tmp_path):
    site, digest = _publish(tmp_path, tmp_path / "rvllm")
    r = run("proxy", binary=str(site))
    cached = tmp_path / "home" / ".cache" / "rvllm" / "bin" / digest / "rvllm"
    assert r.execs == [[str(cached), "proxy"]]
    assert os.access(cached, os.X_OK)
    # Served from the cache once the download location is gone.
    (site / "rvllm").unlink()
    run("proxy", "again", binary=str(site))
    assert r.execs[-1] == [str(cached), "proxy", "again"]


def test_downloaded_binary_with_the_wrong_sha256_is_rejected(run, tmp_path, capsys):
    site, _ = _publish(tmp_path, tmp_path / "rvllm", sha="0" * 64)
    with pytest.raises(SystemExit):
        run("proxy", binary=str(site))
    assert "expected " + "0" * 64 in capsys.readouterr().err
    assert not list((tmp_path / "home" / ".cache" / "rvllm" / "bin").glob("*/rvllm"))


def test_python_m_replaces_itself_with_the_binary(tmp_path):
    rvllm = tmp_path / "rvllm"
    rvllm.write_text(FAKE_RVLLM)
    rvllm.chmod(0o755)
    log = tmp_path / "args"
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [str(ROOT / "src"), os.environ.get("PYTHONPATH", "")]
        ),
        "FAKE_LOG": str(log),
    }
    subprocess.run(
        [
            sys.executable,
            "-m",
            "vllm_torchtpu.experimental.rust",
            "proxy",
            "--rvllm-binary",
            str(rvllm),
            "--port",
            "1",
        ],
        env=env,
        check=True,
    )
    assert log.read_text().splitlines() == ["proxy", "--port", "1"]
