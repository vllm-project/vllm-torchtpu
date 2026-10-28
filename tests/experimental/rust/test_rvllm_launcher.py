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

import pytest

ROOT = Path(__file__).resolve().parents[3]

try:
    from vllm_torchtpu.experimental.export import cli as export_cli
    from vllm_torchtpu.experimental.rust import launcher

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


def test_the_model_can_be_given_with_model():
    assert split("--model=m", "--max-model-len", "8").model == "m"
    assert split("--model", "m").model == "m"


@pytest.mark.parametrize(
    "argv, message",
    [
        (["m", "--api-key", "k"], "--api-key is a `vllm serve` frontend option"),
        (["m", "--chat-template", "t.jinja"], "--chat-template is a `vllm serve`"),
        (["m", "--enable-auto-tool-choice"], "frontend option that rvllm does not"),
        (["m", "--kv-transfer-config", "{}"], "--role prefill|decode"),
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
