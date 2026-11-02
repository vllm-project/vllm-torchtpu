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
"""Serves a model with the native Rust runtime (rvllm) behind the `vllm serve` CLI.

    python -m vllm_torchtpu.experimental.rust serve <model> [vllm serve arguments]

exports the model with `python -m vllm_torchtpu.experimental.export` (once; the
artifact is cached by model, export arguments and exporter source) and then
replaces this process with `rvllm serve <artifact>`.

vLLM's own `vllm serve` parser splits the command line:
- Engine arguments (the fields of `AsyncEngineArgs`) go to the exporter, which
  records in the artifact what the runtime needs.
- The frontend options rvllm implements, the engine arguments that set its
  runtime behavior, and rvllm's own deployment flags go to `rvllm serve`.
- A frontend option that only affects vLLM's own logs or docs is dropped with a
  warning. Any other option rvllm does not implement fails, so that no option is
  silently ignored.

Both commands are printed before they run. `bench ...` replaces this process
with `vllm bench ...`; other subcommands (`proxy`, `--version`) with the rvllm
binary.

`--rvllm-binary` (required) names the rvllm binary, which is not part of this
repository: a local binary, used as is, or a location holding `rvllm` and a
`metadata.json` whose `sha256` entry is the binary's SHA-256 in lowercase hex
(`gs://...`, `https://...` or a local directory). A downloaded binary is checked
against the sha256 and cached. The launcher takes it out before the split, as
`--rvllm-binary VALUE` or `--rvllm-binary=VALUE`; `vllm serve` has no
`--rvllm-*` options, and any other one fails. Downloads and exports are cached
under `~/.cache/rvllm`.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from vllm_torchtpu.experimental.export import cli as export_cli

PROG = "python -m vllm_torchtpu.experimental.rust"

# vLLM options that `rvllm serve` takes under the same name. host and port are
# frontend options; the others are engine arguments that set runtime behavior,
# not the artifact.
RVLLM_VLLM_DESTS = frozenset(
    {"host", "port", "served_model_name", "async_scheduling", "load_format"}
)
# Of those, the ones the exporter reads too: it exports dummy weights for
# `--load-format dummy`.
SHARED_DESTS = frozenset({"load_format"})

# rvllm's own deployment flags, which `vllm serve` does not have.
RVLLM_VALUE_FLAGS = frozenset(
    {
        "--max-queued-requests",
        "--shutdown-timeout",
        "--structured-decoding-mode",
        "--role",
        "--kv-role",
        "--kv-port",
        "--kv-transfer-port",
        "--kv-listen",
        "--kv-teleport-listen",
        "--kv-advertise",
        "--kv-producers",
        "--debug-max-blocks",
        "--pjrt-cpu-device-count",
        "--prompt",
        "--max-tokens",
    }
)
RVLLM_BOOL_FLAGS = frozenset({"--debug-dummy-weight", "--debug-dummy-compute"})

# Frontend options that only shape vLLM's own logs, docs or HTTP server limits:
# dropped with a warning.
WARN_ONLY_DESTS = frozenset(
    {
        "uvicorn_log_level",
        "disable_uvicorn_access_log",
        "disable_access_log_for_endpoints",
        "max_log_len",
        "enable_log_outputs",
        "enable_log_deltas",
        "log_error_stack",
        "disable_fastapi_docs",
        "enable_offline_docs",
        "h11_max_incomplete_event_size",
        "h11_max_header_count",
        "api_server_count",
    }
)

# Arguments rvllm.bin chooses itself, and internal export flags.
LAUNCHER_DESTS = frozenset({"output_dir", "export_mode"})
INTERNAL_EXPORT_DESTS = frozenset(
    {"export_worker", "jax_coordinator", "checkpoint_data_dir"}
)


class LauncherError(ValueError):
    """A command line or setting the launcher cannot honor."""


@dataclass
class SplitArgs:
    """A `vllm serve` command line, split between the exporter and `rvllm serve`."""

    model: str | None = None
    export_args: list[str] = field(default_factory=list)
    serve_args: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _canonical(option: str) -> str:
    """`--max_model_len` -> `--max-model-len`, as vLLM's parser accepts both."""
    if option.startswith("--"):
        return "--" + option[2:].replace("_", "-")
    return option


def _engine_dests() -> frozenset[str]:
    from vllm.engine.arg_utils import AsyncEngineArgs

    return frozenset(f.name for f in dataclasses.fields(AsyncEngineArgs))


def _value_count(action: argparse.Action, rest: Sequence[str]) -> int:
    """How many of the following tokens `action` takes as values."""
    nargs = action.nargs
    if nargs == 0:
        return 0
    if nargs is None:
        return 1
    if nargs == "?":
        return 1 if rest and not rest[0].startswith("-") else 0
    if nargs in ("*", "+", argparse.REMAINDER):
        n = 0
        while n < len(rest) and not rest[n].startswith("-"):
            n += 1
        return n
    return int(nargs)


def split_args(
    argv: Sequence[str], parser: argparse.ArgumentParser | None = None
) -> SplitArgs:
    """Splits `argv`, a `vllm serve` command line without `serve`.

    Raises `LauncherError` for an option rvllm does not implement, an option
    `vllm serve` does not know, or one rvllm.bin chooses itself.
    """
    parser = parser or export_cli.build_arg_parser()
    engine_dests = _engine_dests()
    actions = parser._option_string_actions
    out = SplitArgs()
    i = 0
    while i < len(argv):
        token = argv[i]
        if not token.startswith("-") or token == "-":
            if out.model is not None:
                raise LauncherError(
                    f"unexpected argument '{token}' (the model is '{out.model}')"
                )
            out.model = token
            i += 1
            continue
        name, has_value, _ = token.partition("=")
        option = _canonical(name)
        if option in RVLLM_VALUE_FLAGS and not has_value:
            if i + 1 >= len(argv):
                raise LauncherError(f"{name} needs a value")
            out.serve_args += argv[i : i + 2]
            i += 2
            continue
        if option in RVLLM_VALUE_FLAGS or option in RVLLM_BOOL_FLAGS:
            out.serve_args.append(token)
            i += 1
            continue
        action = actions.get(option)
        attached = False
        if action is None and not option.startswith("--") and option[:2] in actions:
            # A short option with its value attached, e.g. `-O3`.
            action, attached = actions[option[:2]], True
        if action is None:
            raise LauncherError(
                f"{name} is neither a `vllm serve` option nor an rvllm serve flag"
            )
        count = 0 if has_value or attached else _value_count(action, argv[i + 1 :])
        if action.nargs is None and count == 1 and i + 1 >= len(argv):
            raise LauncherError(f"{name} needs a value")
        group = list(argv[i : i + 1 + count])
        i += 1 + count
        dest = action.dest
        if dest == "model":
            out.model = token.partition("=")[2] if has_value else group[-1]
        elif dest in LAUNCHER_DESTS:
            raise LauncherError(
                f"{name} is chosen by the launcher, which exports to its cache"
            )
        elif dest in INTERNAL_EXPORT_DESTS:
            raise LauncherError(f"{name} is internal to the exporter")
        elif dest == "kv_transfer_config":
            raise LauncherError(
                "--kv-transfer-config is not supported: rvllm disaggregates with "
                "its own flags (--role prefill|decode, --kv-port, --kv-advertise, "
                "--kv-producers)"
            )
        elif dest in RVLLM_VLLM_DESTS:
            if dest == "served_model_name" and count > 1:
                raise LauncherError("rvllm serves one --served-model-name")
            out.serve_args += group
            if dest in SHARED_DESTS:
                out.export_args += group
        elif dest in engine_dests or dest in export_cli._EXPORT_DESTS:
            out.export_args += group
        elif dest in WARN_ONLY_DESTS:
            out.warnings.append(
                f"ignoring {name}: it only configures vLLM's own API server"
            )
        else:
            raise LauncherError(
                f"{name} is a `vllm serve` frontend option that rvllm does not support"
            )
    return out


def _log(message: str) -> None:
    sys.stderr.write(f"rvllm: {message}\n")


def cache_dir() -> Path:
    return Path.home() / ".cache" / "rvllm"


def take_binary_option(args: Sequence[str]) -> tuple[str, list[str]]:
    """Takes `--rvllm-binary` out of `args`; returns its value and the rest."""
    binary, rest = "", []
    i = 0
    while i < len(args):
        name, has_value, value = args[i].partition("=")
        if not name.startswith("--rvllm-"):
            rest.append(args[i])
            i += 1
            continue
        if name != "--rvllm-binary":
            raise LauncherError(
                f"unknown option {name}; the launcher's only option is --rvllm-binary"
            )
        if not has_value:
            if i + 1 >= len(args):
                raise LauncherError(f"{name} needs a value")
            value = args[i + 1]
            i += 1
        binary = value
        i += 1
    return binary, rest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch(location: str, name: str, dest: Path) -> None:
    """Copies `<location>/<name>` to `dest`; location is gs://, https:// or a path."""
    source = f"{location.rstrip('/')}/{name}"
    if location.startswith("gs://"):
        if shutil.which("gcloud") is None:
            raise LauncherError(f"gcloud is needed to download from {location}")
        result = subprocess.run(
            ["gcloud", "--no-user-output-enabled", "storage", "cp", source, str(dest)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise LauncherError(f"could not fetch {source}: {result.stderr.strip()}")
    elif location.startswith(("http://", "https://")):
        try:
            urllib.request.urlretrieve(source, dest)
        except OSError as e:
            raise LauncherError(f"could not fetch {source}: {e}") from e
    else:
        try:
            shutil.copyfile(source, dest)
        except OSError as e:
            raise LauncherError(f"could not fetch {source}: {e}") from e


def resolve_binary(location: str) -> Path:
    """The rvllm binary `--rvllm-binary` names, downloaded and verified if needed."""
    if not location:
        raise LauncherError(
            "pass --rvllm-binary: an rvllm binary, or a location (gs://, https:// "
            "or a directory) holding rvllm and its metadata.json"
        )
    path = Path(location)
    if path.is_file():
        if not os.access(path, os.X_OK):
            raise LauncherError(f"--rvllm-binary {location} is not executable")
        return path
    if not location.startswith(("gs://", "http://", "https://")) and not path.is_dir():
        raise LauncherError(
            f"--rvllm-binary {location} is neither a file, a directory nor a URL"
        )
    with tempfile.TemporaryDirectory() as tmp:
        meta = Path(tmp) / "metadata.json"
        _fetch(location, "metadata.json", meta)
        try:
            expected = json.loads(meta.read_text())["sha256"]
        except (KeyError, ValueError) as e:
            raise LauncherError(f"metadata.json from {location} has no sha256") from e
        cached = cache_dir() / "bin" / expected / "rvllm"
        if cached.is_file() and _sha256(cached) == expected:
            return cached
        download = Path(tmp) / "rvllm"
        _fetch(location, "rvllm", download)
        actual = _sha256(download)
        if actual != expected:
            raise LauncherError(
                f"rvllm from {location} has sha256 {actual}, expected {expected}"
            )
        download.chmod(0o755)
        cached.parent.mkdir(parents=True, exist_ok=True)
        os.replace(download, cached)
        return cached


def _exporter_source_hash() -> str:
    root = Path(export_cli.__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def export_command(
    model: str, output_dir: Path, export_args: Sequence[str]
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "vllm_torchtpu.experimental.export",
        model,
        "--output-dir",
        str(output_dir),
        "--export-mode",
        "all",
        *export_args,
    ]


def run_export(command: Sequence[str]) -> None:
    """Runs the exporter; a failed export ends the launcher with its status."""
    status = subprocess.run(list(command)).returncode
    if status != 0:
        raise SystemExit(status)


def artifact_for(split: SplitArgs) -> Path:
    """The exported artifact to serve: cached, or exported now."""
    if split.model is None:
        raise LauncherError(f"usage: {PROG} serve <model> [vllm serve arguments]")
    key = hashlib.sha256(
        "\0".join([split.model, *split.export_args, _exporter_source_hash()]).encode()
    ).hexdigest()[:24]
    artifact = cache_dir() / "artifacts" / key
    if (artifact / ".complete").is_file():
        _log(f"serving the cached export {artifact}")
        return artifact
    partial = artifact.with_name(artifact.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.parent.mkdir(parents=True, exist_ok=True)
    command = export_command(split.model, partial, split.export_args)
    _log(f"exporting {split.model} to {artifact}")
    _log(f"export command: {shlex.join(command)}")
    run_export(command)
    shutil.rmtree(artifact, ignore_errors=True)
    os.replace(partial, artifact)
    (artifact / ".complete").touch()
    return artifact


def _exec(argv: Sequence[str]) -> None:
    """Replaces this process with `argv`."""
    os.execv(argv[0], list(argv))


def main(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        if not args or args[0] in ("-h", "--help"):
            sys.stdout.write(__doc__ or "")
            return
        if args[0] == "bench":
            _exec([sys.executable, "-m", "vllm.entrypoints.cli.main", *args])
            return
        command, (location, rest) = args[0], take_binary_option(args[1:])
        if command != "serve":
            _exec([str(resolve_binary(location)), command, *rest])
            return
        if "-h" in rest or "--help" in rest:
            _exec([str(resolve_binary(location)), "serve", "--help"])
            return
        split = split_args(rest)
        for warning in split.warnings:
            _log(f"warning: {warning}")
        binary = resolve_binary(location)
        artifact = artifact_for(split)
        serve = [str(binary), "serve", str(artifact), *split.serve_args]
        _log(f"serve command: {shlex.join(serve)}")
        _exec(serve)
    except LauncherError as e:
        sys.stderr.write(f"rvllm: error: {e}\n")
        raise SystemExit(2) from None
