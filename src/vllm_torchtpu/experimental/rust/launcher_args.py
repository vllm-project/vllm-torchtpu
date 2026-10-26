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
"""Splits a `vllm serve` command line between the AOT exporter and `rvllm serve`.

`bin/rvllm.bin serve` takes the `vllm serve` command line. vLLM's own parser
(`vllm_torchtpu.experimental.export.cli.build_arg_parser`) classifies each
option:

- Engine arguments (the fields of `AsyncEngineArgs`) shape the model, so they go
  to the exporter, which records what the runtime needs in the artifact.
- The frontend options rvllm implements, and the engine arguments that set its
  runtime behavior, go to `rvllm serve`.
- rvllm's own deployment flags, which `vllm serve` does not have, go to
  `rvllm serve`.
- A frontend option that only affects vLLM's own logging or docs is dropped with
  a warning. Any other option rvllm does not implement fails, so that no option
  is silently ignored.

Run as `python -m vllm_torchtpu.experimental.rust.launcher_args ARGS...`. It
writes NUL-separated `KIND`, `VALUE` pairs to stdout: `M` is the model, `E` an
exporter argument, `S` an `rvllm serve` argument and `W` a warning. On an error
it writes the reason to stderr and exits with status 2.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field

from vllm_torchtpu.experimental.export import cli as export_cli

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


class LauncherArgsError(ValueError):
    """An option `rvllm.bin serve` cannot honor."""


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

    Raises `LauncherArgsError` for an option rvllm does not implement, an option
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
                raise LauncherArgsError(
                    f"unexpected argument '{token}' (the model is '{out.model}')"
                )
            out.model = token
            i += 1
            continue
        name, has_value, _ = token.partition("=")
        option = _canonical(name)
        if option in RVLLM_VALUE_FLAGS and not has_value:
            if i + 1 >= len(argv):
                raise LauncherArgsError(f"{name} needs a value")
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
            raise LauncherArgsError(
                f"{name} is neither a `vllm serve` option nor an rvllm serve flag"
            )
        count = 0 if has_value or attached else _value_count(action, argv[i + 1 :])
        if action.nargs is None and count == 1 and i + 1 >= len(argv):
            raise LauncherArgsError(f"{name} needs a value")
        group = list(argv[i : i + 1 + count])
        i += 1 + count
        dest = action.dest
        if dest == "model":
            out.model = token.partition("=")[2] if has_value else group[-1]
        elif dest in LAUNCHER_DESTS:
            raise LauncherArgsError(
                f"{name} is chosen by rvllm.bin; set RVLLM_ARTIFACT_DIR to serve an "
                "existing artifact"
            )
        elif dest in INTERNAL_EXPORT_DESTS:
            raise LauncherArgsError(f"{name} is internal to the exporter")
        elif dest == "kv_transfer_config":
            raise LauncherArgsError(
                "--kv-transfer-config is not supported: rvllm disaggregates with "
                "its own flags (--role prefill|decode, --kv-port, --kv-advertise, "
                "--kv-producers)"
            )
        elif dest in RVLLM_VLLM_DESTS:
            if dest == "served_model_name" and count > 1:
                raise LauncherArgsError("rvllm serves one --served-model-name")
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
            raise LauncherArgsError(
                f"{name} is a `vllm serve` frontend option that rvllm does not support"
            )
    return out


def main(argv: Sequence[str] | None = None) -> int:
    try:
        split = split_args(list(sys.argv[1:] if argv is None else argv))
    except LauncherArgsError as e:
        sys.stderr.write(f"rvllm.bin: error: {e}\n")
        return 2
    records: list[tuple[str, str]] = []
    if split.model is not None:
        records.append(("M", split.model))
    records += [("E", a) for a in split.export_args]
    records += [("S", a) for a in split.serve_args]
    records += [("W", w) for w in split.warnings]
    sys.stdout.write("".join(f"{k}\0{v}\0" for k, v in records))
    return 0


if __name__ == "__main__":
    sys.exit(main())
