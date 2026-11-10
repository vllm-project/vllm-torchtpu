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

# Source provenance: inferact/k3-megakernel@30c9ba966b
# tools/megakernel/__main__.py
"""Run a megakernel program without a vLLM server.

From the login node the tool runs itself as 32 processes (8 per host) on the
allocation ``--jobid`` (default ``$JOBID``). ``--set`` overrides a
``KernelConfig`` field to compare configs; every result line and JSON record
names the config digest and git revision that produced it.
"""

from __future__ import annotations

import argparse
import os
import sys

from .. import config
from . import _settings as settings
from . import bench, check, harness, k3, preshard
from .coverage import LineCoverage


def parser() -> argparse.ArgumentParser:
    """Build and return the top-level CLI argument parser."""
    top = argparse.ArgumentParser(
        prog="python -m vllm_torchtpu.models.kimi_k3.megakernel.tools",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    top.add_argument(
        "--jobid",
        default=settings.JOBID,
        help="the four-host allocation to run on",
    )
    top.add_argument(
        "--venv",
        default=settings.K3_VENV_DIR,
        help="the Python environment the workers run (default: this one)",
    )
    top.add_argument(
        "--presharded",
        default="",
        help="the TP32 pre-shard directory (``preshard k3 --output``)",
    )
    top.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help="override a KernelConfig field (repeatable)",
    )
    top.add_argument(
        "--coverage",
        default="",
        help="merge the kernel lines this run executes into this JSON file",
    )
    top.add_argument(
        "--json",
        default="",
        help="append one JSON record per result to this file",
    )
    top.add_argument(
        "--kv-cache-dtype",
        default="auto",
        choices=("auto", "bfloat16", "fp8", "fp8_e4m3"),
        help="cache encoding, as vLLM's --kv-cache-dtype (FP8 uses 512-token pages)",
    )
    commands = top.add_subparsers(dest="command", required=True)
    bench.add_arguments(
        commands.add_parser("bench", help="kernel-only timing of the serving ops")
    )
    check.add_arguments(commands.add_parser("check", help="numerics checks"))
    profile = commands.add_parser("profile", help="a bench with an xprof trace")
    bench.add_arguments(profile)
    preshard.add_arguments(
        commands.add_parser(
            "preshard",
            help="write the checkpoints in the kernels' per-rank layout",
        )
    )
    return top


def kernel_config(overrides: list[str]) -> config.KernelConfig:
    """Parse ``FIELD=VALUE`` ``overrides`` and install ``KernelConfig``."""
    fields: dict[str, str] = {}
    for item in overrides:
        name, _, value = item.partition("=")
        if not value or name not in config.KernelConfig.__dataclass_fields__:
            msg = f"--set {item!r}: expected FIELD=VALUE with a KernelConfig field"
            raise SystemExit(msg)
        fields[name] = value
    return config.install(config.KernelConfig().replace(**fields))


def _launch_remote(argv: list[str], args: argparse.Namespace) -> int:
    """Prepare worker environment variables and launch across the Slurm allocation."""
    if not args.jobid:
        msg = "set --jobid (or $JOBID) to a running four-host allocation"
        raise SystemExit(msg)
    env: dict[str, str] = {}
    if args.coverage:
        env["VLLM_XLA_CACHE_PATH"] = ""
    return harness.launch(argv, args.jobid, args.venv, env)


def main(argv: list[str]) -> int:
    """CLI entry point for ``vllm_torchtpu.models.kimi_k3.megakernel.tools``."""
    args = parser().parse_args(argv)
    if args.command == "preshard":
        return run_preshard(argv, args)
    if not args.presharded:
        msg = "set --presharded to the TP32 pre-shard directory"
        raise SystemExit(msg)
    if not harness.is_worker():
        return _launch_remote(argv, args)
    harness.init()
    coverage = None
    if args.coverage and harness.rank() == 0:
        coverage = LineCoverage(args.coverage)
        coverage.start()
    k3.configure_cache(args.kv_cache_dtype)
    installed = kernel_config(args.set)
    harness.log(
        f"kernel config {installed.key()}"
        + (f" ({', '.join(args.set)})" if args.set else " (defaults)")
    )
    sub_target = str(vars(args).get("role", "") or vars(args).get("check", ""))
    records = harness.Records(
        args.json,
        f"{args.command} {sub_target}",
        installed,
    )
    records.context.update(kv_cache_dtype=args.kv_cache_dtype, page_tokens=k3.PAGE)
    if args.command == "check":
        check.run(args, records)
    else:
        if args.command == "profile" and not args.trace:
            args.trace = "profile-traces"
        bench.run(args, records)
    if coverage is not None:
        coverage.stop()
        harness.log(f"kernel line coverage merged into {args.coverage}")
    harness.log("done")
    return 0


def run_preshard(argv: list[str], args: argparse.Namespace) -> int:
    """Run host-side checkpoint presharding locally or across the allocation."""
    if preshard.runs_on_allocation(args) and not harness.is_worker():
        if not args.jobid:
            msg = "set --jobid (or $JOBID) to a running four-host allocation"
            raise SystemExit(msg)
        return harness.launch(
            argv,
            args.jobid,
            args.venv,
            {"JAX_PLATFORMS": "cpu"},
            per_host=1,
        )
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    kernel_config(args.set)
    status = preshard.run(args, harness.log_host)
    harness.log("done")
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
