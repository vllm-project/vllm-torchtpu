# SPDX-License-Identifier: Apache-2.0
"""Run a megakernel program without a vLLM server.

    python -m tools.megakernel [--jobid J] [--set field=value ...] [--json FILE] \\
        COMMAND ...

Commands (Kimi-K3; see each module's docstring):

    bench decode --batches 1,8,16 --contexts 1152,32768 [--dspark [CKPT]] \\
        [--mix 1152x4,131072x4]
    bench prefill --prompts 8192x1,2048x4,3000x1@32768
    check spec                  DSpark against plain decode, batch transitions,
                                seeded sampling
    check handoff               prefill -> pools -> decode against a decode-only replay
    check greedy --output FILE  prompts replayed on the decoder, then greedy tokens
    check masked                grammar-masked (structured outputs) vs unmasked decode
    profile decode|prefill ...  the bench's cases with an xprof trace (--trace) and,
                                with --llo DIR, the compiled LLO of every op (fresh
                                compile caches, so nothing is skipped)
    preshard k3|prefill|dspark --output DIR   kernel-layout checkpoints and projections
    preshard verify             a rebuilt sample of one rank against an existing
                                pre-shard

From the login node the tool runs itself as 32 processes (8 per host) on the
allocation ``--jobid`` (default ``$JOBID``). ``--set`` overrides a
``KernelConfig`` field (``vllm_torchtpu.megakernel.kimi_k3.config``) to compare
configs; every result line and JSON record names the config digest and git
revision that produced it.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

from tools.megakernel import bench, check, harness, k3, preshard

LLO_FLAGS = (
    "--xla_jf_dump_to={dir} --xla_jf_dump_llo_text=true "
    "--xla_jf_dump_llo_pass_label_regex=(vliw-packed-bundles|final_bundles) "
    "--xla_jf_dump_llo_static_gaps=true --xla_mosaic_enable_llo_source_annotations=true"
)


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(
        prog="python -m tools.megakernel",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    top.add_argument(
        "--jobid",
        default=os.environ.get("JOBID", ""),
        help="the four-host allocation to run on",
    )
    top.add_argument(
        "--venv",
        default=os.environ.get("K3_VENV_DIR", sys.prefix),
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
        "--json", default="", help="append one JSON record per result to this file"
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
    profile = commands.add_parser(
        "profile", help="a bench with an xprof trace and LLO dumps"
    )
    bench.add_arguments(profile, profile=True)
    preshard.add_arguments(
        commands.add_parser(
            "preshard", help="write the checkpoints in the kernels' per-rank layout"
        )
    )
    return top


def kernel_config(overrides: list[str]):
    from vllm_torchtpu.megakernel.kimi_k3 import config

    fields = {}
    for item in overrides:
        name, _, value = item.partition("=")
        if not value or name not in config.KernelConfig.__dataclass_fields__:
            raise SystemExit(
                f"--set {item!r}: expected FIELD=VALUE with a KernelConfig field"
            )
        fields[name] = value
    return config.install(config.KernelConfig().replace(**fields))


def main(argv: list[str]) -> int:
    args = parser().parse_args(argv)
    if args.command == "preshard":
        return run_preshard(argv, args)
    if not args.presharded:
        raise SystemExit("set --presharded to the TP32 pre-shard directory")
    if not harness.is_worker():
        if not args.jobid:
            raise SystemExit(
                "set --jobid (or $JOBID) to a running four-host allocation"
            )
        env = {}
        llo = getattr(args, "llo", "")
        if args.command == "profile" and llo:
            os.makedirs(llo, exist_ok=True)
            env["LIBTPU_INIT_ARGS"] = (
                os.environ.get("LIBTPU_INIT_ARGS", "")
                + " "
                + LLO_FLAGS.format(dir=os.path.abspath(llo))
            ).strip()
            # Compile everything afresh: a cached executable dumps nothing.
            fresh = tempfile.mkdtemp(
                prefix="megakernel-profile-", dir=os.path.abspath(llo)
            )
            env.update(
                TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT=fresh,
                TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT=fresh,
                TORCH_TPU_TIER2_COMPILATION_CACHE=os.path.basename(fresh),
                TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE=os.path.basename(fresh),
                VLLM_XLA_CACHE_PATH=os.path.join(fresh, "xla"),
            )
        if args.coverage:
            # The kernel builders run only while their ops are traced; an
            # exported op loaded from the cache would record nothing.
            env["VLLM_XLA_CACHE_PATH"] = ""
        return harness.launch(argv, args.jobid, args.venv, env)
    harness.init()
    coverage = None
    if args.coverage and harness.rank() == 0:
        from tools.megakernel.coverage import LineCoverage

        coverage = LineCoverage(args.coverage)
        coverage.start()  # before the kernels import: their module-level lines count
    k3.configure_cache(args.kv_cache_dtype)
    config = kernel_config(args.set)  # before any kernel module is imported
    harness.log(
        f"kernel config {config.key()}"
        + (f" ({', '.join(args.set)})" if args.set else " (defaults)")
    )
    records = harness.Records(
        args.json,
        args.command + " " + (getattr(args, "role", "") or getattr(args, "check", "")),
        config,
    )
    records.context.update(kv_cache_dtype=args.kv_cache_dtype, page_tokens=k3.PAGE)
    if args.command == "check":
        check.run(args, records)
    else:
        if args.command == "profile" and not args.trace:
            args.trace = args.llo or "profile-traces"
        bench.run(args, records)
    if coverage is not None:
        coverage.stop()
        harness.log(f"kernel line coverage merged into {args.coverage}")
    harness.log("done")
    return 0


def run_preshard(argv: list[str], args) -> int:
    """Host-side conversion: no TPU, one process per host on the allocation
    for the model's checkpoint, in place otherwise."""
    if preshard.runs_on_allocation(args) and not harness.is_worker():
        if not args.jobid:
            raise SystemExit(
                "set --jobid (or $JOBID) to a running four-host allocation"
            )
        return harness.launch(
            argv, args.jobid, args.venv, {"JAX_PLATFORMS": "cpu"}, per_host=1
        )
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    kernel_config(args.set)  # before any kernel module is imported
    status = preshard.run(args, harness.log_host)
    harness.log("done")
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
