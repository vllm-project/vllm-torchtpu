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
"""Run a megakernel program without a vLLM server.

    python -m tools.megakernel [--jobid J] [--set field=value ...] [--json FILE] \\
        COMMAND ...

Commands (Kimi-K3; see each module's docstring):

    bench decode --batches 1,8,16 --contexts 1152,32768 [--dspark [CKPT]] \\
        [--mix 1152x4,131072x4]
    bench prefill --prompts 8192x1,2048x4,3000x1@32768
    bench dsa-micro --cases copy,barrier --rows 1,8,32,64
                                Kimi-K3.1 DSA decode microbenchmarks (no weights)
    check spec                  DSpark against plain decode, batch transitions,
                                seeded sampling
    check handoff               prefill -> pools -> decode against a decode-only replay
    check greedy --output FILE  prompts replayed on the decoder, then greedy tokens
    check resume                preempted and resumed decodes vs an uninterrupted one
    check masked                grammar-masked (structured outputs) vs unmasked decode
    check dsa-keys              Kimi-K3.1 index keys vs their latent rows (--dsa-config,
                                --synthetic-indexer rope-probe)
    check dsa-encode            Kimi-K3.1 in-kernel FP8 encodings (keys and queries)
                                on each rank's chip vs XLA on the host
    check dsa-scores --parts inputs
                                Kimi-K3.1 decode indexer's check outputs vs the JAX
                                reference (--dsa-config, --synthetic-indexer random)
    check arena-order           the prefill arena keeps each page's token order (the
                                Kimi-K3.1 selection's contract; no weights)
    stage bench|reference ...   the prefill as four pipeline stages (k3_stage)
    stage dsa-micro --prompts 8192x1 --rows 1024 --candidates 8192,147456
                                Kimi-K3.1 prefill DSA scoring and selection per
                                stage core (stage_dsa_micro; no weights)
    profile decode|prefill ...  the bench's cases with an xprof trace (--trace) and,
                                with --llo DIR, the compiled LLO of every op (fresh
                                compile caches, so nothing is skipped)
    preshard k3|prefill|dspark|indexer --output DIR   kernel-layout checkpoints and
                                projections (indexer: added to an existing k3 cache)
    preshard verify             a rebuilt sample of one rank against an existing
                                pre-shard
    preshard verify-indexer     a sample of ranks' indexer files against the checkpoint
    preshard verify-prefill     the prefill projection cache at a few layers
    preshard prefill-experts    the TP32 prefill's expert files, from a pre-shard
    preshard serve-view         link pre-shards written in separate places

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

from tools.megakernel import bench, check, harness, k3, k3_stage, preshard

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
        "--host-worlds",
        action="store_true",
        help="run every host as its own 8-process world (stage commands: host i runs stage i)",
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
    top.add_argument(
        "--dsa-config",
        default="",
        help="a Kimi-K3.1 model directory (or config.json): decode with its DSA "
        "indexer",
    )
    top.add_argument(
        "--synthetic-indexer",
        default=None,
        choices=("random", "rope-probe"),
        help="with --dsa-config: synthetic index-key weights instead of the "
        "pre-shard's own",
    )
    top.add_argument("--indexer-seed", type=int, default=0)
    top.add_argument(
        "--dsa-topk",
        type=int,
        default=0,
        help="with --dsa-config: the index_topk to run (0: the config's)",
    )
    commands = top.add_subparsers(dest="command", required=True)
    bench.add_arguments(
        commands.add_parser("bench", help="kernel-only timing of the serving ops")
    )
    check.add_arguments(commands.add_parser("check", help="numerics checks"))
    k3_stage.add_arguments(
        commands.add_parser("stage", help="the prefill as four pipeline stages")
    )
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
    if not args.presharded and not (
        (args.command == "bench" and args.role == "dsa-micro")
        or (args.command == "stage" and args.stage_command == "dsa-micro")
        or (args.command == "check" and args.check == "arena-order")
    ):
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
        return harness.launch(argv, args.jobid, args.venv, env, host_worlds=args.host_worlds)
    harness.init()
    coverage = None
    if args.coverage and harness.rank() == 0:
        from tools.megakernel.coverage import LineCoverage

        coverage = LineCoverage(args.coverage)
        coverage.start()  # before the kernels import: their module-level lines count
    k3.configure_cache(args.kv_cache_dtype)
    k3.configure_dsa(
        args.dsa_config,
        args.synthetic_indexer,
        args.indexer_seed,
        args.dsa_topk or None,
    )
    config = kernel_config(args.set)  # before any kernel module is imported
    harness.log(
        f"kernel config {config.key()}"
        + (f" ({', '.join(args.set)})" if args.set else " (defaults)")
    )
    records = harness.Records(
        args.json,
        args.command
        + " "
        + (
            getattr(args, "role", "")
            or getattr(args, "check", "")
            or getattr(args, "stage_command", "")
        ),
        config,
    )
    records.context.update(kv_cache_dtype=args.kv_cache_dtype, page_tokens=k3.PAGE)
    if k3.DSA is not None:
        records.context.update(synthetic_indexer=k3.indexer_mode() or "pre-shard")
    if args.command == "check":
        check.run(args, records)
    elif args.command == "stage":
        k3_stage.run(args, records)
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
