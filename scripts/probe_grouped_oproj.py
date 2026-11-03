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
"""Device time of the collectives a 4-way grouped o_proj would put on the MLA
rows block's critical path, next to the ones it would replace. GLM-5.2 at an
8192-token bucket on 16 devices: R=512 rows per device, K=16384 (64 heads x
v 256), N=6144.

  * a2a_v8: inside a group of 4, each device sends the other three their K/4
    columns of its R rows of fp8 V (6.3 MB) and keeps the group's 4R rows of
    its own K/4 columns; a2a_s: the per-(row, 1024-column shard) f32 scales;
    a2a_v8s: both in one all-to-all of bytes; a2a_v16: the same V in bf16.
  * rs16 / rs32: the reduce-scatter of the [4R, N] partial products inside
    the group (bf16 / f32), each device keeping its own R rows.
  * ref_ag_wo: the 16-way all-gather of the [K/16, N] fp8 o_proj shards (the
    rows block's per-layer weight gather today); ref_a2a_tok: the 16-way
    tokens->heads all-to-all of [R, 64, 512] bf16 (the token-parallel split's
    before the rows block).

Groups are strided ({j, j+4, j+8, j+12}: device j+4m holds K slice m, which
contains its own tensor-parallel shard) and contiguous ({4g..4g+3}), both timed.
Device ranks are positions in jax.devices(); the device list is recorded.

Each program chains CHAIN copies of one collective (each one's input is the
last one's output, so they run back to back on lockstepped devices). A
collective's device time is its span in a profiler trace (an async one's from
its start to its done), median over every copy of --iters runs after warmup.

As a serving job's pre-command, before the server boots (the job's Ray
cluster is up on both hosts):

  python3 scripts/probe_grouped_oproj.py --launch --out $OUT/grouped_probe.json

`--launch` starts this script once per host through Ray; the two processes
join the slice with jax.distributed and process 0 writes the results.
"""

import argparse
import glob
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile

R, K, N = 512, 16384, 6144
GROUP, WORLD, SHARD = 4, 16, 1024
HEADS, KV_LORA = 64, 512
CHAIN = 10
COLLECTIVE = re.compile(r"all-to-all|all_to_all|reduce-scatter|all-gather|all_gather")
# libtpu's multi-process port for the probe, not the server's TPU_PROCESS_PORT.
PROBE_PORT = 8477
COORD_PORT = 8480


def launch(args: argparse.Namespace) -> int:
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    ray.init(address="auto", logging_level="error")
    nodes = [
        n
        for n in ray.nodes()
        if n["Alive"] and n.get("Resources", {}).get("TPU", 0) > 0
    ]
    if len(nodes) != 2:
        raise SystemExit(f"expected 2 TPU hosts in the Ray cluster, found {len(nodes)}")
    # Process 0 (TPU worker 0) hosts jax.distributed's coordinator and writes
    # --out, so it has to be this host, where the job collects $OUT.
    if os.environ.get("TPU_WORKER_ID") != "0":
        raise SystemExit("--launch runs on TPU worker 0")
    hosts = [
        a.rsplit(":", 1)[0] for a in os.environ["TPU_PROCESS_ADDRESSES"].split(",")
    ]
    script = os.path.abspath(__file__)
    argv = ["--out", args.out, "--iters", str(args.iters)]

    @ray.remote(num_cpus=0)
    def run() -> tuple[str, int, str]:
        env = dict(os.environ, PROBE_COORD=f"{hosts[0]}:{COORD_PORT}", PROBE_NPROC="2")
        env["TPU_PROCESS_ADDRESSES"] = ",".join(f"{h}:{PROBE_PORT}" for h in hosts)
        env["TPU_PROCESS_PORT"] = str(PROBE_PORT)
        # Every chip of the host, whatever Ray assigned this task (none).
        env.pop("TPU_VISIBLE_CHIPS", None)
        p = subprocess.run(
            [sys.executable, script, *argv],
            env=env,
            capture_output=True,
            text=True,
            timeout=args.timeout,
        )
        return (
            env.get("TPU_WORKER_ID", "?"),
            p.returncode,
            (p.stdout + p.stderr)[-6000:],
        )

    refs = [
        run.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(n["NodeID"], soft=False)
        ).remote()
        for n in nodes
    ]
    rc = 0
    for worker, code, log in ray.get(refs):
        print(f"--- worker {worker} rc={code}\n{log}", flush=True)
        rc = rc or code
    return rc


def _collective_spans_us(trace_dir: str, names: list[str]) -> dict[str, list[float]]:
    """Every collective's span inside each named program, on this process's
    first TensorCore: a synchronous one's duration, an async one's time from
    its start to its done."""
    from jax.profiler import ProfileData

    (path,) = glob.glob(os.path.join(trace_dir, "**", "*.xplane.pb"), recursive=True)
    planes = [
        p
        for p in ProfileData.from_file(path).planes
        if p.name.startswith("/device:TPU:") and "SparseCore" not in p.name
    ]
    plane = min(planes, key=lambda p: int(p.name.rsplit(":", 1)[1]))
    lines = {line.name: line for line in plane.lines}
    mods = sorted(lines["XLA Modules"].events, key=lambda e: e.start_ns)
    # An op event is named by its HLO text, `%name = shape opcode(operands)`.
    ops = sorted(
        (
            (e, re.match(r"%?([^\s=]+)", e.name).group(1))
            for e in lines["XLA Ops"].events
        ),
        key=lambda pair: pair[0].start_ns,
    )
    ops = [(e, name) for e, name in ops if COLLECTIVE.match(name)]
    spans: dict[str, list[float]] = {n: [] for n in names}
    for m in mods:
        owner = next((n for n in names if m.name.split("(")[0] == f"jit_{n}"), None)
        if owner is None:
            continue
        opened: dict[str, int] = {}
        for e, name in ops:
            if e.start_ns < m.start_ns or e.start_ns >= m.end_ns:
                continue
            # An async collective's events share a base: all-gather.4.cloned.1
            # .call-start / -update / -done (all-gather-start.3 / -done.3).
            kind = re.sub(r"(\.\d+)+$", "", name)
            base = re.sub(r"[.-](call-)?(start|update|done)", "", name)
            base = re.sub(r"(\.\d+)+$", "", base) if kind != name else base
            if kind.endswith("start"):
                opened[base] = e.start_ns
            elif kind.endswith("done"):
                if base in opened:
                    spans[owner].append((e.end_ns - opened.pop(base)) / 1e3)
            elif not kind.endswith("update"):
                spans[owner].append((e.end_ns - e.start_ns) / 1e3)
    return spans


def probe(args: argparse.Namespace) -> None:
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax import lax
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    # The serving LIBTPU_INIT_ARGS (env_override), set before libtpu loads.
    import vllm_torchtpu  # noqa: F401

    if os.environ.get("PROBE_NPROC"):
        jax.distributed.initialize(
            coordinator_address=os.environ["PROBE_COORD"],
            num_processes=int(os.environ["PROBE_NPROC"]),
            process_id=int(os.environ["TPU_WORKER_ID"]),
        )
    devs = np.array(jax.devices())
    if devs.size != WORLD:
        raise SystemExit(f"expected {WORLD} devices, found {devs.size}")
    fp8 = jnp.float8_e4m3fn
    meshes = {
        # rank 4m + j: groups {j, j+4, j+8, j+12} along "m".
        "strided": Mesh(devs.reshape(GROUP, WORLD // GROUP), ("m", "j")),
        # rank 4j + m: groups {4j..4j+3} along "m".
        "contiguous": Mesh(devs.reshape(WORLD // GROUP, GROUP), ("j", "m")),
    }

    def a2a_v(v):  # [R, K] -> the group's 4R rows of this device's K/4 columns
        x = v.reshape(R, GROUP, K // GROUP)
        y = lax.all_to_all(x, "m", split_axis=1, concat_axis=0, tiled=True)
        return y.reshape(GROUP * R, K // GROUP)

    def a2a_s(s):  # [R, K/SHARD] f32 scales, the same exchange
        x = s.reshape(R, GROUP, K // SHARD // GROUP)
        y = lax.all_to_all(x, "m", split_axis=1, concat_axis=0, tiled=True)
        return y.reshape(GROUP * R, K // SHARD // GROUP)

    def a2a_vs(v, s):  # V bytes and scale bytes in one exchange
        vb = lax.bitcast_convert_type(v, jnp.uint8).reshape(R, GROUP, K // GROUP)
        sb = lax.bitcast_convert_type(s, jnp.uint8).reshape(R, GROUP, -1)
        x = jnp.concatenate([vb, sb], axis=2)
        y = lax.all_to_all(x, "m", split_axis=1, concat_axis=0, tiled=True)
        return y.reshape(GROUP * R, -1)

    def rs(p):  # [4R, N] partials -> this device's R rows, summed over the group
        return lax.psum_scatter(p, "m", scatter_dimension=0, tiled=True)

    def ag_wo(w):  # [K/16, N] fp8 shard -> every shard, 16-way
        return lax.all_gather(w, ("m", "j"), tiled=True)

    def a2a_tok(x):  # [R, 64, 512] -> 4 heads of every row, 16-way
        y = lax.all_to_all(
            x.reshape(R, WORLD, -1), ("m", "j"), split_axis=1, concat_axis=0, tiled=True
        )
        return y.reshape(WORLD * R, HEADS // WORLD, KV_LORA)

    def program(mesh, name, fn, back, n_in):
        # `back` turns a collective's output into its next copy's input.
        def chained(*xs):
            for _ in range(CHAIN):
                xs = back(fn(*xs))
            return xs

        spec = P(mesh.axis_names)
        f = jax.shard_map(
            chained,
            mesh=mesh,
            in_specs=(spec,) * n_in,
            out_specs=(spec,) * n_in,
            check_vma=False,
        )
        f.__name__ = name
        return jax.jit(f)

    def data(mesh, shape, dtype, seed):
        sharding = NamedSharding(mesh, P(mesh.axis_names))
        make = jax.jit(
            lambda key: (
                jax.random.normal(key, (WORLD * shape[0], *shape[1:])) * 0.1
            ).astype(dtype),
            out_shardings=sharding,
        )
        return make(jax.random.key(seed))

    def same_count(shape):  # a collective's output, reread at its input's shape
        return lambda y: (y.reshape(shape),)

    def unpack(y):  # a2a_vs's bytes back into V and scales
        v = lax.bitcast_convert_type(y[:, : K // GROUP].reshape(R, K), fp8)
        sc = lax.bitcast_convert_type(
            y[:, K // GROUP :].reshape(R, K // SHARD, 4), jnp.float32
        )
        return v, sc

    cases = []  # (label, jitted, inputs, MB sent per device per collective)
    sent = (GROUP - 1) / GROUP / 1e6
    for gname, mesh in meshes.items():
        v8 = data(mesh, (R, K), fp8, 1)
        v16 = data(mesh, (R, K), jnp.bfloat16, 2)
        sc = data(mesh, (R, K // SHARD), jnp.float32, 3)
        p16 = data(mesh, (GROUP * R, N), jnp.bfloat16, 4)
        p32 = data(mesh, (GROUP * R, N), jnp.float32, 5)
        tile = lambda y: (jnp.tile(y, (GROUP, 1)),)  # noqa: E731
        for label, fn, back, inputs, mb in (
            ("a2a_v8", a2a_v, same_count((R, K)), (v8,), sent * R * K),
            (
                "a2a_s",
                a2a_s,
                same_count((R, K // SHARD)),
                (sc,),
                sent * R * (K // SHARD) * 4,
            ),
            ("a2a_v8s", a2a_vs, unpack, (v8, sc), sent * R * (K + K // SHARD * 4)),
            ("a2a_v16", a2a_v, same_count((R, K)), (v16,), sent * R * K * 2),
            ("rs16", rs, tile, (p16,), sent * GROUP * R * N * 2),
            ("rs32", rs, tile, (p32,), sent * GROUP * R * N * 4),
        ):
            name = f"{gname}_{label}"
            cases.append((name, program(mesh, name, fn, back, len(inputs)), inputs, mb))
    mesh = meshes["strided"]
    w = data(mesh, (K // WORLD, N), fp8, 6)
    tok = data(mesh, (R, HEADS, KV_LORA), jnp.bfloat16, 7)
    sent16 = (WORLD - 1) / WORLD / 1e6
    cases += [
        (
            "ref_ag_wo",
            program(mesh, "ref_ag_wo", ag_wo, lambda y: (y[: K // WORLD],), 1),
            (w,),
            sent16 * K * N,
        ),
        (
            "ref_a2a_tok",
            program(mesh, "ref_a2a_tok", a2a_tok, same_count((R, HEADS, KV_LORA)), 1),
            (tok,),
            sent16 * R * HEADS * KV_LORA * 2,
        ),
    ]

    for _, f, inputs, _ in cases:  # compile and warm up outside the trace
        for _ in range(3):
            jax.block_until_ready(f(*inputs))
    # Process 0 keeps its trace next to --out; the others' go away at the end.
    if jax.process_index() == 0:
        trace_dir = os.path.join(
            os.path.dirname(os.path.abspath(args.out)), "grouped_probe_trace"
        )
        shutil.rmtree(trace_dir, ignore_errors=True)  # one run's trace only
    else:
        trace_dir = tempfile.mkdtemp(prefix="grouped_probe_")
    with jax.profiler.trace(trace_dir):
        for _, f, inputs, _ in cases:
            for _ in range(args.iters):
                jax.block_until_ready(f(*inputs))
    if jax.process_index() == 0:
        names = [label for label, *_ in cases]
        spans = _collective_spans_us(trace_dir, names)
        results = {}
        for label, _, _, mb in cases:
            t = sorted(spans[label])
            if len(t) != CHAIN * args.iters:
                raise SystemExit(
                    f"{label}: {len(t)} collectives in the trace, "
                    f"expected {CHAIN * args.iters}"
                )
            results[label] = dict(
                median_us=round(statistics.median(t), 1),
                p10_us=round(t[len(t) // 10], 1),
                p90_us=round(t[(9 * len(t)) // 10], 1),
                n=len(t),
                mb_sent_per_device=round(mb, 2),
            )
        out = dict(
            libtpu_init_args=os.environ.get("LIBTPU_INIT_ARGS", ""),
            jax=jax.__version__,
            chain=CHAIN,
            iters=args.iters,
            devices=[
                dict(
                    rank=i,
                    id=d.id,
                    process=d.process_index,
                    coords=list(getattr(d, "coords", ())),
                    core=getattr(d, "core_on_chip", None),
                )
                for i, d in enumerate(devs)
            ],
            results=results,
        )
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=1)
        for label, r in results.items():
            mb = r["mb_sent_per_device"]
            print(f"{label:24s} {r['median_us']:8.1f} us  ({mb} MB/device)")
    else:
        shutil.rmtree(trace_dir)
    if os.environ.get("PROBE_NPROC"):
        jax.distributed.shutdown()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--launch", action="store_true", help="start one process per host via Ray"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--timeout", type=int, default=900)
    a = ap.parse_args()
    if a.launch:
        sys.exit(launch(a))
    probe(a)


if __name__ == "__main__":
    main()
