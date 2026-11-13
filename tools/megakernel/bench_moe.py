# SPDX-License-Identifier: Apache-2.0
"""Single-layer fused-MoE timing (``bench moe``).

Times one MoE layer's fused op on the 32-rank ``k3_sp`` mesh: the serving
prefill's residual mixture + RMSNorm + MoE
(``make_indexed_mixture_moe_sp_ring``, ``--op mixture``) or the plain MoE
(``make_indexed_moe_sp_ring``, ``--op plain``), with only MoE layer
``--layer``'s weights read (``rank_loader.read_moe_layer``), for every
``--windows`` launch size and ``--routes`` routing case (``moe_layer``).

By default every family holds that layer alone and the op runs at slot 0.
``--family-layers N`` / ``--dense-layers M`` / ``--slot S`` lay the families
out as serving holds them: the four expert families with N layers and the
others with M, zero but for the layer's weights at the index the kernel takes
for slot S (``prefill_megakernel.expert_family_slot``: the prefill role's 3/8/1
expert banks, the aggregate's 92-layer families), so the op sees serving's
shapes and byte offsets.

Each case's inputs make the kernel's router select the case's experts: every
residual block and the prefix of a row are the same BF16 residual, so the
mixture returns it and RMSNorm gives the hidden state ``moe_layer`` solved
for (the plain op takes that hidden state directly). The selection is checked
on the host before the case runs, and on the device as well when
``--set prefill_route_counts=1`` makes the op return its chip's expert counts.

Timing: ``--launches`` launches on the same inputs back to back with one host
sync, median of ``--repeats`` (``--spaced MS``: a host sync after every launch
and MS ms idle, median launch from dispatch to completion). A line per case
gives us per launch and the routing's busiest-chip load (routed rows,
GU-bucket-padded rows, grid visits), a digest of every rank's output (sha256
over the ranks' digests), a float checksum, and whether the last launch
reproduced the first. ``--reference`` adds the error of rank 0's first rows
against the host reference (``moe_layer.finish``). ``--save-output`` keeps
rank 0's output rows and 8 rows of every rank per case, for an offline
comparison of two runs:

    python -m tools.megakernel.bench_moe A.npz B.npz

A counted case runs at the windows that hold its tokens (a served 16K
window's counts only at 16384). ``--trace DIR``: rank 0 saves an xprof trace
of each case's last repeat. The log also gives each family's byte range of
the layer's entry, and brackets each window's compile between two lines: when
libtpu logs its overlay partitioner there
(``[libtpu:--vmodule=overlay_partitioner=1,overlay=1]`` with libtpu's log on
stderr), ``python -m tools.megakernel.bench_moe overlays RANK_LOG`` reports
each window's compiled overlays from the process's log.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from tools.megakernel import harness

AXIS = "k3_sp"
SAMPLE_ROWS = 8  # rows of every rank kept by --save-output


def add_arguments(sub, profile: bool = False) -> None:
    moe = sub.add_parser("moe", help="one MoE layer's fused op per routing case")
    moe.add_argument(
        "--windows",
        default="16384",
        help="launch sizes in tokens (512, 1024, multiples of 2048 up to 16384)",
    )
    moe.add_argument(
        "--routes",
        default="uniform",
        help="routing cases: uniform[:SEED], serving:WINDOW:LAYER, sweep:N, "
        "counts:FILE.npy (tools/megakernel/moe_layer.py)",
    )
    moe.add_argument(
        "--routes-file",
        default="",
        help="served route counts (npz: counts [K, 896] per logical expert, "
        "window [K], layer [K]) for serving: and sweep: cases",
    )
    moe.add_argument(
        "--layer", type=int, default=47, help="the MoE layer (0..91) whose weights run"
    )
    moe.add_argument("--op", choices=("mixture", "plain"), default="mixture")
    moe.add_argument(
        "--family-layers",
        type=int,
        default=1,
        help="layers of the four expert families (8: a prefill-role bank, 92: "
        "the aggregate's families); zero but for the layer's weights",
    )
    moe.add_argument(
        "--dense-layers",
        type=int,
        default=0,
        help="layers of the router, latent and shared families (default: "
        "--slot + 1); zero but for the layer's weights at --slot",
    )
    moe.add_argument(
        "--slot",
        type=int,
        default=-1,
        help="the MoE slot the op runs (default: --family-layers - 1)",
    )
    moe.add_argument("--launches", type=int, default=32, help="launches per repeat")
    moe.add_argument("--repeats", type=int, default=5)
    moe.add_argument(
        "--spaced",
        type=float,
        default=None,
        metavar="MS",
        help="a host sync after every launch, then MS ms idle (default: back "
        "to back, one sync per repeat)",
    )
    moe.add_argument(
        "--length",
        type=int,
        default=0,
        help="routed tokens of a uniform case (default: the whole launch)",
    )
    moe.add_argument(
        "--margin",
        type=float,
        default=0.1,
        help="designed gap between every token's 16th and 17th biased score",
    )
    moe.add_argument("--seed", type=int, default=0, help="seed of uniform cases")
    moe.add_argument(
        "--zero-router-bias",
        action="store_true",
        help="run with a zero router bias (any expert set is reachable; the "
        "routes and the op's work are the same), for a bias that leaves some "
        "case's experts unreachable",
    )
    moe.add_argument(
        "--reference",
        action="store_true",
        help="also check rank 0's first --reference-rows rows of each case "
        "against the host reference of the layer",
    )
    moe.add_argument("--reference-rows", type=int, default=128)
    moe.add_argument(
        "--save-output",
        default="",
        help="npz file: rank 0's output rows and 8 rows of every rank per case",
    )
    moe.add_argument(
        "--trace",
        default="",
        help="directory: rank 0 saves an xprof trace of each case",
    )
    if profile:
        moe.add_argument(
            "--llo", default="", help="directory for the compiled LLO of every op"
        )


def _active_blocks(layer: int) -> int:
    """Residual blocks the mixture of MoE layer ``layer`` (model layer
    ``layer + 1``) reads, as the prefill groups pass them."""
    return (layer + 1) // 12 + 1


def _bits(tensor) -> np.ndarray:
    """This rank's rows of a ``[1, rows, hidden]`` BF16 device tensor, as host
    uint16 bit patterns."""
    import torch

    return tensor.cpu()[0].view(torch.int16).numpy().view(np.uint16)


def _float(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits).astype(np.uint32) << 16).view(np.float32)


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    """BF16 values held as FP32, as uint16 bit patterns."""
    return (np.asarray(values, np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _gather(value) -> list:
    """Every rank's ``value``, in rank order."""
    import torch

    gathered: list = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, value)
    return gathered


def _fail_on(errors: list, what: str) -> None:
    """Fail every rank with the first rank's error, if any rank has one."""
    failed = [error for error in errors if error]
    if failed:
        harness.fail(f"{what}: {failed[0]}")


def _on_every_rank(what: str, step, *args):
    """``step(*args)``; when it raises on any rank, every rank fails with the
    first rank's error instead of waiting in a collective for a rank that
    died."""
    result, error = None, ""
    try:
        result = step(*args)
    except Exception as exc:  # noqa: BLE001 -- failed on every rank below
        error = f"rank {harness.rank()}: {type(exc).__name__}: {exc}"
    _fail_on(_gather(error), what)
    return result


def _tag(window: int, label: str) -> str:
    return f"w{window}_" + "".join(c if c.isalnum() else "_" for c in label)


def _device_i32(value: int):
    """A replicated [1] int32 operand on the device."""
    import torch

    return torch.tensor([value], dtype=torch.int32).to("tpu")


class _Layer:
    """MoE layer ``layer``'s weights for this rank: on the host (routing,
    reference) and resident on the device as ``layout`` holds them."""

    def __init__(
        self, presharded: str, rank: int, layer: int, zero_bias: bool, layout
    ) -> None:
        from torch_tpu._internal.sync import synchronize

        from tools.megakernel import moe_layer
        from vllm_torchtpu.megakernel import bridge
        from vllm_torchtpu.megakernel.kimi_k3.kernels import prefill_megakernel as k
        from vllm_torchtpu.megakernel.kimi_k3.kernels import rank_loader

        started = time.perf_counter()
        self.host = rank_loader.read_moe_layer(Path(presharded), rank, layer)
        self.read_s = time.perf_counter() - started
        self.bias_span = (
            float(self.host["router_bias"].min()),
            float(self.host["router_bias"].max()),
        )
        if zero_bias:
            self.host["router_bias"] = np.zeros_like(self.host["router_bias"])
        self.families, self.device_bytes = [], 0
        for name in k.MOE_FAMILIES:
            if name in rank_loader.EXPERT_FAMILIES:
                held = moe_layer.held_family(
                    self.host[name], layout.experts, layout.expert_index
                )
            else:
                held = moe_layer.held_family(self.host[name], layout.dense, layout.slot)
            self.families.append(bridge.upload(held))
            self.device_bytes += held.nbytes
            # torch_tpu compiles a tensor's pending work into the next op
            # taking it; settled operands keep every launch on one graph.
            synchronize(self.families[-1:], wait=True)
            del held
        self.mixture = [bridge.upload(self.host[n]) for n in ("mlp_res", "ffn_norm")]
        self.slot = _device_i32(layout.slot)
        synchronize(self.mixture + [self.slot], wait=True)
        self.resident_s = time.perf_counter() - started
        self.router = self.host["router"][0].astype(np.float32)
        self.bias = self.host["router_bias"][0, 0].astype(np.float32)
        self.norm = self.host["ffn_norm"][0].astype(np.float32)
        self.solver = moe_layer.HiddenSolver(self.router, self.norm)

    def gib(self) -> float:
        return sum(value.nbytes for value in self.host.values()) / 2**30

    def offsets(self, layout) -> dict[str, tuple[int, int, int]]:
        """Per family on the device: the byte range of the layer's entry and
        the family's bytes."""
        from vllm_torchtpu.megakernel.kimi_k3.kernels import prefill_megakernel as k

        result = {}
        for name, (layers, index) in layout.entries(k.MOE_FAMILIES).items():
            entry = self.host[name][0].nbytes
            result[name] = (index * entry, (index + 1) * entry, layers * entry)
        return result


@dataclasses.dataclass
class _Inputs:
    """One case's routing and this rank's rows of the op's inputs."""

    length: int  # routed tokens of the launch
    sets: np.ndarray  # [length, 16] every routed token's experts
    valid: np.ndarray  # [rows] this rank's routed rows
    residual: np.ndarray  # [rows, hidden] BF16 values
    hidden: np.ndarray  # [rows, hidden] the kernel's hidden rows (BF16 values)
    margin: float  # smallest 16th-17th biased score gap of this rank's rows
    scaled: int  # rows whose target logits were scaled to stay reachable

    def digest(self) -> str:
        from tools.megakernel import moe_layer

        return moe_layer.inputs_digest(self.length, self.residual, self.hidden)


def _synthesize(case, window: int, rank: int, layer: _Layer, args) -> _Inputs:
    """This rank's rows of ``case`` in a ``window``-token launch (raises when
    the routing cannot be built or does not check out)."""
    from tools.megakernel import moe_layer
    from vllm_torchtpu.megakernel.kimi_k3.kernels import prefill_megakernel as k

    rows = window // 32
    tokens = k.sp_token_index(rank, np.arange(rows), k.window_stripe(rows))
    length = case.tokens(window, args.length or None)
    sets = case.sets(length)
    valid = tokens < length
    mine = sets[np.minimum(tokens, length - 1)]
    rng = np.random.default_rng([moe_layer.seed_of(case.label, window), rank])
    logits = moe_layer.target_logits(mine, valid, layer.bias, rng, margin=args.margin)
    residual, scaled = layer.solver.residual(logits, rng)
    hidden = moe_layer.post_norm(residual, layer.norm)
    margin = moe_layer.check_selection(hidden, layer.router, layer.bias, mine, valid)
    return _Inputs(length, sets, valid, residual, hidden, margin, scaled)


def _build_op(mesh, window: int, mixture: bool, counting: bool):
    """The fused MoE of a ``window``-token launch as a torch op over
    ``mesh``: ``(prefix, blocks, mlp_res, ffn_norm, active, length, slot,
    *families)`` with the mixture, else ``(hidden, prefix, length, slot,
    *families)``; it returns the output rows (and with route counting the
    chip's expert counts)."""
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.megakernel.bridge import register_global_op
    from vllm_torchtpu.megakernel.kimi_k3.kernels import prefill_megakernel as k

    sharded, replicated = P(AXIS), P()
    names = k.MOE_FAMILIES
    if mixture:
        moe = k.make_indexed_mixture_moe_sp_ring(mesh, seq_len=window)

        def run(prefix, blocks, mlp_res, ffn_norm, active, length, slot, *families):
            out = moe(
                prefix, blocks, mlp_res, ffn_norm, active,
                dict(zip(names, families)), length, slot,
            )  # fmt: skip
            return out if counting else (out,)

        in_specs = (sharded,) * 4 + (replicated,) * 3 + (sharded,) * len(names)
    else:
        moe = k.make_indexed_moe_sp_ring(mesh, seq_len=window)

        def run(hidden, prefix, length, slot, *families):
            out = moe(hidden, prefix, dict(zip(names, families)), length, slot)
            return out if counting else (out,)

        in_specs = (sharded,) * 2 + (replicated,) * 2 + (sharded,) * len(names)
    kind = "mixture" if mixture else "plain"
    # The export cache keys on the package's sources; this wrapper is not one.
    wrapper = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
    return register_global_op(
        f"pallas::k3_bench_moe_{kind}_w{window}",
        mesh,
        run,
        in_specs,
        (sharded,) * (2 if counting else 1),
        export_key=repr(("bench-moe", kind, window, names, wrapper)),
    )


def _operands(inputs: _Inputs, layer: _Layer, mixture: bool, active):
    """The op's operands for a case, settled on the device."""
    import ml_dtypes
    from torch_tpu._internal.sync import synchronize

    from vllm_torchtpu.megakernel import bridge

    residual = inputs.residual.astype(ml_dtypes.bfloat16)
    length, slot = _device_i32(inputs.length), layer.slot
    if mixture:
        blocks = np.broadcast_to(residual, (8, *residual.shape))
        operands = (
            bridge.upload(residual),
            bridge.upload(np.ascontiguousarray(blocks)),
            *layer.mixture,
            active,
            length,
            slot,
            *layer.families,
        )
    else:
        hidden = inputs.hidden.astype(ml_dtypes.bfloat16)
        operands = (
            bridge.upload(hidden),
            bridge.upload(residual),
            length,
            slot,
            *layer.families,
        )
    synchronize(list(operands), wait=True)
    return operands


def _routes_error(counted, inputs: _Inputs, rank: int) -> str:
    """How the op's per-chip expert counts differ from the case's, if they
    do (``prefill_route_counts``)."""
    from tools.megakernel import moe_layer

    chip = rank // 2
    expected = np.bincount(inputs.sets.reshape(-1), minlength=moe_layer.EXPERTS)
    expected = expected[moe_layer.LOCAL * chip : moe_layer.LOCAL * (chip + 1)]
    counted = counted.cpu().numpy().reshape(-1)
    if np.array_equal(counted, expected):
        return ""
    wrong = int(np.flatnonzero(counted != expected)[0])
    return (
        f"rank {rank}: chip {chip} routed {counted[wrong]} rows to local expert "
        f"{wrong}, the case {expected[wrong]}"
    )


def _launch(op, operands, count: int) -> tuple:
    """``count`` launches dispatched back to back and waited for once; the
    last launch's outputs. The warm-up uses it too, so every launch runs the
    graphs the warm-up compiled."""
    from torch_tpu._internal.sync import synchronize

    from vllm_torchtpu.megakernel import bridge

    for _ in range(count):
        out = op(*operands)
        synchronize(list(out), wait=False)
    bridge.materialize(out[0])
    return out


def _time(op, operands, args, tag: str) -> tuple[list[float], tuple]:
    """us per launch of each repeat, and the last launch's outputs."""
    from tools.megakernel.bench import _trace

    times = []
    for repeat in range(args.repeats):
        harness.sync()
        with _trace(args.trace, tag, repeat == args.repeats - 1):
            if args.spaced is None:
                began = time.perf_counter()
                out = _launch(op, operands, args.launches)
                times.append(1e6 * (time.perf_counter() - began) / args.launches)
            else:
                each = []
                for _ in range(args.launches):
                    began = time.perf_counter()
                    out = _launch(op, operands, 1)
                    each.append(1e6 * (time.perf_counter() - began))
                    time.sleep(args.spaced / 1e3)
                times.append(statistics.median(each))
    return times, out


def _reference(inputs: _Inputs, layer: _Layer, window: int, rank: int, args):
    """Mesh position 0's first ``--reference-rows`` rows as the host reference
    computes them, and the residual rows they add to, on that rank (every rank
    computes its chip's routed part); None elsewhere."""
    from tools.megakernel import moe_layer
    from vllm_torchtpu.megakernel.kimi_k3.kernels import prefill_megakernel as k

    rows = min(args.reference_rows, window // 32)
    residual = _gather(inputs.residual[:rows] if rank == 0 else None)[0]
    tokens = k.sp_token_index(0, np.arange(rows), k.window_stripe(window // 32))
    host = layer.host
    hidden = moe_layer.post_norm(residual, layer.norm)

    def partial():
        logits = moe_layer.matmul(hidden, layer.router)
        experts, weights = moe_layer.route(logits, layer.bias)
        latent = moe_layer.latents(hidden, host["latent_down"][0].astype(np.float32))
        subtotal = moe_layer.routed_partial(latent, experts, weights, rank // 2, host)
        if rank >= 2:
            return _bf16_bits(subtotal), None
        # Rank 0's chip: the two halves of its rows' shared expert.
        shared = moe_layer.shared_partial(
            hidden,
            host["shared_gu"][0].astype(np.float32),
            host["shared_down"][0].astype(np.float32),
        )
        return _bf16_bits(subtotal), shared

    parts = _gather(_on_every_rank("reference partials", partial))

    def finish():
        return moe_layer.finish(
            [_float(bits) for bits, _ in parts],
            [half for _, half in parts if half is not None],
            residual,
            tokens < inputs.length,
            host["latent_norm"][0, 0],
            host["latent_up"][0].astype(np.float32),
        )

    out = _on_every_rank("reference", finish if rank == 0 else lambda: None)
    return (out, residual) if rank == 0 else None


def run(args, records: harness.Records) -> None:
    from torch_tpu._internal.sync import synchronize

    from tools.megakernel import moe_layer
    from vllm_torchtpu.megakernel import bridge
    from vllm_torchtpu.megakernel.kimi_k3 import config
    from vllm_torchtpu.megakernel.kimi_k3.kernels import prefill_megakernel as k
    from vllm_torchtpu.megakernel.kimi_k3.kernels import rank_loader
    from vllm_torchtpu.megakernel.kimi_k3.kernels.prefill_moe import fused

    windows = list(dict.fromkeys(int(w) for w in args.windows.split(",")))
    for window in windows:
        if window not in k.SMALL_WINDOWS and (
            window % k.SP_BLOCK or not k.SP_BLOCK <= window <= 16384
        ):
            harness.fail(f"--windows {window}: 512, 1024 or 2048..16384 by 2048")
    if not 0 <= args.layer < 92:
        harness.fail(f"--layer {args.layer}: a MoE layer is 0..91")
    if args.launches < 1 or args.repeats < 1 or args.reference_rows < 1:
        harness.fail("--launches, --repeats and --reference-rows are at least 1")
    slot = args.family_layers - 1 if args.slot < 0 else args.slot
    layout = moe_layer.Layout(args.family_layers, args.dense_layers or slot + 1, slot)
    if layout.problem():
        harness.fail(layout.problem())
    mixture = args.op == "mixture"
    counting = config.active().prefill_route_counts
    mesh, rank = bridge.kernel_mesh(AXIS, 32)
    order = rank_loader.expert_plan(93)
    if order is None:
        order = np.tile(np.arange(moe_layer.EXPERTS), (92, 1))
    cases = moe_layer.parse(
        args.routes, order=order, routes_file=args.routes_file, seed=args.seed
    )
    layer = _on_every_rank(
        f"MoE layer {args.layer} weights",
        _Layer,
        args.presharded,
        rank,
        args.layer,
        args.zero_router_bias,
        layout,
    )
    logger = harness.rank() == 0  # the process whose lines the log keeps
    harness.log(
        f"MoE layer {args.layer}: {layer.gib():.2f} GiB per rank read in "
        f"{layer.read_s:.1f} s; families resident as {layout.experts} expert / "
        f"{layout.dense} dense layers ({layer.device_bytes / 2**30:.1f} GiB per "
        f"rank; slot {layout.slot}, expert index {layout.expert_index}) after "
        f"{layer.resident_s:.1f} s; router bias spans "
        "{:.4f}..{:.4f}".format(*layer.bias_span)
        + (" (run with zeros)" if args.zero_router_bias else "")
        + f"; {len(cases)} routing cases x windows {windows}"
    )
    offsets = layer.offsets(layout)
    harness.log(
        "the layer's entry in each family (GiB from the family's start, of its "
        "size): "
        + ", ".join(
            f"{name} [{a / 2**30:.3f}, {b / 2**30:.3f}) of {c / 2**30:.3f}"
            for name, (a, b, c) in offsets.items()
        )
    )
    active = _device_i32(_active_blocks(args.layer))
    synchronize([active], wait=True)
    mode = "back to back" if args.spaced is None else f"spaced {args.spaced:g} ms"
    saved: dict[str, np.ndarray] = {}
    for window in windows:
        op = _build_op(mesh, window, mixture, counting)
        parts = fused._COMPOUND_PARTS.get(window, (window,))
        compile_s = None
        for case in cases:
            tag = _tag(window, case.label)
            if not case.fits(window):
                harness.log(
                    f"{case.label}: its routed tokens exceed w{window}; not run"
                )
                continue
            inputs = _on_every_rank(
                f"{tag}: routing inputs", _synthesize, case, window, rank, layer, args
            )
            loads = moe_layer.case_loads(inputs.sets, inputs.length, parts)
            operands = _operands(inputs, layer, mixture, active)
            harness.sync()
            if compile_s is None:
                harness.log(f"{moe_layer.COMPILE_BEGINS} w{window}")
            began = time.perf_counter()
            first = _launch(op, operands, 1)
            if compile_s is None:
                compile_s = time.perf_counter() - began
                harness.log(
                    f"{moe_layer.COMPILE_ENDS} w{window}: first launch (compile + "
                    f"run) {compile_s:.1f} s"
                )
            first_bits = _bits(first[0])
            routes_error = _routes_error(first[1], inputs, rank) if counting else ""
            _fail_on(_gather(routes_error), f"{tag}: the op routed elsewhere")
            del first
            times, out = _time(op, operands, args, tag)
            bits = _bits(out[0])
            values = _float(bits).astype(np.float64)
            reference = (
                _reference(inputs, layer, window, rank, args)
                if args.reference
                else None
            )
            # Mesh position 0's rows (the reference's) reach the logging process.
            summary = _gather(
                {
                    "position": rank,
                    "digest": hashlib.sha256(bits.tobytes()).hexdigest(),
                    "same": bool(np.array_equal(bits, first_bits)),
                    "sum": float(values.sum()),
                    "abs": float(np.abs(values).sum()),
                    "margin": inputs.margin,
                    "scaled": inputs.scaled,
                    "inputs": inputs.digest(),
                    "sample": bits[:SAMPLE_ROWS] if args.save_output else None,
                    "rows": bits if rank == 0 and args.save_output else None,
                    "reference": reference if rank == 0 else None,
                    "own": bits[: len(reference[0])]
                    if rank == 0 and reference
                    else None,
                }
            )
            summary.sort(key=lambda s: s["position"])
            digest = hashlib.sha256(
                "".join(s["digest"] for s in summary).encode()
            ).hexdigest()[:16]
            same = all(s["same"] for s in summary)
            inputs_digest = hashlib.sha256(
                "".join(s["inputs"] for s in summary).encode()
            ).hexdigest()[:16]
            checksum = sum(s["sum"] for s in summary)
            magnitude = sum(s["abs"] for s in summary)
            margin = min(s["margin"] for s in summary)
            scaled = sum(s["scaled"] for s in summary)
            accuracy = {}
            if args.reference:
                want, residual = summary[0]["reference"]
                accuracy = moe_layer.error(_float(summary[0]["own"]), want, residual)
            us = statistics.median(times)
            served = (
                f" (served window {case.window} layer {case.layer})"
                if case.window is not None
                else ""
            )
            harness.log(
                f"moe w{window} {args.op} {case.label}{served}: {us:.1f} us per "
                f"launch, {mode} x{args.launches} (repeats "
                f"{[round(t, 1) for t in times]}) | busiest chip: routed "
                f"{loads['routed']}, GU rows {loads['gu_rows']}, visits "
                f"{loads['visits']}, down rows {loads['down_rows']} (mean chip GU "
                f"{loads['mean_gu_rows']:.0f})"
                + (
                    " OVER THE COMPACT ROUTE CAPACITY (the op takes its overflow "
                    "path: at 16K two 8K launches, at 8K the uncapped routed "
                    "experts; the loads describe the compact launch)"
                    if loads["overflow"]
                    else ""
                )
                + f" | margin {margin:.4f} | inputs {inputs_digest} | digest "
                f"{digest} sum {checksum:.6e} abs {magnitude:.6e}"
                + ("" if same else " | LAST LAUNCH DIFFERS FROM THE FIRST")
                + (
                    " | vs reference: max rel {max_rel:.3e} rms rel {rms_rel:.3e} "
                    "identical {identical:.3f}".format(**accuracy)
                    if accuracy
                    else ""
                )
            )
            records.add(
                case=tag,
                window=window,
                op=args.op,
                routes=case.label,
                served_window=case.window,
                served_layer=case.layer,
                layer=args.layer,
                family_layers=layout.experts,
                dense_layers=layout.dense,
                slot=layout.slot,
                expert_index=layout.expert_index,
                offsets=offsets,
                length=inputs.length,
                mode=mode,
                launches=args.launches,
                us=round(us, 1),
                repeats=[round(t, 1) for t in times],
                compile_s=round(compile_s, 1),
                busiest_chip=loads["chip"],
                busiest_routed=loads["routed"],
                busiest_gu_rows=loads["gu_rows"],
                busiest_visits=loads["visits"],
                busiest_down_rows=loads["down_rows"],
                mean_routed=round(loads["mean_routed"], 1),
                mean_gu_rows=round(loads["mean_gu_rows"], 1),
                over_capacity=loads["overflow"],
                margin=round(margin, 5),
                scaled_rows=scaled,
                inputs_digest=inputs_digest,
                digest=digest,
                checksum=checksum,
                magnitude=magnitude,
                deterministic=same,
                routes_checked_on_device=counting,
                zero_router_bias=args.zero_router_bias,
                reference={n: round(v, 6) for n, v in accuracy.items()} or None,
            )
            if args.save_output and logger:
                saved[f"{tag}/rank0"] = summary[0]["rows"]
                saved[f"{tag}/sample"] = np.stack([s["sample"] for s in summary])
                if args.reference:
                    want, residual = summary[0]["reference"]
                    saved[f"{tag}/reference"] = _bf16_bits(want)
                    saved[f"{tag}/prefix"] = _bf16_bits(residual)
                meta = dict(
                    window=window,
                    op=args.op,
                    routes=case.label,
                    layer=args.layer,
                    length=inputs.length,
                    digest=digest,
                )
                saved[f"{tag}/meta"] = np.array(json.dumps(meta))
            del out, operands
    if args.save_output and logger:
        np.savez(args.save_output, **saved)
        harness.log(f"outputs saved to {args.save_output}")


# ----------------------------------------------------------------- compare


def _difference(x: np.ndarray, y: np.ndarray) -> str:
    fx, fy = _float(x).astype(np.float64), _float(y).astype(np.float64)
    differ = int(np.sum(x != y))
    delta = np.abs(fx - fy)
    scale = max(float(np.abs(fx).max()), 1e-30)
    rms = float(np.sqrt(np.mean((fx - fy) ** 2) / max(np.mean(fx**2), 1e-30)))
    verdict = "bit-identical" if not differ else f"{differ} of {x.size} differ"
    return (
        f"{verdict}; max |B-A| {delta.max():.4e} ({delta.max() / scale:.3e} of "
        f"max |A|), rms rel {rms:.3e}"
    )


def compare(path_a: str, path_b: str) -> int:
    """Print, per case both ``--save-output`` files hold, how B's outputs
    differ from A's (and each from its reference rows, when saved); 0 when
    every case is bit-identical."""
    from tools.megakernel import moe_layer

    a, b = np.load(path_a), np.load(path_b)
    tags = sorted(
        {n.rsplit("/", 1)[0] for n in a.files} & {n.rsplit("/", 1)[0] for n in b.files}
    )
    if not tags:
        print("no common cases")
        return 2
    status = 0
    for tag in tags:
        for part in ("rank0", "sample"):
            x, y = a[f"{tag}/{part}"], b[f"{tag}/{part}"]
            if x.shape != y.shape:
                print(f"{tag} {part}: shapes {x.shape} vs {y.shape}")
                status = 1
                continue
            print(f"{tag} {part}: {_difference(x, y)}")
            status = status or int(np.any(x != y))
        for name, run in (("A", a), ("B", b)):
            if f"{tag}/reference" in run.files:
                want = _float(run[f"{tag}/reference"])
                got = _float(run[f"{tag}/rank0"][: len(want)])
                accuracy = moe_layer.error(got, want, _float(run[f"{tag}/prefix"]))
                print(
                    f"{tag} {name} vs its reference: max rel "
                    "{max_rel:.3e} rms rel {rms_rel:.3e} identical "
                    "{identical:.3f}".format(**accuracy)
                )
    return status


def overlays(path: str) -> int:
    """Print each window's compiled overlays from a bench moe process log."""
    from tools.megakernel import moe_layer

    by_window = moe_layer.overlays_by_window(Path(path).read_text(errors="replace"))
    if not by_window:
        print(f"{path}: no bench moe compile in this log")
        return 2
    for window, programs in by_window.items():
        if not programs:
            print(f"w{window}: no overlay records (a cached executable, or no vmodule)")
        for program in programs:
            print(f"w{window}: " + json.dumps(program))
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "overlays":
        sys.exit(overlays(sys.argv[2]))
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: python -m tools.megakernel.bench_moe A.npz B.npz\n"
            "       python -m tools.megakernel.bench_moe overlays RANK_LOG"
        )
    sys.exit(compare(sys.argv[1], sys.argv[2]))
