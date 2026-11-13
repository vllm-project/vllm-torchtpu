# SPDX-License-Identifier: Apache-2.0
"""The host side of the single-layer fused-MoE bench (``bench moe``): routing
cases, their load, inputs that make the kernel route them, and a numpy
reference of the layer.

**Routing.** A case fixes the 16 experts (physical slots of the installed
``expert_plan``) of every token of a launch, from one of:

* ``uniform[:SEED]``: 16 distinct experts per token, uniform and independent;
* ``capacity:R[:CHIP]``: R of the launch's routes on one chip (by default chip
  0; R near or past the fused op's compact route capacity), evenly over its 56
  experts, the rest evenly over the other chips' experts;
* per-expert route counts (a served window's, or ``counts:FILE``), assigned
  exactly: the routes are laid out expert-major, experts in a random order,
  and route ``i`` goes to token ``perm[i % tokens]``, so every token gets 16
  routes and, since no expert holds more routes than there are tokens, 16
  distinct experts.

:func:`launch_loads` prices a case with the routed expert loop's tiling (the
busiest chip's routed rows, GU-bucket-padded rows and grid visits).
:class:`HiddenSolver` builds the residual rows whose RMS-normalized hidden
states make the kernel's router select exactly a case's experts, and
:func:`check_selection` checks that with :func:`top16`, the selection of the
kernel's ``fused._Top16`` (biased sigmoid scores, ties to the lowest expert).

**Reference.** :func:`routed_partial`, :func:`shared_partial` and
:func:`finish` compute the layer's output rows from the same weights and
inputs as the kernel. They keep the kernel's BF16 boundaries (latents,
route-weighted GU rows, each down row, every rank's routed subtotal, the
reduced and normalized rows, latent-up, the shared expert) but sum in FP32
and in their own order (the kernel's reduce ring rounds to BF16 after every
hop), so the two agree to a tolerance, not bit for bit. The routed part splits
as the kernel's does: each rank computes its chip's experts at its half of
the intermediate columns, and :func:`finish` sums every rank's subtotal.
"""

from __future__ import annotations

import dataclasses
import hashlib

import ml_dtypes
import numpy as np

EXPERTS, TOP_K, CHIPS, LOCAL = 896, 16, 16, 56
HIDDEN = 7168
SUBLANE = 16  # BF16 rows: expert groups start at any row, tiles at a 16-row boundary
BUCKET = 64  # row bucket of the MXFP4 grouped matmul
EPS = 1e-5
LATENT = 3584
LANES = 128  # gate/up columns alternate in chunks of 128 (rank_loader)


def bf16(values: np.ndarray) -> np.ndarray:
    """``values`` rounded to BF16, as FP32."""
    return np.asarray(values, np.float32).astype(ml_dtypes.bfloat16).astype(np.float32)


def seed_of(*parts) -> int:
    """A stable 63-bit seed for a tuple of labels."""
    digest = hashlib.sha256(repr(parts).encode()).digest()
    return int.from_bytes(digest[:8], "little") >> 1


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``a @ b`` on the host through torch, in the operands' (common) float
    type: numpy's BLAS threads stall on the skinny products used here."""
    import torch

    left = torch.from_numpy(np.ascontiguousarray(a))
    return (left @ torch.from_numpy(np.ascontiguousarray(b))).numpy()


# ----------------------------------------------------------------- cases


@dataclasses.dataclass(frozen=True)
class Case:
    """One routing: ``label`` names it in logs and records; ``counts`` (per
    physical slot), ``capacity`` routes on ``chip`` (sized by the launch) or
    ``seed`` (``uniform``) fixes every token's experts; the served
    ``window``/``layer`` it replays, if any."""

    label: str
    counts: np.ndarray | None = None
    seed: int = 0
    window: int | None = None
    layer: int | None = None
    capacity: int | None = None
    chip: int = 0

    def tokens(self, launch: int, length: int | None) -> int:
        """The routed (valid) tokens of a ``launch``-token launch: a counted
        case's own, else ``length`` or the whole launch."""
        if self.counts is None:
            tokens = launch if length is None else length
        else:
            total = int(self.counts.sum())
            if total % TOP_K:
                raise ValueError(f"{self.label}: {total} routes are not 16 per token")
            tokens = total // TOP_K
        if not 0 < tokens <= launch:
            raise ValueError(
                f"{self.label}: {tokens} routed tokens do not fit a "
                f"{launch}-token launch"
            )
        return tokens

    def fits(self, launch: int) -> bool:
        """Whether the case's routes fit a ``launch``-token launch."""
        if self.capacity is not None:
            return self.capacity <= TOP_K * launch
        return self.counts is None or int(self.counts.sum()) <= TOP_K * launch

    def sets(self, tokens: int) -> np.ndarray:
        """[tokens, 16] distinct experts per token."""
        if self.capacity is not None:
            counts = capacity_counts(self.capacity, tokens, self.chip)
            return sets_from_counts(counts, seed_of(self.label))
        if self.counts is None:
            return uniform_sets(tokens, self.seed)
        return sets_from_counts(self.counts, seed_of(self.label))


def capacity_counts(routes: int, tokens: int, chip: int) -> np.ndarray:
    """Per-slot counts of ``tokens`` tokens with ``routes`` of their routes on
    ``chip``, evenly over its experts, and the rest evenly over the others."""
    if not 0 <= chip < CHIPS or not 0 <= routes <= TOP_K * tokens:
        raise ValueError(f"capacity: {routes} routes of {tokens} tokens on chip {chip}")

    def even(total, experts):
        return total // experts + (np.arange(experts) < total % experts)

    counts = np.empty(EXPERTS, np.int64)
    mine = np.arange(LOCAL * chip, LOCAL * (chip + 1))
    others = np.setdiff1d(np.arange(EXPERTS), mine)
    counts[mine] = even(routes, LOCAL)
    counts[others] = even(TOP_K * tokens - routes, len(others))
    return counts


def uniform_sets(tokens: int, seed: int) -> np.ndarray:
    """16 distinct experts per token, uniform and independent."""
    keys = np.random.default_rng(seed).random((tokens, EXPERTS), dtype=np.float32)
    return np.argpartition(keys, TOP_K - 1, axis=1)[:, :TOP_K].astype(np.int32)


def sets_from_counts(counts: np.ndarray, seed: int) -> np.ndarray:
    """[tokens, 16] experts whose per-expert route counts are ``counts``
    exactly (see the module docstring), ``tokens = counts.sum() / 16``."""
    counts = np.asarray(counts, np.int64)
    tokens = int(counts.sum()) // TOP_K
    if counts.shape != (EXPERTS,) or counts.min() < 0 or tokens * TOP_K != counts.sum():
        raise ValueError("route counts: 896 non-negative counts, 16 per token")
    if counts.max() > tokens:
        raise ValueError(
            f"route counts: an expert holds {counts.max()} routes of {tokens} tokens"
        )
    rng = np.random.default_rng(seed)
    experts = rng.permutation(EXPERTS)
    routes = np.repeat(experts, counts[experts])
    token = rng.permutation(tokens)
    position = np.arange(TOP_K * tokens)
    sets = np.empty((tokens, TOP_K), np.int32)
    sets[token[position % tokens], position // tokens] = routes
    return sets


def parse(
    specs: str,
    *,
    order: np.ndarray,
    routes_file: str = "",
    seed: int = 0,
) -> list[Case]:
    """Cases of a comma-separated ``--routes`` list:

    * ``uniform`` / ``uniform:SEED``;
    * ``serving:W:L``: served window ``W``'s route counts at MoE layer ``L``
      (``routes_file``), placed by ``order`` [92, 896] (the installed plan's
      logical expert at each physical slot);
    * ``sweep:N``: ``routes_file``'s cases nearest to ``N`` busiest-chip
      GU-padded row loads evenly spaced over its range (16K tiling);
    * ``capacity:R[:CHIP]``: R routes on one chip (:func:`capacity_counts`);
    * ``counts:FILE``: an ``.npy`` of per-physical-slot counts, [896] or
      [K, 896] (K cases).
    """
    served = None
    cases = []
    for spec in filter(None, (s.strip() for s in specs.split(","))):
        kind, _, rest = spec.partition(":")
        if kind == "uniform":
            cases.append(Case(spec, seed=int(rest) if rest else seed))
        elif kind in ("serving", "sweep"):
            if served is None:
                served = ServedRoutes(routes_file)
            if kind == "serving":
                window, _, layer = rest.partition(":")
                cases.append(served.case(int(window), int(layer), order))
            else:
                cases.extend(served.sweep(int(rest), order))
        elif kind == "capacity":
            routes, _, chip = rest.partition(":")
            cases.append(Case(spec, capacity=int(routes), chip=int(chip or 0)))
        elif kind == "counts":
            counts = np.load(rest)
            for k, row in enumerate(np.atleast_2d(counts)):
                label = spec if counts.ndim == 1 else f"{spec}#{k}"
                cases.append(Case(label, counts=row.astype(np.int64)))
        else:
            raise ValueError(
                f"--routes {spec!r}: uniform, serving, sweep, capacity or counts"
            )
    if not cases:
        raise ValueError("--routes names no case")
    unique: dict[str, Case] = {}
    for case in cases:  # a case named twice (e.g. also picked by a sweep) runs once
        unique.setdefault(case.label, case)
    return list(unique.values())


class ServedRoutes:
    """Served route counts: an ``.npz`` with ``counts`` [K, 896] per LOGICAL
    expert, ``window`` [K] and ``layer`` [K] (the served window and MoE layer
    of each row)."""

    def __init__(self, path: str) -> None:
        if not path:
            raise ValueError("serving/sweep routes need --routes-file")
        data = np.load(path)
        self.path = path
        self.counts = data["counts"].astype(np.int64)
        self.window, self.layer = data["window"], data["layer"]

    def _case(self, row: int, order: np.ndarray) -> Case:
        window, layer = int(self.window[row]), int(self.layer[row])
        return Case(
            f"serving:{window}:{layer}",
            counts=self.counts[row][order[layer]],
            window=window,
            layer=layer,
        )

    def case(self, window: int, layer: int, order: np.ndarray) -> Case:
        rows = np.flatnonzero((self.window == window) & (self.layer == layer))
        if not rows.size:
            held = sorted(set(self.window[self.layer == layer].tolist()))
            raise ValueError(
                f"{self.path} holds no window {window} at layer {layer} "
                f"(windows with that layer: {held[:24]}...)"
            )
        return self._case(int(rows[0]), order)

    def sweep(self, count: int, order: np.ndarray) -> list[Case]:
        """The cases nearest to ``count`` busiest-chip GU-row loads evenly
        spaced from the lightest to the heaviest case (16K tiling)."""
        cases = [self._case(row, order) for row in range(len(self.counts))]
        loads = np.array(
            [busiest(launch_loads(c.counts, 16384))["gu_rows"] for c in cases]
        )
        targets = np.linspace(loads.min(), loads.max(), count)
        picks = dict.fromkeys(int(np.argmin(np.abs(loads - t))) for t in targets)
        return [cases[i] for i in sorted(picks, key=lambda i: loads[i])]


# ----------------------------------------------------------------- layout


@dataclasses.dataclass(frozen=True)
class Layout:
    """How the device holds a layer's families: the four expert families with
    ``experts`` layers (8: a prefill-role bank, 92: the aggregate's), the
    others with ``dense`` layers, and the MoE ``slot`` the op runs. The
    layer's weights sit at :attr:`expert_index` (the kernel's index for the
    slot) and at ``slot``; every other entry is zero."""

    experts: int
    dense: int
    slot: int

    @property
    def expert_index(self) -> int:
        from vllm_torchtpu.megakernel.kimi_k3.kernels import prefill_megakernel as k

        if self.experts == 1:  # expert_family_slot's single-layer bank
            return 0
        return int(k.expert_family_slot(self.experts, self.slot))

    def problem(self) -> str:
        """Why the op cannot run at ``slot``, or an empty string."""
        if min(self.experts, self.dense) < 1 or self.slot < 0:
            return "--family-layers and --dense-layers are at least 1, --slot >= 0"
        if self.slot >= self.dense:
            return f"--slot {self.slot} is past the {self.dense} dense layers"
        if not 0 <= self.expert_index < self.experts:
            return f"--slot {self.slot} maps past the {self.experts} expert layers"
        return ""

    def entries(self, names) -> dict[str, tuple[int, int]]:
        """Each family's (layers, index of the layer's entry)."""
        experts = {"expert_gu", "expert_gus", "expert_down", "expert_ds"}
        return {
            name: (self.experts, self.expert_index)
            if name in experts
            else (self.dense, self.slot)
            for name in names
        }


COMPILE_BEGINS = "bench moe compile begins:"  # bench_moe's lines around a compile
COMPILE_ENDS = "bench moe compile ends:"


def overlays_by_window(log: str) -> dict[int, list[dict[str, int | str]]]:
    """:func:`overlay_programs` of each window's compile in a bench moe process
    log: the lines between its compile markers."""
    import re

    found, window, lines = {}, None, []
    for line in log.splitlines():
        if match := re.search(re.escape(COMPILE_BEGINS) + r" w(\d+)", line):
            window, lines = int(match[1]), []
        elif window is not None and COMPILE_ENDS in line:
            found[window] = overlay_programs("\n".join(lines))
            window = None
        elif window is not None:
            lines.append(line)
    return found


def overlay_programs(log: str) -> list[dict[str, int | str]]:
    """The overlay layout of every program compiled in a libtpu log written
    with ``--vmodule=overlay_partitioner=1,overlay=1``: its input bundle count,
    the regular/jumbo/hlo overlay sizes and the partitions of each kind."""
    import re

    programs: list[dict[str, int | str]] = []
    for line in log.splitlines():
        if match := re.search(
            r"OverlayProgram\(\), input program bundle count: (\d+)", line
        ):
            programs.append({"bundles": int(match[1]), "sizes": "", "partitions": 0})
        elif programs and (
            match := re.search(r"regular/jumbo/hlo overlay size (\d+/\d+/\d+)", line)
        ):
            programs[-1]["sizes"] = match[1]
        elif programs and (
            match := re.search(r"OverlayPartition \d+: .*kind=(\d+)", line)
        ):
            kind = f"kind{match[1]}"
            programs[-1][kind] = int(programs[-1].get(kind, 0)) + 1
            programs[-1]["partitions"] = int(programs[-1]["partitions"]) + 1
    return programs


def inputs_digest(length: int, *rows: np.ndarray) -> str:
    """sha256 of a rank's routed length and operand rows."""
    digest = hashlib.sha256(str(length).encode())
    for part in rows:
        digest.update(np.ascontiguousarray(part, np.float32).tobytes())
    return digest.hexdigest()


def held_family(value: np.ndarray, layers: int, index: int) -> np.ndarray:
    """``value`` ([1, ...]) as entry ``index`` of ``layers`` zeroed entries
    (numpy's zeros: pages stay untouched until something reads them)."""
    if layers == 1:
        return value
    held = np.zeros((layers, *value.shape[1:]), value.dtype)
    held[index] = value[0]
    return held


# ----------------------------------------------------------------- load


def chip_loads(counts: np.ndarray, tile: int) -> dict[str, np.ndarray]:
    """Per-chip work of one launch's routed expert loop from per-slot route
    counts [896]: routed rows, grid visits (``fill_metadata``: an expert's
    routes start at the running offset and its tiles of ``tile`` rows at
    16-row boundaries), GU rows with each tail tile at its row bucket
    (128/192/256 at ``tile`` 256, whose 64-row bucket is not compiled for GU;
    64/128 at 128) and down rows (64-row buckets)."""
    groups = np.asarray(counts, np.int64).reshape(CHIPS, LOCAL)
    start = np.cumsum(groups, axis=1) - groups
    aligned = groups + start % SUBLANE
    live = groups > 0
    visits = np.where(live, -(-aligned // tile), 0)
    last = np.where(live, aligned - tile * (visits - 1), 0)
    down_last = BUCKET * -(-last // BUCKET)
    gu_last = np.maximum(down_last, 2 * BUCKET) if tile == 256 else down_last
    full = tile * np.maximum(visits - 1, 0)
    return {
        "routed": groups.sum(1),
        "visits": visits.sum(1),
        "gu_rows": np.where(live, full + gu_last, 0).sum(1),
        "down_rows": np.where(live, full + down_last, 0).sum(1),
    }


def launch_tile(tokens: int) -> int:
    """The routed loop's row tile of a ``tokens``-token launch."""
    return 256 if tokens >= 8192 else 128


def launch_loads(counts: np.ndarray, tokens: int) -> dict[str, np.ndarray]:
    """``chip_loads`` of a single ``tokens``-token launch."""
    return chip_loads(counts, launch_tile(tokens))


def busiest(loads: dict[str, np.ndarray]) -> dict[str, int]:
    """Each metric's busiest chip value."""
    return {name: int(value.max()) for name, value in loads.items()}


def case_loads(
    sets: np.ndarray, length: int, parts: tuple[int, ...]
) -> dict[str, float | int]:
    """A case's load summary for a launch made of ``parts`` (token counts,
    consecutive token ranges: one part, or a compound window's 8K/4K/2K
    launches): per metric, the sum over parts of the busiest chip, and the
    mean chip's routed and GU rows; ``chip`` is the busiest chip by GU rows
    of the largest part; ``overflow`` whether a part's busiest chip holds
    more routes than the op's compact capacity."""
    from vllm_torchtpu.megakernel.kimi_k3.kernels.prefill_moe import fused

    summary: dict[str, float | int] = {"overflow": False}
    begin = 0
    for tokens in parts:
        part = sets[begin : min(begin + tokens, length)]
        begin += tokens
        counts = np.bincount(part.reshape(-1), minlength=EXPERTS)
        loads = launch_loads(counts, tokens)
        # A chip past the compact route capacity sends the launch to its
        # uncapped path (fused._layer_kernel).
        overflow = loads["routed"].max() > fused._route_capacity(tokens)
        summary["overflow"] = bool(summary["overflow"] or overflow)
        if "chip" not in summary:
            summary["chip"] = int(np.argmax(loads["gu_rows"]))
        for name, value in busiest(loads).items():
            summary[name] = summary.get(name, 0) + value
        for name in ("routed", "gu_rows"):
            mean = float(loads[name].mean())
            summary[f"mean_{name}"] = summary.get(f"mean_{name}", 0.0) + mean
    return summary


# ----------------------------------------------------------------- hidden


def post_norm(residual: np.ndarray, norm: np.ndarray) -> np.ndarray:
    """The fused MoE's hidden rows when every residual block and the prefix
    are ``residual`` (BF16 values): the mixture returns them unchanged, then
    RMSNorm and ``norm``, rounded to BF16 as the kernel does."""
    rows = np.asarray(residual, np.float32)
    mean = np.mean(rows * rows, axis=1, keepdims=True, dtype=np.float32)
    unit = rows * (np.float32(1) / np.sqrt(mean + np.float32(EPS)))
    return bf16(unit * np.asarray(norm, np.float32))


def target_logits(
    sets: np.ndarray,
    valid: np.ndarray,
    bias: np.ndarray,
    rng: np.random.Generator,
    *,
    margin: float,
    least: float = 0.01,
    jitter: float = 0.5,
) -> np.ndarray:
    """[rows, 896] router logits whose biased sigmoid scores put each valid
    row's 16 experts ``sets`` above the rest by ``margin``, or by what the
    bias leaves (sigmoids span less than one) if that is less, but at least
    ``least``. Per row a threshold between the chosen and the other scores is
    picked on a grid to keep the logits small (unit-RMS rows must reach
    them); ``jitter`` spreads them so route weights differ. Invalid rows get
    zeros."""
    rows = sets.shape[0]
    chosen = np.zeros((rows, EXPERTS), bool)
    chosen[np.arange(rows)[:, None], sets] = True
    bias = np.asarray(bias, np.float64)[None, :]
    tiny = 1e-3  # keeps every target sigmoid inside (0, 1)
    top_other = np.where(chosen, -np.inf, bias).max(1)
    low_chosen = np.where(chosen, bias, np.inf).min(1)
    gap = np.minimum(margin, low_chosen + 1 - top_other - 4 * tiny)
    short = valid & (gap < least)
    if short.any():
        row = int(np.flatnonzero(short)[0])
        raise ValueError(
            f"router bias spans {bias.min():.4f}..{bias.max():.4f}: no sigmoid "
            f"logits put row {row}'s experts {sorted(sets[row].tolist())} "
            f"{least} above the rest (bench moe --zero-router-bias reaches any set)"
        )
    gap = np.maximum(gap, least)[:, None]
    low = top_other[:, None] + gap / 2 + tiny
    high = low_chosen[:, None] + 1 - gap / 2 - tiny

    def logit(p):
        p = np.clip(p, tiny, 1 - tiny)
        return np.log(p) - np.log1p(-p)

    best, best_cost = None, None
    for t in np.linspace(0, 1, 17):
        threshold = low + t * (high - low)
        up = np.maximum(logit(threshold + gap / 2 - bias), 0)
        down = np.minimum(logit(threshold - gap / 2 - bias), 0)
        logits = np.where(chosen, up, down)
        cost = np.sum(logits * logits, axis=1)
        if best is None:
            best, best_cost = logits, cost
        else:
            better = cost < best_cost
            best = np.where(better[:, None], logits, best)
            best_cost = np.where(better, cost, best_cost)
    spread = rng.uniform(0, jitter, (rows, EXPERTS))
    best = best + np.where(chosen, spread, -spread)
    return np.where(valid[:, None], best, 0.0)


class HiddenSolver:
    """Residual rows whose kernel hidden states (:func:`post_norm`) have given
    router logits: unit-RMS rows ``u`` with ``u @ scaled = logits``
    (``scaled = norm[:, None] * router``), the minimum-norm solution plus a
    random component orthogonal to ``scaled``'s 896 columns, sized for an RMS
    of exactly one (the mixture and RMSNorm then return ``u * norm`` up to
    rounding). Projections go through the inverse of the 896 x 896 Gram
    matrix (torch's Cholesky: numpy's LAPACK inverse is far slower with many
    BLAS threads).
    """

    def __init__(self, router: np.ndarray, norm: np.ndarray) -> None:
        import torch

        self.router = np.asarray(router, np.float32)  # [hidden, 896], BF16 values
        self.norm = np.asarray(norm, np.float32)  # [hidden]
        self.scaled = self.norm.astype(np.float64)[:, None] * self.router
        gram = torch.from_numpy(matmul(self.scaled.T, self.scaled))
        self.inverse = torch.cholesky_inverse(torch.linalg.cholesky(gram)).numpy()

    def _span(self, coefficients: np.ndarray) -> np.ndarray:
        """Rows of ``scaled``'s column span with ``row @ scaled`` given."""
        return matmul(matmul(coefficients, self.inverse), self.scaled.T)

    def residual(
        self, logits: np.ndarray, rng: np.random.Generator, *, fill: float = 0.9
    ) -> tuple[np.ndarray, int]:
        """BF16 residual rows [rows, hidden] (as FP32) and how many rows had
        their logits scaled down to stay reachable (the minimum-norm part may
        take at most ``fill`` of the unit RMS)."""
        exact = self._span(logits)
        energy = np.sum(exact * exact, axis=1)
        scale = np.minimum(1.0, np.sqrt(fill * HIDDEN / np.maximum(energy, 1e-30)))
        exact *= scale[:, None]
        noise = rng.standard_normal(exact.shape)
        noise -= self._span(matmul(noise, self.scaled))
        left = HIDDEN - np.sum(exact * exact, axis=1)
        alpha = np.sqrt(left / np.sum(noise * noise, axis=1))
        return bf16(exact + alpha[:, None] * noise), int(np.sum(scale < 1))


def top16(logits: np.ndarray, bias: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The experts ``fused._Top16`` selects, in its order, and every row's
    biased scores sorted descending: FP32 sigmoid plus bias, ties to the
    lowest expert id."""
    logits = np.asarray(logits, np.float32)
    raw = np.float32(1) / (np.float32(1) + np.exp(-logits))
    scores = raw + np.asarray(bias, np.float32)[None, :]
    ranked = np.argsort(-scores, axis=1, kind="stable")
    return ranked[:, :TOP_K], np.take_along_axis(scores, ranked, axis=1)


def check_selection(
    hidden: np.ndarray,
    router: np.ndarray,
    bias: np.ndarray,
    sets: np.ndarray,
    valid: np.ndarray,
    least: float = 1e-3,
) -> float:
    """Check that every valid row of ``hidden`` (BF16 values) selects exactly
    its 16 experts ``sets`` through ``router`` and ``bias`` (:func:`top16`)
    with a gap of at least ``least`` between its 16th and 17th score (so the
    device's own rounding cannot reorder them), and return the smallest gap.
    Raises with the first row that fails."""
    logits = matmul(np.asarray(hidden, np.float32), np.asarray(router, np.float32))
    selected, scores = top16(logits, bias)
    selected = np.sort(selected, axis=1)
    expected = np.sort(sets, axis=1)
    wrong = np.flatnonzero(valid & np.any(selected != expected, axis=1))
    if wrong.size:
        row = int(wrong[0])
        raise ValueError(
            f"{wrong.size} rows route elsewhere; row {row} selects "
            f"{selected[row].tolist()} instead of {expected[row].tolist()}"
        )
    gaps = np.where(valid, scores[:, TOP_K - 1] - scores[:, TOP_K], np.inf)
    if gaps.min() < least:
        row = int(np.argmin(gaps))
        raise ValueError(f"row {row}'s 16th and 17th scores are {gaps[row]:.2e} apart")
    return float(gaps.min())


# ----------------------------------------------------------------- reference

FP4 = np.array(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float32
)


def dequantize(packed: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """MXFP4 weights [K/8, N] uint32 (eight FP4 codes per word, low nibble
    first along K) with E8M0 scales [K/32, N] as FP32 [K, N]."""
    shifts = np.arange(8, dtype=np.uint32) * 4
    codes = (packed[:, None, :] >> shifts[None, :, None]) & 15
    values = FP4[codes.reshape(-1, packed.shape[-1])]
    scale = np.exp2(scales.astype(np.float32) - 127)
    return values * np.repeat(scale, 32, axis=0)


def situ(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    """The SiTU activation (beta 4, linear beta 25) of FP32 gate/up."""
    with np.errstate(over="ignore"):  # exp(-gate) -> inf: the sigmoid is 0
        sigmoid = 1 / (1 + np.exp(-gate))
    return 4 * np.tanh(gate / 4) * sigmoid * (25 * np.tanh(up / 25))


def route(logits: np.ndarray, bias: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each row's 16 experts and route weights (their sigmoids, normalized)."""
    experts, _ = top16(logits, bias)
    raw = 1 / (1 + np.exp(-np.take_along_axis(logits, experts, axis=1)))
    return experts, raw / raw.sum(axis=1, keepdims=True)


def latents(hidden: np.ndarray, latent_down: np.ndarray) -> np.ndarray:
    return bf16(matmul(hidden, latent_down[:, :LATENT]))


def expert_rows(
    rows: np.ndarray,
    weights: np.ndarray,
    gu: np.ndarray,
    down: np.ndarray,
) -> np.ndarray:
    """One expert at one rank's half of the intermediate: BF16 latent ``rows``
    [n, 3584], route ``weights`` [n], dequantized gate/up ``gu`` [3584, W]
    (lane-interleaved) and ``down`` [W/2, 3584]; the BF16 down rows."""
    product = matmul(rows, gu)
    chunks = product.reshape(len(rows), -1, 2, LANES)
    middle = situ(chunks[:, :, 0], chunks[:, :, 1]).reshape(len(rows), -1)
    return bf16(matmul(bf16(middle * weights[:, None]), down))


def routed_partial(
    rows: np.ndarray,
    experts: np.ndarray,
    weights: np.ndarray,
    chip: int,
    families: dict[str, np.ndarray],
) -> np.ndarray:
    """One rank's routed subtotal [n, 3584] (BF16 values): every route of the
    BF16 latent ``rows`` to a local expert of ``chip`` (``experts``/``weights``
    [n, 16], physical slots), through the rank's ``expert_*`` families
    ([1, 56, ...] as ``rank_loader.read_moe_layer`` returns them)."""
    sums = np.zeros((len(rows), LATENT), np.float32)
    local = experts - LOCAL * chip
    for expert in np.unique(local[(local >= 0) & (local < LOCAL)]):
        token, column = np.nonzero(local == expert)
        gu = dequantize(
            families["expert_gu"][0, expert], families["expert_gus"][0, expert]
        )
        down = dequantize(
            families["expert_down"][0, expert], families["expert_ds"][0, expert]
        )
        out = expert_rows(rows[token], weights[token, column], gu, down)
        np.add.at(sums, token, out)
    return bf16(sums)


def shared_partial(
    hidden: np.ndarray, shared_gu: np.ndarray, shared_down: np.ndarray
) -> np.ndarray:
    """One rank's half of the shared expert, FP32 [n, hidden]: ``shared_gu``
    [hidden, 2 W] (gate then up), ``shared_down`` [W, hidden]."""
    width = shared_down.shape[0]
    gate, up = (
        matmul(hidden, shared_gu[:, :width]),
        matmul(hidden, shared_gu[:, width:]),
    )
    return matmul(bf16(situ(gate, up)), shared_down)


def finish(
    subtotals: list[np.ndarray],
    shared: list[np.ndarray],
    prefix: np.ndarray,
    valid: np.ndarray,
    latent_norm: np.ndarray,
    latent_up: np.ndarray,
) -> np.ndarray:
    """The layer's output rows (BF16 values) from every rank's routed
    subtotal, the two shared halves, the residual ``prefix`` rows and the
    latent norm [>= 3584] / latent-up [3584, hidden] weights."""
    reduced = bf16(np.sum(subtotals, axis=0, dtype=np.float32))
    mean = np.mean(reduced * reduced, axis=1, keepdims=True, dtype=np.float32)
    normed = bf16(reduced / np.sqrt(mean + np.float32(EPS)) * latent_norm[:LATENT])
    value = bf16(bf16(matmul(normed, latent_up)) + bf16(np.sum(shared, axis=0)))
    return bf16(prefix + np.where(valid[:, None], value, 0))


def error(kernel: np.ndarray, reference: np.ndarray, prefix: np.ndarray) -> dict:
    """How the kernel's output rows differ from the reference's, on the
    layer's contribution (output minus the residual prefix)."""
    got, want = kernel - prefix, reference - prefix
    delta = np.abs(got.astype(np.float64) - want)
    scale = max(float(np.abs(want).max()), 1e-30)
    power = max(float(np.mean(want.astype(np.float64) ** 2)), 1e-30)
    return {
        "max_abs": float(delta.max()),
        "max_rel": float(delta.max() / scale),
        "rms_rel": float(np.sqrt(np.mean(delta**2) / power)),
        "identical": float(np.mean(kernel == reference)),
    }
