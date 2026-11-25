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
"""Routed MoE expert activation envelopes and concave hull loader."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal, get_args

from .constants import (
    ACTIVATION_HULL_SCHEMA,
    HULL_BERNOULLI_CAP_TOLERANCE,
    HULL_CONCAVITY_SLOPE_TOLERANCE,
)
from .geometry import layer_ffn_kinds

if TYPE_CHECKING:
    from .geometry import ModelGeometry

type ActivationRegime = Literal["pigeonhole", "bernoulli", "empirical_hull"]

_VALID_ACTIVATION_REGIMES: frozenset[str] = frozenset(
    get_args(ActivationRegime.__value__)
)
_HULL_SPEC_PREFIX: str = "hull:"


def _validate_curve_geometry(
    token_grid: Sequence[int],
    active_experts: Sequence[float],
    curve_name: str,
) -> None:
    """Validate strictly increasing token grid, monotonicity, and concavity."""
    if not token_grid or len(token_grid) != len(active_experts):
        msg = (
            f"Activation hull {curve_name!r} requires non-empty equal-length "
            "'token_grid' and 'active_experts' sequences"
        )
        raise ValueError(msg)
    prev_tokens = 0
    prev_active = 0.0
    prev_slope: float | None = None
    for idx, (tokens, active_val) in enumerate(
        zip(token_grid, active_experts, strict=True)
    ):
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
            msg = (
                f"Activation hull {curve_name!r} token_grid entries "
                "must be positive ints"
            )
            raise ValueError(msg)
        if (
            isinstance(active_val, bool)
            or not math.isfinite(active_val)
            or active_val < 0.0
        ):
            msg = (
                f"Activation hull {curve_name!r} active_experts entries must be finite"
            )
            raise ValueError(msg)
        if tokens <= prev_tokens:
            msg = f"Activation hull {curve_name!r} token_grid must strictly increase"
            raise ValueError(msg)
        if active_val < prev_active:
            msg = (
                f"Activation hull {curve_name!r} active_experts must be non-decreasing"
            )
            raise ValueError(msg)
        slope = (active_val - prev_active) / float(tokens - prev_tokens)
        if (
            prev_slope is not None
            and slope > prev_slope + HULL_CONCAVITY_SLOPE_TOLERANCE
        ):
            msg = (
                f"Activation hull {curve_name!r} is not concave at index {idx}: "
                f"slope {slope:.6f} > previous slope {prev_slope:.6f}"
            )
            raise ValueError(msg)
        prev_tokens = tokens
        prev_active = active_val
        prev_slope = slope


@dataclass(frozen=True, slots=True)
class ExpertActivationModel:
    """Routed MoE active-expert envelope specification across regimes.

    Regimes:
      - ``"pigeonhole"``: strict lower bound assuming all tokens hit the same
        ``min(num_experts, experts_per_token)`` experts.
      - ``"bernoulli"``: uniform i.i.d. routing expectation (mean, not a lower
        bound; reported for diagnostic envelopes without pruning ``slo_verdict``).
      - ``"empirical_hull"``: trace-conditioned validated concave empirical
        lower hull (applies only to workloads at least as routing-diverse as the
        recorded trace); drives ``ruled_out`` SLO pruning and ordered TPOT
        ordering.
    """

    regime: ActivationRegime = "bernoulli"
    decode_token_grid: tuple[int, ...] = ()
    decode_active_experts: tuple[float, ...] = ()
    prefill_token_grid: tuple[int, ...] = ()
    prefill_active_experts: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        """Validate regime name and empirical hull curve geometry."""
        if self.regime not in _VALID_ACTIVATION_REGIMES:
            msg = f"Unsupported activation regime: {self.regime!r}"
            raise ValueError(msg)
        if self.regime == "empirical_hull":
            _validate_curve_geometry(
                self.decode_token_grid,
                self.decode_active_experts,
                "decode_hull",
            )
            _validate_curve_geometry(
                self.prefill_token_grid,
                self.prefill_active_experts,
                "prefill_hull",
            )

    @property
    def prunes_on_envelope(self) -> bool:
        """Return True when envelope metrics govern SLO pruning and layout order."""
        return self.regime == "empirical_hull"

    @property
    def matches_hard_floor(self) -> bool:
        """Return True when the envelope matches the strict pigeonhole floor."""
        return self.regime == "pigeonhole"


@dataclass(frozen=True, slots=True)
class ExpertDispatch:
    """Token dispatch dimensions for routed MoE per-shard expert occupancy."""

    num_experts: int
    experts_per_token: int
    dispatched_tokens: int
    ep_shards: int
    is_prefill: bool = False
    tokens_per_seq: int = 1


@lru_cache(maxsize=8192)
def bernoulli_expected_experts(
    num_experts: int,
    experts_per_token: int,
    dispatched_tokens: int,
) -> float:
    """Return expected distinct active experts under uniform i.i.d. routing."""
    if num_experts <= 0 or experts_per_token <= 0 or dispatched_tokens <= 0:
        return 0.0
    if experts_per_token >= num_experts:
        return float(num_experts)
    miss_prob = 1.0 - float(experts_per_token) / float(num_experts)
    return float(num_experts) * (1.0 - miss_prob**dispatched_tokens)


def _interpolate_concave_hull(
    token_grid: tuple[int, ...],
    active_experts: tuple[float, ...],
    token_count: int,
) -> float:
    """Lower-chord linear interpolation from (0, 0.0) through concave hull knots."""
    if token_count >= token_grid[-1]:
        return active_experts[-1]
    prev_tokens = 0
    prev_active = 0.0
    for grid_tokens, grid_active in zip(token_grid, active_experts, strict=True):
        if token_count <= grid_tokens:
            span = float(grid_tokens - prev_tokens)
            frac = float(token_count - prev_tokens) / span
            return prev_active + frac * (grid_active - prev_active)
        prev_tokens = grid_tokens
        prev_active = grid_active
    return active_experts[-1]


def active_experts_envelope(
    activation_model: ExpertActivationModel,
    dispatch: ExpertDispatch,
) -> float:
    """Return global active-expert count for the configured activation regime.

    Returns a strict lower bound under ``pigeonhole`` and ``empirical_hull``
    (applying set-union monotonicity across independent sequences when
    ``tokens_per_seq > 1`` during speculative verification), and the uniform
    i.i.d. routing mean under ``bernoulli``.
    """
    if (
        dispatch.num_experts <= 0
        or dispatch.experts_per_token <= 0
        or dispatch.dispatched_tokens <= 0
    ):
        return 0.0
    min_active = float(min(dispatch.num_experts, dispatch.experts_per_token))
    if activation_model.matches_hard_floor:
        return min_active
    if activation_model.regime == "bernoulli":
        expected = bernoulli_expected_experts(
            dispatch.num_experts,
            dispatch.experts_per_token,
            dispatch.dispatched_tokens,
        )
        return max(min_active, expected)
    effective_tokens = (
        max(1, dispatch.dispatched_tokens // dispatch.tokens_per_seq)
        if (not dispatch.is_prefill and dispatch.tokens_per_seq > 1)
        else dispatch.dispatched_tokens
    )
    grid = (
        activation_model.prefill_token_grid
        if dispatch.is_prefill
        else activation_model.decode_token_grid
    )
    values = (
        activation_model.prefill_active_experts
        if dispatch.is_prefill
        else activation_model.decode_active_experts
    )
    interpolated = _interpolate_concave_hull(grid, values, effective_tokens)
    return min(float(dispatch.num_experts), max(min_active, interpolated))


def _binomial_pmf(trials: int, successes: int, prob: float) -> float:
    """Compute binomial PMF P(X = successes) safely in log-space for large trials."""
    if prob <= 0.0:
        return 1.0 if successes == 0 else 0.0
    if prob >= 1.0:
        return 1.0 if successes == trials else 0.0
    if trials < 1025:
        return (
            math.comb(trials, successes)
            * (prob**successes)
            * ((1.0 - prob) ** (trials - successes))
        )
    log_comb = (
        math.lgamma(trials + 1)
        - math.lgamma(successes + 1)
        - math.lgamma(trials - successes + 1)
    )
    log_pmf = (
        log_comb + successes * math.log(prob) + (trials - successes) * math.log1p(-prob)
    )
    return math.exp(log_pmf)


@lru_cache(maxsize=8192)
def ep_busiest_experts(num_experts: int, top_k: int, rows: int, ranks: int) -> float:
    """Expected distinct routed experts on the busiest rank under uniform EP."""
    if num_experts <= 0 or top_k <= 0 or rows <= 0 or ranks <= 0:
        return 0.0
    if ranks == 1:
        return bernoulli_expected_experts(num_experts, top_k, rows)
    base_n = num_experts // ranks
    remainder = num_experts % ranks
    max_shard = base_n + (1 if remainder > 0 else 0)
    hit_prob = (
        1.0 if top_k >= num_experts else 1.0 - (1.0 - top_k / num_experts) ** rows
    )
    cdf_base = 0.0
    cdf_big = 0.0
    expected_max = 0.0
    for count in range(max_shard):
        if count < base_n:
            cdf_base += _binomial_pmf(base_n, count, hit_prob)
        else:
            cdf_base = 1.0
        if remainder > 0:
            big_n = base_n + 1
            cdf_big += _binomial_pmf(big_n, count, hit_prob)
        p_all_le_count = (
            (min(1.0, cdf_big) ** remainder)
            * (min(1.0, cdf_base) ** (ranks - remainder))
            if remainder > 0
            else min(1.0, cdf_base) ** ranks
        )
        expected_max += 1.0 - p_all_le_count
    pigeonhole_min = float(math.ceil(min(num_experts, top_k) / ranks))
    upper = float(min(max_shard, num_experts, rows * top_k))
    return min(upper, max(expected_max, pigeonhole_min))


def min_experts_per_ep_shard(
    activation_model: ExpertActivationModel,
    dispatch: ExpertDispatch,
) -> float:
    """Return busiest-shard active-expert envelope for one routed MoE dispatch.

    Uses ``max(ceil(min(E, K) / shards), total_active / shards)`` on pruning
    envelopes (``pigeonhole``, ``empirical_hull``) for Jensen lower-bound
    soundness, and ``ep_busiest_experts`` on the report-only ``bernoulli`` regime.
    """
    if (
        dispatch.num_experts <= 0
        or dispatch.experts_per_token <= 0
        or dispatch.dispatched_tokens <= 0
    ):
        return 0.0
    shards = max(1, dispatch.ep_shards)
    min_active = min(dispatch.num_experts, dispatch.experts_per_token)
    box_floor = float(math.ceil(min_active / shards))
    if activation_model.matches_hard_floor:
        return box_floor
    if activation_model.regime == "bernoulli":
        return max(
            box_floor,
            ep_busiest_experts(
                dispatch.num_experts,
                dispatch.experts_per_token,
                dispatch.dispatched_tokens,
                shards,
            ),
        )
    total_active = active_experts_envelope(activation_model, dispatch)
    return max(box_floor, total_active / float(shards))


def _require_exact_int(
    payload: Mapping[str, object],
    key: str,
    expected: int,
) -> None:
    """Assert that a JSON integer field matches expected without accepting bool."""
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"Activation hull field {key!r} must be an integer, got {value!r}"
        raise TypeError(msg)
    if value != expected:
        msg = (
            f"Activation hull field {key!r} mismatch: "
            f"file has {value}, model has {expected}"
        )
        raise ValueError(msg)


def _extract_curve_arrays(
    raw_curve: object,
    curve_name: str,
) -> tuple[list[int], list[float]]:
    """Extract JSON token_grid and active_experts lists and check element types."""
    if not isinstance(raw_curve, Mapping):
        msg = f"Activation hull {curve_name!r} must be a JSON object"
        raise TypeError(msg)
    raw_grid = raw_curve.get("token_grid")
    raw_active = raw_curve.get("active_experts")
    if (
        not isinstance(raw_grid, Sequence)
        or isinstance(raw_grid, (str, bytes))
        or not isinstance(raw_active, Sequence)
        or isinstance(raw_active, (str, bytes))
    ):
        msg = (
            f"Activation hull {curve_name!r} requires "
            "'token_grid' and 'active_experts' arrays"
        )
        raise TypeError(msg)
    grid: list[int] = []
    for item in raw_grid:
        if isinstance(item, bool) or not isinstance(item, int):
            msg = (
                f"Activation hull {curve_name!r} token_grid entries "
                "must be positive ints"
            )
            raise TypeError(msg)
        grid.append(item)
    active: list[float] = []
    for item in raw_active:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            msg = (
                f"Activation hull {curve_name!r} active_experts entries must be finite"
            )
            raise TypeError(msg)
        active.append(float(item))
    return (grid, active)


def _validate_hull_curve(
    raw_curve: object,
    curve_name: str,
    num_experts: int,
    experts_per_token: int,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Validate geometry, minimum expert floor, and Bernoulli cap for one hull."""
    grid, active = _extract_curve_arrays(raw_curve, curve_name)
    _validate_curve_geometry(grid, active, curve_name)
    min_active = float(min(num_experts, experts_per_token))
    for idx, (tokens, active_val) in enumerate(zip(grid, active, strict=True)):
        if active_val < min_active:
            msg = (
                f"Activation hull {curve_name!r} active_experts must be "
                f">= {min_active:g}"
            )
            raise ValueError(msg)
        bernoulli_cap = bernoulli_expected_experts(
            num_experts, experts_per_token, tokens
        )
        if active_val > bernoulli_cap + HULL_BERNOULLI_CAP_TOLERANCE:
            msg = (
                f"Activation hull {curve_name!r} point {idx} ({active_val:.3f}) "
                f"exceeds Bernoulli cap ({bernoulli_cap:.3f}) at {tokens} tokens"
            )
            raise ValueError(msg)
    return (tuple(grid), tuple(active))


def load_activation_hull(
    path: Path | str,
    geometry: ModelGeometry,
) -> ExpertActivationModel:
    """Load and validate an empirical concave MoE activation hull JSON file.

    Schema ``vllm_torchtpu.roofline.moe_activation_hull/v1`` contract:
      - ``schema_version``: must equal ``ACTIVATION_HULL_SCHEMA``.
      - ``num_experts``, ``experts_per_token``, ``moe_layers``: exact integers
        matching ``geom`` (guards against reusing a hull across models).
      - ``decode_hull``: lower concave hull of per-MoE-layer distinct active
        experts (unit: experts/layer) versus independent single-token decode
        sequence count (``token_grid``, unit: sequences). Multi-token verify
        steps evaluate at ``seq_count = dispatched_tokens // tokens_per_seq``
        via set-union monotonicity.
      - ``prefill_hull``: lower concave hull of per-MoE-layer distinct active
        experts (unit: experts/layer) versus prefill chunk token count
        (``token_grid``, unit: tokens).
      - Each knot must lie in ``[min(E, K), bernoulli_expected_experts(E, K, t)]``
        (with ``HULL_BERNOULLI_CAP_TOLERANCE`` for up to 6-decimal JSON exports)
        with non-increasing secant slopes from ``(0, 0.0)`` (skewed or i.i.d.
        routing; actively super-uniform balanced routers are rejected).
    """
    hull_path = Path(path)
    raw = json.loads(hull_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        msg = f"Activation hull file {hull_path} must contain a JSON object"
        raise TypeError(msg)
    schema = raw.get("schema_version")
    if schema != ACTIVATION_HULL_SCHEMA:
        msg = (
            f"Unsupported activation hull schema_version {schema!r}; "
            f"expected {ACTIVATION_HULL_SCHEMA!r}"
        )
        raise ValueError(msg)
    if geometry.num_experts <= 0 or geometry.top_k <= 0:
        msg = "Empirical MoE activation hull requires a routed MoE model"
        raise ValueError(msg)
    moe_layers = layer_ffn_kinds(geometry).count("moe")
    _require_exact_int(raw, "num_experts", geometry.num_experts)
    _require_exact_int(raw, "experts_per_token", geometry.top_k)
    _require_exact_int(raw, "moe_layers", moe_layers)
    decode_grid, decode_active = _validate_hull_curve(
        raw.get("decode_hull"),
        "decode_hull",
        geometry.num_experts,
        geometry.top_k,
    )
    prefill_grid, prefill_active = _validate_hull_curve(
        raw.get("prefill_hull"),
        "prefill_hull",
        geometry.num_experts,
        geometry.top_k,
    )
    return ExpertActivationModel(
        regime="empirical_hull",
        decode_token_grid=decode_grid,
        decode_active_experts=decode_active,
        prefill_token_grid=prefill_grid,
        prefill_active_experts=prefill_active,
    )


def parse_activation_spec(
    spec: str,
    geometry: ModelGeometry,
) -> ExpertActivationModel:
    """Parse CLI --moe-activation spec into an ExpertActivationModel."""
    cleaned = spec.strip()
    if cleaned == "pigeonhole":
        return ExpertActivationModel(regime="pigeonhole")
    if cleaned == "bernoulli":
        return ExpertActivationModel(regime="bernoulli")
    if cleaned.startswith(_HULL_SPEC_PREFIX):
        hull_path_str = cleaned[len(_HULL_SPEC_PREFIX) :].strip()
        if not hull_path_str:
            msg = "--moe-activation hull:<path> requires a non-empty file path"
            raise ValueError(msg)
        return load_activation_hull(hull_path_str, geometry)
    msg = (
        f"Unsupported --moe-activation spec {spec!r}; "
        "expected 'pigeonhole', 'bernoulli', or 'hull:<path>'"
    )
    raise ValueError(msg)
