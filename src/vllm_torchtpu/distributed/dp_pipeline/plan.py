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
"""Which engines run which decoder layers of one pipelined prefill."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vllm_torchtpu import envs

# SamplingParams.extra_args key holding a pipelined request's plan.
PLAN_KEY = "tpu_dp_pipeline"

# A plan id packs the API server's id and a counter; a chunk token packs the
# plan id and the chunk's first token position.
_SERVER_BITS = 16
_COUNTER_BITS = 20
_POSITION_BITS = 20


def server_id_of(pid: int) -> int:
    """The id the API server process ``pid`` names its plans and rings by."""
    return pid % (1 << _SERVER_BITS)


def make_pipeline_id(server_id: int, counter: int) -> int:
    """A nonzero plan id; ``counter`` wraps within its bits."""
    if not 0 <= server_id < 1 << _SERVER_BITS:
        raise ValueError(f"server id out of range: {server_id}")
    return server_id << _COUNTER_BITS | counter % ((1 << _COUNTER_BITS) - 1) + 1


@dataclass(frozen=True)
class PipelinePlan:
    """One prompt's pipeline: stage ``s`` runs on DP engine ``ranks[s]`` and
    computes layers ``[bounds[s], bounds[s + 1])`` of the prompt's first
    ``tokens`` tokens (``pipelined_tokens``); ``ranks[-1]`` owns the request
    and computes any rest of the prompt alone. With ``kv`` the helpers send
    the owner their layers' K/V (``kv_plan``), so the owner can go on alone."""

    pipeline_id: int
    ranks: tuple[int, ...]
    bounds: tuple[int, ...]
    tokens: int
    kv: bool = False

    def __post_init__(self) -> None:
        if len(self.ranks) < 2 or len(set(self.ranks)) != len(self.ranks):
            raise ValueError(f"a pipeline needs >= 2 distinct engines: {self.ranks}")
        if self.tokens <= 0:
            raise ValueError(f"a pipeline computes some tokens, not {self.tokens}")
        if len(self.bounds) != len(self.ranks) + 1 or self.bounds[0] != 0:
            raise ValueError(f"bounds {self.bounds} do not fit stages {self.ranks}")
        if any(a >= b for a, b in zip(self.bounds, self.bounds[1:])):
            raise ValueError(f"layer bounds must increase: {self.bounds}")
        if not 0 < self.pipeline_id < 1 << (63 - _POSITION_BITS):
            raise ValueError(f"pipeline id out of range: {self.pipeline_id}")

    @property
    def num_stages(self) -> int:
        return len(self.ranks)

    @property
    def server_id(self) -> int:
        """The API server that made the plan; its hand-off rings are its own."""
        return self.pipeline_id >> _COUNTER_BITS

    def stage_of(self, dp_rank: int) -> int:
        return self.ranks.index(dp_rank)

    def layers(self, stage: int) -> tuple[int, int]:
        return self.bounds[stage], self.bounds[stage + 1]

    def chunk_token(self, first_position: int) -> int:
        """Identifies one chunk of this prompt on every stage."""
        if not 0 <= first_position < 1 << _POSITION_BITS:
            raise ValueError(f"chunk position out of range: {first_position}")
        return self.pipeline_id << _POSITION_BITS | first_position

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.pipeline_id,
            "ranks": list(self.ranks),
            "bounds": list(self.bounds),
            "tokens": self.tokens,
            "kv": self.kv,
        }

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> PipelinePlan:
        return cls(
            int(value["id"]),
            tuple(value["ranks"]),
            tuple(value["bounds"]),
            int(value["tokens"]),
            bool(value["kv"]),
        )


def pipelined_tokens(num_tokens: int, chunk: int, sizes: list[int]) -> int:
    """The prompt tokens a pipeline computes: its full ``chunk``-token chunks,
    and a final partial chunk only when it pads to the full chunk's bucket
    (``sizes``: the token buckets), the one bucket the layer segments are
    compiled at; the owner computes a smaller rest as an ordinary step.

    A segment program at another bucket need not reproduce the whole-model
    program's numerics there (on the 31B, bucket 2048 did not), while every
    full chunk of every pipelined prompt so far has matched at the full one.
    """
    full = num_tokens - num_tokens % chunk
    rest = num_tokens - full
    if rest and min(s for s in sizes if s >= rest) == chunk:
        return num_tokens
    return full


def plan_of(sampling_params: Any) -> PipelinePlan | None:
    """The plan a request carries, or None for an ordinary request."""
    extra = getattr(sampling_params, "extra_args", None) or {}
    value = extra.get(PLAN_KEY)
    return None if value is None else PipelinePlan.from_json(value)


def is_pipelined(sampling_params: Any) -> bool:
    """Whether a request carries a plan, without parsing it."""
    return PLAN_KEY in (getattr(sampling_params, "extra_args", None) or {})


def _parse_per_stages(name: str, value: str) -> dict[int, list[int]]:
    """``S:n1,n2,...[;...]`` (the format of ``name``): per stage count S, its
    integers."""
    entries: dict[int, list[int]] = {}
    for part in filter(None, (p.strip() for p in value.split(";"))):
        stages, colon, numbers = part.partition(":")
        try:
            num_stages, parsed = int(stages), [int(n) for n in numbers.split(",")]
        except ValueError as exc:
            raise ValueError(f"malformed {name} entry {part!r}") from exc
        if not colon or num_stages in entries:
            raise ValueError(f"malformed or repeated {name} entry {part!r}")
        entries[num_stages] = parsed
    return entries


def parse_splits(value: str) -> dict[int, tuple[int, ...]]:
    """``S:b1,...,b(S-1)[;...]``: for an S-stage pipeline, the layers where
    stages change. Raises on a malformed value."""
    splits: dict[int, tuple[int, ...]] = {}
    for num_stages, cuts in _parse_per_stages("TPU_DP_PIPELINE_SPLITS", value).items():
        if num_stages < 2 or len(cuts) != num_stages - 1:
            raise ValueError(
                f"TPU_DP_PIPELINE_SPLITS entry for {num_stages} stages needs S-1 cuts"
            )
        if any(a >= b for a, b in zip(cuts, cuts[1:])) or cuts[0] <= 0:
            raise ValueError(f"TPU_DP_PIPELINE_SPLITS cuts must increase: {cuts}")
        splits[num_stages] = tuple(cuts)
    return splits


def parse_stage_tokens(value: str) -> dict[int, int]:
    """``S:tokens[;...]``: per stage count S > 2, the shortest prompt split
    into S stages; the thresholds rise with S. Raises on a malformed value."""
    name = "TPU_DP_PIPELINE_STAGE_TOKENS"
    thresholds: dict[int, int] = {}
    for num_stages, numbers in sorted(_parse_per_stages(name, value).items()):
        if num_stages < 3 or len(numbers) != 1 or numbers[0] <= 0:
            raise ValueError(f"{name}: one token count > 0 per stage count S >= 3")
        if any(numbers[0] < t for t in thresholds.values()):
            raise ValueError(f"{name} thresholds must rise with S: {value!r}")
        thresholds[num_stages] = numbers[0]
    return thresholds


def max_stages(num_tokens: int) -> int:
    """The most stages a prompt of ``num_tokens`` tokens is split into: every
    stage adds a hand-off to the pipeline's fill, which a short prompt does
    not earn back (TPU_DP_PIPELINE_STAGE_TOKENS)."""
    thresholds = parse_stage_tokens(envs.TPU_DP_PIPELINE_STAGE_TOKENS)
    stages = envs.TPU_DP_PIPELINE_STAGES
    for num_stages, tokens in thresholds.items():
        if num_tokens < tokens:
            stages = min(stages, num_stages - 1)
    return stages


def stage_bounds(num_stages: int, num_layers: int) -> tuple[int, ...]:
    """Layer boundaries of an ``num_stages``-stage pipeline, ends included."""
    cuts = parse_splits(envs.TPU_DP_PIPELINE_SPLITS).get(num_stages)
    if cuts is None:
        raise ValueError(f"TPU_DP_PIPELINE_SPLITS has no {num_stages}-stage split")
    if cuts[-1] >= num_layers:
        raise ValueError(
            f"TPU_DP_PIPELINE_SPLITS {cuts} does not fit {num_layers} layers (the "
            "default fits Gemma4-31B's 60)"
        )
    return (0, *cuts, num_layers)


def segment_bounds(num_layers: int) -> tuple[int, ...]:
    """Boundaries of the compiled layer segments: every cut of every split up
    to TPU_DP_PIPELINE_STAGES stages, so each stage is a run of segments."""
    cuts: set[int] = set()
    for num_stages in range(2, envs.TPU_DP_PIPELINE_STAGES + 1):
        cuts.update(stage_bounds(num_stages, num_layers)[1:-1])
    return (0, *sorted(cuts), num_layers)
