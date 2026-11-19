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
"""Cross-layer reuse of the MSA block selection on decode steps (IndexCache,
arXiv 2603.12201), set by ``TPU_MINIMAX_M3_MSA_INDEX_REUSE_DECODE``.

On a decode-sized step, "Full" layers run the lightning indexer, the local
top-k and the DCP candidate merge; "Shared" layers write their K/V (and
index keys, which later prefill steps read) and attend over the selection
of the nearest Full layer before them in the same step (per KV head: head
``h`` reuses head ``h``'s blocks). Lossy: a Shared layer's own index
queries are ignored. Layer 0 is always Full (it seeds the selection).
Prefill-sized steps run every layer Full.

A step is decode-sized when it has at most ``max_num_seqs * (1 + spec
tokens)`` tokens (padded shapes, so each compiled bucket has one kind).

Pattern grammar:

- ``""`` or ``"1"``: every layer Full (off).
- ``"N"`` / ``"stride:N"``: Full layers ``0, N, 2N, ...``.
- ``"full:0,1,5"``: these layers Full (must include 0).
- ``"FFSF..."``: one letter per layer (the paper's ``index_topk_pattern``).
"""

import dataclasses

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

ENV = "TPU_MINIMAX_M3_MSA_INDEX_REUSE_DECODE"


def parse_pattern(spec: str, num_layers: int) -> frozenset[int]:
    """The Full layers of pattern `spec` over `num_layers` layers."""
    raw = spec.strip()
    if not raw:
        return frozenset(range(num_layers))
    try:
        if raw.startswith("stride:") or raw.isdigit():
            stride = int(raw.removeprefix("stride:"))
            if stride <= 0:
                raise ValueError
            return frozenset(range(0, num_layers, stride))
        if raw.startswith("full:"):
            full = frozenset(
                int(x) for x in raw.removeprefix("full:").split(",") if x.strip()
            )
        elif set(raw.upper()) <= {"F", "S"}:
            if len(raw) != num_layers:
                raise ValueError(f"{len(raw)} letters for {num_layers} layers")
            full = frozenset(i for i, c in enumerate(raw.upper()) if c == "F")
        else:
            raise ValueError
    except ValueError as e:
        detail = f" ({e})" if str(e) else ""
        raise ValueError(
            f"{ENV}: bad layer pattern {spec!r}{detail}; expected 'N', "
            "'stride:N', 'full:0,a,b,...' or an F/S string"
        ) from None
    if 0 not in full:
        raise ValueError(
            f"{ENV}: layer 0 must be Full (it seeds the selection), got {spec!r}"
        )
    if not all(0 <= i < num_layers for i in full):
        raise ValueError(f"{ENV}: layers out of range [0, {num_layers}) in {spec!r}")
    return full


def pattern_string(full: frozenset[int], num_layers: int) -> str:
    return "".join("F" if i in full else "S" for i in range(num_layers))


@dataclasses.dataclass(frozen=True)
class IndexReusePlan:
    """The Full layers of decode-sized steps."""

    num_layers: int
    full_decode: frozenset[int]

    @property
    def active(self) -> bool:
        return self.full_decode != frozenset(range(self.num_layers))

    def is_full(self, layer: int) -> bool:
        return layer in self.full_decode


_logged: set[tuple] = set()


def plan_from_env(num_layers: int, spec: str | None = None) -> IndexReusePlan | None:
    """The plan of ``TPU_MINIMAX_M3_MSA_INDEX_REUSE_DECODE`` (or `spec`), or
    None when off. Logged once per pattern."""
    if spec is None:
        spec = envs.TPU_MINIMAX_M3_MSA_INDEX_REUSE_DECODE
    plan = IndexReusePlan(num_layers, parse_pattern(spec, num_layers))
    if not plan.active:
        return None
    key = (num_layers, plan.full_decode)
    if key not in _logged:
        _logged.add(key)
        logger.info(
            "MiniMax-M3 MSA index reuse on (%s=%r): decode steps %d/%d Full "
            "layers %s; prefill steps unchanged",
            ENV,
            spec,
            len(plan.full_decode),
            num_layers,
            pattern_string(plan.full_decode, num_layers),
        )
    return plan


def is_decode_step(num_tokens: int, max_num_seqs: int, decode_rows: int) -> bool:
    """Whether a step of `num_tokens` (padded) rows over `max_num_seqs`
    (padded) request slots is decode-sized: static per compiled bucket."""
    return num_tokens <= max_num_seqs * decode_rows
