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
"""Cross-layer reuse of the MSA block selection (IndexCache, arXiv
2603.12201), set from `--additional-config` keys.

"Full" layers run the lightning indexer, the top-k candidates and their DCP
all-gather and merge; "Shared" layers write their K/V and attend over the
merged selection of the nearest Full layer before them in the same step
(per KV head: head ``h`` reuses head ``h``'s blocks). Lossy: a Shared layer's
own index queries are ignored. Layer 0 is always Full (it seeds the
selection).

Keys (`vllm_torchtpu.opt_in_config`, all default off):

- ``minimax_msa_index_reuse``: the pattern of both step kinds.
- ``minimax_msa_index_reuse_prefill`` / ``..._decode``: the pattern of
  prefill-sized / decode-sized steps (overrides the first; "" inherits it).
  A step is decode-sized when it has at most ``max_num_seqs * (1 + spec
  tokens)`` tokens (a static shape, so each compiled bucket has one kind).
- ``minimax_msa_index_reuse_skip_index_write``: layers Shared in both kinds
  never read their index keys, so they do not write them either.

Pattern grammar:

- ``""`` or ``"1"``: every layer Full (off).
- ``"N"`` / ``"stride:N"``: Full layers ``0, N, 2N, ...``.
- ``"full:0,1,5"``: these layers Full (must include 0).
- ``"FFSF..."``: one letter per layer (the paper's ``index_topk_pattern``).
"""

import dataclasses
from typing import Any

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.opt_in_config import option

logger = init_logger(__name__)

KEY = "minimax_msa_index_reuse"
PREFILL_KEY = "minimax_msa_index_reuse_prefill"
DECODE_KEY = "minimax_msa_index_reuse_decode"
SKIP_INDEX_WRITE_KEY = "minimax_msa_index_reuse_skip_index_write"

_logged: set[tuple] = set()


def parse_pattern(spec: str, num_layers: int, key: str = KEY) -> frozenset[int]:
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
            f"additional_config.{key}: bad layer pattern {spec!r}{detail}; "
            "expected 'N', 'stride:N', 'full:0,a,b,...' or an F/S string"
        ) from None
    if 0 not in full:
        raise ValueError(
            f"additional_config.{key}: layer 0 must be Full (it seeds the "
            f"selection), got {spec!r}"
        )
    if not all(0 <= i < num_layers for i in full):
        raise ValueError(
            f"additional_config.{key}: layers out of range [0, {num_layers}) "
            f"in {spec!r}"
        )
    return full


@dataclasses.dataclass(frozen=True)
class IndexReusePlan:
    num_layers: int
    full_prefill: frozenset[int]
    full_decode: frozenset[int]
    skip_index_write: bool = False

    @property
    def active(self) -> bool:
        every = frozenset(range(self.num_layers))
        return self.full_prefill != every or self.full_decode != every

    def is_full(self, layer: int, prefill: bool) -> bool:
        return layer in (self.full_prefill if prefill else self.full_decode)

    def writes_index(self, layer: int) -> bool:
        """Whether `layer` must keep writing its index keys: it scores in
        some step kind (or the skip key is off)."""
        if not self.skip_index_write:
            return True
        return layer in self.full_prefill or layer in self.full_decode

    @staticmethod
    def pattern_string(full: frozenset[int], num_layers: int) -> str:
        return "".join("F" if i in full else "S" for i in range(num_layers))


def plan_from_config(additional_config: Any, num_layers: int) -> IndexReusePlan:
    """The plan of the keys in `additional_config` (logged once when on)."""
    base = option(additional_config, KEY)
    prefill = option(additional_config, PREFILL_KEY) or base
    decode = option(additional_config, DECODE_KEY) or base
    plan = IndexReusePlan(
        num_layers,
        parse_pattern(prefill, num_layers, PREFILL_KEY if prefill != base else KEY),
        parse_pattern(decode, num_layers, DECODE_KEY if decode != base else KEY),
        bool(option(additional_config, SKIP_INDEX_WRITE_KEY)),
    )
    if plan.skip_index_write and not plan.active:
        logger.warning(
            "additional_config.%s without an index reuse pattern: no effect.",
            SKIP_INDEX_WRITE_KEY,
        )
    log_key = (plan.full_prefill, plan.full_decode, plan.skip_index_write)
    if plan.active and log_key not in _logged:
        _logged.add(log_key)
        logger.info(
            "MiniMax-M3 MSA index reuse on (additional_config %s=%r %s=%r "
            "%s=%r %s=%r): prefill %d/%d Full layers %s, decode %d/%d Full "
            "layers %s; index-key writes skipped in %d layers",
            KEY,
            base,
            PREFILL_KEY,
            option(additional_config, PREFILL_KEY),
            DECODE_KEY,
            option(additional_config, DECODE_KEY),
            SKIP_INDEX_WRITE_KEY,
            plan.skip_index_write,
            len(plan.full_prefill),
            num_layers,
            plan.pattern_string(plan.full_prefill, num_layers),
            len(plan.full_decode),
            num_layers,
            plan.pattern_string(plan.full_decode, num_layers),
            sum(not plan.writes_index(i) for i in range(num_layers)),
        )
    return plan
