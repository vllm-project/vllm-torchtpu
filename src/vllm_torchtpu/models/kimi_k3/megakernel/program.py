# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
# src/vllm_torchtpu/megakernel/kimi_k3/program.py
"""Kimi-K3 as a megakernel program for prefill and decode roles.

The K3 programs have no vLLM dependencies; ``megakernel.runner`` builds them
from vLLM's configuration and the standalone tools build them directly.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING

from .layout import cache_layout
from .runtime_shim.program import Capabilities

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import ModuleType

    from . import decode, prefill
    from .runtime_shim.config import MegakernelEngineConfig

TP_SIZE = 32
DCP_SIZE = 32
HIDDEN_SIZE = 7168
VOCAB = 163840
DRAFT_TOKENS = 7
SHORT_DRAFT_TOKENS = 3
MAX_WIDE_DRAFT_BATCH = 8
MAX_DP_GROUPS = 2
DRAFT_POS_SLACK = 16
SPEC_BATCHES = (1, 2, 4, 8, 16)
DP_SPEC_BATCHES = (2, 4, 8, 16, 32)
MAX_STEP_TOKENS = 8192
SAMPLING = frozenset({"temperature", "top_p", "top_k", "seed"})

_SPEC_SUPPORT = (
    f"{SHORT_DRAFT_TOKENS} or {DRAFT_TOKENS} drafts at batches "
    f"1-{MAX_WIDE_DRAFT_BATCH}, and {SHORT_DRAFT_TOKENS} at batches "
    f"9-{max(SPEC_BATCHES)} (attention DP2: 9-{max(DP_SPEC_BATCHES)}) through "
    f"the compiled speculation batches {SPEC_BATCHES} (attention DP2: "
    f"{DP_SPEC_BATCHES}; up to --max-num-seqs), 0 anywhere"
)


def kernel_modules(*names: str) -> list[ModuleType]:
    """Import ``vllm_torchtpu.models.kimi_k3.megakernel.kernels.<name>``."""
    return [
        importlib.import_module(
            f"vllm_torchtpu.models.kimi_k3.megakernel.kernels.{name}"
        )
        for name in names
    ]


def supports_spec_tokens(batch: int, drafts: int, attention_dp: int = 1) -> bool:
    """Return True if ``drafts`` speculative width is supported at ``batch``."""
    largest = max(DP_SPEC_BATCHES if attention_dp > 1 else SPEC_BATCHES)
    return drafts == 0 or (
        1 <= batch <= largest
        and (
            drafts == SHORT_DRAFT_TOKENS
            or (batch <= MAX_WIDE_DRAFT_BATCH and drafts == DRAFT_TOKENS)
        )
    )


def _validate_spec_entries(entries: list[tuple[int, int, int]]) -> None:
    """Validate sorted speculative schedule ``(min_batch, max_batch, drafts)`` rows."""
    previous_end = 0
    for lo, hi, k in entries:
        if lo <= 0 or hi < lo or k < 0:
            msg = (
                "num_speculative_tokens_per_batch_size entries are "
                "(min_batch, max_batch, drafts) with 1 <= min <= max and "
                f"drafts >= 0, not ({lo}, {hi}, {k})"
            )
            raise NotImplementedError(msg)
        if lo <= previous_end:
            msg = (
                "num_speculative_tokens_per_batch_size ranges must be "
                f"non-overlapping: ({lo}, {hi}, {k})"
            )
            raise NotImplementedError(msg)
        previous_end = hi
    if entries[0][0] != 1:
        msg = (
            "num_speculative_tokens_per_batch_size's first range must "
            f"start at batch size 1: {entries[0]}"
        )
        raise NotImplementedError(msg)


def _expand_spec_schedule(
    schedule: Sequence[tuple[int, int, int]] | None,
    num_spec: int,
    max_seqs: int,
) -> list[int]:
    """Expand a sparse speculative schedule into a dense ``max_seqs + 1`` table."""
    dense = [0] * (max_seqs + 1)
    if schedule is None:
        for b in range(1, max_seqs + 1):
            dense[b] = num_spec
        return dense
    entries = sorted((int(lo), int(hi), int(k)) for lo, hi, k in schedule)
    _validate_spec_entries(entries)
    next_b, last_k = 1, 0
    for lo, hi, k in entries:
        for b in range(next_b, min(lo, max_seqs + 1)):
            dense[b] = min(num_spec, last_k)
        for b in range(max(lo, next_b), min(hi, max_seqs) + 1):
            dense[b] = min(num_spec, k)
        next_b, last_k = max(next_b, hi + 1), k
        if next_b > max_seqs:
            break
    for b in range(next_b, max_seqs + 1):
        dense[b] = min(num_spec, last_k)
    return dense


def spec_tokens_per_batch(config: MegakernelEngineConfig) -> tuple[int, ...] | None:
    """Return dense speculative draft counts per batch size, or None."""
    if config.role != "decode" or not config.num_speculative_tokens:
        return None
    schedule = config.num_speculative_tokens_per_batch_size
    num_spec, max_seqs = config.num_speculative_tokens, config.max_num_seqs
    dense = _expand_spec_schedule(schedule, num_spec, max_seqs)
    for b in range(1, max_seqs + 1):
        k = dense[b]
        if k == 0:
            continue
        supported = supports_spec_tokens(b, k, config.attention_dp)
        if b > 1 and supported and config.attention_dp == 1:
            supported = any(b <= s <= max_seqs for s in SPEC_BATCHES)
        if not supported:
            given = (
                "None: num_speculative_tokens everywhere"
                if schedule is None
                else str(schedule)
            )
            msg = (
                f"DSpark on the K3 decoder supports {_SPEC_SUPPORT}; the "
                f"schedule gives {k} drafts at batch size {b} "
                f"(num_speculative_tokens_per_batch_size={given}; "
                f"--max-num-seqs {max_seqs})"
            )
            raise NotImplementedError(msg)
    return tuple(dense)


def recurrent_state_dtype(cache_dtype: str) -> str:
    """Resolve decode KDA recurrent bank storage dtype from ``cache_dtype``."""
    if cache_dtype not in ("auto", "float32", "bfloat16"):
        msg = (
            "K3 megakernel --mamba-ssm-cache-dtype supports auto, float32, "
            f"or bfloat16, not {cache_dtype!r}"
        )
        raise ValueError(msg)
    return "float32" if cache_dtype == "auto" else cache_dtype


def _check_parallel_and_cache(config: MegakernelEngineConfig) -> None:
    """Validate role, TP, attention DP, DCP, and block-size configuration."""
    recurrent_state_dtype(config.mamba_ssm_cache_dtype)
    if config.role == "prefill" and config.mamba_ssm_cache_dtype == "bfloat16":
        msg = (
            "K3 megakernel --mamba-ssm-cache-dtype bfloat16 is decode-only; "
            "the prefill worker requires auto or float32 for its FP32 state "
            "and transfer format"
        )
        raise NotImplementedError(msg)
    if config.role not in ("prefill", "decode"):
        msg = f"the K3 megakernel serves one role of a P/D pair, not {config.role!r}"
        raise NotImplementedError(msg)
    if config.tensor_parallel_size != TP_SIZE:
        msg = f"the K3 megakernel needs --tensor-parallel-size {TP_SIZE}"
        raise NotImplementedError(msg)
    if config.attention_dp not in (1, MAX_DP_GROUPS) or (
        config.attention_dp > 1 and config.role != "decode"
    ):
        msg = "the K3 decoder runs attention_dp 1 or 2; the prefill runs 1"
        raise NotImplementedError(msg)
    dcp = DCP_SIZE // config.attention_dp
    if config.decode_context_parallel_size != dcp:
        suffix = f" under attention_dp {config.attention_dp}" if dcp != DCP_SIZE else ""
        msg = f"the K3 megakernel needs --decode-context-parallel-size {dcp}{suffix}"
        raise NotImplementedError(msg)
    layout = cache_layout(config.kv_cache_dtype)
    if config.block_size != layout.tokens_per_block:
        msg = (
            f"the K3 megakernel with cache dtype {config.kv_cache_dtype} requires "
            f"--block-size {layout.tokens_per_block}"
        )
        raise NotImplementedError(msg)
    if config.cp_kv_cache_interleave_size not in (None, layout.tokens_per_block):
        msg = (
            "the K3 megakernel requires --cp-kv-cache-interleave-size "
            f"{layout.tokens_per_block} with cache dtype {config.kv_cache_dtype}"
        )
        raise NotImplementedError(msg)


def check(config: MegakernelEngineConfig) -> None:
    """Raise ``NotImplementedError`` for an unsupported K3 configuration."""
    _check_parallel_and_cache(config)
    if config.role == "prefill" and config.max_num_batched_tokens > MAX_STEP_TOKENS:
        msg = (
            f"the K3 prefill takes at most {MAX_STEP_TOKENS} tokens per step: "
            f"--max-num-batched-tokens <= {MAX_STEP_TOKENS}"
        )
        raise NotImplementedError(msg)
    if config.role == "decode" and config.num_speculative_tokens not in (
        0,
        SHORT_DRAFT_TOKENS,
        DRAFT_TOKENS,
    ):
        msg = (
            "DSpark on the K3 decoder supports "
            f"num_speculative_tokens=0, {SHORT_DRAFT_TOKENS} or {DRAFT_TOKENS}"
        )
        raise NotImplementedError(msg)
    dense = spec_tokens_per_batch(config)
    if config.synthetic_acceptance_rates_by_width is not None:
        by_width = dict(config.synthetic_acceptance_rates_by_width)
        missing = sorted({k for k in dense or () if k} - set(by_width))
        if missing:
            msg = (
                f"synthetic_acceptance_lengths gives no length for the {missing} "
                f"draft widths the schedule runs (it has {sorted(by_width)})"
            )
            raise NotImplementedError(msg)
        own = by_width.get(config.num_speculative_tokens)
        if own is not None and own != config.synthetic_acceptance_rates:
            msg = (
                f"synthetic_acceptance_lengths' {config.num_speculative_tokens}-draft "
                f"rates {list(own)} differ from --speculative-config's "
                f"{list(config.synthetic_acceptance_rates or ())}"
            )
            raise NotImplementedError(msg)
    if config.role == "decode" and not config.async_scheduling:
        msg = (
            "the K3 decoder needs async scheduling (it keeps each request's next "
            "input on the device)"
        )
        raise NotImplementedError(msg)


def capabilities(config: MegakernelEngineConfig) -> Capabilities:
    """Return engine ``Capabilities`` supported by ``config``."""
    check(config)
    dense = spec_tokens_per_batch(config)
    speculating = dense is not None and any(dense)
    return Capabilities(
        roles=frozenset({config.role}),
        sampling=SAMPLING,
        speculation="dspark" if speculating else None,
        num_speculative_tokens=config.num_speculative_tokens if speculating else 0,
        max_step_tokens=MAX_STEP_TOKENS
        if config.role == "prefill"
        else config.max_num_batched_tokens,
        page_aligned_chunks=config.role == "prefill",
        block_copies=config.role == "prefill",
        structured_outputs=True,
    )


def build(
    config: MegakernelEngineConfig,
    presharded: Path,
    draft_checkpoint: Path | None = None,
    *,
    tp_rank: int,
) -> prefill.KimiPrefiller | decode.KimiDecoder:
    """Build the program for ``config.role`` on TP rank ``tp_rank``."""
    check(config)
    if config.role == "prefill":
        prefill_mod = importlib.import_module(
            "vllm_torchtpu.models.kimi_k3.megakernel.prefill"
        )
        return prefill_mod.KimiPrefiller(
            config,
            presharded,
            draft_checkpoint,
            tp_rank=tp_rank,
        )
    dense = spec_tokens_per_batch(config)
    drafter = None
    if draft_checkpoint is not None and dense is not None and any(dense):
        draft_mod = importlib.import_module(
            "vllm_torchtpu.models.kimi_k3.megakernel.draft"
        )
        drafter = draft_mod.draft_class(draft_checkpoint)(
            max_position=config.max_model_len + DRAFT_POS_SLACK,
            checkpoint=str(draft_checkpoint),
            presharded=Path(presharded),
            synthetic_rates=config.synthetic_acceptance_rates,
            synthetic_rates_by_width=dict(config.synthetic_acceptance_rates_by_width)
            if config.synthetic_acceptance_rates_by_width is not None
            else None,
        )
    sub_base = 2 * config.max_num_seqs + 1
    decode_ops_mod = importlib.import_module(
        "vllm_torchtpu.models.kimi_k3.megakernel.decode_ops"
    )
    decode_mod = importlib.import_module(
        "vllm_torchtpu.models.kimi_k3.megakernel.decode"
    )
    ops = decode_ops_mod.DecodeOps(
        presharded,
        num_slots=decode_ops_mod.Control.total_slots(sub_base) if drafter else sub_base,
        batch_sizes=decode_ops_mod.decode_batch_sizes(config.max_num_seqs),
        mamba_ssm_cache_dtype=config.mamba_ssm_cache_dtype,
    )
    return decode_mod.KimiDecoder(config, ops, drafter, tp_rank=tp_rank)
