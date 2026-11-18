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
#   tests/megakernel/test_structured_outputs.py
"""Tests for sampling mask_logits unpacking, grammar masking, and sharding."""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.models.kimi_k3.megakernel.common.constants import BITS_PER_WORD

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from vllm_torchtpu.models.kimi_k3.megakernel.decode_ops import DecodeOps

from vllm_torchtpu.models.kimi_k3.megakernel import decode
from vllm_torchtpu.models.kimi_k3.megakernel.decode import (
    CompileCall,
    Control,
    KimiDecoder,
)
from vllm_torchtpu.models.kimi_k3.megakernel.program import TP_SIZE, VOCAB
from vllm_torchtpu.models.kimi_k3.megakernel.runtime_shim.config import (
    MegakernelEngineConfig,
)
from vllm_torchtpu.models.kimi_k3.megakernel.sampling import (
    AXIS,
    apply_grammar_mask,
    mask_logits,
)

_NEG_INF = -1e30
_INT32_BOUND = 2**31
VOCAB_WORDS = VOCAB // BITS_PER_WORD
_EXPECTED_PAGES = 16
_NUM_SPEC_TOKENS = 7


def test_mask_logits_unpacking_and_sharding() -> None:
    """Unpack bitmask shards per TP rank and set disallowed logits to -1e30."""
    width = VOCAB // TP_SIZE
    rows = 3
    rng = np.random.default_rng(0)
    mask = rng.integers(
        -_INT32_BOUND,
        _INT32_BOUND,
        (rows, VOCAB_WORDS),
        dtype=np.int32,
    )
    logits = rng.normal(size=(rows, width)).astype(np.float32)

    masked = jax.jit(mask_logits)
    for rank in (0, 1, 3, TP_SIZE - 1):
        out = np.asarray(masked(jnp.asarray(logits), jnp.asarray(mask), rank))
        words = mask[
            :,
            rank * width // BITS_PER_WORD : (rank + 1) * width // BITS_PER_WORD,
        ]
        bit_shifts = np.arange(BITS_PER_WORD, dtype=np.int32)
        allowed = ((words[:, :, None] >> bit_shifts) & 1) != 0
        allowed = allowed.reshape(rows, width)
        assert bool(((out == _NEG_INF) == ~allowed).all())
        np.testing.assert_array_equal(out[allowed], logits[allowed])


def test_mask_logits_all_ones_is_identity_and_finite_when_rank_empty() -> None:
    """Preserve finite logits unchanged when all-ones, and mask out other ranks."""
    width = 163840 // TP_SIZE
    logits = np.random.default_rng(1).normal(size=(2, width)).astype(np.float32)
    masked = jax.jit(mask_logits)
    ones = np.full((2, VOCAB_WORDS), -1, dtype=np.int32)
    for rank in (0, TP_SIZE - 1):
        np.testing.assert_array_equal(
            np.asarray(masked(jnp.asarray(logits), jnp.asarray(ones), rank)), logits
        )
    # A schema allowing only rank 3's tokens fully masks every other rank:
    # finite sentinel, never NaN or -inf (the per-rank normalizers stay sane).
    narrow = np.zeros((2, VOCAB_WORDS), np.int32)
    narrow[:, 3 * width // 32 : 4 * width // 32] = -1
    out = np.asarray(masked(jnp.asarray(logits), jnp.asarray(narrow), 0))
    assert (out == _NEG_INF).all() and np.isfinite(out).all()
    out = np.asarray(masked(jnp.asarray(logits), jnp.asarray(narrow), 3))
    np.testing.assert_array_equal(out, logits)


def test_apply_grammar_mask_sharded_round_trip() -> None:
    """Verify apply_grammar_mask under shard_map matches rank-0 mask_logits."""
    width = VOCAB // TP_SIZE
    rows = 2
    rng = np.random.default_rng(7)
    mask = rng.integers(
        -_INT32_BOUND,
        _INT32_BOUND,
        (rows, VOCAB_WORDS),
        dtype=np.int32,
    )
    logits = rng.normal(size=(rows, width)).astype(np.float32)
    mesh = Mesh(np.array([jax.devices()[0]]), (AXIS,))
    sharded_fn = jax.shard_map(
        apply_grammar_mask,
        mesh=mesh,
        in_specs=(P(AXIS), P()),
        out_specs=P(AXIS),
    )
    out = np.asarray(sharded_fn(jnp.asarray(logits), jnp.asarray(mask)))
    expected = np.asarray(mask_logits(jnp.asarray(logits), jnp.asarray(mask), 0))
    np.testing.assert_array_equal(out, expected)


class _DecodeOpsStub:
    """Minimal decode ops double recording step invocations."""

    def __init__(self, num_slots: int, recorded_events: list[str]) -> None:
        """Initialize mock buffers, mesh, and event log."""
        self.batch_sizes = (1, 2)
        self.attention_dp = 1
        self.num_slots = num_slots
        self.rank = 0
        self.slots = [
            torch.zeros((2, num_slots, 4)),
            torch.zeros((2, num_slots, 4)),
        ]
        self.unified_pools = [torch.zeros((4, 4))]
        self._recorded_events = recorded_events

    def cached_op(
        self,
        _key: tuple[object, ...],
        _factory: Callable[[], object],
    ) -> Callable[..., tuple[torch.Tensor, ...]]:
        """Return lightweight identity callable for control tensor operations."""
        return lambda *args, **_kwargs: (args[0],)

    def compilation_calls(
        self,
        *_args: object,
        **_kwargs: object,
    ) -> list[CompileCall]:
        """Return placeholder compilation descriptor for precompilation."""
        return [CompileCall(lambda: (), (), {}, priority=1)]

    def plain_step(
        self,
        control: torch.Tensor,
        shape_tuple: tuple[int, object, int],
        *,
        sampled: bool,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Record plain step batch and return placeholder outputs."""
        batch = shape_tuple[0]
        masked = mask is not None
        self._recorded_events.append(
            f"plain:{batch}:sampled={sampled}:masked={masked}",
        )
        return torch.zeros(batch, dtype=torch.int32), control

    def verify_packed(
        self,
        *_args: object,
        **_kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return dummy triple for packed verification."""
        return (torch.zeros(1), torch.zeros(1), torch.zeros(1))

    def verify_batched(
        self,
        *_args: object,
        **_kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return dummy triple for batched verification."""
        return (torch.zeros(1), torch.zeros(1), torch.zeros(1))

    def load_weights(self) -> None:
        """No-op weight loader for test double."""


@dataclasses.dataclass
class _StubDraftConfig:
    """Minimal config stub for _DraftStub."""


@dataclasses.dataclass
class _StubDraftState:
    """Typed draft state double for _DraftStub."""

    state: torch.Tensor
    vector: torch.Tensor


class _DraftStub:
    """Mock draft runtime program double without bind_pools or imports_ring."""

    uses_unified_pools: bool = False

    def __init__(self, recorded_events: list[str]) -> None:
        """Initialize mock draft state and recording sink."""
        self.weights = True
        self.config = _StubDraftConfig()
        self.aux_layers: tuple[object, ...] = ()
        self._recorded_events = recorded_events

    def synthetic_rates_for(self, _drafts: int) -> tuple[float, ...] | None:
        """Return mock synthetic acceptance rates."""
        return None

    def load_weights(self) -> None:
        """No-op weight loader for test double."""

    def compilation_calls(
        self,
        *_args: object,
        **_kwargs: object,
    ) -> list[CompileCall]:
        """Return empty list of draft compilation descriptors."""
        return []

    def from_ring(
        self,
        _offered_cache: torch.Tensor,
        _offered_positions: torch.Tensor,
        _zero: torch.Tensor,
    ) -> _StubDraftState:
        """Record from_ring call and return mock draft state."""
        self._recorded_events.append("from_ring")
        return _StubDraftState(state=torch.zeros(1), vector=torch.zeros(1))

    def new_state(self, _zero: torch.Tensor) -> _StubDraftState:
        """Return new mock draft state."""
        return _StubDraftState(state=torch.zeros(1), vector=torch.zeros(1))

    def zero(self) -> torch.Tensor:
        """Return mock zero draft state tensor."""
        return torch.zeros(1)

    def step_single(
        self,
        state: _StubDraftState,
        *_args: object,
        **_kwargs: object,
    ) -> tuple[_StubDraftState, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return mock single step outputs."""
        return (
            state,
            torch.zeros(1),
            torch.zeros(1, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
        )

    def step_batched(
        self,
        states: Sequence[_StubDraftState],
        *_args: object,
        **_kwargs: object,
    ) -> tuple[list[_StubDraftState], torch.Tensor, torch.Tensor]:
        """Return mock batched step outputs."""
        return (
            list(states),
            torch.zeros(1),
            torch.zeros(1, dtype=torch.int32),
        )


class _ImporterStub:
    """Mock KDA importer double."""

    def import_state(
        self,
        conv: torch.Tensor,
        recurrent: torch.Tensor,
        _blocks: torch.Tensor,
        _slot: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return unchanged recurrent state pair."""
        return (conv, recurrent)

    def import_ring(
        self,
        _draft_config: object,
        _blocks: torch.Tensor,
        _computed: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return mock ring cache and positions."""
        return (torch.zeros(1), torch.zeros(1, dtype=torch.int32))


_BATCH_PLAIN_FIRST = 1
_BATCH_PLAIN_SECOND = 2


def _verify_plain_step_batching(recorded_events: list[str]) -> None:
    """Verify that in every plain pass, batch steps precede materialize calls."""
    plain_batches: list[int] = []
    plain_passes = 0
    for event in recorded_events:
        if event.startswith("plain:"):
            batch_number = int(event.split(":")[1])
            plain_batches.append(batch_number)
        elif event == "materialize" and plain_batches:
            if plain_batches == [_BATCH_PLAIN_FIRST]:
                plain_batches.clear()
                continue
            assert _BATCH_PLAIN_FIRST in plain_batches
            assert _BATCH_PLAIN_SECOND in plain_batches
            plain_batches.clear()
            plain_passes += 1
    assert plain_passes > 0


def test_decoder_warmup_initializes_layout_precompiles_and_batches_plain_steps() -> (
    None
):
    """Verify decoder warmup initializes control layout and batches plain steps."""
    config = MegakernelEngineConfig(
        role="decode",
        tensor_parallel_size=32,
        decode_context_parallel_size=1,
        block_size=128,
        max_num_seqs=2,
        max_model_len=4096,
        max_num_batched_tokens=4096,
        num_speculative_tokens=_NUM_SPEC_TOKENS,
    )
    num_slots = Control.total_slots(2 * config.max_num_seqs + 1)
    events: list[str] = []
    ops = _DecodeOpsStub(num_slots, events)
    draft = _DraftStub(events)
    importer = _ImporterStub()

    original_to = cast("Callable[..., torch.Tensor]", torch.Tensor.to)

    def _safe_to(
        tensor_self: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        """Route tpu placement requests to cpu."""
        if args and args[0] == "tpu":
            return original_to(tensor_self, "cpu", *args[1:], **kwargs)
        if kwargs.get("device") == "tpu":
            kwargs["device"] = "cpu"
        return original_to(tensor_self, *args, **kwargs)

    original_full = cast("Callable[..., torch.Tensor]", torch.full)

    def _safe_full(*args: object, **kwargs: object) -> torch.Tensor:
        """Route tpu device allocations to cpu."""
        if kwargs.get("device") == "tpu":
            kwargs["device"] = "cpu"
        return original_full(*args, **kwargs)

    compile_counts = [0]

    def _record_compile_ops(_calls: object) -> None:
        """Record invocations of compile_ops."""
        compile_counts[0] += 1

    def _record_materialize(_target: object) -> None:
        """Record invocations of materialize."""
        events.append("materialize")

    def _noop_hook(*_args: object, **_kwargs: object) -> None:
        """No-op callback for synchronization and distributed hooks."""

    with (
        patch.object(torch.Tensor, "to", _safe_to),
        patch("torch.full", _safe_full),
        patch.object(decode, "logger", logging.getLogger("decode_test")),
        patch.object(decode, "synchronize", _noop_hook),
        patch.object(decode, "prepare_distributed", _noop_hook),
        patch.object(decode, "compile_ops", _record_compile_ops),
        patch.object(decode, "materialize", _record_materialize),
    ):
        decoder = KimiDecoder(config, cast("DecodeOps", ops), draft, tp_rank=0)
        decoder._pages = _EXPECTED_PAGES
        decoder._importer = cast("decode.KdaImporter", importer)

        decoder.warmup()

        assert decoder.layout is not None
        assert decoder.layout.pages == _EXPECTED_PAGES
        assert compile_counts[0] == 1
        assert decoder._compiled_layout == decoder.layout.key()
        assert "from_ring" in events

        _verify_plain_step_batching(events)
