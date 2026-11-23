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
"""Tests for sampling mask_logits unpacking, grammar masking, and warmup contracts."""

from __future__ import annotations

import contextlib
import dataclasses
import logging
from typing import TYPE_CHECKING, Protocol, cast
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.models.kimi_k3.megakernel import decode, prefill
from vllm_torchtpu.models.kimi_k3.megakernel.common.constants import (
    AXIS_TP,
    BITS_PER_WORD,
)
from vllm_torchtpu.models.kimi_k3.megakernel.decode import (
    CompileCall,
    Control,
    KimiDecoder,
)
from vllm_torchtpu.models.kimi_k3.megakernel.kernels import dspark
from vllm_torchtpu.models.kimi_k3.megakernel.program import TP_SIZE, VOCAB
from vllm_torchtpu.models.kimi_k3.megakernel.runtime_shim.config import (
    MegakernelEngineConfig,
)
from vllm_torchtpu.models.kimi_k3.megakernel.runtime_shim.program import (
    KVGroup,
    StepInput,
)
from vllm_torchtpu.models.kimi_k3.megakernel.sampling import (
    apply_grammar_mask,
    mask_logits,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from pathlib import Path

    from vllm_torchtpu.models.kimi_k3.megakernel.decode_ops import DecodeOps

_NEG_INF = -1e30
_INT32_BOUND = 2**31
VOCAB_WORDS = VOCAB // BITS_PER_WORD
_EXPECTED_PAGES = 16
_NUM_SPEC_TOKENS = 7
_WARM_BATCHED_CALLS_BEFORE_TRANSITIONS = 6
_BATCH_PLAIN_FIRST = 1
_BATCH_PLAIN_SECOND = 2


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
    """Verify all-ones bitmask is identity and all-zeros shard stays finite."""
    width = VOCAB // TP_SIZE
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(2, width)).astype(np.float32)
    masked = jax.jit(mask_logits)

    all_ones = np.full((2, VOCAB_WORDS), -1, np.int32)
    for rank in (0, TP_SIZE - 1):
        np.testing.assert_array_equal(
            np.asarray(masked(jnp.asarray(logits), jnp.asarray(all_ones), rank)),
            logits,
        )

    # A schema allowing only rank 3's tokens fully masks every other rank:
    # finite sentinel, never NaN or -inf (the per-rank normalizers stay sane).
    narrow = np.zeros((2, VOCAB_WORDS), np.int32)
    narrow[:, 3 * width // BITS_PER_WORD : 4 * width // BITS_PER_WORD] = -1
    out = np.asarray(masked(jnp.asarray(logits), jnp.asarray(narrow), 0))
    assert bool((out == _NEG_INF).all())
    assert bool(np.isfinite(out).all())
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
    mesh = Mesh(np.array([jax.devices()[0]]), (AXIS_TP,))
    sharded_fn = jax.shard_map(
        apply_grammar_mask,
        mesh=mesh,
        in_specs=(P(AXIS_TP), P()),
        out_specs=P(AXIS_TP),
        check_vma=False,
    )
    with pltpu.force_tpu_interpret_mode():
        out = np.asarray(sharded_fn(jnp.asarray(logits), jnp.asarray(mask)))
    expected = np.asarray(mask_logits(jnp.asarray(logits), jnp.asarray(mask), 0))
    np.testing.assert_array_equal(out, expected)


class _DecodeOpsStub:
    """Minimal decode ops double recording step invocations."""

    def __init__(
        self,
        num_slots: int,
        recorded_events: list[str],
        importer: _ImporterStub | None = None,
    ) -> None:
        """Initialize mock buffers, mesh, and event log."""
        self.batch_sizes = (1, 2)
        self.attention_dp = 1
        self.num_slots = num_slots
        self.rank = 0
        self.mesh = object()
        self.slots = [
            torch.zeros((2, num_slots, 4)),
            torch.zeros((2, num_slots, 4)),
        ]
        self.unified_pools = [torch.zeros((4, 4))]
        self._recorded_events = recorded_events
        self._importer = importer

    def cached_op(
        self,
        key: tuple[object, ...],
        _factory: Callable[[], object],
    ) -> object:
        """Return the cached importer double or a lightweight identity op."""
        if key == ("kda_importer", None) and self._importer is not None:
            return self._importer
        return lambda *args, **_kwargs: (args[0],)

    def use_unified_pools(self, _raw_pools: object, _mla_raw: object) -> None:
        """No-op unified pool binding for test double."""

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
class _StubDraftState:
    """Typed draft state double for _DraftStub."""

    state: torch.Tensor
    vector: torch.Tensor


class _DraftStub:
    """Mock draft runtime program double without unified pool binding."""

    uses_unified_pools: bool = False

    def __init__(self, recorded_events: list[str]) -> None:
        """Initialize mock draft state and recording sink."""
        self.weights: dict[str, torch.Tensor] = {"stub": torch.zeros(1)}
        self.config = dspark.DraftConfig()
        self.aux_layers: tuple[int, ...] = ()
        self._recorded_events = recorded_events

    def synthetic_rates_for(self, _drafts: int) -> tuple[float, ...] | None:
        """Return mock synthetic acceptance rates."""
        return None

    def load_weights(self) -> float:
        """No-op weight loader for test double."""
        return 0.0

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


@dataclasses.dataclass
class _ImporterStub:
    """Mock KDA importer double supporting public bind_kv_caches."""

    raw_pools: list[torch.Tensor] = dataclasses.field(default_factory=list)
    mla_raw: list[int] = dataclasses.field(default_factory=list)
    mla_layer_names: tuple[str, ...] = ("model.layers.0.self_attn",)
    kda_layer_names: tuple[str, ...] = ("model.layers.1.self_attn",)

    def bind(self, _caches: object, _dtype: str) -> int:
        """Return fixed block count for KV binding."""
        return 4

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


def _make_kv_groups(pages: int) -> list[KVGroup]:
    """Build minimal MLA and KDA KVGroup descriptors for bind_kv_caches."""
    return [
        KVGroup(
            layer_names=("model.layers.0.self_attn",),
            block_size=128,
            table_width=pages,
        ),
        KVGroup(
            layer_names=("model.layers.1.self_attn",),
            block_size=128,
            table_width=pages,
        ),
    ]


@contextlib.contextmanager
def _cpu_decode_patches(
    events: list[str],
    compile_counts: list[int],
) -> Iterator[None]:
    """Patch runtime hooks for CPU-hermetic decoder warmup tests."""

    def _record_compile_ops(_calls: object) -> None:
        """Record invocations of compile_ops."""
        compile_counts[0] += 1

    def _record_materialize(_target: object) -> None:
        """Record invocations of materialize."""
        events.append("materialize")

    def _noop_hook(*_args: object, **_kwargs: object) -> None:
        """No-op callback for synchronization and distributed hooks."""

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            patch.object(decode, "logger", logging.getLogger("decode_test"))
        )
        stack.enter_context(patch.object(decode, "synchronize_tpu", _noop_hook))
        stack.enter_context(
            patch.object(decode, "tp_rank_permutation", return_value=None)
        )
        stack.enter_context(patch.object(decode, "prepare_distributed", _noop_hook))
        stack.enter_context(patch.object(decode, "compile_ops", _record_compile_ops))
        stack.enter_context(patch.object(decode, "materialize", _record_materialize))
        yield


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
    importer = _ImporterStub()
    ops = _DecodeOpsStub(num_slots, events, importer=importer)
    draft = _DraftStub(events)
    compile_counts = [0]

    with _cpu_decode_patches(events, compile_counts):
        decoder = KimiDecoder(config, cast("DecodeOps", ops), draft, tp_rank=0)
        decoder.bind_kv_caches({}, _make_kv_groups(_EXPECTED_PAGES))
        decoder.warmup()

        assert decoder.layout is not None
        assert decoder.layout.pages == _EXPECTED_PAGES
        assert compile_counts[0] == 1
        assert decoder.compiled_layout == decoder.layout.key()
        assert "from_ring" in events

        _verify_plain_step_batching(events)


class _ClearDraftOnBatchStub(_DraftStub):
    """Draft stub that clears decoder.draft just before transition warmup."""

    def __init__(self, recorded_events: list[str]) -> None:
        """Initialize stub with deferred decoder binding."""
        super().__init__(recorded_events)
        self.decoder: KimiDecoder | None = None
        self._batched_calls = 0

    def step_batched(
        self,
        states: Sequence[_StubDraftState],
        *args: object,
        **kwargs: object,
    ) -> tuple[list[_StubDraftState], torch.Tensor, torch.Tensor]:
        """Clear decoder.draft on the final batched step before transitions."""
        result = super().step_batched(states, *args, **kwargs)
        self._batched_calls += 1
        if (
            self.decoder is not None
            and self._batched_calls == _WARM_BATCHED_CALLS_BEFORE_TRANSITIONS
        ):
            self.decoder.draft = None
        return result


class _DropDraftStateOnTransitionStub(_DraftStub):
    """Draft stub that returns None on the final transition single-spec step."""

    def __init__(self, recorded_events: list[str]) -> None:
        """Track single and batched invocations to trigger transition failure."""
        super().__init__(recorded_events)
        self._batched_seen = False
        self._single_calls = 0

    def step_batched(
        self,
        states: Sequence[_StubDraftState],
        *args: object,
        **kwargs: object,
    ) -> tuple[list[_StubDraftState], torch.Tensor, torch.Tensor]:
        """Mark that batched warmup ran before transition single-spec calls."""
        self._batched_seen = True
        return super().step_batched(states, *args, **kwargs)

    def step_single(
        self,
        state: _StubDraftState,
        *args: object,
        **kwargs: object,
    ) -> tuple[_StubDraftState, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return None draft state on the final step of transition single-spec."""
        if self._batched_seen:
            self._single_calls += 1
            if self._single_calls == decode.WARM_SINGLE_STEPS:
                return (
                    cast("_StubDraftState", None),
                    torch.zeros(1),
                    torch.zeros(1, dtype=torch.int32),
                    torch.zeros(1, dtype=torch.int32),
                )
        return super().step_single(state, *args, **kwargs)


def test_warm_spec_transitions_fails_closed_without_draft() -> None:
    """Raise RuntimeError instead of returning or continuing without draft state."""
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

    events_missing_draft: list[str] = []
    importer_a = _ImporterStub()
    ops_a = _DecodeOpsStub(num_slots, events_missing_draft, importer=importer_a)
    draft_a = _ClearDraftOnBatchStub(events_missing_draft)
    with _cpu_decode_patches(events_missing_draft, [0]):
        decoder_a = KimiDecoder(config, cast("DecodeOps", ops_a), draft_a, tp_rank=0)
        draft_a.decoder = decoder_a
        decoder_a.bind_kv_caches({}, _make_kv_groups(_EXPECTED_PAGES))
        with pytest.raises(
            RuntimeError,
            match="K3 decode: decode warmup requires initialized draft state",
        ):
            decoder_a.warmup()

    events_missing_state: list[str] = []
    importer_b = _ImporterStub()
    ops_b = _DecodeOpsStub(num_slots, events_missing_state, importer=importer_b)
    draft_b = _DropDraftStateOnTransitionStub(events_missing_state)
    with _cpu_decode_patches(events_missing_state, [0]):
        decoder_b = KimiDecoder(config, cast("DecodeOps", ops_b), draft_b, tp_rank=0)
        decoder_b.bind_kv_caches({}, _make_kv_groups(_EXPECTED_PAGES))
        with pytest.raises(
            RuntimeError,
            match="K3 decode: draft state missing during speculative warmup",
        ):
            decoder_b.warmup()


@dataclasses.dataclass(frozen=True)
class _StubLayoutShape:
    """Minimal KV layout descriptor exposing block_shape."""

    block_shape: tuple[int, ...] = (1,)


class _StubPrefillPoolWriter:
    """Minimal PoolWriter double for CPU-hermetic prefill warmup testing."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        """Initialize placeholder pool buffers and layout."""
        self.raw_pools = [torch.zeros(1)]
        self.layout = _StubLayoutShape()

    def mla_pools(self) -> list[torch.Tensor]:
        """Return single placeholder MLA pool."""
        return [torch.zeros(1)]

    def write_packed(self, _write: object) -> None:
        """Ignore packed pool writes during warmup."""

    def read_states(self, _blocks: torch.Tensor) -> tuple[torch.Tensor]:
        """Return placeholder KDA state read."""
        return (torch.zeros(1),)

    def copy_blocks(self, _pairs: Sequence[tuple[int, int]]) -> None:
        """Ignore block copies during warmup."""


class _StubDraftLatents:
    """Minimal PrefillDraftLatents double recording warmup contract calls."""

    def __init__(self) -> None:
        """Initialize call logs for bind, latents, write, and copy_blocks."""
        self.loaded = False
        self.bound_count = 0
        self.latents_windows: list[int] = []
        self.written_shapes: list[tuple[int, ...]] = []
        self.copied: list[tuple[list[tuple[int, int]], int]] = []

    def load_weights(self) -> None:
        """Record weight loading."""
        self.loaded = True

    def bind(self, pools_arg: Sequence[torch.Tensor]) -> None:
        """Record number of bound draft pools."""
        self.bound_count = len(pools_arg)

    def latents(
        self,
        window: int,
        *_args: object,
        **_kwargs: object,
    ) -> torch.Tensor:
        """Record ``window`` and return dummy draft latent tensor."""
        self.latents_windows.append(window)
        return torch.zeros(1)

    def write(
        self,
        _latents: torch.Tensor,
        _table: torch.Tensor,
        block_shape: tuple[int, ...],
    ) -> None:
        """Record target ``block_shape`` passed to draft latent write."""
        self.written_shapes.append(block_shape)

    def copy_blocks(
        self,
        pairs: Sequence[tuple[int, int]],
        per_call: int,
    ) -> None:
        """Record copy-on-write pairs and chunk size."""
        self.copied.append((list(pairs), per_call))


class _RingInputLike(Protocol):
    """Minimal protocol exposing the optional ring hook on packed window inputs."""

    ring: Callable[..., None] | None


class _StubPrefillProgram:
    """Minimal PrefillProgram double exercising the warmup draft hook."""

    def __init__(
        self,
        windows: tuple[int, ...],
        *_args: object,
        **_kwargs: object,
    ) -> None:
        """Store single warmup window."""
        self.windows = windows[:1]
        self.rank = 0
        self.mesh = object()

    def load_weights(self, _presharded: Path) -> None:
        """No-op prefill weight loader."""

    def window_slots(self, _window: int) -> int:
        """Return one sequence slot for warmup."""
        return 1

    def zero_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return dummy zero KDA state pair."""
        return (torch.zeros(1), torch.zeros(1))

    def stack_states(
        self,
        _pairs: Sequence[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return stacked dummy KDA state pair."""
        return (torch.zeros(1), torch.zeros(1))

    def split_states(
        self,
        _recurrent: torch.Tensor,
        _conv: torch.Tensor,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Return single split dummy KDA state pair."""
        return [(torch.zeros(1), torch.zeros(1))]

    def sp_order(self, window: int) -> np.ndarray:
        """Return sequential SP token order."""
        return np.arange(window, dtype=np.int32)

    def run_packed(
        self,
        inputs: _RingInputLike,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Run the draft hook and return dummy CPU tensors."""
        dummy = torch.zeros(1)
        if inputs.ring is not None:
            inputs.ring(dummy, dummy, dummy, dummy, dummy)
        return (dummy, dummy, dummy, dummy, dummy, dummy)


def test_prefill_warmup_writes_draft_latents_on_resident_device(tmp_path: Path) -> None:
    """Verify prefill warmup fails before bind and runs draft-latent path on CPU."""
    config = MegakernelEngineConfig(
        role="prefill",
        tensor_parallel_size=32,
        decode_context_parallel_size=32,
        block_size=256,
        max_num_seqs=1,
        max_model_len=512,
        max_num_batched_tokens=512,
    )

    def _noop(*_args: object, **_kwargs: object) -> None:
        """No-op runtime hook."""

    draft_stub = _StubDraftLatents()
    with (
        patch.object(prefill, "PrefillProgram", _StubPrefillProgram),
        patch.object(prefill, "PoolWriter", _StubPrefillPoolWriter),
        patch.object(
            prefill,
            "bind_raw_pools",
            return_value=([torch.zeros(1)], [0], [0], ["mla0"], ["kda0"]),
        ),
        patch.object(prefill, "draft_raw_pools", return_value=[torch.zeros(1)]),
        patch.object(prefill, "tp_rank_permutation", return_value=None),
        patch.object(prefill, "prepare_distributed", _noop),
        patch.object(prefill, "compile_ops", _noop),
        patch.object(prefill, "materialize", _noop),
        patch.object(
            prefill,
            "register_op",
            return_value=lambda *_a: (torch.zeros(1),),
        ),
        patch.object(prefill, "kernel_mesh", return_value=(object(), 0)),
        patch.object(prefill.KimiPrefiller, "compilation_calls", return_value=[]),
        patch.object(prefill, "logger", logging.getLogger("prefill_test")),
    ):
        prefiller = prefill.KimiPrefiller(config, tmp_path, tp_rank=0)
        with pytest.raises(RuntimeError, match="bind the KV caches"):
            prefiller.warmup()
        prefiller.draft_latents = cast(
            "prefill.PrefillDraftLatents",
            draft_stub,
        )
        prefiller.bind_kv_caches(
            {},
            [
                KVGroup(layer_names=("mla0",), block_size=256, table_width=4),
                KVGroup(layer_names=("kda0",), block_size=256, table_width=4),
            ],
        )
        prefiller.warmup()

    assert draft_stub.loaded is True
    assert draft_stub.bound_count == 1
    assert draft_stub.latents_windows == [prefiller.program.windows[0]]
    assert draft_stub.written_shapes == [(1,)]
    assert draft_stub.copied == [([(0, 0)], prefill.COPY_PAIRS)]


def test_decode_zero_requests_returns_placeholder_tensor() -> None:
    """Verify KimiDecoder.step with zero requests returns a (1, 8) -1 int32 tensor."""
    config = MegakernelEngineConfig(
        role="decode",
        tensor_parallel_size=32,
        decode_context_parallel_size=1,
        block_size=128,
        max_num_seqs=2,
        max_model_len=4096,
        max_num_batched_tokens=4096,
        num_speculative_tokens=0,
    )
    num_slots = Control.total_slots(2 * config.max_num_seqs + 1)
    events: list[str] = []
    importer = _ImporterStub()
    ops = _DecodeOpsStub(num_slots, events, importer=importer)

    with (
        patch.object(decode, "tp_rank_permutation", return_value=None),
        patch.object(decode, "logger", logging.getLogger("decode_test")),
    ):
        decoder = KimiDecoder(config, cast("DecodeOps", ops), None, tp_rank=0)
        decoder.bind_kv_caches({}, _make_kv_groups(_EXPECTED_PAGES))
        step_output = decoder.step(
            StepInput(requests=(), finished=(), copies=(), mask=None)
        )

    assert isinstance(step_output.tokens, torch.Tensor)
    assert step_output.tokens.shape == (1, KimiDecoder.WIDTH)
    assert step_output.tokens.dtype == torch.int32
    assert step_output.tokens.device == torch.device("cpu")
    assert bool((step_output.tokens == -1).all())
