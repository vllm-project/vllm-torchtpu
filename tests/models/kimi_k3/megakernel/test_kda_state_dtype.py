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
#   tests/megakernel/kimi_k3/test_kda_state_dtype.py
"""State-storage selection, pool round-trip, and verifier row selection tests."""

from __future__ import annotations

import dataclasses
import importlib
from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.experimental import pallas as pl
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.models.kimi_k3.megakernel import (
    config as kernel_config,
)
from vllm_torchtpu.models.kimi_k3.megakernel import (
    decode,
    decode_ops,
    draft,
    draft_mla,
    draft_ring,
    pools,
    program,
)
from vllm_torchtpu.models.kimi_k3.megakernel import layout as k3_layout
from vllm_torchtpu.models.kimi_k3.megakernel.common.constants import AXIS_TP
from vllm_torchtpu.models.kimi_k3.megakernel.kernels import (
    decode_megakernel as kernel,
)
from vllm_torchtpu.models.kimi_k3.megakernel.kernels import dspark
from vllm_torchtpu.models.kimi_k3.megakernel.kernels import (
    dspark_mla as dspark_mla_mod,
)
from vllm_torchtpu.models.kimi_k3.megakernel.layout import (
    CONV_HEAD_ROWS,
    CONV_TAPS,
    HEAD_DIM,
    KDA_HEADS,
)
from vllm_torchtpu.models.kimi_k3.megakernel.runtime_shim import backend, bridge
from vllm_torchtpu.models.kimi_k3.megakernel.runtime_shim import (
    program as shim_program,
)
from vllm_torchtpu.models.kimi_k3.megakernel.runtime_shim.config import (
    MegakernelEngineConfig,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from jax.typing import DTypeLike

_PROJECTED_DIM = 1664
_ATTENDED_DIM = 384
_CONV_FILTER_TAPS = CONV_TAPS + 1
_RNG_SEED = 413
_POOL_BLOCKS = 4
_REAL_PALLAS_CALL = pl.pallas_call
# Mask bit 14 of float32 low-16 half so CPU Pallas interpret does not
# canonicalize bfloat16 NaN payloads (0x7F81..0x7FFF) on the bfloat16 pool.
_NON_BF16_NAN_LOW16_MASK = ~0x4000 & 0xFFFFFFFF
_TEST_ANCHOR_TOKEN = 17
_TEST_MAX_POS = 1024
_TEST_RING_LEN = 12
_REAL_TORCH_TENSOR = torch.tensor
_REAL_TORCH_ZEROS = torch.zeros
_REAL_TORCH_ONES = torch.ones
_REAL_TORCH_FULL = torch.full
_REAL_TORCH_FROM_NUMPY = torch.from_numpy
_REAL_TORCH_DEVICE = torch.device


@dataclasses.dataclass(frozen=True)
class _SyntheticDevice:
    """Synthetic torch-like device descriptor for CPU-backed TPU tests."""

    type: str = "tpu"


@dataclasses.dataclass(frozen=True)
class _FakeDecodeOpsDouble:
    """Minimal DecodeOps double captured by test_engine_config_passes_state_dtype."""

    presharded: Path
    num_slots: int
    mamba_ssm_cache_dtype: str
    batch_sizes: tuple[int, ...] = ()


@dataclasses.dataclass(frozen=True)
class _ParallelConfigDouble:
    """Parallel configuration stub for MegakernelEngineConfig.from_vllm_config."""

    tensor_parallel_size: int
    decode_context_parallel_size: int
    cp_kv_cache_interleave_size: int


@dataclasses.dataclass(frozen=True)
class _SchedulerConfigDouble:
    """Scheduler configuration stub for MegakernelEngineConfig.from_vllm_config."""

    max_num_seqs: int
    max_num_batched_tokens: int
    async_scheduling: bool


@dataclasses.dataclass(frozen=True)
class _ModelConfigDouble:
    """Model configuration stub for MegakernelEngineConfig.from_vllm_config."""

    max_model_len: int


@dataclasses.dataclass(frozen=True)
class _CacheConfigDouble:
    """Cache configuration stub for MegakernelEngineConfig.from_vllm_config."""

    block_size: int
    cache_dtype: str
    mamba_ssm_cache_dtype: str


@dataclasses.dataclass(frozen=True)
class _VllmConfigDouble:
    """Top-level vLLM config stub for MegakernelEngineConfig.from_vllm_config."""

    parallel_config: _ParallelConfigDouble
    scheduler_config: _SchedulerConfigDouble
    model_config: _ModelConfigDouble
    cache_config: _CacheConfigDouble
    speculative_config: object = None


@dataclasses.dataclass(frozen=True)
class _NoOpAsyncCopy:
    """No-op async copy stub for KDA kernel tests."""

    def wait(self) -> None:
        """Complete immediately."""


@dataclasses.dataclass(frozen=True)
class _StubProjectionCopies:
    """Projection copy handles stub for KDA kernel weights."""

    gate: _NoOpAsyncCopy
    feature_gate: _NoOpAsyncCopy


@dataclasses.dataclass
class _StubKdaWeights:
    """Structural KDA weight container stub for interpret-mode tests."""

    gate_projection: jax.Array
    input_projection: jax.Array
    feature_gate_projection_vmem: jax.Array
    exp_log_decay_rate: jax.Array
    time_step_bias: jax.Array
    convolution_filter: jax.Array
    output_normalization: jax.Array

    def make_projection_copies(self) -> _StubProjectionCopies:
        """Return no-op projection async copy handles."""
        no_copy = _NoOpAsyncCopy()
        return _StubProjectionCopies(gate=no_copy, feature_gate=no_copy)


@dataclasses.dataclass
class _StubCommWorkspace:
    """Minimal collective workspace stub carrying reduction_phase."""

    reduction_phase: jax.Array


@dataclasses.dataclass(frozen=True)
class _StubKdaImporter:
    """Importer double wrapping test callbacks for decoder bind and step tests."""

    import_state: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, int],
        tuple[torch.Tensor, torch.Tensor],
    ]
    mla_layer_names: tuple[str, ...]
    kda_layer_names: tuple[str, ...]
    raw_pools: list[torch.Tensor] = dataclasses.field(default_factory=list)
    mla_raw: list[int] = dataclasses.field(default_factory=list)

    def bind(self, _caches: object, _dtype: str) -> int:
        """Return synthetic unified block count."""
        return _POOL_BLOCKS

    def import_ring(
        self,
        draft_config: dspark.DraftConfig | dspark_mla_mod.MlaDraftConfig,
        _kda_blocks: torch.Tensor,
        _length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a cold synthetic TPU draft ring pair."""
        _ = draft_config
        cfg = dspark.DraftConfig()
        ring_cache = _as_fake_tpu(
            _REAL_TORCH_ZEROS(
                (1, cfg.layers, cfg.window, 2 * cfg.head_dim),
                dtype=torch.bfloat16,
            ),
        )
        ring_pos = _as_fake_tpu(
            _REAL_TORCH_FULL((1, cfg.window), -1, dtype=torch.int32)
        )
        return ring_cache, ring_pos


@dataclasses.dataclass
class _AdmitDecodeOpsDouble:
    """DecodeOps double for decoder admission and batched speculation sync tests."""

    rank: int
    num_slots: int
    attention_dp: int
    batch_sizes: tuple[int, ...]
    slots: list[torch.Tensor]
    importer: _StubKdaImporter
    recorded_modes: list[torch.Tensor] = dataclasses.field(default_factory=list)
    events: list[str] = dataclasses.field(default_factory=list)
    mesh: object = dataclasses.field(default_factory=object)
    unified_pools: list[torch.Tensor] = dataclasses.field(default_factory=list)
    pool_map: list[int] | None = None

    def load_weights(self) -> None:
        """No-op weight load for CPU decode-admission tests."""

    def use_unified_pools(
        self,
        _raw: Sequence[torch.Tensor],
        _mla: Sequence[int],
    ) -> None:
        """No-op unified pool bind for CPU decode-admission tests."""

    def cached_op(
        self,
        key: object,
        _factory: Callable[[], object],
    ) -> object:
        """Return the stub KDA importer or a pass-through control/enter op."""
        if key == ("kda_importer", None):
            return self.importer
        return lambda control, *_rest: (control,)

    def batch_for(self, _rows: int) -> int:
        """Return the single compiled batch size."""
        return self.batch_sizes[0]

    def plain_step(
        self,
        control: torch.Tensor,
        _shape: object,
        *,
        sampled: bool,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return synthetic greedy emissions and pass control through."""
        _ = (sampled, mask)
        return (
            _REAL_TORCH_ZEROS((1, decode.KimiDecoder.WIDTH), dtype=torch.int32),
            control,
        )

    def verify_batched(
        self,
        _spec: object,
        _layout: object,
        control_and_modes: tuple[torch.Tensor, torch.Tensor],
        *,
        synthetic_rates: tuple[float, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Record modes tensor passed into batched verify and return dummy outputs."""
        _ = synthetic_rates
        _, modes_t = control_and_modes
        self.events.append("verify_batched")
        self.recorded_modes.append(modes_t)
        dummy = _as_fake_tpu(_REAL_TORCH_ZEROS((1,), dtype=torch.int32))
        return dummy, dummy, dummy


@dataclasses.dataclass
class _StubSpecDraft:
    """Minimal DSpark draft double exercising batched speculation modes sync."""

    config: dspark.DraftConfig = dataclasses.field(default_factory=dspark.DraftConfig)
    aux_layers: tuple[int, ...] = (0,)
    uses_unified_pools: bool = False
    weights: dict[str, torch.Tensor] = dataclasses.field(
        default_factory=lambda: {"w": _as_fake_tpu(_REAL_TORCH_ZEROS((1,)))},
    )

    def load_weights(self) -> float:
        """No-op weight load returning zero wall time."""
        return 0.0

    def new_state(self, _zero: torch.Tensor, /) -> draft_mla.MlaDraftState:
        """Return a minimal synthetic draft state."""
        return draft_mla.MlaDraftState(
            _as_fake_tpu(
                _REAL_TORCH_FULL(
                    (1, draft_mla.DRAFT_VECTOR_LEN),
                    -1,
                    dtype=torch.int32,
                ),
            ),
        )

    def from_ring(
        self,
        _cache: torch.Tensor,
        _positions: torch.Tensor,
        zero: torch.Tensor,
        /,
    ) -> draft_mla.MlaDraftState:
        """Return a minimal synthetic draft state from an offered ring."""
        return self.new_state(zero)

    def synthetic_rates_for(self, _drafts: int, /) -> tuple[float, ...] | None:
        """Return no synthetic acceptance rates."""
        return None

    def step_single(
        self,
        state: draft_mla.MlaDraftState,
        inputs: decode_ops.DraftStepInputs,
    ) -> tuple[draft_mla.MlaDraftState, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return unchanged state and dummy single-request outputs."""
        dummy = _as_fake_tpu(_REAL_TORCH_ZEROS((1,), dtype=torch.int32))
        return state, dummy, inputs.control_state, dummy

    def step_batched(
        self,
        states: Sequence[draft_mla.MlaDraftState],
        inputs: decode_ops.BatchedDraftStepInputs,
    ) -> tuple[list[draft_mla.MlaDraftState], torch.Tensor, torch.Tensor]:
        """Return unchanged states and dummy batched emissions."""
        emitted = _REAL_TORCH_ZEROS(
            (len(states), decode.KimiDecoder.WIDTH),
            dtype=torch.int32,
        )
        return list(states), emitted, inputs.control_state

    def compilation_calls(
        self,
        _control: decode_ops.Control,
        _spec_shapes: Sequence[tuple[int, int]],
        *,
        spec_alone: bool = False,
    ) -> list[object]:
        """Return no precompilation calls for stub draft."""
        _ = spec_alone
        return []


class _UnboundImporterDecoder(decode.KimiDecoder):
    """KimiDecoder subclass allowing tests to drop the bound importer cleanly."""

    def drop_importer(self) -> None:
        """Clear the bound importer to test fail-closed zero-slot admission."""
        self._importer = None


class _CpuTpuCapableTensor(torch.Tensor):
    """CPU tensor that upgrades to _FakeTpuTensor when .to('tpu') is called."""

    def to(self, *args: object, **kwargs: object) -> torch.Tensor:
        """Upgrade to _FakeTpuTensor when target device is TPU, else stay on CPU."""
        target = kwargs.get("device")
        for arg in args:
            if isinstance(arg, (_REAL_TORCH_DEVICE, str, _SyntheticDevice)):
                target = arg
        is_tpu = target == "tpu" or (
            isinstance(target, (_REAL_TORCH_DEVICE, _SyntheticDevice))
            and target.type == "tpu"
        )
        clean_kwargs = {k: v for k, v in kwargs.items() if k != "device"}
        clean_args = tuple(
            arg
            for arg in args
            if not isinstance(arg, (_REAL_TORCH_DEVICE, str, _SyntheticDevice))
        )
        converted = super().to(*clean_args, **clean_kwargs)
        if is_tpu:
            return converted.as_subclass(_FakeTpuTensor)
        return converted.as_subclass(torch.Tensor)


class _FakeTpuTensor(_CpuTpuCapableTensor):
    """CPU-backed tensor double reporting device.type == 'tpu'."""

    @property
    def device(self) -> _SyntheticDevice:
        """Return a synthetic TPU device descriptor."""
        return _SyntheticDevice(type="tpu")

    def to(self, *args: object, **kwargs: object) -> _FakeTpuTensor:
        """Convert dtype on CPU while preserving _FakeTpuTensor."""
        clean_kwargs = {k: v for k, v in kwargs.items() if k != "device"}
        clean_args = tuple(
            arg
            for arg in args
            if not isinstance(arg, (_REAL_TORCH_DEVICE, str, _SyntheticDevice))
        )
        return torch.Tensor.to(self, *clean_args, **clean_kwargs).as_subclass(
            _FakeTpuTensor,
        )

    def contiguous(
        self,
        memory_format: torch.memory_format = torch.contiguous_format,
    ) -> _FakeTpuTensor:
        """Return a contiguous _FakeTpuTensor on CPU."""
        return (
            super().contiguous(memory_format=memory_format).as_subclass(_FakeTpuTensor)
        )

    def clone(
        self,
        *,
        memory_format: torch.memory_format | None = None,
    ) -> _FakeTpuTensor:
        """Return a cloned _FakeTpuTensor on CPU."""
        return super().clone(memory_format=memory_format).as_subclass(_FakeTpuTensor)

    def reshape(self, *shape: int) -> _FakeTpuTensor:
        """Return a reshaped _FakeTpuTensor on CPU."""
        return super().reshape(*shape).as_subclass(_FakeTpuTensor)


def _as_fake_tpu(tensor: torch.Tensor) -> _FakeTpuTensor:
    """Wrap a CPU torch.Tensor as a _FakeTpuTensor."""
    return tensor.as_subclass(_FakeTpuTensor)


def _is_tpu_device(device: object) -> bool:
    """Return True when device represents a TPU device."""
    return device == "tpu" or (
        isinstance(device, (_REAL_TORCH_DEVICE, _SyntheticDevice))
        and device.type == "tpu"
    )


def _fake_torch_device(device: object) -> object:
    """Return a synthetic TPU device descriptor for 'tpu', else a torch.device."""
    if _is_tpu_device(device):
        return _SyntheticDevice(type="tpu")
    return _REAL_TORCH_DEVICE(cast("torch.device | str", device))


def _fake_torch_tensor(
    data: object,
    *,
    dtype: torch.dtype | None = None,
    device: object = None,
) -> torch.Tensor:
    """Construct a _FakeTpuTensor when device is TPU, else a CPU tensor."""
    if _is_tpu_device(device):
        return _as_fake_tpu(_REAL_TORCH_TENSOR(data, dtype=dtype))
    target_dev = cast("torch.device | str | None", device)
    return _REAL_TORCH_TENSOR(
        data,
        dtype=dtype,
        device=target_dev,
    ).as_subclass(_CpuTpuCapableTensor)


def _fake_torch_zeros(
    *size: object,
    dtype: torch.dtype | None = None,
    device: object = None,
) -> torch.Tensor:
    """Construct a zero tensor supporting synthetic TPU device routing."""
    shape = (
        cast("tuple[int, ...]", size[0])
        if len(size) == 1 and isinstance(size[0], tuple)
        else cast("tuple[int, ...]", size)
    )
    if _is_tpu_device(device):
        return _as_fake_tpu(_REAL_TORCH_ZEROS(shape, dtype=dtype))
    target_dev = cast("torch.device | str | None", device)
    return _REAL_TORCH_ZEROS(
        shape,
        dtype=dtype,
        device=target_dev,
    ).as_subclass(_CpuTpuCapableTensor)


def _fake_torch_ones(
    *size: object,
    dtype: torch.dtype | None = None,
    device: object = None,
) -> torch.Tensor:
    """Construct a ones tensor supporting synthetic TPU device routing."""
    shape = (
        cast("tuple[int, ...]", size[0])
        if len(size) == 1 and isinstance(size[0], tuple)
        else cast("tuple[int, ...]", size)
    )
    if _is_tpu_device(device):
        return _as_fake_tpu(_REAL_TORCH_ONES(shape, dtype=dtype))
    target_dev = cast("torch.device | str | None", device)
    return _REAL_TORCH_ONES(
        shape,
        dtype=dtype,
        device=target_dev,
    ).as_subclass(_CpuTpuCapableTensor)


def _fake_torch_full(
    size: tuple[int, ...],
    fill_value: float,
    *,
    dtype: torch.dtype | None = None,
    device: object = None,
) -> torch.Tensor:
    """Construct a filled tensor supporting synthetic TPU device routing."""
    if _is_tpu_device(device):
        return _as_fake_tpu(_REAL_TORCH_FULL(size, fill_value, dtype=dtype))
    target_dev = cast("torch.device | str | None", device)
    return _REAL_TORCH_FULL(
        size,
        fill_value,
        dtype=dtype,
        device=target_dev,
    ).as_subclass(_CpuTpuCapableTensor)


def _fake_torch_from_numpy(ndarray: np.ndarray) -> torch.Tensor:
    """Wrap torch.from_numpy output so .to('tpu') returns _FakeTpuTensor."""
    return _REAL_TORCH_FROM_NUMPY(ndarray).as_subclass(_CpuTpuCapableTensor)


def _jax_to_fake_tpu(arr: jax.Array) -> _FakeTpuTensor:
    """Convert a JAX array to a CPU-backed _FakeTpuTensor."""
    np_arr = np.array(arr, copy=True)
    if np_arr.dtype.name == "bfloat16":
        tensor = _REAL_TORCH_FROM_NUMPY(np_arr.view(np.uint16)).view(torch.bfloat16)
    else:
        tensor = _REAL_TORCH_FROM_NUMPY(np_arr)
    return _as_fake_tpu(tensor)


def _expected_cold_draft_vector(
    anchor: int,
    rows: int,
    bonus_idx: int,
) -> torch.Tensor:
    """Build expected cold draft output vector directly from layout invariants."""
    expected = _REAL_TORCH_FULL(
        (1, draft_mla.DRAFT_VECTOR_LEN),
        -1,
        dtype=torch.int32,
    )
    for col in range(rows - 1):
        expected[0, col] = anchor
    expected[0, bonus_idx] = anchor
    return expected


def _expected_dspark_cold_state(
    cfg: dspark.DraftConfig,
    anchor: int,
) -> draft.DraftState:
    """Build an independent expected cold DSpark state from config and layout."""
    cache = _REAL_TORCH_ZEROS(
        (1, cfg.layers, cfg.window, 2 * cfg.head_dim),
        dtype=torch.bfloat16,
    )
    positions = _REAL_TORCH_FULL((1, cfg.window), -1, dtype=torch.int32)
    vector = _expected_cold_draft_vector(anchor, dspark.ROWS, dspark.OUT_BONUS)
    return draft.DraftState(cache, positions, vector)


@pytest.mark.parametrize(
    ("setting", "expected"),
    [("auto", "float32"), ("float32", "float32"), ("bfloat16", "bfloat16")],
)
def test_engine_config_passes_state_dtype_to_decoder(
    setting: str,
    expected: str,
) -> None:
    """Verify MegakernelEngineConfig forwards mamba_ssm_cache_dtype to DecodeOps."""
    captured: dict[str, str] = {}

    def _fake_decode_ops(
        path: Path,
        *,
        num_slots: int = 0,
        mamba_ssm_cache_dtype: str = "auto",
        batch_sizes: tuple[int, ...] = (),
    ) -> _FakeDecodeOpsDouble:
        """Capture keyword arguments passed to DecodeOps."""
        captured["dtype"] = mamba_ssm_cache_dtype
        return _FakeDecodeOpsDouble(
            presharded=path,
            num_slots=num_slots,
            mamba_ssm_cache_dtype=mamba_ssm_cache_dtype,
            batch_sizes=batch_sizes,
        )

    def _fake_decoder(
        _cfg: MegakernelEngineConfig,
        ops_obj: _FakeDecodeOpsDouble,
        *_args: object,
        **_kwargs: object,
    ) -> object:
        """Return the underlying DecodeOps stub."""
        return ops_obj

    with (
        patch.object(backend, "role", return_value="decode"),
        patch.object(backend, "attention_dp", return_value=2),
        patch.object(backend, "synthetic_acceptance_lengths", return_value=None),
        patch.object(decode_ops, "DecodeOps", side_effect=_fake_decode_ops),
        patch.object(decode, "KimiDecoder", side_effect=_fake_decoder),
    ):
        config = MegakernelEngineConfig.from_vllm_config(
            _VllmConfigDouble(
                parallel_config=_ParallelConfigDouble(
                    tensor_parallel_size=32,
                    decode_context_parallel_size=16,
                    cp_kv_cache_interleave_size=256,
                ),
                scheduler_config=_SchedulerConfigDouble(
                    max_num_seqs=128,
                    max_num_batched_tokens=128,
                    async_scheduling=True,
                ),
                model_config=_ModelConfigDouble(max_model_len=139264),
                cache_config=_CacheConfigDouble(
                    block_size=256,
                    cache_dtype="auto",
                    mamba_ssm_cache_dtype=setting,
                ),
                speculative_config=None,
            )
        )
        program.build(config, Path("unused"), tp_rank=0)
        assert captured["dtype"] == setting
        assert program.recurrent_state_dtype(captured["dtype"]) == expected


@pytest.mark.parametrize("setting", ["float16", "fp8", "invalid"])
def test_unsupported_state_dtype_rejected_before_allocation(setting: str) -> None:
    """Reject unsupported mamba_ssm_cache_dtype strings before buffer allocation."""
    with pytest.raises(ValueError, match="mamba-ssm-cache-dtype"):
        decode_ops.DecodeOps(
            Path("unused"),
            num_slots=2,
            mamba_ssm_cache_dtype=setting,
        )


def kda_recurrence_call(
    projected: jax.Array,
    history: jax.Array,
    state: jax.Array,
    sequence_rows: int,
    *,
    interpret: bool = True,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Run KDA recurrence with fixed projections, excluding cross-rank work."""
    batch = projected.shape[0]
    kda_fn: Callable[..., object] = kernel.kda

    def body(*refs: jax.Array) -> None:
        """Execute single-call KDA kernel body inside Pallas interpret mode."""
        (
            projected_ref,
            history_ref,
            state_ref,
            feature_ref,
            next_history,
            next_state,
            output,
        ) = refs
        # No projection DMA: these small constant weights supply the gate and
        # normalization; the per-token input projections are supplied directly.
        weights = _StubKdaWeights(
            gate_projection=jnp.zeros((HEAD_DIM, HEAD_DIM), jnp.bfloat16),
            input_projection=jnp.zeros((HEAD_DIM, _PROJECTED_DIM), jnp.bfloat16),
            feature_gate_projection_vmem=feature_ref,
            exp_log_decay_rate=jnp.zeros((KDA_HEADS, 1), jnp.float32),
            time_step_bias=jnp.full((KDA_HEADS, HEAD_DIM), -3.0, jnp.float32),
            convolution_filter=jnp.full(
                (_CONV_FILTER_TAPS, pools.CONV_CHANNEL_GROUPS, KDA_HEADS, HEAD_DIM),
                0.25,
                jnp.bfloat16,
            ),
            output_normalization=jnp.ones((HEAD_DIM,), jnp.bfloat16),
        )
        kda_fn(
            jnp.zeros((batch, HEAD_DIM), jnp.bfloat16),
            jnp.zeros((batch, HEAD_DIM), jnp.bfloat16),
            next_history,
            next_state,
            weights=weights,
            convolution_history_ref=history_ref,
            recurrent_state_ref=state_ref,
            communication_workspace=_StubCommWorkspace(
                reduction_phase=jnp.array([0], jnp.int32)
            ),
            synchronize_devices=False,
            weights_are_prefetched=True,
            sequence_rows=sequence_rows,
            shard_index=0,
            first_call=False,
            attended_out=output,
            gate_features_vmem=jnp.zeros((batch, HEAD_DIM), jnp.bfloat16),
            projected_rows=projected_ref[...],
        )

    runner: Callable[..., tuple[jax.Array, jax.Array, jax.Array]] = pl.pallas_call(
        body,
        out_shape=(
            jax.ShapeDtypeStruct(history.shape, history.dtype),
            jax.ShapeDtypeStruct(state.shape, state.dtype),
            jax.ShapeDtypeStruct((batch, _ATTENDED_DIM), jnp.bfloat16),
        ),
        interpret=interpret,
    )
    return runner(
        projected,
        history,
        state,
        jnp.zeros((HEAD_DIM, _ATTENDED_DIM), jnp.bfloat16),
    )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("width", [4, 8])
def test_verifier_state_matches_sequential_decode(
    dtype: DTypeLike,
    width: int,
) -> None:
    """Confirm batched KDA recurrence matches step-by-step sequential decode."""
    rng = np.random.default_rng(_RNG_SEED)
    requests = 2
    projected = jnp.asarray(
        rng.normal(size=(requests, width, _PROJECTED_DIM)),
        jnp.bfloat16,
    )
    history = jnp.asarray(
        rng.normal(
            size=(
                requests,
                CONV_TAPS,
                pools.CONV_CHANNEL_GROUPS,
                CONV_HEAD_ROWS,
                HEAD_DIM,
            ),
        ),
        jnp.bfloat16,
    )
    state = jnp.asarray(
        rng.normal(size=(requests, KDA_HEADS, HEAD_DIM, HEAD_DIM)),
        dtype,
    )
    batched = kda_recurrence_call(
        projected.reshape(-1, _PROJECTED_DIM),
        jnp.repeat(history, width, axis=0),
        jnp.repeat(state, width, axis=0),
        width,
    )
    cur_history = history
    cur_state = state
    for offset in range(width):
        cur_history, cur_state, attended = kda_recurrence_call(
            projected[:, offset],
            cur_history,
            cur_state,
            1,
        )
        # Only the three real conv heads are written (the fourth is padding).
        np.testing.assert_array_equal(
            batched[0][offset::width, :, :, :KDA_HEADS],
            cur_history[:, :, :, :KDA_HEADS],
        )
        np.testing.assert_array_equal(batched[1][offset::width], cur_state)
        np.testing.assert_array_equal(batched[2][offset::width], attended)


def test_prefill_rejects_bf16_instead_of_ignoring_the_flag() -> None:
    """Raise NotImplementedError when bfloat16 KDA state is requested on prefill."""
    config = MegakernelEngineConfig(
        role="prefill",
        tensor_parallel_size=32,
        decode_context_parallel_size=32,
        block_size=256,
        max_num_seqs=1,
        max_model_len=8192,
        max_num_batched_tokens=8192,
        mamba_ssm_cache_dtype="bfloat16",
    )
    with pytest.raises(NotImplementedError, match="decode-only"):
        program.check(config)


def test_slot_map_round_trip_dp1_and_dp4() -> None:
    """Verify SlotMap allocation, DP owner routing, and release for DP=1 and DP=4."""
    slot_map_dp1 = pools.SlotMap(num_slots=9, groups=1)
    allocated_dp1 = [slot_map_dp1.allocate(f"req-{idx}") for idx in range(4)]
    assert allocated_dp1 == [1, 2, 3, 4]
    assert "req-0" in slot_map_dp1
    assert slot_map_dp1.group("req-0") == 0
    slot_map_dp1.release(["req-0", "req-1", "req-2", "req-3"])
    assert "req-0" not in slot_map_dp1

    slot_map_dp4 = pools.SlotMap(num_slots=9, groups=4)
    g0 = slot_map_dp4.allocate("dp4-g0", group=0)
    g2 = slot_map_dp4.allocate("dp4-g2", group=2)
    assert (g0, g2) == (1, 5)
    assert slot_map_dp4.group("dp4-g0") == 0
    expected_dp_group = 2
    expected_owners = 4
    assert slot_map_dp4.group("dp4-g2") == expected_dp_group
    slot_map_dp4.release(["dp4-g0", "dp4-g2"])

    assert pools.dp_row_owners(4) == 1
    assert pools.dp_row_owners(8) == expected_owners
    rows_dp2 = pools.dp_group_rows(8, expected_dp_group)
    assert rows_dp2.tolist() == [0, 2, 4, 6, 1, 3, 5, 7]
    assert pools.dp_batch_sizes((32, 64), expected_owners) == (32, 64)
    with patch.object(
        kernel_config,
        "active",
        return_value=kernel_config.KernelConfig(attention_dp=expected_dp_group),
    ):
        opts = decode_ops.decode_options(8)
    assert opts["attention_dp"] == expected_dp_group


def _torch_to_jax_bit_exact(tensor: torch.Tensor) -> jax.Array:
    """Convert a CPU torch.Tensor to a JAX array preserving bfloat16 bit patterns."""
    if tensor.dtype == torch.bfloat16:
        return jnp.asarray(tensor.detach().view(torch.uint16).numpy()).view(
            jnp.bfloat16,
        )
    return jnp.asarray(tensor.detach().numpy())


def _jax_to_torch_bit_exact(array: jax.Array) -> torch.Tensor:
    """Convert a JAX array to a CPU torch.Tensor preserving bfloat16 bit patterns."""
    if array.dtype == jnp.bfloat16:
        raw_u16 = np.array(array.view(jnp.uint16), copy=True)
        return torch.from_numpy(raw_u16).view(torch.bfloat16)
    return torch.from_numpy(np.array(array, copy=True))


def _interpret_pallas_call(*args: object, **kwargs: object) -> object:
    """Run ``pl.pallas_call`` in CPU interpret mode for unit tests."""
    return cast("Callable[..., object]", _REAL_PALLAS_CALL)(
        *args,
        **kwargs,
        interpret=True,
    )


def _pack_kda_state_into_unified_pool(
    rec_state: torch.Tensor,
    conv_state: torch.Tensor,
) -> torch.Tensor:
    """Pack ``(rec_state, conv_state)`` into block 1 of a 4-D unified KDA pool."""
    block_shape = k3_layout.LAYOUT.block_shape
    _, tokens_per_row, pool_lanes = block_shape
    lane_split = pool_lanes // HEAD_DIM
    raw_u16 = np.zeros((_POOL_BLOCKS, *block_shape), dtype=np.uint16)
    # Fill null block 0 with non-zero bytes to prove block 0 is masked to zero.
    raw_u16[0] = np.uint16(0x3F80)
    ssm_f32 = np.zeros((pools.SSM_ROWS * lane_split, HEAD_DIM), dtype=np.float32)
    ssm_f32[: KDA_HEADS * HEAD_DIM] = rec_state.numpy().reshape(
        KDA_HEADS * HEAD_DIM,
        HEAD_DIM,
    )
    ssm_u32 = ssm_f32.reshape(pools.SSM_ROWS, pool_lanes).view(np.uint32)
    raw_u16[1, : pools.SSM_ROWS, 0, :] = (ssm_u32 & 0xFFFF).astype(np.uint16)
    raw_u16[1, : pools.SSM_ROWS, 1, :] = (ssm_u32 >> 16).astype(np.uint16)
    conv_u16 = np.zeros(
        pools.CONV_ROWS * tokens_per_row * pool_lanes,
        dtype=np.uint16,
    )
    conv_flat = conv_state.view(torch.uint16).numpy().reshape(-1)
    conv_u16[: conv_flat.size] = conv_flat
    raw_u16[1, pools.SSM_ROWS : pools.SSM_ROWS + pools.CONV_ROWS] = conv_u16.reshape(
        pools.CONV_ROWS,
        tokens_per_row,
        pool_lanes,
    )
    return torch.from_numpy(raw_u16).view(torch.bfloat16)


def test_kda_importer_round_trip_bit_exact() -> None:
    """Verify real JAX KdaImporter gather/import path is bit-exact and zeros block 0."""
    rng = np.random.default_rng(_RNG_SEED)
    rec_raw = np.asarray(
        rng.standard_normal((KDA_HEADS, HEAD_DIM, HEAD_DIM)),
        dtype=np.float32,
    )
    rec_u32 = rec_raw.view(np.uint32) & np.uint32(_NON_BF16_NAN_LOW16_MASK)
    rec_expected = torch.from_numpy(rec_u32.view(np.float32).copy())
    conv_expected = torch.from_numpy(
        np.asarray(
            rng.standard_normal(
                (CONV_TAPS, pools.CONV_CHANNEL_GROUPS, KDA_HEADS, HEAD_DIM),
            ),
            dtype=np.float32,
        ),
    ).to(torch.bfloat16)
    raw_pool = _pack_kda_state_into_unified_pool(rec_expected, conv_expected)

    mesh = Mesh(np.array([jax.devices()[0]]), (AXIS_TP,))
    importer = pools.KdaImporter(mesh, attention_dp=1, num_slots=4)
    importer.raw_pools = [raw_pool]
    importer.kda_raw = [0] * pools.KDA_LAYERS

    def fake_register_op(
        _name: str,
        op_mesh: Mesh,
        body: Callable[..., tuple[jax.Array, jax.Array]],
        spec: bridge.OpRegistrationSpec,
    ) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
        """Execute the real sharded JAX importer body directly on CPU."""
        mapped = jax.shard_map(
            body,
            mesh=op_mesh,
            in_specs=spec.in_specs,
            out_specs=spec.out_specs,
            check_vma=False,
        )

        def runner(*args: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Convert tensor arguments, run JAX shard_map body, and convert back."""
            jax_args = tuple(_torch_to_jax_bit_exact(arg) for arg in args)
            conv_out, rec_out = mapped(*jax_args)
            return _jax_to_torch_bit_exact(conv_out), _jax_to_torch_bit_exact(rec_out)

        return runner

    conv_bank = torch.zeros(
        (
            1,
            4,
            pools.KDA_LAYERS,
            CONV_TAPS,
            pools.CONV_CHANNEL_GROUPS,
            CONV_HEAD_ROWS,
            HEAD_DIM,
        ),
        dtype=torch.bfloat16,
    )
    rec_bank = torch.zeros(
        (1, 4, pools.KDA_LAYERS, KDA_HEADS, HEAD_DIM, HEAD_DIM),
        dtype=torch.float32,
    )
    kda_blocks = torch.ones((pools.KDA_LAYERS,), dtype=torch.int32)
    kda_blocks[1] = 0
    with (
        patch.object(bridge, "register_op", side_effect=fake_register_op),
        patch.object(pl, "pallas_call", side_effect=_interpret_pallas_call),
    ):
        conv_after, rec_after = importer.import_state(
            conv_bank,
            rec_bank,
            kda_blocks,
            slot=2,
        )

    bank_conv_diff = float(
        (conv_after[0, 2, 0, :, :, :KDA_HEADS, :].float() - conv_expected.float())
        .abs()
        .max()
        .item()
    )
    bank_rec_diff = float((rec_after[0, 2, 0] - rec_expected).abs().max().item())
    assert bank_conv_diff == 0.0
    assert bank_rec_diff == 0.0
    assert float(conv_after[0, 2, 0, :, :, KDA_HEADS:, :].abs().max().item()) == 0.0
    assert float(conv_after[0, 2, 1].abs().max().item()) == 0.0
    assert float(rec_after[0, 2, 1].abs().max().item()) == 0.0
    assert float(conv_after[0, 1].abs().max().item()) == 0.0


def test_select_verify_kda_rows_matches_accepted_token_offsets() -> None:
    """Verify select_verify_kda_rows selects exact source rows and target slots."""
    slots_dp1 = jnp.array([3, 3, 3, 3, 0, 0, 0, 0], dtype=jnp.int32)
    accepted_dp1 = jnp.array([2, 1], dtype=jnp.int32)
    sources_dp1, targets_dp1 = pools.select_verify_kda_rows(
        slots_dp1,
        accepted_dp1,
        pools.VerifyKdaSelectionConfig(
            requests=2,
            rows_per=4,
            attention_dp=1,
            bank_slots=64,
        ),
    )
    assert sources_dp1.tolist() == [2, 7]
    assert targets_dp1.tolist() == [3, 0]

    slots_dp2 = jnp.array(
        [1, 1, 1, 1, 2, 2, 2, 2, 32, 32, 32, 32, 0, 0, 0, 0],
        dtype=jnp.int32,
    )
    accepted_dp2 = jnp.array([1, 2, 3, 0], dtype=jnp.int32)
    sources_g0, targets_g0 = pools.select_verify_kda_rows(
        slots_dp2,
        accepted_dp2,
        pools.VerifyKdaSelectionConfig(
            requests=4,
            rows_per=4,
            attention_dp=2,
            bank_slots=64,
            group=0,
        ),
    )
    sources_g1, targets_g1 = pools.select_verify_kda_rows(
        slots_dp2,
        accepted_dp2,
        pools.VerifyKdaSelectionConfig(
            requests=4,
            rows_per=4,
            attention_dp=2,
            bank_slots=64,
            group=1,
        ),
    )
    assert sources_g0.tolist() == [1, 6, 9, 14]
    assert targets_g0.tolist() == [1, 2, 33, 34]
    assert sources_g1.tolist() == [3, 7, 11, 15]
    assert targets_g1.tolist() == [1, 0, 33, 32]
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        cast("Callable[..., object]", pools.VerifyKdaSelectionConfig)(
            requests=2,
            rows_per=4,
            unknown_option=1,
        )


def _make_dspark_global_op_registrar(
    registered_specs: dict[str, bridge.OpRegistrationSpec],
) -> Callable[
    [str, Mesh, Callable[..., Sequence[jax.Array]], bridge.OpRegistrationSpec],
    Callable[..., tuple[torch.Tensor, ...]],
]:
    """Build a fake register_global_op slicing P(AXIS_TP) outputs to rank 0."""

    def fake_register_global_op(
        name: str,
        _mesh: Mesh,
        fn: Callable[..., Sequence[jax.Array]],
        spec: bridge.OpRegistrationSpec,
    ) -> Callable[..., tuple[torch.Tensor, ...]]:
        """Capture global op registration and slice TP-sharded outputs to 1 rank."""
        registered_specs[name] = spec

        def runner(*args: torch.Tensor) -> tuple[torch.Tensor, ...]:
            """Run global JAX init op and shard P(AXIS_TP) outputs to local rank 0."""
            jax_args = tuple(jnp.asarray(arg.detach().cpu().numpy()) for arg in args)
            raw_outs = fn(*jax_args)
            sliced: list[torch.Tensor] = []
            for out, out_spec in zip(raw_outs, spec.out_specs, strict=True):
                if out_spec == P(AXIS_TP):
                    assert out.shape[0] == program.TP_SIZE
                    local = out[:1]
                else:
                    local = out
                sliced.append(_jax_to_fake_tpu(local))
            return tuple(sliced)

        return runner

    return fake_register_global_op


def _assert_dspark_state_matches(
    actual: draft.DraftState,
    expected: draft.DraftState,
    synced_batch: list[torch.Tensor],
) -> None:
    """Verify a TPU DSpark draft state matches independent reference and was settled."""
    assert actual.cache.device.type == "tpu"
    assert torch.equal(actual.cache, expected.cache)
    assert torch.equal(actual.positions, expected.positions)
    assert torch.equal(actual.vector, expected.vector)
    assert synced_batch == [actual.cache, actual.positions, actual.vector]


def test_mla_draft_new_state_uses_mesh_init_op_and_matches_cpu() -> None:
    """Verify MlaDsparkDraft.new_state matches independent oracle and syncs on TPU."""
    mesh = Mesh(np.array([jax.devices()[0]]), (AXIS_TP,))
    registered_specs: dict[str, bridge.OpRegistrationSpec] = {}
    sync_batches: list[tuple[list[torch.Tensor], bool]] = []
    sync_mod = importlib.import_module("torch_tpu._internal.sync")

    def fake_register_global_op(
        name: str,
        _mesh: Mesh,
        fn: Callable[..., Sequence[jax.Array]],
        spec: bridge.OpRegistrationSpec,
    ) -> Callable[..., tuple[torch.Tensor, ...]]:
        """Capture global op registration and run JAX body on CPU arrays."""
        registered_specs[name] = spec

        def runner(*args: torch.Tensor) -> tuple[torch.Tensor, ...]:
            """Convert tensor arguments to JAX arrays and wrap outputs as fake TPU."""
            jax_args = tuple(jnp.asarray(arg.detach().cpu().numpy()) for arg in args)
            return tuple(_jax_to_fake_tpu(out) for out in fn(*jax_args))

        return runner

    def fake_sync(tensors: Sequence[torch.Tensor], *, wait: bool = True) -> None:
        """Record synchronized tensor lists and wait flags."""
        sync_batches.append((list(tensors), wait))

    expected_vector = _expected_cold_draft_vector(
        _TEST_ANCHOR_TOKEN,
        dspark.ROWS,
        dspark.OUT_BONUS,
    )
    with (
        patch.object(bridge, "kernel_mesh", return_value=(mesh, 0)),
        patch.object(
            dspark_mla_mod.MlaDraftConfig,
            "from_checkpoint",
            return_value=dspark_mla_mod.MlaDraftConfig(),
        ),
        patch.object(
            bridge,
            "register_global_op",
            side_effect=fake_register_global_op,
        ),
        patch.object(sync_mod, "synchronize", side_effect=fake_sync),
        patch.object(torch, "device", side_effect=_fake_torch_device),
        patch.object(torch, "tensor", side_effect=_fake_torch_tensor),
        patch.object(torch, "zeros", side_effect=_fake_torch_zeros),
        patch.object(torch, "full", side_effect=_fake_torch_full),
    ):
        mla_draft = draft_mla.MlaDsparkDraft(_TEST_MAX_POS, "unused", Path("unused"))
        mla_draft.pools = [torch.zeros((1,), dtype=torch.int32)]
        cpu_state = mla_draft.new_state(_TEST_ANCHOR_TOKEN)
        assert torch.equal(cpu_state.vector, expected_vector)
        assert "pallas::k3_dspark_mla_init_state" not in registered_specs
        assert not sync_batches

        mla_draft.pools = [_as_fake_tpu(torch.zeros((1,), dtype=torch.int32))]
        tpu_state_int = mla_draft.new_state(_TEST_ANCHOR_TOKEN)
        tpu_anchor = _as_fake_tpu(
            _REAL_TORCH_TENSOR([_TEST_ANCHOR_TOKEN], dtype=torch.int32),
        )
        tpu_state_tensor = mla_draft.new_state(tpu_anchor)

        assert "pallas::k3_dspark_mla_init_state" in registered_specs
        init_spec = registered_specs["pallas::k3_dspark_mla_init_state"]
        assert init_spec.in_specs == (P(),)
        assert init_spec.out_specs == (P(),)
        tpu_states = (tpu_state_int, tpu_state_tensor)
        for state in tpu_states:
            assert torch.equal(state.vector, expected_vector)
        assert len(sync_batches) == 2 * len(tpu_states)
        assert all(wait is False for _, wait in sync_batches)
        assert sync_batches[1][0] == [tpu_state_int.vector]
        assert sync_batches[3][0] == [tpu_state_tensor.vector]


def test_dspark_draft_new_state_and_from_ring_use_mesh_init_ops() -> None:
    """Verify DsparkDraft new_state and from_ring match oracle and sync on TPU."""
    mesh = Mesh(np.array([jax.devices()[0]]), (AXIS_TP,))
    registered_specs: dict[str, bridge.OpRegistrationSpec] = {}
    sync_batches: list[tuple[list[torch.Tensor], bool]] = []
    sync_mod = importlib.import_module("torch_tpu._internal.sync")

    def fake_sync(tensors: Sequence[torch.Tensor], *, wait: bool = True) -> None:
        """Record synchronized tensor lists and wait flags."""
        sync_batches.append((list(tensors), wait))

    with (
        patch.object(draft, "kernel_mesh", return_value=(mesh, 0)),
        patch.object(
            dspark.DraftConfig,
            "from_checkpoint",
            return_value=dspark.DraftConfig(),
        ),
        patch.object(
            draft,
            "register_global_op",
            side_effect=_make_dspark_global_op_registrar(registered_specs),
        ),
        patch.object(sync_mod, "synchronize", side_effect=fake_sync),
        patch.object(torch, "device", side_effect=_fake_torch_device),
        patch.object(torch, "tensor", side_effect=_fake_torch_tensor),
        patch.object(torch, "zeros", side_effect=_fake_torch_zeros),
        patch.object(torch, "full", side_effect=_fake_torch_full),
    ):
        dspark_draft = draft.DsparkDraft(_TEST_MAX_POS, "unused", Path("unused"))
        cfg = dspark_draft.config
        expected_state = _expected_dspark_cold_state(cfg, _TEST_ANCHOR_TOKEN)
        cpu_anchor = torch.tensor([_TEST_ANCHOR_TOKEN], dtype=torch.int32)
        cpu_new = dspark_draft.new_state(cpu_anchor)
        assert cpu_new.cache.device.type == "cpu"
        assert torch.equal(cpu_new.cache, expected_state.cache)
        assert torch.equal(cpu_new.positions, expected_state.positions)
        assert torch.equal(cpu_new.vector, expected_state.vector)

        cpu_ring_cache = torch.ones(
            (1, cfg.layers, cfg.window, 2 * cfg.head_dim),
            dtype=torch.bfloat16,
        )
        cpu_ring_pos = torch.zeros((1, cfg.window), dtype=torch.int32)
        cpu_from_ring = dspark_draft.from_ring(
            cpu_ring_cache,
            cpu_ring_pos,
            _TEST_ANCHOR_TOKEN,
        )
        assert torch.equal(cpu_from_ring.vector, expected_state.vector)
        assert not registered_specs
        assert not sync_batches

        tpu_new = dspark_draft.new_state(_as_fake_tpu(cpu_anchor))
        tpu_from_ring = dspark_draft.from_ring(
            _as_fake_tpu(cpu_ring_cache),
            _as_fake_tpu(cpu_ring_pos),
            _TEST_ANCHOR_TOKEN,
        )
        dspark_draft.weights = {
            "probe": _as_fake_tpu(_REAL_TORCH_ZEROS((1,), dtype=torch.bfloat16)),
        }
        tpu_from_cpu_anchor = dspark_draft.new_state(cpu_anchor)

        assert registered_specs["pallas::k3_dspark_init_state"].in_specs == (P(),)
        assert registered_specs["pallas::k3_dspark_init_state"].out_specs == (
            P(AXIS_TP),
            P(),
            P(),
        )
        assert registered_specs["pallas::k3_dspark_init_vector"].in_specs == (P(),)
        assert registered_specs["pallas::k3_dspark_init_vector"].out_specs == (P(),)
        tpu_calls = (tpu_new, tpu_from_ring, tpu_from_cpu_anchor)
        assert len(sync_batches) == 2 * len(tpu_calls)
        assert all(wait is False for _, wait in sync_batches)
        _assert_dspark_state_matches(tpu_new, expected_state, sync_batches[1][0])
        assert torch.equal(tpu_from_ring.vector, expected_state.vector)
        assert sync_batches[3][0] == [
            tpu_from_ring.cache,
            tpu_from_ring.positions,
            tpu_from_ring.vector,
        ]
        _assert_dspark_state_matches(
            tpu_from_cpu_anchor,
            expected_state,
            sync_batches[5][0],
        )


def test_draft_compilation_calls_include_mesh_init_ops() -> None:
    """Verify DsparkDraft and MlaDsparkDraft compilation_calls include mesh init ops."""
    mesh = Mesh(np.array([jax.devices()[0]]), (AXIS_TP,))
    dspark_specs: dict[str, bridge.OpRegistrationSpec] = {}
    mla_specs: dict[str, bridge.OpRegistrationSpec] = {}
    control = decode_ops.Control(slots=4, rows=1, pages=1)

    with (
        patch.object(draft, "kernel_mesh", return_value=(mesh, 0)),
        patch.object(bridge, "kernel_mesh", return_value=(mesh, 0)),
        patch.object(
            dspark.DraftConfig,
            "from_checkpoint",
            return_value=dspark.DraftConfig(),
        ),
        patch.object(
            dspark_mla_mod.MlaDraftConfig,
            "from_checkpoint",
            return_value=dspark_mla_mod.MlaDraftConfig(),
        ),
        patch.object(
            draft,
            "register_global_op",
            side_effect=_make_dspark_global_op_registrar(dspark_specs),
        ),
        patch.object(
            bridge,
            "register_global_op",
            side_effect=_make_dspark_global_op_registrar(mla_specs),
        ),
    ):
        dspark_draft = draft.DsparkDraft(_TEST_MAX_POS, "unused", Path("unused"))
        dspark_draft.names = ()
        dspark_calls = dspark_draft.compilation_calls(control, ())

        mla_draft = draft_mla.MlaDsparkDraft(_TEST_MAX_POS, "unused", Path("unused"))
        mla_draft.names = ()
        mla_calls = mla_draft.compilation_calls(control, ())

    expected_dspark_ops = {
        "pallas::k3_dspark_init_state",
        "pallas::k3_dspark_init_vector",
    }
    assert set(dspark_specs) == expected_dspark_ops
    assert len(dspark_calls) == len(expected_dspark_ops)
    for call in dspark_calls:
        assert call.priority == 1
        assert len(call.operands) == 1
        operand = call.operands[0]
        assert isinstance(operand, torch.Tensor)
        assert operand.shape == (1,)
        assert operand.dtype == torch.int32

    assert set(mla_specs) == {"pallas::k3_dspark_mla_init_state"}
    assert len(mla_calls) == 1
    assert mla_calls[0].priority == 1
    assert len(mla_calls[0].operands) == 1
    mla_operand = mla_calls[0].operands[0]
    assert isinstance(mla_operand, torch.Tensor)
    assert mla_operand.shape == (1,)
    assert mla_operand.dtype == torch.int32


def test_kda_importer_synchronizes_tpu_state_and_ring_transfers() -> None:
    """Verify KdaImporter synchronizes TPU state and ring import buffers."""
    mesh = Mesh(np.array([jax.devices()[0]]), (AXIS_TP,))
    importer = pools.KdaImporter(mesh, attention_dp=1, num_slots=4)
    raw_pool = _as_fake_tpu(
        torch.zeros(
            (_POOL_BLOCKS, *k3_layout.LAYOUT.block_shape), dtype=torch.bfloat16
        ),
    )
    importer.raw_pools = [raw_pool]
    importer.kda_raw = [0] * pools.KDA_LAYERS

    conv_bank = _as_fake_tpu(
        torch.zeros(
            (
                1,
                4,
                pools.KDA_LAYERS,
                CONV_TAPS,
                pools.CONV_CHANNEL_GROUPS,
                CONV_HEAD_ROWS,
                HEAD_DIM,
            ),
            dtype=torch.bfloat16,
        ),
    )
    rec_bank = _as_fake_tpu(
        torch.zeros(
            (1, 4, pools.KDA_LAYERS, KDA_HEADS, HEAD_DIM, HEAD_DIM),
            dtype=torch.float32,
        ),
    )
    kda_blocks = _as_fake_tpu(torch.ones((pools.KDA_LAYERS,), dtype=torch.int32))
    events: list[str] = []
    sync_batches: list[tuple[list[torch.Tensor], bool]] = []
    sync_mod = importlib.import_module("torch_tpu._internal.sync")

    def fake_register_op(
        name: str,
        _mesh: Mesh,
        _body: Callable[..., object],
        _spec: bridge.OpRegistrationSpec,
    ) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
        """Return a stub TPU op recording invocation order relative to synchronize."""

        def op_runner(*args: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Record custom op execution and return fake TPU outputs."""
            if "import_kda" in name:
                events.append("state_op")
                return (_as_fake_tpu(args[0].clone()), _as_fake_tpu(args[1].clone()))
            events.append("ring_op")
            cfg = dspark.DraftConfig()
            ring_out = _as_fake_tpu(
                torch.zeros(
                    (1, cfg.layers, cfg.window, 2 * cfg.head_dim),
                    dtype=torch.bfloat16,
                ),
            )
            pos_out = _as_fake_tpu(torch.full((1, cfg.window), -1, dtype=torch.int32))
            return (ring_out, pos_out)

        return op_runner

    def fake_sync(tensors: Sequence[torch.Tensor], *, wait: bool = True) -> None:
        """Record synchronize call ordering and arguments."""
        events.append("sync")
        sync_batches.append((list(tensors), wait))

    with (
        patch.object(bridge, "register_op", side_effect=fake_register_op),
        patch.object(sync_mod, "synchronize", side_effect=fake_sync),
        patch.object(torch, "tensor", side_effect=_fake_torch_tensor),
    ):
        conv_out, rec_out = importer.import_state(
            conv_bank,
            rec_bank,
            kda_blocks,
            slot=2,
        )
        ring_out, pos_out = importer.import_ring(
            dspark.DraftConfig(),
            kda_blocks,
            length=_TEST_RING_LEN,
        )

    assert events == ["sync", "state_op", "sync", "sync", "ring_op", "sync"]
    assert len(sync_batches) == events.count("sync")
    assert all(wait is False for _, wait in sync_batches)
    assert sync_batches[1][0] == [conv_out, rec_out]
    assert sync_batches[3][0] == [ring_out, pos_out]


def _make_kv_groups() -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[shim_program.KVGroup, shim_program.KVGroup],
]:
    """Return MLA and KDA layer names and matching KVGroup descriptors."""
    mla_layers = ("model.layers.0.self_attn",)
    kda_layers = tuple(
        f"model.layers.{layer_idx}.self_attn"
        for layer_idx in range(1, pools.KDA_LAYERS + 1)
    )
    kv_groups = (
        shim_program.KVGroup(layer_names=mla_layers, block_size=256, table_width=1),
        shim_program.KVGroup(layer_names=kda_layers, block_size=256, table_width=1),
    )
    return mla_layers, kda_layers, kv_groups


def test_decoder_admit_zero_kda_routes_through_tpu_importer() -> None:
    """Verify KimiDecoder cold admission at computed=0 uses KdaImporter on TPU."""
    config = MegakernelEngineConfig(
        role="decode",
        tensor_parallel_size=32,
        decode_context_parallel_size=32,
        block_size=256,
        max_num_seqs=1,
        max_model_len=8192,
        max_num_batched_tokens=8,
    )
    conv_slot = _as_fake_tpu(torch.ones((1, 4, 2), dtype=torch.bfloat16))
    rec_slot = _as_fake_tpu(torch.ones((1, 4, 2), dtype=torch.float32))
    conv_imported = _as_fake_tpu(torch.zeros((1, 4, 2), dtype=torch.bfloat16))
    rec_imported = _as_fake_tpu(torch.zeros((1, 4, 2), dtype=torch.float32))
    imported_calls: list[tuple[torch.Tensor, int]] = []
    sync_batches: list[tuple[list[torch.Tensor], bool]] = []
    sync_mod = importlib.import_module("torch_tpu._internal.sync")

    def fake_import_state(
        conv: torch.Tensor,
        recurrent: torch.Tensor,
        kda_blocks: torch.Tensor,
        slot: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Record zero-block KdaImporter call and return fresh slot buffers."""
        assert conv is conv_slot
        assert recurrent is rec_slot
        imported_calls.append((kda_blocks, int(slot)))
        return (conv_imported, rec_imported)

    def fake_sync(tensors: Sequence[torch.Tensor], *, wait: bool = True) -> None:
        """Record synchronized tensor lists and wait flags."""
        sync_batches.append((list(tensors), wait))

    mla_layers, kda_layers, kv_groups = _make_kv_groups()
    importer_stub = _StubKdaImporter(
        import_state=fake_import_state,
        mla_layer_names=mla_layers,
        kda_layer_names=kda_layers,
    )
    ops_stub = cast(
        "decode_ops.DecodeOps",
        _AdmitDecodeOpsDouble(
            rank=0,
            num_slots=4,
            attention_dp=1,
            batch_sizes=(1,),
            slots=[conv_slot, rec_slot],
            importer=importer_stub,
        ),
    )
    decoder = decode.KimiDecoder(config, ops_stub, tp_rank=0)
    request = shim_program.RequestInput(
        req_id="req-zero",
        computed=0,
        scheduled=1,
        token_ids=np.array([1], dtype=np.int32),
        num_tokens=1,
        num_prompt_tokens=1,
        block_tables=(
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
        ),
        sampling=shim_program.GREEDY,
    )

    with (
        patch.object(decode, "tp_rank_permutation", return_value=None),
        patch.object(sync_mod, "synchronize", side_effect=fake_sync),
        patch.object(torch, "zeros", side_effect=_fake_torch_zeros),
        patch.object(torch, "ones", side_effect=_fake_torch_ones),
        patch.object(torch, "from_numpy", side_effect=_fake_torch_from_numpy),
    ):
        decoder.bind_kv_caches({}, kv_groups)
        output = decoder.step(shim_program.StepInput(requests=[request]))

    assert output.tokens.shape == (1, decode.KimiDecoder.WIDTH)
    assert len(imported_calls) == 1
    blocks_arg, slot_arg = imported_calls[0]
    assert blocks_arg.shape == (pools.KDA_LAYERS,)
    assert int(blocks_arg.abs().max().item()) == 0
    assert slot_arg == 1
    assert float(conv_slot[0, 1].min().item()) == 1.0
    assert decoder.ops.slots == [conv_imported, rec_imported]
    assert all(wait is False for _, wait in sync_batches)
    assert any(
        any(tensor is blocks_arg for tensor in batch) for batch, _ in sync_batches
    )


def test_decoder_zero_kda_tpu_slot_fails_closed_without_importer() -> None:
    """Verify cold TPU zero-slot admission raises RuntimeError without importer."""
    config = MegakernelEngineConfig(
        role="decode",
        tensor_parallel_size=32,
        decode_context_parallel_size=32,
        block_size=256,
        max_num_seqs=1,
        max_model_len=8192,
        max_num_batched_tokens=8,
    )
    conv_slot = _as_fake_tpu(torch.ones((1, 4, 2), dtype=torch.bfloat16))
    rec_slot = _as_fake_tpu(torch.ones((1, 4, 2), dtype=torch.float32))
    sync_mod = importlib.import_module("torch_tpu._internal.sync")

    def unused_import_state(
        conv: torch.Tensor,
        recurrent: torch.Tensor,
        _kda_blocks: torch.Tensor,
        _slot: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Unused callback when importer is dropped before step."""
        return conv, recurrent

    mla_layers, kda_layers, kv_groups = _make_kv_groups()
    importer_stub = _StubKdaImporter(
        import_state=unused_import_state,
        mla_layer_names=mla_layers,
        kda_layer_names=kda_layers,
    )
    ops_stub = cast(
        "decode_ops.DecodeOps",
        _AdmitDecodeOpsDouble(
            rank=0,
            num_slots=4,
            attention_dp=1,
            batch_sizes=(1,),
            slots=[conv_slot, rec_slot],
            importer=importer_stub,
        ),
    )
    decoder = _UnboundImporterDecoder(config, ops_stub, tp_rank=0)
    request = shim_program.RequestInput(
        req_id="req-unbound",
        computed=0,
        scheduled=1,
        token_ids=np.array([1], dtype=np.int32),
        num_tokens=1,
        num_prompt_tokens=1,
        block_tables=(
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
        ),
        sampling=shim_program.GREEDY,
    )

    with (
        patch.object(decode, "tp_rank_permutation", return_value=None),
        patch.object(sync_mod, "synchronize"),
        patch.object(torch, "zeros", side_effect=_fake_torch_zeros),
        patch.object(torch, "ones", side_effect=_fake_torch_ones),
        patch.object(torch, "from_numpy", side_effect=_fake_torch_from_numpy),
    ):
        decoder.bind_kv_caches({}, kv_groups)
        decoder.drop_importer()
        with pytest.raises(RuntimeError, match="KDA importer is not bound"):
            decoder.step(shim_program.StepInput(requests=[request]))

    assert float(conv_slot[0, 1].min().item()) == 1.0


def test_decoder_continuing_admit_and_batched_spec_synchronize_tpu_inputs() -> None:
    """Verify continuing KDA blocks and batched-spec modes synchronize on TPU."""
    config = MegakernelEngineConfig(
        role="decode",
        tensor_parallel_size=32,
        decode_context_parallel_size=32,
        block_size=256,
        max_num_seqs=1,
        max_model_len=8192,
        max_num_batched_tokens=8,
        num_speculative_tokens=3,
    )
    needed_slots = decode_ops.Control.total_slots(2 * config.max_num_seqs + 1)
    conv_slot = _as_fake_tpu(torch.ones((1, needed_slots, 2), dtype=torch.bfloat16))
    rec_slot = _as_fake_tpu(torch.ones((1, needed_slots, 2), dtype=torch.float32))
    imported_blocks: list[torch.Tensor] = []
    sync_batches: list[tuple[list[torch.Tensor], bool]] = []
    events: list[str] = []
    sync_mod = importlib.import_module("torch_tpu._internal.sync")

    def fake_import_state(
        conv: torch.Tensor,
        recurrent: torch.Tensor,
        kda_blocks: torch.Tensor,
        _slot: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Record continuing-request KDA block tensor and event ordering."""
        events.append("import_state")
        imported_blocks.append(kda_blocks)
        return conv, recurrent

    def fake_sync(tensors: Sequence[torch.Tensor], *, wait: bool = True) -> None:
        """Record synchronized tensors, wait flag, and event ordering."""
        events.append("sync")
        sync_batches.append((list(tensors), wait))

    mla_layers, kda_layers, kv_groups = _make_kv_groups()
    importer_stub = _StubKdaImporter(
        import_state=fake_import_state,
        mla_layer_names=mla_layers,
        kda_layer_names=kda_layers,
    )
    ops_double = _AdmitDecodeOpsDouble(
        rank=0,
        num_slots=needed_slots,
        attention_dp=1,
        batch_sizes=(1,),
        slots=[conv_slot, rec_slot],
        importer=importer_stub,
        events=events,
    )
    decoder = decode.KimiDecoder(
        config,
        cast("decode_ops.DecodeOps", ops_double),
        cast("draft.DsparkDraft", _StubSpecDraft()),
        tp_rank=0,
    )
    expected_block = 2
    continuing_request = shim_program.RequestInput(
        req_id="req-continuing",
        computed=1,
        scheduled=1,
        token_ids=np.array([10, 20], dtype=np.int32),
        num_tokens=2,
        num_prompt_tokens=2,
        block_tables=(
            np.array([1], dtype=np.int32),
            np.array([expected_block], dtype=np.int32),
        ),
        sampling=shim_program.GREEDY,
    )

    with (
        patch.object(decode, "tp_rank_permutation", return_value=None),
        patch.object(sync_mod, "synchronize", side_effect=fake_sync),
        patch.object(torch, "tensor", side_effect=_fake_torch_tensor),
        patch.object(torch, "zeros", side_effect=_fake_torch_zeros),
        patch.object(torch, "ones", side_effect=_fake_torch_ones),
        patch.object(torch, "from_numpy", side_effect=_fake_torch_from_numpy),
    ):
        decoder.bind_kv_caches({}, kv_groups)
        decoder.step(shim_program.StepInput(requests=[continuing_request]))

    assert len(imported_blocks) == 1
    kda_blocks_arg = imported_blocks[0]
    assert kda_blocks_arg.shape == (pools.KDA_LAYERS,)
    assert int(kda_blocks_arg.min().item()) == expected_block
    assert int(kda_blocks_arg.max().item()) == expected_block
    assert len(ops_double.recorded_modes) == 1
    modes_arg = ops_double.recorded_modes[0]
    assert modes_arg.shape == (1,)
    assert all(wait is False for _, wait in sync_batches)
    assert any(
        any(tensor is kda_blocks_arg for tensor in batch) for batch, _ in sync_batches
    )
    assert any(
        any(tensor is modes_arg for tensor in batch) for batch, _ in sync_batches
    )
    import_idx = events.index("import_state")
    verify_idx = events.index("verify_batched")
    assert "sync" in events[:import_idx]
    assert "sync" in events[import_idx:verify_idx]


_DRAFT_STATE_BLOCKS = 15


def test_spec_schedule_validation_and_prefill_draft_ring_seam() -> None:
    """Verify speculation schedule rules and prefill-to-draft ring pool seam."""
    base_cfg = MegakernelEngineConfig(
        role="decode",
        tensor_parallel_size=32,
        decode_context_parallel_size=32,
        block_size=256,
        max_num_seqs=16,
        max_model_len=8192,
        max_num_batched_tokens=64,
        num_speculative_tokens=7,
        num_speculative_tokens_per_batch_size=((1, 8, 7), (9, 16, 3)),
    )
    caps = program.capabilities(base_cfg)
    assert caps.speculation == "dspark"

    bad_schedules = (
        (((2, 8, 7),), "first range must start at batch size 1"),
        (((1, 6, 7), (4, 8, 3)), "non-overlapping"),
        (((4, 2, 7),), "1 <= min <= max"),
        (((1, 16, 7),), "schedule gives 7 drafts at batch size 9"),
    )
    for schedule, match in bad_schedules:
        bad_cfg = MegakernelEngineConfig(
            role="decode",
            tensor_parallel_size=32,
            decode_context_parallel_size=32,
            block_size=256,
            max_num_seqs=16,
            max_model_len=8192,
            max_num_batched_tokens=64,
            num_speculative_tokens=7,
            num_speculative_tokens_per_batch_size=schedule,
        )
        with pytest.raises(NotImplementedError, match=match):
            program.capabilities(bad_cfg)

    with pytest.raises(ValueError, match="supports auto, float32, or bfloat16"):
        program.recurrent_state_dtype("float16")

    draft_region = next(r for r in k3_layout.LAYOUT.state_regions if r.name == "draft")
    assert draft_region.rows == _DRAFT_STATE_BLOCKS
    draft_cfg = dspark.DraftConfig()
    assert tuple(draft_cfg.aux_layers) == draft_ring.AUX_LAYERS
    ring = jax.numpy.zeros(
        (1, draft_cfg.layers, draft_cfg.window, 2 * draft_cfg.head_dim),
        dtype=jax.numpy.bfloat16,
    )
    meta = jax.numpy.array([[64, 0]], dtype=jax.numpy.int32)
    kda_blocks = len(k3_layout.LAYOUT.state_layers)
    packed = dspark.pack_ring(ring, meta, blocks=kda_blocks)
    assert packed.shape[1] == kda_blocks
