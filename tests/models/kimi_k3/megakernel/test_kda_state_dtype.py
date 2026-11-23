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

from vllm_torchtpu.models.kimi_k3.megakernel import (
    config as kernel_config,
)
from vllm_torchtpu.models.kimi_k3.megakernel import (
    decode,
    decode_ops,
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
from vllm_torchtpu.models.kimi_k3.megakernel.layout import (
    CONV_HEAD_ROWS,
    CONV_TAPS,
    HEAD_DIM,
    KDA_HEADS,
)
from vllm_torchtpu.models.kimi_k3.megakernel.runtime_shim import backend, bridge
from vllm_torchtpu.models.kimi_k3.megakernel.runtime_shim.config import (
    MegakernelEngineConfig,
)

if TYPE_CHECKING:
    from collections.abc import Callable

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
    """Run ``pl.pallas_call`` in CPU interpret mode, overriding ``interpret``."""
    # Override interpret in-place so mesh_axis_index(..., interpret=False) does
    # not raise duplicate-keyword TypeError, while keeping generic CPU interpret
    # mode required by BitcastTransform inside pool_adapters.gather_region.
    merged_kwargs = dict(kwargs)
    merged_kwargs["interpret"] = True
    return cast("Callable[..., object]", _REAL_PALLAS_CALL)(*args, **merged_kwargs)


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
    importer.bind(_make_full_unified_caches(raw_pool), "auto")

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
        patch.object(pools, "register_op", side_effect=fake_register_op),
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


def _make_full_unified_caches(raw_pool: torch.Tensor) -> dict[str, torch.Tensor]:
    """Build a 24-pool, 93-layer cache mapping with ``raw_pool`` on pool 0."""
    pool_list = [raw_pool] + [
        torch.zeros_like(raw_pool) for _ in range(pools.MLA_LAYERS - 1)
    ]
    caches: dict[str, torch.Tensor] = {}
    for mla_idx, layer in enumerate(k3_layout.MLA):
        caches[f"model.layers.{layer}.self_attn"] = pool_list[mla_idx]
    for kda_idx, layer in enumerate(k3_layout.KDA):
        caches[f"model.layers.{layer}.self_attn"] = pool_list[kda_idx // 3]
    return caches


def test_kda_importer_rebind_clears_cache_and_keys_ring_op_on_draft_config() -> None:
    """Verify KdaImporter requires bind, clears cached ops on rebind, and keys rings."""
    mesh = Mesh(np.array([jax.devices()[0]]), (AXIS_TP,))
    importer = pools.KdaImporter(mesh, attention_dp=1, num_slots=4)
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
    kda_blocks = torch.zeros((pools.KDA_LAYERS,), dtype=torch.int32)
    cfg_a = dspark.DraftConfig(window=2048)
    cfg_b = dspark.DraftConfig(window=1024)

    with pytest.raises(RuntimeError, match="bind"):
        importer.import_state(conv_bank, rec_bank, kda_blocks, slot=1)
    with pytest.raises(RuntimeError, match="bind"):
        importer.import_ring(cfg_a, kda_blocks, length=64)

    pool_a = torch.zeros(
        (_POOL_BLOCKS, *k3_layout.LAYOUT.block_shape), dtype=torch.bfloat16
    )
    pool_b = torch.zeros(
        (_POOL_BLOCKS, *k3_layout.LAYOUT.block_shape), dtype=torch.bfloat16
    )
    registered: list[tuple[str, str | None]] = []

    def _fake_register_sharded_op(
        name: str,
        _fn: object,
        _mesh: Mesh,
        spec: bridge.OpRegistrationSpec,
        _configure_callable: object = None,
    ) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
        """Record each op registration and return a dummy tensor-pair runner."""
        registered.append((name, spec.export_key))
        return lambda *_args: (torch.zeros(1), torch.zeros(1))

    with patch.object(
        bridge, "_register_sharded_op", side_effect=_fake_register_sharded_op
    ):
        importer.bind(_make_full_unified_caches(pool_a), "auto")
        importer.import_state(conv_bank, rec_bank, kda_blocks, slot=1)
        importer.import_state(conv_bank, rec_bank, kda_blocks, slot=2)
        importer.import_ring(cfg_a, kda_blocks, length=64)
        importer.import_ring(cfg_a, kda_blocks, length=128)
        importer.import_ring(cfg_b, kda_blocks, length=64)

        first_bind_count = 3
        assert len(registered) == first_bind_count
        ring_a_name, ring_a_key = registered[1]
        ring_b_name, ring_b_key = registered[2]
        assert ring_a_name != ring_b_name
        assert ring_a_key != ring_b_key

        importer.bind(_make_full_unified_caches(pool_b), "auto")
        importer.import_state(conv_bank, rec_bank, kda_blocks, slot=1)
        importer.import_ring(cfg_a, kda_blocks, length=64)

        rebound_count = 5
        assert len(registered) == rebound_count
