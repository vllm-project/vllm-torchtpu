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
"""State-storage selection, pool round-trip, and verifier row selection tests."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.experimental import pallas as pl
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.models.kimi_k3.megakernel import (
    decode,
    decode_ops,
    draft,
    draft_mla,
    pools,
    program,
)
from vllm_torchtpu.models.kimi_k3.megakernel.kernels import (
    decode_megakernel as kernel,
)
from vllm_torchtpu.models.kimi_k3.megakernel.kernels import (
    dspark as dspark_kernel,
)
from vllm_torchtpu.models.kimi_k3.megakernel.kernels import (
    dspark_mla as dspark_mla_mod,
)
from vllm_torchtpu.models.kimi_k3.megakernel.runtime_shim import (
    backend,
    bridge,
)
from vllm_torchtpu.models.kimi_k3.megakernel.runtime_shim import (
    program as shim_program,
)
from vllm_torchtpu.models.kimi_k3.megakernel.runtime_shim.config import (
    MegakernelEngineConfig,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

_PROJECTED_DIM = 1664
_ATTENDED_DIM = 384
_HEAD_DIM = 128
_NUM_HEADS = 3
_CONV_WINDOW = 4
_CONV_TAPS = 3
_RNG_SEED = 413
_POOL_BLOCKS = 4
_POOL_ROWS = 96
_POOL_COLS = 1280
_SSM_FLOATS = _NUM_HEADS * _HEAD_DIM * _HEAD_DIM
_CONV_BF16S = _CONV_TAPS * _NUM_HEADS * _NUM_HEADS * _HEAD_DIM
_TEST_ANCHOR_TOKEN = 17
_TEST_MAX_POS = 1024
_TEST_RING_LEN = 12
_EXPECTED_IMPORTER_SYNC_CALLS = 4
_REAL_TORCH_TENSOR = torch.tensor
_REAL_TORCH_ZEROS = torch.zeros
_REAL_TORCH_FULL = torch.full
_REAL_TORCH_DEVICE = torch.device


class _CpuTpuCapableTensor(torch.Tensor):
    """CPU tensor that upgrades to _FakeTpuTensor when .to('tpu') is called."""

    def to(self, *args: object, **kwargs: object) -> torch.Tensor:
        """Upgrade to _FakeTpuTensor when target device is TPU, else stay on CPU."""
        target = kwargs.get("device")
        for arg in args:
            if isinstance(arg, (torch.device, str, SimpleNamespace)):
                target = arg
        is_tpu = target == "tpu" or getattr(target, "type", None) == "tpu"
        clean_kwargs = {k: v for k, v in kwargs.items() if k != "device"}
        clean_args = tuple(
            arg
            for arg in args
            if not isinstance(arg, (torch.device, str, SimpleNamespace))
        )
        converted = super().to(*clean_args, **clean_kwargs)
        if is_tpu:
            return converted.as_subclass(_FakeTpuTensor)
        return converted.as_subclass(torch.Tensor)


class _FakeTpuTensor(_CpuTpuCapableTensor):
    """CPU-backed tensor double reporting device.type == 'tpu'."""

    @property
    def device(self) -> SimpleNamespace:
        """Return a synthetic TPU device namespace."""
        return SimpleNamespace(type="tpu")

    def to(self, *args: object, **kwargs: object) -> _FakeTpuTensor:
        """Convert dtype on CPU while preserving _FakeTpuTensor."""
        clean_kwargs = {k: v for k, v in kwargs.items() if k != "device"}
        clean_args = tuple(
            arg
            for arg in args
            if not isinstance(arg, (torch.device, str, SimpleNamespace))
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
        memory_format: torch.memory_format = torch.preserve_format,
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
    return device == "tpu" or getattr(device, "type", None) == "tpu"


def _fake_torch_device(device: object) -> object:
    """Return a synthetic TPU device namespace for 'tpu', else a torch.device."""
    if _is_tpu_device(device):
        return SimpleNamespace(type="tpu")
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


def _jax_to_fake_tpu(arr: jax.Array) -> _FakeTpuTensor:
    """Convert a JAX array to a CPU-backed _FakeTpuTensor."""
    np_arr = np.array(arr, copy=True)
    if np_arr.dtype.name == "bfloat16":
        tensor = torch.from_numpy(np_arr.view(np.uint16)).view(torch.bfloat16)
    else:
        tensor = torch.from_numpy(np_arr)
    return _as_fake_tpu(tensor)


def _assert_raises(
    expected_exc: type[BaseException],
    match: str,
    fn: Callable[[], object],
) -> None:
    """Assert that calling fn raises expected_exc containing match."""
    try:
        fn()
    except expected_exc as exc:
        if match not in str(exc):
            msg = f"Expected {match!r} in {exc!r}"
            raise AssertionError(msg) from exc
        return
    msg = f"Expected {expected_exc.__name__} matching {match!r}"
    raise AssertionError(msg)


def test_engine_config_passes_state_dtype_to_decoder() -> None:
    """Verify MegakernelEngineConfig forwards mamba_ssm_cache_dtype to DecodeOps."""
    cases = (
        ("auto", "float32"),
        ("float32", "float32"),
        ("bfloat16", "bfloat16"),
    )
    for setting, expected in cases:
        captured: dict[str, str] = {}

        def _fake_decode_ops(
            _path: Path,
            *,
            out_box: dict[str, str] = captured,
            **kwargs: object,
        ) -> SimpleNamespace:
            """Capture keyword arguments passed to DecodeOps."""
            out_box["dtype"] = str(kwargs["mamba_ssm_cache_dtype"])
            return SimpleNamespace(**kwargs)

        def _fake_decoder(
            _config: object,
            ops_obj: object,
            _draft: object,
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
                SimpleNamespace(
                    parallel_config=SimpleNamespace(
                        tensor_parallel_size=32,
                        decode_context_parallel_size=16,
                        cp_kv_cache_interleave_size=256,
                    ),
                    scheduler_config=SimpleNamespace(
                        max_num_seqs=128,
                        max_num_batched_tokens=128,
                        async_scheduling=True,
                    ),
                    model_config=SimpleNamespace(max_model_len=139264),
                    cache_config=SimpleNamespace(
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


def test_unsupported_state_dtype_rejected_before_allocation() -> None:
    """Reject unsupported mamba_ssm_cache_dtype strings before buffer allocation."""
    for setting in ("float16", "fp8", "invalid"):
        _assert_raises(
            ValueError,
            "mamba-ssm-cache-dtype",
            lambda s=setting: decode_ops.DecodeOps(
                Path("unused"),
                num_slots=2,
                mamba_ssm_cache_dtype=s,
            ),
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
        no_copy = SimpleNamespace(wait=lambda: None)
        weights = SimpleNamespace(
            gate_projection=jnp.zeros((_HEAD_DIM, _HEAD_DIM), jnp.bfloat16),
            input_projection=jnp.zeros((_HEAD_DIM, _PROJECTED_DIM), jnp.bfloat16),
            feature_gate_projection_vmem=feature_ref,
            exp_log_decay_rate=jnp.zeros((_NUM_HEADS, 1), jnp.float32),
            time_step_bias=jnp.full((_NUM_HEADS, _HEAD_DIM), -3.0, jnp.float32),
            convolution_filter=jnp.full(
                (_CONV_WINDOW, _NUM_HEADS, _NUM_HEADS, _HEAD_DIM),
                0.25,
                jnp.bfloat16,
            ),
            output_normalization=jnp.ones((_HEAD_DIM,), jnp.bfloat16),
            make_projection_copies=lambda: SimpleNamespace(
                gate=no_copy,
                feature_gate=no_copy,
            ),
        )
        kda_fn(
            jnp.zeros((batch, _HEAD_DIM), jnp.bfloat16),
            jnp.zeros((batch, _HEAD_DIM), jnp.bfloat16),
            next_history,
            next_state,
            weights=weights,
            convolution_history_ref=history_ref,
            recurrent_state_ref=state_ref,
            communication_workspace=SimpleNamespace(
                reduction_phase=jnp.array([0], jnp.int32)
            ),
            synchronize_devices=False,
            weights_are_prefetched=True,
            sequence_rows=sequence_rows,
            shard_index=0,
            first_call=False,
            attended_out=output,
            gate_features_vmem=jnp.zeros((batch, _HEAD_DIM), jnp.bfloat16),
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
        jnp.zeros((_HEAD_DIM, _ATTENDED_DIM), jnp.bfloat16),
    )


def test_verifier_state_matches_sequential_decode() -> None:
    """Confirm batched KDA recurrence matches step-by-step sequential decode."""
    for dtype in (jnp.float32, jnp.bfloat16):
        for width in (4, 8):
            rng = np.random.default_rng(_RNG_SEED)
            requests = 2
            projected = jnp.asarray(
                rng.normal(size=(requests, width, _PROJECTED_DIM)),
                jnp.bfloat16,
            )
            history = jnp.asarray(
                rng.normal(
                    size=(requests, _NUM_HEADS, _NUM_HEADS, _CONV_WINDOW, _HEAD_DIM)
                ),
                jnp.bfloat16,
            )
            state = jnp.asarray(
                rng.normal(size=(requests, _NUM_HEADS, _HEAD_DIM, _HEAD_DIM)),
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
                np.testing.assert_array_equal(
                    batched[0][offset::width, :, :, :_NUM_HEADS],
                    cur_history[:, :, :, :_NUM_HEADS],
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
    _assert_raises(
        NotImplementedError,
        "decode-only",
        lambda: program.check(config),
    )


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
    opts = pools.decode_step_options(8, attention_dp=expected_dp_group)
    assert opts["attention_dp"] == expected_dp_group


def test_kda_importer_round_trip_bit_exact() -> None:
    """Verify KdaImporter CPU unpack/import round-trip has zero max_abs_diff."""
    raw_pool = torch.zeros(
        (_POOL_BLOCKS, _POOL_ROWS, _POOL_COLS),
        dtype=torch.bfloat16,
    )
    gen = torch.Generator().manual_seed(_RNG_SEED)
    rec_expected = torch.randn(
        (_NUM_HEADS, _HEAD_DIM, _HEAD_DIM),
        generator=gen,
        dtype=torch.float32,
    )
    conv_expected = torch.randn(
        (_CONV_TAPS, _NUM_HEADS, _NUM_HEADS, _HEAD_DIM),
        generator=gen,
        dtype=torch.float32,
    ).to(torch.bfloat16)

    block_bytes = raw_pool[1].view(torch.uint8)
    ssm_slice = block_bytes[: pools.SSM_ROWS].view(torch.float32).reshape(-1)
    ssm_slice[:_SSM_FLOATS] = rec_expected.reshape(-1)
    conv_slice = (
        block_bytes[pools.SSM_ROWS : pools.SSM_ROWS + pools.CONV_ROWS]
        .view(torch.bfloat16)
        .reshape(-1)
    )
    conv_slice[:_CONV_BF16S] = conv_expected.reshape(-1)

    conv_unpacked, rec_unpacked = pools.unpack_kda_state(raw_pool, 1)
    max_conv_diff = float(
        (conv_unpacked[:, :, :_NUM_HEADS, :].float() - conv_expected.float())
        .abs()
        .max()
        .item()
    )
    max_rec_diff = float((rec_unpacked - rec_expected).abs().max().item())
    assert max_conv_diff == 0.0
    assert max_rec_diff == 0.0

    mesh = Mesh(np.array([jax.devices()[0]]), (pools.MESH_AXIS,))
    importer = pools.KdaImporter(mesh, attention_dp=1, num_slots=4)
    importer.raw_pools = [raw_pool]
    importer.kda_raw = [0] * pools.KDA_LAYERS
    conv_bank = torch.zeros(
        (4, pools.KDA_LAYERS, _CONV_TAPS, _NUM_HEADS, _CONV_WINDOW, _HEAD_DIM),
        dtype=torch.bfloat16,
    )
    rec_bank = torch.zeros(
        (4, pools.KDA_LAYERS, _NUM_HEADS, _HEAD_DIM, _HEAD_DIM),
        dtype=torch.float32,
    )
    kda_blocks = torch.ones((pools.KDA_LAYERS,), dtype=torch.int32)
    conv_after, rec_after = importer.import_state(
        conv_bank,
        rec_bank,
        kda_blocks,
        slot=2,
    )
    bank_conv_diff = float(
        (conv_after[2, 0, :, :, :_NUM_HEADS, :].float() - conv_expected.float())
        .abs()
        .max()
        .item()
    )
    bank_rec_diff = float((rec_after[2, 0] - rec_expected).abs().max().item())
    assert bank_conv_diff == 0.0
    assert bank_rec_diff == 0.0


def test_select_verify_kda_rows_matches_accepted_token_offsets() -> None:
    """Verify select_verify_kda_rows selects exact source rows and target slots."""
    slots_dp1 = jnp.array([3, 3, 3, 3, 0, 0, 0, 0], dtype=jnp.int32)
    accepted_dp1 = jnp.array([2, 1], dtype=jnp.int32)
    sources_dp1, targets_dp1 = pools.select_verify_kda_rows(
        slots_dp1,
        accepted_dp1,
        requests=2,
        rows_per=4,
        attention_dp=1,
        bank_slots=64,
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
        requests=4,
        rows_per=4,
        attention_dp=2,
        bank_slots=64,
        group=0,
    )
    sources_g1, targets_g1 = pools.select_verify_kda_rows(
        slots_dp2,
        accepted_dp2,
        requests=4,
        rows_per=4,
        attention_dp=2,
        bank_slots=64,
        group=1,
    )
    assert sources_g0.tolist() == [1, 6, 9, 14]
    assert targets_g0.tolist() == [1, 2, 33, 34]
    assert sources_g1.tolist() == [3, 7, 11, 15]
    assert targets_g1.tolist() == [1, 0, 33, 32]
    _assert_raises(
        TypeError,
        "Unexpected select_verify_kda_rows options",
        lambda: pools.select_verify_kda_rows(
            slots_dp1,
            accepted_dp1,
            requests=2,
            rows_per=4,
            unknown_option=1,
        ),
    )


def test_mla_draft_new_state_uses_mesh_init_op_and_matches_cpu() -> None:
    """Verify MlaDsparkDraft.new_state uses mesh register_global_op on TPU."""
    mesh = Mesh(np.array([jax.devices()[0]]), (draft_mla.AXIS,))
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
        expected_cpu = draft_mla.make_draft_state(_TEST_ANCHOR_TOKEN, device="cpu")
        assert torch.equal(cpu_state.vector, expected_cpu.vector)
        assert "pallas::k3_dspark_mla_init_state" not in registered_specs

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
        assert torch.equal(tpu_state_int.vector, expected_cpu.vector)
        assert torch.equal(tpu_state_tensor.vector, expected_cpu.vector)
        assert all(wait is False for _, wait in sync_batches)


def test_dspark_draft_new_state_and_from_ring_use_mesh_init_ops() -> None:
    """Verify DsparkDraft new_state and from_ring run mesh init ops on TPU."""
    mesh = Mesh(np.array([jax.devices()[0]]), (draft.AXIS,))
    registered_specs: dict[str, bridge.OpRegistrationSpec] = {}
    sync_batches: list[tuple[list[torch.Tensor], bool]] = []
    sync_mod = importlib.import_module("torch_tpu._internal.sync")

    def fake_register_global_op(
        name: str,
        _mesh: Mesh,
        fn: Callable[..., Sequence[jax.Array]],
        spec: bridge.OpRegistrationSpec,
    ) -> Callable[..., tuple[torch.Tensor, ...]]:
        """Capture global op registration and slice TP-sharded outputs to 1 rank."""
        registered_specs[name] = spec

        def runner(*args: torch.Tensor) -> tuple[torch.Tensor, ...]:
            """Run global JAX init op and shard P(AXIS) outputs to local rank 0."""
            jax_args = tuple(jnp.asarray(arg.detach().cpu().numpy()) for arg in args)
            raw_outs = fn(*jax_args)
            sliced: list[torch.Tensor] = []
            for out, out_spec in zip(raw_outs, spec.out_specs, strict=True):
                if out_spec == P(draft.AXIS):
                    assert out.shape[0] == program.TP_SIZE
                    local = out[:1]
                else:
                    local = out
                sliced.append(_jax_to_fake_tpu(local))
            return tuple(sliced)

        return runner

    def fake_sync(tensors: Sequence[torch.Tensor], *, wait: bool = True) -> None:
        """Record synchronized tensor lists and wait flags."""
        sync_batches.append((list(tensors), wait))

    with (
        patch.object(draft, "kernel_mesh", return_value=(mesh, 0)),
        patch.object(
            dspark_kernel.DraftConfig,
            "from_checkpoint",
            return_value=dspark_kernel.DraftConfig(),
        ),
        patch.object(
            draft,
            "register_global_op",
            side_effect=fake_register_global_op,
        ),
        patch.object(sync_mod, "synchronize", side_effect=fake_sync),
        patch.object(torch, "device", side_effect=_fake_torch_device),
        patch.object(torch, "tensor", side_effect=_fake_torch_tensor),
        patch.object(torch, "zeros", side_effect=_fake_torch_zeros),
        patch.object(torch, "full", side_effect=_fake_torch_full),
    ):
        d_draft = draft.DsparkDraft(_TEST_MAX_POS, "unused", Path("unused"))
        cfg = d_draft.config
        cpu_anchor = _REAL_TORCH_TENSOR([_TEST_ANCHOR_TOKEN], dtype=torch.int32)
        cpu_new = d_draft.new_state(cpu_anchor)
        expected_vec = draft_mla.make_draft_state(
            _TEST_ANCHOR_TOKEN,
            device="cpu",
        ).vector
        assert cpu_new.cache.device.type == "cpu"
        assert torch.equal(cpu_new.vector, expected_vec)

        cpu_ring_cache = torch.ones(
            (1, cfg.layers, cfg.window, 2 * cfg.head_dim),
            dtype=torch.bfloat16,
        )
        cpu_ring_pos = torch.zeros((1, cfg.window), dtype=torch.int32)
        cpu_from_ring = d_draft.from_ring(
            cpu_ring_cache,
            cpu_ring_pos,
            _TEST_ANCHOR_TOKEN,
        )
        assert torch.equal(cpu_from_ring.vector, expected_vec)
        assert not registered_specs

        tpu_anchor = _as_fake_tpu(cpu_anchor)
        tpu_new = d_draft.new_state(tpu_anchor)
        tpu_from_ring = d_draft.from_ring(
            _as_fake_tpu(cpu_ring_cache),
            _as_fake_tpu(cpu_ring_pos),
            _TEST_ANCHOR_TOKEN,
        )

        assert registered_specs["pallas::k3_dspark_init_state"].in_specs == (P(),)
        assert registered_specs["pallas::k3_dspark_init_state"].out_specs == (
            P(draft.AXIS),
            P(),
            P(),
        )
        assert registered_specs["pallas::k3_dspark_init_vector"].in_specs == (P(),)
        assert registered_specs["pallas::k3_dspark_init_vector"].out_specs == (P(),)
        assert tpu_new.cache.shape == (1, cfg.layers, cfg.window, 2 * cfg.head_dim)
        assert torch.equal(tpu_new.positions, cpu_new.positions)
        assert torch.equal(tpu_new.vector, expected_vec)
        assert torch.equal(tpu_from_ring.vector, expected_vec)
        assert all(wait is False for _, wait in sync_batches)


def test_kda_importer_synchronizes_tpu_state_and_ring_transfers() -> None:
    """Verify KdaImporter synchronizes TPU state and ring import buffers."""
    mesh = Mesh(np.array([jax.devices()[0]]), (pools.MESH_AXIS,))
    importer = pools.KdaImporter(mesh, attention_dp=1, num_slots=4)
    raw_pool = _as_fake_tpu(
        torch.zeros((_POOL_BLOCKS, _POOL_ROWS, _POOL_COLS), dtype=torch.bfloat16),
    )
    importer.raw_pools = [raw_pool]
    importer.kda_raw = [0] * pools.KDA_LAYERS

    conv_bank = _as_fake_tpu(
        torch.zeros(
            (1, 4, pools.KDA_LAYERS, _CONV_TAPS, _NUM_HEADS, _CONV_WINDOW, _HEAD_DIM),
            dtype=torch.bfloat16,
        ),
    )
    rec_bank = _as_fake_tpu(
        torch.zeros(
            (1, 4, pools.KDA_LAYERS, _NUM_HEADS, _HEAD_DIM, _HEAD_DIM),
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
            cfg = dspark_kernel.DraftConfig()
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
            dspark_kernel.DraftConfig(),
            kda_blocks,
            length=_TEST_RING_LEN,
        )

    assert events == ["sync", "state_op", "sync", "sync", "ring_op", "sync"]
    assert len(sync_batches) == _EXPECTED_IMPORTER_SYNC_CALLS
    assert all(wait is False for _, wait in sync_batches)
    assert sync_batches[1][0] == [conv_out, rec_out]
    assert sync_batches[3][0] == [ring_out, pos_out]


def test_decoder_admit_zero_kda_routes_through_tpu_importer() -> None:
    """Verify KimiDecoder._admit at computed=0 uses KdaImporter on TPU slots."""
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
    ops_stub = cast(
        "decode_ops.DecodeOps",
        SimpleNamespace(
            rank=0,
            num_slots=4,
            attention_dp=1,
            batch_sizes=(1,),
            slots=[conv_slot, rec_slot],
        ),
    )
    decoder = decode.KimiDecoder(config, ops_stub, tp_rank=0)
    imported_calls: list[tuple[torch.Tensor, int]] = []

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

    vars(decoder)["_importer"] = SimpleNamespace(import_state=fake_import_state)
    vars(decoder)["_kda_groups"] = [(0, 256)] * pools.KDA_LAYERS
    request = shim_program.RequestInput(
        req_id="req-zero",
        computed=0,
        scheduled=1,
        token_ids=np.array([1], dtype=np.int32),
        num_tokens=1,
        num_prompt_tokens=1,
        block_tables=(np.array([0], dtype=np.int32),),
        sampling=shim_program.GREEDY,
    )
    admit_fn = cast(
        "Callable[[decode.KimiDecoder, shim_program.RequestInput, int], None]",
        vars(decode.KimiDecoder)["_admit"],
    )

    with (
        patch.object(decode, "synchronize"),
        patch.object(torch, "zeros", side_effect=_fake_torch_zeros),
    ):
        admit_fn(decoder, request, 1)

    assert len(imported_calls) == 1
    blocks_arg, slot_arg = imported_calls[0]
    assert blocks_arg.shape == (pools.KDA_LAYERS,)
    assert int(blocks_arg.abs().max().item()) == 0
    assert slot_arg == 1
    assert float(conv_slot[0, 1].min().item()) == 1.0
    assert decoder.ops.slots == [conv_imported, rec_imported]
