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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.experimental import pallas as pl
from jax.sharding import Mesh

from vllm_torchtpu.models.kimi_k3.megakernel import decode, decode_ops, pools, program
from vllm_torchtpu.models.kimi_k3.megakernel.kernels import decode_megakernel as kernel
from vllm_torchtpu.models.kimi_k3.megakernel.runtime_shim import backend
from vllm_torchtpu.models.kimi_k3.megakernel.runtime_shim.config import (
    MegakernelEngineConfig,
)

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
