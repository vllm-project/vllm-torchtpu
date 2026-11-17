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
"""Tests for GDN v3 batched decode tile SMEM bounds and decode tile clamping."""

import sys
from contextlib import contextmanager
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.state import indexing
from jax._src.state import types as state_types
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.gdn.v3 import config, metadata, tiling, wrapper


def _batched_config(
    batch_size: int, tile_size: int, window_size: int = 1
) -> config.GDNConfig:
    return config.GDNConfig(
        mode=config.GDNMode.BATCHED,
        dtypes=config.Dtypes(
            act_in=jnp.float32.dtype,
            act_out=jnp.bfloat16.dtype,
            compute=jnp.float32.dtype,
            recurrent_state=jnp.float32.dtype,
            conv_state=jnp.float32.dtype,
        ),
        batch_size=batch_size,
        dim_size=512,
        kernel_size=4,
        tile_size=tile_size,
        window_size=window_size,
        num_kq_heads=1,
        num_v_heads=2,
        kq_head_dim=128,
        v_head_dim=128,
    )


@contextmanager
def _cpu_gdn_interpret_env():
    """Runs GDN v3 Pallas kernels on CPU with bounds and DMA race checks."""
    mosaic_interpret = sys.modules[pltpu.reset_tpu_interpret_mode_state.__module__]
    interpret_utils = mosaic_interpret.interpret_utils
    race_detection_state = sys.modules[
        f"{mosaic_interpret.__package__}.race_detection_state"
    ]
    shared_memory = sys.modules[f"{mosaic_interpret.__package__}.shared_memory"]
    races_per_call: list[bool] = []
    orig_clear = mosaic_interpret._clear_shared_memory
    orig_slices_overlap = race_detection_state._slices_overlap
    orig_store = shared_memory.SharedMemory.store_buffer_content

    def _slices_overlap_nonempty(s1, s2):
        # Zero-length DMAs (`pl.ds(idx, 0)`) transfer 0 bytes; JAX 0.11.x
        # `_slices_overlap` treats `slice(k, k)` as overlapping `slice(k, k+1)`.
        for s in (s1, s2):
            if (
                isinstance(s, slice)
                and s.start is not None
                and s.stop is not None
                and s.start >= s.stop
            ):
                return False
        return orig_slices_overlap(s1, s2)

    def _record_and_clear():
        races = getattr(mosaic_interpret, "races", None)
        if races is not None:
            races_per_call.append(bool(races.races_found))
        orig_clear()

    def _resolve_to_range(transforms):
        resolved, viewed = (), False
        for transform in transforms:
            if isinstance(transform, indexing.NDIndexer):
                indices = tuple(
                    interpret_utils._transform_slice_or_index(i)
                    for i in transform.indices
                )
                if viewed:
                    full = tuple(slice(0, d, 1) for d in transform.shape[-3:])
                    assert indices[-3:] == full, f"partial-state ref: {indices}"
                    indices = indices[:-3]
                resolved = interpret_utils._compose_slice_or_index(resolved, indices)
            elif isinstance(
                transform,
                (state_types.BitcastTransform, state_types.ReshapeTransform),
            ):
                if not viewed:
                    resolved, viewed = resolved[:-3], True
            else:
                raise NotImplementedError(f"unsupported ref transform: {transform}")
        return resolved

    def _store_buffer_content(buf_self, key, rnge, value, *args, **kwargs):
        buff_dtype = buf_self.mem[key].dtype
        assert value.itemsize == buff_dtype.itemsize, (value.dtype, buff_dtype)
        return orig_store(buf_self, key, rnge, value.view(buff_dtype), *args, **kwargs)

    mesh = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    pltpu.reset_tpu_interpret_mode_state()
    mosaic_interpret._clear_shared_memory = _record_and_clear
    try:
        with (
            use_abstract_mesh(mesh),
            mock.patch.object(envs, "TPU_ENABLE_GDN_DYNAMIC_TILING", False),
            mock.patch.object(interpret_utils, "to_range", _resolve_to_range),
            mock.patch.object(
                race_detection_state, "_slices_overlap", _slices_overlap_nonempty
            ),
            mock.patch.object(
                shared_memory.SharedMemory,
                "store_buffer_content",
                _store_buffer_content,
            ),
            pltpu.force_tpu_interpret_mode(
                pltpu.InterpretParams(
                    detect_races=True,
                    dma_execution_mode="eager",
                    skip_floating_point_ops=False,
                    vector_clock_size=128,
                )
            ),
        ):
            yield races_per_call
    finally:
        mosaic_interpret._clear_shared_memory = orig_clear
        pltpu.reset_tpu_interpret_mode_state()


@pytest.mark.parametrize(
    ("max_seqs", "tile_size", "end_seq"),
    [
        (6, 4, 6),
        (6, 4, 5),
        (6, 4, 0),
        (6, 4, 9),
        (5, 2, 5),
        (2, 4, 2),
    ],
)
def test_batched_metadata_pads_smem_tables_to_full_tiles(
    max_seqs: int, tile_size: int, end_seq: int
):
    cfg = _batched_config(batch_size=max_seqs, tile_size=tile_size)
    seq_lens = jnp.arange(1, max_seqs + 1, dtype=jnp.int32) + 4
    query_start_loc = jnp.arange(max_seqs + 1, dtype=jnp.int32)
    state_indices = jnp.arange(1, max_seqs + 1, dtype=jnp.int32)
    read_offsets = jnp.zeros((max_seqs,), dtype=jnp.int32)

    batched_metadata = metadata.compute_batched_seq_metadata(
        cfg=cfg,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        state_indices=state_indices,
        read_offsets=read_offsets,
        end_seq=jnp.int32(end_seq),
    )

    clamped_end_seq = min(max(end_seq, 0), max_seqs)
    expected_len = max(1, pl.cdiv(max_seqs, tile_size)) * tile_size
    expected_num_tiles = pl.cdiv(clamped_end_seq, tile_size)
    assert int(batched_metadata.num_tiles) == expected_num_tiles

    for wrapper_field in (
        batched_metadata.p_id_to_s_idx,
        batched_metadata.p_id_to_r_base,
        batched_metadata.p_id_to_r_size,
        batched_metadata.p_id_is_first_tile,
        batched_metadata.p_id_is_last_tile,
    ):
        assert wrapper_field.data.shape == (expected_len,)

    for p_id in range(max(1, expected_num_tiles)):
        for idx in range(tile_size):
            flat_slot = p_id * tile_size + idx
            s_idx = int(batched_metadata.p_id_to_s_idx[p_id, idx])
            r_base = int(batched_metadata.p_id_to_r_base[p_id, idx])
            r_size = int(batched_metadata.p_id_to_r_size[p_id, idx])
            is_first = bool(batched_metadata.p_id_is_first_tile[p_id, idx])
            is_last = bool(batched_metadata.p_id_is_last_tile[p_id, idx])
            if flat_slot < clamped_end_seq:
                assert s_idx == flat_slot
                assert r_base == flat_slot
                assert r_size == 1
                assert is_first is True
                assert is_last is True
            else:
                assert s_idx == 0
                assert r_base == 0
                assert r_size == 0
                assert is_first is False
                assert is_last is False


@pytest.mark.parametrize("explicit_decode_tile", [8, None])
def test_get_tile_sizes_clamps_decode_tile_to_active_num_seqs(
    explicit_decode_tile: int | None,
):
    mesh = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    with use_abstract_mesh(mesh):
        limit = config.GDNConfig.get_vmem_limit_bytes(1)
        decode_tile, mixed_tile = tiling.get_tile_sizes(
            batch_size=2048,
            num_seqs=2,
            padded_batch_size=2048,
            n_kq=2,
            n_v=8,
            d_k=128,
            d_v=128,
            kernel_size=4,
            conv_state_dim_size=1536,
            act_out_dtype=jnp.dtype(jnp.bfloat16),
            recurrent_state_dtype=jnp.dtype(jnp.float32),
            num_lanes=128,
            decode_vmem_limit_bytes=limit,
            mixed_vmem_limit_bytes=limit,
            window_size=1,
            decode_tile_size=explicit_decode_tile,
        )
    assert decode_tile == 2
    assert mixed_tile == 128


def _fresh_jitted_gdn():
    return jax.jit(
        wrapper.fused_conv1d_gdn.__wrapped__,
        donate_argnames=("states",),
        static_argnames=(
            "n_kq",
            "n_v",
            "d_k",
            "d_v",
            "kernel_size",
            "num_spec_tokens",
            "decode_tile_size",
            "mixed_tile_size",
            "batched_only",
            "prefill_only",
            "zero_initialize_out",
            "compute_precision",
            "recurrent_dtype",
            "conv_dtype",
            "attention_mode",
            "gate_lower_bound",
            "kernel_act_dtype",
            "kernel_conv_weight_dtype",
        ),
    )


def _build_gdn_inputs(
    key: jax.Array,
    query_lens_list: list[int],
    num_decode: int,
    num_spec_tokens: int = 0,
    recurrent_dtype=jnp.float32,
    use_ckpt_indices: bool = False,
    padded_tokens: int | None = None,
):
    n_kq, n_v, d_k, d_v, kernel_size = 1, 2, 128, 128, 4
    dim = 2 * n_kq * d_k + n_v * d_v
    num_seqs = len(query_lens_list)
    active_tokens = sum(query_lens_list)
    total_tokens = max(active_tokens, padded_tokens or active_tokens)
    window_size = num_spec_tokens + 1
    slots_per_seq = window_size + (1 if num_spec_tokens > 0 else 0)

    keys = iter(jax.random.split(key, 10))
    qkv = jax.random.normal(next(keys), (total_tokens, dim), dtype=jnp.bfloat16)
    b = jax.random.normal(next(keys), (total_tokens, n_v), dtype=jnp.bfloat16)
    a = jax.random.normal(next(keys), (total_tokens, n_v), dtype=jnp.bfloat16)
    conv_weight = jax.random.normal(
        next(keys), (dim, 1, kernel_size), dtype=jnp.bfloat16
    )
    conv_bias = jax.random.normal(next(keys), (dim,), dtype=jnp.bfloat16)
    a_log = jax.random.normal(next(keys), (n_v,), dtype=jnp.float32)
    dt_bias = jax.random.normal(next(keys), (n_v,), dtype=jnp.float32)

    total_state_slots = num_seqs * slots_per_seq + 1
    init_conv = jax.random.normal(
        next(keys), (total_state_slots, kernel_size - 1, dim), dtype=jnp.float32
    )
    init_rec = jax.random.normal(
        next(keys), (total_state_slots, n_v, d_k, d_v), dtype=recurrent_dtype
    )

    query_start_loc = jnp.array(
        [0, *np.cumsum(query_lens_list).tolist()], dtype=jnp.int32
    )
    state_indices = 1 + jnp.arange(num_seqs, dtype=jnp.int32) * slots_per_seq
    read_offsets = (
        jnp.arange(num_seqs, dtype=jnp.int32) % window_size
        if num_spec_tokens > 0
        else jnp.zeros((num_seqs,), dtype=jnp.int32)
    )
    ckpt_indices = (
        state_indices[:, None] + jnp.arange(window_size, dtype=jnp.int32)[None, :]
        if use_ckpt_indices
        else None
    )
    distribution = jnp.array([num_decode, num_seqs, num_seqs], dtype=jnp.int32)
    seq_lens = jnp.array([q_len + 8 for q_len in query_lens_list], dtype=jnp.int32)
    return dict(
        qkv=qkv,
        b=b,
        a=a,
        states=(init_conv, init_rec),
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        a_log=a_log,
        dt_bias=dt_bias,
        query_start_loc=query_start_loc,
        state_indices=state_indices,
        read_offsets=read_offsets,
        ckpt_indices=ckpt_indices,
        distribution=distribution,
        seq_lens=seq_lens,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
        num_spec_tokens=num_spec_tokens,
    )


def test_batched_non_divisible_and_mixed_step_cpu_interpret():
    if jax.default_backend() != "cpu":
        pytest.skip("Pallas interpret mode requires a local JAX CPU device")
    cases = [
        # 6 decode sequences with decode_tile_size=4: tail tile reads slots 4..7.
        ([1, 1, 1, 1, 1, 1], 6, 0, 4, False),
        # Mixed step with 2 sequences and 4 total tokens: decode_tile_size=4
        # exceeds num_seqs=2 unless clamped against num_seqs.
        ([1, 3], 1, 0, 4, False),
        # Pure prefill step (end_seq == 0) with prefill_only=False: BATCHED pass
        # emits num_tiles == 0 before PER_SEQ runs.
        ([2, 3], 0, 0, 4, False),
        # Speculative verify (W=4) with 5 sequences, non-zero read_offsets, and
        # explicit ckpt_indices: tail tile reads slots 4..5.
        ([4, 3, 4, 2, 4], 5, 3, 2, True),
    ]

    for case_idx, (
        query_lens_list,
        num_decode,
        num_spec,
        decode_tile,
        use_ckpt,
    ) in enumerate(cases):
        base_inputs = _build_gdn_inputs(
            jax.random.key(case_idx),
            query_lens_list=query_lens_list,
            num_decode=num_decode,
            num_spec_tokens=num_spec,
            use_ckpt_indices=use_ckpt,
        )
        cand_states = (
            base_inputs["states"][0].copy(),
            base_inputs["states"][1].copy(),
        )
        ref_states = (
            base_inputs["states"][0].copy(),
            base_inputs["states"][1].copy(),
        )

        with _cpu_gdn_interpret_env() as cand_races:
            cand_fn = _fresh_jitted_gdn()
            (cand_conv, cand_rec), cand_out = cand_fn(
                **{**base_inputs, "states": cand_states},
                decode_tile_size=decode_tile,
                mixed_tile_size=1,
            )
            jax.block_until_ready((cand_conv, cand_rec, cand_out))
        assert cand_races and not any(cand_races)

        with _cpu_gdn_interpret_env() as ref_races:
            ref_fn = _fresh_jitted_gdn()
            (ref_conv, ref_rec), ref_out = ref_fn(
                **{**base_inputs, "states": ref_states},
                decode_tile_size=1,
                mixed_tile_size=1,
            )
            jax.block_until_ready((ref_conv, ref_rec, ref_out))
        assert ref_races and not any(ref_races)

        np.testing.assert_array_equal(np.asarray(cand_out), np.asarray(ref_out))
        np.testing.assert_array_equal(np.asarray(cand_conv), np.asarray(ref_conv))
        np.testing.assert_array_equal(np.asarray(cand_rec), np.asarray(ref_rec))

    # Exercise TPU_ENABLE_GDN_DYNAMIC_TILING=True with batched_only=True on CPU
    # (6 decode sequences auto-select decode_tile_size >= 4 -> non-divisible tail).
    dyn_inputs = _build_gdn_inputs(
        jax.random.key(99), query_lens_list=[1, 1, 1, 1, 1, 1], num_decode=6
    )
    dyn_states = (dyn_inputs["states"][0].copy(), dyn_inputs["states"][1].copy())
    ref_states = (dyn_inputs["states"][0].copy(), dyn_inputs["states"][1].copy())
    with (
        _cpu_gdn_interpret_env() as dyn_races,
        mock.patch.object(envs, "TPU_ENABLE_GDN_DYNAMIC_TILING", True),
    ):
        dyn_fn = _fresh_jitted_gdn()
        (dyn_conv, dyn_rec), dyn_out = dyn_fn(
            **{**dyn_inputs, "states": dyn_states},
            batched_only=True,
        )
        jax.block_until_ready((dyn_conv, dyn_rec, dyn_out))
    assert dyn_races and not any(dyn_races)
    with _cpu_gdn_interpret_env() as ref_dyn_races:
        ref_fn = _fresh_jitted_gdn()
        (ref_conv, ref_rec), ref_out = ref_fn(
            **{**dyn_inputs, "states": ref_states},
            decode_tile_size=1,
            mixed_tile_size=1,
            batched_only=True,
        )
        jax.block_until_ready((ref_conv, ref_rec, ref_out))
    assert ref_dyn_races and not any(ref_dyn_races)
    np.testing.assert_array_equal(np.asarray(dyn_out), np.asarray(ref_out))
    np.testing.assert_array_equal(np.asarray(dyn_conv), np.asarray(ref_conv))
    np.testing.assert_array_equal(np.asarray(dyn_rec), np.asarray(ref_rec))


@pytest.mark.tpu_test
@pytest.mark.parametrize("recurrent_dtype", ["float32", "bfloat16"])
def test_batched_non_divisible_and_mixed_step_tpu(recurrent_dtype: str):
    rec_dtype = getattr(jnp, recurrent_dtype)
    tol = (
        dict(rtol=1e-5, atol=1e-5)
        if rec_dtype == jnp.float32
        else dict(rtol=1e-2, atol=1e-2)
    )
    cases = [
        ([1, 1, 1, 1, 1, 1], 6, 0, 4, 64, False, False, None),
        ([1, 63], 1, 0, 4, 64, False, False, 64),
        ([2, 30], 0, 0, 4, 64, False, False, 64),
        ([4, 3, 4, 2, 4], 5, 3, 2, 64, True, False, 64),
        ([1, 1, 1, 1, 1, 1], 6, 0, 8, 64, False, True, 64),
        ([1, 63], 1, 0, 8, 64, False, True, 64),
    ]
    for case_idx, (
        query_lens_list,
        num_decode,
        num_spec,
        decode_tile,
        mixed_tile,
        use_ckpt,
        dyn_tiling,
        padded_tokens,
    ) in enumerate(cases):
        base_inputs = _build_gdn_inputs(
            jax.random.key(100 + case_idx),
            query_lens_list=query_lens_list,
            num_decode=num_decode,
            num_spec_tokens=num_spec,
            recurrent_dtype=rec_dtype,
            use_ckpt_indices=use_ckpt,
            padded_tokens=padded_tokens,
        )
        cand_states = (
            base_inputs["states"][0].copy(),
            base_inputs["states"][1].copy(),
        )
        ref_states = (
            base_inputs["states"][0].copy(),
            base_inputs["states"][1].copy(),
        )
        with mock.patch.object(envs, "TPU_ENABLE_GDN_DYNAMIC_TILING", dyn_tiling):
            cand_fn = _fresh_jitted_gdn()
            (cand_conv, cand_rec), cand_out = cand_fn(
                **{**base_inputs, "states": cand_states},
                decode_tile_size=decode_tile,
                mixed_tile_size=mixed_tile,
            )
        with mock.patch.object(envs, "TPU_ENABLE_GDN_DYNAMIC_TILING", False):
            ref_fn = _fresh_jitted_gdn()
            (ref_conv, ref_rec), ref_out = ref_fn(
                **{**base_inputs, "states": ref_states},
                decode_tile_size=1,
                mixed_tile_size=64,
            )
        active_tokens = sum(query_lens_list)
        np.testing.assert_allclose(
            np.asarray(cand_out[:active_tokens], dtype=np.float32),
            np.asarray(ref_out[:active_tokens], dtype=np.float32),
            rtol=1e-2,
            atol=1e-2,
        )
        np.testing.assert_array_equal(np.asarray(cand_conv), np.asarray(ref_conv))
        # Null block 0 must remain bitwise untouched despite padded tail slots
        # indexing s_idx = 0 with dma_size = 0.
        np.testing.assert_array_equal(
            np.asarray(cand_rec[0]), np.asarray(base_inputs["states"][1][0])
        )
        np.testing.assert_allclose(
            np.asarray(cand_rec[1:], dtype=np.float32),
            np.asarray(ref_rec[1:], dtype=np.float32),
            **tol,
        )
