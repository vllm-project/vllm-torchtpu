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
"""CPU unit tests for `vllm_torchtpu.kernels.glm53_megakernel`."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.glm53_megakernel import (
    Glm53MegakernelConfig,
    kernel_cache_token_loc,
    kv_bytes,
    kv_words,
    latent_pool_shapes,
    layer_tables,
    make_decode_step,
    place_rank_weights_on_mesh,
    prepare_kernel_weights_local,
    superpages_per_slot,
    synthetic_rank_weights,
    to_per_channel_fp8,
)

_NO_FRAMEWORKS = """
import sys
for name in ("torch", "jax", "jaxlib", "vllm", "numpy"):
    sys.modules[name] = None
import vllm_torchtpu.kernels.glm53_megakernel as gm
assert gm.SPEC.name == "glm53"
assert gm.SPEC.tp_sizes == (8,)
"""


def test_glm53_megakernel_spec_imports_without_frameworks() -> None:
    """`vllm_torchtpu.kernels.glm53_megakernel` is stdlib-only at import time."""
    import vllm_torchtpu.kernels.glm53_megakernel as gm

    src = Path(gm.__file__).resolve().parents[3]
    res = subprocess.run(
        [sys.executable, "-c", _NO_FRAMEWORKS],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(src)},
        timeout=60,
    )
    assert res.returncode == 0, res.stderr[-2000:]


def test_config_invariants_and_small_config() -> None:
    small = Glm53MegakernelConfig.small_lane_aligned_for_test(world_size=2, page_size=8)
    assert small.num_hidden_layers == 4
    assert small.kda_layers == (0, 1, 2)
    assert small.mla_layers == (3,)
    assert small.first_k_dense_replace == 1

    tiny = Glm53MegakernelConfig.tiny_for_test(world_size=1, page_size=8)
    assert tiny.num_hidden_layers == 4
    assert tiny.hidden_size == 32

    with pytest.raises(ValueError, match="kda_layers"):
        bad = Glm53MegakernelConfig.tiny_for_test(world_size=1, page_size=8)
        object.__setattr__(bad, "kda_layers", (0,))
        bad.validate(world_size=1)


def test_kv_words_and_kv_bytes_roundtrip() -> None:
    rng = np.random.default_rng(42)
    raw = rng.integers(0, 256, size=(2, 3, 16, 64), dtype=np.uint8)
    fp8 = jax.lax.bitcast_convert_type(jnp.asarray(raw), jnp.float8_e4m3fn)
    words = kv_words(fp8)
    assert words.shape == (2, 3, 4, 64)
    assert words.dtype == jnp.int32
    back = kv_bytes(words)
    assert back.shape == fp8.shape
    assert back.dtype == jnp.float8_e4m3fn
    np.testing.assert_array_equal(
        np.asarray(jax.lax.bitcast_convert_type(back, jnp.uint8)), raw
    )


def test_small_config_weight_preparation_and_interpret_step() -> None:
    cfg = Glm53MegakernelConfig.small_lane_aligned_for_test(world_size=1, page_size=8)
    mesh = jax.sharding.Mesh(np.asarray(jax.devices("cpu")[:1]), ("tp",))
    per_rank = [
        synthetic_rank_weights(cfg, rank=0, world_size=1, seed=0, include_mtp=True)
    ]
    w_mesh = place_rank_weights_on_mesh(per_rank, mesh)
    w_local = {k: v[0] for k, v in w_mesh.items()}
    w_pc = to_per_channel_fp8(cfg, w_local)
    kw_local = prepare_kernel_weights_local(
        cfg, w_pc, world_size=1, experts_are_tp=False
    )
    assert "mla_a_proj_q" in kw_local and "kda_in_proj" in kw_local
    assert "moe_expert_gate_up_q" in kw_local and "moe_expert_down_q" in kw_local
    assert "mtp_eh_proj" in kw_local and "meta" in kw_local

    # Run a 1-step interpret decode on CPU with TP=1.
    kw = {k: v[None] for k, v in kw_local.items()}
    n_kpages = 2
    kvq_sh, kvs_sh = latent_pool_shapes(cfg, len(cfg.mla_layers), n_kpages)
    kvq = jnp.zeros((1, *kvq_sh), dtype=jnp.int32)
    kvs = jnp.ones((1, *kvs_sh), dtype=jnp.float32)
    lk = len(cfg.kda_layers)
    hk = cfg.kda_num_heads
    dk = cfg.kda_head_dim
    conv = jnp.zeros(
        (1, lk, 2, cfg.kda_conv_kernel_size - 1, 3 * hk * dk), dtype=jnp.bfloat16
    )
    rec = jnp.zeros((1, lk, 2, hk, dk, dk), dtype=jnp.float32)
    step = make_decode_step(cfg, mesh, batch=1, max_batch=2, n_pages=4, interpret=True)
    tok = jnp.array([3], dtype=jnp.int32)
    pos = jnp.array([0], dtype=jnp.int32)
    slots = jnp.array([0], dtype=jnp.int32)
    valid = jnp.array([1], dtype=jnp.int32)
    top_lp, top_id, nxt, conv_out, rec_out, kvq_out, kvs_out = step(
        kw, tok, pos, slots, valid, conv, rec, kvq, kvs
    )
    assert top_lp.shape == (1, 64)
    assert top_id.shape == (1, 64)
    assert nxt.shape == (1,)
    assert bool(jnp.all(jnp.isfinite(top_lp)))
    assert conv_out.shape == conv.shape and rec_out.shape == rec.shape
    assert kvq_out.shape == kvq.shape and kvs_out.shape == kvs.shape

    # Masked superpage tail positions (`k_pos >= pos`) holding stale NaN scales
    # must never poison finite prefix positions via IEEE-754 `0.0 * NaN = NaN`.
    # (`step` donates `conv/rec/kvq/kvs`, so clone before each call.)
    pos2 = jnp.array([2], dtype=jnp.int32)
    lp_clean, id_clean, nxt_clean, *_ = step(
        kw,
        tok,
        pos2,
        slots,
        valid,
        jnp.array(conv_out, copy=True),
        jnp.array(rec_out, copy=True),
        jnp.array(kvq_out, copy=True),
        jnp.array(kvs_out, copy=True),
    )
    kvs_poison = jnp.array(kvs_out, copy=True).at[:, :, :, :, 2:].set(jnp.nan)
    lp_nan_tail, id_nan_tail, nxt_nan_tail, *_ = step(
        kw,
        tok,
        pos2,
        slots,
        valid,
        jnp.array(conv_out, copy=True),
        jnp.array(rec_out, copy=True),
        jnp.array(kvq_out, copy=True),
        kvs_poison,
    )
    assert bool(jnp.all(jnp.isfinite(lp_nan_tail)))
    np.testing.assert_allclose(np.asarray(lp_nan_tail), np.asarray(lp_clean))
    np.testing.assert_array_equal(np.asarray(id_nan_tail), np.asarray(id_clean))
    np.testing.assert_array_equal(np.asarray(nxt_nan_tail), np.asarray(nxt_clean))


def test_layer_tables_and_cache_loc_helpers() -> None:
    cfg = Glm53MegakernelConfig.small_lane_aligned_for_test(world_size=2, page_size=8)
    ltab = layer_tables(cfg)
    assert ltab.shape == (3 * cfg.num_hidden_layers,)
    owner, grp, row = kernel_cache_token_loc(
        jnp.array([0, cfg.page_size]), cfg.page_size, world_size=2
    )
    assert owner.tolist() == [0, 1]
    assert grp.tolist() == [0, 0]
    assert row.tolist() == [0, 0]
    assert superpages_per_slot(cfg.page_size, 1024, world_size=8) >= 1


def _prepare_test_step_inputs(
    world_size: int, devices: list[jax.Device]
) -> tuple[
    Glm53MegakernelConfig,
    jax.sharding.Mesh,
    dict[str, jax.Array],
    dict[str, jax.Array],
    tuple[jax.Array, jax.Array, jax.Array, jax.Array],
]:
    from vllm_torchtpu.kernels.glm53_megakernel import kernel as mk_kernel

    cfg = Glm53MegakernelConfig.small_lane_aligned_for_test(
        world_size=world_size, page_size=8
    )
    mesh = jax.sharding.Mesh(np.asarray(devices), ("tp",))
    per_rank = [
        synthetic_rank_weights(
            cfg, rank=r, world_size=world_size, seed=0, include_mtp=True
        )
        for r in range(world_size)
    ]
    w_mesh = place_rank_weights_on_mesh(per_rank, mesh)
    p = jax.sharding.PartitionSpec
    kw_tp = jax.jit(
        jax.shard_map(
            lambda wl: {
                k: v[None]
                for k, v in prepare_kernel_weights_local(
                    cfg,
                    to_per_channel_fp8(cfg, {k: v[0] for k, v in wl.items()}),
                    world_size=world_size,
                    experts_are_tp=False,
                ).items()
            },
            mesh=mesh,
            in_specs=(p("tp"),),
            out_specs=p("tp"),
            check_vma=False,
        )
    )(w_mesh)
    kw_km = mk_kernel.experts_tp_to_kmajor(cfg, mesh, kw_tp, donate=False)

    n_kpages = 4
    kvq_sh, kvs_sh = latent_pool_shapes(cfg, len(cfg.mla_layers) + 1, n_kpages)
    lk = len(cfg.kda_layers)
    hk = cfg.kda_num_heads // world_size
    dk = cfg.kda_head_dim
    tp_sh = jax.sharding.NamedSharding(mesh, p("tp"))
    conv = jax.device_put(
        jnp.zeros(
            (world_size, lk, 2, cfg.kda_conv_kernel_size - 1, 3 * hk * dk),
            dtype=jnp.bfloat16,
        ),
        tp_sh,
    )
    rec = jax.device_put(
        jnp.zeros((world_size, lk, 2, hk, dk, dk), dtype=jnp.float32), tp_sh
    )
    kvq = jax.device_put(jnp.zeros((world_size, *kvq_sh), dtype=jnp.int32), tp_sh)
    kvs = jax.device_put(jnp.ones((world_size, *kvs_sh), dtype=jnp.float32), tp_sh)
    return cfg, mesh, kw_tp, kw_km, (conv, rec, kvq, kvs)


def test_kmajor_and_spec_decode_step_interpret_detect_races() -> None:
    """CPU `detect_races=True` exercises both `make_decode_step` and
    `make_spec_decode_step` (`sp.lk=3` KDA rollback DMA pipeline + `experts_kmajor`)."""
    from jax.experimental.pallas import tpu as pltpu

    from vllm_torchtpu.kernels.glm53_megakernel import make_spec_decode_step

    cfg, mesh, kw_tp, kw_km, (conv, rec, kvq, kvs) = _prepare_test_step_inputs(
        1, jax.devices("cpu")[:1]
    )
    race_interp = pltpu.InterpretParams(detect_races=True)
    step_nm = make_decode_step(
        cfg,
        mesh,
        batch=1,
        max_batch=2,
        n_pages=4,
        interpret=race_interp,
        experts_kmajor=False,
    )
    step_km = make_decode_step(
        cfg,
        mesh,
        batch=1,
        max_batch=2,
        n_pages=4,
        interpret=race_interp,
        experts_kmajor=True,
    )
    tok = jnp.array([5], dtype=jnp.int32)
    pos = jnp.array([0], dtype=jnp.int32)
    slots = jnp.array([0], dtype=jnp.int32)
    valid = jnp.array([1], dtype=jnp.int32)
    lp_nm, id_nm, nxt_nm, *_ = step_nm(
        kw_tp,
        tok,
        pos,
        slots,
        valid,
        jnp.array(conv[:, :, :, :, :], copy=True),
        jnp.array(rec, copy=True),
        jnp.array(kvq[:, : len(cfg.mla_layers)], copy=True),
        jnp.array(kvs[:, : len(cfg.mla_layers)], copy=True),
    )
    lp_km, id_km, nxt_km, *_ = step_km(
        kw_km,
        tok,
        pos,
        slots,
        valid,
        jnp.array(conv, copy=True),
        jnp.array(rec, copy=True),
        jnp.array(kvq[:, : len(cfg.mla_layers)], copy=True),
        jnp.array(kvs[:, : len(cfg.mla_layers)], copy=True),
    )
    np.testing.assert_allclose(
        np.asarray(lp_km), np.asarray(lp_nm), rtol=1e-4, atol=1e-4
    )
    np.testing.assert_array_equal(np.asarray(id_km[:, 0]), np.asarray(id_nm[:, 0]))
    np.testing.assert_array_equal(np.asarray(nxt_km), np.asarray(nxt_nm))

    spec_step = make_spec_decode_step(
        cfg,
        mesh,
        batch=1,
        max_batch=2,
        n_pages=4,
        interpret=race_interp,
        experts_kmajor=True,
    )
    v_toks = jnp.array([[5, int(nxt_km[0]), 2, 3]], dtype=jnp.int32)
    s_lp, _s_id, s_tgt, s_acc, s_nxt_v, s_nxt_p, *_ = spec_step(
        kw_km,
        v_toks,
        pos,
        slots,
        valid,
        jnp.array(conv, copy=True),
        jnp.array(rec, copy=True),
        jnp.array(kvq, copy=True),
        jnp.array(kvs, copy=True),
    )
    assert bool(jnp.all(jnp.isfinite(s_lp)))
    # Verify token 0 must match the plain decode step's greedy sample and logprobs,
    # and because `v_toks[0, 1] == nxt_km[0]`, at least 2 tokens are accepted.
    np.testing.assert_allclose(
        np.asarray(s_lp[:, 0]), np.asarray(lp_km), rtol=1e-3, atol=1e-3
    )
    assert int(s_tgt[0, 0]) == int(nxt_km[0])
    assert int(s_acc[0]) >= 2
    assert int(s_nxt_p[0]) == int(s_acc[0])
    assert int(s_nxt_v[0, 0]) == int(s_tgt[0, int(s_acc[0]) - 1])


@pytest.mark.tpu_test
def test_two_device_tpu_megakernel_decode_and_spec_decode_parity() -> None:
    """2-device TPU execution test for `make_decode_step` and `make_spec_decode_step`
    (`experts_kmajor=True`, peer barrier + ring/all-reduce collectives)."""
    from vllm_torchtpu.kernels.glm53_megakernel import make_spec_decode_step

    tpus = jax.devices("tpu")
    assert len(tpus) >= 2
    cfg, mesh, kw_tp, kw_km, (conv, rec, kvq, kvs) = _prepare_test_step_inputs(
        2, tpus[:2]
    )
    tp_sh = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("tp"))
    rep_sh = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    def clone_tp(x: jax.Array) -> jax.Array:
        return jax.device_put(np.array(x), tp_sh)

    step_nm = make_decode_step(
        cfg, mesh, batch=1, max_batch=2, n_pages=4, experts_kmajor=False
    )
    step_km = make_decode_step(
        cfg, mesh, batch=1, max_batch=2, n_pages=4, experts_kmajor=True
    )
    spec_step = make_spec_decode_step(
        cfg, mesh, batch=1, max_batch=2, n_pages=4, experts_kmajor=True
    )

    tok = jax.device_put(jnp.array([7], dtype=jnp.int32), rep_sh)
    pos = jax.device_put(jnp.array([0], dtype=jnp.int32), rep_sh)
    slots = jax.device_put(jnp.array([0], dtype=jnp.int32), rep_sh)
    valid = jax.device_put(jnp.array([1], dtype=jnp.int32), rep_sh)

    lp_nm, _id_nm, nxt_nm, *_ = step_nm(
        kw_tp,
        tok,
        pos,
        slots,
        valid,
        clone_tp(conv),
        clone_tp(rec),
        clone_tp(kvq[:, : len(cfg.mla_layers)]),
        clone_tp(kvs[:, : len(cfg.mla_layers)]),
    )
    lp_km, _id_km, nxt_km, conv1, rec1, kvq1, kvs1 = step_km(
        kw_km,
        tok,
        pos,
        slots,
        valid,
        clone_tp(conv),
        clone_tp(rec),
        clone_tp(kvq[:, : len(cfg.mla_layers)]),
        clone_tp(kvs[:, : len(cfg.mla_layers)]),
    )
    np.testing.assert_allclose(
        np.asarray(lp_km), np.asarray(lp_nm), rtol=1e-2, atol=1e-2
    )
    np.testing.assert_array_equal(np.asarray(nxt_km), np.asarray(nxt_nm))

    # Step 2 on `step_km` at pos=1 gives the exact greedy token at pos=1.
    _, _, nxt_km2, *_ = step_km(
        kw_km,
        jax.device_put(jnp.asarray(nxt_km, dtype=jnp.int32), rep_sh),
        jax.device_put(jnp.array([1], dtype=jnp.int32), rep_sh),
        slots,
        valid,
        clone_tp(conv1),
        clone_tp(rec1),
        clone_tp(kvq1),
        clone_tp(kvs1),
    )

    v_toks = jax.device_put(
        jnp.array([[7, int(nxt_km[0]), int(nxt_km2[0]), 0]], dtype=jnp.int32), rep_sh
    )
    s_lp, _s_id, s_tgt, s_acc, _s_nv, s_np, *_ = spec_step(
        kw_km,
        v_toks,
        pos,
        slots,
        valid,
        clone_tp(conv),
        clone_tp(rec),
        clone_tp(kvq),
        clone_tp(kvs),
    )
    assert bool(jnp.all(jnp.isfinite(s_lp)))
    assert int(s_tgt[0, 0]) == int(nxt_km[0])
    assert int(s_tgt[0, 1]) == int(nxt_km2[0])
    assert int(s_acc[0]) >= 3
    assert int(s_np[0]) == int(s_acc[0])
