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

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3 import config as k3_config
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3 import prefill as k3_prefill
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.config import KernelConfig
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.kernels import (
    prefill_megakernel as pm,
)
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.kernels.mla_a2a import (
    HEAD_COLS,
    RANKS,
    can_fuse_mla_ag_and_gate_chip_pair,
    gather_gate_shares,
    gather_gate_shares_chip_pair,
    gather_gate_shares_fp8,
    gather_output_projection_shares,
    gather_output_projection_shares_chip_pair,
    gather_output_projection_shares_fp8,
    make_indexed_mla_layer_sp_a2a,
    mla_attended_all_to_all_pallas,
    mla_local_normed_a2a_attention,
    mla_qkv_up_pallas,
    mla_qkv_up_unstriped_and_prefix_gather_pallas,
    mla_qkv_up_unstriped_pallas,
    owner_mla_gate_and_output_chip_pair_pallas,
    owner_mla_gate_and_output_chip_pair_ref,
    owner_mla_gate_and_output_pallas,
    owner_mla_gate_and_output_ref,
    owner_mla_output_gated_chip_pair_pallas,
    rank_half_gather_and_mla_gate_chip_pair_pallas,
    unstripe_sp_window,
)


def _bf16(x):
    return np.asarray(jnp.asarray(x, jnp.bfloat16).astype(jnp.float32))


def _numpy_owner_reference(
    local_normed, owner_rows, m_gate, m_o, prefix, *, stripe, length, token_owner
):
    num_ranks, rows, cols = owner_rows.shape
    blocks = rows // stripe
    og = _bf16(local_normed @ m_gate)
    gate = _bf16(1.0 / (1.0 + np.exp(-og)))
    attended = owner_rows.transpose(1, 0, 2).reshape(rows, num_ranks * cols)
    gated = _bf16(attended * gate)
    proj = gated @ m_o
    out = np.zeros_like(prefix)
    for b in range(blocks):
        for s in range(stripe):
            r = b * stripe + s
            token = b * num_ranks * stripe + token_owner * stripe + s
            out[r] = prefix[r] + (proj[r] if token < length else 0.0)
    return _bf16(out)


def test_should_keep_ring_mla_when_env_disabled(monkeypatch):
    monkeypatch.setenv("K3_PREFILL_MLA_A2A", "0")

    assert KernelConfig().prefill_mla_a2a is False


def test_should_select_mla_a2a_when_env_set(monkeypatch):
    monkeypatch.setenv("K3_PREFILL_MLA_A2A", "1")

    assert KernelConfig().prefill_mla_a2a is True


def test_should_validate_mesh_and_segments_in_make_indexed_mla_layer_sp_a2a():
    one_dev_mesh = Mesh(np.asarray(jax.devices()[:1]), ("k3_sp",))

    with pytest.raises(ValueError, match="packed windows"):
        make_indexed_mla_layer_sp_a2a(
            one_dev_mesh, seq_len=512, segments=0, prefix_capacity=0
        )

    with pytest.raises(ValueError, match="TP32 mesh"):
        make_indexed_mla_layer_sp_a2a(
            one_dev_mesh, seq_len=512, segments=1, prefix_capacity=0
        )


def test_should_reject_invalid_shapes_for_unstripe_a2a_and_owner_pallas():
    with pytest.raises(ValueError, match="divisible"):
        unstripe_sp_window(jnp.zeros((10, 8), jnp.bfloat16), num_ranks=4, stripe=4)

    with pytest.raises(ValueError, match="attended"):
        mla_attended_all_to_all_pallas(
            jnp.zeros((1, 3, 2), jnp.int32),
            jnp.zeros((16, 128), jnp.bfloat16),
            num_ranks=2,
            stripe=8,
        )

    with pytest.raises(ValueError, match="owner_rows"):
        owner_mla_gate_and_output_pallas(
            jnp.zeros((16, 3584), jnp.bfloat16),
            jnp.zeros((4, 16, HEAD_COLS), jnp.bfloat16),
            jnp.zeros((2, 3584, RANKS * HEAD_COLS), jnp.bfloat16),
            jnp.zeros((2, RANKS * HEAD_COLS, 3584), jnp.bfloat16),
            jnp.zeros((16, 3584), jnp.bfloat16),
            layer=0,
            length=16,
            token_owner=0,
            stripe=16,
            interpret=True,
        )


@pytest.mark.parametrize(
    ("token_owner", "length"), [(5, 10_000), (5, 32 * 4 + 5 * 4 + 2)]
)
def test_should_match_numpy_oracle_for_owner_mla_gate_and_output_ref(
    token_owner, length
):
    rng = np.random.default_rng(11)
    num_ranks, blocks, stripe, cols, hidden = 32, 2, 4, 8, 16
    rows = blocks * stripe
    local_normed = _bf16(rng.standard_normal((rows, hidden)) * 0.25)
    owner_rows = _bf16(rng.standard_normal((num_ranks, rows, cols)) * 0.25)
    m_gate = _bf16(rng.standard_normal((hidden, num_ranks * cols)) * 0.15)
    m_o = _bf16(rng.standard_normal((num_ranks * cols, hidden)) * 0.15)
    prefix = _bf16(rng.standard_normal((rows, hidden)) + 4.0)

    actual = np.asarray(
        owner_mla_gate_and_output_ref(
            jnp.asarray(local_normed, jnp.bfloat16),
            jnp.asarray(owner_rows, jnp.bfloat16),
            jnp.asarray(m_gate, jnp.bfloat16),
            jnp.asarray(m_o, jnp.bfloat16),
            jnp.asarray(prefix, jnp.bfloat16),
            stripe=stripe,
            length=length,
            token_owner=token_owner,
        ),
        np.float32,
    )
    expected = _numpy_owner_reference(
        local_normed,
        owner_rows,
        m_gate,
        m_o,
        prefix,
        stripe=stripe,
        length=length,
        token_owner=token_owner,
    )

    np.testing.assert_allclose(actual, expected, rtol=2e-2, atol=2e-1)


def test_should_break_owner_gate_and_output_when_gate_rank_slices_permuted():
    rng = np.random.default_rng(12)
    num_ranks, rows, stripe, cols, hidden = 8, 8, 4, 8, 16
    local_normed = _bf16(rng.standard_normal((rows, hidden)) * 0.5)
    owner_rows = _bf16(rng.standard_normal((num_ranks, rows, cols)) * 0.5)
    m_gate = _bf16(rng.standard_normal((hidden, num_ranks * cols)) * 0.5)
    m_o = _bf16(rng.standard_normal((num_ranks * cols, hidden)) * 0.5)
    prefix = _bf16(rng.standard_normal((rows, hidden)))
    wrong_gate = m_gate.reshape(hidden, num_ranks, cols)[
        :, np.roll(np.arange(num_ranks), 1), :
    ].reshape(hidden, num_ranks * cols)

    actual = np.asarray(
        owner_mla_gate_and_output_ref(
            jnp.asarray(local_normed, jnp.bfloat16),
            jnp.asarray(owner_rows, jnp.bfloat16),
            jnp.asarray(wrong_gate, jnp.bfloat16),
            jnp.asarray(m_o, jnp.bfloat16),
            jnp.asarray(prefix, jnp.bfloat16),
            stripe=stripe,
            length=1000,
            token_owner=0,
        ),
        np.float32,
    )
    expected = _numpy_owner_reference(
        local_normed,
        owner_rows,
        m_gate,
        m_o,
        prefix,
        stripe=stripe,
        length=1000,
        token_owner=0,
    )

    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) > 0.25


def test_should_match_reference_when_projecting_mla_owner_in_pallas():
    rng = np.random.default_rng(13)
    rows, stripe, hidden, slots = 16, 8, 3584, 3
    layer, token_owner = 1, 6
    length = 32 * stripe + token_owner * stripe + 5
    local_normed = jnp.asarray(rng.standard_normal((rows, hidden)) * 0.15, jnp.bfloat16)
    owner_rows = jnp.asarray(
        rng.standard_normal((RANKS, rows, HEAD_COLS)) * 0.15, jnp.bfloat16
    )
    m_gate_fam = jnp.asarray(
        rng.standard_normal((slots, hidden, RANKS * HEAD_COLS)) * 0.04, jnp.bfloat16
    )
    m_o_fam = jnp.asarray(
        rng.standard_normal((slots, RANKS * HEAD_COLS, hidden)) * 0.04, jnp.bfloat16
    )
    prefix = jnp.asarray(rng.standard_normal((rows, hidden)) + 3.0, jnp.bfloat16)

    actual = np.asarray(
        owner_mla_gate_and_output_pallas(
            local_normed,
            owner_rows,
            m_gate_fam,
            m_o_fam,
            prefix,
            layer=layer,
            length=length,
            token_owner=token_owner,
            stripe=stripe,
            interpret=True,
        ),
        np.float32,
    )
    expected = np.asarray(
        owner_mla_gate_and_output_ref(
            local_normed,
            owner_rows,
            m_gate_fam[layer],
            m_o_fam[layer],
            prefix,
            stripe=stripe,
            length=length,
            token_owner=token_owner,
        ),
        np.float32,
    )

    rel_l2 = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
    assert rel_l2 < 1e-3
    np.testing.assert_array_equal(
        actual[stripe + 5 :], np.asarray(prefix, np.float32)[stripe + 5 :]
    )


def test_should_match_reference_for_mla_qkv_up_pallas_with_tail_mask():
    rng = np.random.default_rng(14)
    tokens, q_rank, length = 256, 256, 193
    both_576 = jnp.asarray(
        rng.standard_normal((tokens, q_rank + 576)) * 0.2, jnp.bfloat16
    )
    both_640 = jnp.pad(both_576, ((0, 0), (0, 64)))
    qu = jnp.asarray(rng.standard_normal((q_rank, 3 * 256)) * 0.08, jnp.bfloat16)
    kvu = jnp.asarray(rng.standard_normal((512, 3 * 256)) * 0.08, jnp.bfloat16)

    q_act, k_act, v_act = mla_qkv_up_pallas(
        both_640,
        qu,
        kvu,
        q_rank=q_rank,
        tile=256,
        query_scale=pm.SCALE_LOG2,
        length=jnp.asarray(length, jnp.int32),
        interpret=True,
    )

    q_lat = both_640[:, :q_rank]
    kv_lat = both_640[:, q_rank : q_rank + 512]
    rope = both_640[:, q_rank + 512 : q_rank + 576]
    real = (jnp.arange(tokens) < length)[:, None]
    rope_m = jnp.where(real, rope, 0)
    zeros_64 = jnp.zeros((tokens, 64), jnp.bfloat16)
    q_ref, k_ref, v_ref = [], [], []
    for h in range(3):
        sl = slice(h * 256, (h + 1) * 256)
        q_ref.append(
            (
                jnp.dot(q_lat, qu[:, sl], preferred_element_type=jnp.float32)
                * pm.SCALE_LOG2
            ).astype(jnp.bfloat16)
        )
        kv_h = jnp.dot(kv_lat, kvu[:, sl], preferred_element_type=jnp.float32).astype(
            jnp.bfloat16
        )
        k_ref.append(
            jnp.concatenate(
                (jnp.where(real, kv_h[:, :128], 0), rope_m, zeros_64), axis=1
            )
        )
        v_ref.append(jnp.where(real, kv_h[:, 128:], 0))

    q_ref_arr = np.asarray(jnp.stack(q_ref, axis=0), np.float32)
    k_ref_arr = np.asarray(jnp.stack(k_ref, axis=0), np.float32)
    v_ref_arr = np.asarray(jnp.stack(v_ref, axis=0), np.float32)
    assert (
        np.linalg.norm(np.asarray(q_act, np.float32) - q_ref_arr)
        / np.linalg.norm(q_ref_arr)
        < 1e-3
    )
    assert (
        np.linalg.norm(np.asarray(k_act, np.float32) - k_ref_arr)
        / np.linalg.norm(k_ref_arr)
        < 1e-3
    )
    assert (
        np.linalg.norm(np.asarray(v_act, np.float32) - v_ref_arr)
        / np.linalg.norm(v_ref_arr)
        < 1e-3
    )
    np.testing.assert_array_equal(np.asarray(k_act, np.float32)[:, length:], 0.0)
    np.testing.assert_array_equal(np.asarray(v_act, np.float32)[:, length:], 0.0)


def test_should_replicate_mla_gate_and_output_shares_preserving_rank_order():
    devices = jax.devices()
    ranks = len(devices)
    mesh = Mesh(np.asarray(devices), ("k3_sp",))
    gate_shares = np.arange(ranks * 2 * 4 * 6, dtype=np.float32).reshape(ranks, 2, 4, 6)
    o_shares = np.arange(ranks * 2 * 6 * 4, dtype=np.float32).reshape(ranks, 2, 6, 4)

    gather_gate = jax.jit(
        jax.shard_map(
            functools.partial(gather_gate_shares, axis="k3_sp"),
            mesh=mesh,
            in_specs=(P("k3_sp"),),
            out_specs=(P("k3_sp"),),
            check_vma=False,
        )
    )
    gather_o = jax.jit(
        jax.shard_map(
            functools.partial(gather_output_projection_shares, axis="k3_sp"),
            mesh=mesh,
            in_specs=(P("k3_sp"),),
            out_specs=(P("k3_sp"),),
            check_vma=False,
        )
    )

    (gate_rep,) = gather_gate(jnp.asarray(gate_shares))
    (o_rep,) = gather_o(jnp.asarray(o_shares))

    expected_gate = gate_shares.transpose(1, 2, 0, 3).reshape(2, 4, ranks * 6)
    expected_o = o_shares.transpose(1, 0, 2, 3).reshape(2, ranks * 6, 4)
    np.testing.assert_array_equal(
        np.asarray(gate_rep), np.broadcast_to(expected_gate, (ranks, 2, 4, ranks * 6))
    )
    np.testing.assert_array_equal(
        np.asarray(o_rep), np.broadcast_to(expected_o, (ranks, 2, ranks * 6, 4))
    )


def test_should_match_full_window_mla_baseline_across_sp_ranks_with_tail():
    devices = jax.devices()
    num_ranks = len(devices)
    mesh = Mesh(np.asarray(devices), ("k3_sp",))
    rng = np.random.default_rng(15)
    rows, hidden, q_rank, slots = 128, 128, 128, 2
    stripe = pm.window_stripe(rows)
    tokens = num_ranks * rows
    blocks = rows // stripe
    length = tokens - 5
    slot = 1
    token_ranks = tuple(np.roll(np.arange(num_ranks), 1).tolist())

    full_hidden = jnp.asarray(rng.standard_normal((tokens, hidden)) * 0.2, jnp.bfloat16)
    full_prefix = jnp.asarray(rng.standard_normal((tokens, hidden)) + 2.0, jnp.bfloat16)
    m_qa = jnp.asarray(rng.standard_normal((hidden, q_rank)) * 0.08, jnp.bfloat16)
    m_ka = jnp.asarray(rng.standard_normal((hidden, 576)) * 0.08, jnp.bfloat16)
    m_qnorm = jnp.asarray(rng.uniform(0.8, 1.2, (1, q_rank)), jnp.bfloat16)
    m_knorm = jnp.asarray(rng.uniform(0.8, 1.2, (1, 512)), jnp.bfloat16)
    m_qb_all = jnp.asarray(
        rng.standard_normal((num_ranks, q_rank, 3 * 256)) * 0.06, jnp.bfloat16
    )
    m_kb_all = jnp.asarray(
        rng.standard_normal((num_ranks, 512, 3 * 256)) * 0.06, jnp.bfloat16
    )
    m_gate_all = jnp.asarray(
        rng.standard_normal((num_ranks, slots, hidden, HEAD_COLS)) * 0.06, jnp.bfloat16
    )
    m_o_all = jnp.asarray(
        rng.standard_normal((num_ranks, slots, HEAD_COLS, hidden)) * 0.06, jnp.bfloat16
    )
    m_gate_full = jnp.broadcast_to(
        m_gate_all.transpose(1, 2, 0, 3).reshape(slots, hidden, num_ranks * HEAD_COLS)[
            None
        ],
        (num_ranks, slots, hidden, num_ranks * HEAD_COLS),
    )
    m_o_full = jnp.broadcast_to(
        m_o_all.transpose(1, 0, 2, 3).reshape(slots, num_ranks * HEAD_COLS, hidden)[
            None
        ],
        (num_ranks, slots, num_ranks * HEAD_COLS, hidden),
    )

    def _stripe_for_ranks(window):
        by_owner = window.reshape(blocks, num_ranks, stripe, hidden).transpose(
            1, 0, 2, 3
        )
        return jnp.asarray(by_owner)[jnp.asarray(token_ranks, jnp.int32)].reshape(
            num_ranks, rows, hidden
        )

    local_hidden = _stripe_for_ranks(full_hidden)
    local_prefix = _stripe_for_ranks(full_prefix)

    def _mock_flash(
        query, new_key, new_value, length_arg, seg_desc, *, count, tile, arena
    ):
        del length_arg, seg_desc, count, tile, arena
        score = jnp.einsum(
            "htd,hsd->hts", query.astype(jnp.float32), new_key.astype(jnp.float32)
        )
        mask = jnp.tril(jnp.ones((tokens, tokens), dtype=jnp.bool_))
        attn = jax.nn.softmax(jnp.where(mask[None], score * 0.05, -1e9), axis=-1)
        return jnp.einsum("hts,hsd->htd", attn, new_value.astype(jnp.float32)).astype(
            jnp.bfloat16
        )

    def _a2a_fn(lh, px, mqa, mqb, mka, mkb, mqn, mkn, mg, mo, seg, lng, slt):
        out, lat = mla_local_normed_a2a_attention(
            lh[0],
            px[0],
            query_down_projection=mqa[0],
            query_up_projection=mqb[0],
            key_value_down_projection=mka[0],
            key_value_up_projection=mkb[0],
            query_normalization=mqn[0],
            key_value_normalization=mkn[0],
            m_gate_family=mg[0],
            m_o_family=mo[0],
            slot=slt,
            length=lng,
            segment_desc=seg,
            num_ranks=num_ranks,
            sp_axis="k3_sp",
            segments=1,
            flash_tile=64,
            token_ranks=token_ranks,
            tables=None,
            flash_fn=_mock_flash,
            use_pallas_a2a=False,
            use_pallas_owner=False,
            interpret=True,
        )
        return out[None], lat[None]

    def _bcast(x):
        return jnp.broadcast_to(x[None], (num_ranks, *x.shape))

    seg_desc = jnp.asarray([0, length, 0, 0, 0], jnp.int32)
    after_a2a, lat_a2a = jax.shard_map(
        _a2a_fn,
        mesh=mesh,
        in_specs=(P("k3_sp"),) * 10 + (P(), P(), P()),
        out_specs=(P("k3_sp"), P("k3_sp")),
        check_vma=False,
    )(
        local_hidden,
        local_prefix,
        _bcast(m_qa),
        m_qb_all,
        _bcast(m_ka),
        m_kb_all,
        _bcast(m_qnorm),
        _bcast(m_knorm),
        m_gate_full,
        m_o_full,
        seg_desc,
        jnp.asarray(length, jnp.int32),
        jnp.asarray(slot, jnp.int32),
    )

    full_proj_sum = np.zeros((tokens, hidden), dtype=np.float32)
    latent_ref = None
    for r in range(num_ranks):
        pw = pm.ProjectionWeights(
            m_qa,
            m_qb_all[r],
            m_ka,
            m_kb_all[r],
            m_gate_all[r, slot],
            m_qnorm[0],
            m_knorm[0],
        )
        q, k, v, og, lat = pm.mla_projections(
            full_hidden,
            pw,
            return_latent=True,
            query_scale=pm.SCALE_LOG2,
            head_major=True,
            length=jnp.asarray(length, jnp.int32),
            interpret=True,
        )
        if latent_ref is None:
            latent_ref = np.asarray(lat, np.float32)
        att = _mock_flash(q, k, v, length, seg_desc, count=1, tile=64, arena=None)
        att_flat = att.transpose(1, 0, 2).reshape(tokens, HEAD_COLS)
        gate = jax.nn.sigmoid(og.astype(jnp.float32)).astype(jnp.bfloat16)
        gated = (att_flat * gate).astype(jnp.bfloat16)
        full_proj_sum += np.asarray(
            jnp.dot(gated, m_o_all[r, slot], preferred_element_type=jnp.float32),
            np.float32,
        )
    valid = (np.arange(tokens) < length)[:, None]
    expected_full = _bf16(
        np.asarray(full_prefix, np.float32) + np.where(valid, full_proj_sum, 0.0)
    )
    expected_local = np.asarray(
        _stripe_for_ranks(jnp.asarray(expected_full, jnp.bfloat16)), np.float32
    )

    actual_local = np.asarray(after_a2a, np.float32)
    rel_l2 = np.linalg.norm(actual_local - expected_local) / np.linalg.norm(
        expected_local
    )
    lat_rel = np.linalg.norm(
        np.asarray(lat_a2a[0], np.float32) - latent_ref
    ) / np.linalg.norm(latent_ref)
    assert rel_l2 < 5e-3
    assert lat_rel < 5e-3


@pytest.mark.parametrize(
    ("a2a", "builder_tag"), [(True, "MLA (all-to-all)"), (False, "norm/gather")]
)
def test_should_dispatch_indexed_mla_step_on_flag(monkeypatch, a2a, builder_tag):
    monkeypatch.setattr(k3_config, "_ACTIVE", KernelConfig(prefill_mla_a2a=a2a))
    one_device_mesh = Mesh(np.asarray(jax.devices()[:1]), ("k3_sp",))

    with pytest.raises(ValueError, match="TP32 mesh") as error:
        pm.make_indexed_mla_step(
            one_device_mesh,
            seq_len=16384,
            segments=2,
            prefix_capacity=4096,
            flash_tile=256,
        )

    assert builder_tag in str(error.value)
    assert ("(all-to-all)" in str(error.value)) is a2a


def test_should_replicate_mla_gate_and_output_in_prefill_load_without_mutating_shared(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(k3_config, "_ACTIVE", KernelConfig(prefill_mla_a2a=True))
    one_mesh = Mesh(np.asarray(jax.devices()[:1]), ("k3_sp",))
    monkeypatch.setattr(k3_prefill, "kernel_mesh", lambda axis, size: (one_mesh, 1))

    prog = k3_prefill.PrefillProgram(windows=(512,), capacity=512, slots=1)
    prog.names = {"mla": ("m_gate", "m_o"), "moe": ("expert_ds",)}

    shared_gate = object()
    shared = {"m_gate": shared_gate, "expert_ds": object()}
    loaded_mo = object()

    def _fake_stream(iterator, store):
        del iterator
        store("mla", "m_o", loaded_mo)
        return {"gib": 0.1, "wall": 0.1}

    registered_ops = []

    def _fake_register(name, mesh, fn, in_specs, out_specs):
        del mesh, fn, in_specs, out_specs
        registered_ops.append(name)
        return lambda x: ((f"replicated:{name}", x),)

    monkeypatch.setattr(k3_prefill, "stream_to_device", _fake_stream)
    monkeypatch.setattr(k3_prefill, "register_op", _fake_register)
    monkeypatch.setattr(k3_prefill, "materialize", lambda x: None)
    monkeypatch.setattr(k3_prefill, "report_load_stats", lambda *a, **kw: None)

    prog.load_weights(tmp_path, shared=shared)

    assert registered_ops == [
        "pallas::k3_prefill_mla_gate_projection_replicate",
        "pallas::k3_prefill_mla_output_projection_replicate",
    ]
    assert prog.weights["mla"]["m_gate"] == (
        "replicated:pallas::k3_prefill_mla_gate_projection_replicate",
        shared_gate,
    )
    assert prog.weights["mla"]["m_o"] == (
        "replicated:pallas::k3_prefill_mla_output_projection_replicate",
        loaded_mo,
    )
    assert shared["m_gate"] is shared_gate


def test_should_keep_mla_a2a_fp8_off_by_default_and_require_mla_a2a_when_enabled(
    monkeypatch,
):
    monkeypatch.setenv("K3_PREFILL_MLA_A2A", "0")
    monkeypatch.delenv("K3_PREFILL_MLA_A2A_FP8", raising=False)
    assert KernelConfig().prefill_mla_a2a_fp8 is False

    monkeypatch.setenv("K3_PREFILL_MLA_A2A", "1")
    monkeypatch.setenv("K3_PREFILL_MLA_A2A_FP8", "1")
    cfg = KernelConfig()
    assert cfg.prefill_mla_a2a is True
    assert cfg.prefill_mla_a2a_fp8 is True

    with pytest.raises(ValueError, match="prefill_mla_a2a=True"):
        KernelConfig(prefill_mla_a2a=False, prefill_mla_a2a_fp8=True)


def test_should_reject_mismatched_or_missing_fp8_scales_in_owner_mla_pallas():
    rows, hidden, slots = 16, 3584, 2
    ln = jnp.zeros((rows, hidden), jnp.bfloat16)
    orw = jnp.zeros((RANKS, rows, HEAD_COLS), jnp.bfloat16)
    px = jnp.zeros((rows, hidden), jnp.bfloat16)
    mg_fp8 = jnp.zeros((slots, hidden, RANKS * HEAD_COLS), jnp.float8_e4m3fn)
    mo_fp8 = jnp.zeros((slots, RANKS * HEAD_COLS, hidden), jnp.float8_e4m3fn)
    mg_bf16 = jnp.zeros((slots, hidden, RANKS * HEAD_COLS), jnp.bfloat16)
    mo_bf16 = jnp.zeros((slots, RANKS * HEAD_COLS, hidden), jnp.bfloat16)
    gs_ok = jnp.ones((slots, 8, RANKS * HEAD_COLS), jnp.float32)
    os_ok = jnp.ones((slots, 8, hidden), jnp.float32)

    with pytest.raises(ValueError, match="share BF16 or FP8 dtype"):
        owner_mla_gate_and_output_pallas(
            ln, orw, mg_fp8, mo_bf16, px, layer=0, length=16, token_owner=0, stripe=8
        )

    with pytest.raises(ValueError, match="require m_gate_scales and m_o_scales"):
        owner_mla_gate_and_output_pallas(
            ln, orw, mg_fp8, mo_fp8, px, layer=0, length=16, token_owner=0, stripe=8
        )

    with pytest.raises(ValueError, match="scales must be"):
        owner_mla_gate_and_output_pallas(
            ln,
            orw,
            mg_fp8,
            mo_fp8,
            px,
            layer=0,
            length=16,
            token_owner=0,
            stripe=8,
            m_gate_scales=jnp.ones((slots, 1, RANKS * HEAD_COLS), jnp.float32),
            m_o_scales=os_ok,
        )

    with pytest.raises(ValueError, match="require float8_e4m3fn"):
        owner_mla_gate_and_output_pallas(
            ln,
            orw,
            mg_bf16,
            mo_bf16,
            px,
            layer=0,
            length=16,
            token_owner=0,
            stripe=8,
            m_gate_scales=gs_ok,
            m_o_scales=os_ok,
        )


def test_should_reject_incompatible_shapes_in_mla_qkv_up_unstriped_pallas():
    qu = jnp.zeros((128, 3 * 256), jnp.bfloat16)
    kvu = jnp.zeros((512, 3 * 256), jnp.bfloat16)
    with pytest.raises(ValueError, match="incompatible shapes"):
        mla_qkv_up_unstriped_pallas(
            jnp.zeros((512, 128 + 640), jnp.bfloat16),
            qu,
            kvu,
            num_ranks=4,
            stripe=64,
            q_rank=128,
            tile=512,
            query_scale=pm.SCALE_LOG2,
            length=512,
            interpret=True,
        )


@pytest.mark.parametrize(
    ("num_ranks", "blocks", "tile", "length"),
    [(4, 1, 256, 193), (8, 4, 256, 1701)],
)
def test_should_match_unstripe_and_qkv_up_bitwise_across_multi_block_tiles_with_tail(
    num_ranks, blocks, tile, length
):
    rng = np.random.default_rng(3901 + blocks)
    stripe, q_rank = 64, 128
    tokens = num_ranks * blocks * stripe
    raw_np = (
        rng.standard_normal((num_ranks, blocks, stripe, q_rank + 576)).astype(
            np.float32
        )
        * 0.25
    )
    owner_bias = (np.arange(num_ranks, dtype=np.float32) * 0.3)[:, None, None, None]
    block_bias = (np.arange(blocks, dtype=np.float32) * 0.7)[None, :, None, None]
    raw_np = raw_np + owner_bias - block_bias
    raw_padded = np.pad(raw_np, ((0, 0), (0, 0), (0, 0), (0, 64))).reshape(
        tokens, q_rank + 640
    )
    both_raw = jnp.asarray(raw_padded, jnp.bfloat16)
    qu = jnp.asarray(rng.standard_normal((q_rank, 3 * 256)) * 0.08, jnp.bfloat16)
    kvu = jnp.asarray(rng.standard_normal((512, 3 * 256)) * 0.08, jnp.bfloat16)

    q_act, k_act, v_act, lat_act = mla_qkv_up_unstriped_pallas(
        both_raw,
        qu,
        kvu,
        num_ranks=num_ranks,
        stripe=stripe,
        q_rank=q_rank,
        tile=tile,
        query_scale=pm.SCALE_LOG2,
        length=jnp.asarray(length, jnp.int32),
        interpret=True,
    )

    both_unstriped = unstripe_sp_window(both_raw, num_ranks, stripe)
    q_ref, k_ref, v_ref = mla_qkv_up_pallas(
        both_unstriped,
        qu,
        kvu,
        q_rank=q_rank,
        tile=tile,
        query_scale=pm.SCALE_LOG2,
        length=jnp.asarray(length, jnp.int32),
        interpret=True,
    )
    lat_ref = both_unstriped[:, q_rank : q_rank + 576]

    np.testing.assert_array_equal(np.asarray(q_act), np.asarray(q_ref))
    np.testing.assert_array_equal(np.asarray(k_act), np.asarray(k_ref))
    np.testing.assert_array_equal(np.asarray(v_act), np.asarray(v_ref))
    np.testing.assert_array_equal(np.asarray(lat_act), np.asarray(lat_ref))
    np.testing.assert_array_equal(np.asarray(k_act, np.float32)[:, length:], 0.0)
    np.testing.assert_array_equal(np.asarray(v_act, np.float32)[:, length:], 0.0)

    if blocks > 1:
        wrong_lat = both_raw[:, q_rank : q_rank + 576]
        diff_wrong = np.linalg.norm(
            np.asarray(lat_act, np.float32) - np.asarray(wrong_lat, np.float32)
        )
        assert diff_wrong > 10.0


def test_should_quantize_and_gather_mla_shares_with_cross_rank_outliers():
    devices = jax.devices()
    ranks = len(devices)
    mesh = Mesh(np.asarray(devices), ("k3_sp",))
    rng = np.random.default_rng(3902)
    slots, hidden, cols = 2, 16, 8

    gate_np = (
        rng.standard_normal((ranks, slots, hidden, cols)).astype(np.float32) * 0.05
    )
    gate_np[-1, 0, :, 3] *= 20.0
    o_np = rng.standard_normal((ranks, slots, cols, hidden)).astype(np.float32) * 0.05
    o_np[-1, 0, 2, 5] *= 20.0

    gather_g_fp8 = jax.jit(
        jax.shard_map(
            functools.partial(gather_gate_shares_fp8, axis="k3_sp"),
            mesh=mesh,
            in_specs=(P("k3_sp"),),
            out_specs=(P("k3_sp"), P("k3_sp")),
            check_vma=False,
        )
    )
    gather_o_fp8 = jax.jit(
        jax.shard_map(
            functools.partial(gather_output_projection_shares_fp8, axis="k3_sp"),
            mesh=mesh,
            in_specs=(P("k3_sp"),),
            out_specs=(P("k3_sp"), P("k3_sp")),
            check_vma=False,
        )
    )

    gate_sh = jnp.asarray(gate_np, jnp.bfloat16)
    o_sh = jnp.asarray(o_np, jnp.bfloat16)
    g_fp8, g_scale = gather_g_fp8(gate_sh)
    o_fp8, o_scale = gather_o_fp8(o_sh)

    assert g_fp8.dtype == jnp.float8_e4m3fn
    assert o_fp8.dtype == jnp.float8_e4m3fn
    assert g_scale.shape == (ranks, slots, 8, ranks * cols)
    assert o_scale.shape == (ranks, slots, 8, hidden)

    g_deq = np.asarray(g_fp8[0], np.float32) * np.asarray(
        g_scale[0, :, 0:1, :], np.float32
    )
    o_deq = np.asarray(o_fp8[0], np.float32) * np.asarray(
        o_scale[0, :, 0:1, :], np.float32
    )
    g_ref = (
        np.asarray(gate_sh, np.float32)
        .transpose(1, 2, 0, 3)
        .reshape(slots, hidden, ranks * cols)
    )
    o_ref = (
        np.asarray(o_sh, np.float32)
        .transpose(1, 0, 2, 3)
        .reshape(slots, ranks * cols, hidden)
    )

    o_rank_diff = np.max(
        np.abs(np.asarray(o_fp8, np.float32) - np.asarray(o_fp8[:1], np.float32))
    )
    assert o_rank_diff == 0.0
    assert np.linalg.norm(g_deq - g_ref) / np.linalg.norm(g_ref) < 0.03
    assert np.linalg.norm(o_deq - o_ref) / np.linalg.norm(o_ref) < 0.03


def test_should_match_fp32_oracle_for_fp8_mla_owner_pallas_with_outliers_and_padding():
    rng = np.random.default_rng(3903)
    rows, stripe, hidden, slots = 16, 8, 3584, 3
    layer, token_owner = 1, 6
    length = 32 * stripe + token_owner * stripe + 5

    ln_np = rng.standard_normal((rows, hidden)).astype(np.float32) * 0.15
    ln_np[2, :] *= 10.0
    ln_np[stripe + 5 :, :] = 777.0
    or_np = rng.standard_normal((RANKS, rows, HEAD_COLS)).astype(np.float32) * 0.15
    or_np[4, :, :] *= 10.0
    or_np[:, stripe + 5 :, :] = 888.0
    mg_np = (
        rng.standard_normal((slots, hidden, RANKS * HEAD_COLS)).astype(np.float32)
        * 0.04
    )
    mg_np[layer, :, 513] *= 12.0
    mo_np = (
        rng.standard_normal((slots, RANKS * HEAD_COLS, hidden)).astype(np.float32)
        * 0.04
    )
    mo_np[layer, :, 1025] *= 12.0
    mg_np[0] = 400.0
    mo_np[2] = -400.0
    pref_np = rng.standard_normal((rows, hidden)).astype(np.float32) + 3.0

    mg_bf16 = jnp.asarray(mg_np, jnp.bfloat16)
    mo_bf16 = jnp.asarray(mo_np, jnp.bfloat16)
    one_mesh = Mesh(np.asarray(jax.devices()[:1]), ("k3_sp",))
    mg_fp8, mg_scale = jax.shard_map(
        functools.partial(gather_gate_shares_fp8, axis="k3_sp"),
        mesh=one_mesh,
        in_specs=(P("k3_sp"),),
        out_specs=(P("k3_sp"), P("k3_sp")),
        check_vma=False,
    )(mg_bf16[None])
    mo_fp8, mo_scale = jax.shard_map(
        functools.partial(gather_output_projection_shares_fp8, axis="k3_sp"),
        mesh=one_mesh,
        in_specs=(P("k3_sp"),),
        out_specs=(P("k3_sp"), P("k3_sp")),
        check_vma=False,
    )(mo_bf16[None])

    local_normed = jnp.asarray(ln_np, jnp.bfloat16)
    owner_rows = jnp.asarray(or_np, jnp.bfloat16)
    prefix = jnp.asarray(pref_np, jnp.bfloat16)

    actual = np.asarray(
        owner_mla_gate_and_output_pallas(
            local_normed,
            owner_rows,
            mg_fp8[0],
            mo_fp8[0],
            prefix,
            layer=layer,
            length=length,
            token_owner=token_owner,
            stripe=stripe,
            m_gate_scales=mg_scale[0],
            m_o_scales=mo_scale[0],
            interpret=True,
        ),
        np.float32,
    )
    expected = _numpy_owner_reference(
        _bf16(ln_np),
        _bf16(or_np),
        _bf16(mg_np[layer]),
        _bf16(mo_np[layer]),
        _bf16(pref_np),
        stripe=stripe,
        length=length,
        token_owner=token_owner,
    )

    rel_l2 = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
    assert rel_l2 < 0.025
    np.testing.assert_array_equal(
        actual[stripe + 5 :], np.asarray(prefix, np.float32)[stripe + 5 :]
    )

    bad_mo_scale = mo_scale[0].at[layer].multiply(0.1)
    bad_actual = np.asarray(
        owner_mla_gate_and_output_pallas(
            local_normed,
            owner_rows,
            mg_fp8[0],
            mo_fp8[0],
            prefix,
            layer=layer,
            length=length,
            token_owner=token_owner,
            stripe=stripe,
            m_gate_scales=mg_scale[0],
            m_o_scales=bad_mo_scale,
            interpret=True,
        ),
        np.float32,
    )
    assert np.linalg.norm(bad_actual - expected) / np.linalg.norm(expected) > 0.15


def test_should_match_window_mla_baseline_when_mla_a2a_runs_fp8_and_fused_unstripe():
    devices = jax.devices()
    num_ranks = len(devices)
    mesh = Mesh(np.asarray(devices), ("k3_sp",))
    rng = np.random.default_rng(3904)
    rows, hidden, q_rank, slots = 128, 128, 128, 2
    stripe = pm.window_stripe(rows)
    tokens = max(256, num_ranks * rows)
    rows = tokens // num_ranks
    stripe = pm.window_stripe(rows)
    blocks = rows // stripe
    length = tokens - 5
    slot = 1
    token_ranks = tuple(np.roll(np.arange(num_ranks), 1).tolist())

    full_hidden = jnp.asarray(rng.standard_normal((tokens, hidden)) * 0.2, jnp.bfloat16)
    full_prefix = jnp.asarray(rng.standard_normal((tokens, hidden)) + 2.0, jnp.bfloat16)
    m_qa = jnp.asarray(rng.standard_normal((hidden, q_rank)) * 0.08, jnp.bfloat16)
    m_ka = jnp.asarray(rng.standard_normal((hidden, 576)) * 0.08, jnp.bfloat16)
    m_qnorm = jnp.asarray(rng.uniform(0.8, 1.2, (1, q_rank)), jnp.bfloat16)
    m_knorm = jnp.asarray(rng.uniform(0.8, 1.2, (1, 512)), jnp.bfloat16)
    m_qb_all = jnp.asarray(
        rng.standard_normal((num_ranks, q_rank, 3 * 256)) * 0.06, jnp.bfloat16
    )
    m_kb_all = jnp.asarray(
        rng.standard_normal((num_ranks, 512, 3 * 256)) * 0.06, jnp.bfloat16
    )
    m_gate_sh = jnp.asarray(
        rng.standard_normal((num_ranks, slots, hidden, HEAD_COLS)) * 0.06, jnp.bfloat16
    )
    m_o_sh = jnp.asarray(
        rng.standard_normal((num_ranks, slots, HEAD_COLS, hidden)) * 0.06, jnp.bfloat16
    )
    m_gate_fp8, m_gate_scale = jax.shard_map(
        functools.partial(gather_gate_shares_fp8, axis="k3_sp"),
        mesh=mesh,
        in_specs=(P("k3_sp"),),
        out_specs=(P("k3_sp"), P("k3_sp")),
        check_vma=False,
    )(m_gate_sh)
    m_o_fp8, m_o_scale = jax.shard_map(
        functools.partial(gather_output_projection_shares_fp8, axis="k3_sp"),
        mesh=mesh,
        in_specs=(P("k3_sp"),),
        out_specs=(P("k3_sp"), P("k3_sp")),
        check_vma=False,
    )(m_o_sh)

    def _stripe_for_ranks(window):
        by_owner = window.reshape(blocks, num_ranks, stripe, hidden).transpose(
            1, 0, 2, 3
        )
        return jnp.asarray(by_owner)[jnp.asarray(token_ranks, jnp.int32)].reshape(
            num_ranks, rows, hidden
        )

    local_hidden = _stripe_for_ranks(full_hidden)
    local_prefix = _stripe_for_ranks(full_prefix)

    def _mock_flash(
        query, new_key, new_value, length_arg, seg_desc, *, count, tile, arena
    ):
        del length_arg, seg_desc, count, tile, arena
        score = jnp.einsum(
            "htd,hsd->hts", query.astype(jnp.float32), new_key.astype(jnp.float32)
        )
        mask = jnp.tril(jnp.ones((tokens, tokens), dtype=jnp.bool_))
        attn = jax.nn.softmax(jnp.where(mask[None], score * 0.05, -1e9), axis=-1)
        return jnp.einsum("hts,hsd->htd", attn, new_value.astype(jnp.float32)).astype(
            jnp.bfloat16
        )

    def _a2a_fp8_fn(
        lh, px, mqa, mqb, mka, mkb, mqn, mkn, mg, mo, mgs, mos, seg, lng, slt
    ):
        out, lat = mla_local_normed_a2a_attention(
            lh[0],
            px[0],
            query_down_projection=mqa[0],
            query_up_projection=mqb[0],
            key_value_down_projection=mka[0],
            key_value_up_projection=mkb[0],
            query_normalization=mqn[0],
            key_value_normalization=mkn[0],
            m_gate_family=mg[0],
            m_o_family=mo[0],
            m_gate_scales=mgs[0],
            m_o_scales=mos[0],
            slot=slt,
            length=lng,
            segment_desc=seg,
            num_ranks=num_ranks,
            sp_axis="k3_sp",
            segments=1,
            flash_tile=64,
            token_ranks=token_ranks,
            tables=None,
            flash_fn=_mock_flash,
            use_pallas_a2a=False,
            use_pallas_owner=False,
            interpret=True,
        )
        return out[None], lat[None]

    def _bcast(x):
        return jnp.broadcast_to(x[None], (num_ranks, *x.shape))

    seg_desc = jnp.asarray([0, length, 0, 0, 0], jnp.int32)
    after_a2a, lat_a2a = jax.shard_map(
        _a2a_fp8_fn,
        mesh=mesh,
        in_specs=(P("k3_sp"),) * 12 + (P(), P(), P()),
        out_specs=(P("k3_sp"), P("k3_sp")),
        check_vma=False,
    )(
        local_hidden,
        local_prefix,
        _bcast(m_qa),
        m_qb_all,
        _bcast(m_ka),
        m_kb_all,
        _bcast(m_qnorm),
        _bcast(m_knorm),
        m_gate_fp8,
        m_o_fp8,
        m_gate_scale,
        m_o_scale,
        seg_desc,
        jnp.asarray(length, jnp.int32),
        jnp.asarray(slot, jnp.int32),
    )

    full_proj_sum = np.zeros((tokens, hidden), dtype=np.float32)
    latent_ref = None
    for r in range(num_ranks):
        pw = pm.ProjectionWeights(
            m_qa,
            m_qb_all[r],
            m_ka,
            m_kb_all[r],
            m_gate_sh[r, slot],
            m_qnorm[0],
            m_knorm[0],
        )
        q, k, v, og, lat = pm.mla_projections(
            full_hidden,
            pw,
            return_latent=True,
            query_scale=pm.SCALE_LOG2,
            head_major=True,
            length=jnp.asarray(length, jnp.int32),
            interpret=True,
        )
        if latent_ref is None:
            latent_ref = np.asarray(lat, np.float32)
        att = _mock_flash(q, k, v, length, seg_desc, count=1, tile=64, arena=None)
        att_flat = att.transpose(1, 0, 2).reshape(tokens, HEAD_COLS)
        gate = jax.nn.sigmoid(og.astype(jnp.float32)).astype(jnp.bfloat16)
        gated = (att_flat * gate).astype(jnp.bfloat16)
        full_proj_sum += np.asarray(
            jnp.dot(gated, m_o_sh[r, slot], preferred_element_type=jnp.float32),
            np.float32,
        )
    valid = (np.arange(tokens) < length)[:, None]
    expected_full = _bf16(
        np.asarray(full_prefix, np.float32) + np.where(valid, full_proj_sum, 0.0)
    )
    expected_local = np.asarray(
        _stripe_for_ranks(jnp.asarray(expected_full, jnp.bfloat16)), np.float32
    )

    actual_local = np.asarray(after_a2a, np.float32)
    rel_l2 = np.linalg.norm(actual_local - expected_local) / np.linalg.norm(
        expected_local
    )
    lat_rel = np.linalg.norm(
        np.asarray(lat_a2a[0], np.float32) - latent_ref
    ) / np.linalg.norm(latent_ref)
    assert rel_l2 < 0.025
    assert lat_rel < 5e-3


def test_should_replicate_fp8_mla_shares_in_prefill_load_without_mutating_shared(
    monkeypatch, tmp_path
):
    fp8_cfg = KernelConfig(prefill_mla_a2a=True, prefill_mla_a2a_fp8=True)
    monkeypatch.setattr(k3_config, "active", lambda: fp8_cfg)
    monkeypatch.setattr(k3_prefill, "active", lambda: fp8_cfg)
    one_mesh = Mesh(np.asarray(jax.devices()[:1]), ("k3_sp",))
    monkeypatch.setattr(k3_prefill, "kernel_mesh", lambda axis, size: (one_mesh, 1))

    prog = k3_prefill.PrefillProgram(windows=(512,), capacity=512, slots=1)
    assert "m_gate_scale" in prog.names["mla"]
    assert "m_o_scale" in prog.names["mla"]
    prog.names = {
        "mla": ("m_gate", "m_o", "m_gate_scale", "m_o_scale"),
        "moe": ("expert_ds",),
    }

    shared_gate = object()
    shared = {"m_gate": shared_gate, "expert_ds": object()}
    loaded_mo = object()

    def _fake_stream(iterator, store):
        del iterator
        store("mla", "m_o", loaded_mo)
        return {"gib": 0.1, "wall": 0.1}

    registered_ops = []
    materialized = []

    def _fake_register(name, mesh, fn, in_specs, out_specs):
        del mesh, fn, in_specs
        assert len(out_specs) == 2
        registered_ops.append(name)
        return lambda x: ((f"fp8:{name}", x), (f"scale:{name}", x))

    monkeypatch.setattr(k3_prefill, "stream_to_device", _fake_stream)
    monkeypatch.setattr(k3_prefill, "register_op", _fake_register)
    monkeypatch.setattr(k3_prefill, "materialize", materialized.append)
    monkeypatch.setattr(k3_prefill, "report_load_stats", lambda *a, **kw: None)

    prog.load_weights(tmp_path, shared=shared)

    assert registered_ops == [
        "pallas::k3_prefill_mla_gate_projection_replicate_fp8",
        "pallas::k3_prefill_mla_output_projection_replicate_fp8",
    ]
    assert prog.weights["mla"]["m_gate"] == (
        "fp8:pallas::k3_prefill_mla_gate_projection_replicate_fp8",
        shared_gate,
    )
    assert prog.weights["mla"]["m_gate_scale"] == (
        "scale:pallas::k3_prefill_mla_gate_projection_replicate_fp8",
        shared_gate,
    )
    assert prog.weights["mla"]["m_o"] == (
        "fp8:pallas::k3_prefill_mla_output_projection_replicate_fp8",
        loaded_mo,
    )
    assert prog.weights["mla"]["m_o_scale"] == (
        "scale:pallas::k3_prefill_mla_output_projection_replicate_fp8",
        loaded_mo,
    )
    assert len(materialized) == 5
    assert shared["m_gate"] is shared_gate


def test_should_reject_invalid_mla_chip_pair_shapes_and_dtypes():
    with pytest.raises(ValueError, match="mutually exclusive"):
        KernelConfig(
            prefill_mla_a2a=True,
            prefill_mla_a2a_fp8=True,
            prefill_a2a_chip_shard=True,
        )

    with pytest.raises(ValueError, match="chip-pair MLA projection"):
        owner_mla_gate_and_output_chip_pair_pallas(
            jnp.zeros((16, 7168), jnp.bfloat16),
            jnp.zeros((4, 16, HEAD_COLS), jnp.bfloat16),
            jnp.zeros((2, 7168, 6144), jnp.bfloat16),
            jnp.zeros((2, 12288, 3584), jnp.bfloat16),
            jnp.zeros((16, 7168), jnp.bfloat16),
            layer=0,
            length=512,
            own_even=0,
            own_odd=1,
            stripe=16,
        )

    with pytest.raises(ValueError, match="chip-pair MLA projection"):
        owner_mla_gate_and_output_chip_pair_pallas(
            jnp.zeros((16, 7168), jnp.bfloat16),
            jnp.zeros((RANKS, 16, HEAD_COLS), jnp.bfloat16),
            jnp.zeros((2, 7168, 6144), jnp.float8_e4m3fn),
            jnp.zeros((2, 12288, 3584), jnp.bfloat16),
            jnp.zeros((16, 7168), jnp.bfloat16),
            layer=0,
            length=512,
            own_even=0,
            own_odd=1,
            stripe=16,
        )

    with pytest.raises(ValueError, match="even R"):
        owner_mla_gate_and_output_chip_pair_ref(
            jnp.zeros((8, 16), jnp.bfloat16),
            jnp.zeros((3, 8, 4), jnp.bfloat16),
            jnp.zeros((16, 6), jnp.bfloat16),
            jnp.zeros((12, 8), jnp.bfloat16),
            jnp.zeros((8, 16), jnp.bfloat16),
            stripe=4,
            length=20,
            own_even=0,
            own_odd=1,
        )


def test_should_shard_mla_shares_in_prefill_load_without_mutating_shared(
    monkeypatch, tmp_path
):
    chip_cfg = KernelConfig(prefill_mla_a2a=True, prefill_a2a_chip_shard=True)
    monkeypatch.setattr(k3_config, "active", lambda: chip_cfg)
    monkeypatch.setattr(k3_prefill, "active", lambda: chip_cfg)
    one_mesh = Mesh(np.asarray(jax.devices()[:1]), ("k3_sp",))
    monkeypatch.setattr(k3_prefill, "kernel_mesh", lambda axis, size: (one_mesh, 1))

    prog = k3_prefill.PrefillProgram(windows=(512,), capacity=512, slots=1)
    assert "m_gate_scale" not in prog.names["mla"]
    assert "m_o_scale" not in prog.names["mla"]
    prog.names = {"mla": ("m_gate", "m_o"), "moe": ("expert_ds",)}

    shared_gate = object()
    shared = {"m_gate": shared_gate, "expert_ds": object()}
    loaded_mo = object()

    def _fake_stream(iterator, store):
        del iterator
        store("mla", "m_o", loaded_mo)
        return {"gib": 0.1, "wall": 0.1}

    registered_ops = []
    materialized = []

    def _fake_register(name, mesh, fn, in_specs, out_specs):
        del mesh, fn, in_specs
        assert len(out_specs) == 1
        registered_ops.append(name)
        return lambda x: ((f"chip_pair:{name}", x),)

    monkeypatch.setattr(k3_prefill, "stream_to_device", _fake_stream)
    monkeypatch.setattr(k3_prefill, "register_op", _fake_register)
    monkeypatch.setattr(k3_prefill, "materialize", materialized.append)
    monkeypatch.setattr(k3_prefill, "report_load_stats", lambda *a, **kw: None)

    prog.load_weights(tmp_path, shared=shared)

    assert registered_ops == [
        "pallas::k3_prefill_mla_gate_projection_chip_pair",
        "pallas::k3_prefill_mla_output_projection_chip_pair",
    ]
    assert prog.weights["mla"]["m_gate"] == (
        "chip_pair:pallas::k3_prefill_mla_gate_projection_chip_pair",
        shared_gate,
    )
    assert prog.weights["mla"]["m_o"] == (
        "chip_pair:pallas::k3_prefill_mla_output_projection_chip_pair",
        loaded_mo,
    )
    assert len(materialized) == 3
    assert shared["m_gate"] is shared_gate


def test_should_match_replicated_mla_a2a_bitwise_when_chip_pair_sharded():
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("chip-pair MLA test requires >= 2 JAX devices")
    even_devs = devices[: len(devices) - (len(devices) % 2)]
    num_ranks = len(even_devs)
    mesh = Mesh(np.asarray(even_devs), ("k3_sp",))
    sh = jax.sharding.NamedSharding(mesh, P("k3_sp"))
    rep = jax.sharding.NamedSharding(mesh, P())
    rng = np.random.default_rng(412)

    rows, hidden, q_rank, slots = 128, 128, 128, 2
    tokens = num_ranks * rows
    length = tokens - 7
    slot = 1
    token_ranks = tuple(np.roll(np.arange(num_ranks), 1).tolist())

    local_hidden = jax.device_put(
        jnp.asarray(rng.standard_normal((num_ranks, rows, hidden)) * 0.2, jnp.bfloat16),
        sh,
    )
    local_prefix = jax.device_put(
        jnp.asarray(rng.standard_normal((num_ranks, rows, hidden)) + 2.0, jnp.bfloat16),
        sh,
    )
    m_qa = jax.device_put(
        jnp.broadcast_to(
            jnp.asarray(rng.standard_normal((hidden, q_rank)) * 0.08, jnp.bfloat16)[
                None
            ],
            (num_ranks, hidden, q_rank),
        ),
        sh,
    )
    m_ka = jax.device_put(
        jnp.broadcast_to(
            jnp.asarray(rng.standard_normal((hidden, 576)) * 0.08, jnp.bfloat16)[None],
            (num_ranks, hidden, 576),
        ),
        sh,
    )
    m_qnorm = jax.device_put(
        jnp.broadcast_to(
            jnp.asarray(rng.uniform(0.8, 1.2, (1, q_rank)), jnp.bfloat16)[None],
            (num_ranks, 1, q_rank),
        ),
        sh,
    )
    m_knorm = jax.device_put(
        jnp.broadcast_to(
            jnp.asarray(rng.uniform(0.8, 1.2, (1, 512)), jnp.bfloat16)[None],
            (num_ranks, 1, 512),
        ),
        sh,
    )
    m_qb = jax.device_put(
        jnp.asarray(
            rng.standard_normal((num_ranks, q_rank, 3 * 256)) * 0.06, jnp.bfloat16
        ),
        sh,
    )
    m_kb = jax.device_put(
        jnp.asarray(
            rng.standard_normal((num_ranks, 512, 3 * 256)) * 0.06, jnp.bfloat16
        ),
        sh,
    )
    gate_shares = jax.device_put(
        jnp.asarray(
            rng.standard_normal((num_ranks, slots, hidden, HEAD_COLS)) * 0.05,
            jnp.bfloat16,
        ),
        sh,
    )
    o_shares = jax.device_put(
        jnp.asarray(
            rng.standard_normal((num_ranks, slots, HEAD_COLS, hidden)) * 0.05,
            jnp.bfloat16,
        ),
        sh,
    )

    (gate_rep,) = jax.shard_map(
        functools.partial(gather_gate_shares, axis="k3_sp"),
        mesh=mesh,
        in_specs=(P("k3_sp"),),
        out_specs=(P("k3_sp"),),
    )(gate_shares)
    (o_rep,) = jax.shard_map(
        functools.partial(gather_output_projection_shares, axis="k3_sp"),
        mesh=mesh,
        in_specs=(P("k3_sp"),),
        out_specs=(P("k3_sp"),),
    )(o_shares)

    (gate_pair,) = jax.shard_map(
        functools.partial(gather_gate_shares_chip_pair, axis="k3_sp"),
        mesh=mesh,
        in_specs=(P("k3_sp"),),
        out_specs=(P("k3_sp"),),
    )(gate_shares)
    (o_pair,) = jax.shard_map(
        functools.partial(gather_output_projection_shares_chip_pair, axis="k3_sp"),
        mesh=mesh,
        in_specs=(P("k3_sp"),),
        out_specs=(P("k3_sp"),),
    )(o_shares)
    assert gate_pair.shape == (num_ranks, slots, hidden, num_ranks * HEAD_COLS // 2)
    assert o_pair.shape == (num_ranks, slots, num_ranks * HEAD_COLS, hidden // 2)

    seg_desc = jax.device_put(jnp.asarray([0, length, 0], jnp.int32), rep)

    def _simple_flash(q, k, v, ln, desc, *, count, tile, arena):
        del desc, count, tile, arena
        h, t, _ = q.shape
        valid = jnp.arange(t) < ln
        scale = 1.0 / np.sqrt(192.0)
        outs = []
        for head in range(h):
            scores = (
                jnp.dot(
                    q[head, :, :192],
                    k[head, :, :192].T,
                    preferred_element_type=jnp.float32,
                )
                * scale
            )
            causal = jnp.tril(jnp.ones((t, t), dtype=bool)) & valid[None, :]
            scores = jnp.where(causal, scores, -1e30)
            probs = jax.nn.softmax(scores, axis=-1).astype(jnp.bfloat16)
            outs.append(
                jnp.dot(probs, v[head], preferred_element_type=jnp.float32).astype(
                    jnp.bfloat16
                )
            )
        return jnp.stack(outs, axis=0)

    def _run_attn(lh, px, qa, qb, ka, kb, qn, kn, mg, mo, sd):
        out, lat = mla_local_normed_a2a_attention(
            lh[0],
            px[0],
            query_down_projection=qa[0],
            query_up_projection=qb[0],
            key_value_down_projection=ka[0],
            key_value_up_projection=kb[0],
            query_normalization=qn[0],
            key_value_normalization=kn[0],
            m_gate_family=mg[0],
            m_o_family=mo[0],
            slot=slot,
            length=length,
            segment_desc=sd,
            num_ranks=num_ranks,
            sp_axis="k3_sp",
            segments=1,
            flash_tile=64,
            token_ranks=token_ranks,
            flash_fn=_simple_flash,
            use_pallas_a2a=False,
            use_pallas_owner=False,
            interpret=True,
        )
        return out[None], lat[None]

    out_rep, lat_rep = jax.shard_map(
        _run_attn,
        mesh=mesh,
        in_specs=(P("k3_sp"),) * 10 + (P(),),
        out_specs=(P("k3_sp"), P("k3_sp")),
        check_vma=False,
    )(
        local_hidden,
        local_prefix,
        m_qa,
        m_qb,
        m_ka,
        m_kb,
        m_qnorm,
        m_knorm,
        gate_rep,
        o_rep,
        seg_desc,
    )

    out_pair, lat_pair = jax.shard_map(
        _run_attn,
        mesh=mesh,
        in_specs=(P("k3_sp"),) * 10 + (P(),),
        out_specs=(P("k3_sp"), P("k3_sp")),
        check_vma=False,
    )(
        local_hidden,
        local_prefix,
        m_qa,
        m_qb,
        m_ka,
        m_kb,
        m_qnorm,
        m_knorm,
        gate_pair,
        o_pair,
        seg_desc,
    )

    np.testing.assert_array_equal(
        np.asarray(out_pair, np.float32), np.asarray(out_rep, np.float32)
    )
    np.testing.assert_array_equal(
        np.asarray(lat_pair, np.float32), np.asarray(lat_rep, np.float32)
    )

    swapped_np = np.asarray(o_pair, np.float32)[[i ^ 1 for i in range(num_ranks)]]
    o_swapped = jax.device_put(jnp.asarray(swapped_np, jnp.bfloat16), sh)
    out_bad, _ = jax.shard_map(
        _run_attn,
        mesh=mesh,
        in_specs=(P("k3_sp"),) * 10 + (P(),),
        out_specs=(P("k3_sp"), P("k3_sp")),
        check_vma=False,
    )(
        local_hidden,
        local_prefix,
        m_qa,
        m_qb,
        m_ka,
        m_kb,
        m_qnorm,
        m_knorm,
        gate_pair,
        o_swapped,
        seg_desc,
    )
    max_bad = np.max(
        np.abs(np.asarray(out_bad, np.float32) - np.asarray(out_rep, np.float32))
    )
    assert max_bad > 0.05


def test_should_report_fuse_eligibility_for_all_tp32_windows_and_reject_bad():
    windows = (512, 1024, 2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384)
    for tokens in windows:
        stripe = pm.window_stripe(tokens // RANKS)
        assert pm.can_fuse_mla_flash_a2a(tokens, RANKS, stripe, tile=256), (
            tokens,
            stripe,
        )
    assert not pm.can_fuse_mla_flash_a2a(512, 32, 4, tile=256)
    assert not pm.can_fuse_mla_flash_a2a(512, 32, 0, tile=256)
    assert not pm.can_fuse_mla_flash_a2a(640, 32, 16, tile=256)
    assert not pm.can_fuse_mla_flash_a2a(4096, 64, 64, tile=256)


def test_should_reject_invalid_shapes_in_mla_flash_prefill_packed_a2a():
    q = jnp.zeros((3, 512, 256), jnp.bfloat16)
    k = jnp.zeros((3, 512, 256), jnp.bfloat16)
    v = jnp.zeros((3, 512, 128), jnp.bfloat16)
    tables = jnp.zeros((3, 3, 1), jnp.int32)
    length = jnp.asarray([512], jnp.int32)
    segs = jnp.asarray([0, 512, 0], jnp.int32)

    with pytest.raises(ValueError, match="query"):
        pm.mla_flash_prefill_packed_a2a(
            tables,
            jnp.zeros((512, 256), jnp.bfloat16),
            k,
            v,
            length,
            segs,
            count=1,
            num_ranks=1,
            stripe=64,
        )
    with pytest.raises(ValueError, match="3 heads"):
        pm.mla_flash_prefill_packed_a2a(
            tables,
            jnp.zeros((2, 512, 256), jnp.bfloat16),
            k,
            v,
            length,
            segs,
            count=1,
            num_ranks=1,
            stripe=64,
        )
    with pytest.raises(ValueError, match="keys/values"):
        pm.mla_flash_prefill_packed_a2a(
            tables,
            q,
            k,
            jnp.zeros((3, 512, 64), jnp.bfloat16),
            length,
            segs,
            count=1,
            num_ranks=1,
            stripe=64,
        )
    with pytest.raises(ValueError, match="tables"):
        pm.mla_flash_prefill_packed_a2a(
            jnp.zeros((3, 3, 2), jnp.int32),
            q,
            k,
            v,
            length,
            segs,
            count=1,
            num_ranks=1,
            stripe=64,
        )
    with pytest.raises(ValueError, match="segments"):
        pm.mla_flash_prefill_packed_a2a(
            tables,
            q,
            k,
            v,
            length,
            jnp.zeros((4,), jnp.int32),
            count=1,
            num_ranks=1,
            stripe=64,
        )
    with pytest.raises(ValueError, match="incompatible with num_ranks"):
        pm.mla_flash_prefill_packed_a2a(
            tables, q, k, v, length, segs, count=1, num_ranks=1, stripe=4
        )
    with pytest.raises(ValueError, match="arena"):
        pm.mla_flash_prefill_packed_a2a(
            tables,
            q,
            k,
            v,
            length,
            jnp.zeros((5,), jnp.int32),
            count=1,
            num_ranks=1,
            stripe=64,
            arena=(
                jnp.zeros((1, 3, 64, 256), jnp.bfloat16),
                jnp.zeros((1, 3, 128, 64), jnp.bfloat16),
            ),
        )


@pytest.mark.parametrize(("prefixed", "stripe"), [(False, 512), (True, 64)])
def test_should_match_mla_flash_prefill_packed_bitwise_on_single_rank(prefixed, stripe):
    tokens = 512
    rng = np.random.default_rng(77 if prefixed else 42)
    q = jnp.asarray(rng.standard_normal((3, tokens, 256)) * 0.25, jnp.bfloat16)
    k = jnp.asarray(rng.standard_normal((3, tokens, 256)) * 0.25, jnp.bfloat16)
    v = jnp.asarray(rng.standard_normal((3, tokens, 128)) * 0.25, jnp.bfloat16)
    length = jnp.asarray([480], jnp.int32)
    tables = jnp.zeros((3, 3, 1), jnp.int32)
    if prefixed:
        cap = 512
        pk = jnp.asarray(rng.standard_normal((1, 3, cap, 256)) * 0.2, jnp.float8_e4m3fn)
        pv = jnp.asarray(rng.standard_normal((1, 3, 128, cap)) * 0.2, jnp.float8_e4m3fn)
        arena = (pk, pv)
        segs = jnp.asarray([0, 256, 224, 224, 0, 0, 0, 256, 128, 192], jnp.int32)
    else:
        arena = None
        segs = jnp.asarray([0, 256, 224, 224, 0, 0], jnp.int32)

    ref_heads = pm.mla_flash_prefill_packed(
        q, k, v, length, segs, count=2, tile=128, arena=arena, interpret=True
    )
    ref = ref_heads.transpose(1, 0, 2).reshape(1, tokens, HEAD_COLS)
    fused = pm.mla_flash_prefill_packed_a2a(
        tables,
        q,
        k,
        v,
        length,
        segs,
        count=2,
        num_ranks=1,
        stripe=stripe,
        tile=128,
        arena=arena,
        interpret=True,
    )
    np.testing.assert_array_equal(
        np.asarray(fused.astype(jnp.float32)), np.asarray(ref.astype(jnp.float32))
    )
    np.testing.assert_array_equal(
        np.asarray(fused[:, 480:, :].astype(jnp.float32)), 0.0
    )
    bad_segs = segs.at[2].set(128)
    bad_ref = (
        pm.mla_flash_prefill_packed(
            q, k, v, length, bad_segs, count=2, tile=128, arena=arena, interpret=True
        )
        .transpose(1, 0, 2)
        .reshape(1, tokens, HEAD_COLS)
    )
    diff_bad = float(
        jnp.max(jnp.abs(fused.astype(jnp.float32) - bad_ref.astype(jnp.float32)))
    )
    assert diff_bad > 0.05


def _synthetic_ring_tables(num_ranks: int) -> np.ndarray:
    tables = np.zeros((3, 3, num_ranks), dtype=np.int32)
    base = np.arange(num_ranks, dtype=np.int32)
    rings = [
        base,
        np.roll(base, 1) if num_ranks > 1 else base,
        base[::-1] if num_ranks > 1 else base,
    ]
    for f, order in enumerate(rings):
        tables[f, 1, :] = order
        for pos, rk in enumerate(order):
            tables[f, 0, int(rk)] = pos
    tables[:, 2, :] = base[None, :]
    return tables


@pytest.mark.parametrize("pool_dtype", [jnp.bfloat16, jnp.float8_e4m3fn])
@pytest.mark.parametrize("page_count", [0, 5])
def test_should_match_unfused_qkv_up_and_prefix_gather_single_rank(
    pool_dtype, page_count
):
    rng = np.random.default_rng(701 + page_count)
    tokens, stripe, q_rank, tile = 512, 256, 128, 256
    pool_pages, slots = 10, 7
    page_shape = (16, 2, 128) if pool_dtype == jnp.bfloat16 else (8, 4, 128)
    both_raw = jnp.asarray(
        rng.standard_normal((tokens, q_rank + 640)) * 0.2, jnp.bfloat16
    )
    qu = jnp.asarray(rng.standard_normal((q_rank, 768)) * 0.1, jnp.bfloat16)
    kvu = jnp.asarray(rng.standard_normal((512, 768)) * 0.1, jnp.bfloat16)

    pool_np = np.full((pool_pages, *page_shape), 3.0, dtype=np.float32)
    chosen = [6, 1, 8, 3, 4][:page_count]
    for p_idx in chosen:
        pool_np[p_idx] = rng.standard_normal(page_shape) * 0.5
    pool = jnp.asarray(pool_np, pool_dtype)

    ids_np = np.zeros((slots,), dtype=np.int32)
    ids_np[0] = page_count
    if page_count > 0:
        ids_np[1 : 1 + page_count] = chosen
    ids = jnp.asarray(ids_np)
    counts = jnp.asarray([page_count], jnp.int32)
    order = jnp.zeros((1 + (slots - 1),), jnp.int32).at[0].set(page_count)
    tables = jnp.asarray(_synthetic_ring_tables(1))
    length = jnp.asarray([430], jnp.int32)

    q_ref, k_ref, v_ref, lat_ref = mla_qkv_up_unstriped_pallas(
        both_raw,
        qu,
        kvu,
        num_ranks=1,
        stripe=stripe,
        q_rank=q_rank,
        tile=tile,
        query_scale=0.125,
        length=length,
        interpret=True,
    )
    pages_ref, _ = pm.gathered_prefix_pages(
        pool,
        ids,
        jnp.zeros((2,), jnp.int32),
        tables,
        "k3_sp",
        count=1,
        capacity=(slots - 1) * page_shape[0] * page_shape[1],
        counts=counts,
        order=order,
        interpret=True,
    )
    q_f, k_f, v_f, lat_f, pages_f = mla_qkv_up_unstriped_and_prefix_gather_pallas(
        both_raw,
        qu,
        kvu,
        pool,
        tables,
        counts,
        ids,
        order,
        num_ranks=1,
        stripe=stripe,
        q_rank=q_rank,
        tile=tile,
        query_scale=0.125,
        length=length,
        interpret=True,
    )

    for fused, ref in ((q_f, q_ref), (k_f, k_ref), (v_f, v_ref), (lat_f, lat_ref)):
        np.testing.assert_array_equal(
            np.asarray(fused, np.float32), np.asarray(ref, np.float32)
        )
    for q_idx in range(page_count):
        expected = np.asarray(pool[chosen[q_idx]], np.float32)
        np.testing.assert_array_equal(
            np.asarray(pages_f[1 + q_idx], np.float32), expected
        )
        np.testing.assert_array_equal(
            np.asarray(pages_ref[1 + q_idx], np.float32), expected
        )
    if page_count > 1:
        bad_ids = ids.at[1].set(chosen[-1]).at[page_count].set(chosen[0])
        _, _, _, _, bad_pages = mla_qkv_up_unstriped_and_prefix_gather_pallas(
            both_raw,
            qu,
            kvu,
            pool,
            tables,
            counts,
            bad_ids,
            order,
            num_ranks=1,
            stripe=stripe,
            q_rank=q_rank,
            tile=tile,
            query_scale=0.125,
            length=length,
            interpret=True,
        )
        assert (
            np.max(
                np.abs(
                    np.asarray(bad_pages[1 : 1 + page_count], np.float32)
                    - np.asarray(pages_f[1 : 1 + page_count], np.float32)
                )
            )
            > 0.1
        )


@pytest.mark.parametrize("counts_per_rank", [(0, 0, 0, 0), (0, 3, 1, 5)])
def test_should_match_unfused_prefix_gather_over_four_rank_ring_with_asymmetric_counts(
    counts_per_rank,
):
    num_ranks = 4
    devs = jax.devices()
    if len(devs) < num_ranks:
        pytest.skip(
            f"needs {num_ranks} host devices "
            f"(XLA_FLAGS=--xla_force_host_platform_device_count={num_ranks})"
        )
    mesh = Mesh(np.array(devs[:num_ranks]), ("k3_sp",))
    sh_r = jax.sharding.NamedSharding(mesh, P("k3_sp"))
    sh_rep = jax.sharding.NamedSharding(mesh, P())

    rng = np.random.default_rng(733 + sum(counts_per_rank))
    tokens, stripe, q_rank, tile = 512, 64, 128, 256
    pool_pages, slots = 10, 7
    page_shape = (16, 2, 128)
    page_tokens = page_shape[0] * page_shape[1]
    segments = 2
    capacity = num_ranks * (slots - 1) * page_tokens

    both_raw = jax.device_put(
        jnp.asarray(
            rng.standard_normal((num_ranks, tokens, q_rank + 640)) * 0.2, jnp.bfloat16
        ),
        sh_r,
    )
    qu = jax.device_put(
        jnp.asarray(rng.standard_normal((num_ranks, q_rank, 768)) * 0.1, jnp.bfloat16),
        sh_r,
    )
    kvu = jax.device_put(
        jnp.asarray(rng.standard_normal((num_ranks, 512, 768)) * 0.1, jnp.bfloat16),
        sh_r,
    )
    pool_np = rng.standard_normal((num_ranks, pool_pages, *page_shape)) * 0.4
    pool = jax.device_put(jnp.asarray(pool_np, jnp.bfloat16), sh_r)

    ids_np = np.zeros((num_ranks, slots), dtype=np.int32)
    local_np = np.zeros((num_ranks, 2 * segments), dtype=np.int32)
    for r, cnt in enumerate(counts_per_rank):
        ids_np[r, 0] = cnt
        if cnt > 0:
            ids_np[r, 1 : 1 + cnt] = rng.permutation(pool_pages)[:cnt]
            s0 = cnt // 2
            s1 = cnt - s0
            local_np[r] = [0, s0 * page_tokens, s0 * page_tokens, s1 * page_tokens]
    ids = jax.device_put(jnp.asarray(ids_np), sh_r)
    local = jax.device_put(jnp.asarray(local_np), sh_r)
    tables = jax.device_put(jnp.asarray(_synthetic_ring_tables(num_ranks)), sh_rep)
    length = jax.device_put(jnp.asarray([433], jnp.int32), sh_rep)

    def _run_fused(b, qu_r, kvu_r, p, ids_r, loc_r, t, lng):
        cnts, ord_ = pm.resolve_prefix_gather_meta(
            p[0], ids_r[0], loc_r[0], "k3_sp", count=segments, capacity=capacity
        )
        q, k, v, lat, pgs = mla_qkv_up_unstriped_and_prefix_gather_pallas(
            b[0],
            qu_r[0],
            kvu_r[0],
            p[0],
            t,
            cnts,
            ids_r[0],
            ord_,
            num_ranks=num_ranks,
            stripe=stripe,
            q_rank=q_rank,
            tile=tile,
            query_scale=0.125,
            length=lng,
            interpret=pltpu.InterpretParams(),
        )
        return q[None], k[None], v[None], lat[None], pgs[None], ord_[None]

    def _run_unfused(b, qu_r, kvu_r, p, ids_r, loc_r, t, lng):
        q, k, v, lat = mla_qkv_up_unstriped_pallas(
            b[0],
            qu_r[0],
            kvu_r[0],
            num_ranks=num_ranks,
            stripe=stripe,
            q_rank=q_rank,
            tile=tile,
            query_scale=0.125,
            length=lng,
            interpret=True,
        )
        pgs, ord_ = pm.gathered_prefix_pages(
            p[0],
            ids_r[0],
            loc_r[0],
            t,
            "k3_sp",
            count=segments,
            capacity=capacity,
            interpret=pltpu.InterpretParams(),
        )
        return q[None], k[None], v[None], lat[None], pgs[None], ord_[None]

    in_sp = (P("k3_sp"),) * 6 + (P(), P())
    out_sp = (P("k3_sp"),) * 6
    fused_fn = jax.jit(
        jax.shard_map(
            _run_fused, mesh=mesh, in_specs=in_sp, out_specs=out_sp, check_vma=False
        )
    )
    unfused_fn = jax.jit(
        jax.shard_map(
            _run_unfused, mesh=mesh, in_specs=in_sp, out_specs=out_sp, check_vma=False
        )
    )
    qf, kf, vf, latf, pgf, ordf = fused_fn(
        both_raw, qu, kvu, pool, ids, local, tables, length
    )
    qr, kr, vr, latr, pgr, ordr = unfused_fn(
        both_raw, qu, kvu, pool, ids, local, tables, length
    )

    for fused, ref in ((qf, qr), (kf, kr), (vf, vr), (latf, latr), (ordf, ordr)):
        np.testing.assert_array_equal(
            np.asarray(fused, np.float32), np.asarray(ref, np.float32)
        )

    pgf_np = np.asarray(pgf, np.float32)
    pgr_np = np.asarray(pgr, np.float32)
    pool_bf16 = _bf16(pool_np)
    for recv_rank in range(num_ranks):
        for owner, cnt in enumerate(counts_per_rank):
            for q_idx in range(cnt):
                slot_idx = owner * slots + 1 + q_idx
                expected = pool_bf16[owner, ids_np[owner, 1 + q_idx]]
                np.testing.assert_array_equal(pgf_np[recv_rank, slot_idx], expected)
                np.testing.assert_array_equal(pgr_np[recv_rank, slot_idx], expected)

    if sum(counts_per_rank) > 0:
        swapped_pool = jax.device_put(
            jnp.asarray(pool_np[[3, 2, 1, 0]], jnp.bfloat16), sh_r
        )
        _, _, _, _, bad_pg, _ = fused_fn(
            both_raw, qu, kvu, swapped_pool, ids, local, tables, length
        )
        bad_np = np.asarray(bad_pg, np.float32)
        valid_slots = [
            owner * slots + 1 + q_idx
            for owner, cnt in enumerate(counts_per_rank)
            for q_idx in range(cnt)
        ]
        assert np.max(np.abs(bad_np[:, valid_slots] - pgf_np[:, valid_slots])) > 0.1


def test_should_reject_incompatible_prefix_gather_shapes_and_conflicting_inputs():
    both_raw = jnp.zeros((512, 128 + 640), jnp.bfloat16)
    qu = jnp.zeros((128, 768), jnp.bfloat16)
    kvu = jnp.zeros((512, 768), jnp.bfloat16)
    pool = jnp.zeros((4, 16, 2, 128), jnp.bfloat16)
    ids = jnp.zeros((5,), jnp.int32)
    order = jnp.zeros((5,), jnp.int32)

    with pytest.raises(ValueError, match="incompatible shapes"):
        mla_qkv_up_unstriped_and_prefix_gather_pallas(
            both_raw,
            qu,
            kvu,
            pool,
            jnp.zeros((3, 3, 3), jnp.int32),
            jnp.zeros((3,), jnp.int32),
            ids,
            order,
            num_ranks=3,
            stripe=64,
            q_rank=128,
            tile=256,
            query_scale=0.125,
            length=None,
            interpret=True,
        )

    with pytest.raises(ValueError, match="1 or even ranks"):
        pm.gathered_prefix_pages(
            pool,
            ids,
            jnp.zeros((2,), jnp.int32),
            jnp.zeros((3, 3, 3), jnp.int32),
            "k3_sp",
            count=1,
            capacity=128,
            counts=jnp.zeros((3,), jnp.int32),
            order=order,
            interpret=True,
        )

    hidden = jnp.zeros((64, 128), jnp.bfloat16)
    with pytest.raises(ValueError, match="at most one of prefix_gather_fn"):
        mla_local_normed_a2a_attention(
            hidden,
            hidden,
            query_down_projection=jnp.zeros((128, 128), jnp.bfloat16),
            query_up_projection=qu,
            key_value_down_projection=jnp.zeros((128, 576), jnp.bfloat16),
            key_value_up_projection=kvu,
            query_normalization=jnp.ones((128,), jnp.bfloat16),
            key_value_normalization=jnp.ones((512,), jnp.bfloat16),
            m_gate_family=jnp.zeros((1, 128, 384), jnp.bfloat16),
            m_o_family=jnp.zeros((1, 384, 128), jnp.bfloat16),
            slot=jnp.asarray([0], jnp.int32),
            length=jnp.asarray([64], jnp.int32),
            segment_desc=jnp.zeros((5,), jnp.int32),
            num_ranks=1,
            sp_axis="k3_sp",
            segments=1,
            flash_tile=64,
            token_ranks=(0,),
            tables=jnp.zeros((3, 3, 1), jnp.int32),
            prefix_gather_fn=lambda: (pool, order),
            prefix_gather_inputs=(
                pool,
                ids,
                jnp.zeros((2,), jnp.int32),
                128,
                jnp.zeros((1,), jnp.int32),
                order,
            ),
            interpret=True,
        )

    with pytest.raises(ValueError, match="requires ring tables"):
        mla_local_normed_a2a_attention(
            hidden,
            hidden,
            query_down_projection=jnp.zeros((128, 128), jnp.bfloat16),
            query_up_projection=qu,
            key_value_down_projection=jnp.zeros((128, 576), jnp.bfloat16),
            key_value_up_projection=kvu,
            query_normalization=jnp.ones((128,), jnp.bfloat16),
            key_value_normalization=jnp.ones((512,), jnp.bfloat16),
            m_gate_family=jnp.zeros((1, 128, 384), jnp.bfloat16),
            m_o_family=jnp.zeros((1, 384, 128), jnp.bfloat16),
            slot=jnp.asarray([0], jnp.int32),
            length=jnp.asarray([64], jnp.int32),
            segment_desc=jnp.zeros((5,), jnp.int32),
            num_ranks=1,
            sp_axis="k3_sp",
            segments=1,
            flash_tile=64,
            token_ranks=(0,),
            tables=None,
            prefix_gather_inputs=(
                pool,
                ids,
                jnp.zeros((2,), jnp.int32),
                128,
                jnp.zeros((1,), jnp.int32),
                order,
            ),
            interpret=True,
        )


def test_should_validate_eligibility_and_reject_bad_mla_ag_gate_chip_pair():
    tables_32 = jnp.zeros((3, 3, RANKS), jnp.int32)
    mg_32 = jax.ShapeDtypeStruct((24, 7168, 6144), jnp.bfloat16)
    mo_32 = jax.ShapeDtypeStruct((24, 12288, 3584), jnp.bfloat16)
    base_kw = dict(
        tables=tables_32,
        sp_axis="k3_sp",
        num_ranks=RANKS,
        rows=512,
        width=7168,
        packed_width=2176,
        m_gate_family=mg_32,
        m_o_family=mo_32,
        m_gate_scales=None,
        m_o_scales=None,
        use_pallas_owner=True,
        interpret=False,
    )
    for w in k3_prefill.WINDOWS:
        assert can_fuse_mla_ag_and_gate_chip_pair(**{**base_kw, "rows": w // RANKS})

    for override in (
        {"tables": None},
        {"tables": jnp.zeros((3, 3, 16), jnp.int32)},
        {"sp_axis": "tp"},
        {"use_pallas_owner": False},
        {"interpret": True},
        {"num_ranks": 6},
        {"rows": 12},
        {"width": 128},
        {"packed_width": 256},
        {"packed_width": 576},
        {"m_gate_scales": jnp.ones((24, 1, 6144), jnp.float32)},
        {"m_gate_family": jax.ShapeDtypeStruct((24, 7168, 6144), jnp.float8_e4m3fn)},
        {"m_gate_family": jax.ShapeDtypeStruct((24, 7168, 12288), jnp.bfloat16)},
    ):
        assert not can_fuse_mla_ag_and_gate_chip_pair(**{**base_kw, **override})

    with pytest.raises(
        ValueError, match="rank_half_gather_and_mla_gate_chip_pair_pallas"
    ):
        rank_half_gather_and_mla_gate_chip_pair_pallas(
            jnp.zeros((16, 256), jnp.bfloat16),
            jnp.zeros((16, 256), jnp.bfloat16),
            jnp.zeros((2, 256, 768), jnp.bfloat16),
            jnp.zeros((3, 3, 4), jnp.int32),
            layer=0,
            num_ranks=4,
        )

    with pytest.raises(ValueError, match="owner_mla_output_gated_chip_pair_pallas"):
        owner_mla_output_gated_chip_pair_pallas(
            jnp.zeros((4, 12, 384), jnp.bfloat16),
            jnp.zeros((2, 12, 1536), jnp.bfloat16),
            jnp.zeros((2, 1536, 128), jnp.bfloat16),
            jnp.zeros((12, 256), jnp.bfloat16),
            layer=0,
            length=48,
            own_even=0,
            own_odd=1,
            stripe=12,
            num_ranks=4,
        )


def test_should_match_oracle_bitwise_when_gathering_and_gating_in_pallas_chip_pair():
    num_ranks = 4
    devs = jax.devices()
    if len(devs) < num_ranks:
        pytest.skip(
            f"needs {num_ranks} host devices "
            f"(XLA_FLAGS=--xla_force_host_platform_device_count={num_ranks})"
        )
    mesh = Mesh(np.asarray(devs[:num_ranks]), ("k3_sp",))
    sh_sp = jax.sharding.NamedSharding(mesh, P("k3_sp"))
    sh_rep = jax.sharding.NamedSharding(mesh, P())
    rng = np.random.default_rng(4821)

    rows, width, packed_width, slots = 16, 256, 384, 3
    stripe = 8
    tokens = num_ranks * rows
    length = tokens - 5
    slot = 2
    token_ranks = (2, 3, 1, 0)

    tables_np = _synthetic_ring_tables(num_ranks)
    tables_np[:, 2, :] = np.asarray(token_ranks, dtype=np.int32)[None, :]
    tables_dev = jax.device_put(jnp.asarray(tables_np), sh_rep)

    packed_np = (rng.standard_normal((num_ranks, rows, packed_width)) * 0.3).astype(
        np.float32
    )
    normed_np = (rng.standard_normal((num_ranks, rows, width)) * 0.25).astype(
        np.float32
    )
    prefix_np = (rng.standard_normal((num_ranks, rows, width)) + 1.5).astype(np.float32)
    ow_np = (rng.standard_normal((num_ranks, num_ranks, rows, HEAD_COLS)) * 0.2).astype(
        np.float32
    )
    for r, own in enumerate(token_ranks):
        for loc in range(rows):
            win_tok = (
                (loc // stripe) * (num_ranks * stripe) + own * stripe + (loc % stripe)
            )
            if win_tok >= length:
                ow_np[r, :, loc, :] = 999.0

    packed_dev = jax.device_put(jnp.asarray(packed_np, jnp.bfloat16), sh_sp)
    normed_dev = jax.device_put(jnp.asarray(normed_np, jnp.bfloat16), sh_sp)
    prefix_dev = jax.device_put(jnp.asarray(prefix_np, jnp.bfloat16), sh_sp)
    ow_dev = jax.device_put(jnp.asarray(ow_np, jnp.bfloat16), sh_sp)

    gate_shares = jax.device_put(
        jnp.asarray(
            rng.standard_normal((num_ranks, slots, width, HEAD_COLS)) * 0.06,
            jnp.bfloat16,
        ),
        sh_sp,
    )
    o_shares = jax.device_put(
        jnp.asarray(
            rng.standard_normal((num_ranks, slots, HEAD_COLS, width)) * 0.06,
            jnp.bfloat16,
        ),
        sh_sp,
    )
    (gate_pair,) = jax.shard_map(
        functools.partial(gather_gate_shares_chip_pair, axis="k3_sp"),
        mesh=mesh,
        in_specs=(P("k3_sp"),),
        out_specs=(P("k3_sp"),),
    )(gate_shares)
    (o_pair,) = jax.shard_map(
        functools.partial(gather_output_projection_shares_chip_pair, axis="k3_sp"),
        mesh=mesh,
        in_specs=(P("k3_sp"),),
        out_specs=(P("k3_sp"),),
    )(o_shares)

    def _run_fused(pk, nm, ow, mg, mo, px, tbl):
        rank = jax.lax.axis_index("k3_sp")
        own_ev = tbl[0, 2, rank & ~1]
        own_od = tbl[0, 2, rank | 1]
        raw_ag, gate_both = rank_half_gather_and_mla_gate_chip_pair_pallas(
            pk[0],
            nm[0],
            mg[0],
            tbl,
            layer=slot,
            num_ranks=num_ranks,
            interpret=pltpu.InterpretParams(),
        )
        out = owner_mla_output_gated_chip_pair_pallas(
            ow[0],
            gate_both,
            mo[0],
            px[0],
            layer=slot,
            length=length,
            own_even=own_ev,
            own_odd=own_od,
            stripe=stripe,
            num_ranks=num_ranks,
            interpret=pltpu.InterpretParams(),
        )
        return raw_ag[None], gate_both[None], out[None]

    def _run_ref(nm, ow, mg, mo, px, tbl):
        rank = jax.lax.axis_index("k3_sp")
        out = owner_mla_gate_and_output_chip_pair_ref(
            nm[0],
            ow[0],
            mg[0, slot],
            mo[0, slot],
            px[0],
            stripe=stripe,
            length=length,
            own_even=tbl[0, 2, rank & ~1],
            own_odd=tbl[0, 2, rank | 1],
            sp_axis="k3_sp",
        )
        return out[None]

    raw_ag_got, gate_both_got, out_got = jax.jit(
        jax.shard_map(
            _run_fused,
            mesh=mesh,
            in_specs=(P("k3_sp"),) * 6 + (P(),),
            out_specs=(P("k3_sp"), P("k3_sp"), P("k3_sp")),
            check_vma=False,
        )
    )(packed_dev, normed_dev, ow_dev, gate_pair, o_pair, prefix_dev, tables_dev)

    out_ref = jax.jit(
        jax.shard_map(
            _run_ref,
            mesh=mesh,
            in_specs=(P("k3_sp"),) * 5 + (P(),),
            out_specs=P("k3_sp"),
            check_vma=False,
        )
    )(normed_dev, ow_dev, gate_pair, o_pair, prefix_dev, tables_dev)

    owner_inv = np.argsort(np.asarray(token_ranks))
    packed_bf16 = np.asarray(jnp.asarray(packed_np, jnp.bfloat16), np.float32)
    exp_ag = packed_bf16[owner_inv].reshape(tokens, packed_width)
    for r in range(num_ranks):
        np.testing.assert_array_equal(np.asarray(raw_ag_got[r], np.float32), exp_ag)
    out_got_f32 = np.asarray(out_got, np.float32)
    out_ref_f32 = np.asarray(out_ref, np.float32)
    rel_l2 = float(
        np.linalg.norm(out_got_f32 - out_ref_f32) / np.linalg.norm(out_ref_f32)
    )
    assert rel_l2 < 1e-4
    got_bits = np.asarray(jax.device_get(out_got).view(jnp.uint16), np.int32)
    ref_bits = np.asarray(jax.device_get(out_ref).view(jnp.uint16), np.int32)
    assert int(np.max(np.abs(got_bits - ref_bits))) <= 1
    prefix_bf16 = np.asarray(jnp.asarray(prefix_np, jnp.bfloat16), np.float32)
    for r, own in enumerate(token_ranks):
        for loc in range(rows):
            win_tok = (
                (loc // stripe) * (num_ranks * stripe) + own * stripe + (loc % stripe)
            )
            if win_tok >= length:
                np.testing.assert_array_equal(out_got_f32[r, loc], prefix_bf16[r, loc])

    def _run_corrupted_gate(ow, gb, mo, px, tbl):
        rank = jax.lax.axis_index("k3_sp")
        out = owner_mla_output_gated_chip_pair_pallas(
            ow[0],
            1.0 - gb[0],
            mo[0],
            px[0],
            layer=slot,
            length=length,
            own_even=tbl[0, 2, rank & ~1],
            own_odd=tbl[0, 2, rank | 1],
            stripe=stripe,
            num_ranks=num_ranks,
            out_collective_id=67,
            interpret=pltpu.InterpretParams(),
        )
        return out[None]

    out_bad = jax.jit(
        jax.shard_map(
            _run_corrupted_gate,
            mesh=mesh,
            in_specs=(P("k3_sp"),) * 4 + (P(),),
            out_specs=P("k3_sp"),
            check_vma=False,
        )
    )(ow_dev, gate_both_got, o_pair, prefix_dev, tables_dev)
    assert (
        float(
            np.max(
                np.abs(
                    np.asarray(out_bad, np.float32) - np.asarray(out_ref, np.float32)
                )
            )
        )
        > 0.1
    )


def test_should_match_unfused_mla_a2a_when_pipeline_fuses_ag_and_chip_pair_gate():
    num_ranks = 4
    devs = jax.devices()
    if len(devs) < num_ranks:
        pytest.skip(
            f"needs {num_ranks} host devices "
            f"(XLA_FLAGS=--xla_force_host_platform_device_count={num_ranks})"
        )
    mesh = Mesh(np.asarray(devs[:num_ranks]), ("k3_sp",))
    sh_sp = jax.sharding.NamedSharding(mesh, P("k3_sp"))
    sh_rep = jax.sharding.NamedSharding(mesh, P())
    rng = np.random.default_rng(4822)

    rows, hidden, q_rank, slots = 64, 256, 128, 2
    tokens = num_ranks * rows
    length = tokens - 7
    slot = 1
    token_ranks = (1, 3, 0, 2)
    tables_np = _synthetic_ring_tables(num_ranks)
    tables_np[:, 2, :] = np.asarray(token_ranks, dtype=np.int32)[None, :]
    tables_dev = jax.device_put(jnp.asarray(tables_np), sh_rep)
    seg_desc = jax.device_put(jnp.asarray([0, length, 0], jnp.int32), sh_rep)

    lh_dev = jax.device_put(
        jnp.asarray(rng.standard_normal((num_ranks, rows, hidden)) * 0.2, jnp.bfloat16),
        sh_sp,
    )
    px_dev = jax.device_put(
        jnp.asarray(rng.standard_normal((num_ranks, rows, hidden)) + 1.0, jnp.bfloat16),
        sh_sp,
    )
    qa_dev = jax.device_put(
        jnp.broadcast_to(
            jnp.asarray(rng.standard_normal((hidden, q_rank)) * 0.08, jnp.bfloat16)[
                None
            ],
            (num_ranks, hidden, q_rank),
        ),
        sh_sp,
    )
    ka_dev = jax.device_put(
        jnp.broadcast_to(
            jnp.asarray(rng.standard_normal((hidden, 576)) * 0.08, jnp.bfloat16)[None],
            (num_ranks, hidden, 576),
        ),
        sh_sp,
    )
    qn_dev = jax.device_put(
        jnp.broadcast_to(
            jnp.asarray(rng.uniform(0.8, 1.2, (1, q_rank)), jnp.bfloat16)[None],
            (num_ranks, 1, q_rank),
        ),
        sh_sp,
    )
    kn_dev = jax.device_put(
        jnp.broadcast_to(
            jnp.asarray(rng.uniform(0.8, 1.2, (1, 512)), jnp.bfloat16)[None],
            (num_ranks, 1, 512),
        ),
        sh_sp,
    )
    qb_dev = jax.device_put(
        jnp.asarray(rng.standard_normal((num_ranks, q_rank, 768)) * 0.06, jnp.bfloat16),
        sh_sp,
    )
    kb_dev = jax.device_put(
        jnp.asarray(rng.standard_normal((num_ranks, 512, 768)) * 0.06, jnp.bfloat16),
        sh_sp,
    )
    gate_shares = jax.device_put(
        jnp.asarray(
            rng.standard_normal((num_ranks, slots, hidden, HEAD_COLS)) * 0.05,
            jnp.bfloat16,
        ),
        sh_sp,
    )
    o_shares = jax.device_put(
        jnp.asarray(
            rng.standard_normal((num_ranks, slots, HEAD_COLS, hidden)) * 0.05,
            jnp.bfloat16,
        ),
        sh_sp,
    )
    (gate_pair,) = jax.shard_map(
        functools.partial(gather_gate_shares_chip_pair, axis="k3_sp"),
        mesh=mesh,
        in_specs=(P("k3_sp"),),
        out_specs=(P("k3_sp"),),
    )(gate_shares)
    (o_pair,) = jax.shard_map(
        functools.partial(gather_output_projection_shares_chip_pair, axis="k3_sp"),
        mesh=mesh,
        in_specs=(P("k3_sp"),),
        out_specs=(P("k3_sp"),),
    )(o_shares)

    def _simple_flash(q, k, v, ln, desc, *, count, tile, arena):
        del desc, count, tile, arena
        h, t, _ = q.shape
        valid = jnp.arange(t) < ln
        scale = 1.0 / np.sqrt(192.0)
        outs = []
        for head in range(h):
            scores = (
                jnp.dot(
                    q[head, :, :192],
                    k[head, :, :192].T,
                    preferred_element_type=jnp.float32,
                )
                * scale
            )
            causal = jnp.tril(jnp.ones((t, t), dtype=bool)) & valid[None, :]
            scores = jnp.where(causal, scores, -1e30)
            probs = jax.nn.softmax(scores, axis=-1).astype(jnp.bfloat16)
            outs.append(
                jnp.dot(probs, v[head], preferred_element_type=jnp.float32).astype(
                    jnp.bfloat16
                )
            )
        return jnp.stack(outs, axis=0)

    def _run_composed(use_fused_ag_gate: bool):
        interp = pltpu.InterpretParams() if use_fused_ag_gate else True

        def _local(lh, px, qa, qb, ka, kb, qn, kn, mg, mo, sd, tbl):
            out, lat = mla_local_normed_a2a_attention(
                lh[0],
                px[0],
                query_down_projection=qa[0],
                query_up_projection=qb[0],
                key_value_down_projection=ka[0],
                key_value_up_projection=kb[0],
                query_normalization=qn[0],
                key_value_normalization=kn[0],
                m_gate_family=mg[0],
                m_o_family=mo[0],
                slot=slot,
                length=length,
                segment_desc=sd,
                num_ranks=num_ranks,
                sp_axis="k3_sp",
                segments=1,
                flash_tile=64,
                token_ranks=token_ranks,
                tables=tbl if use_fused_ag_gate else None,
                flash_fn=_simple_flash,
                use_pallas_a2a=False,
                use_pallas_owner=use_fused_ag_gate,
                interpret=interp,
            )
            return out[None], lat[None]

        return jax.jit(
            jax.shard_map(
                _local,
                mesh=mesh,
                in_specs=(P("k3_sp"),) * 10 + (P(), P()),
                out_specs=(P("k3_sp"), P("k3_sp")),
                check_vma=False,
            )
        )(
            lh_dev,
            px_dev,
            qa_dev,
            qb_dev,
            ka_dev,
            kb_dev,
            qn_dev,
            kn_dev,
            gate_pair,
            o_pair,
            seg_desc,
            tables_dev,
        )

    out_fused, lat_fused = _run_composed(use_fused_ag_gate=True)
    out_ref, lat_ref = _run_composed(use_fused_ag_gate=False)
    np.testing.assert_array_equal(
        np.asarray(out_fused, np.float32), np.asarray(out_ref, np.float32)
    )
    np.testing.assert_array_equal(
        np.asarray(lat_fused, np.float32), np.asarray(lat_ref, np.float32)
    )
