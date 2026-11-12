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
"""VLLM_TPU_GDN_STEP_KERNEL_METADATA: the packed per-step kernel metadata
hands the GDN kernel the same integers, in every slot it reads, as the
per-layer derivation.

CPU only: the kernel launch is replaced by a probe that records its
scalar-prefetch operands; the TPU info the wrapper sizes tiles with is faked.
"""

import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.gdn.v3 import memory_ref, metadata, wrapper
from vllm_torchtpu.layers.core import gdn_attention

N_KQ, N_V, D = 2, 6, 128
CONV = 2 * N_KQ * D + N_V * D
NUM_SPEC = 3
MAX_SEQS = 24
KBS, PAYLOAD, LANES, POOL_BLOCKS = 512, (1, 4), 256, 256


def _case(kind):
    if kind == "pf1":
        lens, ctx, ndec = [8192], [75776], 0
    elif kind == "mix4":
        lens, ctx, ndec = [4096, 2048, 1536, 512], [60000, 20000, 0, 3000], 0
    elif kind == "mixdec":
        lens, ctx, ndec = [4] * 20 + [8112], [30000 + i for i in range(20)] + [75776], 20
    elif kind == "ragged":
        lens, ctx, ndec = [3, 1, 4, 300, 129, 7, 1000], [9, 0, 5, 0, 77, 1, 0], 3
    else:
        raise ValueError(kind)
    n, tokens = len(lens), sum(lens)
    tokens = -(-tokens // 16) * 16
    qsl = np.full(MAX_SEQS + 1, sum(lens), np.int32)
    qsl[: n + 1] = np.concatenate([[0], np.cumsum(lens)])
    seq = np.zeros(MAX_SEQS, np.int32)
    seq[:n] = np.asarray(ctx) + np.asarray(lens)
    st = np.zeros(MAX_SEQS, np.int32)
    st[:n] = 4 + 5 * np.arange(n)
    ck = np.zeros((MAX_SEQS, NUM_SPEC + 1), np.int32)
    ck[:n] = st[:n, None] + np.arange(NUM_SPEC + 1)[None]
    return tokens, dict(
        query_start_loc=jnp.asarray(qsl),
        state_indices=jnp.asarray(st),
        distribution=jnp.asarray([ndec, n, n], jnp.int32),
        seq_lens=jnp.asarray(seq),
        read_offsets=jnp.asarray((np.arange(MAX_SEQS) % 2) * (st > 0), jnp.int32),
        ckpt_indices=jnp.asarray(ck),
    )


@pytest.fixture
def fake_tpu(monkeypatch):
    info = types.SimpleNamespace(
        num_lanes=128, num_sublanes=8, vmem_capacity_bytes=64 * 1024**2
    )
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda *a, **k: info)
    for name in (
        "TPU_ENABLE_GDN_DYNAMIC_TILING",
        "TPU_GDN_CHUNK_PAIR",
        "TPU_GDN_RAW_QKV",
        "TPU_GDN_RAW_BA",
        "TPU_GDN_KERNEL_ZERO_OUT",
        "VLLM_TPU_GDN_V3_SINGLE_LAUNCH",
    ):
        monkeypatch.setattr(envs, name, True)
    calls = []

    class _PL:
        def __getattr__(self, name):
            return getattr(pl, name)

        def pallas_call(self, kernel, *, out_shape, input_output_aliases, **kw):
            def probe(*args):
                calls.append(args[0])
                flat = jax.tree.leaves(args)
                outs = []
                for i, o in enumerate(out_shape):
                    if o is None:
                        outs.append(None)
                        continue
                    src = [k for k, v in input_output_aliases.items() if v == i]
                    if src:
                        outs.append(flat[src[0]])
                    else:
                        outs.append(jnp.zeros(o.shape, o.dtype))
                return tuple(outs)

            return probe

    monkeypatch.setattr(wrapper, "pl", _PL())
    # Eager wrapper, so the probe records concrete operands.
    monkeypatch.setattr(
        gdn_attention.gdn_v3_wrapper,
        "fused_conv1d_gdn",
        wrapper.fused_conv1d_gdn.__wrapped__,
    )
    return calls


def _run(tokens, md, **kw):
    rng = np.random.default_rng(0)
    pool = jnp.zeros((POOL_BLOCKS, KBS) + PAYLOAD + (LANES,), jnp.float8_e4m3fn)
    return gdn_attention.run_jax_gdn_attention_pooled_local(
        jnp.asarray(rng.normal(size=(tokens, CONV)), jnp.bfloat16),
        jnp.asarray(rng.normal(size=(tokens, N_V)), jnp.bfloat16),
        jnp.asarray(rng.normal(size=(tokens, N_V)), jnp.bfloat16),
        pool,
        jnp.ones((4, 1, CONV), jnp.float32),
        None,
        jnp.zeros((N_V,), jnp.float32),
        jnp.zeros((N_V,), jnp.float32),
        md["query_start_loc"],
        md["state_indices"],
        md["distribution"],
        md["seq_lens"],
        md["read_offsets"],
        md["ckpt_indices"],
        None,
        n_kq=N_KQ,
        n_v=N_V,
        d_k=D,
        d_v=D,
        kernel_size=4,
        pool_block_tokens=KBS,
        num_spec_tokens=NUM_SPEC,
        conv_weight_kmajor=True,
        **kw,
    )


def _assert_kernel_view_equal(got, want):
    """Bit-equal in every slot the kernel reads: per-tile arrays up to the
    tile-sized length (past `num_tiles` + lookahead), the rest whole."""
    assert jax.tree.structure(got) == jax.tree.structure(want)
    num_tiles = int(want.num_tiles)
    for (path, g), w in zip(
        jax.tree_util.tree_leaves_with_path(got), jax.tree.leaves(want), strict=True
    ):
        g, w = np.asarray(g), np.asarray(w)
        assert g.dtype == w.dtype, (path, g.dtype, w.dtype)
        if g.shape != w.shape:
            assert "p_id" in jax.tree_util.keystr(path), path
            assert g.ndim == w.ndim == 1 and g.size >= num_tiles + 2, path
            w = w[: g.size]
        np.testing.assert_array_equal(g, w, err_msg=jax.tree_util.keystr(path))


@pytest.mark.parametrize("kind", ["pf1", "mix4", "mixdec", "ragged"])
def test_packed_step_metadata_matches_per_layer(kind, fake_tpu):
    tokens, md = _case(kind)
    _run(tokens, md)
    per_layer = list(fake_tpu)
    assert len(per_layer) == 1  # single launch: (BATCHED, PER_SEQ) metas
    fake_tpu.clear()
    exact = _run(tokens, md, metadata_only=True)
    assert exact.dtype == jnp.int32 and exact.ndim == 1
    assert not fake_tpu  # no kernel launch
    # Tile-sized: about 1 KB, not five token-long arrays.
    assert exact.size < 5 * tokens
    # The served op pads to a static bound; the layer reads the prefix.
    bound = wrapper.packed_kernel_metadata_size(8192, MAX_SEQS, NUM_SPEC)
    packed = _run(tokens, md, metadata_only=True, packed_size=bound)
    assert packed.shape == (bound,)
    np.testing.assert_array_equal(np.asarray(packed[: exact.size]), np.asarray(exact))
    assert not np.asarray(packed[exact.size :]).any()
    _run(tokens, md, packed_metadata=packed)
    (stepped,) = fake_tpu
    for got, want in zip(stepped, per_layer[0], strict=True):
        _assert_kernel_view_equal(got, want)
    if kind != "mixdec":
        assert int(per_layer[0][1].num_tiles) > 0


def test_metadata_only_call_never_donates_the_pool(monkeypatch):
    """The step op does not own the pool: outside any outer jit, the
    donating kernel wrapper must not consume it (seen on TPU as 'Buffer has
    been deleted or donated' on the next layer's op)."""
    info = types.SimpleNamespace(
        num_lanes=128, num_sublanes=8, vmem_capacity_bytes=64 * 1024**2
    )
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda *a, **k: info)
    for name in ("TPU_GDN_RAW_QKV", "TPU_GDN_RAW_BA", "VLLM_TPU_GDN_V3_SINGLE_LAUNCH"):
        monkeypatch.setattr(envs, name, True)
    seen = []
    real = wrapper.fused_conv1d_gdn

    def spy(*args, state_source=None, **kw):
        seen.append(state_source)
        return real(*args, state_source=state_source, **kw)

    monkeypatch.setattr(gdn_attention.gdn_v3_wrapper, "fused_conv1d_gdn", spy)
    tokens, md = _case("mix4")
    packed = _run(tokens, md, metadata_only=True)
    assert packed.ndim == 1
    (source,) = seen
    assert source.shape[0] == 1  # a copy, not the caller's pool


def test_tile_slots_cover_every_tile():
    cfg = types.SimpleNamespace(batch_size=8192, chunk_size=256)
    for lens in ([8192], [1] * 23 + [8169], [255, 257] * 12):
        tiles = sum(-(-q // 256) for q in lens)
        assert metadata.per_seq_tile_slots(cfg, MAX_SEQS, True) >= tiles + 2
    assert metadata.per_seq_tile_slots(cfg, MAX_SEQS, False) == 8192


def test_pack_round_trip_is_exact():
    rng = np.random.default_rng(3)
    meta = memory_ref.MetadataRef(
        num_tiles=jnp.int32(7),
        p_id_to_s_idx=memory_ref.SmemWrapper(jnp.asarray(rng.integers(0, 9, 40), jnp.int32), (1, 1)),
        p_id_to_r_base=memory_ref.SmemWrapper(jnp.asarray(rng.integers(0, 9999, 40), jnp.int32), (1, 1)),
        p_id_to_r_size=memory_ref.SmemWrapper(jnp.asarray(rng.integers(0, 256, 40), jnp.int32), (1, 1)),
        p_id_is_first_tile=memory_ref.SmemWrapper(jnp.asarray(rng.integers(0, 2, 40), bool), (1, 1)),
        p_id_is_last_tile=memory_ref.SmemWrapper(jnp.asarray(rng.integers(0, 2, 40), bool), (1, 1)),
        s_idx_has_initial_state=jnp.asarray(rng.integers(0, 2, 24), bool),
        s_idx_to_state_indices=jnp.asarray(rng.integers(0, 2**30, 24), jnp.int32),
        s_idx_to_read_offset=jnp.asarray(rng.integers(0, 4, 24), jnp.int32),
        s_idx_to_ckpt_indices=jnp.asarray(rng.integers(-5, 2**30, (24, 4)), jnp.int32),
    )
    metas = (meta, meta)
    got = wrapper.unpack_kernel_metadata(
        wrapper.pack_kernel_metadata(metas), jax.eval_shape(lambda: metas)
    )
    for g, w in zip(jax.tree.leaves(got), jax.tree.leaves(metas), strict=True):
        assert g.dtype == w.dtype
        np.testing.assert_array_equal(np.asarray(g), np.asarray(w))


def test_flag_requires_both_segments(fake_tpu):
    tokens, md = _case("mix4")
    with pytest.raises(ValueError, match="pooled calls running both"):
        md2 = dict(md)
        gdn_attention.gdn_v3_wrapper.fused_conv1d_gdn(
            jnp.zeros((tokens, CONV), jnp.bfloat16),
            jnp.zeros((tokens, N_V), jnp.bfloat16),
            jnp.zeros((tokens, N_V), jnp.bfloat16),
            jnp.zeros((MAX_SEQS + 1, 3, CONV), jnp.float32),
            jnp.zeros((MAX_SEQS + 1, N_V, D, D), jnp.float32),
            jnp.ones((CONV, 1, 4), jnp.bfloat16),
            None,
            jnp.zeros((N_V,), jnp.float32),
            jnp.zeros((N_V,), jnp.float32),
            md2["query_start_loc"],
            md2["state_indices"],
            md2["distribution"],
            md2["seq_lens"],
            n_kq=N_KQ,
            n_v=N_V,
            d_k=D,
            d_v=D,
            kernel_size=4,
            metadata_only=True,
        )
