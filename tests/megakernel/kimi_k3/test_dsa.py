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
"""Kimi-K3.1 DSA on the K3 decode megakernel (CPU): the configuration, the
index-key encoding and commit, the synthetic indexer weights, the JAX
references of the indexer and the sparse MLA attention, the kernel's index key,
the decode step's key commit, the tools' flags and ``check dsa-keys``, and the
decode stack's static VMEM plan against Kimi-K3's."""

import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from tools.megakernel import __main__ as tool_main
from tools.megakernel import check as tool_check
from tools.megakernel import dsa_reference as ref
from tools.megakernel import k3 as tool_k3
from tools.megakernel import preshard as tool_preshard
from vllm_torchtpu.megakernel.kimi_k3 import dsa
from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_cache

pytestmark = pytest.mark.cpu_test

BF16 = np.dtype(ml_dtypes.bfloat16)

FULL = [*range(4, 93, 4), 93]
PINNED = dict(
    index_topk=2048,
    index_num_heads=32,
    index_head_dim=128,
    index_rope_dim=0,
    index_k_norm_type="layernorm",
    indexer_use_fp8=False,
    index_topk_freq=1,
    index_skip_topk_offset=1,
    dsa_layers=FULL,
)


def text_config(**dsa_overrides):
    return {
        "rms_norm_eps": 1e-5,
        "linear_attn_config": {"full_attn_layers": FULL},
        "dsa_config": dict(PINNED, **dsa_overrides),
    }


def test_pinned_config_parses():
    assert dsa.DsaConfig.from_text_config(text_config()) == dsa.DsaConfig(2048, 1e-5)
    # The released checkpoint's dsa_config leaves indexer_use_fp8 out (default false).
    config = text_config()
    del config["dsa_config"]["indexer_use_fp8"]
    assert dsa.DsaConfig.from_text_config(config) == dsa.DsaConfig(2048, 1e-5)


def test_no_dsa_config_is_k3():
    assert dsa.DsaConfig.from_text_config({"rms_norm_eps": 1e-5}) is None


@pytest.mark.parametrize(
    "override",
    [
        {"index_num_heads": 64},
        {"index_head_dim": 64},
        {"index_rope_dim": 64},
        {"index_k_norm_type": "rmsnorm"},
        {"indexer_use_fp8": True},
        {"index_topk_freq": 2},
        {"index_skip_topk_offset": 2},
        {"index_topk_pattern": "F" * 92 + "S"},
        {"index_topk": 1024},
        {"dsa_layers": FULL[:-1]},
    ],
)
def test_unsupported_config_raises(override):
    with pytest.raises(NotImplementedError):
        dsa.DsaConfig.from_text_config(text_config(**override))


def test_a_none_topk_pattern_is_every_layer_selecting():
    config = text_config(index_topk_pattern=None)
    assert dsa.DsaConfig.from_text_config(config) == dsa.DsaConfig(2048, 1e-5)
    assert dsa._SUPPORTED["index_topk_pattern"] is None


def test_object_config_and_model_directory(tmp_path):
    obj = SimpleNamespace(**text_config())
    assert dsa.DsaConfig.from_text_config(obj) == dsa.DsaConfig(2048, 1e-5)
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"text_config": text_config()}))
    assert dsa.load_dsa_config(tmp_path) == dsa.DsaConfig(2048, 1e-5)
    assert dsa.load_dsa_config(path) == dsa.DsaConfig(2048, 1e-5)


def test_engine_config_reads_dsa():
    from vllm_torchtpu.megakernel.config import _dsa_config

    k31 = SimpleNamespace(architectures=["KimiK31ForConditionalGeneration"])
    k3 = SimpleNamespace(architectures=["KimiK3ForConditionalGeneration"])
    with_dsa = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=k31, hf_text_config=SimpleNamespace(**text_config())
        )
    )
    bare = SimpleNamespace(rms_norm_eps=1e-5)
    without = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=k3, hf_text_config=bare)
    )
    assert _dsa_config(with_dsa) == dsa.DsaConfig(2048, 1e-5)
    assert _dsa_config(without) is None
    for hf_config, text in ((k3, SimpleNamespace(**text_config())), (k31, bare)):
        mismatch = SimpleNamespace(
            model_config=SimpleNamespace(hf_config=hf_config, hf_text_config=text)
        )
        with pytest.raises(ValueError, match="Kimi-K3.1 needs both"):
            _dsa_config(mismatch)


@pytest.mark.parametrize(
    "change,message",
    [
        (
            dict(attention_dp=2, num_speculative_tokens=3, max_num_seqs=64),
            "at most 64 rows at attention DP2",
        ),
        (
            dict(attention_dp=2, num_speculative_tokens=3, max_num_seqs=24),
            "at most 64 rows at attention DP2",
        ),
        (
            dict(attention_dp=2, num_speculative_tokens=7, max_num_seqs=16),
            "at most 64 rows at attention DP2",
        ),
        (dict(attention_dp=2, max_num_seqs=128), "at most 64 rows at attention DP2"),
    ],
)
def test_program_check_rejects_unimplemented_dsa(change, message):
    from vllm_torchtpu.megakernel.config import MegakernelEngineConfig
    from vllm_torchtpu.megakernel.kimi_k3 import program

    fields = dict(
        role="decode",
        tensor_parallel_size=32,
        decode_context_parallel_size=32,
        block_size=256,
        max_num_seqs=8,
        max_model_len=65536,
        max_num_batched_tokens=512,
        dsa=dsa.DsaConfig(),
        synthetic_indexer=("random", 0),
    )
    fields.update(change)
    if change.get("attention_dp") == 2:
        fields["decode_context_parallel_size"] = 16
    with pytest.raises(NotImplementedError, match=message):
        program.check(MegakernelEngineConfig(**fields))


@pytest.mark.parametrize("seqs, drafts", [(16, 0), (64, 0), (16, 3), (32, 3), (8, 7)])
def test_program_check_takes_dsa_decode_at_attention_dp2(seqs, drafts):
    """Plain decode and DSpark verify at attention DP2: to 64 rows, 128 with
    3 drafts."""
    from vllm_torchtpu.megakernel.config import MegakernelEngineConfig
    from vllm_torchtpu.megakernel.kimi_k3 import program

    program.check(
        MegakernelEngineConfig(
            role="decode",
            tensor_parallel_size=32,
            decode_context_parallel_size=16,
            attention_dp=2,
            block_size=256,
            max_num_seqs=seqs,
            max_model_len=65536,
            max_num_batched_tokens=512,
            dsa=dsa.DsaConfig(),
            synthetic_indexer=("random", 0),
            num_speculative_tokens=drafts,
        )
    )


def gpu_scale(amax, floor):
    """The GPU rule: 2^ceil(log2(max(amax, floor) / 448))."""
    return 2.0 ** np.ceil(np.log2(np.maximum(amax, floor) / 448.0))


@pytest.mark.parametrize("amax", [56.0, 57.0, 448.0, 1.0, 0.0, 3.7e-3])
def test_key_scale_matches_gpu_rule(amax):
    key = np.zeros((128,), np.float32)
    key[5] = amax
    _, exponent = dsa_cache.quantize_index_keys(jnp.asarray(key))
    scale = 2.0 ** (int(exponent) - 127)
    assert scale == gpu_scale(amax, 1e-4)


def test_key_encoding_pins_bytes_and_exponents():
    key = np.zeros((3, 128), np.float32)
    key[0, :6] = [448.0, -2.0, 1.1, 1.0625, -0.3, 2.0**-9]  # amax 448: scale 1
    key[1, :3] = [57.0, -1.0, 0.1]  # amax 57: scale 1/4
    q, e = (np.asarray(a) for a in dsa_cache.quantize_index_keys(jnp.asarray(key)))
    # 2^0, 2^-2 and the floor's 2^ceil(log2(1e-4 / 448)) = 2^-22 as UE8M0 bytes.
    np.testing.assert_array_equal(e, [0x7F, 0x7D, 0x69])
    # E4M3: 448 -> 0x7E, -2 -> 0xC0, 1.1 rounds up to 1.125 (0x39), the tie
    # 1.0625 to even 1.0 (0x38), -0.3 to -0.3125 (0xAA), 2^-9 is the least
    # subnormal (0x01); at scale 1/4, 228 rounds down to 224 (0x76), -4 is
    # 0xC8, 0.4 rounds to 0.40625 (0x2D).
    np.testing.assert_array_equal(q[0, :6], [0x7E, 0xC0, 0x39, 0x38, 0xAA, 0x01])
    np.testing.assert_array_equal(q[1, :3], [0x76, 0xC8, 0x2D])
    assert not q[0, 6:].any() and not q[1, 3:].any() and not q[2].any()


def test_query_scales_are_per_head():
    queries = np.zeros((2, 3, 128), np.float32)
    queries[0, 0, 3] = 448.0
    queries[0, 1, :2] = [57.0, -1.0]
    queries[0, 2, 7] = 1e-12  # below the floor: 2^ceil(log2(1e-10 / 448)) = 2^-42
    queries[1] = 2 * queries[0]
    q, scale = (
        np.asarray(a) for a in dsa_cache.quantize_index_queries(jnp.asarray(queries))
    )
    np.testing.assert_array_equal(scale, [[1.0, 0.25, 2.0**-42], [2.0, 0.5, 2.0**-42]])
    np.testing.assert_array_equal(q[:, 0, 3], [448.0, 448.0])
    np.testing.assert_array_equal(q[0, 1, :2], [224.0, -4.0])  # 228 rounds down
    np.testing.assert_array_equal(q[:, 2, 7], [4.5, 9.0])  # 4.398 and 8.796
    # Each head is scaled alone: growing one leaves the others' codes.
    grown = queries.copy()
    grown[0, 1] *= 64
    q2, scale2 = (
        np.asarray(a) for a in dsa_cache.quantize_index_queries(jnp.asarray(grown))
    )
    assert scale2[0, 1] == 16.0
    keep = np.ones((2, 3), bool)
    keep[0, 1] = False
    np.testing.assert_array_equal(q2[keep], q[keep])
    np.testing.assert_array_equal(scale2[keep], scale[keep])


def test_key_round_trip_within_fp8_precision():
    keys = jax.random.normal(jax.random.PRNGKey(0), (64, 128), jnp.float32) * 3
    q, e = dsa_cache.quantize_index_keys(keys.astype(jnp.bfloat16))
    back = dsa_cache.dequantize_index_keys(q, e)
    ref = np.asarray(keys.astype(jnp.bfloat16).astype(jnp.float32))
    amax = np.abs(ref).max(-1, keepdims=True)
    # E4M3: 3 mantissa bits -> relative error <= 2^-4 of the value,
    # plus subnormals near zero.
    assert np.all(
        np.abs(np.asarray(back) - ref) <= np.abs(ref) * 2**-4 + amax * 2**-9 * 2
    )


def test_query_scale_uses_its_own_floor():
    q = jnp.zeros((2, 128), jnp.float32)
    _, scale = dsa_cache.quantize_index_queries(q)
    assert np.all(np.asarray(scale) == gpu_scale(0.0, 1e-10))


def test_block_shape():
    assert dsa_cache.index_block_shape(256) == (288, 128)
    assert dsa_cache.index_block_shape(512) == (544, 128)
    with pytest.raises(ValueError):
        dsa_cache.index_block_shape(100)


@pytest.mark.parametrize("tokens", [256, 512])  # BF16 and FP8 pages
@pytest.mark.parametrize("global_pages", [True, False])
@pytest.mark.parametrize("member", [0, 1])
def test_commit_places_bytes_and_exponents(tokens, global_pages, member):
    group = 2
    cache = jnp.zeros((4, 2, *dsa_cache.index_block_shape(tokens)), jnp.uint8)
    keys = jax.random.normal(jax.random.PRNGKey(1), (3, 2, 128), jnp.float32).astype(
        jnp.bfloat16
    )
    q, e = dsa_cache.quantize_index_keys(keys)
    # Rows on pages member, member + 1 and member + 2 of a group of 2: the
    # first and last are this member's, the middle one its peer's. The last
    # row's slot is in the block's last scale row (515 for 512 tokens).
    pages = member + np.arange(3)
    slots = np.asarray([5, 7, tokens - 2])
    positions = jnp.asarray(pages * tokens + slots, jnp.int32)
    # Row r's block is r + 1, at its page's table entry (global tables) or
    # its group of pages' (local tables); the other entries are null.
    tables = np.zeros((3, 4), np.int32)
    tables[np.arange(3), pages if global_pages else pages // group] = [1, 2, 3]
    out = dsa_cache.commit_index_rows(
        cache,
        q,
        e,
        jnp.asarray(tables),
        positions,
        member=member,
        group=group,
        global_pages=global_pages,
    )
    q, e = np.asarray(q), np.asarray(e)
    expected = np.zeros(cache.shape, np.uint8)
    for r in (0, 2):
        for m in range(2):
            expected[r + 1, m, slots[r]] = q[r, m]
            expected[r + 1, m, tokens + slots[r] // 128, slots[r] % 128] = e[r, m]
    np.testing.assert_array_equal(np.asarray(out), expected)


def k3_families(seed=0, layers=3, hidden=256):
    rng = np.random.default_rng(seed)
    return {
        "m_ka": rng.standard_normal((layers, hidden, 640)).astype(BF16)
        * BF16.type(0.01),
        "m_knorm": rng.standard_normal((layers, 1, 512)).astype(BF16),
        "m_qa": np.zeros((layers, hidden, 1536), BF16),
        "m_qb": np.zeros((layers, 64, 768), BF16),
    }


def test_presharded_indexer_lands_where_the_synthetic_one_does():
    layers, hidden, seed, head = 3, 256, 4, 7
    stored = dict(
        k3_families(),
        index_wk=np.stack(
            [dsa.synthetic_key_projection(seed, m, hidden, BF16) for m in range(layers)]
        ),
        index_wq=np.stack(
            [
                dsa.synthetic_query_heads(seed, m, 64, BF16, heads=(head,))
                for m in range(layers)
            ]
        ),
        index_wv=np.stack(
            [
                dsa.synthetic_weight_rows(seed, m, hidden, BF16, heads=(head,))
                for m in range(layers)
            ]
        ),
    )
    stored["index_knorm_weight"], stored["index_knorm_bias"] = dsa.synthetic_key_norm(
        seed, layers, BF16
    )
    out = dsa.with_presharded_indexer(stored)
    expected = dsa.with_synthetic_indexer(k3_families(), "random", seed, head=head)
    assert set(out) == set(expected)
    for name, array in expected.items():
        assert out[name].dtype == array.dtype
        np.testing.assert_array_equal(out[name], array)


def test_stage_presharded_indexer_is_the_stage_synthetic_layout():
    seed, ordinals, q_lora, hidden = 3, range(6, 12), 64, 96
    weight, bias = dsa.synthetic_key_norm(seed, len(dsa.MLA), BF16)
    keys = np.stack(
        [dsa.synthetic_key_projection(seed, m, hidden, BF16) for m in ordinals]
    )
    heads = [
        dict(
            index_wk=keys,
            index_knorm_weight=weight[list(ordinals)],
            index_knorm_bias=bias[list(ordinals)],
            index_wq=np.stack(
                [
                    dsa.synthetic_query_heads(seed, m, q_lora, BF16, heads=(head,))
                    for m in ordinals
                ]
            ),
            index_wv=np.stack(
                [
                    dsa.synthetic_weight_rows(seed, m, hidden, BF16, heads=(head,))
                    for m in ordinals
                ]
            ),
        )
        for head in range(dsa.INDEX_HEADS)
    ]
    out = dsa.stage_presharded_indexer(heads)
    expected = dsa.stage_synthetic_indexer(
        "random", seed, ordinals, q_lora, hidden, BF16
    )
    assert set(out) == set(expected) == set(dsa.STAGE_DSA_FAMILIES)
    for name, array in expected.items():
        assert out[name].dtype == array.dtype
        np.testing.assert_array_equal(out[name], array)
    with pytest.raises(ValueError, match="32 heads"):
        dsa.stage_presharded_indexer(heads[:31])


@pytest.mark.parametrize("mode", dsa.SYNTHETIC_INDEXERS)
def test_synthetic_indexer_widens_and_keeps_k3_columns(mode):
    families = k3_families()
    out = dsa.with_synthetic_indexer(families, mode, seed=3, head=2)
    assert out["m_ka"].shape == (3, 256, 768) and out["m_ka"].dtype == BF16
    assert out["m_qb"].shape == (3, 64, 896) and out["m_qb"].dtype == BF16
    assert out["m_knorm"].shape == (3, 1, 768 + 256) and out["m_knorm"].dtype == BF16
    np.testing.assert_array_equal(out["m_ka"][..., :640], families["m_ka"])
    np.testing.assert_array_equal(out["m_qb"][..., :768], families["m_qb"])
    np.testing.assert_array_equal(out["m_knorm"][..., :512], families["m_knorm"])
    assert out["m_qa"] is families["m_qa"]
    # Another rank: the same replicated key projection and LayerNorm, its own
    # index-query head and weights_proj row.
    other = dsa.with_synthetic_indexer(k3_families(), mode, seed=3, head=5)
    np.testing.assert_array_equal(out["m_ka"], other["m_ka"])
    np.testing.assert_array_equal(
        out["m_knorm"][..., :768], other["m_knorm"][..., :768]
    )
    assert not np.array_equal(out["m_qb"][..., 768:], other["m_qb"][..., 768:])
    assert not np.array_equal(out["m_knorm"][..., 768:], other["m_knorm"][..., 768:])


def test_rope_probe_copies_the_rope_columns():
    families = k3_families()
    out = dsa.with_synthetic_indexer(families, "rope-probe", seed=0)
    rope = families["m_ka"][..., 512:576]
    np.testing.assert_array_equal(out["m_ka"][..., 640:704], rope)
    np.testing.assert_array_equal(out["m_ka"][..., 704:768], -rope)
    assert (
        np.all(out["m_knorm"][..., 512:640] == 1)
        and not out["m_knorm"][..., 640:768].any()
    )
    assert np.all(out["m_knorm"][..., 768:].any(axis=-1))  # the weights_proj row


def test_random_indexer_keys_are_float32_math():
    # sqrt(300) is irrational: a float64 quotient rounds to BF16 differently
    # for some of these draws.
    out = dsa.with_synthetic_indexer(k3_families(hidden=300), "random", seed=5)
    for layer in range(3):
        draw = np.random.default_rng((5, layer)).standard_normal((300, 128), np.float32)
        expected = np.divide(draw, np.sqrt(300), dtype=np.float32).astype(BF16)
        np.testing.assert_array_equal(
            out["m_ka"][layer, :, 640:].view(np.uint16), expected.view(np.uint16)
        )


def test_random_indexer_differs_by_layer_and_seed():
    a = dsa.with_synthetic_indexer(k3_families(), "random", seed=0)["m_ka"][..., 640:]
    b = dsa.with_synthetic_indexer(k3_families(), "random", seed=1)["m_ka"][..., 640:]
    assert not np.array_equal(a[0], a[1]) and not np.array_equal(a, b)
    assert 0.5 < float(np.std(a.astype(np.float32))) * np.sqrt(256) < 2.0


def test_synthetic_index_heads_are_each_ranks_own():
    out = dsa.with_synthetic_indexer(k3_families(), "random", seed=4, head=7)
    for layer in range(3):
        heads = dsa.synthetic_query_heads(4, layer, 64, BF16)
        rows = dsa.synthetic_weight_rows(4, layer, 256, BF16)
        assert heads.shape == (64, 32 * 128) and rows.shape == (32, 256)
        np.testing.assert_array_equal(
            out["m_qb"][layer, :, 768:], heads[:, 7 * 128 : 8 * 128]
        )
        np.testing.assert_array_equal(out["m_knorm"][layer, 0, 768:], rows[7])
        np.testing.assert_array_equal(
            dsa.synthetic_query_heads(4, layer, 64, BF16, heads=(7,)),
            heads[:, 7 * 128 : 8 * 128],
        )
        assert 0.5 < float(np.std(heads.astype(np.float32))) * np.sqrt(64) < 2.0
        assert 0.5 < float(np.std(rows.astype(np.float32))) * np.sqrt(256) < 2.0
    weight, bias = dsa.synthetic_key_norm(4, 3, BF16)
    np.testing.assert_array_equal(out["m_knorm"][..., 512:640], weight)
    np.testing.assert_array_equal(out["m_knorm"][..., 640:768], bias)
    np.testing.assert_array_equal(
        out["m_ka"][1, :, 640:], dsa.synthetic_key_projection(4, 1, 256, BF16)
    )


def test_mla_norms_pack_into_one_family_and_split_back():
    qnorm = np.arange(3 * 1536, dtype=np.float32).reshape(3, 1, 1536).astype(BF16)
    families = dsa.with_synthetic_indexer(
        {**k3_families(), "m_qnorm": qnorm}, "random", seed=0
    )
    packed = dsa.pack_mla_norms(families)
    assert "m_qnorm" not in packed and "m_knorm" not in packed
    assert packed["m_norms"].shape == (3, 1, 1536 + 768 + 256)
    assert packed["m_norms"].dtype == BF16 and packed["m_ka"] is families["m_ka"]
    split = dsa.unpack_mla_norms(packed)
    assert "m_norms" not in split
    for name in ("m_qnorm", "m_knorm"):
        np.testing.assert_array_equal(split[name], families[name])
    k3 = k3_families()
    assert dsa.unpack_mla_norms(k3) is k3  # Kimi-K3's operands are its families


def test_k31_decode_rank_packs_the_mla_norms(monkeypatch):
    from vllm_torchtpu.megakernel.kimi_k3.kernels import rank_loader

    dense = {**k3_families(), "m_qnorm": np.ones((3, 1, 1536), BF16)}

    class RankDir:
        def __init__(self, *args, **kwargs):
            pass

        def read_dense_all(self):
            return dict(dense)

    monkeypatch.setattr(rank_loader, "RankDir", RankDir)
    monkeypatch.setattr(rank_loader, "_expert_jobs", lambda rank_dir: [])
    monkeypatch.setattr(rank_loader, "_dense_bytes", lambda rank_dir: 1)

    def names(**options):
        return {name for _, name, _ in rank_loader.iter_decode_rank("x", 0, **options)}

    assert names() == set(dense)
    k31 = names(synthetic_indexer=("random", 0))
    assert k31 == (set(dense) - {"m_qnorm", "m_knorm"}) | {"m_norms"}
    arrays = {
        name: array
        for _, name, array in rank_loader.iter_decode_rank(
            "x", 3, synthetic_indexer=("random", 0)
        )
    }
    heads = dsa.synthetic_query_heads(0, 1, 64, BF16)
    rows = dsa.synthetic_weight_rows(0, 1, 256, BF16)
    np.testing.assert_array_equal(arrays["m_norms"][1, 0, 1536 + 768 :], rows[3])
    np.testing.assert_array_equal(
        arrays["m_qb"][1, :, 768:], heads[:, 3 * 128 : 4 * 128]
    )


def test_k31_decode_rank_holds_each_dp_shards_index_head(monkeypatch):
    from vllm_torchtpu.megakernel.kimi_k3.kernels import rank_loader
    from vllm_torchtpu.megakernel.kimi_k3.kernels.decode_megakernel import (
        ATTENTION_DP_WEIGHTS,
    )

    def dense_of(rank):
        rng = np.random.default_rng(rank)
        families = {**k3_families(), "m_qnorm": np.ones((3, 1, 1536), BF16)}
        for name in ATTENTION_DP_WEIGHTS:
            families.setdefault(name, rng.standard_normal((3, 4, 8)).astype(BF16))
        return families

    class RankDir:
        def __init__(self, presharded, rank, layers=93):
            self.dense = dense_of(rank)
            self.families = dict.fromkeys(self.dense)

        def read_dense_all(self):
            return dict(self.dense)

        def read_dense(self, name):
            return self.dense[name]

    monkeypatch.setattr(rank_loader, "RankDir", RankDir)
    monkeypatch.setattr(rank_loader, "_expert_jobs", lambda rank_dir: [])
    monkeypatch.setattr(rank_loader, "_dense_bytes", lambda rank_dir: 1)
    arrays = {
        name: array
        for _, name, array in rank_loader.iter_decode_rank(
            "x", 3, attention_dp=2, synthetic_indexer=("random", 0)
        )
    }
    heads = dsa.synthetic_query_heads(0, 1, 64, BF16)
    rows = dsa.synthetic_weight_rows(0, 1, 256, BF16)
    width = 768 + 128
    assert arrays["m_qb"].shape == (3, 64, 2 * width)
    # m_norms: m_qnorm (1536), then m_knorm: its norms, then the rows of both shards.
    assert arrays["m_norms"].shape == (3, 1, 1536 + 768 + 2 * 256)
    # Rank 3 holds TP32 shards 3 and 19 (attention DP2: groups of 16).
    for shard, head in enumerate((3, 19)):
        np.testing.assert_array_equal(
            arrays["m_qb"][1, :, shard * width + 768 : (shard + 1) * width],
            heads[:, head * 128 : (head + 1) * 128],
        )
        lane = 1536 + 768 + shard * 256
        np.testing.assert_array_equal(
            arrays["m_norms"][1, 0, lane : lane + 256], rows[head]
        )


INDEXER_LAYERS = (3, 7, 92)  # checkpoint MLA layers of the small fixture
INDEXER_HIDDEN, INDEXER_Q_LORA = 256, 64


def indexer_tensors(seed=11, synthetic=False):
    """``{layer: (wq_b, wk, weights_proj, k_norm.weight, k_norm.bias)}``."""
    out = {}
    for m, layer in enumerate(INDEXER_LAYERS):
        if synthetic:
            weight, bias = dsa.synthetic_key_norm(seed, len(INDEXER_LAYERS), BF16)
            out[layer] = (
                dsa.synthetic_query_heads(seed, m, INDEXER_Q_LORA, BF16).T,
                dsa.synthetic_key_projection(seed, m, INDEXER_HIDDEN, BF16).T,
                dsa.synthetic_weight_rows(seed, m, INDEXER_HIDDEN, BF16),
                weight[m, 0],
                bias[m, 0],
            )
            continue
        rng = np.random.default_rng((seed, layer))
        # Scales unlike the synthetic family's, per head of wq_b and weights_proj.
        heads = np.repeat(np.exp2(rng.integers(-6, 7, 32)), 128)[:, None]
        rows = np.exp2(rng.integers(-6, 7, 32))[:, None]
        out[layer] = tuple(
            (scale * rng.standard_normal(shape, np.float32) + shift).astype(BF16)
            for shape, scale, shift in (
                ((4096, INDEXER_Q_LORA), heads, 0),
                ((128, INDEXER_HIDDEN), 0.02, 0),
                ((32, INDEXER_HIDDEN), rows, 0),
                ((128,), 0.1, 1),
                ((128,), 0.1, 0),
            )
        )
    return out


class IndexerCheckpoint:
    """As much of ``load.Checkpoint`` as the indexer's pre-shard reads."""

    def __init__(self, path, tensors):
        self.path = Path(path)
        self.config = SimpleNamespace(
            dim=INDEXER_HIDDEN,
            q_rank=INDEXER_Q_LORA,
            full_attention=INDEXER_LAYERS,
            vocab=1,
        )
        self.tensors = {
            f"language_model.model.layers.{layer}.self_attn.indexer.{name}": value
            for layer, values in tensors.items()
            for name, value in zip(tool_preshard.INDEXER_TENSORS, values, strict=True)
        }
        self.index = dict.fromkeys(self.tensors, "fixture")

    READ_THREADS = 2

    def read(self, key):
        return self.tensors[key]


def indexer_load(tensors, **extra):
    from vllm_torchtpu.megakernel.kimi_k3.kernels import load

    return SimpleNamespace(
        PREFIX=load.PREFIX,
        BF16=load.BF16,
        _family_layers=load._family_layers,
        Checkpoint=lambda path: IndexerCheckpoint(path, tensors),
        **extra,
    )


def stored_indexer(tensors, rank):
    """A rank's indexer families as the pre-shard stores them."""
    per_layer = [dsa.rank_indexer_values(*tensors[i], rank) for i in INDEXER_LAYERS]
    return {
        name: np.stack([layer[name] for layer in per_layer])
        for name in dsa.PRESHARD_INDEXER
    }


@pytest.mark.parametrize("rank", [0, 7, 31])
def test_real_indexer_families_are_the_synthetic_contract(rank):
    # The synthetic tensors in checkpoint form, through the real path, give the
    # synthetic path's arrays exactly: names, shapes, dtypes and head per rank.
    want = dsa.with_synthetic_indexer(k3_families(), "random", seed=11, head=rank)
    got = dsa.with_presharded_indexer(
        {**k3_families(), **stored_indexer(indexer_tensors(synthetic=True), rank)}
    )
    assert set(got) == set(want)
    for name, array in want.items():
        assert got[name].dtype == array.dtype and got[name].shape == array.shape
        np.testing.assert_array_equal(got[name], array)


def test_per_rank_indexer_slices_match_the_reference():
    eps = 1e-5
    keys = jax.random.split(jax.random.PRNGKey(5), 2)
    x = jax.random.normal(keys[0], (4, INDEXER_HIDDEN), jnp.float32).astype(
        jnp.bfloat16
    )
    q_c = jax.random.normal(keys[1], (4, INDEXER_Q_LORA), jnp.float32).astype(
        jnp.bfloat16
    )
    for layer, (wq_b, wk, wv, weight, bias) in indexer_tensors().items():
        m = INDEXER_LAYERS.index(layer)
        want_key = ref.index_key(x, jnp.asarray(wk.T), weight, bias, eps)
        want_q, want_w = ref.index_query(q_c, jnp.asarray(wq_b.T), x, jnp.asarray(wv.T))
        for rank in (0, 13, 31):
            held = stored_indexer(indexer_tensors(), rank)
            key = ref.index_key(
                x,
                jnp.asarray(held["index_wk"][m]),
                held["index_knorm_weight"][m],
                held["index_knorm_bias"][m],
                eps,
            )
            np.testing.assert_array_equal(key, want_key)
            # This rank's head alone gives the reference's values of that head.
            projected = ref._bf16_dot(q_c, jnp.asarray(held["index_wq"][m])).astype(
                jnp.float32
            )
            q, scale = dsa_cache.quantize_index_queries(projected)
            np.testing.assert_array_equal(q, want_q[:, rank])
            v = ref._bf16_dot(x, jnp.asarray(held["index_wv"][m].T)).astype(jnp.float32)
            v = (v * 32.0**-0.5).astype(jnp.bfloat16).astype(jnp.float32)
            np.testing.assert_array_equal(
                v[:, 0] * scale * 128.0**-0.5, want_w[:, rank]
            )


def indexer_cache(tmp_path, ranks=(0, 5)):
    """A k3 rank cache (one dense family per rank) under ``tmp_path``."""
    output = tmp_path / "cache"
    output.mkdir()
    checkpoint = tmp_path / "checkpoint"
    (output / "layout.json").write_text(json.dumps({"checkpoint": str(checkpoint)}))
    dense = np.arange(12, dtype=np.float32).astype(BF16).reshape(3, 1, 4)
    for rank in ranks:
        rank_dir = output / f"rank{rank:02d}"
        rank_dir.mkdir()
        (rank_dir / "dense.bin").write_bytes(dense.tobytes())
        spec = {
            "file": "dense.bin",
            "offset": 0,
            "shape": list(dense.shape),
            "dtype": str(dense.dtype),
        }
        (rank_dir / "index.json").write_text(
            json.dumps(
                {"format": 1, "rank": rank, "layers": 93, "families": {"k_gate": spec}}
            )
        )
    return output, str(checkpoint)


def preshard_args(*argv):
    return tool_main.parser().parse_args(["preshard", *argv])


def add_indexer(checkpoint, output, ranks):
    return tool_preshard.run(
        preshard_args(
            "indexer",
            "--checkpoint",
            checkpoint,
            "--output",
            str(output),
            "--ranks",
            ranks,
        ),
        print,
    )


def test_preshard_indexer_adds_each_ranks_families_and_refuses_to_overwrite(
    tmp_path, monkeypatch
):
    from vllm_torchtpu.megakernel.kimi_k3.kernels import rank_loader

    tensors = indexer_tensors()
    monkeypatch.setattr(
        tool_preshard, "_modules", lambda: (indexer_load(tensors), None)
    )
    output, checkpoint = indexer_cache(tmp_path)
    before = (output / "rank00" / "dense.bin").read_bytes()
    k3_index = json.loads((output / "rank00" / "index.json").read_text())

    # A rank without its cache: refused before any rank is written.
    with pytest.raises(SystemExit, match="rank07"):
        add_indexer(checkpoint, output, "0,7")
    assert not (output / "rank00" / dsa.INDEXER_FILE).exists()
    with pytest.raises(SystemExit, match="weights of"):
        add_indexer("other", output, "0")

    assert add_indexer(checkpoint, output, "0,5") == 0
    assert (output / "rank00" / "dense.bin").read_bytes() == before
    index = json.loads((output / "rank00" / "index.json").read_text())
    assert index["families"]["k_gate"] == k3_index["families"]["k_gate"]
    assert {k: v for k, v in index.items() if k != "families"} == {
        k: v for k, v in k3_index.items() if k != "families"
    }
    for rank in (0, 5):
        rank_dir = rank_loader.RankDir(output, rank)
        assert rank_dir.has_indexer() and rank_dir.dense_names() == ["k_gate"]
        read = rank_dir.read_indexer()
        for name, want in stored_indexer(tensors, rank).items():
            assert read[name].dtype == BF16
            np.testing.assert_array_equal(read[name], want)
    assert not list(output.glob("rank*/*.partial"))
    with pytest.raises(SystemExit, match="already has an indexer"):
        add_indexer(checkpoint, output, "5")


def test_preshard_indexer_refuses_a_k3_checkpoint(tmp_path, monkeypatch):
    k3_checkpoint = indexer_load({})
    monkeypatch.setattr(tool_preshard, "_modules", lambda: (k3_checkpoint, None))
    output, checkpoint = indexer_cache(tmp_path)
    with pytest.raises(SystemExit, match="not a Kimi-K3.1 checkpoint"):
        add_indexer(checkpoint, output, "0")


def test_preshard_verify_indexer_compares_every_rank_and_layer(tmp_path, monkeypatch):
    tensors = indexer_tensors()
    monkeypatch.setattr(
        tool_preshard, "_modules", lambda: (indexer_load(tensors), None)
    )
    output, checkpoint = indexer_cache(tmp_path)
    add_indexer(checkpoint, output, "0,5")
    argv = preshard_args(
        "verify-indexer",
        "--against",
        str(output),
        "--checkpoint",
        checkpoint,
        "--ranks",
        "0,5",
        "--layers",
        "3,92",
    )
    assert tool_preshard.run(argv, print) == 0
    path = output / "rank05" / dsa.INDEXER_FILE
    good = path.read_bytes()
    index = json.loads((output / "rank05" / "index.json").read_text())["families"]
    # One byte of layer 92's head, the last sampled family of the last rank.
    spec = index["index_wq"]
    at = spec["offset"] + 2 * (2 * spec["shape"][1] * spec["shape"][2])
    path.write_bytes(good[:at] + bytes([good[at] ^ 1]) + good[at + 1 :])
    assert tool_preshard.run(argv, print) == 1
    # Another rank's head: the same shapes, in the wrong place.
    other = output / "rank00" / dsa.INDEXER_FILE
    path.write_bytes(other.read_bytes())
    assert tool_preshard.run(argv, print) == 1
    path.write_bytes(good + b"0")
    assert tool_preshard.run(argv, print) == 1
    # A sampled layer that is not an MLA layer would compare nothing.
    path.write_bytes(good)
    with pytest.raises(SystemExit, match="MLA layers"):
        tool_preshard.run(
            preshard_args(
                "verify-indexer",
                "--against",
                str(output),
                "--checkpoint",
                checkpoint,
                "--layers",
                "4",
            ),
            print,
        )


def test_preshard_verify_takes_a_cache_with_an_indexer_and_no_draft(
    tmp_path, monkeypatch
):
    tensors = indexer_tensors()
    dense = np.arange(12, dtype=np.float32).astype(BF16).reshape(3, 1, 4)

    class Checkpoint(IndexerCheckpoint):
        def dense_selected(self, name, layer, rank):
            return dense[0]

    load = indexer_load(
        tensors, selected_local_shapes=lambda *a, **k: {"k_gate": dense.shape}
    )
    load.Checkpoint = lambda path: Checkpoint(path, tensors)
    monkeypatch.setattr(tool_preshard, "_modules", lambda: (load, None))
    output, checkpoint = indexer_cache(tmp_path)
    # Layer 0 samples no expert file (the fixture has none) and the first MLA layer.
    argv = preshard_args(
        "verify",
        "--against",
        str(output),
        "--checkpoint",
        checkpoint,
        "--rank",
        "5",
        "--layers",
        "0",
    )
    assert tool_preshard.run(argv, print) == 0  # before the indexer
    add_indexer(checkpoint, output, "5")
    lines = []
    assert tool_preshard.run(argv, lines.append) == 0
    compared = [line for line in lines if "rank 5 layer 3" in line]
    assert len(compared) == len(dsa.PRESHARD_INDEXER)
    path = output / "rank05" / dsa.INDEXER_FILE
    path.write_bytes(bytes([path.read_bytes()[0] ^ 1]) + path.read_bytes()[1:])
    assert tool_preshard.run(argv, print) == 1


@pytest.mark.parametrize("spread, shift", [(5.0, 1.0), (1e-3, 0.01)])
def test_key_from_projection_is_float32_layernorm_rounded_to_bf16(spread, shift):
    rng = jax.random.split(jax.random.PRNGKey(2), 3)
    projected = (jax.random.normal(rng[0], (64, 128)) * spread + shift).astype(
        jnp.bfloat16
    )
    weight = jax.random.normal(rng[1], (128,)) * 0.5 + 1
    bias = jax.random.normal(rng[2], (128,)) * 0.1
    out = ref.key_from_projection(projected, weight, bias, 1e-5)
    x = np.asarray(projected, np.float32)
    centered = x - x.mean(-1, keepdims=True, dtype=np.float32)
    variance = (centered * centered).mean(-1, keepdims=True, dtype=np.float32)
    normed = centered / np.sqrt(variance + np.float32(1e-5))
    expected = (normed * np.asarray(weight) + np.asarray(bias)).astype(BF16)
    assert out.dtype == jnp.bfloat16
    np.testing.assert_array_equal(
        np.asarray(out).view(np.uint16), expected.view(np.uint16)
    )


def test_scores_match_a_loop():
    rng = jax.random.split(jax.random.PRNGKey(3), 3)
    q = jax.random.normal(rng[0], (2, 32, 128))
    w = jax.random.normal(rng[1], (2, 32))
    keys = jax.random.normal(rng[2], (5, 128))
    out = np.asarray(ref.scores(q, w, keys))
    loop = [
        [
            sum(float(w[t, h]) * max(float(q[t, h] @ keys[j]), 0.0) for h in range(32))
            for j in range(5)
        ]
        for t in range(2)
    ]
    np.testing.assert_allclose(out, loop, rtol=1e-4, atol=1e-4)


def test_select_topk_order_ties_and_padding():
    s = jnp.asarray([[1.0, 2.0, 2.0, 0.0], [5.0, 4.0, 3.0, 9.0]])
    out = np.asarray(ref.select_topk(s, jnp.asarray([4, 2]), 3))
    np.testing.assert_array_equal(out, [[1, 2, 0], [0, 1, -1]])


def test_index_query_folds_scales():
    rng = jax.random.split(jax.random.PRNGKey(4), 4)
    q_c = jax.random.normal(rng[0], (3, 1536)).astype(jnp.bfloat16)
    wq = (jax.random.normal(rng[1], (1536, 4096)) / 40).astype(jnp.bfloat16)
    x = jax.random.normal(rng[2], (3, 7168)).astype(jnp.bfloat16)
    wv = (jax.random.normal(rng[3], (7168, 32)) / 85).astype(jnp.bfloat16)
    q, w = ref.index_query(q_c, wq, x, wv)
    highest = jax.lax.Precision.HIGHEST
    raw = jnp.dot(q_c, wq, precision=highest, preferred_element_type=jnp.float32)
    raw = raw.astype(jnp.bfloat16).astype(jnp.float32).reshape(3, 32, 128)
    q_expected, scale = dsa_cache.quantize_index_queries(raw)
    np.testing.assert_array_equal(np.asarray(q), np.asarray(q_expected))
    v = jnp.dot(x, wv, precision=highest, preferred_element_type=jnp.float32)
    v = (v.astype(jnp.bfloat16).astype(jnp.float32) * 32**-0.5).astype(jnp.bfloat16)
    ratio = np.asarray(w) / (np.asarray(v, np.float32) * np.float32(128**-0.5))
    np.testing.assert_array_equal(np.frexp(ratio)[0], 0.5)
    np.testing.assert_array_equal(ratio, np.asarray(scale))
    np.testing.assert_allclose(
        np.asarray(q) * ratio[..., None],
        np.asarray(raw),
        rtol=2**-4,
        atol=float(np.abs(raw).max()) * 2**-8,
    )


def mla_case(tokens=40, heads=3, seed=7):
    """One query row's MLA inputs: ``q_nope``, ``q_rope``, per-head ``W_UK``
    and ``W_UV`` ``[heads, 512, 128]``, cached latent rows ``[tokens, 576]``
    and the new token's row (FP32)."""
    rng = np.random.default_rng(seed)

    def normal(*shape):
        return rng.standard_normal(shape).astype(np.float32)

    return (
        normal(heads, 128),
        normal(heads, 64),
        normal(heads, 512, 128) / 16,
        normal(heads, 512, 128) / 16,
        normal(tokens, 576),
        normal(576),
    )


def test_sparse_attention_over_every_token_is_dense_mla():
    q_nope, q_rope, w_uk, w_uv, latents, new = mla_case()
    out = ref.sparse_mla_attention(q_nope, q_rope, w_uk, latents, jnp.arange(40), new)
    # Standard attention in float64 over the expanded keys [C @ W_UK | rope]
    # and values C @ W_UV of the cached tokens and the new one.
    c = np.concatenate((latents, new[None])).astype(np.float64)
    keys = np.concatenate(
        (
            np.einsum("tl,hld->htd", c[:, :512], w_uk),
            np.broadcast_to(c[:, 512:], (3, 41, 64)),
        ),
        axis=-1,
    )
    values = np.einsum("tl,hld->htd", c[:, :512], w_uv)
    logits = np.einsum("hd,htd->ht", np.concatenate((q_nope, q_rope), -1), keys)
    logits /= np.sqrt(192.0)
    weights = np.exp(logits - logits.max(-1, keepdims=True))
    dense = np.einsum("ht,htd->hd", weights / weights.sum(-1, keepdims=True), values)
    # W_UV after the latent-space output; FP32 against float64.
    np.testing.assert_allclose(
        np.einsum("hl,hld->hd", np.asarray(out, np.float64), w_uv),
        dense,
        rtol=1e-5,
        atol=1e-5,
    )


def test_sparse_attention_ignores_order_and_unselected_tokens():
    q_nope, q_rope, w_uk, _, latents, new = mla_case()

    def attend(selected, cached=latents):
        return np.asarray(
            ref.sparse_mla_attention(
                q_nope, q_rope, w_uk, cached, jnp.asarray(selected), new
            )
        )

    selected = [3, 17, 29, 11, 38]
    out = attend(selected)
    np.testing.assert_allclose(attend(selected[::-1]), out, rtol=1e-6, atol=1e-6)
    padded = [3, -1, 17, 29, -1, 11, 38, -1]
    np.testing.assert_allclose(attend(padded), out, rtol=1e-6, atol=1e-6)
    # Unselected rows (row 0 too, which the -1 entries gather) never count.
    unselected = np.ones(40, bool)
    unselected[selected] = False
    poisoned = latents.copy()
    poisoned[unselected] = np.nan
    np.testing.assert_array_equal(attend(padded, poisoned), attend(padded))
    # The new token is always a candidate: without cached tokens it is the output.
    np.testing.assert_allclose(
        attend([-1, -1]), np.broadcast_to(new[:512], (3, 512)), rtol=1e-6
    )


def test_global_token_interleaves_members_page_by_page():
    """Local page j of member o is global page j group + o."""
    page, group = 4, 3
    local = np.arange(8)
    assert ref.global_token(local, 1, page, group).tolist() == [
        4,
        5,
        6,
        7,
        16,
        17,
        18,
        19,
    ]


def _split_scores(values, position, page, group, tokens):
    """Each member's local scores [1, tokens] of a global row ``values``, NaN past
    its local length."""
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_decode

    out = []
    for o in range(group):
        row = np.full((1, tokens), np.nan, np.float32)
        length = int(dsa_decode.local_length(position, o, page, group))
        row[0, :length] = values[ref.global_token(np.arange(length), o, page, group)]
        out.append(row)
    return out


@pytest.mark.parametrize(
    "page, group, tokens, k, positions",
    [
        (4, 3, 40, 16, (10, 15, 50, 119)),
        (512, 32, 1536, 2048, (2047, 40000)),  # the decode geometry (smoke)
    ],
)
@pytest.mark.parametrize("ties", [False, True])
def test_decode_selection_is_select_topk_in_global_order(
    ties, page, group, tokens, k, positions
):
    """Members' local scores reassemble in global order; the top k with ties to
    the lower global index; the new token (index position) as a flag, never in
    a mask, and its page owner's last local token; short rows select every
    token."""
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_decode

    rng = np.random.default_rng(5)
    for position in positions:
        owner = position // page % group
        last = int(dsa_decode.local_length(position, owner, page, group)) - 1
        assert ref.global_token(last, owner, page, group) == position
        values = rng.standard_normal(position + 1).astype(np.float32)
        if ties:
            values = np.round(values).astype(np.float32)
        local = _split_scores(values, position, page, group, tokens)
        masks, new = ref.decode_selection(
            local, [position], page_tokens=page, group=group, k=k
        )
        if position + 1 <= k:
            chosen = set(range(position + 1))
        else:
            chosen = set(
                np.asarray(
                    ref.select_topk(
                        jnp.asarray(values[None]), jnp.asarray([position + 1]), k
                    )
                )[0].tolist()
            )
        got = {
            int(ref.global_token(t, o, page, group))
            for o in range(group)
            for t in np.flatnonzero(masks[o][0])
        }
        assert position not in got
        assert got | ({position} if new[0] else set()) == chosen
        if position + 1 > k:
            assert len(got) + int(new[0]) == k


def _pad_lanes(rows):
    """Latent rows [n, 576] -> [n, 640], lanes 576:640 NaN (ignored)."""
    return np.concatenate((rows, np.full((len(rows), 64), np.nan)), axis=1)


def test_latent_attention_is_sparse_mla_attention_over_absorbed_queries():
    """latent_attention over absorbed queries (W_UK and the score scale folded
    in, lanes 576:640 ignored) equals sparse_mla_attention over the same rows
    and the new token, its lse dsa_micro.fold_exact_lse's; NEG and zeros with
    no rows."""
    from tools.megakernel import dsa_micro
    from vllm_torchtpu.megakernel.kimi_k3.kernels.mla_dcp import NEG

    q_nope, q_rope, w_uk, _, latents, new = mla_case()
    selected = [3, 17, 29, 11, 38]
    want = ref.sparse_mla_attention(
        q_nope, q_rope, w_uk, latents, jnp.asarray(selected), new
    )
    absorbed = np.einsum(
        "hd,hld->hl", np.asarray(q_nope, np.float64), np.asarray(w_uk, np.float64)
    )
    query = np.concatenate(
        (absorbed, np.asarray(q_rope, np.float64), np.zeros((len(absorbed), 64))),
        axis=1,
    ) / np.sqrt(192.0)
    rows = np.asarray(latents, np.float64)[selected]
    new_row = np.asarray(new, np.float64)
    o, lse = ref.latent_attention(query, _pad_lanes(rows), _pad_lanes(new_row[None])[0])
    np.testing.assert_allclose(o, np.asarray(want), rtol=1e-5, atol=1e-5)
    every = np.concatenate((rows, new_row[None]))
    want_lse = dsa_micro.fold_exact_lse(query, np.ones(len(query)), every)
    np.testing.assert_allclose(lse, want_lse, rtol=1e-12)
    o0, lse0 = ref.latent_attention(query, np.zeros((0, 640)))
    assert (o0 == 0).all() and (lse0 == NEG).all()


def test_selection_mismatches_count_missing_extra_and_the_flag():
    want = [np.array([[1, 0, 1, 0]], bool), np.array([[0, 1, 0, 0]], bool)]
    got = [np.array([[1, 0, 0, 1]], bool), np.array([[0, 1, 0, 0]], bool)]
    assert tool_check.selection_mismatches(got, [True], want, [False]) == (1, 2)
    assert tool_check.selection_mismatches(want, [False], want, [False]) == (0, 0)


def test_attention_error_is_relative_to_the_reference_maximum():
    want = np.array([[2.0, -4.0]])
    assert tool_check.attention_error(np.array([[2.0, -3.0]]), want) == 0.25
    assert tool_check.attention_error(np.array([[np.nan, 0.0]]), want) == float("inf")


def test_attend_summary_reports_a_wrong_head():
    """Over sampled heads ``[..., 512]``: one head off the reference is the one
    head over the tolerance, the worst error is that head's and names its
    sample; the long rows' heads count by their new-token flag."""
    rng = np.random.default_rng(5)
    want = rng.standard_normal((2, 3, 3, 512))
    merged = want.copy()
    merged[1, 2, 0] += 0.5 * np.abs(want[1, 2, 0]).max()
    flags = {(0, 0): True, (0, 1): False, (1, 0): False, (1, 1): True}
    samples = [
        (7, m, b, h, b < 2, flags.get((m, b), True))
        for m in range(2)
        for b in range(3)
        for h in range(3)
    ]
    summary = tool_check.attend_summary(merged, want, 0.1, samples)
    assert summary == {
        "attend_samples": 18,
        "attend_worst": tool_check.attention_error(merged[1, 2, 0], want[1, 2, 0]),
        "attend_worst_at": "r07 layer 1 row 2 head 0 short flagged",
        "attend_over": 1,
        "attend_flagged": 6,
        "attend_unflagged": 6,
    }
    assert tool_check.attend_summary(want, want, 0.1, samples)["attend_over"] == 0
    with pytest.raises(ValueError, match="samples"):
        tool_check.attend_summary(want, want, 0.1, samples[1:])


def test_tie_keys_repeat_a_few_keys_on_every_rank():
    """``tie_index_keys`` gives every MLA layer's token slots the same few keys
    (slot ``s`` the ``s % patterns``-th, bytes and exponent), the same on every
    rank, so a row's scores take at most that many values; ``needs_tie_quota``
    tells a row whose ties at its ``k``-th score overflow the top-k."""
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_cache

    page, patterns = 256, 7
    index = np.random.default_rng(0).integers(
        0, 255, (3, 2, *dsa_cache.index_block_shape(page)), np.uint8
    )
    tied = tool_check.tie_index_keys(index, page, patterns)
    assert tied.shape == index.shape and tied.dtype == np.uint8
    used = slice(0, page + page // 128)  # the keys and their exponent rows
    np.testing.assert_array_equal(
        tool_check.tie_index_keys(index * 0, page, 7)[:, :, used], tied[:, :, used]
    )
    slots = np.arange(page)
    keys = tied[:, :, slots]
    exponents = tied[:, :, page + slots // 128, slots % 128]
    for s in range(page):
        np.testing.assert_array_equal(keys[:, :, s], keys[:, :, s % patterns])
        np.testing.assert_array_equal(exponents[:, :, s], exponents[:, :, s % patterns])
    assert len({keys[0, 0, s].tobytes() for s in range(patterns)}) == patterns
    assert tool_check.needs_tie_quota(np.array([3.0, 2, 2, 2, 1]), 2)
    assert not tool_check.needs_tie_quota(np.array([3.0, 2, 2, 2, 1]), 4)
    assert not tool_check.needs_tie_quota(np.array([3.0, 2, 1, 0]), 2)


def test_index_cache_routes_apart_from_the_unified_pools(monkeypatch):
    import torch

    from vllm_torchtpu.megakernel.kimi_k3 import layout, pools

    # Stand-in tensors: skip the unified-pool layout check (it wants real block
    # shapes).
    monkeypatch.setattr(
        pools, "cache_layout", lambda *_: SimpleNamespace(check=lambda _pools: None)
    )
    unified = [torch.zeros((2, 4)) for _ in range(24)]
    index = torch.zeros((2, 24, 3, 4), dtype=torch.uint8)
    caches = {
        f"model.layers.{layer}.self_attn.mla_attn.attn": unified[m]
        for m, layer in enumerate(layout.MLA)
    }
    caches.update(
        {
            f"model.layers.{layer}.self_attn": unified[k // 3]
            for k, layer in enumerate(layout.KDA)
        }
    )
    assert pools.index_raw_cache(caches) is None
    names = [f"model.layers.{layer}{pools.INDEX_CACHE}" for layer in layout.MLA]
    caches.update(dict.fromkeys(names, index))
    assert pools.index_raw_cache(caches) is index
    # The index names must not count as MLA layers (else: "expected 24 MLA and
    # 69 KDA layers").
    raw, mla_raw, *_ = pools.bind_raw_pools(caches)
    assert len(raw) == 24
    assert [raw[i].data_ptr() for i in mla_raw] == [t.data_ptr() for t in unified]
    narrow = torch.zeros((2, 23, 3, 4), dtype=torch.uint8)
    for bad in (
        {**caches, names[3]: torch.zeros_like(index)},  # two tensors
        {**caches, names[3]: index.view(index.shape)},  # two objects, one storage
        {name: cache for name, cache in caches.items() if name != names[5]},
        {**caches, **dict.fromkeys(names, narrow)},  # not 24 pages a block
    ):
        with pytest.raises(RuntimeError, match="index-key cache"):
            pools.index_raw_cache(bad)


@pytest.mark.parametrize("spread, shift", [(5.0, 1.0), (1e-3, 0.01), (0.05, 3.0)])
def test_kernel_index_key_matches_the_reference(spread, shift):
    from vllm_torchtpu.megakernel.kimi_k3.kernels import decode_megakernel as kernel

    rng = jax.random.split(jax.random.PRNGKey(5), 4)
    # The index-key projection's own dot output [rows, 128] (BF16), and the
    # key-value normalization row [latent 512 | weight 128 | bias 128].
    projected = (jax.random.normal(rng[0], (64, 128)) * spread + shift).astype(
        jnp.bfloat16
    )
    weight = (jax.random.normal(rng[1], (128,)) * 0.5 + 1).astype(jnp.bfloat16)
    bias = (jax.random.normal(rng[2], (128,)) * 0.1).astype(jnp.bfloat16)
    latent_norm = jax.random.normal(rng[3], (512,)).astype(jnp.bfloat16)
    normalization = jnp.concatenate((latent_norm, weight, bias))[None]
    # Both compiled: an eager reference can differ from a fused one by 1 ulp.
    got = jax.jit(kernel._index_key, static_argnums=2)(projected, normalization, 1e-5)
    want = jax.jit(ref.key_from_projection, static_argnums=3)(
        projected, weight, bias, 1e-5
    )
    assert got.dtype == jnp.bfloat16
    np.testing.assert_array_equal(
        np.asarray(got).view(np.uint16), np.asarray(want).view(np.uint16)
    )


@pytest.mark.parametrize("make", [ref.finite_bf16_keys, ref.every_amax_keys])
def test_kernel_key_encoding_is_quantize_index_keys(make):
    keys = jnp.asarray(make())
    codes, exponents = jax.jit(dsa_cache.index_key_rows)(keys)
    assert codes.dtype == exponents.dtype == jnp.bfloat16
    assert np.all(np.asarray(exponents) == np.asarray(exponents)[:, :1])
    got = jax.jit(dsa_cache.index_key_bytes)(codes, exponents)
    for a, b in zip(got, dsa_cache.quantize_index_keys(keys), strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize("make", [ref.finite_bf16_keys, ref.every_amax_keys])
def test_query_encoding_is_quantize_index_queries(make):
    queries = jnp.asarray(make())
    values, exponent = jax.jit(
        lambda q: dsa_cache.encode_index_keys(q, floor=dsa_cache.QUERY_AMAX_FLOOR)
    )(queries)
    want, scale = dsa_cache.quantize_index_queries(queries)
    np.testing.assert_array_equal(np.asarray(values), np.asarray(want))
    np.testing.assert_array_equal(
        np.asarray(dsa_cache.power_of_two(exponent[..., 0] - 127)), np.asarray(scale)
    )


def test_dsa_encode_check_counts_wrong_rows():
    keys = ref.finite_bf16_keys()
    outputs = tool_check.reference_encodings(keys)
    assert tool_check.encode_mismatches(keys, *outputs) == (0, 0)
    codes, exponents, values, value_exponents = outputs
    codes[3, 5] = 448.0 if codes[3, 5] != 448.0 else 0.0
    values[7, 0] = 448.0 if values[7, 0] != 448.0 else 0.0
    value_exponents[9, 4] += 1  # the exponent must fill every lane
    assert tool_check.encode_mismatches(keys, *outputs) == (1, 2)


def test_dsa_encode_check_compares_query_values_bit_for_bit():
    keys = ref.finite_bf16_keys()
    outputs = tool_check.reference_encodings(keys)
    values = outputs[2]
    row, lane = np.argwhere(values == 0)[0]
    values[row, lane] = -values[row, lane]  # equal to its other zero as a float
    assert tool_check.encode_mismatches(keys, *outputs) == (0, 1)


def test_plain_step_commits_index_keys_where_the_latents_go(monkeypatch):
    """The plain step's body on one CPU device, the fused decoder stubbed:
    K3.1's 64 operands (35 families and the packed MLA norms, the KDA banks,
    24 unified pools, the index cache, the control), the norms split back for
    the decoder, and the kernel's encoded keys (new-value rows 0 and 1)
    committed into the donated index cache at the tokens the latent rows take,
    a padding row's at null block 0, slot 0; Kimi-K3's op unchanged."""
    import torch
    from jax.sharding import Mesh

    from vllm_torchtpu.megakernel.kimi_k3 import decode_ops
    from vllm_torchtpu.megakernel.kimi_k3.kernels import paged

    batch, layers, slots, page = 4, 24, 4, 256
    rng = jax.random.split(jax.random.PRNGKey(6), 2)
    new_keys = jax.random.normal(rng[0], (batch, layers, 3, 256)).astype(jnp.bfloat16)
    keys = jax.random.normal(rng[1], (batch, layers, 128)).astype(jnp.bfloat16)
    codes, exponents = dsa_cache.index_key_rows(keys)
    new_values = jnp.zeros((batch, layers, 3, 128), jnp.bfloat16)
    new_values = new_values.at[:, :, 0].set(codes).at[:, :, 1].set(exponents)
    seen = {}
    unpacked = []

    def decode_local(weights, states, tokens, positions, **_):
        seen.update(weights)
        logits = jnp.zeros((1, batch, 128), jnp.float32)
        return logits, (*states[:2], new_keys, new_values)

    registered = {}
    export_keys = {}

    def register_op(
        name, mesh, body, in_specs, out_specs, donate_argnums=(), export_key=None
    ):
        registered[name] = (body, in_specs, out_specs, donate_argnums)
        export_keys[name] = export_key
        return name

    def unpack_spy(weights):
        out = dsa.unpack_mla_norms(weights)
        if "m_norms" in weights:
            unpacked.append(weights["m_norms"].shape)
        return out

    monkeypatch.setattr(paged, "decode_local", decode_local)
    monkeypatch.setattr(decode_ops, "register_op", register_op)
    monkeypatch.setattr(decode_ops, "unpack_mla_norms", unpack_spy)
    # The latent rows' generic scatter: the fused commit (four rows) is a TPU kernel.
    cpu_config = paged._config().replace(fused_cache_commit=False)
    monkeypatch.setattr(paged, "_config", lambda: cpu_config)
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("tp",))

    def run(name, *args):
        body, in_specs, out_specs, _ = registered[name]
        return jax.jit(
            jax.shard_map(
                body,
                mesh=mesh,
                in_specs=in_specs,
                out_specs=out_specs,
                check_vma=False,
            )
        )(*args)

    ops = decode_ops.DecodeOps.__new__(decode_ops.DecodeOps)
    ops.paged, ops.mesh, ops.batch_sizes, ops._ops = paged, mesh, (batch,), {}
    # K3.1's decode operands: 35 of K3's families and the packed MLA norms.
    ops.names = (*(f"w{i}" for i in range(35)), "m_norms")
    ops.pool_map, ops.owner_table = tuple(range(layers)), None
    ops.unified_pools = [torch.zeros(1)] * layers  # counted only
    ops.index_cache = torch.zeros(1)
    control = decode_ops.Control(slots, batch, pages=2)
    plain = f"pallas::k3_decode_owned_b{batch}"
    ops._build_plain(batch, control, slots, False, False)
    k31_key = export_keys[plain]
    _, plain_in, plain_out, plain_donated = registered[plain]
    # Torch ops take at most 64 operands.
    assert len(plain_in) == 36 + 2 + layers + 1 + 1 == 64
    assert len(plain_out) == 4 + layers + 1
    assert plain_donated == tuple(range(36, 36 + 2 + layers + 1))
    assert list(registered) == [plain]  # no separate commit op

    # Pages 0 and 32 are this rank's (blocks 1 and 2 of the rows' tables);
    # page 1 is not. Row 3 is padding (slot 0, position 0, a zero table row).
    positions = [5, 32 * page + 7, page + 9]
    entries = [(r + 1, True, 0, p, decode_ops.GREEDY) for r, p in enumerate(positions)]
    plan = control.plan(entries, [[1, 0], [0, 2], [1, 2]])
    state = jnp.asarray(np.concatenate((np.zeros(3 * slots, np.int32), plan)))
    norms = jnp.arange(layers * 2304, dtype=jnp.float32).reshape(1, layers, 1, 2304)
    norms = norms.astype(jnp.bfloat16)
    banks_and_pools = (
        jnp.zeros((1, slots, 1), jnp.bfloat16),
        jnp.zeros((1, slots, 1), jnp.float32),
        *[jnp.zeros((3, 128, 2, 640), jnp.bfloat16) for _ in range(layers)],
    )
    index = jnp.zeros((3, layers, *dsa_cache.index_block_shape(page)), jnp.uint8)
    outputs = run(
        plain,
        *[jnp.zeros((1, 1), jnp.bfloat16)] * 35,
        norms,
        *banks_and_pools,
        index,
        state,
    )
    # The decoder sees the norms split back. The stub runs under the trace, so
    # the split's values are covered on concrete arrays by the pack test and
    # the two widths here tell the halves apart.
    assert unpacked == [(layers, 1, 2304)]
    assert seen["m_qnorm"].shape == (layers, 1, 1536)
    assert seen["m_knorm"].shape == (layers, 1, 768)
    assert "m_norms" not in seen
    latents = np.asarray(paged.latent_rows_from_handoff(new_keys))
    q, e = (np.asarray(a) for a in dsa_cache.quantize_index_keys(keys))
    committed = np.asarray(outputs[-1]).copy()
    for m in range(layers):
        latent = np.asarray(outputs[4 + m])
        for r, block, token in [(0, 1, 5), (1, 2, 7), (3, 0, 0)]:
            np.testing.assert_array_equal(
                latent[block, token // 2, token % 2], latents[r, m]
            )
            np.testing.assert_array_equal(committed[block, m, token], q[r, m])
            scale = (block, m, page + token // 128, token % 128)
            assert committed[scale] == e[r, m]
            committed[block, m, token] = committed[scale] = 0
    assert not committed.any()  # row 2 dropped, nothing else written

    # Kimi-K3: its 37 families and no index cache: the same step, no index
    # output, and an export key (its persistent cache entry) without one. Built
    # over K3's names, K3.1's key is K3's plus the index-cache marker.
    k3_names = tuple(f"w{i}" for i in range(37))
    ops.names, ops._ops = k3_names, {}
    ops._build_plain(batch, control, slots, False, False)
    k31_key_over_k3_names = export_keys[plain]
    ops.index_cache, ops._ops = None, {}
    ops._build_plain(batch, control, slots, False, False)
    assert "index_cache" in k31_key and "index_cache" not in export_keys[plain]
    assert k31_key_over_k3_names == export_keys[plain][:-1] + ", 'index_cache')"
    assert len(registered[plain][1]) == 37 + 2 + layers + 1
    k3 = run(plain, *[jnp.zeros((1, 1), jnp.bfloat16)] * 37, *banks_and_pools, state)
    assert len(registered[plain][2]) == len(k3) == 4 + layers
    for got, want in zip(k3, outputs[:-1], strict=True):
        np.testing.assert_array_equal(got, want)


def test_plain_step_hands_the_index_debug_outputs_on(monkeypatch):
    """With index_debug the K3.1 plain op asks the decoder for its check
    outputs and returns them after the donated index cache (an extra output
    each, the export key marked); plain_step keeps them apart from the
    tensors it adopts."""
    import torch
    from jax.sharding import Mesh

    from vllm_torchtpu.megakernel.kimi_k3 import decode_ops
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_decode, paged

    batch, layers, slots = 2, 24, 4
    count = len(dsa_decode.DEBUG_OUTPUTS)
    asked = []

    def decode_paged(weights, conv, recurrent, pool, *args, index_cache=None, **kw):
        asked.append(kw.get("index_debug", False))
        debug = tuple(jnp.full((1, 2), i, jnp.float32) for i in range(count))
        logits = jnp.zeros((batch, 128), jnp.float32)
        return (logits, conv, recurrent, tuple(pool), index_cache, *debug)

    registered, export_keys = {}, {}

    def register_op(
        name, mesh, body, in_specs, out_specs, donate_argnums=(), export_key=None
    ):
        registered[name] = (body, in_specs, out_specs)
        export_keys[name] = export_key
        return name

    monkeypatch.setattr(paged, "decode_paged", decode_paged)
    monkeypatch.setattr(decode_ops, "register_op", register_op)
    ops = decode_ops.DecodeOps.__new__(decode_ops.DecodeOps)
    ops.paged, ops.batch_sizes, ops._ops = paged, (batch,), {}
    ops.mesh = Mesh(np.asarray(jax.devices()[:1]), ("tp",))
    ops.names = (*(f"w{i}" for i in range(35)), "m_norms")
    ops.pool_map, ops.owner_table = tuple(range(layers)), None
    ops.unified_pools = [torch.zeros(1) for _ in range(layers)]
    ops.index_cache = torch.zeros(1)
    ops.index_debug, ops.index_debug_outputs = True, ()
    control = decode_ops.Control(slots, batch, pages=2)
    name = f"pallas::k3_decode_owned_b{batch}"
    ops._build_plain(batch, control, slots, False, False)
    body, in_specs, out_specs = registered[name]
    assert len(out_specs) == 4 + layers + 1 + count
    assert export_keys[name].endswith("'index_cache', 'index_debug')")
    plan = control.plan([(1, True, 0, 5, decode_ops.GREEDY)], [[1, 0]])
    state = jnp.asarray(np.concatenate((np.zeros(3 * slots, np.int32), plan)))
    outputs = jax.jit(
        jax.shard_map(
            body, mesh=ops.mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False
        )
    )(
        *[jnp.zeros((1, 1), jnp.bfloat16)] * 36,
        jnp.zeros((1, slots, 1), jnp.bfloat16),
        jnp.zeros((1, slots, 1), jnp.float32),
        *[jnp.zeros((3, 128, 2, 640), jnp.bfloat16) for _ in range(layers)],
        jnp.zeros((3, layers, 288, 128), jnp.uint8),
        state,
    )
    assert asked == [True]
    for i, output in enumerate(outputs[-count:]):
        np.testing.assert_array_equal(output, np.full((1, 2), i))

    fresh = [torch.ones(1) for _ in range(layers + 1)]
    debug = [torch.full((1,), float(i)) for i in range(count)]
    adopted = []
    ops.weights = {n: torch.zeros(1) for n in ops.names}
    ops.slots = [torch.zeros(1), torch.zeros(1)]
    ops._adopt = lambda conv, recurrent, pools: adopted.append(list(pools))
    ops._run = lambda op, rows, *args: (*[torch.zeros(1)] * 4, *fresh, *debug)
    ops.plain_step(torch.zeros(1), batch, control, slots, False)
    assert len(adopted) == 1
    assert all(a is b for a, b in zip(adopted[0], fresh, strict=True))
    assert all(a is b for a, b in zip(ops.index_debug_outputs, debug, strict=True))


def test_index_cache_binds_beside_the_unified_pools():
    import torch

    from vllm_torchtpu.megakernel.kimi_k3 import decode_ops

    ops = decode_ops.DecodeOps.__new__(decode_ops.DecodeOps)
    ops._ops, ops.dsa, ops.index_cache = {}, dsa.DsaConfig(), None
    ops.weights, ops.names = {}, ()
    ops.slots = [torch.zeros(1), torch.zeros(1)]
    ops.unified_pools = [torch.zeros((5, 128, 2, 640), dtype=torch.bfloat16)]
    cache = torch.zeros((5, 24, 288, 128), dtype=torch.uint8)
    for bad in (
        None,
        cache[:, :23],
        cache.to(torch.int8),
        torch.zeros((5, 24, 544, 128), dtype=torch.uint8),  # FP8 pages' blocks
    ):
        with pytest.raises(RuntimeError, match="uint8 index cache"):
            ops.use_index_cache(bad)
    ops.use_index_cache(cache)
    assert ops.index_cache is cache and ops._operands()[-1] is cache
    # The plain step donates it after the pools; adoption keeps its identity.
    fresh = [torch.ones_like(t) for t in (*ops.unified_pools, cache)]
    ops._adopt(*ops.slots, fresh)
    assert ops.index_cache is cache and bool(cache.eq(1).all())
    ops._ops = {"built": None}
    with pytest.raises(RuntimeError, match="before building"):
        ops.use_index_cache(cache)
    ops._ops, ops.dsa = {}, None
    with pytest.raises(RuntimeError, match="DSA"):
        ops.use_index_cache(cache)


def test_synthetic_index_keys_differ_per_rank_block_layer_and_token():
    blocks, tokens, layers = 3, 256, 2
    a = tool_k3.synthetic_index_keys(blocks, tokens, rank=0, layers=layers)
    assert a.shape == (blocks, layers, 288, 128) and a.dtype == np.uint8
    np.testing.assert_array_equal(
        a, tool_k3.synthetic_index_keys(blocks, tokens, rank=0, layers=layers)
    )
    assert not np.array_equal(
        a, tool_k3.synthetic_index_keys(blocks, tokens, rank=1, layers=layers)
    )
    assert not np.array_equal(a[:, 0], a[:, 1])
    assert not any(np.array_equal(a[i], a[j]) for i in range(blocks) for j in range(i))
    keys = a[:, :, :tokens]
    exponents = a[:, :, tokens : tokens + 2].reshape(blocks, layers, tokens)
    assert not a[:, :, tokens + 2 :].any() and len(np.unique(exponents)) > 4
    decoded = dsa_cache.dequantize_index_keys(jnp.asarray(keys), jnp.asarray(exponents))
    assert np.isfinite(np.asarray(decoded)).all()
    for layer in range(layers):  # no two tokens of a layer share a key
        rows = np.concatenate(
            (keys[:, layer].reshape(-1, 128), exponents[:, layer].reshape(-1, 1)), 1
        )
        assert len(np.unique(rows, axis=0)) == len(rows)
    # A rank's keys are quantize_index_keys of its seeded draw; layer l holds
    # them in the order of its own seeded permutation of the token rows.
    rng = np.random.default_rng((0, 0))
    draw = rng.standard_normal((blocks, tokens, 128), np.float32) * np.exp2(
        rng.integers(-4, 5, (blocks, tokens, 1))
    ).astype(np.float32)
    want, want_exponent = (
        np.asarray(x)
        for x in dsa_cache.quantize_index_keys(jnp.asarray(draw.astype(BF16)))
    )
    for layer in range(layers):
        order = np.random.default_rng((0, 0, layer + 1)).permutation(blocks * tokens)
        np.testing.assert_array_equal(
            keys[:, layer], want.reshape(-1, 128)[order].reshape(want.shape)
        )
        np.testing.assert_array_equal(
            exponents[:, layer],
            want_exponent.reshape(-1)[order].reshape(blocks, tokens),
        )


def test_pools_fill_the_index_cache_with_this_ranks_keys_only_under_pattern(
    monkeypatch,
):
    import torch

    from vllm_torchtpu.megakernel import bridge
    from vllm_torchtpu.megakernel.kimi_k3 import layout
    from vllm_torchtpu.megakernel.kimi_k3.pools import INDEX_CACHE

    to = torch.Tensor.to
    monkeypatch.setattr(
        torch.Tensor,
        "to",
        lambda self, *args, **kwargs: self
        if args == ("tpu",)
        else to(self, *args, **kwargs),
    )
    monkeypatch.setattr(bridge, "kernel_mesh", lambda axis, world_size=None: (None, 7))
    monkeypatch.setattr(tool_k3, "DSA", (dsa.DsaConfig(), ("random", 0)))
    zeros = tool_k3.Pools(3, 1, 1).index_tensor
    assert zeros.shape == (3, 24, tool_k3.PAGE + 32, 128) and not zeros.any()
    pools = tool_k3.Pools(3, 1, 1, pattern=True)
    np.testing.assert_array_equal(
        pools.index_tensor.numpy(), tool_k3.synthetic_index_keys(3, tool_k3.PAGE, 7)
    )
    assert all(
        pools.caches[f"model.layers.{layer}{INDEX_CACHE}"] is pools.index_tensor
        for layer in layout.MLA
    )


def test_k31_decode_ops_refuse_k3_mixes_consistently():
    from vllm_torchtpu.megakernel.kimi_k3 import decode_ops

    # The indexer exists only for K3.1: a synthetic one without `dsa` is a typo.
    with pytest.raises(ValueError, match="synthetic_indexer"):
        decode_ops.DecodeOps(Path("unused"), 8, synthetic_indexer=("random", 0))
    # K3's cached verifier ops are returned as before.
    ops = decode_ops.DecodeOps.__new__(decode_ops.DecodeOps)
    control = SimpleNamespace(key=lambda: "control")
    ops.dsa, ops.index_cache = None, None
    ops._ops = {
        ("verify", (), "control", None): "single",
        ("verify", (), 2, 4, "control", None): "batched",
    }
    assert ops._build_verify((), control) == "single"
    assert ops._build_verify_batched((), 2, 4, control) == "batched"


def test_index_keys_reject_unimplemented_paths():
    from vllm_torchtpu.megakernel.config import MegakernelEngineConfig
    from vllm_torchtpu.megakernel.kimi_k3 import decode_ops, program
    from vllm_torchtpu.megakernel.kimi_k3.kernels import decode_megakernel, paged

    # A served K3.1 config has no index-key weights yet: it must not run as K3.
    config = MegakernelEngineConfig(
        role="decode",
        tensor_parallel_size=32,
        decode_context_parallel_size=32,
        block_size=256,
        max_num_seqs=8,
        max_model_len=65536,
        max_num_batched_tokens=512,
        dsa=dsa.DsaConfig(),
    )
    with pytest.raises(NotImplementedError, match="needs indexer weights"):
        program.build(config, Path("unused"), tp_rank=0)
    with pytest.raises(NotImplementedError, match="index-key weights"):
        decode_ops.DecodeOps(Path("unused"), 8, dsa=dsa.DsaConfig())
    # An attention DP group's rows must be a scorer batch class.
    rows = jnp.zeros(6, jnp.int32)
    for options in (dict(attention_dp=2), dict(attention_dp=4)):
        with pytest.raises(NotImplementedError, match="K3.1 index keys"):
            paged.decode_paged(
                {},
                None,
                None,
                (),
                rows,
                rows,
                None,
                rows,
                batch_size=6,
                dcp_pool_map=(),
                index_cache=jnp.zeros((1, 24, 288, 128), jnp.uint8),
                **options,
            )
    # K3.1 is built and checked for the batch classes of decode_batch_sizes(64).
    for batch in (48, 96):
        rows = jnp.zeros(batch, jnp.int32)
        with pytest.raises(NotImplementedError, match="K3.1 index keys"):
            paged.decode_paged(
                {},
                None,
                None,
                (),
                rows,
                rows,
                None,
                rows,
                batch_size=batch,
                dcp_pool_map=(),
                index_cache=jnp.zeros((1, 24, 288, 128), jnp.uint8),
            )
    weights = SimpleNamespace(
        index_key_projection=jax.ShapeDtypeStruct((7168, 128), jnp.bfloat16)
    )
    with pytest.raises(ValueError, match="index scratch"):
        decode_megakernel.mla_dcp_shards(
            None,
            None,
            None,
            None,
            None,
            weights=weights,
            streams={},
            communication_workspace=None,
            dcp=None,
        )


def test_k31_batch_classes_are_the_decode_batch_sizes_to_64():
    from vllm_torchtpu.megakernel.kimi_k3 import decode_ops
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_decode

    assert decode_ops.decode_batch_sizes(64) == dsa_decode.BATCH_CLASSES
    assert BATCH_CLASSES == dsa_decode.BATCH_CLASSES


# Decode weight families stored in FP32 or as packed experts (others BF16).
FAMILY_DTYPES = {
    **dict.fromkeys(
        ("attn_res", "mlp_res", "output_res", "latent_norm", "router_bias"),
        jnp.float32,
    ),
    **dict.fromkeys(("k_conv", "k_a_log", "k_dt", "k_norm"), jnp.float32),
    **dict.fromkeys(("expert_gu", "expert_down"), jnp.uint32),
    **dict.fromkeys(("expert_gus", "expert_ds"), jnp.uint8),
}
# Every TP32 batch K3.1 builds (decode_batch_sizes(64)).
BATCH_CLASSES = (1, 2, 4, 8, 16, 32, 64)


class _Planned(Exception):
    pass


def decode_step(
    batch,
    cache_dtype,
    k31,
    *,
    pages=2,
    index_debug=False,
    widths=None,
    index_cache=None,
    attention_dp=1,
    options=None,
):
    """One TP32 rank's plain decode step at ``batch`` on one CPU device,
    ``widths`` overriding families' last dimension, ``index_cache`` (default
    ``k31``) whether the step takes an index cache, ``attention_dp`` the held
    shards' families (the installed kernel config's degree): ``(step,
    operands)``, the operands abstract shapes."""
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.megakernel.kimi_k3 import layout
    from vllm_torchtpu.megakernel.kimi_k3.decode_ops import decode_options
    from vllm_torchtpu.megakernel.kimi_k3.kernels import load, paged
    from vllm_torchtpu.megakernel.kimi_k3.kernels.decode_megakernel import (
        ATTENTION_DP_WEIGHTS,
    )

    shapes = load.selected_local_shapes()
    if k31:
        shapes["m_ka"] = (*shapes["m_ka"][:-1], 640 + 128)
        shapes["m_qb"] = (*shapes["m_qb"][:-1], 768 + 128)
        shapes["m_knorm"] = (*shapes["m_knorm"][:-1], 512 + 256 + shapes["m_ka"][-2])
    for name, width in (widths or {}).items():
        shapes[name] = (*shapes[name][:-1], width)
    if attention_dp > 1:
        for name, (_, axis) in ATTENTION_DP_WEIGHTS.items():
            shape = list(shapes[name])
            if axis is None:
                shape.insert(1, attention_dp)
            else:
                shape[1 + axis] *= attention_dp
            shapes[name] = tuple(shape)
        if k31:  # every held shard's weights_proj row after the norms
            shapes["m_knorm"] = (
                *shapes["m_knorm"][:-1],
                768 + attention_dp * shapes["m_ka"][-2],
            )
    names = tuple(sorted(shapes))
    cache = layout.cache_layout(cache_dtype)
    slots = 2 * batch + 1 if attention_dp == 1 else attention_dp * (batch + 1)
    count = len(names)
    operands = (
        *[
            jax.ShapeDtypeStruct((1, *shapes[n]), FAMILY_DTYPES.get(n, jnp.bfloat16))
            for n in names
        ],
        jax.ShapeDtypeStruct(
            (1, slots, len(layout.KDA), 3, 3, layout.CONV_HEAD_ROWS, 128),
            jnp.bfloat16,
        ),
        jax.ShapeDtypeStruct((1, slots, len(layout.KDA), 3, 128, 128), jnp.float32),
        *[
            jax.ShapeDtypeStruct((4, *cache.block_shape), jnp.dtype(cache.dtype))
            for _ in layout.MLA
        ],
        jax.ShapeDtypeStruct((batch,), jnp.int32),
        jax.ShapeDtypeStruct((batch,), jnp.int32),
        jax.ShapeDtypeStruct((batch, pages), jnp.int32),
        jax.ShapeDtypeStruct((batch,), jnp.int32),
    )

    def body(*args):
        weights = {n: args[i][0] for i, n in enumerate(names)}
        paged.decode_paged(
            weights,
            args[count][0],
            args[count + 1][0],
            args[count + 2 : count + 26],
            *args[count + 26 :],
            batch_size=batch,
            layers=93,
            dcp_pool_map=tuple(range(24)),
            index_cache=(
                jnp.zeros(
                    (4, 24, *dsa_cache.index_block_shape(cache.tokens_per_block)),
                    jnp.uint8,
                )
                if (k31 if index_cache is None else index_cache)
                else None
            ),
            **decode_options(batch),
            **({"index_debug": True} if index_debug else {}),
            **(options or {}),
        )
        return ()

    step = jax.shard_map(
        body,
        mesh=Mesh(np.asarray(jax.devices()[:1]), ("tp",)),
        in_specs=(P("tp"),) * (count + 26) + (P(),) * 4,
        out_specs=(),
        check_vma=False,
    )
    return step, operands


def decode_vmem_plans(
    monkeypatch,
    batch,
    cache_dtype,
    k31,
    *,
    pages=2,
    index_debug=False,
    widths=None,
    index_cache=None,
):
    """Every static VMEM plan the decode stack checks while
    :func:`decode_step` (same arguments) traces, stopped at the stack's
    ``pallas_call``: ``(plans, layouts, calls)``: plans ``[(persistent,
    {region: (held, [nested])})]``, entries ``(name, shape, dtype, bytes)``;
    the K3.1 indexer layouts the trace placed (``_dsa_layout``); the stack
    calls' ``(operands, outputs)``."""
    from jax.experimental import pallas as pl

    from vllm_torchtpu.megakernel.kimi_k3.kernels import decode_megakernel as kernel

    step, operands = decode_step(
        batch,
        cache_dtype,
        k31,
        pages=pages,
        index_debug=index_debug,
        widths=widths,
        index_cache=index_cache,
    )

    def entries(specs):
        return [
            (
                name,
                tuple(getattr(spec, "shape", ())),
                str(getattr(spec, "dtype", "")),
                kernel._vmem_bytes(spec),
            )
            for name, spec in specs
        ]

    plans, layouts, calls = [], [], []
    budget, place, pallas_call = kernel._vmem_budget, kernel._dsa_layout, pl.pallas_call

    def record(persistent, regions, **options):
        plans.append(
            (
                entries(persistent),
                {
                    region: (entries(held), [entries(n) for n in nested])
                    for region, (held, nested) in regions.items()
                },
            )
        )
        return budget(persistent, regions, **options)

    def record_layout(*args, **options):
        layouts.append(place(*args, **options))
        return layouts[-1]

    def stop(*args, **options):
        if options.get("name") == "kimi_decoder_stack":
            calls.append(
                (len(options["grid_spec"].in_specs), len(options["out_shape"]))
            )
            raise _Planned
        return pallas_call(*args, **options)

    monkeypatch.setattr(kernel, "_vmem_budget", record)
    monkeypatch.setattr(kernel, "_dsa_layout", record_layout)
    monkeypatch.setattr(pl, "pallas_call", stop)
    with pytest.raises(_Planned):
        jax.eval_shape(step, *operands)
    monkeypatch.undo()
    return plans, layouts, calls


@pytest.mark.filterwarnings("ignore:The decoder's VMEM plan leaves less")
@pytest.mark.parametrize("index_debug", [False, True])
@pytest.mark.parametrize("cache_dtype", ["auto", "fp8"])
@pytest.mark.parametrize("batch", BATCH_CLASSES)
def test_k31_decode_stack_body_traces_at_every_batch_class(
    monkeypatch, batch, cache_dtype, index_debug
):
    """K3.1's decode stack kernel body traces (abstract shapes, one CPU device,
    no Mosaic lowering) at every batch class it is built for."""
    from jax.experimental import pallas as pl

    step, operands = decode_step(batch, cache_dtype, True, index_debug=index_debug)
    traced, pallas_call = [], pl.pallas_call

    def counting(kernel, *args, **options):
        if options.get("name") == "kimi_decoder_stack":
            inner = kernel

            def kernel(*refs):
                traced.append(batch)
                return inner(*refs)

        return pallas_call(kernel, *args, **options)

    monkeypatch.setattr(pl, "pallas_call", counting)
    jax.eval_shape(step, *operands)
    assert traced == [batch]


def _dp2_stack_traces(
    batches, cache_dtype="fp8", index_debug=False, sequence_rows=1, pages=None
):
    """Trace K3.1's decode stack body at attention DP2 x DCP16 for each
    batch, with ``sequence_rows`` > 1 a verify of that many rows a sequence
    and ``pages`` table pages a row (default ``decode_step``'s), and return
    each batch's select rows a pass (run in a fresh process: kernel modules
    read the kernel config at import)."""
    import warnings

    from jax.experimental import pallas as pl

    from vllm_torchtpu.megakernel.kimi_k3 import config as kernel_config

    kernel_config.install(kernel_config.KernelConfig(attention_dp=2, dcp_group=16))
    from vllm_torchtpu.megakernel.kimi_k3.kernels import decode_megakernel, dsa_decode

    warnings.filterwarnings("ignore", "The decoder's VMEM plan leaves less")
    pallas_call = pl.pallas_call
    layout = decode_megakernel._dsa_layout
    passes = {}
    for batch in batches:
        options = {"sequence_rows": sequence_rows} if sequence_rows > 1 else None
        step, operands = decode_step(
            batch,
            cache_dtype,
            True,
            index_debug=index_debug,
            attention_dp=2,
            options=options,
            **({} if pages is None else {"pages": pages}),
        )

        def recording(*args, batch=batch, **kwargs):
            passes[batch] = dsa_decode.select_rows(
                kwargs["batch_size"], kwargs["local_tokens"]
            )
            return layout(*args, **kwargs)

        traced = []

        def counting(kernel, *args, batch=batch, traced=traced, **options):
            if options.get("name") == "kimi_decoder_stack":
                inner = kernel

                def kernel(*refs):
                    traced.append(batch)
                    return inner(*refs)

            return pallas_call(kernel, *args, **options)

        pl.pallas_call = counting
        decode_megakernel._dsa_layout = recording
        try:
            jax.eval_shape(step, *operands)
        finally:
            pl.pallas_call = pallas_call
            decode_megakernel._dsa_layout = layout
        assert traced == [batch], (batch, traced)
        print(f"B{batch} traced", flush=True)
    return passes


@pytest.mark.parametrize(
    "cache_dtype, index_debug", [("auto", False), ("fp8", False), ("fp8", True)]
)
def test_k31_decode_stack_body_traces_at_attention_dp2(cache_dtype, index_debug):
    """K3.1's decode stack body traces at attention DP2 x DCP16 (held shards'
    index heads, a 16-member scorer and select over the group's rows; with
    ``index_debug`` the check outputs of the group's rows and six heads) at a
    packet-field batch, a chunked one and B64 (32 group rows: FP8 steps of
    2048 tokens read the select in two calls), in a subprocess (the kernel
    config is process-wide)."""
    import os
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[3]
    path = [str(root / "src"), str(root), os.environ.get("PYTHONPATH", "")]
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": os.pathsep.join(path)}
    code = (
        "from tests.megakernel.kimi_k3.test_dsa import _dp2_stack_traces; "
        f"_dp2_stack_traces((4, 16, 64), {cache_dtype!r}, {index_debug!r})"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-3000:]


@pytest.mark.parametrize("sequence_rows, batches", [(4, (8, 64, 128)), (8, (16, 64))])
def test_k31_decode_stack_body_traces_dspark_verify_at_attention_dp2(
    sequence_rows, batches
):
    """K3.1's decode stack body traces a DSpark verify at attention DP2 x
    DCP16 (each verify row its own selection over its group's DCP16 pages,
    earlier verify tokens cached as at DP1) for 3 and 7 drafts, to 64 group
    rows, in a subprocess (the kernel config is process-wide)."""
    import os
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[3]
    path = [str(root / "src"), str(root), os.environ.get("PYTHONPATH", "")]
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": os.pathsep.join(path)}
    code = (
        "from tests.megakernel.kimi_k3.test_dsa import _dp2_stack_traces; "
        f"_dp2_stack_traces({batches!r}, 'fp8', False, {sequence_rows})"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-3000:]


def test_a_decoder_attention_refuses_host_combine_buffers_it_was_not_handed(
    monkeypatch,
):
    """KernelConfig.dcp_host_combine: an attention whose shape takes the
    two-stage combine must be handed the stack's buffers, and one that does not
    take it must not be, or the switch would be silently ignored."""
    from vllm_torchtpu.megakernel.kimi_k3.kernels import decode_megakernel, mla_dcp

    attention = SimpleNamespace(host_combine=None)
    bind = decode_megakernel._bind_host_combine
    monkeypatch.setattr(mla_dcp, "host_combine_scratch", lambda *a, **k: (1, 2, 3))
    with pytest.raises(ValueError, match="dcp_host_combine"):
        bind(attention, None, 32, 6, jnp.float8_e4m3fn, 4)
    bind(attention, ("recv", "send", "sems"), 32, 6, jnp.float8_e4m3fn, 4)
    assert attention.host_combine == ("recv", "send", "sems")
    monkeypatch.setattr(mla_dcp, "host_combine_scratch", lambda *a, **k: ())
    with pytest.raises(ValueError, match="dcp_host_combine"):
        bind(attention, ("recv", "send", "sems"), 32, 6, jnp.float8_e4m3fn, 4)


def _dp2_host_combine_traces(batches, sequence_rows):
    """Trace the DP2 stack with the two-stage combine on, and assert each
    decoder attention runs with its buffers (a fresh process: the kernel config
    is read at import)."""
    import warnings

    from vllm_torchtpu.megakernel.kimi_k3 import config as kernel_config

    kernel_config.install(
        kernel_config.KernelConfig(attention_dp=2, dcp_group=16, dcp_host_combine=True)
    )
    from vllm_torchtpu.megakernel.kimi_k3.kernels import mla_dcp

    warnings.filterwarnings("ignore", "The decoder's VMEM plan leaves less")
    attend = mla_dcp.DcpAttention.attend
    seen = []

    def recording(self, *args, **kwargs):
        seen.append(self.host_combine is not None)
        return attend(self, *args, **kwargs)

    mla_dcp.DcpAttention.attend = recording
    for batch in batches:
        options = {"sequence_rows": sequence_rows} if sequence_rows > 1 else None
        step, operands = decode_step(
            batch, "fp8", True, attention_dp=2, options=options
        )
        seen.clear()
        jax.eval_shape(step, *operands)
        assert seen and all(seen), (batch, seen)
        print(f"B{batch} traced", flush=True)


@pytest.mark.parametrize("sequence_rows, batches", [(1, (32, 64)), (4, (64, 128))])
def test_k31_decode_stack_runs_the_host_combine_at_attention_dp2(
    sequence_rows, batches
):
    """dcp_host_combine at attention DP2 x DCP16 reaches each decoder attention
    of the chunked shapes, plain and with verify rows, in a subprocess (the
    kernel config is process-wide)."""
    import os
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[3]
    path = [str(root / "src"), str(root), os.environ.get("PYTHONPATH", "")]
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": os.pathsep.join(path)}
    code = (
        "from tests.megakernel.kimi_k3.test_dsa import _dp2_host_combine_traces; "
        f"_dp2_host_combine_traces({batches!r}, {sequence_rows})"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-3000:]


def test_k31_decode_stack_body_traces_two_pass_verify_at_attention_dp2():
    """B32 with 3 drafts at attention DP2 (64 group rows) over 88 pages a row
    (45056 local tokens a DCP16 member): its rows exceed the row-token budget,
    so the indexer scores and selects them in two passes of 32, each marking
    bit words, and the views fit the attention allocation; in a subprocess
    (the kernel config is process-wide)."""
    import os
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[3]
    path = [str(root / "src"), str(root), os.environ.get("PYTHONPATH", "")]
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": os.pathsep.join(path)}
    code = (
        "from tests.megakernel.kimi_k3.test_dsa import _dp2_stack_traces; "
        "passes = _dp2_stack_traces((128,), 'fp8', False, 4, pages=88); "
        "assert passes == {128: 32}, passes"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-3000:]


# Kimi-K3's own plans warn of a thin spill reserve at some batches.
@pytest.mark.filterwarnings("ignore:The decoder's VMEM plan leaves less")
@pytest.mark.parametrize("cache_dtype", ["auto", "fp8"])
@pytest.mark.parametrize("batch", BATCH_CLASSES)
def test_index_keys_keep_k3s_decode_vmem_plan(monkeypatch, batch, cache_dtype):
    """K3.1's indexer weights and buffers overlay the projection/expert pool:
    every scoped region's VMEM is Kimi-K3's; the persistent key-value
    normalization slot (m_knorm, now with the key LayerNorm and the index
    head's weights_proj row) grows, the select's inboxes (and at chunked
    batches the index fields' inbox) are added and the pool shrinks by at
    most their bytes, the MLA region adds the indexer's and the select's SMEM
    and semaphores (no VMEM), and the stack reads one more operand (the
    index cache)."""
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_decode, dsa_select, mla_dcp

    k3, _, k3_calls = decode_vmem_plans(monkeypatch, batch, cache_dtype, k31=False)
    k31, layouts, k31_calls = decode_vmem_plans(
        monkeypatch, batch, cache_dtype, k31=True
    )
    new = (
        *dsa_select.persistent_names(dsa_select.SELECT_FORMS[batch], order_rows=batch),
        *(dsa_decode.FIELD_NAMES if mla_dcp.chunked(batch) else ()),
    )
    assert len(k31) == len(k3) >= 1 and len(layouts) >= 1
    assert k31_calls == [(k3_calls[0][0] + 1, k3_calls[0][1])]
    for (persistent, regions), (k3_persistent, k3_regions) in zip(k31, k3, strict=True):
        added = [e for e in persistent if e[0] in new]
        kept = [e for e in persistent if e[0] not in new]
        assert tuple(e[0] for e in added) == new
        assert [e[0] for e in kept] == [e[0] for e in k3_persistent]
        grown = {
            a[0]: (b, a) for a, b in zip(kept, k3_persistent, strict=True) if a != b
        }
        k3_pool, pool = grown.pop("projection_expert_pool", (None, None))
        assert {name: (b[1], a[1]) for name, (b, a) in grown.items()} == {
            "m_knorm": ((2, 1, 512), (2, 1, 7936))
        }
        if pool is not None:  # the managed arena's growth gives up the new bytes
            assert 0 < k3_pool[3] - pool[3] <= sum(e[3] for e in added)
        assert regions.keys() == k3_regions.keys()
        for region, (held, nested) in regions.items():
            k3_held, k3_nested = k3_regions[region]
            scoped = [entry for entry in held if entry not in k3_held]
            assert [entry for entry in held if entry in k3_held] == k3_held
            assert nested == k3_nested
            names = (
                set(dsa_decode.SCOPED_NAMES) | set(dsa_select.SCOPED_NAMES)
                if region == "MLA"
                else set()
            )
            assert {entry[0] for entry in scoped} == names
            assert all(entry[3] == 0 for entry in scoped)


@pytest.mark.filterwarnings("ignore:The decoder's VMEM plan leaves less")
@pytest.mark.parametrize("cache_dtype", ["auto", "fp8"])
@pytest.mark.parametrize("batch", BATCH_CLASSES)
def test_k31_indexer_views_fit_beside_the_mla_weights_to_1m(
    monkeypatch, batch, cache_dtype
):
    """At a table of 1M tokens a row, the indexer's views stay in pool space an
    MLA layer leaves free, in order and never overlapping: the index-query
    head and the ring in region B after the up/gate projections; the scorer's
    buffers, the select's and the check's q-down latent, absorbed queries and
    merged outputs over the consumed input projections after the output and
    index-key projections. With index_debug the stack returns the nine check
    outputs."""
    from vllm_torchtpu.megakernel.kimi_k3 import layout

    logical = layout.cache_layout(cache_dtype).tokens_per_block * 32
    pages = -(-(1 << 20) // logical)
    _, layouts, calls = decode_vmem_plans(
        monkeypatch, batch, cache_dtype, k31=True, pages=pages, index_debug=True
    )
    assert len(layouts) >= 1 and calls[0][1] == 7 + 10
    regions = (
        ("region_b", ("query_head", "ring")),
        (
            "region_a",
            (
                "queries",
                "weights",
                "fields",
                "scores",
                "new_scores",
                "mask",
                "bounds",
                "state",
                "histogram",
                "query_latent",
                "absorbed",
                "merged",
            ),
        ),
    )
    for placed in layouts:
        for region, names in regions:
            start, end = placed[region]
            offsets = [placed[name] for name in names] + [placed[f"{region}_end"]]
            assert offsets[0] == start and offsets == sorted(set(offsets))
            assert offsets[-1] <= end


def test_k31_stack_refuses_tables_over_128_local_pages(monkeypatch):
    """The select's tie packet holds at most 128 local pages a row: a wider
    table is refused before the stack plans its VMEM."""
    with pytest.raises(ValueError, match="128"):
        decode_vmem_plans(monkeypatch, 8, "auto", k31=True, pages=129)


@pytest.mark.parametrize("family, width", [("m_qb", 768), ("m_knorm", 768)])
def test_k31_stack_refuses_narrow_index_head_families(monkeypatch, family, width):
    """K3.1's index-key projection without the index-query head in m_qb or the
    widened m_knorm row is refused before the stack plans its VMEM."""
    with pytest.raises(
        ValueError, match="m_qb carries each held shard's index-query head"
    ):
        decode_vmem_plans(monkeypatch, 8, "auto", k31=True, widths={family: width})


def test_k3_stack_refuses_an_index_cache(monkeypatch):
    """An index cache beside Kimi-K3's weights (no index-key projection) is
    refused before the stack plans its VMEM, not silently ignored."""
    with pytest.raises(ValueError, match="index_cache is Kimi-K3.1's"):
        decode_vmem_plans(monkeypatch, 8, "auto", k31=False, index_cache=True)


def test_k31_stack_refuses_a_streamed_m_qb(monkeypatch):
    from vllm_torchtpu.megakernel.kimi_k3.kernels import decode_megakernel

    monkeypatch.setattr(decode_megakernel, "WEIGHT_STREAMS", {"m_qb": (2, 2)})
    with pytest.raises(NotImplementedError, match="unstreamed m_qb"):
        decode_vmem_plans(monkeypatch, 8, "auto", k31=True)


def tool_engine_config():
    return tool_k3.engine_config(
        "decode", max_num_seqs=8, max_model_len=8192, max_num_batched_tokens=512
    )


def test_tool_flags_configure_the_decoder(tmp_path, monkeypatch):
    monkeypatch.setattr(tool_k3, "DSA", None)
    argv = ["--dsa-config", str(tmp_path), "--synthetic-indexer", "rope-probe"]
    args = tool_main.parser().parse_args(
        [*argv, "--indexer-seed", "3", "check", "dsa-keys", "--model", "m"]
    )
    assert (args.dsa_config, args.synthetic_indexer, args.indexer_seed) == (
        str(tmp_path),
        "rope-probe",
        3,
    )
    assert args.check == "dsa-keys"
    # No --dsa-config: Kimi-K3, nothing for the decoder.
    tool_k3.configure_dsa("", None, 0)
    assert tool_k3.DSA is None and tool_k3.decode_ops_options() == {}
    assert tool_engine_config().dsa is None

    (tmp_path / "config.json").write_text(json.dumps({"text_config": text_config()}))
    tool_k3.configure_dsa(args.dsa_config, args.synthetic_indexer, args.indexer_seed)
    config = dsa.DsaConfig(2048, 1e-5)
    assert tool_k3.decode_ops_options() == {
        "dsa": config,
        "synthetic_indexer": ("rope-probe", 3),
    }
    assert tool_engine_config().dsa == config
    # --dsa-topk replaces the config's index_topk.
    args = tool_main.parser().parse_args(
        [*argv, "--dsa-topk", "512", "check", "dsa-keys", "--model", "m"]
    )
    tool_k3.configure_dsa(args.dsa_config, args.synthetic_indexer, 0, args.dsa_topk)
    assert tool_engine_config().dsa == dsa.DsaConfig(512, 1e-5)


def test_tool_dsa_config_never_runs_silently_as_k3(tmp_path, monkeypatch):
    monkeypatch.setattr(tool_k3, "DSA", None)
    k31 = tmp_path / "k31"
    k31.mkdir()
    (k31 / "config.json").write_text(json.dumps({"text_config": text_config()}))
    k3_dir = tmp_path / "k3"
    k3_dir.mkdir()
    (k3_dir / "config.json").write_text(
        json.dumps({"text_config": {"rms_norm_eps": 1e-5}})
    )
    # Without --synthetic-indexer K3.1 takes the pre-shard's own indexer weights.
    tool_k3.configure_dsa(str(k31), None, 0)
    assert tool_k3.DSA[1] is None and tool_k3.indexer_mode() is None
    assert tool_k3.decode_ops_options()["synthetic_indexer"] is None
    tool_k3.DSA = None
    with pytest.raises(SystemExit, match="dsa_config"):
        tool_k3.configure_dsa(str(k3_dir), "random", 0)
    with pytest.raises(SystemExit, match="needs --dsa-config"):
        tool_k3.configure_dsa("", "random", 0)
    with pytest.raises(SystemExit, match="--dsa-topk needs --dsa-config"):
        tool_k3.configure_dsa("", None, 0, 512)
    assert tool_k3.DSA is None and tool_k3.decode_ops_options() == {}


def test_check_batch_fills_the_rows_with_distinct_prompts():
    for check, extra in (("greedy", ["--output", "o"]), ("dsa-keys", [])):
        argv = ["check", check, "--model", "m", *extra]
        assert tool_main.parser().parse_args(argv).batch == 8
        assert tool_main.parser().parse_args([*argv, "--batch", "64"]).batch == 64
    ids = list(range(5000))
    five = [np.asarray(ids)] + [np.full(12, n) for n in range(4)]
    assert tool_check.batch_prompts(five, ids, 8) == five  # B8: the five prompts
    assert tool_check.batch_prompts(five, ids, 1) == five[:1]
    filled = tool_check.batch_prompts(five[:1], ids, 4, fill_above=1)  # dsa-scores
    assert len(filled) == 4 and len({tuple(row.tolist()) for row in filled}) == 4
    rows = tool_check.batch_prompts(five, ids, 64)
    assert len(rows) == 64 and rows[:5] == five
    assert len({tuple(row.tolist()) for row in rows}) == 64
    assert min(len(row) for row in rows[5:]) >= 32
    with pytest.raises(SystemExit):
        tool_check.batch_prompts(five, ids[:1000], 64)


def committed_rope_probe_keys(seed=0, blocks=3, page=256):
    """A layer's latent pool and its pages of the index cache as a rope-probe
    decode commits them: random BF16 rope lanes, and the index bytes and
    exponents of the keys they imply."""
    import torch

    rope = np.random.default_rng(seed).standard_normal((blocks, page, 64))
    latent = torch.zeros((blocks, page // 2, 2, 640), dtype=torch.bfloat16)
    latent[..., 512:576] = torch.from_numpy(rope.astype(np.float32)).reshape(
        blocks, page // 2, 2, 64
    )
    rows = latent[..., 512:576].reshape(blocks, page, 64)
    projected = jnp.asarray(torch.cat((rows, -rows), dim=-1).float().numpy())
    key = ref.key_from_projection(
        projected.astype(jnp.bfloat16), jnp.ones(128), jnp.zeros(128), 1e-5
    )
    q, e = dsa_cache.quantize_index_keys(key)
    index = np.zeros((blocks, *dsa_cache.index_block_shape(page)), np.uint8)
    index[:, :page] = np.asarray(q)
    index[:, page : page + page // 128] = np.asarray(e).reshape(blocks, -1, 128)
    return latent, index


def test_dsa_keys_check_tells_a_wrong_key_from_a_rounding_step():
    page = 256
    latent, index = committed_rope_probe_keys(page=page)
    blocks, slots = np.asarray([1, 2, 1, 2]), np.asarray([5, 7, 200, 255])

    def check(index):
        return tool_check.index_key_mismatches(latent, index, blocks, slots, page, 1e-5)

    assert check(index) == (0, 0, 0.0)
    # One FP8 code off in one lane: a difference, not a wrong key.
    step = index.copy()
    lane = int(np.flatnonzero((step[2, 7] & 0x7F) < 0x70)[0])
    step[2, 7, lane] += 1
    differing, beyond, worst = check(step)
    assert (differing, beyond) == (1, 0) and worst > 0
    # A key never committed (zero), a foreign key and a wrong scale exponent.
    for wrong in range(3):
        bad = index.copy()
        if wrong == 0:
            bad[1, 5] = 0
        elif wrong == 1:
            bad[1, 200] = index[1, 201]
        else:
            bad[2, page + 255 // 128, 255 % 128] += 1
        assert check(bad)[:2] == (1, 1), wrong
    # Not finite: a NaN code (0x7F) in one lane, an infinite scale (0xFF).
    nan = index.copy()
    nan[1, 200, 3] = 0x7F
    differing, beyond, worst = check(nan)
    assert (differing, beyond) == (1, 1) and np.isnan(worst)
    infinite = index.copy()
    infinite[1, page + 5 // 128, 5 % 128] = 0xFF
    differing, beyond, worst = check(infinite)
    assert (differing, beyond) == (1, 1) and not np.isfinite(worst)


@pytest.mark.parametrize("rows", [1, 2, 3, 8, 16])
def test_score_buffer_places_each_rows_tokens_in_the_select_layout(rows):
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_decode

    planes, vrows, lanes = dsa_decode.score_shape(1536, rows)
    _, parts = dsa_decode.virtual_rows(rows)
    part = planes * 128
    assert parts * part >= 1536 and lanes == 128 and vrows % 8 == 0
    if rows < 8:
        assert vrows == 8 and parts == 8 // (1 << (rows - 1).bit_length())
    else:
        assert vrows >= rows and parts == 1
    buffer = np.full((planes, vrows, lanes), np.nan, np.float32)
    values = np.arange(rows * parts * part, dtype=np.float32).reshape(rows, -1)
    for r in range(rows):  # the placement IndexScorer._store writes
        for t in range(0, parts * part, 128):
            buffer[t % part // 128, r * parts + t // part] = values[r, t : t + 128]
    np.testing.assert_array_equal(dsa_decode.scores_by_row(buffer, rows), values)


def test_steps_score_once_a_rows_context_exceeds_the_top_k():
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_decode

    assert not bool(dsa_decode.scores_needed(jnp.asarray([5, 2047, 0]), 3))
    assert bool(dsa_decode.scores_needed(jnp.asarray([5, 2048, 0]), 3))


def test_row_tokens_are_the_ranks_pages_below_the_position():
    page, group = 4, 3
    for rank in range(group):
        for position in range(40):
            cached, new = tool_check.row_tokens(position, rank, page, group)
            want = [
                j * page + o
                for j in range(10)
                for o in range(page)
                if (j * group + rank) * page + o < position
            ]
            assert cached.tolist() == want
            owner = position // page % group == rank
            assert (new is not None) == owner
            if owner:
                assert new == position // page // group * page + position % page


@pytest.mark.parametrize("page_tokens", [256, 512])
def test_local_length_is_each_ranks_count_of_the_first_position_plus_one(page_tokens):
    """dsa_decode.local_length against dsa_micro.local_counts(position + 1) over
    32 ranks, every page edge and a stride, also traced."""
    from tools.megakernel import dsa_micro
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_decode

    span = 32 * page_tokens
    edges = {span * k + d for k in range(1, 4) for d in (-1, 0, 1)}
    positions = sorted({*range(0, 3 * span, 37), *edges})
    traced = jax.jit(dsa_decode.local_length, static_argnums=(2, 3))
    for position in positions:
        want = dsa_micro.local_counts(position + 1, 4 * page_tokens, page_tokens)
        got = [
            int(dsa_decode.local_length(position, r, page_tokens, 32))
            for r in range(32)
        ]
        assert got == want.tolist(), position
        assert (
            int(traced(jnp.int32(position), jnp.int32(5), page_tokens, 32)) == want[5]
        )


def test_masked_tokens_are_the_contracts_minus_inf_tail():
    """check.masked_tokens is [L, L rounded up to 128) with L the row's cached
    local tokens plus its new token on the owner."""
    page, group = 256, 4
    for rank in range(group):
        for position in range(0, 6 * group * page, 97):
            cached, new = tool_check.row_tokens(position, rank, page, group)
            length = len(cached) + (new is not None)
            want = np.arange(length, -(-length // 128) * 128)
            got = tool_check.masked_tokens(position, rank, page, group)
            np.testing.assert_array_equal(got, want)


def test_key_step_mismatch_tells_a_step_from_a_wrong_key():
    key = jnp.asarray(np.random.default_rng(2).standard_normal((1, 128)), jnp.bfloat16)
    want, exponent = (np.asarray(a)[0] for a in dsa_cache.quantize_index_keys(key))
    check = tool_check.key_step_mismatch
    assert check(want, exponent, want, exponent) == (False, False)
    step = want.copy()
    lane = int(np.flatnonzero((step & 0x7F) < 0x70)[0])
    step[lane] += 1
    assert check(step, exponent, want, exponent) == (True, False)
    assert check(np.zeros_like(want), exponent, want, exponent) == (True, True)


def finite_e4m3_codes():
    codes = np.arange(256, dtype=np.uint8)
    return codes[(codes & 0x7F) != 0x7F]


def test_e4m3_values_decode_every_finite_code():
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_decode

    codes = finite_e4m3_codes()
    got = np.asarray(dsa_decode.e4m3_values(jnp.asarray(codes, jnp.int32)))
    want = codes.view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    np.testing.assert_array_equal(got.view(np.uint32), want.view(np.uint32))


def test_query_fields_are_the_reference_indexer_query():
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_decode

    rng = jax.random.split(jax.random.PRNGKey(8), 4)
    q_c = jax.random.normal(rng[0], (5, 1536)).astype(jnp.bfloat16)
    wq = (jax.random.normal(rng[1], (1536, 128)) / 40).astype(jnp.bfloat16)
    wq = jnp.tile(wq, 32)  # one head, 32 times
    x = jax.random.normal(rng[2], (5, 7168)).astype(jnp.bfloat16)
    wv = jnp.tile((jax.random.normal(rng[3], (7168, 1)) / 85).astype(jnp.bfloat16), 32)
    highest = jax.lax.Precision.HIGHEST
    # The reference's own projections, head 0's columns.
    q = jnp.dot(q_c, wq, precision=highest, preferred_element_type=jnp.float32)
    v = jnp.dot(x, wv, precision=highest, preferred_element_type=jnp.float32)
    q_head = q[:, :128].astype(jnp.bfloat16)
    values, v, exponent = dsa_decode.query_fields(q_head, v[:, :1])
    want_q, want_w = ref.index_query(q_c, wq, x, wv)
    np.testing.assert_array_equal(np.asarray(values), np.asarray(want_q)[:, 0])
    np.testing.assert_array_equal(
        np.asarray(dsa_decode.head_weight(v, exponent))[:, 0], np.asarray(want_w)[:, 0]
    )


def test_row_dot_is_the_rows_float32_product():
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_decode

    rng = np.random.default_rng(10)
    x = jnp.asarray(rng.standard_normal((3, 7168)), jnp.bfloat16)
    row = jnp.asarray(rng.standard_normal((1, 7168)), jnp.bfloat16)
    want = np.asarray(x, np.float64) @ np.asarray(row, np.float64).T
    np.testing.assert_allclose(
        np.asarray(dsa_decode.row_dot(x, row)), want, rtol=1e-5, atol=1e-3
    )


@pytest.mark.parametrize("uint8_packets", [False, True])
def test_index_fields_round_trip_through_the_wire_lanes(uint8_packets):
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_decode

    rng = np.random.default_rng(9)
    picks = rng.choice(finite_e4m3_codes(), (6, 128))
    values = jnp.asarray(picks.view(ml_dtypes.float8_e4m3fn).astype(np.float32))
    v = jnp.asarray(rng.standard_normal((6, 1)).astype(BF16).astype(np.float32))
    exponent = jnp.asarray(rng.integers(1, 255, (6, 1)), jnp.int32)
    if uint8_packets:
        encode, read = dsa_decode.fp8_fields, dsa_decode.read_fp8_fields
        first = dsa_decode.FP8_FIELD_LANE
    else:
        encode, read = dsa_decode.bf16_fields, dsa_decode.read_bf16_fields
        first = dsa_decode.BF16_FIELD_LANE
    fields = encode(values, v, exponent, roll=jnp.roll)
    if uint8_packets:
        assert all(0 <= int(np.min(f)) <= int(np.max(f)) < 256 for f in fields)
    else:  # the BF16 wire carries them exactly
        fields = tuple(f.astype(jnp.bfloat16).astype(jnp.float32) for f in fields)
    for field in fields:
        assert not np.asarray(field)[:, :first].any()
    got_values, got_weight = read(*fields, roll=jnp.roll)
    np.testing.assert_array_equal(
        np.asarray(got_values).view(np.uint32), np.asarray(values).view(np.uint32)
    )
    np.testing.assert_array_equal(
        np.asarray(got_weight), np.asarray(dsa_decode.head_weight(v, exponent))
    )


def test_field_packet_round_trips_values_and_weight():
    """The fields' own all-gather packet: FP8 query values (as BF16 pairs),
    v and the exponent come back exactly; lanes past the exponent stay zero."""
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_decode

    rng = np.random.default_rng(12)
    queries = jnp.asarray(rng.standard_normal((16, 128)) * 5, jnp.float32)
    values, _ = dsa_cache.quantize_index_queries(queries)
    v = jnp.asarray(rng.standard_normal((16, 1)).astype(BF16).astype(np.float32))
    exponent = jnp.asarray(rng.integers(1, 255, (16, 1)), jnp.int32)
    packet = np.asarray(dsa_decode.field_packet(values, v, exponent, roll=jnp.roll))
    assert packet.shape == (16, 128) and not packet[:, 66:].any()
    got_values, got_weight = dsa_decode.read_field_packet(
        jnp.asarray(packet), roll=jnp.roll
    )
    np.testing.assert_array_equal(
        np.asarray(got_values).view(np.uint32), np.asarray(values).view(np.uint32)
    )
    np.testing.assert_array_equal(
        np.asarray(got_weight), np.asarray(dsa_decode.head_weight(v, exponent))
    )


@pytest.mark.parametrize("races", [False, True])
@pytest.mark.parametrize("rows", [16, 32])
def test_index_fields_all_gather_on_8_interpreted_ranks(rows, races):
    """At chunked batches IndexScorer.set_fields, send_fields and
    receive_fields (Pallas interpret mode, 8 ranks) give every rank every
    member's query values and head weight of every row, member order."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_decode

    group = 8
    rng = np.random.default_rng(rows)
    q_projection = jnp.asarray(rng.standard_normal((group * rows, 128)), jnp.bfloat16)
    v_projection = jnp.asarray(rng.standard_normal((group * rows, 1)), jnp.float32)
    values, v, exponent = dsa_decode.query_fields(q_projection, v_projection)

    def kernel(values_ref, v_ref, e_ref, q_ref, w_ref, inbox, semaphores):
        scratch = dsa_decode.IndexScratch(
            *(None,) * 3, q_ref, w_ref, *(None,) * 7,
            field_inbox=inbox, field_semaphores=semaphores,
        )  # fmt: skip
        scorer = dsa_decode.IndexScorer(
            scratch,
            position_ref=None,
            batch=rows,
            active=e_ref[0, 0] >= 0,
            table_ref=None,
            pages=1,
            layer=0,
            page_tokens=256,
            group=group,
            global_pages=False,
            owner=0,
            slots=3,
            uint8_packets=False,
            new_value_ref=None,
            member=jax.lax.axis_index("tp"),
            base=0,
        )
        assert not scorer.packet_fields
        scorer.set_fields(values_ref[...], v_ref[...], e_ref[...])
        scorer.send_fields()
        scorer.receive_fields()

    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    call = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((rows, group, 128), jnp.bfloat16),
            jax.ShapeDtypeStruct((rows, group, 128), jnp.float32),
        ),
        in_specs=[vmem] * 3,
        out_specs=[vmem] * 2,
        scratch_shapes=dsa_decode.field_persistent_scratch(rows, group),
        interpret=pltpu.InterpretParams(
            dma_execution_mode="eager" if races else "on_wait", detect_races=races
        ),
    )
    mesh = Mesh(np.array(jax.devices("cpu")[:group]), ("tp",))
    queries, weights = jax.shard_map(
        call, mesh=mesh, in_specs=(P("tp"),) * 3, out_specs=(P("tp"),) * 2,
        check_vma=False,
    )(values, v, exponent)  # fmt: skip
    if races:
        from jax._src.pallas.mosaic.interpret import interpret_pallas_call

        assert not interpret_pallas_call.races.races_found
    want_q = np.asarray(values).reshape(group, rows, 128).transpose(1, 0, 2)
    want_w = np.asarray(dsa_decode.head_weight(v, exponent)).reshape(group, rows)
    for member in range(group):
        np.testing.assert_array_equal(
            np.asarray(queries, np.float32)[member * rows : (member + 1) * rows], want_q
        )
        np.testing.assert_array_equal(
            np.asarray(weights)[member * rows : (member + 1) * rows],
            np.broadcast_to(want_w.T[:, :, None], (rows, group, 128)),
        )


@pytest.mark.parametrize("uint8_packets", [False, True])
@pytest.mark.parametrize("rows, slots", [(1, 3), (2, 4), (8, 3)])
def test_index_fields_ride_the_query_packet(uint8_packets, rows, slots):
    """IndexScorer.set_fields, send_packet, take_fields and clean in Pallas
    interpret mode on CPU (pltpu.roll as in the kernel): each row's query
    values and head weight come back exactly, the packet's latent-query lanes
    and scale are Kimi-K3's, and the receiver's clean zeroes the BF16 field
    lanes; take_fields books its work to the index_fields scope."""
    import contextlib

    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_decode, mla_dcp

    names = []

    @contextlib.contextmanager
    def scope(name):
        names.append(name)
        yield

    rng = np.random.default_rng(rows * 10 + slots)
    q_projection = jnp.asarray(rng.standard_normal((rows, 128)) * 3, jnp.bfloat16)
    v_projection = jnp.asarray(rng.standard_normal((rows, 1)), jnp.float32)
    values, v, exponent = dsa_decode.query_fields(q_projection, v_projection)
    own = np.zeros((slots * rows, 640), np.float32)
    own[:, :576] = rng.standard_normal((slots * rows, 576))
    if slots == 4:
        own[3::4] = 0  # the zero pad slot
    packet_dtype = jnp.uint8 if uint8_packets else jnp.bfloat16
    fields_dtype = jnp.int32 if uint8_packets else jnp.float32

    def kernel(own_ref, values_ref, v_ref, e_ref, packet, q_ref, w_ref, clean, f_ref):
        scratch = dsa_decode.IndexScratch(
            None, None, None, q_ref, w_ref, f_ref, None, None, None, None, None, None
        )
        scorer = dsa_decode.IndexScorer(
            scratch,
            position_ref=None,
            batch=rows,
            active=e_ref[0, 0] >= 0,
            table_ref=None,
            pages=1,
            layer=0,
            page_tokens=256,
            group=32,
            global_pages=False,
            owner=0,
            slots=slots,
            uint8_packets=uint8_packets,
            new_value_ref=None,
            scope=scope,
        )
        scorer.set_fields(values_ref[...], v_ref[...], e_ref[...])
        scorer.send_packet(packet.at[0], own_ref[...], 0, rows)
        scorer.take_fields(packet, 0, rows)
        if uint8_packets:
            clean[...] = jnp.zeros(clean.shape, jnp.float32)
        else:
            clean[...] = scorer.clean(packet[0].astype(jnp.float32))

    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    packet, queries, weights, cleaned = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((1, slots * rows, 640), packet_dtype),
            jax.ShapeDtypeStruct((rows, 1, 128), jnp.bfloat16),
            jax.ShapeDtypeStruct((rows, 1, 128), jnp.float32),
            jax.ShapeDtypeStruct((slots * rows, 640), jnp.float32),
        ),
        in_specs=[vmem] * 4,
        out_specs=[vmem] * 4,
        scratch_shapes=[pltpu.VMEM((rows, dsa_decode.FIELD_SLOTS, 128), fields_dtype)],
        interpret=pltpu.InterpretParams(dma_execution_mode="on_wait"),
    )(jnp.asarray(own), values, v, exponent)
    assert "index_fields" in names
    np.testing.assert_array_equal(
        np.asarray(queries, np.float32)[:, 0], np.asarray(values)
    )
    np.testing.assert_array_equal(
        np.asarray(weights)[:, 0],
        np.broadcast_to(np.asarray(dsa_decode.head_weight(v, exponent)), (rows, 128)),
    )
    if uint8_packets:
        got = mla_dcp._unpack_fp8_query(jnp.asarray(packet[0]))
        want = mla_dcp._unpack_fp8_query(mla_dcp._pack_fp8_query(jnp.asarray(own)))
        for a, b in zip(got, want, strict=True):
            np.testing.assert_array_equal(
                np.asarray(a.astype(jnp.float32)), np.asarray(b.astype(jnp.float32))
            )
    else:
        packet = np.asarray(packet[0], np.float32)
        latent = np.asarray(jnp.asarray(own[:, :576]).astype(jnp.bfloat16), np.float32)
        np.testing.assert_array_equal(packet[:, :576], latent)
        np.testing.assert_array_equal(packet[0, 576:640], np.asarray(values)[0, :64])
        cleaned = np.asarray(cleaned)
        assert not cleaned[:, 576:].any()
        np.testing.assert_array_equal(cleaned[:, :576], latent)
        if slots == 4:
            assert not packet[3::4].any()


@pytest.mark.parametrize("rows", [16, 32])
def test_two_index_heads_a_rank_all_gather_on_8_interpreted_ranks(rows):
    """Attention DP2: each rank's two index heads go on the fields'
    all-gather (two heads a packet row) and come back on every rank as
    queries ``k G + m``, head ``k`` of member ``m``."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_decode

    group, heads = 8, 2
    rng = np.random.default_rng(rows + 1)
    q_projection = jnp.asarray(
        rng.standard_normal((heads, group * rows, 128)), jnp.bfloat16
    )
    v_projection = jnp.asarray(
        rng.standard_normal((heads, group * rows, 1)), jnp.float32
    )
    fields = [
        dsa_decode.query_fields(q_projection[k], v_projection[k]) for k in range(heads)
    ]
    values, v, exponent = (jnp.stack([f[i] for f in fields]) for i in range(3))

    def kernel(values_ref, v_ref, e_ref, q_ref, w_ref, inbox, semaphores):
        scratch = dsa_decode.IndexScratch(
            *(None,) * 3, q_ref, w_ref, *(None,) * 7,
            field_inbox=inbox, field_semaphores=semaphores,
        )  # fmt: skip
        scorer = dsa_decode.IndexScorer(
            scratch, position_ref=None, batch=rows, active=e_ref[0, 0, 0] >= 0,
            table_ref=None, pages=1, layer=0, page_tokens=256, group=group,
            global_pages=False, owner=0, slots=6, uint8_packets=False,
            new_value_ref=None, member=jax.lax.axis_index("tp"), base=0, heads=heads,
        )  # fmt: skip
        scorer.set_fields(
            [values_ref[k] for k in range(heads)],
            [v_ref[k] for k in range(heads)],
            [e_ref[k] for k in range(heads)],
        )
        scorer.send_fields()
        scorer.receive_fields()

    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    call = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((rows, heads * group, 128), jnp.bfloat16),
            jax.ShapeDtypeStruct((rows, heads * group, 128), jnp.float32),
        ),
        in_specs=[vmem] * 3,
        out_specs=[vmem] * 2,
        scratch_shapes=dsa_decode.field_persistent_scratch(rows, group, heads),
        interpret=pltpu.InterpretParams(dma_execution_mode="on_wait"),
    )
    mesh = Mesh(np.array(jax.devices("cpu")[:group]), ("tp",))
    queries, weights = jax.shard_map(
        call, mesh=mesh, in_specs=(P(None, "tp"),) * 3, out_specs=(P("tp"),) * 2,
        check_vma=False,
    )(values, v, exponent)  # fmt: skip
    queries = np.asarray(queries, np.float32).reshape(group, rows, heads, group, 128)
    weights = np.asarray(weights).reshape(group, rows, heads, group, 128)
    for k in range(heads):
        want_q = np.asarray(values[k]).reshape(group, rows, 128).transpose(1, 0, 2)
        weight = np.asarray(dsa_decode.head_weight(v[k], exponent[k]))
        want_w = weight.reshape(group, rows).T
        for member in range(group):
            np.testing.assert_array_equal(queries[member, :, k], want_q)
            np.testing.assert_array_equal(
                weights[member, :, k],
                np.broadcast_to(want_w[:, :, None], (rows, group, 128)),
            )


@pytest.mark.parametrize("uint8_packets", [False, True])
@pytest.mark.parametrize("rows", [1, 2])
def test_two_index_heads_ride_a_six_slot_query_packet(uint8_packets, rows):
    """Attention DP2's packets (six head slots): head ``k``'s fields ride
    slots ``3 k .. 3 k + 2`` and take_fields returns both heads exactly."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_decode

    heads, slots = 2, 6
    rng = np.random.default_rng(rows + 7)
    q_projection = jnp.asarray(
        rng.standard_normal((heads, rows, 128)) * 3, jnp.bfloat16
    )
    v_projection = jnp.asarray(rng.standard_normal((heads, rows, 1)), jnp.float32)
    fields = [
        dsa_decode.query_fields(q_projection[k], v_projection[k]) for k in range(heads)
    ]
    values, v, exponent = (jnp.stack([f[i] for f in fields]) for i in range(3))
    own = np.zeros((slots * rows, 640), np.float32)
    own[:, :576] = rng.standard_normal((slots * rows, 576))
    packet_dtype = jnp.uint8 if uint8_packets else jnp.bfloat16
    fields_dtype = jnp.int32 if uint8_packets else jnp.float32

    def kernel(own_ref, values_ref, v_ref, e_ref, packet, q_ref, w_ref, f_ref):
        scratch = dsa_decode.IndexScratch(
            None, None, None, q_ref, w_ref, f_ref, None, None, None, None, None, None
        )
        scorer = dsa_decode.IndexScorer(
            scratch, position_ref=None, batch=rows, active=e_ref[0, 0, 0] >= 0,
            table_ref=None, pages=1, layer=0, page_tokens=256, group=16,
            global_pages=False, owner=0, slots=slots, uint8_packets=uint8_packets,
            new_value_ref=None, heads=heads,
        )  # fmt: skip
        scorer.set_fields(
            [values_ref[k] for k in range(heads)],
            [v_ref[k] for k in range(heads)],
            [e_ref[k] for k in range(heads)],
        )
        scorer.send_packet(packet.at[0], own_ref[...], 0, rows)
        scorer.take_fields(packet, 0, rows)

    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    _, queries, weights = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((1, slots * rows, 640), packet_dtype),
            jax.ShapeDtypeStruct((rows, heads, 128), jnp.bfloat16),
            jax.ShapeDtypeStruct((rows, heads, 128), jnp.float32),
        ),
        in_specs=[vmem] * 4,
        out_specs=[vmem] * 3,
        scratch_shapes=[pltpu.VMEM((rows, dsa_decode.FIELD_SLOTS, 128), fields_dtype)],
        interpret=pltpu.InterpretParams(dma_execution_mode="on_wait"),
    )(jnp.asarray(own), values, v, exponent)
    for k in range(heads):
        np.testing.assert_array_equal(
            np.asarray(queries, np.float32)[:, k], np.asarray(values[k])
        )
        weight = np.asarray(dsa_decode.head_weight(v[k], exponent[k]))
        np.testing.assert_array_equal(
            np.asarray(weights)[:, k], np.broadcast_to(weight, (rows, 128))
        )


def test_query_pair_mismatches_allow_a_step_or_a_moved_scale_not_a_swap():
    rng = np.random.default_rng(11)
    raw = jnp.asarray(rng.standard_normal((2, 32, 128)), jnp.float32)
    q = np.asarray(dsa_cache.quantize_index_queries(raw)[0])
    w = rng.standard_normal((2, 32)).astype(np.float32)
    check = tool_check.query_pair_mismatches
    assert check(q, w, q, w) == (0, 0)
    moved_q, moved_w = q.copy(), w.copy()
    moved_q[0, 5] *= 2
    moved_w[0, 5] /= 2  # a power-of-two scale moved between q and w
    assert check(moved_q, moved_w, q, w) == (1, 0)
    step = q.copy()
    step[1, 3, 7] += np.abs(q[1, 3]).max() / 16
    assert check(step, w, q, w) == (1, 0)
    swap = q.copy()
    swap[0, [1, 2]] = q[0, [2, 1]]
    assert check(swap, w, q, w) == (2, 2)


def test_score_errors_and_index_keys_at():
    rng = np.random.default_rng(12)
    page = 256
    keys = jnp.asarray(rng.standard_normal((2, page, 128)), jnp.bfloat16)
    q, e = (np.asarray(a) for a in dsa_cache.quantize_index_keys(keys))
    index = np.zeros((3, 24, page + 32, 128), np.uint8)
    index[1:, 5, :page] = q
    index[1:, 5, page : page + 2] = e.reshape(2, 2, 128)
    tokens = np.asarray([0, 7, page + 3, 2 * page - 1])
    got = tool_check.index_keys_at(index, np.asarray([1, 2]), 5, tokens, page)
    want = dsa_cache.dequantize_index_keys(jnp.asarray(q), jnp.asarray(e))
    np.testing.assert_array_equal(got, np.asarray(want).reshape(-1, 128)[tokens])
    raw = jnp.asarray(rng.standard_normal((32, 128)), jnp.float32)
    queries = np.asarray(dsa_cache.quantize_index_queries(raw)[0])
    weights = rng.standard_normal(32).astype(np.float32)
    scores = np.asarray(
        ref.scores(jnp.asarray(queries)[None], jnp.asarray(weights)[None], got)
    )[0]
    assert tool_check.score_errors(scores, queries, weights, got) == 0.0
    assert tool_check.score_errors(scores * 1.01, queries, weights, got) > 1e-3
    nan = np.full_like(scores, np.nan)
    assert tool_check.score_errors(nan, queries, weights, got) == float("inf")


@pytest.mark.parametrize("global_pages, pages", [(False, 3), (True, 6), (True, 8)])
@pytest.mark.parametrize("races", [False, True])
@pytest.mark.parametrize("page_tokens", [256, 512])
@pytest.mark.parametrize("rows, chunk", [(1, 1), (2, 1), (3, 3), (8, 8), (16, 8)])
def test_index_scorer_scores_each_rows_own_pages(
    page_tokens, rows, chunk, races, global_pages, pages
):
    """The scorer (Pallas interpret mode on CPU, in calls of ``chunk`` rows
    as the decode layer makes them) against dsa_reference.scores over each
    row's own local pages below its position, plus its new token on the rank
    owning that page; the entries from there to the next 128-token boundary
    -inf (check.masked_tokens), and the rest of a page the new token opens
    unwritten (NaN here; the scores contract leaves it arbitrary). NaN codes
    fill each row's own page past its position, random bytes the unused scale
    rows and the other layers. Tables have ``pages`` entries a row, local or
    global (``global_pages``; widths that are and are not a multiple of the
    group), and the scores hold the most local pages any rank holds. With
    ``races``, DMAs run eagerly under the interpreter's race detector, which
    must find none."""
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call
    from jax.experimental.pallas import tpu as pltpu

    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_decode

    group, owner, layer = 4, 1, 2
    rng = np.random.default_rng(page_tokens + rows)
    span = (pages if global_pages else pages * group) * page_tokens
    special = [
        0,  # a padding row
        page_tokens,  # its new token opens the owner's first page
        owner * page_tokens + 5,
        span - 1,
        (group + owner) * page_tokens + 17,
        page_tokens - 1,  # nothing on this rank, its page on member 0
    ]
    extra = rng.integers(1, span, max(rows - len(special), 0)).tolist()
    positions = np.asarray((special + extra)[:rows], np.int32)
    table = (1 + np.arange(rows * pages, dtype=np.int32)).reshape(rows, pages)
    local = table[:, owner::group] if global_pages else table
    blocks = 1 + rows * pages
    cache = rng.integers(0, 256, (blocks, 4, page_tokens + 32, 128), dtype=np.uint8)
    draw = rng.standard_normal((blocks, page_tokens, 128)) * np.exp2(
        rng.integers(-4, 5, (blocks, page_tokens, 1))
    )
    key_bytes, exponent = (
        np.asarray(a)
        for a in dsa_cache.quantize_index_keys(jnp.asarray(draw, jnp.bfloat16))
    )
    cache[:, layer, :page_tokens] = key_bytes
    rows_of_scales = page_tokens // 128
    cache[:, layer, page_tokens : page_tokens + rows_of_scales] = exponent.reshape(
        blocks, rows_of_scales, 128
    )
    for r, position in enumerate(positions.tolist()):  # stale NaN keys to mask
        if position // page_tokens % group == owner:
            block = local[r, position // page_tokens // group]
            cache[block, layer, position % page_tokens : page_tokens] = 0x7F
    raw = jnp.asarray(rng.standard_normal((rows, 32, 128)), jnp.float32)
    queries = np.asarray(dsa_cache.quantize_index_queries(raw)[0])
    weights = rng.standard_normal((rows, 32)).astype(np.float32)
    new = jnp.asarray(rng.standard_normal((rows, 128)), jnp.bfloat16)
    codes, exponents = dsa_cache.index_key_rows(new)
    new_values = jnp.zeros((rows, 3, 128), jnp.bfloat16)
    new_values = new_values.at[:, 0].set(codes).at[:, 1].set(exponents)
    scores = dsa_decode.score_rows(
        jnp.asarray(cache),
        jnp.asarray(table),
        jnp.asarray(positions),
        jnp.asarray(queries, jnp.bfloat16),
        jnp.asarray(np.broadcast_to(weights[..., None], (rows, 32, 128))),
        new_values,
        layer=layer,
        page_tokens=page_tokens,
        group=group,
        owner=owner,
        chunk=chunk,
        global_pages=global_pages,
        interpret=pltpu.InterpretParams(
            dma_execution_mode="eager" if races else "on_wait",
            detect_races=races,
            uninitialized_memory="nan",
        ),
    )
    held = max(
        dsa_decode.local_pages(span, rank, page_tokens, group) for rank in range(group)
    )
    assert scores.shape == dsa_decode.score_shape(held * page_tokens, rows)
    by_row = dsa_decode.scores_by_row(np.asarray(scores), rows)
    if races:
        assert not interpret_pallas_call.races.races_found
    new_keys = np.asarray(
        dsa_cache.dequantize_index_keys(*dsa_cache.quantize_index_keys(new))
    )
    for r, position in enumerate(positions.tolist()):
        cached, own = tool_check.row_tokens(position, owner, page_tokens, group)
        keys = tool_check.index_keys_at(cache, local[r], layer, cached, page_tokens)
        tokens = cached
        if own is not None:
            tokens = np.append(cached, own)
            keys = np.concatenate((keys, new_keys[r : r + 1]))
        if len(tokens):
            error = tool_check.score_errors(
                by_row[r, tokens], queries[r], weights[r], keys
            )
            assert error <= tool_check.SCORE_TOLERANCE, (r, position, error)
        masked = tool_check.masked_tokens(position, owner, page_tokens, group)
        assert (by_row[r, masked] == -np.inf).all(), (r, position)
        if own is not None and position % page_tokens == 0:  # a page it opens
            assert np.isnan(by_row[r, own + 128 : own + page_tokens]).all(), r


def _scores_and_new_scores(rows, width, passes, group_new_tokens, sort, seed=5):
    """The scorer (Pallas interpret mode on CPU) on ``rows // width`` verify
    sequences of ``width`` rows at consecutive positions, each sharing one
    table and its first row streaming the pages below its last row, scored in
    ``passes`` passes into a scores view of ``rows // passes`` rows as the
    decode layer's halves, with ``sort`` in dsa_select.sort_rows's order
    (sequences by first position, highest first): each batch row's scores
    ``[rows, T]`` (read at its slot) and the new-token scores."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_decode

    group, owner, layer, page_tokens, pages = 4, 1, 1, 256, 12
    sequences, half = rows // width, rows // passes
    rng = np.random.default_rng(seed + rows + width)
    span = pages * page_tokens
    firsts = rng.integers(page_tokens // 2, span - width - 1, sequences)
    firsts[0] = (
        group + owner
    ) * page_tokens - width // 2  # a sequence across an owned page's start
    positions = np.concatenate([f + np.arange(width) for f in firsts]).astype(np.int32)
    table = np.repeat(
        (1 + np.arange(sequences * pages, dtype=np.int32)).reshape(sequences, pages),
        width,
        axis=0,
    )
    blocks = 1 + sequences * pages
    cache = rng.integers(0, 256, (blocks, 2, page_tokens + 32, 128), dtype=np.uint8)
    draw = rng.standard_normal((blocks, page_tokens, 128)) * np.exp2(
        rng.integers(-4, 5, (blocks, page_tokens, 1))
    )
    key_bytes, exponent = (
        np.asarray(a)
        for a in dsa_cache.quantize_index_keys(jnp.asarray(draw, jnp.bfloat16))
    )
    cache[:, layer, :page_tokens] = key_bytes
    cache[:, layer, page_tokens : page_tokens + page_tokens // 128] = exponent.reshape(
        blocks, page_tokens // 128, 128
    )
    raw = jnp.asarray(rng.standard_normal((rows, 32, 128)), jnp.float32)
    queries = jnp.asarray(dsa_cache.quantize_index_queries(raw)[0], jnp.bfloat16)
    weights = jnp.asarray(
        np.broadcast_to(rng.standard_normal((rows, 32, 1)), (rows, 32, 128)),
        jnp.float32,
    )
    codes, exponents = dsa_cache.index_key_rows(
        jnp.asarray(rng.standard_normal((rows, 128)), jnp.bfloat16)
    )
    new_values = (
        jnp.zeros((rows, 3, 128), jnp.bfloat16)
        .at[:, 0]
        .set(codes)
        .at[:, 1]
        .set(exponents)
    )
    order = np.zeros((3, rows), np.int32)
    ranked = (
        sorted(range(sequences), key=lambda q: (-int(firsts[q]), q))
        if sort
        else range(sequences)
    )
    for place, q in enumerate(ranked):
        for w in range(width):
            row, slot = q * width + w, place * width + w
            order[0, row], order[1, slot], order[2, slot] = slot, row, positions[row]
    held = -(-pages // group)
    shapes = dsa_decode.scratch_shapes(rows, held * page_tokens, page_tokens, False)
    plane_rows = dsa_decode.score_shape(held * page_tokens, half)

    def kernel(
        table_ref,
        position_ref,
        cache_ref,
        queries_ref,
        weights_ref,
        new_value_ref,
        order_ref,
        scores_out,
        new_scores_ref,
        view,
        ring,
        *scoped,
    ):
        scratch = dsa_decode.IndexScratch(
            cache_ref,
            None,
            ring,
            queries_ref,
            weights_ref,
            None,
            view,
            *scoped,
            new_scores=new_scores_ref,
        )
        counts = [
            dsa_decode.local_pages(
                position_ref[b + width - 1], owner, page_tokens, group
            )
            if b % width == 0
            else 0
            for b in range(rows)
        ]
        scorer = dsa_decode.IndexScorer(
            scratch,
            position_ref=position_ref,
            batch=rows,
            active=position_ref[0] >= 0,
            table_ref=table_ref,
            pages=pages,
            layer=layer,
            page_tokens=page_tokens,
            group=group,
            global_pages=True,
            owner=owner,
            slots=3,
            uint8_packets=False,
            new_value_ref=new_value_ref,
            counts=counts,
            sequence_rows=width,
            order=order_ref if sort else None,
            group_new_tokens=group_new_tokens,
        )
        assert scorer._groups_new_tokens(half) == group_new_tokens
        scorer.prime()
        for g in range(passes):
            view[...] = jnp.full(view.shape, jnp.nan, jnp.float32)
            scorer.score(g * half, half)
            scores_out[g] = view[...]

    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    scores, new_scores = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((passes, *plane_rows), jnp.float32),
            jax.ShapeDtypeStruct(shapes["new_scores"][0], jnp.float32),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                vmem,
                vmem,
                vmem,
                pl.BlockSpec(memory_space=pltpu.SMEM),
            ],
            out_specs=(vmem, vmem),
            scratch_shapes=(
                pltpu.VMEM(plane_rows, jnp.float32),
                pltpu.VMEM(*shapes["ring"]),
                *dsa_decode.scoped_scratch(rows, False),
            ),
        ),
        interpret=pltpu.InterpretParams(uninitialized_memory="nan"),
    )(
        jnp.asarray(table.reshape(-1)),
        jnp.asarray(positions),
        jnp.asarray(cache),
        queries,
        weights,
        new_values,
        jnp.asarray(order),
    )
    by_pass = [dsa_decode.scores_by_row(np.asarray(view), half) for view in scores]
    slots = order[0]
    by_row = np.stack([by_pass[slots[r] // half][slots[r] % half] for r in range(rows)])
    return by_row, np.asarray(new_scores)


@pytest.mark.parametrize(
    "rows, width, passes, sort",
    [
        (8, 1, 1, False),
        (16, 1, 1, False),
        (16, 2, 1, False),
        (16, 4, 1, False),
        (8, 8, 1, False),
        (16, 8, 1, False),
        (32, 4, 2, False),
        (32, 1, 2, False),
        (16, 1, 1, True),
        (16, 4, 1, True),
        (16, 8, 1, True),
        (32, 4, 2, True),
    ],
)
def test_grouped_new_tokens_are_the_per_row_scores(rows, width, passes, sort):
    """IndexScorer's new tokens per group of 8 rows in the scoring order (one
    dot of the group's queries against its 8 new keys,
    KernelConfig.dsa_group_new_tokens) store the per-row path's scores and
    new-token scores bit for bit, for verify sequences that divide 8, two
    score-and-select passes and the select's row order (each row read at its
    slot), as the per-row path in that order does."""
    want = _scores_and_new_scores(rows, width, passes, False, False)
    arms = [(True, sort)] + ([(False, True)] if sort else [])
    for group_new_tokens, order in arms:
        got = _scores_and_new_scores(rows, width, passes, group_new_tokens, order)
        for a, b in zip(want, got, strict=True):
            assert np.isfinite(a).any()
            np.testing.assert_array_equal(b.view(np.uint32), a.view(np.uint32))


def test_score_rows_refuses_calls_that_do_not_cover_every_row():
    """Calls of ``chunk`` rows that leave rows unscored are refused."""
    from jax.experimental.pallas import tpu as pltpu

    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_decode

    rows, page_tokens = 3, 256
    with pytest.raises(ValueError, match="chunk must divide the rows"):
        dsa_decode.score_rows(
            jnp.zeros((1 + rows, 1, page_tokens + 32, 128), jnp.uint8),
            jnp.ones((rows, 1), jnp.int32),
            jnp.zeros(rows, jnp.int32),
            jnp.zeros((rows, 32, 128), jnp.bfloat16),
            jnp.zeros((rows, 32, 128), jnp.float32),
            jnp.zeros((rows, 3, 128), jnp.bfloat16),
            layer=0,
            page_tokens=page_tokens,
            group=4,
            owner=0,
            chunk=2,
            interpret=pltpu.InterpretParams(dma_execution_mode="on_wait"),
        )


def _small_scorer_case(rows=8, page_tokens=256, group=4, owner=1, pages=3, seed=3):
    """Inputs of dsa_decode.score_rows: rows at several pages on ``owner``."""
    rng = np.random.default_rng(seed)
    blocks = 1 + rows * pages
    cache = rng.integers(0, 256, (blocks, 1, page_tokens + 32, 128), dtype=np.uint8)
    table = (1 + np.arange(rows * pages, dtype=np.int32)).reshape(rows, pages)
    positions = rng.integers(1, pages * group * page_tokens, rows).astype(np.int32)
    queries = jnp.asarray(rng.integers(-8, 8, (rows, 32, 128)), jnp.bfloat16)
    weights = jnp.asarray(rng.standard_normal((rows, 32, 128)), jnp.float32)
    new_values = jnp.zeros((rows, 3, 128), jnp.bfloat16)
    return (
        jnp.asarray(cache),
        jnp.asarray(table),
        jnp.asarray(positions),
        queries,
        weights,
        new_values,
    )


def test_scorer_in_the_select_order_is_the_unsorted_scorer_at_each_slot():
    """With dsa_select.sort_rows's order (rows by position, highest first) the
    scorer takes its rows in slot order and stores each at its slot: every
    page DMA meets its wait (the interpreter would hang otherwise) and slot
    ``i`` holds row ``order[1, i]``'s unsorted scores, bit for bit (every
    sublane holds the same head-sum bits)."""
    from jax.experimental.pallas import tpu as pltpu

    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_decode

    rows = 16
    case = _small_scorer_case(rows=rows)
    positions = np.asarray(case[2])
    by_slot = sorted(range(rows), key=lambda r: (-int(positions[r]), r))
    order = np.zeros((3, rows), np.int32)
    for slot, row in enumerate(by_slot):
        order[0, row], order[1, slot], order[2, slot] = slot, row, positions[row]
    options = dict(layer=0, page_tokens=256, group=4, owner=1, chunk=rows)
    params = pltpu.InterpretParams(
        dma_execution_mode="on_wait", uninitialized_memory="nan"
    )
    unsorted, in_order = (
        dsa_decode.scores_by_row(
            np.asarray(
                dsa_decode.score_rows(*case, **options, order=o, interpret=params)
            ),
            rows,
        )
        for o in (None, order)
    )
    assert by_slot != list(range(rows))
    for slot, row in enumerate(by_slot):
        want, got = unsorted[row], in_order[slot]
        assert np.array_equal(np.isnan(want), np.isnan(got)), (slot, row)
        known = ~np.isnan(want)
        assert np.array_equal(
            want[known].view(np.uint32), got[known].view(np.uint32)
        ), (slot, row)


@pytest.mark.parametrize(
    "rows, chunk, passes, sort",
    [
        (8, 8, 1, False),
        (32, 8, 1, False),
        (16, 16, 2, False),
        (4, 4, 1, False),
        (16, 8, 1, True),
        (32, 8, 2, True),
    ],
)
def test_rows_selected_new_is_row_selected_new_of_each_row(rows, chunk, passes, sort):
    """IndexScorer.rows_selected_new (whole state tiles in batch order, else
    row by row) is row_selected_new of every row, over calls of ``chunk`` rows
    at traced offsets: with early rows, short rows, keys above, at and below
    the threshold, and ties taken or not, under one or two select passes and
    the select's row order."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_decode, dsa_select

    rng = np.random.default_rng(rows * 10 + passes + sort)
    vrows, parts = dsa_decode.virtual_rows(rows // passes)
    new_scores = np.repeat(
        rng.standard_normal((max(rows, 8), 1)).astype(np.float32), 128, axis=1
    )
    key = np.asarray(dsa_select._key(jnp.asarray(new_scores.view(np.int32))))
    slots = rng.permutation(rows) if sort else np.arange(rows)
    state = np.zeros((passes, dsa_select.STATE_PLANES, vrows, 128), np.int32)
    for row in range(rows):
        g, vrow = divmod(int(slots[row]), rows // passes)
        tau = key[row] + rng.choice([-1, 0, 0, 1])
        state[g, dsa_select.BASE, vrow * parts] = tau ^ np.int32(-(2**31))
        state[g, dsa_select.EARLY, vrow * parts] = rng.random() < 0.1
        state[g, dsa_select.AT, vrow * parts] = rng.choice(
            [dsa.INDEX_TOPK, dsa.INDEX_TOPK + 3]
        )
    positions = rng.choice([100, 2047, 2048, 50_000], size=rows).astype(np.int32)
    order = np.zeros((3, rows), np.int32)
    order[0] = slots
    order[1, slots] = np.arange(rows)

    def kernel(position_ref, order_ref, state_ref, new_ref, want_ref, got_ref):
        selections = []
        for g in range(passes):
            selection = object.__new__(dsa_select.TopkSelect)
            selection.s, selection.parts = SimpleNamespace(state=state_ref.at[g]), parts
            selections.append(selection)
        scorer = object.__new__(dsa_decode.IndexScorer)
        scorer.s, scorer.position_ref, scorer.batch = (
            SimpleNamespace(new_scores=new_ref),
            position_ref,
            rows,
        )
        scorer.order, scorer.parts, scorer.runs = (
            order_ref if sort else None,
            parts,
            True,
        )
        scorer.selection, scorer.select_rows = (
            selections if passes > 1 else selections[0],
            rows // passes,
        )
        for row in range(rows):
            want_ref[pl.ds(row, 1), :] = jnp.broadcast_to(
                scorer.row_selected_new(row).astype(jnp.int32), (1, 128)
            )

        @pl.loop(0, rows // chunk)
        def call(c):
            first = pl.multiple_of(c * chunk, chunk)
            got_ref[pl.ds(first, chunk), :] = jnp.broadcast_to(
                scorer.rows_selected_new(first, chunk), (chunk, 128)
            )

    smem, vmem = (
        pl.BlockSpec(memory_space=pltpu.SMEM),
        pl.BlockSpec(memory_space=pltpu.VMEM),
    )
    want, got = pl.pallas_call(
        kernel,
        out_shape=[jax.ShapeDtypeStruct((rows, 128), jnp.int32)] * 2,
        in_specs=[smem, smem, vmem, vmem],
        out_specs=[vmem, vmem],
        interpret=pltpu.InterpretParams(),
    )(
        jnp.asarray(positions),
        jnp.asarray(order),
        jnp.asarray(state),
        jnp.asarray(new_scores),
    )
    want, got = np.asarray(want), np.asarray(got)
    assert 0 < want[:, 0].sum() < rows
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("rows, chunk", [(2, 1), (8, 8)])
def test_index_scorer_ablations_validate_and_run(rows, chunk):
    """The scorer's dcp_skip ablations: unknown or unbalanced sets are refused,
    and every arm runs to the end (each DMA it starts is waited, so the
    interpreter does not hang) with its own output over each row's ``[0, L)``
    and -inf tail (check.masked_tokens). Unwritten entries are NaN here, as
    are scores of the never-filled ring (uint8 255 is an E4M3 NaN): index_off
    and the arms without a store write nothing; the store-only arm stores
    -inf; the compute arms (no DMA) store every chunk, the tail -inf and the
    new token's entry finite (scored from its handoff)."""
    from jax.experimental.pallas import tpu as pltpu

    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_decode

    case = _small_scorer_case(rows=rows)
    options = dict(layer=0, page_tokens=256, group=4, owner=1, chunk=chunk)
    params = pltpu.InterpretParams(
        dma_execution_mode="on_wait", uninitialized_memory="nan"
    )
    for skip, message in (
        (("index_nothing",), "unknown scorer ablations"),
        (("index_store",), "index_store needs index_dots"),
        (("index_fetch",), "index_fetch needs index_dma"),
    ):
        with pytest.raises(ValueError, match=message):
            dsa_decode.score_rows(*case, **options, skip=skip, interpret=params)

    def run(*skip):
        scores = dsa_decode.score_rows(*case, **options, skip=skip, interpret=params)
        return np.asarray(scores)

    for skip in (
        ("index_off",),
        ("index_dots", "index_store"),
        ("index_dma", "index_dots", "index_store"),
    ):
        assert np.isnan(run(*skip)).all(), skip
    store = dsa_decode.scores_by_row(run("index_dma", "index_dots"), rows)
    computes = [
        dsa_decode.scores_by_row(run(*skip), rows)
        for skip in (("index_dma",), ("index_dma", "index_fetch"))
    ]
    for r, position in enumerate(np.asarray(case[2]).tolist()):
        length = int(dsa_decode.local_length(position, 1, 256, 4))
        masked = tool_check.masked_tokens(position, 1, 256, 4)
        assert (store[r, :length] == -np.inf).all(), r
        assert (store[r, masked] == -np.inf).all(), r
        new = position // 256 % 4 == 1  # its new token is local token length - 1
        for got in computes:
            finite = np.flatnonzero(np.isfinite(got[r, :length]))
            assert finite.tolist() == ([length - 1] if new else []), r
            assert (got[r, masked] == -np.inf).all(), r


def test_index_scorer_scopes_its_work():
    """score books each call to the index_score scope; the test's recording
    scope sees one per call."""
    import contextlib

    from jax.experimental.pallas import tpu as pltpu

    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_decode

    names = []

    @contextlib.contextmanager
    def scope(name):
        names.append(name)
        yield

    dsa_decode.score_rows(
        *_small_scorer_case(rows=16),
        layer=0,
        page_tokens=256,
        group=4,
        owner=1,
        chunk=8,
        scope=scope,
        interpret=pltpu.InterpretParams(dma_execution_mode="on_wait"),
    )
    assert names == ["index_score"]  # one rolled call body, traced once


def _lowered_scorer_ops(skip):
    """Op counts of dsa_decode.score_rows' Mosaic module for one TPU7x core,
    lowered on the host (no TPU; pallas_call with ``debug=True`` prints it):
    page-table loads, DMA starts and waits, MXU dots and loads of the dots'
    operands (ring, queries, weights, new keys)."""
    import contextlib
    import io

    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_decode

    rows, pages, page_tokens = 8, 3, 256
    operands = (
        jax.ShapeDtypeStruct((1 + rows * pages, 1, page_tokens + 32, 128), jnp.uint8),
        jax.ShapeDtypeStruct((rows, pages), jnp.int32),
        jax.ShapeDtypeStruct((rows,), jnp.int32),
        jax.ShapeDtypeStruct((rows, 32, 128), jnp.bfloat16),
        jax.ShapeDtypeStruct((rows, 32, 128), jnp.float32),
        jax.ShapeDtypeStruct((rows, 3, 128), jnp.bfloat16),
    )

    def score(*args):
        return dsa_decode.score_rows(
            *args,
            layer=0,
            page_tokens=page_tokens,
            group=4,
            owner=1,
            chunk=8,
            skip=skip,
        )

    device = jax.sharding.AbstractDevice(
        device_kind="TPU7x", num_cores=1, platform="tpu"
    )
    printed = io.StringIO()
    with (
        jax.sharding.use_abstract_mesh(
            jax.sharding.AbstractMesh((1,), ("x",), abstract_device=device)
        ),
        contextlib.redirect_stdout(printed),
    ):
        jax.jit(score).trace(*operands).lower(lowering_platforms=("tpu",))
    text = printed.getvalue()
    module = text[text.index("module for pallas_call") :]
    loads = ("%ring", "%queries_ref", "%weights_ref", "%new_value_ref")
    return {
        "table loads": module.count("memref.load %table_ref["),
        "dma starts": module.count("tpu.enqueue_dma"),
        "dma waits": module.count("tpu.wait_dma"),
        "dots": module.count("tpu.matmul"),
        "dot loads": sum(module.count(f"vector.load {ref}[") for ref in loads),
    }


def test_index_scorer_ablations_remove_what_they_name(monkeypatch):
    """Lowered for a TPU (after JAX's dead-code elimination), each scorer
    ablation removes its own work and keeps the rest: index_dma every
    page-table load (sunk) but no DMA; index_fetch also the table loads;
    index_dots every MXU dot and every load that feeds one. Relations, not
    totals, so that a rewrite of the page loop keeps what each arm times."""
    from jax.experimental import pallas as pl

    pallas_call = pl.pallas_call
    monkeypatch.setattr(
        pl, "pallas_call", lambda *a, **k: pallas_call(*a, **k, debug=True)
    )
    ops = {
        skip: _lowered_scorer_ops(skip)
        for skip in (
            (),
            ("index_dma",),
            ("index_dma", "index_fetch"),
            ("index_dots", "index_store"),
            ("index_dma", "index_dots"),
            ("index_dma", "index_dots", "index_store"),
        )
    }
    default = ops[()]
    assert min(default.values()) > 0, default
    no_dma = {"dma starts": 0, "dma waits": 0}
    no_dots = {"dots": 0, "dot loads": 0}
    assert ops["index_dma",] == {**default, **no_dma}
    assert ops["index_dma", "index_fetch"] == {**default, **no_dma, "table loads": 0}
    assert ops["index_dots", "index_store"] == {**default, **no_dots}
    for skip in (
        ("index_dma", "index_dots"),
        ("index_dma", "index_dots", "index_store"),
    ):
        assert ops[skip] == {**default, **no_dma, **no_dots}, skip


def test_k31_mla_refuses_verify_rows_over_a_bf16_cache():
    """K3.1's verify rows (``sequence_rows > 1``) attend cached FP8
    sequences, whose pages hold the earlier rows' latents: a BF16 cache with
    the indexer's buffers is refused."""
    from vllm_torchtpu.megakernel.kimi_k3.kernels import decode_megakernel

    cache = SimpleNamespace(
        page_tokens=256, pages=2, global_pages=False, group=32, dtype=jnp.bfloat16
    )
    hidden = np.zeros((2, 128), np.float32)
    with pytest.raises(ValueError, match="cached FP8 sequences only"):
        decode_megakernel.mla(
            np.zeros(2, np.int32),
            hidden,
            hidden,
            None,
            None,
            weights=None,
            workspace=None,
            communication_workspace=None,
            dcp=(cache, None),
            sequence_rows=2,
            index=object(),
        )


@pytest.mark.parametrize(
    "context, refused", [(dsa.INDEX_TOPK, True), (dsa.INDEX_TOPK + 1, False)]
)
def test_dsa_scores_check_refuses_a_last_step_that_does_not_score(
    monkeypatch, capsys, context, refused
):
    """``check dsa-scores`` stops before it builds a decoder when no row's
    context in the last step exceeds ``INDEX_TOPK`` (that step scores
    nothing)."""
    from vllm_torchtpu.megakernel.kimi_k3 import decode_ops

    class Built(Exception):
        pass

    def build(*args, **kwargs):
        raise Built

    monkeypatch.setattr(tool_k3, "DSA", (dsa.DsaConfig(), ("random", 0)))
    monkeypatch.setattr(
        tool_k3, "fixed_prompt", lambda count, model: (list(range(count)), None)
    )
    monkeypatch.setattr(decode_ops, "DecodeOps", build)
    generate = 4
    args = SimpleNamespace(
        parts="inputs,queries,scores",
        prompt_tokens=context - generate + 1,  # the last step's context
        generate=generate,
        batch=1,
        model="m",
        presharded="p",
    )
    with pytest.raises(SystemExit if refused else Built):
        tool_check.run_dsa_scores(args, None)
    assert ("needs a last step that scores" in capsys.readouterr().out) == refused


def _loader(value, role="decode"):
    kv = SimpleNamespace(
        is_kv_producer=role == "prefill", is_kv_consumer=role == "decode"
    )
    return SimpleNamespace(
        load_config=SimpleNamespace(
            load_format="megakernel", model_loader_extra_config=value
        ),
        kv_transfer_config=kv,
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["KimiK31ForConditionalGeneration"])
        ),
    )


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_synthetic_indexer_loader_keys(role):
    from vllm_torchtpu.megakernel import backend

    base = {"presharded": "/x"}
    assert backend.synthetic_indexer(_loader(base, role)) is None
    value = dict(base, synthetic_indexer="random", indexer_seed=3)
    assert backend.synthetic_indexer(_loader(value, role)) == ("random", 3)
    assert backend.synthetic_indexer(
        _loader(dict(base, synthetic_indexer="rope-probe"), role)
    ) == ("rope-probe", 0)
    with pytest.raises(ValueError, match="synthetic indexer"):
        backend.synthetic_indexer(_loader(dict(base, synthetic_indexer="zeros"), role))
    with pytest.raises(ValueError, match="indexer_seed"):
        backend.synthetic_indexer(_loader(dict(base, indexer_seed=1), role))


def _k31_config(**changes):
    from vllm_torchtpu.megakernel.config import MegakernelEngineConfig
    from vllm_torchtpu.megakernel.kimi_k3.dsa import DsaConfig

    stage = dict(
        role="prefill",
        tensor_parallel_size=1,
        decode_context_parallel_size=1,
        block_size=2048,
        kv_cache_dtype="fp8",
        cp_kv_cache_interleave_size=2048,
        max_num_seqs=4,
        max_model_len=131072,
        max_num_batched_tokens=8192,
        async_scheduling=False,
        pipeline_parallel_size=4,
        prefill_context_parallel_size=8,
        dsa=DsaConfig(),
        synthetic_indexer=("random", 0),
    )
    stage.update(changes)
    return MegakernelEngineConfig(**stage)


def test_program_check_takes_k31_prefill_on_the_stages_and_the_slice():
    from vllm_torchtpu.megakernel.kimi_k3 import program

    program.check(_k31_config())
    program.check(
        _k31_config(
            tensor_parallel_size=32,
            decode_context_parallel_size=32,
            block_size=512,
            cp_kv_cache_interleave_size=512,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
        )
    )
    with pytest.raises(NotImplementedError, match="synthetic_indexer"):
        program.check(_k31_config(synthetic_indexer=None))
    program.check(_k31_config(synthetic_indexer=None, presharded_indexer=True))


def test_program_build_hands_the_decoder_its_synthetic_indexer(monkeypatch):
    from vllm_torchtpu.megakernel.kimi_k3 import decode, decode_ops, program

    seen = {}

    class Ops:
        def __init__(self, *args, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(decode_ops, "DecodeOps", Ops)
    monkeypatch.setattr(decode, "KimiDecoder", lambda config, ops, draft, tp_rank: ops)
    config = _k31_config(
        role="decode",
        tensor_parallel_size=32,
        decode_context_parallel_size=32,
        block_size=512,
        cp_kv_cache_interleave_size=512,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        max_num_batched_tokens=512,
        async_scheduling=True,
    )
    program.build(config, "/x", tp_rank=0)
    assert seen["synthetic_indexer"] == ("random", 0)


@pytest.mark.parametrize("short", [True, False])
def test_dsa_select_check_refuses_a_prompt_shorter_than_asked(
    monkeypatch, capsys, short
):
    """``check dsa-select`` stops before it builds a decoder when the fixed
    prompt holds fewer tokens than ``--prompt-tokens`` (it would cut its long
    rows silently)."""
    from vllm_torchtpu.megakernel.kimi_k3 import decode_ops

    class Built(Exception):
        pass

    def build(*args, **kwargs):
        raise Built

    held = 5000 - short
    monkeypatch.setattr(tool_k3, "DSA", (dsa.DsaConfig(), ("random", 0)))
    monkeypatch.setattr(
        tool_k3,
        "fixed_prompt",
        lambda count, model: (list(range(min(count, held))), None),
    )
    monkeypatch.setattr(decode_ops, "DecodeOps", build)
    args = SimpleNamespace(
        parts="select",
        prompt_tokens=5000,
        generate=4,
        batch=8,
        long_rows=2,
        tie_keys=0,
        model="m",
        presharded="p",
        token_file="",
    )
    with pytest.raises(SystemExit if short else Built):
        tool_check.run_dsa_select(args, None)
    assert ("fewer than --prompt-tokens" in capsys.readouterr().out) == short


def test_dsa_checks_refuse_an_owner_table(monkeypatch, capsys):
    """The DSA checks key a rank's pages by its mesh position: a decoder whose
    pages follow an owner table is refused before it decodes."""
    from vllm_torchtpu.megakernel.kimi_k3 import decode_ops

    class Ops:
        owner_table = tuple(reversed(range(32)))

        def __init__(self, *args, **kwargs):
            pass

    class Decoded(Exception):
        pass

    def decoder(*args, **kwargs):
        return SimpleNamespace(warmup=lambda: (_ for _ in ()).throw(Decoded()))

    monkeypatch.setattr(decode_ops, "DecodeOps", Ops)
    monkeypatch.setattr(tool_k3, "Pools", lambda *args: None)
    monkeypatch.setattr(tool_k3, "decoder", decoder)
    monkeypatch.setattr(tool_k3, "decode_ops_options", lambda: {})
    monkeypatch.setattr(tool_k3, "logical", lambda: 8192)
    monkeypatch.setattr(tool_check.harness, "sync", lambda: None)
    args = SimpleNamespace(batch=1, presharded="p", generate=4)
    with pytest.raises(SystemExit):
        tool_check._index_debug_decode(args, [np.zeros(4, np.int32)], 2)
    assert "owner table" in capsys.readouterr().out
    Ops.owner_table = None
    with pytest.raises(Decoded):
        tool_check._index_debug_decode(args, [np.zeros(4, np.int32)], 2)


def test_index_scorer_refuses_a_field_inbox_of_another_group():
    """The field all-gather waits for ``group - 1`` peers' slots: an inbox
    sized for another group is refused when the scorer is built."""
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_decode

    def scorer(members):
        inbox = SimpleNamespace(shape=(members, 16, 128))
        scratch = dsa_decode.IndexScratch(
            *(None,) * 12, field_inbox=inbox, field_semaphores=None
        )
        return dsa_decode.IndexScorer(
            scratch, position_ref=None, batch=16, active=True, table_ref=None,
            pages=1, layer=0, page_tokens=256, group=8, global_pages=False,
            owner=0, slots=3, uint8_packets=False, new_value_ref=None, member=0,
            base=0,
        )  # fmt: skip

    assert not scorer(8).packet_fields
    with pytest.raises(ValueError, match="inbox"):
        scorer(4)


def test_chunked_dcp_attention_refuses_index_fields_on_the_query_packets():
    """``_attend_chunked`` reads the index fields from their own all-gather:
    a scorer whose fields ride the query packets (an unchunked batch forced
    into chunks) is refused."""
    from vllm_torchtpu.megakernel.kimi_k3.kernels import mla_dcp

    attention = SimpleNamespace(index=SimpleNamespace(packet_fields=True))
    with pytest.raises(ValueError, match="packet"):
        mla_dcp._attend_chunked(attention, None, None, None)


def test_masked_step_folds_a_sparse_rows_selected_tokens_only():
    """A page step's visible tokens: below the row's position, and for a row
    of more than ``INDEX_TOPK`` tokens only those its selection marks (asked
    for the step's local tokens, from ``local_step span page_tokens``); a row
    within the top-k sees every token below its position, as Kimi-K3's."""
    from vllm_torchtpu.megakernel.kimi_k3.kernels import mla_dcp

    span, page, group, owner = 2, 512, 32, 3
    tokens = span * page
    # Row 0 ends inside its second step's first page.
    positions = jnp.asarray([34500, dsa.INDEX_TOPK - 1], jnp.int32)
    chosen = np.random.default_rng(1).integers(0, 2, (1, tokens), np.int32)
    asked = []

    def lanes(row, first, count):
        asked.append((int(row), int(first), count))
        return jnp.asarray(chosen)

    attention = SimpleNamespace(
        span=span, page_tokens=page, group=group, owner=owner, buffers=3,
        position_ref=positions, sequence_rows=1, cache_verify_rows=False,
        index=SimpleNamespace(lanes=lanes),
        _clean_step=lambda *args: None, _pages=lambda slot: None,
    )  # fmt: skip
    attention._limit = lambda row: mla_dcp.DcpAttention._limit(attention, row)
    attention._sparse = lambda row: mla_dcp.DcpAttention._sparse(attention, row)
    attention._chosen = lambda row, step: mla_dcp.DcpAttention._chosen(
        attention, row, step
    )
    offset = np.arange(tokens)
    for local_step in (0, 1):
        page_of = (local_step * span + offset // page) * group + owner
        token = page_of * page + offset % page
        for row in (0, 1):
            attention._locate = lambda step, row=row, local_step=local_step: (
                jnp.int32(row),
                jnp.int32(local_step),
                2,
            )
            got_row, visible, _ = mla_dcp.DcpAttention._step_pages(attention, 0)
            below = token[None] < int(positions[row])
            want = below & (chosen != 0) if row == 0 else below
            assert int(got_row) == row
            np.testing.assert_array_equal(np.asarray(visible), want)
    assert asked == [(0, 0, tokens), (1, 0, tokens)] + [
        (0, tokens, tokens),
        (1, tokens, tokens),
    ]
    attention.index = None
    assert mla_dcp.DcpAttention._sparse(attention, 0) is False


def test_merge_received_weights_the_new_token_by_its_flag():
    """``_merge_received``: with ``new_selected`` a row's new token joins the
    log-sum-exp merge of the peers' partials only when its flag is set (a peer
    with no tokens, lse ``NEG``, never counts); without it every new token
    joins, as Kimi-K3's."""
    from vllm_torchtpu.megakernel.kimi_k3.kernels import mla_dcp

    group, rows = 4, 6
    rng = np.random.default_rng(3)
    outputs = rng.standard_normal((group, rows, 512)).astype(np.float32)
    lse = rng.standard_normal((group, rows)).astype(np.float32)
    lse[1, 0], outputs[1, 0] = mla_dcp.NEG, 0
    received = np.zeros((group, rows, 640), np.float32)
    received[..., :512], received[..., 512] = outputs, lse
    own = rng.standard_normal((rows, 640)).astype(np.float32) * 0.1
    new = rng.standard_normal((rows, 640)).astype(np.float32)
    flags = np.array([True, False, True, False, True, True])[:, None]
    score = (own.astype(np.float64) * new).sum(1)

    def want(take):
        weights = np.exp(np.concatenate((lse.T, score[:, None]), 1).astype(np.float64))
        weights[:, -1] *= take
        values = np.concatenate((outputs.transpose(1, 0, 2), new[:, None, :512]), 1)
        return (weights[..., None] * values).sum(1) / weights.sum(1, keepdims=True)

    def merge(**options):
        return np.asarray(
            mla_dcp._merge_received(
                jnp.asarray(received), jnp.asarray(own), jnp.asarray(new), group,
                **options,
            )
        )  # fmt: skip

    np.testing.assert_allclose(
        merge(new_selected=jnp.asarray(flags)), want(flags[:, 0]), rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(merge(), want(np.ones(rows)), rtol=1e-5, atol=1e-6)


def test_new_selected_rows_flag_each_rows_head_slots():
    """``DcpAttention._new_selected_rows(first, rows)``: packet row ``slots j +
    h`` holds batch row ``first + j``'s new-token flag, ``first`` traced."""
    from vllm_torchtpu.megakernel.kimi_k3.kernels import mla_dcp

    slots, rows = 4, 5
    pattern = np.array([True, False, False, True, True, False, True, False, True])
    attention = SimpleNamespace(
        slots=slots,
        heads=3,
        index=SimpleNamespace(
            rows_selected_new=lambda first, count: jax.lax.dynamic_slice(
                jnp.asarray(pattern, jnp.int32), (first,), (count,)
            )[:, None]
        ),
    )

    @jax.jit
    def flags(first):
        return mla_dcp.DcpAttention._new_selected_rows(attention, first, rows)

    for first in (0, 3):
        got = np.asarray(flags(jnp.int32(first)))
        want = np.repeat(pattern[first : first + rows], slots)[:, None]
        assert got.shape == (slots * rows, 1)
        np.testing.assert_array_equal(got, want)


def test_merge_partials_weights_the_new_token_by_its_flag():
    """The unchunked attention's merge (``_merge_partials``): with
    ``new_selected`` a row's new token joins the log-sum-exp merge of the
    peers' partials only when its flag is set (a row whose every peer but one
    holds lse ``NEG``, with its flag clear, is that peer's output); without it
    every new token joins, as Kimi-K3's."""
    from vllm_torchtpu.megakernel.kimi_k3.kernels import mla_dcp

    group, rows = 4, 6
    rng = np.random.default_rng(4)
    outputs = rng.standard_normal((group, rows, 512)).astype(np.float32)
    lse = rng.standard_normal((group, rows, 1)).astype(np.float32)
    lse[:3, 1], outputs[:3, 1] = mla_dcp.NEG, 0
    partials = np.zeros((group, rows, 640), np.float32)
    partials[..., :512] = outputs
    new = rng.standard_normal((rows, 640)).astype(np.float32)
    score = rng.standard_normal((rows, 1)).astype(np.float32)
    flags = np.array([True, False, True, False, True, True])[:, None]

    def want(take):
        logits = np.concatenate((lse[..., 0].T, score), 1).astype(np.float64)
        weights = np.exp(logits - logits.max(1, keepdims=True))
        weights[:, -1] *= take
        values = np.concatenate((outputs.transpose(1, 0, 2), new[:, None, :512]), 1)
        return (weights[..., None] * values).sum(1) / weights.sum(1, keepdims=True)

    def merge(flag=None):
        return np.asarray(
            mla_dcp._merge_partials(
                jnp.asarray(partials), jnp.asarray(lse), jnp.asarray(score),
                jnp.asarray(new), None if flag is None else jnp.asarray(flag),
            )
        )  # fmt: skip

    got = merge(flags)
    np.testing.assert_allclose(got, want(flags[:, 0]), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(got[1], outputs[3, 1], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(merge(), want(np.ones(rows)), rtol=1e-5, atol=1e-6)


def test_select_row_summary_counts_missing_extra_and_bad_rows():
    """A row's selection over two members: a swapped token is one missing and
    one extra; a sparse row a token short of ``INDEX_TOPK`` is a bad count; a
    short row with a marked token or a clear flag is bad."""
    k = dsa.INDEX_TOPK
    want = [np.zeros(1500, bool), np.zeros(1500, bool)]
    want[0][:1000], want[1][: k - 1001] = True, True  # plus the new token

    def summary(masks, flag, want_masks=want, want_flag=True, sparse=True):
        return tool_check.select_row_summary(
            masks, flag, want_masks, want_flag, sparse=sparse
        )

    exact = {"missing": 0, "extra": 0, "count_bad": 0, "short_bad": 0}
    assert summary([m.copy() for m in want], True) == exact
    swapped = [m.copy() for m in want]
    swapped[1][0], swapped[1][1400] = False, True
    assert summary(swapped, True) == {**exact, "missing": 1, "extra": 1}
    assert summary([m.copy() for m in want], False) == {
        **exact,
        "missing": 1,
        "count_bad": 1,
    }
    empty = [np.zeros(1500, bool), np.zeros(1500, bool)]
    assert summary(empty, True, empty, True, sparse=False) == exact
    marked = [empty[0].copy(), empty[1]]
    marked[0][3] = True
    assert summary(marked, True, empty, True, sparse=False) == {
        **exact,
        "extra": 1,
        "short_bad": 1,
    }
    assert summary(empty, False, empty, True, sparse=False) == {
        **exact,
        "missing": 1,
        "short_bad": 1,
    }


def test_kda_gate_comes_from_either_pre_shard_layout():
    """``rank_loader.read_kda_gate``: Kimi-K3.1's pre-shard stores ``k_gate``
    apart (``k_projection [.., 1664]``), an older one inside ``k_projection
    [.., 1792]`` at columns 1152..1280; in the split layout those columns are
    the projection's, not the gate."""
    import types

    from vllm_torchtpu.megakernel.kimi_k3.kernels import rank_loader

    rng = np.random.default_rng(5)
    combined = rng.standard_normal((2, 16, 1792)).astype(BF16)
    gate = np.ascontiguousarray(combined[..., 1152:1280])
    rest = np.concatenate((combined[..., :1152], combined[..., 1280:]), axis=-1)

    def rank_dir(families):
        return types.SimpleNamespace(
            families=families,
            read_dense_layers=lambda name, lo, hi: families[name][lo:hi],
        )

    older = rank_dir({"k_projection": combined})
    split = rank_dir({"k_projection": rest, "k_gate": gate})
    assert rank_loader.kda_gate_source(older) == "k_projection"
    assert rank_loader.kda_gate_source(split) == "k_gate"
    for layers in (older, split):
        np.testing.assert_array_equal(
            rank_loader.read_kda_gate(layers, 1, 2), gate[1:2]
        )
    assert not np.array_equal(rest[..., 1152:1280], gate)
