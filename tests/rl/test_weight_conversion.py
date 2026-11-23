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
"""CPU unit tests for pure-PyTorch 12-op executor, validator, and inverter."""

from __future__ import annotations

import ast
import inspect
import json
from types import MappingProxyType
from typing import Any

import pytest
import torch

from vllm_torchtpu.rl import (
    SCHEMA_VERSION,
    execute_manifest,
    from_hf,
    from_hf_impl,
    invert_manifest,
    to_hf,
    to_hf_impl,
    validate_manifest,
)


def _make_manifest(
    rules: list[dict[str, Any]],
    *,
    expert_axis_grouped: bool = False,
    stripes: dict[str, Any] | None = None,
    tied_keys: list[list[str]] | None = None,
    dropped_keys: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "expert_axis_grouped": expert_axis_grouped,
        "stripes": {} if stripes is None else stripes,
        "tied_keys": [] if tied_keys is None else tied_keys,
        "dropped_keys": [] if dropped_keys is None else dropped_keys,
        "rules": rules,
    }


def _assert_round_trip(
    manifest: dict[str, Any],
    src_weights: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    validate_manifest(manifest)
    dst = from_hf(json.dumps(manifest), src_weights)
    inv = invert_manifest(manifest)
    validate_manifest(inv)
    recovered = to_hf(json.dumps(inv), dst)
    assert set(recovered.keys()) == set(src_weights.keys())
    for k, orig in src_weights.items():
        assert recovered[k].dtype == orig.dtype
        assert recovered[k].shape == orig.shape
        assert torch.equal(recovered[k], orig), f"Mismatch on key {k}"
    return dst, recovered


# ---------------------------------------------------------------------------
# All 12 primitive ops: forward & inverse
# ---------------------------------------------------------------------------


def test_op_stack_and_unstack() -> None:
    e0 = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    e1 = torch.arange(6, 12, dtype=torch.float32).reshape(2, 3)
    e2 = torch.arange(12, 18, dtype=torch.float32).reshape(2, 3)

    manifest = _make_manifest(
        [
            {
                "source_keys": ["e0", "e1", "e2"],
                "target_keys": ["stacked"],
                "ops": [{"stack": {"axis": 0}}],
                "sharding": ["ep", None, None],
            }
        ]
    )
    dst, _ = _assert_round_trip(manifest, {"e0": e0, "e1": e1, "e2": e2})
    assert torch.equal(dst["stacked"], torch.stack([e0, e1, e2], dim=0))

    # Test negative axis stack/unstack
    manifest_neg = _make_manifest(
        [
            {
                "source_keys": ["e0", "e1"],
                "target_keys": ["stacked"],
                "ops": [{"stack": {"axis": -1}}],
            }
        ]
    )
    dst_neg, _ = _assert_round_trip(manifest_neg, {"e0": e0, "e1": e1})
    assert torch.equal(dst_neg["stacked"], torch.stack([e0, e1], dim=-1))


def test_op_concat_and_split() -> None:
    q = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    k = torch.arange(12, 18, dtype=torch.float32).reshape(2, 3)
    v = torch.arange(18, 24, dtype=torch.float32).reshape(2, 3)

    manifest = _make_manifest(
        [
            {
                "source_keys": ["q", "k", "v"],
                "target_keys": ["qkv"],
                "ops": [{"concat": {"axis": 0, "sizes": [4, 2, 2]}}],
                "sharding": ["tp", None],
            }
        ]
    )
    dst, _ = _assert_round_trip(manifest, {"q": q, "k": k, "v": v})
    assert torch.equal(dst["qkv"], torch.cat([q, k, v], dim=0))


def test_op_reshape() -> None:
    t = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    manifest = _make_manifest(
        [
            {
                "source_keys": ["x"],
                "target_keys": ["y"],
                "ops": [{"reshape": {"src_shape": [2, 3, 4], "dst_shape": [6, 4]}}],
            }
        ]
    )
    dst, _ = _assert_round_trip(manifest, {"x": t})
    assert torch.equal(dst["y"], t.reshape(6, 4))


def test_op_transpose_asymmetric_permutation() -> None:
    # Non-self-inverse permutation: [2, 0, 1] -> inverse is [1, 2, 0]
    t = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    manifest = _make_manifest(
        [
            {
                "source_keys": ["x"],
                "target_keys": ["y"],
                "ops": [{"transpose": {"axes": [2, 0, 1]}}],
            }
        ]
    )
    dst, _ = _assert_round_trip(manifest, {"x": t})
    assert dst["y"].shape == (4, 2, 3)
    assert torch.equal(dst["y"], t.permute(2, 0, 1))
    inv = invert_manifest(manifest)
    assert inv["rules"][0]["ops"] == [{"transpose": {"axes": [1, 2, 0]}}]


def test_op_scale_float32_compute_and_inverse() -> None:
    # Power-of-two scale factor gives bit-exact round-trip even in bfloat16.
    t = torch.tensor([1.5, -2.25, 0.5, 4.0], dtype=torch.bfloat16)
    manifest = _make_manifest(
        [
            {
                "source_keys": ["x"],
                "target_keys": ["y"],
                "ops": [{"scale": {"factor": 0.25}}],
            }
        ]
    )
    dst, _ = _assert_round_trip(manifest, {"x": t})
    assert dst["y"].dtype == torch.bfloat16
    expected = (t.to(torch.float32) * 0.25).to(torch.bfloat16)
    assert torch.equal(dst["y"], expected)

    # Verify computation happens in float32 (not bfloat16) before casting back.
    # For a value where f32 multiply differs from bf16 multiply:
    v = torch.tensor([1.0078125], dtype=torch.bfloat16)  # 1 + 1/128
    factor = 1.0 / 3.0
    m = _make_manifest(
        [
            {
                "source_keys": ["x"],
                "target_keys": ["y"],
                "ops": [{"scale": {"factor": factor}}],
            }
        ]
    )
    out = from_hf_impl(m, {"x": v})
    assert out["y"].dtype == torch.bfloat16
    assert torch.equal(out["y"], (v.to(torch.float32) * factor).to(torch.bfloat16))


def test_op_pad_and_unpad() -> None:
    t = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    # Asymmetric [left, right] padding
    manifest = _make_manifest(
        [
            {
                "source_keys": ["x"],
                "target_keys": ["y"],
                "ops": [{"pad": {"axis": 1, "sizes": [2, 3]}}],
            }
        ]
    )
    dst, _ = _assert_round_trip(manifest, {"x": t})
    assert dst["y"].shape == (3, 9)
    assert torch.all(dst["y"][:, :2] == 0)
    assert torch.all(dst["y"][:, 6:] == 0)
    assert torch.equal(dst["y"][:, 2:6], t)

    # Right == 0 edge case (must not trigger slice(left, -0) empty slice bug)
    manifest_left_only = _make_manifest(
        [
            {
                "source_keys": ["x"],
                "target_keys": ["y"],
                "ops": [{"pad": {"axis": 0, "sizes": [1, 0]}}],
            }
        ]
    )
    dst_l, _ = _assert_round_trip(manifest_left_only, {"x": t})
    assert dst_l["y"].shape == (4, 4)
    assert torch.all(dst_l["y"][0] == 0)
    assert torch.equal(dst_l["y"][1:], t)


def test_op_cast() -> None:
    t = torch.tensor([1.0, -2.5, 3.25], dtype=torch.bfloat16)
    manifest = _make_manifest(
        [
            {
                "source_keys": ["x"],
                "target_keys": ["y"],
                "ops": [{"cast": {"src_dtype": "bfloat16", "dst_dtype": "float32"}}],
            }
        ]
    )
    dst, _ = _assert_round_trip(manifest, {"x": t})
    assert dst["y"].dtype == torch.float32
    assert torch.equal(dst["y"], t.to(torch.float32))

    fp8 = torch.tensor([1.0, -2.0], dtype=torch.float32).to(torch.float8_e4m3fn)
    m_fp8 = _make_manifest(
        [
            {
                "source_keys": ["x"],
                "target_keys": ["y"],
                "ops": [
                    {"cast": {"src_dtype": "float8_e4m3fn", "dst_dtype": "float32"}}
                ],
            }
        ]
    )
    dst_fp8, _ = _assert_round_trip(m_fp8, {"x": fp8})
    assert dst_fp8["y"].dtype == torch.float32


def test_op_repeat_and_unrepeat() -> None:
    # GQA KV head replication: 2 KV heads repeated 4x -> 8 heads
    kv = torch.arange(2 * 4 * 8, dtype=torch.float32).reshape(2, 4, 8)
    manifest = _make_manifest(
        [
            {
                "source_keys": ["kv"],
                "target_keys": ["kv_rep"],
                "ops": [{"repeat": {"axis": 0, "times": 4}}],
            }
        ]
    )
    dst, _ = _assert_round_trip(manifest, {"kv": kv})
    assert dst["kv_rep"].shape == (8, 4, 8)
    assert torch.equal(dst["kv_rep"], torch.repeat_interleave(kv, 4, dim=0))


# ---------------------------------------------------------------------------
# Showcases
# ---------------------------------------------------------------------------


def test_showcase_1_rope_complex_pair_permutation() -> None:
    """Showcase 1: RoPE complex-pair permutation (reshape -> transpose -> reshape)."""
    num_heads, head_dim, hidden = 4, 8, 16
    w_q = torch.arange(num_heads * head_dim * hidden, dtype=torch.float32).reshape(
        num_heads * head_dim, hidden
    )

    manifest = _make_manifest(
        [
            {
                "source_keys": ["model.layers.0.self_attn.q_proj.weight"],
                "target_keys": ["layers.0.attn.q_proj.weight"],
                "ops": [
                    {
                        "reshape": {
                            "src_shape": [num_heads * head_dim, hidden],
                            "dst_shape": [num_heads, 2, head_dim // 2, hidden],
                        }
                    },
                    {"transpose": {"axes": [0, 2, 1, 3]}},
                    {
                        "reshape": {
                            "src_shape": [num_heads, head_dim // 2, 2, hidden],
                            "dst_shape": [num_heads * head_dim, hidden],
                        }
                    },
                ],
                "sharding": ["tp", None],
            }
        ]
    )

    dst, _ = _assert_round_trip(
        manifest, {"model.layers.0.self_attn.q_proj.weight": w_q}
    )
    expected = (
        w_q.reshape(num_heads, 2, head_dim // 2, hidden)
        .permute(0, 2, 1, 3)
        .reshape(num_heads * head_dim, hidden)
    )
    assert torch.equal(dst["layers.0.attn.q_proj.weight"], expected)


def test_showcase_2_moe_tp_shard_local_gate_up_interleaving() -> None:
    """Showcase 2: MoE TP shard-local gate/up interleaving.

    Pipeline: pad -> concat -> reshape -> transpose -> reshape -> transpose.
    Produces per-TP-shard interleaved [gate_shard_i, up_shard_i] layout with
    transposed inner dims (E, H, 2 * inter_padded).
    """
    num_experts = 4
    inter = 6
    pad_inter = 2  # inter_padded = 8, divisible by tp_size = 2
    inter_padded = inter + pad_inter
    tp_size = 2
    hidden = 10

    gate = torch.arange(num_experts * inter * hidden, dtype=torch.bfloat16).reshape(
        num_experts, inter, hidden
    )
    up = (
        torch.arange(num_experts * inter * hidden, dtype=torch.bfloat16).reshape(
            num_experts, inter, hidden
        )
        + 1000
    )

    manifest = _make_manifest(
        [
            {
                "source_keys": ["experts.gate_proj", "experts.up_proj"],
                "target_keys": ["experts.w13_weight"],
                "ops": [
                    {"pad": {"axis": 1, "sizes": [0, pad_inter]}},
                    {
                        "concat": {
                            "axis": 1,
                            "sizes": [inter_padded, inter_padded],
                        }
                    },
                    {
                        "reshape": {
                            "src_shape": [num_experts, 2 * inter_padded, hidden],
                            "dst_shape": [
                                num_experts,
                                2,
                                tp_size,
                                inter_padded // tp_size,
                                hidden,
                            ],
                        }
                    },
                    {"transpose": {"axes": [0, 2, 1, 3, 4]}},
                    {
                        "reshape": {
                            "src_shape": [
                                num_experts,
                                tp_size,
                                2,
                                inter_padded // tp_size,
                                hidden,
                            ],
                            "dst_shape": [num_experts, 2 * inter_padded, hidden],
                        }
                    },
                    {"transpose": {"axes": [0, 2, 1]}},
                ],
                "sharding": ["ep", None, "moe_tp"],
            }
        ],
        expert_axis_grouped=True,
    )

    dst, _ = _assert_round_trip(
        manifest, {"experts.gate_proj": gate, "experts.up_proj": up}
    )
    w13 = dst["experts.w13_weight"]
    assert w13.shape == (num_experts, hidden, 2 * inter_padded)

    # Verify each TP shard has [gate_shard, up_shard] transposed as (H, 2*shard).
    gate_pad = torch.cat(
        [gate, torch.zeros(num_experts, pad_inter, hidden, dtype=gate.dtype)], dim=1
    )
    up_pad = torch.cat(
        [up, torch.zeros(num_experts, pad_inter, hidden, dtype=up.dtype)], dim=1
    )
    shard_w = inter_padded // tp_size
    for rank in range(tp_size):
        s_start = rank * 2 * shard_w
        shard = w13[:, :, s_start : s_start + 2 * shard_w]
        expected_shard = torch.cat(
            [
                gate_pad[:, rank * shard_w : (rank + 1) * shard_w, :],
                up_pad[:, rank * shard_w : (rank + 1) * shard_w, :],
            ],
            dim=1,
        ).transpose(1, 2)
        assert torch.equal(shard, expected_shard)


# ---------------------------------------------------------------------------
# Cardinalities: 1->1, M->1, 1->N, M->N and API wrappers
# ---------------------------------------------------------------------------


def test_round_trip_all_cardinalities() -> None:
    a = torch.randn(4, 6)
    b = torch.randn(2, 6)
    c = torch.randn(6, 4)
    d = torch.randn(6, 4)

    manifest = _make_manifest(
        [
            # 1 -> 1 identity
            {
                "source_keys": ["id_in"],
                "target_keys": ["id_out"],
                "ops": [],
                "sharding": [None, None],
            },
            # M -> 1 concat
            {
                "source_keys": ["m1_a", "m1_b"],
                "target_keys": ["m1_out"],
                "ops": [{"concat": {"axis": 0, "sizes": [4, 2]}}],
            },
            # 1 -> N split
            {
                "source_keys": ["s_in"],
                "target_keys": ["s_out0", "s_out1"],
                "ops": [{"split": {"axis": 0, "sizes": [2, 4]}}],
            },
            # M -> N (stack -> transpose -> unstack)
            {
                "source_keys": ["mn_0", "mn_1"],
                "target_keys": ["mn_out0", "mn_out1", "mn_out2", "mn_out3"],
                "ops": [
                    {"stack": {"axis": 0}},
                    {"transpose": {"axes": [2, 1, 0]}},
                    {"unstack": {"axis": 0}},
                ],
            },
        ]
    )
    src = {
        "id_in": a,
        "m1_a": a,
        "m1_b": b,
        "s_in": torch.cat([a, b], dim=0),
        "mn_0": c,
        "mn_1": d,
    }
    _assert_round_trip(manifest, src)


def test_api_wrappers_and_mesh_compat() -> None:
    manifest = _make_manifest(
        [{"source_keys": ["w"], "target_keys": ["w_out"], "ops": []}]
    )
    manifest_json = json.dumps(manifest)
    t = torch.tensor([1.0, 2.0])

    # mesh argument is accepted for protocol compatibility
    out1 = from_hf(manifest_json, {"w": t}, mesh=object())
    out2 = from_hf_impl(manifest, {"w": t}, mesh=object())
    out3 = execute_manifest(manifest_json, {"w": t}, mesh=object())
    assert torch.equal(out1["w_out"], t)
    assert torch.equal(out2["w_out"], t)
    assert torch.equal(out3["w_out"], t)

    inv = invert_manifest(manifest_json)
    back1 = to_hf(json.dumps(inv), out1)
    back2 = to_hf_impl(inv, out1)
    assert torch.equal(back1["w"], t)
    assert torch.equal(back2["w"], t)

    with pytest.raises(TypeError):
        from_hf(manifest, {"w": t})  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        to_hf(inv, out1)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# tied_keys and dropped_keys
# ---------------------------------------------------------------------------


def test_tied_keys_and_dropped_keys() -> None:
    embed = torch.randn(16, 8)
    norm = torch.randn(8)

    manifest = _make_manifest(
        [
            {
                "source_keys": ["model.embed_tokens.weight"],
                "target_keys": ["model.embed_tokens.weight"],
                "ops": [],
            },
            {
                "source_keys": ["model.norm.weight"],
                "target_keys": ["model.norm.weight"],
                "ops": [],
            },
            {
                "source_keys": ["visual.patch_embed.weight"],
                "target_keys": ["visual.patch_embed.weight"],
                "ops": [],
            },
        ],
        tied_keys=[
            ["model.embed_tokens.weight", "lm_head.weight"],
            ["model.embed_tokens.weight", "dropped_tie.weight"],
            ["missing_src.weight", "ignored_tie.weight"],
        ],
        dropped_keys=["visual.patch_embed.weight", "dropped_tie.weight"],
    )

    # Note: dropped source key is intentionally omitted from src_weights.
    out = from_hf_impl(
        manifest,
        {"model.embed_tokens.weight": embed, "model.norm.weight": norm},
    )
    assert set(out.keys()) == {
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.norm.weight",
    }
    assert torch.equal(out["lm_head.weight"], embed)
    # Tied copy is independent of the source tensor.
    out["lm_head.weight"].zero_()
    assert torch.equal(out["model.embed_tokens.weight"], embed)

    # Inverse skips the dropped rule cleanly.
    inv = invert_manifest(manifest)
    recovered = to_hf_impl(inv, out)
    assert set(recovered.keys()) == {
        "model.embed_tokens.weight",
        "model.norm.weight",
    }
    assert torch.equal(recovered["model.embed_tokens.weight"], embed)
    assert torch.equal(recovered["model.norm.weight"], norm)


# ---------------------------------------------------------------------------
# Strict ValueError validation checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_manifest",
    [
        "not a mapping",
        {"schema_version": "wrong_version", "rules": []},
        {"rules": []},
        {"schema_version": SCHEMA_VERSION},
        {"schema_version": SCHEMA_VERSION, "rules": [], "extra_top": 1},
        {"schema_version": SCHEMA_VERSION, "rules": [], "expert_axis_grouped": 1},
        {"schema_version": SCHEMA_VERSION, "rules": [], "stripes": [1, 2]},
        {"schema_version": SCHEMA_VERSION, "rules": [], "stripes": None},
        {
            "schema_version": SCHEMA_VERSION,
            "rules": [],
            "stripes": {"qkv": {"axis": 0}},
        },
        {"schema_version": SCHEMA_VERSION, "rules": [], "stripes": {"tp_size": 0}},
        {"schema_version": SCHEMA_VERSION, "rules": [], "stripes": {"tp_size": -2}},
        {"schema_version": SCHEMA_VERSION, "rules": [], "stripes": {"tp_size": True}},
        {"schema_version": SCHEMA_VERSION, "rules": [], "stripes": {"tp_size": "4"}},
        {"schema_version": SCHEMA_VERSION, "rules": [], "stripes": {"tp_size": 2.0}},
        {"schema_version": SCHEMA_VERSION, "rules": [], "stripes": {1: 4}},
        {"schema_version": SCHEMA_VERSION, "rules": [], "tied_keys": ["not_a_pair"]},
        {"schema_version": SCHEMA_VERSION, "rules": [], "tied_keys": [["a"]]},
        {"schema_version": SCHEMA_VERSION, "rules": [], "dropped_keys": [123]},
        {"schema_version": SCHEMA_VERSION, "rules": "not_a_list"},
        {
            "schema_version": SCHEMA_VERSION,
            "rules": [
                {
                    "source_keys": ["a"],
                    "target_keys": ["b"],
                    "ops": [],
                    "unknown_rule_key": True,
                }
            ],
        },
        {
            "schema_version": SCHEMA_VERSION,
            "rules": [{"source_keys": [], "target_keys": ["b"], "ops": []}],
        },
        {
            "schema_version": SCHEMA_VERSION,
            "rules": [{"source_keys": ["a"], "target_keys": [1], "ops": []}],
        },
        {
            "schema_version": SCHEMA_VERSION,
            "rules": [
                {
                    "source_keys": ["a"],
                    "target_keys": ["b"],
                    "ops": [],
                    "sharding": [123],
                }
            ],
        },
        {
            "schema_version": SCHEMA_VERSION,
            "rules": [
                {
                    "source_keys": ["a"],
                    "target_keys": ["b"],
                    "ops": [{"unknown_op": {}}],
                }
            ],
        },
        {
            "schema_version": SCHEMA_VERSION,
            "rules": [
                {
                    "source_keys": ["a"],
                    "target_keys": ["b"],
                    "ops": [{"stack": {"axis": 0}, "unstack": {"axis": 0}}],
                }
            ],
        },
        {
            "schema_version": SCHEMA_VERSION,
            "rules": [
                {
                    "source_keys": ["a"],
                    "target_keys": ["b"],
                    "ops": [{"stack": {"axis": True}}],
                }
            ],
        },
        {
            "schema_version": SCHEMA_VERSION,
            "rules": [
                {
                    "source_keys": ["a"],
                    "target_keys": ["b"],
                    "ops": [{"scale": {"factor": 0.0}}],
                }
            ],
        },
        {
            "schema_version": SCHEMA_VERSION,
            "rules": [
                {
                    "source_keys": ["a"],
                    "target_keys": ["b"],
                    "ops": [{"scale": {"factor": float("nan")}}],
                }
            ],
        },
        {
            "schema_version": SCHEMA_VERSION,
            "rules": [
                {
                    "source_keys": ["a"],
                    "target_keys": ["b"],
                    "ops": [{"scale": {"factor": True}}],
                }
            ],
        },
        {
            "schema_version": SCHEMA_VERSION,
            "rules": [
                {
                    "source_keys": ["a"],
                    "target_keys": ["b"],
                    "ops": [{"reshape": {"src_shape": [2, 3], "dst_shape": [2, 4]}}],
                }
            ],
        },
        {
            "schema_version": SCHEMA_VERSION,
            "rules": [
                {
                    "source_keys": ["a"],
                    "target_keys": ["b"],
                    "ops": [{"reshape": {"src_shape": [-1, 3], "dst_shape": [2, 3]}}],
                }
            ],
        },
        {
            "schema_version": SCHEMA_VERSION,
            "rules": [
                {
                    "source_keys": ["a"],
                    "target_keys": ["b"],
                    "ops": [{"transpose": {"axes": [0, 0]}}],
                }
            ],
        },
        {
            "schema_version": SCHEMA_VERSION,
            "rules": [
                {
                    "source_keys": ["a"],
                    "target_keys": ["b"],
                    "ops": [{"pad": {"axis": 0, "sizes": [1]}}],
                }
            ],
        },
        {
            "schema_version": SCHEMA_VERSION,
            "rules": [
                {
                    "source_keys": ["a"],
                    "target_keys": ["b"],
                    "ops": [{"pad": {"axis": 0, "sizes": [-1, 2]}}],
                }
            ],
        },
        {
            "schema_version": SCHEMA_VERSION,
            "rules": [
                {
                    "source_keys": ["a"],
                    "target_keys": ["b"],
                    "ops": [{"cast": {"src_dtype": "float32", "dst_dtype": "int999"}}],
                }
            ],
        },
        {
            "schema_version": SCHEMA_VERSION,
            "rules": [
                {
                    "source_keys": ["a"],
                    "target_keys": ["b"],
                    "ops": [{"repeat": {"axis": 0, "times": 0}}],
                }
            ],
        },
        {
            "schema_version": SCHEMA_VERSION,
            "rules": [
                {
                    "source_keys": ["a"],
                    "target_keys": ["b"],
                    "ops": [{"concat": {"axis": 0, "sizes": [0, 2]}}],
                }
            ],
        },
        {
            "schema_version": SCHEMA_VERSION,
            "rules": [
                {
                    "source_keys": ["a"],
                    "target_keys": ["b"],
                    "ops": [{"stack": {"axis": 0, "extra": 1}}],
                }
            ],
        },
    ],
)
def test_validate_manifest_rejects_invalid_schemas(bad_manifest: Any) -> None:
    with pytest.raises(ValueError):
        validate_manifest(bad_manifest)


def test_validate_manifest_accepts_strict_mode_edge_cases() -> None:
    """MappingProxyType, tuple sequences, int scale.factor, and 0-D ops."""
    manifest = MappingProxyType(
        {
            "schema_version": SCHEMA_VERSION,
            "expert_axis_grouped": False,
            "stripes": MappingProxyType({"tp_size": 2}),
            "tied_keys": (("a", "b"),),
            "dropped_keys": ("unused",),
            "rules": (
                MappingProxyType(
                    {
                        "source_keys": ("x",),
                        "target_keys": ("y",),
                        "sharding": ("tp", None),
                        "ops": (
                            MappingProxyType(
                                {"scale": MappingProxyType({"factor": 2})}
                            ),
                            {"reshape": {"src_shape": (), "dst_shape": ()}},
                            {"transpose": {"axes": ()}},
                            {"pad": {"axis": 0, "sizes": (0, 1)}},
                            {"concat": {"axis": 0, "sizes": (2, 2)}},
                        ),
                    }
                ),
            ),
        }
    )
    validate_manifest(manifest)


@pytest.mark.parametrize(
    "bad_manifest",
    [
        {"schema_version": SCHEMA_VERSION, "rules": [], "expert_axis_grouped": None},
        {"schema_version": SCHEMA_VERSION, "rules": [], "tied_keys": None},
        {"schema_version": SCHEMA_VERSION, "rules": [], "dropped_keys": None},
        {
            "schema_version": SCHEMA_VERSION,
            "rules": [
                {
                    "source_keys": ["a"],
                    "target_keys": ["b"],
                    "ops": [],
                    "sharding": None,
                }
            ],
        },
        {
            "schema_version": SCHEMA_VERSION,
            "rules": [
                {
                    "source_keys": ["a"],
                    "target_keys": ["b"],
                    "ops": [{"scale": {"factor": float("inf")}}],
                }
            ],
        },
        {
            "schema_version": SCHEMA_VERSION,
            "rules": [
                {
                    "source_keys": ["a"],
                    "target_keys": ["b"],
                    "ops": [{"scale": {"factor": -float("inf")}}],
                }
            ],
        },
        {
            "schema_version": SCHEMA_VERSION,
            "rules": [
                {
                    "source_keys": ["a"],
                    "target_keys": ["b"],
                    "ops": [{"scale": {"factor": -0.0}}],
                }
            ],
        },
        {
            "schema_version": SCHEMA_VERSION,
            "rules": [
                {
                    "source_keys": ["a"],
                    "target_keys": ["b"],
                    "ops": [{}],
                }
            ],
        },
        {
            "schema_version": SCHEMA_VERSION,
            "rules": [
                {
                    "source_keys": ["a"],
                    "target_keys": ["b"],
                    "ops": [{123: {"axis": 0}}],
                }
            ],
        },
    ],
)
def test_validate_manifest_rejects_strict_mode_edge_cases(bad_manifest: Any) -> None:
    with pytest.raises(ValueError):
        validate_manifest(bad_manifest)


def test_runtime_shape_size_and_dtype_mismatches_raise_value_error() -> None:
    # reshape src_shape mismatch
    m_reshape = _make_manifest(
        [
            {
                "source_keys": ["x"],
                "target_keys": ["y"],
                "ops": [{"reshape": {"src_shape": [2, 3], "dst_shape": [3, 2]}}],
            }
        ]
    )
    with pytest.raises(ValueError, match="reshape"):
        execute_manifest(m_reshape, {"x": torch.zeros(3, 2)})

    # concat width mismatch
    m_concat = _make_manifest(
        [
            {
                "source_keys": ["a", "b"],
                "target_keys": ["c"],
                "ops": [{"concat": {"axis": 0, "sizes": [4, 2]}}],
            }
        ]
    )
    with pytest.raises(ValueError, match="concat"):
        execute_manifest(m_concat, {"a": torch.zeros(3, 2), "b": torch.zeros(2, 2)})

    # split sum(sizes) mismatch
    m_split = _make_manifest(
        [
            {
                "source_keys": ["c"],
                "target_keys": ["a", "b"],
                "ops": [{"split": {"axis": 0, "sizes": [4, 2]}}],
            }
        ]
    )
    with pytest.raises(ValueError, match="split"):
        execute_manifest(m_split, {"c": torch.zeros(5, 2)})

    # cast src_dtype mismatch
    m_cast = _make_manifest(
        [
            {
                "source_keys": ["x"],
                "target_keys": ["y"],
                "ops": [{"cast": {"src_dtype": "bfloat16", "dst_dtype": "float32"}}],
            }
        ]
    )
    with pytest.raises(ValueError, match="cast"):
        execute_manifest(m_cast, {"x": torch.zeros(2, dtype=torch.float32)})

    # unpad exceeding dim
    m_unpad = _make_manifest(
        [
            {
                "source_keys": ["x"],
                "target_keys": ["y"],
                "ops": [{"unpad": {"axis": 0, "sizes": [3, 3]}}],
            }
        ]
    )
    with pytest.raises(ValueError, match="unpad"):
        execute_manifest(m_unpad, {"x": torch.zeros(4, 2)})

    # unrepeat non-divisible dim
    m_unrepeat = _make_manifest(
        [
            {
                "source_keys": ["x"],
                "target_keys": ["y"],
                "ops": [{"unrepeat": {"axis": 0, "times": 3}}],
            }
        ]
    )
    with pytest.raises(ValueError, match="unrepeat"):
        execute_manifest(m_unrepeat, {"x": torch.zeros(5, 2)})

    # invalid JSON string
    with pytest.raises(ValueError, match="Invalid manifest JSON"):
        execute_manifest("{not valid json", {})


def test_chained_rules_gqa_qkv_tp_interleaving() -> None:
    """Chained rules for heterogeneous GQA qkv_proj with KV replication."""
    tp_size = 4
    num_q_heads = 8  # 2 Q heads per TP rank
    num_kv_heads = 2  # replicated 2x across 4 TP ranks
    kv_rep = tp_size // num_kv_heads
    head_dim = 4
    hidden = 6

    q_shard = (num_q_heads // tp_size) * head_dim
    k_shard = head_dim
    v_shard = head_dim

    q = torch.arange(num_q_heads * head_dim * hidden, dtype=torch.bfloat16).reshape(
        num_q_heads * head_dim, hidden
    )
    k = (
        torch.arange(num_kv_heads * head_dim * hidden, dtype=torch.bfloat16).reshape(
            num_kv_heads * head_dim, hidden
        )
        + 1000
    )
    v = (
        torch.arange(num_kv_heads * head_dim * hidden, dtype=torch.bfloat16).reshape(
            num_kv_heads * head_dim, hidden
        )
        + 2000
    )

    manifest = _make_manifest(
        [
            {
                "source_keys": ["q_proj.weight"],
                "target_keys": ["__q_tp"],
                "ops": [
                    {
                        "reshape": {
                            "src_shape": [num_q_heads * head_dim, hidden],
                            "dst_shape": [tp_size, q_shard, hidden],
                        }
                    }
                ],
            },
            {
                "source_keys": ["k_proj.weight"],
                "target_keys": ["__k_tp"],
                "ops": [
                    {
                        "reshape": {
                            "src_shape": [num_kv_heads * head_dim, hidden],
                            "dst_shape": [num_kv_heads, k_shard, hidden],
                        }
                    },
                    {"repeat": {"axis": 0, "times": kv_rep}},
                ],
            },
            {
                "source_keys": ["v_proj.weight"],
                "target_keys": ["__v_tp"],
                "ops": [
                    {
                        "reshape": {
                            "src_shape": [num_kv_heads * head_dim, hidden],
                            "dst_shape": [num_kv_heads, v_shard, hidden],
                        }
                    },
                    {"repeat": {"axis": 0, "times": kv_rep}},
                ],
            },
            {
                "source_keys": ["__q_tp", "__k_tp", "__v_tp"],
                "target_keys": ["qkv_proj.weight"],
                "ops": [
                    {"concat": {"axis": 1, "sizes": [q_shard, k_shard, v_shard]}},
                    {
                        "reshape": {
                            "src_shape": [
                                tp_size,
                                q_shard + k_shard + v_shard,
                                hidden,
                            ],
                            "dst_shape": [
                                tp_size * (q_shard + k_shard + v_shard),
                                hidden,
                            ],
                        }
                    },
                ],
                "sharding": ["tp", None],
            },
        ],
        stripes={"tp_size": tp_size, "kv_tp_size": tp_size},
    )

    src = {"q_proj.weight": q, "k_proj.weight": k, "v_proj.weight": v}
    dst, recovered = _assert_round_trip(manifest, src)

    # Intermediate keys must be pruned from output.
    assert set(dst.keys()) == {"qkv_proj.weight"}
    assert set(recovered.keys()) == set(src.keys())

    # Each contiguous TP shard holds [Q_r; K_{r // kv_rep}; V_{r // kv_rep}].
    qkv = dst["qkv_proj.weight"]
    shard_rows = q_shard + k_shard + v_shard
    assert qkv.shape == (tp_size * shard_rows, hidden)
    for rank in range(tp_size):
        actual_shard = qkv[rank * shard_rows : (rank + 1) * shard_rows]
        kv_idx = rank // kv_rep
        expected_shard = torch.cat(
            [
                q[rank * q_shard : (rank + 1) * q_shard],
                k[kv_idx * k_shard : (kv_idx + 1) * k_shard],
                v[kv_idx * v_shard : (kv_idx + 1) * v_shard],
            ],
            dim=0,
        )
        assert torch.equal(actual_shard, expected_shard)


def test_same_name_source_key_reused_across_rules_not_pruned() -> None:
    """A source key written to the same name and reused by a later rule stays in out."""
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    manifest = _make_manifest(
        [
            {
                "source_keys": ["x"],
                "target_keys": ["x"],
                "ops": [{"scale": {"factor": 2.0}}],
            },
            {
                "source_keys": ["x"],
                "target_keys": ["y"],
                "ops": [{"scale": {"factor": 3.0}}],
            },
        ]
    )

    out = execute_manifest(manifest, {"x": x})
    assert set(out.keys()) == {"x", "y"}
    assert torch.equal(out["x"], x * 2.0)
    assert torch.equal(out["y"], x * 3.0)


def test_dropped_downstream_rule_prunes_tmp_intermediates() -> None:
    """Skipping a downstream rule via dropped_keys must not leak __tmp__:* keys."""
    q = torch.randn(4, 6)
    norm = torch.randn(6)
    manifest = _make_manifest(
        [
            {
                "source_keys": ["q_proj.weight"],
                "target_keys": ["__tmp__:q_tp"],
                "ops": [{"reshape": {"src_shape": [4, 6], "dst_shape": [2, 2, 6]}}],
            },
            {
                "source_keys": ["__tmp__:q_tp", "k_proj.weight"],
                "target_keys": ["qkv_proj.weight"],
                "ops": [{"concat": {"axis": 1, "sizes": [2, 2]}}],
            },
            {
                "source_keys": ["norm.weight"],
                "target_keys": ["norm.weight"],
                "ops": [],
            },
        ],
        dropped_keys=["k_proj.weight"],
    )

    out = execute_manifest(manifest, {"q_proj.weight": q, "norm.weight": norm})
    assert set(out.keys()) == {"norm.weight"}
    assert torch.equal(out["norm.weight"], norm)


def test_zero_jax_imports() -> None:
    import vllm_torchtpu.rl as rl_pkg
    import vllm_torchtpu.rl.weight_conversion as wc_mod

    for mod in (rl_pkg, wc_mod):
        tree = ast.parse(inspect.getsource(mod))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("jax")
            elif isinstance(node, ast.ImportFrom):
                assert not (node.module and node.module.startswith("jax"))
