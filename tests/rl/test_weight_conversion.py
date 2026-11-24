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
import json
from pathlib import Path
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


def _m(rules: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "expert_axis_grouped": False,
        "stripes": {},
        "tied_keys": [],
        "dropped_keys": [],
        "rules": rules,
        **kw,
    }


def _op_m(
    op: dict[str, Any], src: list[str] | None = None, dst: list[str] | None = None
) -> dict[str, Any]:
    return _m([{"source_keys": src or ["a"], "target_keys": dst or ["b"], "ops": [op]}])


def _round_trip(
    manifest: dict[str, Any], src: dict[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    validate_manifest(manifest)
    dst = from_hf(json.dumps(manifest), src)
    inv = invert_manifest(manifest)
    validate_manifest(inv)
    rec = to_hf(json.dumps(inv), dst)
    assert set(rec) == set(src)
    for k, v in src.items():
        assert rec[k].dtype == v.dtype and rec[k].shape == v.shape
        if v.dtype == torch.bfloat16:
            assert torch.equal(rec[k], v)
        else:
            torch.testing.assert_close(rec[k], v, rtol=0, atol=0)
    return dst, rec


def test_all_12_ops_round_trip() -> None:
    g = torch.Generator().manual_seed(0)

    # stack / unstack (positive & negative axis)
    es = [torch.randn(4, 6, generator=g, dtype=torch.bfloat16) for _ in range(3)]
    for axis in (0, -1):
        m = _op_m({"stack": {"axis": axis}}, ["e0", "e1", "e2"], ["stacked"])
        dst, _ = _round_trip(m, {"e0": es[0], "e1": es[1], "e2": es[2]})
        assert torch.equal(dst["stacked"], torch.stack(es, dim=axis))

    # concat / split
    a = torch.randn(4, 8, generator=g, dtype=torch.bfloat16)
    b = torch.randn(6, 8, generator=g, dtype=torch.bfloat16)
    dst, _ = _round_trip(
        _op_m({"concat": {"axis": 0, "sizes": [4, 6]}}, ["a", "b"], ["c"]),
        {"a": a, "b": b},
    )
    assert torch.equal(dst["c"], torch.cat([a, b], dim=0))

    # reshape & asymmetric transpose
    x = torch.randn(2, 3, 4, generator=g, dtype=torch.bfloat16)
    dst, _ = _round_trip(
        _op_m({"reshape": {"src_shape": [2, 3, 4], "dst_shape": [6, 4]}}), {"a": x}
    )
    assert torch.equal(dst["b"], x.reshape(6, 4))
    dst, _ = _round_trip(_op_m({"transpose": {"axes": [2, 0, 1]}}), {"a": x})
    assert torch.equal(dst["b"], x.permute(2, 0, 1))

    # pad / slice
    p = torch.randn(5, 4, generator=g, dtype=torch.bfloat16)
    dst, _ = _round_trip(_op_m({"pad": {"axis": 0, "sizes": [1, 2]}}), {"a": p})
    assert dst["b"].shape == (8, 4) and torch.equal(dst["b"][1:6], p)
    assert torch.all(dst["b"][:1] == 0) and torch.all(dst["b"][6:] == 0)

    # repeat / mean_pool
    r = torch.randn(2, 4, generator=g, dtype=torch.bfloat16)
    dst, _ = _round_trip(_op_m({"repeat": {"axis": 0, "times": 3}}), {"a": r})
    assert torch.equal(dst["b"], r.repeat_interleave(3, dim=0))

    # scale (bf16 via fp32 compute + fp64 precision preserved)
    s32 = torch.randn(4, 4, generator=g, dtype=torch.float32)
    dst, _ = _round_trip(_op_m({"scale": {"factor": 4.0}}), {"a": s32})
    torch.testing.assert_close(dst["b"], s32 * 4.0)
    s64 = torch.tensor([1.0 + 1e-12, -2.0 - 3e-13], dtype=torch.float64)
    dst64, _ = _round_trip(_op_m({"scale": {"factor": 2.0}}), {"a": s64})
    assert torch.equal(dst64["b"], s64 * 2.0)

    # cast (bf16 <-> fp32, and fp8 if supported)
    c = torch.randn(4, 4, generator=g, dtype=torch.bfloat16)
    dst, _ = _round_trip(
        _op_m({"cast": {"src_dtype": "bfloat16", "dst_dtype": "float32"}}), {"a": c}
    )
    assert dst["b"].dtype == torch.float32
    if hasattr(torch, "float8_e4m3fn"):
        f8 = torch.tensor([1.0, -0.5], dtype=torch.float32).to(torch.float8_e4m3fn)
        dst8, _ = _round_trip(
            _op_m({"cast": {"src_dtype": "float8_e4m3fn", "dst_dtype": "float32"}}),
            {"a": f8},
        )
        assert dst8["b"].dtype == torch.float32


def test_showcases_rope_and_moe_tp_interleaving() -> None:
    """Showcase 1 (RoPE complex-pair perm) and Showcase 2 (multi-tensor unary pad)."""
    g = torch.Generator().manual_seed(7)
    # Showcase 1: [H*D, C] -> swap real/imag halves
    w = torch.randn(4 * 8, 16, generator=g, dtype=torch.bfloat16)
    m1 = _m(
        [
            {
                "source_keys": ["q.weight"],
                "target_keys": ["q_rope.weight"],
                "ops": [
                    {"reshape": {"src_shape": [32, 16], "dst_shape": [4, 2, 4, 16]}},
                    {"transpose": {"axes": [0, 2, 1, 3]}},
                    {"reshape": {"src_shape": [4, 4, 2, 16], "dst_shape": [32, 16]}},
                ],
            }
        ]
    )
    _round_trip(m1, {"q.weight": w})

    # Showcase 2: 2 inputs -> unary pad broadcasts to both -> concat -> interleave
    gate = torch.randn(4, 6, 8, generator=g, dtype=torch.bfloat16)
    up = torch.randn(4, 6, 8, generator=g, dtype=torch.bfloat16)
    m2 = _m(
        [
            {
                "source_keys": ["gate", "up"],
                "target_keys": ["w13"],
                "ops": [
                    {"pad": {"axis": 1, "sizes": [0, 2]}},
                    {"reshape": {"src_shape": [4, 8, 8], "dst_shape": [4, 2, 4, 8]}},
                    {"concat": {"axis": 2, "sizes": [4, 4]}},
                    {"reshape": {"src_shape": [4, 2, 8, 8], "dst_shape": [4, 16, 8]}},
                ],
                "sharding": ["ep", "moe_tp", None],
            }
        ],
        expert_axis_grouped=True,
        stripes={"ep_size": 2, "moe_tp_size": 2},
    )
    dst2, _ = _round_trip(m2, {"gate": gate, "up": up})
    assert dst2["w13"].shape == (4, 16, 8)


def test_cardinalities_and_api_wrappers() -> None:
    """1->1, N->1, 1->N, N->M, plus from_hf/to_hf/mesh and TypeError checks."""
    g = torch.Generator().manual_seed(11)
    a = torch.randn(4, 4, generator=g, dtype=torch.bfloat16)
    b = torch.randn(4, 4, generator=g, dtype=torch.bfloat16)
    m = _m(
        [
            {
                "source_keys": ["a", "b"],
                "target_keys": ["c", "d"],
                "ops": [
                    {"concat": {"axis": 0, "sizes": [4, 4]}},
                    {"split": {"axis": 0, "sizes": [2, 6]}},
                ],
            }
        ]
    )
    dst, _ = _round_trip(m, {"a": a, "b": b})
    assert dst["c"].shape == (2, 4) and dst["d"].shape == (6, 4)
    assert torch.equal(from_hf_impl(m, {"a": a, "b": b}, mesh=object())["c"], dst["c"])
    inv = invert_manifest(m)
    assert torch.equal(to_hf_impl(inv, dst)["a"], a)
    with pytest.raises(TypeError, match="from_hf expects dst_rule to be str"):
        from_hf(m, {"a": a, "b": b})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="to_hf expects src_rule to be str"):
        to_hf(inv, dst)  # type: ignore[arg-type]


def test_tied_keys_dropped_keys_and_chaining_invariants() -> None:
    emb = torch.randn(8, 4, dtype=torch.bfloat16)
    norm = torch.randn(4, dtype=torch.bfloat16)
    m = _m(
        [
            {"source_keys": ["embed"], "target_keys": ["embed"], "ops": []},
            {"source_keys": ["norm"], "target_keys": ["norm"], "ops": []},
            {"source_keys": ["drop_src"], "target_keys": ["__tmp__:d"], "ops": []},
            {"source_keys": ["__tmp__:d"], "target_keys": ["drop_dst"], "ops": []},
        ],
        tied_keys=[["embed", "lm_head"]],
        dropped_keys=["drop_src"],
    )
    out = execute_manifest(m, {"embed": emb, "norm": norm})
    assert set(out) == {"embed", "norm", "lm_head"} and torch.equal(out["lm_head"], emb)
    inv = invert_manifest(
        _m(
            [
                {"source_keys": ["embed"], "target_keys": ["embed"], "ops": []},
                {"source_keys": ["norm"], "target_keys": ["norm"], "ops": []},
            ],
            tied_keys=[["embed", "lm_head"]],
        )
    )
    assert inv["tied_keys"] == []
    rec = execute_manifest(inv, out)
    assert set(rec) == {"embed", "norm"}

    # Same-name source key reused across rules is not pruned or clobbered.
    x = torch.tensor([1.0, 2.0])
    m_reuse = _m(
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
    res = execute_manifest(m_reuse, {"x": x})
    torch.testing.assert_close(res["x"], x * 2.0)
    torch.testing.assert_close(res["y"], x * 3.0)


def test_chained_rules_gqa_qkv_tp_interleaving() -> None:
    """Option 1 __tmp__:* chaining for heterogeneous GQA QKV + KV replication."""
    g = torch.Generator().manual_seed(99)
    q = torch.randn(8, 16, generator=g, dtype=torch.bfloat16)
    k = torch.randn(2, 16, generator=g, dtype=torch.bfloat16)
    v = torch.randn(2, 16, generator=g, dtype=torch.bfloat16)
    m = _m(
        [
            {
                "source_keys": ["q"],
                "target_keys": ["__tmp__:q"],
                "ops": [{"reshape": {"src_shape": [8, 16], "dst_shape": [4, 2, 16]}}],
            },
            {
                "source_keys": ["k"],
                "target_keys": ["__tmp__:k"],
                "ops": [
                    {"reshape": {"src_shape": [2, 16], "dst_shape": [2, 1, 16]}},
                    {"repeat": {"axis": 0, "times": 2}},
                ],
            },
            {
                "source_keys": ["v"],
                "target_keys": ["__tmp__:v"],
                "ops": [
                    {"reshape": {"src_shape": [2, 16], "dst_shape": [2, 1, 16]}},
                    {"repeat": {"axis": 0, "times": 2}},
                ],
            },
            {
                "source_keys": ["__tmp__:q", "__tmp__:k", "__tmp__:v"],
                "target_keys": ["qkv"],
                "ops": [
                    {"concat": {"axis": 1, "sizes": [2, 1, 1]}},
                    {"reshape": {"src_shape": [4, 4, 16], "dst_shape": [16, 16]}},
                ],
                "sharding": ["tp", None],
            },
        ],
        stripes={"tp_size": 4},
    )
    dst, _ = _round_trip(m, {"q": q, "k": k, "v": v})
    assert set(dst) == {"qkv"} and dst["qkv"].shape == (16, 16)


def test_validate_manifest_strict_edge_cases() -> None:
    """MappingProxyType, tuples, int scale.factor, and 0-D reshape/transpose."""
    m = MappingProxyType(
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
    validate_manifest(m)


@pytest.mark.parametrize(
    "bad",
    [
        "{not json",
        "[]",
        {"schema_version": "wrong", "rules": []},
        {"rules": []},
        {"schema_version": SCHEMA_VERSION, "rules": [], "extra_key": 1},
        {"schema_version": SCHEMA_VERSION, "rules": [], "expert_axis_grouped": "yes"},
        {"schema_version": SCHEMA_VERSION, "rules": [], "expert_axis_grouped": None},
        {"schema_version": SCHEMA_VERSION, "rules": [], "stripes": {"tp": 0}},
        {"schema_version": SCHEMA_VERSION, "rules": [], "stripes": {"tp": True}},
        {"schema_version": SCHEMA_VERSION, "rules": [], "tied_keys": [["only_one"]]},
        {"schema_version": SCHEMA_VERSION, "rules": [], "tied_keys": None},
        {"schema_version": SCHEMA_VERSION, "rules": [], "dropped_keys": None},
        {
            "schema_version": SCHEMA_VERSION,
            "rules": [{"source_keys": [], "target_keys": ["b"], "ops": []}],
        },
        {
            "schema_version": SCHEMA_VERSION,
            "rules": [{"source_keys": ["a"], "target_keys": ["b"]}],
        },
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
                {"source_keys": ["a"], "target_keys": ["b"], "ops": [], "extra": 1}
            ],
        },
        _op_m({"unknown_op": {}}),
        _op_m({"scale": {"factor": 0}}),
        _op_m({"scale": {"factor": True}}),
        _op_m({"scale": {"factor": float("nan")}}),
        _op_m({"scale": {"factor": float("inf")}}),
        _op_m({"scale": {"factor": -float("inf")}}),
        _op_m({"reshape": {"src_shape": [2, 3], "dst_shape": [2, 4]}}),
        _op_m({"reshape": {"src_shape": [-1, 3], "dst_shape": [2, 3]}}),
        _op_m({"transpose": {"axes": [0, 0]}}),
        _op_m({"pad": {"axis": 0, "sizes": [1]}}),
        _op_m({"pad": {"axis": 0, "sizes": [-1, 2]}}),
        _op_m({"cast": {"src_dtype": "float32", "dst_dtype": "int999"}}),
        _op_m({"repeat": {"axis": 0, "times": 0}}),
        _op_m({"concat": {"axis": 0, "sizes": [0, 2]}}),
        _op_m({"stack": {"axis": 0, "extra": 1}}),
    ],
)
def test_validate_manifest_rejects_invalid_schemas(bad: Any) -> None:
    with pytest.raises(ValueError):
        validate_manifest(bad)


def test_runtime_mismatches_raise_errors() -> None:
    t = torch.zeros(4, 4, dtype=torch.float32)
    with pytest.raises(KeyError, match="missing source weight"):
        execute_manifest(_op_m({"scale": {"factor": 2.0}}), {})
    with pytest.raises(ValueError, match="reshape"):
        execute_manifest(
            _op_m({"reshape": {"src_shape": [2, 8], "dst_shape": [16]}}), {"a": t}
        )
    with pytest.raises(ValueError, match="cast"):
        execute_manifest(
            _op_m({"cast": {"src_dtype": "bfloat16", "dst_dtype": "float32"}}), {"a": t}
        )
    with pytest.raises(ValueError, match="concat"):
        execute_manifest(
            _op_m({"concat": {"axis": 0, "sizes": [2, 4]}}, ["a", "b"], ["c"]),
            {"a": t, "b": t},
        )
    with pytest.raises(ValueError, match="axis"):
        execute_manifest(_op_m({"pad": {"axis": 5, "sizes": [0, 1]}}), {"a": t})


def test_rl_package_has_zero_jax_imports() -> None:
    rl_dir = Path(__file__).resolve().parents[2] / "src" / "vllm_torchtpu" / "rl"
    for py_file in rl_dir.glob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(not a.name.startswith("jax") for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                assert not node.module.startswith("jax")
