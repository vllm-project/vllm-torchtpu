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
"""Unit tests for compilation guard skipping, epilogue batching, and shape handover."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import sympy
import torch
import torch.fx as fx
from vllm.config import CompilationMode

from vllm_torchtpu.compilation import aot_guard_skip, foreach_epilogue, shape_variants


def _make_config(
    *,
    multimodal: bool = False,
    enforce_eager: bool = False,
    apc: bool = False,
    lora: object = None,
    spec: object = None,
    kv_transfer: object = None,
    dp: int = 1,
    eval_guards: bool = False,
    mode: CompilationMode = CompilationMode.VLLM_COMPILE,
) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            is_multimodal_model=multimodal, enforce_eager=enforce_eager
        ),
        cache_config=SimpleNamespace(enable_prefix_caching=apc),
        lora_config=lora,
        speculative_config=spec,
        kv_transfer_config=kv_transfer,
        parallel_config=SimpleNamespace(data_parallel_size=dp),
        compilation_config=SimpleNamespace(
            mode=mode,
            dynamic_shapes_config=SimpleNamespace(evaluate_guards=eval_guards),
        ),
    )


def test_aot_guard_skip_eligibility_and_install_gate(monkeypatch):
    assert aot_guard_skip.should_enable(_make_config())
    for kw in (
        dict(multimodal=True),
        dict(enforce_eager=True),
        dict(apc=True),
        dict(lora=object()),
        dict(spec=object()),
        dict(kv_transfer=object()),
        dict(dp=2),
        dict(eval_guards=True),
        dict(mode=CompilationMode.STOCK_TORCH_COMPILE),
    ):
        assert not aot_guard_skip.should_enable(_make_config(**kw))

    fn = MagicMock()
    assert aot_guard_skip.disable(fn) and fn.disable_guard_check.call_count == 1
    assert not aot_guard_skip.disable(object())

    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

    compiled_fn = MagicMock()
    monkeypatch.setattr(aot_guard_skip, "_installed", False)
    monkeypatch.setattr(
        TorchCompileWithNoGuardsWrapper,
        "aot_compile",
        lambda self, *a, **kw: compiled_fn,
    )
    aot_guard_skip.install()
    wrapper = SimpleNamespace(vllm_config=_make_config())
    assert TorchCompileWithNoGuardsWrapper.aot_compile(wrapper) is compiled_fn
    assert compiled_fn.disable_guard_check.call_count == 1
    aot_guard_skip.set_active(False)
    assert TorchCompileWithNoGuardsWrapper.aot_compile(wrapper) is compiled_fn
    assert compiled_fn.disable_guard_check.call_count == 1
    eager_fn = torch.compile(lambda x: x + 1.0, backend="eager", fullgraph=True)
    assert torch.equal(eager_fn(torch.ones(2)), torch.full((2,), 2.0))


def test_foreach_epilogue_rewrite_preserves_order_grad_and_overlap_fallback():
    src = (
        "def _apply_mutations(orig_inputs, updated_inputs):\n"
        "    if orig_inputs[2].requires_grad: "
        "orig_inputs[2].detach().copy_(updated_inputs[0])\n"
        "    else: orig_inputs[2].copy_(updated_inputs[0])\n"
        "    orig_inputs[5].copy_(updated_inputs[1])"
    )
    rewritten = foreach_epilogue.rewrite(src)
    assert rewritten is not None and "torch._foreach_copy_(_d, _s)" in rewritten

    fallback_calls: list[str] = []
    ns: dict[str, object] = {
        "torch": torch,
        "_tpu_can_foreach": foreach_epilogue._can_foreach,
        "_tpu_original_apply_mutations": lambda o, u: fallback_calls.append("orig"),
    }
    exec(rewritten, ns)  # noqa: S102
    apply_fn = ns["_apply_mutations"]

    orig = [torch.zeros(2) for _ in range(6)]
    upd = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    apply_fn(orig, upd)  # type: ignore[operator]
    assert (
        torch.equal(orig[2], upd[0])
        and torch.equal(orig[5], upd[1])
        and not fallback_calls
    )

    pool = torch.zeros(6)
    orig[2], orig[5] = pool[0:2], pool[4:6]
    apply_fn(orig, upd)  # type: ignore[operator]
    assert not fallback_calls and pool.tolist() == [1.0, 2.0, 0.0, 0.0, 3.0, 4.0]

    orig[2], orig[5] = pool[0:3], pool[1:4]
    apply_fn(orig, [torch.ones(3), torch.full((3,), 2.0)])  # type: ignore[operator]
    assert fallback_calls == ["orig"]

    fallback_calls.clear()
    orig[2], orig[5] = torch.zeros(2, requires_grad=True), torch.zeros(2)
    apply_fn(orig, upd)  # type: ignore[operator]
    assert fallback_calls == ["orig"]

    for body in (
        "    orig_inputs[1].copy_(updated_inputs[0])\n"
        "    orig_inputs[1].copy_(updated_inputs[1])",
        "    orig_inputs[1].copy_(updated_inputs[0])\n"
        "    orig_inputs[2].copy_(updated_inputs[0])",
    ):
        assert (
            foreach_epilogue.rewrite(
                f"def _apply_mutations(orig_inputs, updated_inputs):\n{body}"
            )
            is None
        )


def test_foreach_epilogue_real_compile_and_deactivation(monkeypatch):
    from torch._functorch._aot_autograd import subclass_codegen as scg
    from torch._functorch.aot_autograd import aot_function

    seen_artifacts: list[str] = []
    orig_exec = scg._compile_and_exec_source

    def spy_exec(source, globals_dict, fn_name, artifact_name, *args, **kwargs):
        seen_artifacts.append(artifact_name)
        return orig_exec(source, globals_dict, fn_name, artifact_name, *args, **kwargs)

    monkeypatch.setattr(foreach_epilogue, "_installed", False)
    monkeypatch.setattr(scg, "_compile_and_exec_source", spy_exec)

    def mut_backend(gm, _example_inputs):
        return aot_function(
            gm, fw_compiler=lambda g, _: g.forward, keep_inference_input_mutations=False
        )

    def mut_step(a: torch.Tensor, b: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        a.add_(delta)
        b.mul_(2.0)
        return a + b

    foreach_epilogue.install()
    try:
        compiled = torch.compile(mut_step, backend=mut_backend, fullgraph=True)
        a, b = torch.ones(4), torch.full((4,), 3.0)
        with torch.no_grad():
            out = compiled(a, b, torch.full((4,), 2.0))
        assert any(name.endswith("_foreach") for name in seen_artifacts)
        assert (a.tolist(), b.tolist(), out.tolist()) == (
            [3.0] * 4,
            [6.0] * 4,
            [9.0] * 4,
        )
    finally:
        foreach_epilogue.set_active(False)

    seen_artifacts.clear()
    compiled_off = torch.compile(
        lambda u, v: (u.add_(1.0), v.add_(2.0))[0], backend=mut_backend, fullgraph=True
    )
    with torch.no_grad():
        compiled_off(torch.zeros(2), torch.zeros(2))
    assert not any(name.endswith("_foreach") for name in seen_artifacts)


def _fx_graph(specs: list[tuple[str, object]]) -> fx.GraphModule:
    g = fx.Graph()
    nodes = [g.placeholder(name) for name, _ in specs]
    for n, (_, val) in zip(nodes, specs):
        n.meta["val"] = val
    g.output(nodes[0] if nodes else None)
    return fx.GraphModule(torch.nn.Module(), g)


def test_bucket_executables_reorders_and_adapts_scalars():
    for b, e in ((0, 0), (1, 0), (64, 32), (512, 256), (96, 95)):
        assert shape_variants.bucket_gate_edge(b) == e
    assert not shape_variants.in_warmup()
    with shape_variants.warmup():
        assert shape_variants.in_warmup()

    sym_s0 = MagicMock(spec=torch.SymInt)
    sym_s0.node = SimpleNamespace(expr=sympy.Symbol("s0") * 2)
    t_a, t_b = (
        torch.zeros((4, 8), dtype=torch.bfloat16),
        torch.zeros((4,), dtype=torch.int32),
    )
    buckets = shape_variants.BucketExecutables()

    g1, g2 = _fx_graph([("a", t_a), ("b", t_b)]), _fx_graph([("b", t_b), ("a", t_a)])
    buckets.put("cache", "k_perm", g1, lambda *args: args)
    assert buckets.get("cache", "k_perm", g2, size=16)("B", "A") == ("A", "B")  # type: ignore[misc]

    buckets.put(
        "cache",
        "k_sym",
        _fx_graph([("a", t_a), ("n2", sym_s0), ("b", t_b)]),
        lambda *a: a,
    )
    assert buckets.get("cache", "k_sym", g1, size=16)("A", "B") == ("A", 32, "B")  # type: ignore[misc]

    buckets.put(
        "cache",
        "k_partial",
        _fx_graph([("a", t_a), ("n2", sym_s0), ("b", t_a), ("c_old", t_b)]),
        lambda *a: a,
    )
    g_curr = _fx_graph([("b", t_a), ("a", t_a), ("c_new", t_b)])
    assert buckets.get("cache", "k_partial", g_curr, size=16)("B", "A", "C") == (  # type: ignore[misc]
        "A",
        32,
        "B",
        "C",
    )

    buckets.put(
        "cache",
        "k_ambig",
        _fx_graph([("u1", t_a), ("n2", sym_s0), ("u2", t_a)]),
        lambda *a: a,
    )
    with pytest.raises(shape_variants.ShapeSpecializationError):
        buckets.get("cache", "k_ambig", _fx_graph([("v1", t_a), ("v2", t_a)]), size=16)

    for bad_expr in (sympy.Symbol("s0") + sympy.Symbol("s1"), sympy.Symbol("s0") / 3):
        bad_sym = MagicMock(spec=torch.SymInt)
        bad_sym.node = SimpleNamespace(expr=bad_expr)
        buckets.put(
            "cache",
            f"k_bad_{bad_expr}",
            _fx_graph([("a", t_a), ("bad", bad_sym)]),
            lambda *a: a,
        )
        with pytest.raises(shape_variants.ShapeSpecializationError):
            buckets.get("cache", f"k_bad_{bad_expr}", _fx_graph([("a", t_a)]), size=16)
    with pytest.raises(shape_variants.ShapeSpecializationError):
        buckets.get("cache", "k_perm", _fx_graph([("a", t_a)]), size=16)
