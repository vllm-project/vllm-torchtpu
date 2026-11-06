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
"""The runner and worker call sites: the dummy-run suppression scope, the
megakernel step inside `forward_model` and the weight-update guard in
`sync_weights`."""

from __future__ import annotations

from typing import Any

import pytest


def test_runner_dummy_run_forward_is_suppressed() -> None:
    """`TPUModelRunner._dummy_run` calls `forward_model` inside `suppressed(self)`."""
    import ast
    from pathlib import Path

    import vllm_torchtpu

    src = (Path(vllm_torchtpu.__file__).parent / "runner" / "tpu_runner.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(src)
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_dummy_run"
    )

    def _is_suppress(item: ast.withitem) -> bool:
        call = item.context_expr
        return (
            isinstance(call, ast.Call)
            and ast.unparse(call.func) == "megakernel_runtime.suppressed"
            and [ast.unparse(a) for a in call.args] == ["self"]
        )

    def _calls_forward(node: ast.AST) -> bool:
        return any(
            isinstance(c, ast.Call) and ast.unparse(c.func) == "self.forward_model"
            for c in ast.walk(node)
        )

    fwd_calls = [
        c
        for c in ast.walk(fn)
        if isinstance(c, ast.Call) and ast.unparse(c.func) == "self.forward_model"
    ]
    assert fwd_calls, "_dummy_run no longer calls self.forward_model"
    guarded = [
        w
        for w in ast.walk(fn)
        if isinstance(w, ast.With)
        and any(_is_suppress(i) for i in w.items)
        and _calls_forward(w)
    ]
    n_guarded = sum(
        1
        for w in guarded
        for c in ast.walk(w)
        if isinstance(c, ast.Call) and ast.unparse(c.func) == "self.forward_model"
    )
    assert n_guarded == len(fwd_calls)


class _FakeIntermediateTensors:
    pass


def _runner_forward_model() -> Any:
    """`TPUModelRunner.forward_model`; compiled from source without torch_tpu."""
    try:
        from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
    except ImportError:
        pass
    else:
        return TPUModelRunner.forward_model
    import ast
    import textwrap
    from pathlib import Path

    import vllm_torchtpu
    from vllm_torchtpu import megakernel_runtime

    path = Path(vllm_torchtpu.__file__).parent / "runner" / "tpu_runner.py"
    src = path.read_text(encoding="utf-8")
    fn = next(
        n
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.FunctionDef) and n.name == "forward_model"
    )
    fn_src = textwrap.dedent(ast.get_source_segment(src, fn) or "")
    ns: dict[str, Any] = {
        "megakernel_runtime": megakernel_runtime,
        "IntermediateTensors": _FakeIntermediateTensors,
    }
    exec(compile(fn_src, str(path), "exec"), ns)
    return ns["forward_model"]


def _raw_outputs() -> list[Any]:
    import torch

    h = torch.arange(6.0).reshape(3, 2)
    aux = [torch.ones(3, 2)]
    return [h, [h], (h,), (h, aux), [h, aux], _FakeIntermediateTensors()]


@pytest.mark.parametrize("raw_idx", range(6))
def test_runner_normalizes_megakernel_output_like_stock(raw_idx: int) -> None:
    """A megakernel step's output goes through the stock normalization."""
    from types import SimpleNamespace

    import torch

    forward_model = _runner_forward_model()
    raw = _raw_outputs()[raw_idx]
    stock_calls: list[dict[str, Any]] = []

    def model(**kw: Any) -> Any:
        stock_calls.append(kw)
        return raw

    class _Driver:
        def __init__(self, hit: bool) -> None:
            self.hit, self.ctx = hit, None

        def try_forward(self, runner: Any, ctx: Any) -> tuple[bool, Any]:
            self.ctx = ctx
            return (True, raw) if self.hit else (False, None)

    ids = torch.zeros((3,), dtype=torch.int32)
    pos = torch.zeros((3,), dtype=torch.int32)
    stock = forward_model(SimpleNamespace(_megakernel=None, model=model), ids, pos)
    assert len(stock_calls) == 1
    drv = _Driver(hit=True)
    mk = forward_model(SimpleNamespace(_megakernel=drv, model=model), ids, pos)
    assert len(stock_calls) == 1, "a megakernel hit must not run the stock model"
    assert drv.ctx.input_ids is ids and drv.ctx.positions is pos
    assert drv.ctx.token_ids is None
    forward_model(
        SimpleNamespace(_megakernel=drv, model=model),
        None,
        pos,
        inputs_embeds=torch.zeros(3, 2),
        megakernel_token_ids=ids,
    )
    assert drv.ctx.input_ids is None and drv.ctx.token_ids is ids
    assert type(mk) is type(stock) and len(mk) == len(stock) == 2
    assert mk[0] is stock[0]
    assert mk[1] is stock[1]


def test_runner_declined_step_passes_model_kwargs_to_stock() -> None:
    from types import SimpleNamespace

    import torch

    forward_model = _runner_forward_model()
    stock_calls: list[dict[str, Any]] = []
    h = torch.zeros(2, 2)

    def model(**kw: Any) -> Any:
        stock_calls.append(kw)
        return h

    class _Decline:
        ctx: Any = None

        def try_forward(self, runner: Any, ctx: Any) -> tuple[bool, Any]:
            self.ctx = ctx
            return False, None

    drv = _Decline()
    runner = SimpleNamespace(_megakernel=drv, model=model)
    ids = torch.zeros((2,), dtype=torch.int32)
    pos = torch.zeros((2,), dtype=torch.int32)
    it = object()
    out = forward_model(
        runner, ids, pos, intermediate_tensors=it, model_kwargs={"extra": 1}
    )
    assert out[0] is h and out[1] is None
    assert drv.ctx.model_kwargs == {"extra": 1}
    assert drv.ctx.intermediate_tensors is it
    assert stock_calls == [
        dict(
            input_ids=ids,
            positions=pos,
            inputs_embeds=None,
            extra=1,
            intermediate_tensors=it,
        )
    ]


def test_worker_sync_weights_runs_the_weight_update_guard() -> None:
    """`TPUWorker.sync_weights` runs the guard before syncing."""
    import ast
    from pathlib import Path

    import vllm_torchtpu

    src = (Path(vllm_torchtpu.__file__).parent / "worker" / "tpu_worker.py").read_text(
        "utf-8"
    )
    fn = next(
        n
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.FunctionDef) and n.name == "sync_weights"
    )
    first = ast.unparse(fn.body[0])
    assert first == "megakernel_runtime.guard_weight_update(self.model_runner)"
