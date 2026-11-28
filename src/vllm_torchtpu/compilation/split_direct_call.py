# SPDX-License-Identifier: Apache-2.0
"""Run TorchTPU's split graph and its compiled pieces without
`nn.Module.__call__` when no hook could run (VLLM_TPU_SPLIT_DIRECT_CALL).

Every graph the "tpu" backend compiles runs as a `_SplitCompiledExecutable`:
its split `GraphModule` calls one `_WrapperModule` per compiled piece, so each
executable call goes through two `nn.Module.__call__`s (the GraphModule's
error-reporting wrapper and `_call_impl`'s hook checks) before it reaches the
executable. With no hooks registered, no compiled call implementation and no
JIT or FX tracing, `nn.Module.__call__` runs exactly `forward()`, so these
calls go to `forward` directly; any other call takes the usual path. Results
are the same; an exception raised inside the split graph's forward is not
re-reported with its generated-code location.
"""

import ast
import inspect
import textwrap

import torch
from torch.fx._symbolic_trace import is_fx_tracing
from torch.nn.modules import module as nn_module

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


# What the patched call replays: TorchTPU's _SplitCompiledExecutable.__call__
# (as ast.unparse prints it) and the first statement of its
# _maybe_handshake_and_recompile.
_SPLIT_CALL_BODY = """self._maybe_handshake_and_recompile()
if len(args) == 1 and isinstance(args[0], (list, tuple)):
    args = args[0]
return self._split_gm(*args)"""
_HANDSHAKE_FIRST = "if self.recompile_fn is None:\n    return"


def _body(fn) -> list:
    """`fn`'s statements, docstring left out."""
    body = ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0].body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
    ):
        body = body[1:]
    return body


def _plain(m: torch.nn.Module) -> bool:
    """Whether `m(*args)` is `m.forward(*args)`: `_wrapped_call_impl` with no
    compiled call goes to `_call_impl`, whose no-hook path calls `forward`
    (`_slow_forward` while JIT tracing; FX tracing patches the call). Mirrors
    the fast-path condition of torch/nn/modules/module.py `_call_impl`."""
    return (
        m._compiled_call_impl is None
        and not (
            m._backward_hooks
            or m._backward_pre_hooks
            or m._forward_hooks
            or m._forward_pre_hooks
            or nn_module._global_backward_pre_hooks
            or nn_module._global_backward_hooks
            or nn_module._global_forward_hooks
            or nn_module._global_forward_pre_hooks
        )
        and not torch._C._get_tracing_state()
        and not is_fx_tracing()
    )


def install(split_cls: type, wrapper_cls: type) -> None:
    """Patches `split_cls.__call__` (a `_SplitCompiledExecutable`) and
    `wrapper_cls.__call__` (its pieces' `_WrapperModule`)."""
    if getattr(split_cls.__call__, "_tpu_direct", False):
        return
    split_call = split_cls.__call__
    checks = (
        (
            split_call,
            "\n".join(ast.unparse(x) for x in _body(split_call)),
            _SPLIT_CALL_BODY,
        ),
        (
            split_cls._maybe_handshake_and_recompile,
            ast.unparse(_body(split_cls._maybe_handshake_and_recompile)[0]),
            _HANDSHAKE_FIRST,
        ),
    )
    for fn, got, want in checks:
        if got != want:
            raise RuntimeError(
                f"VLLM_TPU_SPLIT_DIRECT_CALL: TorchTPU's {fn.__qualname__} is not "
                f"the code this patch replays:\n{got}"
            )
    if "__call__" in vars(wrapper_cls):
        raise RuntimeError(
            f"VLLM_TPU_SPLIT_DIRECT_CALL: TorchTPU's {wrapper_cls.__qualname__} has "
            "its own __call__, which the patch would replace"
        )

    def call(self, *args, **kwargs):
        gm = self._split_gm
        # GraphModule.recompile installs it; a lazy GraphModule (as split_module
        # builds them inside Dynamo) only does so on its first call.
        wrapped = getattr(type(gm), "_wrapped_call", None)
        if (
            kwargs
            # the handshake path of _maybe_handshake_and_recompile
            or self.recompile_fn is not None
            or wrapped is None
            # a GraphModule of a class with its own __call__
            or wrapped.cls_call is not None
            or not _plain(gm)
        ):
            return split_call(self, *args, **kwargs)
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            args = args[0]
        return gm.forward(*args)

    def wrapper_call(self, *args, **kwargs):
        if _plain(self):
            return self.forward(*args, **kwargs)
        # looked up per call: FX symbolic tracing patches nn.Module.__call__
        return torch.nn.Module.__call__(self, *args, **kwargs)

    call._tpu_direct = True
    split_cls.__call__ = call
    wrapper_cls.__call__ = wrapper_call


def patch_split_direct_call() -> None:
    if not envs.VLLM_TPU_SPLIT_DIRECT_CALL:
        return
    from torch_tpu._internal.compile import split_compiler

    install(split_compiler._SplitCompiledExecutable, split_compiler._WrapperModule)
    logger.info("Compiled TPU graphs call their split pieces' forward directly")
