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
"""``tpu_compile``, the decorator for this package's compiled TPU functions.

It compiles a plain function with main's ``torch.compile(backend="tpu",
fullgraph=True, dynamic=False)`` and wraps the result for the guard-checked
direct dispatch of ``compilation.fast_dispatch``, which
``additional_config["tpu_compile"]`` arms (``compilation.compile_options``).
New compile sites of a plain function use it rather than that
``torch.compile`` call (``docs/developers_guide/coding_guidance.md``);
``tests/compilation/test_tpu_compile_sites.py`` checks the package's sites.
"""

import functools
import types
from collections.abc import Callable
from typing import Any, overload

import torch

from vllm_torchtpu.compilation import fast_dispatch


@overload
def tpu_compile[F: Callable[..., Any]](fn: F, /) -> F: ...


@overload
def tpu_compile[F: Callable[..., Any]](*, guard_memo: bool) -> Callable[[F], F]: ...


def tpu_compile(fn: Any = None, /, *, guard_memo: bool = True) -> Any:
    """Compile ``fn`` for TPU with ``torch.compile(fn, backend="tpu",
    fullgraph=True, dynamic=False)`` and wrap the result for fast dispatch.
    Use it as a decorator, or call it on a function;
    ``@tpu_compile(guard_memo=False)`` keeps the function out of the guard
    memo (``compilation.fast_dispatch``).

    The compile call is main's, so the compiled programs and their guards
    are those of the plain ``torch.compile`` call. Disarmed (the default) the
    wrapper calls the compiled function and nothing else. The option is read
    on every call, never at decoration.
    """
    if fn is None:
        return functools.partial(tpu_compile, guard_memo=guard_memo)
    if not isinstance(fn, types.FunctionType):
        raise TypeError(
            f"tpu_compile takes a plain function, got {type(fn).__name__}; "
            "compile a bound method or a module with torch.compile"
        )
    compiled = torch.compile(fn, backend="tpu", fullgraph=True, dynamic=False)
    if compiled is fn:
        # Dynamo is disabled (TORCHDYNAMO_DISABLE=1): torch.compile hands back
        # the function itself, which has no cache entries to dispatch.
        return fn
    return fast_dispatch.wrap(compiled, fn, guard_memo)
