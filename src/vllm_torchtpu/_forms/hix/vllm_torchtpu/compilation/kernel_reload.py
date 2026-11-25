# SPDX-License-Identifier: Apache-2.0
"""Live kernel hot-reload for kernel-iteration mode (TPU_KERNEL_ITER_MODE=1).

Kernel developers iterating on a Pallas kernel normally pay a full server
restart per edit; with the compiled graph split at the Pallas custom ops
(see tpu_platform.py) the kernel is the only piece that changes, so the
server can instead re-import the kernel sources and swap the implementation
behind the already-registered torch op:

- Each reloadable custom op is registered against a ``_KernelDispatch``
  indirection instead of its callable, so the op name baked into captured
  graphs never changes; only the dispatch target is swapped.
- Op owners register a zero-argument *builder* that reconstructs the live
  callable from freshly imported modules (resolve modules with
  ``importlib.import_module`` inside the builder — a closure over module
  attributes would keep pre-reload objects alive).
- ``reload_kernels()`` re-imports the reloadable module list (from
  ``TPU_KERNEL_RELOAD_MODULES`` or the per-request override) and swaps every
  registered op. The swapped kernel lowers and compiles on its next
  invocation.

Signature-changing kernel edits still require a restart: captured graphs and
the ops' fake implementations keep the original schema.
"""

import importlib
import sys
import time
from collections.abc import Callable, Sequence

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Default reloadable modules: the in-tree experimental RPA kernels, listed in
# dependency order (later modules re-bind ``from X import ...`` names from
# earlier ones). Overridable via TPU_KERNEL_RELOAD_MODULES or per reload
# request.
_DEFAULT_RELOAD_MODULES = (
    "vllm_torchtpu.kernels.experimental.batched_rpa.utils",
    "vllm_torchtpu.kernels.experimental.batched_rpa.kernel",
    "vllm_torchtpu.kernels.experimental.batched_rpa.wrapper",
    "vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.pcp_layout",
    "vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.schedule",
    "vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.kernel",
    "vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.reference",
    "vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.wrapper",
    "vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.vllm_adapter",
)

# op name -> live callable invoked by the dispatcher.
_LIVE: dict[str, Callable] = {}
# op name -> zero-arg builder returning a fresh live callable.
_BUILDERS: dict[str, Callable[[], Callable]] = {}


def is_enabled() -> bool:
    return envs.TPU_KERNEL_ITER_MODE


def default_reload_modules() -> tuple[str, ...]:
    configured = envs.TPU_KERNEL_RELOAD_MODULES
    if configured:
        return tuple(m.strip() for m in configured.split(",") if m.strip())
    return _DEFAULT_RELOAD_MODULES


class _KernelDispatch:
    """Per-call indirection so the registered torch op survives reloads.

    ``torch.library.custom_op`` infers the op schema from the callable, which
    requires function-style introspection (``__globals__``, annotations, and
    a signature — resolved through ``__wrapped__``). Copy those from the
    initial callable so the registered schema is identical to registering it
    directly.
    """

    _COPY_ATTRS = (
        "__name__",
        "__qualname__",
        "__module__",
        "__doc__",
        "__globals__",
        "__annotations__",
        "__signature__",
        "__defaults__",
        "__kwdefaults__",
    )

    def __init__(self, name: str, template: Callable) -> None:
        self._name = name
        for attr in self._COPY_ATTRS:
            try:
                value = getattr(template, attr)
            except AttributeError:
                continue
            setattr(self, attr, value)
        self.__wrapped__ = template

    def __call__(self, *args, **kwargs):
        return _LIVE[self._name](*args, **kwargs)


def make_dispatcher(name: str, template: Callable) -> _KernelDispatch:
    return _KernelDispatch(name, template)


def set_live(name: str, callable_: Callable) -> None:
    """Set the current dispatch target for op ``name``."""
    _LIVE[name] = callable_


def get_live(name: str) -> Callable | None:
    return _LIVE.get(name)


def register_builder(name: str, builder: Callable[[], Callable]) -> None:
    """Register the rebuild recipe for op ``name``.

    ``builder`` reconstructs the live callable from freshly imported modules;
    reload_kernels() calls it after re-importing the reloadable module list.
    """
    _BUILDERS[name] = builder


def reload_kernels(modules: Sequence[str] | None = None) -> dict:
    """Re-import kernel sources and swap every registered op.

    Returns per-phase timings. The swapped callables lower and device-compile
    lazily on their next invocation, so the caller's next request pays the
    kernel compile and nothing else.
    """
    module_names = tuple(modules) if modules else default_reload_modules()

    t0 = time.perf_counter()
    for mod_name in module_names:
        mod = sys.modules.get(mod_name)
        if mod is not None:
            importlib.reload(mod)
        else:
            logger.warning("Reload module %s is not imported; skipping.", mod_name)
    reimport_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    rebuilt = []
    for name, builder in _BUILDERS.items():
        _LIVE[name] = builder()
        rebuilt.append(name)
    rebuild_s = time.perf_counter() - t0

    logger.info(
        "Kernel hot-reload: re-imported %d modules (%.3fs), rebuilt ops %s (%.3fs).",
        len(module_names),
        reimport_s,
        rebuilt,
        rebuild_s,
    )
    return {
        "reimport_s": round(reimport_s, 3),
        "rebuild_s": round(rebuild_s, 3),
        "reloaded_modules": list(module_names),
        "rebuilt_ops": rebuilt,
    }
