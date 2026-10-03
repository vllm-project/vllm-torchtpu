"""Bind device implementations into reused upstream Python code."""

from contextlib import contextmanager
from threading import RLock

_lock = RLock()


@contextmanager
def device_hooks(module, **replacements):
    with _lock:
        previous = {name: getattr(module, name) for name in replacements}
        try:
            for name, value in replacements.items():
                setattr(module, name, value)
            yield
        finally:
            for name, value in previous.items():
                setattr(module, name, value)


def bind_device_ops(function, **operations):
    """Reuse upstream bytecode with per-adapter device function bindings.

    Unlike runtime module monkeypatches, bindings are immutable during forward
    and TorchDynamo can trace them without locks or global mutation.
    """
    import sys
    from types import FunctionType, ModuleType

    # Dynamo guards resolve global functions by module name. A copied globals
    # dict with the upstream name would make guards inspect the GPU operation.
    name = f"{__name__}.{function.__qualname__.replace('.', '_')}"
    namespace = ModuleType(name)
    namespace.__dict__.update(function.__globals__ | operations | {"__name__": name})
    sys.modules[name] = namespace
    bound = FunctionType(
        function.__code__,
        namespace.__dict__,
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    bound.__kwdefaults__ = function.__kwdefaults__
    bound.__annotations__ = function.__annotations__
    bound.__qualname__ = function.__qualname__
    setattr(namespace, function.__name__, bound)
    return bound
