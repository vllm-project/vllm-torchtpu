"""pytest plugin (-p q38_origin): print where vllm_torchtpu and the modules under test were imported from, so a
mutant leg that silently imported another tree (e.g. through the recipe's PYTHONPATH=/tmp/src/src) reads INVALID.
Only modules the leg actually imported are listed; vllm_torchtpu itself is always listed (None if never imported)."""

MODULES = ("vllm_torchtpu", "vllm_torchtpu.runner.tpu_runner", "vllm_torchtpu.spec_decode.eagle3",
           "vllm_torchtpu.runner.tpu_runner_async_output", "vllm_torchtpu.core.prompt_ids_ndarray",
           "vllm_torchtpu.runner.owned_args")


def pytest_terminal_summary(terminalreporter):
    import sys

    parts = []
    for name in MODULES:
        m = sys.modules.get(name)
        f = getattr(m, "__file__", None) if m is not None else None
        if f is not None or name == "vllm_torchtpu":
            parts.append(f"{name}={f}")
    terminalreporter.write_line("Q38_ORIGIN " + " ".join(parts))
