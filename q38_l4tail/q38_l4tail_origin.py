"""pytest plugin (-p q38_l4tail_origin): print which sources the L4 TAIL tests ran: the runner source the ast harness
extracted from (test_vps_wiring.RUNNER, set by Q38MM_SRC_ROOT), the envs.py / tpu_compiler.py the harness read
(test_vps_wiring.PKG / .TC), and where vllm_torchtpu's tpu_runner / attention_metadata / envs / tpu_compiler were
imported from, so a leg that read another tree reads INVALID. Unused entries print None.
D83S-BUILD (L1-D83-SERVED): + pkg / tc / envs / tc_mod entries (the runner asserts every non-None entry is inside the
tree's src)."""


def pytest_terminal_summary(terminalreporter):
    import sys

    w = sys.modules.get("test_vps_wiring")
    m = sys.modules.get("vllm_torchtpu.runner.tpu_runner")
    a = sys.modules.get("vllm_torchtpu.layers.core.attention_metadata")
    e = sys.modules.get("vllm_torchtpu.envs")
    t = sys.modules.get("vllm_torchtpu.compilation.tpu_compiler")
    terminalreporter.write_line("Q38_L4TAIL_ORIGIN runner=%s tpu_runner=%s am=%s pkg=%s tc=%s envs=%s tc_mod=%s" % (
        getattr(w, "RUNNER", None), getattr(m, "__file__", None), getattr(a, "__file__", None),
        getattr(w, "PKG", None), getattr(w, "TC", None), getattr(e, "__file__", None), getattr(t, "__file__", None)))
