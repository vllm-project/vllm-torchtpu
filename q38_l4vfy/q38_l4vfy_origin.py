"""pytest plugin (-p q38_l4vfy_origin), VFY-BUILD: print which sources the VFY-ONE tests ran, so a leg that read
another tree reads INVALID: the runner / eagle3 sources the ast harness extracted from (test_vps_wiring.RUNNER /
.EAGLE, set by Q38MM_SRC_ROOT), every module copy tests/l4vfy/vfy_harness.py loaded (vfy_one, l6_ab, served_ab),
and where vllm_torchtpu's envs / tpu_compiler were imported from. Unused entries print None."""


def pytest_terminal_summary(terminalreporter):
    import sys

    w = sys.modules.get("test_vps_wiring")
    h = sys.modules.get("vfy_harness")
    e = sys.modules.get("vllm_torchtpu.envs")
    t = sys.modules.get("vllm_torchtpu.compilation.tpu_compiler")
    loaded = sorted(set((getattr(h, "LOADED", None) or {}).values())) or [None]
    terminalreporter.write_line("Q38_L4VFY_ORIGIN runner=%s eagle=%s envs=%s tc_mod=%s %s" % (
        getattr(w, "RUNNER", None), getattr(w, "EAGLE", None), getattr(e, "__file__", None),
        getattr(t, "__file__", None), " ".join("loaded=%s" % p for p in loaded)))
