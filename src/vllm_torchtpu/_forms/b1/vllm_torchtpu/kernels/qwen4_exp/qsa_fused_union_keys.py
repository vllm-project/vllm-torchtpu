# SPDX-License-Identifier: Apache-2.0
"""q38 QSAF keys (phase 1, kernels/qwen4_exp/qsa_fused_union.py), read straight from the environment.

They are deliberately NOT registered in ``vllm_torchtpu.envs`` (the q38 AR-FIX / IDXX pattern): a registered key enters
``tpu_compiler._tpu_compile_env_factors`` even when unset, so a tree that merely carries QSAF would miss every warm compile
cache. Here they key the cache only when the lever is on (``compile_factors`` is ``{}`` when off), so with QSAF off the
env factors are exactly those of a tree without it.

- ``VLLM_TPU_QSA_DECODE_FUSED_UNION``: unset, "" or "0" off (default); "1" on; anything else raises.
- ``VLLM_TPU_QSA_DECODE_FUSED_UNION_ISSUE``: unset / "" / "serial" (default) or "overlap"; anything else raises. It changes
  the program only when the lever is on, so it keys the cache only then.

Both are read at trace time inside the compiled forward (``qsa_rpa.union_decode_attention``)."""

import os

ENV = "VLLM_TPU_QSA_DECODE_FUSED_UNION"
ISSUE_ENV = "VLLM_TPU_QSA_DECODE_FUSED_UNION_ISSUE"
ISSUES = ("serial", "overlap")


def enabled() -> bool:
    value = os.environ.get(ENV)
    if value is None or value in ("", "0"):
        return False
    if value == "1":
        return True
    raise ValueError(f"Invalid value '{value}' for {ENV}. Valid options: '', '0', '1'.")


def issue() -> str:
    value = os.environ.get(ISSUE_ENV)
    if value is None or value == "":
        return "serial"
    if value in ISSUES:
        return value
    raise ValueError(f"Invalid value '{value}' for {ISSUE_ENV}. Valid options: '', 'serial', 'overlap'.")


def compile_factors() -> dict[str, str]:
    """The lever's compile-key factors: ``{}`` when off (every key unchanged); both keys when on."""
    issue_value = issue()  # validate even when off: a bad value never passes silently
    if not enabled():
        return {}
    return {ENV: "1", ISSUE_ENV: issue_value}
