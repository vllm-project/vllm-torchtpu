# SPDX-License-Identifier: Apache-2.0
"""q38 PSTEP-WARMTWIN P7-TWIN: a per-process, warm-safe line for the libtpu P-state flag.

libtpu reads LIBTPU_INIT_ARGS (assembled in env_override.py) when the PjRt client starts.
LIBTPU_INIT_ARGS is a native compile factor (compilation/tpu_compiler.py
_NATIVE_TPU_COMPILE_ENV_VARS), so the executables a process compiles or loads are keyed
by it. ``report`` runs at worker init, outside any compiled region, and prints on every
worker process (plain logger.info) what this process's env carries:

  q38 P7-TWIN: rank=<r> pid=<p> p7=<0|1> dvfs_tokens=<t,..> libtpu_args_sha=<s> compile_factor_sha=<s>

``p7=1`` claims the flag ``--xla_tpu_dvfs_p_state=7`` reached this process's env and the
compile-key factors; it says NOTHING about the device clock. libtpu's own per-process
argv lines stay the clock evidence. Silent (but recorded) when LIBTPU_INIT_ARGS carries
no ``--xla_tpu_dvfs_p_state`` token. tpu_compiler.compute_tpu_compilation_hash compares
the record with the native factors it actually hashes (``_p7_twin_check``) and prints a
'q38 P7-TWIN check' or 'q38 P7-TWIN MISMATCH' line. Log only: nothing enters a program.
"""

import hashlib
import json
import os

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

P7_FLAG = "--xla_tpu_dvfs_p_state=7"
DVFS_PREFIX = "--xla_tpu_dvfs_p_state"
LINE = (
    "q38 P7-TWIN: rank=%d pid=%d p7=%d dvfs_tokens=%s libtpu_args_sha=%s "
    "compile_factor_sha=%s"
)

_CLAIM: dict | None = None


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _dvfs_tokens(args: str) -> list[str]:
    return [t for t in args.split() if t.startswith(DVFS_PREFIX)]


def native_env() -> dict:
    """The native compile factors exactly as compute_tpu_compilation_hash reads them."""
    from vllm_torchtpu.compilation.tpu_compiler import _NATIVE_TPU_COMPILE_ENV_VARS

    return {name: os.getenv(name) for name in _NATIVE_TPU_COMPILE_ENV_VARS}


def report(rank: int) -> dict:
    """Record this process's claim and print the P7-TWIN line when a dvfs token is set."""
    global _CLAIM
    args = os.environ.get("LIBTPU_INIT_ARGS", "")
    tokens = _dvfs_tokens(args)
    claim = {
        "rank": int(rank),
        "pid": os.getpid(),
        "p7": int(P7_FLAG in tokens),
        "dvfs_tokens": ",".join(tokens) or "-",
        "libtpu_args_sha": _sha(args),
        "compile_factor_sha": _sha(
            json.dumps(native_env(), sort_keys=True, separators=(",", ":"))
        ),
    }
    _CLAIM = claim
    if claim["dvfs_tokens"] != "-":
        logger.info(
            LINE,
            claim["rank"],
            claim["pid"],
            claim["p7"],
            claim["dvfs_tokens"],
            claim["libtpu_args_sha"],
            claim["compile_factor_sha"],
        )
    return dict(claim)


def claim() -> dict | None:
    """This process's recorded claim, or None if ``report`` never ran here."""
    return None if _CLAIM is None else dict(_CLAIM)
