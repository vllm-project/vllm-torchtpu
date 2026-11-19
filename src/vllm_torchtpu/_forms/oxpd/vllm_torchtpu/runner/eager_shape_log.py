# SPDX-License-Identifier: Apache-2.0
"""q38 L4 EAGER-FIX: log-only names for serving-time eager recompiles
(VLLM_TPU_EAGER_SHAPE_LOG=N, 0 = off; compile-ignored).

torch_tpu compiles each eager graph once per input layout, and a served
decode run logged new eager compiles (sub, copy__as_strided_inverse, uniform_,
index) in steady state with no Python-side name. Each suspect site calls
``note(site, key)`` with the layout it is about to issue; every key seen
before ``mark_serving()`` (warm-up) is recorded silently, and each key first
seen after it is logged once, up to N lines per process, as

  EAGERSHAPE rank=R site=<site> key=<layout> site_new=<k>

so a served run's TpuCompiler compile lines can be matched to a call site by
time. With N = 0 every call returns at once.
"""

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_LIMIT = envs.VLLM_TPU_EAGER_SHAPE_LOG
_seen: dict[str, set] = {}
_new: dict[str, int] = {}
_logged = 0
_serving = False


def enabled() -> bool:
    return _LIMIT > 0


def mark_serving() -> None:
    global _serving
    _serving = True


def note(site: str, key: tuple) -> None:
    global _logged
    if not _LIMIT:
        return
    seen = _seen.setdefault(site, set())
    if key in seen:
        return
    seen.add(key)
    if not _serving:
        return
    _new[site] = _new.get(site, 0) + 1
    if _logged < _LIMIT:
        _logged += 1
        from vllm_torchtpu.runner import host_phases

        logger.info(
            "EAGERSHAPE rank=%d site=%s key=%s site_new=%d",
            host_phases._rank,
            site,
            key,
            _new[site],
        )


def _reset_for_tests(limit: int) -> None:
    global _LIMIT, _logged, _serving
    _LIMIT = limit
    _logged = 0
    _serving = False
    _seen.clear()
    _new.clear()
