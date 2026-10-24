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
"""Find set environment variables that nothing reads.

``envs.py`` lists every knob this package reads, so a set variable that is
neither there nor in vLLM's registry does nothing. Two cases are reported:

- ``VLLM_*`` names, which only vLLM and this package use, when neither
  registry has them (for example a removed ``VLLM_TPU_*`` knob).
- Names that share a prefix with this package's knobs (``TPU_``, ``MOE_``,
  ...) and look like a typo of a registered name: a close ``difflib`` match
  with the same number of ``_``-separated tokens that differs by more than
  digits. Sibling names in a namespace another reader owns, such as
  ``TORCH_TPU_TIER2_COMPILATION_CACHE`` next to
  ``TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT``, are left alone, as are other
  ``TPU_*`` names: libtpu, JAX and GKE read and inject many of them, and no
  list of those stays current.

``PACKAGE_WRITTEN_ENV_VARS`` lists the names this package sets itself;
``tests/test_env_validation.py`` checks it against every literal
``os.environ`` write under ``src/``.
"""

import difflib
from collections.abc import Collection, Iterable
from dataclasses import dataclass

# High enough that a one- or two-letter slip matches and unrelated names
# sharing a prefix do not.
_TYPO_CUTOFF = 0.9

# Names this package writes into os.environ for itself or for the libraries it
# drives (torch_tpu, libtpu, vLLM, Ray). They are set on purpose, so they are
# never reported.
PACKAGE_WRITTEN_ENV_VARS = frozenset(
    {
        "ALLOW_MULTIPLE_LIBTPU_LOAD",
        "LIBTPU_INIT_ARGS",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "NODE_RANK",
        "RAIDEN_DISABLE_SINGLETON_WORKER",
        "RANK",
        "RAY_USAGE_STATS_ENABLED",
        "TORCH_TPU_DP_MASTER_ADDR",
        "TORCH_TPU_DP_MASTER_PORT",
        "TORCH_TPU_DP_SIZE",
        "TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS",
        "TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE",
        "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT",
        "TORCH_TPU_SLICEBUILDER_ADDRESSES",
        "TORCH_TPU_TIER2_COMPILATION_CACHE",
        "TORCH_TPU_TOPOLOGY",
        "TORCH_TPU_XPROF_SESSION_ID",
        "TPU_GDN_CONV_QK_PAIR_LAYOUT",
        "TPU_NUM_HOSTS",
        "TPU_PREMAPPED_BUFFER_SIZE",
        "TPU_VISIBLE_CHIPS",
        "TPU_VISIBLE_DEVICES",
        "VLLM_CACHE_ROOT",
        "VLLM_DISABLE_SHARED_EXPERTS_STREAM",
        "VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT",
        "VLLM_USE_BREAKABLE_CUDAGRAPH",
        "VLLM_USE_MEGA_AOT_ARTIFACT",
        "VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE",
        "WORLD_SIZE",
    }
)


@dataclass(frozen=True)
class UnreadEnvVar:
    """A set environment variable that no registry reads."""

    name: str
    suggestion: str | None = None

    def __str__(self) -> str:
        if self.suggestion is None:
            return self.name
        return f"{self.name} (did you mean {self.suggestion}?)"


def _prefix(name: str) -> str:
    return name.split("_", 1)[0]


def _closest(name: str, candidates: Collection[str]) -> str | None:
    matches = difflib.get_close_matches(name, candidates, n=1, cutoff=_TYPO_CUTOFF)
    return matches[0] if matches else None


def _looks_like_typo(name: str, registered: str) -> bool:
    """A different token count or a digit-only difference names a sibling."""
    if name.count("_") != registered.count("_"):
        return False
    strip_digits = str.maketrans("", "", "0123456789")
    return name.translate(strip_digits) != registered.translate(strip_digits)


def find_unread_env_vars(
    set_names: Iterable[str],
    torchtpu_names: Collection[str],
    vllm_names: Collection[str],
    external_prefixes: tuple[str, ...] = (),
) -> list[UnreadEnvVar]:
    """Returns the set names that neither registry reads, sorted by name.

    Args:
        set_names: Names of the variables set in the environment.
        torchtpu_names: Names this package reads or writes.
        vllm_names: Names vLLM reads.
        external_prefixes: Prefixes owned by a reader without a registry (for
            example ``VLLM_OMNI_``). Names with them are never reported.
    """
    known = set(torchtpu_names) | set(vllm_names)
    known_vllm = sorted(n for n in known if n.startswith("VLLM_"))
    by_prefix: dict[str, list[str]] = {}
    for registered in torchtpu_names:
        if not registered.startswith("VLLM_"):
            by_prefix.setdefault(_prefix(registered), []).append(registered)

    unread = []
    for name in sorted(set(set_names) - known):
        if name.startswith(external_prefixes):
            continue
        if name.startswith("VLLM_"):
            unread.append(UnreadEnvVar(name, _closest(name, known_vllm)))
            continue
        candidates = by_prefix.get(_prefix(name))
        if candidates:
            suggestion = _closest(name, candidates)
            if suggestion is not None and _looks_like_typo(name, suggestion):
                unread.append(UnreadEnvVar(name, suggestion))
    return unread
