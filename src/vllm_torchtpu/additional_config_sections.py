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
"""The top-level ``--additional-config`` sections this package reads.

vLLM hands ``additional_config`` to the platform untouched, so a section whose
reader is not in the tree is silently ignored, and a deployment that sets one
runs without it. For the two namespaces this package owns, ``qwen3_5`` and
every ``tpu_*`` key, ``check`` fails instead: each change that adds a
section adds its key to ``KNOWN_SECTIONS``. Keys outside those namespaces
(``abort_on_preemption``, ``raiden_pd_send_completion``, ``megakernel``,
the ``phased_profiler_*`` keys, or another plugin's) are not checked.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# Each entry names the module that parses the section.
KNOWN_SECTIONS: frozenset[str] = frozenset(
    {
        # keep-sorted start
        "qwen3_5",  # models/qwen3_5/settings.py
        "tpu_online_fp8",  # layers/adapter/quantization/online_fp8.py
        # keep-sorted end
    }
)

OWNED_KEYS = frozenset({"qwen3_5"})
OWNED_PREFIX = "tpu_"


def is_owned(key: str) -> bool:
    """Whether the key is in a namespace this package owns."""
    return key in OWNED_KEYS or key.startswith(OWNED_PREFIX)


def unknown_sections(additional_config: Any) -> list[str]:
    """The owned top-level keys that no section reader in the tree claims."""
    if not isinstance(additional_config, Mapping):
        return []
    return sorted(
        str(key)
        for key in additional_config
        if isinstance(key, str) and is_owned(key) and key not in KNOWN_SECTIONS
    )


def check(additional_config: Any) -> None:
    """Raise ValueError naming every owned key this tree does not read."""
    unknown = unknown_sections(additional_config)
    if unknown:
        raise ValueError(
            f"Unknown --additional-config section(s) {unknown}: this build reads "
            f"{sorted(KNOWN_SECTIONS)} in the qwen3_5 and tpu_* namespaces. "
            "The change that adds a section may not be in this build."
        )
