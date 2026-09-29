# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Functions for finding which devices share a physical chip.

Re-exports core primitives from `llm_tpu_core.distributed.chip_topology`.
"""

from __future__ import annotations

from vllm_torchtpu.llm_tpu_core.distributed.chip_topology import *  # noqa: F403
