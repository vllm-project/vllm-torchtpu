# SPDX-License-Identifier: Apache-2.0
"""Model implementations this plugin registers with vLLM.

`register_models` is the target of the `torchtpu_models` entry point in
`vllm.general_plugins`, which is the earliest hook that runs before vLLM
resolves a config's architecture to a model class. It is re-exported here rather
than only from `models.vllm` so that the entry point names this package.
"""

from vllm_torchtpu.models.vllm import register_models

__all__ = ["register_models"]
