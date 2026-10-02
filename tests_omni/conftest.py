# SPDX-License-Identifier: Apache-2.0
"""Shared pytest configuration for the Omni tests.

tests_omni/ holds tests that need the real vllm-omni package, which only the
``-omni`` CI image installs. It sits outside tests/ so the general CI lanes,
which glob ``tests/**`` in the dev image, never collect it; only the Omni lanes
in .buildkite/pipeline_tests.yml run it.

tests/conftest.py does not apply here, so the environment defaults that
engine-starting tests rely on are repeated below. Keep them in sync with it.
"""

import os

os.environ.setdefault("TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS", "false")
os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
os.environ.setdefault("VLLM_DO_NOT_TRACK", "1")
