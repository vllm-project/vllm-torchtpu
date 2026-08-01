# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from vllm_torchtpu.kernels.kimi_k3.chunk_kda import (chunk_kda_fwd_custom,
                                                     kda_forward)
from vllm_torchtpu.kernels.kimi_k3.decode_kda import kda_decode
from vllm_torchtpu.kernels.kimi_k3.naive_kda import kda_step, naive_ragged_kda

# Three ways to evaluate the same KDA recurrence, see README.md:
#   kda_forward       chunked Pallas kernel, the prefill path
#   kda_decode        single-token recurrent Pallas kernel, the decode path
#   naive_ragged_kda  plain-XLA recurrence over ragged requests
# `kda_decode` and `naive_ragged_kda` do not yet share an ABI -- see
# decode_kda.py's docstring for the five differences.
__all__ = [
    "chunk_kda_fwd_custom",
    "kda_decode",
    "kda_forward",
    "kda_step",
    "naive_ragged_kda",
]
