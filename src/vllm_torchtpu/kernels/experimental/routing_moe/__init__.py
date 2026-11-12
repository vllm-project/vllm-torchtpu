# SPDX-License-Identifier: Apache-2.0
"""Experimental TPU routing_moe kernel package ported from openxla/tokamax.

Provenance (PR-3):
- Source: https://github.com/openxla/tokamax
  (`tokamax/_src/ops/experimental/routing_moe/`).
- Local changes relative to upstream `openxla/tokamax`:
  1. Rewrote package imports from `tokamax._src.ops.experimental.routing_moe.*`
     to `vllm_torchtpu.kernels.experimental.routing_moe.*` so `vllm-torchtpu`
     works standalone before a released `tokamax` wheel exposes this package.
  2. Admitted lane-aligned 128-multiple partial-chunk widths (e.g. 640 = 5 * 128)
     in `gmm/plan.py` / `gmm/memory_ref.py` alongside powers of two so sharded
     hidden/intermediate widths such as 2560 -> 640 at TP=4 / EP=4 tile cleanly.
"""
