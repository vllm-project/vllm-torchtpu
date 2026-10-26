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
"""The weight-stream decode projection (`kernel_stream`) against the emit_pipeline one.

The stream's stored state is not byte-equal to the pipeline kernel's (last-bit
differences, measured on TPU, at every k tile including the pipeline's 3584),
so both are checked against an f64 oracle of the state the projection stores:
bf16 weights must sit as close to it as the pipeline kernel does, fp8 weights
within the e4m3 error. The stream's k tiling itself does not change the bytes
(f32 accumulation over k tiles in order), which the first test pins.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu

from vllm_torchtpu.kernels.deepseek_v4.compress_and_store import (
    compressor_v1,
    proj_and_save_state as pk,
)

from .compressor_prefill_paths_test import VARIANTS, _build, _run

pytestmark = pytest.mark.tpu_test

BASE = dict(proj_runs_min_tokens=128)


def _fp8(w):
    wf = w.astype(jnp.float32)
    s = jnp.max(jnp.abs(wf), axis=0) / 448.0
    s = jnp.where(s == 0, 1.0, s)
    return jnp.clip(wf / s[None, :], -448.0, 448.0).astype(jnp.float8_e4m3fn), s


def _states(args, **kw):
    """Run only the projection; return (stored f32 state per real token, f64 oracle)."""
    head, ratio, overlap, _ = None, args["compress_ratio"], args["overlap"], None
    t2r, slots, _ = compressor_v1.derive_metadata(
        positions=args["positions"],
        block_table=args["block_table"],
        query_start_loc=args["query_start_loc"],
        kv_block_table=args["kv_block_table"],
        cache=args["cache"],
        compress_ratio=ratio,
        state_block_size=args["state_block_size"],
        head_dim=args["head_dim"],
        overlap=overlap,
        cos_sin_cache=args["cos_sin_cache"],
        state_cache=args["state_cache"],
        seq_along_lane=args["seq_along_lane"],
    )
    n_real = int(args["query_start_loc"][args["distribution"][2]])
    slots = jnp.where(jnp.arange(slots.shape[0]) < n_real, slots, -1)
    tgt = args["cache"] if args["state_cache"] is None else args["state_cache"]
    w, ws = args["wkv_wgate"], None
    if kw.pop("fp8", False):
        w, ws = _fp8(w)
    out = jax.jit(
        lambda x, w, ws, a, p, s, c: pk.proj_and_save_state(
            x, w, a, p, s, c, compress_ratio=ratio, wkv_scale=ws, **kw
        )
    )(args["hidden_states"], w, ws, args["ape"], args["positions"], slots, tgt)
    return _extract(args, out, slots)


def _slots(args):
    return compressor_v1._state_slots(
        **{k: args[k] for k in ("positions", "block_table", "query_start_loc",
                                "kv_block_table", "cache", "state_cache", "distribution",
                                "cos_sin_cache", "compress_ratio", "state_block_size",
                                "head_dim", "overlap", "seq_along_lane")}
    )


def _extract(args, out, slots):
    """(stored f32 state per real token, its f64 oracle)."""
    ratio = args["compress_ratio"]
    out = np.asarray(out)
    lanes, page_rows = out.shape[-1], out.shape[1]
    sw = args["wkv_wgate"].shape[1] // 2
    rows = 2 * sw // lanes
    flat = out.reshape(out.shape[0] * page_rows, -1).view(np.uint8)
    x = np.asarray(args["hidden_states"].astype(jnp.float32)).astype(np.float64)
    wf = np.asarray(args["wkv_wgate"].astype(jnp.float32)).astype(np.float64)
    ref = x @ wf
    pos = np.asarray(args["positions"])
    ref[:, sw:] += np.asarray(args["ape"]).astype(np.float64)[pos % ratio]
    got, want = [], []
    for t, s0 in enumerate(np.asarray(slots)):
        if s0 < 0:
            continue
        g = flat[s0 : s0 + rows]
        if out.dtype == np.uint8:  # byte-transposed rows
            g = g.reshape(rows, 4, lanes).transpose(0, 2, 1)
        got.append(g.copy().view(np.float32).reshape(-1))
        want.append(ref[t])
    return np.stack(got), np.stack(want)


class CompressorProjStreamTest(jtu.JaxTestCase):
    def setUp(self):
        super().setUp()
        if not jtu.test_device_matches(["tpu"]):
            self.skipTest("TPU only")

    @parameterized.product(
        variant=tuple(VARIANTS),
        decode=((1, 7), (2, 7), (4, 7), (8, 7), (3, 2)),
        nbuf=(2, 1),
    )
    def test_k_tiling_is_byte_neutral(self, variant, decode, nbuf):
        args = _build(variant, decode, ())
        ref = _run(args, **BASE, proj_stream_max_tokens=64, proj_stream_cfg=(3584, 2, 1))
        new = _run(
            args, **BASE, proj_stream_max_tokens=64, proj_stream_cfg=(512, 4 + nbuf, 1)
        )
        for r, n in zip(ref, new):
            if r is not None:
                self.assertArraysEqual(r, n)
        inputs = (args["cache"], args["rope_cache"], args["state_cache"])
        self.assertTrue(
            any(
                r is not None and not np.array_equal(r, np.asarray(i))
                for r, i in zip(ref, inputs)
            )
        )

    @parameterized.product(
        variant=tuple(VARIANTS),
        decode=((2, 7), (8, 7)),
        cfg=((512, 6, 1), (1024, 4, 2), (256, 8, 1)),
        fp8=(False, True),
    )
    def test_oracle(self, variant, decode, cfg, fp8):
        args = _build(variant, decode, ())
        got, want = _states(
            args, stream_max_tokens=64, stream_cfg=cfg, fp8=fp8
        )
        base, _ = _states(args)
        scale = np.abs(want).max(axis=1, keepdims=True)
        err = np.abs(got - want) / scale
        err_base = np.abs(base - want) / scale
        # bf16 weights: same distance to f64 as the pipeline kernel.
        # fp8 (per-channel e4m3, 3 mantissa bits): a few % of the row max.
        tol = 5e-2 if fp8 else max(4 * err_base.max(), 1e-5)
        self.assertLess(err.max(), tol)
        if fp8:
            # Positive control: the weight really is quantized.
            self.assertGreater(err.max(), 10 * err_base.max())


class CompressorPairProjTest(jtu.JaxTestCase):
    """The two-group stream call (CSA KV + indexer projections of the same
    tokens) against the f64 oracle, and the skip_proj plumbing. Not byte-equal
    to two single calls at T >= 32 (measured, j-1e33105f: ~1% of state words in
    2 of a token's 16 rows) -- Mosaic lowers the same dot differently in the two
    kernels -- so it is held to the oracle tolerance of the single stream."""

    def setUp(self):
        super().setUp()
        if not jtu.test_device_matches(["tpu"]):
            self.skipTest("TPU only")

    @parameterized.product(
        decode=((1, 7), (2, 7), (4, 7), (8, 7), (3, 2)),
        cfg=((512, 6, 1), (3584, 2, 1)),
        fp8=(False, True),
    )
    def test_pair(self, decode, cfg, fp8):
        a = _build("csa", decode, ())
        b = _build("indexer", decode, ())
        np.testing.assert_array_equal(a["hidden_states"], b["hidden_states"])
        np.testing.assert_array_equal(a["positions"], b["positions"])
        groups = []
        for args in (a, b):
            w, ws = _fp8(args["wkv_wgate"]) if fp8 else (args["wkv_wgate"], None)
            groups.append(dict(
                wkv_wgate=w, wkv_scale=ws, ape=args["ape"], slot_mapping=_slots(args),
                cache=args["cache"], compress_ratio=args["compress_ratio"]))
        hosts = pk._proj_and_save_state_stream_multi(
            a["hidden_states"], a["positions"], groups,
            t=cfg[0], nbuf=cfg[1], ndma=cfg[2], interpret=False)
        for args, host, g in zip((a, b), hosts, groups):
            got, want = _extract(args, host, g["slot_mapping"])
            base, _ = _states(args)
            scale = np.abs(want).max(axis=1, keepdims=True)
            err = (np.abs(got - want) / scale).max()
            err_base = (np.abs(base - want) / scale).max()
            self.assertLess(err, 5e-2 if fp8 else max(4 * err_base, 1e-5))
            if fp8:
                self.assertGreater(err, 10 * err_base)
            # Nothing outside the real tokens' state rows changed.
            n_rows = 2 * (args["wkv_wgate"].shape[1] // 2) // host.shape[-1]
            own = np.zeros(host.shape[0] * host.shape[1], bool)
            for s0 in np.asarray(g["slot_mapping"]):
                if s0 >= 0:
                    own[s0 : s0 + n_rows] = True
            h = np.asarray(host).reshape(own.shape[0], -1)
            i = np.asarray(args["cache"]).reshape(own.shape[0], -1)
            self.assertFalse(((h != i).any(axis=1) & ~own).any())

    @parameterized.product(variant=("csa", "indexer"), decode=((1, 7), (4, 7)))
    def test_skip_proj_reads_the_saved_state(self, variant, decode):
        """compressor_forward(skip_proj=True) on a state saved by the stream
        call == compressor_forward that saves it itself."""
        args = _build(variant, decode, ())
        kw = dict(BASE, proj_stream_max_tokens=64, proj_stream_cfg=(512, 6, 1))
        ref = _run(args, **kw)
        host = pk._proj_and_save_state_stream(
            args["hidden_states"], args["wkv_wgate"], None, args["ape"],
            args["positions"], _slots(args), args["cache"],
            compress_ratio=args["compress_ratio"], t=512, nbuf=6, ndma=1,
            interpret=False)
        new = _run(dict(args, cache=host), **kw, skip_proj=True)
        for r, n in zip(ref, new):
            if r is not None:
                self.assertArraysEqual(r, n)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
