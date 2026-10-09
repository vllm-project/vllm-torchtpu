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

"""Tests of fused_qkv: the GQA attention prologue."""

from __future__ import annotations

import dataclasses
import re
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from vllm_torchtpu.kernels.experimental.batched_rpa import configs as rpa_configs
from vllm_torchtpu.kernels.experimental.batched_rpa import wrapper
from vllm_torchtpu.kernels.experimental.fused_qkv import models, reference
from vllm_torchtpu.kernels.experimental.fused_qkv.kernel import configs, rpa

CASES = [
    (f"_{m.name}_T{t}", m.name, t)
    for m in models.MODELS
    for t in (32, 64, 128, 1024, 16384)
]
KV_LAYOUTS = (
    rpa_configs.KVLayout.SEQ_ALONG_LANE,
    rpa_configs.KVLayout.HEAD_ALONG_SUBLANE,
)
LAYOUT_CASES = [
    (f"{case}_{layout.symbol}", *args, layout)
    for case, *args in CASES
    for layout in KV_LAYOUTS
]
OUTPUTS = ("q", "kv", "gate", "residual")
# test_accuracy: how far the kernel may fall below vLLM's SNR.
SNR_TOLERANCE_DB = 0.15
# test_accuracy: below this many tokens, the tolerance grows as 1/sqrt(T).
SNR_TOLERANCE_TOKENS = 128
# vLLM (torch) rounds every bf16 op; XLA must not elide those roundings.
strict_reference = jax.jit(
    reference.gqa_reference,
    static_argnums=0,
    static_argnames=("kv_layout", "numerics", "page_size"),
    compiler_options={"xla_allow_excess_precision": False},
)
Numerics = reference.Numerics


def run_kernel(
    model: models.Model,
    x: jax.Array,
    residual: jax.Array | None,
    weights: configs.QkvWeights,
    cos: jax.Array | None,
    sin: jax.Array | None,
    **kwargs: Any,
) -> tuple[jax.Array | None, ...]:
    """fused_qkv on weights as loaded."""
    weights = rpa.prepare_weights(model.config, weights)
    return rpa.fused_qkv(model.config, x, residual, weights, cos, sin, **kwargs)


def snr_db(got: jax.Array, want: jax.Array) -> float:
    """10 log10(|want|^2 / |got - want|^2), in f64."""
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    return 10 * np.log10(np.sum(want**2) / np.sum((got - want) ** 2))


def fused_qkv_rpa(
    model: models.Model,
    kv_layout: rpa_configs.KVLayout,
    args: models.Inputs,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    dist: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array | None]:
    """fused_qkv feeding batched_rpa directly (no prepare_inputs)."""
    s = model.config
    q, kv, gate, _ = rpa.fused_qkv(s, *args, kv_layout=kv_layout)
    t = args[0].shape[0]
    qs = jax.ShapeDtypeStruct((t, s.num_q_heads, s.head_dim), s.q_dtype)
    kvs = jax.ShapeDtypeStruct((t, s.num_kv_heads, s.head_dim), s.kv_dtype)
    out, kv_cache = wrapper.ragged_paged_attention(
        qs,
        kvs,
        kvs,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        dist,
        sm_scale=s.head_dim**-0.5,
        k_scale=s.k_scale,
        v_scale=s.v_scale,
        kv_layout=kv_layout,
        prepared_inputs=(q, kv),
    )
    return out, kv_cache, gate


class FusedQkvTest(parameterized.TestCase):
    def assert_exact(
        self,
        label: str,
        got: tuple[jax.Array | None, ...],
        want: tuple[jax.Array | None, ...],
    ) -> None:
        for out, g, w in zip(OUTPUTS, got, want):
            self.assertEqual(g is None, w is None, out)
            if g is None:
                continue
            g, w = np.asarray(g, np.float32), np.asarray(w, np.float32)
            self.assertEqual(g.shape, w.shape, out)
            diff = np.abs(g - w)
            first = np.unravel_index(diff.argmax(), diff.shape)
            print(
                f"{label} {out}: max_abs_diff={diff.max():.3e} "
                f"mismatches={(diff > 0).sum()}/{diff.size} "
                f"worst@{first} got={g[first]} want={w[first]}",
                flush=True,
            )
            self.assertLess(diff.max(), 1e-5, out)

    @parameterized.named_parameters(*LAYOUT_CASES)
    def test_kernel_matches_reference(
        self, name: str, t: int, kv_layout: rpa_configs.KVLayout
    ) -> None:
        """The kernel computes exactly Numerics.KERNEL."""
        model = models.BY_NAME[name]
        args = models.make_exact_inputs(model, t)
        self.assert_exact(
            f"{name} T={t} {kv_layout}",
            run_kernel(model, *args, kv_layout=kv_layout),
            strict_reference(model.config, *args, kv_layout=kv_layout),
        )

    @parameterized.named_parameters(
        *(
            (f"_{m.name}_T{t}_tile{tile}", m.name, t, tile)
            for m in models.MODELS
            for t, tile in ((64, 64), (128, 128), (384, 128))
        )
    )
    def test_kv_padded_to_page_size(self, name: str, t: int, tile: int) -> None:
        """SEQ_ALONG_LANE KV is zero-padded to the page size past the grid."""
        model = models.BY_NAME[name]
        layout, page_size = rpa_configs.KVLayout.SEQ_ALONG_LANE, 256
        args = models.make_exact_inputs(model, t)
        tiling_ = rpa.choose_tiling(
            model.config, t, args[2], kv_layout=layout, tile_size=tile
        )
        self.assert_exact(
            f"{name} T={t} tile={tile} page_size={page_size}",
            run_kernel(
                model, *args, kv_layout=layout, page_size=page_size, tiling=tiling_
            ),
            strict_reference(
                model.config, *args, kv_layout=layout, page_size=page_size
            ),
        )

    @parameterized.named_parameters(*((f"_{m.name}", m.name) for m in models.MODELS))
    def test_kernel_numerics_match_vllm_when_exact(self, name: str) -> None:
        """Numerics.KERNEL and VLLM agree exactly on exact inputs.

        They differ only in where they round. With eps = 0, 1/rms is exactly 1 and
        nothing rounds, so a difference is a bug in the reference.
        """
        model = models.BY_NAME[name]
        config = dataclasses.replace(model.config, eps=0.0)
        args = models.make_exact_inputs(model, 1024)
        self.assert_exact(
            name,
            strict_reference(config, *args, numerics=Numerics.KERNEL),
            strict_reference(config, *args, numerics=Numerics.VLLM),
        )

    @parameterized.named_parameters(*CASES)
    def test_accuracy(self, name: str, t: int) -> None:
        """On random data, the kernel's SNR vs. UNQUANTIZED is close to vLLM's.

        The kernel's bf16 prep math costs up to ~0.1 dB, hence SNR_TOLERANCE_DB.
        Fewer tokens make the measured gap noisier (KV: -0.19 dB at T=32 vs -0.11
        dB at T>=128), so below SNR_TOLERANCE_TOKENS the tolerance grows as
        1/sqrt(T). bf16 configs have no activation quant, so they are only
        reported. The residual must match exactly.
        """
        model = models.BY_NAME[name]
        args = models.make_inputs(model, t)
        got = run_kernel(model, *args)
        vllm = strict_reference(model.config, *args, numerics=Numerics.VLLM)
        truth = strict_reference(model.config, *args, numerics=Numerics.UNQUANTIZED)
        tolerance = SNR_TOLERANCE_DB * max(1, (SNR_TOLERANCE_TOKENS / t) ** 0.5)
        for out, g, v, e in zip(OUTPUTS, got, vllm, truth):
            if g is None:
                continue
            if out == "residual":
                np.testing.assert_array_equal(
                    np.asarray(g, np.float32), np.asarray(v, np.float32)
                )
                continue
            snr_kernel, snr_vllm = snr_db(g, e), snr_db(v, e)
            print(
                f"{name} T={t} {out}: SNR vs unquantized kernel={snr_kernel:.2f} dB"
                f" vllm={snr_vllm:.2f} dB",
                flush=True,
            )
            if model.config.fp8:
                self.assertGreaterEqual(snr_kernel, snr_vllm - tolerance, out)

    @parameterized.named_parameters(
        *(
            (f"_{m.name}_{layout.symbol}", m.name, layout)
            for m in models.MODELS
            for layout in KV_LAYOUTS
        )
    )
    def test_no_relayout_into_rpa(
        self, name: str, kv_layout: rpa_configs.KVLayout
    ) -> None:
        """fused_qkv's q and kv reach batched_rpa with no relayout in between."""
        model = models.BY_NAME[name]
        s = model.config
        t, seqs, page, pages = 1024, 8, 128, 64
        kv_cache = jnp.zeros(
            wrapper.get_kv_cache_shape(
                pages,
                page,
                s.num_kv_heads,
                s.head_dim,
                s.kv_dtype,
                kv_layout,
            ),
            s.kv_dtype,
        )
        q_lens = jnp.full((seqs,), t // seqs, jnp.int32)
        cu_q_lens = jnp.concatenate([jnp.zeros(1, jnp.int32), jnp.cumsum(q_lens)])
        page_indices = jnp.arange(pages, dtype=jnp.int32)
        dist = jnp.array([0, 0, seqs], jnp.int32)
        x, residual, weights, cos, sin = models.make_inputs(model, t)
        args = (x, residual, rpa.prepare_weights(s, weights), cos, sin)
        hlo = (
            jax.jit(fused_qkv_rpa, static_argnums=(0, 1))
            .lower(
                model,
                kv_layout,
                args,
                kv_cache,
                q_lens,
                page_indices,
                cu_q_lens,
                dist,
            )
            .compile()
            .as_text()
        )

        lines, users = _hlo_graph(hlo)
        producers = [n for n in lines if n.startswith("fused_qkv")]
        self.assertLen(producers, 1, hlo)
        rpa_consumers, bad, frontier = set(), [], list(users[producers[0]])
        while frontier:
            n = frontier.pop()
            line = lines[n]
            op = _opcode(line)
            if op in ("get-tuple-element", "bitcast", "copy-done") or (
                op == "copy-start" and _is_same_layout_copy(line)
            ):
                frontier += users[n]
            elif op == "custom-call" and n.startswith(("RPA", "rpa_prefetch_start")):
                rpa_consumers.add(n)
            elif not line.lstrip().startswith("ROOT"):
                bad.append(line.strip()[:400])
        print(f"{name}: fused_qkv -> RPA consumers:", sorted(rpa_consumers))
        self.assertEmpty(bad, "relayouts between fused_qkv and RPA")
        self.assertNotEmpty(rpa_consumers)


def _hlo_graph(hlo: str) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Returns ({name: line}, {name: [user names]}) for compiled HLO text."""
    lines = {}
    for line in hlo.splitlines():
        if m := re.match(r"\s*(ROOT )?%([\w.\-]+) = ", line):
            lines[m[2]] = line
    users = {n: [] for n in lines}
    for n, line in lines.items():
        for operand in re.findall(r"%([\w.\-]+)", _rhs(line)):
            if operand in users:
                users[operand].append(n)
    return lines, users


def _rhs(line: str) -> str:
    return line.split(" = ", 1)[1]


def _opcode(line: str) -> str:
    return re.search(r" ([a-z][a-z\-]*)\(%", line)[1]


def _is_same_layout_copy(line: str) -> bool:
    """Whether line is an HBM->VMEM prefetch, not a relayout.

    A prefetch's result and operand differ only in memory space.
    """
    shapes = re.findall(r"\w+\[[\d,]*\]\{[^}]*\}", _rhs(line))[:2]
    return len({re.sub(r"S\(\d\)", "", s) for s in shapes}) == 1


if __name__ == "__main__":
    absltest.main()
