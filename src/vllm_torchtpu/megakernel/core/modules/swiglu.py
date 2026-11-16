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
"""Streamed SwiGLU MLP (`SwiGLU`): the MoE shared expert and the dense FFN."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax

from ..decls import HFParamDecl, ParamTransform, VLLMAlias
from ..ops import BF16, F32, FP8, round_bf16
from ..spec import SmallField, WeightDecl


@dataclass(frozen=True)
class SwiGLU:
    """`down(silu(gate(x)) * up(x))` on the static dense weight ring.

    `gu` names the fused `[gate | up]` kernel weight (`K=H`, `N=2*IR`), `down`
    the down projection (`K=IR`, `N=H`); `IR` is the per-rank intermediate
    width. An fp8 weight carries per-column f32 scales in smalls `<name>_s`.
    """

    gu: str
    down: str
    gu_fp8: bool
    down_fp8: bool

    def weights(self, H: int, IR: int) -> list[WeightDecl]:
        return [
            WeightDecl(self.gu, K=H, N=2 * IR, dtype=FP8 if self.gu_fp8 else BF16),
            WeightDecl(self.down, K=IR, N=H, dtype=FP8 if self.down_fp8 else BF16),
        ]

    def smalls(self, H: int, IR: int) -> list[SmallField]:
        out = []
        if self.gu_fp8:
            out.append(SmallField(f"{self.gu}_s", 2 * IR))
        if self.down_fp8:
            out.append(SmallField(f"{self.down}_s", H))
        return out

    def emit(self, ctx: Any, slot: Any, IR: int, H: int) -> Any:
        """Return the `[MB, H]` f32 output for the bf16 input in `scr['xb']`."""
        stream, scr = ctx.stream, ctx.scr
        g13 = stream.dmatmul(self.gu, scr["xb"], 2 * IR)
        if self.gu_fp8:
            g13 = g13 * ctx.sm(slot, f"{self.gu}_s")
        mid = round_bf16(
            round_bf16(jax.nn.silu(round_bf16(g13[:, :IR]))) * round_bf16(g13[:, IR:])
        )
        scr["xb2"][:, :IR] = mid.astype(BF16)
        y = stream.dmatmul(self.down, scr["xb2"], H)
        return y * ctx.sm(slot, f"{self.down}_s") if self.down_fp8 else y

    def hf_params(
        self,
        n: Callable[[str], str],
        src: str,
        H: int,
        inter: int,
        *,
        canon: str = "",
        down_aliases: tuple[VLLMAlias, ...] = (),
    ) -> list[HFParamDecl]:
        """Decls for HF `<src>{gate,up,down}_proj`, named `n(<canon>gate_proj)`,
        `n(<canon>up_proj)` (fused into `gu`) and `n(down)`; `down_aliases` go
        before the vLLM `<src>down_proj` alias."""
        gate, up = n(f"{canon}gate_proj"), n(f"{canon}up_proj")
        gu_alias = VLLMAlias(
            source=f"{src}gate_up_proj",
            targets=(gate, up),
            scale_targets=(
                (n(f"{canon}gate_proj_s"), n(f"{canon}up_proj_s"))
                if self.gu_fp8
                else ()
            ),
            scale_source=f"{src}gate_up_proj.weight_scale",
            shard_axis=-1,
            already_sharded=True,
            already_fp8=True,
            flipped=True,
        )
        out: list[HFParamDecl] = []
        for order, (w_n, stem) in enumerate(((gate, "gate_proj"), (up, "up_proj"))):
            out.append(
                HFParamDecl(
                    name=w_n,
                    hf_name=f"{src}{stem}.weight",
                    shape=(H, inter),
                    hf_shape=(inter, H),
                    dtype=FP8 if self.gu_fp8 else BF16,
                    transform=ParamTransform(
                        fuse_into=self.gu,
                        fuse_order=order,
                        fuse_axis=-1,
                        transpose=(1, 0),
                        quantize="fp8" if self.gu_fp8 else None,
                        shard_axis=-1,
                    ),
                    vllm_aliases=(gu_alias,) if order == 0 else (),
                )
            )
            if self.gu_fp8:
                out.append(
                    HFParamDecl(
                        name=n(f"{canon}{stem}_s"),
                        hf_name=f"{src}{stem}.weight_scale_inv",
                        shape=(inter,),
                        hf_shape=(inter,),
                        dtype=F32,
                        transform=ParamTransform(
                            fuse_into=f"{self.gu}_s",
                            fuse_order=order,
                            fuse_axis=-1,
                            scale_for=w_n,
                            shard_axis=-1,
                            small_field=f"{self.gu}_s",
                        ),
                    )
                )
        d_n, ds_n = n(self.down), n(f"{self.down}_s")
        out.append(
            HFParamDecl(
                name=d_n,
                hf_name=f"{src}down_proj.weight",
                shape=(inter, H),
                hf_shape=(H, inter),
                dtype=FP8 if self.down_fp8 else BF16,
                transform=ParamTransform(
                    transpose=(1, 0),
                    quantize="fp8" if self.down_fp8 else None,
                    shard_axis=0,
                ),
                vllm_aliases=(
                    *down_aliases,
                    VLLMAlias(
                        source=f"{src}down_proj",
                        targets=(d_n,),
                        scale_targets=(ds_n,) if self.down_fp8 else (),
                        scale_source=f"{src}down_proj.weight_scale",
                        shard_axis=0,
                        already_sharded=True,
                        already_fp8=True,
                        flipped=True,
                    ),
                ),
            )
        )
        if self.down_fp8:
            out.append(
                HFParamDecl(
                    name=ds_n,
                    hf_name=f"{src}down_proj.weight_scale_inv",
                    shape=(H,),
                    hf_shape=(H,),
                    dtype=F32,
                    transform=ParamTransform(
                        scale_for=d_n,
                        shard_axis=None,
                        replicate_ranks=True,
                        small_field=f"{self.down}_s",
                    ),
                )
            )
        return out


def swiglu_reference(
    xb: Any, w_gu: Any, w_down: Any, IR: int, s_gu: Any = None, s_down: Any = None
) -> jax.Array:
    """One rank's f32 `SwiGLU` output for bf16-valued f32 rows `xb`."""
    g13 = xb @ w_gu.astype(F32)
    if s_gu is not None:
        g13 = g13 * s_gu
    mid = (
        round_bf16(
            round_bf16(jax.nn.silu(round_bf16(g13[:, :IR]))) * round_bf16(g13[:, IR:])
        )
        .astype(BF16)
        .astype(F32)
    )
    y = mid @ w_down.astype(F32)
    return y * s_down if s_down is not None else y


__all__ = ["SwiGLU", "swiglu_reference"]
