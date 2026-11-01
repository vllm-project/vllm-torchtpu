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
"""`TorchStepMetaOps`: the library's `StepMetaOps` step-metadata primitives
over `torch.Tensor` (Dynamo `fullgraph=True` safe), for programs that pack
their step metadata with `megakernel.runtime.step_meta`."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from vllm_torchtpu.megakernel.runtime.step_meta import check_ckpt_write_cols_distinct


class TorchStepMetaOps:
    """`StepMetaOps` primitives over `torch.Tensor` (Dynamo `fullgraph=True` safe)."""

    def shape(self, x: torch.Tensor) -> tuple[int, ...]:
        return tuple(int(d) for d in x.shape)

    def ndim(self, x: torch.Tensor) -> int:
        return int(x.ndim)

    def numel(self, x: torch.Tensor) -> int:
        return int(x.numel())

    def reshape(self, x: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
        return x.reshape(shape)

    def slice_or_pad_1d(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        x1 = x.reshape(-1)
        n = int(x1.shape[0])
        out = x1[:target_len] if n >= target_len else F.pad(x1, (0, target_len - n))
        return out.to(torch.int32)

    def slice_or_pad_2d(
        self, x: torch.Tensor, target_rows: int, target_cols: int
    ) -> torch.Tensor:
        r, c = int(x.shape[0]), int(x.shape[1])
        out = x[: min(r, target_rows), : min(c, target_cols)]
        pad_r, pad_c = max(0, target_rows - r), max(0, target_cols - c)
        if pad_r > 0 or pad_c > 0:
            out = F.pad(out, (0, pad_c, 0, pad_r), value=0)
        return out.to(torch.int32)

    def zeros_i32(self, shape: tuple[int, ...], *, like: torch.Tensor) -> torch.Tensor:
        return torch.zeros(shape, dtype=torch.int32, device=like.device)

    def zeros_like_i32(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    def arange_i32(self, start: int, stop: int, *, like: torch.Tensor) -> torch.Tensor:
        return torch.arange(start, stop, dtype=torch.int32, device=like.device)

    def clip(self, x: torch.Tensor, lo: int, hi: int | None = None) -> torch.Tensor:
        return x.clamp(min=lo) if hi is None else x.to(torch.long).clamp(min=lo, max=hi)

    def take_1d(self, table_1d: torch.Tensor, idx_1d: torch.Tensor) -> torch.Tensor:
        return table_1d[idx_1d].to(torch.int32)

    def take_col_2d(
        self, table_2d: torch.Tensor, col_idx_1d: torch.Tensor
    ) -> torch.Tensor:
        return table_2d.gather(1, col_idx_1d.unsqueeze(1))

    def expand_1d_to_2d(
        self, x_1d: torch.Tensor, cols: int, *, contiguous: bool = True
    ) -> torch.Tensor:
        out = x_1d.unsqueeze(1).expand(int(x_1d.shape[0]), int(cols))
        return out.contiguous() if contiguous else out

    def concat_cols(self, col0: torch.Tensor, col1: torch.Tensor) -> torch.Tensor:
        return torch.cat([col0, col1], dim=1)

    def stack_dim0(self, xs: Sequence[torch.Tensor]) -> torch.Tensor:
        return torch.stack(list(xs), dim=0)

    def where(
        self, cond: torch.Tensor, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        return torch.where(cond, x, y)

    def gt_zero(self, x: torch.Tensor) -> torch.Tensor:
        return x > 0

    def lt_zero(self, x: torch.Tensor) -> torch.Tensor:
        return x < 0

    def logical_and(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a & b

    def check_distinct_cols_2d(self, ckpt_g: torch.Tensor, T: int) -> None:
        if torch.compiler.is_compiling() or ckpt_g.device.type != "cpu":
            return
        check_ckpt_write_cols_distinct(ckpt_g, T)


__all__ = ["TorchStepMetaOps"]
