# SPDX-License-Identifier: Apache-2.0
"""PLE storage extensions using the common table and checkpoint shard loader."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from vllm.distributed import GroupCoordinator
from vllm.models.qwen4_exp.common.ngram_embedding import Qwen4ExpPLEEmbeddingMethod
from vllm.models.qwen4_exp.common.ple import compute_ple_shard_overlap

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter.custom_ops.ple_op import (
    build_decode_op,
    build_raw_lookup_op,
)

from .ple import Qwen4ExpPLETable
from .ple_host import SharedPLETable


@dataclass(frozen=True)
class SharedPLEOptions:
    directory: Path
    key: str
    identity: dict[str, object]
    participants: int


class ResidentPLETable(Qwen4ExpPLETable):
    """Load CPU rows through upstream, then transfer one compact encoded table."""

    weight: torch.nn.Parameter
    table: torch.Tensor

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        embedding_method: Qwen4ExpPLEEmbeddingMethod,
        padding_size: int,
        prefix: str,
        parallel_group: GroupCoordinator | None = None,
    ) -> None:
        if num_embeddings >= 2**31:
            raise ValueError("PLE row indices must fit int32")
        self._loaded_ranges: set[tuple[int, int]] = set()
        self._owned_ranges: list[tuple[int, int]] = []
        self._finalized = False
        self._decode: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None
        super().__init__(
            num_embeddings,
            embedding_dim,
            embedding_method=embedding_method,
            padding_size=padding_size,
            prefix=prefix,
            parallel_group=parallel_group,
        )
        if self.output_dtype != torch.bfloat16:
            raise ValueError("Compact PLE storage requires BF16 model activations")
        self.register_buffer("table", torch.empty(0, dtype=torch.uint8))
        # This shard's first row, an operand of the lookup (see build_raw_lookup_op).
        self.register_buffer("vocab_start", torch.zeros(1, dtype=torch.int32))
        self._raw_lookup: (
            Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] | None
        ) = None

    def allocate_embedding_weight(
        self, num_embeddings: int, embedding_dim: int, dtype: torch.dtype
    ) -> torch.Tensor:
        if num_embeddings * embedding_dim // 128 >= 2**31:
            raise ValueError("PLE compact segment indices must fit int32")
        if dtype not in (torch.bfloat16, torch.float8_e4m3fn):
            raise ValueError(
                "Compact PLE storage requires BF16 or FP8 checkpoint weights"
            )
        self.storage_dtype = dtype
        return torch.empty(num_embeddings, embedding_dim, dtype=dtype, device="cpu")

    def _apply(self, fn, recurse=True):
        # The loader owns CPU storage until finalize_weights. A model-wide
        # device move must not copy a second, padded embedding table to HBM.
        weight = self._parameters.pop("weight")
        try:
            return super()._apply(fn, recurse)
        finally:
            self._parameters["weight"] = weight

    def weight_loader(
        self,
        param: torch.Tensor,
        loaded_weight: torch.Tensor,
        checkpoint_start: int | None = None,
    ) -> None:
        if self._finalized:
            raise ValueError("PLE table is already finalized")
        if checkpoint_start is None:
            raise NotImplementedError(
                "PLE table loading requires checkpoint row offsets"
            )
        if loaded_weight.dtype != self.storage_dtype:
            raise ValueError("PLE checkpoint dtype must match its embedding method")
        key = (checkpoint_start, loaded_weight.shape[0])
        if key in self._loaded_ranges:
            raise ValueError("Duplicate PLE checkpoint row range")
        self._copy_checkpoint(param, loaded_weight, checkpoint_start)
        overlap = compute_ple_shard_overlap(
            checkpoint_start=checkpoint_start,
            checkpoint_rows=loaded_weight.shape[0],
            tp_start=self.shard_indices.org_vocab_start_index,
            tp_end=self.shard_indices.org_vocab_end_index,
        )
        if overlap is not None:
            self._owned_ranges.append(
                (
                    overlap.destination_start,
                    overlap.destination_start + overlap.row_count,
                )
            )
        self._loaded_ranges.add(key)

    def _copy_checkpoint(self, param, loaded_weight, checkpoint_start):
        super().weight_loader(param, loaded_weight, checkpoint_start)

    def finalize_weights(self) -> None:
        if self._finalized:
            return
        end = 0
        for start, stop in sorted(self._owned_ranges):
            if start > end:
                raise ValueError("Missing PLE checkpoint rows")
            end = max(end, stop)
        if end != self.shard_indices.num_org_elements:
            raise ValueError("Missing PLE checkpoint rows")
        self.embedding_method.process_weights_after_loading(self)
        self._finish_storage()
        self._decode = build_decode_op()
        self._finalized = True

    def _finish_storage(self) -> None:
        if self.weight.numel() % 128:
            raise ValueError("Compact PLE table must fill 128-element storage segments")
        carrier = torch.int16 if self.storage_dtype == torch.bfloat16 else torch.uint8
        self.table = (
            self.weight.detach().view(carrier).reshape(-1, 128).to(self.table.device)
        )
        self.weight.data = torch.empty(0, dtype=self.storage_dtype, device="cpu")
        self.vocab_start = torch.tensor(
            [self.shard_indices.org_vocab_start_index],
            dtype=torch.int32,
            device=self.table.device,
        )
        self._raw_lookup = build_raw_lookup_op(
            self.shard_indices.num_org_elements,
            self.embedding_dim,
            envs.VLLM_TPU_PLE_LOOKUP_CHUNK_TOKENS,
        )

    def lookup_raw(self, ids: torch.Tensor) -> torch.Tensor:
        assert self._finalized and self._raw_lookup is not None
        return self._raw_lookup(self.table, ids, self.vocab_start)

    def decode(self, raw: torch.Tensor) -> torch.Tensor:
        assert self._finalized and self._decode is not None
        if self.storage_dtype == torch.bfloat16:
            scale = raw.new_ones((), dtype=self.output_dtype)
        else:
            scale = self.get_parameter("weight_scale")
        return self._decode(raw, scale)

    def local_rows(self, ids: torch.Tensor) -> torch.Tensor:
        rows = self.decode(self.lookup_raw(ids))
        return rows.reshape(*ids.shape, self.embedding_dim)


class HostPLETable(ResidentPLETable):
    """Keep checkpoint bytes on CPU, optionally in one sealed shared mapping."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        embedding_method: Qwen4ExpPLEEmbeddingMethod,
        padding_size: int,
        prefix: str,
        parallel_group: GroupCoordinator | None = None,
        shared: SharedPLEOptions | None = None,
    ) -> None:
        self._shared_options = shared
        self._shared_table: SharedPLETable | None = None
        super().__init__(
            num_embeddings,
            embedding_dim,
            embedding_method=embedding_method,
            padding_size=padding_size,
            prefix=prefix,
            parallel_group=parallel_group,
        )

    def allocate_embedding_weight(
        self, num_embeddings: int, embedding_dim: int, dtype: torch.dtype
    ) -> torch.Tensor:
        if dtype not in (torch.bfloat16, torch.float8_e4m3fn):
            raise ValueError(
                "Compact PLE storage requires BF16 or FP8 checkpoint weights"
            )
        self.storage_dtype = dtype
        options = self._shared_options
        if options is None:
            return super().allocate_embedding_weight(
                num_embeddings, embedding_dim, dtype
            )
        prototype = torch.empty((), dtype=dtype, device="cpu")
        self._shared_table = SharedPLETable(
            directory=options.directory,
            key=options.key,
            identity=dict(
                options.identity,
                dtype=str(dtype),
                rows=num_embeddings,
                head_dim=embedding_dim,
                start=self.shard_indices.org_vocab_start_index,
            ),
            nbytes=num_embeddings * embedding_dim * prototype.element_size(),
            participants=options.participants,
        )
        # No per-process staging allocation and no persistent writable view:
        # each loader call borrows the mapping, allowing finish() to seal it.
        return torch.empty(num_embeddings, embedding_dim, dtype=dtype, device="meta")

    def _copy_checkpoint(self, param, loaded_weight, checkpoint_start):
        table = self._shared_table
        if table is None:
            return super()._copy_checkpoint(param, loaded_weight, checkpoint_start)
        if table.is_writer:
            with table.writable_buffer() as buffer:
                destination = torch.frombuffer(
                    buffer, dtype=self.storage_dtype
                ).reshape(self.num_embeddings_per_partition, self.embedding_dim)
                try:
                    super()._copy_checkpoint(
                        destination, loaded_weight, checkpoint_start
                    )
                finally:
                    del destination

    def _finish_storage(self) -> None:
        if self._shared_table is not None:
            scale = (
                None
                if self.storage_dtype == torch.bfloat16
                else float(self.get_parameter("weight_scale").detach().cpu().item())
            )
            self._shared_table.finish(metadata={"scale": scale})

    def lookup_host(self, ids: torch.Tensor) -> torch.Tensor:
        assert self._finalized
        indices = ids.numpy().reshape(-1)
        start = self.shard_indices.org_vocab_start_index
        end = self.shard_indices.org_vocab_end_index
        dtype = np.int16 if self.storage_dtype == torch.bfloat16 else np.uint8
        if envs.VLLM_TPU_PLE_HOST_FAST_LOOKUP:
            raw = self._lookup_into_buffer(indices, start, end, dtype)
        else:
            owned = (indices >= start) & (indices < end)
            output = np.empty((indices.size, self.embedding_dim), dtype=dtype)
            if self._shared_table is None:
                table = (
                    self.weight.detach()
                    .view(torch.int16 if dtype == np.int16 else torch.uint8)
                    .numpy()
                )
                np.take(table, indices - start, axis=0, out=output, mode="clip")
            else:
                with self._shared_table.buffer() as buffer:
                    table = np.frombuffer(buffer, dtype=dtype).reshape(
                        -1, self.embedding_dim
                    )
                    np.take(table, indices - start, axis=0, out=output, mode="clip")
                    del table
            output[~owned] = 0
            raw = torch.from_numpy(output)
        if self.storage_dtype != torch.bfloat16:
            raw = raw.view(torch.int8)
        return raw.reshape(*ids.shape[:-1], ids.shape[-1] * self.embedding_dim)

    def _lookup_into_buffer(
        self, indices: np.ndarray, start: int, end: int, dtype
    ) -> torch.Tensor:
        """The same rows, gathered into one of two reused (pinned if possible)
        host buffers, so the step allocates nothing and the device copy can
        start without staging. Two buffers alternate: a step's lookup runs
        only after the previous step's forward, and so its copy, has run."""
        n = indices.size
        bufs = self.__dict__.setdefault("_lookup_bufs", [None, None])
        slot = self.__dict__.get("_lookup_slot", 0)
        self._lookup_slot = slot ^ 1
        buf = bufs[slot]
        if buf is None or buf.shape[0] < n:
            rows = max(n, 2 * (buf.shape[0] if buf is not None else 0))
            carrier = torch.int16 if dtype == np.int16 else torch.uint8
            try:
                buf = torch.empty(
                    (rows, self.embedding_dim), dtype=carrier, pin_memory=True
                )
            except RuntimeError:
                buf = torch.empty((rows, self.embedding_dim), dtype=carrier)
            bufs[slot] = buf
        out_t = buf[:n]
        output = out_t.numpy()
        local = indices - start
        if self._shared_table is None:
            table = self.__dict__.get("_np_table")
            if table is None:
                table = self.weight.detach().view(buf.dtype).numpy()
                self._np_table = table
            np.take(table, local, axis=0, out=output, mode="clip")
        else:
            with self._shared_table.buffer() as buffer:
                table = np.frombuffer(buffer, dtype=dtype).reshape(
                    -1, self.embedding_dim
                )
                np.take(table, local, axis=0, out=output, mode="clip")
                del table
        stray = (local < 0) | (local >= end - start)
        if stray.any():
            output[stray] = 0
        return out_t

    def lookup_raw(self, ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Host PLE requires the explicit host lookup boundary")

    def close(self) -> None:
        if self._shared_table is not None:
            self._shared_table.close()
        self.weight.data = torch.empty(
            0, dtype=self.storage_dtype, device=self.weight.device
        )
        self._finalized = False
