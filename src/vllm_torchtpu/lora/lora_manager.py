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
"""LoRA managers that load an adapter into its TPU slot with one program."""

import contextlib
from collections.abc import Iterator

import torch
from vllm.lora.layers import (
    BaseLayerWithLoRA,
    ColumnParallelLinearWithLoRA,
    LogitsProcessorWithLoRA,
    MergedColumnParallelLinearVariableSliceWithLoRA,
    MergedColumnParallelLinearWithLoRA,
    MergedQKVParallelLinearWithLoRA,
    QKVParallelLinearWithLoRA,
    ReplicatedLinearWithLoRA,
    RowParallelLinearWithLoRA,
    VocabParallelEmbeddingWithLoRA,
)
from vllm.lora.lora_model import LoRAModel
from vllm.lora.model_manager import LRUCacheLoRAModelManager
from vllm.lora.request import LoRARequest
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager

from vllm_torchtpu.layers.adapter.custom_ops.lora_slot_update_op import (
    write_adapter_to_slot,
)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Layers whose reset_lora and set_lora write only through stacked[index], so
# their writes can be redirected to a host buffer that holds one slot.
_SUPPORTED_LAYERS = (
    ColumnParallelLinearWithLoRA,
    MergedColumnParallelLinearWithLoRA,
    MergedColumnParallelLinearVariableSliceWithLoRA,
    QKVParallelLinearWithLoRA,
    MergedQKVParallelLinearWithLoRA,
    RowParallelLinearWithLoRA,
    ReplicatedLinearWithLoRA,
    VocabParallelEmbeddingWithLoRA,
    LogitsProcessorWithLoRA,
)


def _as_tuple(
    weights: torch.Tensor | tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    return weights if isinstance(weights, tuple) else (weights,)


def _stacked_weights(modules: list[BaseLayerWithLoRA]) -> list[torch.Tensor]:
    return [
        weights
        for module in modules
        for stacked in (module.lora_a_stacked, module.lora_b_stacked)
        for weights in _as_tuple(stacked)
    ]


def _host_weights_like(
    device_weights: torch.Tensor | tuple[torch.Tensor, ...],
    host_views: Iterator[torch.Tensor],
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Takes the next host views, grouped like device_weights (a tuple or one
    tensor), so set_lora can index them as it indexes the TPU tensors."""
    if isinstance(device_weights, tuple):
        return tuple(next(host_views) for _ in device_weights)
    return next(host_views)


def _warn_fallback(reason: str) -> None:
    """Warns once that adapters keep vLLM's slice-by-slice writes."""
    logger.warning_once(
        "[LoRA] %s, so adapters are written to the TPU one slice at a "
        "time; adapter swaps and the decode steps after them run slower.",
        reason,
    )


class _AdapterSlotWriter:
    """Loads an adapter into one slot of every stacked LoRA tensor at once.

    TPULoRAModelManager wraps activate_adapter with it: while activate_adapter
    runs, each layer's set_lora writes into a host buffer that holds one slot
    instead of the TPU tensors, and write_slot then copies that buffer into
    the chosen slot with one program.
    """

    @classmethod
    def create(
        cls,
        modules: list[BaseLayerWithLoRA],
        max_loras: int,
        device: torch.device,
    ) -> "_AdapterSlotWriter | None":
        """Returns None, after a warning, when the writes cannot be staged."""
        if not modules:
            return None  # No LoRA layers on this worker, so nothing to write.
        unsupported = {
            type(module).__name__
            for module in modules
            if type(module) not in _SUPPORTED_LAYERS
        }
        if unsupported:
            _warn_fallback(f"LoRA layers {sorted(unsupported)} are not supported")
            return None
        stacked_weights = _stacked_weights(modules)
        # _host_slot_weights holds one slot of every tensor in one buffer, so
        # they all need one dtype. vLLM creates every lora_a_stacked and
        # lora_b_stacked in lora_config.lora_dtype, so this does not happen
        # today. Revisit this if vLLM gives LoRA A and B, or some layers,
        # different dtypes.
        if len({weights.dtype for weights in stacked_weights}) > 1:
            _warn_fallback("the stacked LoRA tensors mix dtypes")
            return None
        if len({id(weights) for weights in stacked_weights}) < len(stacked_weights):
            # A tensor donated twice in one call fails.
            _warn_fallback("two LoRA modules share a stacked tensor")
            return None
        return cls(modules, stacked_weights, max_loras, device)

    def __init__(
        self,
        modules: list[BaseLayerWithLoRA],
        stacked_weights: list[torch.Tensor],
        max_loras: int,
        device: torch.device,
    ) -> None:
        self._modules = modules
        # The TPU tensors that write_slot updates: all LoRA A and B tensors of
        # all layers in one list, e.g. [q_a, k_a, v_a, q_b, k_b, v_b, o_a, ...].
        # Each has shape [max_loras, ...]: index 0 picks the slot.
        self._stacked_weights = stacked_weights
        # Collects the new adapter's weights on the CPU, so they go to the TPU
        # in one copy instead of hundreds of small writes. Holds one slot of
        # each tensor above, in the same order:
        # [q_a[slot] | k_a[slot] | ... | o_b[slot]]
        self._host_slot_weights = torch.empty(
            sum(weights[0].numel() for weights in stacked_weights),
            dtype=stacked_weights[0].dtype,
        )
        # Tells the write program which slot to write. slot_indices[i] is slot
        # number i as a TPU tensor; as a program input (not a constant), one
        # compile works for every slot.
        self._slot_indices = [
            torch.tensor([slot], dtype=torch.int32, device=device)
            for slot in range(max_loras)
        ]
        self._device = device

        # Each layer's (lora_a_stacked, lora_b_stacked) on the TPU. Used to put
        # them back on the layers after set_lora is done.
        self._device_weights = [
            (module.lora_a_stacked, module.lora_b_stacked) for module in modules
        ]
        # Put on the layers while set_lora runs, so set_lora writes into
        # _host_slot_weights instead of the TPU. Same layout as
        # _device_weights.
        host_views = self._host_views()
        self._host_weights = [
            (
                _host_weights_like(lora_a, host_views),
                _host_weights_like(lora_b, host_views),
            )
            for lora_a, lora_b in self._device_weights
        ]

    def _host_views(self) -> Iterator[torch.Tensor]:
        """Makes the CPU views that set_lora writes into, one per tensor in
        _stacked_weights: its part of _host_slot_weights, shaped like the full
        tensor.

        All slots share the same memory. E.g. for q_a of shape [8, 1, 16, 1024],
        view[0], view[1], ..., view[7] are the same 1 x 16 x 1024 block. So
        set_lora can write to whichever slot vLLM picked and it lands in that
        block.
        """
        offset = 0
        for weights in self._stacked_weights:
            slot_size = weights[0].numel()
            yield (
                self._host_slot_weights[offset : offset + slot_size]
                .view(weights.shape[1:])
                .expand(weights.shape)
            )
            offset += slot_size

    @contextlib.contextmanager
    def redirect_writes_to_host(self) -> Iterator[None]:
        for module, (lora_a, lora_b) in zip(self._modules, self._host_weights):
            # nn.Module.__setattr__ is typed for tensors and modules, but the
            # linear layers store a tuple here, as vLLM does.
            # pyrefly: ignore[bad-argument-type]
            module.lora_a_stacked, module.lora_b_stacked = lora_a, lora_b
        try:
            yield
        finally:
            for module, (lora_a, lora_b) in zip(self._modules, self._device_weights):
                module.lora_a_stacked, module.lora_b_stacked = lora_a, lora_b

    def write_slot(self, slot: int) -> None:
        # dynamic_update_slice clamps an out-of-range index to a valid slot.
        assert 0 <= slot < len(self._slot_indices), slot
        slot_weights = self._host_slot_weights.to(self._device)
        write_adapter_to_slot(
            self._stacked_weights, slot_weights, self._slot_indices[slot]
        )

    def precompile(self) -> None:
        self._host_slot_weights.zero_()
        self.write_slot(0)

    @property
    def slot_bytes(self) -> int:
        """Size of one adapter slot, which write_slot holds on the TPU while
        it writes."""
        return self._host_slot_weights.nbytes


class TPULoRAModelManager(LRUCacheLoRAModelManager):
    """LRU LoRA model manager that loads an adapter into its slot at once.

    TPUWorkerLoRAManager creates it. vLLM's activate_adapter writes each
    layer's weights into the TPU slot slice by slice, which costs one small
    program per write and slows every later step. This manager keeps vLLM's
    slot choice and per-layer write logic, but lands those writes on the host
    and copies the finished slot to the TPU with one program.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._slot_writer = _AdapterSlotWriter.create(
            list(self.modules.values()), self.lora_slots, self.device
        )

    def activate_adapter(self, lora_id: int) -> bool:
        # vLLM calls this every step for each adapter in the batch; one that
        # already has a slot needs no write.
        if self._slot_writer is None or lora_id in self.lora_index_to_id:
            return super().activate_adapter(lora_id)
        try:
            with self._slot_writer.redirect_writes_to_host():
                activated = super().activate_adapter(lora_id)
            if activated:
                self._slot_writer.write_slot(self.lora_index_to_id.index(lora_id))
        except BaseException:
            # vLLM gives the adapter its slot before the weights land; without
            # this, a failed write would leave the slot serving stale weights.
            self.deactivate_adapter(lora_id)
            raise
        return activated

    def precompile_activation(self) -> None:
        """Compiles the slot write program at startup, before any adapter is
        active, by writing zeros into slot 0."""
        if self._slot_writer is None:
            return
        assert all(lora_id is None for lora_id in self.lora_index_to_id), (
            self.lora_index_to_id
        )
        self._slot_writer.precompile()

    def slot_write_hbm_bytes(self) -> int:
        """HBM that writing an adapter into its slot takes on top of the LoRA
        weights.

        The memory profile does not see it, since it is freed once the write
        is done.

        Without the slot writer, vLLM's writes copy each whole stacked tensor
        instead of updating it in place: reset_lora and set_lora each make a
        new copy while the old one still exists, and set_lora also uploads the
        adapter's weights. The writes run asynchronously, so the copies of
        earlier tensors can still exist while later tensors are written. How
        many of them exist at the same time depends on timing, so this counts
        all of them.
        """
        if self._slot_writer is not None:
            return self._slot_writer.slot_bytes
        stacked_weights = _stacked_weights(list(self.modules.values()))
        return sum(
            2 * weights.nbytes + weights[0].nbytes for weights in stacked_weights
        )


class TPUWorkerLoRAManager(LRUCacheWorkerLoRAManager):
    """Worker LoRA manager that uses TPULoRAModelManager.

    TPUModelRunner.load_lora_model creates it in place of the
    LRUCacheWorkerLoRAManager that vLLM's runner mixin hardcodes.
    """

    _manager_cls = TPULoRAModelManager

    def _load_adapter(self, lora_request: LoRARequest) -> LoRAModel:
        """Loads the adapter and rejects it if any LoRA weight is NaN or Inf.

        Without this check, NaN or Inf in one adapter's LoRA B breaks every
        request, not only the requests that use the adapter and not only the
        requests in the same batch. The expand in lora_ops multiplies every
        slot's LoRA B and adds the results, with 0 for the slots a token does
        not use, and 0 * NaN is NaN. So while the adapter sits in a slot, every
        step returns NaN for every token, even for requests without LoRA, until
        the adapter is evicted.

        NaN or Inf in LoRA A breaks only the adapter's own requests, but it is
        checked too, so such an adapter fails at load with a clear error. vLLM
        calls this before it evicts an old adapter to make room, or removes the
        adapter that a reload replaces, so after a rejection the adapters that
        were loaded stay loaded.
        """
        lora = super()._load_adapter(lora_request)
        for module_name, weights in lora.loras.items():
            for tensor in (weights.lora_a, weights.lora_b):
                if not torch.isfinite(tensor).all():
                    raise ValueError(
                        f"LoRA adapter {lora_request.lora_name!r} has NaN or Inf "
                        f"weights in {module_name}. TPU rejects such adapters, "
                        "since NaN or Inf in one adapter's LoRA B would reach "
                        "every request. Load an adapter with finite weights."
                    )
        return lora

    def precompile_activation(self) -> None:
        manager = self._adapter_manager
        assert isinstance(manager, TPULoRAModelManager)
        manager.precompile_activation()

    def slot_write_hbm_bytes(self) -> int:
        manager = self._adapter_manager
        assert isinstance(manager, TPULoRAModelManager)
        return manager.slot_write_hbm_bytes()
