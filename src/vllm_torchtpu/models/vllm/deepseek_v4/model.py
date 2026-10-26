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
"""Native DeepSeek-V4 TPU model implementation."""

import re
import sys
import typing
from collections.abc import Callable, Iterable
from itertools import islice

import torch
import torch.nn as nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import (
    get_dp_group,
    get_pp_group,
    get_tp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.fused_moe import fused_moe_make_expert_params_mapping
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
)
from vllm.model_executor.offloader import NoopOffloader, set_offloader
from vllm.sequence import IntermediateTensors

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4.deepseek_v4_compressor import (
    VllmDeepseekCompressor,  # noqa: E501
)
from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4.deepseek_v4_mhc_op import (  # noqa: E501
    KernelOpts,
    MHCOps,
    get_mhc_ops,
    get_mhc_post_head_op,
    get_mhc_post_op,
    get_mhc_split_fn_op,
    get_mhc_stack_fn_op,
)
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.deepseek_v4.attention import VllmDeepseekV4MLAAttention
from vllm_torchtpu.models.vllm.deepseek_v4.layers import mhc_collapse_head
from vllm_torchtpu.layers.adapter.fused_moe_ep import MOE_DEFER_GATE, SP_GATE
from vllm_torchtpu.models.vllm.deepseek_v4.moe import (
    DeepseekV4MoE,
    dsv4_sequence_parallel,
)
from vllm_torchtpu.utils import fresh_buffer, host_rss_gib, release_memory_to_os

logger = init_logger(__name__)


class DeepseekV4DecoderLayer(nn.Module):
    """Single transformer decoder layer for DeepSeek-V4 with MHC residual stream."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream_list: list | None = None,
        seq_parallel: bool | None = None,
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.hidden_size = config.hidden_size
        self.prefix = prefix

        _m = re.search(r"layers\.(\d+)", prefix or "")
        self.layer_idx = int(_m.group(1)) if _m else -1

        self.rms_norm_eps = config.rms_norm_eps
        attn_cls = VllmDeepseekV4MLAAttention
        if envs.TPU_DSV4_ATTN_INTRA_CHIP_TP:
            from vllm_torchtpu.models.vllm.deepseek_v4.attn_intra_chip import (
                VllmDeepseekV4MLAAttentionIntraChip as attn_cls,
            )
        self.attn = attn_cls(
            vllm_config,
            prefix=f"{prefix}.attn",
            topk_indices_buffer=topk_indices_buffer,
            aux_stream_list=aux_stream_list,
        )
        # Sequence parallel: the layer's residual stream holds this rank's
        # 1/tp of the tokens; attention gathers its input and scatters its
        # (TP-partial) output, so wo_b must not all-reduce. The DSpark draft
        # layers (blocks of K + 1 rows) pass False.
        self.seq_parallel = (
            dsv4_sequence_parallel() if seq_parallel is None else seq_parallel
        )
        self.ffn = DeepseekV4MoE(
            vllm_config, prefix=f"{prefix}.ffn", seq_parallel=self.seq_parallel
        )
        if self.seq_parallel:
            if envs.TPU_DSV4_ATTN_INTRA_CHIP_TP:
                raise ValueError(
                    "TPU_DSV4_SEQ_PARALLEL and TPU_DSV4_ATTN_INTRA_CHIP_TP are "
                    "exclusive"
                )
            self.attn.wo_b.reduce_results = False

        self.attn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.ffn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        # The mHC kernels then emit attn_norm/ffn_norm-normalized inputs.
        self.fuse_norm = envs.VLLM_TPU_DSV4_MHC_FUSE_NORM
        self.mhc_opts = KernelOpts(
            gates_in_kernel=envs.VLLM_TPU_DSV4_MHC_GATES_IN_KERNEL,
            fuse_norm=self.fuse_norm,
            stacked_fn=envs.VLLM_TPU_DSV4_MHC_STACKED_FN,
            token_block=envs.VLLM_TPU_DSV4_MHC_TOKEN_BLOCK,
            gates_max_tokens=envs.VLLM_TPU_DSV4_MHC_GATES_MAX_TOKENS,
        )
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.hc_post_alpha = getattr(config, "hc_post_alpha", 2.0)
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * self.hidden_size
        self.hc_attn_fn = nn.Parameter(
            torch.empty(mix_hc, hc_dim, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_ffn_fn = nn.Parameter(
            torch.empty(mix_hc, hc_dim, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_attn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_ffn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_attn_scale = nn.Parameter(
            torch.empty(3, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_ffn_scale = nn.Parameter(
            torch.empty(3, dtype=torch.float32),
            requires_grad=False,
        )

    @property
    def mhc_ops(self) -> MHCOps:
        """This layer's mHC Pallas ops, built once outside the trace.

        The runner materializes this before the first compiled forward; see
        ``get_mhc_ops`` for why it cannot happen inside one.
        """
        ops = self.__dict__.get("_mhc_ops_instance")
        if ops is None:
            ops = get_mhc_ops(
                self.rms_norm_eps,
                self.hc_eps,
                self.hc_eps,
                self.hc_post_alpha,
                self.hc_sinkhorn_iters,
                opts=self.mhc_opts,
            )
            object.__setattr__(self, "_mhc_ops_instance", ops)
            # VLLM_TPU_DSV4_MHC_SEAM_FN_HBM: per seam site, a variant of the
            # fused op that streams fn from HBM (see envs).
            fh = envs.VLLM_TPU_DSV4_MHC_SEAM_FN_HBM
            fh_ops = None
            if fh in ("attn", "ffn", "both"):
                fh_ops = get_mhc_ops(
                    self.rms_norm_eps,
                    self.hc_eps,
                    self.hc_eps,
                    self.hc_post_alpha,
                    self.hc_sinkhorn_iters,
                    opts=self.mhc_opts._replace(fn_hbm=True),
                )
            seam_ops = {
                name: fh_ops.fused if fh_ops is not None and fh in (site, "both") else ops.fused
                for name, site in (("hc_attn_fn", "attn"), ("hc_ffn_fn", "ffn"))
            }
            object.__setattr__(self, "_seam_ops", seam_ops)
            self._arm_ar_seam()
            self._arm_moe_ar_seam()
            if self.seq_parallel:
                # Built outside the trace alongside the mHC ops.
                from vllm_torchtpu.models.vllm.deepseek_v4.seq_parallel import (
                    build_sp_ops,
                )

                object.__setattr__(self, "_sp_ops", build_sp_ops())
                shared = getattr(self.ffn, "shared_experts", None)
                if (
                    (
                        envs.TPU_DSV4_SEQ_PARALLEL_MIN_TOKENS > 0
                        or envs.TPU_DSV4_SP_SHARED_TP
                    )
                    and shared is not None
                    and not shared.arm_tp_slice(
                        get_tensor_model_parallel_rank(),
                        get_tensor_model_parallel_world_size(),
                    )
                ):
                    raise RuntimeError(
                        "TPU_DSV4_SEQ_PARALLEL_MIN_TOKENS / TPU_DSV4_SP_SHARED_TP: "
                        "the shared expert's "
                        "weight layout cannot be TP-sliced; unset it"
                    )
        return ops

    def _seam(self, fn_name: str):
        """The fused seam op for the seam that gates ``fn_name``'s sublayer."""
        ops = self.mhc_ops
        return self.__dict__.get("_seam_ops", {}).get(fn_name, ops.fused)

    def _arm_moe_ar_seam(self) -> None:
        """TPU_DSV4_MHC_MOE_AR_SEAM: build the op that sums the previous
        layer's fused-EP partial (its in-op shared all-reduce, deferred) and
        runs this layer's attention seam on the sum, in one Pallas call with
        hc_attn_fn DMA'd under the all-reduce. The model asks the previous
        layer's MoE for its partial only when this op exists."""
        object.__setattr__(self, "_moe_ar_seam_op", None)
        mode = envs.TPU_DSV4_MHC_MOE_AR_SEAM
        opts = self.mhc_opts
        if not (
            mode in (1, 2)
            and opts.gates_in_kernel
            and opts.fuse_norm
            and opts.stacked_fn
            and get_tp_group().world_size > 1
        ):
            return
        from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4 import (
            deepseek_v4_mhc_ar_seam_op as ar_seam,
        )

        op = ar_seam.get_ar_seam_op(
            self._ar_seam_consts(),
            self._ar_seam_kernel_kw(),
            moe_fn_mode="pin" if mode == 1 else "vmem",
        )
        if op is None or tuple(ar_seam.tp_ranks()) != tuple(
            get_tp_group().ranks
        ):
            return
        object.__setattr__(self, "_moe_ar_seam_op", op)

    def _ar_seam_consts(self) -> tuple:
        return (
            self.rms_norm_eps,
            self.hc_eps,
            self.hc_eps,
            self.hc_post_alpha,
            self.hc_sinkhorn_iters,
        )

    def _ar_seam_kernel_kw(self) -> dict:
        opts = self.mhc_opts
        kernel_kw = dict(
            gates_in_kernel=True,
            small_block=envs.VLLM_TPU_DSV4_MHC_SMALL_BLOCK,
            fn_chunks=envs.VLLM_TPU_DSV4_MHC_FN_CHUNKS,
        )
        if opts.gates_max_tokens >= 0:
            kernel_kw["gates_max_tokens"] = opts.gates_max_tokens
        if opts.token_block:
            kernel_kw["token_block_size"] = opts.token_block
        return kernel_kw

    def _arm_ar_seam(self) -> None:
        """TPU_DSV4_MHC_AR_SEAM: build the fused attention-AR + ffn-seam op
        (outside the trace, with the mHC ops) and hand wo_b's all-reduce to
        it. Only the served mHC form (in-kernel gates, fused norm, stacked
        fn) on a TP group the small all-reduce kernel serves."""
        object.__setattr__(self, "_ar_seam_op", None)
        opts = self.mhc_opts
        wo_b = self.attn.wo_b
        if not (
            envs.TPU_DSV4_MHC_AR_SEAM
            and opts.gates_in_kernel
            and opts.fuse_norm
            and opts.stacked_fn
            and not envs.TPU_DSV4_ATTN_INTRA_CHIP_TP
            and getattr(wo_b, "tp_size", 1) > 1
        ):
            return
        from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4 import (
            deepseek_v4_mhc_ar_seam_op as ar_seam,
        )

        kernel_kw = dict(
            gates_in_kernel=True,
            small_block=envs.VLLM_TPU_DSV4_MHC_SMALL_BLOCK,
            fn_chunks=envs.VLLM_TPU_DSV4_MHC_FN_CHUNKS,
        )
        if opts.gates_max_tokens >= 0:
            kernel_kw["gates_max_tokens"] = opts.gates_max_tokens
        if opts.token_block:
            kernel_kw["token_block_size"] = opts.token_block
        consts = (
            self.rms_norm_eps,
            self.hc_eps,
            self.hc_eps,
            self.hc_post_alpha,
            self.hc_sinkhorn_iters,
        )
        op = ar_seam.get_ar_seam_op(consts, kernel_kw)
        if op is None or tuple(ar_seam.tp_ranks()) != tuple(
            get_tp_group().ranks
        ):
            return
        object.__setattr__(self, "_ar_seam_op", op)
        # The fused op sums wo_b's partial; the replicated path below sums it
        # itself whenever the op is not taken.
        wo_b.reduce_results = False

    def _attn_seam(self, positions, x, residual, post_mix, res_mix):
        """Attention, its TP all-reduce and the ffn seam, as
        ``mhc_ops.fused(all_reduce(attn(x)), ...)``; one op when armed."""
        op = self.__dict__.get("_ar_seam_op")
        fn = (*self._fn3("hc_ffn_fn"), self.hc_ffn_scale, self.hc_ffn_base)
        norm = self._norm_arg(self.ffn_norm)
        x_p = self.attn(positions, x)
        if torch.compiler.is_compiling():
            from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4 import (
                deepseek_v4_mhc_ar_seam_op as ar_seam,
            )

            rank, peers = ar_seam.rank_peers()
            return op(x_p, residual, post_mix, res_mix, *fn, *norm, rank, peers)
        x = tensor_model_parallel_all_reduce(x_p)
        return self.mhc_ops.fused(x, residual, post_mix, res_mix, *fn, *norm)

    def split_hc_fns(self) -> None:
        """Split hc_attn_fn / hc_ffn_fn into the bf16 chunks the mHC ops take.

        The split depends only on the weight; doing it here, once after the
        weights land, keeps it out of every forward. Each buffer goes through
        `fresh_buffer`: a view or an aliased clone would otherwise be
        re-materialized as a `tt_jit_as_strided` program before every forward
        (~62 per decode step measured, j-decf9ff4 vs mHC off j-4eccc700).
        """
        if self.mhc_opts.stacked_fn:
            stack = get_mhc_stack_fn_op()
            for name in ("hc_attn_fn", "hc_ffn_fn"):
                self.register_buffer(
                    f"{name}_cat",
                    fresh_buffer(stack(getattr(self, name))),
                    persistent=False,
                )
        else:
            split = get_mhc_split_fn_op()
            for name in ("hc_attn_fn", "hc_ffn_fn"):
                for suffix, chunk in zip(
                    ("hi", "mid", "lo"), split(getattr(self, name))
                ):
                    self.register_buffer(
                        f"{name}_{suffix}", fresh_buffer(chunk), persistent=False
                    )
        if self.fuse_norm:
            # The kernels read the norm weight as (1, H); reshaping (H,) in
            # the forward is a relayout copy per seam (~1.4 us at decode).
            for norm_name in ("attn_norm", "ffn_norm"):
                w = getattr(self, norm_name).weight
                self.register_buffer(
                    f"{norm_name}_w2d",
                    fresh_buffer(w.detach().reshape(1, -1)),
                    persistent=False,
                )

    def _fn3(self, name: str) -> tuple[torch.Tensor, ...]:
        if self.mhc_opts.stacked_fn:
            return (getattr(self, f"{name}_cat"),)
        return tuple(getattr(self, f"{name}_{s}") for s in ("hi", "mid", "lo"))

    def _norm_arg(self, norm: RMSNorm) -> tuple[torch.Tensor, ...]:
        if not self.fuse_norm:
            return ()
        name = "attn_norm" if norm is self.attn_norm else "ffn_norm"
        return (getattr(self, f"{name}_w2d"),)

    def _norm(self, norm: RMSNorm, x: torch.Tensor) -> torch.Tensor:
        return x if self.fuse_norm else norm(x)

    def hc_pre(
        self,
        x: torch.Tensor,
        hc_fn3: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gate the residual streams into this sublayer's input.

        The gate constants are baked into the op, so only the tensors are
        passed; this just puts `layer_input` first for the caller.
        """
        post_mix, res_mix, layer_input = self.mhc_ops.pre(
            x, *hc_fn3, hc_scale, hc_base, *self._norm_arg(self.attn_norm)
        )
        return layer_input, post_mix, res_mix

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        x_partial: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Defer each post into the next pre so the pair is one kernel.

        x_partial (TPU_DSV4_MHC_MOE_AR_SEAM): `x` is the previous MoE's
        partial over the TP group (see the model loop); this layer's seam op
        sums it.

        `x` arrives as the previous sublayer's raw output and the caller
        carries `(residual, post_mix, res_mix)` -- the recombine that output
        still owes. The chain is opened by the layer-0 attention pre, which
        has no preceding post, and closed by the caller.
        """
        if residual is None:
            # Chain start: a standalone pre, nothing to fuse it with.
            residual = x
            x, post_mix, res_mix = self.hc_pre(
                x, self._fn3("hc_attn_fn"), self.hc_attn_scale, self.hc_attn_base
            )
        elif x_partial:
            from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4 import (
                deepseek_v4_mhc_ar_seam_op as ar_seam,
            )

            rank, peers = ar_seam.rank_peers()
            residual, post_mix, res_mix, x = self.__dict__["_moe_ar_seam_op"](
                x,
                residual,
                post_mix,
                res_mix,
                *self._fn3("hc_attn_fn"),
                self.hc_attn_scale,
                self.hc_attn_base,
                *self._norm_arg(self.attn_norm),
                rank,
                peers,
            )
        else:
            residual, post_mix, res_mix, x = self._seam("hc_attn_fn")(
                x,
                residual,
                post_mix,
                res_mix,
                *self._fn3("hc_attn_fn"),
                self.hc_attn_scale,
                self.hc_attn_base,
                *self._norm_arg(self.attn_norm),
            )

        x = self._norm(self.attn_norm, x)
        if self.__dict__.get("_ar_seam_op") is not None and (
            not self.seq_parallel or SP_GATE["replicated"]
        ):
            residual, post_mix, res_mix, x = self._attn_seam(
                positions, x, residual, post_mix, res_mix
            )
            x = self._norm(self.ffn_norm, x)
            x = self.ffn(x, input_ids)
            return x, residual, post_mix, res_mix
        if self.seq_parallel and SP_GATE["replicated"]:
            # A bucket TPU_DSV4_SEQ_PARALLEL_MIN_TOKENS keeps replicated: the
            # whole batch on every TP rank, wo_b's partial summed here.
            x = tensor_model_parallel_all_reduce(self.attn(positions, x))
        elif self.seq_parallel:
            # The views pin the row counts to the trace's own symbols (the
            # token count T of `positions`, and the local count of the
            # residual): the collectives' fakes return (T // tp) * tp and
            # T // tp, and TorchTPU refuses an allocation sized by an
            # expression it cannot resolve to an input dimension.
            sp_ops = self.__dict__["_sp_ops"]
            n_local = x.shape[0]
            x = self.attn(positions, sp_ops["ag"](x).view(positions.shape[0], -1))
            x = sp_ops["rs"](x).view(n_local, -1)
        else:
            x = self.attn(positions, x)

        residual, post_mix, res_mix, x = self._seam("hc_ffn_fn")(
            x,
            residual,
            post_mix,
            res_mix,
            *self._fn3("hc_ffn_fn"),
            self.hc_ffn_scale,
            self.hc_ffn_base,
            *self._norm_arg(self.ffn_norm),
        )
        x = self._norm(self.ffn_norm, x)
        x = self.ffn(x, input_ids)
        return x, residual, post_mix, res_mix


class _HostStagedParams:
    """Redirects weight-loading writes into host memory, one device write out.

    On this TPU backend, ``copy_()`` into an already-materialized slice/view of
    a device tensor allocates a fresh buffer per call instead of overwriting in
    place, so every per-expert / per-shard slice write during weight loading
    leaks its own size. A whole-tensor ``copy_()`` into the same destination is
    free.

    Every loader reached from ``load_weights`` writes through the parameter it
    is handed, so handing it a host-backed stand-in of the same class turns
    those slice writes into ordinary in-place host writes.
    Each staged parameter is then flushed with exactly one whole-tensor device
    ``copy_()``.
    """

    def __init__(self):
        # name -> (parameter, its real device tensor).
        self._staged: dict[str, tuple[torch.nn.Parameter, torch.nn.Parameter]] = {}

    def stage(self, name: str, param: torch.nn.Parameter) -> torch.nn.Parameter:
        """Return a host-backed stand-in for `param` for loaders to write."""
        if name not in self._staged:
            # ``_make_subclass`` is how nn.Parameter builds itself from a
            # tensor; using it keeps the stand-in the parameter's own class, so
            # upstream's isinstance dispatch (BlockQuantScaleParameter in
            # MergedColumnParallelLinear.weight_loader, say) still picks the
            # right branch and class-level loader methods bind to the stand-in.
            # ``param.data`` cannot simply be re-pointed at host storage:
            # this backend's tensors carry their own TensorImpl and set_data
            # rejects the dispatch-key change.
            host = typing.cast(
                torch.nn.Parameter,
                torch.Tensor._make_subclass(type(param), param.data.to("cpu"), False),
            )
            host.__dict__.update(param.__dict__)
            self._staged[name] = (param, host)
        return self._staged[name][1]

    def _flush(self, name: str) -> None:
        param, host = self._staged.pop(name)
        param.data.copy_(host.data)
        # `host` was the only reference to the staging buffer; dropping it
        # here returns that host memory to the allocator.
        del host

    def flush_all(self) -> None:
        """Flush every staged parameter and release all host buffers."""
        staged_bytes = sum(
            h.numel() * h.element_size() for _, h in self._staged.values()
        )
        logger.info(
            "Writing %d staged parameters (%.1f GiB of host memory) to the device.",
            len(self._staged),
            staged_bytes / 2**30,
        )
        for name in list(self._staged):
            self._flush(name)
        # The staging is tens of GiB per worker, eight workers a host, and
        # dropping the last reference does not by itself shrink the process:
        # the allocator keeps the freed memory for reuse (see
        # release_memory_to_os). Held, it is host DRAM the KV-offload store
        # cannot have, so hand it back now and log what came back.
        rss_before = host_rss_gib()
        release_memory_to_os()
        logger.info(
            "Released host memory after writing staged parameters: RSS "
            "%.1f GiB -> %.1f GiB.",
            rss_before,
            host_rss_gib(),
        )


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
        "inputs_embeds": 0,
    }
)
class DeepseekV4Model(nn.Module):
    """DeepSeek-V4 backbone model containing token embedding, layers, and head."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        set_offloader(NoopOffloader())
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.vocab_size = config.vocab_size
        self.hc_eps = config.hc_eps
        self.hc_mult = config.hc_mult
        self.hc_dim = self.hc_mult * config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps

        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
        )

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: DeepseekV4DecoderLayer(
                vllm_config=vllm_config,
                prefix=prefix,
                topk_indices_buffer=self.topk_indices_buffer,
                aux_stream_list=None,
            ),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, self.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.hc_head_fn = nn.Parameter(
            torch.empty(self.hc_mult, self.hc_dim, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(self.hc_mult, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32),
            requires_grad=False,
        )

        if get_pp_group().is_last_rank:
            self._mtp_hidden_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                self.hc_dim,
                dtype=typing.cast(torch.dtype, vllm_config.model_config.dtype),
            )
        else:
            self._mtp_hidden_buffer = None

        # Layer tags for eagle-family draft conditioning. Set via
        # ``DeepseekV4ForCausalLM.set_aux_hidden_state_layers`` before warmup.
        self.aux_hidden_state_layers: tuple[int, ...] = ()

        # TPU_DSV4_SEQ_PARALLEL: the layers run on this rank's 1/tp of the
        # tokens (contiguous block `tp_rank`); the outputs are gathered back.
        self.seq_parallel = dsv4_sequence_parallel()
        if self.seq_parallel and get_pp_group().world_size > 1:
            raise ValueError("TPU_DSV4_SEQ_PARALLEL does not support PP")

    def _sp_local_rows(self, tensors, rank):
        """This rank's contiguous 1/tp block of each tensor's rows. The rank is
        device data, so every TP rank traces the same graph."""
        tp = get_tp_group().world_size
        return [
            None
            if t is None
            else t.reshape(tp, -1, *t.shape[1:]).index_select(0, rank)[0]
            for t in tensors
        ]

    @property
    def mhc_post_op(self):
        """The op that settles the post the last decoder layer deferred.

        Only ``post`` is needed here, and it carries no gate constants, so
        this neither duplicates the layers' constants nor registers a new op
        -- it shares their cache entry.
        """
        op = self.__dict__.get("_mhc_post_op_instance")
        if op is None:
            op = get_mhc_post_op()
            object.__setattr__(self, "_mhc_post_op_instance", op)
            if getattr(self, "seq_parallel", False):
                from vllm_torchtpu.models.vllm.deepseek_v4.seq_parallel import (
                    build_sp_ops,
                )

                object.__setattr__(self, "_sp_ops", build_sp_ops())
        return op

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def _dp_gather_hash_moe_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """All-gather input_ids over the DP group for hash-MoE routing.

        Hash layers route on hash_indices_table[input_ids], so the ids must
        line up row-for-row with the hidden states the experts are handed.
        That is the DP-gathered global batch whenever the MoE runner gathers
        it, and this rank's own rows when the layer owns its dispatch -- the
        fused EP kernel does, and then gathering here would hand a layer ids
        for dp_size times the rows it computes. Uses the same
        all_gather(dim=0) as hidden_states to keep rows aligned.
        """
        dp_group = get_dp_group()
        if dp_group.world_size == 1:
            return input_ids
        # One gather serves every hash layer, so they have to agree on which
        # rows they want. They are built from one config and armed by one
        # weight-load path, so disagreeing is a bug here rather than a case to
        # handle -- and it would be a silent shape mismatch deep in a layer.
        hash_layers = typing.cast(
            Iterable[DeepseekV4DecoderLayer],
            self.layers[
                self.start_layer : min(self.end_layer, self.config.num_hash_layers)
            ],
        )
        wants = {layer.ffn.routes_dp_gathered_tokens for layer in hash_layers}
        assert len(wants) == 1, (
            "DeepSeek-V4 hash MoE layers disagree on whether the MoE runner "
            "DP-gathers the token rows, so no single gather of input_ids can "
            "serve them all"
        )
        if not wants.pop():
            return input_ids
        return dp_group.all_gather(input_ids, dim=0)

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (batch_size, self.hc_mult, self.config.hidden_size),
                    dtype=dtype,
                ),
            }
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        # TPU_DSV4_SEQ_PARALLEL_MIN_TOKENS: a shape branch (T <= cap stays
        # replicated), so each side of it is its own trace (shape_variants);
        # the layers and the fused EP read the switch while being traced.
        seq_parallel = self.seq_parallel and not (
            envs.TPU_DSV4_SEQ_PARALLEL_MIN_TOKENS > 0
            and positions.shape[0] <= envs.TPU_DSV4_SEQ_PARALLEL_MIN_TOKENS
        )
        if self.seq_parallel:
            SP_GATE["replicated"] = not seq_parallel
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            if seq_parallel:
                from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4.deepseek_v4_indexer import (  # noqa: E501
                    tp_rank_as_data,
                )

                sp_rank = tp_rank_as_data(get_tp_group(), hidden_states.device)
                (hidden_states,) = self._sp_local_rows([hidden_states], sp_rank)
            hidden_states = hidden_states.unsqueeze(-2).repeat(1, self.hc_mult, 1)
        else:
            assert intermediate_tensors is not None
            hidden_states = typing.cast(
                torch.Tensor, intermediate_tensors["hidden_states"]
            )

        # Hash routing is confined to the first `num_hash_layers`; only those
        # layers need ids gathered across DP.
        moe_input_ids = input_ids
        if input_ids is not None and self.start_layer < self.config.num_hash_layers:
            moe_input_ids = self._dp_gather_hash_moe_input_ids(input_ids)
            if seq_parallel:
                (moe_input_ids,) = self._sp_local_rows([moe_input_ids], sp_rank)

        residual, post_mix, res_mix = None, None, None
        # The fused tail needs the head here and at least one layer to defer
        # a post into it.
        fuse_tail = (
            envs.VLLM_TPU_DSV4_MHC_POST_HEAD
            and get_pp_group().is_last_rank
            and self.end_layer > self.start_layer
        )
        aux_hidden_states: list[torch.Tensor] = []
        final_aux_recon: torch.Tensor | None = None
        # TPU_DSV4_MHC_MOE_AR_SEAM: a layer's MoE hands its pre-all-reduce
        # partial to the next layer's attention seam, which sums it. Only on
        # the replicated layout, never into a chain end (the last layer, an
        # aux hidden state), and only where the next layer armed the op.
        # A shape branch (T <= cap, the small all-reduce's power-of-2 row cap):
        # above it every trace keeps the stock MoE all-reduce and seam ops.
        moe_seam = bool(envs.TPU_DSV4_MHC_MOE_AR_SEAM) and not seq_parallel
        if moe_seam:
            from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4 import (
                deepseek_v4_mhc_ar_seam_op as ar_seam,
            )

            moe_seam = positions.shape[0] <= ar_seam.moe_cap()
        x_partial = False
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            nxt = idx + 1
            defer = (
                moe_seam
                and nxt < self.end_layer
                and nxt not in self.aux_hidden_state_layers
                and self.layers[nxt].__dict__.get("_moe_ar_seam_op") is not None
            )
            MOE_DEFER_GATE["want"] = defer
            MOE_DEFER_GATE["done"] = False
            hidden_states, residual, post_mix, res_mix = layer(
                hidden_states,
                positions,
                moe_input_ids,
                post_mix,
                res_mix,
                residual,
                x_partial=x_partial,
            )
            x_partial = defer and MOE_DEFER_GATE["done"]
            MOE_DEFER_GATE["want"] = False
            MOE_DEFER_GATE["done"] = False
            if idx + 1 in self.aux_hidden_state_layers:
                if fuse_tail and idx + 1 == self.end_layer:
                    continue  # the fused tail below emits this aux mean
                if fuse_tail:
                    (aux_mean,) = self._post_head_op(False, True)(
                        hidden_states, residual, post_mix, res_mix,
                        self.hc_head_fn, self.hc_head_scale, self.hc_head_base,
                    )
                    aux_hidden_states.append(aux_mean)
                    continue
                # Reconstruct the full hc-stream state and collapse it the
                # way the drafts were trained on: settle the deferred post,
                # then mean over the hc streams (upstream numerics).
                aux_recon = self.mhc_post_op(hidden_states, residual, post_mix, res_mix)
                aux_hidden_states.append(aux_recon.mean(dim=1))
                final_aux_recon = aux_recon

        if fuse_tail:
            # One kernel: last post + hc head (+ the final aux mean).
            want_mean = self.end_layer in self.aux_hidden_state_layers
            outs = self._post_head_op(True, want_mean)(
                hidden_states, residual, post_mix, res_mix,
                self.hc_head_fn, self.hc_head_scale, self.hc_head_base,
            )
            if want_mean:
                aux_hidden_states.append(outs[1])
            hidden_states = self.norm(outs[0])
            hidden_states, aux_hidden_states = self._sp_gather(
                hidden_states, aux_hidden_states, positions.shape[0], seq_parallel
            )
            if aux_hidden_states:
                return hidden_states, aux_hidden_states
            return hidden_states

        if post_mix is not None:
            if (
                final_aux_recon is not None
                and self.end_layer in self.aux_hidden_state_layers
            ):
                hidden_states = final_aux_recon
            else:
                # Fused path: settle the post the last layer deferred.
                hidden_states = self.mhc_post_op(
                    hidden_states, residual, post_mix, res_mix
                )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        hidden_states = mhc_collapse_head(
            hidden_states,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.rms_norm_eps,
            self.hc_eps,
        )
        hidden_states = self.norm(hidden_states)
        hidden_states, aux_hidden_states = self._sp_gather(
            hidden_states, aux_hidden_states, positions.shape[0], seq_parallel
        )
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states

    def _sp_gather(self, hidden_states, aux_hidden_states, rows, seq_parallel):
        """Undo the sequence-parallel split on the backbone's outputs. `rows`
        is the step's token count (an input dimension of the trace)."""
        if not seq_parallel:
            return hidden_states, aux_hidden_states
        ag = self.__dict__["_sp_ops"]["ag"]

        def gather(t):
            return ag(t).view(rows, *t.shape[1:])

        return gather(hidden_states), [gather(a) for a in aux_hidden_states]

    def _post_head_op(self, want_head: bool, want_mean: bool):
        """The fused post + head / aux-mean op; built in the runner's op
        prebuild (``post_head_ops``) before compile."""
        ops = self.__dict__.setdefault("_post_head_ops", {})
        key = (want_head, want_mean)
        if key not in ops:
            ops[key] = get_mhc_post_head_op(
                self.rms_norm_eps, self.hc_eps, want_head=want_head,
                want_mean=want_mean,
            )
        return ops[key]

    @property
    def post_head_ops(self):
        """Prebuild every fused-tail variant outside the trace."""
        if envs.VLLM_TPU_DSV4_MHC_POST_HEAD:
            for key in ((True, False), (True, True), (False, True)):
                self._post_head_op(*key)
        return self.__dict__.get("_post_head_ops")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load checkpoint tensors into stacked, expert, sink and plain params."""
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
            ("attn.fused_wqa_wkv", "attn.wq_a", 0),
            ("attn.fused_wqa_wkv", "attn.wkv", 1),
            ("compressor.fused_wkv_wgate", "compressor.wkv", 0),
            ("compressor.fused_wkv_wgate", "compressor.wgate", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        # Weight loading writes each parameter in slices (per expert, per
        # shard, per TP rank), which leaks TPU HBM on this backend -- see
        # _HostStagedParams. Staging redirects those writes to host memory and
        # flushes each parameter to the device in one whole-tensor copy_.
        staged = _HostStagedParams()

        # attn_sink is stored per global head; each rank keeps its own slice.
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        n_head = self.config.num_attention_heads
        n_local_head = n_head // tp_size
        head_rank_start = n_local_head * tp_rank
        head_rank_end = n_local_head * (tp_rank + 1)

        # Pre-compute expert mapping ONCE.
        expert_mapping = self.get_expert_mapping()

        try:
            for name, loaded_weight in weights:
                for param_name, weight_name, shard_id in stacked_params_mapping:
                    # Skip non-stacked layers and experts (experts handled below).
                    if ".experts." in name:
                        continue
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)

                    if is_pp_missing_parameter(name, self):
                        break
                    if name not in params_dict:
                        continue
                    param = staged.stage(name, params_dict[name])
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                    loaded_params.add(name)
                    break
                else:
                    if ".experts." in name:
                        # E8M0 scales are stored as float8_e8m0fnu in
                        # checkpoints but the MoE param is uint8. copy_()
                        # would do a numeric conversion (e.g. 2^-7 → 0),
                        # destroying the raw exponent bytes.
                        if (
                            "weight_scale" in name
                            and loaded_weight.dtype == torch.float8_e8m0fnu
                        ):
                            loaded_weight = loaded_weight.view(torch.uint8)
                        for mapping in expert_mapping:
                            param_name, weight_name, expert_id, expert_shard_id = (
                                mapping
                            )
                            if weight_name not in name:
                                continue
                            name_mapped = name.replace(weight_name, param_name)
                            if is_pp_missing_parameter(name_mapped, self):
                                continue
                            if name_mapped not in params_dict:
                                continue
                            param = staged.stage(name_mapped, params_dict[name_mapped])
                            weight_loader = typing.cast(
                                Callable[..., bool], param.weight_loader
                            )
                            success = weight_loader(
                                param,
                                loaded_weight,
                                name_mapped,
                                shard_id=expert_shard_id,
                                expert_id=expert_id,
                                return_success=True,
                            )
                            if success:
                                loaded_params.add(name_mapped)
                                break
                        continue
                    elif "attn_sink" in name:
                        if is_pp_missing_parameter(name, self):
                            continue
                        if name not in params_dict:
                            continue
                        narrow_weight = loaded_weight[head_rank_start:head_rank_end]
                        n = narrow_weight.shape[0]
                        param = staged.stage(name, params_dict[name])
                        param.data[:n].copy_(narrow_weight)
                        loaded_params.add(name)
                        continue
                    else:
                        if is_pp_missing_parameter(name, self):
                            continue
                        if name not in params_dict:
                            continue
                        param = staged.stage(name, params_dict[name])
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                        loaded_params.add(name)
                        continue
        finally:
            staged.flush_all()
        for layer in self.layers:
            if isinstance(layer, DeepseekV4DecoderLayer):
                layer.split_hc_fns()

        return loaded_params

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        """Generate MoE expert parameter mappings for weight loading."""
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.n_routed_experts,
        )


def _make_deepseek_v4_weights_mapper(expert_dtype: str) -> WeightsMapper:
    """Weights mapper adapting checkpoint naming to vllm-torchtpu layer names."""
    if expert_dtype == "fp4":
        scale_regex = {
            re.compile(r"(\.experts\.\d+\.w[123])\.scale$"): r"\1.weight_scale",
            re.compile(r"\.scale$"): ".weight_scale_inv",
        }
    else:
        scale_regex = {
            re.compile(r"\.scale$"): ".weight_scale_inv",
        }
    return WeightsMapper(
        orig_to_new_prefix={
            "layers.": "model.layers.",
            "embed.": "model.embed.",
            "norm.": "model.norm.",
            "hc_head": "model.hc_head",
            "mtp.": "model.mtp.",
        },
        orig_to_new_regex=scale_regex,
        orig_to_new_suffix={
            "head.weight": "lm_head.weight",
            "embed.weight": "embed_tokens.weight",
            ".ffn.gate.bias": ".ffn.gate.e_score_correction_bias",
        },
        orig_to_new_substr={
            ".shared_experts.w2": ".shared_experts.down_proj",
        },
    )


class DeepseekV4ForCausalLM(nn.Module, SupportsPP):
    """DeepSeek-V4 causal language model registered for TPU inference."""

    model_cls = DeepseekV4Model
    hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper("fp4")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        # Tracing 61 layers at long context exceeds CPython's 1000 frame limit;
        # bumping recursion limit prevents recursion errors during AOT capture.
        sys.setrecursionlimit(50000)
        config = vllm_config.model_config.hf_config
        self.config = config
        expert_dtype = getattr(config, "expert_dtype", "fp4")
        if expert_dtype != "fp4":
            self.hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper(expert_dtype)

        self.model = self.model_cls(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        return getattr(self.model, "_mtp_hidden_buffer", None)

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        """Tag layers whose post-layer state feeds an eagle-family draft.
        Must run before warmup so the captured forward is the one that compiles.
        """
        invalid = [i for i in layers if i < 1 or i > len(self.model.layers)]
        if invalid:
            raise ValueError(f"Invalid DeepseekV4 aux hidden-state layers: {invalid}")
        self.model.aux_hidden_state_layers = tuple(layers)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        import safetensors.torch  # pyrefly: ignore

        safetensors.torch._TYPES["F8_E8M0"] = torch.uint8

        loader = AutoWeightsLoader(self)
        loaded_params = loader.load_weights(
            self.hf_to_vllm_mapper.apply(weights),
            mapper=WeightsMapper(orig_to_new_substr={"mtp.": None}),
        )

        # Post-load weight surgery goes here.
        # `fused_wkv_wgate` is built with `quant_config=None`, so it never
        # reaches a TPU linear method and the canonical (k, n) flip does not
        # apply; the compress-and-store kernel still wants
        # [hidden_size, 2 * coff * head_dim], so transpose it once here.
        for module in self.modules():
            if isinstance(module, VllmDeepseekCompressor):
                module.transpose_wkv_wgate()

        return loaded_params

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()
