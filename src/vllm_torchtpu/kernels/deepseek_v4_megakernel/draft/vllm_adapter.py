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
"""The DSpark draft forward as a second megakernel call (B9 Stage 1).

When the target step ran on the megakernel, `DSparkProposer.propose` hands
the draft block (the stock input ids / positions / seq lens), the target
step's aux hidden states and verify-row positions, and the draft layers'
block tables to `DraftMegakernel.forward` instead of running the context-KV
insert and the draft model; the stock logits + Markov sampling then run on
the returned hidden (`_sample_from_hidden`). An idle DP rank runs the same
step with every request inactive (`idle`), so in-kernel collectives pair.

The draft bucket of a target bucket holds the same requests: q = the draft
block (K rows per request), `ctx_q` = the target's verify rows.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any

import torch

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.deepseek_v4_megakernel import kernel_interface as KI
from vllm_torchtpu.kernels.deepseek_v4_megakernel import vllm_adapter as VA
from vllm_torchtpu.kernels.deepseek_v4_megakernel import weight_map as WM
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# meta, input_ids, positions, seq_lens, ctx_aux, ctx_positions, ctx_seq_lens
_HEAD_OPERANDS = 7


def draft_bucket(target: KI.Bucket, q: int) -> KI.Bucket:
    """The draft step serving `target`'s requests with q draft rows each."""
    reqs_group = target.rows // KI.request_block_rows(target.q)
    return dataclasses.replace(target, q=q, rows=reqs_group * KI.request_block_rows(q))


class DraftMegakernel:
    """Built with the target megakernel at load, bound with it at capture."""

    def __init__(
        self,
        *,
        spec: KI.ModelSpec,
        ctx_q: int,
        n_aux: int,
        kernel: Any,
        mesh: Any,
        rank: int,
        expert_block: int,
        weights: list[torch.Tensor],
        draft_layers: list[Any],
        device: torch.device | str,
        compile_fn: Callable[[Callable], Callable] | None = None,
        op_namespace: str = "dsv4_megakernel_draft",
        direct_dispatch: bool = True,
    ) -> None:
        self.spec, self.ctx_q, self.n_aux, self.kernel = spec, ctx_q, n_aux, kernel
        self.mesh, self.rank, self.weights = mesh, rank, weights
        self.draft_layers = draft_layers
        # forward returns the draft rows' logits (compute_draft_logits)
        self.logits = bool(getattr(kernel, "logits", False))
        self.device = torch.device(device)
        self.op_namespace = op_namespace
        self.direct_dispatch = direct_dispatch
        self._compile = compile_fn or VA.DSv4Megakernel._default_compile
        meta = torch.zeros(KI.META_LEN, dtype=torch.int32)
        meta[KI.META_RANK] = rank
        meta[KI.META_Q] = spec.q
        meta[KI.META_EBLK] = expert_block
        self.meta = meta.to(self.device)
        self._metas: dict[int, torch.Tensor] = {spec.q: self.meta}
        self.cfg: KI.StepConfig | None = None
        self.caches: VA.BoundCaches | None = None
        self._steps: dict[KI.Bucket, Callable] = {}
        self._preps: dict[KI.Bucket, Callable] = {}
        self._idle: dict[KI.Bucket, list[torch.Tensor]] = {}
        self._draft_buckets: dict[KI.Bucket, KI.Bucket] = {}
        # the target's VA.RunTrace (maybe_create)
        self.trace: Any = VA.RunTrace(rank, 0, False)

    def _draft_q_for(self, target: KI.Bucket) -> int:
        if target.q != self.ctx_q and target.q > 1:
            return target.q - 1
        return self.spec.q

    def _meta_for(self, target: KI.Bucket) -> torch.Tensor:
        dq = self._draft_q_for(target)
        meta = self._metas.get(dq)
        if meta is None:
            m = self.meta.detach().clone()
            m[KI.META_Q] = dq
            meta = self._metas[dq] = m.to(self.device)
        return meta

    @classmethod
    def maybe_create(cls, runner: Any, target: Any) -> DraftMegakernel | None:
        """With VLLM_TPU_DSV4_MK_DRAFT and a DSpark drafter; None otherwise."""
        if not envs.VLLM_TPU_DSV4_MK_DRAFT:
            return None
        drafter = getattr(runner, "drafter", None)
        VA._need(
            hasattr(drafter, "_sample_from_hidden"),
            f"draft megakernel: the drafter {type(drafter).__name__} is not DSpark",
        )
        VA._need(
            not target.spec.moe_placement,
            "draft megakernel with a static expert placement (its layers' "
            "placement rows are not passed yet)",
        )
        VA._need(
            target.spec.attn_tp == 1,
            "draft megakernel under vLLM TP (its weights are not mapped from TP "
            "shards yet)",
        )
        draft_model = drafter.draft_model
        layers = list(draft_model.model.layers)
        dcfg = draft_model.config
        tspec = target.spec
        spec = dataclasses.replace(
            tspec,
            num_layers=len(layers),
            compress_ratios=(1,) * len(layers),
            q=int(drafter.block_size),
            aux_layers=(),
            num_hash_layers=0,
        )
        n_aux = len(tuple(dcfg.dspark_target_layer_ids))
        VA._need(
            n_aux == len(tspec.aux_layers),
            f"draft reads {n_aux} target aux layers, the target emits "
            f"{len(tspec.aux_layers)}",
        )
        g = WM.Geometry.from_hf(runner.model_config.hf_config, tspec.compress_ratios)
        shared = {}
        VA._need(
            not (target.logits and getattr(draft_model, "has_own_lm_head", False)),
            "logits mode: the draft has its own lm_head (the megakernel applies "
            "the target's)",
        )
        if target.logits:  # one lm_head shard for both steps
            names = [e.name for e in KI.weight_manifest(tspec, logits=True)]
            shared["lm_head"] = target.weights[names.index("lm_head")]
        with torch.no_grad():
            weights = target.kernel.collect_draft_weights(
                draft_model, spec, g, n_aux, shared
            )
        if weights and weights[0].device.type == "tpu":
            VA._sync(weights)
        logger.info(
            "DeepSeek-V4 draft megakernel: %d layers, %d weight operands",
            len(layers),
            len(weights),
        )
        draft = cls(
            spec=spec,
            ctx_q=tspec.q,
            n_aux=n_aux,
            kernel=target.kernel,
            mesh=target.mesh,
            rank=target.rank,
            expert_block=int(target.meta[KI.META_EBLK].item()),
            weights=weights,
            draft_layers=layers,
            device=target.device,
            direct_dispatch=target.direct_dispatch,
        )
        draft.trace = target.trace
        return draft

    # ------------------------------------------------------------ binding

    def bind_caches(self, runner: Any, target_buckets: Sequence[KI.Bucket]) -> None:
        groups = runner.kv_cache_config.kv_cache_groups
        group_of = {n: gid for gid, g in enumerate(groups) for n in g.layer_names}
        caches = VA.bind_layer_caches(self.draft_layers, group_of)
        widths = [
            int(runner.input_batch.block_table[gid].max_num_blocks_per_req)
            for gid in caches.group_ids
        ]
        pages = {int(lyr.attn.swa_cache_layer.block_size) for lyr in self.draft_layers}
        VA._need(len(pages) == 1, f"draft SWA page sizes differ: {pages}")
        self.bind(caches, widths, target_buckets, swa_page=pages.pop())

    def bind(
        self,
        caches: VA.BoundCaches,
        group_widths: Sequence[int],
        target_buckets: Sequence[KI.Bucket],
        swa_page: int = 128,
    ) -> None:
        for i, t in enumerate([*caches.arrays, *self.weights]):
            got = VA.settled_layout(t)
            VA._need(
                got is True,
                f"draft operand {i} {tuple(t.shape)} {t.dtype}: "
                + ("a view" if got is None else "not in the default device layout"),
            )
        self.caches = caches
        self.cfg = KI.StepConfig(
            swa_page=swa_page,
            model=self.spec,
            layers=caches.layers,
            cache_shapes=tuple(
                (tuple(t.shape), str(t.dtype).removeprefix("torch."))
                for t in caches.arrays
            ),
            group_widths=tuple(group_widths),
        )
        for tb in target_buckets:
            b = draft_bucket(tb, self._draft_q_for(tb))
            self._draft_buckets[tb] = b
            self._steps[tb] = self._build_step(b, ctx_q=tb.q)
            self._preps[tb] = self._build_prep(b, tb)
            self._idle[tb] = self._idle_operands(b, ctx_q=tb.q)
        logger.info(
            "DeepSeek-V4 draft megakernel bound: %d cache arrays, KV groups %s",
            len(caches.arrays),
            caches.group_ids,
        )

    @property
    def bound(self) -> bool:
        return self.cfg is not None

    def _build_step(self, bucket: KI.Bucket, ctx_q: int | None = None) -> Callable:
        cfg, caches = self.cfg, self.caches
        assert cfg is not None and caches is not None
        VA._need(self.kernel.cache_inplace, "draft megakernel: in-place caches only")
        eff_ctx_q = self.ctx_q if ctx_q is None else ctx_q
        fn = self.kernel.build_draft_step(cfg, bucket, eff_ctx_q)
        n_bt = len(cfg.group_widths)
        n_in = _HEAD_OPERANDS + n_bt + len(self.weights) + len(caches.arrays)
        tag = f"q{bucket.q}_t{bucket.rows}"
        flat = VA.export_step(fn, self.mesh, n_in, 1, f"dsv4_draft_{tag}",
                              f"{self.op_namespace}::step_{tag}")  # fmt: skip
        call = VA.inplace_call(flat, _HEAD_OPERANDS + n_bt, self.direct_dispatch)

        def step(head, block_tables, weights, caches):
            return call((*head, *block_tables), weights, caches)[0]

        return step

    def _build_prep(self, bucket: KI.Bucket, target: KI.Bucket) -> Callable:
        """The proposer's padded draft block and the target step's verify rows
        -> the draft bucket's fixed shapes; requests past `num_reqs` (the
        step's request count) are inactive, their rows zero."""
        K, B, M = bucket.q, bucket.reqs, bucket.own_rows
        ctx_q = target.q
        Mc = B * ctx_q
        widths = self.cfg.group_widths
        fit = VA._fit_rows

        def prep(input_ids, positions, seq_lens, ctx_aux, ctx_positions,
                 ctx_seq_lens, num_reqs, *block_tables):  # fmt: skip
            dev = seq_lens.device
            active = torch.arange(B, dtype=torch.int32, device=dev) < num_reqs
            rows = fit(active.repeat_interleave(K), M)
            crows = fit(active.repeat_interleave(ctx_q), Mc)
            zi = torch.zeros((), dtype=torch.int32, device=dev)
            ids = torch.where(rows, fit(input_ids, M).int(), zi)
            pos = torch.where(rows, fit(positions, M).int(), zi)
            seq = torch.where(active, fit(seq_lens, B).int(), zi)
            aux = fit(torch.cat(list(ctx_aux), dim=-1), Mc).to(torch.bfloat16)
            aux = torch.where(crows[:, None], aux, torch.zeros_like(aux))
            cpos = torch.where(crows, fit(ctx_positions, Mc).int(), zi)
            cseq = torch.where(active, fit(ctx_seq_lens, B).int(), zi)
            tables = []
            for bt, width in zip(block_tables, widths):
                bt = fit(fit(bt.reshape(ctx_seq_lens.shape[0], -1), B), width, 1)
                tables.append(torch.where(active[:, None], bt.int(), zi))
            return (ids, pos, seq, aux, cpos, cseq, *tables)

        prep.__code__ = prep.__code__.replace(
            co_name=f"dsv4_draft_prep_q{K}_t{bucket.rows}"
        )
        return self._compile(prep)

    def _idle_operands(
        self, bucket: KI.Bucket, ctx_q: int | None = None
    ) -> list[torch.Tensor]:
        z = torch.zeros
        eff_ctx_q = self.ctx_q if ctx_q is None else ctx_q
        B, Mc, i32 = bucket.reqs, bucket.reqs * eff_ctx_q, torch.int32
        n_aux_h = self.n_aux * self.spec.hidden_size
        ops = [z(bucket.own_rows, dtype=i32), z(bucket.own_rows, dtype=i32),
               z(B, dtype=i32), z((Mc, n_aux_h), dtype=torch.bfloat16),
               z(Mc, dtype=i32), z(B, dtype=i32)]  # fmt: skip
        ops += [z((B, w), dtype=i32) for w in self.cfg.group_widths]
        return [t.to(self.device) for t in ops]

    # ------------------------------------------------------------ per step

    def forward(
        self,
        target: KI.Bucket,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        seq_lens: torch.Tensor,
        ctx_aux: Sequence[torch.Tensor],
        ctx_positions: torch.Tensor,
        ctx_seq_lens: torch.Tensor,
        num_reqs: torch.Tensor,
        block_tables: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """The draft hidden (pre-norm; with `logits`, the draft rows' logits) at
        the proposer's padded draft rows.
        `block_tables` follow `caches.group_layer_names` (the draft layers'
        groups, from the target step's metadata)."""
        ops = self._preps[target](input_ids, positions, seq_lens, ctx_aux,
                                  ctx_positions, ctx_seq_lens, num_reqs,
                                  *block_tables)  # fmt: skip
        with self.trace(f"draft {target.key} busy"):
            hidden = self._run(target, ops)
        return VA._fit_rows(hidden, int(input_ids.shape[0]))

    def idle(self, target: KI.Bucket) -> None:
        """An idle DP rank's draft step (every request inactive)."""
        with self.trace(f"draft {target.key} idle"):
            self._run(target, self._idle[target])

    def _run(self, target: KI.Bucket, ops: Sequence[torch.Tensor]) -> torch.Tensor:
        assert self.caches is not None
        head = (self._meta_for(target), *ops[:6])
        return self._steps[target](
            head, list(ops[6:]), self.weights, self.caches.arrays
        )

    # ------------------------------------------------------------ warmup

    def warmup(self, drafter: Any = None) -> None:
        """Compile and run every draft step idle, synced (torch_tpu compiles
        at the first sync); every rank runs the same sequence. Then the
        drafter's sampling graph on this step's output."""
        for tb in self._steps:
            with torch.no_grad():
                VA._sync([self._run(tb, self._idle[tb])])
            logger.info("DeepSeek-V4 draft megakernel: warmed bucket %s", tb.key)
        if drafter is not None:
            width = self.spec.vocab_size if self.logits else self.spec.hidden_size
            drafter.precompile_megakernel_sampling(self.logits, width)
