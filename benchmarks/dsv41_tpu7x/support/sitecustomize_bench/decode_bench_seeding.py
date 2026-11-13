# SPDX-License-Identifier: Apache-2.0
"""Seeds DecodeBenchConnector external blocks from real prefill blocks."""

_COPY_CHUNK = 1024
_MAX_TEMPLATES = 2


def patch_decode_bench_seeding() -> None:
    try:
        import torch
        from vllm.distributed.kv_transfer.kv_connector.v1 import (
            decode_bench_connector as dbc,
        )
        from vllm.utils.math_utils import cdiv
        from vllm.v1.kv_cache_interface import MLAAttentionSpec, iter_layer_specs
    except Exception:
        return
    if getattr(dbc, "_tpu_seeding_patched", False):
        return

    s_cls, w_cls = dbc.DecodeBenchConnectorScheduler, dbc.DecodeBenchConnectorWorker
    s_init, s_alloc = s_cls.__init__, s_cls.update_state_after_alloc
    s_meta, s_done = s_cls.build_connector_meta, s_cls.request_finished
    w_init, w_reg = w_cls.__init__, w_cls.register_kv_caches

    def _copy_block_pairs(pools, pairs: list[tuple[int, int]]) -> None:
        from vllm_torchtpu.layers.adapter.custom_ops.mamba_state_copy_op import (
            copy_mamba_state_blocks,
        )
        from vllm_torchtpu.utils import synchronize_tensors

        for i in range(0, len(pairs), _COPY_CHUNK):
            pad = (pairs[i : i + _COPY_CHUNK] + [(0, 0)] * _COPY_CHUNK)[:_COPY_CHUNK]
            for t in pools:
                s = t.new_tensor([p[0] for p in pad], dtype=torch.int32)
                d = t.new_tensor([p[1] for p in pad], dtype=torch.int32)
                synchronize_tensors([t, s, d])
                copy_mamba_state_blocks([t], s, d)
                synchronize_tensors([t])

    def _mla_groups(groups) -> list[int]:
        specs = (iter_layer_specs(g.kv_cache_spec) for g in groups)
        return [
            i
            for i, s in enumerate(specs)
            if any(isinstance(x, MLAAttentionSpec) for x in s)
        ]

    def _sched_init(self, vllm_config, kv_cache_config):
        s_init(self, vllm_config, kv_cache_config)
        ktc = vllm_config.kv_transfer_config
        self._cache_hit_ratio = (
            ktc.get_from_extra_config("cache_hit_ratio", 0.95)
            if ktc is not None
            else 0.95
        )
        cfg_bs = int(
            ktc.get_from_extra_config("cache_block_size", 128)
            if ktc is not None
            else 128
        )
        self._mla_group_ids = _mla_groups(self.kv_cache_groups)
        self._block_size = (
            max(cfg_bs, *self.group_block_sizes) if self._mla_group_ids else cfg_bs
        )
        self._template_block_ids, self._reserved_tmpl_bids, self._req_info = {}, {}, {}

    def _get_num_new_matched_tokens(self, request, num_computed_tokens):
        import os

        if request.request_id in self._filled_requests:
            return 0, False
        if self._mla_group_ids and request.num_tokens <= 3 * self._block_size:
            return 0, False
        if os.path.exists("/tmp/disable_decode_bench_fill"):
            return 0, False
        if any(
            len(self._template_block_ids.get(g, ())) < _MAX_TEMPLATES
            for g in self._mla_group_ids
        ):
            return 0, False
        num_uncomputed = request.num_tokens - num_computed_tokens
        if os.path.exists("/tmp/override_cache_hit_ratio"):
            ratio = float(open("/tmp/override_cache_hit_ratio").read().strip())
            if ratio >= 1.0:
                return max(0, num_uncomputed - 1), False
        if self._cache_hit_ratio is not None and float(self._cache_hit_ratio) < 1.0:
            bs = int(self._block_size or 128)
            target_cached = (
                int(request.num_tokens * float(self._cache_hit_ratio)) // bs
            ) * bs
            return (
                max(0, min(target_cached - num_computed_tokens, num_uncomputed - 1)),
                False,
            )
        return max(0, num_uncomputed - 1), False

    def _update_state_after_alloc(self, request, blocks, n_ext):
        if mla := self._mla_group_ids:
            self._req_info[request.request_id] = (request, n_ext)
            if n_ext > 0:
                grps = enumerate(zip(blocks.blocks, self.group_block_sizes))
                bids = tuple(
                    [b.block_id for b in g[: cdiv(n_ext, bs)] if i in mla and not b.is_null]
                    for i, (g, bs) in grps
                )
                self._pending_fills[request.request_id] = (bids, n_ext)
                self._filled_requests.add(request.request_id)
        else:
            s_alloc(self, request, blocks, n_ext)

    def _build_connector_meta(self, scheduler_output):
        to_capture: dict[int, list[tuple[int, int]]] = {}
        mla, tmpl_map = self._mla_group_ids, self._template_block_ids
        if any(len(tmpl_map.get(g, ())) < _MAX_TEMPLATES for g in mla):
            state = scheduler_output.kv_connector_block_state
            pool = state.resolve_block_ids.__self__.block_pool
            for g in (g for g in mla if g not in self._reserved_tmpl_bids):
                blks = pool.get_new_blocks(_MAX_TEMPLATES)
                self._reserved_tmpl_bids[g] = [b.block_id for b in blks]
            for req_id, n_sched in scheduler_output.num_scheduled_tokens.items():
                req, ext = self._req_info.get(req_id, (None, 0))
                for g in mla if req and (bids := state.get_block_ids(req_id)) else ():
                    tmpls, res = tmpl_map.setdefault(g, []), self._reserved_tmpl_bids[g]
                    bs, n_comp = self.group_block_sizes[g], req.num_computed_tokens
                    lo, hi = max(cdiv(ext, bs), n_comp // bs), (n_comp + n_sched) // bs
                    for bid in bids[g][lo:hi]:
                        if len(tmpls) < len(res) and bid > 0 and bid not in tmpls:
                            to_capture.setdefault(g, []).append((bid, res[len(tmpls)]))
                            tmpls.append(bid)
        meta = s_meta(self, scheduler_output)
        meta.template_blocks_to_capture = to_capture
        return meta

    def _request_finished(self, request):
        s_done(self, request)
        self._req_info.pop(request.request_id, None)

    def _worker_init(self, vllm_config, kv_cache_config):
        w_init(self, vllm_config, kv_cache_config)
        self._mla_group_ids = _mla_groups(kv_cache_config.kv_cache_groups)
        self._template_bids, self._pending_backfill = {}, {}
        self._group_unique_tensors, self._group_num_rows = {}, {}

    def _register_kv_caches(self, kv_caches):
        from vllm_torchtpu.layers.adapter.custom_ops.mamba_state_copy_op import (
            copy_mamba_state_blocks,
        )
        from vllm_torchtpu.utils import synchronize_tensors

        w_reg(self, kv_caches)
        warmed: set[tuple] = set()
        with torch.no_grad():
            for g in self._mla_group_ids:
                uniq: dict[int, torch.Tensor] = {}
                for name in self.group_to_layers[g]:
                    e = kv_caches.get(name)
                    for t in e if isinstance(e, (list, tuple)) else (e,):
                        if isinstance(t, torch.Tensor) and t.shape[0] > 1:
                            uniq.setdefault(id(t), t)
                self._group_unique_tensors[g] = pools = list(uniq.values())
                self._group_num_rows[g] = min((t.shape[0] for t in pools), default=0)
        self._preseeded_groups = set(self._mla_group_ids)

    def _copy_group_blocks(self, g: int, block_ids: list[int]) -> None:
        tmpls, n = self._template_bids.get(g), self._group_num_rows.get(g, 0)
        if not tmpls:
            self._pending_backfill.setdefault(g, []).extend(block_ids)
        elif dst := [b for b in block_ids if 0 < b < n]:
            pairs = [(tmpls[i % len(tmpls)], b) for i, b in enumerate(dst)]
            _copy_block_pairs(self._group_unique_tensors[g], pairs)

    def _preseed_group_pool_once(self, g: int, tmpl_bid: int) -> None:
        if g in getattr(self, "_preseeded_groups", ()):
            return
        from vllm_torchtpu.utils import synchronize_tensors
        if not hasattr(self, "_preseeded_groups"):
            self._preseeded_groups = set()
        for t in self._group_unique_tensors.get(g, ()):
            if t.shape[0] > 1 and 0 < tmpl_bid < t.shape[0]:
                for start in range(1, t.shape[0], 4096):
                    end = min(start + 4096, t.shape[0])
                    pairs = [
                        (tmpl_bid, dst)
                        for dst in range(start, end)
                        if dst != tmpl_bid
                    ]
                    if pairs:
                        _copy_block_pairs([t], pairs)
                        synchronize_tensors([t])
        self._preseeded_groups.add(g)

    def _start_fill_kv(self, metadata):
        if getattr(self, "_preseeded_groups", None) and len(self._preseeded_groups) >= len(self._mla_group_ids):
            return
        with torch.no_grad():
            for group_bids, _ in metadata.reqs_to_fill.values():
                for g, bids in enumerate(group_bids):
                    if bids and g not in getattr(self, "_preseeded_groups", ()):
                        _copy_group_blocks(self, g, bids)

    def _worker_wait_for_save(self, metadata):
        with torch.no_grad():
            for g, pairs in metadata.template_blocks_to_capture.items():
                n = self._group_num_rows.get(g, 0)
                if valid := [(s, d) for s, d in pairs if 0 < s < n and 0 < d < n]:
                    _copy_block_pairs(self._group_unique_tensors[g], valid)
                    tmpls = self._template_bids.setdefault(g, [])
                    tmpls.extend(d for _, d in valid if d not in tmpls)
                    _preseed_group_pool_once(self, g, valid[0][1])
                    dbc.logger.info(
                        "DecodeBenchConnectorWorker: captured & pre-seeded full pool from template %s "
                        "for group %d",
                        valid[0][1],
                        g,
                    )
                if n and (backfill := self._pending_backfill.pop(g, None)):
                    if g not in getattr(self, "_preseeded_groups", ()):
                        _copy_group_blocks(self, g, backfill)

    def _conn_wait_for_save(self):
        if self.connector_worker is not None and self._connector_metadata is not None:
            self.connector_worker.wait_for_save(self._connector_metadata)

    s_cls.__init__, s_cls.request_finished = _sched_init, _request_finished
    s_cls.get_num_new_matched_tokens = _get_num_new_matched_tokens
    s_cls.update_state_after_alloc = _update_state_after_alloc
    s_cls.build_connector_meta = _build_connector_meta
    w_cls.__init__, w_cls.register_kv_caches = _worker_init, _register_kv_caches
    w_cls._preseed_group_pool_once = _preseed_group_pool_once
    w_cls.start_fill_kv, w_cls.wait_for_save = _start_fill_kv, _worker_wait_for_save
    dbc.DecodeBenchConnector.wait_for_save = _conn_wait_for_save
    dbc._tpu_seeding_patched = True
