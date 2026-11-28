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
"""Debug dumps of MSA prefill indexer inputs (4-bit DCP path) for offline
selection studies. Off unless TPU_MINIMAX_M3_MSA_PREFILL_DUMP is set, e.g.
``layers=4/20/36/52+buckets=20000:45000/180000:280000/330000:2000000+per=2``.

Inside the compiled forward the 4-bit DCP op of each selected layer copies
the scoring op's index queries ``[T, G, D]``, this rank's queries, the
packed candidates and the step metadata into fixed-size module buffers (on
every step and path, so every trace takes the same inputs). After each model
step the runner hook picks one prefill request of at least `minrows` new
rows whose length falls in a bucket that still needs dumps (the same choice
on every rank) and writes, per selected layer, its rows (all index queries
and candidates, `qrows` sampled queries) and its pages of this rank (index,
K/V and scale pages) to $CDK_OUTPUT_DIR/msa_pf_dump, and optionally to
`gcs` (gs://bucket/prefix, background upload).
"""

import json
import os
import re
import threading
import time

import numpy as np
import torch

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

ENV = "TPU_MINIMAX_M3_MSA_PREFILL_DUMP"
_SPEC = os.environ.get(ENV, "")
_CFG = dict(
    layers="4/20/36/52",
    buckets="20000:45000/180000:280000/330000:2000000",
    per="2",
    minrows="2048",
    qrows="256",
    ranks="4",
    gcs="",
    maxwait="5400",
)
for _item in filter(None, _SPEC.replace("+", ",").split(",")):
    _k, _v = _item.split("=", 1)
    _CFG[_k] = _v
ENABLED = bool(_SPEC)
LAYERS = {int(x) for x in _CFG["layers"].split("/")}
BUCKETS = [tuple(int(y) for y in x.split(":")) for x in _CFG["buckets"].split("/")]
PER, MIN_ROWS, Q_ROWS = int(_CFG["per"]), int(_CFG["minrows"]), int(_CFG["qrows"])
RANKS, GCS, MAX_WAIT = int(_CFG["ranks"]), _CFG["gcs"], float(_CFG["maxwait"])

_MODULES: list = []
_STATE = {"step": 0, "patched": False, "t0": None, "done": False, "n": 0}
_COUNTS = [0] * len(BUCKETS)
_UPLOADS: list = []


def layer_index(prefix: str) -> int:
    m = re.search(r"(?:^|\.)layers\.(\d+)\.self_attn$", prefix)
    if m is None or any(w in prefix for w in ("draft", "dspark", "mtp", "eagle")):
        return -1  # main model only (not the draft)
    return int(m.group(1))


def wanted(prefix: str) -> bool:
    return ENABLED and layer_index(prefix) in LAYERS


def alloc(
    module, vllm_config, prefix: str, num_heads: int, head_dim: int, kv_heads: int, topk: int
):
    """Register the layer's dump buffers (call from __init__)."""
    dev = str(torch.get_default_device())
    rows = vllm_config.scheduler_config.max_num_batched_tokens
    mns = vllm_config.scheduler_config.max_num_seqs
    dcp = max(1, vllm_config.parallel_config.decode_context_parallel_size)
    block = vllm_config.cache_config.block_size
    pps = -(-vllm_config.model_config.max_model_len // (block * dcp)) + 8

    def buf(name, shape, dtype):
        module.register_buffer(
            name, torch.zeros(shape, dtype=dtype, device=dev), persistent=False
        )

    buf("pfd_iq", (rows, kv_heads, head_dim), torch.bfloat16)
    buf("pfd_q", (rows, num_heads * head_dim), torch.bfloat16)
    buf("pfd_cand", (rows, kv_heads * 2 * topk), torch.float32)
    buf("pfd_meta", (4 + (mns + 8) + (mns + 9) + (mns + 8) * pps,), torch.int32)
    module.pfd_sizes = (mns + 8, mns + 9, (mns + 8) * pps)
    module.pfd_layer = layer_index(prefix)
    _MODULES.append(module)
    _patch_runner()


def _fit(x: torch.Tensor, shape) -> torch.Tensor:
    """``x`` cut or zero-padded to ``shape`` (same rank)."""
    x = x[tuple(slice(0, s) for s in shape)]
    pad = []
    for have, want in zip(reversed(x.shape), reversed(shape)):
        pad += [0, want - have]
    if any(pad):
        x = torch.nn.functional.pad(x, pad)
    return x


def _flat(x: torch.Tensor, n: int) -> torch.Tensor:
    return _fit(x.reshape(-1).to(torch.int32), (n,))


def stash(module, index_q, q, candidates, metadata, num_tokens: int) -> None:
    """In the compiled forward: copy this step's scoring inputs/outputs into
    the buffers. ``index_q [T, h, D]`` (any h), ``q`` this rank's queries
    (token- or head-major per the module), ``candidates [T, C]`` or None."""
    module.pfd_iq.copy_(_fit(index_q.to(torch.bfloat16), module.pfd_iq.shape))
    if module.q_head_major:
        q = q.transpose(0, 1)
    q = q.reshape(q.shape[0], -1).to(torch.bfloat16)
    module.pfd_q.copy_(_fit(q, module.pfd_q.shape))
    if candidates is None:
        candidates = torch.zeros(
            (num_tokens, 1), dtype=torch.float32, device=module.pfd_cand.device
        )
    module.pfd_cand.copy_(_fit(candidates.to(torch.float32), module.pfd_cand.shape))
    n_sl, n_qsl, n_bt = module.pfd_sizes
    mns = int(metadata.seq_lens.shape[0])
    cols = int(metadata.block_tables.numel()) // mns
    head = torch.tensor(
        [num_tokens, mns, cols], dtype=torch.int32, device=metadata.seq_lens.device
    )
    dist = metadata.request_distribution.reshape(-1).to(torch.int32)[2:3]
    module.pfd_meta.copy_(
        torch.cat(
            [
                head,
                dist,
                _flat(metadata.seq_lens, n_sl),
                _flat(metadata.query_start_loc, n_qsl),
                _flat(metadata.block_tables, n_bt),
            ]
        )
    )


def _patch_runner() -> None:
    if _STATE["patched"]:
        return
    _STATE["patched"] = True
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    orig = TPUModelRunner.execute_model

    def execute_model(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        try:
            _after_step()
        except Exception as e:  # never kill the server for a debug dump
            logger.warning("msa_prefill_dump failed: %s", e)
        return out

    TPUModelRunner.execute_model = execute_model


def _token():
    import urllib.request as r

    q = r.Request(
        "http://metadata.google.internal/computeMetadata/v1/instance/"
        "service-accounts/default/token",
        headers={"Metadata-Flavor": "Google"},
    )
    return json.load(r.urlopen(q, timeout=30))["access_token"]


def _upload(path: str) -> None:
    import urllib.parse as p
    import urllib.request as r

    bucket, prefix = GCS[5:].split("/", 1)
    name = prefix.rstrip("/") + "/" + os.path.basename(path)
    for i in range(5):
        try:
            with open(path, "rb") as f:
                data = f.read()
            q = r.Request(
                "https://storage.googleapis.com/upload/storage/v1/b/%s/o?uploadType=media&name=%s"
                % (bucket, p.quote(name, safe="")),
                data=data,
                headers={
                    "Authorization": "Bearer " + _token(),
                    "Content-Type": "application/octet-stream",
                },
                method="POST",
            )
            r.urlopen(q, timeout=600).read()
            return
        except Exception as e:
            logger.warning("msa_prefill_dump upload %s try %d: %s", name, i, e)
            time.sleep(10 * (i + 1))


def _save(path: str, rec: dict) -> None:
    np.savez(path, **rec)
    if GCS:
        t = threading.Thread(target=_upload, args=(path,), daemon=True)
        t.start()
        _UPLOADS.append(t)


def _finish(rank: int, out_dir: str, why: str) -> None:
    _STATE["done"] = True
    path = os.path.join(out_dir, f"r{rank}_DONE.json")
    with open(path, "w") as f:
        json.dump(
            dict(why=why, counts=_COUNTS, buckets=BUCKETS, step=_STATE["step"],
                 records=_STATE["n"]), f
        )  # fmt: skip
    if GCS:
        for t in _UPLOADS:
            t.join()
        _upload(path)
    logger.warning("msa_prefill_dump finished (%s): %s", why, _COUNTS)


def _meta(m):
    meta = m.pfd_meta.cpu().numpy()
    T, mns, cols, nreq = (int(x) for x in meta[:4])
    n_sl, n_qsl, _ = m.pfd_sizes
    sl = meta[4 : 4 + n_sl][:mns]
    qsl = meta[4 + n_sl : 4 + n_sl + n_qsl][: mns + 1]
    bt = meta[4 + n_sl + n_qsl :][: mns * cols].reshape(mns, cols)
    return T, mns, cols, nreq, sl, qsl, bt


def _after_step() -> None:
    if _STATE["done"] or not _MODULES:
        return
    _STATE["step"] += 1
    from vllm.distributed import get_tensor_model_parallel_rank

    rank = get_tensor_model_parallel_rank()
    if rank >= RANKS:
        _STATE["done"] = True
        return
    out_dir = os.path.join(os.environ.get("CDK_OUTPUT_DIR", "/tmp"), "msa_pf_dump")
    os.makedirs(out_dir, exist_ok=True)
    T, mns, cols, nreq, sl, qsl, bt = _meta(_MODULES[0])
    if T < MIN_ROWS or nreq <= 0:
        return
    if _STATE["t0"] is None:
        _STATE["t0"] = time.time()
    pick = None
    for r in range(min(nreq, mns)):
        qlen = int(qsl[r + 1] - qsl[r])
        L = int(sl[r])
        if qlen < MIN_ROWS or qlen > L:
            continue
        for b, (lo, hi) in enumerate(BUCKETS):
            if lo <= L < hi and _COUNTS[b] < PER:
                pick = (r, b)
                break
        if pick:
            break
    if pick is not None:
        r, b = pick
        _COUNTS[b] += 1
        L, q0, q1 = int(sl[r]), int(qsl[r]), int(qsl[r + 1])
        for m in _MODULES:
            Tm, _, colsm, _, slm, qslm, btm = _meta(m)
            if Tm != T or int(slm[r]) != L:
                logger.warning("msa_prefill_dump: layer %d meta mismatch", m.pfd_layer)
                continue
            dcp = max(1, m.dcp_size)
            rows_pp = int(m.kv_cache.kv_cache.shape[1])
            block = (rows_pp - rows_pp % 2) * 8  # data word rows x 8 tokens
            nlog = -(-L // (block * dcp))
            ids = btm[r, : min(nlog, colsm)].astype(np.int64)
            dev = m.index_cache.kv_cache.device
            tid = torch.from_numpy(ids).to(dev)
            rows = np.unique(np.linspace(0, q1 - q0 - 1, Q_ROWS).astype(np.int64))
            rec = dict(
                layer=m.pfd_layer, step=_STATE["step"], bucket=b, tp_rank=rank,
                dcp_rank=int(getattr(m, "pfd_dcp_rank", -1)), dcp=dcp, seq_len=L,
                q_len=q1 - q0, T=T, nreq=nreq, mns=mns, page_ids=ids,
                msa_cfg=np.asarray(m.pfd_msa_cfg, np.float64),
                scale_row0=int(m.scale_row0), kv_head_replicas=int(m.kv_head_replicas),
                num_heads=int(m.num_heads), head_dim=int(m.head_dim),
                index_q=m.pfd_iq[q0:q1].float().cpu().numpy(),
                cand=m.pfd_cand[q0:q1].cpu().numpy(),
                q_rows=rows,
                q=m.pfd_q[q0:q1][torch.from_numpy(rows).to(m.pfd_q.device)]
                .float().cpu().numpy(),
                index_pages=torch.index_select(m.index_cache.kv_cache, 0, tid).cpu().numpy(),
                kv_pages=torch.index_select(m.kv_cache.kv_cache, 0, tid).cpu().numpy(),
            )  # fmt: skip
            if m.scale_cache is not None:
                rec["scale_pages"] = (
                    torch.index_select(m.scale_cache.kv_cache, 0, tid).cpu().numpy()
                )
            name = f"r{rank}_l{m.pfd_layer:02d}_s{_STATE['step']:06d}_b{b}.npz"
            _save(os.path.join(out_dir, name), rec)
            _STATE["n"] += 1
        logger.warning(
            "msa_prefill_dump step %d req len %d rows %d bucket %d counts %s",
            _STATE["step"], L, q1 - q0, b, _COUNTS,
        )  # fmt: skip
    if all(c >= PER for c in _COUNTS):
        _finish(rank, out_dir, "full")
    elif time.time() - _STATE["t0"] > MAX_WAIT:
        _finish(rank, out_dir, "maxwait")
