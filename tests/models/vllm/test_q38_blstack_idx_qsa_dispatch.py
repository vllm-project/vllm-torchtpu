# SPDX-License-Identifier: Apache-2.0
"""q38 BLSTACK-IDX (VLLM_TPU_QSA_COMPRESSED_DCP_ROW_SHARD): module wiring.

- Mode: on only for target QSA layers under compressed-key DCP with TP > DCP;
  off by default, for the MTP draft, without compressed DCP, under PCP.
- Flag off: no operand, no op, and the DCP dispatch is Benliang's (the plain
  compressed-DCP op with ``dcp_rank`` only).
- Flag on: decode-shaped graphs at or above the min rows take the row-shard op,
  every other graph the plain op, both with the same ``row_shard_index``
  operand (one input list for every compile range).
- Op builders (stand-in jax ops): the row-shard op hands the token select
  ``shard_index // DCP`` with ``shard_size = TP / DCP`` and ``decode_rows`` and
  gathers the local lists over TP; the plain op hands the same divided operand
  to the owned-group select step's operand slot and gathers over the DCP group;
  the prefill token-shard op (``decode_rows`` off) passes exactly its old
  keywords.
"""

from types import SimpleNamespace as NS

import pytest
import torch
from qwen4_exp_test_utils import q4exp_text_config as attention_config

from vllm_torchtpu.layers.adapter.custom_ops import qsa_op
from vllm_torchtpu.models.vllm.qwen4_exp import qsa as qsa_mod
from vllm_torchtpu.models.vllm.qwen4_exp.qsa import (
    QSAMetadata,
    Qwen4ExpQSA,
    compressed_row_shard_applies,
)

TARGET = "model.layers.3.self_attn"
DRAFT = "mtp.layers.0.self_attn"
CK_ENV = {
    "VLLM_TPU_QSA_MAIN_KV_DCP": "4",
    "VLLM_TPU_QSA_MTP_KV_DCP": "1",
    "VLLM_TPU_QSA_COMPRESSED_DCP": "1",
    "VLLM_TPU_QSA_COMPRESSED_DCP_LOCAL_MERGE": "1",
    "VLLM_TPU_QSA_COMPRESSED_DCP_PACKED_GATHER": "1",
}


@pytest.fixture(autouse=True)
def cpu_vllm_config_context():
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    resolve_kv_cache_layout(config, [["LBNHC", "LBHNC"]])
    with set_current_vllm_config(config):
        yield


class Group:
    def __init__(self, size, rank, log=None, name=""):
        self.world_size, self.rank_in_group, self.log, self.name = size, rank, log, name

    def all_gather(self, x, dim=0):
        if self.log is not None:
            self.log.append((self.name, tuple(x.shape), dim))
        return torch.cat([x] * self.world_size, dim=dim)

    def all_reduce(self, x):
        return x


def _env(monkeypatch, on, min_rows=None, extra=()):
    for k in list(CK_ENV) + ["VLLM_TPU_QSA_COMPRESSED_DCP_ROW_SHARD",
                             "VLLM_TPU_QSA_COMPRESSED_DCP_ROW_SHARD_MIN_ROWS",
                             "VLLM_TPU_QSA_DCP_MERGE", "VLLM_TPU_MEGAKERNEL"]:
        monkeypatch.delenv(k, raising=False)
    for k, v in CK_ENV.items():
        monkeypatch.setenv(k, v)
    if on:
        monkeypatch.setenv("VLLM_TPU_QSA_COMPRESSED_DCP_ROW_SHARD", "1")
    if min_rows is not None:
        monkeypatch.setenv("VLLM_TPU_QSA_COMPRESSED_DCP_ROW_SHARD_MIN_ROWS", str(min_rows))
    for k, v in extra:
        monkeypatch.setenv(k, v)


def _module(monkeypatch, rank=5, prefix=TARGET, bind=True):
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_rank",
                        lambda: rank)
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
                        lambda: 8)
    import vllm.distributed

    monkeypatch.setattr(vllm.distributed, "get_tp_group", lambda: Group(8, rank))
    monkeypatch.setattr(qsa_mod, "main_kv_dcp_group", lambda size: Group(size, rank % size))
    with torch.device("cpu"):
        module = Qwen4ExpQSA(config=attention_config(), tp_size=8, tp_rank=rank,
                             max_model_len=8192, pooled=True, max_query_len=4,
                             max_num_seqs=64, prefix=prefix)
        if bind:
            module.bind_pool(torch.zeros(16, 128, dtype=torch.uint8))
    return module


def test_applies_rule():
    # decode-shaped, divisible by the shards, at least the min rows
    assert compressed_row_shard_applies(96, 64, 2, 4, 0)
    assert compressed_row_shard_applies(256, 64, 2, 4, 48)
    assert not compressed_row_shard_applies(32, 64, 2, 4, 48)    # below min rows
    assert not compressed_row_shard_applies(258, 64, 2, 4, 0)    # prefill-shaped
    assert not compressed_row_shard_applies(97, 64, 2, 4, 0)     # not divisible
    assert not compressed_row_shard_applies(96, 64, 1, 4, 0)     # one shard: nothing to split
    assert compressed_row_shard_applies(16, 64, 2, 4, 0)


def test_mode_rule(monkeypatch):
    mode = Qwen4ExpQSA._compressed_row_shard_mode
    _env(monkeypatch, on=True)
    assert mode(4, 8, 1, TARGET)
    assert not mode(4, 8, 1, DRAFT)       # the MTP draft never
    assert not mode(4, 4, 1, TARGET)      # TP == DCP: one group, nothing to split
    assert not mode(4, 6, 1, TARGET)      # TP not a multiple of DCP
    assert not mode(1, 8, 1, TARGET)      # no main-KV DCP
    assert not mode(4, 8, 2, TARGET)      # PCP
    monkeypatch.setenv("VLLM_TPU_QSA_COMPRESSED_DCP", "0")
    assert not mode(4, 8, 1, TARGET)      # needs compressed-key DCP
    _env(monkeypatch, on=False)
    assert not mode(4, 8, 1, TARGET)      # default off


def test_flag_off_registers_nothing(monkeypatch):
    _env(monkeypatch, on=False)
    module = _module(monkeypatch)
    assert module.main_kv_dcp == 4 and module._compressed_dcp
    assert not module.compressed_row_shard
    assert not hasattr(module, "row_shard_index")
    assert module._row_shard_dcp_op is None and module._row_plain_dcp_op is None


def test_flag_on_binds_both_ops_and_logs(monkeypatch):
    _env(monkeypatch, on=True, min_rows=48)
    monkeypatch.setattr(qsa_mod, "_Q38_ROWSHARD_LOGGED", False)
    lines = []
    monkeypatch.setattr(qsa_mod.logger, "info", lambda msg, *a: lines.append(msg % a))
    built = []
    real_rs = qsa_op.build_pooled_qsa_token_shard_compressed_dcp_op
    real_plain = qsa_op.build_pooled_qsa_row_plain_compressed_dcp_op

    def rs(tp_group, pair_group, size, **kw):
        built.append(("row_shard", int(tp_group.world_size), int(pair_group.world_size), size, kw))
        return real_rs(tp_group, pair_group, size, **kw)

    def plain(pair_group, size, **kw):
        built.append(("plain", int(pair_group.world_size), size, kw))
        return real_plain(pair_group, size, **kw)

    monkeypatch.setattr(qsa_op, "build_pooled_qsa_token_shard_compressed_dcp_op", rs)
    monkeypatch.setattr(qsa_op, "build_pooled_qsa_row_plain_compressed_dcp_op", plain)
    module = _module(monkeypatch, rank=6)
    # The row-shard op is the token-shard compressed op with decode rows, over TP 8 / DCP 4.
    assert built == [("row_shard", 8, 4, 4, {"decode_rows": True}), ("plain", 4, 4, {})], built
    assert module.compressed_row_shard and module.row_shard_min_rows == 48
    assert module.row_shard_index.tolist() == [6] and module.row_shard_index.dtype == torch.int32
    assert module._row_shard_dcp_op is not None and module._row_plain_dcp_op is not None
    bind = [x for x in lines if "BLSTACK-IDX compressed-DCP row shard: on" in x]
    assert len(bind) == 1 and "TP rank 6 of 8, DCP 4, row shard 1 of 2, min rows 48" in bind[0]
    print("Q38_BLIDX ENGAGE_BIND_LINE", bind[0])


def test_draft_layer_stays_off(monkeypatch):
    _env(monkeypatch, on=True, extra=[("VLLM_TPU_QSA_DRAFT_SHARED_SELECTION", "1")])
    module = _module(monkeypatch, prefix=DRAFT, bind=False)
    assert not module.compressed_row_shard and not hasattr(module, "row_shard_index")


def _dispatch(monkeypatch, module, tokens, slots=64):
    calls = []

    def recorder(name):
        def op(*args, **kwargs):
            calls.append((name, kwargs))
            return torch.zeros(1), torch.zeros(1)

        return op

    for name in ("_dcp_op", "_row_shard_dcp_op", "_row_plain_dcp_op"):
        if getattr(module, name, None) is not None:
            setattr(module, name, recorder(name))
    table = torch.zeros((slots, 4), dtype=torch.int32)
    per_layer = {
        module.prefix + ".raw_ring": NS(block_tables=torch.zeros((slots, 1), dtype=torch.int32)),
        module.main_cache_name: NS(block_tables=table),
        module.compressed_cache_name: NS(block_tables=table),
    }
    monkeypatch.setattr(qsa_mod, "get_forward_context", lambda: NS(attn_metadata=per_layer))
    meta = QSAMetadata(
        query_start_loc=torch.zeros(slots + 1, dtype=torch.int32),
        context_lens=torch.zeros(slots, dtype=torch.int32),
        state_slots=torch.zeros(slots, dtype=torch.int32),
        main_block_table=table,
        compressed_block_table=table,
    )
    hidden = torch.zeros(tokens, 2560, dtype=torch.bfloat16)
    module.attend_with_indices(hidden, torch.zeros(tokens, dtype=torch.int64), meta)
    assert len(calls) == 1, calls
    return calls[0]


@pytest.mark.parametrize("tokens,want", [(96, "_row_shard_dcp_op"), (256, "_row_shard_dcp_op"),
                                         (48, "_row_shard_dcp_op"), (32, "_row_plain_dcp_op"),
                                         (16, "_row_plain_dcp_op"), (264, "_row_plain_dcp_op"),
                                         (98, "_row_shard_dcp_op")])
def test_flag_on_dispatch(monkeypatch, tokens, want):
    _env(monkeypatch, on=True, min_rows=48)
    module = _module(monkeypatch)
    name, kwargs = _dispatch(monkeypatch, module, tokens)
    assert name == want, (tokens, name)
    assert kwargs["shard_index"] is module.row_shard_index
    assert kwargs["dcp_rank"] is module.dcp_rank
    assert set(kwargs) == {"shard_index", "dcp_rank"}, kwargs
    print(f"Q38_BLIDX DISPATCH tokens={tokens} op={name}")


def test_flag_off_dispatch_is_the_plain_compressed_op(monkeypatch):
    _env(monkeypatch, on=False)
    module = _module(monkeypatch)
    name, kwargs = _dispatch(monkeypatch, module, 96)
    assert name == "_dcp_op" and set(kwargs) == {"dcp_rank"}, (name, kwargs)


def test_sink_layer_keeps_the_global_selection(monkeypatch):
    # The layer whose selection the MTP draft reads gets global_selection=True
    # on the row-shard op as on the plain compressed op.
    _env(monkeypatch, on=True)
    module = _module(monkeypatch)
    module.register_buffer("_selection_sink", torch.full((256, 2051), -1, dtype=torch.int32),
                           persistent=False)
    calls = []

    def op(*args, **kwargs):
        calls.append(kwargs)
        return torch.zeros(1), torch.zeros((96, 2051), dtype=torch.int32)

    module._row_shard_dcp_op = op
    monkeypatch.setattr(qsa_mod, "update_rows_", lambda dst, src: None)
    slots, table = 64, torch.zeros((64, 4), dtype=torch.int32)
    per_layer = {
        module.prefix + ".raw_ring": NS(block_tables=torch.zeros((slots, 1), dtype=torch.int32)),
        module.main_cache_name: NS(block_tables=table),
        module.compressed_cache_name: NS(block_tables=table),
    }
    monkeypatch.setattr(qsa_mod, "get_forward_context", lambda: NS(attn_metadata=per_layer))
    meta = QSAMetadata(torch.zeros(65, dtype=torch.int32), torch.zeros(64, dtype=torch.int32),
                       torch.zeros(64, dtype=torch.int32), table, table)
    module.attend_with_indices(torch.zeros(96, 2560, dtype=torch.bfloat16),
                               torch.zeros(96, dtype=torch.int64), meta)
    assert calls and calls[0].get("global_selection") is True, calls


# ------------------------------------------------------------------ op builders


def _stub_ops(monkeypatch, log):
    T, K = 96, 16

    def token_select(*args, **kwargs):
        log.append(("token_select", args, kwargs))
        rows = args[0].shape[0] // kwargs["shard_size"]
        return (args[10], torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1),
                torch.zeros(1), torch.zeros((rows, K)), torch.zeros((rows, K), dtype=torch.int32))

    def select(*args, **kwargs):
        log.append(("select", args, kwargs))
        return (args[10], torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1),
                torch.zeros(1), torch.zeros((T, K)), torch.zeros((T, K), dtype=torch.int32))

    def attend(*args, **kwargs):
        log.append(("attend", args, kwargs))
        return args[7], torch.zeros(1), torch.zeros(1), torch.zeros(1)

    monkeypatch.setattr(qsa_op, "_compressed_dcp_ops", lambda: (select, attend, token_select))
    monkeypatch.setattr(qsa_op, "_row_plain_compressed_dcp_op", lambda: select)
    monkeypatch.setattr(qsa_op, "_dcp_ops", lambda: (None, None, None, None))
    monkeypatch.setattr(qsa_op, "_dcp_merge_setup", lambda: None)
    monkeypatch.setattr(qsa_op, "_dcp_finish", lambda *a, **k: "finished")
    monkeypatch.setenv("VLLM_TPU_QSA_COMPRESSED_DCP_PACKED_GATHER", "1")
    return T


def _op_args(T):
    args = [torch.zeros(T, 12)] + [torch.zeros(T, 4)] * 3 + [torch.zeros(8)] * 4
    args += [torch.zeros(T, dtype=torch.int64), torch.zeros(4), torch.zeros(16, 128)]
    args += [torch.zeros(65, dtype=torch.int64), torch.zeros(64, dtype=torch.int64),
             torch.zeros(64, dtype=torch.int64), torch.zeros(64, 4, dtype=torch.int64),
             torch.zeros(64, 4, dtype=torch.int64)]
    return args


KW = dict(kv_heads=1, compress_ratio=4, token_topk=64, norm_eps=1e-6, rotary_dim=64,
          max_query_len=4)


def test_row_shard_builder_routes_decode_rows_over_tp(monkeypatch):
    log, gathers = [], []
    T = _stub_ops(monkeypatch, log)
    tp, pair = Group(8, 5, gathers, "tp"), Group(4, 1, gathers, "pair")
    run = qsa_op.build_pooled_qsa_token_shard_compressed_dcp_op(tp, pair, 4, decode_rows=True)
    shard_index, dcp_rank = torch.tensor([5], dtype=torch.int32), torch.tensor([1], dtype=torch.int32)
    assert run(*_op_args(T), shard_index=shard_index, dcp_rank=dcp_rank, **KW)[0] == "finished"
    (name, args, kwargs), = [x for x in log if x[0] == "token_select"]
    assert args[16].tolist() == [1] and args[17] is dcp_rank          # shard = 5 // 4
    assert kwargs["shard_size"] == 2 and kwargs["dcp_size"] == 4
    assert kwargs["decode_rows"] is True and kwargs["max_query_len"] == 4
    assert ("tp", (1, T // 2, 32), 0) in gathers                      # one packed TP gather
    assert not [g for g in gathers if g[0] == "pair" and g[2] == 0]   # no pair list gather


def test_prefill_token_shard_builder_keeps_its_keywords(monkeypatch):
    log = []
    T = _stub_ops(monkeypatch, log)
    run = qsa_op.build_pooled_qsa_token_shard_compressed_dcp_op(Group(8, 5), Group(4, 1), 4)
    run(*_op_args(T), shard_index=torch.tensor([5], dtype=torch.int32),
        dcp_rank=torch.tensor([1], dtype=torch.int32), **{**KW, "max_query_len": 1})
    (name, args, kwargs), = [x for x in log if x[0] == "token_select"]
    assert "decode_rows" not in kwargs and kwargs["shard_size"] == 2


def test_plain_builder_passes_the_operand_in_the_select_slot(monkeypatch):
    log, gathers = [], []
    T = _stub_ops(monkeypatch, log)
    pair = Group(4, 1, gathers, "pair")
    run = qsa_op.build_pooled_qsa_row_plain_compressed_dcp_op(pair, 4)
    shard_index, dcp_rank = torch.tensor([5], dtype=torch.int32), torch.tensor([1], dtype=torch.int32)
    assert run(*_op_args(T), shard_index=shard_index, dcp_rank=dcp_rank, **KW)[0] == "finished"
    (name, args, kwargs), = [x for x in log if x[0] == "select"]
    assert len(args) == 18 and args[16].tolist() == [1] and args[17] is dcp_rank
    assert kwargs["dcp_size"] == 4 and "shard_size" not in kwargs
    assert ("pair", (1, T, 32), 0) in gathers                         # the pair list gather
