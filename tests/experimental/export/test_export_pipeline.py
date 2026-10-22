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
"""Unit and integration tests for `vllm_torchtpu.experimental.export`."""

from __future__ import annotations

import json
import os
import struct
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest
import torch
import torch.nn as nn

from vllm_torchtpu.experimental.export import (
    api,
    cli,
    exporter,
    sampling,
    tpu_lowering,
    universal_adapter,
    validator,
    weight_mapper,
)


def _mock_lower_torch_callable_to_mlir(
    fn: Any,
    arg_specs: Sequence[api.ShapeDtypeStruct],
    *,
    donate_argnums: Sequence[int] = (),
    module_name: str | None = None,
) -> str:
    """Test helper: traces `fn` on CPU tensors and emits matching StableHLO MLIR."""
    del module_name
    donated_set = {int(i) for i in donate_argnums}
    cpu_args = [
        torch.zeros(
            [int(d) for d in s.shape],
            dtype=tpu_lowering.torch_dtype(s.dtype),
            device="cpu",
        )
        for s in arg_specs
    ]
    with torch.no_grad():
        raw_out = fn(*[t.clone() for t in cpu_args])
    out_tensors = (
        [raw_out]
        if isinstance(raw_out, torch.Tensor)
        else [t for t in raw_out if isinstance(t, torch.Tensor)]
    )

    def _fmt(shape: Sequence[int], dt: Any) -> str:
        c = validator.canonical_dtype(tpu_lowering.dtype_name(dt))
        dims = "x".join(str(int(d)) for d in shape)
        return f"tensor<{dims}x{c}>" if dims else f"tensor<{c}>"

    arg_strs: list[str] = []
    for i, s in enumerate(arg_specs):
        attr = " {jax.buffer_donor = true}" if i in donated_set else ""
        arg_strs.append(f"%arg{i}: {_fmt(s.shape, s.dtype)}{attr}")
    out_strs = [_fmt(tuple(int(d) for d in t.shape), t.dtype) for t in out_tensors]
    ret_sig = f"({', '.join(out_strs)})" if len(out_strs) != 1 else out_strs[0]
    # Read every argument, as a traced graph reads every input it uses.
    body_lines: list[str] = [
        f'    %r{i} = "stablehlo.optimization_barrier"(%arg{i}) : ({t}) -> {t}'
        for i, t in enumerate(_fmt(s.shape, s.dtype) for s in arg_specs)
    ]
    ret_vals: list[str] = []
    for i, t_str in enumerate(out_strs):
        body_lines.append(f"    %c{i} = stablehlo.constant dense<0> : {t_str}")
        ret_vals.append(f"%c{i}")
    ret_stmt = (
        f"    return {', '.join(ret_vals)} : {', '.join(out_strs)}"
        if out_strs
        else "    return"
    )
    return "\n".join(
        [
            "module {",
            f"  func.func public @main({', '.join(arg_strs)}) -> {ret_sig} {{",
            *body_lines,
            ret_stmt,
            "  }",
            "}",
            "",
        ]
    )


class _TinyPyTorchBackbone(nn.Module):
    def __init__(
        self, vocab_size: int = 64, hidden_size: int = 32, num_layers: int = 2
    ) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: list[torch.Tensor] | None = None,
        attn_metadata: Any = None,
    ) -> torch.Tensor:
        del positions, kv_caches, attn_metadata
        x = self.embed_tokens(
            input_ids.clamp(min=0, max=self.embed_tokens.num_embeddings - 1)
        )
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class _TinyPyTorchCausalLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 64,
        hidden_size: int = 32,
        num_layers: int = 2,
        mrope_section: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.model = _TinyPyTorchBackbone(vocab_size, hidden_size, num_layers)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        if mrope_section is not None:
            from tests.experimental.export.test_rope_patch import (
                _SyntheticRotaryEmbedding,
            )

            self.rotary_emb = _SyntheticRotaryEmbedding(
                head_size=32,
                rotary_dim=32,
                mrope_section=mrope_section,
            )


class _TinyTorchExportableLLM(api.ExportableTextOnlyLLM):
    """Minimal PyTorch-backed ExportableTextOnlyLLM for protocol testing."""

    def __init__(self, cfg: api.TextOnlyLLMExportConfig) -> None:
        self._cfg = cfg

    @property
    def pallas_interpret(self) -> bool:
        return False

    @property
    def export_config(self) -> api.TextOnlyLLMExportConfig:
        return self._cfg

    def abstract_params(self) -> Mapping[str, api.ShapeDtypeStruct]:
        cfg = self._cfg
        return {
            "model.embed_tokens.weight": api.ShapeDtypeStruct(
                (cfg.vocab_size, cfg.hidden_size), cfg.dtype
            ),
            "model.layers.0.self_attn.qkv_proj.weight": api.ShapeDtypeStruct(
                (cfg.hidden_size, cfg.hidden_size), cfg.dtype
            ),
            "lm_head.weight": api.ShapeDtypeStruct(
                (cfg.hidden_size, cfg.vocab_size), cfg.dtype
            ),
        }

    def forward(
        self,
        weights: Mapping[str, torch.Tensor],
        kv_caches: Sequence[torch.Tensor],
        *,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_lens: torch.Tensor,
        block_tables: torch.Tensor,
        query_start_loc: torch.Tensor,
        req_dist: torch.Tensor,
    ) -> tuple[Sequence[torch.Tensor], torch.Tensor]:
        del positions, kv_lens, block_tables, query_start_loc, req_dist
        emb = weights["model.embed_tokens.weight"]
        proj = weights["model.layers.0.self_attn.qkv_proj.weight"]
        hidden = torch.matmul(emb[input_ids.to(torch.int64)], proj)
        return list(kv_caches), hidden

    def compute_logits(
        self, hidden_states: torch.Tensor, lm_head_weight: torch.Tensor
    ) -> torch.Tensor:
        return torch.matmul(hidden_states, lm_head_weight)

    def sample_tokens(
        self,
        logits: torch.Tensor,
        *,
        temperature: torch.Tensor | None = None,
        top_k: torch.Tensor | None = None,
        top_p: torch.Tensor | None = None,
        min_p: torch.Tensor | None = None,
        uniform: torch.Tensor | None = None,
        repetition_penalty: torch.Tensor | None = None,
        presence_penalty: torch.Tensor | None = None,
        frequency_penalty: torch.Tensor | None = None,
        history: torch.Tensor | None = None,
        output_start: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return sampling.dispatch_sample_tokens(
            logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            uniform=uniform,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            history=history,
            output_start=output_start,
        )


def test_exportable_protocol_completeness() -> None:
    cfg = api.TextOnlyLLMExportConfig(
        model_name="tiny-torch",
        vocab_size=32,
        hidden_size=16,
        num_layers=1,
        num_q_heads=2,
        num_kv_heads=2,
        head_dim=8,
        block_size=8,
        num_blocks=16,
        max_num_seqs=8,
        max_model_len=64,
        token_buckets=(8, 16),
        req_buckets=(4, 8),
        emit_byte_tokenizer=True,
    )
    model = _TinyTorchExportableLLM(cfg)
    assert isinstance(model, api.ExportableModel)
    assert isinstance(model, api.ExportableTextOnlyLLM)

    logits = torch.tensor(
        [[0.1, 2.5, -1.0, 0.4], [3.0, 0.2, 0.1, -0.5]], dtype=torch.float32
    )
    greedy = model.sample_tokens(logits)
    assert greedy.shape == (2, 1)
    assert greedy.dtype == torch.int64
    assert greedy[:, 0].tolist() == [1, 0]

    temp = torch.tensor([1.0, 1.0], dtype=torch.float32)
    top_k = torch.tensor([1, 1], dtype=torch.int32)
    top_p = torch.tensor([1.0, 1.0], dtype=torch.float32)
    min_p = torch.tensor([0.0, 0.0], dtype=torch.float32)
    unif = torch.tensor([0.5, 0.5], dtype=torch.float32)
    stoch = model.sample_tokens(
        logits,
        temperature=temp,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        uniform=unif,
    )
    assert stoch[:, 0].tolist() == [1, 0]

    pen = model.sample_tokens(
        logits,
        temperature=temp,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        uniform=unif,
        repetition_penalty=torch.tensor([1.0, 1.0], dtype=torch.float32),
        presence_penalty=torch.tensor([10.0, 0.0], dtype=torch.float32),
        frequency_penalty=torch.tensor([0.0, 0.0], dtype=torch.float32),
        history=torch.tensor([[1, -1, -1, -1], [0, -1, -1, -1]], dtype=torch.int32),
        output_start=torch.tensor([0, 0], dtype=torch.int32),
    )
    assert pen.shape == (2, 1)
    assert int(pen[0, 0].item()) == 3


def test_end_to_end_export_declarative_weights_only(tmp_path) -> None:
    cfg = api.TextOnlyLLMExportConfig(
        model_name="tiny-torch",
        vocab_size=32,
        hidden_size=16,
        num_layers=1,
        num_q_heads=2,
        num_kv_heads=2,
        head_dim=8,
        block_size=8,
        num_blocks=16,
        max_num_seqs=8,
        max_model_len=64,
        token_buckets=(8, 16),
        req_buckets=(4, 8),
        emit_byte_tokenizer=True,
    )
    model = _TinyTorchExportableLLM(cfg)

    out_dir = str(tmp_path / "exported_torch")
    with mock.patch.object(
        tpu_lowering,
        "lower_torch_callable_to_mlir",
        side_effect=_mock_lower_torch_callable_to_mlir,
    ):
        exporter.export(model, out_dir)
    validator.validate_exported_model_dir(out_dir)

    assert os.path.isfile(os.path.join(out_dir, "metadata.json"))
    assert not os.path.exists(os.path.join(out_dir, "config.json"))
    assert os.path.isfile(os.path.join(out_dir, "tokenizer.json"))
    assert os.path.isfile(os.path.join(out_dir, "subgraphs", "abstract_params.mlir"))
    assert os.path.isfile(os.path.join(out_dir, "subgraphs", "forward_bucket8.mlir"))
    assert os.path.isfile(os.path.join(out_dir, "subgraphs", "forward_bucket16.mlir"))
    assert os.path.isfile(
        os.path.join(out_dir, "subgraphs", "compute_logits_req4.mlir")
    )
    assert os.path.isfile(os.path.join(out_dir, "subgraphs", "sample_tokens_req4.mlir"))
    assert os.path.isfile(
        os.path.join(out_dir, "subgraphs", "sample_tokens_stochastic_req4.mlir")
    )
    assert os.path.isfile(
        os.path.join(out_dir, "subgraphs", "sample_tokens_penalized_req4.mlir")
    )
    # Only declarative weight.json is emitted; no constants.safetensors is written
    assert os.path.isfile(os.path.join(out_dir, "serving_checkpoint", "weight.json"))
    assert not os.path.exists(
        os.path.join(out_dir, "serving_checkpoint", "constants.safetensors")
    )


@pytest.fixture
def single_rank_tp_group() -> Any:
    """A one-rank tensor-parallel group: gathering over it returns the input.

    This is what vLLM's all-gather does at world size 1, without needing a TPU
    process group on the CPU test host.
    """
    with mock.patch(
        "vllm.distributed.tensor_model_parallel_all_gather",
        lambda x, dim=-1: x,
    ):
        yield


@pytest.mark.usefixtures("single_rank_tp_group")
def test_universal_adapter_1d_and_2d_mrope_export_and_cli(tmp_path) -> None:
    hf_dir = tmp_path / "hf_ckpt"
    hf_dir.mkdir()
    with open(hf_dir / "config.json", "w") as f:
        json.dump(
            {
                "architectures": ["Qwen2_5_VLForConditionalGeneration"],
                "vocab_size": 64,
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "head_dim": 16,
            },
            f,
        )
    header = {
        "model.embed_tokens.weight": {
            "dtype": "BF16",
            "shape": [64, 32],
            "data_offsets": [0, 4096],
        },
        "model.layers.0.weight": {
            "dtype": "BF16",
            "shape": [32, 32],
            "data_offsets": [4096, 6144],
        },
        "model.layers.1.weight": {
            "dtype": "BF16",
            "shape": [32, 32],
            "data_offsets": [6144, 8192],
        },
        "lm_head.weight": {
            "dtype": "BF16",
            "shape": [64, 32],
            "data_offsets": [8192, 12288],
        },
    }
    hb = json.dumps(header).encode("utf-8")
    with open(hf_dir / "model.safetensors", "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)

    hf_meta, default_st = weight_mapper.read_hf_safetensors_metadata(str(hf_dir))
    assert default_st == "model.safetensors"
    assert hf_meta["model.embed_tokens.weight"] == "model.safetensors"
    st_header = weight_mapper.read_safetensors_header(hf_dir / "model.safetensors")
    assert st_header["model.embed_tokens.weight"]["shape"] == [64, 32]

    pt_model = _TinyPyTorchCausalLM(
        vocab_size=64,
        hidden_size=32,
        num_layers=2,
        mrope_section=[4, 6, 6],
    ).to(torch.bfloat16)
    vllm_cfg = SimpleNamespace(
        model_config=SimpleNamespace(
            model=str(hf_dir),
            dtype="bfloat16",
            max_model_len=128,
            hf_config=SimpleNamespace(
                vocab_size=64,
                hidden_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=16,
                eos_token_id=[2, 0],
                architectures=["Qwen2_5_VLForConditionalGeneration"],
            ),
        ),
        cache_config=SimpleNamespace(block_size=8, num_gpu_blocks=16),
        scheduler_config=SimpleNamespace(max_num_seqs=8),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            data_parallel_size=1,
            pipeline_parallel_size=1,
            use_sequence_parallel_moe=False,
            is_moe_model=False,
        ),
        compilation_config=SimpleNamespace(
            compile_sizes=[8, 16],
            fast_moe_cold_start=False,
            static_forward_context={},
        ),
    )

    out_dir = str(tmp_path / "exported_mrope")
    args = cli.parse_args(
        [
            "--model",
            str(hf_dir),
            "--output-dir",
            out_dir,
            "--target-tpu",
            "v6e",
            "--token-buckets",
            "8,16",
            "--req-buckets",
            "4,8",
        ]
    )
    with mock.patch.object(
        tpu_lowering,
        "lower_torch_callable_to_mlir",
        side_effect=_mock_lower_torch_callable_to_mlir,
    ):
        exported_cfg = cli.run_export_from_args(
            args, vllm_config=vllm_cfg, model=pt_model
        )
    assert exported_cfg.uses_mrope is True
    assert "cos_sin_cache" not in pt_model.rotary_emb._buffers

    with open(os.path.join(out_dir, "metadata.json")) as f:
        meta = json.load(f)
    fwd8_inputs = meta["functions"]["forward"]["8"]["inputs"]
    pos_entry = next(inp for inp in fwd8_inputs if inp["name"] == "positions")
    assert pos_entry["shape"] == [3, 8]

    with open(os.path.join(out_dir, "serving_checkpoint", "weight.json")) as f:
        wjson = json.load(f)
    assert wjson["mesh"] == {"mesh_shape": [1, 1], "axis_names": ["dp", "tp"]}
    assert wjson["lm_head.weight"]["sources"][0]["path"] == "model.safetensors"
    assert wjson["hf_checkpoint_path"] == os.path.abspath(hf_dir)


def test_build_export_config_raises_block_size_to_fit_smem(tmp_path: Path) -> None:
    def cfg(max_num_seqs: int, max_model_len: int, block_size: int = 16) -> Any:
        return SimpleNamespace(
            model_config=SimpleNamespace(
                model=str(tmp_path), dtype="bfloat16", max_model_len=max_model_len
            ),
            cache_config=SimpleNamespace(block_size=block_size),
            scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        )

    build = universal_adapter.build_export_config_from_vllm
    # 128 seqs x 40960 tokens at page 16 is a 1.25 MiB block table.
    assert build(cfg(128, 40960)).block_size == 64
    assert build(cfg(256, 40960)).block_size == 128
    assert build(cfg(8, 128, block_size=8)).block_size == 8
    big = build(cfg(128, 40960))
    assert big.max_num_pages * 4 <= 1024 * 1024 // 2


def test_module_derived_gate_up_sources_beat_name_heuristic() -> None:
    # Qwen3-0.6B: gate_up_proj is stored `[in=1024, 2*inter=6144]`, so the
    # shape-only rule would mistake it for `[2*half, in]` and skip the transpose.
    tensor_to_file = {
        "model.layers.0.mlp.gate_proj.weight": "model.safetensors",
        "model.layers.0.mlp.up_proj.weight": "model.safetensors",
    }
    module_sources = [
        {
            "path": "",
            "source_tensor": f"model.layers.0.mlp.{sub}.weight",
            "source_shape": [3072, 1024],
            "transpose": [1, 0],
            "dst_slices": [{"dim": 1, "start": start, "length": 3072}],
        }
        for sub, start in (("gate_proj", 0), ("up_proj", 3072))
    ]
    sources = weight_mapper.expand_or_resolve_weight_sources(
        "model.layers.0.mlp.gate_up_proj.weight",
        {"shape": [1024, 6144], "sources": module_sources},
        tensor_to_file,
        "model.safetensors",
    )
    assert [s["source_shape"] for s in sources] == [[3072, 1024], [3072, 1024]]
    assert all(s["transpose"] == [1, 0] for s in sources)
    assert [s["dst_slices"][0]["start"] for s in sources] == [0, 3072]
    assert all(s["path"] == "model.safetensors" for s in sources)


def test_companion_files_are_copied_and_declared_in_serving_config(tmp_path) -> None:
    hf_dir = tmp_path / "hf_ckpt"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text(json.dumps({"eos_token_id": 2}))
    (hf_dir / "tokenizer.json").write_text(json.dumps({"version": "1.0"}))
    (hf_dir / "generation_config.json").write_text(
        json.dumps({"eos_token_id": [2, 0], "do_sample": False})
    )
    out_dir = tmp_path / "artifact"
    out_dir.mkdir()
    cfg = api.TextOnlyLLMExportConfig(
        model_name="tiny", vocab_size=64, hf_checkpoint_dir=str(hf_dir)
    )

    serving_config = exporter._emit_model_companion_files(cfg, str(out_dir))

    assert serving_config["tokenizer_config"] == {
        "source": {"TokenizerPath": {"path": "tokenizer.json"}}
    }
    assert serving_config["generation_config"] == {"path": "generation_config.json"}
    for name in ("tokenizer.json", "generation_config.json"):
        assert (out_dir / name).read_text() == (hf_dir / name).read_text()
    assert not (out_dir / "config.json").exists()


def test_generation_config_is_not_declared_when_the_checkpoint_has_none(
    tmp_path,
) -> None:
    hf_dir = tmp_path / "hf_ckpt"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text(json.dumps({"eos_token_id": 2}))
    out_dir = tmp_path / "artifact"
    out_dir.mkdir()
    cfg = api.TextOnlyLLMExportConfig(
        model_name="tiny", vocab_size=64, hf_checkpoint_dir=str(hf_dir)
    )

    serving_config = exporter._emit_model_companion_files(cfg, str(out_dir))

    assert "generation_config" not in serving_config
    assert not (out_dir / "generation_config.json").exists()


def _parallel_vllm_cfg(tmp_path: Path, tp: int, dp: int, pp: int) -> Any:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            model=str(tmp_path), dtype="bfloat16", max_model_len=128
        ),
        cache_config=SimpleNamespace(block_size=16),
        scheduler_config=SimpleNamespace(max_num_seqs=8),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp,
            data_parallel_size=dp,
            pipeline_parallel_size=pp,
        ),
    )


@pytest.mark.parametrize(("dp", "pp"), [(2, 1), (1, 2)])
def test_build_export_config_rejects_data_and_pipeline_parallelism(
    tmp_path: Path, dp: int, pp: int
) -> None:
    with pytest.raises(NotImplementedError, match=f"dp={dp} pp={pp}"):
        universal_adapter.build_export_config_from_vllm(
            _parallel_vllm_cfg(tmp_path, 1, dp, pp)
        )


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_build_export_config_spans_one_partition_per_tp_rank(
    tmp_path: Path, tp: int
) -> None:
    cfg = universal_adapter.build_export_config_from_vllm(
        _parallel_vllm_cfg(tmp_path, tp, 1, 1)
    )
    assert (cfg.num_replicas, cfg.num_partitions) == (1, tp)
    assert cfg.tensor_parallel_size == tp
    assert cfg.use_spmd_partitioning is True


def test_main_args_read_finds_the_arguments_main_reads() -> None:
    args = ", ".join(f"%arg{i}: tensor<i64>" for i in range(11))
    mlir = "\n".join(
        [
            "module {",
            f"  func.func public @main({args}) -> tensor<i64> {{",
            "    %0 = stablehlo.add %arg2, %arg10 : tensor<i64>",
            "    return %0 : tensor<i64>",
            "  }",
            "  func.func private @helper(%arg1: tensor<i64>) -> tensor<i64> {",
            "    return %arg1 : tensor<i64>",
            "  }",
            "}",
        ]
    )
    # %arg10 does not count as %arg1, and @helper's own %arg1 is not @main's.
    assert exporter._main_args_read(mlir) == {2, 10}


def test_params_read_by_forward_drops_unread_forward_weights() -> None:
    cfg = api.TextOnlyLLMExportConfig(
        model_name="tiny", vocab_size=64, token_buckets=(8, 16), num_layers=1
    )
    params = {
        "model.embed_tokens.weight": api.ShapeDtypeStruct((64, 8), "bf16"),
        "model.embed_tokens.org_vocab_start_index": api.ShapeDtypeStruct((), "i64"),
        "model.norm.weight": api.ShapeDtypeStruct((8,), "bf16"),
        "lm_head.weight": api.ShapeDtypeStruct((64, 8), "bf16"),
    }
    model = SimpleNamespace(
        export_config=cfg,
        abstract_states=lambda: [api.ShapeDtypeStruct((4, 8, 2, 2, 4), "bf16")],
    )
    num_user = len(exporter._user_inputs(cfg, 8))

    def fake_lower(_model: Any, *, forward_names: Sequence[str], **_: Any) -> str:
        assert "lm_head.weight" not in forward_names
        read = [
            num_user + forward_names.index(n)
            for n in ("model.embed_tokens.weight", "model.norm.weight")
        ]
        args = ", ".join(f"%arg{i}: tensor<i32>" for i in range(num_user + 4))
        uses = "\n".join(
            f"    %u{i} = stablehlo.abs %arg{i} : tensor<i32>" for i in read
        )
        header = f"  func.func public @main({args}) -> () {{"
        return "\n".join(["module {", header, uses, "  }", "}", ""])

    with mock.patch.object(tpu_lowering, "lower_torch_forward_to_mlir", fake_lower):
        kept = exporter._params_read_by_forward(model, params)
    assert list(kept) == [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    ]


def test_resolve_hf_checkpoint_dir_uses_cached_hub_snapshot(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshots" / "main"
    snapshot.mkdir(parents=True)

    def fake_cached_file(repo: str, filename: str, **kwargs: Any) -> str:
        assert kwargs["local_files_only"] is True
        if repo != "org/tiny":
            raise OSError(f"{repo} is not cached")
        return str(snapshot / filename)

    resolve = universal_adapter.resolve_hf_checkpoint_dir
    assert resolve(str(tmp_path)) == str(tmp_path)
    assert resolve(None) is None
    with mock.patch("transformers.utils.cached_file", fake_cached_file):
        assert resolve("org/tiny") == str(snapshot)
        with pytest.raises(FileNotFoundError, match="org/missing"):
            resolve("org/missing")


def test_weight_json_declares_constants_and_full_source_shapes(tmp_path: Path) -> None:
    hf_dir = tmp_path / "hf"
    hf_dir.mkdir()
    header = {
        "model.embed_tokens.weight": {
            "dtype": "BF16",
            "shape": [64, 8],
            "data_offsets": [0, 1024],
        }
    }
    hb = json.dumps(header).encode("utf-8")
    with open(hf_dir / "model.safetensors", "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
    cfg = api.TextOnlyLLMExportConfig(
        model_name="tiny", vocab_size=64, hf_checkpoint_dir=str(hf_dir)
    )
    # Rank-local shapes of a tensor-parallel size 2 export.
    params = {
        "model.embed_tokens.weight": api.ShapeDtypeStruct(
            (32, 8), "bf16", sharding=["tp", None]
        ),
        "model.embed_tokens.org_vocab_start_index": api.ShapeDtypeStruct((), "i64"),
    }
    model = SimpleNamespace(
        export_config=cfg,
        model=None,
        constant_axis_values=lambda name: [0, 32],
    )
    out_dir = tmp_path / "artifact"
    exporter._emit_declarative_weight_json(
        model,
        params,
        {"model.embed_tokens.weight": 6, "model.embed_tokens.org_vocab_start_index": 7},
        str(out_dir),
        1,
        2,
        constants=frozenset({"model.embed_tokens.org_vocab_start_index"}),
    )

    wjson = json.loads((out_dir / "serving_checkpoint" / "weight.json").read_text())
    embed = wjson["model.embed_tokens.weight"]
    assert embed["shape"] == [32, 8]
    assert embed["sources"][0]["source_shape"] == [64, 8]
    assert wjson["model.embed_tokens.org_vocab_start_index"] == {
        "index": 7,
        "shape": [],
        "dtype": "int64",
        "axis_values": {"axis": "tp", "values": [0, 32]},
    }


@pytest.mark.usefixtures("single_rank_tp_group")
def test_compute_logits_returns_the_vocabulary_without_padding() -> None:
    # The lm_head shard carries 2 padding rows past the 10-token vocabulary.
    exportable = SimpleNamespace(
        _export_config=SimpleNamespace(vocab_size=10),
        _local_logits=lambda h, w: torch.matmul(h, w.T),
    )
    hidden = torch.ones(3, 4)
    lm_head = torch.arange(48.0).reshape(12, 4)
    logits = universal_adapter.UniversalVllmExportableLLM.compute_logits(
        exportable, hidden, lm_head
    )
    assert logits.shape == (3, 10)
    assert torch.equal(logits, torch.matmul(hidden, lm_head.T)[:, :10])


def test_worker_env_takes_the_first_tp_devices_of_the_slice() -> None:
    launch = {
        "TORCH_TPU_SLICEBUILDER_ADDRESSES": "h:8070,h:8071,h:8072,h:8073",
        "TORCH_TPU_TOPOLOGY": "1,1,1,2",
        "WORLD_SIZE": "8",
        "MASTER_ADDR": "h",
    }
    done = SimpleNamespace(stdout="log line\n" + json.dumps(launch) + "\n")
    with mock.patch.object(cli.subprocess, "run", return_value=done):
        env = cli._worker_env(2)
    assert env["TORCH_TPU_SLICEBUILDER_ADDRESSES"] == "h:8070,h:8071"
    assert env["TORCH_TPU_TOPOLOGY"] == "1,1,1,2"
    assert env["WORLD_SIZE"] == "2"
    assert env["MASTER_ADDR"] == "h"
    with (
        mock.patch.object(cli.subprocess, "run", return_value=done),
        pytest.raises(ValueError, match="needs 8 TPU devices"),
    ):
        cli._worker_env(8)


class _FakeWorker:
    """Stands in for a worker process: writes a file named after its rank."""

    fail_ranks: tuple[int, ...] = ()

    def __init__(self, cmd: Sequence[str], env: Mapping[str, str]) -> None:
        rank = int(env["RANK"])
        assert cmd[-1] == "--export-worker"
        out_dir = Path(cmd[cmd.index("--output-dir", len(cmd) - 4) + 1])
        out_dir.mkdir(parents=True)
        (out_dir / "metadata.json").write_text(json.dumps({"rank": rank}))
        self._code = 1 if rank in self.fail_ranks else 0

    def wait(self) -> int:
        return self._code

    def poll(self) -> int:
        return self._code


def test_launch_workers_keeps_rank0_output(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    with (
        mock.patch.object(cli, "_worker_env", lambda tp: {}),
        mock.patch.object(cli.subprocess, "Popen", _FakeWorker),
    ):
        cli._launch_workers(["--model", "m"], 4, str(out_dir))
    assert json.loads((out_dir / "metadata.json").read_text()) == {"rank": 0}
    assert not list(tmp_path.glob(".export_ranks_*"))


def test_launch_workers_fails_when_a_worker_fails(tmp_path: Path) -> None:
    class Failing(_FakeWorker):
        fail_ranks = (1,)

    out_dir = tmp_path / "out"
    with (
        mock.patch.object(cli, "_worker_env", lambda tp: {}),
        mock.patch.object(cli.subprocess, "Popen", Failing),
        pytest.raises(RuntimeError, match=r"ranks \[1\] failed"),
    ):
        cli._launch_workers(["--model", "m"], 2, str(out_dir))
    assert not out_dir.exists()
    assert not list(tmp_path.glob(".export_ranks_*"))
