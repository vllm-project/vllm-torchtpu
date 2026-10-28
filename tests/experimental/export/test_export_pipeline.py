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
    checkpoint,
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


def test_end_to_end_export_writes_no_weights(tmp_path) -> None:
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
        os.path.join(out_dir, "subgraphs", "sample_tokens_temperature_req4.mlir")
    )
    assert os.path.isfile(
        os.path.join(out_dir, "subgraphs", "sample_tokens_stochastic_req4.mlir")
    )
    assert os.path.isfile(
        os.path.join(out_dir, "subgraphs", "sample_tokens_penalized_req4.mlir")
    )
    # Weights ship only as the checkpoint of `--export-mode checkpoint|all`.
    assert not os.path.exists(os.path.join(out_dir, "serving_checkpoint"))


class _TinyRecurrentExportableLLM(_TinyTorchExportableLLM):
    """A tiny hybrid model: one paged attention layer and one recurrent layer."""

    def abstract_states(self) -> Sequence[api.ShapeDtypeStruct]:
        cfg = self._cfg
        return [
            api.ShapeDtypeStruct(
                (cfg.num_blocks, cfg.block_size, 2 * cfg.num_kv_heads, cfg.head_dim),
                cfg.dtype,
            ),
            api.ShapeDtypeStruct(
                (cfg.num_state_slots, 3, 1, 8),
                "f32",
                sharding=[None, None, None, "model"],
            ),
            api.ShapeDtypeStruct(
                (cfg.num_state_slots, 2, 4, 4),
                "f32",
                sharding=[None, "model", None, None],
            ),
        ]

    def extra_user_inputs(self) -> list[tuple[str, tuple[int, ...]]]:
        return [("mamba_state_indices", (self._cfg.max_num_seqs,))]

    def forward(  # type: ignore[override]
        self,
        weights: Mapping[str, torch.Tensor],
        kv_caches: Sequence[torch.Tensor],
        *,
        mamba_state_indices: torch.Tensor,
        **controls: torch.Tensor,
    ) -> tuple[Sequence[torch.Tensor], torch.Tensor]:
        kvs, hidden = super().forward(weights, kv_caches, **controls)
        kv, conv, recurrent = kvs
        rows = mamba_state_indices.to(torch.int64)
        return [kv, conv.index_fill(0, rows, 1.0), recurrent], hidden


def test_end_to_end_export_of_a_recurrent_model(tmp_path: Path) -> None:
    cfg = api.TextOnlyLLMExportConfig(
        model_name="tiny-hybrid",
        vocab_size=32,
        hidden_size=16,
        num_layers=2,
        num_q_heads=2,
        num_kv_heads=2,
        head_dim=8,
        block_size=8,
        num_blocks=16,
        max_num_seqs=8,
        max_model_len=64,
        token_buckets=(8,),
        req_buckets=(4, 8),
        emit_byte_tokenizer=True,
    )
    out_dir = str(tmp_path / "exported_hybrid")
    with mock.patch.object(
        tpu_lowering,
        "lower_torch_callable_to_mlir",
        side_effect=_mock_lower_torch_callable_to_mlir,
    ):
        exporter.export(_TinyRecurrentExportableLLM(cfg), out_dir)
    validator.validate_exported_model_dir(out_dir)

    meta = json.loads(Path(out_dir, "metadata.json").read_text())
    forward = meta["functions"]["forward"]["8"]
    states = [t for t in forward["inputs"] if t["argument_type"] == "state"]
    assert [(t["shape"], t["dtype"], t["sharding"]) for t in states] == [
        ([16, 8, 4, 8], "bf16", [None, None, None, None]),
        ([9, 3, 1, 8], "f32", [None, None, None, "model"]),
        ([9, 2, 4, 4], "f32", [None, "model", None, None]),
    ]
    # Recurrent state pools hold one slot per sequence plus null slot 0.
    assert meta["serving_config"]["num_state_slots"] == 9
    # The recurrent control input follows the state, which keeps its indices.
    last = forward["inputs"][-1]
    assert last["name"] == "mamba_state_indices"
    assert last["argument_type"] == "user_io" and last["shape"] == [8]
    assert last["arg_index"] == states[-1]["arg_index"] + 1
    outputs = forward["outputs"]
    assert [t["argument_type"] for t in outputs] == ["state"] * 3 + ["intermediate"]
    assert outputs[-1]["name"] == "hidden_states" and outputs[-1]["arg_index"] == 3
    mlir = Path(out_dir, forward["file_path"]).read_text()
    assert f"%arg{last['arg_index']}: tensor<8xi32>" in mlir


def test_lm_head_name_finds_a_nested_language_model_head() -> None:
    params = {
        "visual.blocks.0.attn.qkv.weight": None,
        "language_model.model.embed_tokens.weight": None,
        "language_model.lm_head.weight": None,
    }
    assert exporter._lm_head_name(params) == "language_model.lm_head.weight"
    del params["language_model.lm_head.weight"]
    assert exporter._lm_head_name(params) == "language_model.model.embed_tokens.weight"


@pytest.mark.parametrize(
    ("names", "alias"),
    [
        # A tied head is named after the embedding, nested or not.
        (
            ["model.embed_tokens.weight"],
            ("lm_head.weight", "model.embed_tokens.weight"),
        ),
        (
            [
                "language_model.model.embed_tokens.weight",
                "visual.patch_embed.proj.weight",
                "mtp.embed_tokens.weight",
            ],
            (
                "language_model.lm_head.weight",
                "language_model.model.embed_tokens.weight",
            ),
        ),
        # An untied head has its own weight.
        (["model.embed_tokens.weight", "lm_head.weight"], None),
        (
            [
                "language_model.model.embed_tokens.weight",
                "language_model.lm_head.weight",
            ],
            None,
        ),
    ],
)
def test_tied_lm_head_alias(names: list[str], alias: tuple[str, str] | None) -> None:
    assert weight_mapper.tied_lm_head_alias(names) == alias


def test_a_nested_tied_lm_head_keeps_the_embedding_a_forward_weight() -> None:
    # Qwen3.5-0.8B: a multimodal model whose language model ties lm_head to the
    # embedding.
    model = nn.Module()
    model.language_model = nn.Module()
    model.language_model.model = nn.Module()
    embed = nn.Embedding(32, 8)
    model.language_model.model.embed_tokens = embed
    tensors = checkpoint.local_tensors(model)
    assert tensors["language_model.lm_head.weight"] is embed.weight
    # compute_logits projects with the alias, so forward keeps the embedding.
    assert exporter._lm_head_name(tensors) == "language_model.lm_head.weight"


class _FakeRecurrentLayer:
    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return (3, 1, 64), (4, 16, 16)

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return torch.float32, torch.bfloat16


@pytest.mark.parametrize("tp_size", [1, 2])
def test_recurrent_state_specs_split_conv_channels_and_value_heads(
    tp_size: int,
) -> None:
    with mock.patch(
        "vllm.model_executor.layers.mamba.mamba_utils.is_conv_state_dim_first",
        return_value=False,
    ):
        conv, recurrent = universal_adapter._recurrent_state_specs(
            _FakeRecurrentLayer(), num_slots=32, tp_size=tp_size
        )
    split = "model" if tp_size > 1 else None
    assert (conv.shape, conv.dtype, conv.sharding) == (
        (32, 3, 1, 64),
        "f32",
        [None, None, None, split],
    )
    assert (recurrent.shape, recurrent.dtype, recurrent.sharding) == (
        (32, 4, 16, 16),
        "bf16",
        [None, split, None, None],
    )


def test_metadata_has_one_serving_config_with_what_the_runtime_reads(tmp_path) -> None:
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
        token_buckets=(8,),
        req_buckets=(4, 8),
        emit_byte_tokenizer=True,
    )
    out_dir = str(tmp_path / "exported")
    with mock.patch.object(
        tpu_lowering,
        "lower_torch_callable_to_mlir",
        side_effect=_mock_lower_torch_callable_to_mlir,
    ):
        exporter.export(_TinyTorchExportableLLM(cfg), out_dir)
    meta = json.loads(Path(out_dir, "metadata.json").read_text())
    assert "config" not in meta
    serving = meta["serving_config"]
    assert set(serving) == {
        "vocab_size",
        "eos_token_id",
        "max_num_seqs",
        "max_num_pages",
        "max_cu_seqs",
        "block_size",
        "num_blocks",
        "max_model_len",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
        "tokenizer_config",
    }
    assert (serving["vocab_size"], serving["num_blocks"]) == (32, 16)


def test_metadata_declares_the_speculative_config_it_was_exported_with(
    tmp_path,
) -> None:
    speculative = {
        "method": "ngram",
        "num_speculative_tokens": 4,
        "prompt_lookup_max": 3,
        "prompt_lookup_min": 3,
    }
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
        token_buckets=(8,),
        req_buckets=(4, 8),
        emit_byte_tokenizer=True,
        speculative_config=speculative,
    )
    out_dir = str(tmp_path / "exported")
    with mock.patch.object(
        tpu_lowering,
        "lower_torch_callable_to_mlir",
        side_effect=_mock_lower_torch_callable_to_mlir,
    ):
        exporter.export(_TinyTorchExportableLLM(cfg), out_dir)
    meta = json.loads(Path(out_dir, "metadata.json").read_text())
    assert meta["serving_config"]["speculative_config"] == speculative


def _export_metadata(tmp_path: Path) -> dict[str, Any]:
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
        token_buckets=(8,),
        req_buckets=(4, 8),
        emit_byte_tokenizer=True,
    )
    out_dir = str(tmp_path / "exported")
    with mock.patch.object(
        tpu_lowering,
        "lower_torch_callable_to_mlir",
        side_effect=_mock_lower_torch_callable_to_mlir,
    ):
        exporter.export(_TinyTorchExportableLLM(cfg), out_dir)
    return json.loads(Path(out_dir, "metadata.json").read_text())


def test_metadata_declares_the_xla_flags_in_libtpu_init_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "LIBTPU_INIT_ARGS",
        "--xla_tpu_use_dynamic_smem_negotiation=true"
        " --xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes=67108864"
        " --xla_tpu_scoped_vmem_limit_kib=0.5 --xla_tpu_mode=fast"
        " --xla_tpu_enable_foo --tpu_megacore=false --xla_jf_bar=false",
    )
    overrides = _export_metadata(tmp_path)["compile_options"]["env_option_overrides"]
    # Typed JSON values; a bare flag is true; non-XLA libtpu flags are dropped.
    assert overrides == {
        "xla_tpu_use_dynamic_smem_negotiation": True,
        "xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes": 67108864,
        "xla_tpu_scoped_vmem_limit_kib": 0.5,
        "xla_tpu_mode": "fast",
        "xla_tpu_enable_foo": True,
        "xla_jf_bar": False,
    }


def test_metadata_has_no_xla_flags_without_libtpu_init_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LIBTPU_INIT_ARGS", raising=False)
    assert "env_option_overrides" not in _export_metadata(tmp_path)["compile_options"]


def test_parse_args_takes_the_vllm_serve_command_line(capsys) -> None:
    # What a `vllm serve` benchmark config passes, model tag included: engine
    # arguments shape the artifact, frontend options are reported and ignored.
    parser = cli.build_arg_parser()
    if "model_tag" not in {a.dest for a in parser._actions}:
        pytest.skip(
            "vLLM cannot build its serve parser here (its platform's device is "
            "unavailable); run with VLLM_TARGET_DEVICE=cpu VLLM_PLUGINS="
        )
    args = cli.parse_args(
        [
            "Qwen/Qwen3-0.6B",
            "--output-dir",
            "/tmp/out",
            "--max-model-len",
            "1024",
            "--block-size",
            "32",
            "--port",
            "8001",
            "--default-chat-template-kwargs",
            '{"enable_thinking": false}',
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "hermes",
        ]
    )
    assert args.model == "Qwen/Qwen3-0.6B"
    assert (args.max_model_len, args.block_size) == (1024, 32)
    ignored = capsys.readouterr().err
    for flag in [
        "--port",
        "--default-chat-template-kwargs",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
    ]:
        assert flag in ignored, ignored
    for flag in ["--max-model-len", "--block-size", "--output-dir"]:
        assert flag not in ignored, ignored


def test_export_speculative_config_keeps_vllm_field_names() -> None:
    assert cli.export_speculative_config(None) is None
    assert cli.export_speculative_config(
        '{"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_max": 3}'
    ) == {
        "method": "ngram",
        "num_speculative_tokens": 4,
        "prompt_lookup_max": 3,
        "prompt_lookup_min": 3,
    }


@pytest.mark.parametrize(
    "raw, error",
    [
        ({"method": "mtp", "num_speculative_tokens": 1}, "not supported"),
        ({"method": "ngram", "prompt_lookup_max": 3}, "num_speculative_tokens"),
        ({"method": "ngram", "num_speculative_tokens": 2}, "prompt_lookup_max"),
        (
            {"method": "ngram", "num_speculative_tokens": 0, "prompt_lookup_max": 3},
            "num_speculative_tokens",
        ),
        (
            {
                "method": "ngram",
                "num_speculative_tokens": 2,
                "prompt_lookup_max": 3,
                "model": "x",
            },
            "fields",
        ),
    ],
)
def test_export_speculative_config_rejects_what_the_runtime_cannot_run(
    raw: dict[str, Any], error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        cli.export_speculative_config(raw)


def test_compute_logits_output_is_the_samplers_intermediate_input(tmp_path) -> None:
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
        token_buckets=(8,),
        req_buckets=(4, 8),
        emit_byte_tokenizer=True,
    )
    out_dir = str(tmp_path / "exported")
    with mock.patch.object(
        tpu_lowering,
        "lower_torch_callable_to_mlir",
        side_effect=_mock_lower_torch_callable_to_mlir,
    ):
        exporter.export(_TinyTorchExportableLLM(cfg), out_dir)
    functions = json.loads(Path(out_dir, "metadata.json").read_text())["functions"]
    samplers = [
        "sample_tokens",
        "sample_tokens_temperature",
        "sample_tokens_stochastic",
        "sample_tokens_penalized",
    ]
    for req, spec in functions["compute_logits"].items():
        (logits,) = spec["outputs"]
        assert (logits["name"], logits["argument_type"]) == ("logits", "intermediate")
        for fn in samplers:
            inputs = {t["name"]: t for t in functions[fn][req]["inputs"]}
            assert inputs["logits"]["argument_type"] == "intermediate", (fn, req)
            assert inputs["logits"]["shape"] == logits["shape"], (fn, req)


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
    (hf_dir / "tokenizer.json").write_text(json.dumps({"version": "1.0"}))
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
            "--export-mode",
            "model",
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

    assert meta["mlir_shapes"] == "shard_local"
    assert meta["mesh"] == {"mesh_shape": [1, 1], "axis_names": ["data", "model"]}
    assert not os.path.exists(os.path.join(out_dir, "serving_checkpoint"))


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


def _adapter_without_model(cfg: api.TextOnlyLLMExportConfig) -> Any:
    adapter = object.__new__(universal_adapter.UniversalVllmExportableLLM)
    adapter._export_config = cfg
    adapter._vllm_config = None
    adapter._model = nn.Module()
    return adapter


def test_fit_num_blocks_to_hbm_fills_the_budget_with_exported_kv_blocks() -> None:
    from vllm_torchtpu import utils

    cfg = api.TextOnlyLLMExportConfig(
        model_name="tiny",
        num_layers=2,
        num_kv_heads=4,
        head_dim=128,
        block_size=128,
        num_blocks=512,
    )
    adapter = _adapter_without_model(cfg)
    # Two layers of [block, block_size, kv_heads * 2, head_dim] bf16 states.
    block_bytes = 2 * 128 * 4 * 2 * 128 * 2
    budget = utils.HbmBudget(
        total_limit=100 * block_bytes,
        total_used=20 * block_bytes,
        cap=90 * block_bytes,
        available=70 * block_bytes + 7,
    )
    with mock.patch.object(utils, "compute_hbm_budget", return_value=budget) as hbm:
        assert adapter.fit_num_blocks_to_hbm(0.9, devices=["tpu:0"]) == 70
    assert hbm.call_args.args == (["tpu:0"], 0.9)
    assert adapter.export_config.num_blocks == 70
    assert adapter.abstract_states()[0].shape[0] == 70


def test_fit_num_blocks_to_hbm_uses_kv_cache_memory_bytes_without_the_device() -> None:
    from vllm_torchtpu import utils

    cfg = api.TextOnlyLLMExportConfig(
        model_name="tiny", num_layers=2, num_kv_heads=4, head_dim=128, block_size=128
    )
    adapter = _adapter_without_model(cfg)
    block_bytes = 2 * 128 * 4 * 2 * 128 * 2
    with mock.patch.object(utils, "compute_hbm_budget") as hbm:
        assert adapter.fit_num_blocks_to_hbm(0.9, 33 * block_bytes + 1) == 33
    hbm.assert_not_called()
    assert adapter.export_config.num_blocks == 33


def test_fit_num_blocks_to_hbm_takes_recurrent_state_off_the_budget_first() -> None:
    cfg = api.TextOnlyLLMExportConfig(
        model_name="tiny", num_layers=1, num_kv_heads=4, head_dim=128, block_size=128
    )
    adapter = _adapter_without_model(cfg)
    state_bytes = 1000
    kv_block_bytes = 128 * 4 * 2 * 128 * 2

    def abstract_states() -> list[api.ShapeDtypeStruct]:
        c = adapter.export_config
        return [
            api.ShapeDtypeStruct((c.num_blocks, 128, 8, 128), "bf16"),
            # A recurrent state pool: one f32 slot of `state_bytes` per sequence.
            api.ShapeDtypeStruct((c.num_state_slots, state_bytes // 4), "f32"),
        ]

    adapter.abstract_states = abstract_states
    fixed = cfg.num_state_slots * state_bytes
    assert adapter.fit_num_blocks_to_hbm(0.9, fixed + 33 * kv_block_bytes + 1) == 33
    assert adapter.abstract_states()[1].shape[0] == cfg.num_state_slots


def test_fit_num_blocks_to_hbm_rejects_a_budget_without_room_for_a_block() -> None:
    from vllm_torchtpu import utils

    adapter = _adapter_without_model(
        api.TextOnlyLLMExportConfig(model_name="tiny", num_layers=1)
    )
    budget = utils.HbmBudget(total_limit=100, total_used=95, cap=90, available=-5)
    with (
        mock.patch.object(utils, "compute_hbm_budget", return_value=budget),
        pytest.raises(ValueError, match="gpu-memory-utilization"),
    ):
        adapter.fit_num_blocks_to_hbm(0.9, devices=["tpu:0"])


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
    (hf_dir / "tokenizer.json").write_text(json.dumps({"version": "1.0"}))
    out_dir = tmp_path / "artifact"
    out_dir.mkdir()
    cfg = api.TextOnlyLLMExportConfig(
        model_name="tiny", vocab_size=64, hf_checkpoint_dir=str(hf_dir)
    )

    serving_config = exporter._emit_model_companion_files(cfg, str(out_dir))

    assert "generation_config" not in serving_config
    assert not (out_dir / "generation_config.json").exists()


@pytest.mark.parametrize(
    "files,expected",
    [
        (
            {
                "chat_template.jinja": "jinja file",
                "tokenizer_config.json": json.dumps({"chat_template": "config"}),
            },
            "jinja file",
        ),
        ({"tokenizer_config.json": json.dumps({"chat_template": "config"})}, "config"),
        (
            {
                "tokenizer_config.json": json.dumps(
                    {
                        "chat_template": [
                            {"name": "tool_use", "template": "tools"},
                            {"name": "default", "template": "named default"},
                        ]
                    }
                )
            },
            "named default",
        ),
        ({"tokenizer_config.json": json.dumps({"bos_token": "<s>"})}, None),
        ({}, None),
    ],
)
def test_chat_template_is_written_and_declared_like_huggingface_finds_it(
    tmp_path, files: dict[str, str], expected: str | None
) -> None:
    hf_dir = tmp_path / "hf_ckpt"
    hf_dir.mkdir()
    (hf_dir / "tokenizer.json").write_text(json.dumps({"version": "1.0"}))
    for name, text in files.items():
        (hf_dir / name).write_text(text)
    out_dir = tmp_path / "artifact"
    out_dir.mkdir()
    cfg = api.TextOnlyLLMExportConfig(
        model_name="tiny", vocab_size=64, hf_checkpoint_dir=str(hf_dir)
    )

    serving_config = exporter._emit_model_companion_files(cfg, str(out_dir))

    if expected is None:
        assert "chat_template" not in serving_config
        assert not (out_dir / "chat_template.jinja").exists()
    else:
        assert serving_config["chat_template"] == {"path": "chat_template.jinja"}
        assert (out_dir / "chat_template.jinja").read_text() == expected


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
    fetched = []

    def fake_cached_file(repo: str, filename: str, **kwargs: Any) -> str | None:
        if repo != "org/tiny":
            raise OSError(f"{repo} is not on the Hub")
        fetched.append((filename, kwargs.get("local_files_only", False)))
        if filename == "generation_config.json":
            assert kwargs["_raise_exceptions_for_missing_entries"] is False
            return None
        return str(snapshot / filename)

    resolve = universal_adapter.resolve_hf_checkpoint_dir
    assert resolve(str(tmp_path)) == str(tmp_path)
    assert resolve(None) is None
    with mock.patch("transformers.utils.cached_file", fake_cached_file):
        assert resolve("org/tiny") == str(snapshot)
        with pytest.raises(FileNotFoundError, match="org/missing"):
            resolve("org/missing")
    # The companion files are fetched, not only looked up in the cache: a dummy
    # weight load caches config.json alone.
    assert fetched == [
        ("config.json", False),
        ("tokenizer.json", False),
        ("generation_config.json", False),
    ]


def test_companion_files_require_a_tokenizer(tmp_path: Path) -> None:
    hf_dir = tmp_path / "hf_ckpt"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text(json.dumps({"eos_token_id": 2}))
    out_dir = tmp_path / "artifact"
    out_dir.mkdir()
    cfg = api.TextOnlyLLMExportConfig(
        model_name="tiny", vocab_size=64, hf_checkpoint_dir=str(hf_dir)
    )
    with pytest.raises(FileNotFoundError, match="tokenizer.json"):
        exporter._emit_model_companion_files(cfg, str(out_dir))


def _tp2_model(constant_values: Mapping[str, list[Any]]) -> Any:
    cfg = api.TextOnlyLLMExportConfig(
        model_name="tiny", vocab_size=64, num_partitions=2
    )
    return SimpleNamespace(
        export_config=cfg,
        model=None,
        constant_axis_values=lambda name: constant_values[name],
    )


def test_tensor_layouts_declare_global_shapes_of_weights_and_constants() -> None:
    values = {
        "model.embed_tokens.org_vocab_start_index": [0, 32],
        "model.embed_tokens.added_vocab_start_index": [64, 64],
    }
    model = _tp2_model(values)
    # Rank-local shapes of a tensor-parallel size 2 export.
    params = {
        "model.embed_tokens.weight": api.ShapeDtypeStruct(
            (32, 8), "bf16", sharding=["tp", None]
        ),
        "model.embed_tokens.org_vocab_start_index": api.ShapeDtypeStruct((), "i64"),
        "model.embed_tokens.added_vocab_start_index": api.ShapeDtypeStruct((), "i64"),
    }
    layouts = exporter._tensor_layouts(
        model, params, frozenset(values), exporter._axis_sizes(model.export_config)
    )

    assert layouts["model.embed_tokens.weight"] == exporter._TensorLayout(
        shape=[64, 8], sharding=["model", None], lowered_shape=(32, 8)
    )
    # A per-device value is a sharded global array; each device holds `[1]`.
    assert layouts["model.embed_tokens.org_vocab_start_index"] == (
        exporter._TensorLayout(
            shape=[2], sharding=["model"], lowered_shape=(1,), reshape=()
        )
    )
    # A value shared by every device stays a replicated scalar.
    assert layouts["model.embed_tokens.added_vocab_start_index"] == (
        exporter._TensorLayout(shape=[], sharding=[], lowered_shape=())
    )


def test_set_aliasing_outputs_pairs_each_donor_with_its_result() -> None:
    mlir = (
        "module {\n"
        "  func.func public @main(%arg0: tensor<4xi32>, "
        "%arg1: tensor<2x4xbf16> {jax.buffer_donor = true}, "
        "%arg2: tensor<2x4xbf16> {jax.buffer_donor = true}) -> "
        "(tensor<2x4xbf16>, tensor<2x4xbf16>, tensor<4xbf16>) {\n"
        "  }\n}\n"
    )
    out = tpu_lowering.set_aliasing_outputs(mlir, {1: 0, 2: 1})
    assert "jax.buffer_donor" not in out
    assert "%arg1: tensor<2x4xbf16> {tf.aliasing_output = 0 : i32}" in out
    assert "%arg2: tensor<2x4xbf16> {tf.aliasing_output = 1 : i32}" in out
    with pytest.raises(ValueError, match="no aliased result"):
        tpu_lowering.set_aliasing_outputs(mlir, {1: 0})


def test_validator_rejects_a_donor_without_aliasing_output(tmp_path: Path) -> None:
    mlir = (
        "module {\n"
        "  func.func public @main(%arg0: tensor<2xbf16> {jax.buffer_donor = true}) "
        "-> tensor<2xbf16> {\n"
        "    return %arg0 : tensor<2xbf16>\n"
        "  }\n}\n"
    )
    (tmp_path / "abstract_params.mlir").write_text(mlir)
    entry = {
        "arg_index": 0,
        "index": 0,
        "name": "w",
        "shape": [2],
        "dtype": "bf16",
        "sharding": [None],
        "argument_type": "weight",
        "buffer_donor": True,
    }
    meta = {
        "format_version": 2,
        "mlir_shapes": "shard_local",
        "mesh": {"mesh_shape": [1, 1], "axis_names": ["data", "model"]},
        "functions": {
            "abstract_params": {
                "default": {
                    "file_path": "abstract_params.mlir",
                    "inputs": [entry],
                    "outputs": [dict(entry, buffer_donor=False)],
                }
            }
        },
    }
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="has no tf.aliasing_output"):
        validator.validate_exported_model_dir(str(tmp_path))
    (tmp_path / "abstract_params.mlir").write_text(
        mlir.replace("jax.buffer_donor = true", "tf.aliasing_output = 0 : i32")
    )
    validator.validate_exported_model_dir(str(tmp_path))
    meta["functions"]["abstract_params"]["default"]["inputs"][0]["dtype"] = "bfloat16"
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="not a StableHLO element type"):
        validator.validate_exported_model_dir(str(tmp_path))


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


class _LlamaLikeBackbone(nn.Module):
    """Backbone whose `forward` takes `intermediate_tensors` with no default."""

    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 4)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Any,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert intermediate_tensors is None
        return self.embed_tokens(input_ids)


class _KvCacheArgBackbone(_LlamaLikeBackbone):
    """Backbone that takes the KV caches and fails inside its forward."""

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Sequence[torch.Tensor] | None = None,
        attn_metadata: Any = None,
    ) -> torch.Tensor:
        if kv_caches is not None:
            raise TypeError("bug inside the model")
        return self.embed_tokens(input_ids)


def _run_backbone_forward(backbone: nn.Module) -> tuple[Sequence[Any], Any]:
    cfg = api.TextOnlyLLMExportConfig(model_name="tiny", num_layers=0)
    adapter = _adapter_without_model(cfg)
    adapter._model = nn.Module()
    adapter._model.model = backbone
    weight = torch.arange(32.0).reshape(8, 4)
    return adapter.forward(
        {"model.embed_tokens.weight": weight},
        [],
        input_ids=torch.tensor([1, 2, 3]),
        positions=torch.tensor([0, 1, 2]),
        kv_lens=torch.tensor([3]),
        block_tables=torch.zeros(1, 1, dtype=torch.int32),
        query_start_loc=torch.tensor([0, 3]),
        req_dist=torch.tensor([0, 0, 1]),
    )


def test_forward_passes_intermediate_tensors_to_the_backbone() -> None:
    _, hidden = _run_backbone_forward(_LlamaLikeBackbone())
    assert torch.equal(hidden, torch.arange(32.0).reshape(8, 4)[[1, 2, 3]])


def test_forward_propagates_type_errors_raised_inside_the_backbone() -> None:
    with pytest.raises(TypeError, match="bug inside the model"):
        _run_backbone_forward(_KvCacheArgBackbone())


def _pallas_impl(kv_dtype: torch.dtype | None) -> Any:
    from vllm_torchtpu.layers.adapter.attention import PallasAttentionBackendImpl

    impl = object.__new__(PallasAttentionBackendImpl)
    impl.num_kv_heads, impl.head_size = 2, 128
    impl.kv_cache_quantized_dtype = kv_dtype
    return impl


def _bare_layer(cls_path: str) -> Any:
    module_name, cls_name = cls_path.rsplit(".", 1)
    module = __import__(module_name, fromlist=[cls_name])
    return object.__new__(getattr(module, cls_name))


@pytest.mark.parametrize(
    "cls_path",
    [
        "vllm.model_executor.layers.mamba.mamba_mixer2.MambaMixer2",
        "vllm.model_executor.layers.mamba.short_conv.ShortConv",
        "vllm.model_executor.layers.mamba.mamba_mixer.MambaMixer",
    ],
)
def test_layer_state_specs_reject_unsupported_recurrent_layers(cls_path: str) -> None:
    layer = _bare_layer(cls_path)
    exportable = SimpleNamespace(
        _export_config=SimpleNamespace(
            dtype="bf16", num_blocks=16, block_size=32, tensor_parallel_size=1
        ),
        _attention_layers=lambda: [layer],
    )
    cls_name = cls_path.rsplit(".", 1)[1]
    with pytest.raises(
        NotImplementedError,
        match=rf"recurrent layer {cls_name} is not supported by export; "
        r"supported: GatedDeltaNetAttention",
    ):
        universal_adapter.UniversalVllmExportableLLM._layer_state_specs(exportable)


@pytest.mark.parametrize("kv_dtype", [None, torch.float8_e4m3fn])
def test_layer_state_specs_declare_kv_and_recurrent_state(
    kv_dtype: torch.dtype | None,
) -> None:
    from vllm_torchtpu.layers.adapter.attention import PallasAttentionBackend

    attention = SimpleNamespace(impl=_pallas_impl(kv_dtype))
    recurrent = _FakeRecurrentLayer()
    exportable = SimpleNamespace(
        _export_config=SimpleNamespace(
            dtype="bf16",
            num_blocks=16,
            num_state_slots=9,
            block_size=32,
            tensor_parallel_size=1,
        ),
        _attention_layers=lambda: [recurrent, attention],
    )
    with (
        mock.patch.object(
            universal_adapter,
            "_is_recurrent_layer",
            side_effect=lambda m: m is recurrent,
        ),
        mock.patch.object(universal_adapter, "_check_recurrent_layer_supported"),
        mock.patch(
            "vllm.model_executor.layers.mamba.mamba_utils.is_conv_state_dim_first",
            return_value=False,
        ),
    ):
        recurrent_states, kv_states = (
            universal_adapter.UniversalVllmExportableLLM._layer_state_specs(exportable)
        )
    # Recurrent state is per sequence slot; the KV cache is per block.
    assert [s.shape for s in recurrent_states] == [(9, 3, 1, 64), (9, 4, 16, 16)]
    # An fp8 KV cache is declared in fp8, with fp8's packing.
    stored = kv_dtype or torch.bfloat16
    (kv,) = kv_states
    assert kv.dtype == tpu_lowering.dtype_name(stored)
    assert kv.shape == PallasAttentionBackend.get_kv_cache_shape(16, 32, 2, 128, stored)


@pytest.mark.parametrize("kv_dtype", [None, torch.float8_e4m3fn])
def test_layer_state_specs_declare_the_mla_latent_cache(
    kv_dtype: torch.dtype | None,
) -> None:
    from vllm_torchtpu.layers.adapter.attention import (
        PallasMLAttentionBackend,
        PallasMLAttentionBackendImpl,
    )

    # One latent KV head of kv_lora_rank + qk_rope_head_dim = 576, which the
    # MLA kernel stores packed: [blocks, block_size / packing, packing, 640].
    mla = SimpleNamespace(
        impl=object.__new__(PallasMLAttentionBackendImpl),
        head_size=576,
        num_kv_heads=1,
        kv_cache_quantized_dtype=kv_dtype,
        use_sparse=False,
    )
    exportable = SimpleNamespace(
        _export_config=SimpleNamespace(
            dtype="bf16", num_blocks=16, block_size=1024, tensor_parallel_size=1
        ),
        _attention_layers=lambda: [mla],
    )
    per_layer = universal_adapter.UniversalVllmExportableLLM._layer_state_specs(
        exportable
    )
    assert per_layer is not None
    ((kv,),) = per_layer
    stored = kv_dtype or torch.bfloat16
    assert kv.dtype == tpu_lowering.dtype_name(stored)
    assert kv.shape == PallasMLAttentionBackend.get_kv_cache_shape(
        16, 1024, 1, 576, stored
    )
    packing = 4 if kv_dtype else 2
    assert kv.shape == (16, 1024 // packing, packing, 640)


def test_layer_state_specs_reject_sparse_mla() -> None:
    from vllm_torchtpu.layers.adapter.attention import PallasMLAttentionBackendImpl

    indexer_cache = SimpleNamespace(impl=None)
    sparse_mla = SimpleNamespace(
        impl=object.__new__(PallasMLAttentionBackendImpl),
        head_size=576,
        num_kv_heads=1,
        kv_cache_quantized_dtype=None,
        use_sparse=True,
    )
    exportable = SimpleNamespace(
        _export_config=SimpleNamespace(
            dtype="bf16", num_blocks=16, block_size=1024, tensor_parallel_size=1
        ),
        _attention_layers=lambda: [indexer_cache, sparse_mla],
    )
    with pytest.raises(
        NotImplementedError,
        match="sparse MLA is not supported by export",
    ):
        universal_adapter.UniversalVllmExportableLLM._layer_state_specs(exportable)


@pytest.fixture
def clean_launch_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "TORCH_TPU_SLICEBUILDER_ADDRESSES",
        "TORCH_TPU_TOPOLOGY",
        cli.UNIFIED_BLOCK_POOL_ENV,
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.mark.usefixtures("clean_launch_env")
def test_worker_env_bootstraps_a_tp_slice() -> None:
    from vllm_torchtpu.platforms.tpu_platform import TpuPlatform

    before = dict(os.environ)
    with mock.patch.object(TpuPlatform, "_get_tpu_topology", return_value="1,1,1,2"):
        env = cli._worker_env(2)
    assert env["WORLD_SIZE"] == "2"
    assert env["TORCH_TPU_TOPOLOGY"] == "1,1,1,2"
    assert len(env["TORCH_TPU_SLICEBUILDER_ADDRESSES"].split(",")) == 2
    assert env["MASTER_ADDR"] and env["MASTER_PORT"]
    assert dict(os.environ) == before


@pytest.mark.usefixtures("clean_launch_env")
def test_worker_env_single_device_skips_the_tpu_bootstrap() -> None:
    env = cli._worker_env(1)
    assert env[cli.UNIFIED_BLOCK_POOL_ENV] == "0"
    assert "WORLD_SIZE" not in env
    assert "TORCH_TPU_TOPOLOGY" not in env
    assert env["MASTER_ADDR"] and env["MASTER_PORT"]


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


def test_main_rejects_data_parallelism_before_starting_workers(
    tmp_path: Path,
) -> None:
    argv = ["--model", "m", "--output-dir", str(tmp_path), "--data-parallel-size", "8"]
    with (
        mock.patch.object(cli, "_launch_workers") as launch,
        pytest.raises(NotImplementedError, match="dp=8 pp=1"),
    ):
        cli.main(argv)
    launch.assert_not_called()


def test_launch_workers_keeps_rank0_output(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    with (
        mock.patch.object(cli, "_worker_env", lambda tp: {}),
        mock.patch.object(cli.subprocess, "Popen", _FakeWorker),
    ):
        cli._launch_workers(["--model", "m"], 4, str(out_dir))
    assert json.loads((out_dir / "metadata.json").read_text()) == {"rank": 0}
    assert not list(tmp_path.glob(".export_ranks_*"))


def test_launch_workers_creates_the_output_parent_dir(tmp_path: Path) -> None:
    out_dir = tmp_path / "new" / "out"
    with (
        mock.patch.object(cli, "_worker_env", lambda tp: {}),
        mock.patch.object(cli.subprocess, "Popen", _FakeWorker),
    ):
        cli._launch_workers(["--model", "m"], 1, str(out_dir))
    assert json.loads((out_dir / "metadata.json").read_text()) == {"rank": 0}
    assert not list((tmp_path / "new").glob(".export_ranks_*"))


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


def test_launch_options_read_the_dispatch_flags() -> None:
    launch = cli._launch_options(
        ["--model", "m", "--export-mode=checkpoint", "-tp", "4", "--output-dir", "o"]
    )
    assert launch.export_mode == "checkpoint"
    assert launch.tensor_parallel_size == 4
    assert launch.output_dir == "o"
    assert not launch.export_worker
    assert cli._launch_options(["--output-dir", "o"]).export_mode == "all"


def test_main_checkpoint_mode_runs_on_cpu_without_parsing_engine_args(
    tmp_path: Path,
) -> None:
    argv = [
        "--model",
        "m",
        "--output-dir",
        str(tmp_path),
        "--export-mode",
        "checkpoint",
    ]
    with (
        mock.patch.object(cli, "parse_args") as parse,
        mock.patch.object(checkpoint, "export_checkpoint") as export_ckpt,
    ):
        cli.main([*argv, "-tp", "2"])
    parse.assert_not_called()
    export_ckpt.assert_called_once_with([*argv, "-tp", "2"], 2, str(tmp_path))

    with (
        mock.patch.object(cli, "parse_args", return_value="args"),
        mock.patch.object(checkpoint, "run_cpu_load_worker") as worker,
    ):
        cli.main([*argv, "--export-worker"])
    worker.assert_called_once_with("args")


def test_export_checkpoint_needs_an_exported_artifact(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no metadata.json"):
        checkpoint.export_checkpoint(["--model", "m"], 1, str(tmp_path))


def test_launch_workers_all_mode_saves_into_the_output_dir(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    stale = out_dir / "serving_checkpoint" / "data"
    stale.mkdir(parents=True)
    cmds: list[Sequence[str]] = []

    class Recording(_FakeWorker):
        def __init__(self, cmd: Sequence[str], env: Mapping[str, str]) -> None:
            cmds.append(cmd)
            super().__init__(cmd, env)

    def flag(cmd: Sequence[str], name: str) -> str:
        return cmd[list(cmd).index(name) + 1]

    with (
        mock.patch.object(cli, "_worker_env", lambda tp: {}),
        mock.patch.object(cli.subprocess, "Popen", Recording),
    ):
        cli._launch_workers(["--model", "m"], 2, str(out_dir), save_checkpoint=True)
    assert {flag(c, checkpoint.DATA_DIR_FLAG) for c in cmds} == {os.path.abspath(stale)}
    # Every rank joins the same JAX coordinator.
    assert len({flag(c, checkpoint.JAX_COORDINATOR_FLAG) for c in cmds}) == 1
    assert not stale.exists()


class _Router:
    """A plain object holding a module's parameter, as vLLM MoE routers do."""

    def __init__(self, bias: torch.Tensor) -> None:
        self.bias = bias


class _BackboneWithRouter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate = nn.Module()
        self.gate.bias = nn.Parameter(torch.zeros(4), requires_grad=False)
        self.experts = nn.Module()
        # Mirrors vLLM's MoERunner/RoutedExperts layout: the same `gate`
        # submodule is reachable at two module paths (`gate` and
        # `experts.gate`), `routed_experts` holds the shared Parameter directly,
        # and `router` is a plain object holding the Parameter outside the
        # module tree.
        self.experts.gate = self.gate
        self.experts.routed_experts = nn.Module()
        self.experts.routed_experts.bias = self.gate.bias
        self.experts.router = _Router(self.gate.bias)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> Any:
        del positions
        assert self.gate.bias is self.experts.routed_experts.bias
        assert self.gate.bias is self.experts.router.bias
        return input_ids.float()[:, None] + self.experts.router.bias[None, :]


def test_forward_binds_weights_held_outside_the_module_tree() -> None:
    wrapper = nn.Module()
    wrapper.model = _BackboneWithRouter()
    original = wrapper.model.gate.bias
    exportable = object.__new__(universal_adapter.UniversalVllmExportableLLM)
    exportable._model = wrapper
    exportable._vllm_config = None
    exportable._export_config = SimpleNamespace(dtype="bf16", max_num_seqs=1)
    controls = {
        name: torch.zeros(1, dtype=torch.int32)
        for name in ("kv_lens", "block_tables", "query_start_loc", "req_dist")
    }
    for step in (1.0, 2.0):
        bias = torch.arange(4, dtype=torch.float32) + step
        _, hidden = exportable.forward(
            {"model.gate.bias": bias},
            [],
            input_ids=torch.tensor([10, 20]),
            positions=torch.tensor([0, 1]),
            **controls,
        )
        # Every holder reads the bound weight, and repeated forward calls
        # (probe pass + bucket lowering) leave the shared parameter intact.
        assert torch.equal(hidden, torch.tensor([[10.0], [20.0]]) + bias)
        assert wrapper.model.gate.bias is original
        assert wrapper.model.experts.routed_experts.bias is original
        assert wrapper.model.experts.router.bias is original
        assert "bias" in wrapper.model.gate._parameters
    assert torch.equal(original, torch.zeros(4))


@pytest.mark.parametrize(
    "max_num_seqs,expected",
    [
        (512, (8, 16, 32, 64, 128, 256, 512)),
        (300, (8, 16, 32, 64, 128, 256, 300)),
        (8, (8,)),
        (4, (4,)),
    ],
)
def test_default_req_buckets_pad_like_the_vllm_runner(
    max_num_seqs: int, expected: tuple[int, ...]
) -> None:
    assert universal_adapter.default_req_buckets(max_num_seqs) == expected


def test_uniform_per_logit_is_in_range_and_depends_only_on_the_row() -> None:
    uniform = torch.tensor([0.1, 0.5, 0.9, 0.5])
    u = sampling.uniform_per_logit(uniform, 1000)
    assert u.shape == (4, 1000) and u.dtype == torch.float32
    assert bool((u > 0).all()) and bool((u < 1).all())
    assert torch.equal(u[1], u[3])  # same seed, same values
    assert not torch.equal(u[0], u[1])
    alone = sampling.uniform_per_logit(uniform[2:3], 1000)
    assert torch.equal(alone[0], u[2])  # batch-invariant
    assert abs(float(u.mean()) - 0.5) < 0.02


def _sample_frequencies(sample, logits_row, rows=20000, **params):
    torch.manual_seed(0)
    logits = logits_row.expand(rows, -1).contiguous()
    uniform = torch.rand(rows)
    tokens = sample(logits, uniform=uniform, **params).squeeze(-1)
    return torch.bincount(tokens, minlength=logits.shape[-1]).float() / rows


def test_temperature_and_stochastic_samplers_follow_the_softmax() -> None:
    logits_row = torch.tensor([[2.0, 1.0, 0.5, 0.0, -1.0, -3.0]])
    rows = 20000
    temp = torch.full((rows,), 0.7)
    expected = torch.softmax(logits_row[0] / 0.7, dim=-1)
    freq = _sample_frequencies(
        lambda logits, uniform, **p: sampling.temperature_tokens(logits, temp, uniform),
        logits_row,
        rows,
    )
    assert torch.allclose(freq, expected, atol=0.015), (freq, expected)
    no_k = torch.zeros(rows, dtype=torch.int32)
    no_p = torch.ones(rows)
    freq = _sample_frequencies(
        lambda logits, uniform, **p: sampling.stochastic_tokens(
            logits, temp, no_k, no_p, torch.zeros(rows), uniform
        ),
        logits_row,
        rows,
    )
    assert torch.allclose(freq, expected, atol=0.015), (freq, expected)


def test_stochastic_sampler_keeps_only_the_top_k_and_top_p_tokens() -> None:
    logits_row = torch.tensor([[3.0, 2.5, 2.0, 0.0, -1.0, -2.0]])
    rows = 4000
    temp = torch.ones(rows)
    top_k = torch.full((rows,), 2, dtype=torch.int32)
    freq = _sample_frequencies(
        lambda logits, uniform, **p: sampling.stochastic_tokens(
            logits, temp, top_k, torch.ones(rows), torch.zeros(rows), uniform
        ),
        logits_row,
        rows,
    )
    assert float(freq[2:].sum()) == 0.0 and float(freq[:2].min()) > 0.2
    top_p = torch.full((rows,), 0.5)
    freq = _sample_frequencies(
        lambda logits, uniform, **p: sampling.stochastic_tokens(
            logits,
            temp,
            torch.zeros(rows, dtype=torch.int32),
            top_p,
            torch.zeros(rows),
            uniform,
        ),
        logits_row,
        rows,
    )
    # softmax([3, 2.5, 2, ...]) = [0.47, 0.29, ...]: the top-p set is tokens 0 and 1.
    assert float(freq[2:].sum()) == 0.0 and float(freq[1]) > 0.2


def test_zero_temperature_rows_take_the_argmax() -> None:
    logits = torch.tensor([[0.0, 5.0, 1.0], [4.0, 0.0, 1.0]])
    temp = torch.tensor([0.0, 0.0])
    uniform = torch.tensor([0.3, 0.8])
    assert sampling.temperature_tokens(logits, temp, uniform).squeeze(-1).tolist() == [
        1,
        0,
    ]
    tokens = sampling.stochastic_tokens(
        logits,
        temp,
        torch.tensor([1, 1], dtype=torch.int32),
        torch.ones(2),
        torch.zeros(2),
        uniform,
    )
    assert tokens.squeeze(-1).tolist() == [1, 0]
