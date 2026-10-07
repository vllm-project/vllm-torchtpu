# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`relayout_rope_tables` re-materializes model-owned RoPE tables row-major."""

from unittest import mock

import torch

from vllm_torchtpu.runner import utils
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


class _Attention(torch.nn.Module):
    def __init__(self, table: torch.Tensor):
        super().__init__()
        self.register_buffer("_tpu_rope_cache", table, persistent=False)


def _relayout(model: torch.nn.Module) -> tuple[list[torch.Tensor], bool]:
    """Runs `relayout_rope_tables` with a fake row-major copy.

    Returns the tensors handed to `_to_row_major` and whether the entry
    layout pin was installed.
    """
    seen: list[torch.Tensor] = []

    def fake_row_major(tensor: torch.Tensor) -> torch.Tensor:
        seen.append(tensor)
        return tensor.clone()

    with (
        mock.patch.object(utils, "_to_row_major", side_effect=fake_row_major),
        mock.patch.object(utils, "_install_entry_layout_pin") as pin,
    ):
        utils.relayout_rope_tables(model)
    return seen, pin.called


def test_shared_tables_relaid_once_and_still_shared():
    shared, other = torch.randn(8, 4), torch.randn(8, 4)
    model = torch.nn.ModuleList(
        [_Attention(shared), _Attention(shared), _Attention(other)]
    )
    seen, pinned = _relayout(model)
    assert [id(t) for t in seen] == [id(shared), id(other)]
    assert pinned
    first, second, third = (m._tpu_rope_cache for m in model)
    assert first is second and first is not shared and third is not other
    assert torch.equal(first, shared) and torch.equal(third, other)


def test_non_2d_or_missing_tables_are_skipped():
    attention = _Attention(torch.randn(8))
    model = torch.nn.ModuleList([attention, torch.nn.Linear(2, 2)])
    seen, pinned = _relayout(model)
    assert seen == [] and not pinned
    assert attention._tpu_rope_cache.ndim == 1


def _make_dummy_runner() -> TPUModelRunner:
    runner = TPUModelRunner.__new__(TPUModelRunner)
    runner.device_config = mock.MagicMock()
    runner.vllm_config = mock.MagicMock()
    runner.model_config = mock.MagicMock(
        quantization=None,
        hf_config=mock.MagicMock(architectures=[]),
    )
    runner.load_config = mock.MagicMock()
    runner.mesh = mock.MagicMock()
    runner.parallel_config = mock.MagicMock(pipeline_parallel_size=1)
    return runner


def test_relayout_draft_model_rope_tables():
    shared, other = torch.randn(8, 4), torch.randn(8, 4)
    draft_model = torch.nn.ModuleList(
        [_Attention(shared), _Attention(shared), _Attention(other)]
    )
    seen, pinned = _relayout(draft_model)
    assert [id(t) for t in seen] == [id(shared), id(other)]
    assert pinned
    first, second, third = (m._tpu_rope_cache for m in draft_model)
    assert first is second and first is not shared and third is not other
    assert torch.equal(first, shared) and torch.equal(third, other)


def test_runner_load_model_relayouts_draft_model_rope_tables():
    drafter = mock.MagicMock()
    draft_table = torch.randn(8, 4)
    draft_model = torch.nn.ModuleList([_Attention(draft_table)])
    drafter.draft_model = draft_model

    target_table = torch.randn(8, 4)
    target_model = torch.nn.ModuleList([_Attention(target_table)])

    runner = _make_dummy_runner()

    real_torch_device = torch.device

    def fake_torch_device(d):
        if d == "tpu":
            return real_torch_device("cpu")
        return real_torch_device(d)

    with (
        mock.patch("torch.device", side_effect=fake_torch_device),
        mock.patch.object(
            TPUModelRunner,
            "_is_async_drafter",
            new_callable=mock.PropertyMock,
            return_value=True,
        ),
        mock.patch.object(
            TPUModelRunner,
            "_async_drafter",
            new_callable=mock.PropertyMock,
            return_value=drafter,
        ),
        mock.patch("vllm_torchtpu.runner.tpu_runner.get_tpu_quantization_config"),
        mock.patch("vllm_torchtpu.runner.tpu_runner.get_model_loader") as mock_loader,
        mock.patch("vllm_torchtpu.runner.tpu_runner.validate_online_fp8"),
        mock.patch(
            "vllm_torchtpu.model_loader_patches.evict_loaded_weights_page_cache"
        ),
        mock.patch.object(runner, "_initialize_pallas_kernels"),
        mock.patch.object(
            utils, "relayout_rope_tables", wraps=utils.relayout_rope_tables
        ) as mock_relayout,
    ):
        mock_loader.return_value.load_model.return_value = target_model
        runner.load_model()

        drafter.load_model.assert_called_once_with(target_model)
        assert mock_relayout.call_count == 2
        mock_relayout.assert_has_calls(
            [mock.call(target_model), mock.call(draft_model)]
        )


def test_runner_load_model_skips_when_draft_model_is_none():
    drafter = mock.MagicMock()
    drafter.draft_model = None

    target_table = torch.randn(8, 4)
    target_model = torch.nn.ModuleList([_Attention(target_table)])

    runner = _make_dummy_runner()

    real_torch_device = torch.device

    def fake_torch_device(d):
        if d == "tpu":
            return real_torch_device("cpu")
        return real_torch_device(d)

    with (
        mock.patch("torch.device", side_effect=fake_torch_device),
        mock.patch.object(
            TPUModelRunner,
            "_is_async_drafter",
            new_callable=mock.PropertyMock,
            return_value=True,
        ),
        mock.patch.object(
            TPUModelRunner,
            "_async_drafter",
            new_callable=mock.PropertyMock,
            return_value=drafter,
        ),
        mock.patch("vllm_torchtpu.runner.tpu_runner.get_tpu_quantization_config"),
        mock.patch("vllm_torchtpu.runner.tpu_runner.get_model_loader") as mock_loader,
        mock.patch("vllm_torchtpu.runner.tpu_runner.validate_online_fp8"),
        mock.patch(
            "vllm_torchtpu.model_loader_patches.evict_loaded_weights_page_cache"
        ),
        mock.patch.object(runner, "_initialize_pallas_kernels"),
        mock.patch.object(
            utils, "relayout_rope_tables", wraps=utils.relayout_rope_tables
        ) as mock_relayout,
    ):
        mock_loader.return_value.load_model.return_value = target_model
        runner.load_model()

        drafter.load_model.assert_called_once_with(target_model)
        assert mock_relayout.call_count == 1
        mock_relayout.assert_called_once_with(target_model)


def test_runner_load_model_skips_when_no_draft_model_attr():
    drafter = mock.MagicMock(spec=["load_model"])

    target_table = torch.randn(8, 4)
    target_model = torch.nn.ModuleList([_Attention(target_table)])

    runner = _make_dummy_runner()

    real_torch_device = torch.device

    def fake_torch_device(d):
        if d == "tpu":
            return real_torch_device("cpu")
        return real_torch_device(d)

    with (
        mock.patch("torch.device", side_effect=fake_torch_device),
        mock.patch.object(
            TPUModelRunner,
            "_is_async_drafter",
            new_callable=mock.PropertyMock,
            return_value=True,
        ),
        mock.patch.object(
            TPUModelRunner,
            "_async_drafter",
            new_callable=mock.PropertyMock,
            return_value=drafter,
        ),
        mock.patch("vllm_torchtpu.runner.tpu_runner.get_tpu_quantization_config"),
        mock.patch("vllm_torchtpu.runner.tpu_runner.get_model_loader") as mock_loader,
        mock.patch("vllm_torchtpu.runner.tpu_runner.validate_online_fp8"),
        mock.patch(
            "vllm_torchtpu.model_loader_patches.evict_loaded_weights_page_cache"
        ),
        mock.patch.object(runner, "_initialize_pallas_kernels"),
        mock.patch.object(
            utils, "relayout_rope_tables", wraps=utils.relayout_rope_tables
        ) as mock_relayout,
    ):
        mock_loader.return_value.load_model.return_value = target_model
        runner.load_model()

        drafter.load_model.assert_called_once_with(target_model)
        assert mock_relayout.call_count == 1
        mock_relayout.assert_called_once_with(target_model)


def test_runner_load_model_skips_when_not_async_drafter():
    target_table = torch.randn(8, 4)
    target_model = torch.nn.ModuleList([_Attention(target_table)])

    runner = _make_dummy_runner()

    real_torch_device = torch.device

    def fake_torch_device(d):
        if d == "tpu":
            return real_torch_device("cpu")
        return real_torch_device(d)

    with (
        mock.patch("torch.device", side_effect=fake_torch_device),
        mock.patch.object(
            TPUModelRunner,
            "_is_async_drafter",
            new_callable=mock.PropertyMock,
            return_value=False,
        ),
        mock.patch("vllm_torchtpu.runner.tpu_runner.get_tpu_quantization_config"),
        mock.patch("vllm_torchtpu.runner.tpu_runner.get_model_loader") as mock_loader,
        mock.patch("vllm_torchtpu.runner.tpu_runner.validate_online_fp8"),
        mock.patch(
            "vllm_torchtpu.model_loader_patches.evict_loaded_weights_page_cache"
        ),
        mock.patch.object(runner, "_initialize_pallas_kernels"),
        mock.patch.object(
            utils, "relayout_rope_tables", wraps=utils.relayout_rope_tables
        ) as mock_relayout,
    ):
        mock_loader.return_value.load_model.return_value = target_model
        runner.load_model()

        assert mock_relayout.call_count == 1
        mock_relayout.assert_called_once_with(target_model)
