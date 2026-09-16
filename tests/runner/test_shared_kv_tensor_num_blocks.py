# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Block count of a KV tensor shared by attention and Mamba layers.

Numbers are Kimi-Linear-48B on tpu7x-8 (TP 8, 4 of 32 KDA heads per rank)
with fp8 KV and 256-token pages: an MLA page is 256 x 640 B = 163,840 B and
each KDA layer keeps 280,576 B of state per request (conv window
3 x 3 x 4 x 128 fp32 plus recurrent state 4 x 128 x 128 fp32). vLLM pads
every layer's page to 163,840 + 3 x 280,576 = 1,005,568 B and sizes the
shared tensor as num_blocks x that page.
"""
import dataclasses
from unittest.mock import patch

import pytest
import torch
from vllm.v1.kv_cache_interface import (FullAttentionSpec, MambaSpec,
                                        MLAAttentionSpec, SlidingWindowSpec)

from vllm_torchtpu.layers.adapter.attention import (PallasAttentionBackend,
                                                    PallasMLAttentionBackend)
from vllm_torchtpu.runner.kv_cache_manager import shared_kv_tensor_num_blocks

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 256
NUM_BLOCKS = 2167
MLA_PAGE = 256 * 640
KDA_SLOT = 280_576


def _mla_spec() -> MLAAttentionSpec:
    return MLAAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.float8_e4m3fn,
        page_size_padded=None,
    )


def _kda_spec() -> MambaSpec:
    return MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((3, 3, 4, 128), (4, 128, 128)),
        dtypes=(torch.float32, torch.float32),
        page_size_padded=None,
    )


def _unpadded(spec: MambaSpec) -> int:
    return dataclasses.replace(spec, page_size_padded=None).page_size_bytes


def test_page_sizes_match_the_serving_layout():
    assert PallasMLAttentionBackend.get_kv_cache_page_size_bytes(
        BLOCK_SIZE, 1, 576, torch.float8_e4m3fn) == MLA_PAGE
    assert _unpadded(_kda_spec()) == KDA_SLOT
    # The RPA layout of the same spec: two packed heads x four fp8 values
    # per word, four times the MLA page.
    assert PallasAttentionBackend.get_kv_cache_page_size_bytes(
        BLOCK_SIZE, 1, 576, torch.float8_e4m3fn) == 4 * MLA_PAGE


def test_mla_and_kda_tensor_holds_every_block_vllm_hands_out():
    padded_page = MLA_PAGE + 3 * KDA_SLOT
    num_blocks = shared_kv_tensor_num_blocks(
        NUM_BLOCKS * padded_page,
        [_mla_spec(), _kda_spec(),
         _kda_spec(), _kda_spec()],
        PallasAttentionBackend,  # the non-SSM backend; MLA specs ignore it
        NUM_BLOCKS)
    assert num_blocks == NUM_BLOCKS


def test_mla_spec_is_sized_by_the_mla_backend_whatever_backend_is_passed():
    padded_page = MLA_PAGE + 3 * KDA_SLOT
    specs = [_mla_spec(), _kda_spec(), _kda_spec(), _kda_spec()]
    for backend in (PallasAttentionBackend, PallasMLAttentionBackend, None):
        assert shared_kv_tensor_num_blocks(NUM_BLOCKS * padded_page, specs,
                                           backend, NUM_BLOCKS) == NUM_BLOCKS


def test_tensor_too_small_for_vllm_block_ids_is_rejected():
    # An MLA page counted at the RPA size would leave about a third of the
    # block ids without a page.
    padded_page = MLA_PAGE + 3 * KDA_SLOT
    too_small = NUM_BLOCKS * padded_page * 2 // 3
    with pytest.raises(ValueError, match="out of bounds"):
        shared_kv_tensor_num_blocks(
            too_small,
            [_mla_spec(), _kda_spec(),
             _kda_spec(), _kda_spec()], PallasAttentionBackend, NUM_BLOCKS)


def test_tensor_with_a_padding_layer_is_capped_at_vllm_count():
    # vLLM pads the last group to the same page even when it has fewer
    # Mamba layers; the divided count exceeds vLLM's and is capped.
    padded_page = MLA_PAGE + 3 * KDA_SLOT
    num_blocks = shared_kv_tensor_num_blocks(
        NUM_BLOCKS * padded_page,
        [_mla_spec(), _kda_spec(), _kda_spec()], PallasAttentionBackend,
        NUM_BLOCKS)
    assert num_blocks == NUM_BLOCKS


def test_rpa_and_kda_tensor_uses_the_rpa_page():
    attn = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.float8_e4m3fn,
        page_size_padded=None,
    )
    rpa_page = PallasAttentionBackend.get_kv_cache_page_size_bytes(
        BLOCK_SIZE, 8, 128, torch.float8_e4m3fn)
    padded_page = rpa_page + 3 * KDA_SLOT
    num_blocks = shared_kv_tensor_num_blocks(
        NUM_BLOCKS * padded_page,
        [attn, _kda_spec(), _kda_spec(),
         _kda_spec()], PallasAttentionBackend, NUM_BLOCKS)
    assert num_blocks == NUM_BLOCKS


def test_attention_only_tensor_keeps_the_divided_count():
    # Sliding-window and full-attention layers sharing one tensor: vLLM
    # pads to one layer's page, the runner divides by all six. The count is
    # returned as is and the shortfall is logged, not raised.
    full = FullAttentionSpec(block_size=BLOCK_SIZE,
                             num_kv_heads=8,
                             head_size=128,
                             dtype=torch.float8_e4m3fn,
                             page_size_padded=None)
    window = SlidingWindowSpec(block_size=BLOCK_SIZE,
                               num_kv_heads=8,
                               head_size=128,
                               sliding_window=1024,
                               dtype=torch.float8_e4m3fn,
                               page_size_padded=None)
    page = PallasAttentionBackend.get_kv_cache_page_size_bytes(
        BLOCK_SIZE, 8, 128, torch.float8_e4m3fn)
    specs = [full, window, window, window, window, window]
    with patch("vllm_torchtpu.runner.kv_cache_manager.logger") as mock_logger:
        num_blocks = shared_kv_tensor_num_blocks(NUM_BLOCKS * page, specs,
                                                 PallasAttentionBackend,
                                                 NUM_BLOCKS)
    assert num_blocks == NUM_BLOCKS // 6
    mock_logger.warning.assert_called_once()
    assert "attention-only shared KV tensor" in (
        mock_logger.warning.call_args[0][0])
