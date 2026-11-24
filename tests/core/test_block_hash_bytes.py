# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""The byte block hasher names the same blocks as vLLM's pickled hasher."""

import random
from types import SimpleNamespace

import pytest
import torch
from vllm.lora.request import LoRARequest
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256, sha256_cbor
from vllm.v1.core import kv_cache_utils
from vllm.v1.request import Request

from vllm_torchtpu.core import block_hash_bytes

BLOCK = 16


@pytest.fixture(autouse=True)
def none_hash(monkeypatch):
    monkeypatch.setattr(kv_cache_utils, "NONE_HASH", None, raising=False)
    kv_cache_utils.init_none_hash(sha256)


def vllm_hasher(hash_fn=sha256):
    return kv_cache_utils.get_request_block_hasher(BLOCK, hash_fn)


def byte_hasher(hash_fn=sha256):
    return block_hash_bytes.get_request_block_hasher(BLOCK, hash_fn)


def make_request(token_ids, hasher, request_id="r", **kwargs):
    return Request(
        request_id,
        list(token_ids),
        SamplingParams(max_tokens=64),
        None,
        block_hasher=hasher,
        **kwargs,
    )


def prompts_with_shared_prefixes(seed=0, count=24):
    rng = random.Random(seed)
    trunks = [[rng.randrange(150_000) for _ in range(BLOCK * 6)] for _ in range(3)]
    prompts = []
    for _ in range(count):
        trunk = rng.choice(trunks)
        keep = rng.randrange(len(trunk) + 1)
        tail = [rng.randrange(150_000) for _ in range(rng.randrange(BLOCK * 5))]
        prompts.append(trunk[:keep] + tail)
    prompts.append(list(prompts[0]))
    same_block = list(range(BLOCK))
    prompts += [trunk[:BLOCK] + same_block for trunk in trunks]
    return prompts


def equality(hashes):
    flat = [h for request_hashes in hashes for h in request_hashes]
    return [[a == b for b in flat] for a in flat]


def test_same_blocks_match_as_with_vllm_hashes():
    prompts = prompts_with_shared_prefixes()
    vllm_hashes = [make_request(p, vllm_hasher()).block_hashes for p in prompts]
    byte_hashes = [make_request(p, byte_hasher()).block_hashes for p in prompts]

    assert [len(h) for h in byte_hashes] == [len(p) // BLOCK for p in prompts]
    assert equality(byte_hashes) == equality(vllm_hashes)
    assert sum(map(sum, equality(vllm_hashes))) > sum(map(len, vllm_hashes))
    assert not set(sum(byte_hashes, [])) & set(sum(vllm_hashes, []))


def test_decode_tokens_extend_the_chain_as_one_pass_does():
    prompt = prompts_with_shared_prefixes()[1]
    output = list(range(1000, 1000 + 3 * BLOCK + 5))
    request = make_request(prompt, byte_hasher())
    for token in output:
        request.append_output_token_ids(token)

    whole = make_request(prompt + output, byte_hasher())
    assert request.block_hashes == whole.block_hashes


def test_blocks_with_extra_keys_keep_vllm_hashes():
    prompt = prompts_with_shared_prefixes()[2] + list(range(BLOCK * 2))
    lora = LoRARequest("adapter", 1, "/adapters/a")
    for kwargs in ({"cache_salt": "salt"}, {"lora_request": lora}):
        ours = make_request(prompt, byte_hasher(), **kwargs).block_hashes
        theirs = make_request(prompt, vllm_hasher(), **kwargs).block_hashes
        assert ours[0] == theirs[0]
        if "lora_request" in kwargs:
            assert ours == theirs

    plain = make_request(prompt, byte_hasher()).block_hashes
    salted = make_request(prompt, byte_hasher(), cache_salt="salt").block_hashes
    assert not set(plain) & set(salted)


def images(layout):
    return [
        MultiModalFeatureSpec(
            data=None,
            modality="image",
            identifier=identifier,
            mm_position=PlaceholderRange(offset=offset, length=length),
        )
        for identifier, offset, length in layout
    ]


def test_image_blocks_match_as_with_vllm_hashes():
    """Including images that end past the prompt's last full block, so the
    first decode-time call picks the image index to start from; the twins
    differ only in the block that call hashes."""
    rng = random.Random(1)
    trunk = [rng.randrange(150_000) for _ in range(BLOCK * 8 + 5)]
    layouts = [
        [],
        [("a", BLOCK * 3 + 2, BLOCK)],
        [("b", BLOCK * 3 + 2, BLOCK)],
        [("a", BLOCK * 3 + 2, BLOCK), ("c", BLOCK * 6, BLOCK - 3)],
        [("a", 0, BLOCK * 2 + 5), ("b", BLOCK * 4, 3)],
        [("d", BLOCK * 8 - 2, 6)],
        [("d", BLOCK * 8 + 2, 3)],
        [("e", BLOCK * 8 + 2, 3)],
        [("a", BLOCK * 3 + 2, BLOCK), ("d", BLOCK * 8 + 2, 3)],
        [("a", BLOCK * 3 + 2, BLOCK), ("e", BLOCK * 8 + 2, 3)],
    ]
    output = [rng.randrange(150_000) for _ in range(BLOCK * 2 + 3)]

    def hashes(hasher):
        result = []
        for layout in layouts:
            request = make_request(trunk, hasher(), mm_features=images(layout))
            for token in output:
                request.append_output_token_ids(token)
            result.append(request.block_hashes)
        return result

    byte_hashes = hashes(byte_hasher)
    vllm_hashes = hashes(vllm_hasher)
    assert [len(h) for h in byte_hashes] == [len(h) for h in vllm_hashes]
    assert equality(byte_hashes) == equality(vllm_hashes)


def test_prompt_embeds_keep_vllm_hashes():
    prompt = [0] * (BLOCK * 3)
    embeds = torch.arange(len(prompt) * 4, dtype=torch.float32).reshape(-1, 4)
    assert (
        make_request(prompt, byte_hasher(), prompt_embeds=embeds).block_hashes
        == make_request(prompt, vllm_hasher(), prompt_embeds=embeds).block_hashes
    )


def test_token_ids_beyond_32_bits_keep_vllm_hashes():
    prompt = [2**32 + 7] + list(range(BLOCK * 2))
    assert (
        make_request(prompt, byte_hasher()).block_hashes
        == make_request(prompt, vllm_hasher()).block_hashes
    )


def test_other_hash_algorithms_keep_vllm_hasher():
    kv_cache_utils.init_none_hash(sha256_cbor)
    prompt = list(range(BLOCK * 3))
    assert (
        make_request(prompt, byte_hasher(sha256_cbor)).block_hashes
        == make_request(prompt, vllm_hasher(sha256_cbor)).block_hashes
    )


def test_hasher_arguments_read_from_vllm_closure():
    assert block_hash_bytes._hasher_arguments(vllm_hasher()) == (BLOCK, sha256)
    assert block_hash_bytes._hasher_arguments(lambda request: []) is None


@pytest.mark.parametrize(
    ("config", "leaves"),
    [
        (SimpleNamespace(kv_transfer_config=None, kv_events_config=None), False),
        (SimpleNamespace(kv_transfer_config=object(), kv_events_config=None), True),
        (
            SimpleNamespace(
                kv_transfer_config=None,
                kv_events_config=SimpleNamespace(enable_kv_cache_events=True),
            ),
            True,
        ),
    ],
)
def test_hashes_that_leave_the_process_keep_vllm_hasher(config, leaves):
    assert block_hash_bytes._hashes_leave_the_process(config) is leaves


@pytest.mark.parametrize(
    ("kv_transfer_config", "hash_fn", "byte_hashing"),
    [(None, sha256, True), (object(), sha256, False), (None, sha256_cbor, False)],
)
def test_install_gives_engine_core_the_byte_hasher(
    monkeypatch, kv_transfer_config, hash_fn, byte_hashing
):
    from vllm.v1.engine.core import EngineCore

    kv_cache_utils.init_none_hash(hash_fn)
    vllm_engine_hasher = vllm_hasher(hash_fn)

    def engine_core_init(self, vllm_config):
        self.request_block_hasher = vllm_engine_hasher

    monkeypatch.setattr(EngineCore, "__init__", engine_core_init)
    monkeypatch.setattr(EngineCore, "_tpu_block_hash_bytes_patch", False, raising=False)
    block_hash_bytes.install_byte_block_hasher()

    config = SimpleNamespace(
        kv_transfer_config=kv_transfer_config, kv_events_config=None
    )
    hasher = EngineCore(config).request_block_hasher
    assert (hasher is not vllm_engine_hasher) is byte_hashing
    prompt = prompts_with_shared_prefixes()[3]
    expected = byte_hasher() if byte_hashing else vllm_engine_hasher
    assert (
        make_request(prompt, hasher).block_hashes
        == make_request(prompt, expected).block_hashes
    )
