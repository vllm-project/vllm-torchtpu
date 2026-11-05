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
"""Kimi-K3 fixtures for the standalone tools: stand-in unified pools laid out
as vLLM's, their KV cache groups, the fixed prompt."""

from __future__ import annotations

import dataclasses
import glob
import hashlib
import json
from pathlib import Path

import numpy as np

PAGE, DCP = 256, 32
LOGICAL = PAGE * DCP  # tokens per vLLM DCP block
CACHE_DTYPE = "auto"


def configure_cache(dtype: str) -> None:
    """Configure standalone fixtures from the same cache option as serving."""
    from vllm_torchtpu.megakernel.kimi_k3.layout import cache_layout

    global PAGE, LOGICAL, CACHE_DTYPE
    PAGE = cache_layout(dtype).tokens_per_block
    LOGICAL = PAGE * DCP
    CACHE_DTYPE = dtype


DSA = None  # (DsaConfig, (synthetic mode, seed)) from configure_dsa


def configure_dsa(
    path: str, mode: str | None, seed: int, topk: int | None = None
) -> None:
    """Decode Kimi-K3.1 (the DSA config at ``path``) with a synthetic indexer;
    ``topk`` replaces the config's ``index_topk``."""
    global DSA
    if not path:
        if mode is not None:
            raise SystemExit("--synthetic-indexer needs --dsa-config")
        if topk is not None:
            raise SystemExit("--dsa-topk needs --dsa-config")
        return
    from vllm_torchtpu.megakernel.kimi_k3.dsa import load_dsa_config

    config = load_dsa_config(path)
    if config is None:
        raise SystemExit("--dsa-config needs a config with a dsa_config")
    if topk is not None:
        config = dataclasses.replace(config, index_topk=topk)
    DSA = (config, None if mode is None else (mode, seed))


def indexer_mode() -> str | None:
    """The synthetic indexer's mode, or None (Kimi-K3, or the pre-shard's
    own indexer weights)."""
    return None if DSA is None or DSA[1] is None else DSA[1][0]


def decode_ops_options() -> dict:
    """``DecodeOps`` keywords for the configured model (none for Kimi-K3)."""
    return {} if DSA is None else {"dsa": DSA[0], "synthetic_indexer": DSA[1]}


def logical() -> int:
    """Tokens per DCP block under the installed kernel config's DCP group."""
    from vllm_torchtpu.megakernel.kimi_k3 import config

    return PAGE * config.active().dcp_group


def fixed_prompt(count: int, checkpoint: str):
    """``(tokens, tokenizer)``: the first ``count`` tokens of a fixed text, the
    model repository's own Python sources (``checkpoint``: the Kimi-K3
    checkpoint directory)."""
    from vllm_torchtpu.megakernel.kimi_k3.program import kernel_modules

    (load,) = kernel_modules("load")
    tokenizer = load.load_tokenizer(checkpoint)
    text = "".join(
        Path(path).read_text() for path in sorted(glob.glob(f"{checkpoint}/*.py"))
    )
    return tokenizer.encode(text, disallowed_special=())[:count], tokenizer


def synthetic_index_keys(
    num_blocks: int, tokens_per_block: int, rank: int, layers: int = 24, seed: int = 0
) -> np.ndarray:
    """``[num_blocks, layers, tokens_per_block + 32, 128]`` uint8: Kimi-K3.1
    index pages of real-valued synthetic keys that differ per rank, block,
    layer and token (repeated keys would make the selection a tie), encoded
    with ``dsa_cache.quantize_index_keys`` and laid out as the ``dsa_cache``
    module docstring says. A rank draws one slab of keys from a generator
    seeded by ``(seed, rank)``: each is BF16 N(0, 1) times ``2^k`` per token
    (``k`` uniform in [-4, 4]). Layer ``l`` holds the slab's tokens (a key's
    bytes and its exponent together) in the order of a permutation seeded by
    ``(seed, rank, l + 1)`` over all the blocks' tokens, so a layer has no
    repeated key and each layer orders the keys differently. The host CPU
    encodes them: a worker's TPU devices belong to the torch_tpu runtime."""
    import jax
    import jax.numpy as jnp
    import ml_dtypes

    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_cache

    rows = tokens_per_block // 128
    out = np.zeros(
        (num_blocks, layers, *dsa_cache.index_block_shape(tokens_per_block)), np.uint8
    )
    shape = (num_blocks, tokens_per_block, 128)
    rng = np.random.default_rng((seed, rank))
    keys = rng.standard_normal(shape, np.float32) * np.exp2(
        rng.integers(-4, 5, (*shape[:2], 1))
    ).astype(np.float32)
    with jax.default_device(jax.devices("cpu")[0]):
        key_bytes, exponent = dsa_cache.quantize_index_keys(
            jnp.asarray(keys.astype(ml_dtypes.bfloat16))
        )
    key_bytes = np.asarray(key_bytes).reshape(-1, 128)
    exponent = np.asarray(exponent).reshape(-1)
    for layer in range(layers):
        # numpy pads seeds with zeros: (seed, rank, 0) would repeat the slab's.
        order = np.random.default_rng((seed, rank, layer + 1)).permutation(
            len(exponent)
        )
        out[:, layer, :tokens_per_block] = key_bytes[order].reshape(shape)
        out[:, layer, tokens_per_block : tokens_per_block + rows] = exponent[
            order
        ].reshape(num_blocks, rows, 128)
    return out


class Pools:
    """24 unified pools laid out per ``cache_layout(CACHE_DTYPE)`` (one MLA
    layer and three KDA layers each, as vLLM allocates them) and the KV cache
    groups: the MLA layers' (``logical()``-token blocks, ``mla_width`` entries a
    row) and three KDA groups (a state column per ``PAGE`` tokens,
    ``state_width`` entries), KDA layer k in group k % 3. Block 0 is the null
    block. ``draft_layers``: that many more pools, the MLA DSpark draft's
    layers (after the target's, in the MLA group). With Kimi-K3.1, the
    index-key cache: zeros, or with ``pattern`` seeded keys
    (:func:`synthetic_index_keys`)."""

    def __init__(
        self,
        num_blocks: int,
        mla_width: int,
        state_width: int,
        pattern: bool = False,
        draft_layers: int = 0,
    ) -> None:
        import torch

        from vllm_torchtpu.megakernel import program as mk
        from vllm_torchtpu.megakernel.kimi_k3 import layout

        cache = layout.cache_layout(CACHE_DTYPE)
        dtype = getattr(torch, cache.dtype)
        if pattern:  # synthetic latents: a random bf16 pattern (the rope lanes zero)
            # One seed on every rank: every page holds the same tokens, whatever
            # the DCP layout that places it.
            generator = torch.Generator().manual_seed(0)
            page = (
                torch.randn((1, 128, 2, 640), dtype=torch.float32, generator=generator)
                * 0.5
            )
            page[..., 576:] = 0
            page = page.to(torch.bfloat16)
            # Keep the same BF16 fixture across page geometries: a 512-token
            # FP8 page repeats the 256-token tile.
            page = (
                page.reshape(1, 256, 640)
                .repeat(1, cache.tokens_per_block // 256, 1)
                .reshape(1, *cache.block_shape)
                .to(dtype)
            )
            self.tensors = [
                page.to("tpu").repeat(num_blocks, 1, 1, 1) for _ in range(24)
            ]
        else:
            self.tensors = [
                torch.zeros((num_blocks, *cache.block_shape), dtype=dtype).to("tpu")
                for _ in range(24)
            ]
        self.tensors += [
            torch.zeros((num_blocks, *cache.block_shape), dtype=dtype).to("tpu")
            for _ in range(draft_layers)
        ]
        mla_names = [
            f"model.layers.{layer}.self_attn.mla_attn.attn" for layer in layout.MLA
        ]
        kda_names = [f"model.layers.{layer}.self_attn" for layer in layout.KDA]
        first_draft = len(layout.MLA) + len(layout.KDA)
        mla_names += [
            f"model.layers.{first_draft + d}.self_attn.mla_attn.attn"
            for d in range(draft_layers)
        ]
        self.caches = {name: self.tensors[m] for m, name in enumerate(mla_names)}
        self.caches.update(
            {name: self.tensors[k // 3] for k, name in enumerate(kda_names)}
        )
        # Kimi-K3.1's index-key cache: one block-major tensor (block b's page of
        # MLA layer m at [b, m]) under every MLA layer's index-cache name,
        # addressed by the MLA layers' block tables (the group lists only the
        # latent names).
        self.index_tensor = None
        if DSA is not None:
            from vllm_torchtpu.megakernel.kimi_k3.kernels.dsa_cache import (
                index_block_shape,
            )
            from vllm_torchtpu.megakernel.kimi_k3.pools import INDEX_CACHE

            shape = (
                num_blocks,
                len(layout.MLA),
                *index_block_shape(cache.tokens_per_block),
            )
            if pattern:  # seeded keys that differ per rank, block, layer and token
                from vllm_torchtpu.megakernel import bridge

                rank = bridge.kernel_mesh("tp", DCP)[1]
                keys = synthetic_index_keys(num_blocks, cache.tokens_per_block, rank)
                self.index_tensor = torch.from_numpy(keys).to("tpu")
            else:
                self.index_tensor = torch.zeros(shape, dtype=torch.uint8).to("tpu")
            self.caches.update(
                dict.fromkeys(
                    (f"model.layers.{layer}{INDEX_CACHE}" for layer in layout.MLA),
                    self.index_tensor,
                )
            )
        self.groups = [mk.KVGroup(tuple(mla_names), logical(), mla_width)]
        self.groups += [
            mk.KVGroup(
                tuple(n for k, n in enumerate(kda_names) if k % 3 == g),
                PAGE,
                state_width,
            )
            for g in range(3)
        ]
        self.num_blocks = num_blocks
        self.mla_width, self.state_width = mla_width, state_width
        self._next = 1

    def scheduled(self, table, index: int) -> np.ndarray:
        """MLA ``table`` as the decode scheduler issues it to the ``index``-th
        request admitted: under attention DP that request is in DP group
        ``g = index % dp`` (fewest requests first) and its pages are ids
        ``g * num_blocks + row`` (``megakernel.dp_blocks``); the null block
        stays shared."""
        from vllm_torchtpu.megakernel.kimi_k3 import config

        group = index % config.active().attention_dp
        table = np.asarray(table, np.int32)
        return np.where(table > 0, table + group * self.num_blocks, 0).astype(np.int32)

    def fresh(self, n: int) -> list[int]:
        out = list(range(self._next, self._next + n))
        self._next += n
        return out

    def tables(self, mla: list[int], kda: list[list[int]]) -> tuple[np.ndarray, ...]:
        def padded(table, width):
            return np.asarray(list(table) + [0] * (width - len(table)), np.int32)

        return (padded(mla, self.mla_width),) + tuple(
            padded(t, self.state_width) for t in kda
        )


def engine_config(
    role: str,
    *,
    max_num_seqs: int,
    max_model_len: int,
    max_num_batched_tokens: int,
    num_speculative_tokens: int = 0,
    num_speculative_tokens_per_batch_size: tuple[tuple[int, int, int], ...]
    | None = None,
):
    """The serving configuration's megakernel view, for the TP32 / DCP32
    deployment (DCP over each group under the installed kernel config's
    attention DP) with the configured cache (``configure_cache``)."""
    from vllm_torchtpu.megakernel.config import MegakernelEngineConfig
    from vllm_torchtpu.megakernel.kimi_k3 import config

    dp = config.active().attention_dp
    return MegakernelEngineConfig(
        role=role,
        tensor_parallel_size=32,
        decode_context_parallel_size=DCP // dp,
        block_size=PAGE,
        kv_cache_dtype=CACHE_DTYPE,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        num_speculative_tokens=num_speculative_tokens,
        num_speculative_tokens_per_batch_size=num_speculative_tokens_per_batch_size,
        attention_dp=dp,
        dsa=None if DSA is None else DSA[0],
        synthetic_indexer=None if DSA is None else DSA[1],
        presharded_indexer=DSA is not None and DSA[1] is None,
    )


def decoder(
    ops,
    pools: Pools,
    *,
    max_num_seqs: int,
    max_model_len: int,
    draft=None,
    schedule: tuple[tuple[int, int, int], ...] | None = None,
):
    """A :class:`KimiDecoder` over shared ``ops`` (and ``draft``) and the
    fixture ``pools``, bound and ready to warm. ``schedule``: vLLM's
    per-batch-size draft count (None: plain steps only). Pages are owned by
    mesh position, as the fixtures lay them out."""
    from vllm_torchtpu.megakernel.kimi_k3.decode import KimiDecoder

    config = engine_config(
        "decode",
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        max_num_batched_tokens=512,
        num_speculative_tokens=7 if schedule else 0,
        num_speculative_tokens_per_batch_size=schedule,
    )
    decoder = KimiDecoder(config, ops, draft if schedule else None, tp_rank=ops.rank)
    decoder.bind_kv_caches(pools.caches, pools.groups)
    return decoder


def request(
    req_id: str,
    tokens,
    table: np.ndarray,
    sampling: tuple[float, float, int, int] = (0.0, 1.0, 0, 0),
    computed: int = 0,
    kda: tuple[int, int, int] | None = None,
):
    """A decode step's request scheduling ``tokens`` after ``computed`` ones,
    MLA pages ``table``. Its KDA tables hold only the null block, so a
    decoder that has not seen it starts it from a clean state (no import);
    with ``kda`` each KDA group's table is that one state block, which the
    decoder imports the state after ``computed`` tokens from."""
    from vllm_torchtpu.megakernel import program as mk

    tokens = np.asarray(tokens, np.int32)
    end = computed + len(tokens)
    token_ids = np.zeros(end, np.int32)
    token_ids[computed:] = tokens
    states = [np.asarray([block], np.int32) for block in (kda or (0, 0, 0))]
    return mk.RequestInput(
        req_id=req_id,
        computed=computed,
        scheduled=len(tokens),
        token_ids=token_ids,
        num_tokens=end,
        num_prompt_tokens=end,
        block_tables=(np.asarray(table, np.int32), *states),
        sampling=mk.SamplingSpec(*sampling),
    )


# Bump when the bench's real-text rows (prompt offsets, request layout) change.
INGEST_FORMAT = 1


class IngestCache:
    """A real-text decode case's ingested state on network storage (``bench
    decode --token-file --ingest-cache``). After the case's first step (each
    row's prompt replayed and its first token decoded) every row's KDA state
    goes into its own three state blocks, the decoder's export as a P/D
    transfer leaves it, and each rank saves its pools, its index-key cache and
    the rows' first tokens; a later run with the same key loads them in place
    and resumes each row at its context: the decoder imports the state from
    those blocks and decodes the first token, which is the fresh run's second
    step. The key covers everything that shapes the bytes: the case and its
    rows, the corpus, the pre-shard, the layout and the decode code.

    ``scope="base"`` leaves the kernel config and the decode code out of the
    key, keeping only what places the bytes (the DCP group and tables), so
    flag arms of one tree reuse one ingest. It is valid only for arms whose
    ingested state is bit-identical to the run that saved it (flags that do
    not change the replay's numerics); otherwise the arm decodes from another
    kernel's state."""

    def __init__(self, root: str, parts: dict, scope: str = "full") -> None:
        from vllm_torchtpu.megakernel import bridge
        from vllm_torchtpu.megakernel.export_cache import _source_fingerprint
        from vllm_torchtpu.megakernel.kimi_k3 import config

        if scope not in ("full", "base"):
            raise ValueError(f"ingest key scope is full or base, not {scope}")
        key = {
            **parts,
            "format": INGEST_FORMAT,
            "cache_dtype": CACHE_DTYPE,
            "dsa": None if DSA is None else repr(DSA),
        }
        if scope == "full":
            key.update(source=_source_fingerprint(), kernel=config.active().key())
        else:
            active = config.active()
            key.update(
                scope=scope,
                placement=[active.dcp_group, active.dcp_tables, active.attention_dp],
            )
        digest = hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode())
        self.key, self.scope = key, scope
        self.root = Path(root) / digest.hexdigest()[:24]
        self.rank = bridge.kernel_mesh("tp", DCP)[1]

    def _rank_dir(self, tag: str) -> Path:
        return self.root / tag / f"rank{self.rank:02d}"

    def ready(self, tag: str) -> bool:
        """Whether every rank holds a complete save of ``tag`` (a collective)."""
        import torch

        mine = (self._rank_dir(tag) / "done").is_file()
        every = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(every, mine)
        return all(every)

    @staticmethod
    def _tensors(pools: Pools) -> list:
        return [
            *pools.tensors,
            *([pools.index_tensor] if pools.index_tensor is not None else []),
        ]

    @staticmethod
    def _bytes_view(tensor):
        """``tensor`` as same-sized integers (numpy has no FP8 or BF16)."""
        import torch

        return tensor.view(
            {1: torch.uint8, 2: torch.int16, 4: torch.int32}[tensor.element_size()]
        )

    def save(self, tag: str, pools: Pools, first_tokens: np.ndarray) -> None:
        """This rank's pools, index cache and the rows' first tokens; the
        ``done`` marker last."""
        folder = self._rank_dir(tag)
        folder.mkdir(parents=True, exist_ok=True)
        for index, tensor in enumerate(self._tensors(pools)):
            self._bytes_view(tensor).cpu().numpy().tofile(
                folder / f"tensor{index:02d}.bin"
            )
        np.save(folder / "first_tokens.npy", np.asarray(first_tokens, np.int32))
        if self.rank == 0:
            (self.root / "key.json").write_text(
                json.dumps(self.key, indent=1, default=str) + "\n"
            )
        (folder / "done").write_text("")

    def load(self, tag: str, pools: Pools) -> np.ndarray:
        """Replace this rank's pools and index cache with the save, one tensor
        at a time (each keeps its identity: the decoder, its importer and the
        ops hold them); returns the rows' first tokens."""
        import torch

        from vllm_torchtpu.megakernel.kimi_k3.kernels.rank_loader import read_into

        folder = self._rank_dir(tag)
        for index, tensor in enumerate(self._tensors(pools)):
            dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[tensor.element_size()]
            host = read_into(
                folder / f"tensor{index:02d}.bin",
                0,
                np.empty(tuple(tensor.shape), dtype),
            )
            tensor.set_(torch.from_numpy(host).view(tensor.dtype).to(tensor.device))
        return np.load(folder / "first_tokens.npy")
