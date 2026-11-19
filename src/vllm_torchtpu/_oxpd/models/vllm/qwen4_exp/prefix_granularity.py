# SPDX-License-Identifier: Apache-2.0
"""Fine prefix-hit granularity for Qwen4Exp (``--prefix-match-unit``).

A Qwen4Exp pool block is one GDN state page, 3,200 tokens at TP1
(``pool_layout.py``), so with the default hash unit every prefix hit lands on
a 3,200-token boundary. On the served AgentX trace that costs about 1.8% of
all prompt tokens, i.e. 16-22% of the tokens a warm turn computes (CACHE lane,
``reports/cache_hits.md``).

The serving image's vLLM already implements finer hits for hybrid models
when ``cache_config.prefix_match_unit`` is set below the block size:
request hashes every ``unit`` tokens, a fine right-to-left Mamba lookup, a
partial-tail entry for the prompt's last ``unit`` boundary (the scheduler
ends a chunk there so the state is materialised), and copy-on-write of the
shared partial block, which the TPU runner applies page-locally in the
unified pool (``TPUModelRunner._apply_kv_cache_block_copies``). Nothing in
the Qwen4Exp kernels assumes a block-aligned resume: a resume point only has
to be a whole QSA kernel page (128 rows), a whole indexer compression group
(``indexer_compress_ratio`` tokens, so the per-request raw ring is empty
there) and a whole GDN tile (64 tokens).

``validate_prefix_match_unit`` admits exactly those units and refuses the
combinations nobody has validated (MTP on the same engine, context
parallelism). It only checks; it never sets the unit.
"""

QSA_KERNEL_ROWS = 128   # qsa.QSA_BLOCK_SIZE: main-KV kernel page rows
GDN_TILE = 64           # kernels/gdn/v3/wrapper.py mixed_tile_size default


def prefix_match_unit_problems(
    unit: int | None,
    *,
    enable_prefix_caching: bool,
    mamba_cache_mode: str,
    pooled: bool,
    compress_ratio: int,
    num_speculative_tokens: int,
    dcp: int,
    pcp: int,
    f2_producer: bool = False,
) -> list[str]:
    """Why ``unit`` cannot be used on this engine (empty list: it can).

    ``f2_producer`` (q38 CACHE-3): the engine is q38 F2's prefill-only MTP
    producer (config.pd_prefill_mtp_producer: mtp, K = 1, never verifies) and
    VLLM_TPU_PREFIX_UNIT_ALLOW_F2_PRODUCER=1 is set. Its MTP draft caches are
    non-prefix-cacheable (config.native_cache_specs, F2 option (a)), so a
    partial-tail hit never copies or reuses draft pages; the draft rows of a
    hit are F2 gap rows, as with block-aligned hits. MTP is then admitted with
    K = 1 only; every other rule still applies."""
    if unit is None:
        return []
    problems = []
    if unit <= 0:
        return [f"prefix_match_unit must be positive, got {unit}"]
    if not enable_prefix_caching:
        problems.append("needs --enable-prefix-caching")
    if mamba_cache_mode != "align":
        problems.append(f"needs --mamba-cache-mode align, got {mamba_cache_mode!r}")
    if not pooled:
        problems.append("needs the Qwen4Exp unified pool")
    for name, step in (
        ("a QSA kernel page", QSA_KERNEL_ROWS),
        ("an indexer compression group", compress_ratio),
        ("a GDN tile", GDN_TILE),
    ):
        if step > 0 and unit % step:
            problems.append(f"must be a whole {name} ({step} tokens), got {unit}")
    if num_speculative_tokens and not (f2_producer and num_speculative_tokens == 1):
        problems.append("is not validated together with MTP on the same engine")
    if dcp > 1 or pcp > 1:
        problems.append(f"is not validated with context parallelism (dcp={dcp}, pcp={pcp})")
    return problems


def validate_prefix_match_unit(cfg, pooled: bool, num_speculative_tokens: int,
                               f2_producer: bool = False) -> None:
    c, p = cfg.cache_config, cfg.parallel_config
    unit = getattr(c, "prefix_match_unit", None)
    problems = prefix_match_unit_problems(
        unit,
        enable_prefix_caching=bool(c.enable_prefix_caching),
        mamba_cache_mode=str(getattr(c, "mamba_cache_mode", "none")),
        pooled=pooled,
        compress_ratio=int(getattr(cfg.model_config.hf_text_config, "indexer_compress_ratio", 0) or 0),
        num_speculative_tokens=num_speculative_tokens,
        dcp=int(getattr(p, "decode_context_parallel_size", 1) or 1),
        pcp=int(getattr(p, "prefill_context_parallel_size", 1) or 1),
        f2_producer=f2_producer,
    )
    if problems:
        raise ValueError("Qwen4Exp prefix_match_unit=%s: %s" % (unit, "; ".join(problems)))
