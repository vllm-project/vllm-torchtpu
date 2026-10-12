# SPDX-License-Identifier: Apache-2.0
"""Freeze DSv4.1 token IDs from LongBench code documents, including decode C128.

Design reference: sgl-jax-glm52-dsa/benchmark/glm52/bench_dsa_cache_hit.py.
Uses this model's tokenizer, independent prefix windows and a content hash;
never imports GLM token IDs or generates random speculative-decode prompts.
"""

import argparse
import hashlib
import json
from pathlib import Path

DATA_SHA256 = "15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    digest = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    if digest != DATA_SHA256:
        raise ValueError(f"Unexpected LongBench dataset: {digest}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    branches = [
        tokenizer.encode(s, add_special_tokens=False) for s in (" the", " a", " and")
    ]
    branches = [s[0] for s in branches if len(s) == 1]
    if len(set(branches)) < 2:
        raise ValueError("Need two distinct single-token prime suffixes")
    documents = sorted(json.loads(args.dataset.read_text()), key=lambda d: d["_id"])
    rows, sources, suffixes, hashes = [], [], [], set()
    for doc in documents:
        if doc.get("domain") != "Code Repository Understanding":
            continue
        ids = tokenizer.encode(doc["context"][:1000000], add_special_tokens=False)
        # Bounded, deterministic windows retain real code continuation statistics.
        for offset in range(0, min(max(0, len(ids) - 81920 + 1), 16 * 256), 256):
            row = ids[offset : offset + 81920]
            key = hashlib.sha256(json.dumps(row[:65536]).encode()).hexdigest()
            if key in hashes:
                continue
            hashes.add(key)
            rows.append(row)
            sources.append(
                {"document": doc["_id"], "offset": offset, "prefix_sha256": key}
            )
            branch = next(t for t in branches if t != row[65536])
            suffixes.append([branch, branch])
            if len(rows) == 128:
                break
        if len(rows) == 128:
            break
    if len(rows) != 128:
        raise ValueError(f"Need 128 distinct long-context windows; found {len(rows)}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(
            {
                "prompts": rows,
                "prime_suffix": suffixes,
                "manifest": {
                    "dataset_sha256": digest,
                    "tokenizer": args.tokenizer,
                    "sources": sources,
                },
            },
            stream,
        )


if __name__ == "__main__":
    main()
