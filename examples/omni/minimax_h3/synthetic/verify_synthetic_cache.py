#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify a synthetic MiniMax-H3 cache produced by gen_minimax_h3_cache.py.

Checks (each prints PASS/FAIL; exit status 1 if any check fails):
  layout       refs/main == sha1(revision seed); snapshot exists; every
               snapshot file is a relative symlink into ../blobs; blob names
               are sha256 of their content
  stamp        every *.safetensors header carries
               {"h3_synthetic": "apache2-random-v1"}
  manifest     the set of snapshot files == SYNTHETIC_MANIFEST.json
               (+ the manifest itself); sizes/sha match
  index        every model.safetensors.index.json maps exactly the tensors
               stored in its shards
  origin       no file names a diffusers/transformers value source or an HF
               repo origin
  provenance   SYNTHETIC_PROVENANCE.json covers every key of every JSON
               config file, and each source is a vllm-omni citation or
               SYNTHETIC
  readonly     no write permission bits on snapshot files/dirs and blobs
Use --snapshot-dir for a plain directory build; the layout and readonly
checks are skipped then.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path
from typing import Any

REPO_FOLDER = "models--MiniMaxAI--MiniMax-H3"
REVISION_SEED = "minimax-h3-vllm-omni-only-synthetic-v2"
REVISION = hashlib.sha1(REVISION_SEED.encode("utf-8")).hexdigest()
STAMP_KEY = "h3_synthetic"
STAMP_VALUE = "apache2-random-v1"
SYNTHETIC_INDEX_VERSION = "0.0.0+h3.synthetic"
MANIFEST_NAME = "SYNTHETIC_MANIFEST.json"
PROVENANCE_NAME = "SYNTHETIC_PROVENANCE.json"
# Files whose provenance is one entry for the whole file (generated, not
# key/value configs).
FILE_LEVEL_SUFFIXES = ("tokenizer.json", "vocab.json", "model.safetensors.index.json")
ORIGIN_MARKERS = (
    "huggingface.co",
    '"_name_or_path"',
    "Qwen/",
    "MiniMaxAI/",
    "hf_cache/hub/models--",
    "transformers_version",
)

RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(
        f"{'PASS' if ok else 'FAIL'} {name}" + (f": {detail}" if detail else ""),
        flush=True,
    )


def snapshot_files(root: Path) -> list[str]:
    out = []
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            out.append(os.path.relpath(os.path.join(dirpath, name), root))
    return sorted(out)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


def safetensors_header(path: Path) -> dict[str, Any]:
    with open(path, "rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        return json.loads(handle.read(length))


def check_layout(repo: Path, quick: bool) -> Path | None:
    ref = repo / "refs" / "main"
    if not ref.is_file():
        record("layout.refs_main", False, f"missing {ref}")
        return None
    revision = ref.read_text().strip()
    record(
        "layout.refs_main",
        revision == REVISION,
        f"refs/main={revision} expected={REVISION}",
    )
    snap = repo / "snapshots" / revision
    if not snap.is_dir():
        record("layout.snapshot_dir", False, f"missing {snap}")
        return None
    record("layout.snapshot_dir", True, str(snap))
    blobs = (repo / "blobs").resolve()
    bad_links, bad_hash = [], []
    referenced = set()
    for rel in snapshot_files(snap):
        path = snap / rel
        target = os.readlink(path) if path.is_symlink() else None
        if target is None or os.path.isabs(target) or path.resolve().parent != blobs:
            bad_links.append(rel)
            continue
        referenced.add(path.resolve().name)
    record(
        "layout.relative_symlinks_into_blobs",
        not bad_links,
        f"bad={bad_links[:5]}" if bad_links else "",
    )
    for blob in sorted((repo / "blobs").iterdir()):
        if blob.name.startswith(".h3tmp-"):
            bad_hash.append(f"leftover temp {blob.name}")
            continue
        if not quick and sha256_file(blob) != blob.name:
            bad_hash.append(blob.name)
    record(
        "layout.blob_names_are_sha256",
        not bad_hash,
        ("skipped hashing (--quick); " if quick else "")
        + (f"bad={bad_hash[:5]}" if bad_hash else ""),
    )
    orphans = sorted({b.name for b in (repo / "blobs").iterdir()} - referenced)
    record(
        "layout.no_orphan_blobs",
        not orphans,
        f"orphans={orphans[:5]}" if orphans else "",
    )
    return snap


def check_stamp(snap: Path) -> None:
    bad = []
    count = 0
    for rel in snapshot_files(snap):
        if rel.endswith(".safetensors"):
            count += 1
            meta = safetensors_header(snap / rel).get("__metadata__") or {}
            if meta.get(STAMP_KEY) != STAMP_VALUE:
                bad.append(rel)
    record(
        "stamp.safetensors_metadata",
        count > 0 and not bad,
        f"{count} weight files; unstamped={bad[:5]}",
    )


def check_manifest(snap: Path, quick: bool) -> dict[str, Any]:
    path = snap / MANIFEST_NAME
    if not path.is_file():
        record("manifest.present", False)
        return {}
    manifest = json.loads(path.read_text())
    files = manifest.get("files", {})
    actual = set(snapshot_files(snap)) - {MANIFEST_NAME}
    extra, missing = sorted(actual - set(files)), sorted(set(files) - actual)
    record(
        "manifest.covers_every_file",
        not extra and not missing,
        f"{len(actual)} files; extra={extra[:5]} missing={missing[:5]}",
    )
    bad = []
    for rel, entry in files.items():
        full = snap / rel
        if not full.exists():
            continue
        if full.stat().st_size != entry["size"]:
            bad.append(f"{rel}:size")
        elif full.is_symlink():
            if full.resolve().name != entry["sha256"]:
                bad.append(f"{rel}:blob")
        elif not quick and sha256_file(full) != entry["sha256"]:
            bad.append(f"{rel}:sha")
    record("manifest.sizes_and_hashes", not bad, f"bad={bad[:5]}" if bad else "")
    return manifest


def check_index(snap: Path) -> None:
    bad = []
    for rel in snapshot_files(snap):
        if not rel.endswith("model.safetensors.index.json"):
            continue
        index = json.loads((snap / rel).read_text())
        weight_map = index["weight_map"]
        base = (snap / rel).parent
        stored: dict[str, str] = {}
        total = 0
        for shard in sorted(set(weight_map.values())):
            header = safetensors_header(base / shard)
            for name, info in header.items():
                if name == "__metadata__":
                    continue
                stored[name] = shard
                total += info["data_offsets"][1] - info["data_offsets"][0]
        if stored != weight_map:
            bad.append(
                f"{rel}: map mismatch ({len(stored)} stored vs "
                f"{len(weight_map)} mapped)"
            )
        if index["metadata"].get("total_size") != total:
            bad.append(
                f"{rel}: total_size {index['metadata'].get('total_size')} != {total}"
            )
    record("index.weight_maps_match_shards", not bad, "; ".join(bad[:3]))


def _walk_json(obj: Any, prefix: str = ""):
    if isinstance(obj, dict):
        if not obj and prefix:
            yield prefix
        for key, value in obj.items():
            yield from _walk_json(value, f"{prefix}.{key}" if prefix else str(key))
    else:
        yield prefix


def _covered(path: str, prov: dict[str, str]) -> bool:
    parts = path.split(".")
    return any(".".join(parts[:i]) in prov for i in range(1, len(parts) + 1))


def check_origin(snap: Path) -> None:
    bad = []
    for rel in snapshot_files(snap):
        if rel.endswith(".safetensors"):
            meta = safetensors_header(snap / rel).get("__metadata__") or {}
            text = json.dumps(meta)
        else:
            text = (snap / rel).read_text(encoding="utf-8")
        for marker in ORIGIN_MARKERS:
            if marker in text:
                bad.append(f"{rel}: contains {marker!r}")
        if "diffusers" in text:
            if rel == PROVENANCE_NAME:
                prov = json.loads(text)
                strings = [s for s in _strings(prov) if "diffusers" in s]
                offenders = [
                    s
                    for s in strings
                    if not s.startswith("vllm-omni vllm_omni/")
                    and s.split(".")[-1] != "_diffusers_version"
                ]
                if offenders:
                    bad.append(
                        f"{rel}: 'diffusers' outside vllm-omni citations: "
                        f"{offenders[:2]}"
                    )
            elif rel.endswith(".json"):
                obj = json.loads(text)
                residue = json.dumps(obj).replace('"_diffusers_version"', "")
                if (
                    "diffusers" in residue
                    or obj.get("_diffusers_version", SYNTHETIC_INDEX_VERSION)
                    != SYNTHETIC_INDEX_VERSION
                ):
                    bad.append(f"{rel}: 'diffusers' used as a value source")
            else:
                bad.append(f"{rel}: mentions 'diffusers'")
        if rel.endswith(".py") and (
            "import transformers" in text or "from transformers" in text
        ):
            bad.append(f"{rel}: remote code imports transformers")
    record("origin.no_upstream_value_sources", not bad, "; ".join(bad[:5]))


def _strings(obj: Any):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield str(key)
            yield from _strings(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _strings(value)
    elif isinstance(obj, str):
        yield obj


def check_provenance(snap: Path) -> None:
    path = snap / PROVENANCE_NAME
    if not path.is_file():
        record("provenance.present", False)
        return
    prov = json.loads(path.read_text())["files"]
    missing = []
    for rel in snapshot_files(snap):
        if rel in (MANIFEST_NAME, PROVENANCE_NAME) or rel.endswith(
            (".safetensors", ".py", ".txt")
        ):
            if rel.endswith(".txt") and rel not in prov:
                missing.append(rel)
            continue
        entry = prov.get(rel)
        if entry is None:
            missing.append(rel)
            continue
        if "__file__" in entry:
            continue
        if rel.endswith(FILE_LEVEL_SUFFIXES):
            missing.append(f"{rel} (needs file-level provenance)")
            continue
        for key_path in _walk_json(json.loads((snap / rel).read_text())):
            if not _covered(key_path, entry):
                missing.append(f"{rel}:{key_path}")
    record(
        "provenance.covers_all_config_keys",
        not missing,
        f"uncovered={missing[:8]}" if missing else "",
    )
    sources = [s for e in prov.values() for s in e.values()]
    malformed = [
        s
        for s in sources
        if not (s.startswith("vllm-omni vllm_omni/") or s.startswith("SYNTHETIC"))
    ]
    record(
        "provenance.sources_are_vllm_omni_or_synthetic",
        not malformed,
        f"bad={malformed[:3]}" if malformed else "",
    )


def check_readonly(paths: list[Path]) -> None:
    writable = []
    for base in paths:
        for dirpath, dirs, files in os.walk(base):
            if os.stat(dirpath).st_mode & 0o222:
                writable.append(dirpath)
            for name in files:
                full = os.path.join(dirpath, name)
                if os.path.islink(full):
                    continue
                if os.stat(full).st_mode & 0o222:
                    writable.append(full)
    record(
        "readonly.no_write_bits",
        not writable,
        f"writable={writable[:5]}" if writable else "",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--hub-dir",
        default="/mnt/workspace/hf_cache/hub",
        help="HF hub cache dir that holds models--MiniMaxAI--MiniMax-H3",
    )
    parser.add_argument(
        "--snapshot-dir", default=None, help="verify a plain directory build instead"
    )
    parser.add_argument(
        "--quick", action="store_true", help="skip full content hashing"
    )
    args = parser.parse_args()
    if args.snapshot_dir:
        snap = Path(args.snapshot_dir)
        record("layout.plain_dir", snap.is_dir(), str(snap))
        repo = None
    else:
        repo = Path(args.hub_dir) / REPO_FOLDER
        snap = check_layout(repo, args.quick)
    if snap is not None and snap.is_dir():
        check_stamp(snap)
        check_manifest(snap, args.quick)
        check_index(snap)
        check_origin(snap)
        check_provenance(snap)
        if repo is not None:
            check_readonly([repo / "snapshots", repo / "blobs"])
    failed = [name for name, ok, _ in RESULTS if not ok]
    print(
        f"SUMMARY: {len(RESULTS) - len(failed)} passed, {len(failed)} failed"
        + (f" ({', '.join(failed)})" if failed else "")
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
