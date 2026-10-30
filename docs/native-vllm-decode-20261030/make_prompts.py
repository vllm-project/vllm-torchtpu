# SPDX-License-Identifier: Apache-2.0

"""Create substitute token-ID prompts; these are not the measured dataset."""

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
tokenizer = AutoTokenizer.from_pretrained(a.model)
base = tokenizer.encode(
    "This is synthetic context for a decode performance benchmark. ",
    add_special_tokens=False,
)
assert base
items = []
for index in range(128):
    head = tokenizer.encode(f"Document {index}. ", add_special_tokens=False)
    suffix = tokenizer.encode(
        f"\nQuestion {index}: continue the text.\n", add_special_tokens=False
    )
    ids = (head + base * (262144 // len(base) + 1))[:262144]
    items.append({"token_ids": ids, "suffix_ids": suffix})
payload = json.dumps(items).encode()
Path(a.output).write_bytes(payload)
print("sha256", hashlib.sha256(payload).hexdigest())
