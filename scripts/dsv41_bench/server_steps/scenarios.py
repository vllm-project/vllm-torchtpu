# SPDX-License-Identifier: Apache-2.0
"""Fixed, node-local DSv4.1 serving workloads and server-side accounting."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Scenario:
    name: str
    concurrency: int
    prefix_tokens: int = 65536
    extend_tokens: int = 0
    output_tokens: int = 1
    dp_size: int = 8
    chunk_tokens: int = 2048
    dspark_tokens: int = 5
    physical_chips: int = 4

    @property
    def local_concurrency(self) -> int:
        return self.concurrency // self.dp_size

    def to_dict(self) -> dict:
        return asdict(self)


SCENARIOS = {
    "encoder_extend": Scenario("encoder_extend", 8, extend_tokens=16384),
    "swa_replay": Scenario("swa_replay", 8),
    "swa_replay_c16": Scenario("swa_replay_c16", 16),
    "swa_replay_c64": Scenario("swa_replay_c64", 64, chunk_tokens=4096),
    "decode": Scenario("decode", 128, output_tokens=512),
}


def classify_rows(starts: list[int], ends: list[int], prompts: list[int]) -> str:
    if not starts:
        return "idle"
    if all(start >= prompt for start, prompt in zip(starts, prompts, strict=True)):
        return "decode"
    if all(end <= prompt for end, prompt in zip(ends, prompts, strict=True)):
        return "prefill"
    return "mixed"


def token_counts(starts: list[int], ends: list[int], prefix: int) -> dict:
    return {
        "computed_tokens": sum(b - a for a, b in zip(starts, ends, strict=True)),
        "new_tokens": sum(
            max(0, b - max(a, prefix)) for a, b in zip(starts, ends, strict=True)
        ),
        "replay_tokens": sum(
            max(0, min(b, prefix) - a) for a, b in zip(starts, ends, strict=True)
        ),
    }
