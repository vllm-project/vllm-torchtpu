# SPDX-License-Identifier: Apache-2.0
"""Optional logical-to-physical expert permutations for DSV41 FP4 serving.

Checkpoint bytes keep their original names during fetch. The loader filters
and yields the corresponding physical expert names; native top-k selection
keeps logical IDs until the fused EP boundary applies the same permutation.
"""

import re

ExpertPermutations = dict[str, tuple[int, ...]]

_CHECKPOINT_EXPERT = re.compile(
    r"^(?P<prefix>(?:model\.)?layers\.(?P<layer>\d+)\.ffn\.experts\.)"
    r"(?P<expert>\d+)(?P<suffix>\..+)$"
)
_MODULE_EXPERT = re.compile(r"(?:^|\.)layers\.(\d+)\.ffn\.experts$")


def parse_expert_permutations(value: object) -> ExpertPermutations | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError("expert_permutations must map layer indices to permutations")
    result = {}
    for layer, permutation in value.items():
        if (
            not isinstance(layer, str)
            or not layer.isdecimal()
            or str(int(layer)) != layer
        ):
            raise ValueError("Expert permutation keys must be canonical layer indices")
        if not isinstance(permutation, (list, tuple)) or not permutation:
            raise ValueError(f"Layer {layer} needs a nonempty expert permutation")
        if any(type(index) is not int for index in permutation):
            raise TypeError(f"Layer {layer} expert IDs must be integers")
        if sorted(permutation) != list(range(len(permutation))):
            raise ValueError(f"Layer {layer} expert permutation must be a bijection")
        result[layer] = tuple(permutation)
    return result or None


def validate_expert_topology(
    permutations: ExpertPermutations, *, layers: int, experts: int
) -> None:
    for layer, permutation in permutations.items():
        if int(layer) >= layers or len(permutation) != experts:
            raise ValueError(
                f"Expert permutation {layer} does not match {layers} layers / "
                f"{experts} experts; draft layers cannot be remapped"
            )


def permutation_for_module(
    prefix: str, permutations: ExpertPermutations | None
) -> tuple[int, ...] | None:
    if not permutations:
        return None
    match = _MODULE_EXPERT.search(prefix)
    if match is None:
        raise ValueError(f"Unrecognized DSV41 expert module prefix: {prefix}")
    permutation = permutations.get(match.group(1))
    if permutation is None or permutation == tuple(range(len(permutation))):
        return None
    return permutation


def remap_checkpoint_expert(name: str, permutations: ExpertPermutations) -> str:
    match = _CHECKPOINT_EXPERT.fullmatch(name)
    if match is None or (permutation := permutations.get(match["layer"])) is None:
        return name
    logical = int(match["expert"])
    if logical >= len(permutation):
        raise ValueError(f"Checkpoint expert outside configured permutation: {name}")
    return f"{match['prefix']}{permutation[logical]}{match['suffix']}"
