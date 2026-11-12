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
"""The query layout of a step's tokens before and after a split point."""


def split_query_layout(query_start_loc, seq_lens, split: int):
    """``(first_query_start_loc, first_seq_lens, second_query_start_loc,
    second_seq_lens)``.

    The attention of the tokens before `split` and of the rest, run in that
    order (the first writes its keys before the second attends), equals the
    attention of the whole step. A request straddling the split keeps its
    pages: the first half sees only its tokens before the split, its length
    shortened by the rest; the second half sees the remaining tokens at the
    full length. A request without tokens in a half has length 0 there, so
    its context is not scanned for queries it does not have. Works on numpy,
    JAX and torch arrays alike.
    """
    first = query_start_loc.clip(max=split)
    second = query_start_loc.clip(min=split) - split
    tail = _lens(second)
    return (
        first,
        (_lens(first) > 0) * (seq_lens - tail),
        second,
        (tail > 0) * seq_lens,
    )


def join_seq_lens(first_seq_lens, second_query_start_loc, second_seq_lens):
    """The whole step's sequence lengths from its halves'."""
    joined = first_seq_lens + _lens(second_query_start_loc)
    return joined + (second_seq_lens - joined).clip(min=0)


def _lens(query_start_loc):
    return query_start_loc[1:] - query_start_loc[:-1]
