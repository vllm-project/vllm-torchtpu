# How DeepSeek-V4's KV cache gets from the prefill machine to the decode machine

Written alongside PR #684, which admitted DSv4 into Raiden Stage 3.

## 1. Some quick definitions

```
  LAYER        a block of the neural network. DSv4 layer 0, layer 1, ...
               one layer can own SEVERAL caches (attention, SWA, compressor state)

  ARRAY        one actual tensor in HBM. Has a shape and a byte size.
               several caches can be overlaid on ONE array.

  CACHE GROUP  vLLM's unit of paging. Layers whose caches page identically get
               bundled into one group, and that group has ONE block table
               shared by every layer in it.

  BLOCK TABLE  where a request's pages physically live. [17, 4, 92] means
               "my first page is in slot 17, my second in slot 4, my third
               in slot 92".

  POOL TAG     Raiden's name for a transferable set of bytes, e.g. "dsv4.swa.g1"
```

The relationship is many-to-many in both directions:

```
   layer 0  ---owns--->  attention cache  ---lives on--->  array A0  ---in--->  group 0
            ---owns--->  swa cache        ---lives on--->  array A0  ---in--->  group 1
            ---owns--->  compressor state ---lives on--->  array A0  ---in--->  group 4
```

## 2. Why DeepSeek-V4 breaks the old assumption

Stage 3 was built for one cache group with one page size, and it refused to
start on anything else, with the error *"requires exactly one full-attention KV
cache group"* (`tpu_connector.py`, as of base commit `293a4737`). DSv4 breaks
that twice over.

### 2a. Six groups, six page sizes, six windows

DSv4 mixes several kinds of attention in one model, so vLLM ends up with six
cache groups. Here is the real geometry, taken from block tables captured off a
running DP4/DP4 DeepSeek-V4-Flash pair
(`test_stage3_window_recycled_pages.py`):

```
  group  pool tag              page size    sliding window
  -----  --------------------  -----------  --------------
    0    dsv4.csa.*, hca, idx  1024 tokens  none (full attention)
    1    dsv4.swa.g1            128 tokens   128 tokens
    2    dsv4.swa.g2            128 tokens   128 tokens
    3    dsv4.state.idx.g3       32 tokens     8 tokens
    4    dsv4.state.csa.g4       16 tokens     8 tokens
    5    dsv4.state.hca.g5       32 tokens   128 tokens
```

Page size in tokens is just `group.kv_cache_spec.block_size`. The 128 comes from
`_swa_block_size` in `deepseek_v4_attention_op.py`, which clamps a page to the
window.

Every group's page is the same size in bytes, and vLLM asserts it
(`get_uniform_page_size` in `kv_cache_utils.py`). What differs is how many
tokens fit in those bytes, because each group compresses at a different ratio.
So a 2000-token
request needs 2 pages in group 0 and 125 in group 4, and a single
`self.block_size` cannot say both.

### 2b. The caches share memory

DSv4 does not give each cache its own memory. It overlays them, in three
separate places in the allocator (`runner/kv_cache_dsv4.py`): sliding-window
caches are bound to the same tensor object as a CSA NoPE array, a compressor
state is bound to its own compressed-KV layer's entry, and HCA states land on
CSA NoPE arrays too.

Dumping the arrays of a short DSv4 config makes it concrete:

```
   array A0, 16x256x4x128, 2 MiB
   +--------+--------+--------+--------+--------+
   | page 0 | page 1 | page 2 | page 3 | page 4 |  ...
   +--------+--------+--------+--------+--------+
       ^  all four of these begin at byte 0 of A0:
       |
       +-- layer 0  attention (CSA NoPE)   group 0
       +-- layer 0  swa_cache              group 1   <- overlaid
       +-- layer 0  compressor state       group 4   <- overlaid
       +-- layer 2  compressor state       group 4   <- overlaid, different layer!
```

Now say you hand Raiden "array A0, pages 3 and 7." **That instruction is
ambiguous.** Four caches begin at byte 0 of A0, and nothing in the bytes says
who owns page 3.

What saves it is that groups own disjoint pages: every page comes from one
shared pool (`kv_cache_coordinator.py`), so if group 1's SWA cache holds
page 3, no other group does. Raiden cannot see that, because all it sees is a
byte range and a list of integers.

## 3. The plumbing that was added

**Each cache group gets its own transfer record.** Stage 3 used to hold a single
page size for the whole model. It now builds one `_Stage3TransferGroup` per
group, carrying that group's index, page size and window
(`_Stage3TransferGroup` in `tpu_connector.py`):

```python
@dataclass(frozen=True)
class _Stage3TransferGroup:
    cache_group_index: int
    page_tokens: int
    window_tokens: int | None = None
```

**Every pool tag names its group,** via a `.g<N>` suffix (`dsv4_group_tag` in
`tags.py`). This is the fix for 2b: `dsv4.swa.g1` reads page
numbers out of group 1's block table and no other, so the ambiguous "A0, page 3"
becomes the unambiguous "group 1's page 3", which disjointness guarantees is
unique.

**A refusal at startup** if two layers of the *same* group ever land on the same
array (`pool_manifest.py`, with the runner keeping its own copy in
`_validate_ds_v4_overlay`). Same group means one shared block table, and then
the pages really would collide. Better to die at startup than corrupt KV
quietly.

**Per-group block tables on the wire:** `group_block_ids: list[list[int]]`
threaded through the send metadata (`tpu_connector.py`). One list per
group instead of one flat list for the model:

```
  BEFORE: one group, so one of everything.

    page size 128     block table [17, 4, 92]     pool tag "kv.fa"

  AFTER: one row per cache group.

    group  page  window  block table        pool tag
    -----  ----  ------  -----------------  -----------------
      0    1024  none    [17, 4]            dsv4.csa.nope.g0
      1     128  128     [9, 31, 8]         dsv4.swa.g1
      2     128  128     [22, 6, 44]        dsv4.swa.g2
      3      32  8       [3, 19]            dsv4.state.idx.g3
      4      16  8       [12, 5, 7, 40]     dsv4.state.csa.g4
      5      32  128     [61, 2]            dsv4.state.hca.g5
```

One send now carries all six block tables at once. Raiden reads row N's table
only through row N's tag, so a page number is never interpreted against a group
that does not own it. The tables have different lengths because the groups have
different page sizes, all describing the same request.

That is where most of the PR goes: every place that handled one group
now loops over six, and each of those places exists twice, once where the
producer registers what it is sending and once where the consumer works out
where it lands.

## 4. Sliding window

Three of DSv4's six groups attend over a sliding window, and that is the one
piece of geometry the transfer cannot copy straight. A windowed group's block
table is mostly holes.

### 4a. vLLM hands the old pages back

A window of W tokens cannot reach further back than W tokens, so vLLM returns
every page behind it to the pool. It does not shorten the block table; it
writes page 0, the reserved null block, over the entry
(`remove_skipped_blocks` in `single_type_kv_cache_manager.py`, `block_pool.py`).
The count is

```
  pages_freed = max(0, num_computed_tokens - W + 1) // page_size
```

which is `SlidingWindowManager.get_num_skipped_tokens`. Group 4 pages at 16
tokens under an 8-token window, so a 375-token
request keeps one page and nulls the other 23. That is correct, not a bug.

### 4b. The transfer makes the same cut

Those nulls are all page 0, so shipping them hands Raiden one destination named
23 times. It rejects a duplicate destination, and the rejection raises out of
`execute_model` on the producer worker, killing the engine and every unrelated
request riding on it.

So the connector drops the dead prefix using the allocator's own formula
(`_stage3_skipped_window_pages` in `tpu_connector.py`, with the trim in
`_stage3_trim_window_pages`). Producer and consumer both run it on the same
request length and page geometry, which keeps their page lists paired entry for
entry. Each group carries its own `window_tokens` in `_Stage3TransferGroup`.
DSv4 wraps its groups in `UniformTypeKVCacheSpecs`, which has no
`sliding_window`, so `_stage3_group_window_tokens` unwraps it and asks a layer.

### 4c. The two numbers have to be one number

The allocator frees against `num_computed_tokens`, how far the producer ran.
The connector skips against `num_tokens`, the transfer prefix, which is one
token shorter because the last prompt token is withheld for the decode replica
to recompute. One token is enough to cross a page boundary:

```
  prompt 375, computed 375, num_tokens 374

    allocator frees   (375 - 8 + 1) // 16 = 23 pages
    connector skips   (374 - 8 + 1) // 16 = 22 pages
```

Entry 22 is then a null listed as a real destination. For group 4 that happens
on one request in sixteen.

The fix is to stop the producer where the handoff stops. It drops the final
prompt token from its own request before prefill, so it computes N-1 and
`num_computed_tokens` really is N-1 (`_maybe_truncate_prompt_for_handoff`).
Both formulas then read 374 and cannot disagree. Nothing on the
wire moves, because `num_tokens` was 374 either way. The decode replica runs
that same formula on that same 374, so it frees the pages the producer skipped
and expects exactly what arrived. The two sides match by construction, not by
separate arithmetic landing on the same answer.

Mamba needed this first, for its own reason: its state advances every token, so
a producer running to N ships h(N) and the decode replica advances the
recurrence twice. The trigger was keyed to Mamba groups, and
`_stage3_truncates_prompt` widens it to any group with a window.

### 4d. When prefill runs past the handoff

Truncation only controls a producer that stops at the prompt. Hit the prefill
port directly, outside the proxy, and it generates a full answer instead: the
window slides forward, the pages it was going to send get recycled, and by
`request_finished` there is nothing left to transfer.

The connector catches that by counting how many pages in the range are the null
block (`_stage3_recycled_transfer_pages`) and drops the send, so the
decode replica prefills the request itself. Slower, but correct. Without it two
recycled pages arrive as page 0 twice, one destination named by two sources,
which Raiden rejects and which takes the engine down with it. That rejection is
inferred from the crash; `tpu_sync`'s native code is not in this tree.

A real disaggregated deployment never gets here, because everything goes through
the proxy. We got here because our accuracy oracle queries prefill alone for a
non-disaggregated baseline.

### 4e. Why no other model hits this

Full attention never reports skipped tokens
(`get_num_skipped_tokens` in `single_type_kv_cache_manager.py`), so a group
without a window never gets a null in its block table. Qwen3.5 and GLM have one
full-attention group,
so nothing in section 4 can reach them. Before this PR no windowed model went
down the Raiden path at all.
