# Main-KV DCP-2 equality harness (`VLLM_TPU_QSA_MAIN_KV_DCP=2`)

Two simulated ranks run the DCP path on projections of a replicated pool: the owned B-pages sit under the 2B-token table (`table[t // 2B]`, row `t % B`, owner `(t // B) % 2`); all other pages are fp8 NaN. Results are compared with the replicated path.

**Adapters** (`dcp2_harness.py`; each has the signature of the production function it stands in for):
- write: `pooled_qsa.qsa_cache_writes(pool, k, v, raw, positions, nik, trig, starts, context, ring_ids, main_table, compressed_table, *, ..., dcp=(rank_i32[1], 2)) -> (pool, req, logical, valid)`
- metadata: `qsa_rpa.virtual_page_metadata(blocks, lengths, table, *, ratio, page_size, dcp) -> (pages, kv_lens, cu, dist)`. Expected value: `reference_metadata`, owned groups sorted ascending, then the owned tail.
- attend: `fn(q, gate, blocks, lengths, table, cache, *, kv_heads, ratio, k_scale, v_scale, dcp[, kv_head_offset]) -> (out f32 [T,2H,Dh], lse f32 [T,2H])`. Real: `rpa_attention(..., return_lse=True)[:2]` and `_prefill_attention(..., starts, return_lse=True)[:2]`. Reference: `reference_attend` (float32).
- combine: `qsa_rpa.dcp_combine(out_pair [2,T,2H,Dh], lse_pair [2,T,2H], gate [T,H,Dh], rank_i32[1]) -> bf16 [T,H,Dh]`.

**Tolerance.** Writes, ownership and per-rank compacted metadata must match exactly (raw bytes / ints). Attention is not bit-equal because the split softmax reorders the sums:
- Pre-gate f32 merge vs the replicated route's `return_lse` output: err / max|head vector| <= 1e-4 (CPU) or 4e-3 (TPU).
- Gated bf16 <= 2 ulps. The kernels round twice (`bf16(out)`, then `bf16(bf16(out) * bf16(sigmoid(gate)))`), so a 1-ulp flip of `bf16(out)` can become 2 ulps after the gate.
- Any NaN fails.

**Poison.** Unallocated pages and the replicated layout's pages are NaN. On CPU, the null page 0 and rows past the sequence end hold 448: they may be loaded and masked, and the CPU RPA stand-in multiplies by 0. With `DCP2_HARNESS_DEVICE=tpu` they are NaN (0x7F) too.

**Run (CPU):** `PYTHONPATH=src python3 -m pytest -q tests/kernels/qwen4_exp/dcp2`. Pallas kernels run in interpret mode; batched RPA uses the CPU stand-in `conftest.reference_rpa_lse`.
**Run (TPU):** `DCP2_HARNESS_DEVICE=tpu python3 -m pytest -q tests/kernels/qwen4_exp/dcp2/test_dcp2_tpu.py`. This uses the real Mosaic kernels.

**Tests**
- `test_dcp2_writes.py` (c): owned writes are byte-exact (whole-row and paired slots 0/1, fused writer, 2048-token chunk); compressed keys and rings stay replicated. Fault controls: unowned writes, flipped owner, B-span.
- `test_dcp2_decode.py` (a): verify rows 4/8/32/96 plus NaN padding, idle seats, tail phases 0/1/B-1/B/B+1, 3-40 pages, a rank with no keys, the tail on either owner; metadata is exact. Fault controls: flipped owner, token-parity owner, rank-0 only, plain average, metadata order/owner.
- `test_dcp2_prefill.py` (b): chunks of 2048 (prefix 1234), 8192 (prefix 3001, budget 64 on CPU) and ragged pieces; the reference is tied to the real kernel; the real DCP kernel at a small shape; the same 4 fault controls.
- `test_dcp2_combine.py` (d): one-sided rows, rows with no owner (0, no NaN), lse gaps, head pick. Fault controls: rank-0 only, plain average.
- `test_dcp2_op.py`: `pooled_qsa_dcp_step` x2 + `dcp_combine` vs `pooled_qsa_step` (verify, mixed); identical selection; exact pools.
- `test_dcp2_tpu.py`: real RPA (served env, slots 0/1, 32/96 rows) and block-sparse prefill (2048/8192 chunks with prefixes 1234/5000/9003, ragged mix).
