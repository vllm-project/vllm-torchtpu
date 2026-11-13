// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
/**
 * Client-side ES2022 roofline and on-chip VMEM physics evaluator.
 */

(function(global) {
  function createAdvisorEngine(seedData) {
    const D = seedData || {};
    const SEED_DEF = global.__MK_ENGINE_DEFAULTS || {};
    const DATA = Object.assign({}, SEED_DEF, D);
  const CORE_LIM = Object.assign(
    {}, SEED_DEF.core_limits || {}, D.core_limits || {}
  );
  const WL_DEF = Object.assign(
    {}, SEED_DEF.workload_defaults || {}, D.workload_defaults || {}
  );
  const PROF = DATA.model_profile || {}, CHIP = DATA.chip_spec || {};
  const REGIMES = DATA.regimes || [];
  const SUPPORTED_FAM = new Set(CORE_LIM.supported_onchip_families || []);
  const RECOGNIZED_TYPES = new Set(CORE_LIM.recognized_model_types || []);
  const VERIFIED_RANKS = new Set(CORE_LIM.verified_ranks || []);
  const CHIP_KINDS = new Set(
    (CORE_LIM.chip_kinds || []).map(s => String(s).toLowerCase())
  );
  // Preset names resolve the way Python resolves them: case, '.', and '-' do not matter.
  const normPresetName = s => String(s || '').toLowerCase().replace(/[.-]/g, '_');
  const DECLARED_MK = {};
  for (const [name, rows] of Object.entries(CORE_LIM.declared_megakernels || {})) {
    DECLARED_MK[normPresetName(name)] = rows;
  }
  const declaredFor = name => DECLARED_MK[normPresetName(name)] || [];
  const DTYPE_BYTES = Object.assign({
    fp32: 4.0, bf16: 2.0, fp16: 2.0, fp8: 1.0, int8: 1.0,
    nvfp4: 0.5625, mxfp4: 0.53125, fp4: 0.5625
  }, DATA.dtype_bytes || SEED_DEF.dtype_bytes || {});
  function roundEven(x) {
    const f = Math.floor(x), d = x - f;
    if (Math.abs(d - 0.5) < 1e-12) return (f % 2 === 0) ? f : f + 1;
    return Math.round(x);
  }
  function gcd(a, b) {
    let x = Math.abs(Math.trunc(a)), y = Math.abs(Math.trunc(b));
    while (y !== 0) { const t = y; y = x % y; x = t; }
    return x || 1;
  }
  function peakFlops(dtype, chipObj, actDtype) {
    const c = chipObj || CHIP;
    const d = String(dtype || 'bf16').toLowerCase();
    const act = String(actDtype || '').toLowerCase();
    const bf16 = c.bf16_flops_per_s || 0.0;
    if (act === 'bf16' || act === 'bfloat16' || act === 'fp16' || act === 'float16') {
      return bf16;
    }
    if (act === 'fp32' || act === 'float32') {
      return 0.5 * bf16;
    }
    if (d === 'fp8' || d === 'nvfp4' || d === 'mxfp4' || d === 'fp4') {
      return c.fp8_flops_per_s || (c.native_ocp_fp8 !== false ? 2.0 * bf16 : bf16);
    }
    if (d === 'int8') return c.int8_flops_per_s || (2.0 * bf16);
    if (d === 'fp32') return 0.5 * bf16;
    return bf16;
  }
  function rowFloor(dtype, chipObj) {
    const c = chipObj || CHIP;
    const dt = String(dtype || 'fp8').toLowerCase();
    const map = c.matmul_row_tile_by_dtype || {};
    if (map[dt]) return map[dt];
    const hasFp8 = c.native_ocp_fp8 !== false;
    return (hasFp8 && (dt === 'fp8' || dt.includes('fp4'))) ? 32 : 16;
  }
  function allReduceCost(msgBytes, effTp, collLatS, chipObj, wl) {
    const c = chipObj || CHIP;
    if (effTp <= 1 || msgBytes <= 0 || (wl && wl.dcnBwBytesPerS < 0)) {
      return { xferS: 0, xferStrictS: 0, latS: 0, hops: 0, intraHops: 0, interLatS: 0 };
    }
    // A ring or dimension-ordered all-reduce moves each rank's share over one link per
    // step. The strict floor keeps the egress sum, the limit for any algorithm.
    const hops = roundEven(Math.log2(effTp));
    const vol = 2.0 * (effTp - 1) / effTp * msgBytes;
    return {
      xferS: vol / c.ici_link_bw_bytes_per_s,
      xferStrictS: vol / c.ici_egress_bw_bytes_per_s,
      latS: hops * collLatS, hops, intraHops: hops, interLatS: 0
    };
  }
  // A layer requests its routed-expert weights only after its router runs, so that
  // stream overlaps the MoE matmuls but not the dense compute before them.
  function routedChain(tDense, routedHbmS, tMoe, tVecMoe = 0.0) {
    return tDense + Math.max(routedHbmS, tMoe, tVecMoe);
  }
  // A fused kernel overlaps the whole HBM stream with the chain, but each all-reduce
  // waits for its layer's result. Separate kernels run collectives after the compute.
  function coreTime(hbmS, chainS, commS, fused) {
    return fused ? Math.max(hbmS, chainS + commS) : (Math.max(hbmS, chainS) + commS);
  }
  // Rows one request adds to a decode or verify call.
  function requestRows(wl) {
    const rawSpec = wl.specTokens ?? wl.spec_tokens;
    const specTok = (rawSpec !== undefined && rawSpec !== null)
      ? rawSpec : Math.max(0, (wl.seqTokens || 1) - 1);
    return wl.phase === 'verify' ? Math.max(1, specTok) + 1 : 1;
  }
  const DEFAULT_DECLARED_PHASES = CORE_LIM.default_declared_phases || ['decode', 'verify'];
  // Deployment blocks that declare scope `sc` for the phase. A block without phases
  // covers decode and verify only, and a block needs a source and a rank count.
  function declaredBlocks(p, sc, phase) {
    return declaredFor(p.name).filter(
      d => d && d.source && d.ranks > 0 && Array.isArray(d.scopes) && d.scopes.includes(sc) &&
        (Array.isArray(d.phases) && d.phases.length ? d.phases : DEFAULT_DECLARED_PHASES)
          .includes(phase)
    );
  }
  // The block that declares scope `sc` at this rank count for the phase.
  function findDeclaration(p, sc, effTp, phase) {
    return declaredBlocks(p, sc, phase).find(d => effTp === d.ranks) || null;
  }
  // Most rows the megakernel serves in one decode or verify call (0: no declared limit).
  // The workload's own limit wins over a preset's.
  function singleCallRowLimit(p, wl, effTp) {
    if (wl.phase !== 'decode' && wl.phase !== 'verify') return 0;
    const decl = findDeclaration(p, 'whole_step', effTp, wl.phase);
    return (wl.maxRowsPerCall ?? wl.max_rows_per_call ?? 0) ||
      (decl ? decl.max_rows : 0);
  }
  // Most rows a whole step may hold. A workload limit means a kernel that stops there
  // and does not split; a declaration allows max_total_rows through split calls.
  function wholeStepRowLimit(decl, wl) {
    return (wl.maxRowsPerCall ?? wl.max_rows_per_call ?? 0) ||
      (decl ? (decl.max_total_rows || decl.max_rows) : 0) ||
      CORE_LIM.max_batch_rows;
  }
  function evaluateSupport(sc, p, wl, ch, ctx) {
    const batchRaw = wl.batch !== undefined ? wl.batch : WL_DEF.batch;
    if (sc === 'per_op' || batchRaw <= 0) {
      return { support: 'baseline', blockers: [] };
    }
    // A declaration states that a runtime exists for this model, scope, rank count and
    // phase, so it supplies the row limit and layer families checked below.
    const blocks = declaredBlocks(p, sc, wl.phase);
    const decl = blocks.find(d => ctx.effTp === d.ranks) || null;
    const fams = p.layer_families || [];
    const hasUnmodeled = !decl && fams.some(f => !SUPPORTED_FAM.has(f));
    const maxAr = CORE_LIM.max_ar_slots_per_layer;
    const isPeriodic = p.is_periodic !== undefined ? Boolean(p.is_periodic) : true;
    const arSlots = p.ar_slots_per_layer !== undefined ? p.ar_slots_per_layer : 2;
    const plannerBad = Boolean(ctx.planError) || (!decl && !isPeriodic) || (arSlots > maxAr);
    const chipName = String(ch.name || '').toLowerCase();
    const chipKind = (chipName === 'tpu7x' || chipName === '7x') ? 'tpu7x' : chipName;
    const mt = String(p.model_type || '');
    const archUnv = Boolean(p.architecture_unverified) ||
      (Boolean(mt) && RECOGNIZED_TYPES.size > 0 && !RECOGNIZED_TYPES.has(mt));
    const blockers = [];
    if (sc === 'single_layer' || sc === 'layer_group') {
      blockers.push('scope_without_core_counterpart');
    }
    if (ctx.activeTokens > wholeStepRowLimit(decl, wl)) {
      blockers.push('rows_exceed_core_max_batch_rows');
    }
    // Declared rank counts restrict a model. Without a declaration for this scope and
    // phase, the core rank grid applies.
    const rankBad = blocks.length > 0 ? !decl : !VERIFIED_RANKS.has(ctx.effTp);
    if (rankBad) {
      blockers.push('rank_count_outside_core_grid');
    }
    if (!CHIP_KINDS.has(chipKind)) {
      blockers.push('chip_without_core_budget');
    }
    if (ctx.epDp > 1 && !(decl && decl.attention_dp_in_call)) {
      blockers.push('all_to_all_not_in_core');
    }
    if (hasUnmodeled) blockers.push('unmodeled_layer_family');
    if (archUnv) blockers.push('model_type_unrecognized');
    if (plannerBad) blockers.push('planner_rejected');
    if (blockers.length > 0) {
      return { support: 'hypothetical', blockers };
    }
    if (decl && ctx.activeTokens > 0) {
      return { support: 'declared', blockers: [] };
    }
    return { support: 'within_core_limits', blockers: [] };
  }
  function computePerTokenVmem(wl, profObj) {
    const p = profObj || PROF;
    const H = p.hidden_size, tpRaw = Math.max(1, wl.tp || WL_DEF.tp);
    const dcp = Math.max(1, wl.dcp || 1);
    const effTp = tpRaw === 1 ? Math.max(1, tpRaw, dcp) : tpRaw;
    const ep = Math.max(1, wl.ep || 1);
    const kvSplit = Math.max(1, wl.kvSplit || wl.kv_split || 1, dcp);
    const vmemAlias = wl.vmemAlias !== undefined
      ? wl.vmemAlias : (wl.vmem_alias !== undefined ? wl.vmem_alias : true);
    const moeMode = wl.moeMode || wl.moe_mode || 'masked_on_chip';
    const resStreams = Math.max(1, p.residual_streams || 1);
    const spine = (4 * resStreams + 2) * H + (effTp > 1 ? 6 * H : 0);
    const nhLoc = Math.max(1, Math.ceil(p.num_attention_heads / effTp));
    const hd = p.head_dim;
    let attn = 0, lin = 0, moe = 0;
    if ((p.kv_lora_rank || 0) > 0) {
      const qlr = p.q_lora_rank || 0, qkr = p.qk_rope_head_dim || 0;
      attn = 2 * (qlr + p.kv_lora_rank + qkr) + 6 * (nhLoc * hd);
    } else {
      const tpPerKv = Math.max(1, Math.floor(effTp / kvSplit));
      const nkvLoc = Math.max(1, Math.ceil(p.num_kv_heads / tpPerKv));
      attn = 2 * (2 * nhLoc + 2 * nkvLoc) * hd + 4 * nhLoc * hd;
    }
    if ((p.num_linear_layers || 0) > 0 && (p.linear_value_heads || 0) > 0) {
      const khL = Math.max(1, Math.ceil((p.linear_key_heads || 0) / effTp));
      const vhL = Math.max(1, Math.ceil(p.linear_value_heads / effTp));
      const kd = p.linear_key_dim || 0, vd = p.linear_value_dim || 0;
      lin = 2 * (2 * khL * kd + 2 * vhL * vd) + 4 * vhL * vd;
    }
    const nExp = p.num_experts || 0;
    if ((p.num_moe_layers || 0) > 0 && nExp > 0) {
      const topK = p.top_k || 1, router = 4 * nExp + 8 * topK;
      const interDiv = Math.max(1, Math.floor(effTp / gcd(ep, effTp)));
      const interLoc = Math.ceil((p.moe_intermediate_size || 0) / interDiv);
      const shI = p.shared_expert_intermediate_size || 0;
      const sharedLoc = shI > 0 ? Math.ceil(shI / effTp) : 0;
      const expH = p.routed_expert_hidden_size || H;
      const effK = ep > 1 ? Math.min(topK, 2 * ep) : 1;
      const grp = 8 * expH +
        Math.ceil(effK * (6 * expH + 8 * interLoc) / ep) + 8 * sharedLoc;
      const isGrp = moeMode === 'grouped_on_chip';
      const tile = isGrp ? grp : (16 * expH + 8 * (interLoc + sharedLoc));
      moe = vmemAlias ? Math.max(router + 2 * expH, tile) : (router + tile);
    } else {
      moe = 6 * H + 8 * Math.ceil((p.intermediate_size || 0) / effTp);
    }
    const sumOp = attn + lin + moe;
    const opBytes = vmemAlias ? Math.max(attn, lin, moe) : sumOp;
    return {
      spineBytes: spine, attnBytes: attn, linBytes: lin, moeBytes: moe,
      opBytes, unaliasedOpBytes: sumOp, totalPerTokenBytes: spine + opBytes
    };
  }
  function expertsHit(tokensEp, topK, numExperts, expertsLocal) {
    if (numExperts <= 0 || expertsLocal <= 0) return 0.0;
    const miss = Math.pow(1.0 - Math.min(1.0, topK / numExperts), tokensEp);
    return Math.max(1.0, expertsLocal * (1.0 - miss));
  }
  function selectAdaptiveRing(
    vmemBudget, fixedNonRing, alpha, activeTokens, hasMoe, wl, dBank, eBank
  ) {
    if (wl.tiles) {
      const nbd = wl.tiles.nbd, nbe = hasMoe ? wl.tiles.nbe : 0;
      const rb = nbd * wl.tiles.bank + nbe * wl.tiles.ebank;
      return { nbd, nbe, ringB: rb, minRingB: rb };
    }
    const rp = wl.ringPlan || wl.ring_plan;
    if (rp) {
      const nbd = rp.dense_slots !== undefined ? rp.dense_slots : 2;
      const nbe = hasMoe ? (rp.expert_slots !== undefined ? rp.expert_slots : 2) : 0;
      const db = rp.dense_slot_bytes || CORE_LIM.bank;
      const eb = rp.expert_slot_bytes || CORE_LIM.ebank;
      const rb = nbd * db + nbe * eb;
      return { nbd, nbe, ringB: rb, minRingB: rb };
    }
    const bufSlots = wl.dmaBuffers || wl.dma_buffers || 2;
    const minNbd = bufSlots, extra = minNbd - 2, maxNbd = CORE_LIM.nbd + extra;
    const minNbe = hasMoe ? minNbd : 0, maxNbe = hasMoe ? (CORE_LIM.nbe + extra) : 0;
    const minRingB = minNbd * dBank + minNbe * eBank;
    const avail = vmemBudget - fixedNonRing - alpha * activeTokens;
    if (avail < minRingB) {
      return { nbd: minNbd, nbe: minNbe, ringB: minRingB, minRingB };
    }
    let bestRing = -1, bestBal = -1, bestNbd = minNbd, bestNbe = minNbe;
    for (let nbd = minNbd; nbd <= maxNbd; nbd++) {
      for (let nbe = minNbe; nbe <= maxNbe; nbe++) {
        const rb = nbd * dBank + nbe * eBank;
        if (rb <= avail) {
          const bal = hasMoe
            ? Math.min(nbd - minNbd, nbe - minNbe) : (nbd - minNbd);
          if (rb > bestRing || (rb === bestRing && bal > bestBal) ||
              (rb === bestRing && bal === bestBal && nbd > bestNbd)) {
            bestRing = rb; bestBal = bal; bestNbd = nbd; bestNbe = nbe;
          }
        }
      }
    }
    return { nbd: bestNbd, nbe: bestNbe, ringB: bestRing, minRingB };
  }
  function computeHbmUsedBytes(p, wl) {
    const dtB = k => DTYPE_BYTES[String(p[k] || 'fp8').toLowerCase()] || 1.0;
    const bwW = dtB('weight_dtype'), bwE = dtB('expert_dtype');
    const bwKv = dtB('kv_dtype');
    const bwSt = DTYPE_BYTES[String(p.state_dtype || 'bf16').toLowerCase()] || 2.0;
    const L = p.num_layers, H = p.hidden_size;
    const nExp = p.num_experts || 0;
    const tp = Math.max(1, wl.tp || WL_DEF.tp), pp = Math.max(1, wl.pp || 1);
    const dp = Math.max(1, wl.dp || 1), ep = Math.max(1, wl.ep || 1);
    const dcp = Math.max(1, wl.dcp || 1);
    const effTp = tp === 1 ? Math.max(tp, dcp) : tp;
    const kvSplit = Math.max(wl.kvSplit || wl.kv_split || 1, dcp);
    const batchLocal = Math.ceil((wl.batch || WL_DEF.batch) / dp);
    const layersLocal = Math.ceil(L / pp);
    const islVal = (wl.isl !== undefined && wl.isl !== null && wl.isl > 0)
      ? wl.isl : WL_DEF.isl;
    const oslVal = (wl.osl !== undefined && wl.osl !== null) ? wl.osl : WL_DEF.osl;
    const effCtx = (wl.context && wl.context > 0)
      ? wl.context : (islVal + (wl.phase === 'prefill' ? 0 : Math.floor(oslVal / 2)));
    const lAttn = Math.min(
      layersLocal, Math.ceil((p.num_attention_layers || 0) * layersLocal / L)
    );
    const lMoe = Math.min(
      layersLocal, Math.ceil((p.num_moe_layers || 0) * layersLocal / L)
    );
    const lLin = Math.min(
      layersLocal - lAttn, Math.ceil((p.num_linear_layers || 0) * layersLocal / L)
    );
    const hd = p.head_dim, kvLora = p.kv_lora_rank || 0;
    const qLora = p.q_lora_rank || 0, qkRope = p.qk_rope_head_dim || 0;
    const qkNope = p.qk_nope_head_dim || hd;
    const kd = p.linear_key_dim || 0, vd = p.linear_value_dim || 0;
    const nhLoc = Math.max(1, Math.ceil(p.num_attention_heads / effTp));
    const tpKvDiv = Math.max(1, Math.floor(effTp / kvSplit));
    const kvhLoc = Math.max(1, Math.ceil(p.num_kv_heads / tpKvDiv));
    const qDim = nhLoc * hd, qTot = nhLoc * (qkNope + qkRope);
    const mlaProj = (qLora ? qLora * (H + qTot) : H * qTot) +
      H * (kvLora + qkRope + qDim);
    const attnGate = (p.has_attn_output_gate && !kvLora) ? (H * qDim) : 0;
    const attnParams = kvLora
      ? (mlaProj + kvLora * nhLoc * (qkNope + hd))
      : (H * 2 * (qDim + kvhLoc * hd) + attnGate);
    const khLoc = lLin > 0
      ? Math.max(1, Math.ceil((p.linear_key_heads || 1) / effTp)) : 0;
    const vhLoc = lLin > 0
      ? Math.max(1, Math.ceil((p.linear_value_heads || 1) / effTp)) : 0;
    const zDim = vhLoc * vd, qkvLin = 2 * khLoc * kd + vhLoc * vd;
    const epTp = gcd(ep, effTp);
    const linParams = lLin > 0 ? H * (qkvLin + 2 * zDim) : 0;
    const expertsLoc = lMoe > 0 ? Math.floor(nExp / ep) : 0;
    const denseFfnParams = 3 * H * Math.max(
      1, Math.ceil((p.intermediate_size || 0) / effTp)
    );
    const sharedExpParams = 3 * H * Math.ceil(
      (p.shared_expert_intermediate_size || 0) / effTp
    );
    const iExpLoc = lMoe > 0 ? Math.ceil(
      (p.moe_intermediate_size || 0) / Math.max(1, Math.floor(effTp / epTp))
    ) : 0;
    const routedExpParams = 3 * (p.routed_expert_hidden_size || H) * iExpLoc;
    const baseDenseParams = lAttn * attnParams + lLin * linParams +
      (layersLocal - lMoe) * denseFfnParams + lMoe * sharedExpParams;
    const wStatic = baseDenseParams * bwW + lMoe * H * nExp * 2.0;
    const expBytes = routedExpParams * bwE;
    const kvComp = Math.min(1.0, Math.max(1e-6, p.kv_compress_factor ?? 1.0));
    const kvRawTok = kvLora ? (kvLora + qkRope) : 2.0 * kvhLoc * hd;
    const kvReadBytes = lAttn * batchLocal * effCtx *
      ((kvRawTok * bwKv * kvComp) / kvSplit);
    const recRwBytes = 2.0 * lLin * batchLocal *
      (zDim * kd + qkvLin * (p.linear_conv_kernel || 0)) * bwSt;
    return wStatic + lMoe * expertsLoc * expBytes + kvReadBytes + 0.5 * recRwBytes;
  }
  function isSliceTopologySupported(sliceCores, chipObj) {
    const minSlice = (chipObj && chipObj.min_slice_cores) || 1;
    if (sliceCores < minSlice || (sliceCores & (sliceCores - 1)) !== 0) {
      return false;
    }
    const topos = Object.keys((chipObj && chipObj.topologies) || {}).map(Number);
    if (!topos.length) return true;
    const maxTopo = Math.max(...topos);
    return topos.includes(sliceCores) ||
      (sliceCores > maxTopo && sliceCores % minSlice === 0);
  }
  function validateWorkload(a, b, c) {
    const p = (c !== undefined) ? (a || PROF) : PROF;
    const ch = (c !== undefined) ? (b || CHIP) : CHIP;
    const wl = ((c !== undefined) ? c : a) || {};
    const batch = wl.batch !== undefined ? wl.batch : WL_DEF.batch;
    const tp = wl.tp !== undefined ? wl.tp : WL_DEF.tp;
    const pp = wl.pp !== undefined ? wl.pp : 1;
    const dp = wl.dp !== undefined ? wl.dp : 1;
    const ep = wl.ep !== undefined ? wl.ep : 1;
    const kvSplit = wl.kvSplit ?? wl.kv_split ?? 1;
    const dcp = wl.dcp !== undefined ? wl.dcp : 1;
    const pcp = wl.pcp !== undefined ? wl.pcp : 1;
    const isl = wl.isl !== undefined ? wl.isl : WL_DEF.isl;
    const osl = wl.osl !== undefined ? wl.osl : WL_DEF.osl;
    const ctxV = wl.context !== undefined ? wl.context : 0;
    const specV = wl.specTokens ?? wl.spec_tokens ?? 0;
    const chunkV = wl.chunkSize ?? wl.chunk_size ?? 0;
    const mbV = wl.microbatches !== undefined ? wl.microbatches : 0;
    const segV = wl.segmentLayers ?? wl.segment_layers ?? 0;
    const opOv = wl.opOverheadUs ?? wl.op_overhead_us ?? 0.0;
    if (Math.min(
          p.num_layers, p.hidden_size, p.num_attention_heads, p.head_dim,
          batch, tp, pp, dp, ep, kvSplit, dcp, pcp, isl
        ) < 1 ||
        Math.min(osl, ctxV, specV, chunkV, mbV, segV) < 0 || opOv < 0.0) {
      return ['invalid_dimensions'];
    }
    const reasons = [];
    const effTp = tp === 1 ? Math.max(tp, dcp) : tp;
    if ((effTp & (effTp - 1)) !== 0) reasons.push('tp_not_power_of_two');
    if ((dp & (dp - 1)) !== 0) reasons.push('dp_not_power_of_two');
    if ((pcp & (pcp - 1)) !== 0) reasons.push('pcp_not_power_of_two');
    if (!isSliceTopologySupported(effTp * dp * pp * pcp, ch)) {
      reasons.push('slice_topology_unsupported');
    }
    if (p.num_attention_heads % effTp !== 0) {
      reasons.push('attn_heads_not_divisible_by_tp');
    }
    const kvHeads = p.num_kv_heads || 0;
    const kvTp = Math.max(1, Math.floor(effTp / Math.max(1, kvSplit)));
    if (effTp % kvSplit !== 0 ||
        (kvHeads === 0 && kvSplit > 1) ||
        (kvHeads > 0 && kvHeads % kvTp !== 0 && kvTp % kvHeads !== 0)) {
      reasons.push('kv_heads_not_divisible_by_tp');
    }
    if (effTp % dcp !== 0) reasons.push('dcp_not_dividing_tp');
    if ((p.num_experts || 0) > 0 &&
        (p.num_experts % ep !== 0 || !(1 <= p.top_k && p.top_k <= p.num_experts))) {
      reasons.push('experts_not_divisible_by_ep');
    }
    if ((effTp * dp * pcp) % ep !== 0) reasons.push('ep_exceeds_slice_ranks');
    if (pp > p.num_layers) reasons.push('pp_exceeds_layers');
    if (batch < dp || batch % dp !== 0) {
      reasons.push('batch_not_divisible_by_dp');
    }
    if ((p.num_linear_layers || 0) > 0) {
      if ((p.linear_value_heads || 0) > 0 && p.linear_value_heads % effTp !== 0) {
        reasons.push('linear_v_heads_not_divisible_by_tp');
      }
      if ((p.linear_key_heads || 0) > 0 &&
          p.linear_key_heads % effTp !== 0 && effTp % p.linear_key_heads !== 0) {
        reasons.push('linear_k_heads_not_divisible_by_tp');
      }
    }
    if (kvSplit > 1 && kvHeads === 0) reasons.push('kv_split_without_kv_heads');
    if (wl.phase === 'verify' && specV < 1) reasons.push('verify_without_spec_tokens');
    // A call holds whole requests. Python rejects this workload before it sizes HBM.
    const rowLimit = singleCallRowLimit(p, wl, effTp);
    if (rowLimit > 0 && rowLimit < requestRows(wl)) {
      reasons.push('row_limit_below_request');
      return reasons;
    }
    if (reasons.includes('batch_not_divisible_by_dp') ||
        reasons.includes('verify_without_spec_tokens')) {
      return reasons;
    }
    const hbmUsed = computeHbmUsedBytes(p, wl);
    const capFrac = wl.hbmUtilizationCap ?? wl.hbm_utilization_cap ??
      (DATA.hbm_utilization_cap !== undefined ? DATA.hbm_utilization_cap : 0.80);
    if (hbmUsed > capFrac * ch.hbm_capacity_bytes) reasons.push('hbm_pool_overflow');
    return reasons;
  }
  function resolveDispatchBox(wl) {
    const dr = wl.dispatchRegime || wl.dispatch_regime || 'auto';
    const rawOp = wl.opOverheadUs ?? wl.op_overhead_us ?? wl.launchGapUs ?? 0.0;
    const xlaP50 = DATA.compiled_xla_gap_p50_us !== undefined ? DATA.compiled_xla_gap_p50_us : 1.0;
    const xlaP90 = DATA.compiled_xla_gap_p90_us !== undefined ? DATA.compiled_xla_gap_p90_us : 1.5;
    const eagerGap = DATA.unfused_eager_gap_us !== undefined ? DATA.unfused_eager_gap_us : 6.0;
    const defaultGapUpper = DATA.default_launch_gap_upper_us !== undefined ? DATA.default_launch_gap_upper_us : 8.0;
    if (dr === 'zero_floor') return { lo: 0.0, hi: 0.0 };
    if (dr === 'compiled_xla') {
      if (wl.calibrated && wl.launchGapUs !== undefined) {
        const lo = wl.launchGapUs;
        return { lo, hi: Math.max(lo, wl.launchGapHiUs ?? lo) };
      }
      return { lo: xlaP50, hi: Math.max(xlaP50, wl.launchGapHiUs ?? xlaP90) };
    }
    if (dr === 'unfused_eager') {
      const lo = rawOp > 0.0 ? rawOp : eagerGap;
      return { lo, hi: Math.max(lo, defaultGapUpper) };
    }
    if (wl.calibrated && wl.launchGapUs !== undefined) {
      const lo = wl.launchGapUs;
      return { lo, hi: Math.max(lo, wl.launchGapHiUs ?? lo) };
    }
    return { lo: rawOp, hi: Math.max(rawOp, wl.launchGapHiUs ?? defaultGapUpper) };
  }
  function evaluateWorkload(wl, profOverride, chipOverride) {
    const p = profOverride || PROF, ch = chipOverride || CHIP;
    const dtB = k => DTYPE_BYTES[String(p[k] || 'fp8').toLowerCase()] || 1.0;
    const bwW = dtB('weight_dtype'), bwE = dtB('expert_dtype');
    const bwKv = dtB('kv_dtype');
    const bwSt = DTYPE_BYTES[String(p.state_dtype || 'bf16').toLowerCase()] || 2.0;
    const L = p.num_layers, H = p.hidden_size;
    const nExp = p.num_experts || 0, topK = p.top_k || 0;
    const tp = Math.max(1, wl.tp || WL_DEF.tp), pp = Math.max(1, wl.pp || 1);
    const dp = Math.max(1, wl.dp || 1), ep = Math.max(1, wl.ep || 1);
    const dcp = Math.max(1, wl.dcp || 1), pcp = Math.max(1, wl.pcp || 1);
    const effTp = tp === 1 ? Math.max(tp, dcp) : tp;
    const kvSplit = Math.max(wl.kvSplit || wl.kv_split || 1, dcp);
    const isPrefOrMix = wl.phase === 'prefill' || wl.phase === 'mixed';
    const effPcp = isPrefOrMix ? pcp : 1;
    const batchVal = wl.batch || WL_DEF.batch;
    const batchLocal = Math.ceil(batchVal / dp), layersLocal = Math.ceil(L / pp);
    const islVal = (wl.isl !== undefined && wl.isl !== null && wl.isl > 0)
      ? wl.isl : WL_DEF.isl;
    const oslVal = (wl.osl !== undefined && wl.osl !== null) ? wl.osl : WL_DEF.osl;
    const fallbackCtx = islVal +
      (wl.phase === 'prefill' ? 0 : Math.floor(oslVal / 2));
    const effCtx = (wl.context && wl.context > 0) ? wl.context : fallbackCtx;
    const pfxHit = wl.prefixHitRate ?? wl.prefix_hit_rate ?? 0.0;
    const uncachedIsl = Math.max(1, roundEven(islVal * (1.0 - pfxHit)));
    const cSize = wl.chunkSize ?? wl.chunk_size ?? 0;
    const chunkRaw = cSize || (wl.chunkSize || uncachedIsl);
    const chunkTokens = Math.ceil(chunkRaw / effPcp);
    const rawSpec = wl.specTokens ?? wl.spec_tokens;
    const specTok = (rawSpec !== undefined && rawSpec !== null)
      ? rawSpec : Math.max(0, (wl.seqTokens || 1) - 1);
    const verifyTok = wl.phase === 'verify'
      ? (Math.max(1, specTok) + 1) : (specTok > 0 ? specTok + 1 : 1);
    const mbVal = wl.microbatches || 0;
    const mbDenom = Math.max(1, (mbVal || 1) * effPcp);
    const prefillTok = cSize
      ? Math.ceil(cSize / effPcp)
      : Math.ceil((batchLocal * uncachedIsl) / mbDenom);
    const decodeTok = batchLocal * (wl.phase === 'decode' ? 1 : verifyTok);
    const activeTokens = wl.phase === 'prefill'
      ? prefillTok : (decodeTok + (wl.phase === 'mixed' ? chunkTokens : 0));
    const lAttn = Math.min(
      layersLocal, Math.ceil((p.num_attention_layers || 0) * layersLocal / L)
    );
    const lMoe = Math.min(
      layersLocal, Math.ceil((p.num_moe_layers || 0) * layersLocal / L)
    );
    const lLin = Math.min(
      layersLocal - lAttn,
      Math.ceil((p.num_linear_layers || 0) * layersLocal / L)
    );
    const tokResBytes = activeTokens * H * 2.0;
    const hd = p.head_dim, kvLora = p.kv_lora_rank || 0;
    const qLora = p.q_lora_rank || 0, qkRope = p.qk_rope_head_dim || 0;
    const qkNope = p.qk_nope_head_dim || hd;
    const kd = p.linear_key_dim || 0, vd = p.linear_value_dim || 0;
    const nhLoc = Math.max(1, Math.ceil(p.num_attention_heads / effTp));
    const tpKvDiv = Math.max(1, Math.floor(effTp / kvSplit));
    const kvhLoc = Math.max(1, Math.ceil(p.num_kv_heads / tpKvDiv));
    const qDim = nhLoc * hd, qTot = nhLoc * (qkNope + qkRope);
    const mlaProj = (qLora ? qLora * (H + qTot) : H * qTot) +
      H * (kvLora + qkRope + qDim);
    const attnGate = (p.has_attn_output_gate && !kvLora) ? (H * qDim) : 0;
    const attnParams = kvLora
      ? (mlaProj + kvLora * nhLoc * (qkNope + hd))
      : (H * 2 * (qDim + kvhLoc * hd) + attnGate);
    const khLoc = lLin > 0
      ? Math.max(1, Math.ceil((p.linear_key_heads || 1) / effTp)) : 0;
    const vhLoc = lLin > 0
      ? Math.max(1, Math.ceil((p.linear_value_heads || 1) / effTp)) : 0;
    const zDim = vhLoc * vd, qkvLin = 2 * khLoc * kd + vhLoc * vd;
    const epTp = gcd(ep, effTp), epDp = Math.max(1, Math.floor(ep / epTp));
    const linParams = lLin > 0 ? H * (qkvLin + 2 * zDim) : 0;
    const expertsLoc = lMoe > 0 ? Math.floor(nExp / ep) : 0;
    const denseFfnParams = 3 * H * Math.max(
      1, Math.ceil((p.intermediate_size || 0) / effTp)
    );
    const sharedExpParams = 3 * H * Math.ceil(
      (p.shared_expert_intermediate_size || 0) / effTp
    );
    const iExpLoc = lMoe > 0 ? Math.ceil(
      (p.moe_intermediate_size || 0) / Math.max(1, Math.floor(effTp / epTp))
    ) : 0;
    const routedExpParams = 3 * (p.routed_expert_hidden_size || H) * iExpLoc;
    const baseDenseParams = lAttn * attnParams + lLin * linParams +
      (layersLocal - lMoe) * denseFfnParams + lMoe * sharedExpParams;
    const wStatic = baseDenseParams * bwW + lMoe * H * nExp * 2.0;
    const expBytes = routedExpParams * bwE;
    const kvComp = Math.min(1.0, Math.max(1e-6, p.kv_compress_factor ?? 1.0));
    const resStreams = Math.max(1, p.residual_streams || 1);
    const kvRawTok = kvLora ? (kvLora + qkRope) : 2.0 * kvhLoc * hd;
    const kvBytesPerTok = (kvRawTok * bwKv * kvComp) / kvSplit;
    const kvReadBytes = lAttn * batchLocal * effCtx * kvBytesPerTok;
    const recRwBytes = 2.0 * lLin * batchLocal *
      (zDim * kd + qkvLin * (p.linear_conv_kernel || 0)) * bwSt;
    const vmemUsable = ch.vmem_usable_bytes;
    const rawVmemMib = wl.vmemMib !== undefined
      ? wl.vmemMib : (wl.vmem_budget_mib !== undefined
      ? wl.vmem_budget_mib : undefined);
    const rpObj = wl.ringPlan || wl.ring_plan;
    const spillRes = (rpObj && rpObj.compiler_spill_reserve_bytes) || 0;
    const fitHead = Boolean(wl.vmemFitHeadroom || wl.vmem_fit_headroom);
    const baseVmemBytes = (rawVmemMib !== undefined && rawVmemMib !== null)
      ? Math.trunc(rawVmemMib * 1048576)
      : (wl.tiles ? (wl.tiles.vmem_mib * 1048576) : vmemUsable);
    const vmemBudget = Math.min(
      vmemUsable, Math.max(0, baseVmemBytes - (fitHead ? spillRes : 0))
    );
    const ffnChoose = lMoe > 0 ? sharedExpParams : denseFfnParams;
    const layerDenseBytes = Math.trunc(
      (Math.max(attnParams, linParams) + ffnChoose) * bwW
    );
    const tPrefill = wl.phase === 'prefill'
      ? activeTokens : (wl.phase === 'mixed' ? chunkTokens : 0);
    const dMla = (activeTokens - tPrefill) * (2 * kvLora + qkRope) +
      tPrefill * (qkNope + qkRope + hd);
    const attnCtx = (wl.phase === 'prefill' && tPrefill > 1)
      ? (Math.max(0, effCtx - tPrefill) + (tPrefill + 1) / 2.0) : effCtx;
    const flopsKv = 2.0 * lAttn * attnCtx * kvComp *
      (kvLora ? nhLoc * dMla : 2.0 * activeTokens * qDim);
    const routerFlops = lMoe * H * nExp;
    const flopsDense = 2.0 * activeTokens * (baseDenseParams + routerFlops) + flopsKv;
    const flopsMoe = expertsLoc > 0
      ? (2.0 * activeTokens * epDp * lMoe * topK *
         routedExpParams * expertsLoc / nExp) : 0.0;
    const bufSlots = wl.dmaBuffers || wl.dma_buffers || 2;
    const dBank = (wl.tiles && wl.tiles.bank)
      ? wl.tiles.bank : ((rpObj && rpObj.dense_slot_bytes)
      ? rpObj.dense_slot_bytes : (wl.denseSlotBytes || CORE_LIM.bank));
    const eBank = (wl.tiles && wl.tiles.ebank)
      ? wl.tiles.ebank : ((rpObj && rpObj.expert_slot_bytes)
      ? rpObj.expert_slot_bytes : (wl.expertSlotBytes || CORE_LIM.ebank));
    const slDenseStaging = Math.min(bufSlots * dBank, bufSlots * layerDenseBytes);
    const slExpertStaging = Math.min(bufSlots * eBank, Math.trunc(bufSlots * expBytes));
    const firstSlot = Math.min(
      dBank, Math.max(262144, Math.floor(layerDenseBytes / 4))
    );
    const gqaScratch = lAttn > 0
      ? (2 * 128 * (4 * Math.max(hd, kvLora)) + 2 * 128 * 128 * 4) : 0;
    const gdnScratch = lLin > 0 ? (4 * Math.min(vhLoc, 8) * kd * vd) : 0;
    const ffnScratch = lMoe > 0 ? (2 * 128 * iExpLoc * 4) : (2 * 128 * 1024 * 4);
    const vmemParts = computePerTokenVmem(wl, p);
    const defCollUs = (ch.collective_latency_s || 1e-6) * 1e6;
    const bracketHi = ch.collective_latency_us_bracket
      ? ch.collective_latency_us_bracket[1] : defCollUs;
    const collUs = wl.collHopUs !== undefined ? wl.collHopUs : defCollUs;
    const collHiUs = wl.calibrated
      ? Math.max(collUs, wl.collHopHiUs ?? collUs)
      : Math.max(collUs, wl.collHopHiUs ?? bracketHi);
    const latColl = collUs * 1e-6, latCollHi = collHiUs * 1e-6;
    const arInfo = allReduceCost(tokResBytes, effTp, latColl, ch, wl);
    const numAr = (p.ar_slots_per_layer || 2) * layersLocal;
    const bwHbm = ch.hbm_bw_bytes_per_s;
    const peakW = peakFlops(p.weight_dtype, ch, p.act_dtype);
    const peakE = peakFlops(p.expert_dtype || p.weight_dtype, ch, p.act_dtype);
    const bf16Peak = ch.bf16_flops_per_s || peakW;
    const rFloor = rowFloor(lMoe > 0 ? p.expert_dtype : p.weight_dtype, ch);
    const iciEg = ch.ici_egress_bw_bytes_per_s;
    const iciLink = ch.ici_link_bw_bytes_per_s;
    const a2aTime = epDp > 1
      ? (2.0 * (epDp - 1) / epDp * Math.min(Math.max(1, topK), epDp) *
         tokResBytes / iciEg)
      : 0.0;
    const ppStageTime = pp > 1
      ? (tokResBytes / iciLink + latColl) : 0.0;
    const pcpRingTime = (effPcp > 1 && lAttn > 0)
      ? ((effPcp - 1) * lAttn * chunkTokens * (kvBytesPerTok * kvSplit) / iciLink)
      : 0.0;
    const wlLg = wl.layerGroups || wl.layer_groups;
    const lgArr = (Array.isArray(wlLg) && wlLg.length > 0)
      ? wlLg : (p.layer_groups || []);
    const presetGroups = lgArr.length > 0
      ? Math.ceil(lgArr.length * layersLocal / L) : 0;
    const segLayers = wl.segmentLayers ?? wl.segment_layers ?? 0;
    const hasSeg = segLayers > 0 && segLayers < layersLocal;
    const segCount = hasSeg ? Math.ceil(layersLocal / segLayers) : 1;
    const noWholeStep = pp > 1 || segCount > 1 || lgArr.length > 1 ||
      L === 1 || Boolean(wl.planError);
    const hbmUsed = wStatic + lMoe * expertsLoc * expBytes +
      kvReadBytes + 0.5 * recRwBytes;
    const hbmCapBytes = ch.hbm_capacity_bytes;
    const hbmUtilCap = wl.hbmUtilizationCap ?? wl.hbm_utilization_cap ?? 0.80;
    const hbmPoolCapBytes = hbmUtilCap * hbmCapBytes;
    const hbmFits = hbmUsed <= hbmCapBytes;
    const invalidReasons = validateWorkload(p, ch, wl);
    const hbmPoolOverflow = invalidReasons.includes('hbm_pool_overflow');
    const vmemAlias = wl.vmemAlias !== undefined
      ? wl.vmemAlias : (wl.vmem_alias !== undefined ? wl.vmem_alias : true);
    const moeMode = wl.moeMode || wl.moe_mode || 'masked_on_chip';
    const lgBox = resolveDispatchBox(wl);
    const etaDense = wl.etaDense ??
      (wl.efficiency && wl.efficiency.eta_dense) ?? ch.eta_dense ?? 0.72;
    const etaMoeGrp = wl.etaMoeGrouped ??
      (wl.efficiency && wl.efficiency.eta_moe_grouped) ??
      ch.eta_moe_grouped ?? 0.50;
    const etaHbm = wl.etaHbm ??
      (wl.efficiency && wl.efficiency.eta_hbm) ?? ch.eta_hbm ?? 0.82;
    const etaIci = wl.etaIci ??
      (wl.efficiency && wl.efficiency.eta_ici) ?? ch.eta_ici ?? 0.75;

    // The single-call row limit floored to whole requests (0: no declared limit). A limit
    // below one request is an invalid workload (validateWorkload), not "no limit".
    const tokPerReq = requestRows(wl);
    const singleCallRows =
      Math.floor(singleCallRowLimit(p, wl, effTp) / tokPerReq) * tokPerReq;
    // Above the limit the kernel serves microbatches, and every call streams its weights
    // again: full calls run first, then the remainder, each priced as its own call.
    function microbatchedWholeStep() {
      const perCall = singleCallRows / tokPerReq;
      const fullCalls = Math.floor(batchLocal / perCall), restReq = batchLocal % perCall;
      const callScope = n => evaluateWorkload(
        Object.assign({}, wl, { batch: n * dp }), p, ch
      ).scopes.find(s => s.scope === 'whole_step');
      const full = callScope(perCall), rest = restReq > 0 ? callScope(restReq) : null;
      const calls = Array(fullCalls).fill(full).concat(rest ? [rest] : []);
      const sum = (key, digits) => +calls.reduce((a, c) => a + c[key], 0).toFixed(digits);
      // Support belongs to the whole step: its total rows are checked against the rows
      // a declaration serves through split calls.
      const stepSupport = evaluateSupport('whole_step', p, wl, ch, {
        activeTokens, effTp, epDp, planError: wl.planError || ''
      });
      return Object.assign({}, full, {
        support: stepSupport.support, blockers: stepSupport.blockers,
        feasible: calls.every(c => c.feasible),
        reason: (calls.find(c => !c.feasible) || full).reason,
        nCalls: calls.length,
        ops_per_step: sum('ops_per_step', 0), n_splits: sum('n_splits', 0),
        rawStepUs: fullCalls * full.rawStepUs + (rest ? rest.rawStepUs : 0.0),
        pureStepUs: fullCalls * full.pureStepUs + (rest ? rest.pureStepUs : 0.0),
        w_step_gb: sum('w_step_gb', 4), w_step_strict_gb: sum('w_step_strict_gb', 5),
        act_hbm_mb: sum('act_hbm_mb', 3),
        compute_us: sum('compute_us', 2), chain_us: sum('chain_us', 2),
        hbm_us: sum('hbm_us', 2), comm_us: sum('comm_us', 2),
        fill_drain_us: sum('fill_drain_us', 2), repass_spill_us: sum('repass_spill_us', 2),
        launch_us: sum('launch_us', 2), roof_floor_us: sum('roof_floor_us', 2),
        step_bound_us: sum('step_bound_us', 2),
        step_floor_strict_us: sum('step_floor_strict_us', 2),
        step_achievable_us: sum('step_achievable_us', 2),
        step_achievable_upper_us: sum('step_achievable_upper_us', 2),
        step_upper_us: sum('step_upper_us', 2)
      });
    }

    const scopes = [];
    for (const sc of ['per_op', 'single_layer', 'layer_group', 'whole_step']) {
      if (sc === 'whole_step' && singleCallRows > 0 && singleCallRows < activeTokens) {
        scopes.push(microbatchedWholeStep());
        continue;
      }
      let feasible = true, reason = 'ok';
      if (sc === 'layer_group' && presetGroups === 0 && pp === 1 && segCount === 1) {
        feasible = false;
        reason = 'No layer grouping: needs a model preset, pp > 1, or segment_layers';
      } else if (sc === 'whole_step' && noWholeStep) {
        feasible = false;
        reason = wl.planError ||
          'Bounded by layer groups, single layers, or unmodeled layer family';
      }
      let staticBytes = 0, actBytes = 0, nOps = 1, tMax = 1, nSplits = 1;
      let denseStage = 0, expertStage = 0, opScratch = 0, padBytes = 0;
      let selNbd = 0, selNbe = 0;
      if (sc === 'per_op') {
        denseStage = Math.max(1048576, slDenseStaging);
        staticBytes = denseStage;
        const perOpAlpha = 4 * H;
        if (staticBytes + perOpAlpha > vmemBudget) {
          tMax = 0;
          feasible = false;
          reason = 'per_op_vmem_overflow';
        } else {
          const perOpMax = Math.floor((vmemBudget - staticBytes) / perOpAlpha);
          tMax = Math.max(activeTokens, perOpMax);
        }
        nSplits = 1; nOps = 2 * layersLocal;
        actBytes = 4.0 * resStreams * layersLocal * tokResBytes;
      } else {
        const isSl = sc === 'single_layer';
        const isSlOrGrp = isSl || (sc === 'layer_group' && presetGroups > 0);
        const mul = sc === 'layer_group' ? (presetGroups || segCount) : 1;
        const sumScratch = gqaScratch + gdnScratch + ffnScratch;
        opScratch = vmemAlias
          ? Math.max(gqaScratch, gdnScratch, ffnScratch) : sumScratch;
        const alpha = vmemParts.totalPerTokenBytes;
        let minStaticBytes = 0;
        if (isSlOrGrp) {
          selNbd = bufSlots;
          selNbe = lMoe > 0 ? bufSlots : 0;
          denseStage = slDenseStaging;
          expertStage = lMoe > 0 ? slExpertStaging : 0;
          padBytes = 0;
          staticBytes = denseStage + expertStage + opScratch;
          minStaticBytes = staticBytes;
        } else {
          padBytes = 2097152;
          const ringSel = selectAdaptiveRing(
            vmemBudget, opScratch + padBytes, alpha, activeTokens,
            lMoe > 0, wl, dBank, eBank
          );
          selNbd = ringSel.nbd;
          selNbe = ringSel.nbe;
          denseStage = selNbd * dBank;
          expertStage = selNbe * eBank;
          staticBytes = ringSel.ringB + opScratch + padBytes;
          minStaticBytes = ringSel.minRingB + opScratch + padBytes;
        }
        actBytes = 2.0 * resStreams * (isSl ? layersLocal : mul) * tokResBytes;
        const subMul = (effTp * ep > 1 && (lAttn + lLin) > 0) ? 3 : 1;
        nOps = isSl ? layersLocal : (
          sc === 'layer_group' && presetGroups > 0 ? mul : mul * subMul
        );
        tMax = (minStaticBytes + alpha > vmemBudget)
          ? 0 : Math.max(1, Math.floor((vmemBudget - minStaticBytes) / alpha));
        // A declared single-call limit caps rows per call below the VMEM ceiling.
        if (sc === 'whole_step' && singleCallRows > 0) tMax = Math.min(tMax, singleCallRows);
        nSplits = tMax === 0 ? activeTokens : Math.ceil(activeTokens / tMax);
        if (tMax === 0) { feasible = false; reason = 'Weight buffers exceed VMEM'; }
      }
      const isFusedMulti = sc === 'whole_step' || sc === 'layer_group';
      const tiled = isFusedMulti && nSplits > 1;
      // A fused scope above its row limit and a single-layer kernel stream weights once:
      // their row tiles move the residual through HBM at every layer instead.
      const oneWeightPass = tiled || sc === 'single_layer';
      const weightPasses = oneWeightPass ? 1 : nSplits;
      const tileTokens = Math.ceil(activeTokens / Math.max(1, nSplits));
      const isGrpMode = moeMode === 'grouped_on_chip';
      const canGroupMoe = !isFusedMulti || epDp === 1 ||
        ((effTp * epDp <= 8) && isGrpMode);
      const moeDecl = findDeclaration(p, sc, effTp, wl.phase);
      const declLayout = moeDecl ? String(moeDecl.moe_layout || '') : '';
      // A declared megakernel runs the expert layout its deployment states.
      const groupedMoe = declLayout === 'grouped' || (declLayout !== 'masked' &&
        canGroupMoe && (sc === 'per_op' || isGrpMode || tileTokens * epDp >= rFloor));
      const moeTok = (groupedMoe && sc !== 'single_layer')
        ? activeTokens : tileTokens;
      let wStep = wStatic, moePadFactor = 1.0;
      if (expertsLoc > 0) {
        const tEp = moeTok * epDp;
        const eHit = expertsHit(tEp, topK, nExp, expertsLoc);
        const pairs = Math.max(1.0, tEp * topK * expertsLoc / nExp);
        const avgM = Math.max(1.0, pairs / eHit);
        const hit = oneWeightPass
          ? expertsHit(activeTokens * epDp, topK, nExp, expertsLoc) : eHit;
        wStep = wStatic + lMoe * hit * expBytes;
        const maskedPhi = Math.max(1.0, (Math.floor((tEp + 15) / 16) * 16 * eHit) / pairs);
        moePadFactor = groupedMoe ? (Math.ceil(avgM / rFloor) * rFloor / avgM) : maskedPhi;
        // A per-op kernel may run each expert on all rows (masked) or on its own rows
        // (grouped). The cheaper padding of the two applies.
        if (sc === 'per_op') moePadFactor = Math.min(moePadFactor, maskedPhi);
      }
      // The per-layer residual traffic already holds the call-boundary activations.
      if (tiled) {
        actBytes = Math.max(actBytes, 2.0 * resStreams * layersLocal * tokResBytes);
      }
      const xlaArFactor = (sc === 'per_op' && effTp > 1) ? 2 : 1;
      const xlaHbmCollBytes = (sc === 'per_op' && effTp > 1)
        ? 2.0 * numAr * tokResBytes : 0.0;
      const overlappedComm = isFusedMulti && nSplits === 1;
      const arAndPcpS = numAr * (
        arInfo.xferS + nSplits * xlaArFactor * arInfo.latS
      ) + xlaHbmCollBytes / bwHbm + pcpRingTime;
      const a2aSyncS = (lMoe > 0 && epDp > 1)
        ? lMoe * (a2aTime + 2 * nSplits * xlaArFactor * latColl) : 0.0;
      const dcnSyncS = numAr * nSplits * xlaArFactor * arInfo.interLatS;
      const syncStageS = overlappedComm
        ? (dcnSyncS + ppStageTime + a2aSyncS)
        : (isFusedMulti ? (arAndPcpS + ppStageTime + a2aSyncS) : 0.0);
      const commS = overlappedComm
        ? (arAndPcpS - dcnSyncS)
        : (isFusedMulti ? 0.0 : (arAndPcpS + ppStageTime + a2aSyncS));
      const effStateBytes = kvReadBytes + recRwBytes;
      let fillDrainS = (nOps - 1) * (firstSlot / bwHbm + 2.0 * latColl) + syncStageS;
      if (tiled) {
        // Each tile boundary adds a sync hop per layer. The residual bytes moved there
        // are already in actBytes.
        fillDrainS += (nSplits - 1) * 2 * layersLocal * 2.0 * latColl;
      }
      const vecRate = ch.vector_f32_ops_per_s || (ch.bf16_flops_per_s / 128.0);
      const subchannelKThresh = DATA.subchannel_block_k_threshold || 512;
      const subchannelVecOps = DATA.subchannel_vec_ops_per_block || 2.0;
      const weightBlockK = Number(p.weight_block_k || 0);
      const subchannelVecDense = (weightBlockK > 0 && weightBlockK < subchannelKThresh)
        ? (subchannelVecOps * activeTokens * baseDenseParams / weightBlockK)
        : 0.0;
      const ffnDenseInterLocal = Math.max(1, Math.ceil((p.intermediate_size || 0) / effTp));
      const sharedInterLocal = (p.shared_expert_intermediate_size || 0) > 0
        ? Math.ceil(p.shared_expert_intermediate_size / effTp) : 0;
      const vecOpsDense = 8.0 * Math.max(1, p.residual_streams || 1) * activeTokens * (p.hidden_size / effTp) * layersLocal +
        4.0 * activeTokens * ffnDenseInterLocal * (layersLocal - lMoe) +
        4.0 * activeTokens * sharedInterLocal * lMoe + subchannelVecDense;
      const routedBytesStreamed = weightPasses * (wStep - wStatic);
      const streamedRoutedParams = Math.max(0.0, routedBytesStreamed / Math.max(1e-6, bwE));
      const epTp = gcd(wl.ep || 1, effTp);
      const iExpLocal = (p.num_experts > 0 && lMoe > 0)
        ? Math.ceil((p.moe_intermediate_size || 0) / Math.max(1, Math.floor(effTp / epTp)))
        : 0;
      const moeTokensLocal = activeTokens * epDp * expertsLoc / Math.max(1, nExp);
      const isFp4Expert = ['nvfp4', 'mxfp4', 'fp4', 'int4'].includes(String(p.expert_dtype || '').toLowerCase());
      const subchannelVecMoe = (weightBlockK > 0 && weightBlockK < subchannelKThresh)
        ? ((moePadFactor * flopsMoe) / weightBlockK)
        : 0.0;
      const vecOpsMoe = 4.0 * moeTokensLocal * lMoe * topK * iExpLocal + (isFp4Expert ? 2.0 * streamedRoutedParams : 0.0) + subchannelVecMoe;
      const actDt = String(p.act_dtype || '').toLowerCase();
      const kvDt = String(p.kv_dtype || '').toLowerCase();
      const attnIsBf16 = actDt !== 'fp8' || kvDt !== 'fp8';
      const tMatmulDense = (attnIsBf16 && flopsKv > 0)
        ? ((flopsDense - flopsKv) / peakW + flopsKv / bf16Peak)
        : (flopsDense / peakW);
      const tMatmulDenseUpper = ((attnIsBf16 || isPrefOrMix) && flopsKv > 0)
        ? ((flopsDense - flopsKv) / peakW + flopsKv / bf16Peak)
        : tMatmulDense;
      const tDense = Math.max(tMatmulDense, vecOpsDense / vecRate);
      const tDenseUpper = Math.max(tMatmulDenseUpper, vecOpsDense / vecRate);
      const tMoe = (moePadFactor * flopsMoe) / peakE;
      const tVecMoe = vecOpsMoe / vecRate;
      const tMatmul = tDense + tMoe;
      const tMatmulHi = tDenseUpper + tMoe;
      const tHbm = (weightPasses * wStep + effStateBytes + actBytes) / bwHbm;
      // Routed-expert bytes are requested after each layer's router, so they stream
      // under neither the dense compute nor the collectives.
      const routedHbmS = weightPasses * (wStep - wStatic) / bwHbm;
      const chainS = routedChain(tDense, routedHbmS, tMoe, tVecMoe);
      const roofCoreS = coreTime(tHbm, chainS, commS, isFusedMulti);
      const roofUs = (roofCoreS + fillDrainS) * 1e6;
      const baseAi = (flopsDense + flopsMoe) / Math.max(1.0, wStep + effStateBytes);
      const repassUs = (baseAi >= peakW / bwHbm && weightPasses > 1)
        ? ((actBytes + (nSplits - 1) * wStep) / bwHbm * 1e6) : 0.0;
      const pureStepUs = roofUs + repassUs;
      const launchUs = nOps * lgBox.lo, stepUs = pureStepUs + launchUs;
      const etaMoe = (groupedMoe && lMoe > 0) ? etaMoeGrp : etaDense;
      const tMatmulDenseAch = (attnIsBf16 && flopsKv > 0)
        ? (((flopsDense - flopsKv) / peakW) / Math.max(1e-3, etaDense) +
           (flopsKv / bf16Peak) / Math.max(1e-3, etaDense))
        : (flopsDense / peakW) / Math.max(1e-3, etaDense);
      const tDenseAch = Math.max(
        tMatmulDenseAch,
        (vecOpsDense / vecRate) / Math.max(1e-3, etaDense)
      );
      const tMoeAch = ((moePadFactor * flopsMoe) / peakE) / Math.max(1e-3, etaMoe);
      const tVecMoeAch = (vecOpsMoe / vecRate) / Math.max(1e-3, etaDense);
      const tCompAch = tDenseAch + tMoeAch;
      const tHbmAch = tHbm / Math.max(1e-3, etaHbm);
      const tCommAch = commS / Math.max(1e-3, etaIci);
      const syncExtraS = isFusedMulti
        ? syncStageS * (1.0 / Math.max(1e-3, etaIci) - 1.0) : 0.0;
      const routedHbmAchS = routedHbmS / Math.max(1e-3, etaHbm);
      const chainAchS = routedChain(tDenseAch, routedHbmAchS, tMoeAch, tVecMoeAch);
      const achCoreS = coreTime(tHbmAch, chainAchS, tCommAch, isFusedMulti) + syncExtraS;
      const stepAchUs = (achCoreS + fillDrainS) * 1e6 + repassUs + launchUs;
      const arExtraS = (numAr * nSplits * xlaArFactor * arInfo.intraHops) *
        Math.max(0.0, latCollHi - latColl);
      const a2aCnt = (lMoe > 0 && epDp > 1) ? (2 * lMoe * nSplits * xlaArFactor) : 0;
      const a2aExtraS = a2aCnt * Math.max(0.0, latCollHi - latColl);
      const chainHiS = routedChain(tDenseUpper, routedHbmS, tMoe, tVecMoe);
      const roofHiCoreS = coreTime(tHbm, chainHiS, commS + arExtraS, overlappedComm) +
        a2aExtraS;
      const roofHiUs = (roofHiCoreS + fillDrainS) * 1e6;
      const stepHiUs = Math.max(stepUs, roofHiUs + repassUs + nOps * lgBox.hi);
      const tMatmulDenseUpperAch = ((attnIsBf16 || isPrefOrMix) && flopsKv > 0)
        ? (((flopsDense - flopsKv) / peakW) / Math.max(1e-3, etaDense) +
           (flopsKv / bf16Peak) / Math.max(1e-3, etaDense))
        : (flopsDense / peakW) / Math.max(1e-3, etaDense);
      const tDenseHiAch = Math.max(
        tMatmulDenseUpperAch,
        (vecOpsDense / vecRate) / Math.max(1e-3, etaDense)
      );
      const tCommHiAch = (commS + arExtraS) / Math.max(1e-3, etaIci);
      const chainHiAchS = routedChain(tDenseHiAch, routedHbmAchS, tMoeAch, tVecMoeAch);
      const achHiCoreS = coreTime(tHbmAch, chainHiAchS, tCommHiAch, overlappedComm) +
        syncExtraS + (a2aExtraS / Math.max(1e-3, etaIci));
      const stepAchHiUs = Math.max(
        stepAchUs, stepHiUs,
        (achHiCoreS + fillDrainS) * 1e6 + repassUs + nOps * lgBox.hi
      );
      const eStrict = expertsLoc > 0
        ? Math.min(expertsLoc, Math.max(1.0, topK * expertsLoc / Math.max(1, nExp)))
        : 0.0;
      const wStepStrict = expertsLoc > 0
        ? (wStatic + lMoe * eStrict * expBytes) : wStatic;
      const tMatmulDenseStrict = (attnIsBf16 && flopsKv > 0)
        ? ((flopsDense - flopsKv) / peakW + flopsKv / bf16Peak)
        : (flopsDense / peakW);
      const tMatmulStrict = tMatmulDenseStrict + flopsMoe / peakE;
      const tHbmStrict = (weightPasses * wStepStrict + effStateBytes + actBytes) / bwHbm;
      const tMoeStrict = flopsMoe / peakE;
      const routedStrictS = weightPasses * (wStepStrict - wStatic) / bwHbm;
      const routedBytesStrict = weightPasses * (wStepStrict - wStatic);
      const streamedRoutedParamsStrict = Math.max(0.0, routedBytesStrict / Math.max(1e-6, bwE));
      const vecOpsMoeStrict = 4.0 * moeTokensLocal * lMoe * topK * iExpLocal +
        (isFp4Expert ? 2.0 * streamedRoutedParamsStrict : 0.0);
      // Vector counts are per element. A 32-bit word holds up to 4 packed elements, and 8
      // when it unpacks weights below one byte, so these rates bound every dtype.
      const strictVecRate = (DATA.vector_max_elements_per_word || 4) * vecRate;
      const unpackOpsStrict = isFp4Expert ? 2.0 * streamedRoutedParamsStrict : 0.0;
      const tVecMoeStrict = (vecOpsMoeStrict - unpackOpsStrict) / strictVecRate +
        unpackOpsStrict / ((DATA.vector_sub_byte_elements_per_word || 8) * vecRate);
      const tDenseStrict = Math.max(
        tMatmulStrict - tMoeStrict, vecOpsDense / strictVecRate
      );
      const chainStrictS = routedChain(
        tDenseStrict, routedStrictS, tMoeStrict, tVecMoeStrict
      );
      // The strict floor prices all-reduce transfer at the egress sum. That transfer sits
      // in commS, or in the sync stage of fillDrainS when a fused scope runs several passes.
      const arStrictSavingS = numAr * (arInfo.xferS - arInfo.xferStrictS);
      const commInCore = overlappedComm || !isFusedMulti;
      const commStrictS = commS - (commInCore ? arStrictSavingS : 0.0);
      const fillDrainStrictS = fillDrainS - (commInCore ? 0.0 : arStrictSavingS);
      const roofCoreStrictS = coreTime(tHbmStrict, chainStrictS, commStrictS, isFusedMulti);
      const repassStrictUs = (baseAi >= peakW / bwHbm && weightPasses > 1)
        ? ((actBytes + (nSplits - 1) * wStepStrict) / bwHbm * 1e6) : 0.0;
      const stepFloorStrictUs = (roofCoreStrictS + fillDrainStrictS) * 1e6 +
        repassStrictUs + launchUs;
      const supInfo = evaluateSupport(sc, p, wl, ch, {
        activeTokens, effTp, epDp, planError: wl.planError || ''
      });
      const resTok = Math.min(activeTokens, Math.max(1, tMax));
      const aliasDiff = vmemParts.unaliasedOpBytes - vmemParts.opBytes;
      scopes.push({
        scope: sc, feasible, reason, ops_per_step: nOps, t_vmem_max: tMax,
        n_splits: nSplits, tile_tokens: tileTokens,
        selected_dense_slots: selNbd, selected_expert_slots: selNbe,
        rawStepUs: stepUs, pureStepUs,
        vmem_static_mib: +(staticBytes / 1048576).toFixed(3),
        dense_stage_mib: +(denseStage / 1048576).toFixed(2),
        expert_stage_mib: +(expertStage / 1048576).toFixed(2),
        op_scratch_mib: +(opScratch / 1048576).toFixed(2),
        pad_mib: +(padBytes / 1048576).toFixed(2),
        spine_dyn_mib: +((resTok * vmemParts.spineBytes) / 1048576).toFixed(2),
        op_dyn_mib: +((resTok * vmemParts.opBytes) / 1048576).toFixed(2),
        alias_saved_kib: +(aliasDiff / 1024).toFixed(1),
        moe_flop_factor: +moePadFactor.toFixed(3), w_step_gb: +(wStep / 1e9).toFixed(4),
        w_step_strict_gb: +(wStepStrict / 1e9).toFixed(5),
        act_hbm_mb: +(actBytes / 1e6).toFixed(3),
        rowsPerCall: sc === 'whole_step' ? singleCallRows : 0,
        compute_us: +((tDense + Math.max(tMoe, tVecMoe)) * 1e6).toFixed(2),
        chain_us: +(chainS * 1e6).toFixed(2),
        hbm_us: +(tHbm * 1e6).toFixed(2), comm_us: +(commS * 1e6).toFixed(2),
        fill_drain_us: +(fillDrainS * 1e6).toFixed(2),
        repass_spill_us: +repassUs.toFixed(2),
        launch_us: +launchUs.toFixed(2), roof_floor_us: +roofUs.toFixed(2),
        step_bound_us: +stepUs.toFixed(2),
        step_floor_strict_us: +stepFloorStrictUs.toFixed(2),
        step_achievable_us: +stepAchUs.toFixed(2),
        step_achievable_upper_us: +stepAchHiUs.toFixed(2),
        step_upper_us: +stepHiUs.toFixed(2),
        support: supInfo.support,
        blockers: supInfo.blockers
      });
    }
    const feas = scopes.filter(s => s.feasible);
    const candidates = hbmFits
      ? (feas.length ? feas : [scopes[0]])
      : [scopes[0]];
    candidates.sort((a, b) => (a.rawStepUs - b.rawStepUs) ||
      (a.ops_per_step - b.ops_per_step));
    const winner = candidates[0], runnerUp = candidates[1] || winner;
    const supFeas = candidates.filter(s =>
      s.support === 'baseline' ||
      s.support === 'within_core_limits' ||
      s.support === 'declared'
    );
    const divReasons = invalidReasons.filter(r => r !== 'hbm_pool_overflow');
    const isPhysValid = hbmFits && divReasons.length === 0;
    const recWinner = (isPhysValid && supFeas.length) ? supFeas[0] : scopes[0];
    const fastestScope = hbmFits ? winner.scope : 'per_op';
    const recScope = recWinner.scope;
    const perOp = scopes[0];
    const compilerProofGainUs = perOp.pureStepUs - winner.pureStepUs;
    const gapPr = wl.launchGapUs !== undefined ? wl.launchGapUs : lgBox.lo;
    const dispatchBubbleGainUs = (perOp.ops_per_step - winner.ops_per_step) * gapPr;
    const compilerProofGainPct = +(100.0 * Math.max(0.0, compilerProofGainUs) /
      Math.max(1e-6, perOp.pureStepUs)).toFixed(2);
    const achievableGainPct = +(100.0 * (perOp.step_achievable_us - winner.step_achievable_us) /
      Math.max(1e-6, perOp.step_achievable_us)).toFixed(2);
    const tokensPerChip = (activeTokens * effPcp * (ch.devices_per_chip || 1)) /
      Math.max(1, effTp * pp);
    const canRun = isPhysValid && recWinner.feasible;
    const tpsChip = canRun
      ? (tokensPerChip / Math.max(1e-9, recWinner.step_bound_us * 1e-6))
      : 0.0;
    const winFusedMulti = fastestScope === 'whole_step' || fastestScope === 'layer_group';
    const winCanGroup = !winFusedMulti || epDp === 1 ||
      ((effTp * epDp <= 8) && (moeMode === 'grouped_on_chip'));
    const tPerEp = Math.ceil(activeTokens / Math.max(1, winner.n_splits)) * epDp;
    const moeModeRec = lMoe === 0 ? 'dense_mlp'
      : (fastestScope === 'per_op' ? 'grouped_in_hbm'
      : (winCanGroup && (moeMode === 'grouped_on_chip' || tPerEp >= rowFloor(p.expert_dtype, ch))
         ? 'grouped_on_chip' : 'masked_on_chip'));
    const marginPct = runnerUp && runnerUp.scope !== winner.scope
      ? ((runnerUp.step_bound_us - winner.step_bound_us) /
         Math.max(0.01, runnerUp.step_bound_us)) * 100.0 : 100.0;
    const wsScratch = vmemAlias
      ? Math.max(gqaScratch, gdnScratch, ffnScratch)
      : (gqaScratch + gdnScratch + ffnScratch);
    const wsAlpha = vmemParts.totalPerTokenBytes;
    const extraSlots = bufSlots - 2;
    const wideRing = (CORE_LIM.nbd + extraSlots) * dBank +
      (lMoe > 0 ? (CORE_LIM.nbe + extraSlots) * eBank : 0);
    const shrunkRing = bufSlots * dBank + (lMoe > 0 ? bufSlots * eBank : 0);
    const wsNonRing = wsScratch + 2097152;
    const tWideMax = (wideRing + wsNonRing + wsAlpha > vmemBudget)
      ? 0 : Math.max(1, Math.floor((vmemBudget - wideRing - wsNonRing) / wsAlpha));
    const tShrunkMax = (shrunkRing + wsNonRing + wsAlpha > vmemBudget)
      ? 0 : Math.max(1, Math.floor((vmemBudget - shrunkRing - wsNonRing) / wsAlpha));
    return {
      activeTokens, effCtx, epDp, scopes, winner, recWinner, runnerUp, fastestScope,
      recScope, tpsChip, hbmFits, hbmPoolOverflow, isPhysValid,
      hbmUsedBytes: hbmUsed, hbmPoolCapBytes,
      invalidReasons, nonHomogeneousReasons: p.non_homogeneous_reasons || [],
      compilerProofGainUs, dispatchBubbleGainUs,
      compilerProofGainPct, achievableGainPct, tWideMax, tShrunkMax,
      ridgeFb: peakW / bwHbm, moeModeRec, marginPct,
      launchGapLoUs: lgBox.lo, launchGapHiUs: lgBox.hi, launchGapPriorUs: gapPr,
      vmemBudgetMib: +(vmemBudget / 1048576).toFixed(2), vmemParts
    };
  }
    function evaluateScenario(params) {
      if (!params) return null;
      const p = params.profile || PROF;
      const ch = params.chip || CHIP;
      const wl = params.wl || params;
      const inv = validateWorkload(p, ch, wl);
      if (params.only_validate) {
        return { invalid_reasons: inv, scopes: {} };
      }
      const res = evaluateWorkload(wl, p, ch);
      const scopesOut = {};
      if (res && res.scopes) {
        for (const sc of res.scopes) {
          scopesOut[sc.scope] = sc;
        }
      }
      return { invalid_reasons: inv, scopes: scopesOut, full: res };
    }

    return {
      DATA, PROF, CHIP, REGIMES, roundEven, gcd, peakFlops, rowFloor,
      allReduceCost, computePerTokenVmem,
      expertsHit, selectAdaptiveRing, validateWorkload, evaluateWorkload,
      evaluateScenario
    };
  }

  global.__MK_CREATE_ADVISOR_ENGINE = createAdvisorEngine;
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { createAdvisorEngine };
  }
  if (typeof window !== 'undefined') {
    window.__MK_CREATE_ADVISOR_ENGINE = createAdvisorEngine;
    if (typeof document !== 'undefined' && document && typeof document.getElementById === 'function') {
      const el = document.getElementById('advisor-data');
      if (el && el.textContent) {
        window.__MK_ADVISOR_ENGINE = createAdvisorEngine(JSON.parse(el.textContent));
      }
    }
  }
})(typeof globalThis !== 'undefined' ? globalThis : this);
