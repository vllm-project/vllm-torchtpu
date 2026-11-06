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
"""Client-side ES2022 roofline and on-chip VMEM physics evaluator."""

from __future__ import annotations


def render_workbench_engine_js() -> str:
    """Return dependency-free ES2022 evaluator script hydrated from #advisor-data."""
    return """
(function() {
  const rawEl = document.getElementById('advisor-data');
  if (!rawEl) return;
  const DATA = JSON.parse(rawEl.textContent || '{}');
  const PROF = DATA.model_profile || {}, CHIP = DATA.chip_spec || {};
  const REGIMES = DATA.regimes || [];
  const SUPPORTED_FAM = new Set([
    'gqa', 'mla', 'dsa', 'swa', 'gdn', 'kda', 'dense_mlp', 'moe', 'mtp', 'ple'
  ]);
  const DTYPE_BYTES = {
    fp32: 4.0, bf16: 2.0, fp16: 2.0, fp8: 1.0, int8: 1.0,
    nvfp4: 0.5625, mxfp4: 0.53125, fp4: 0.5625
  };
  function gcd(a, b) {
    let x = Math.abs(Math.trunc(a)), y = Math.abs(Math.trunc(b));
    while (y !== 0) { const t = y; y = x % y; x = t; }
    return x || 1;
  }
  function peakFlops(dtype) {
    const dt = String(dtype || 'fp8').toLowerCase();
    if (dt === 'fp8' || dt === 'nvfp4' || dt === 'mxfp4' || dt === 'fp4') {
      return CHIP.fp8_flops_per_s || 2.3e15;
    }
    if (dt === 'int8') return CHIP.int8_flops_per_s || 1.155e15;
    if (dt === 'fp32') return 0.5 * (CHIP.bf16_flops_per_s || 1.155e15);
    return CHIP.bf16_flops_per_s || 1.155e15;
  }
  function rowFloor(dtype) {
    const dt = String(dtype || 'fp8').toLowerCase();
    const map = CHIP.matmul_row_tile_by_dtype || {};
    if (map[dt]) return map[dt];
    const hasFp8 = CHIP.native_ocp_fp8 !== false;
    return (hasFp8 && (dt === 'fp8' || dt.includes('fp4'))) ? 32 : 16;
  }
  function allReduceCost(msgBytes, effTp, collLatS) {
    if (effTp <= 1 || msgBytes <= 0) return { xferS: 0, latS: 0, hops: 0 };
    const iciBw = CHIP.ici_egress_bw_bytes_per_s || 3e11;
    const hops = Math.max(1, Math.round(Math.log2(effTp)));
    if (effTp <= 8) {
      const vol = 2.0 * (effTp - 1) / effTp * msgBytes;
      return { xferS: vol / iciBw, latS: hops * collLatS, hops };
    }
    const numHosts = Math.ceil(effTp / 8), intraVol = 1.75 * msgBytes;
    const interVol = 2.0 * (numHosts - 1) / numHosts * (msgBytes / 8.0);
    const dcnBw = 12.5 * 1073741824.0;
    const latS = 3 * collLatS + Math.round(Math.log2(numHosts)) * 5.0e-6;
    return { xferS: intraVol / iciBw + interVol / dcnBw, latS, hops };
  }
  function computePerTokenVmem(wl) {
    const H = PROF.hidden_size || 4096, tpRaw = Math.max(1, wl.tp || 1);
    const effTp = tpRaw === 1 ? Math.max(1, tpRaw, wl.dcp || 1) : tpRaw;
    const ep = Math.max(1, wl.ep || 1);
    const kvSplit = Math.max(1, wl.kvSplit || 1, wl.dcp || 1);
    const spine = 6 * H + (effTp > 1 ? 6 * H : 0);
    const nhLoc = Math.max(1, Math.ceil((PROF.num_attention_heads || 32) / effTp));
    const hd = PROF.head_dim || 128;
    let attn = 0, lin = 0, moe = 0;
    if ((PROF.kv_lora_rank || 0) > 0) {
      const qlr = PROF.q_lora_rank || 0, qkr = PROF.qk_rope_head_dim || 0;
      attn = 2 * (qlr + PROF.kv_lora_rank + qkr) + 6 * (nhLoc * hd);
    } else {
      const tpPerKv = Math.max(1, Math.floor(effTp / kvSplit));
      const nkvLoc = Math.max(1, Math.ceil((PROF.num_kv_heads || 8) / tpPerKv));
      attn = 2 * (2 * nhLoc + 2 * nkvLoc) * hd + 4 * nhLoc * hd;
    }
    if ((PROF.num_linear_layers || 0) > 0 && (PROF.linear_value_heads || 0) > 0) {
      const khL = Math.max(1, Math.ceil((PROF.linear_key_heads || 1) / effTp));
      const vhL = Math.max(1, Math.ceil(PROF.linear_value_heads / effTp));
      const kd = PROF.linear_key_dim || 128, vd = PROF.linear_value_dim || 128;
      lin = 2 * (2 * khL * kd + 2 * vhL * vd) + 4 * vhL * vd;
    }
    const nExp = PROF.num_experts || 0;
    if ((PROF.num_moe_layers || 0) > 0 && nExp > 0) {
      const topK = PROF.top_k || 1, router = 4 * nExp + 8 * topK;
      const interDiv = Math.max(1, Math.floor(effTp / gcd(ep, effTp)));
      const interLoc = Math.ceil((PROF.moe_intermediate_size || 0) / interDiv);
      const shI = PROF.shared_expert_intermediate_size || 0;
      const sharedLoc = shI > 0 ? Math.ceil(shI / effTp) : 0;
      const expH = PROF.routed_expert_hidden_size || H;
      const effK = ep > 1 ? Math.min(topK, 2 * ep) : 1;
      const grp = 8 * expH +
        Math.ceil(effK * (6 * expH + 8 * interLoc) / ep) + 8 * sharedLoc;
      const tile = wl.moeMode === 'grouped_on_chip'
        ? grp : (16 * expH + 8 * (interLoc + sharedLoc));
      moe = wl.vmemAlias ? Math.max(router + 2 * expH, tile) : (router + tile);
    } else {
      moe = 6 * H + 8 * Math.ceil((PROF.intermediate_size || 4 * H) / effTp);
    }
    const sumOp = attn + lin + moe;
    const opBytes = wl.vmemAlias ? Math.max(attn, lin, moe) : sumOp;
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
  function evaluateWorkload(wl) {
    const dtB = k => DTYPE_BYTES[String(PROF[k] || 'fp8').toLowerCase()] || 1.0;
    const bwW = dtB('weight_dtype'), bwE = dtB('expert_dtype');
    const bwKv = dtB('kv_dtype');
    const bwSt = DTYPE_BYTES[String(PROF.state_dtype || 'bf16').toLowerCase()] || 2.0;
    const L = PROF.num_layers || 1, H = PROF.hidden_size || 4096;
    const nExp = PROF.num_experts || 0, topK = PROF.top_k || 0;
    const tp = Math.max(1, wl.tp || 1), pp = Math.max(1, wl.pp || 1);
    const dp = Math.max(1, wl.dp || 1), ep = Math.max(1, wl.ep || 1);
    const dcp = Math.max(1, wl.dcp || 1), pcp = Math.max(1, wl.pcp || 1);
    const effTp = tp === 1 ? Math.max(tp, dcp) : tp;
    const kvSplit = Math.max(wl.kvSplit || 1, dcp);
    const isPrefOrMix = wl.phase === 'prefill' || wl.phase === 'mixed';
    const effPcp = isPrefOrMix ? pcp : 1;
    const batchLocal = Math.ceil(wl.batch / dp), layersLocal = Math.ceil(L / pp);
    const fallbackCtx = (wl.isl || 4096) +
      (wl.phase === 'prefill' ? 0 : Math.floor((wl.osl || 0) / 2));
    const effCtx = Math.max(1, wl.context || fallbackCtx);
    const islVal = wl.isl || wl.seqTokens || effCtx;
    const uncachedIsl = Math.max(
      1, Math.round(islVal * (1.0 - (wl.prefixHitRate || 0)))
    );
    const chunkRaw = wl.chunkSize || wl.seqTokens || uncachedIsl;
    const chunkTokens = Math.ceil(chunkRaw / effPcp);
    const specTok = wl.specTokens !== undefined
      ? wl.specTokens : Math.max(0, (wl.seqTokens || 1) - 1);
    const verifyTok = wl.phase === 'verify'
      ? (Math.max(1, specTok) + 1) : (specTok > 0 ? specTok + 1 : 1);
    const mbDenom = Math.max(1, (wl.microbatches || pp) * effPcp);
    const prefillTok = wl.chunkSize
      ? Math.ceil(wl.chunkSize / effPcp)
      : Math.ceil((batchLocal * uncachedIsl) / mbDenom);
    const decodeTok = batchLocal * (wl.phase === 'decode' ? 1 : verifyTok);
    const activeTokens = wl.phase === 'prefill'
      ? prefillTok : (decodeTok + (wl.phase === 'mixed' ? chunkTokens : 0));
    const lAttn = Math.min(
      layersLocal, Math.ceil((PROF.num_attention_layers || L) * layersLocal / L)
    );
    const lMoe = Math.min(
      layersLocal, Math.ceil((PROF.num_moe_layers || 0) * layersLocal / L)
    );
    const lLin = Math.min(
      layersLocal - lAttn,
      Math.ceil((PROF.num_linear_layers || 0) * layersLocal / L)
    );
    const tokResBytes = activeTokens * H * 2.0;
    const hd = PROF.head_dim || 128, kvLora = PROF.kv_lora_rank || 0;
    const qLora = PROF.q_lora_rank || 0, qkRope = PROF.qk_rope_head_dim || 0;
    const qkNope = PROF.qk_nope_head_dim || hd;
    const kd = PROF.linear_key_dim || 128, vd = PROF.linear_value_dim || 128;
    const nhLoc = Math.max(1, Math.ceil((PROF.num_attention_heads || 32) / effTp));
    const tpKvDiv = Math.max(1, Math.floor(effTp / kvSplit));
    const kvhLoc = Math.max(1, Math.ceil((PROF.num_kv_heads || 8) / tpKvDiv));
    const qDim = nhLoc * hd, qTot = nhLoc * (qkNope + qkRope);
    const mlaProj = (qLora ? qLora * (H + qTot) : H * qTot) +
      H * (kvLora + qkRope + qDim);
    const attnParams = kvLora
      ? (mlaProj + kvLora * nhLoc * (qkNope + hd))
      : (H * 2 * (qDim + kvhLoc * hd));
    const khLoc = lLin > 0
      ? Math.max(1, Math.ceil((PROF.linear_key_heads || 1) / effTp)) : 0;
    const vhLoc = lLin > 0
      ? Math.max(1, Math.ceil((PROF.linear_value_heads || 1) / effTp)) : 0;
    const zDim = vhLoc * vd, qkvLin = 2 * khLoc * kd + vhLoc * vd;
    const epTp = gcd(ep, effTp), epDp = Math.max(1, Math.floor(ep / epTp));
    const linParams = lLin > 0 ? H * (qkvLin + 2 * zDim) : 0;
    const expertsLoc = lMoe > 0 ? Math.floor(nExp / ep) : 0;
    const denseFfnParams = 3 * H * Math.max(
      1, Math.ceil((PROF.intermediate_size || 4 * H) / effTp)
    );
    const sharedExpParams = 3 * H * Math.ceil(
      (PROF.shared_expert_intermediate_size || 0) / effTp
    );
    const iExpLoc = lMoe > 0 ? Math.ceil(
      (PROF.moe_intermediate_size || 0) / Math.max(1, Math.floor(effTp / epTp))
    ) : 0;
    const routedExpParams = 3 * (PROF.routed_expert_hidden_size || H) * iExpLoc;
    const baseDenseParams = lAttn * attnParams + lLin * linParams +
      (layersLocal - lMoe) * denseFfnParams + lMoe * sharedExpParams;
    const wStatic = baseDenseParams * bwW + lMoe * H * nExp * 2.0;
    const expBytes = routedExpParams * bwE;
    const kvRawTok = kvLora ? (kvLora + qkRope) : 2.0 * kvhLoc * hd;
    const kvReadBytes = lAttn * batchLocal * effCtx * ((kvRawTok * bwKv) / kvSplit);
    const recRwBytes = 2.0 * lLin * batchLocal *
      (zDim * kd + qkvLin * (PROF.linear_conv_kernel || 4)) * bwSt;
    const vmemUsable = CHIP.vmem_usable_bytes || (62 * 1048576);
    const vmemBudget = Math.min(
      vmemUsable, Math.max(0, Math.trunc(wl.vmemMip * 1048576))
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
    const flopsKv = 2.0 * lAttn * attnCtx *
      (kvLora ? nhLoc * dMla : 2.0 * activeTokens * qDim);
    const routerFlops = lMoe * H * nExp;
    const flopsDense = 2.0 * activeTokens * (baseDenseParams + routerFlops) + flopsKv;
    const flopsMoe = expertsLoc > 0
      ? (2.0 * activeTokens * epDp * lMoe * topK *
         routedExpParams * expertsLoc / nExp) : 0.0;
    const denseSlotBytes = wl.denseSlotBytes || (2 * 1048576);
    const expertSlotBytes = wl.expertSlotBytes || (2 * 1048576);
    const wsDenseStaging = Math.min(28 * 1048576, 2 * layerDenseBytes) +
      (wl.dmaBuffers - 2) * denseSlotBytes;
    const wsExpertStaging = 10 * 1048576 + (wl.dmaBuffers - 2) * expertSlotBytes;
    const slDenseStaging = Math.min(4 * 1048576, layerDenseBytes);
    const slExpertStaging = Math.min(
      4 * 1048576, Math.trunc(wl.dmaBuffers * expBytes)
    );
    const firstSlot = Math.min(
      denseSlotBytes, Math.max(262144, Math.floor(layerDenseBytes / 4))
    );
    const gqaScratch = lAttn > 0 ? 3072 * Math.max(hd, kvLora) : 0;
    const gdnScratch = 16 * Math.min(vhLoc, 8) * kd * vd;
    const ffnScratch = lMoe > 0
      ? 1024 * Math.max(1, iExpLoc || H) : (2 * 1048576);
    const vmemParts = computePerTokenVmem(wl);
    const latColl = wl.collHopUs * 1e-6;
    const latCollHi = Math.max(latColl, (wl.collHopHiUs || 5.13) * 1e-6);
    const arInfo = allReduceCost(tokResBytes, effTp, latColl);
    const numAr = (PROF.ar_slots_per_layer || 2) * layersLocal;
    const bwHbm = CHIP.hbm_bw_bytes_per_s || 3.7e12;
    const peakW = peakFlops(PROF.weight_dtype);
    const peakE = peakFlops(PROF.expert_dtype || PROF.weight_dtype);
    const bf16Peak = CHIP.bf16_flops_per_s || peakW;
    const rFloor = rowFloor(lMoe > 0 ? PROF.expert_dtype : PROF.weight_dtype);
    const iciEg = CHIP.ici_egress_bw_bytes_per_s || 3e11;
    const a2aTime = epDp > 1
      ? (2.0 * (epDp - 1) / epDp * Math.max(1, topK) * tokResBytes / iciEg) : 0.0;
    const ppStageTime = pp > 1
      ? (tokResBytes / (CHIP.ici_link_bw_bytes_per_s || 1e11) + latColl) : 0.0;
    const lgArr = wl.layerGroups || [];
    const presetGroups = lgArr.length > 0
      ? Math.ceil(lgArr.length * layersLocal / L) : 0;
    const hasSeg = wl.segmentLayers &&
      wl.segmentLayers > 0 && wl.segmentLayers < layersLocal;
    const segCount = hasSeg ? Math.ceil(layersLocal / wl.segmentLayers) : 1;
    const fams = PROF.layer_families || [];
    const hasUnmodeled = fams.some(f => !SUPPORTED_FAM.has(f));
    const noWholeStep = pp > 1 || segCount > 1 || lgArr.length > 1 ||
      L === 1 || hasUnmodeled;
    const hbmUsed = wStatic + lMoe * expertsLoc * expBytes +
      kvReadBytes + 0.5 * recRwBytes;
    const hbmFits = hbmUsed <= (CHIP.hbm_capacity_bytes || 103e9);

    const scopes = [];
    for (const sc of ['per_op', 'single_layer', 'layer_group', 'whole_step']) {
      let feasible = true, reason = 'ok';
      if (sc === 'layer_group' && presetGroups === 0 && pp === 1 && segCount === 1) {
        feasible = false; reason = 'No layer grouping configured';
      } else if (sc === 'whole_step' && noWholeStep) {
        feasible = false; reason = 'Bounded by layer groups or PP > 1';
      }
      let staticBytes = 0, actBytes = 0, nOps = 1, tMax = 1, nSplits = 1;
      let denseStage = 0, expertStage = 0, opScratch = 0, padBytes = 0;
      if (sc === 'per_op') {
        denseStage = Math.max(1048576, slDenseStaging);
        staticBytes = denseStage;
        const perOpMax = Math.floor((vmemBudget - staticBytes) / (4 * H));
        tMax = Math.max(activeTokens, perOpMax);
        nSplits = 1; nOps = 2 * layersLocal;
        actBytes = 4.0 * layersLocal * tokResBytes;
      } else {
        const isSl = sc === 'single_layer';
        const isSlOrGrp = isSl || (sc === 'layer_group' && presetGroups > 0);
        const mul = sc === 'layer_group' ? (presetGroups || segCount) : 1;
        denseStage = isSlOrGrp ? slDenseStaging : wsDenseStaging;
        expertStage = lMoe > 0
          ? (isSlOrGrp ? slExpertStaging : wsExpertStaging) : 0;
        const aliasScratch = isSl
          ? Math.max(gqaScratch, gdnScratch, ffnScratch)
          : Math.max(gqaScratch + gdnScratch, ffnScratch);
        const sumScratch = gqaScratch + gdnScratch + ffnScratch;
        opScratch = wl.vmemAlias ? aliasScratch : sumScratch;
        const moePad = lMoe > 0 ? (12 * 1048576) : (2 * 1048576);
        padBytes = isSlOrGrp ? 0 : (moePad + (lMoe === 0 ? 2304 * H : 0));
        staticBytes = denseStage + expertStage + opScratch + padBytes;
        actBytes = 2.0 * (isSl ? layersLocal : mul) * tokResBytes;
        const subMul = (effTp * ep > 1 && (lAttn + lLin) > 0) ? 3 : 1;
        nOps = isSl ? layersLocal : (
          sc === 'layer_group' && presetGroups > 0 ? mul : mul * subMul
        );
        const alpha = vmemParts.totalPerTokenBytes;
        tMax = (staticBytes + alpha > vmemBudget)
          ? 0 : Math.max(1, Math.floor((vmemBudget - staticBytes) / alpha));
        nSplits = tMax === 0 ? activeTokens : Math.ceil(activeTokens / tMax);
        if (tMax === 0) { feasible = false; reason = 'Weight buffers exceed VMEM'; }
      }
      const isWsOrLg = sc === 'whole_step' || sc === 'layer_group';
      const tiled = isWsOrLg && nSplits > 1;
      const tileTokens = Math.ceil(activeTokens / Math.max(1, nSplits));
      const groupedMoe = !(isWsOrLg && epDp > 1) && (
        sc === 'per_op' || wl.moeMode === 'grouped_on_chip' ||
        tileTokens * epDp >= rFloor
      );
      const moeTok = (groupedMoe && sc !== 'single_layer')
        ? activeTokens : tileTokens;
      let wStep = wStatic, phi = 1.0;
      if (expertsLoc > 0) {
        const tEp = moeTok * epDp;
        const eHit = expertsHit(tEp, topK, nExp, expertsLoc);
        const pairs = Math.max(1.0, tEp * topK * expertsLoc / nExp);
        const avgM = Math.max(1.0, pairs / eHit);
        const hit = tiled
          ? expertsHit(activeTokens * epDp, topK, nExp, expertsLoc) : eHit;
        wStep = wStatic + lMoe * hit * expBytes;
        phi = groupedMoe ? (Math.ceil(avgM / rFloor) * rFloor / avgM)
          : Math.max(1.0, (Math.floor((tEp + 15) / 16) * 16 * eHit) / pairs);
      }
      if (tiled) actBytes += 2.0 * layersLocal * tokResBytes;
      const xlaArFactor = (sc === 'per_op' && effTp > 1) ? 2 : 1;
      const xlaHbmCollBytes = (sc === 'per_op' && effTp > 1)
        ? 2.0 * numAr * tokResBytes : 0.0;
      const a2aTerm = (lMoe > 0 && epDp > 1)
        ? lMoe * (a2aTime + 2 * nSplits * xlaArFactor * latColl) : 0.0;
      const tComm = numAr * (arInfo.xferS + nSplits * xlaArFactor * arInfo.latS) +
        xlaHbmCollBytes / bwHbm + ppStageTime + a2aTerm;
      const effStateBytes = kvReadBytes + recRwBytes;
      let fillDrainS = (nOps - 1) * (firstSlot / bwHbm + 2.0 * latColl);
      if (tiled) {
        fillDrainS += (nSplits - 1) * layersLocal *
          ((tileTokens * H * 2.0) / bwHbm + 2.0 * latColl);
      }
      const tMatmul = flopsDense / peakW + (phi * flopsMoe) / peakE;
      const tMatmulHi = (isPrefOrMix && flopsKv > 0)
        ? ((flopsDense - flopsKv) / peakW + flopsKv / bf16Peak +
           (phi * flopsMoe) / peakE)
        : tMatmul;
      const tHbm = ((tiled ? 1 : nSplits) * wStep + effStateBytes + actBytes) / bwHbm;
      const roofUs = (Math.max(tMatmul, tHbm) + tComm + fillDrainS) * 1e6;
      const baseAi = (flopsDense + flopsMoe) / Math.max(1.0, wStep + effStateBytes);
      const repassUs = (baseAi >= peakW / bwHbm && nSplits > 1 && !tiled)
        ? ((actBytes + (nSplits - 1) * wStep) / bwHbm * 1e6) : 0.0;
      const launchUs = nOps * wl.launchGapUs, stepUs = roofUs + repassUs + launchUs;
      const a2aHops = (lMoe > 0 && epDp > 1) ? 2 * lMoe * nSplits * xlaArFactor : 0;
      const collExtraS = (numAr * nSplits * xlaArFactor * arInfo.hops + a2aHops) *
        Math.max(0.0, latCollHi - latColl);
      const roofHiUs = (Math.max(tMatmulHi, tHbm) + tComm +
        collExtraS + fillDrainS) * 1e6;
      const launchHi = Math.max(wl.launchGapUs, wl.launchGapHiUs || 8.0);
      const stepHiUs = Math.max(stepUs, roofHiUs + repassUs + nOps * launchHi);
      const resTok = Math.min(activeTokens, Math.max(1, tMax));
      const aliasDiff = vmemParts.unaliasedOpBytes - vmemParts.opBytes;
      scopes.push({
        scope: sc, feasible, reason, ops_per_step: nOps, t_vmem_max: tMax,
        n_splits: nSplits, tile_tokens: tileTokens, rawStepUs: stepUs,
        vmem_static_mib: +(staticBytes / 1048576).toFixed(3),
        dense_stage_mib: +(denseStage / 1048576).toFixed(2),
        expert_stage_mib: +(expertStage / 1048576).toFixed(2),
        op_scratch_mib: +(opScratch / 1048576).toFixed(2),
        pad_mib: +(padBytes / 1048576).toFixed(2),
        spine_dyn_mib: +((resTok * vmemParts.spineBytes) / 1048576).toFixed(2),
        op_dyn_mib: +((resTok * vmemParts.opBytes) / 1048576).toFixed(2),
        alias_saved_kib: +(aliasDiff / 1024).toFixed(1),
        moe_flop_factor: +phi.toFixed(3), w_step_gb: +(wStep / 1e9).toFixed(4),
        act_hbm_mb: +(actBytes / 1e6).toFixed(3),
        compute_us: +(tMatmul * 1e6).toFixed(2),
        hbm_us: +(tHbm * 1e6).toFixed(2), comm_us: +(tComm * 1e6).toFixed(2),
        fill_drain_us: +(fillDrainS * 1e6).toFixed(2),
        repass_spill_us: +repassUs.toFixed(2),
        launch_us: +launchUs.toFixed(2), roof_floor_us: +roofUs.toFixed(2),
        step_bound_us: +stepUs.toFixed(2), step_upper_us: +stepHiUs.toFixed(2)
      });
    }
    const feas = hbmFits ? scopes.filter(s => s.feasible) : [scopes[0]];
    feas.sort((a, b) => (a.rawStepUs - b.rawStepUs) ||
      (a.ops_per_step - b.ops_per_step));
    const winner = feas[0] || scopes[0], runnerUp = feas[1] || winner;
    const recScope = hbmFits ? winner.scope : 'per_op';
    const tokensPerChip = (activeTokens * effPcp * (CHIP.devices_per_chip || 1)) /
      Math.max(1, effTp * pp);
    const tpsChip = tokensPerChip / Math.max(1e-9, winner.step_bound_us * 1e-6);
    const tPerEp = Math.ceil(activeTokens / Math.max(1, winner.n_splits)) * epDp;
    const moeModeRec = lMoe === 0 ? 'dense_mlp'
      : (recScope === 'per_op' ? 'grouped_in_hbm'
      : (tPerEp >= rowFloor(PROF.expert_dtype)
         ? 'grouped_on_chip' : 'masked_on_chip'));
    const marginPct = runnerUp && runnerUp.scope !== winner.scope
      ? ((runnerUp.step_bound_us - winner.step_bound_us) /
         Math.max(0.01, runnerUp.step_bound_us)) * 100.0 : 100.0;
    return {
      activeTokens, effCtx, scopes, winner, runnerUp, tpsChip, hbmFits,
      ridgeFb: peakW / bwHbm, moeModeRec, marginPct,
      vmemBudgetMib: +(vmemBudget / 1048576).toFixed(2), vmemParts
    };
  }
  window.__MK_ADVISOR_ENGINE = {
    DATA, PROF, CHIP, REGIMES, computePerTokenVmem, evaluateWorkload
  };
})();
"""
