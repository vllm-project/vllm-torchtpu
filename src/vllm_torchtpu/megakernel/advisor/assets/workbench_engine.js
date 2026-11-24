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
  const CORE_FAM = new Set(CORE_LIM.core_module_families || []),
    CORE_DTYPES = new Set(CORE_LIM.core_supported_dtypes || []),
    STRUCTURAL_TAGS = new Set(CORE_LIM.structural_layer_tags || []);
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
    fp32: 4.0, bf16: 2.0, fp16: 2.0, fp8: 1.0, int8: 1.0, int4: 0.5,
    int4_per_token_head: 0.5, turboquant_4bit_nc: 0.5,
    nvfp4: 0.5625, mxfp4: 0.53125, fp4: 0.5625
  }, DATA.dtype_bytes || SEED_DEF.dtype_bytes || {});
  const KV_ONLY_DTYPES = new Set(
    DATA.kv_only_dtypes || SEED_DEF.kv_only_dtypes || ['int4_per_token_head', 'turboquant_4bit_nc']
  );
  const KV_DTYPE_ALIASES = Object.assign({
    auto: 'bf16', bfloat16: 'bf16', bf16: 'bf16', float16: 'fp16', fp16: 'fp16',
    float32: 'fp32', fp32: 'fp32', fp8: 'fp8', fp8_e4m3: 'fp8', fp8_e4m3fn: 'fp8',
    fp8_e5m2: 'fp8_e5m2', int8: 'int8', int4: 'int4',
    int4_per_token_head: 'int4_per_token_head', turboquant_4bit_nc: 'turboquant_4bit_nc'
  }, DATA.kv_dtype_aliases || SEED_DEF.kv_dtype_aliases || {});
  function normalizeKvDtype(raw) {
    const low = String(raw || 'fp8').trim().toLowerCase();
    return Object.prototype.hasOwnProperty.call(KV_DTYPE_ALIASES, low)
      ? KV_DTYPE_ALIASES[low] : low;
  }
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
    const dt = String(dtype || '').toLowerCase();
    const map = c.matmul_row_tile_by_dtype || {};
    if (Object.prototype.hasOwnProperty.call(map, dt) && map[dt] > 0) {
      return map[dt];
    }
    throw new Error(
      `Chip '${c.name || 'unknown'}' has no matmul_row_tile_by_dtype entry for dtype='${dtype}'`
    );
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
  function readPositiveSeed(key) {
    const value = Number(DATA[key]);
    if (!Number.isFinite(value) || value <= 0) {
      throw new Error(`Missing or invalid positive seed constant: ${key}`);
    }
    return value;
  }
  function routedChain(tDense, routedHbmS, tMoe, tVecMoe = 0.0, dsaStageS = 0.0) {
    return Math.max(tDense, dsaStageS) + Math.max(routedHbmS, tMoe, tVecMoe);
  }
  function hasDsaProfile(p) {
    if (p && p.sparse_attn != null) return false;
    return (p.index_topk || 0) > 0 && (p.index_n_heads || 0) > 0 && (p.index_head_dim || 0) > 0;
  }
  function wideDcpKvHeadsPerRank(numKvHeads, tp, dcp, dcpSpan = 'kv_heads') {
    if (numKvHeads <= 0) return 0;
    const effTp = Math.max(1, tp), effDcp = Math.max(1, dcp);
    if (dcpSpan === 'tp_group') {
      return Math.min(numKvHeads, Math.max(1, Math.ceil((numKvHeads * effDcp) / effTp)));
    }
    const kvSplitTp = Math.min(effDcp, effTp);
    return Math.max(1, Math.ceil(numKvHeads / Math.max(1, Math.floor(effTp / kvSplitTp))));
  }
  function dcpDividesMesh(p, effTp, dcp, pcp) {
    if (p.dcp_span === 'tp_group') {
      if (effTp % dcp !== 0) return false;
      const kvh = p.num_kv_heads || 0;
      if (kvh > 0) {
        const replicas = Math.max(1, Math.floor(effTp / kvh));
        return replicas % dcp === 0 || dcp % replicas === 0;
      }
      return true;
    }
    return (effTp * pcp) % dcp === 0;
  }
  function normalizeHostTableOverrides(raw) {
    const out = {};
    if (!raw) return out;
    if (Array.isArray(raw)) {
      for (const item of raw) {
        if (Array.isArray(item) && item.length === 2) {
          out[String(item[0])] = Boolean(item[1]);
        }
      }
      return out;
    }
    if (typeof raw === 'object') {
      for (const [k, v] of Object.entries(raw)) out[String(k)] = Boolean(v);
    }
    return out;
  }
  function effectiveHostTableHbmResident(table, wl) {
    const rawOv = wl.host_table_hbm_overrides ?? wl.hostTableHbmOverrides;
    const ov = normalizeHostTableOverrides(rawOv);
    if (Object.prototype.hasOwnProperty.call(ov, table.name)) return ov[table.name];
    return Boolean(table.hbm_resident);
  }
  function activeFusionBreakLayers(p, wl) {
    const breaks = new Set();
    const numLayers = p.num_layers || 0;
    for (const tbl of (p.host_tables || [])) {
      if (!effectiveHostTableHbmResident(tbl, wl)) {
        for (const lyr of (tbl.layers || [])) {
          if (lyr > 0 && lyr < numLayers) breaks.add(lyr);
        }
      }
    }
    const execMode = String(wl.execution || WL_DEF.execution || 'full');
    const cedBoundary = p.ced_boundary || 0;
    if (execMode === 'ced' && cedBoundary > 0 &&
        (wl.phase === 'prefill' || wl.phase === 'mixed') &&
        cedBoundary < numLayers) {
      breaks.add(cedBoundary);
    }
    return Array.from(breaks).sort((a, b) => a - b);
  }
  function fusionBreakLayerGroups(numLayers, breakLayers) {
    if (!breakLayers || breakLayers.length === 0 || numLayers <= 1) return [];
    const cuts = Array.from(
      new Set(breakLayers.filter(l => l > 0 && l < numLayers))
    ).sort((a, b) => a - b);
    if (cuts.length === 0) return [];
    const bounds = [0, ...cuts, numLayers];
    const groups = [];
    for (let i = 0; i < bounds.length - 1; i++) groups.push(bounds[i + 1] - bounds[i]);
    return groups;
  }
  function turboquantSlotBytes(headDim, kvPageSize) {
    const pageSz = Math.max(1, kvPageSize || readPositiveSeed('turboquant_default_page_size'));
    const hdr = readPositiveSeed('turboquant_header_bytes');
    const wordB = readPositiveSeed('turboquant_word_bytes');
    const subW = readPositiveSeed('turboquant_sublane_words');
    const rawWords = Math.ceil((pageSz * (headDim + hdr)) / wordB);
    const aligned = Math.floor((rawWords + subW - 1) / subW) * subW;
    return (aligned * wordB) / pageSz;
  }
  function computeHadamardKvVecOps(kvDtype, nhLoc, kvhLoc, activeTokens, lAttn) {
    const dt = normalizeKvDtype(kvDtype);
    let headsRot = 0;
    if (dt === 'int4_per_token_head') headsRot = 2 * nhLoc + 2 * kvhLoc;
    else if (dt === 'turboquant_4bit_nc') headsRot = nhLoc + kvhLoc;
    else return 0.0;
    return readPositiveSeed('hadamard_128_vec_ops_per_head') * headsRot * activeTokens * lAttn;
  }
  function computeHyperConnectionTerms(p, layersLocal, effTp, activeTokens) {
    const hc = p.hyper_connection;
    if (!hc || (hc.num_streams || 0) <= 1 || layersLocal <= 0) {
      return { params: 0, flops: 0.0, vecOps: 0.0 };
    }
    const tp = Math.max(1, effTp), streams = hc.num_streams;
    const hLoc = Math.ceil(p.hidden_size / tp);
    const subPerLayer = readPositiveSeed('hc_sublayers_per_layer');
    if (hc.kind === 'lowrank') {
      const rank = hc.lowrank_dim || 0;
      const perSub = 2 * streams * hLoc * rank + streams * streams * hLoc;
      let params = layersLocal * subPerLayer * perSub;
      if (hc.has_final_mixer) params += 2 * streams * hLoc * rank;
      return { params, flops: 2.0 * params * activeTokens, vecOps: 0.0 };
    }
    const dynTerms = readPositiveSeed('mhc_dynamic_proj_terms');
    const perSub = (streams * streams + dynTerms * streams) * (streams * hLoc);
    const params = layersLocal * subPerLayer * perSub;
    const flops = 2.0 * params * activeTokens +
      2.0 * subPerLayer * layersLocal * activeTokens * streams * streams * hLoc;
    const vecOps = subPerLayer * layersLocal * activeTokens * streams * streams *
      readPositiveSeed('sinkhorn_vec_ops_per_entry_iter') * (hc.sinkhorn_iters || 0);
    return { params, flops, vecOps };
  }
  function computeAttentionParams(p, effTp, kvSplit, dcp = 1) {
    const H = p.hidden_size, hd = p.head_dim, kvLora = p.kv_lora_rank || 0;
    const qLora = p.q_lora_rank || 0, qkRope = p.qk_rope_head_dim || 0;
    const qkNope = p.qk_nope_head_dim || hd;
    const nhLoc = Math.max(1, Math.ceil(p.num_attention_heads / effTp));
    const kvhLoc = p.dcp_span === 'tp_group'
      ? wideDcpKvHeadsPerRank(p.num_kv_heads || 0, effTp, dcp, 'tp_group')
      : Math.max(1, Math.ceil((p.num_kv_heads || 0) / Math.max(1, Math.floor(effTp / kvSplit))));
    const kvhWLoc = p.dcp_span === 'tp_group'
      ? Math.max(1, Math.ceil((p.num_kv_heads || 0) / effTp))
      : kvhLoc;
    const qDim = nhLoc * hd;
    const attnGate = (p.has_attn_output_gate && !kvLora) ? (H * qDim) : 0;
    if ((p.o_lora_rank || 0) > 0 && kvLora === 0) {
      const qIn = qLora > 0 ? (qLora * H + nhLoc * hd * qLora) : (nhLoc * hd * H);
      const sharedMqa = Boolean(p.sparse_attn && p.sparse_attn.shared_kv_mqa);
      const wkv = (sharedMqa ? 1 : 2) * kvhWLoc * hd * H;
      const woA = p.o_lora_rank * nhLoc * hd;
      const woB = H * Math.max(1, Math.ceil((p.o_groups || 1) * p.o_lora_rank / effTp));
      return { nhLoc, kvhLoc, qDim, attnParams: qIn + wkv + woA + woB + attnGate };
    }
    const qTot = nhLoc * (qkNope + qkRope);
    const mlaProj = (qLora ? qLora * (H + qTot) : H * qTot) + H * (kvLora + qkRope + qDim);
    const attnParams = kvLora
      ? (mlaProj + kvLora * nhLoc * (qkNope + hd))
      : (H * 2 * (qDim + kvhWLoc * hd) + attnGate);
    return { nhLoc, kvhLoc, qDim, attnParams };
  }
  function ownerRecordBytes(owner, p, wl, kvDt, bwKv, effTp, kvGroupHeads, isIndex) {
    const repMult = owner.replication === 'kv_head_group'
      ? kvGroupHeads : (owner.replication === 'tp_sharded' ? (1.0 / Math.max(1, effTp)) : 1.0);
    const ovB = Number(owner.record_overhead_bytes || 0.0);
    if (owner.elem_bytes !== null && owner.elem_bytes !== undefined) {
      return (owner.record_elems * Number(owner.elem_bytes) + ovB) * repMult;
    }
    if (isIndex) {
      if (kvDt === 'int4_per_token_head') {
        return (owner.record_elems * 0.5 + readPositiveSeed('int4_index_scale_overhead_bytes')) *
          repMult + ovB;
      }
      const elemW = kvDt === 'turboquant_4bit_nc' ? 1.0 : bwKv;
      return owner.record_elems * elemW * repMult + ovB;
    }
    const kvPageSz = wl.kv_page_size ?? wl.kvPageSize ?? WL_DEF.kv_page_size ?? 64;
    if (kvDt === 'turboquant_4bit_nc') {
      const headsRec = owner.record_elems / Math.max(1, 2 * p.head_dim);
      return headsRec * turboquantSlotBytes(p.head_dim, kvPageSz) * repMult + ovB;
    }
    if (kvDt === 'int4_per_token_head') {
      return owner.record_elems * 0.5 * repMult +
        readPositiveSeed('int4_scale_page_bytes_per_tok') + ovB;
    }
    return owner.record_elems * bwKv * repMult + ovB;
  }
  function synthesizeHomogeneousOwners(sp, p, lAttn) {
    const hasKv = Array.isArray(sp.kv_owners) && sp.kv_owners.length > 0;
    const hasIdx = Array.isArray(sp.index_owners) && sp.index_owners.length > 0;
    if (hasKv || hasIdx) return { kvOwners: sp.kv_owners || [], idxOwners: sp.index_owners || [] };
    if (sp.kind !== 'qsa' && sp.kind !== 'msa') return { kvOwners: [], idxOwners: [] };
    const kvOwners = [], idxOwners = [];
    const ratio = sp.kind === 'qsa' ? Math.max(1, sp.index_compress_ratio || 1) : 1.0;
    const idxElemB = sp.kind === 'qsa' ? readPositiveSeed('fp32_bytes') : null;
    for (let lyr = 0; lyr < lAttn; lyr++) {
      kvOwners.push({
        layer: lyr, record_tokens: 1.0, record_elems: 2 * p.head_dim, elem_bytes: null,
        record_overhead_bytes: 0.0, replication: 'kv_head_group', users: [lyr]
      });
      if ((sp.index_dim || 0) > 0) {
        idxOwners.push({
          layer: lyr, record_tokens: ratio, record_elems: sp.index_dim, elem_bytes: idxElemB,
          record_overhead_bytes: 0.0, replication: 'per_rank', users: [lyr]
        });
      }
    }
    return { kvOwners, idxOwners };
  }
  function sparseIndexerProjParams(sp, p, effTp, lAttn) {
    const idxH = sp.index_heads || 0, idxD = sp.index_dim || 0;
    if (idxH <= 0 || idxD <= 0 || lAttn <= 0) return { qParams: 0, kwParams: 0 };
    if (sp.kind === 'qsa' || sp.kind === 'msa') {
      const proj = Math.ceil(((idxH + (sp.index_kv_heads || 1)) * idxD) / effTp) * p.hidden_size;
      return sp.kind === 'qsa' ? { qParams: 0, kwParams: proj } : { qParams: proj, kwParams: 0 };
    }
    const qParams = Math.ceil((idxH * idxD) / effTp) * (p.q_lora_rank || p.hidden_size);
    const idxOwnersLen = (sp.index_owners || []).length;
    const kwParams = roundEven(
      (idxOwnersLen / Math.max(1, p.num_layers)) * p.hidden_size *
      Math.ceil((idxD + idxH) / effTp)
    );
    return { qParams, kwParams };
  }
  function computeSparseAttnTerms(p, wl, g, kvhLoc, bwKv, rawAttnCtx) {
    const sp = p.sparse_attn;
    const kvDt = normalizeKvDtype(p.kv_dtype);
    const capTok = (sp.window_tokens || 0) + (sp.select_tokens || 0);
    const attnCtx = capTok > 0 ? Math.min(rawAttnCtx, capTok) : rawAttnCtx;
    const isSparseGather = capTok > 0 && rawAttnCtx > capTok;
    const tokDiv = p.dcp_span === 'tp_group' ? g.dcp : g.kvSplit;
    const hasExp = (sp.kv_owners && sp.kv_owners.length > 0) ||
      (sp.index_owners && sp.index_owners.length > 0);
    const ppScale = hasExp ? (g.layersLocal / Math.max(1, p.num_layers)) : 1.0;
    const { kvOwners, idxOwners } = synthesizeHomogeneousOwners(sp, p, g.lAttn);
    const winLayers = (sp.window_layers && sp.window_layers.length > 0)
      ? Math.ceil(sp.window_layers.length / Math.max(1, wl.pp || 1))
      : ((sp.window_bytes_per_token || 0) > 0 ? g.lAttn : 0);
    const swaRes = g.batchLocal * winLayers * Math.min(g.effCtx, sp.window_tokens || 0) *
      ((sp.window_bytes_per_token || 0) / Math.max(1, g.effTp));
    const ringRes = g.batchLocal * (sp.kv_owner_ring_bytes_per_req || 0) * ppScale;
    let kvRes = 0.0, kvReadStrict = 0.0, kvReadGather = 0.0;
    for (const o of kvOwners) {
      const recB = ownerRecordBytes(o, p, wl, kvDt, bwKv, g.effTp, kvhLoc, false);
      const recTok = Math.max(1.0, o.record_tokens || 1.0);
      kvRes += g.batchLocal * ((g.effCtx / recTok) / tokDiv) * recB * ppScale;
      const selR = (attnCtx / recTok) / tokDiv;
      const usersN = Math.max(1, (o.users || []).length);
      const strictB = usersN * g.batchLocal * selR * recB * ppScale;
      kvReadStrict += strictB;
      kvReadGather += isSparseGather
        ? (usersN * g.activeTokens * selR * padCachelineBytes(recB) * ppScale) : strictB;
    }
    let idxRes = 0.0, idxRead = 0.0, flopsIdx = 0.0;
    for (const o of idxOwners) {
      const recB = ownerRecordBytes(o, p, wl, kvDt, bwKv, g.effTp, kvhLoc, true);
      const recTok = Math.max(1.0, o.record_tokens || 1.0);
      idxRes += g.batchLocal * ((g.effCtx / recTok) / tokDiv) * recB * ppScale;
      let scanR = rawAttnCtx / recTok;
      if (o.max_scan_records !== null && o.max_scan_records !== undefined) {
        scanR = Math.min(scanR, Number(o.max_scan_records));
      }
      const locScan = scanR / tokDiv;
      const usersN = Math.max(1, (o.users || []).length);
      idxRead += usersN * g.batchLocal * locScan * recB * ppScale;
      flopsIdx += usersN * 2.0 * g.activeTokens * (sp.index_heads || 0) *
        (sp.index_dim || 0) * locScan * ppScale;
    }
    const { qParams, kwParams } = sparseIndexerProjParams(sp, p, g.effTp, g.lAttn);
    return {
      hasDsa: true, isSparseGather, rawAttnCtx, indexerCtxLocal: rawAttnCtx / tokDiv,
      attnCtx, indexerQueryProjParams: qParams, indexerKeyWeightProjParams: kwParams,
      indexerKReadBytes: idxRead,
      indexerKResidentBytes: swaRes + ringRes + kvRes + idxRes,
      dsaKvGatherBytes: swaRes + kvReadGather,
      dsaKvStrictBytes: swaRes + kvReadStrict,
      flopsIndexerScore: flopsIdx
    };
  }
  function cedTokenScale(p, wl, activeTokens, chunkTokens, layersLocal) {
    const execMode = String(wl.execution || WL_DEF.execution || 'full');
    const cedB = p.ced_boundary || 0;
    if (execMode !== 'ced' || cedB <= 0 ||
        (wl.phase !== 'prefill' && wl.phase !== 'mixed') ||
        activeTokens <= 0 || layersLocal <= 0) {
      return 1.0;
    }
    const tPrefPart = wl.phase === 'prefill' ? activeTokens : chunkTokens;
    const tDecPart = Math.max(0, activeTokens - tPrefPart);
    const winCap = (p.sparse_attn && (p.sparse_attn.window_tokens || 0) > 0)
      ? p.sparse_attn.window_tokens : tPrefPart;
    const decTokens = tDecPart + Math.min(tPrefPart, winCap);
    const encL = Math.min(layersLocal, Math.ceil((cedB * layersLocal) / Math.max(1, p.num_layers)));
    const decL = Math.max(0, layersLocal - encL);
    return (encL * activeTokens + decL * decTokens) / (layersLocal * activeTokens);
  }
  function hostTableShardDivisor(shardAxis, effTp, dp) {
    const tp = Math.max(1, effTp);
    const coresHost = readPositiveSeed('cores_per_host');
    if (shardAxis === 'tp') return tp;
    if (shardAxis === 'dp_local') return Math.max(1, Math.min(dp, Math.max(1, Math.floor(coresHost / tp))));
    return Math.max(1, Math.min(tp * dp, coresHost));
  }
  function computeHostTableContextTerms(p, wl, effTp, dp, activeTokens, batchLocal, ch) {
    const tables = p.host_tables || [];
    if (tables.length === 0) return { tblHbmB: 0.0, tblReadB: 0.0, hostH2dS: 0.0 };
    let tblHbmB = 0.0, tblReadB = 0.0, offH2dB = 0.0;
    for (const tbl of tables) {
      tblHbmB += (tbl.state_bytes_per_request || 0) * batchLocal;
      const stepLookup = activeTokens * (tbl.h2d_bytes_per_token || 0.0);
      if (effectiveHostTableHbmResident(tbl, wl)) {
        const div = hostTableShardDivisor(tbl.shard_axis || 'tp', effTp, dp);
        tblHbmB += (tbl.table_bytes_total || 0) / div;
        tblReadB += stepLookup;
      } else {
        offH2dB += stepLookup;
      }
    }
    const pcieBw = Math.max(1.0, (ch && ch.pcie_bw_bytes_per_s) || 32e9);
    return { tblHbmB, tblReadB, hostH2dS: offH2dB / pcieBw };
  }
  function dsaQueryGatherDim(p) {
    return (p.kv_lora_rank || 0) > 0
      ? ((p.kv_lora_rank || 0) + (p.qk_rope_head_dim || 0))
      : p.head_dim;
  }
  function indexerRecordBytes(indexHeadDim) {
    if (indexHeadDim <= 0) return 0;
    const scaleBlock = readPositiveSeed('indexer_scale_block_size');
    const wordAlign = readPositiveSeed('word_align_bytes');
    const scaleBytes = Math.ceil(indexHeadDim / scaleBlock);
    const rawBytes = indexHeadDim + scaleBytes;
    return Math.floor((rawBytes + wordAlign - 1) / wordAlign) * wordAlign;
  }
  function padCachelineBytes(byteCount) {
    const lineBytes = readPositiveSeed('hbm_cacheline_bytes');
    return Math.ceil(byteCount / lineBytes) * lineBytes;
  }
  function padVectorLanes(elementCount) {
    const laneWidth = readPositiveSeed('attn_tile_tokens');
    return Math.ceil(elementCount / laneWidth) * laneWidth;
  }
  function computeIndexerStaticAndResidentTerms(
    p, layersLocal, attnLayersLocal, batchLocal, effCtx, kvSplit
  ) {
    const hasDsa = hasDsaProfile(p);
    if (!hasDsa) {
      return {
        hasDsa: false, lIndexer: 0, indexerQueryProjParams: 0,
        indexerKeyWeightProjParams: 0, indexerKeyRecordBytes: 0,
        indexerKResidentBytes: 0
      };
    }
    const numLayers = p.num_layers;
    const hidden = p.hidden_size;
    const qLora = p.q_lora_rank || 0;
    const lIndexer = Math.min(
      attnLayersLocal, Math.ceil((p.num_indexer_layers || 0) * layersLocal / numLayers)
    );
    const indexerQueryProjParams = (qLora || hidden) * (p.index_n_heads || 0) * (p.index_head_dim || 0);
    const indexerKeyWeightProjParams = hidden * ((p.index_head_dim || 0) + (p.index_n_heads || 0));
    const indexerKeyRecordBytes = indexerRecordBytes(p.index_head_dim || 0);
    const indexerKResidentBytes = lIndexer * batchLocal * (effCtx / kvSplit) * indexerKeyRecordBytes;
    return {
      hasDsa: true, lIndexer, indexerQueryProjParams,
      indexerKeyWeightProjParams, indexerKeyRecordBytes, indexerKResidentBytes
    };
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
  function sameLayout(d, wl) {
    const wlTp = wl.tp ?? WL_DEF.tp ?? 8;
    const wlEp = wl.ep ?? WL_DEF.ep ?? 1;
    const wlDp = wl.dp ?? WL_DEF.dp ?? 1;
    const wlPp = wl.pp ?? WL_DEF.pp ?? 1;
    const wlDcp = wl.dcp ?? WL_DEF.dcp ?? 1;
    const wlPcp = wl.pcp ?? WL_DEF.pcp ?? 1;
    return (d.tp ?? 1) === wlTp &&
      (d.ep ?? 1) === wlEp &&
      (d.dp ?? 1) === wlDp &&
      (d.pp ?? 1) === wlPp &&
      (d.dcp ?? 1) === wlDcp &&
      (d.pcp ?? 1) === wlPcp;
  }
  function resolveDeclaredExpertMode(decl, wl, activeTokens, lMoe) {
    if (lMoe <= 0) return 'none';
    if (decl && Array.isArray(decl.expert_mode_by_rows) && decl.expert_mode_by_rows.length > 0) {
      for (const rule of decl.expert_mode_by_rows) {
        if (activeTokens <= Number(rule.max_rows)) return String(rule.mode);
      }
      return String(decl.expert_mode_by_rows[decl.expert_mode_by_rows.length - 1].mode);
    }
    return (wl.ep || 1) > 1 ? 'ep' : 'tp';
  }
  function computeCollectiveStepTerms(step, p, ctx, nSplits, callMult, expertMode, collectiveSeed) {
    const groupSize = Number(step.group_size);
    const callsPerLayer = Number(step.calls_per_layer || collectiveSeed.defaultCallsPerLayer);
    const hopsPerCall = roundEven(Math.log2(groupSize));
    let wireBytes = 0.0, totalCalls = 0;
    if (step.kind === 'routed_transport') {
      if (ctx.lMoe <= 0 || expertMode === 'tp') {
        return { transferS: 0.0, strictEgressS: 0.0, hops: 0 };
      }
      const tpInGroup = Math.max(1, Math.floor(ctx.effTp / groupSize));
      const fanout = Math.min(Math.max(1, p.top_k || 1), groupSize);
      const routedHidden = p.routed_expert_hidden_size || p.hidden_size;
      const bytesPerLayer = (
        2.0 * ((groupSize - 1) / groupSize) * ctx.activeTokens * fanout *
        routedHidden * collectiveSeed.bf16Bytes
      ) / tpInGroup;
      wireBytes = ctx.lMoe * bytesPerLayer;
      totalCalls = ctx.lMoe * callsPerLayer * nSplits * callMult;
    } else if (step.kind === 'split_kv') {
      if (ctx.lAttn <= 0) return { transferS: 0.0, strictEgressS: 0.0, hops: 0 };
      const bytesPerLayer = (groupSize - 1) * ctx.activeTokens * ctx.nhLoc *
        p.head_dim * (collectiveSeed.bf16Bytes + collectiveSeed.fp32Bytes);
      wireBytes = ctx.lAttn * bytesPerLayer;
      totalCalls = ctx.lAttn * callsPerLayer * nSplits * callMult;
    } else {
      if (ctx.lAttn <= 0) return { transferS: 0.0, strictEgressS: 0.0, hops: 0 };
      const qHeadDim = ((p.qk_nope_head_dim || 0) + (p.qk_rope_head_dim || 0)) || p.head_dim;
      const bytesPerHeadTok = collectiveSeed.bf16Bytes * qHeadDim +
        collectiveSeed.fp32Bytes * (p.head_dim + collectiveSeed.lseScalarsPerHead);
      const bytesPerLayer = (groupSize - 1) * ctx.activeTokens * ctx.nhLoc * bytesPerHeadTok;
      wireBytes = ctx.lAttn * bytesPerLayer;
      totalCalls = ctx.lAttn * callsPerLayer * nSplits * callMult;
    }
    return {
      transferS: wireBytes / ctx.iciLink,
      strictEgressS: wireBytes / ctx.iciEg,
      hops: totalCalls * hopsPerCall
    };
  }
  // Deployment blocks that declare scope `sc` for the phase. A block without phases
  // covers decode and verify only, and a block needs a source and a rank count.
  function declaredBlocks(p, sc, phase) {
    return declaredFor(p.name).filter(
      d => d && String(d.source || '').trim() && d.ranks > 0 &&
        Array.isArray(d.scopes) && d.scopes.includes(sc) &&
        (Array.isArray(d.phases) && d.phases.length ? d.phases : DEFAULT_DECLARED_PHASES)
          .includes(phase)
    );
  }
  // The block that declares scope `sc` at this rank count, phase, and layout.
  function findDeclaration(p, sc, effTp, wl) {
    return declaredBlocks(p, sc, wl.phase).find(
      d => effTp === d.ranks && sameLayout(d, wl)
    ) || null;
  }
  // Most rows the megakernel serves in one decode or verify call (0: no declared limit).
  // The workload's own limit wins over a preset's.
  function singleCallRowLimit(p, wl, effTp) {
    if (wl.phase !== 'decode' && wl.phase !== 'verify') return 0;
    const decl = findDeclaration(p, 'whole_step', effTp, wl);
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
  // Whether one call is larger than the core's default bucket list.
  function exceedsDefaultBuckets(wl, rowsPerCall) {
    if (wl.phase === 'decode') {
      return rowsPerCall > CORE_LIM.default_decode_bucket_top;
    }
    if (wl.phase === 'verify') {
      return rowsPerCall > CORE_LIM.default_verify_bucket_top;
    }
    return false;
  }
  // Return { familyGap, dtypeGap } matching support.py core gap rules.
  function evaluateCoreGaps(p, hasDecl) {
    if (hasDecl) return { familyGap: false, dtypeGap: false };
    const fams = (p.layer_families || []).filter(f => !STRUCTURAL_TAGS.has(f));
    const familyGap = fams.some(f => SUPPORTED_FAM.has(f) && !CORE_FAM.has(f));
    const weightDtype = String(p.weight_dtype || '').toLowerCase();
    const expertDtype = String(p.expert_dtype || p.weight_dtype || '').toLowerCase();
    const dtypeGap = !CORE_DTYPES.has(weightDtype) || !CORE_DTYPES.has(expertDtype);
    return { familyGap, dtypeGap };
  }
  // Return declared expert_owner_groups (> 0) for p matching wl by sameLayout, else undefined.
  function declaredExpertOwnerGroups(p, wl) {
    const match = declaredFor(p ? p.name : '').find(
      d => d && String(d.source || '').trim() && d.ranks > 0 && sameLayout(d, wl)
    );
    return (match && match.expert_owner_groups > 0) ? match.expert_owner_groups : undefined;
  }
  // Return "Attn: ... · MoE: ..." string for layout and optional expert owner groups.
  function sublayerLabel(layout, expertOwnerGroups) {
    const layoutAxes = layout || {};
    const tp = layoutAxes.tp !== undefined ? layoutAxes.tp : 1;
    const ep = layoutAxes.ep !== undefined ? layoutAxes.ep : 1;
    const attnAxes = [
      ['TP', tp],
      ['DP', layoutAxes.dp !== undefined ? layoutAxes.dp : 1],
      ['PP', layoutAxes.pp !== undefined ? layoutAxes.pp : 1],
      ['PCP', layoutAxes.pcp !== undefined ? layoutAxes.pcp : 1],
      ['DCP', layoutAxes.dcp !== undefined ? layoutAxes.dcp : 1],
    ];
    const attnParts = [];
    for (const [axisName, degree] of attnAxes) {
      if (degree > 1) attnParts.push(`${axisName}${degree}`);
    }
    const attention = attnParts.length ? attnParts.join(' x ') : 'TP1';
    if (ep <= 1) return `Attn: ${attention} \u00b7 MoE: TP${tp}`;
    const rawGroups = (expertOwnerGroups !== undefined && expertOwnerGroups !== null)
      ? expertOwnerGroups : ep;
    const parsedGroups = Number(rawGroups);
    const groups = Number.isFinite(parsedGroups) ? Math.trunc(parsedGroups) : ep;
    const ranks = Math.max(ep, tp);
    const innerTp = Math.max(1, Math.floor(ranks / Math.max(1, groups)));
    const moe = innerTp === 1 ? `EP${groups}` : `TP${innerTp} x EP${groups}`;
    return `Attn: ${attention} \u00b7 MoE: ${moe}`;
  }
  function evaluateSupport(sc, p, wl, ch, ctx) {
    const batchRaw = wl.batch !== undefined ? wl.batch : WL_DEF.batch;
    if (sc === 'per_op' || batchRaw <= 0) {
      return { support: 'baseline', blockers: [] };
    }
    // A declaration states that a runtime exists for this model, scope, phase, rank
    // count, and layout, so it supplies the row limit and layer families checked below.
    const blocks = declaredBlocks(p, sc, wl.phase);
    const rankMatches = blocks.filter(d => ctx.effTp === d.ranks);
    const decl = rankMatches.find(d => sameLayout(d, wl)) || null;
    const gaps = evaluateCoreGaps(p, Boolean(decl));
    const maxAr = CORE_LIM.max_ar_slots_per_layer;
    const isPeriodic = p.is_periodic !== undefined ? Boolean(p.is_periodic) : true;
    const arSlots = p.ar_slots_per_layer !== undefined ? p.ar_slots_per_layer : 2;
    const plannerBad = Boolean(ctx.planError) || (!decl && !isPeriodic) || (arSlots > maxAr);
    const chipName = String(ch.name || '').toLowerCase();
    const chipKind = (chipName === 'tpu7x' || chipName === '7x') ? 'tpu7x' : chipName;
    const rawWlOv = wl.host_table_hbm_overrides ?? wl.hostTableHbmOverrides;
    const hasWlOv = Array.isArray(rawWlOv) ? rawWlOv.length > 0
      : (rawWlOv && typeof rawWlOv === 'object' && Object.keys(rawWlOv).length > 0);
    const effWl = (!hasWlOv && decl && decl.host_table_hbm_overrides)
      ? Object.assign({}, wl, { host_table_hbm_overrides: decl.host_table_hbm_overrides })
      : wl;
    const blockers = [];
    if (sc === 'whole_step' && activeFusionBreakLayers(p, effWl).length > 0) {
      blockers.push('fusion_break');
    }
    if (sc === 'single_layer' || sc === 'layer_group') {
      blockers.push('scope_without_core_counterpart');
    }
    if (ctx.activeTokens > wholeStepRowLimit(decl, wl)) {
      blockers.push('rows_exceed_core_max_batch_rows');
    }
    // Declared rank counts restrict a model. Without a declaration for this scope and
    // phase, the core rank grid applies.
    const rankBad = blocks.length > 0
      ? rankMatches.length === 0 : !VERIFIED_RANKS.has(ctx.effTp);
    if (rankBad) {
      blockers.push('rank_count_outside_core_grid');
    }
    if (rankMatches.length > 0 && !decl) {
      blockers.push('layout_not_declared');
    }
    if (!CHIP_KINDS.has(chipKind)) {
      blockers.push('chip_without_core_budget');
    }
    if (ctx.epDp > 1 && !(decl && decl.attention_dp_in_call)) {
      blockers.push('all_to_all_not_in_core');
    }
    if (gaps.familyGap) blockers.push('family_not_in_core');
    if (gaps.dtypeGap) blockers.push('dtype_not_in_core');
    if (plannerBad) blockers.push('planner_rejected');
    const maxRowsPerCall = wl.maxRowsPerCall ?? wl.max_rows_per_call ?? 0;
    if (sc === 'whole_step' && !decl && maxRowsPerCall === 0 &&
        exceedsDefaultBuckets(wl, ctx.activeTokens)) {
      blockers.push('row_limit_not_stated');
    }
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
    const hasDsa = hasDsaProfile(p);
    const wordAlignBytes = hasDsa ? readPositiveSeed('word_align_bytes') : 0;
    const bf16Bytes = hasDsa ? readPositiveSeed('bf16_bytes') : 0;
    const indexerIndicesBytes = hasDsa
      ? wordAlignBytes * Math.ceil((p.index_topk || 0) / kvSplit) : 0;
    const numIndexerLayers = p.num_indexer_layers || 0;
    const numAttentionLayers = p.num_attention_layers || 0;
    const sharedIndexerSpineBytes = (
      hasDsa && numIndexerLayers > 0 && numIndexerLayers < numAttentionLayers
    ) ? indexerIndicesBytes : 0;
    const hc = p.hyper_connection;
    const resStreams = Math.max(1, (hc && hc.num_streams) || p.residual_streams || 1);
    const seedBf16 = readPositiveSeed('bf16_bytes');
    const seedFp32 = readPositiveSeed('fp32_bytes');
    const seedWordAlign = readPositiveSeed('word_align_bytes');
    const hcScratch = hc
      ? (hc.kind === 'lowrank'
         ? (2 * (hc.lowrank_dim || 0) * seedBf16)
         : (4 * resStreams * resStreams * seedFp32))
      : 0;
    const spine = (4 * resStreams + 2) * H + (effTp > 1 ? 6 * H : 0) +
      sharedIndexerSpineBytes + hcScratch;
    const nhLoc = Math.max(1, Math.ceil(p.num_attention_heads / effTp));
    const hd = p.head_dim;
    let attn = 0, lin = 0, moe = 0;
    if ((p.kv_lora_rank || 0) > 0) {
      const qlr = p.q_lora_rank || 0, qkr = p.qk_rope_head_dim || 0;
      attn = 2 * (qlr + p.kv_lora_rank + qkr) + 6 * (nhLoc * hd);
    } else {
      const nkvLoc = p.dcp_span === 'tp_group'
        ? wideDcpKvHeadsPerRank(p.num_kv_heads || 0, effTp, dcp, 'tp_group')
        : Math.max(1, Math.ceil(p.num_kv_heads / Math.max(1, Math.floor(effTp / kvSplit))));
      attn = 2 * (2 * nhLoc + 2 * nkvLoc) * hd + 4 * nhLoc * hd;
    }
    if ((p.o_lora_rank || 0) > 0) {
      attn += (p.o_groups || 1) * p.o_lora_rank * seedBf16;
    }
    if (p.sparse_attn) {
      const sp = p.sparse_attn;
      attn += (sp.index_heads || 0) * (sp.index_dim || 0) * seedBf16 +
        seedWordAlign * Math.ceil(((sp.window_tokens || 0) + (sp.select_tokens || 0)) / kvSplit);
    }
    if (KV_ONLY_DTYPES.has(normalizeKvDtype(p.kv_dtype))) {
      attn += 2 * hd * seedFp32;
    }
    if (hasDsa) {
      const queryGatherDim = dsaQueryGatherDim(p);
      attn += bf16Bytes * (
        (p.index_n_heads || 0) * (p.index_head_dim || 0) +
        (p.index_head_dim || 0) +
        (p.index_n_heads || 0)
      ) + (sharedIndexerSpineBytes ? 0 : indexerIndicesBytes) +
        (dcp > 1 ? bf16Bytes * (dcp - 1) * nhLoc * queryGatherDim : 0);
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
  const LOG_FACT_CACHE = [0.0];
  function logFactorial(n) {
    for (let i = LOG_FACT_CACHE.length; i <= n; i++) {
      LOG_FACT_CACHE.push(LOG_FACT_CACHE[i - 1] + Math.log(i));
    }
    return LOG_FACT_CACHE[n];
  }
  // Expected per-rank active experts on the busiest of ownerGroups independent
  // Binomial(E_g, hit) expert groups (E_g = numExperts / ownerGroups), evaluated
  // via sum_{m=0}^{E_g - 1} (1 - F(m)^ownerGroups) in log space and scaled by
  // expertsLocal / E_g (floored at 1.0).
  function expertsHitBusiest(tokensEp, topK, numExperts, expertsLocal, ownerGroups) {
    if (numExperts <= 0 || expertsLocal <= 0 || ownerGroups > numExperts) {
      throw new Error(
        `expertsHitBusiest invalid inputs: numExperts=${numExperts}, expertsLocal=${expertsLocal}, ownerGroups=${ownerGroups}`
      );
    }
    if (ownerGroups <= 1) {
      return expertsHit(tokensEp, topK, numExperts, expertsLocal);
    }
    const expertsPerGroup = Math.floor(numExperts / ownerGroups);
    const hit = 1.0 - Math.pow(1.0 - Math.min(1.0, topK / numExperts), tokensEp);
    if (hit <= 0.0) return 1.0;
    if (hit >= 1.0) return Math.max(1.0, expertsLocal);
    const logFactTotal = logFactorial(expertsPerGroup);
    const logHit = Math.log(hit);
    const logMiss = Math.log1p(-hit);
    let cdf = 0.0, expectedGroupMax = 0.0;
    for (let count = 0; count < expertsPerGroup; count++) {
      const logPmf = logFactTotal - logFactorial(count) -
        logFactorial(expertsPerGroup - count) +
        count * logHit + (expertsPerGroup - count) * logMiss;
      cdf = Math.min(1.0, cdf + Math.exp(logPmf));
      expectedGroupMax += 1.0 - Math.pow(cdf, ownerGroups);
    }
    return Math.max(1.0, expectedGroupMax * (expertsLocal / expertsPerGroup));
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
  function computeHbmUsedBytes(p, wl, chipObj) {
    const ch = chipObj || CHIP;
    const dtB = k => DTYPE_BYTES[String(p[k] || 'fp8').toLowerCase()] || 1.0;
    const bwW = dtB('weight_dtype'), bwE = dtB('expert_dtype');
    const kvDt = normalizeKvDtype(p.kv_dtype);
    const bwKv = DTYPE_BYTES[kvDt] || 1.0;
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
    const attnInfo = computeAttentionParams(p, effTp, kvSplit, dcp);
    const nhLoc = attnInfo.nhLoc, kvhLoc = attnInfo.kvhLoc, attnParams = attnInfo.attnParams;
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
    const bf16Bytes = readPositiveSeed('bf16_bytes');
    const indexerStatic = p.sparse_attn
      ? computeSparseAttnTerms(
          p, wl,
          { effTp, kvSplit, dcp, lAttn, layersLocal, batchLocal, activeTokens: batchLocal, effCtx },
          kvhLoc, bwKv, effCtx
        )
      : computeIndexerStaticAndResidentTerms(p, layersLocal, lAttn, batchLocal, effCtx, kvSplit);
    const lIdx = p.sparse_attn
      ? ((p.sparse_attn.index_dim || 0) > 0 ? lAttn : 0) : indexerStatic.lIndexer;
    const hcTerms = computeHyperConnectionTerms(p, layersLocal, effTp, batchLocal);
    const tblProjLocal = (p.host_tables || []).reduce(
      (s, t) => s + Math.ceil((t.dense_proj_params || 0) / effTp), 0
    );
    const tblCtx = computeHostTableContextTerms(p, wl, effTp, dp, batchLocal, batchLocal, ch);
    const baseDenseParams = lAttn * attnParams +
      lIdx * indexerStatic.indexerQueryProjParams +
      lLin * linParams + (layersLocal - lMoe) * denseFfnParams + lMoe * sharedExpParams;
    // Indexer key/weight, HC, and host-table projections stay unquantized in BF16.
    const wStatic = baseDenseParams * bwW + lMoe * H * nExp * bf16Bytes +
      lIdx * indexerStatic.indexerKeyWeightProjParams * bf16Bytes +
      (hcTerms.params + tblProjLocal) * bf16Bytes;
    const expBytes = routedExpParams * bwE;
    const kvComp = Math.min(1.0, Math.max(1e-6, p.kv_compress_factor ?? 1.0));
    const tokSplit = p.dcp_span === 'tp_group' ? dcp : kvSplit;
    const kvPageSz = wl.kv_page_size ?? wl.kvPageSize ?? WL_DEF.kv_page_size ?? 64;
    let kvPerTok;
    if (kvLora === 0 && kvDt === 'turboquant_4bit_nc') {
      kvPerTok = (kvhLoc * turboquantSlotBytes(hd, kvPageSz) * kvComp) / tokSplit;
    } else if (kvLora === 0 && kvDt === 'int4_per_token_head') {
      kvPerTok = ((2.0 * kvhLoc * hd * 0.5 + readPositiveSeed('int4_scale_page_bytes_per_tok')) *
        kvComp) / tokSplit;
    } else {
      const kvRawTok = kvLora ? (kvLora + qkRope) : 2.0 * kvhLoc * hd;
      kvPerTok = (kvRawTok * bwKv * kvComp) / tokSplit;
    }
    const fullKvBytes = lAttn * batchLocal * effCtx * kvPerTok;
    const residentKvBytes = p.sparse_attn
      ? indexerStatic.indexerKResidentBytes
      : (indexerStatic.hasDsa ? (fullKvBytes + indexerStatic.indexerKResidentBytes) : fullKvBytes);
    const recRwBytes = 2.0 * lLin * batchLocal *
      (zDim * kd + qkvLin * (p.linear_conv_kernel || 0)) * bwSt;
    return wStatic + lMoe * expertsLoc * expBytes + residentKvBytes + 0.5 * recRwBytes + tblCtx.tblHbmB;
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
    if (!dcpDividesMesh(p, effTp, dcp, pcp)) reasons.push('dcp_not_dividing_tp');
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
    const hbmUsed = computeHbmUsedBytes(p, wl, ch);
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
  function resolveMeasuredArS(callsRaw, effTp, path, wl) {
    if (effTp <= 1 || !Array.isArray(callsRaw) || callsRaw.length === 0) return null;
    if (wl && (wl.source_kind === 'architectural_prior' ||
               wl.sourceKind === 'architectural_prior')) {
      return null;
    }
    const matched = [];
    for (const c of callsRaw) {
      if (!c || typeof c !== 'object') continue;
      const cPath = (c.path === undefined || c.path === null) ? 'xla' : String(c.path);
      const cOp = (c.op === undefined || c.op === null)
        ? '' : String(c.op).trim().toLowerCase().replace(/_/g, '-');
      const cGrp = Math.max(2, Math.trunc(Number(c.group_size ?? c.groupSize ?? 2)) || 2);
      if (cPath !== path || cOp !== 'all-reduce' || cGrp !== effTp) continue;
      const hasMean = typeof c.mean_latency_us === 'number' &&
        Number.isFinite(c.mean_latency_us);
      const hasMed = typeof c.latency_us === 'number' && Number.isFinite(c.latency_us);
      if (!hasMean && !hasMed) continue;
      const valUs = hasMean ? c.mean_latency_us : c.latency_us;
      const nVal = Math.max(0, Math.trunc(Number(c.n ?? 1)));
      matched.push({ valUs, n: nVal });
    }
    if (matched.length === 0) return null;
    const totalN = matched.reduce((acc, m) => acc + m.n, 0);
    const combinedUs = totalN > 0
      ? (matched.reduce((acc, m) => acc + m.valUs * m.n, 0) / totalN)
      : (matched.reduce((acc, m) => acc + m.valUs, 0) / matched.length);
    return combinedUs * 1e-6;
  }
  function evaluateWorkload(wl, profOverride, chipOverride) {
    const p = profOverride || PROF, ch = chipOverride || CHIP;
    const dtB = k => DTYPE_BYTES[String(p[k] || 'fp8').toLowerCase()] || 1.0;
    const bwW = dtB('weight_dtype'), bwE = dtB('expert_dtype');
    const kvDtNorm = normalizeKvDtype(p.kv_dtype);
    const bwKv = DTYPE_BYTES[kvDtNorm] || 1.0;
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
    const attnInfo = computeAttentionParams(p, effTp, kvSplit, dcp);
    const nhLoc = attnInfo.nhLoc, kvhLoc = attnInfo.kvhLoc;
    const qDim = attnInfo.qDim, attnParams = attnInfo.attnParams;
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
    const seedBf16Bytes = readPositiveSeed('bf16_bytes');
    const seedFp32Bytes = readPositiveSeed('fp32_bytes');
    const indexerRadixPasses = readPositiveSeed('indexer_radix_passes');
    const indexerRadixBuckets = readPositiveSeed('indexer_radix_buckets');
    const indexerHeadReduceOps = readPositiveSeed('indexer_head_reduce_vec_ops');
    const indexerScaleOps = readPositiveSeed('indexer_scale_vec_ops');
    const indexerRadixPassOps = readPositiveSeed('indexer_radix_pass_vec_ops');
    const dsaUnpackMaskOps = readPositiveSeed('dsa_unpack_mask_vec_ops');
    const dsaSoftmaxOps = readPositiveSeed('dsa_softmax_vec_ops');
    const dsaOutputRescaleOps = readPositiveSeed('dsa_output_rescale_vec_ops');
    const dcpLseMergeOps = readPositiveSeed('dcp_lse_merge_vec_ops');
    const perOpIndexerScoreRoundtripBytes = readPositiveSeed('per_op_indexer_score_roundtrip_bytes');
    const perOpDsaKvRoundtripFactor = readPositiveSeed('per_op_dsa_kv_roundtrip_factor');
    const tPrefill = wl.phase === 'prefill'
      ? activeTokens : (wl.phase === 'mixed' ? chunkTokens : 0);
    const rawAttnCtx = (wl.phase === 'prefill' && tPrefill > 1)
      ? (Math.max(0, effCtx - tPrefill) + (tPrefill + 1) / 2.0) : effCtx;
    const kvComp = Math.min(1.0, Math.max(1e-6, p.kv_compress_factor ?? 1.0));
    const resStreams = Math.max(
      1, (p.hyper_connection && p.hyper_connection.num_streams) || p.residual_streams || 1
    );
    const hasSparseSpec = p.sparse_attn != null;
    const tokSplit = p.dcp_span === 'tp_group' ? dcp : kvSplit;
    const kvPageSz = wl.kv_page_size ?? wl.kvPageSize ?? WL_DEF.kv_page_size ?? 64;
    const rawKvElements = kvLora ? (kvLora + qkRope) : 2.0 * kvhLoc * hd;
    let kvBytesPerTok;
    if (kvLora === 0 && kvDtNorm === 'turboquant_4bit_nc') {
      kvBytesPerTok = (kvhLoc * turboquantSlotBytes(hd, kvPageSz) * kvComp) / tokSplit;
    } else if (kvLora === 0 && kvDtNorm === 'int4_per_token_head') {
      kvBytesPerTok = (
        (2.0 * kvhLoc * hd * 0.5 + readPositiveSeed('int4_scale_page_bytes_per_tok')) * kvComp
      ) / tokSplit;
    } else {
      kvBytesPerTok = (rawKvElements * bwKv * kvComp) / tokSplit;
    }
    const sparseTerms = hasSparseSpec
      ? computeSparseAttnTerms(
          p, wl,
          { effTp, kvSplit, dcp, lAttn, layersLocal, batchLocal, activeTokens, effCtx },
          kvhLoc, bwKv, rawAttnCtx
        )
      : null;
    const indexerStatic = hasSparseSpec
      ? sparseTerms
      : computeIndexerStaticAndResidentTerms(p, layersLocal, lAttn, batchLocal, effCtx, kvSplit);
    const hasLegacyDsa = !hasSparseSpec && indexerStatic.hasDsa;
    const hasDsa = hasSparseSpec ? sparseTerms.hasDsa : hasLegacyDsa;
    const lIndexer = hasSparseSpec
      ? ((p.sparse_attn.index_dim || 0) > 0 ? lAttn : 0) : indexerStatic.lIndexer;
    const indexerQueryProjParams = indexerStatic.indexerQueryProjParams;
    const indexerKeyWeightProjParams = indexerStatic.indexerKeyWeightProjParams;
    const isSparseGather = hasSparseSpec
      ? sparseTerms.isSparseGather : (hasLegacyDsa && (rawAttnCtx > (p.index_topk || 0)));
    const indexerCtxLocal = hasSparseSpec
      ? sparseTerms.indexerCtxLocal : (hasLegacyDsa ? (rawAttnCtx / kvSplit) : 0.0);
    const selectedTopk = hasSparseSpec
      ? sparseTerms.attnCtx : (hasLegacyDsa ? Math.min(rawAttnCtx, p.index_topk || 0) : rawAttnCtx);
    const localSelectedTopk = hasLegacyDsa ? (selectedTopk / kvSplit) : 0.0;
    const indexerKReadBytes = hasSparseSpec
      ? sparseTerms.indexerKReadBytes
      : (hasLegacyDsa
         ? (lIndexer * batchLocal * indexerCtxLocal * indexerStatic.indexerKeyRecordBytes) : 0.0);
    const dsaEntryBytes = hasLegacyDsa
      ? (kvLora > 0
         ? (padCachelineBytes(kvLora * bwKv) + padCachelineBytes(qkRope * bwKv))
         : padCachelineBytes(rawKvElements * bwKv))
      : 0.0;
    const dsaKvStrictBytes = hasSparseSpec
      ? sparseTerms.dsaKvStrictBytes
      : (hasLegacyDsa
         ? (lAttn * batchLocal * localSelectedTopk * rawKvElements * bwKv * kvComp) : 0.0);
    const dsaKvGatherBytes = hasSparseSpec
      ? sparseTerms.dsaKvGatherBytes
      : (hasLegacyDsa
         ? (isSparseGather ? (lAttn * activeTokens * localSelectedTopk * dsaEntryBytes) : dsaKvStrictBytes)
         : 0.0);
    const hcTerms = computeHyperConnectionTerms(p, layersLocal, effTp, activeTokens);
    const tblProjLocal = (p.host_tables || []).reduce(
      (s, t) => s + Math.ceil((t.dense_proj_params || 0) / effTp), 0
    );
    const tblCtx = computeHostTableContextTerms(p, wl, effTp, dp, activeTokens, batchLocal, ch);
    const baseDenseParams = lAttn * attnParams + lIndexer * indexerQueryProjParams +
      lLin * linParams + (layersLocal - lMoe) * denseFfnParams + lMoe * sharedExpParams;
    // Indexer key/weight, HC, and host-table projections stay unquantized in BF16.
    const wStatic = baseDenseParams * bwW + lMoe * H * nExp * seedBf16Bytes +
      lIndexer * indexerKeyWeightProjParams * seedBf16Bytes +
      (hcTerms.params + tblProjLocal) * seedBf16Bytes;
    const expBytes = routedExpParams * bwE;
    const fullKvBytes = lAttn * batchLocal * effCtx * kvBytesPerTok;
    const residentKvBytes = hasSparseSpec
      ? sparseTerms.indexerKResidentBytes
      : (hasLegacyDsa ? (fullKvBytes + indexerStatic.indexerKResidentBytes) : fullKvBytes);
    const kvReadBytes = (hasDsa ? (indexerKReadBytes + dsaKvGatherBytes) : fullKvBytes) +
      tblCtx.tblReadB;
    const recRwBytes = 2.0 * lLin * batchLocal *
      (zDim * kd + qkvLin * (p.linear_conv_kernel || 0)) * bwSt;
    const vmemUsable = ch.vmem_usable_bytes;
    const vmemHeadroom = ch.vmem_fit_headroom_bytes || 0;
    const rawVmemMib = wl.vmemMib !== undefined
      ? wl.vmemMib : (wl.vmem_budget_mib !== undefined
      ? wl.vmem_budget_mib : undefined);
    const rpObj = wl.ringPlan || wl.ring_plan;
    const spillRes = (rpObj && rpObj.compiler_spill_reserve_bytes) || 0;
    const fitHead = Boolean(wl.vmemFitHeadroom || wl.vmem_fit_headroom);
    const baseVmemBytes = (rawVmemMib !== undefined && rawVmemMib !== null)
      ? Math.trunc(rawVmemMib * 1048576)
      : Math.max(
        0,
        (wl.tiles && wl.tiles.vmem_mib > 0
          ? (wl.tiles.vmem_mib * 1048576) : vmemUsable) - vmemHeadroom
      );
    const vmemBudget = Math.min(
      vmemUsable, Math.max(0, baseVmemBytes - (fitHead ? spillRes : 0))
    );
    const ffnChoose = lMoe > 0 ? sharedExpParams : denseFfnParams;
    const indexerLayerDenseParams = lIndexer > 0 ? indexerQueryProjParams : 0;
    const indexerLayerBf16Bytes = lIndexer > 0
      ? indexerKeyWeightProjParams * seedBf16Bytes : 0.0;
    const hcPerLayerBf16 = hcTerms.params > 0
      ? (Math.ceil(hcTerms.params / Math.max(1, layersLocal)) * seedBf16Bytes) : 0.0;
    const layerDenseBytes = Math.trunc(
      (Math.max(attnParams + indexerLayerDenseParams, linParams) + ffnChoose) * bwW +
      indexerLayerBf16Bytes + hcPerLayerBf16
    );
    const dMla = (activeTokens - tPrefill) * (2 * kvLora + qkRope) +
      tPrefill * (qkNope + qkRope + hd);
    const attnCtx = selectedTopk;
    const cedScale = cedTokenScale(p, wl, activeTokens, chunkTokens, layersLocal);
    const rawFlopsKv = 2.0 * lAttn * attnCtx * kvComp *
      (kvLora ? nhLoc * dMla : 2.0 * activeTokens * qDim);
    const flopsKv = rawFlopsKv * cedScale;
    const flopsIndexerScore = hasSparseSpec
      ? sparseTerms.flopsIndexerScore
      : (hasLegacyDsa
         ? (2.0 * lIndexer * activeTokens * (p.index_n_heads || 0) *
            (p.index_head_dim || 0) * indexerCtxLocal)
         : 0.0);
    const routerFlops = lMoe * H * nExp;
    const flopsDenseBase = 2.0 * activeTokens * (
      baseDenseParams + routerFlops + lIndexer * indexerKeyWeightProjParams
    ) + rawFlopsKv + flopsIndexerScore +
      hcTerms.flops + 2.0 * activeTokens * tblProjLocal;
    const flopsDense = flopsDenseBase * cedScale;
    const vecOpsIndexerRadix = (hasLegacyDsa && isSparseGather)
      ? (indexerRadixPasses * indexerRadixPassOps * lIndexer * activeTokens * indexerCtxLocal)
      : 0.0;
    const hadamardVecOps = computeHadamardKvVecOps(kvDtNorm, nhLoc, kvhLoc, activeTokens, lAttn);
    const vecOpsPreDsaExtra = (hasLegacyDsa
      ? ((indexerHeadReduceOps * (p.index_n_heads || 0) + indexerScaleOps) *
         lIndexer * activeTokens * indexerCtxLocal + vecOpsIndexerRadix)
      : 0.0) + hcTerms.vecOps + hadamardVecOps;
    let vecOpsDsa = 0.0;
    if (hasLegacyDsa) {
      const selectedTopkPadded = padVectorLanes(localSelectedTopk);
      const kvElementsPadded = padVectorLanes(rawKvElements);
      const headsWithDcp = nhLoc * dcp;
      const dsaValueDim = kvLora > 0 ? kvLora : hd;
      const dcpMergeOps = dcp > 1 ? (dcpLseMergeOps * dcp * nhLoc * dsaValueDim) : 0.0;
      vecOpsDsa = lAttn * activeTokens * (
        dsaUnpackMaskOps * selectedTopkPadded * kvElementsPadded +
        dsaSoftmaxOps * headsWithDcp * selectedTopkPadded +
        dsaOutputRescaleOps * headsWithDcp * dsaValueDim +
        dcpMergeOps
      );
    }
    const bufSlots = wl.dmaBuffers || wl.dma_buffers || 2;
    const dBank = (wl.tiles && wl.tiles.bank)
      ? wl.tiles.bank : ((rpObj && rpObj.dense_slot_bytes)
      ? rpObj.dense_slot_bytes : (wl.denseSlotBytes || CORE_LIM.bank));
    const eBank = (wl.tiles && wl.tiles.ebank)
      ? wl.tiles.ebank : ((rpObj && rpObj.expert_slot_bytes)
      ? rpObj.expert_slot_bytes : (wl.expertSlotBytes || CORE_LIM.ebank));
    const slDenseStaging = Math.min(bufSlots * dBank, bufSlots * layerDenseBytes);
    const firstSlot = Math.min(
      dBank, Math.max(262144, Math.floor(layerDenseBytes / 4))
    );
    const gqaScratch = lAttn > 0
      ? (2 * 128 * (4 * Math.max(hd, kvLora)) + 2 * 128 * 128 * 4) : 0;
    const gdnScratch = lLin > 0 ? (4 * Math.min(vhLoc, 8) * kd * vd) : 0;
    const attnTileTokens = readPositiveSeed('attn_tile_tokens');
    const denseFfnTileCap = readPositiveSeed('dense_ffn_tile_cap');
    const computeFfnScratchBytes = moeInterLocal => lMoe > 0
      ? (2 * attnTileTokens * moeInterLocal * seedFp32Bytes)
      : (2 * attnTileTokens * denseFfnTileCap * seedFp32Bytes);
    const ffnScratch = computeFfnScratchBytes(iExpLoc);
    const vmemParts = computePerTokenVmem(wl, p);
    const defCollUs = (ch.collective_latency_s || 1e-6) * 1e6;
    const bracketHi = ch.collective_latency_us_bracket
      ? ch.collective_latency_us_bracket[1] : defCollUs;
    const collUs = wl.collHopUs !== undefined ? wl.collHopUs : defCollUs;
    const collHiUs = Math.max(collUs, wl.collHopHiUs ?? bracketHi);
    const latColl = collUs * 1e-6, latCollHi = collHiUs * 1e-6;
    const rawCollectiveCalls = wl.collectiveCalls || wl.collective_calls || [];
    const measuredArXlaS = resolveMeasuredArS(rawCollectiveCalls, effTp, 'xla', wl);
    const measuredArInKernelS = resolveMeasuredArS(
      rawCollectiveCalls, effTp, 'in_kernel', wl
    );
    const arInfo = allReduceCost(tokResBytes, effTp, latColl, ch, wl);
    const numAr = (p.ar_slots_per_layer || 2) * layersLocal;
    const bwHbm = ch.hbm_bw_bytes_per_s;
    const peakW = peakFlops(p.weight_dtype, ch, p.act_dtype);
    const peakE = peakFlops(
      p.expert_dtype || p.weight_dtype, ch, p.expert_act_dtype || p.act_dtype
    );
    const bf16Peak = ch.bf16_flops_per_s || peakW;
    const rFloor = rowFloor(lMoe > 0 ? p.expert_dtype : p.weight_dtype, ch);
    const iciEg = ch.ici_egress_bw_bytes_per_s;
    const iciLink = ch.ici_link_bw_bytes_per_s;
    const ppStageTime = pp > 1
      ? (tokResBytes / iciLink + latColl) : 0.0;
    const pcpRingTime = (effPcp > 1 && lAttn > 0)
      ? ((effPcp - 1) * lAttn * chunkTokens * (kvBytesPerTok * kvSplit) / iciLink)
      : 0.0;
    const fusionBreaks = activeFusionBreakLayers(p, wl);
    const hasFusionBreak = fusionBreaks.length > 0;
    const breakGroups = fusionBreakLayerGroups(L, fusionBreaks);
    const wlLg = wl.layerGroups || wl.layer_groups;
    const lgArr = (Array.isArray(wlLg) && wlLg.length > 0)
      ? wlLg : (breakGroups.length > 0 ? breakGroups : (p.layer_groups || []));
    const presetGroups = lgArr.length > 0
      ? Math.ceil(lgArr.length * layersLocal / L) : 0;
    const segLayers = wl.segmentLayers ?? wl.segment_layers ?? 0;
    const hasSeg = segLayers > 0 && segLayers < layersLocal;
    const segCount = hasSeg ? Math.ceil(layersLocal / segLayers) : 1;
    const noWholeStep = pp > 1 || segCount > 1 || lgArr.length > 1 ||
      hasFusionBreak || L === 1 || Boolean(wl.planError);
    const hbmUsed = wStatic + lMoe * expertsLoc * expBytes +
      residentKvBytes + 0.5 * recRwBytes + tblCtx.tblHbmB;
    let dcpHops = 0, dcpQueryAllGatherTransferS = 0.0, dcpQueryAllGatherHops = 0;
    let dcpPostAttnTransferS = 0.0, dcpPostAttnHops = 0;
    if (hasLegacyDsa && dcp > 1) {
      dcpHops = roundEven(Math.log2(dcp));
      const queryGatherDim = dsaQueryGatherDim(p);
      const outputMergeDim = kvLora > 0 ? kvLora : hd;
      const dcpQueryAllGatherBytes = lAttn * (dcp - 1) * activeTokens * nhLoc *
        queryGatherDim * seedBf16Bytes;
      const dcpPartialOutputAllToAllBytes = lAttn * (dcp - 1) * activeTokens * nhLoc *
        (outputMergeDim * seedBf16Bytes + seedFp32Bytes);
      const indexerMinExchangeElements = Math.min(
        p.index_topk || 0, indexerRadixPasses * indexerRadixBuckets
      );
      const dcpIndexerMinBytes = isSparseGather
        ? (lIndexer * (dcp - 1) * activeTokens * indexerMinExchangeElements * seedFp32Bytes)
        : 0.0;
      dcpQueryAllGatherTransferS = dcpQueryAllGatherBytes / iciEg;
      dcpQueryAllGatherHops = lAttn * dcpHops;
      dcpPostAttnTransferS = (dcpPartialOutputAllToAllBytes + dcpIndexerMinBytes) / iciEg;
      dcpPostAttnHops = (lAttn + (isSparseGather ? lIndexer : 0)) * dcpHops;
    }
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
    const bf16Bytes = Number(DATA.bf16_bytes);
    const fp32Bytes = Number(DATA.fp32_bytes);
    const lseScalarsPerHead = Number(DATA.lse_scalars_per_head);
    const defaultCallsPerLayer = Number(DATA.default_declared_calls_per_layer);
    const perOpCollFactor = Number(DATA.per_op_collective_call_factor);
    if (
      !Number.isFinite(bf16Bytes) ||
      !Number.isFinite(fp32Bytes) ||
      !Number.isFinite(lseScalarsPerHead) ||
      !Number.isFinite(defaultCallsPerLayer) ||
      !Number.isFinite(perOpCollFactor)
    ) {
      throw new Error('Missing required collective seed constants in DATA');
    }
    const collectiveSeed = {
      bf16Bytes, fp32Bytes, lseScalarsPerHead, defaultCallsPerLayer, perOpCollFactor
    };
    const hostHeadPerOpUs = Number(wl.host_head_per_op_us ?? wl.hostHeadPerOpUs ?? 0.0);
    const hostHeadFusedUs = Number(wl.host_head_fused_us ?? wl.hostHeadFusedUs ?? 0.0);

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
        Object.assign({}, wl, {
          batch: n * dp,
          host_head_fused_us: 0.0,
          hostHeadFusedUs: 0.0
        }), p, ch
      ).scopes.find(s => s.scope === 'whole_step');
      const full = callScope(perCall), rest = restReq > 0 ? callScope(restReq) : null;
      const calls = Array(fullCalls).fill(full).concat(rest ? [rest] : []);
      const sum = (key, digits) => +calls.reduce((a, c) => a + c[key], 0).toFixed(digits);
      const stepDecl = findDeclaration(p, 'whole_step', effTp, wl);
      const stepExpertMode = resolveDeclaredExpertMode(stepDecl, wl, activeTokens, lMoe);
      const stepEpDp = stepExpertMode === 'tp' ? 1 : epDp;
      // Support belongs to the whole step: its total rows are checked against the rows
      // a declaration serves through split calls.
      const stepSupport = evaluateSupport('whole_step', p, wl, ch, {
        activeTokens, effTp, epDp: stepEpDp, planError: wl.planError || ''
      });
      const stepHeadUs = hostHeadFusedUs;
      return Object.assign({}, full, {
        support: stepSupport.support, blockers: stepSupport.blockers,
        expert_mode: stepExpertMode,
        host_head_us: +stepHeadUs.toFixed(2),
        feasible: calls.every(c => c.feasible),
        reason: (calls.find(c => !c.feasible) || full).reason,
        nCalls: calls.length,
        ops_per_step: sum('ops_per_step', 0), n_splits: sum('n_splits', 0),
        rawStepUs: fullCalls * full.rawStepUs + (rest ? rest.rawStepUs : 0.0) + stepHeadUs,
        pureStepUs: fullCalls * full.pureStepUs + (rest ? rest.pureStepUs : 0.0),
        w_step_gb: sum('w_step_gb', 4), w_step_strict_gb: sum('w_step_strict_gb', 5),
        act_hbm_mb: sum('act_hbm_mb', 3),
        compute_us: sum('compute_us', 2), chain_us: sum('chain_us', 2),
        hbm_us: sum('hbm_us', 2), comm_us: sum('comm_us', 2),
        fill_drain_us: sum('fill_drain_us', 2),
        launch_us: sum('launch_us', 2), roof_floor_us: sum('roof_floor_us', 2),
        step_bound_us: +(sum('step_bound_us', 2) + stepHeadUs).toFixed(2),
        step_floor_strict_us: sum('step_floor_strict_us', 2),
        step_achievable_us: +(sum('step_achievable_us', 2) + stepHeadUs).toFixed(2),
        step_achievable_upper_us: +(sum('step_achievable_upper_us', 2) + stepHeadUs).toFixed(2),
        step_upper_us: +(sum('step_upper_us', 2) + stepHeadUs).toFixed(2)
      });
    }

    const scopes = [];
    for (const sc of ['per_op', 'single_layer', 'layer_group', 'whole_step']) {
      if (sc === 'whole_step' && singleCallRows > 0 && singleCallRows < activeTokens) {
        scopes.push(microbatchedWholeStep());
        continue;
      }
      const moeDecl = findDeclaration(p, sc, effTp, wl);
      const expertMode = resolveDeclaredExpertMode(moeDecl, wl, activeTokens, lMoe);
      const scopeEp = (expertMode === 'tp' && ep > 1) ? 1 : ep;
      const scopeWl = scopeEp === ep ? wl : Object.assign({}, wl, { ep: 1 });
      const scopeEpTp = gcd(scopeEp, effTp);
      const scopeEpDp = Math.max(1, Math.floor(scopeEp / scopeEpTp));
      const scopeExpertsLoc = lMoe > 0 ? Math.floor(nExp / scopeEp) : 0;
      const scopeIExpLoc = lMoe > 0
        ? Math.ceil((p.moe_intermediate_size || 0) / Math.max(1, Math.floor(effTp / scopeEpTp)))
        : 0;
      const scopeRoutedExpParams = 3 * (p.routed_expert_hidden_size || H) * scopeIExpLoc;
      const scopeExpBytes = scopeRoutedExpParams * bwE;
      const scopeFlopsMoe = scopeExpertsLoc > 0
        ? ((2.0 * activeTokens * scopeEpDp * lMoe * topK * scopeRoutedExpParams * scopeExpertsLoc / nExp) * cedScale)
        : 0.0;
      const scopeSlExpertStaging = Math.min(bufSlots * eBank, Math.trunc(bufSlots * scopeExpBytes));
      const scopeFfnScratch = computeFfnScratchBytes(scopeIExpLoc);
      const scopeVmemParts = scopeEp === ep ? vmemParts : computePerTokenVmem(scopeWl, p);
      const scopeA2aTime = scopeEpDp > 1
        ? (2.0 * (scopeEpDp - 1) / scopeEpDp * Math.min(Math.max(1, topK), scopeEpDp) * tokResBytes / iciEg)
        : 0.0;
      const scopeHostHeadUs = sc === 'per_op' ? hostHeadPerOpUs : hostHeadFusedUs;
      let feasible = true, reason = 'ok';
      if (sc === 'layer_group' && presetGroups === 0 && pp === 1 && segCount === 1) {
        feasible = false;
        reason = 'No layer grouping: needs a model preset, pp > 1, or segment_layers';
      } else if (sc === 'whole_step' && noWholeStep) {
        feasible = false;
        if (wl.planError) reason = wl.planError;
        else if (hasFusionBreak) reason = 'Fusion break at host-table or CED boundary';
        else reason = 'Bounded by layer groups or single layers';
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
        const sumScratch = gqaScratch + gdnScratch + scopeFfnScratch;
        opScratch = vmemAlias
          ? Math.max(gqaScratch, gdnScratch, scopeFfnScratch) : sumScratch;
        const alpha = scopeVmemParts.totalPerTokenBytes;
        let minStaticBytes = 0;
        if (isSlOrGrp) {
          selNbd = bufSlots;
          selNbe = lMoe > 0 ? bufSlots : 0;
          denseStage = slDenseStaging;
          expertStage = lMoe > 0 ? scopeSlExpertStaging : 0;
          padBytes = 0;
          staticBytes = denseStage + expertStage + opScratch;
          minStaticBytes = staticBytes;
        } else {
          padBytes = 2097152;
          const ringSel = selectAdaptiveRing(
            vmemBudget, opScratch + padBytes, alpha, activeTokens,
            lMoe > 0, scopeWl, dBank, eBank
          );
          selNbd = ringSel.nbd;
          selNbe = ringSel.nbe;
          denseStage = selNbd * dBank;
          expertStage = selNbe * eBank;
          staticBytes = ringSel.ringB + opScratch + padBytes;
          minStaticBytes = ringSel.minRingB + opScratch + padBytes;
        }
        actBytes = 2.0 * resStreams * (isSl ? layersLocal : mul) * tokResBytes;
        const subMul = (effTp * scopeEp > 1 && (lAttn + lLin) > 0) ? 3 : 1;
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
      const tileTokens = Math.ceil(activeTokens / Math.max(1, nSplits));
      const isGrpMode = moeMode === 'grouped_on_chip';
      const canGroupMoe = !isFusedMulti || scopeEpDp === 1 ||
        ((effTp * scopeEpDp <= 8) && isGrpMode);
      const declLayout = moeDecl ? String(moeDecl.moe_layout || '') : '';
      // A declared megakernel runs the expert layout its deployment states.
      const groupedMoe = declLayout === 'grouped' || (declLayout !== 'masked' &&
        canGroupMoe && (sc === 'per_op' || isGrpMode || tileTokens * scopeEpDp >= rFloor));
      const moeTok = (groupedMoe && sc !== 'single_layer')
        ? activeTokens : tileTokens;
      let wStep = wStatic, moePadFactor = 1.0;
      if (scopeExpertsLoc > 0) {
        const tEp = moeTok * scopeEpDp;
        const eHit = expertsHit(tEp, topK, nExp, scopeExpertsLoc);
        const pairs = Math.max(1.0, tEp * topK * scopeExpertsLoc / nExp);
        const avgM = Math.max(1.0, pairs / eHit);
        const hit = (tiled || sc === 'single_layer')
          ? expertsHit(activeTokens * scopeEpDp, topK, nExp, scopeExpertsLoc) : eHit;
        wStep = wStatic + lMoe * hit * scopeExpBytes;
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
      if (sc === 'per_op' && hasDsa && isSparseGather) {
        actBytes += perOpIndexerScoreRoundtripBytes * lIndexer * activeTokens * indexerCtxLocal +
          perOpDsaKvRoundtripFactor * dsaKvGatherBytes;
      }
      const layoutCollDecl = moeDecl || findDeclaration(p, 'whole_step', effTp, wl);
      const declColl = (
        layoutCollDecl && layoutCollDecl.collectives && typeof layoutCollDecl.collectives === 'object'
      ) ? layoutCollDecl.collectives : null;
      const effNumAr = (declColl && typeof declColl.all_reduces === 'number')
        ? Math.ceil(declColl.all_reduces * layersLocal / Math.max(1, L))
        : numAr;
      const overlappedComm = isFusedMulti && nSplits === 1;
      const computeScopeCollectives = (callFactorVal, allowMeasuredAr) => {
        const measuredArS = allowMeasuredAr
          ? (sc === 'per_op' ? measuredArXlaS : measuredArInKernelS)
          : null;
        const hasMeasuredAr = measuredArS !== null && effTp > 1;
        const measuredArTotalS = hasMeasuredAr ? (effNumAr * nSplits * measuredArS) : 0.0;
        const xlaArFactor = (sc === 'per_op' && effTp > 1) ? callFactorVal : 1;
        const xlaHbmCollBytes = (!hasMeasuredAr && sc === 'per_op' && effTp > 1)
          ? 2.0 * effNumAr * tokResBytes : 0.0;
        let stepsToPrice = [];
        let stepCallMult = 1;
        if (declColl) {
          stepsToPrice = Array.isArray(declColl.steps) ? declColl.steps : [];
          stepCallMult = sc === 'per_op' ? callFactorVal : 1;
        } else if (
          !hasLegacyDsa &&
          dcp > 1 &&
          (dcp & (dcp - 1)) === 0 &&
          (wl.phase === 'decode' || wl.phase === 'verify')
        ) {
          stepsToPrice = [{
            kind: 'dcp_gather', group_size: dcp, calls_per_layer: defaultCallsPerLayer
          }];
          stepCallMult = sc === 'per_op' ? callFactorVal : 1;
        }
        const hasDeclaredRouted = stepsToPrice.some(st => st && st.kind === 'routed_transport');
        const hasDeclaredDcp = stepsToPrice.some(
          st => st && (st.kind === 'dcp_gather' || st.kind === 'split_kv')
        );
        let stepTransferS = 0.0, stepStrictS = 0.0, stepHops = 0;
        const stepCtx = { activeTokens, effTp, lMoe, lAttn, nhLoc, iciLink, iciEg };
        for (const stItem of stepsToPrice) {
          const stepTerms = computeCollectiveStepTerms(
            stItem, p, stepCtx, nSplits, stepCallMult, expertMode, collectiveSeed
          );
          stepTransferS += stepTerms.transferS;
          stepStrictS += stepTerms.strictEgressS;
          stepHops += stepTerms.hops;
        }
        const priceDsaDcp = hasDsa && dcpPostAttnHops > 0 && !hasDeclaredDcp;
        const dcpPostHopsTotal = priceDsaDcp
          ? (nSplits * xlaArFactor * dcpPostAttnHops) : 0;
        stepHops += dcpPostHopsTotal;
        const dcpPostTransferScopeS = priceDsaDcp ? dcpPostAttnTransferS : 0.0;
        const stepCollS = stepTransferS + dcpPostTransferScopeS + stepHops * latColl;
        const arAndPcpS = (hasMeasuredAr
          ? measuredArTotalS
          : (effNumAr * (arInfo.xferS + nSplits * xlaArFactor * arInfo.latS) +
             xlaHbmCollBytes / bwHbm)
        ) + pcpRingTime + stepCollS;
        const a2aSyncS = hasDeclaredRouted
          ? 0.0
          : ((lMoe > 0 && scopeEpDp > 1)
             ? lMoe * (scopeA2aTime + 2 * nSplits * xlaArFactor * latColl) : 0.0);
        const dcnSyncS = hasMeasuredAr
          ? 0.0 : (effNumAr * nSplits * xlaArFactor * arInfo.interLatS);
        const syncStageS = overlappedComm
          ? (dcnSyncS + ppStageTime + a2aSyncS)
          : (isFusedMulti ? (arAndPcpS + ppStageTime + a2aSyncS) : 0.0);
        const commS = overlappedComm
          ? (arAndPcpS - dcnSyncS)
          : (isFusedMulti ? 0.0 : (arAndPcpS + ppStageTime + a2aSyncS));
        return {
          syncStageS, commS, xlaArFactor, measuredArTotalS, hasMeasuredAr,
          stepHops, hasDeclaredRouted, hasDeclaredDcp, stepTransferS, stepStrictS
        };
      };
      const floorColl = computeScopeCollectives(1, false);
      const upperColl = computeScopeCollectives(perOpCollFactor, false);
      const achColl = computeScopeCollectives(perOpCollFactor, true);
      const commS = floorColl.commS;
      const hasDeclaredRouted = floorColl.hasDeclaredRouted;
      const hasDeclaredDcp = floorColl.hasDeclaredDcp;
      const commInCore = overlappedComm || !isFusedMulti;
      const measuredInCommS = commInCore ? achColl.measuredArTotalS : 0.0;
      const measuredInSyncS = commInCore ? 0.0 : achColl.measuredArTotalS;
      const effStateBytes = kvReadBytes + recRwBytes;
      const minEffFloor = Number(DATA.min_efficiency_floor);
      const syncHopFactor = Number(DATA.boundary_sync_hop_factor);
      const perOpMixerKernels = Number(DATA.per_op_mixer_kernels_per_layer);
      const perOpBaseFfnKernels = Number(DATA.per_op_base_ffn_kernels_per_layer);
      const perOpSharedExpertKernels = Number(DATA.per_op_shared_expert_kernels_per_layer);
      if (
        !Number.isFinite(minEffFloor) ||
        !Number.isFinite(syncHopFactor) ||
        !Number.isFinite(perOpMixerKernels) ||
        !Number.isFinite(perOpBaseFfnKernels) ||
        !Number.isFinite(perOpSharedExpertKernels)
      ) {
        throw new Error('Missing required per-op kernel or boundary seed constants in DATA');
      }
      const boundaryFillDrainS = (
        boundaries, syncStageVal, effHbmVal, effIciVal, syncHopVal, includeFirstSlotHbm
      ) => {
        const firstSlotS = includeFirstSlotHbm ? ((firstSlot / bwHbm) / effHbmVal) : 0.0;
        const perCallS = firstSlotS + (syncHopVal * latColl) / effIciVal;
        const tiledSyncS = tiled
          ? (((nSplits - 1) * 2 * layersLocal * syncHopVal * latColl) / effIciVal)
          : 0.0;
        return (boundaries * perCallS + syncStageVal) + tiledSyncS;
      };
      // Floor excludes cold-slot HBM fill and boundary sync-hop priors.
      const fillDrainS = boundaryFillDrainS(
        nOps - 1, floorColl.syncStageS, 1.0, 1.0, 0.0, false
      );
      const upperFillDrainS = boundaryFillDrainS(
        nOps - 1, upperColl.syncStageS, 1.0, 1.0, syncHopFactor, true
      );
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
      const vecOpsDense = 8.0 * resStreams * activeTokens * (p.hidden_size / effTp) * layersLocal +
        4.0 * activeTokens * ffnDenseInterLocal * (layersLocal - lMoe) +
        4.0 * activeTokens * sharedInterLocal * lMoe + subchannelVecDense +
        vecOpsPreDsaExtra + vecOpsDsa;
      const routedBytesStreamed = wStep - wStatic;
      const streamedRoutedParams = Math.max(0.0, routedBytesStreamed / Math.max(1e-6, bwE));
      const iExpLocal = scopeIExpLoc;
      const moeTokensLocal = activeTokens * scopeEpDp * scopeExpertsLoc / Math.max(1, nExp);
      const isFp4Expert = ['nvfp4', 'mxfp4', 'fp4', 'int4'].includes(String(p.expert_dtype || '').toLowerCase());
      const subchannelVecMoe = (weightBlockK > 0 && weightBlockK < subchannelKThresh)
        ? ((moePadFactor * scopeFlopsMoe) / weightBlockK)
        : 0.0;
      const vecOpsMoe = 4.0 * moeTokensLocal * lMoe * topK * iExpLocal + (isFp4Expert ? 2.0 * streamedRoutedParams : 0.0) + subchannelVecMoe;
      const actDt = String(p.act_dtype || '').toLowerCase();
      const kvDt = kvDtNorm;
      const attnIsBf16 = actDt !== 'fp8' || kvDt !== 'fp8';
      const tMatmulDense = (attnIsBf16 && flopsKv > 0)
        ? ((flopsDense - flopsKv) / peakW + flopsKv / bf16Peak)
        : (flopsDense / peakW);
      const tMatmulDenseUpper = ((attnIsBf16 || isPrefOrMix) && flopsKv > 0)
        ? ((flopsDense - flopsKv) / peakW + flopsKv / bf16Peak)
        : tMatmulDense;
      const tDense = Math.max(tMatmulDense, vecOpsDense / vecRate);
      const tDenseUpper = Math.max(tMatmulDenseUpper, vecOpsDense / vecRate);
      const tMoe = (moePadFactor * scopeFlopsMoe) / peakE;
      const tVecMoe = vecOpsMoe / vecRate;
      const tHbm = Math.max((wStep + effStateBytes + actBytes) / bwHbm, tblCtx.hostH2dS);
      // Routed-expert bytes are requested after each layer's router, so they stream
      // under neither the dense compute nor the collectives.
      const routedHbmS = (wStep - wStatic) / bwHbm;
      const dsaStageSeconds = (
        latCollVal, bf16Upper, strictMode, effDenseVal, effHbmVal, effIciVal, xlaFactorVal
      ) => {
        if (!hasDsa) return 0.0;
        const effVecRate = strictMode
          ? ((DATA.vector_max_elements_per_word || 4) * vecRate) : vecRate;
        let vecPreDsa = Math.max(0.0, vecOpsDense - vecOpsDsa);
        if (strictMode) vecPreDsa = Math.max(0.0, vecPreDsa - vecOpsIndexerRadix);
        const tPreMatmul = ((flopsDense - flopsKv) / peakW) / effDenseVal;
        const useBf16 = attnIsBf16 || (bf16Upper && isPrefOrMix);
        const kvPeak = useBf16 ? bf16Peak : peakW;
        const tDsaMatmul = flopsKv > 0 ? ((flopsKv / kvPeak) / effDenseVal) : 0.0;
        const tPreDsa = Math.max(tPreMatmul, (vecPreDsa / effVecRate) / effDenseVal);
        const tDsaCompute = Math.max(tDsaMatmul, (vecOpsDsa / effVecRate) / effDenseVal);
        const dcpQueryAllGatherS = (dcpQueryAllGatherHops > 0 && !hasDeclaredDcp)
          ? ((dcpQueryAllGatherTransferS + nSplits * xlaFactorVal * dcpQueryAllGatherHops * latCollVal) / effIciVal)
          : 0.0;
        const dsaBytes = strictMode ? dsaKvStrictBytes : dsaKvGatherBytes;
        const dsaHbmS = (dsaBytes / bwHbm) / effHbmVal;
        return Math.max(tPreDsa, dcpQueryAllGatherS) + Math.max(tDsaCompute, dsaHbmS);
      };
      const dsaStageS = dsaStageSeconds(
        latColl, false, false, 1.0, 1.0, 1.0, floorColl.xlaArFactor
      );
      const chainS = routedChain(tDense, routedHbmS, tMoe, tVecMoe, dsaStageS);
      const roofCoreS = coreTime(tHbm, chainS, commS, isFusedMulti);
      const roofUs = (roofCoreS + fillDrainS) * 1e6;
      const pureStepUs = roofUs;
      const launchUs = nOps * lgBox.lo;
      const stepUs = pureStepUs + launchUs + scopeHostHeadUs;
      const etaMoe = (groupedMoe && lMoe > 0) ? etaMoeGrp : etaDense;
      const effHbm = Math.max(minEffFloor, etaHbm);
      const effIci = Math.max(minEffFloor, etaIci);
      const effDense = Math.max(minEffFloor, etaDense);
      const effMoe = Math.max(minEffFloor, etaMoe);
      const tMatmulDenseAch = (attnIsBf16 && flopsKv > 0)
        ? (((flopsDense - flopsKv) / peakW) / effDense +
           (flopsKv / bf16Peak) / effDense)
        : (flopsDense / peakW) / effDense;
      const tDenseAch = Math.max(
        tMatmulDenseAch,
        (vecOpsDense / vecRate) / effDense
      );
      const declOwnerGroups = moeDecl ? Number(moeDecl.expert_owner_groups || 0) : 0;
      const ownerGroups = expertMode === 'tp'
        ? 1 : (declOwnerGroups > 0 ? declOwnerGroups : scopeEp);
      const tokEpStep = activeTokens * scopeEpDp;
      const routedAchRatio = scopeExpertsLoc > 0
        ? (expertsHitBusiest(tokEpStep, topK, nExp, scopeExpertsLoc, ownerGroups) /
           expertsHit(tokEpStep, topK, nExp, scopeExpertsLoc))
        : 1.0;
      const perOpCalls = Math.max(
        1,
        (perOpMixerKernels + perOpBaseFfnKernels) * layersLocal +
          ((p.shared_expert_intermediate_size || 0) > 0
            ? perOpSharedExpertKernels * lMoe
            : 0)
      );
      const boundaryCount = sc === 'per_op' ? (perOpCalls - 1) : (nOps - 1);
      const achFillDrainS = boundaryFillDrainS(
        boundaryCount, achColl.syncStageS, effHbm, effIci, syncHopFactor, true
      );
      const tMoeAch = ((moePadFactor * routedAchRatio * scopeFlopsMoe) / peakE) / effMoe;
      const tVecMoeAch = ((vecOpsMoe * routedAchRatio) / vecRate) / effDense;
      const tHbmAch = (tHbm + (routedAchRatio - 1.0) * routedHbmS) / effHbm;
      const indexerAchievableExtraHops = (
        sc !== 'per_op' && !hasDeclaredDcp && hasDsa && isSparseGather && dcpHops > 0
      ) ? ((indexerRadixPasses - 1) * lIndexer * nSplits * dcpHops) : 0;
      const indexerAchievableExtraS = indexerAchievableExtraHops * latColl;
      const unmeasuredCommS = Math.max(0.0, achColl.commS - measuredInCommS);
      const tCommAch = measuredInCommS + (unmeasuredCommS + indexerAchievableExtraS) / effIci;
      const unmeasuredSyncS = Math.max(0.0, achColl.syncStageS - measuredInSyncS);
      const syncExtraS = isFusedMulti
        ? unmeasuredSyncS * (1.0 / effIci - 1.0) : 0.0;
      const routedHbmAchS = (routedHbmS * routedAchRatio) / effHbm;
      const dsaStageAchS = dsaStageSeconds(
        latColl, false, false, effDense, effHbm, effIci, achColl.xlaArFactor
      );
      const chainAchS = routedChain(
        tDenseAch, routedHbmAchS, tMoeAch, tVecMoeAch, dsaStageAchS
      );
      const achCoreS = Math.max(tHbmAch, chainAchS) + tCommAch + syncExtraS;
      const stepAchUs = (achCoreS + achFillDrainS) * 1e6 + launchUs + scopeHostHeadUs;
      const extraLatS = Math.max(0.0, latCollHi - latColl);
      const arHopsUpper = effNumAr * nSplits * upperColl.xlaArFactor * arInfo.intraHops;
      const arExtraUpperS = (arHopsUpper + upperColl.stepHops) * extraLatS;
      const a2aCntUpper = hasDeclaredRouted
        ? 0 : ((lMoe > 0 && scopeEpDp > 1) ? (2 * lMoe * nSplits * upperColl.xlaArFactor) : 0);
      const a2aExtraUpperS = a2aCntUpper * extraLatS;
      const dsaStageHiS = dsaStageSeconds(
        latCollHi, true, false, 1.0, 1.0, 1.0, upperColl.xlaArFactor
      );
      const chainHiS = routedChain(
        tDenseUpper, routedHbmS, tMoe, tVecMoe, dsaStageHiS
      );
      const roofHiCoreS = coreTime(
        tHbm, chainHiS, upperColl.commS + arExtraUpperS, overlappedComm
      ) + a2aExtraUpperS;
      const roofHiUs = (roofHiCoreS + upperFillDrainS) * 1e6;
      const stepHiUs = Math.max(
        stepUs, roofHiUs + nOps * lgBox.hi + scopeHostHeadUs
      );
      const arHopsAch = achColl.hasMeasuredAr
        ? 0 : (effNumAr * nSplits * achColl.xlaArFactor * arInfo.intraHops);
      const arExtraAchS = (arHopsAch + achColl.stepHops) * extraLatS;
      const a2aCntAch = hasDeclaredRouted
        ? 0 : ((lMoe > 0 && scopeEpDp > 1) ? (2 * lMoe * nSplits * achColl.xlaArFactor) : 0);
      const a2aExtraAchS = a2aCntAch * extraLatS;
      const tMatmulDenseUpperAch = ((attnIsBf16 || isPrefOrMix) && flopsKv > 0)
        ? (((flopsDense - flopsKv) / peakW) / effDense +
           (flopsKv / bf16Peak) / effDense)
        : (flopsDense / peakW) / effDense;
      const tDenseHiAch = Math.max(
        tMatmulDenseUpperAch,
        (vecOpsDense / vecRate) / effDense
      );
      const indexerAchievableExtraHiS = indexerAchievableExtraHops * latCollHi;
      const tCommHiAch = measuredInCommS +
        (unmeasuredCommS + arExtraAchS + indexerAchievableExtraHiS) / effIci;
      const dsaStageHiAchS = dsaStageSeconds(
        latCollHi, true, false, effDense, effHbm, effIci, achColl.xlaArFactor
      );
      const chainHiAchS = routedChain(
        tDenseHiAch, routedHbmAchS, tMoeAch, tVecMoeAch, dsaStageHiAchS
      );
      const achHiCoreS = Math.max(tHbmAch, chainHiAchS) + tCommHiAch +
        syncExtraS + (a2aExtraAchS / effIci);
      const stepAchHiUs = Math.max(
        stepAchUs, stepHiUs,
        (achHiCoreS + achFillDrainS) * 1e6 + nOps * lgBox.hi + scopeHostHeadUs
      );
      const eStrict = scopeExpertsLoc > 0
        ? Math.min(scopeExpertsLoc, Math.max(1.0, topK * scopeExpertsLoc / Math.max(1, nExp)))
        : 0.0;
      const wStepStrict = scopeExpertsLoc > 0
        ? (wStatic + lMoe * eStrict * scopeExpBytes) : wStatic;
      const tMatmulDenseStrict = (attnIsBf16 && flopsKv > 0)
        ? ((flopsDense - flopsKv) / peakW + flopsKv / bf16Peak)
        : (flopsDense / peakW);
      const tMatmulStrict = tMatmulDenseStrict + scopeFlopsMoe / peakE;
      const effStateStrictBytes = hasDsa
        ? (effStateBytes - dsaKvGatherBytes + dsaKvStrictBytes) : effStateBytes;
      const tHbmStrict = Math.max(
        (wStepStrict + effStateStrictBytes + actBytes) / bwHbm, tblCtx.hostH2dS
      );
      const tMoeStrict = scopeFlopsMoe / peakE;
      const routedBytesStrict = wStepStrict - wStatic;
      const routedStrictS = routedBytesStrict / bwHbm;
      const streamedRoutedParamsStrict = Math.max(0.0, routedBytesStrict / Math.max(1e-6, bwE));
      const subchannelVecMoeStrict = (weightBlockK > 0 && weightBlockK < subchannelKThresh)
        ? (scopeFlopsMoe / weightBlockK)
        : 0.0;
      const vecOpsMoeStrict = 4.0 * moeTokensLocal * lMoe * topK * iExpLocal +
        (isFp4Expert ? 2.0 * streamedRoutedParamsStrict : 0.0) + subchannelVecMoeStrict;
      // Vector counts are per element. A 32-bit word holds up to 4 packed elements, and 8
      // when it unpacks weights below one byte, so these rates bound every dtype.
      const strictVecRate = (DATA.vector_max_elements_per_word || 4) * vecRate;
      const unpackOpsStrict = isFp4Expert ? 2.0 * streamedRoutedParamsStrict : 0.0;
      const tVecMoeStrict = (vecOpsMoeStrict - unpackOpsStrict) / strictVecRate +
        unpackOpsStrict / ((DATA.vector_sub_byte_elements_per_word || 8) * vecRate);
      const vecOpsDenseStrict = Math.max(0.0, vecOpsDense - vecOpsIndexerRadix);
      const tDenseStrict = Math.max(
        tMatmulStrict - tMoeStrict, vecOpsDenseStrict / strictVecRate
      );
      const dsaStageStrictS = dsaStageSeconds(
        latColl, false, true, 1.0, 1.0, 1.0, floorColl.xlaArFactor
      );
      const chainStrictS = routedChain(
        tDenseStrict, routedStrictS, tMoeStrict, tVecMoeStrict, dsaStageStrictS
      );
      // The strict floor prices all-reduce transfer at the egress sum. That transfer sits
      // in commS, or in the sync stage of fillDrainS when a fused scope runs several passes.
      const arStrictSavingS = effNumAr * (arInfo.xferS - arInfo.xferStrictS) +
        (floorColl.stepTransferS - floorColl.stepStrictS);
      const commStrictS = commS - (commInCore ? arStrictSavingS : 0.0);
      const fillDrainStrictS = fillDrainS - (commInCore ? 0.0 : arStrictSavingS);
      const roofCoreStrictS = coreTime(tHbmStrict, chainStrictS, commStrictS, isFusedMulti);
      const stepFloorStrictUs = (roofCoreStrictS + fillDrainStrictS) * 1e6 +
        launchUs;
      const supInfo = evaluateSupport(sc, p, wl, ch, {
        activeTokens, effTp, epDp: scopeEpDp, planError: wl.planError || ''
      });
      const resTok = Math.min(activeTokens, Math.max(1, tMax));
      const aliasDiff = scopeVmemParts.unaliasedOpBytes - scopeVmemParts.opBytes;
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
        spine_dyn_mib: +((resTok * scopeVmemParts.spineBytes) / 1048576).toFixed(2),
        op_dyn_mib: +((resTok * scopeVmemParts.opBytes) / 1048576).toFixed(2),
        alias_saved_kib: +(aliasDiff / 1024).toFixed(1),
        moe_flop_factor: +moePadFactor.toFixed(3), w_step_gb: +(wStep / 1e9).toFixed(4),
        w_step_strict_gb: +(wStepStrict / 1e9).toFixed(5),
        act_hbm_mb: +(actBytes / 1e6).toFixed(3),
        rowsPerCall: sc === 'whole_step' ? singleCallRows : 0,
        compute_us: +((tDense + Math.max(tMoe, tVecMoe)) * 1e6).toFixed(2),
        chain_us: +(chainS * 1e6).toFixed(2),
        hbm_us: +(tHbm * 1e6).toFixed(2), comm_us: +(commS * 1e6).toFixed(2),
        fill_drain_us: +(fillDrainS * 1e6).toFixed(2),
        launch_us: +launchUs.toFixed(2), roof_floor_us: +roofUs.toFixed(2),
        step_bound_us: +stepUs.toFixed(2),
        step_floor_strict_us: +stepFloorStrictUs.toFixed(2),
        step_achievable_us: +stepAchUs.toFixed(2),
        step_achievable_upper_us: +stepAchHiUs.toFixed(2),
        step_upper_us: +stepHiUs.toFixed(2),
        expert_mode: expertMode,
        host_head_us: +scopeHostHeadUs.toFixed(2),
        break_even_host_head_delta_us: 0.0,
        support: supInfo.support,
        blockers: supInfo.blockers
      });
    }
    const perOpDevUs = scopes[0].rawStepUs - hostHeadPerOpUs;
    for (const s of scopes) {
      if (s.scope !== 'per_op' && s.feasible) {
        const scDevUs = s.rawStepUs - hostHeadFusedUs;
        s.break_even_host_head_delta_us = +Math.max(0.0, scDevUs - perOpDevUs).toFixed(2);
      } else {
        s.break_even_host_head_delta_us = 0.0;
      }
    }
    const feas = scopes.filter(s => s.feasible);
    const candidates = hbmFits
      ? (feas.length ? feas : [scopes[0]])
      : [scopes[0]];
    const floorTieEpsUs = Number(DATA.floor_tie_epsilon_us || 1e-6);
    candidates.sort((a, b) => (
      Math.round(a.rawStepUs / floorTieEpsUs) - Math.round(b.rawStepUs / floorTieEpsUs)
    ) || (a.step_upper_us - b.step_upper_us) ||
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
      ? +(tokensPerChip / Math.max(1e-9, recWinner.step_bound_us * 1e-6)).toFixed(2)
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
      evaluateScenario, sublayerLabel, declaredExpertOwnerGroups
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
