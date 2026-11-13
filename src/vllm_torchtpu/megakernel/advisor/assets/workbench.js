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
(function() { const ENG = window.__MK_ADVISOR_ENGINE; if (!ENG) return;
  const { DATA, PROF, CHIP, REGIMES, evaluateWorkload } = ENG;
  const el = id => document.getElementById(id);
  const escH = s => String(s == null ? '' : s).replace(/&/g, '&amp;')
  .replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  const fmtCtx = c => c >= 1048576 ? (c / 1048576).toFixed(1) + 'M'
  : (c / 1024).toFixed(1) + 'K';
  const rSvg = (x, y, w, h, c) => `<rect x="${x.toFixed(1)}" y="${y}" ` +
  `width="${w.toFixed(1)}" height="${h}" fill="${c}"/>`;
  const dAcc = (t, tag, body, open) => `<details class="wb-acc"${open ? ' open' : ''}>` +
  `<summary><span>${t}</span><span class="prov-tag prov-tag-${tag}">${tag}</span>` +
  `</summary><div class="wb-acc-body">${body}</div></details>`;
  const SCOPE_COLORS = { whole_step: '#009E73', layer_group: '#56B4E9',
  single_layer: '#E69F00', per_op: '#CC79A7' };
  const REASON_LABELS = {
    tp_not_power_of_two: 'TP must be a power of 2',
    attn_heads_not_divisible_by_tp: 'Attention heads % TP != 0',
    kv_heads_not_divisible_by_tp: 'KV heads % TP != 0',
    dcp_not_dividing_tp: 'DCP must divide effective TP',
    experts_not_divisible_by_ep: 'E % EP != 0',
    ep_exceeds_slice_ranks: 'EP exceeds TP*DP*PCP slice ranks',
    pp_exceeds_layers: 'PP exceeds model num_layers',
    invalid_dimensions: 'Dimensions must be positive'
  };
  const STATE = { pinA: 0, pinB: -1, customDirty: false, hwDirty: false,
  meshDirty: false, calibrated: false, dispatchRegime: 'compiled_xla',
  selectedTopic: { kind: 'scope', scope: 'whole_step', sub: 'total' },
  calSource: '', calStepTimeUs: null, launchGapHiUs: 1.5,
  denseSlotBytes: 2097152, collHopHiUs: (CHIP.collective_latency_us_bracket ||
  [1.0, 5.0])[1], expertSlotBytes: 1048576, presets: [], resetSeq: 0,
  maxVmemMib: Math.round(((CHIP.vmem_usable_bytes || 65011712) / 1048576) * 10) / 10 };
  function ensureSelectOption(selEl, val) {
  if (!selEl) return; const contextTokens = String(val);
  if (!Array.from(selEl.options).some(o => o.value === contextTokens)) {
  const opt = document.createElement('option');
  opt.value = contextTokens; opt.textContent = contextTokens;
  selEl.appendChild(opt); } selEl.value = contextTokens; }
  function syncMeshSelects(m) {
  for (const k of ['tp', 'ep', 'dp', 'pp', 'dcp', 'pcp']) {
  ensureSelectOption(el('ctl-' + k), m[k] || 1); } }
  function initPresets() {
  if (el('ctl-vmem')) {
    el('ctl-vmem').min = '2'; el('ctl-vmem').max = String(STATE.maxVmemMib);
  }
  const r0 = REGIMES[0] || {}, rw0 = r0.workload || {};
  if (rw0.dispatch_regime !== undefined && rw0.dispatch_regime !== null) {
    STATE.dispatchRegime = rw0.dispatch_regime;
  }
  const defP = DATA.parallelism || { tp: 8, ep: 1, dp: 1, pp: 1, dcp: 1, pcp: 1 };
  const seen = new Set(), addP = (name, p) => { if (!p || !p.tp) return;
  const key = [p.tp, p.ep || 1, p.dp || 1, p.pp || 1, p.dcp || 1, p.pcp || 1].join(':');
  if (seen.has(key)) return; seen.add(key); STATE.presets.push({ name, tp: p.tp,
  ep: p.ep || 1, dp: p.dp || 1, pp: p.pp || 1, dcp: p.dcp || 1, pcp: p.pcp || 1,
  layerGroups: p.layer_groups || p.layerGroups || [] }); };
  addP('Decode Mesh', defP); addP('Prefill Mesh', DATA.prefill_parallelism);
  addP('Saturated Mesh', DATA.saturation_parallelism);
  for (const r of REGIMES) if (r && r.workload) addP(r.layout || 'Mesh', r.workload);
  if (el('ctl-layout')) el('ctl-layout').innerHTML = STATE.presets.map((p, i) =>
  {
    const ax = [p.tp > 1 ? 'TP' + p.tp : '', p.ep > 1 ? 'EP' + p.ep : ''].filter(Boolean);
    return `<option value="${i}">${escH(p.name)} (${ax.join(' x ') || 'TP1'})</option>`;
  }).join(''); }
  function runtimeKnobs(gapFallback) {
  const r0 = REGIMES[0] || {}, rw0 = r0.workload || {};
  const fallbackOp = gapFallback !== undefined ? gapFallback
    : (rw0.op_overhead_us !== undefined ? rw0.op_overhead_us : 1.0);
  const rawV = parseFloat(el('ctl-vmem').value) || DATA.vmem_budget_mib
  || STATE.maxVmemMib;
  const clampedV = Math.min(STATE.maxVmemMib, Math.max(2.0, rawV));
  const useMan = STATE.hwDirty || STATE.calibrated;
  const dr = useMan && !STATE.calibrated ? 'auto' : STATE.dispatchRegime;
  const gapVal = useMan ? (parseFloat(el('ctl-launch-gap').value) || 0.0)
  : (dr === 'zero_floor' ? 0.0 : (dr === 'compiled_xla' ? 1.0
  : (dr === 'unfused_eager' ? 6.0 : fallbackOp)));
  return { vmemMib: clampedV, calibrated: STATE.calibrated,
  dispatchRegime: dr, opOverheadUs: gapVal,
  dmaBuffers: parseInt(el('ctl-dma').value, 10) || DATA.dma_buffers || 2,
  vmemAlias: el('ctl-alias').value === '1', launchGapUs: gapVal,
  collHopUs: parseFloat(el('ctl-coll-hop').value) || 1.0,
  etaDense: parseFloat(el('ctl-eta-dense').value) || 0.72,
  etaMoeGrouped: parseFloat(el('ctl-eta-moe').value) || 0.50,
  etaHbm: parseFloat(el('ctl-eta-hbm').value) || 0.82,
  etaIci: parseFloat(el('ctl-eta-ici').value) || 0.75,
  launchGapHiUs: STATE.launchGapHiUs, collHopHiUs: STATE.collHopHiUs,
  denseSlotBytes: STATE.denseSlotBytes, expertSlotBytes: STATE.expertSlotBytes }; }
  function regimeToWorkload(idx) {
  const reg = REGIMES[idx] || REGIMES[0] || {}, rw = reg.workload || {};
  const ph = rw.phase || reg.phase || 'decode', seqTok = ph === 'decode' ? 1
  : (ph === 'verify' ? ((rw.spec_tokens || 1) + 1) : (rw.chunk_size || rw.isl || 1));
  const gapDef = rw.op_overhead_us !== undefined ? rw.op_overhead_us : 1.0;
  return Object.assign({ label: reg.label || 'Preset', phase: ph,
  batch: rw.batch || reg.active_tokens || 16, seqTokens: seqTok, isl: rw.isl,
  osl: rw.osl, specTokens: rw.spec_tokens,
  context: rw.context || reg.effective_context || 4096, chunkSize: rw.chunk_size,
  prefixHitRate: rw.prefix_hit_rate || 0.0, microbatches: rw.microbatches || 0,
  maxRowsPerCall: rw.max_rows_per_call || 0,
  segmentLayers: rw.segment_layers || 0, tp: rw.tp || 8, ep: rw.ep || 1,
  dp: rw.dp || 1, pp: rw.pp || 1, dcp: rw.dcp || 1, pcp: rw.pcp || 1,
  kvSplit: rw.kv_split || 1, layerGroups: rw.layer_groups || [],
  moeMode: rw.moe_mode || reg.recommended_moe_mode || 'masked_on_chip' },
  runtimeKnobs(gapDef)); }
  function currentWorkloadA() {
  const base = regimeToWorkload(STATE.pinA);
  if (!STATE.customDirty && !STATE.meshDirty && REGIMES[STATE.pinA]) return base;
  const p = STATE.presets[parseInt(el('ctl-layout').value, 10) || 0] || base;
  const tp = STATE.meshDirty ? (parseInt(el('ctl-tp').value, 10) || 1) : p.tp;
  const ep = STATE.meshDirty ? (parseInt(el('ctl-ep').value, 10) || 1) : p.ep;
  const dp = STATE.meshDirty ? (parseInt(el('ctl-dp').value, 10) || 1) : p.dp;
  const pp = STATE.meshDirty ? (parseInt(el('ctl-pp').value, 10) || 1) : p.pp;
  const dcp = STATE.meshDirty ? (parseInt(el('ctl-dcp').value, 10) || 1) : p.dcp;
  const pcp = STATE.meshDirty ? (parseInt(el('ctl-pcp').value, 10) || 1) : p.pcp;
  const phase = el('ctl-phase').value, batch = parseInt(el('ctl-batch').value, 10) || 1;
  const isCh = phase === 'prefill' || phase === 'mixed';
  const rSeq = parseInt(el('ctl-seq-tokens').value, 10) || 1;
  const seqTokens = phase === 'decode' ? 1 : rSeq;
  const ctxVal = parseInt(el('ctl-context').value, 10) || 4096;
  const same = phase === base.phase && seqTokens === base.seqTokens &&
  ctxVal === base.context, actT = Math.ceil(batch / Math.max(1, dp)) *
  (phase === 'decode' ? 1 : seqTokens);
  const effTp = tp === 1 ? Math.max(1, dcp) : tp;
  const epDp = Math.max(1, Math.floor(ep / ENG.gcd(ep, effTp)));
  const rFloor = ENG.rowFloor(PROF.expert_dtype);
  const lbl = `Live (${phase} B=${batch}, seq=${seqTokens})`;
  return Object.assign({}, base, { label: lbl, phase, batch, seqTokens, context: ctxVal,
  chunkSize: same ? base.chunkSize : (isCh ? seqTokens : 0),
  isl: same ? base.isl : (isCh ? Math.max(seqTokens, ctxVal) : ctxVal),
  specTokens: phase === 'verify' ? Math.max(1, seqTokens - 1) : 0,
  prefixHitRate: phase === base.phase ? base.prefixHitRate : 0.0,
  tp, ep, dp, pp, dcp, pcp,
  layerGroups: (p.layerGroups && p.layerGroups.length) ? p.layerGroups
  : base.layerGroups, moeMode: (same && batch === base.batch && !STATE.meshDirty)
  ? base.moeMode : (actT * epDp >= rFloor ? 'grouped_on_chip' : 'masked_on_chip') },
  runtimeKnobs(base.launchGapUs)); }
  function syncControlsFromRegime(idx) { const wl = regimeToWorkload(idx);
  el('ctl-phase').value = wl.phase;
  if (el('ctl-batch')) { el('ctl-batch').max = String(Math.max(1024, wl.batch));
  el('ctl-batch').value = String(wl.batch); }
  el('ctl-seq-tokens').value = Math.min(8192, wl.seqTokens);
  el('ctl-seq-tokens').disabled = wl.phase === 'decode';
  el('ctl-context').value = Math.min(1048576, wl.context);
  syncMeshSelects(wl);
  const pIdx = STATE.presets.findIndex(p => p.tp === wl.tp && p.ep === wl.ep &&
  p.dp === wl.dp && p.pp === wl.pp && p.dcp === wl.dcp && p.pcp === wl.pcp);
  if (pIdx >= 0 && el('ctl-layout')) el('ctl-layout').value = String(pIdx);
  if (!STATE.calibrated && !STATE.hwDirty)
  el('ctl-launch-gap').value = String(wl.launchGapUs); updateControlLabels(wl); }
  function updateControlLabels(wl) {
  el('lbl-phase').textContent = wl.phase; el('lbl-batch').textContent = wl.batch;
  if (el('ctl-seq-tokens')) el('ctl-seq-tokens').disabled = wl.phase === 'decode';
  el('lbl-seq-tokens').textContent = wl.phase === 'decode'
    ? '1 (decode)' : String(wl.seqTokens);
  el('lbl-context').textContent = fmtCtx(wl.context);
  el('lbl-vmem').textContent = wl.vmemMib.toFixed(1) + ' MiB';
  el('lbl-dma').textContent = wl.dmaBuffers + ' slots';
  el('lbl-alias').textContent = wl.vmemAlias ? 'enabled' : 'disabled';
  el('lbl-launch-gap').textContent = wl.launchGapUs.toFixed(2) + ' us';
  el('lbl-coll-hop').textContent = wl.collHopUs.toFixed(2) + ' us';
  if (el('lbl-dispatch-regime')) el('lbl-dispatch-regime').textContent = wl.dispatchRegime;
  for (const [id, mode] of [['btn-dr-zero','zero_floor'],
  ['btn-dr-xla','compiled_xla'],['btn-dr-eager','unfused_eager']]) {
  if (el(id)) el(id).className = 'wb-mini-btn' +
  (STATE.dispatchRegime === mode && !STATE.hwDirty ? ' active-a' : ''); }
  for (const k of ['tp', 'ep', 'dp', 'pp', 'dcp', 'pcp']) {
  if (el('lbl-' + k)) el('lbl-' + k).textContent = String(wl[k]); }
  el('lbl-eta-dense').textContent = wl.etaDense.toFixed(2);
  el('lbl-eta-moe').textContent = wl.etaMoeGrouped.toFixed(2);
  el('lbl-eta-hbm').textContent = wl.etaHbm.toFixed(2);
  el('lbl-eta-ici').textContent = wl.etaIci.toFixed(2);
  const axes = []; for (const k of ['tp', 'dp', 'ep', 'pp', 'pcp', 'dcp'])
  if (wl[k] > 1) axes.push(`${k.toUpperCase()}${wl[k]}`);
  const layStr = axes.length ? axes.join(' x ') : 'TP1';
  el('lbl-layout').textContent = layStr; el('wb-active-layout').textContent = layStr; }
  function renderTopVerdictAndBanner(wlA, evalA) {
  const pill = el('wb-verdict-pill'), sumEl = el('wb-verdict-summary');
  const ban = el('wb-invalid-banner'); if (!pill || !sumEl || !ban) return;
  const zeroEv = evaluateWorkload(Object.assign({}, wlA, {
    dispatchRegime: 'zero_floor', launchGapUs: 0.0, opOverheadUs: 0.0
  }));
  const perOp = evalA.scopes[0];
  const recScope = evalA.recScope || evalA.winner.scope;
  const win = evalA.scopes.find(s => s.scope === recScope) || evalA.winner;
  const fast = evalA.winner;
  const isHyp = evalA.isPhysValid && fast.scope !== recScope;
  const totGain = perOp.rawStepUs - win.rawStepUs;
  const cpGain = perOp.pureStepUs - win.pureStepUs;
  const physShare = totGain > 1e-6
    ? Math.max(0, Math.min(100, (cpGain / totGain) * 100)) : (cpGain > 0 ? 100 : 0);
  const cpPct = perOp.pureStepUs > 1e-6 ? (cpGain / perOp.pureStepUs) * 100 : 0;
  const isFusedWin = win.scope === 'whole_step' || win.scope === 'layer_group';
  const zeroRecScope = zeroEv.recScope || zeroEv.winner.scope;
  let vTxt = 'KEEP PER_OP', vBg = '#475569';
  if (!evalA.isPhysValid) { vTxt = 'INVALID CONFIG'; vBg = '#DC2626'; }
  else if (isFusedWin && (zeroRecScope === 'per_op' || cpGain <= 0.0)) {
    vTxt = 'TUNE XLA FLAGS FIRST'; vBg = '#D97706';
  } else if (win.scope === 'whole_step') { vTxt = 'BUILD WHOLE_STEP'; vBg = '#059669'; }
  else if (win.scope === 'layer_group') { vTxt = 'BUILD LAYER_GROUP'; vBg = '#0284C7'; }
  else if (win.scope === 'single_layer') {
    vTxt = 'BUILD SINGLE_LAYER'; vBg = '#B45309';
  }
  pill.textContent = vTxt; pill.style.background = vBg;
  const tCross = Math.max(1, Math.ceil(
    ENG.rowFloor(PROF.expert_dtype) / Math.max(1, evalA.epDp)
  ));
  const bankM = Math.round((wlA.denseSlotBytes || 2097152) / 1048576);
  const ebankM = Math.round((wlA.expertSlotBytes || 1048576) / 1048576);
  const achGain = perOp.step_achievable_us - win.step_achievable_us;
  const achPct = perOp.step_achievable_us > 1e-6
    ? (achGain / perOp.step_achievable_us) * 100 : 0;
  const isFusedFast = fast.scope === 'whole_step' || fast.scope === 'layer_group';
  const cpFastGain = perOp.pureStepUs - fast.pureStepUs;
  const cpFastPct = perOp.pureStepUs > 1e-6 ? (cpFastGain / perOp.pureStepUs) * 100 : 0;
  const roiCheck = isHyp
    ? (isFusedFast && cpFastGain > 0.0 && cpFastPct < 1.0)
    : (isFusedWin && cpGain > 0.0 && cpPct < 1.0);
  const roiBadge = roiCheck
    ? '<span class="wb-roi-badge">LOW ROI (&lt;1% PHYSICS)</span>' : '';
  const hypBadge = isHyp
    ? ` <span class="badge badge-hypothetical">hypothetical: ` +
      `${escH(fast.scope)} (${fast.step_bound_us.toFixed(1)} us; ` +
      `${cpFastPct.toFixed(1)}% physics)</span>` : '';
  const moeNote = (PROF.num_moe_layers || 0) > 0
    ? ` &middot; MoE: masked &lt;${tCross}, grouped &ge;${tCross}` : '';
  sumEl.innerHTML = `${roiBadge}Saves <strong>${achGain.toFixed(1)} us ` +
  `(${achPct.toFixed(1)}% achievable)</strong> &middot; Fusion gain (estimate): ` +
  `<strong>${cpGain.toFixed(1)} us (${cpPct.toFixed(1)}%)</strong> &middot; ` +
  `Ideal 1-pass: <strong>T&le;${evalA.tWideMax}</strong>${hypBadge}` +
  `<details class="wb-topbar-details"><summary>Blueprint &amp; Tile Specs ` +
  `(nbd=${win.selected_dense_slots}, nbe=${win.selected_expert_slots})</summary>` +
  `Physics share: <strong>${physShare.toFixed(0)}%</strong> &middot; ` +
  `Ideal 2-slot ceiling: <strong>T&le;${evalA.tShrunkMax}</strong> &middot; ` +
  `<code>Tiles(nbd=${win.selected_dense_slots}, nbe=${win.selected_expert_slots}, ` +
  `bank=${bankM}MiB, ebank=${ebankM}MiB)</code>${moeNote}</details>`;
  const alerts = [];
  for (const r of evalA.invalidReasons) {
    if (r === 'hbm_pool_overflow') {
      const uG = (evalA.hbmUsedBytes / 1e9).toFixed(1);
      const pG = (evalA.hbmPoolCapBytes / 1e9).toFixed(1);
      const cG = ((CHIP.hbm_capacity_bytes || 103e9) / 1e9).toFixed(1);
      alerts.push(`HBM exceeds 80% serving pool (${uG} GB > ${pG} GB / ${cG} GB)`);
    } else alerts.push(REASON_LABELS[r] || r);
  }
  if (wlA.pp > 1) {
    alerts.push(`PP=${wlA.pp} > 1 blocks whole_step fusion across stages`);
  }
  if (alerts.length > 0) {
    ban.textContent = 'FEASIBILITY RED FLAG: ' + alerts.join(' | ');
    ban.classList.remove('view-hidden');
  } else { ban.textContent = ''; ban.classList.add('view-hidden'); } }
  function renderVmemArena(evalA) {
  const scopeSel = el('ctl-arena-scope').value || evalA.winner.scope;
  const sc = evalA.scopes.find(s => s.scope === scopeSel) || evalA.winner;
  const budget = Math.max(1.0, evalA.vmemBudgetMib), W = 760, padL = 16, barW = 728;
  const used = sc.dense_stage_mib + sc.expert_stage_mib + sc.op_scratch_mib +
  sc.pad_mib + sc.spine_dyn_mib + sc.op_dyn_mib, selT = STATE.selectedTopic || {};
  const px = m => (m / Math.max(budget * 1.12, used * 1.05)) * barW;
  const blocks = [['dense',`Dense (${sc.selected_dense_slots})`,sc.dense_stage_mib,'#3B82F6'],
  ['exp',`Expert (${sc.selected_expert_slots})`,sc.expert_stage_mib,'#6366F1'],
  ['scr','Op Scratch',sc.op_scratch_mib,'#0EA5E9'],
  ['pad','Align Reserve',sc.pad_mib,'#64748B'],
  ['spn','[T,H] Spine',sc.spine_dyn_mib,'#10B981'],
  ['tok','Token Scratch',sc.op_dyn_mib,'#F59E0B']]; let curX = padL, rects = '';
  for (const [key, label, mib, col] of blocks) { if (mib <= 0.01) continue;
  const w = Math.max(2, px(mib)), mid = (curX + w / 2).toFixed(1);
  const stk = (selT.kind === 'vmem' && selT.sub === key)
  ? ' stroke="#FFF" stroke-width="2"' : '';
  const lbl = w > 58 ? `<text x="${mid}" y="45" text-anchor="middle" fill="#FFF"` +
  ` font-size="11" font-weight="600">${label}</text>` : '';
  rects += `<g class="wb-clickable" onclick="wbSelectTopic('vmem','${sc.scope}',` +
  `'${key}')">${rSvg(curX, 22, w, 36, col).replace('/>', ` rx="3"${stk}><title>` +
  `${label}: ${mib.toFixed(2)} MiB</title></rect>`)}${lbl}</g>`; curX += w; }
  const ceilX = (padL + px(budget)).toFixed(1), ceilLblX = (+ceilX - 6).toFixed(1);
  const nCalls = sc.nCalls || 1;
  const stCol = (sc.n_splits > 1 || nCalls > 1) ? '#F59E0B' : '#10B981';
  const stTxt = nCalls > 1
  ? `Microbatched: ${nCalls} calls of at most ${sc.rowsPerCall} rows, ` +
  `weights streamed again per call`
  : (sc.n_splits > 1
  ? `Multi-pass split: ${sc.n_splits} passes (T_max=${sc.t_vmem_max})`
  : `Single pass under the ideal-kernel ceiling: ${evalA.activeTokens}/` +
  `${sc.t_vmem_max} tok (${used.toFixed(1)}/${budget.toFixed(1)} MiB)`);
  el('wb-vmem-arena-svg').innerHTML = `<svg viewBox="0 0 ${W} 96" class="chart-svg"` +
  ` role="img" aria-label="VMEM Floorplan"><rect x="${padL}" y="22" width="${barW}"` +
  ` height="36" rx="4" fill="var(--chart-bg)" stroke="var(--border)"/>${rects}` +
  `<line x1="${ceilX}" y1="12" x2="${ceilX}" y2="68" stroke="#EF4444" ` +
  `stroke-width="2" stroke-dasharray="4,3"/><text x="${ceilLblX}" y="14" ` +
  `text-anchor="end" fill="#EF4444" font-size="11" font-weight="700">Ceiling ` +
  `${budget.toFixed(1)} MiB</text><text x="${padL}" y="82" fill="${stCol}" ` +
  `font-size="12" font-weight="600">${stTxt} &middot; Aliasing saves ` +
  `${sc.alias_saved_kib} KiB/tok</text></svg>`; }
  function renderScopeTimeline(evalA) { const W = 760, padL = 112, barArea = 510;
  const feasHi = evalA.scopes.map(s => s.feasible
    ? Math.max(s.step_upper_us, s.step_achievable_us) : 0);
  const maxUs = Math.max(1.0, ...feasHi) * 1.06, px = u => (u / maxUs) * barArea;
  let rows = ''; ['whole_step', 'layer_group', 'single_layer', 'per_op']
  .forEach((nm, idx) => { const s = evalA.scopes.find(x => x.scope === nm);
  if (!s) return; const y = 18 + idx * 48,
  isWin = s.scope === (evalA.recScope || evalA.winner.scope)
  && s.feasible; if (!s.feasible) { rows += `<text x="10" y="${y + 18}" ` +
  `fill="var(--muted)" font-size="12">${s.scope}</text><text x="${padL}" ` +
  `y="${y + 18}" fill="var(--muted)" font-size="11">Infeasible: ` +
  `${escH(s.reason)}</text>`; return; }
  const isFused = s.scope === 'whole_step' || s.scope === 'layer_group';
  // Fused: collectives extend the chain while the HBM stream overlaps both.
  // Separate kernels: collectives follow the streamed compute.
  const envW = px(coreStepNoLaunch(s) - (isFused ? 0 : s.comm_us));
  const chainW = px(s.chain_us), commW = px(s.comm_us);
  const fdW = px(s.fill_drain_us), repW = px(s.repass_spill_us);
  const x1 = padL + envW, x2 = x1 + (isFused ? 0 : commW);
  const x3 = x2 + fdW, x4 = x3 + repW;
  const hiX = padL + px(s.step_upper_us), loX = padL + px(s.step_bound_us);
  const bars = rSvg(padL, y + 2, Math.max(1, chainW), 12, '#3B82F6') +
    rSvg(padL, y + 16, Math.max(1, px(s.hbm_us)), 12, '#009E73') +
    (isFused ? rSvg(padL + chainW, y + 2, commW, 12, '#E69F00')
      : rSvg(x1, y + 2, commW, 26, '#E69F00')) +
    rSvg(x2, y + 2, fdW, 26, '#56B4E9') + rSvg(x3, y + 2, repW, 26, '#EF4444') +
    rSvg(x4, y + 2, px(s.launch_us), 26, '#CC79A7');
  rows += `<g class="wb-clickable" onclick="showProof('${s.scope}')">` +
  `<text x="10" y="${y + 15}" fill="${isWin ? '#10B981' : 'var(--text)'}"` +
  ` font-size="12" font-weight="${isWin ? 700 : 500}">${isWin ? '* ' : ''}` +
  `${s.scope}</text><text x="10" y="${y + 29}" fill="var(--muted)" font-size="10">` +
  `${s.ops_per_step} ops, ${s.n_splits}p</text>${bars}<line x1="${loX.toFixed(1)}"` +
  ` y1="${y + 15}" x2="${hiX.toFixed(1)}" y2="${y + 15}" stroke="var(--text)" ` +
  `stroke-width="1.5" stroke-dasharray="2,2"/><text x="${(hiX + 6).toFixed(1)}" ` +
  `y="${y + 15}" fill="var(--text)" font-size="10" font-weight="600" class="tabular">` +
  `floor ${s.step_bound_us.toFixed(1)}~${s.step_upper_us.toFixed(1)}us</text>` +
  `<text x="${(hiX + 6).toFixed(1)}" y="${y + 27}" fill="var(--muted)" ` +
  `font-size="9" class="tabular">ach ${s.step_achievable_us.toFixed(1)}~${(s.step_achievable_upper_us || s.step_achievable_us).toFixed(1)}us</text></g>`; });
  const leg = [['#3B82F6',0,'Compute Chain'],['#009E73',100,'HBM'],
  ['#E69F00',152,'Collectives'],['#56B4E9',236,'Fill/Drain'],
  ['#EF4444',314,'Repass'],['#CC79A7',378,'Launch Gap']].map(([c, x, t]) =>
  `<rect x="${x}" y="-8" width="8" height="8" fill="${c}"/>` +
  `<text x="${x + 11}" y="0">${t}</text>`).join('');
  el('wb-scope-timeline-svg').innerHTML = `<svg viewBox="0 0 ${W} 224" ` +
  `class="chart-svg" role="img" aria-label="4-Scope Step Timeline">${rows}` +
  `<g transform="translate(${padL}, 214)" font-size="10" ` +
  `fill="var(--muted)">${leg}</g></svg>`; }
  // Core step time as the engine priced it: exact for any composition rule and for a
  // scope summed over microbatch calls, with no formula repeated here.
  function coreStepNoLaunch(s) { return s.roof_floor_us - s.fill_drain_us; }
  function exposedCommUs(s) {
    if (!(s.comm_us > 0)) return 0;
    const raw = coreStepNoLaunch(s) - Math.max(s.chain_us, s.hbm_us);
    // Each call rounds its fields to 0.01 us; snap <= 0.01 * nCalls around exact 0.
    const snap = 0.01 * (s.nCalls || 1) + 0.005;
    return raw >= snap ? raw : 0;
  }
  function renderSignedWaterfall(evalA, evalB, labelB) {
  const scA = evalA.winner, perOp = evalA.scopes.find(s => s.scope === 'per_op');
  const scB = evalB ? evalB.winner : (perOp || evalA.runnerUp);
  const hypA = evalA.isPhysValid &&
    evalA.recScope && scA.scope !== evalA.recScope ? ' [hypothetical]' : '';
  const hypB = evalB && evalB.isPhysValid &&
    evalB.recScope && scB.scope !== evalB.recScope ? ' [hypothetical]' : '';
  const cmpB = evalB
    ? `${labelB} (${scB.scope}${hypB})` : `Baseline (${scB.scope})`;
  el('wb-waterfall-title').textContent =
  `Signed Attribution: [A] ${scA.scope}${hypA} vs [B] ${cmpB}`;
  const envA = Math.max(scA.chain_us, scA.hbm_us);
  const envB = Math.max(scB.chain_us, scB.hbm_us);
  const commAddA = exposedCommUs(scA);
  const commAddB = exposedCommUs(scB);
  const selT = STATE.selectedTopic || {};
  const terms = [['Compute/HBM Roofline Core', envA - envB],
  ['Exposed Collective Sync & Bounce', commAddA - commAddB],
  ['Cold Pipeline Fill/Drain', scA.fill_drain_us - scB.fill_drain_us],
  ['Multi-Pass HBM Weight Re-stream', scA.repass_spill_us - scB.repass_spill_us],
  ['Dispatch / Boundary Bubble',
    (scA.ops_per_step - scB.ops_per_step) * (evalA.launchGapPriorUs ?? evalA.launchGapLoUs)]];
  const netD = scA.step_bound_us - scB.step_bound_us;
  const pct = scB.step_bound_us > 0 ? (netD / scB.step_bound_us) * 100.0 : 0.0;
  const sgn = v => (v >= 0 ? '+' : '') + v.toFixed(1);
  el('wb-waterfall-summary').textContent = `Net: ${sgn(netD)} us (${sgn(pct)}%)`;
  const maxAbs = Math.max(1.0, ...terms.map(t => Math.abs(t[1])), Math.abs(netD));
  const W = 760, midX = 430, halfW = 220; const rows = terms.map(([lbl, d], i) => {
  const d0 = Math.abs(d) < 5e-3 ? 0 : d;
  const y = 16 + i * 28, w = Math.max(1.5, (Math.abs(d0) / maxAbs) * halfW);
  const x = d0 < 0 ? midX - w : midX, col = d0 <= 0 ? '#10B981' : '#EF4444';
  const stk = (selT.kind === 'waterfall' && String(selT.sub) === String(i))
  ? ' stroke="#FFF" stroke-width="2"' : '', bar = rSvg(x, y + 2, w, 16, col)
  .replace('/>', stk + '/>');
  return `<g class="wb-clickable" onclick="wbSelectTopic('waterfall',` +
  `'${scA.scope}','${i}')"><text x="12" y="${y + 14}" fill="var(--text)" ` +
  `font-size="11">${lbl}</text>${bar}<text x="${W - 14}" y="${y + 14}" ` +
  `text-anchor="end" fill="${col}" font-size="11" font-weight="600" ` +
  `class="tabular">${d0 > 0 ? '+' : ''}${d0.toFixed(2)} us</text></g>`; }).join('');
  el('wb-waterfall-svg').innerHTML = `<svg viewBox="0 0 ${W} 162" ` +
  `class="chart-svg" role="img" aria-label="Signed Mechanism Waterfall">` +
  `<line x1="${midX}" y1="8" x2="${midX}" y2="154" stroke="var(--border)" ` +
  `stroke-width="1.5"/>${rows}</svg>`; }
  function renderDerivDrawer(wlA, evalA, evalB, lblB) {
  const selSc = (STATE.selectedTopic || {}).scope;
  const sc = evalA.scopes.find(s => s.scope === selSc) || evalA.winner;
  const provTag = STATE.calibrated ? 'MEASURED' : (wlA.dispatchRegime === 'zero_floor' ? 'DERIVED' : 'ASSUMED');
  const isFused = sc.scope === 'whole_step' || sc.scope === 'layer_group';
  const nCalls = sc.nCalls || 1, rowLim = sc.rowsPerCall || 0, f2 = v => v.toFixed(2);
  const serialUs = sc.chain_us + (isFused ? sc.comm_us : 0);
  const bind = sc.hbm_us >= serialUs ? 'HBM Weight/KV Stream'
    : ((sc.chain_us > sc.compute_us ? 'Routed Expert Stream after Router'
    : 'Matrix Compute') + (isFused && sc.comm_us > 0 ? ' + ICI Collectives' : ''));
  const cal = window.wbCalNotes ? window.wbCalNotes(sc) : { line:
  `<span class="prov-tag ` +
  `prov-tag-${provTag}">${provTag}</span> Settle: --trace &lt;p&gt;`,
  anc: '' }, spK = (evalA.vmemParts.spineBytes / 1024).toFixed(1);
  const opK = (evalA.vmemParts.opBytes / 1024).toFixed(1);
  const totK = (evalA.vmemParts.totalPerTokenBytes / 1024).toFixed(1);
  const focus = window.wbFocusedTopicCard ? window.wbFocusedTopicCard(wlA, evalA,
  evalB, lblB, STATE.selectedTopic, sc) : '';
  const r1 = `<div style="font-size:11px;line-height:1.4;">${isFused
    ? `Keeps the [T, H] residual spine in VMEM across layers and overlaps HBM ` +
      `weight streaming (${sc.selected_dense_slots}/${sc.selected_expert_slots} DMA slots) with the ` +
      `compute chain. Routed expert weights stream only after each router, and ICI ` +
      `collectives extend the chain; bound by <strong>${bind}</strong>.`
    : `Materializes activations to HBM at ${sc.ops_per_step} kernel boundaries ` +
      `with serial collective barriers; bound by <strong>${bind}</strong>.`}${nCalls > 1
    ? ` Served as ${nCalls} calls of at most ${sc.rowsPerCall} rows; every call ` +
      `streams its weights again.` : ''}</div>`;
  const r2 = `<div class="wb-formula">T = ${evalA.activeTokens} tok/core | ` +
  `B_vmem = ${evalA.vmemBudgetMib.toFixed(1)} MiB | b_tok = ${totK} KiB/tok\n` +
  `nbd=${sc.selected_dense_slots} (2MiB), nbe=${sc.selected_expert_slots} (1MiB) | ` +
  `phi_moe = ${sc.moe_flop_factor.toFixed(3)}x\n` +
  `eta = (dense ${wlA.etaDense.toFixed(2)}, moe ${wlA.etaMoeGrouped.toFixed(2)}, ` +
  `hbm ${wlA.etaHbm.toFixed(2)}, ici ${wlA.etaIci.toFixed(2)})</div>`;
  const eqHead = (isFused
    ? 't_step = max(t_hbm, t_chain + t_comm) + t_fd + t_rep + N*t_gap'
    : 't_step = max(t_chain, t_hbm) + t_comm + t_fd + t_rep + N*t_gap') +
    '\nt_chain = t_dense + max(t_routed, t_moe)';
  // A microbatched scope is a sum of calls, so its summed terms do not recombine
  // through one max(); list them instead of printing an equation that would not hold.
  const eqNum = nCalls > 1
    ? `per call, summed over ${nCalls} calls: hbm ${f2(sc.hbm_us)}, chain ` +
      `${f2(sc.chain_us)}, comm ${f2(sc.comm_us)}, fd ${f2(sc.fill_drain_us)}, ` +
      `rep ${f2(sc.repass_spill_us)}, gaps ${f2(sc.launch_us)}`
    : (isFused
      ? `= max(${f2(sc.hbm_us)}, ${f2(sc.chain_us)} + ${f2(sc.comm_us)})`
      : `= max(${f2(sc.chain_us)}, ${f2(sc.hbm_us)}) + ${f2(sc.comm_us)}`) +
      ` + ${f2(sc.fill_drain_us)} + ${f2(sc.repass_spill_us)} + ` +
      `${sc.ops_per_step}*${f2(evalA.launchGapLoUs)}`;
  const eqStr = `${eqHead}\n${eqNum}`;
  const stFl = (sc.step_floor_strict_us ?? sc.step_bound_us).toFixed(2);
  const r3 = `<div class="wb-formula">${eqStr}\n= ${sc.step_bound_us.toFixed(2)} us ` +
  `floor (strict ${stFl} us, up ${sc.step_upper_us.toFixed(2)} us) | ` +
  `${sc.step_achievable_us.toFixed(2)}~` +
  `${(sc.step_achievable_upper_us || sc.step_achievable_us).toFixed(2)} us achievable\n` +
  `T_max = ${rowLim ? 'min(' : ''}floor((${evalA.vmemBudgetMib.toFixed(1)} - ` +
  `${sc.vmem_static_mib.toFixed(2)}) MiB / (${spK}+${opK}) KiB)${rowLim
    ? `, ${rowLim} rows per call)` : ''} = ${sc.t_vmem_max} tok (${nCalls > 1
    ? `${nCalls} calls` : `${sc.n_splits}p`})</div>`;
  const perOp = evalA.scopes[0], gapPr = evalA.launchGapPriorUs ?? evalA.launchGapLoUs;
  const cpUs = perOp.pureStepUs - sc.pureStepUs;
  const dbUs = (perOp.ops_per_step - sc.ops_per_step) * gapPr;
  const totUs = cpUs + dbUs;
  const shPct = Math.abs(totUs) > 1e-6 ? ((cpUs / totUs) * 100).toFixed(0) : '100';
  const r4 = `<div class="wb-formula">Fusion gain (estimate): ` +
  `${cpUs >= 0 ? '+' : ''}${cpUs.toFixed(2)} us (HBM + sync + MoE pad)\n` +
  `Dispatch Bubble Gain: ${dbUs >= 0 ? '+' : ''}${dbUs.toFixed(2)} us ` +
  `((${perOp.ops_per_step}-${sc.ops_per_step}) * ` +
  `${gapPr.toFixed(2)} us)\n` +
  `Total vs per_op: ${totUs >= 0 ? '+' : ''}${totUs.toFixed(2)} us ` +
  `(${shPct}% fusion-estimate) | w_step=${sc.w_step_gb} GB</div>` +
  `<div style="font-size:11px;margin-top:4px;">${cal.line}</div>`;
  const achSave = perOp.step_achievable_us - sc.step_achievable_us;
  const achSavePct = perOp.step_achievable_us > 1e-6
    ? (achSave / perOp.step_achievable_us) * 100 : 0;
  const supTxt = escH(sc.support || 'baseline') +
    ((sc.blockers && sc.blockers.length) ? ` (${escH(sc.blockers.join(', '))})` : '');
  const execCard = `<div class="wb-exec-card">` +
  `<div class="wb-exec-row"><span>Inspecting Scope</span>` +
  `<strong>${escH(sc.scope)} (${sc.step_achievable_us.toFixed(1)} us)</strong></div>` +
  `<div class="wb-exec-row"><span>Support</span><strong>${supTxt}</strong></div>` +
  `<div class="wb-exec-row"><span>Primary Bottleneck</span>` +
  `<strong>${bind}</strong></div>` +
  `<div class="wb-exec-row"><span>Net vs per_op</span><strong>` +
  `${achSave >= 0 ? '+' : ''}${achSave.toFixed(1)} us (` +
  `${achSavePct >= 0 ? '+' : ''}${achSavePct.toFixed(1)}%)</strong></div></div>`;
  el('wb-deriv-drawer').innerHTML = focus + execCard +
  dAcc(`1. Plain-English Mechanism (${sc.scope})`, 'DERIVED', r1, true) +
  dAcc('2. Step-Time & VMEM Whiteboard Equations', 'DERIVED', r3, false) +
  dAcc('3. Fusion Gain vs Dispatch Split', provTag, r4, false) +
  dAcc('4. Symbols, Ring Slots & Efficiency Priors', 'DERIVED', r2, false) +
  cal.anc; }
  window.showProof = function(key) {
  const scNames = ['whole_step', 'layer_group', 'single_layer', 'per_op'];
  const sc = scNames.includes(key)
    ? key : evaluateWorkload(currentWorkloadA()).winner.scope;
  window.wbSelectTopic('scope', sc, key); };
  function renderRegimeList() {
  const listEl = el('wb-regime-list'); if (!listEl) return;
  const vsBtn = el('btn-vs-perop'), swBtn = el('btn-swap-ab');
  if (vsBtn) vsBtn.className = 'wb-mini-btn' + (STATE.pinB < 0 ? ' active-b' : '');
  if (swBtn) swBtn.disabled = STATE.pinB < 0;
  listEl.innerHTML = REGIMES.map((r, i) => {
  const ev = evaluateWorkload(regimeToWorkload(i));
  const isA = i === STATE.pinA && !STATE.customDirty && !STATE.meshDirty;
  const isB = i === STATE.pinB;
  const cls = (isA ? ' pin-a' : '') + (isB ? ' pin-b' : '');
  const recScope = ev.recScope || ev.winner.scope;
  const recEst = ev.recWinner || ev.scopes.find(s => s.scope === recScope) || ev.winner;
  const isInvalid = !ev.isPhysValid;
  const scopeLbl = isInvalid ? 'invalid' : recEst.scope;
  const col = isInvalid ? '#EF4444' : (SCOPE_COLORS[recEst.scope] || '#94A3B8');
  const hypTag = !isInvalid && ev.winner && ev.winner.scope !== recEst.scope
    ? ` (hyp: ${escH(ev.winner.scope)} ${ev.winner.step_bound_us}us)` : '';
  return `<div class="wb-regime-item${cls}" onclick="wbPinRegime(${i}, 'A')">` +
  `<div style="min-width:0;flex:1;"><div style="font-weight:600;white-space:nowrap;` +
  `overflow:hidden;text-overflow:ellipsis;">${escH(r.label)}</div><div style=` +
  `"color:var(--muted);font-size:10px;"><span style="color:${col};font-weight:700;">` +
  `${escH(scopeLbl)}</span> &middot; T=${ev.activeTokens} &middot; ` +
  `${recEst.step_bound_us} us${hypTag}</div></div><button class="wb-mini-btn` +
  `${isA ? ' active-a' : ''}" type="button" onclick="event.stopPropagation();` +
  `wbPinRegime(${i}, 'A')">A</button><button class="wb-mini-btn` +
  `${isB ? ' active-b' : ''}" type="button" onclick="event.stopPropagation();` +
  `wbPinRegime(${i}, 'B')">B</button></div>`; }).join(''); }
  window.wbSetDispatchRegime = function(mode) {
  STATE.dispatchRegime = mode; STATE.hwDirty = false; STATE.calibrated = false;
  if (mode === 'zero_floor') {
    el('ctl-launch-gap').value = '0'; STATE.launchGapHiUs = 0.0;
  } else if (mode === 'compiled_xla') {
    el('ctl-launch-gap').value = '1.00'; STATE.launchGapHiUs = 1.5;
  } else {
    el('ctl-launch-gap').value = '6.00'; STATE.launchGapHiUs = 8.0;
  }
  window.wbRenderAll(); };
  window.wbOnLayoutPresetChange = function() {
  const p = STATE.presets[parseInt(el('ctl-layout').value, 10) || 0];
  if (p) syncMeshSelects(p);
  STATE.meshDirty = false; STATE.customDirty = true; window.wbRenderAll(); };
  window.wbOnMeshChange = function() {
  STATE.meshDirty = true; STATE.customDirty = true; window.wbRenderAll(); };
  window.wbPinRegime = function(idx, slot) { if (slot === 'B') {
  STATE.pinB = STATE.pinB === idx ? -1 : idx; } else {
  STATE.pinA = idx; STATE.customDirty = false; STATE.meshDirty = false;
  syncControlsFromRegime(idx);
  const winSc = evaluateWorkload(regimeToWorkload(idx)).winner.scope;
  el('ctl-arena-scope').value = winSc;
  STATE.selectedTopic = { kind: 'scope', scope: winSc, sub: 'total' }; }
  window.wbRenderAll(); };
  window.wbSwapAB = function() {
  if (STATE.pinB < 0) return;
  [STATE.pinA, STATE.pinB] = [STATE.pinB, STATE.pinA];
  STATE.customDirty = false; STATE.meshDirty = false;
  syncControlsFromRegime(STATE.pinA);
  const winSc = evaluateWorkload(regimeToWorkload(STATE.pinA)).winner.scope;
  el('ctl-arena-scope').value = winSc;
  STATE.selectedTopic = { kind: 'scope', scope: winSc, sub: 'total' };
  window.wbRenderAll(); };
  window.wbClearB = function() { STATE.pinB = -1; window.wbRenderAll(); };
  window.wbOnArenaScopeChange = function() {
  const sc = el('ctl-arena-scope').value || 'whole_step';
  STATE.selectedTopic = { kind: 'scope', scope: sc, sub: 'total' };
  window.wbRenderAll(); };
  window.wbResetSliders = function() {
  STATE.pinA = 0; STATE.pinB = -1; STATE.resetSeq = (STATE.resetSeq || 0) + 1;
  STATE.customDirty = false; STATE.meshDirty = false;
  STATE.hwDirty = false; STATE.calibrated = false;
  STATE.dispatchRegime = 'compiled_xla'; STATE.launchGapHiUs = 1.5;
  el('ctl-vmem').value = String(Math.min(STATE.maxVmemMib,
  Math.max(2.0, DATA.vmem_budget_mib || STATE.maxVmemMib)));
  el('ctl-dma').value = String(DATA.dma_buffers || 2); el('ctl-alias').value = '1';
  el('ctl-eta-dense').value = '0.72'; el('ctl-eta-moe').value = '0.50';
  el('ctl-eta-hbm').value = '0.82'; el('ctl-eta-ici').value = '0.75';
  el('ctl-coll-hop').value = ((CHIP.collective_latency_s || 1e-6) * 1e6).toFixed(2);
  syncControlsFromRegime(0);
  const winSc = evaluateWorkload(regimeToWorkload(0)).winner.scope;
  el('ctl-arena-scope').value = winSc;
  STATE.selectedTopic = { kind: 'scope', scope: winSc, sub: 'total' };
  window.wbRenderAll(); };
  window.wbOnPhaseChange = function() {
  const ph = el('ctl-phase').value, seqEl = el('ctl-seq-tokens');
  const cur = parseInt(seqEl.value, 10) || 1;
  if (ph === 'decode') seqEl.value = '1';
  else if (ph === 'verify' && cur <= 1) {
    seqEl.value = String((PROF.default_spec_tokens || 3) + 1);
  } else if ((ph === 'prefill' || ph === 'mixed') && cur <= 16) seqEl.value = '256';
  seqEl.disabled = ph === 'decode';
  STATE.customDirty = true; STATE.topicPinned = false; window.wbRenderAll(); };
  window.wbOnSeqChange = function() {
  if (el('ctl-phase').value === 'decode') { el('ctl-seq-tokens').value = '1'; return; }
  STATE.customDirty = true; STATE.topicPinned = false; window.wbRenderAll(); };
  window.wbRenderAll = function() {
  const wlA = currentWorkloadA(); updateControlLabels(wlA);
  const evalA = evaluateWorkload(wlA), hasB = STATE.pinB >= 0 && REGIMES[STATE.pinB];
  if (STATE.selectedTopic && STATE.selectedTopic.kind === 'waterfall') {
    if (!STATE.topicPinned) {
      STATE.selectedTopic.scope = evalA.winner.scope;
      el('ctl-arena-scope').value = evalA.winner.scope;
    }
  }
  const evalB = hasB ? evaluateWorkload(regimeToWorkload(STATE.pinB)) : null;
  const lblB = hasB ? regimeToWorkload(STATE.pinB).label : '';
  renderTopVerdictAndBanner(wlA, evalA);
  const lblScopeA = !evalA.isPhysValid
    ? 'invalid' : (evalA.recScope || evalA.winner.scope);
  const lblScopeB = evalB
    ? (!evalB.isPhysValid ? 'invalid' : (evalB.recScope || evalB.winner.scope)) : '';
  if (el('wb-ab-status')) el('wb-ab-status').innerHTML = `<strong>[A]</strong> ` +
  `${escH(wlA.label)} (${lblScopeA}) vs <strong>[B]</strong> ` +
  (hasB ? `${escH(lblB)} (${lblScopeB})` : 'per_op') +
  (STATE.resetSeq ? ` &middot; #${STATE.resetSeq}` : '');
  const recKpi = evalA.recWinner ||
    evalA.scopes.find(s => s.scope === (evalA.recScope || evalA.winner.scope)) || evalA.winner;
  el('wb-kpi-scope').textContent = `${recKpi.scope} (${evalA.moeModeRec})`;
  el('wb-kpi-floor').textContent = `${recKpi.step_bound_us.toFixed(1)} ~ ` +
  `${recKpi.step_upper_us.toFixed(1)} us`; el('wb-kpi-tmax').textContent =
  `${recKpi.t_vmem_max} tok (` + (recKpi.nCalls > 1
    ? `${recKpi.nCalls} calls` : `${recKpi.n_splits}p`) + ')';
  el('wb-kpi-tps').textContent = `${evalA.tpsChip.toFixed(1)} tok/s/chip`;
  renderRegimeList(); renderVmemArena(evalA); renderScopeTimeline(evalA);
  renderSignedWaterfall(evalA, evalB, lblB);
  renderDerivDrawer(wlA, evalA, evalB, lblB);
  if (window.wbRenderPhaseExtension) window.wbRenderPhaseExtension(wlA, evalA); };
  window.wbOnControl = function() { STATE.customDirty = true; STATE.topicPinned = false; window.wbRenderAll(); };
  window.wbOnHwControl = function() { STATE.hwDirty = true; window.wbRenderAll(); };
  window.wbSelectTopic = function(kind, scope, sub) {
  STATE.selectedTopic = { kind, scope, sub }; STATE.topicPinned = true; el('ctl-arena-scope').value = scope;
  window.wbRenderAll(); }; window.wbSetView = function(mode) {
  const isWb = mode === 'workbench';
  document.body.classList.toggle('wb-mode-active', isWb);
  el('view-workbench').classList.toggle('view-hidden', !isWb);
  el('view-audit').classList.toggle('view-hidden', isWb);
  el('btn-view-workbench').classList.toggle('active', isWb);
  el('btn-view-audit').classList.toggle('active', !isWb); };
  window.wbSetStageTab = function(tab) {
  const isAr = tab === 'arena', arEl = el('wb-stage-arena');
  const phEl = el('wb-stage-phase');
  if (arEl) arEl.style.display = isAr ? 'flex' : 'none';
  if (phEl) phEl.style.display = isAr ? 'none' : 'flex';
  if (el('btn-stage-arena')) el('btn-stage-arena').classList.toggle('active', isAr);
  if (el('btn-stage-phase')) el('btn-stage-phase').classList.toggle('active', !isAr); };
  window.__WB_API = {
    STATE,
    DATA,
    ENG,
    el,
    fmt: {
      fmtCtx,
      fmtUs: typeof fmtUs !== 'undefined' ? fmtUs : (v => v.toFixed(2)),
      fmtK: typeof fmtK !== 'undefined' ? fmtK : (v => v.toFixed(1)),
    },
    esc: escH,
    escH,
    setText: (id, txt) => {
      const e = el(id);
      if (e) e.textContent = txt;
    },
    setHtml: (id, html) => {
      const e = el(id);
      if (e) e.innerHTML = html;
    },
    recompute: window.wbRenderAll,
    SCOPE_COLORS,
    evaluateWorkload,
    currentWorkloadA,
    regimeToWorkload,
    syncControlsFromRegime,
    coreStepNoLaunch,
    exposedCommUs,
  };
  initPresets();
  if (REGIMES.length > 0) syncControlsFromRegime(0);
  document.body.classList.add('wb-mode-active');
})();
