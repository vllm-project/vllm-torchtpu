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
  var WB = window.__WB_API || {};
  var STATE = WB.STATE || {};
  var DATA = WB.DATA || {};
  const PHASE_GRID_SEED = globalThis.__MK_PHASE_GRID || {};
  const PHASE_T_GRID = Array.isArray(PHASE_GRID_SEED.t_grid)
    ? PHASE_GRID_SEED.t_grid
    : [1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
  const PHASE_S_GRID = Array.isArray(PHASE_GRID_SEED.s_grid)
    ? PHASE_GRID_SEED.s_grid
    : [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 1048576];

  (function mountPhaseDom() {
    const railSlot = document.getElementById('wb-extra-rail-slot');
    if (railSlot) {
      railSlot.innerHTML =
        '<div class="wb-sec-title">4. Trace Calibration Dropzone</div>' +
        '<div class="wb-knob-group" id="wb-cal-dropzone" ' +
        'ondragover="event.preventDefault()" ' +
        'ondrop="wbOnDropCalibration(event)">' +
        '<div style="font-size:10px;color:var(--muted);">' +
        'Drop <code>advisor.measurements.v1</code> or Chrome trace ' +
        '(<code>.json</code> / <code>.json.gz</code>) to upgrade ' +
        '<code>ASSUMED</code> priors to <code>MEASURED</code>.' +
        '</div>' +
        '<input id="cal-file-input" type="file" ' +
        'accept=".json,.gz,application/json,application/gzip" ' +
        'onchange="wbOnPickCalibration(event)" ' +
        'style="font-size:10px;width:100%;"/>' +
        '<div id="wb-cal-status" style="font-size:10px;color:var(--muted);">' +
        'Status: uncalibrated hardware priors</div></div>';
    }
    const stageSlot = document.getElementById('wb-extra-stage-slot');
    if (stageSlot) {
      stageSlot.innerHTML =
        '<div id="wb-stage-phase" ' +
        'style="display:none;flex-direction:column;gap:14px;">' +
        '<div class="wb-panel"><div class="wb-panel-head">' +
        '<h2>Continuous 2D Winning-Scope Phase Map ' +
        '(Active Tokens &times; Context)</h2>' +
        '<div style="display:flex;align-items:center;gap:8px;">' +
        '<span class="kpi-sub">Scope View:</span>' +
        '<select class="wb-inline-sel" id="wb-phase-map-mode" ' +
        'onchange="wbSetPhaseMapMode(this.value)">' +
        '<option value="recommended" selected>Recommended</option>' +
        '<option value="floor">Floor Winner</option></select>' +
        '<span class="kpi-sub">Click any cell to inspect in Arena</span>' +
        '</div></div><div id="wb-phase-map-svg"></div></div>' +
        '<div class="wb-panel"><div class="wb-panel-head">' +
        '<h2>Advisory Token-Bucket Schedule JSON (Offline Export)</h2>' +
        '<button class="wb-mini-btn" id="btn-copy-bucket-schedule" ' +
        'type="button" onclick="wbCopyBucketSchedule()">' +
        'Copy Bucket Schedule JSON</button></div>' +
        '<div class="kpi-sub" style="margin-bottom:6px;">' +
        'Schema <code>advisor.bucket_schedule.v1</code> ' +
        '(advisory offline schedule; unwired to runtime runner dispatch).' +
        '</div><pre class="wb-formula" id="wb-bucket-schedule-pre" ' +
        'style="max-height:200px;overflow-y:auto;margin:0;"></pre>' +
        '</div></div>';
    }
  })();

  window.wbSetPhaseMapMode = function(mode) {
    STATE.phaseMapMode = (mode === 'floor') ? 'floor' : 'recommended';
    const modeSelect = document.getElementById('wb-phase-map-mode');
    if (modeSelect) modeSelect.value = STATE.phaseMapMode;
    const currentWorkload = (WB.currentWorkloadA && WB.currentWorkloadA()) || {};
    const evalA = (window.__WB_API && window.__WB_API.evaluateWorkload)
      ? window.__WB_API.evaluateWorkload(currentWorkload)
      : null;
    window.wbRenderPhaseExtension(currentWorkload, evalA);
  };

  window.wbFocusedTopicCard = function(wlA, evalA, evalB, lblB, topic, sc) {
    if (!topic || !sc) return '';
    const esc = (window.__WB_API && window.__WB_API.escH) || String;
    if (topic.kind === 'vmem') {
      const spK = (evalA.vmemParts.spineBytes / 1024).toFixed(1);
      const vmap = {
        dense: ['Dense Weight Ring', sc.dense_stage_mib,
          `slots=${sc.selected_dense_slots} * 2 MiB (adaptive ring)`],
        exp: ['Routed Expert Ring', sc.expert_stage_mib,
          `slots=${sc.selected_expert_slots} * 1 MiB (adaptive ring)`],
        scr: ['Static Op Scratch', sc.op_scratch_mib,
          'sequential attention / linear / FFN tile scratch'],
        pad: ['Arena Alignment Reserve', sc.pad_mib,
          '2.0 MiB on-chip arena alignment reserve'],
        spn: ['Dynamic [T, H] Residual Spine', sc.spine_dyn_mib,
          `T=${evalA.activeTokens} tok * ${spK} KiB/tok`],
        tok: ['Dynamic Per-Token Op Scratch', sc.op_dyn_mib,
          `alias=${wlA.vmemAlias ? 'enabled (max)' : 'disabled (sum)'} (#ctl-alias)`]
      };
      const info = vmap[topic.sub] || ['VMEM Region', sc.vmem_static_mib, sc.scope];
      const pct = ((info[1] / Math.max(1, evalA.vmemBudgetMib)) * 100).toFixed(1);
      return `<div class="wb-deriv-box"><div style="display:flex;` +
        `justify-content:space-between;"><strong>Selected VMEM: ${esc(info[0])}` +
        `</strong><span class="prov-tag prov-tag-DERIVED">DERIVED</span></div>` +
        `<div class="wb-formula">${info[1].toFixed(2)} MiB (${pct}% of ` +
        `${evalA.vmemBudgetMib.toFixed(1)} MiB budget)\n${esc(info[2])}</div></div>`;
    }
    if (topic.kind === 'waterfall') {
      const scA = (topic && topic.scope)
        ? (evalA.scopes.find(s => s.scope === topic.scope) || evalA.winner)
        : (sc || evalA.winner);
      const perOp = evalA.scopes.find(s => s.scope === 'per_op');
      const scB = evalB ? evalB.winner : (perOp || evalA.runnerUp);
      const cmpLbl = evalB ? `${lblB} (${scB.scope})` : `Baseline (${scB.scope})`;
      const idx = parseInt(topic.sub || '0', 10) || 0;
      // Same split as the waterfall bars: the chain/HBM envelope, then the part of the
      // engine's core time that collectives add on top of it.
      const expComm = window.__WB_API.exposedCommUs;
      const envA = Math.max(scA.chain_us, scA.hbm_us);
      const envB = Math.max(scB.chain_us, scB.hbm_us);
      const cA = expComm(scA), cB = expComm(scB);
      const rows = [
        ['Compute/HBM Roofline Core', envA, envB, 'max(t_chain, t_hbm) overlap floor'],
        ['Exposed Collective Sync & Bounce', cA, cB,
          'Collective time the HBM stream does not cover (#ctl-coll-hop)'],
        ['Cold Pipeline Fill/Drain', scA.fill_drain_us, scB.fill_drain_us,
          'Cold HBM slot fill + synchronous stage barriers'],
        ['Multi-Pass HBM Weight Re-stream', scA.repass_spill_us, scB.repass_spill_us,
          `n_splits=${scA.n_splits} vs ${scB.n_splits} (knob: #ctl-vmem)`],
        ['Dispatch / Boundary Bubble',
          scA.ops_per_step * (evalA.launchGapPriorUs ?? evalA.launchGapLoUs),
          scB.ops_per_step * (evalA.launchGapPriorUs ?? evalA.launchGapLoUs),
          `ops=${scA.ops_per_step} vs ${scB.ops_per_step} * gap (#ctl-launch-gap)`]
      ];
      const r = rows[idx] || rows[0];
      const rawD = r[1] - r[2], d = Math.abs(rawD) < 5e-3 ? 0 : rawD;
      const sgn = d > 0 ? '+' : '';
      return `<div class="wb-deriv-box"><div style="display:flex;` +
        `justify-content:space-between;"><strong>Waterfall: ${esc(r[0])}</strong>` +
        `<span class="prov-tag prov-tag-DERIVED">DERIVED</span></div>` +
        `<div class="wb-formula">[A] ${scA.scope}: ${r[1].toFixed(2)} us\n` +
        `[B] ${esc(cmpLbl)}: ${r[2].toFixed(2)} us\n` +
        `Delta ([A] - [B]): ${sgn}${d.toFixed(2)} us\n${esc(r[3])}</div></div>`;
    }
    return '';
  };

  function resolveEpDpAndRowFloor(wlBase) {
    const ENG = window.__MK_ADVISOR_ENGINE;
    const tp = Math.max(1, wlBase.tp || 1), dcp = Math.max(1, wlBase.dcp || 1);
    const ep = Math.max(1, wlBase.ep || 1);
    const effTp = tp === 1 ? Math.max(1, dcp) : tp;
    const epDp = Math.max(1, Math.floor(ep / ENG.gcd(ep, effTp)));
    const rFloor = ENG.rowFloor(ENG.PROF.expert_dtype);
    return { epDp, rFloor };
  }

  function buildLiveBucketSchedule(wlBase) {
    const intervals = [];
    let cur = null;
    const { epDp, rFloor } = resolveEpDpAndRowFloor(wlBase);
    const dp = Math.max(1, wlBase.dp || 1);
    for (const tVal of PHASE_T_GRID) {
      const wl = Object.assign({}, wlBase, {
        phase: 'decode', batch: tVal * dp, seqTokens: 1,
        moeMode: (tVal * epDp >= rFloor) ? 'grouped_on_chip' : 'masked_on_chip'
      });
      const res = window.__WB_API.evaluateWorkload(wl);
      const recSc = res.recScope || res.winner.scope;
      const recWin = (res.scopes || []).find(s => s.scope === recSc) ||
        res.recWinner || res.winner;
      if (cur && cur.recommended_scope === recSc &&
          cur.recommended_moe_mode === res.moeModeRec) {
        cur.max_active_tokens = tVal;
        cur.step_bound_us = recWin.step_bound_us;
        cur.max_vmem_tokens = recWin.t_vmem_max;
        cur.num_vmem_splits = recWin.n_splits;
      } else {
        cur = {
          phase: 'decode',
          layout: document.getElementById('lbl-layout').textContent,
          min_active_tokens: tVal,
          max_active_tokens: tVal,
          effective_context: wlBase.context,
          recommended_scope: recSc,
          recommended_moe_mode: res.moeModeRec,
          max_vmem_tokens: recWin.t_vmem_max,
          num_vmem_splits: recWin.n_splits,
          step_bound_us: recWin.step_bound_us
        };
        intervals.push(cur);
      }
    }
    return {
      schema: 'advisor.bucket_schedule.v1',
      consumer_status: 'advisory_offline_v1',
      model: DATA.model, chip: DATA.chip,
      default_layout: document.getElementById('lbl-layout').textContent,
      calibrated: STATE.calibrated,
      buckets: intervals
    };
  }

  window.wbCalNotes = function(sc) {
    const esc = (window.__WB_API && window.__WB_API.escH) || String;
    const pTag = t => `<span class="prov-tag prov-tag-${t}">${t}</span>`;
    const line = STATE.calibrated
      ? `${pTag('MEASURED')} Source: ${esc(STATE.calSource)}`
      : `${pTag('ASSUMED')} Settle: calibrate --trace &lt;p&gt;`;
    const ratio = (STATE.calStepTimeUs && STATE.calStepTimeUs > 0)
      ? (sc.step_bound_us / Math.max(0.1, STATE.calStepTimeUs) * 100).toFixed(1) : '';
    const anc = (STATE.calibrated && STATE.calStepTimeUs !== null)
      ? '<div class="wb-deriv-box"><strong>Measured Step:</strong> ' +
        `<span class="tabular">${STATE.calStepTimeUs.toFixed(1)} us</span> ` +
        `(${ratio}%)</div>`
      : '';
    return { line, anc };
  };

  const PHASE_ROLES = PHASE_GRID_SEED.roles || {};
  const PHASE_PREFILL_TOKEN_GRID = Array.isArray(PHASE_GRID_SEED.prefill_token_grid)
    ? PHASE_GRID_SEED.prefill_token_grid
    : [];
  const PHASE_CELL = { padLeft: 72, padTop: 40, width: 92, height: 34 };

  // Loads one map cell into the live controls. A decode column counts rows per replica,
  // a prefill column counts the tokens of one prompt.
  window.wbSelectPhaseCell = function(phase, tokenCount, contextTokens) {
    const currentWorkload = (WB.currentWorkloadA && WB.currentWorkloadA()) || {};
    const batchElement = document.getElementById('ctl-batch');
    const isPrefill = phase === 'prefill';
    const batchValue = isPrefill ? 1 : tokenCount * Math.max(1, currentWorkload.dp || 1);
    STATE.customDirty = true;
    document.getElementById('ctl-phase').value = phase;
    if (batchElement) {
      batchElement.max = String(Math.max(1024, batchValue));
      batchElement.value = String(batchValue);
    }
    document.getElementById('ctl-seq-tokens').value = isPrefill ? String(tokenCount) : '1';
    document.getElementById('ctl-context').value = String(contextTokens);
    window.wbSetStageTab('arena');
    window.wbRenderAll();
  };

  function fmtPhaseCtx(contextTokens) {
    return contextTokens >= 1048576
      ? (contextTokens / 1048576) + 'M'
      : (contextTokens / 1024) + 'K';
  }

  // Workload of one map cell: the current workload with the step of the column and the
  // context of the row. When the model has deployment roles, the cell runs on the layout
  // of its role (decode or prefill).
  function phaseCellWorkload(currentWorkload, phase, tokenCount, contextTokens) {
    const role = PHASE_ROLES[phase];
    const layout = role
      ? { tp: role.tp, ep: role.ep, dp: role.dp, pp: role.pp, dcp: role.dcp, pcp: role.pcp,
          layerGroups: role.layer_groups }
      : {};
    const replicas = Math.max(1, role ? role.dp : (currentWorkload.dp || 1));
    // A prefill cell is one prompt in one pass: no chunking, no cached prefix, no
    // microbatches. A decode cell is tokenCount rows per replica, one row per request.
    const step = phase === 'prefill'
      ? { batch: 1, seqTokens: tokenCount, isl: tokenCount, chunkSize: 0, prefixHitRate: 0,
          microbatches: 0 }
      : { batch: tokenCount * replicas, seqTokens: 1 };
    return Object.assign({}, currentWorkload, layout, step,
      { phase, specTokens: 0, context: contextTokens });
  }

  // SVG of one map cell. It shows the recommended scope (lowest floor among the supported
  // scopes) or, in floor mode, the scope with the lowest floor, which can be hypothetical.
  function phaseCellSvg(cell, evaluated, mode, isActive) {
    const colors = window.__WB_API.SCOPE_COLORS || {};
    const esc = (window.__WB_API && window.__WB_API.escH) || String;
    const isOom = Boolean(
      evaluated.hbmPoolOverflow || !evaluated.hbmFits ||
      (evaluated.invalidReasons && evaluated.invalidReasons.includes('hbm_pool_overflow'))
    );
    const winner = evaluated.winner;
    const recommended = evaluated.scopes.find(scope => scope.scope === evaluated.recScope);
    const shown = mode === 'floor' ? winner : recommended;
    const shownSupport = shown.support || 'baseline';
    const showsWinner = shown.scope === winner.scope;
    const isHypothetical = !isOom && shownSupport === 'hypothetical';
    const isTie = !isOom && showsWinner && evaluated.marginPct < 5.0;
    const subLabel = isHypothetical ? '(hypothetical)' : (isTie ? '&lt;5% margin' : '');
    const scopeLabel = isOom ? 'OOM' : shown.scope;
    const fill = isOom ? 'url(#wb-oom-hatch)' : (colors[shown.scope] || '#64748B');
    let stroke = '';
    if (isActive) {
      stroke = ' stroke="#FFFFFF" stroke-width="2.2"';
      if (isHypothetical) stroke += ' stroke-dasharray="3,2"';
    } else if (isHypothetical) {
      stroke = ' stroke="#FFFFFF" stroke-width="1.2" stroke-dasharray="3,2"';
    }
    const hbmGigabytes = (evaluated.hbmUsedBytes / 1e9).toFixed(1);
    const lowestFloor = showsWinner
      ? ''
      : `; lowest floor: ${winner.scope} [${winner.support || 'baseline'}] ` +
        `${winner.step_bound_us} us`;
    const verdictText = isOom
      ? `OOM > 80% HBM (${hbmGigabytes} GB)`
      : `${shown.scope} [${shownSupport}]` +
        (showsWinner ? ` (${evaluated.moeModeRec})` : '') +
        `, floor=${shown.step_bound_us} us${lowestFloor}`;
    const tooltip = esc(
      `${cell.phase} ${cell.unit}=${cell.tokenCount}, context=${cell.contextLabel}` +
      ` -> ${verdictText}`
    );
    const centerX = cell.x + PHASE_CELL.width / 2;
    const subSvg = subLabel
      ? `<text x="${centerX}" y="${cell.y + 26}" text-anchor="middle" ` +
        `fill="#FFFFFF" font-size="8.5" opacity="0.9">${subLabel}</text>`
      : '';
    const yMain = subLabel ? (cell.y + 15) : (cell.y + 21);
    return `<g class="wb-clickable" data-phase="${cell.phase}" ` +
      `data-column="${cell.tokenCount}" data-context="${cell.contextTokens}" ` +
      `data-scope="${scopeLabel}" data-rec-scope="${recommended.scope}" ` +
      `data-floor-scope="${winner.scope}" data-support="${shownSupport}" ` +
      `data-active-tokens="${evaluated.activeTokens}" ` +
      `onclick="wbSelectPhaseCell('${cell.phase}',${cell.tokenCount},${cell.contextTokens})">` +
      `<rect x="${cell.x + 1}" y="${cell.y + 1}" width="${PHASE_CELL.width - 2}" ` +
      `height="${PHASE_CELL.height - 2}" rx="3" fill="${fill}" fill-opacity="0.88"${stroke}>` +
      `<title>${tooltip}</title></rect>` +
      `<text x="${centerX}" y="${yMain}" text-anchor="middle" ` +
      `fill="#FFFFFF" font-size="10" font-weight="700">` +
      `${scopeLabel}</text>${subSvg}</g>`;
  }

  window.wbRenderPhaseExtension = function(currentWorkload, evalA) {
    const mapElement = document.getElementById('wb-phase-map-svg');
    const preElement = document.getElementById('wb-bucket-schedule-pre');
    if (!mapElement || !preElement) return;
    const colors = window.__WB_API.SCOPE_COLORS || {};
    const esc = (window.__WB_API && window.__WB_API.escH) || String;
    const mode = STATE.phaseMapMode === 'floor' ? 'floor' : 'recommended';
    const hasRoles = Boolean(PHASE_ROLES.decode && PHASE_ROLES.prefill);
    const prefillTokenGrid = hasRoles ? PHASE_PREFILL_TOKEN_GRID : [];
    const columns = PHASE_T_GRID.map(
      tokenCount => ({ phase: 'decode', unit: 'rows', tokenCount })
    ).concat(prefillTokenGrid.map(
      tokenCount => ({ phase: 'prefill', unit: 'tokens', tokenCount })
    ));
    const gridBottom = PHASE_CELL.padTop + PHASE_S_GRID.length * PHASE_CELL.height;
    const svgWidth = PHASE_CELL.padLeft + columns.length * PHASE_CELL.width + 16;
    const svgHeight = gridBottom + 42;
    // The highlighted cell is the live workload: its rows on a decode step, its prompt
    // tokens on a prefill step.
    const liveIsPrefill = currentWorkload.phase === 'prefill';
    const liveTokens = liveIsPrefill
      ? currentWorkload.seqTokens
      : (evalA ? evalA.activeTokens : currentWorkload.batch);
    let cells = '';
    PHASE_S_GRID.forEach((contextTokens, rowIndex) => {
      const y = PHASE_CELL.padTop + (PHASE_S_GRID.length - 1 - rowIndex) * PHASE_CELL.height;
      const contextLabel = fmtPhaseCtx(contextTokens);
      cells += `<text x="${PHASE_CELL.padLeft - 8}" y="${y + 22}" text-anchor="end" ` +
        `fill="var(--muted)" font-size="11" class="tabular">${contextLabel}</text>`;
      columns.forEach((column, columnIndex) => {
        const workload = phaseCellWorkload(
          currentWorkload, column.phase, column.tokenCount, contextTokens
        );
        const isActive = (column.phase === 'prefill') === liveIsPrefill &&
          column.tokenCount === liveTokens && contextTokens === currentWorkload.context;
        const cell = Object.assign(
          { x: PHASE_CELL.padLeft + columnIndex * PHASE_CELL.width, y, contextTokens,
            contextLabel },
          column
        );
        cells += phaseCellSvg(cell, window.__WB_API.evaluateWorkload(workload), mode, isActive);
      });
    });
    const ticks = columns.map((column, columnIndex) => {
      const centerX = PHASE_CELL.padLeft + (columnIndex + 0.5) * PHASE_CELL.width;
      return `<text x="${centerX}" y="${gridBottom + 18}" text-anchor="middle" ` +
        `fill="var(--muted)" font-size="11" class="tabular">${column.tokenCount}</text>`;
    }).join('');

    const defs = '<defs><pattern id="wb-oom-hatch" width="8" height="8" ' +
      'patternUnits="userSpaceOnUse" patternTransform="rotate(45)">' +
      '<rect width="8" height="8" fill="#7F1D1D"/>' +
      '<line x1="0" y1="0" x2="0" y2="8" stroke="#EF4444" stroke-width="3"/>' +
      '</pattern></defs>';

    const legend = `<g class="wb-phase-legend" font-size="11" fill="var(--muted)">` +
      `<text x="${PHASE_CELL.padLeft}" y="16">` +
      `Y: Context Length | X: Active Tokens | Legend: ` +
      `<tspan fill="${colors.whole_step || '#2563EB'}">whole_step</tspan> &middot; ` +
      `<tspan fill="${colors.layer_group || '#059669'}">layer_group</tspan> &middot; ` +
      `<tspan fill="${colors.single_layer || '#D97706'}">single_layer</tspan> &middot; ` +
      `<tspan fill="${colors.per_op || '#475569'}">per_op</tspan> &middot; ` +
      `<tspan fill="#EF4444">OOM = OOM &gt; 80% HBM</tspan> | ` +
      `dashed: (hypothetical) &middot; &lt;5% margin` +
      `</text></g>`;

    // One header per column block. It names the role layout when the model has one.
    const blockHeader = (firstColumn, columnCount, text) => {
      const centerX = PHASE_CELL.padLeft + (firstColumn + columnCount / 2) * PHASE_CELL.width;
      return `<text x="${centerX}" y="32" text-anchor="middle" fill="var(--muted)" ` +
        `font-size="11" font-weight="700">${esc(text)}</text>`;
    };
    const onLayout = role => (role && role.label ? ` on ${role.label}` : '');
    let headers = blockHeader(
      0, PHASE_T_GRID.length, `decode rows${onLayout(PHASE_ROLES.decode)}`
    );
    let divider = '';
    if (prefillTokenGrid.length) {
      headers += blockHeader(
        PHASE_T_GRID.length, prefillTokenGrid.length,
        `prefill tokens${onLayout(PHASE_ROLES.prefill)}`
      );
      const dividerX = PHASE_CELL.padLeft + PHASE_T_GRID.length * PHASE_CELL.width;
      divider = `<line x1="${dividerX}" y1="${PHASE_CELL.padTop - 6}" x2="${dividerX}" ` +
        `y2="${gridBottom}" stroke="var(--border, #373B5E)" stroke-width="1.5" ` +
        `stroke-dasharray="4,2"/>`;
    }
    mapElement.innerHTML = `<svg viewBox="0 0 ${svgWidth} ${svgHeight}" class="chart-svg" ` +
      `role="img" aria-label="2D Winning-Scope Phase Map">${defs}` +
      `${legend}${headers}${divider}${cells}${ticks}</svg>`;
    preElement.textContent = JSON.stringify(buildLiveBucketSchedule(currentWorkload), null, 2);
  };

  function applyCalibrationObject(obj, fileName) {
    if (!obj || obj.schema !== 'advisor.measurements.v1') {
      const st = document.getElementById('wb-cal-status');
      if (st) st.textContent = 'Rejected: expected advisor.measurements.v1';
      return;
    }
    const hasGap = Boolean(obj.launch_gap_us && typeof obj.launch_gap_us.p50 === 'number');
    const hasColls = Array.isArray(obj.collectives) && obj.collectives.length > 0;
    const hasStepTime = Boolean(obj.step_time_us && (
      typeof obj.step_time_us.p50 === 'number' || typeof obj.step_time_us === 'number'
    ));
    const hasRing = Boolean(obj.ring_plan && typeof obj.ring_plan === 'object');
    if (!hasGap && !hasColls && !hasStepTime && !hasRing) {
      const st = document.getElementById('wb-cal-status');
      if (st) st.textContent = 'Rejected: payload lacks usable calibration measurements';
      return;
    }
    if (hasGap) {
      const p50 = obj.launch_gap_us.p50;
      document.getElementById('ctl-launch-gap').value = String(p50);
      STATE.launchGapHiUs = typeof obj.launch_gap_us.p90 === 'number'
        ? Math.max(p50, obj.launch_gap_us.p90) : p50;
    }
    if (hasColls) {
      let minHop = null;
      for (const c of obj.collectives) {
        if (c && typeof c.latency_us === 'number') {
          const grp = Math.max(2, parseInt(c.group_size || 2, 10));
          const hopUs = c.latency_us / Math.max(1, Math.round(Math.log2(grp)));
          if (minHop === null || hopUs < minHop) minHop = hopUs;
        }
      }
      if (minHop !== null) {
        const cVal = Number(minHop.toFixed(3));
        document.getElementById('ctl-coll-hop').value = String(cVal);
        STATE.collHopHiUs = cVal;
      }
    }
    if (hasRing) {
      const rp = obj.ring_plan;
      if (rp.dense_slots) {
        document.getElementById('ctl-dma').value = String(rp.dense_slots);
      }
      if (rp.dense_slot_bytes) STATE.denseSlotBytes = rp.dense_slot_bytes;
      if (rp.expert_slot_bytes) STATE.expertSlotBytes = rp.expert_slot_bytes;
    }
    const stRaw = obj.step_time_us;
    STATE.calStepTimeUs = (stRaw && typeof stRaw.p50 === 'number') ? stRaw.p50
      : (typeof stRaw === 'number' ? stRaw : null);
    STATE.calibrated = true;
    STATE.hwDirty = true;
    STATE.dispatchRegime = 'compiled_xla';
    const drLbl = document.getElementById('lbl-dispatch-regime');
    if (drLbl) drLbl.textContent = 'compiled_xla';
    STATE.calSource = (obj.source && obj.source.file)
      ? obj.source.file : fileName;
    const st = document.getElementById('wb-cal-status');
    if (st) st.textContent = 'Calibrated (MEASURED): ' + STATE.calSource;
    window.wbRenderAll();
  }
  function quantileSorted(arr, p) {
    if (!arr.length) return 0.0;
    const pos = p * (arr.length - 1), lo = Math.floor(pos);
    const hi = Math.min(arr.length - 1, lo + 1), frac = pos - lo;
    return arr[lo] * (1.0 - frac) + arr[hi] * frac;
  }
  function traceEventsToMeas(raw, fileName) {
    const evs = Array.isArray(raw) ? raw
      : (raw && Array.isArray(raw.traceEvents) ? raw.traceEvents : null);
    if (!evs) return raw;
    const valid = evs.filter(e => e && (!e.ph || e.ph === 'X') &&
      typeof e.ts === 'number' && typeof e.dur === 'number' && e.dur > 0);
    const isStep = e => /step/i.test(String(e.name || '')) ||
      /step/i.test(String(e.cat || ''));
    const isWrap = e => /^(jit[(]|pjit[(]|xla_computation[(]|module_main|__main__)/i
      .test(String(e.name || '').trim()) ||
      /xla modules/i.test(String(e.cat || ''));
    const steps = valid.filter(isStep).sort((a, b) => a.ts - b.ts);
    const spans = valid.filter(e => !isStep(e) && !isWrap(e))
      .sort((a, b) => a.ts - b.ts);
    const numLayers = (window.__MK_ADVISOR_ENGINE &&
      window.__MK_ADVISOR_ENGINE.PROF &&
      window.__MK_ADVISOR_ENGINE.PROF.num_layers) || 60;
    const boundaries = 2 * numLayers;
    const MIN_STEP_OPS_FOR_EXPLICIT_NORM = 16;

    let launchGapPayload = null;
    let stepTimePayload = null;

    if (steps.length > 0) {
      const minRest = steps.length > 1
        ? Math.min(...steps.slice(1).map(s => s.dur)) : 0.0;
      const dropWarmup = steps.length >= 3 ||
        (steps.length === 2 && steps[0].dur > 2.0 * minRest);
      const steady = dropWarmup ? steps.slice(1) : steps;
      const perBnd = [], stepIdles = [], stepDurs = [];
      let maxStepOps = 0;

      const spansByRank = new Map();
      for (const sp of spans) {
        const rk = sp.pid || 0;
        if (!spansByRank.has(rk)) spansByRank.set(rk, []);
        spansByRank.get(rk).push(sp);
      }

      for (const st of steady) {
        const s0 = st.ts, s1 = st.ts + st.dur;
        const rankSpans = spansByRank.has(st.pid) ? spansByRank.get(st.pid) : spans;
        const clipped = [];
        for (const sp of rankSpans) {
          const a = Math.max(s0, sp.ts), b = Math.min(s1, sp.ts + sp.dur);
          if (b > a) clipped.push([a, b]);
        }
        maxStepOps = Math.max(maxStepOps, clipped.length);
        if (clipped.length === 0) continue;
        clipped.sort((u, v) => u[0] - v[0]);
        let busy = 0.0;
        let curA = clipped[0][0], curB = clipped[0][1];
        for (let i = 1; i < clipped.length; i++) {
          if (clipped[i][0] <= curB) {
            curB = Math.max(curB, clipped[i][1]);
          } else {
            busy += (curB - curA);
            curA = clipped[i][0];
            curB = clipped[i][1];
          }
        }
        busy += (curB - curA);
        const idle = Math.max(0.0, st.dur - busy);
        stepIdles.push(idle);
        stepDurs.push(st.dur);
        perBnd.push(idle / (2.0 * numLayers));
      }

      if (stepDurs.length > 0) {
        stepDurs.sort((a, b) => a - b);
        stepTimePayload = {
          p50: Number(quantileSorted(stepDurs, 0.5).toFixed(2)),
          p90: Number(quantileSorted(stepDurs, 0.9).toFixed(2)),
          n: stepDurs.length
        };
      }

      const useConserved = stepIdles.length > 0 && (
        maxStepOps >= MIN_STEP_OPS_FOR_EXPLICIT_NORM || maxStepOps >= boundaries
      );
      if (useConserved) {
        perBnd.sort((a, b) => a - b);
        stepIdles.sort((a, b) => a - b);
        launchGapPayload = {
          p50: Number(quantileSorted(perBnd, 0.5).toFixed(4)),
          p90: Number(quantileSorted(perBnd, 0.9).toFixed(4)),
          n: stepIdles.length,
          step_idle_gap_us: Number(quantileSorted(stepIdles, 0.5).toFixed(2)),
          num_layers_norm: numLayers,
          unit: 'us_per_sublayer_boundary'
        };
      }
    }

    if (!launchGapPayload) {
      const streamSpans = new Map();
      for (const sp of spans) {
        const key = `${sp.pid || 0}:${sp.tid || 0}`;
        if (!streamSpans.has(key)) streamSpans.set(key, []);
        streamSpans.get(key).push(sp);
      }
      const gaps = [];
      for (const evList of streamSpans.values()) {
        evList.sort((a, b) => a.ts - b.ts);
        for (let i = 1; i < evList.length; i++) {
          const g = evList[i].ts - (evList[i - 1].ts + evList[i - 1].dur);
          if (g > 0 && g <= 500) gaps.push(g);
        }
      }
      gaps.sort((a, b) => a - b);
      if (gaps.length > 0) {
        launchGapPayload = {
          p50: Number(quantileSorted(gaps, 0.5).toFixed(4)),
          p90: Number(quantileSorted(gaps, 0.9).toFixed(4)),
          n: gaps.length
        };
      }
    }

    const out = {
      schema: 'advisor.measurements.v1',
      source: { file: fileName }
    };
    if (stepTimePayload) out.step_time_us = stepTimePayload;
    if (launchGapPayload) out.launch_gap_us = launchGapPayload;
    return out;
  }
  window.__WB_TRACE_TO_MEAS = traceEventsToMeas;
  window.__WB_APPLY_CALIBRATION = applyCalibrationObject;
  window.__WB_BUILD_LIVE_BUCKET_SCHEDULE = buildLiveBucketSchedule;
  async function readCalibrationFile(file) {
    if (!file) return;
    try {
      let text = '';
      if (file.name.endsWith('.gz') && typeof DecompressionStream !== 'undefined') {
        const ds = file.stream().pipeThrough(new DecompressionStream('gzip'));
        text = await new Response(ds).text();
      } else {
        text = await new Promise((res, rej) => {
          const reader = new FileReader();
          reader.onload = ev => res(String((ev.target && ev.target.result) || '{}'));
          reader.onerror = () => rej(new Error('read error'));
          reader.readAsText(file);
        });
      }
      const parsed = traceEventsToMeas(JSON.parse(text), file.name);
      applyCalibrationObject(parsed, file.name);
    } catch (_err) {
      const st = document.getElementById('wb-cal-status');
      if (st) st.textContent = 'Invalid JSON or gzip trace file';
    }
  }
  window.wbOnPickCalibration = function(ev) {
    const files = ev.target && ev.target.files;
    if (files && files[0]) readCalibrationFile(files[0]);
  };
  window.wbOnDropCalibration = function(ev) {
    ev.preventDefault();
    const files = ev.dataTransfer && ev.dataTransfer.files;
    if (files && files[0]) readCalibrationFile(files[0]);
  };
  window.wbCopyBucketSchedule = function() {
    const preEl = document.getElementById('wb-bucket-schedule-pre');
    const btn = document.getElementById('btn-copy-bucket-schedule');
    if (preEl && navigator.clipboard) {
      navigator.clipboard.writeText(preEl.textContent || '');
    }
    if (btn) {
      btn.textContent = 'Copied!';
      setTimeout(() => { btn.textContent = 'Copy Bucket Schedule JSON'; }, 1200);
    }
  };
