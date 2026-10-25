(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const ACTION_NAMES = ["KEEP", "CREATE", "DROP"];

  const els = {
    connPill: $("connPill"), connText: $("connText"),
    runPill: $("runPill"), runText: $("runText"),
    clock: $("clock"),
    startBtn: $("startBtn"), pauseBtn: $("pauseBtn"),
    stopBtn: $("stopBtn"), resetBtn: $("resetBtn"),
    errorBanner: $("errorBanner"),

    kpiQps: $("kpiQps"), kpiQpsTarget: $("kpiQpsTarget"),
    kpiQueries: $("kpiQueries"), kpiSteps: $("kpiSteps"),
    kpiPhase: $("kpiPhase"), kpiPhaseSub: $("kpiPhaseSub"),
    kpiEpsilon: $("kpiEpsilon"),
    kpiReplay: $("kpiReplay"), kpiReplayCap: $("kpiReplayCap"), kpiUpdates: $("kpiUpdates"),
    kpiDrift: $("kpiDrift"), kpiConsolidations: $("kpiConsolidations"),

    currentQuery: $("currentQuery"), decisionHint: $("decisionHint"),
    exploreBadge: $("exploreBadge"), actionBadge: $("actionBadge"),
    curReward: $("curReward"), curLatency: $("curLatency"), curIndexes: $("curIndexes"),
    qBars: $("qBars"), predicateChips: $("predicateChips"), indexChips: $("indexChips"),

    epsilonValue: $("epsilonValue"), epsilonBar: $("epsilonBar"),
    statUpdates: $("statUpdates"), statReplay: $("statReplay"),
    statLoss: $("statLoss"), statTd: $("statTd"),
    statEwc: $("statEwc"), statCf: $("statCf"),
    phaseName: $("phaseName"), phaseBadge: $("phaseBadge"),
    phaseBar: $("phaseBar"), phaseSteps: $("phaseSteps"), phaseCycle: $("phaseCycle"),

    actionMix: $("actionMix"), phaseBars: $("phaseBars"),
    recentBody: $("recentBody"), trafficChart: $("trafficChart"),

    qpsRange: $("qpsRange"), qpsOut: $("qpsOut"),
    phaseRange: $("phaseRange"), phaseOut: $("phaseOut"),
    learningToggle: $("learningToggle"), ewcToggle: $("ewcToggle"), cwilToggle: $("cwilToggle"),
  };

  let ws = null;
  let reconnectTimer = null;
  let pollTimer = null;
  let latest = null;

  const fmt = (value, digits = 0) => {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
    return Number(value).toLocaleString(undefined, {
      minimumFractionDigits: digits, maximumFractionDigits: digits,
    });
  };
  const pct = (value) => `${fmt(Number(value || 0) * 100, 1)}%`;
  const cls = (action) => String(action || "").toLowerCase();
  const esc = (text) => String(text ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));

  function clock(seconds) {
    const s = Math.max(0, Math.floor(seconds || 0));
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = s % 60;
    const mm = String(m).padStart(2, "0");
    const ss = String(sec).padStart(2, "0");
    return h > 0 ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
  }

  function setPill(pill, textEl, text, kind) {
    pill.className = `signal ${kind}`;
    textEl.textContent = text;
  }

  // ---------------------------------------------------------------- transport
  function connect() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${proto}//${location.host}/ws/live`);

    ws.onopen = () => {
      setPill(els.connPill, els.connText, "live", "live");
      stopPolling();
    };
    ws.onmessage = (event) => {
      try {
        const state = JSON.parse(event.data);
        if (state && state.type !== "error") {
          latest = state;
          render(state);
        }
      } catch (_) { /* ignore malformed frame */ }
    };
    ws.onclose = () => {
      setPill(els.connPill, els.connText, "reconnecting", "warn");
      startPolling();
      scheduleReconnect();
    };
    ws.onerror = () => { try { ws.close(); } catch (_) {} };
  }

  function scheduleReconnect() {
    if (reconnectTimer) return;
    reconnectTimer = setTimeout(() => {
      reconnectTimer = null;
      connect();
    }, 1500);
  }

  function startPolling() {
    if (pollTimer) return;
    pollTimer = setInterval(async () => {
      try {
        const response = await fetch("/api/state", { cache: "no-store" });
        const state = await response.json();
        latest = state;
        render(state);
        setPill(els.connPill, els.connText, "polling", "warn");
      } catch (_) { /* server offline */ }
    }, 1000);
  }

  function stopPolling() {
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
  }

  function control(action, config = {}) {
    const payload = { action, config };
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(payload));
    } else {
      fetch("/api/sim/control", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(payload),
      }).then((r) => r.json()).then((body) => {
        if (body && body.state) { latest = body.state; render(body.state); }
      }).catch(() => {});
    }
  }

  // ------------------------------------------------------------------ render
  function render(s) {
    if (!s) return;
    const status = s.status || {};
    const traffic = s.traffic || {};
    const agent = s.agent || {};
    const phase = s.phase || {};
    const cfg = s.config || {};

    if (status.error) {
      els.errorBanner.textContent = `simulation error · ${status.error}`;
      els.errorBanner.classList.remove("hidden");
    } else {
      els.errorBanner.classList.add("hidden");
    }

    if (status.running && status.paused) setPill(els.runPill, els.runText, "paused", "warn");
    else if (status.running) setPill(els.runPill, els.runText, "running", "live");
    else setPill(els.runPill, els.runText, "idle", "muted");

    els.clock.textContent = clock(status.uptime);
    els.startBtn.disabled = Boolean(status.running && !status.paused);
    els.pauseBtn.disabled = !status.running;
    els.pauseBtn.textContent = status.paused ? "Resume" : "Pause";
    els.stopBtn.disabled = !status.running;

    els.kpiQps.textContent = fmt(traffic.qps_actual, 1);
    els.kpiQpsTarget.textContent = fmt(traffic.qps_target, 0);
    els.kpiQueries.textContent = fmt(traffic.total_queries);
    els.kpiSteps.textContent = fmt(traffic.steps);
    els.kpiPhase.textContent = phase.name || "—";
    els.kpiPhaseSub.textContent = `cycle ${fmt(phase.cycle)} · ${fmt(phase.steps_in_phase)} / ${fmt(phase.steps_per_phase)}`;
    els.kpiEpsilon.textContent = fmt(agent.epsilon, 3);
    els.kpiReplay.textContent = fmt(agent.replay_size);
    els.kpiReplayCap.textContent = `/${fmt(agent.replay_capacity)}`;
    els.kpiUpdates.textContent = fmt(agent.updates);
    els.kpiDrift.textContent = fmt(agent.drift_events);
    els.kpiConsolidations.textContent = fmt(agent.consolidations);

    renderCurrent(s.current || {});
    renderLearning(agent, phase);
    renderActionMix(traffic);
    renderPhases(s.phases || []);
    renderRecent(s.recent || []);
    syncControls(cfg);
    drawChart(s.history || []);
  }

  function renderCurrent(cur) {
    els.currentQuery.textContent = cur.sql || "—";
    els.decisionHint.textContent = cur.sql
      ? `step ${fmt(cur.step)} · ${cur.phase_name || ""}`
      : "waiting for traffic";

    const action = cur.action || "KEEP";
    els.actionBadge.textContent = action;
    els.actionBadge.className = `action-badge ${cls(action)}`;
    els.exploreBadge.classList.toggle("hidden", !cur.exploring);

    els.curReward.textContent = fmt(cur.reward, 3);
    els.curLatency.textContent = fmt(cur.latency, 3);
    els.curIndexes.textContent = fmt(cur.n_indexes);

    const q = cur.q_values || {};
    const mask = cur.mask || {};
    const valid = ACTION_NAMES.filter((name) => mask[name] !== false);
    const values = valid.map((name) => Number(q[name])).filter((v) => Number.isFinite(v) && v > -1e8);
    const lo = values.length ? Math.min(...values) : 0;
    const hi = values.length ? Math.max(...values) : 1;
    const span = Math.max(1e-6, hi - lo);

    els.qBars.innerHTML = ACTION_NAMES.map((name) => {
      const masked = mask[name] === false || Number(q[name]) <= -1e8;
      const value = Number(q[name]);
      const norm = masked ? 0 : Math.max(0.06, Math.min(1, (value - lo) / span));
      const chosen = name === action;
      return `
        <div class="q-row ${chosen ? "chosen" : ""} ${masked ? "masked" : ""}">
          <span class="q-name">${name}</span>
          <div class="q-track"><div class="q-fill" style="width:${(norm * 100).toFixed(1)}%"></div></div>
          <span class="q-val">${masked ? "masked" : fmt(value, 3)}</span>
        </div>`;
    }).join("");

    const predicates = cur.predicate_columns || [];
    els.predicateChips.innerHTML = predicates.length
      ? predicates.map((c) => `<span class="chip">${esc(c)}</span>`).join("")
      : `<span class="chip">no predicate</span>`;

    const indexes = cur.active_indexes || [];
    els.indexChips.innerHTML = indexes.length
      ? indexes.map((cols) => `<span class="chip idx">idx(${esc(cols.join(", "))})</span>`).join("")
      : `<span class="chip">no indexes</span>`;
  }

  function renderLearning(agent, phase) {
    const epsilon = Number(agent.epsilon || 0);
    els.epsilonValue.textContent = fmt(epsilon, 3);
    els.epsilonBar.style.width = `${Math.max(0, Math.min(100, epsilon * 100))}%`;
    els.statUpdates.textContent = fmt(agent.updates);
    els.statReplay.textContent = fmt(agent.replay_size);
    els.statLoss.textContent = fmt(agent.loss, 3);
    els.statTd.textContent = fmt(agent.td, 3);
    els.statEwc.textContent = fmt(agent.ewc_penalty, 3);
    els.statCf.textContent = fmt(agent.cf_updates);

    els.phaseName.textContent = phase.name || "—";
    els.phaseBadge.classList.toggle("hidden", !phase.is_conflict);
    els.phaseBar.style.width = `${Math.max(0, Math.min(100, (phase.progress || 0) * 100))}%`;
    els.phaseSteps.textContent = `${fmt(phase.steps_in_phase)} / ${fmt(phase.steps_per_phase)} steps`;
    els.phaseCycle.textContent = `cycle ${fmt(phase.cycle)}`;
  }

  function renderActionMix(traffic) {
    const counts = traffic.action_counts || {};
    const rates = traffic.action_rates || {};
    const segments = ACTION_NAMES.map((name) => {
      const rate = Number(rates[name] || 0);
      return `<div class="mix-seg ${cls(name)}" style="width:${(rate * 100).toFixed(2)}%"></div>`;
    }).join("");
    const legend = ACTION_NAMES.map((name) => `
      <div class="mix-item">
        <span class="mix-swatch ${cls(name)}"></span>
        <span>${name}</span>
        <span class="mix-numbers">${fmt(counts[name])} · ${pct(rates[name])}</span>
        <span></span>
      </div>`).join("");
    els.actionMix.innerHTML = `<div class="mix-bar">${segments}</div><div class="mix-legend">${legend}</div>`;
  }

  function renderPhases(phases) {
    els.phaseBars.innerHTML = phases.map((p) => {
      const has = p.steps > 0 && p.mean_reward !== null && p.mean_reward !== undefined;
      const reward = has ? Number(p.mean_reward) : 0;
      const barPx = has ? Math.max(4, Math.round(((reward + 1.6) / 1.6) * 68)) : 3;
      const tone = p.is_conflict ? "conflict" : "selective";
      return `
        <div class="phase-cell ${p.is_conflict ? "conflict" : ""} ${has ? "" : "empty"}">
          <div class="phase-cell-top">
            <span>T${p.task}</span>
            <span class="phase-reward">${has ? fmt(reward, 2) : "—"}</span>
          </div>
          <div class="phase-column"><div class="phase-column-fill ${tone}" style="height:${barPx}px"></div></div>
          <div class="phase-cell-name" title="${esc(p.name)}">${esc(p.name)}</div>
          <div class="phase-cell-meta">${has ? `${fmt(p.steps)} steps` : "unvisited"}</div>
        </div>`;
    }).join("");
  }

  function renderRecent(recent) {
    const rows = recent.slice(-18).reverse();
    els.recentBody.innerHTML = rows.map((r) => {
      const q = r.q_values || {};
      const qv = (name) => (Number(q[name]) <= -1e8 ? "—" : fmt(q[name], 2));
      return `
        <div class="log-row ${cls(r.action)}">
          <span class="log-step">${fmt(r.step)}</span>
          <span class="log-phase">T${r.phase}</span>
          <span class="log-query" title="${esc(r.query)}">${esc(r.query)}</span>
          <span class="log-action">${esc(r.action)}</span>
          <span class="log-reward">${fmt(r.reward, 3)}</span>
          <span class="log-q">K ${qv("KEEP")} · C ${qv("CREATE")} · D ${qv("DROP")}</span>
        </div>`;
    }).join("");
  }

  function syncControls(cfg) {
    if (!cfg) return;
    if (document.activeElement !== els.qpsRange) els.qpsRange.value = Math.round(cfg.qps || 60);
    if (document.activeElement !== els.phaseRange) els.phaseRange.value = Math.round(cfg.steps_per_phase || 400);
    els.qpsOut.textContent = `${fmt(cfg.qps, 0)} q/s`;
    els.phaseOut.textContent = `${fmt(cfg.steps_per_phase, 0)} steps`;
    els.learningToggle.checked = Boolean(cfg.learning);
    els.ewcToggle.checked = Boolean(cfg.ewc);
    els.cwilToggle.checked = Boolean(cfg.cwil);
  }

  // ------------------------------------------------------------------- chart
  function drawChart(history) {
    const canvas = els.trafficChart;
    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(rect.width * dpr);
    canvas.height = Math.floor(rect.height * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);

    const pad = { left: 44, right: 12, top: 20, bottom: 22 };
    const width = rect.width - pad.left - pad.right;
    const height = rect.height - pad.top - pad.bottom;

    if (!history.length) {
      ctx.fillStyle = "rgba(127,145,173,0.8)";
      ctx.font = "12px ui-sans-serif, system-ui";
      ctx.textAlign = "center";
      ctx.fillText("waiting for traffic…", rect.width / 2, rect.height / 2);
      return;
    }

    // Downsample for a stable frame budget.
    const maxPoints = 420;
    const step = Math.max(1, Math.ceil(history.length / maxPoints));
    const data = history.filter((_, i) => i % step === 0 || i === history.length - 1);
    const ys = data.map((d) => Number(d.reward_ma ?? d.reward ?? 0));
    let lo = Math.min(-1.5, ...ys);
    let hi = Math.max(0.05, ...ys);
    const span = Math.max(0.2, hi - lo);
    lo -= span * 0.06;
    hi += span * 0.06;
    const x = (i) => pad.left + (i / Math.max(1, data.length - 1)) * width;
    const y = (v) => pad.top + (1 - (v - lo) / (hi - lo)) * height;

    // Phase bands.
    let bandStart = 0;
    for (let i = 1; i <= data.length; i++) {
      const changed = i === data.length || data[i].phase !== data[bandStart].phase;
      if (!changed) continue;
      if (i - bandStart >= 1) {
        ctx.fillStyle = data[bandStart].phase % 2 === 0
          ? "rgba(201,240,74,0.035)" : "rgba(157,180,208,0.028)";
        ctx.fillRect(x(bandStart), pad.top, Math.max(1, x(i - 1) - x(bandStart)), height);
        if (x(i - 1) - x(bandStart) > 42) {
          ctx.fillStyle = "rgba(139,141,136,0.9)";
          ctx.font = "8px ui-monospace, monospace";
          ctx.textAlign = "left";
          ctx.fillText(`T${data[bandStart].phase}`, x(bandStart) + 4, pad.top - 6);
        }
        if (i < data.length) {
          ctx.strokeStyle = "rgba(232,162,58,0.35)";
          ctx.setLineDash([2, 5]);
          ctx.beginPath();
          ctx.moveTo(x(i), pad.top);
          ctx.lineTo(x(i), pad.top + height);
          ctx.stroke();
          ctx.setLineDash([]);
        }
      }
      bandStart = i;
    }

    // Grid.
    ctx.font = "8px ui-monospace, monospace";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let g = 0; g <= 4; g++) {
      const value = lo + ((hi - lo) * g) / 4;
      const gy = y(value);
      ctx.strokeStyle = "rgba(231,227,216,0.07)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(pad.left, gy);
      ctx.lineTo(pad.left + width, gy);
      ctx.stroke();
      ctx.fillStyle = "rgba(139,141,136,0.9)";
      ctx.fillText(value.toFixed(2), pad.left - 7, gy);
    }

    // Raw reward, thin and translucent.
    ctx.beginPath();
    data.forEach((d, i) => {
      const px = x(i);
      const py = y(Math.max(lo, Math.min(hi, Number(d.reward ?? 0))));
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.strokeStyle = "rgba(157,180,208,0.35)";
    ctx.lineWidth = 1;
    ctx.stroke();

    // Smoothed reward with an area fill.
    const gradient = ctx.createLinearGradient(0, pad.top, 0, pad.top + height);
    gradient.addColorStop(0, "rgba(201,240,74,0.16)");
    gradient.addColorStop(1, "rgba(201,240,74,0.005)");
    ctx.beginPath();
    data.forEach((d, i) => {
      const px = x(i);
      const py = y(Math.max(lo, Math.min(hi, Number(d.reward_ma ?? d.reward ?? 0))));
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.lineTo(x(data.length - 1), pad.top + height);
    ctx.lineTo(x(0), pad.top + height);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();

    ctx.beginPath();
    data.forEach((d, i) => {
      const px = x(i);
      const py = y(Math.max(lo, Math.min(hi, Number(d.reward_ma ?? d.reward ?? 0))));
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.strokeStyle = "#c9f04a";
    ctx.lineWidth = 1.5;
    ctx.shadowColor = "rgba(201,240,74,0.35)";
    ctx.shadowBlur = 6;
    ctx.stroke();
    ctx.shadowBlur = 0;

    const last = data[data.length - 1];
    const lastY = y(Math.max(lo, Math.min(hi, Number(last.reward_ma ?? last.reward ?? 0))));
    ctx.beginPath();
    ctx.arc(x(data.length - 1), lastY, 2.6, 0, Math.PI * 2);
    ctx.fillStyle = "#c9f04a";
    ctx.fill();

    // Time axis.
    ctx.fillStyle = "rgba(139,141,136,0.9)";
    ctx.font = "8px ui-monospace, monospace";
    ctx.textBaseline = "alphabetic";
    ctx.textAlign = "left";
    ctx.fillText(`${fmt(data[0].t, 1)}s`, pad.left, rect.height - 6);
    ctx.textAlign = "right";
    ctx.fillText(`${fmt(last.t, 1)}s`, rect.width - pad.right, rect.height - 6);
  }

  // ----------------------------------------------------------------- controls
  els.startBtn.addEventListener("click", () => control("start"));
  els.pauseBtn.addEventListener("click", () => {
    if (!latest) return;
    control(latest.status && latest.status.paused ? "resume" : "pause");
  });
  els.stopBtn.addEventListener("click", () => control("stop"));
  els.resetBtn.addEventListener("click", () => control("reset"));

  els.qpsRange.addEventListener("input", () => {
    els.qpsOut.textContent = `${fmt(els.qpsRange.value)} q/s`;
  });
  els.qpsRange.addEventListener("change", () => {
    control("configure", { qps: Number(els.qpsRange.value) });
  });
  els.phaseRange.addEventListener("input", () => {
    els.phaseOut.textContent = `${fmt(els.phaseRange.value)} steps`;
  });
  els.phaseRange.addEventListener("change", () => {
    control("configure", { steps_per_phase: Number(els.phaseRange.value) });
  });
  els.learningToggle.addEventListener("change", () => {
    control("configure", { learning: els.learningToggle.checked });
  });
  els.ewcToggle.addEventListener("change", () => {
    control("configure", { ewc: els.ewcToggle.checked });
  });
  els.cwilToggle.addEventListener("change", () => {
    control("configure", { cwil: els.cwilToggle.checked });
  });

  window.addEventListener("resize", () => {
    if (latest) drawChart(latest.history || []);
  });

  setPill(els.connPill, els.connText, "connecting", "muted");
  setPill(els.runPill, els.runText, "idle", "muted");
  connect();
})();
