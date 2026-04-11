<script>
  import { onMount, onDestroy, tick } from "svelte";
  import {
    getHealth, startRun, getRun, getDashboard, scoreCustomer
  } from "./lib/api.js";

  // ── State ────────────────────────────────────────────────────
  let health = null;
  let runId = "";
  let run = null;
  let dash = null;         // full dashboard.json payload
  let loading = false;
  let err = "";
  let pollTimer = null;

  // customer table
  let search = "";
  let filterTier = "ALL";
  let filterDecision = "ALL";
  let page = 0;
  const PAGE_SIZE = 20;

  // single scorer
  let scorerOpen = false;
  let scorerResult = null;
  let scorerLoading = false;
  let scorerErr = "";
  const scorerDefaults = {
    occupation: "Salaried", age: 35, customer_tenure_years: 3,
    account_type: "Savings", country_risk: "Low", document_status: "Complete",
    address_verified: "Yes", monthly_txn_count: 10, annual_income: 600000,
    digital_risk_score: 20, fraud_history_flag: "No",
    pep_flag: "No", sanctions_flag: "No", adverse_media_flag: "No",
  };
  let scorerForm = { ...scorerDefaults };

  // ── Lifecycle ────────────────────────────────────────────────
  onMount(async () => {
    try { health = await getHealth(); } catch {}
  });
  onDestroy(() => clearInterval(pollTimer));

  // ── Run management ───────────────────────────────────────────
  async function triggerRun() {
    loading = true; err = ""; dash = null;
    try {
      const res = await startRun({ cv_folds: 5, threshold_target_recall: 0.95 });
      runId = res.run_id;
      run = { status: "queued", stage: "starting" };
      poll();
    } catch (e) { err = e.message; }
    finally { loading = false; }
  }

  function poll() {
    clearInterval(pollTimer);
    pollTimer = setInterval(async () => {
      try {
        run = await getRun(runId);
        if (run?.status === "completed") {
          clearInterval(pollTimer);
          dash = await getDashboard(runId);
          // Wait for Svelte to render {#if dash} canvases into the DOM,
          // then wait for Chart.js CDN script to be available.
          await tick();
          await waitForChartJS();
          drawAllCharts();
        } else if (run?.status === "failed") {
          clearInterval(pollTimer);
          err = run.error || "Pipeline failed";
        }
      } catch { clearInterval(pollTimer); }
    }, 2000);
  }

  // ── Wait for Chart.js CDN ─────────────────────────────────────
  function waitForChartJS() {
    return new Promise((resolve) => {
      const check = () => (window.Chart ? resolve() : setTimeout(check, 80));
      check();
    });
  }

  // ── Chart rendering (Chart.js via CDN) ───────────────────────
  let chartInstances = {};

  function destroyChart(id) {
    if (chartInstances[id]) { chartInstances[id].destroy(); delete chartInstances[id]; }
  }

  function mkChart(id, type, data, options = {}) {
    destroyChart(id);
    const el = document.getElementById(id);
    if (!el || !window.Chart) return;
    chartInstances[id] = new window.Chart(el, { type, data, options });
  }

  const TIER_COLORS   = { LOW: "#2ecc71", MEDIUM: "#f39c12", HIGH: "#e74c3c" };
  const DEC_COLORS    = { APPROVE: "#2ecc71", MANUAL_REVIEW: "#f39c12", EDD: "#e67e22", REJECT: "#e74c3c" };
  const darkPlugin    = { background: "#1a1d27" };

  const baseOpts = (title) => ({
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { labels: { color: "#eef" } },
      title: { display: !!title, text: title, color: "#eef", font: { size: 13, weight: "bold" } },
    },
    scales: {
      x: { ticks: { color: "#aaa" }, grid: { color: "#2b2f3c" } },
      y: { ticks: { color: "#aaa" }, grid: { color: "#2b2f3c" } },
    },
  });

  function drawAllCharts() {
    if (!dash || !window.Chart) return;
    const d = dash;

    // 1. Tier doughnut
    const tierKeys = Object.keys(d.tier_distribution);
    mkChart("c-tier", "doughnut", {
      labels: tierKeys,
      datasets: [{ data: tierKeys.map(k => d.tier_distribution[k]),
                   backgroundColor: tierKeys.map(k => TIER_COLORS[k] || "#555"),
                   borderWidth: 2, borderColor: "#0f1117" }],
    }, { ...baseOpts(null), cutout: "60%", scales: {} });

    // 2. Decision bar
    const decKeys = Object.keys(d.decision_distribution);
    mkChart("c-decision", "bar", {
      labels: decKeys,
      datasets: [{ label: "Count", data: decKeys.map(k => d.decision_distribution[k]),
                   backgroundColor: decKeys.map(k => DEC_COLORS[k] || "#888") }],
    }, baseOpts(null));

    // 3. Score histogram
    const histBins = d.score_histogram.LOW.map(b => b.x);
    mkChart("c-hist", "bar", {
      labels: histBins,
      datasets: ["LOW","MEDIUM","HIGH"].map(t => ({
        label: t,
        data: d.score_histogram[t].map(b => b.count),
        backgroundColor: TIER_COLORS[t] + "bb",
        stack: "hist",
      })),
    }, { ...baseOpts(null), scales: { ...baseOpts().scales, x: { stacked: true, ticks:{color:"#aaa"}, grid:{color:"#2b2f3c"} }, y: { stacked: true, ticks:{color:"#aaa"}, grid:{color:"#2b2f3c"} } } });

    // 4. Model comparison
    const models = d.model_comparison;
    mkChart("c-models", "bar", {
      labels: models.map(m => m.name),
      datasets: [{
        label: "CV F1-Macro",
        data: models.map(m => m.f1_mean),
        backgroundColor: models.map(m => m.is_best ? "#e74c3c" : "#3498db"),
        borderRadius: 4,
      }],
    }, { ...baseOpts(null), scales: { ...baseOpts().scales, y: { min: 0.8, max: 1.0, ticks:{color:"#aaa"}, grid:{color:"#2b2f3c"} }, x: { ticks:{color:"#aaa", maxRotation:20}, grid:{color:"#2b2f3c"} } } });

    // 5. Feature importance
    const fi = d.feature_importance.slice(0, 12);
    mkChart("c-fi", "bar", {
      labels: fi.map(f => f.feature).reverse(),
      datasets: [{ label: "Importance", data: fi.map(f => f.importance).reverse(),
                   backgroundColor: fi.map(f => ["sanctions","fraud","pep"].some(k=>f.feature.includes(k)) ? "#e74c3c" : ["txn","struct","digital"].some(k=>f.feature.includes(k)) ? "#f39c12" : "#3498db").reverse(),
                   borderRadius: 3 }],
    }, { ...baseOpts(null), indexAxis: "y", scales: { ...baseOpts().scales } });

    // 6. Filter scores grouped bar (F1-F5 by tier)
    const flabels = d.filter_labels;
    mkChart("c-filters", "bar", {
      labels: flabels,
      datasets: ["LOW","MEDIUM","HIGH"].map(t => ({
        label: t,
        data: d.filter_scores[t] || [],
        backgroundColor: TIER_COLORS[t] + "cc",
        borderRadius: 3,
      })),
    }, baseOpts(null));

    // 7. Threshold curve
    const tc = d.threshold_curve;
    mkChart("c-thresh", "line", {
      labels: tc.map(p => p.threshold),
      datasets: [
        { label: "Recall (HIGH)", data: tc.map(p => p.recall), borderColor: "#e74c3c", tension: 0.3, pointRadius: 2, fill: false },
        { label: "Precision (HIGH)", data: tc.map(p => p.precision), borderColor: "#3498db", tension: 0.3, pointRadius: 2, fill: false },
      ],
    }, {
      ...baseOpts(null),
      plugins: {
        ...baseOpts(null).plugins,
        annotation: undefined,
      },
    });

    // 8. Occupation avg risk
    const occ = d.occupation_risk;
    mkChart("c-occ", "bar", {
      labels: occ.map(o => o.occupation),
      datasets: [{ label: "Avg Risk Score", data: occ.map(o => o.avg_risk),
                   backgroundColor: occ.map(o => o.avg_risk > 35 ? "#e74c3c" : o.avg_risk > 25 ? "#f39c12" : "#2ecc71"),
                   borderRadius: 4 }],
    }, { ...baseOpts(null), indexAxis: "y" });
  }



  // ── Customer table ───────────────────────────────────────────
  $: customers = (dash?.customers || []).filter(c => {
    const q = search.toLowerCase();
    const matchQ = !q || String(c.customer_id).includes(q) || (c.top_risk_factors||"").toLowerCase().includes(q);
    const matchT = filterTier === "ALL" || c.risk_tier === filterTier;
    const matchD = filterDecision === "ALL" || c.decision === filterDecision;
    return matchQ && matchT && matchD;
  });
  $: totalPages = Math.ceil(customers.length / PAGE_SIZE);
  $: pageCustomers = customers.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);
  $: { search; filterTier; filterDecision; page = 0; }

  // ── Single scorer ────────────────────────────────────────────
  async function doScore() {
    scorerLoading = true; scorerErr = ""; scorerResult = null;
    try {
      scorerResult = await scoreCustomer(scorerForm);
    } catch (e) { scorerErr = e.message; }
    finally { scorerLoading = false; }
  }

  const decColor = (d) => DEC_COLORS[d] || "#888";
  const tierColor = (t) => TIER_COLORS[t] || "#888";
  const fmt = (v, d=2) => v == null ? "N/A" : typeof v === "number" ? v.toFixed(d) : v;
</script>

<svelte:head>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet" />
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
  <title>KYC Risk Dashboard</title>
</svelte:head>

<div class="shell">
  <!-- ── HEADER ─────────────────────────────────────────── -->
  <header>
    <div class="header-left">
      <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
        <circle cx="14" cy="14" r="13" stroke="#e74c3c" stroke-width="2"/>
        <path d="M8 14l4 4 8-8" stroke="#e74c3c" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
      <span class="logo-text">KYC <span class="logo-accent">Risk Engine</span></span>
    </div>
    <div class="header-right">
      {#if health}
        <span class="pill green">● Backend OK</span>
      {:else}
        <span class="pill red">● Backend Down</span>
      {/if}
      <button class="btn-primary" on:click={triggerRun} disabled={loading || (run && run.status === 'running')}>
        {#if loading}⏳ Starting…{:else if run?.status === 'running'}⚙️ Running…{:else}🚀 Run Pipeline{/if}
      </button>
    </div>
  </header>

  <!-- ── RUN STATUS STRIP ───────────────────────────────── -->
  {#if run}
    <div class="status-strip" class:done={run.status==='completed'} class:fail={run.status==='failed'}>
      <span>Run <code>{runId}</code></span>
      <span class="stage">{run.stage || '—'}</span>
      <span class="badge" class:green={run.status==='completed'} class:orange={run.status==='running'} class:red={run.status==='failed'}>
        {run.status}
      </span>
      {#if run.status === 'running'}
        <span class="spinner"></span>
      {/if}
    </div>
  {/if}

  {#if err}
    <div class="alert-error">⚠️ {err}</div>
  {/if}

  <!-- ── KPI CARDS ──────────────────────────────────────── -->
  {#if dash}
    {@const k = dash.kpis}
    <section class="kpi-row">
      <div class="kpi"><div class="kpi-v">{k.total}</div><div class="kpi-l">Total Customers</div></div>
      <div class="kpi green"><div class="kpi-v">{k.approve}</div><div class="kpi-l">Approved (SDD)</div></div>
      <div class="kpi orange"><div class="kpi-v">{k.manual_review}</div><div class="kpi-l">Manual Review</div></div>
      <div class="kpi amber"><div class="kpi-v">{k.edd}</div><div class="kpi-l">EDD Required</div></div>
      <div class="kpi red"><div class="kpi-v">{k.reject}</div><div class="kpi-l">Rejected</div></div>
      <div class="kpi red"><div class="kpi-v">{k.sanctions_hits}</div><div class="kpi-l">Sanctions Hits</div></div>
      <div class="kpi orange"><div class="kpi-v">{k.pep_customers}</div><div class="kpi-l">PEP Customers</div></div>
      <div class="kpi orange"><div class="kpi-v">{k.structuring_alerts}</div><div class="kpi-l">Structuring Alerts</div></div>
      <div class="kpi blue"><div class="kpi-v kpi-sm">{k.best_model}</div><div class="kpi-l">Best Model</div></div>
      <div class="kpi blue"><div class="kpi-v">{fmt(k.roc_auc, 4)}</div><div class="kpi-l">ROC-AUC (OvR)</div></div>
      <div class="kpi blue"><div class="kpi-v">{fmt(k.cv_f1_macro, 4)}</div><div class="kpi-l">CV F1-Macro</div></div>
      <div class="kpi blue"><div class="kpi-v">{fmt(k.threshold, 2)}</div><div class="kpi-l">Threshold</div></div>
    </section>

    <!-- ── CHARTS ROW 1 ──────────────────────────────────── -->
    <section class="charts-grid">
      <div class="chart-card">
        <div class="chart-title">Risk Tier Distribution</div>
        <div class="chart-wrap"><canvas id="c-tier"></canvas></div>
      </div>
      <div class="chart-card">
        <div class="chart-title">Onboarding Decisions</div>
        <div class="chart-wrap"><canvas id="c-decision"></canvas></div>
      </div>
      <div class="chart-card">
        <div class="chart-title">Risk Score Distribution by Tier</div>
        <div class="chart-wrap"><canvas id="c-hist"></canvas></div>
      </div>
    </section>

    <!-- ── CHARTS ROW 2 ──────────────────────────────────── -->
    <section class="charts-grid">
      <div class="chart-card">
        <div class="chart-title">Model Comparison — CV F1-Macro</div>
        <div class="chart-wrap"><canvas id="c-models"></canvas></div>
      </div>
      <div class="chart-card">
        <div class="chart-title">Feature Importance — SHAP/Proxy (HIGH)</div>
        <div class="chart-wrap"><canvas id="c-fi"></canvas></div>
      </div>
      <div class="chart-card">
        <div class="chart-title">Avg Filter Scores (F1–F5) by Tier</div>
        <div class="chart-wrap"><canvas id="c-filters"></canvas></div>
      </div>
    </section>

    <!-- ── CHARTS ROW 3 ──────────────────────────────────── -->
    <section class="charts-grid">
      <div class="chart-card">
        <div class="chart-title">HIGH-Tier Recall &amp; Precision vs Threshold</div>
        <div class="chart-wrap"><canvas id="c-thresh"></canvas></div>
      </div>
      <div class="chart-card">
        <div class="chart-title">Avg Risk Score by Occupation</div>
        <div class="chart-wrap"><canvas id="c-occ"></canvas></div>
      </div>
      <!-- Confusion Matrix -->
      <div class="chart-card">
        <div class="chart-title">Confusion Matrix — {dash.kpis.best_model}</div>
        <div class="cm-wrap">
          {#if dash.confusion_matrix}
            {@const cm = dash.confusion_matrix}
            {@const labels = cm.labels}
            <table class="cm-table">
              <thead>
                <tr><th></th>{#each labels as l}<th class="cm-head">{l}</th>{/each}</tr>
              </thead>
              <tbody>
                {#each cm.matrix as row, i}
                  <tr>
                    <td class="cm-side">{labels[i]}</td>
                    {#each row as cell}
                      {@const maxVal = Math.max(...cm.matrix.flat(), 1)}
                      <td class="cm-cell" style="opacity:{0.3 + cell/maxVal*0.7}">{cell}</td>
                    {/each}
                  </tr>
                {/each}
              </tbody>
            </table>
          {/if}
        </div>
      </div>

    </section>

    <!-- ── CUSTOMER TABLE ────────────────────────────────── -->
    <section class="card">
      <div class="table-toolbar">
        <h3 class="section-title">Customer Records</h3>
        <div class="toolbar-filters">
          <input class="search-input" placeholder="Search ID / factors…" bind:value={search} />
          <select class="sel" bind:value={filterTier}>
            <option value="ALL">All Tiers</option>
            <option>LOW</option><option>MEDIUM</option><option>HIGH</option>
          </select>
          <select class="sel" bind:value={filterDecision}>
            <option value="ALL">All Decisions</option>
            <option>APPROVE</option><option>MANUAL_REVIEW</option><option>EDD</option><option>REJECT</option>
          </select>
          <span class="count-badge">{customers.length} rows</span>
        </div>
      </div>

      <div class="table-scroll">
        <table class="data-table">
          <thead>
            <tr>
              <th>Customer ID</th><th>Risk Score</th><th>Tier</th><th>Decision</th>
              <th>Sanctions</th><th>PEP</th><th>Fraud</th><th>Country Risk</th><th>Top Risk Factors</th>
            </tr>
          </thead>
          <tbody>
            {#each pageCustomers as c}
              <tr>
                <td class="mono">{c.customer_id}</td>
                <td>
                  <div class="score-bar-wrap">
                    <span class="score-num">{fmt(c.risk_score, 1)}</span>
                    <div class="score-bar"><div class="score-fill" style="width:{c.risk_score}%;background:{c.risk_score>60?'#e74c3c':c.risk_score>30?'#f39c12':'#2ecc71'}"></div></div>
                  </div>
                </td>
                <td><span class="badge-pill" style="background:{tierColor(c.risk_tier)}22;color:{tierColor(c.risk_tier)};border:1px solid {tierColor(c.risk_tier)}55">{c.risk_tier}</span></td>
                <td><span class="badge-pill" style="background:{decColor(c.decision)}22;color:{decColor(c.decision)};border:1px solid {decColor(c.decision)}55">{c.decision}</span></td>
                <td class:flag-on={c.sanctions_flag}>{c.sanctions_flag ? "🔴" : "—"}</td>
                <td class:flag-on={c.pep_flag}>{c.pep_flag ? "🟡" : "—"}</td>
                <td class:flag-on={c.fraud_history_flag}>{c.fraud_history_flag ? "🔴" : "—"}</td>
                <td><span class:high-risk={c.country_risk==='High'}>{c.country_risk || "—"}</span></td>
                <td class="factors">{c.top_risk_factors || "—"}</td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>

      <div class="pagination">
        <button on:click={() => page--} disabled={page===0}>◀ Prev</button>
        <span>Page {page+1} of {totalPages || 1}</span>
        <button on:click={() => page++} disabled={page>=totalPages-1}>Next ▶</button>
      </div>
    </section>

    <!-- ── SINGLE CUSTOMER SCORER ────────────────────────── -->
    <section class="card">
      <div class="scorer-header" on:click={() => scorerOpen = !scorerOpen}>
        <h3 class="section-title">🔍 Single Customer Scorer</h3>
        <span class="toggle">{scorerOpen ? "▲" : "▼"}</span>
      </div>
      {#if scorerOpen}
        <div class="scorer-grid">
          <label>Occupation
            <select bind:value={scorerForm.occupation}>
              {#each ["Salaried","Student","Self Employed","Business","Cash Business"] as o}<option>{o}</option>{/each}
            </select>
          </label>
          <label>Age <input type="number" bind:value={scorerForm.age} min=18 max=90 /></label>
          <label>Tenure (years) <input type="number" bind:value={scorerForm.customer_tenure_years} min=0 max=40 /></label>
          <label>Account Type
            <select bind:value={scorerForm.account_type}>
              {#each ["Savings","Current","NRI","Corporate"] as a}<option>{a}</option>{/each}
            </select>
          </label>
          <label>Country Risk
            <select bind:value={scorerForm.country_risk}>
              {#each ["Low","Medium","High"] as r}<option>{r}</option>{/each}
            </select>
          </label>
          <label>Document Status
            <select bind:value={scorerForm.document_status}>
              {#each ["Complete","Partial","Missing"] as d}<option>{d}</option>{/each}
            </select>
          </label>
          <label>Address Verified
            <select bind:value={scorerForm.address_verified}>
              <option>Yes</option><option>No</option>
            </select>
          </label>
          <label>Monthly Txn Count <input type="number" bind:value={scorerForm.monthly_txn_count} min=0 /></label>
          <label>Annual Income (₹) <input type="number" bind:value={scorerForm.annual_income} min=0 /></label>
          <label>Digital Risk Score <input type="number" bind:value={scorerForm.digital_risk_score} min=0 max=100 /></label>
          <label>Fraud History
            <select bind:value={scorerForm.fraud_history_flag}>
              <option>No</option><option>Yes</option>
            </select>
          </label>
          <label>PEP
            <select bind:value={scorerForm.pep_flag}>
              <option>No</option><option>Yes</option>
            </select>
          </label>
          <label>Sanctions
            <select bind:value={scorerForm.sanctions_flag}>
              <option>No</option><option>Yes</option>
            </select>
          </label>
          <label>Adverse Media
            <select bind:value={scorerForm.adverse_media_flag}>
              <option>No</option><option>Yes</option>
            </select>
          </label>
        </div>
        <button class="btn-primary score-btn" on:click={doScore} disabled={scorerLoading}>
          {scorerLoading ? "Scoring…" : "Score Customer"}
        </button>
        {#if scorerErr}<div class="alert-error">{scorerErr}</div>{/if}
        {#if scorerResult}
          {@const sr = scorerResult}
          <div class="score-result">
            <div class="sr-row"><span class="sr-label">Risk Score</span>
              <span class="sr-val" style="color:{sr.risk_score>60?'#e74c3c':sr.risk_score>30?'#f39c12':'#2ecc71'}">{fmt(sr.risk_score, 2)}</span></div>
            <div class="sr-row"><span class="sr-label">Risk Tier</span>
              <span class="sr-val" style="color:{tierColor(sr.risk_tier)}">{sr.risk_tier}</span></div>
            <div class="sr-row"><span class="sr-label">Decision</span>
              <span class="sr-val" style="color:{decColor(sr.decision)}">{sr.decision}</span></div>
            <div class="sr-row"><span class="sr-label">Top Factors</span>
              <span class="sr-facts">{sr.top_risk_factors}</span></div>
            {#if sr.signals}
              <div class="sr-signals">
                {#each Object.entries(sr.signals) as [k,v]}
                  <div class="sig"><div class="sig-name">{k}</div>
                    <div class="sig-bar"><div class="sig-fill" style="width:{v}%;background:{v>60?'#e74c3c':v>30?'#f39c12':'#2ecc71'}"></div></div>
                    <div class="sig-val">{fmt(v,1)}</div>
                  </div>
                {/each}
              </div>
            {/if}
          </div>
        {/if}
      {/if}
    </section>
  {:else if !run}
    <!-- Empty state -->
    <div class="empty-state">
      <div class="empty-icon">🛡️</div>
      <h2>Smart KYC Risk Scoring Engine</h2>
      <p>FATF RBA Framework · Parida &amp; Kumar 5-Filter Model · Leak-Free ML Validation</p>
      <button class="btn-primary btn-lg" on:click={triggerRun} disabled={loading}>🚀 Start Pipeline Run</button>
    </div>
  {:else if run?.status !== 'completed'}
    <div class="running-state">
      <div class="run-spinner"></div>
      <div class="run-label">Running pipeline… <strong>{run?.stage || 'initializing'}</strong></div>
      <p class="run-sub">This typically takes 2–5 minutes for 9 models + SHAP analysis</p>
    </div>
  {/if}
</div>

<style>
  :global(*) { box-sizing: border-box; margin: 0; padding: 0; }
  :global(body) { font-family: "Inter", system-ui, sans-serif; background: #0f1117; color: #eef2ff; min-height: 100vh; }

  .shell { max-width: 1400px; margin: 0 auto; padding: 0 20px 40px; }

  /* ── HEADER ─── */
  header { display: flex; justify-content: space-between; align-items: center; padding: 18px 0; border-bottom: 1px solid #2b2f3c; margin-bottom: 20px; }
  .header-left { display: flex; align-items: center; gap: 12px; }
  .logo-text { font-size: 1.3rem; font-weight: 700; letter-spacing: -0.5px; }
  .logo-accent { color: #e74c3c; }
  .header-right { display: flex; align-items: center; gap: 12px; }
  .pill { font-size: .75rem; padding: 4px 10px; border-radius: 999px; font-weight: 600; }
  .pill.green { background: #2ecc7122; color: #2ecc71; border: 1px solid #2ecc7144; }
  .pill.red   { background: #e74c3c22; color: #e74c3c; border: 1px solid #e74c3c44; }

  /* ── BUTTONS ─── */
  .btn-primary { background: linear-gradient(135deg,#2563eb,#1d4ed8); color: #fff; border: none; border-radius: 8px; padding: 9px 18px; font-size: .875rem; font-weight: 600; cursor: pointer; transition: opacity .2s; }
  .btn-primary:hover { opacity: .9; }
  .btn-primary:disabled { opacity: .5; cursor: not-allowed; }
  .btn-lg { padding: 14px 32px; font-size: 1rem; }

  /* ── STATUS STRIP ─── */
  .status-strip { display: flex; align-items: center; gap: 14px; background: #1a1d27; border: 1px solid #2b2f3c; border-radius: 10px; padding: 10px 16px; margin-bottom: 16px; font-size: .85rem; }
  .status-strip code { font-size: .78rem; color: #facc15; background: #facc1522; padding: 2px 7px; border-radius: 4px; }
  .stage { color: #aaa; }
  .badge { padding: 2px 10px; border-radius: 999px; font-weight: 600; font-size:.78rem; }
  .badge.green { background:#2ecc7122; color:#2ecc71; }
  .badge.orange { background:#f39c1222; color:#f39c12; }
  .badge.red { background:#e74c3c22; color:#e74c3c; }

  /* ── SPINNER ─── */
  .spinner { width: 16px; height: 16px; border: 2px solid #2b2f3c; border-top-color: #3498db; border-radius: 50%; animation: spin 0.8s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ── ALERT ─── */
  .alert-error { background: #e74c3c22; border: 1px solid #e74c3c55; color: #fca5a5; padding: 12px 16px; border-radius: 8px; margin: 12px 0; font-size: .875rem; }

  /* ── KPI ROW ─── */
  .kpi-row { display: grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); gap: 12px; margin-bottom: 24px; }
  .kpi { background: #1a1d27; border: 1px solid #2b2f3c; border-radius: 12px; padding: 16px 14px; text-align: center; transition: transform .15s; }
  .kpi:hover { transform: translateY(-2px); }
  .kpi.green { border-color: #2ecc7144; }
  .kpi.orange, .kpi.amber { border-color: #f39c1244; }
  .kpi.red { border-color: #e74c3c44; }
  .kpi.blue { border-color: #3498db44; }
  .kpi-v { font-size: 1.6rem; font-weight: 700; line-height: 1.2; }
  .kpi-sm { font-size: 1rem; }
  .kpi-l { font-size: .7rem; color: #888; margin-top: 4px; text-transform: uppercase; letter-spacing: .5px; }
  .kpi.green .kpi-v { color: #2ecc71; }
  .kpi.orange .kpi-v, .kpi.amber .kpi-v { color: #f39c12; }
  .kpi.red .kpi-v { color: #e74c3c; }
  .kpi.blue .kpi-v { color: #60a5fa; }

  /* ── CHARTS ─── */
  .charts-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 20px; }
  .chart-card { background: #1a1d27; border: 1px solid #2b2f3c; border-radius: 12px; padding: 16px; }
  .chart-title { font-size: .8rem; font-weight: 600; color: #aaa; text-transform: uppercase; letter-spacing: .5px; margin-bottom: 10px; }
  .chart-wrap { position: relative; height: 220px; }

  /* ── CONFUSION MATRIX ─── */
  .cm-wrap { display: flex; justify-content: center; align-items: center; height: 220px; }
  .cm-table { border-collapse: separate; border-spacing: 4px; }
  .cm-head { text-align: center; font-size: .75rem; color: #aaa; padding: 4px 10px; }
  .cm-side { font-size: .75rem; color: #aaa; padding-right: 8px; text-align: right; }
  .cm-cell { text-align: center; font-size: 1.1rem; font-weight: 700; padding: 14px 18px; border-radius: 6px; color: #eef; min-width: 60px; }

  /* ── CARD ─── */
  .card { background: #1a1d27; border: 1px solid #2b2f3c; border-radius: 12px; padding: 20px; margin-bottom: 20px; }
  .section-title { font-size: 1rem; font-weight: 700; margin-bottom: 14px; }

  /* ── TABLE ─── */
  .table-toolbar { display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px; margin-bottom: 14px; }
  .toolbar-filters { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
  .search-input { background: #272b3a; border: 1px solid #2b2f3c; border-radius: 8px; padding: 7px 12px; color: #eef; font-size: .84rem; width: 220px; }
  .search-input:focus { outline: none; border-color: #3498db; }
  .sel { background: #272b3a; border: 1px solid #2b2f3c; border-radius: 8px; padding: 7px 10px; color: #eef; font-size: .84rem; cursor: pointer; }
  .count-badge { font-size: .75rem; color: #888; background: #272b3a; padding: 4px 10px; border-radius: 999px; border: 1px solid #2b2f3c; }
  .table-scroll { overflow-x: auto; }
  .data-table { width: 100%; border-collapse: collapse; font-size: .82rem; }
  .data-table th { text-align: left; padding: 10px 12px; font-size: .72rem; color: #888; text-transform: uppercase; letter-spacing: .5px; border-bottom: 1px solid #2b2f3c; }
  .data-table td { padding: 9px 12px; border-bottom: 1px solid #1f2230; vertical-align: middle; }
  .data-table tr:hover td { background: #1f2234; }
  .mono { font-family: monospace; color: #facc15; }
  .factors { max-width: 280px; font-size: .78rem; color: #aaa; }
  .high-risk { color: #e74c3c; font-weight: 600; }
  .flag-on { color: #e74c3c; }
  .badge-pill { font-size: .72rem; font-weight: 600; padding: 3px 9px; border-radius: 999px; white-space: nowrap; }
  .score-bar-wrap { display: flex; align-items: center; gap: 8px; min-width: 100px; }
  .score-num { font-weight: 600; width: 32px; }
  .score-bar { flex: 1; height: 6px; background: #2b2f3c; border-radius: 3px; overflow: hidden; }
  .score-fill { height: 100%; border-radius: 3px; transition: width .3s; }
  .pagination { display: flex; align-items: center; gap: 14px; justify-content: center; margin-top: 16px; font-size: .85rem; }
  .pagination button { background: #272b3a; border: 1px solid #2b2f3c; color: #eef; padding: 6px 14px; border-radius: 6px; cursor: pointer; }
  .pagination button:disabled { opacity: .4; cursor: not-allowed; }

  /* ── SCORER ─── */
  .scorer-header { display: flex; justify-content: space-between; align-items: center; cursor: pointer; }
  .toggle { font-size: .9rem; color: #888; }
  .scorer-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 14px; margin-top: 16px; }
  .scorer-grid label { display: flex; flex-direction: column; gap: 6px; font-size: .8rem; color: #aaa; }
  .scorer-grid input, .scorer-grid select { background: #272b3a; border: 1px solid #2b2f3c; border-radius: 7px; padding: 7px 10px; color: #eef; font-size: .85rem; }
  .score-btn { margin-top: 18px; }
  .score-result { background: #131621; border: 1px solid #2b2f3c; border-radius: 10px; padding: 18px; margin-top: 16px; }
  .sr-row { display: flex; align-items: baseline; gap: 12px; margin-bottom: 10px; }
  .sr-label { font-size: .75rem; color: #888; text-transform: uppercase; width: 90px; }
  .sr-val { font-size: 1.3rem; font-weight: 700; }
  .sr-facts { font-size: .85rem; color: #ccc; }
  .sr-signals { margin-top: 14px; display: flex; flex-direction: column; gap: 8px; }
  .sig { display: flex; align-items: center; gap: 10px; font-size: .8rem; }
  .sig-name { width: 120px; color: #888; }
  .sig-bar { flex: 1; height: 6px; background: #2b2f3c; border-radius: 3px; overflow: hidden; }
  .sig-fill { height: 100%; border-radius: 3px; }
  .sig-val { width: 36px; text-align: right; font-weight: 600; }

  /* ── EMPTY / RUNNING STATE ─── */
  .empty-state { text-align: center; padding: 100px 20px; }
  .empty-icon { font-size: 4rem; margin-bottom: 20px; }
  .empty-state h2 { font-size: 1.8rem; margin-bottom: 10px; }
  .empty-state p { color: #888; margin-bottom: 28px; max-width: 480px; margin-left: auto; margin-right: auto; }
  .running-state { text-align: center; padding: 80px 20px; }
  .run-spinner { width: 56px; height: 56px; border: 4px solid #2b2f3c; border-top-color: #3498db; border-radius: 50%; margin: 0 auto 24px; animation: spin .9s linear infinite; }
  .run-label { font-size: 1.1rem; font-weight: 600; }
  .run-sub { color: #888; margin-top: 8px; font-size: .85rem; }

  @media (max-width: 1100px) { .charts-grid { grid-template-columns: repeat(2,1fr); } }
  @media (max-width: 720px)  { .charts-grid { grid-template-columns: 1fr; } .kpi-row { grid-template-columns: repeat(3,1fr); } }
</style>
