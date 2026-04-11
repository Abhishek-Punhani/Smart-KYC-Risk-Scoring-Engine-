<script>
  import { getHealth, startRun, getRun, getRunSummary, getArtifacts } from './lib/api'

  let health = null
  let runId = ''
  let run = null
  let summary = null
  let artifacts = []
  let loading = false
  let err = ''
  let pollTimer

  async function checkHealth() {
    err = ''
    try {
      health = await getHealth()
    } catch (e) {
      err = e.message
    }
  }

  async function triggerRun() {
    loading = true
    err = ''
    summary = null
    artifacts = []
    try {
      const res = await startRun({ cv_folds: 5, threshold_target_recall: 0.95 })
      runId = res.run_id
      await refreshRun()
      poll()
    } catch (e) {
      err = e.message
    } finally {
      loading = false
    }
  }

  async function refreshRun() {
    if (!runId) return
    run = await getRun(runId)
    if (run.status === 'completed') {
      const s = await getRunSummary(runId)
      summary = s.summary
      const a = await getArtifacts(runId)
      artifacts = a.files
    }
  }

  function poll() {
    clearInterval(pollTimer)
    pollTimer = setInterval(async () => {
      try {
        await refreshRun()
        if (run?.status === 'completed' || run?.status === 'failed') {
          clearInterval(pollTimer)
        }
      } catch {
        clearInterval(pollTimer)
      }
    }, 2000)
  }

  checkHealth()
</script>

<main>
  <h1>KYC Risk Dashboard (Flask + Svelte)</h1>

  <section class="card">
    <h2>Backend Health</h2>
    <button on:click={checkHealth}>Check Health</button>
    {#if health}
      <p>Status: <b>{health.status}</b></p>
      <p>Service: <b>{health.service}</b></p>
    {/if}
  </section>

  <section class="card">
    <h2>Run Full Pipeline</h2>
    <button disabled={loading} on:click={triggerRun}>
      {loading ? 'Starting...' : 'Start Run'}
    </button>
    {#if runId}
      <p>Run ID: <code>{runId}</code></p>
    {/if}
    {#if run}
      <p>Status: <b>{run.status}</b> | Stage: <b>{run.stage}</b></p>
    {/if}
  </section>

  {#if summary}
    <section class="card">
      <h2>Summary</h2>
      <ul>
        <li>Best Model: <b>{summary.best_model}</b></li>
        <li>CV F1 Macro: <b>{summary.cv_f1_macro?.toFixed(4)}</b></li>
        <li>HIGH Recall (Holdout): <b>{summary.holdout_high_recall?.toFixed(4)}</b></li>
        <li>ROC-AUC OvR: <b>{summary.roc_auc_ovr_macro?.toFixed?.(4) ?? summary.roc_auc_ovr_macro}</b></li>
      </ul>
    </section>
  {/if}

  {#if artifacts.length}
    <section class="card">
      <h2>Artifacts</h2>
      <ul>
        {#each artifacts as file}
          <li>
            <a href={file.url} target="_blank" rel="noreferrer">{file.name}</a>
            <small> ({Math.round(file.size_bytes / 1024)} KB)</small>
          </li>
        {/each}
      </ul>
    </section>
  {/if}

  {#if err}
    <section class="card error">
      <b>Error:</b> {err}
    </section>
  {/if}
</main>

<style>
  :global(body) {
    margin: 0;
    font-family: Inter, system-ui, sans-serif;
    background: #0f1117;
    color: #eef2ff;
  }
  main {
    max-width: 900px;
    margin: 0 auto;
    padding: 24px;
  }
  h1 {
    margin-top: 0;
  }
  .card {
    background: #1a1d27;
    border: 1px solid #2b2f3c;
    border-radius: 12px;
    padding: 16px;
    margin: 12px 0;
  }
  .error {
    border-color: #a33;
    color: #ffd7d7;
  }
  button {
    background: #2563eb;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 8px 12px;
    cursor: pointer;
  }
  button:disabled {
    opacity: 0.6;
  }
  a {
    color: #93c5fd;
  }
  code {
    color: #facc15;
  }
</style>
