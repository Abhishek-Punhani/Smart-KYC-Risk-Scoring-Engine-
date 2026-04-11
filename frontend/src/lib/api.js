export async function getHealth() {
  const res = await fetch("/api/health");
  if (!res.ok) throw new Error("Health check failed");
  return res.json();
}

export async function startRun(config = {}) {
  const res = await fetch("/api/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ config }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || "Failed to start run");
  }
  return res.json();
}

export async function getRun(runId) {
  const res = await fetch(`/api/runs/${runId}`);
  if (!res.ok) throw new Error("Failed to fetch run");
  return res.json();
}

export async function getRunSummary(runId) {
  const res = await fetch(`/api/runs/${runId}/summary`);
  if (!res.ok) throw new Error("Failed to fetch summary");
  return res.json();
}

export async function getArtifacts(runId) {
  const res = await fetch(`/api/runs/${runId}/artifacts`);
  if (!res.ok) throw new Error("Failed to fetch artifacts");
  return res.json();
}
