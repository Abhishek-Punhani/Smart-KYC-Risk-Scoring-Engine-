const BASE = "/api";

async function _get(url) {
  const res = await fetch(BASE + url);
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).error || `HTTP ${res.status}`);
  return res.json();
}

async function _post(url, body) {
  const res = await fetch(BASE + url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).error || `HTTP ${res.status}`);
  return res.json();
}

export const getHealth = () => _get("/health");
export const listRuns = () => _get("/runs");
export const getRun = (id) => _get(`/runs/${id}`);
export const getRunSummary = (id) => _get(`/runs/${id}/summary`);
export const getArtifacts = (id) => _get(`/runs/${id}/artifacts`);
export const getDashboard = (id) => _get(`/runs/${id}/dashboard`);
export const startRun = (config = {}) => _post("/runs", { config });
export const scoreCustomer = (payload) => _post("/score", payload);
