export interface RunParams {
  source: string;
  seed: number;
  grid: number;
  cpr_min: number;
  dop_max: number;
  fusion_k: number;
  target_prob: number;
  unc_max: number;
  battery_Wh: number;
  max_slope_deg: number;
  proximity_w: number;
  n_mc: number;
  upload_token?: string | null;
}

export const DEFAULTS: RunParams = {
  source: "synthetic",
  seed: 42,
  grid: 224,
  cpr_min: 1.0,
  dop_max: 0.13,
  fusion_k: 4.0,
  target_prob: 0.6,
  unc_max: 0.6,
  battery_Wh: 2000,
  max_slope_deg: 25,
  proximity_w: 0.3,
  n_mc: 800,
  upload_token: null,
};

export async function runPipeline(p: RunParams) {
  const r = await fetch("/api/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(p),
  });
  if (!r.ok) throw new Error((await r.json()).detail || "run failed");
  return r.json();
}

export async function getScatter(runId: string) {
  const r = await fetch(`/api/scatter?run_id=${runId}`);
  return r.json();
}

export function layerUrl(runId: string, name: string) {
  return `/api/layer?run_id=${runId}&name=${name}`;
}

export async function uploadFile(token: string, role: string, file: File) {
  const fd = new FormData();
  fd.append("token", token);
  fd.append("role", role);
  fd.append("file", file);
  const r = await fetch("/api/upload", { method: "POST", body: fd });
  return r.json();
}
