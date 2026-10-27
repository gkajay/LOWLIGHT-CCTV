async function request(path, options) {
  let res;
  try {
    res = await fetch(path, options);
  } catch {
    throw new Error("Cannot reach the backend. Start it with: uvicorn app.main:app --port 8000");
  }
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (body.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* keep default message */
    }
    throw new Error(detail);
  }
  return res.json();
}

export const getHealth = () => request("/api/health");
export const reloadModels = () => request("/api/reload", { method: "POST" });
export const getSamples = () => request("/api/samples");
export const getBenchmarks = () => request("/api/benchmarks");
export const runBenchmark = () => request("/api/benchmarks/run", { method: "POST" });
export const getTraining = () => request("/api/training");

export function enhance({ image, reference, models }) {
  const form = new FormData();
  form.append("image", image);
  if (reference) form.append("reference", reference);
  form.append("models", models.join(","));
  return request("/api/enhance", { method: "POST", body: form });
}

export async function loadSample(name) {
  const get = async (kind) => {
    const res = await fetch(`/api/samples/${kind}/${encodeURIComponent(name)}`);
    if (!res.ok) throw new Error("Could not load the sample image.");
    return new File([await res.blob()], name, { type: "image/png" });
  };
  const [image, reference] = await Promise.all([get("low"), get("high")]);
  return { image, reference };
}
