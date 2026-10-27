import { useCallback, useEffect, useState } from "react";
import { getBenchmarks, runBenchmark } from "../api.js";
import BarCompare from "../components/BarCompare.jsx";

const ORDER = ["cnn", "zerodce", "gan"];

export default function Benchmarks({ health }) {
  const [data, setData] = useState(null);
  const [runId, setRunId] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  const load = useCallback(async (prefer) => {
    try {
      const d = await getBenchmarks();
      setData(d);
      setRunId((cur) => prefer ?? (d.runs.some((r) => r.id === cur) ? cur : d.runs[0]?.id));
    } catch (e) {
      setError(e.message);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const run = async () => {
    setBusy(true);
    setError(null);
    try {
      await runBenchmark();
      await load("live");
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  const current = data?.runs.find((r) => r.id === runId);
  const rows = current ? ORDER.filter((k) => current.models[k]).map((k) => ({ key: k, ...current.models[k] })) : [];
  const metric = (field) => rows.map((m) => ({ key: m.key, label: m.label, value: m[field] }));
  const anyReady = health && Object.values(health.models).some((m) => m.ready);

  const names = rows.length ? [...new Set(rows.flatMap((m) => m.rows.map((r) => r.name)))] : [];
  const cell = (m, name) => m.rows.find((r) => r.name === name);

  return (
    <div className="stack">
      <section className="panel">
        <div className="panel-head row">
          <div>
            <h2>Benchmarks</h2>
            <p>PSNR and SSIM measure similarity to a normal-light reference. Higher is better. Time is milliseconds per image.</p>
          </div>
          <button className="btn primary" onClick={run} disabled={busy || !data?.dataset_available || !anyReady}
            title={!data?.dataset_available ? "LOL dataset not found. Set LOL_DIR." : undefined}>
            {busy ? "Running…" : "Run on eval15"}
          </button>
        </div>
        {data && !data.dataset_available && (
          <p className="muted">The LOL dataset was not found, so live runs are disabled. Place <code>LOLdataset</code> next to <code>backend</code> or set <code>LOL_DIR</code>.</p>
        )}
        {error && <div className="notice bad" role="alert">{error}</div>}

        {data && (
          <div className="seg wide" role="tablist" aria-label="Benchmark source">
            {data.runs.map((r) => (
              <button key={r.id} className={r.id === runId ? "on" : ""} onClick={() => setRunId(r.id)}>{r.label}</button>
            ))}
          </div>
        )}
        {current && <p className="muted">{current.note}{current.device ? ` Device: ${current.device}.` : ""}</p>}
      </section>

      {current && (
        <section className="panel">
          <div className="bars-grid">
            <BarCompare title="Average PSNR" unit="dB" rows={metric("avg_psnr")} />
            <BarCompare title="Average SSIM" unit="" digits={4} rows={metric("avg_ssim")} />
            <BarCompare title="Inference time" unit="ms" digits={1} better="low" rows={metric("avg_time_ms")} />
          </div>
          <p className="muted">
            {rows.map((m) => `${m.label}: ${m.n} images`).join(" · ")}. A small test set and LOL photos are not real CCTV footage, so treat these as a first comparison.
          </p>
        </section>
      )}

      {current && names.length > 0 && (
        <section className="panel">
          <h2>Per-image results</h2>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th rowSpan="2">Image</th>
                  {rows.map((m) => <th key={m.key} colSpan="3" className={`grp ${m.key}`}>{m.label}</th>)}
                </tr>
                <tr>
                  {rows.map((m) => (
                    <FragmentHead key={m.key} />
                  ))}
                </tr>
              </thead>
              <tbody>
                {names.map((n) => (
                  <tr key={n}>
                    <td>{n}</td>
                    {rows.map((m) => {
                      const c = cell(m, n);
                      return (
                        <FragmentCells key={m.key} c={c} />
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  );
}

function FragmentHead() {
  return (
    <>
      <th className="num">PSNR</th>
      <th className="num">SSIM</th>
      <th className="num">ms</th>
    </>
  );
}

function FragmentCells({ c }) {
  const f = (v, d) => (v == null ? "–" : v.toFixed(d));
  return (
    <>
      <td className="num">{f(c?.psnr, 2)}</td>
      <td className="num">{f(c?.ssim, 4)}</td>
      <td className="num">{f(c?.time_ms, 1)}</td>
    </>
  );
}
