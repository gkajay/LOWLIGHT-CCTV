import { useEffect, useMemo, useState } from "react";
import { enhance, getSamples, loadSample } from "../api.js";
import DropZone from "../components/DropZone.jsx";
import CompareSlider from "../components/CompareSlider.jsx";

const LABELS = { cnn: "Custom CNN", zerodce: "Zero-DCE++", gan: "GAN" };
const BLURB = { cnn: "Trained on LOL pairs", zerodce: "Zero-reference curve estimation", gan: "CNN generator trained against a discriminator" };

const fmt = (v, d = 2) => (v == null ? "–" : v.toFixed(d));

export default function Enhance({ health }) {
  const [image, setImage] = useState(null);
  const [reference, setReference] = useState(null);
  const [chosen, setChosen] = useState({ cnn: true, zerodce: true, gan: true });
  const [samples, setSamples] = useState({ available: false, items: [] });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [view, setView] = useState("slider");
  const [leftKey, setLeftKey] = useState("input");
  const [rightKey, setRightKey] = useState("cnn");

  useEffect(() => {
    getSamples().then(setSamples).catch(() => {});
  }, []);

  const [inputUrl, setInputUrl] = useState(null);
  useEffect(() => {
    if (!image) { setInputUrl(null); return undefined; }
    const url = URL.createObjectURL(image);
    setInputUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [image]);

  const ready = (k) => health?.models?.[k]?.ready;
  const selected = Object.keys(chosen).filter((k) => chosen[k]);

  const pick = async (name) => {
    if (!name) return;
    setError(null);
    try {
      const s = await loadSample(name);
      setImage(s.image);
      setReference(s.reference);
      setResult(null);
    } catch (e) {
      setError(e.message);
    }
  };

  const run = async () => {
    setBusy(true);
    setError(null);
    try {
      const r = await enhance({ image, reference, models: selected });
      setResult(r);
      const okKeys = selected.filter((k) => r.results[k] && !r.results[k].error);
      setLeftKey("input");
      setRightKey(okKeys[0] ?? "input");
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  // Everything that can be shown in a panel, in display order.
  const panels = useMemo(() => {
    if (!result) return [];
    const out = [{ key: "input", label: "Original", url: result.urls.input }];
    for (const k of Object.keys(LABELS)) {
      if (result.urls[k]) out.push({ key: k, label: LABELS[k], url: result.urls[k] });
    }
    if (result.urls.reference) out.push({ key: "reference", label: "Reference", url: result.urls.reference });
    return out;
  }, [result]);

  const byKey = Object.fromEntries(panels.map((p) => [p.key, p]));
  const hasRef = Boolean(result?.urls.reference);
  const aspect = result ? `${result.width} / ${result.height}` : "3 / 2";

  return (
    <div className="stack">
      <section className="panel">
        <div className="panel-head">
          <h2>1. Choose an image</h2>
          <p>Upload a dark frame, or use a test image from the LOL evaluation set.</p>
        </div>
        <div className="grid-2">
          <DropZone file={image} onFile={(f) => { setImage(f); setResult(null); }}
            label="Drop a low-light image here" hint="PNG, JPG or BMP · or click to browse" />
          <DropZone file={reference} compact onFile={setReference}
            label="Reference image (optional)" hint="A normal-light version of the same scene. Adds PSNR and SSIM." />
        </div>
        {samples.available && (
          <label className="field">
            <span>Or load a LOL test image</span>
            <select defaultValue="" onChange={(e) => pick(e.target.value)}>
              <option value="">Select a sample…</option>
              {samples.items.map((n) => <option key={n} value={n}>{n}</option>)}
            </select>
          </label>
        )}
      </section>

      <section className="panel">
        <div className="panel-head">
          <h2>2. Pick models</h2>
          <p>Run one model or compare both on the same image.</p>
        </div>
        <div className="models">
          {Object.entries(LABELS).map(([k, label]) => (
            <label key={k} className={`model-opt ${k}${chosen[k] ? " on" : ""}${ready(k) ? "" : " off"}`}>
              <input type="checkbox" checked={chosen[k]} onChange={(e) => setChosen({ ...chosen, [k]: e.target.checked })} />
              <span className="swatch" aria-hidden="true" />
              <span>
                <strong>{label}</strong>
                <small>{ready(k) ? BLURB[k] : "Weights not found"}</small>
              </span>
            </label>
          ))}
        </div>
        <div className="actions">
          <button className="btn primary" disabled={!image || busy || selected.length === 0} onClick={run}>
            {busy ? "Enhancing…" : "Enhance image"}
          </button>
          {!image && <span className="muted">Add an image to continue.</span>}
        </div>
        {error && <div className="notice bad" role="alert">{error}</div>}
      </section>

      {(result || image) && (
        <section className="panel">
          <div className="panel-head row">
            <h2>Result</h2>
            {result && (
              <div className="seg" role="tablist" aria-label="View">
                <button className={view === "slider" ? "on" : ""} onClick={() => setView("slider")}>Slider</button>
                <button className={view === "grid" ? "on" : ""} onClick={() => setView("grid")}>Side by side</button>
              </div>
            )}
          </div>

          {!result && inputUrl && (
            <img className="preview" src={inputUrl} alt="Selected low-light input" />
          )}

          {result && result.downscaled && (
            <p className="muted">Large image resized to fit within {health?.max_side}px before enhancement.</p>
          )}

          {result && view === "slider" && panels.length > 1 && (
            <>
              <div className="pickers">
                <label className="field inline"><span>Left</span>
                  <select value={leftKey} onChange={(e) => setLeftKey(e.target.value)}>
                    {panels.map((p) => <option key={p.key} value={p.key}>{p.label}</option>)}
                  </select>
                </label>
                <label className="field inline"><span>Right</span>
                  <select value={rightKey} onChange={(e) => setRightKey(e.target.value)}>
                    {panels.map((p) => <option key={p.key} value={p.key}>{p.label}</option>)}
                  </select>
                </label>
              </div>
              <CompareSlider left={byKey[leftKey].url} right={byKey[rightKey].url}
                leftLabel={byKey[leftKey].label} rightLabel={byKey[rightKey].label} aspect={aspect} />
            </>
          )}

          {result && view === "grid" && (
            <div className="gallery">
              {panels.map((p) => (
                <figure key={p.key}>
                  <img src={p.url} alt={p.label} />
                  <figcaption>
                    <span>{p.label}</span>
                    {p.key !== "reference" && p.key !== "input" && (
                      <a href={`${p.url}?download=true`} download>Download</a>
                    )}
                  </figcaption>
                </figure>
              ))}
            </div>
          )}

          {result && (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Image</th>
                    <th className="num">PSNR (dB)</th>
                    <th className="num">SSIM</th>
                    <th className="num">Brightness</th>
                    <th className="num">Time (ms)</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>Original</td>
                    <td className="num">{fmt(result.input.psnr)}</td>
                    <td className="num">{fmt(result.input.ssim, 4)}</td>
                    <td className="num">{fmt(result.input.brightness * 100, 1)}%</td>
                    <td className="num">–</td>
                  </tr>
                  {Object.entries(result.results).map(([k, r]) => (
                    <tr key={k}>
                      <td><span className={`dot ${k}`} />{LABELS[k]}</td>
                      {r.error ? (
                        <td colSpan="4" className="err">{r.error}</td>
                      ) : (
                        <>
                          <td className="num">{fmt(r.psnr)}</td>
                          <td className="num">{fmt(r.ssim, 4)}</td>
                          <td className="num">{fmt(r.brightness * 100, 1)}%</td>
                          <td className="num">{fmt(r.time_ms, 1)}</td>
                        </>
                      )}
                    </tr>
                  ))}
                  {hasRef && (
                    <tr>
                      <td>Reference</td><td className="num">–</td><td className="num">–</td>
                      <td className="num">{fmt(result.reference.brightness * 100, 1)}%</td><td className="num">–</td>
                    </tr>
                  )}
                </tbody>
              </table>
              {!hasRef && <p className="muted">Add a reference image to see PSNR and SSIM. Brightness is the average luminance of each image.</p>}
            </div>
          )}
        </section>
      )}
    </div>
  );
}
