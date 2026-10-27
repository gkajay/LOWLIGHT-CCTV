import { useEffect, useState } from "react";
import { getTraining } from "../api.js";

const NAMES = { cnn: "Custom CNN", zerodce: "Zero-DCE++", gan: "GAN" };
const COLORS = { cnn: "var(--cnn)", zerodce: "var(--zdce)", gan: "var(--gan)" };

function LineChart({ epochs, field, label, unit, color }) {
  const pts = epochs.filter((e) => e[field] != null);
  if (pts.length < 2) return null;
  const W = 520, H = 190, P = { l: 46, r: 12, t: 12, b: 26 };
  const xs = pts.map((p) => p.epoch), ys = pts.map((p) => p[field]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  let y0 = Math.min(...ys), y1 = Math.max(...ys);
  if (y0 === y1) { y0 -= 1; y1 += 1; }
  const pad = (y1 - y0) * 0.08; y0 -= pad; y1 += pad;
  const sx = (v) => P.l + ((v - x0) / (x1 - x0)) * (W - P.l - P.r);
  const sy = (v) => H - P.b - ((v - y0) / (y1 - y0)) * (H - P.t - P.b);
  const d = pts.map((p, i) => `${i ? "L" : "M"}${sx(p.epoch).toFixed(1)},${sy(p[field]).toFixed(1)}`).join(" ");
  const ticks = [y0 + pad, (y0 + y1) / 2, y1 - pad];
  const last = pts[pts.length - 1];
  return (
    <figure className="chart">
      <figcaption>{label}<small>{unit}</small></figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${label} over ${pts.length} epochs, ending at ${last[field].toFixed(3)}`}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={P.l} x2={W - P.r} y1={sy(t)} y2={sy(t)} className="grid" />
            <text x={P.l - 6} y={sy(t) + 4} textAnchor="end">{t.toFixed(2)}</text>
          </g>
        ))}
        <text x={P.l} y={H - 6}>epoch {x0}</text>
        <text x={W - P.r} y={H - 6} textAnchor="end">{x1}</text>
        <path d={d} fill="none" stroke={color} strokeWidth="2" strokeLinejoin="round" />
        <circle cx={sx(last.epoch)} cy={sy(last[field])} r="3.5" fill={color} />
      </svg>
    </figure>
  );
}

export default function Training() {
  const [hist, setHist] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    getTraining().then(setHist).catch((e) => setError(e.message));
  }, []);

  if (error) return <div className="notice bad">{error}</div>;
  if (!hist) return <p className="muted">Loading…</p>;
  const keys = Object.keys(hist);

  if (keys.length === 0) {
    return (
      <section className="panel">
        <h2>No training runs yet</h2>
        <p className="muted">Run the training scripts from the <code>backend</code> folder. Each epoch is logged here.</p>
        <pre>{`python -m train.train_cnn --data ../LOLdataset
python -m train.train_zerodce --data ../LOLdataset
python -m train.train_gan --data ../LOLdataset`}</pre>
      </section>
    );
  }

  return (
    <div className="stack">
      {keys.map((k) => {
        const { meta, epochs } = hist[k];
        const best = epochs.filter((e) => e.val_psnr != null).sort((a, b) => b.val_psnr - a.val_psnr)[0];
        const color = COLORS[k] ?? "var(--cnn)";
        return (
          <section className="panel" key={k}>
            <div className="panel-head">
              <h2>{NAMES[k] ?? meta.model}</h2>
              <p>
                {epochs.length} of {meta.epochs} epochs · {meta.train_images} training images · {meta.val_images} validation ·
                batch {meta.batch_size} · learning rate {meta.lr} · {meta.device}
              </p>
              <p className="muted">Loss: {meta.loss}</p>
            </div>
            {best && (
              <p>
                Best validation checkpoint: epoch <strong>{best.epoch}</strong>, PSNR <strong>{best.val_psnr.toFixed(2)} dB</strong>, SSIM{" "}
                <strong>{best.val_ssim.toFixed(4)}</strong>. Validation images come from the training folder, not the eval15 test set.
              </p>
            )}
            <div className="charts">
              <LineChart epochs={epochs} field="loss" label="Training loss" unit="" color={color} />
              <LineChart epochs={epochs} field="val_psnr" label="Validation PSNR" unit=" dB" color={color} />
              <LineChart epochs={epochs} field="val_ssim" label="Validation SSIM" unit="" color={color} />
            </div>
          </section>
        );
      })}
    </div>
  );
}
