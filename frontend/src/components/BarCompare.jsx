/** Horizontal bars comparing one metric across models. `better` is "high" or "low". */
export default function BarCompare({ title, unit, rows, digits = 2, better = "high" }) {
  const valid = rows.filter((r) => r.value != null);
  const max = Math.max(...valid.map((r) => r.value), 0.000001);
  const best = valid.length > 1
    ? valid.reduce((a, b) => ((better === "high" ? b.value > a.value : b.value < a.value) ? b : a)).key
    : null;
  return (
    <figure className="bars">
      <figcaption>{title}</figcaption>
      {rows.map((r) => (
        <div className="bar-row" key={r.key}>
          <span className="bar-name">{r.label}</span>
          <div className="bar-track" role="img" aria-label={`${r.label}: ${r.value == null ? "no data" : r.value.toFixed(digits) + " " + unit}`}>
            {r.value != null && <div className={`bar-fill ${r.key}`} style={{ width: `${(r.value / max) * 100}%` }} />}
          </div>
          <span className="bar-val">
            {r.value == null ? "–" : r.value.toFixed(digits)} <small>{unit}</small>
            {best === r.key && <em className="best">best</em>}
          </span>
        </div>
      ))}
    </figure>
  );
}
