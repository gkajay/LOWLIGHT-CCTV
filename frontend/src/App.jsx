import { useCallback, useEffect, useState } from "react";
import { getHealth, reloadModels } from "./api.js";
import Enhance from "./pages/Enhance.jsx";
import Benchmarks from "./pages/Benchmarks.jsx";
import Training from "./pages/Training.jsx";
import About from "./pages/About.jsx";

const TABS = [
  { id: "enhance", label: "Enhance" },
  { id: "benchmarks", label: "Benchmarks" },
  { id: "training", label: "Training" },
  { id: "about", label: "About" },
];

export default function App() {
  const [tab, setTab] = useState(() => {
    const hash = window.location.hash.slice(1);
    return TABS.some((t) => t.id === hash) ? hash : "enhance";
  });
  const [health, setHealth] = useState(null);
  const [healthError, setHealthError] = useState(null);

  const refresh = useCallback(async (reload = false) => {
    try {
      setHealth(await (reload ? reloadModels() : getHealth()));
      setHealthError(null);
    } catch (e) {
      setHealthError(e.message);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  useEffect(() => {
    window.location.hash = tab;
  }, [tab]);

  const models = health?.models ?? {};

  return (
    <>
      <header className="masthead">
        <div className="wrap masthead-row">
          <div className="brand">
            <span className="lamp" aria-hidden="true" />
            <div>
              <h1>Night Vision Lab</h1>
              <p>Low-light CCTV image enhancement</p>
            </div>
          </div>
          <nav aria-label="Main">
            {TABS.map((t) => (
              <button key={t.id} className={tab === t.id ? "tab active" : "tab"} onClick={() => setTab(t.id)}
                aria-current={tab === t.id ? "page" : undefined}>
                {t.label}
              </button>
            ))}
          </nav>
          <div className="status" role="status">
            {healthError ? (
              <span className="pill bad">Backend offline</span>
            ) : !health ? (
              <span className="pill">Connecting…</span>
            ) : (
              <>
                {Object.entries(models).map(([key, m]) => (
                  <span key={key} className={m.ready ? `pill ok ${key}` : "pill bad"} title={m.error || m.weights_file}>
                    {m.label}: {m.ready ? "ready" : "no weights"}
                  </span>
                ))}
                <span className="pill">{health.device.toUpperCase()}</span>
              </>
            )}
          </div>
        </div>
      </header>

      <main className="wrap">
        {healthError && (
          <div className="notice bad">
            <strong>The backend is not running.</strong> {healthError}
            <button className="btn small" onClick={() => refresh()}>Try again</button>
          </div>
        )}
        {health && Object.values(models).some((m) => !m.ready) && (
          <div className="notice">
            <strong>
              {Object.values(models).every((m) => !m.ready) ? "No trained models found." : "One model is missing weights."}
            </strong>{" "}
            Train with <code>python -m train.train_cnn</code>, <code>python -m train.train_zerodce</code> and <code>python -m train.train_gan</code>, then
            <button className="btn small" onClick={() => refresh(true)}>Reload models</button>
          </div>
        )}
        {tab === "enhance" && <Enhance health={health} />}
        {tab === "benchmarks" && <Benchmarks health={health} />}
        {tab === "training" && <Training />}
        {tab === "about" && <About />}
      </main>
    </>
  );
}
