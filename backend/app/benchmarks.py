"""Benchmark data: reported summary, user CSVs, and a live run on the LOL eval set."""
import csv
import json
from pathlib import Path

from . import config
from .imaging import load_rgb
from .metrics import psnr_ssim

# Averages from the original project review (15 LOL eval images, CPU).
REPORTED = {
    "id": "reported",
    "label": "Reported results",
    "kind": "summary",
    "note": "Averages from the original 15-image LOL evaluation. No per-image data.",
    "models": {
        "cnn": {"label": "Custom CNN", "n": 15, "avg_psnr": 17.3251, "avg_ssim": 0.7225, "avg_time_ms": 56.298, "rows": []},
        "zerodce": {"label": "Zero-DCE++", "n": 15, "avg_psnr": 15.3451, "avg_ssim": 0.5731, "avg_time_ms": 21.0, "rows": []},
    },
}

LIVE_FILE = config.DATA_DIR / "live_benchmark.json"
CSV_FILES = {"cnn": "cnn_evaluation.csv", "zerodce": "zerodce_evaluation.csv", "gan": "gan_evaluation.csv"}
LABELS = {"cnn": "Custom CNN", "zerodce": "Zero-DCE++", "gan": "GAN"}
ALL_MODELS = ("cnn", "zerodce", "gan")


def _avg(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(key: str, rows: list[dict]) -> dict:
    return {
        "label": LABELS[key],
        "n": len(rows),
        "avg_psnr": _avg([r["psnr"] for r in rows if r.get("psnr") is not None]),
        "avg_ssim": _avg([r["ssim"] for r in rows if r.get("ssim") is not None]),
        "avg_time_ms": _avg([r["time_ms"] for r in rows if r.get("time_ms") is not None]),
        "rows": rows,
    }


def _find(header: list[str], *needles: str) -> int | None:
    for i, name in enumerate(header):
        low = name.strip().lower()
        if any(n in low for n in needles):
            return i
    return None


def _float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_csv(path: Path) -> list[dict]:
    """Parse a per-image evaluation CSV. Column names are matched loosely
    (anything containing 'psnr', 'ssim', 'time', 'name'/'image'/'file')."""
    with path.open(newline="", encoding="utf-8-sig") as fh:
        table = list(csv.reader(fh))
    if len(table) < 2:
        return []
    header, body = table[0], table[1:]
    i_name = _find(header, "name", "image", "file")
    i_psnr, i_ssim, i_time = _find(header, "psnr"), _find(header, "ssim"), _find(header, "time")
    time_in_ms = i_time is not None and "ms" in header[i_time].lower()
    rows = []
    for rec in body:
        name = rec[i_name] if i_name is not None and i_name < len(rec) else (rec[0] if rec else "")
        if not name or any(w in name.lower() for w in ("average", "mean", "avg")):
            continue
        t = _float(rec[i_time]) if i_time is not None and i_time < len(rec) else None
        if t is not None and not time_in_ms:
            t *= 1000.0  # seconds -> ms
        rows.append({
            "name": name,
            "psnr": _float(rec[i_psnr]) if i_psnr is not None and i_psnr < len(rec) else None,
            "ssim": _float(rec[i_ssim]) if i_ssim is not None and i_ssim < len(rec) else None,
            "time_ms": t,
        })
    return rows


def csv_run() -> dict | None:
    models = {}
    for key, fname in CSV_FILES.items():
        path = config.DATA_DIR / fname
        if path.exists():
            rows = read_csv(path)
            if rows:
                models[key] = summarize(key, rows)
    if not models:
        return None
    return {"id": "csv", "label": "Your CSV files", "kind": "csv",
            "note": "Loaded from backend/data/*.csv.", "models": models}


def live_run() -> dict | None:
    if LIVE_FILE.exists():
        return json.loads(LIVE_FILE.read_text())
    return None


def all_runs() -> list[dict]:
    return [r for r in (live_run(), csv_run(), REPORTED) if r]


def evaluate_dataset(engine, low_dir: Path, high_dir: Path, keys=ALL_MODELS, limit: int | None = None) -> dict:
    """Run every model on every low/high pair using one shared alignment and metric
    implementation, so the comparison is fair. Returns {key: [row, ...]}."""
    files = sorted(p.name for p in low_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
    if limit:
        files = files[:limit]
    rows = {k: [] for k in keys}
    for name in files:
        low, _ = load_rgb((low_dir / name).read_bytes())
        high, _ = load_rgb((high_dir / name).read_bytes())
        h, w = min(low.shape[0], high.shape[0]), min(low.shape[1], high.shape[1])
        low, high = low[:h, :w], high[:h, :w]
        for key in keys:
            out, ms = engine.run(key, low)
            p, s = psnr_ssim(high, out)
            rows[key].append({"name": name, "psnr": p, "ssim": s, "time_ms": ms})
    return rows


def run_live(engine) -> dict:
    low_dir, high_dir = config.LOL_DIR / "eval15" / "low", config.LOL_DIR / "eval15" / "high"
    if not low_dir.is_dir() or not high_dir.is_dir():
        raise FileNotFoundError(f"LOL eval folder not found at {config.LOL_DIR / 'eval15'}. Set LOL_DIR.")
    status = engine.status()
    keys = [k for k in ALL_MODELS if status[k]["ready"]]
    if not keys:
        raise RuntimeError("No trained model is available yet. Train at least one model first.")
    rows = evaluate_dataset(engine, low_dir, high_dir, keys)
    run = {
        "id": "live", "label": "Live run (this machine)", "kind": "live",
        "note": "Both models evaluated on eval15 with identical alignment and metric code.",
        "device": str(engine.device),
        "models": {k: summarize(k, r) for k, r in rows.items()},
    }
    LIVE_FILE.write_text(json.dumps(run, indent=2))
    return run


def training_histories() -> dict:
    out = {}
    for key in ALL_MODELS:
        f = config.DATA_DIR / f"history_{key}.json"
        if f.exists():
            out[key] = json.loads(f.read_text())
    return out
