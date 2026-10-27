"""Evaluate every trained model (CNN, Zero-DCE++, GAN) on LOL eval15 and write per-image CSVs (same format the UI reads).

    cd backend
    python -m train.evaluate --data ../LOLdataset
"""
import argparse
import csv
from pathlib import Path

from app import benchmarks, config
from app.engine import Engine


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=config.LOL_DIR)
    args = ap.parse_args()
    config.LOL_DIR = args.data

    engine = Engine()
    run = benchmarks.run_live(engine)
    for key, m in run["models"].items():
        path = config.DATA_DIR / benchmarks.CSV_FILES[key]
        with path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["image", "psnr", "ssim", "time_ms"])
            for r in m["rows"]:
                w.writerow([r["name"], f"{r['psnr']:.4f}", f"{r['ssim']:.4f}", f"{r['time_ms']:.3f}"])
        print(f"{m['label']:<12} n={m['n']}  PSNR {m['avg_psnr']:.4f} dB  SSIM {m['avg_ssim']:.4f}  {m['avg_time_ms']:.1f} ms/img  -> {path.name}")


if __name__ == "__main__":
    main()
