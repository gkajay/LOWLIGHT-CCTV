# Night Vision Lab — Low-Light CCTV Image Enhancement

React + Vite frontend, FastAPI backend, three PyTorch models trained locally:

| Model | How it is trained |
|---|---|
| **Custom CNN** | Supervised on LOL `our485` pairs, loss = L1 + 0.2·(1 − SSIM) |
| **Zero-DCE++** | From scratch, zero-reference (exposure, colour, spatial and smoothness losses). No pretrained checkpoint |
| **GAN** | The CNN is the generator; a PatchGAN discriminator is trained against it. Generator loss = L1 + 0.2·(1 − SSIM) + 0.01·adversarial (least-squares GAN) |

Everything runs on your machine. The backend only serves the local models to the UI; it calls no external APIs.

```
lowlight-cctv/
├── backend/
│   ├── app/            FastAPI server, models, metrics, benchmarks
│   ├── train/          train_cnn.py, train_zerodce.py, train_gan.py, evaluate.py
│   ├── weights/        trained .pth files are written here
│   └── data/           training history + evaluation CSVs (read by the UI)
├── frontend/           React + Vite app
└── LOLdataset/         put your dataset here (our485/ and eval15/)
```

## 1. Set up

Python 3.10+ and Node 18+.

```bash
cd backend
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

If `pip install torch` picks a huge CUDA build, use the CPU wheel instead:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`

Copy your **LOLdataset** folder (containing `our485` and `eval15`) into `lowlight-cctv/`, or set `LOL_DIR` to its path.

```bash
cd frontend
npm install
```

## 2. Train

From the `backend` folder:

```bash
python -m train.train_cnn     --data ../LOLdataset --epochs 100
python -m train.train_zerodce --data ../LOLdataset --epochs 100
python -m train.train_gan     --data ../LOLdataset --epochs 50   # train the CNN first
python -m train.evaluate      --data ../LOLdataset
```

- Both scripts hold 20 images out of `our485` to choose the best checkpoint, so **`eval15` stays a clean test set**.
- Weights go to `backend/weights/cnn.pth`, `zerodce.pth` and `gan.pth`. Progress is logged to `backend/data/history_*.json` and appears on the **Training** tab.
- `train_gan` starts the generator from `cnn.pth`, so train the CNN first. `gan.pth` is the generator after the last epoch (used by the app). `gan_best.pth` is the epoch with the best validation PSNR, which is usually less GAN-like.
- `evaluate` runs every trained model on `eval15` with identical cropping and metric code and writes `cnn_evaluation.csv`, `zerodce_evaluation.csv` and `gan_evaluation.csv`.
- On a CPU, a 100-epoch run of each model takes a while. Start with `--epochs 5` to confirm everything works. Add `--device cuda` if you have a GPU.
- Your existing `lowlight_model.pth` also works: copy it to `backend/weights/` and the backend uses it until `cnn.pth` exists.

## 3. Run

Two terminals.

```bash
# Terminal 1 — backend
cd backend
uvicorn app.main:app --port 8000

# Terminal 2 — frontend
cd frontend
npm run dev          # open http://localhost:5173
```

Single-server mode: run `npm run build` in `frontend`, then only start uvicorn and open http://localhost:8000.

After training finishes while the server is running, click **Reload models** in the banner.

## What the UI offers

- **Enhance**: upload an image or load a LOL test image, run any of the three models, compare with a before/after slider or side by side, download results. Add a reference image to get PSNR and SSIM.
- **Benchmarks**: reported results from your review, your CSVs, and a live run on `eval15`. Bars for PSNR, SSIM and time plus a per-image table.
- **Training**: loss and validation curves for each model.
- **About**: plain-language explanation and limitations.

## Settings (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `LOL_DIR` | `./LOLdataset` | Dataset folder |
| `CNN_WEIGHTS`, `ZDCE_WEIGHTS`, `GAN_WEIGHTS` | `backend/weights/...` | Weight file paths |
| `MAX_SIDE` | 1280 | Larger uploads are downscaled to this |
| `ZDCE_SCALE` | 12 | Zero-DCE++ inference downscale factor |

## Honest limits

LOL is not real CCTV footage, and 15 test images is a small sample. Enhancement does not detect objects or recover lost detail.
