"""Local FastAPI server. It only connects the React UI to the PyTorch models on this
machine; it never calls any external service."""
import re
import uuid
from collections import OrderedDict

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from . import benchmarks, config
from .engine import MODELS, Engine
from .imaging import load_rgb, resize_to, to_png_bytes
from .metrics import mean_brightness, psnr_ssim

app = FastAPI(title="Low-Light CCTV Enhancement", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = Engine()
RESULTS: "OrderedDict[str, dict[str, bytes]]" = OrderedDict()
MAX_RESULTS = 24
SAFE_NAME = re.compile(r"^[\w.\-]+$")


def _store(result_id: str, files: dict[str, bytes]) -> None:
    RESULTS[result_id] = files
    while len(RESULTS) > MAX_RESULTS:
        RESULTS.popitem(last=False)


# ------------------------------------------------------------------ status
@app.get("/api/health")
def health():
    return {
        "device": str(engine.device),
        "models": engine.status(),
        "dataset_available": (config.LOL_DIR / "eval15" / "low").is_dir(),
        "max_side": config.MAX_SIDE,
        "zdce_scale": config.ZDCE_SCALE,
    }


@app.post("/api/reload")
def reload_models():
    engine.reload()
    return health()


# ------------------------------------------------------------------ enhance
@app.post("/api/enhance")
async def enhance(
    image: UploadFile = File(...),
    reference: UploadFile | None = File(None),
    models: str = Form("cnn,zerodce,gan"),
):
    keys = [m for m in models.split(",") if m]
    if not keys or any(k not in MODELS for k in keys):
        raise HTTPException(400, "models must be a comma-separated list of: cnn, zerodce, gan")

    try:
        low, resized = load_rgb(await image.read(), config.MAX_SIDE)
    except Exception:
        raise HTTPException(400, "Could not read the image. Use a PNG, JPG or BMP file.")
    h, w = low.shape[:2]

    ref = None
    if reference is not None and reference.filename:
        try:
            ref_full, _ = load_rgb(await reference.read())
        except Exception:
            raise HTTPException(400, "Could not read the reference image.")
        rh, rw = ref_full.shape[:2]
        if abs(rw / rh - w / h) > 0.02:
            raise HTTPException(400, "The reference image must have the same aspect ratio as the input.")
        ref = ref_full if ref_full.shape[:2] == (h, w) else resize_to(ref_full, h, w)

    files = {"input": to_png_bytes(low)}
    response = {
        "width": w,
        "height": h,
        "downscaled": resized,
        "input": {"brightness": mean_brightness(low)},
        "results": {},
    }
    if ref is not None:
        files["reference"] = to_png_bytes(ref)
        p, s = psnr_ssim(ref, low)
        response["input"].update(psnr=p, ssim=s)
        response["reference"] = {"brightness": mean_brightness(ref)}

    for key in keys:
        try:
            out, ms = engine.run(key, low)
        except FileNotFoundError as exc:
            response["results"][key] = {"error": str(exc)}
            continue
        except Exception as exc:
            response["results"][key] = {"error": f"Enhancement failed: {exc}"}
            continue
        entry = {"time_ms": ms, "brightness": mean_brightness(out)}
        if ref is not None:
            entry["psnr"], entry["ssim"] = psnr_ssim(ref, out)
        files[key] = to_png_bytes(out)
        response["results"][key] = entry

    if not any("error" not in r for r in response["results"].values()):
        raise HTTPException(503, "No trained model is available. Train a model first, then reload.")

    rid = uuid.uuid4().hex[:12]
    _store(rid, files)
    response["id"] = rid
    response["urls"] = {name: f"/api/results/{rid}/{name}.png" for name in files}
    return response


@app.get("/api/results/{rid}/{name}.png")
def get_result(rid: str, name: str, download: bool = False):
    files = RESULTS.get(rid)
    if not files or name not in files:
        raise HTTPException(404, "Result expired. Enhance the image again.")
    headers = {"Content-Disposition": f'attachment; filename="{name}_{rid}.png"'} if download else {}
    return Response(files[name], media_type="image/png", headers=headers)


# ------------------------------------------------------------------ samples
def _sample_names() -> list[str]:
    low = config.LOL_DIR / "eval15" / "low"
    if not low.is_dir():
        return []
    return sorted(p.name for p in low.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})


@app.get("/api/samples")
def samples():
    names = _sample_names()
    return {"available": bool(names), "items": names}


@app.get("/api/samples/{kind}/{name}")
def sample_file(kind: str, name: str):
    if kind not in ("low", "high") or not SAFE_NAME.match(name) or name not in _sample_names():
        raise HTTPException(404, "Sample not found")
    return FileResponse(config.LOL_DIR / "eval15" / kind / name)


# --------------------------------------------------------------- benchmarks
@app.get("/api/benchmarks")
def get_benchmarks():
    return {"runs": benchmarks.all_runs(), "dataset_available": bool(_sample_names())}


@app.post("/api/benchmarks/run")
def run_benchmark():
    try:
        run = benchmarks.run_live(engine)
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(409, str(exc))
    return run


@app.get("/api/training")
def training():
    return benchmarks.training_histories()


# --------------------------------------------------- serve built frontend
if config.FRONTEND_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=config.FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        target = (config.FRONTEND_DIST / path).resolve()
        if path and target.is_file() and config.FRONTEND_DIST.resolve() in target.parents:
            return FileResponse(target)
        return FileResponse(config.FRONTEND_DIST / "index.html")
