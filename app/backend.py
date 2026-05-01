from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .predictor import SpeakerPredictor

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = (APP_DIR.parent).resolve()

TRAIN_CSV = REPO_ROOT / "data" / "manifests" / "train.csv"
MODEL_DIR = REPO_ROOT / "pretrained_models" / "spkrec-ecapa-voxceleb"
CENTROIDS_PATH = REPO_ROOT / "data" / "features" / "centroids.pt"

SPEAKER_NAMES = {
    "spk01": "Trump",
    "spk02": "Biden",
    "spk03": "Obama",
}

app = FastAPI(title="Speaker Recognition Demo")

app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")

predictor = SpeakerPredictor(
    repo_root=REPO_ROOT,
    train_csv=TRAIN_CSV,
    model_dir=MODEL_DIR,
    centroids_path=CENTROIDS_PATH,
)


@app.get("/")
async def root() -> FileResponse:
    return FileResponse(APP_DIR / "static" / "index.html")


@app.get("/health")
async def health() -> dict:
    speakers = sorted(predictor.centroids.keys())
    return {
        "status": "ok",
        "device": predictor.device,
        "speakers": [SPEAKER_NAMES.get(spk, spk) for spk in speakers],
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing file name.")

    suffix = Path(file.filename).suffix or ".bin"

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            uploaded_path = tmpdir_path / f"uploaded{suffix}"

            with uploaded_path.open("wb") as f:
                shutil.copyfileobj(file.file, f)

            result = predictor.predict_any_audio_file(uploaded_path)
            
            # Map speaker IDs to human-readable names
            if "predicted_speaker" in result:
                spk = result["predicted_speaker"]
                result["predicted_speaker"] = SPEAKER_NAMES.get(spk, spk)
                
            if "scores" in result:
                result["scores"] = {SPEAKER_NAMES.get(k, k): v for k, v in result["scores"].items()}
                
            if "ranking" in result:
                for item in result["ranking"]:
                    item["speaker"] = SPEAKER_NAMES.get(item["speaker"], item["speaker"])
                    
            result["filename"] = file.filename
            return result

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
