from __future__ import annotations

import csv
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
import torch.nn.functional as F
from speechbrain.inference.classifiers import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy


class SpeakerPredictor:
    def __init__(
        self,
        repo_root: Path,
        train_csv: Path,
        model_dir: Path,
        centroids_path: Path,
        model_source: str = "speechbrain/spkrec-ecapa-voxceleb",
        device: str | None = None,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.train_csv = train_csv.resolve()
        self.model_dir = model_dir.resolve()
        self.centroids_path = centroids_path.resolve()
        self.model_source = model_source
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.classifier = EncoderClassifier.from_hparams(
            source=self.model_source,
            savedir=str(self.model_dir),
            run_opts={"device": self.device},
            local_strategy=LocalStrategy.COPY,
        )

        self.centroids: dict[str, torch.Tensor] = {}
        self.ensure_centroids(allow_build=True)

    def ensure_centroids(self, allow_build: bool = True) -> None:
        if self.centroids_path.exists():
            payload = torch.load(self.centroids_path, map_location="cpu")
            self.centroids = {
                spk: F.normalize(t.float().cpu(), dim=0)
                for spk, t in payload["centroids"].items()
            }
            return

        if not allow_build:
            raise FileNotFoundError(
                f"Centroids not found: {self.centroids_path}. "
                "Run build_centroids.py first."
            )

        self.centroids = self.build_centroids()
        self.centroids_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"centroids": self.centroids}, self.centroids_path)

    def load_rows(self, csv_path: Path) -> list[dict[str, str]]:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        required = {"speaker_id", "filepath"}
        fieldnames = set(reader.fieldnames or [])
        missing = required - fieldnames
        if missing:
            raise ValueError(
                f"CSV {csv_path} is missing required columns: {sorted(missing)}"
            )

        return rows

    def resolve_path(self, path_str: str) -> Path:
        p = Path(path_str)

        if p.is_absolute():
            return p

        candidates = [
            (self.repo_root / p),
            (self.repo_root.parent / p),
            (Path.cwd() / p),
        ]

        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate.exists():
                return candidate

        return (self.repo_root / p).resolve()
    
    def load_audio(self, wav_path: Path) -> tuple[torch.Tensor, int]:
        audio, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(audio.T)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform = waveform.squeeze(0)
        return waveform, sample_rate

    def extract_embedding(self, wav_path: Path) -> torch.Tensor:
        waveform, sample_rate = self.load_audio(wav_path)

        if hasattr(self.classifier, "normalizer"):
            waveform = self.classifier.normalizer(waveform, sample_rate)

        if waveform.ndim != 1:
            waveform = waveform.squeeze()

        wavs = waveform.unsqueeze(0).to(self.device)
        wav_lens = torch.tensor([1.0], device=self.device)

        with torch.no_grad():
            emb = self.classifier.encode_batch(wavs, wav_lens, normalize=True)

        emb = emb.squeeze().detach().cpu().float()
        emb = F.normalize(emb, dim=0)
        return emb

    def build_centroids(self) -> dict[str, torch.Tensor]:
        rows = self.load_rows(self.train_csv)
        grouped: dict[str, list[torch.Tensor]] = defaultdict(list)

        for row in rows:
            spk = row["speaker_id"]
            wav_path = self.resolve_path(row["filepath"])
            emb = self.extract_embedding(wav_path)
            grouped[spk].append(emb)

        centroids: dict[str, torch.Tensor] = {}
        for spk, embs in grouped.items():
            centroid = torch.stack(embs, dim=0).mean(dim=0)
            centroids[spk] = F.normalize(centroid, dim=0)

        return centroids

    def predict_embedding(self, emb: torch.Tensor) -> dict[str, Any]:
        scores: dict[str, float] = {}
        for spk, centroid in self.centroids.items():
            score = F.cosine_similarity(
                emb.unsqueeze(0),
                centroid.unsqueeze(0),
                dim=1,
            ).item()
            scores[spk] = score

        predicted = max(scores, key=scores.get)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

        return {
            "predicted_speaker": predicted,
            "scores": scores,
            "ranking": [{"speaker": spk, "score": score} for spk, score in ranked],
        }

    def predict_wav(self, wav_path: Path) -> dict[str, Any]:
        emb = self.extract_embedding(wav_path)
        return self.predict_embedding(emb)

    def convert_to_demo_wav(self, source_path: Path, output_path: Path) -> None:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "ffmpeg not found in PATH. Install ffmpeg and make sure it is available."
            )

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(source_path),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "ffmpeg conversion failed.\n"
                f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
            )

    def predict_any_audio_file(self, source_path: Path) -> dict[str, Any]:
        source_path = source_path.resolve()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            converted = tmpdir_path / "input_16k_mono.wav"
            self.convert_to_demo_wav(source_path, converted)
            result = self.predict_wav(converted)
            result["converted_wav"] = str(converted)
            return result
