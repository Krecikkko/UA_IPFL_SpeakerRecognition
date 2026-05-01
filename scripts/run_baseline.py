from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
from speechbrain.utils.fetching import LocalStrategy
from speechbrain.inference.classifiers import EncoderClassifier


def load_csv_rows(csv_path: Path) -> list[dict]:
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    required = {"speaker_id", "filepath"}
    missing = required - set(reader.fieldnames or [])
    if missing:
        raise ValueError(
            f"CSV {csv_path} is missing required columns: {sorted(missing)}"
        )
    return rows


def resolve_path(path_str: str, repo_root: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (repo_root / p).resolve()


def load_audio_with_soundfile(path: Path) -> tuple[torch.Tensor, int]:
    """
    Returns:
        waveform: torch.Tensor [time] mono float32
        sample_rate: int
    """
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    # soundfile: [time, channels] -> torch [channels, time]
    waveform = torch.from_numpy(audio.T)

    # downmix to mono if needed
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    waveform = waveform.squeeze(0)  # [time]
    return waveform, sample_rate


def extract_embedding(
    classifier: EncoderClassifier,
    wav_path: Path,
    device: str,
) -> torch.Tensor:
    waveform, sample_rate = load_audio_with_soundfile(wav_path)

    # SpeechBrain docs recommend using normalizer(signal, sample_rate)
    # to convert the signal to the model's expected format.
    if hasattr(classifier, "normalizer"):
        waveform = classifier.normalizer(waveform, sample_rate)

    if waveform.ndim != 1:
        waveform = waveform.squeeze()

    wavs = waveform.unsqueeze(0).to(device)         # [1, time]
    wav_lens = torch.tensor([1.0], device=device)   # full-length utterance

    with torch.no_grad():
        emb = classifier.encode_batch(wavs, wav_lens, normalize=True)

    # Typical shape: [1, 1, dim] -> [dim]
    emb = emb.squeeze().detach().cpu().float()
    emb = F.normalize(emb, dim=0)
    return emb


def build_centroids(
    train_rows: list[dict],
    classifier: EncoderClassifier,
    repo_root: Path,
    device: str,
) -> dict[str, torch.Tensor]:
    grouped = defaultdict(list)

    for row in train_rows:
        spk = row["speaker_id"]
        wav_path = resolve_path(row["filepath"], repo_root)
        emb = extract_embedding(classifier, wav_path, device)
        grouped[spk].append(emb)

    centroids = {}
    for spk, embs in grouped.items():
        centroid = torch.stack(embs, dim=0).mean(dim=0)
        centroid = F.normalize(centroid, dim=0)
        centroids[spk] = centroid

    return centroids


def predict_speaker(
    emb: torch.Tensor,
    centroids: dict[str, torch.Tensor],
) -> tuple[str, dict[str, float]]:
    scores = {}
    for spk, centroid in centroids.items():
        score = F.cosine_similarity(
            emb.unsqueeze(0),
            centroid.unsqueeze(0),
            dim=1,
        ).item()
        scores[spk] = score

    pred = max(scores, key=scores.get)
    return pred, scores


def print_confusion_matrix(y_true: list[str], y_pred: list[str]) -> None:
    labels = sorted(set(y_true) | set(y_pred))
    matrix = {t: {p: 0 for p in labels} for t in labels}

    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1

    print("\nConfusion matrix:")
    header = ["true\\pred"] + labels
    print(" | ".join(f"{h:>12}" for h in header))
    print("-" * (15 * len(header)))

    for t in labels:
        row = [t] + [str(matrix[t][p]) for p in labels]
        print(" | ".join(f"{x:>12}" for x in row))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_csv",
        type=str,
        default="data/manifests/train.csv",
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default="data/manifests/test.csv",
    )
    parser.add_argument(
        "--savedir",
        type=str,
        default="pretrained_models/spkrec-ecapa-voxceleb",
    )
    args = parser.parse_args()

    repo_root = Path.cwd()
    train_csv = (repo_root / args.train_csv).resolve()
    test_csv = (repo_root / args.test_csv).resolve()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=str((repo_root / args.savedir).resolve()),
    run_opts={"device": device},
    local_strategy=LocalStrategy.COPY,
)

    train_rows = load_csv_rows(train_csv)
    test_rows = load_csv_rows(test_csv)

    print(f"Train rows: {len(train_rows)}")
    print(f"Test rows:  {len(test_rows)}")

    print("\nBuilding speaker centroids from train set...")
    centroids = build_centroids(train_rows, classifier, repo_root, device)

    print("\nEvaluating on test set...")
    y_true = []
    y_pred = []

    for row in test_rows:
        true_spk = row["speaker_id"]
        wav_path = resolve_path(row["filepath"], repo_root)

        emb = extract_embedding(classifier, wav_path, device)
        pred_spk, scores = predict_speaker(emb, centroids)

        y_true.append(true_spk)
        y_pred.append(pred_spk)

        utt_id = row.get("utt_id", wav_path.stem)
        print(
            f"{utt_id}: true={true_spk}, pred={pred_spk}, "
            f"scores={{{', '.join(f'{k}: {v:.3f}' for k, v in scores.items())}}}"
        )

    correct = sum(t == p for t, p in zip(y_true, y_pred))
    acc = correct / len(y_true) if y_true else 0.0

    print(f"\nAccuracy: {acc:.4f} ({correct}/{len(y_true)})")
    print_confusion_matrix(y_true, y_pred)


if __name__ == "__main__":
    main()