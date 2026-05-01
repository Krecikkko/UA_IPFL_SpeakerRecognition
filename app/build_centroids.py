from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .predictor import SpeakerPredictor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo_root",
        type=str,
        default=".",
        help="Path to your project root containing data/, pretrained_models/, etc.",
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        default="data/manifests/train.csv",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="pretrained_models/spkrec-ecapa-voxceleb",
    )
    parser.add_argument(
        "--centroids_path",
        type=str,
        default="data/features/centroids.pt",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    train_csv = (repo_root / args.train_csv).resolve()
    model_dir = (repo_root / args.model_dir).resolve()
    centroids_path = (repo_root / args.centroids_path).resolve()

    predictor = SpeakerPredictor(
        repo_root=repo_root,
        train_csv=train_csv,
        model_dir=model_dir,
        centroids_path=centroids_path,
    )

    centroids_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"centroids": predictor.centroids}, centroids_path)

    print(f"Saved centroids to: {centroids_path}")
    print(f"Speakers: {sorted(predictor.centroids)}")


if __name__ == "__main__":
    main()
