from pathlib import Path
import csv
import re

ROOT = Path("data/clips")
OUT_DIR = Path("data/manifests")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_CSV = OUT_DIR / "clips.csv"
TRAIN_CSV = OUT_DIR / "train.csv"
TEST_CSV = OUT_DIR / "test.csv"
DEV_CSV = OUT_DIR / "dev.csv"

pattern = re.compile(r"^(spk\d+)_(int\d+)_(\d+)$")

rows = []

for speaker_dir in sorted(ROOT.iterdir()):
    if not speaker_dir.is_dir():
        continue

    speaker_id = speaker_dir.name

    for split_dir in sorted(speaker_dir.iterdir()):
        if not split_dir.is_dir():
            continue

        split = split_dir.name

        for wav_path in sorted(split_dir.glob("*.wav")):
            utt_id = wav_path.stem
            m = pattern.match(utt_id)

            if m:
                parsed_speaker, source_interview, _ = m.groups()
            else:
                parsed_speaker = speaker_id
                source_interview = ""

            rows.append({
                "utt_id": utt_id,
                "speaker_id": parsed_speaker,
                "source_interview": source_interview,
                "split": split,
                "filepath": wav_path.as_posix(),
            })

fieldnames = ["utt_id", "speaker_id", "source_interview", "split", "filepath"]

def save_csv(path, data):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

save_csv(ALL_CSV, rows)
save_csv(TRAIN_CSV, [r for r in rows if r["split"] == "train"])
save_csv(TEST_CSV, [r for r in rows if r["split"] == "test"])
save_csv(DEV_CSV, [r for r in rows if r["split"] == "dev"])

print(f"Saved {len(rows)} rows to {ALL_CSV}")