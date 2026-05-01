import os
import sys
import torch
import soundfile as sf
from pyannote.audio import Pipeline

if len(sys.argv) < 3:
    print("Usage: python diarize_to_labels.py input.wav output_labels.txt [num_speakers]")
    sys.exit(1)

audio_path = sys.argv[1]
label_path = sys.argv[2]
num_speakers = int(sys.argv[3]) if len(sys.argv) >= 4 else None

token = os.environ.get("HF_TOKEN")
if not token:
    raise RuntimeError("HF_TOKEN environment variable not set.")

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token=token,
)

if torch.cuda.is_available():
    pipeline.to(torch.device("cuda"))

# Read audio with soundfile instead of torchaudio / torchcodec
audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)

# soundfile returns (time, channels) -> pyannote expects (channels, time)
waveform = torch.from_numpy(audio.T)

# Downmix to mono if needed
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

if num_speakers is not None:
    output = pipeline(
        {"waveform": waveform, "sample_rate": sample_rate},
        num_speakers=num_speakers,
    )
else:
    output = pipeline(
        {"waveform": waveform, "sample_rate": sample_rate},
    )

diarization = output.exclusive_speaker_diarization

with open(label_path, "w", encoding="utf-8") as f:
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start = max(0.0, float(turn.start))
        end = max(start, float(turn.end))

        if end - start < 1.0:
            continue

        f.write(f"{start:.3f}\t{end:.3f}\t{speaker}\n")

print(f"Saved labels to: {label_path}")
