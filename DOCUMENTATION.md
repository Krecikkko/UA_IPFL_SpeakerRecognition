# Speaker Recognition Project Documentation

## 1. Project purpose

This project implements a **speaker recognition system** for a small set of known speakers.  
The main goal is to recognize **who is speaking** in an input audio sample by comparing the speaker characteristics of the sample against reference data prepared during enrollment/training.

The project was developed as a practical assignment for a speech-related university course.

## 2. Problem addressed

The system solves a **closed-set speaker identification** task:

- the set of speakers is known in advance
- the system receives a short audio recording
- the output is the most likely speaker among the enrolled speakers

This is different from speech recognition / ASR, because the system does **not** try to recognize the spoken words. It tries to recognize the **speaker identity** from the voice signal.

## 3. General solution idea

The implemented solution follows a modern embedding-based speaker-recognition pipeline:

1. collect speech recordings for several target speakers
2. extract and normalize the audio
3. segment the recordings into shorter speaker-specific clips
4. build training and test sets
5. use a **pretrained ECAPA-TDNN speaker embedding model** from SpeechBrain
6. compute a fixed-length embedding for each clip
7. compute one **centroid embedding** per speaker from the training clips
8. for a new input recording, extract its embedding
9. compare it with enrolled speaker centroids using **cosine similarity**
10. return the speaker with the highest similarity score

## 4. Technologies used

### Core machine learning / audio stack
- **Python**
- **PyTorch**
- **SpeechBrain**
- **soundfile**
- **ffmpeg**

### Data collection and preparation
- **yt-dlp** for downloading audio from YouTube
- **ffmpeg** for converting and standardizing audio
- **pyannote.audio** for speaker diarization
- **Audacity** for manual inspection, cleanup, and export of clips

### Application layer
- **FastAPI** for the backend API
- **Uvicorn** as the ASGI server
- **HTML / CSS / JavaScript** for the frontend
- browser **MediaRecorder API** for microphone recording

## 5. Audio data source

The project dataset was created from **YouTube interview-style recordings** containing the target speakers.

### Why interview recordings were chosen
Interview recordings are useful because they usually contain:
- relatively long speech from one identifiable person
- natural speaking style
- multiple utterances with different phonetic content
- more realistic variability than short scripted samples

### Important note
Only recordings that can legally and ethically be used for the academic task should be included in the dataset.

## 6. How the audio was extracted

Audio was downloaded from YouTube using **yt-dlp** and then standardized with **ffmpeg**.

### Typical normalization step
The audio was converted to:
- **WAV**
- **16 kHz**
- **mono**
- **PCM 16-bit**

## 7. Annotation and dataset creation workflow

The dataset creation process had two stages.

### 7.1 Automatic stage
The full interview recordings were first processed with **pyannote.audio speaker diarization**. This produced files containing speaker segments with:
- start time
- end time
- provisional label such as `SPEAKER_00`, `SPEAKER_01`

### 7.2 Manual cleanup stage
The auto-generated labels were then imported into **Audacity**. In Audacity, the following actions were performed:
- listen to a few example segments of each provisional speaker
- determine which diarized speaker corresponds to the target person
- remove interviewer / non-target segments
- remove noisy or unusable fragments
- split very long segments into shorter clips
- rename labels into a clean convention such as:
  - `spk01_int01_001`
  - `spk01_int01_002`

After cleanup, the clips were exported from Audacity in batch mode.

### Target clip characteristics
Most clips were prepared as:
- clean single-speaker speech
- approximately **2 to 6 seconds** long
- without strong overlap, music, or background interference

## 8. File organization

```text
.
  data/
    raw/
    wav/
    labels/
      auto/
      final/
    clips/
      spk01/
        train/
        test/
      spk02/
        train/
        test/
      spk03/
        train/
        test/
    manifests/
      clips.csv
      train.csv
      test.csv
```

### Meaning of the CSV files
- `clips.csv` contains all clips
- `train.csv` contains training clips
- `test.csv` contains evaluation clips

## 9. Train / test split

The split was done by **interview/session**, not by random clip mixing.

Example:
- `int01` clips of each speaker -> `train`
- `int02` clips of each speaker -> `test`

## 10. Recognition model

### 10.1 Embedding model
The project uses a **pretrained SpeechBrain ECAPA-TDNN speaker-recognition model**:
- model source: `speechbrain/spkrec-ecapa-voxceleb`
- role: extract a fixed-length speaker embedding from each input clip

### 10.2 Why a pretrained model was used
Training a large speaker embedding model from scratch would require:
- many more speakers
- much more training data
- more compute
- more time

For a small academic project with three speakers, a pretrained embedding model is much more appropriate.

## 11. Classification backend

The first implemented backend is a **centroid-based classifier**:
1. compute embeddings for all training clips of one speaker
2. average them to obtain one speaker centroid
3. normalize the centroid
4. for a test clip, compute its embedding
5. compute cosine similarity between the test embedding and each centroid
6. choose the speaker with the highest score

## 12. Current result

A baseline run on the prepared train/test split achieved:
- **Accuracy: 0.9955**
- **891 / 895 correct predictions**

Confusion matrix summary:
- `spk01`: 236 / 236 correct
- `spk02`: 294 / 296 correct
- `spk03`: 361 / 363 correct

## 13. Backend and frontend application

### 13.1 Backend
The backend was built with **FastAPI** and exposes an endpoint for speaker prediction.

Main responsibilities:
- receive uploaded audio
- receive microphone recordings from the frontend
- convert input to 16 kHz mono WAV with ffmpeg
- extract a speaker embedding
- compare it against enrolled centroids
- return predicted speaker and similarity scores

### 13.2 Frontend
The frontend is a small browser-based interface built with:
- HTML
- CSS
- JavaScript

It supports:
- uploading an audio file
- recording a short sample with the laptop microphone
- sending the audio to the backend
- displaying the predicted speaker and similarity scores

## 14. How the system works end-to-end

### Offline preparation stage
1. collect long recordings
2. extract and normalize audio
3. diarize full recordings
4. clean the speaker labels
5. export speaker clips
6. prepare train/test CSV manifests
7. build speaker centroids

### Inference stage
1. user uploads or records audio
2. backend converts audio to the expected format
3. backend extracts a SpeechBrain embedding
4. backend computes cosine similarity to each speaker centroid
5. frontend displays the most likely speaker

## 15. Why this approach makes sense

This solution is well suited to the project because it is:
- **practical**
- **interpretable**
- **fast**
- **modular**
- **easy to demo**

## 16. Limitations

- only **three speakers** are enrolled
- the problem is **closed-set identification**
- the test data comes from a limited number of sessions
- some clips may still contain noise or channel mismatch
- very short or poor-quality utterances are harder to classify
- the current demo is not full real-time streaming

## 17. Possible future improvements

- train a small classifier such as **Logistic Regression** on top of embeddings
- compare centroid classification vs. classifier-based backend
- add confidence thresholds
- support unknown-speaker rejection
- add real-time chunk-based streaming inference
- evaluate performance by clip duration
- add spoofing / replay robustness checks
- improve dataset diversity with more sessions per speaker

## 18. Conclusion

The project successfully implements a compact speaker-recognition system for three speakers using a modern embedding-based approach.

The final system:
- uses YouTube interviews as a speech source
- extracts and cleans speaker-specific clips
- uses SpeechBrain ECAPA-TDNN embeddings
- classifies speakers with centroid-based cosine scoring
- achieves very high accuracy on the prepared test set
- provides a usable demo through a simple backend and frontend application
