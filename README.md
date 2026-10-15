# YTP Finder

A local tool for making YTP (YouTube Poop) edits. It transcribes a folder of videos, lets you search for a word, a syllable or an IPA sound across all of them, and exports the matching clips.

## Features

- Word-level transcription with [faster-whisper](https://github.com/SYSTRAN/faster-whisper), with voice activity detection to avoid hallucinations on music and silence
- Phoneme-level forced alignment (wav2vec2 espeak phoneme model) for precise cut points
- Search by spelling, by sound (espeak-ng phonetics) or directly in IPA
- Per-clip reliability score, sorting and filtering
- Speaker grouping: phrases are clustered by voice, you can listen to each voice, name it after the character and filter results by character
- Basket to order clips, preview the sequence and export it as MP4 or WAV (one file or one file per clip)

## Requirements

- Python 3.10+
- [FFmpeg](https://ffmpeg.org/) on the `PATH`
- [eSpeak NG](https://github.com/espeak-ng/espeak-ng) (needed for phonetic search and alignment)
- An NVIDIA GPU is recommended; CPU works but is much slower

```bash
pip install -r requirements.txt
```

Models (Whisper, the alignment model and the voice embedding model) are downloaded from Hugging Face on first use.

## Usage

```bash
python app.py
```

or double-click `Start YTP Finder.bat` on Windows. The app opens at http://127.0.0.1:8765.

1. Paste the path of a folder containing your videos and click **Index folder**.
2. Type a word, a syllable or IPA in the search bar.
3. Hover a result to preview it, add clips to the basket, then export.
4. Use **Name voices…** to name each detected voice and filter by character.

Transcriptions are cached in `cache/`, exports go to `exports/`.
