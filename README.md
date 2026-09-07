# whisper-tools

Small wrapper around faster-whisper and OpenAI-compatible APIs for transcribing audio files, raw audio, and microphone streams.

## Install

```bash
pip install whisper-tools
```

Optional extras:

```bash
pip install whisper-tools[diarize]   # speaker diarization
pip install whisper-tools[yaml]      # YAML config files
```

## Usage

### Local transcription

```python
from whisper_tools import WhisperLocal

whisper = WhisperLocal(model="base", language="ru")
result = whisper.transcribe("audio.wav")

print(result.text)
```

### API transcription

```python
from whisper_tools import WhisperAPI

whisper = WhisperAPI(api_key="...", base_url="...")
result = whisper.transcribe("audio.wav")

print(result.text)
```

### Microphone streaming

```python
from whisper_tools import WhisperLocal, StreamRecorder

recorder = StreamRecorder(WhisperLocal(), on_text=lambda r: print(r.text))
recorder.start()
try:
    input("Press Enter to stop\n")
finally:
    recorder.stop()
```

Or poll manually:

```python
recorder = StreamRecorder(WhisperLocal())
recorder.start()
try:
    while True:
        result = recorder.process()
        if result:
            print(result.text)
except KeyboardInterrupt:
    pass
finally:
    recorder.stop()
```

### Segments and timestamps

```python
result = whisper.transcribe("audio.wav", word_timestamps=True)
for segment in result.segments:
    print(f"[{segment.start:.1f}-{segment.end:.1f}] {segment.text}")
```

### Language auto-detection

```python
whisper = WhisperLocal(language=None)  # or WhisperAPI(language=None)
```

### Speaker diarization

```python
from whisper_tools import diarize

turns = diarize("audio.wav", hf_token="hf_...")
for turn in turns:
    print(f"{turn.speaker}: {turn.start:.1f}-{turn.end:.1f}")
```

### CLI

```bash
whisper-tools audio.wav
whisper-tools --json audio.wav
whisper-tools --stream
whisper-tools --list-devices
whisper-tools --version
```

## API

### `WhisperLocal(model="base", language="ru", device="auto", chunk_seconds=30.0, overlap_seconds=2.0, noise_reduction=0.0)`

- `transcribe(audio, sample_rate=16000, prompt=None, word_timestamps=False)` — accepts a file path or a numpy array. Returns a `TranscriptionResult`.
- `transcribe_many(audio_files, num_workers=1)` — parallel transcription of multiple files.

### `WhisperAPI(api_key, base_url, model="whisper-large-v3", language="ru", max_retries=2, timeout=60.0, noise_reduction=0.0)`

- `transcribe(audio, sample_rate=16000, prompt=None)` — accepts a file path or a numpy array. Returns a `TranscriptionResult`.
- `transcribe_async(audio, sample_rate=16000, prompt=None)` — async version.
- `transcribe_many(audio_files, num_workers=4)` — parallel transcription of multiple files.

### `StreamRecorder(transcriber, sample_rate=16000, chunk_seconds=2.1, max_chunk_seconds=5.0, adapt_step=0.5, min_interval=1.0, energy_threshold=1e-5, dynamic_threshold=5e-3, max_queue_size=20, on_text=None)`

- `start()` / `stop()` — begin and end microphone capture.
- `process()` — transcribe the next chunk if speech is detected, otherwise return `None`.
- `device` — input device index (see `list_devices()`).
- `lagging` — True when transcription is slower than real time.

### `Diarizer(hf_token, model="pyannote/speaker-diarization-3.1")`

- `diarize(audio, sample_rate=16000)` — returns a list of `SpeakerTurn`.

### `list_devices()`

Returns `(index, name)` pairs for available input devices.

### Result types

- `TranscriptionResult(text, segments, language, duration, finalized)` — `text` is the full transcript, `segments` is a list of `Segment`.
- `Segment(start, end, text, words)` — `words` is a list of `(start, end, word)` tuples when word timestamps are enabled.
- `SpeakerTurn(start, end, speaker)` — a speaker turn from diarization.

## License

MIT