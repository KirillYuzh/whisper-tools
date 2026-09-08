# whisper-tools

Small wrapper around faster-whisper and OpenAI-compatible APIs for transcribing audio files, raw audio, and microphone streams.

[**Practical demo**](https://github.com/KirillYuzh/whisper-tools/blob/main/whisper_tools_demo.ipynb)

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

### WhisperLocal

Transcribes audio locally using faster-whisper.

**Constructor**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | `"base"` | Whisper model name |
| `language` | `str` | `"ru"` | Language code |
| `device` | `str` | `"auto"` | Device to run on |
| `chunk_seconds` | `float` | `30.0` | Audio chunk size in seconds |
| `overlap_seconds` | `float` | `2.0` | Overlap between chunks |
| `noise_reduction` | `float` | `0.0` | Noise reduction level |
| `hf_token` | `str \| None` | `None` | Hugging Face token for diarization |

**Methods**

`transcribe(audio, sample_rate=16000, prompt=None, word_timestamps=False, diarize=False)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `audio` | `str \| numpy.ndarray` | — | File path or audio array |
| `sample_rate` | `int` | `16000` | Audio sample rate |
| `prompt` | `str \| None` | `None` | Initial prompt for decoding |
| `word_timestamps` | `bool` | `False` | Enable word-level timestamps |
| `diarize` | `bool` | `False` | Enable speaker diarization (requires `hf_token`) |

Returns `TranscriptionResult`. When `diarize=True`, each segment gets an optional `speaker` field.

`transcribe_many(audio_files, num_workers=1)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `audio_files` | `list[str]` | — | List of audio file paths |
| `num_workers` | `int` | `1` | Number of parallel workers |

Returns `list[TranscriptionResult]`.

**Example**

```python
from whisper_tools import WhisperLocal

whisper = WhisperLocal(model="base", language="ru")
result = whisper.transcribe("audio.wav", word_timestamps=True)
print(result.text)
```

---

### WhisperAPI

Transcribes audio via an OpenAI-compatible API.

**Constructor**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `api_key` | `str` | — | API key |
| `base_url` | `str` | — | Base URL of the API |
| `model` | `str` | `"whisper-large-v3"` | Model to use |
| `language` | `str` | `"ru"` | Language code |
| `max_retries` | `int` | `2` | Maximum retry attempts |
| `timeout` | `float` | `60.0` | Request timeout in seconds |
| `noise_reduction` | `float` | `0.0` | Noise reduction level |
| `hf_token` | `str \| None` | `None` | Hugging Face token for diarization |

**Methods**

`transcribe(audio, sample_rate=16000, prompt=None, diarize=False)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `audio` | `str \| numpy.ndarray` | — | File path or audio array |
| `sample_rate` | `int` | `16000` | Audio sample rate |
| `prompt` | `str \| None` | `None` | Initial prompt for decoding |
| `diarize` | `bool` | `False` | Enable speaker diarization (requires `hf_token`) |

Returns `TranscriptionResult`. When `diarize=True`, each segment gets an optional `speaker` field.

`transcribe_async(audio, sample_rate=16000, prompt=None, diarize=False)`

Async version of `transcribe`. Returns `TranscriptionResult`.

`transcribe_many(audio_files, num_workers=4)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `audio_files` | `list[str]` | — | List of audio file paths |
| `num_workers` | `int` | `4` | Number of parallel workers |

Returns `list[TranscriptionResult]`.

**Example**

```python
from whisper_tools import WhisperAPI

whisper = WhisperAPI(api_key="...", base_url="...")
result = whisper.transcribe("audio.wav")
print(result.text)
```

---

### StreamRecorder

Records from a microphone and transcribes in real time.

**Constructor**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `transcriber` | `WhisperLocal \| WhisperAPI` | — | Transcription backend |
| `sample_rate` | `int` | `16000` | Audio sample rate |
| `chunk_seconds` | `float` | `2.1` | Chunk duration |
| `max_chunk_seconds` | `float` | `5.0` | Maximum chunk duration |
| `adapt_step` | `float` | `0.5` | Adaptation step |
| `min_interval` | `float` | `1.0` | Minimum interval between transcriptions |
| `energy_threshold` | `float` | `1e-5` | Energy threshold for speech detection |
| `dynamic_threshold` | `float` | `5e-3` | Dynamic threshold multiplier |
| `max_queue_size` | `int` | `20` | Maximum queue size |
| `on_text` | `callable \| None` | `None` | Callback for transcribed text |

**Methods**

`start()` / `stop()` — Begin and end microphone capture.

`process()` — Transcribe the next chunk if speech is detected; otherwise returns `None`.

**Properties**

| Property | Type | Description |
|---|---|---|
| `device` | `int` | Input device index (see `list_devices()`) |
| `lagging` | `bool` | True when transcription is slower than real time |

**Example**

```python
from whisper_tools import WhisperLocal, StreamRecorder

recorder = StreamRecorder(WhisperLocal(), on_text=lambda r: print(r.text))
recorder.start()
try:
    input("Press Enter to stop\n")
finally:
    recorder.stop()
```

---

### Diarizer

Performs speaker diarization on audio.

**Constructor**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `hf_token` | `str` | — | Hugging Face token |
| `model` | `str` | `"pyannote/speaker-diarization-3.1"` | Diarization model |

**Methods**

`diarize(audio, sample_rate=16000)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `audio` | `str` | — | Audio file path |
| `sample_rate` | `int` | `16000` | Audio sample rate |

Returns `list[SpeakerTurn]`.

**Example**

```python
from whisper_tools import diarize

turns = diarize("audio.wav", hf_token="hf_...")
for turn in turns:
    print(f"{turn.speaker}: {turn.start:.1f}-{turn.end:.1f}")
```

---

### list_devices()

Returns available input devices.

**Returns** `list[tuple[int, str]]` — `(index, name)` pairs.

---

### Result Types

**TranscriptionResult**

| Field | Type | Description |
|---|---|---|
| `text` | `str` | Full transcript |
| `segments` | `list[Segment]` | List of transcription segments |
| `language` | `str` | Detected language |
| `duration` | `float` | Audio duration |
| `finalized` | `bool` | Whether transcription is finalized |

**Segment**

| Field | Type | Description |
|---|---|---|
| `start` | `float` | Start time in seconds |
| `end` | `float` | End time in seconds |
| `text` | `str` | Segment text |
| `words` | `list[tuple[float, float, str]]` | Word tuples `(start, end, word)` when word timestamps enabled |

**SpeakerTurn**

| Field | Type | Description |
|---|---|---|
| `start` | `float` | Start time in seconds |
| `end` | `float` | End time in seconds |
| `speaker` | `str` | Speaker label |

## License

MIT