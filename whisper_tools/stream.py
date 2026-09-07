import queue
import threading
import time
import typing as tp

import numpy as np

from whisper_tools.types import TranscriptionResult

_SENTENCE_ENDINGS = (".", "!", "?", "…")


def list_devices() -> tp.List[tp.Tuple[int, str]]:
    """
    List available audio input devices

    Returns
    -------
    devices : list of (int, str)
        Pairs of device index and device name.
    """
    import sounddevice as sd

    return [
        (i, d["name"])
        for i, d in enumerate(sd.query_devices())
        if d["max_input_channels"] > 0
    ]


class StreamRecorder:
    """
    Record from the microphone and transcribe chunks as speech is detected

    Supports both pull-based polling via :meth:`process` and push-based
    delivery via the `on_text` callback.

    Parameters
    ----------
    transcriber : object
        Any object with a ``transcribe(audio, sample_rate)`` method that
        returns a :class:`TranscriptionResult`. For example
        :class:`WhisperLocal` or :class:`WhisperAPI`.
    sample_rate : int, default 16000
        Sample rate for recording.
    chunk_seconds : float, default 2.1
        Initial chunk length in seconds.
    max_chunk_seconds : float, default 5.0
        Upper bound for adaptive chunk growth.
    adapt_step : float, default 0.5
        How much the chunk length grows or shrinks per adaptation.
    min_interval : float, default 1.0
        Minimum time between transcription attempts.
    energy_threshold : float, default 1e-5
        Minimum mean squared energy for speech detection.
    dynamic_threshold : float, default 5e-3
        Minimum peak-to-peak amplitude for speech detection.
    max_queue_size : int, default 20
        Maximum number of buffered audio blocks. When exceeded,
        the oldest blocks are dropped and `lagging` is set to True.
    on_text : callable | None, default None
        Optional callback invoked with each :class:`TranscriptionResult`
        in a background thread. When set, :meth:`process` is not needed.

    See Also
    --------
    WhisperLocal
    WhisperAPI
    """

    def __init__(
        self,
        transcriber: tp.Any,
        sample_rate: int = 16000,
        chunk_seconds: float = 2.1,
        max_chunk_seconds: float = 5.0,
        adapt_step: float = 0.5,
        min_interval: float = 1.0,
        energy_threshold: float = 1e-5,
        dynamic_threshold: float = 5e-3,
        max_queue_size: int = 20,
        on_text: tp.Optional[tp.Callable[[TranscriptionResult], None]] = None,
    ) -> None:
        self.transcriber = transcriber
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds
        self.max_chunk_seconds = max_chunk_seconds
        self.adapt_step = adapt_step
        self.min_interval = min_interval
        self.energy_threshold = energy_threshold
        self.dynamic_threshold = dynamic_threshold
        self.max_queue_size = max_queue_size
        self.on_text = on_text
        self.device: tp.Optional[int] = None
        self.lagging = False

        self._recording = False
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self._buffer = np.array([], dtype=np.float32)
        self._pending: tp.Optional[TranscriptionResult] = None
        self._last = 0.0
        self._record_thread: tp.Optional[threading.Thread] = None
        self._push_thread: tp.Optional[threading.Thread] = None

    def start(self) -> None:
        self._recording = True
        self._buffer = np.array([], dtype=np.float32)
        self._pending = None
        self._last = 0.0
        self.lagging = False

        self._record_thread = threading.Thread(target=self._record, daemon=True)
        self._record_thread.start()

        if self.on_text is not None:
            self._push_thread = threading.Thread(target=self._push_loop, daemon=True)
            self._push_thread.start()

    def stop(self) -> None:
        self._recording = False

    def process(self) -> tp.Optional[TranscriptionResult]:
        """
        Transcribe the next chunk if speech is present

        Returns
        -------
        result : TranscriptionResult | None
            Transcription result, or None when there is no speech yet.
        """
        chunk = self._next_chunk()
        if chunk is None:
            return None

        result = self.transcriber.transcribe(chunk, self.sample_rate)
        if not _is_finalized(result):
            self._pending = result
            return result

        if self._pending is not None:
            merged = _merge_results(self._pending, result)
            self._pending = None
            return merged
        return result

    def _push_loop(self) -> None:
        """Background loop that delivers results through `on_text`"""
        while self._recording:
            result = self.process()
            if result is not None and self.on_text is not None:
                self.on_text(result)
            time.sleep(0.05)

    def _record(self) -> None:
        """Audio capture loop running in a background thread"""
        import sounddevice as sd

        def callback(indata, frames, time, status) -> None:
            if not self._recording:
                return
            if self._queue.qsize() >= self.max_queue_size:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                self.lagging = True
            self._queue.put(indata[:, 0].astype(np.float32))

        with sd.InputStream(
            device=self.device,
            channels=1,
            samplerate=self.sample_rate,
            callback=callback,
            blocksize=int(self.sample_rate * 0.1),
        ):
            while self._recording:
                sd.sleep(100)

    def _next_chunk(self) -> tp.Optional[np.ndarray]:
        """Extract the next audio chunk, adapting chunk length to speech"""
        frames = int(self.sample_rate * self.chunk_seconds)
        while not self._queue.empty() and len(self._buffer) < frames * 2:
            self._buffer = np.concatenate([self._buffer, self._queue.get_nowait()])

        now = time.time()
        if now - self._last < self.min_interval or len(self._buffer) < frames:
            return None

        chunk = self._buffer[:frames]
        self._buffer = self._buffer[frames:]

        if _has_speech(chunk, self.energy_threshold, self.dynamic_threshold):
            self._last = now
            self.chunk_seconds = min(
                self.chunk_seconds + self.adapt_step, self.max_chunk_seconds
            )
            return chunk

        self.chunk_seconds = max(self.chunk_seconds - self.adapt_step, 2.0)
        return None


def _has_speech(
    audio: np.ndarray,
    energy_threshold: float = 1e-5,
    dynamic_threshold: float = 5e-3,
) -> bool:
    if len(audio) == 0:
        return False
    return np.mean(audio**2) > energy_threshold and np.ptp(audio) > dynamic_threshold


def _is_finalized(result: TranscriptionResult) -> bool:
    """Whether a result ends with sentence-ending punctuation"""
    return result.text.rstrip().endswith(_SENTENCE_ENDINGS)


def _merge_results(
    first: TranscriptionResult, second: TranscriptionResult
) -> TranscriptionResult:
    """Merge a partial hypothesis with the following finalized result"""
    return TranscriptionResult(
        text=f"{first.text} {second.text}".strip(),
        segments=first.segments + second.segments,
        language=second.language or first.language,
        duration=second.duration,
        finalized=True,
    )