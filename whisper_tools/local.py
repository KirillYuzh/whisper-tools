import typing as tp
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from whisper_tools.audio import TARGET_SAMPLE_RATE, load_audio, reduce_noise, resample, split_chunks
from whisper_tools.types import Segment, TranscriptionResult


class WhisperLocal:
    """
    Local speech-to-text transcription powered by faster-whisper

    Parameters
    ----------
    model : str, default "base"
        Name or path of the Whisper model.
    language : str | None, default "ru"
        Language code to transcribe in. ``None`` lets faster-whisper
        auto-detect the language.
    device : str, default "auto"
        Device to run inference on. One of "auto", "cpu", "cuda", "mps".
    chunk_seconds : float, default 30.0
        Chunk length in seconds for long-file transcription.
    overlap_seconds : float, default 2.0
        Overlap between neighbouring chunks in seconds.
    noise_reduction : float, default 0.0
        Noise reduction strength. 0 disables the effect.

    See Also
    --------
    WhisperAPI
    StreamRecorder
    """

    def __init__(
        self,
        model: str = "base",
        language: tp.Optional[str] = "ru",
        device: str = "auto",
        chunk_seconds: float = 30.0,
        overlap_seconds: float = 2.0,
        noise_reduction: float = 0.0,
    ) -> None:
        self.model_name = model
        self.language = language
        self.device = device
        self.chunk_seconds = chunk_seconds
        self.overlap_seconds = overlap_seconds
        self.noise_reduction = noise_reduction
        self._model = None

    def _load(self) -> tp.Any:
        """Load and cache the faster-whisper model."""
        if self._model is None:
            from faster_whisper import WhisperModel

            device = self.device
            if device == "auto":
                device = "cuda" if _cuda_available() else "cpu"
            compute = "float16" if device in ("cuda", "mps") else "int8"
            self._model = WhisperModel(
                self.model_name, device=device, compute_type=compute
            )
        return self._model

    def transcribe(
        self,
        audio: tp.Union[str, np.ndarray],
        sample_rate: int = 16000,
        prompt: tp.Optional[str] = None,
        word_timestamps: bool = False,
    ) -> TranscriptionResult:
        """
        Transcribe an audio file or a numpy array

        Parameters
        ----------
        audio : str | ndarray
            Path to an audio file or raw audio samples.
        sample_rate : int, default 16000
            Sample rate of `audio` when it is a numpy array.
        prompt : str | None, default None
            Initial prompt to bias the model vocabulary.
        word_timestamps : bool, default False
            Whether to include word-level timestamps in segments.

        Returns
        -------
        result : TranscriptionResult
            Transcribed text with segments, language and duration.
        """
        if isinstance(audio, str):
            data, rate = load_audio(audio, TARGET_SAMPLE_RATE)
        else:
            data = np.asarray(audio, dtype=np.float32)
            if data.ndim > 1:
                data = data[:, 0]
            if sample_rate != TARGET_SAMPLE_RATE:
                data = resample(data, sample_rate, TARGET_SAMPLE_RATE)
            rate = TARGET_SAMPLE_RATE

        if self.noise_reduction > 0:
            data = reduce_noise(data, rate, self.noise_reduction)

        duration = len(data) / rate
        if duration <= self.chunk_seconds:
            return self._transcribe_chunk(data, rate, prompt, word_timestamps, offset=0.0)

        chunks = split_chunks(data, rate, self.chunk_seconds, self.overlap_seconds)
        segments = []
        language = None
        for i, chunk in enumerate(chunks):
            offset = i * (self.chunk_seconds - self.overlap_seconds)
            result = self._transcribe_chunk(chunk, rate, prompt, word_timestamps, offset)
            segments.extend(result.segments)
            language = language or result.language

        text = " ".join(s.text for s in segments if s.text)
        return TranscriptionResult(text=text, segments=segments, language=language, duration=duration)

    def _transcribe_chunk(
        self,
        audio: np.ndarray,
        sample_rate: int,
        prompt: tp.Optional[str],
        word_timestamps: bool,
        offset: float,
    ) -> TranscriptionResult:
        kwargs = {
            "language": self.language,
            "vad_filter": True,
            "word_timestamps": word_timestamps,
        }
        if prompt is not None:
            kwargs["initial_prompt"] = prompt

        segments_iter, info = self._load().transcribe(audio, **kwargs)

        segments = []
        for seg in segments_iter:
            words = None
            if word_timestamps and seg.words:
                words = [(w.start + offset, w.end + offset, w.word) for w in seg.words]
            segments.append(
                Segment(
                    start=seg.start + offset,
                    end=seg.end + offset,
                    text=seg.text.strip(),
                    words=words,
                )
            )

        text = " ".join(s.text for s in segments if s.text)
        return TranscriptionResult(
            text=text,
            segments=segments,
            language=info.language,
            duration=len(audio) / sample_rate,
        )

    def transcribe_many(
        self,
        audio_files: tp.List[str],
        num_workers: int = 1,
    ) -> tp.List[TranscriptionResult]:
        """
        Transcribe multiple audio files in parallel

        A single model instance is not thread-safe, so each worker
        gets its own :class:`WhisperLocal` instance.

        Parameters
        ----------
        audio_files : list of str
            Paths to audio files.
        num_workers : int, default 1
            Number of parallel workers.

        Returns
        -------
        results : list of TranscriptionResult
            Transcription results in the same order as `audio_files`.
        """
        def _transcribe(path: str) -> TranscriptionResult:
            worker = WhisperLocal(
                model=self.model_name,
                language=self.language,
                device=self.device,
                chunk_seconds=self.chunk_seconds,
                overlap_seconds=self.overlap_seconds,
                noise_reduction=self.noise_reduction,
            )
            return worker.transcribe(path)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            return list(executor.map(_transcribe, audio_files))


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False