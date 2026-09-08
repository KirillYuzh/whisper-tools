import asyncio
import time
import typing as tp
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from whisper_tools.audio import TARGET_SAMPLE_RATE, cleanup_temp, convert_to_wav, load_audio, reduce_noise, resample, save_temp_wav
from whisper_tools.types import Segment, TranscriptionResult


class WhisperAPI:
    """
    Wrapper around an OpenAI-compatible transcription API

    Parameters
    ----------
    api_key : str
        API key for the transcription service.
    base_url : str
        Base URL of the OpenAI-compatible API endpoint.
    model : str, default "whisper-large-v3"
        Whisper model name to use.
    language : str | None, default "ru"
        Language code for transcription. ``None`` lets the server
        auto-detect the language.
    max_retries : int, default 2
        Number of retries for transient failures.
    timeout : float, default 60.0
        Request timeout in seconds.
    noise_reduction : float, default 0.0
        Noise reduction strength. 0 disables the effect.
    hf_token : str | None, default None
        Hugging Face token for speaker diarization. Required when
        ``diarize=True`` is passed to :meth:`transcribe`.

    See Also
    --------
    WhisperLocal
    StreamRecorder
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str = "whisper-large-v3",
        language: tp.Optional[str] = "ru",
        max_retries: int = 2,
        timeout: float = 60.0,
        noise_reduction: float = 0.0,
        hf_token: tp.Optional[str] = None,
    ) -> None:
        from openai import OpenAI

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            max_retries=max_retries,
            timeout=timeout,
        )
        self.model = model
        self.language = language
        self.max_retries = max_retries
        self.timeout = timeout
        self.noise_reduction = noise_reduction
        self.hf_token = hf_token
        self._async_client = None

    def _get_async_client(self) -> tp.Any:
        """
        Create and cache the async OpenAI client

        Returns
        -------
        async_client : object
            An instance of AsyncOpenAI for making asynchronous requests.
        """
        if self._async_client is None:
            from openai import AsyncOpenAI

            self._async_client = AsyncOpenAI(
                api_key=self.client.api_key,
                base_url=self.client.base_url,
                max_retries=self.max_retries,
                timeout=self.timeout,
            )
        return self._async_client

    def transcribe(
        self,
        audio: tp.Union[str, np.ndarray],
        sample_rate: int = 16000,
        prompt: tp.Optional[str] = None,
        diarize: bool = False,
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
        diarize : bool, default False
            Whether to run speaker diarization and assign speaker labels
            to transcription segments.

        Returns
        -------
        result : TranscriptionResult
            Transcribed text with segments, language and duration.
        """
        path = self._prepare_audio(audio, sample_rate)
        try:
            result = self._request_with_retry(path, prompt)
            if diarize and self.hf_token is not None:
                from whisper_tools.diarize import Diarizer, assign_speakers

                diarizer = Diarizer(hf_token=self.hf_token)
                # Load the original audio for diarization
                if isinstance(audio, str):
                    turns = diarizer.diarize(audio)
                else:
                    turns = diarizer.diarize(audio, sample_rate=sample_rate)
                assign_speakers(result.segments, turns)
            return result
        finally:
            cleanup_temp(path)

    async def transcribe_async(
        self,
        audio: tp.Union[str, np.ndarray],
        sample_rate: int = 16000,
        prompt: tp.Optional[str] = None,
        diarize: bool = False,
    ) -> TranscriptionResult:
        """
        Asynchronously transcribe an audio file or a numpy array

        Parameters
        ----------
        audio : str | ndarray
            Path to an audio file or raw audio samples.
        sample_rate : int, default 16000
            Sample rate of `audio` when it is a numpy array.
        prompt : str | None, default None
            Initial prompt to bias the model vocabulary.
        diarize : bool, default False
            Whether to run speaker diarization and assign speaker labels
            to transcription segments.

        Returns
        -------
        result : TranscriptionResult
            Transcribed text with segments, language and duration.
        """
        path = self._prepare_audio(audio, sample_rate)
        try:
            result = await self._request_async_with_retry(path, prompt)
            if diarize:
                # WhisperAPI doesn't support diarization natively
                pass
            return result
        finally:
            cleanup_temp(path)

    def transcribe_many(
        self,
        audio_files: tp.List[str],
        num_workers: int = 4,
    ) -> tp.List[TranscriptionResult]:
        """
        Transcribe multiple audio files in parallel

        The OpenAI client is thread-safe, so a single instance can
        be shared across workers.

        Parameters
        ----------
        audio_files : list of str
            Paths to audio files.
        num_workers : int, default 4
            Number of parallel workers.

        Returns
        -------
        results : list of TranscriptionResult
            Transcription results in the same order as `audio_files`.
        """
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            return list(executor.map(self.transcribe, audio_files))

    def _prepare_audio(
        self,
        audio: tp.Union[str, np.ndarray],
        sample_rate: int,
    ) -> str:
        if isinstance(audio, str):
            converted = None
            if not audio.lower().endswith(".wav"):
                converted = convert_to_wav(audio)
                wav_path = converted
            else:
                wav_path = audio
            data, rate = load_audio(wav_path, TARGET_SAMPLE_RATE)
            if self.noise_reduction > 0:
                data = reduce_noise(data, rate, self.noise_reduction)
            result_path = save_temp_wav(data, TARGET_SAMPLE_RATE)
            if converted is not None:
                cleanup_temp(converted)
            return result_path

        data = np.asarray(audio, dtype=np.float32)
        if data.ndim > 1:
            data = data[:, 0]
        if sample_rate != TARGET_SAMPLE_RATE:
            data = resample(data, sample_rate, TARGET_SAMPLE_RATE)
        if self.noise_reduction > 0:
            data = reduce_noise(data, TARGET_SAMPLE_RATE, self.noise_reduction)
        return save_temp_wav(data, TARGET_SAMPLE_RATE)

    def _request_with_retry(self, path: str, prompt: tp.Optional[str]) -> TranscriptionResult:
        """
        Send a transcription request with exponential backoff

        Parameters
        ----------
        path : str
            Path to the audio file.
        prompt : str | None
            Prompt for the transcription.

        Returns
        -------
        result : TranscriptionResult
            Transcription result.
        """
        for attempt in range(self.max_retries + 1):
            try:
                return self._request(path, prompt)
            except Exception:
                if attempt == self.max_retries:
                    raise
                time.sleep(2**attempt)

    async def _request_async_with_retry(
        self, path: str, prompt: tp.Optional[str]
    ) -> TranscriptionResult:
        """
        Send an async transcription request with exponential backoff
        
        Parameters
        ----------
        path : str
            Path to the audio file.
        prompt : str | None
            Prompt for the transcription.
        
        Returns
        -------
        result : TranscriptionResult
            Transcription result.
        """
        for attempt in range(self.max_retries + 1):
            try:
                return await self._request_async(path, prompt)
            except Exception:
                if attempt == self.max_retries:
                    raise
                await asyncio.sleep(2**attempt)

    def _request(self, path: str, prompt: tp.Optional[str]) -> TranscriptionResult:
        """
        ## Sync
        Perform a single sync transcription request

        Parameters
        ----------
        path : str
            Path to the audio file.
        prompt : str | None
            Prompt for the transcription.

        Returns
        -------
        result : TranscriptionResult
            Transcription result.
        """
        kwargs = {
            "model": self.model,
            "response_format": "verbose_json",
        }
        if self.language is not None:
            kwargs["language"] = self.language
        if prompt is not None:
            kwargs["prompt"] = prompt

        with open(path, "rb") as f:
            result = self.client.audio.transcriptions.create(file=f, **kwargs)

        return _parse_response(result)

    async def _request_async(self, path: str, prompt: tp.Optional[str]) -> TranscriptionResult:
        """
        ## Async
        Perform a single async transcription request

        Parameters
        ----------
        path : str
            Path to the audio file.
        prompt : str | None
            Prompt for the transcription.
    
        Returns
        -------
        result : TranscriptionResult
            Transcription result.
        """
        kwargs = {
            "model": self.model,
            "response_format": "verbose_json",
        }
        if self.language is not None:
            kwargs["language"] = self.language
        if prompt is not None:
            kwargs["prompt"] = prompt

        with open(path, "rb") as f:
            result = await self._get_async_client().audio.transcriptions.create(file=f, **kwargs)

        return _parse_response(result)


def _parse_response(response: tp.Any) -> TranscriptionResult:
    """Convert an OpenAI verbose_json response into a TranscriptionResult"""
    segments = []
    for seg in getattr(response, "segments", []) or []:
        segments.append(
            Segment(
                start=float(seg.get("start", 0.0)),
                end=float(seg.get("end", 0.0)),
                text=str(seg.get("text", "")).strip(),
            )
        )

    text = str(getattr(response, "text", "")).strip()
    if not text and segments:
        text = " ".join(s.text for s in segments if s.text)

    return TranscriptionResult(
        text=text,
        segments=segments,
        language=getattr(response, "language", None),
        duration=float(getattr(response, "duration", 0.0) or 0.0),
    )