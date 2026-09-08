import typing as tp

import numpy as np

from whisper_tools.audio import TARGET_SAMPLE_RATE, cleanup_temp, convert_to_wav, load_audio, resample
from whisper_tools.types import Segment, SpeakerTurn


class Diarizer:
    """
    Wraps pyannote.audio for speaker diarization

    Parameters
    ----------
    hf_token : str
        Hugging Face access token with access to the diarization model.
    model : str, default "pyannote/speaker-diarization-3.1"
        Name of the pyannote diarization pipeline.

    See Also
    --------
    WhisperLocal
    WhisperAPI
    """

    def __init__(
        self,
        hf_token: str,
        model: str = "pyannote/speaker-diarization-3.1",
    ) -> None:
        self.hf_token = hf_token
        self.model_name = model
        self._pipeline = None

    def _load(self) -> tp.Any:
        """Load and cache the pyannote pipeline"""
        if self._pipeline is None:
            from pyannote.audio import Pipeline

            self._pipeline = Pipeline.from_pretrained(
                self.model_name, use_auth_token=self.hf_token
            )
        return self._pipeline

    def diarize(
        self,
        audio: tp.Union[str, np.ndarray],
        sample_rate: int = 16000,
    ) -> tp.List[SpeakerTurn]:
        """
        Split audio into speaker turns

        Parameters
        ----------
        audio : str | ndarray
            Path to an audio file or raw audio samples.
        sample_rate : int, default 16000
            Sample rate of `audio` when it is a numpy array.

        Returns
        -------
        turns : list of SpeakerTurn
            Speaker turns sorted by start time.
        """
        converted_temp = None
        if isinstance(audio, str):
            if not audio.lower().endswith(".wav"):
                converted_temp = convert_to_wav(audio)
                audio = converted_temp
            data, rate = load_audio(audio, TARGET_SAMPLE_RATE)
        else:
            data = np.asarray(audio, dtype=np.float32)
            if data.ndim > 1:
                data = data[:, 0]
            if sample_rate != TARGET_SAMPLE_RATE:
                data = resample(data, sample_rate, TARGET_SAMPLE_RATE)
            rate = TARGET_SAMPLE_RATE

        import torch

        waveform = torch.from_numpy(data).unsqueeze(0)
        diarization = self._load()({"waveform": waveform, "sample_rate": rate})

        turns = [
            SpeakerTurn(start=turn.start, end=turn.end, speaker=speaker)
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]
        if converted_temp is not None:
            cleanup_temp(converted_temp)
        return sorted(turns, key=lambda t: t.start)


def assign_speakers(
    segments: tp.List[Segment],
    turns: tp.List[SpeakerTurn],
) -> None:
    """Assign speaker labels to transcription segments.

    Each segment gets the speaker whose turn overlaps the most with
    the segment's midpoint.

    Parameters
    ----------
    segments : list of Segment
        Transcription segments (mutated in place).
    turns : list of SpeakerTurn
        Speaker turns from diarization.
    """
    for seg in segments:
        mid = (seg.start + seg.end) / 2
        for turn in turns:
            if turn.start <= mid <= turn.end:
                seg.speaker = turn.speaker
                break
        else:
            best = None
            best_overlap = 0.0
            for turn in turns:
                overlap = min(seg.end, turn.end) - max(seg.start, turn.start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best = turn.speaker
            if best is not None:
                seg.speaker = best


def diarize(
    audio: tp.Union[str, np.ndarray],
    hf_token: str,
    sample_rate: int = 16000,
    model: str = "pyannote/speaker-diarization-3.1",
) -> tp.List[SpeakerTurn]:
    """
    Convenience function for one-off speaker diarization

    Parameters
    ----------
    audio : str | ndarray
        Path to an audio file or raw audio samples.
    hf_token : str
        Hugging Face access token with access to the diarization model.
    sample_rate : int, default 16000
        Sample rate of `audio` when it is a numpy array.
    model : str, default "pyannote/speaker-diarization-3.1"
        Name of the pyannote diarization pipeline.

    Returns
    -------
    turns : list of SpeakerTurn
        Speaker turns sorted by start time.
    """
    return Diarizer(hf_token=hf_token, model=model).diarize(audio, sample_rate)