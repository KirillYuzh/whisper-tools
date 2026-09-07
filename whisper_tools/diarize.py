import typing as tp

import numpy as np

from whisper_tools.audio import TARGET_SAMPLE_RATE, load_audio, resample
from whisper_tools.types import SpeakerTurn


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
        """Load and cache the pyannote pipeline."""
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
        import torch

        if isinstance(audio, str):
            data, rate = load_audio(audio, TARGET_SAMPLE_RATE)
        else:
            data = np.asarray(audio, dtype=np.float32)
            if data.ndim > 1:
                data = data[:, 0]
            if sample_rate != TARGET_SAMPLE_RATE:
                data = resample(data, sample_rate, TARGET_SAMPLE_RATE)
            rate = TARGET_SAMPLE_RATE

        waveform = torch.from_numpy(data).unsqueeze(0)
        diarization = self._load()({"waveform": waveform, "sample_rate": rate})

        turns = [
            SpeakerTurn(start=turn.start, end=turn.end, speaker=speaker)
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]
        return sorted(turns, key=lambda t: t.start)


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