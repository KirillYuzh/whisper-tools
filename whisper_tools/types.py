import typing as tp
from dataclasses import dataclass


@dataclass
class Segment:
    """
    A timed piece of transcribed text

    Parameters
    ----------
    start : float
        Start time in seconds.
    end : float
        End time in seconds.
    text : str
        Transcribed text.
    words : list of (start, end, word) | None, default None
        Word-level timestamps when requested.
    """

    start: float
    end: float
    text: str
    words: tp.Optional[tp.List[tp.Tuple[float, float, str]]] = None


@dataclass
class TranscriptionResult:
    """
    Result of a transcription call

    Parameters
    ----------
    text : str
        Full transcribed text.
    segments : list of Segment
        Timed segments.
    language : str | None, default None
        Detected or requested language code.
    duration : float, default 0.0
        Audio duration in seconds.
    finalized : bool, default True
        Whether the text is a complete utterance. Set by
        :class:`StreamRecorder` for partial hypotheses.
    """

    text: str
    segments: tp.List[Segment]
    language: tp.Optional[str] = None
    duration: float = 0.0
    finalized: bool = True


@dataclass
class SpeakerTurn:
    """
    A speaker turn from diarization

    Parameters
    ----------
    start : float
        Start time in seconds.
    end : float
        End time in seconds.
    speaker : str
        Speaker label.
    """

    start: float
    end: float
    speaker: str