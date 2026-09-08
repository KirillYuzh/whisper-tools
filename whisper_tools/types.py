import typing as tp
from dataclasses import dataclass


def _fmt_time(seconds: float) -> str:
    """Format a timestamp as ``HH:MM:SS.mmm``."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def to_markdown(result: "TranscriptionResult") -> str:
    """
    Render a :class:`TranscriptionResult` as a Markdown string.

    The output contains a title, the full text as a blockquote, a
    segments table (Start, End, Speaker, Text) and, when available,
    a word-level timestamps sub-table per segment.

    Parameters
    ----------
    result : TranscriptionResult
        The transcription to render.

    Returns
    -------
    md : str
        Markdown-formatted string.
    """
    lines: tp.List[str] = ["# Transcription", ""]

    meta: tp.List[str] = []
    if result.language:
        meta.append(f"Language: `{result.language}`")
    if result.duration:
        meta.append(f"Duration: `{_fmt_time(result.duration)}`")
    if meta:
        lines.append("*" + " | ".join(meta) + "*")
        lines.append("")

    lines.append("> " + result.text.replace("\n", "\n> "))
    lines.append("")

    lines.append("## Segments")
    lines.append("")
    lines.append("| Start | End | Speaker | Text |")
    lines.append("| --- | --- | --- | --- |")

    for seg in result.segments:
        speaker = seg.speaker or ""
        text = seg.text.replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {_fmt_time(seg.start)} | {_fmt_time(seg.end)} "
            f"| {speaker} | {text} |"
        )

        if seg.words:
            lines.append("")
            lines.append(
                f"| {_fmt_time(seg.start)} | {_fmt_time(seg.end)} "
                f"| {speaker} | Words: |"
            )
            for w_start, w_end, word in seg.words:
                w = word.replace("|", "\\|")
                lines.append(
                    f"| {_fmt_time(w_start)} | {_fmt_time(w_end)} "
                    f"| {speaker} | &nbsp;&nbsp;{w} |"
                )

    lines.append("")
    return "\n".join(lines)


def save_markdown(path: str, result: "TranscriptionResult") -> None:
    """
    Write a :class:`TranscriptionResult` to a Markdown file.

    Parameters
    ----------
    path : str
        Output file path.
    result : TranscriptionResult
        The transcription to write.
    """
    md = to_markdown(result)
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)


@dataclass
class Segment:
    """
    A timed piece of transcribed text

    Data
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
    speaker: tp.Optional[str] = None


@dataclass
class TranscriptionResult:
    """
    Result of a transcription call

    Data
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

    Data
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