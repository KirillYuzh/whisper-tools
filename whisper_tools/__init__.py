from .api import WhisperAPI
from .diarize import Diarizer, diarize
from .local import WhisperLocal
from .stream import StreamRecorder, list_devices
from .types import Segment, SpeakerTurn, TranscriptionResult, save_markdown, to_markdown

__version__ = "0.3.0"
__all__ = [
    "WhisperLocal",
    "WhisperAPI",
    "StreamRecorder",
    "Diarizer",
    "diarize",
    "list_devices",
    "Segment",
    "SpeakerTurn",
    "TranscriptionResult",
    "to_markdown",
    "save_markdown",
]