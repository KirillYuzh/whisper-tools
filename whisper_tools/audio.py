import subprocess
import tempfile
import typing as tp
from pathlib import Path

import numpy as np

TARGET_SAMPLE_RATE = 16000


def convert_to_wav(path: str, output_path: tp.Optional[str] = None) -> str:
    """
    Convert any audio format to WAV (16kHz, mono, f32le) using ffmpeg.

    Parameters
    ----------
    path : str
        Path to the input audio file.
    output_path : str | None, default None
        Path for the output WAV file. If None, a temp file is created.

    Returns
    -------
    str
        Path to the converted WAV file.
    """
    import os

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

    cmd = [
        "ffmpeg", "-v", "error", "-i", str(path),
        "-f", "f32le", "-ac", "1", "-ar", str(TARGET_SAMPLE_RATE), "-y",
        str(output_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return str(output_path)


def load_audio(path: str, sample_rate: int = TARGET_SAMPLE_RATE) -> tp.Tuple[int, np.ndarray]:
    """
    Load an audio file and resample it to the target rate

    Parameters
    ----------
    path : str
        Path to an audio file. WAV files are read directly by soundfile;
        other formats are decoded through ffmpeg.
    sample_rate : int, default 16000
        Target sample rate.

    Returns
    -------
    (sample_rate, audio) : tuple of (int, ndarray)
        Actual sample rate and mono float32 samples.
    """
    path = str(path)
    if path.lower().endswith(".wav"):
        try:
            import soundfile as sf

            data, rate = sf.read(path, dtype="float32")
            data = np.asarray(data, dtype=np.float32)
        except Exception:
            data, rate = _read_with_ffmpeg(path)
            data = np.asarray(data, dtype=np.float32)
    else:
        data, rate = _read_with_ffmpeg(path)
        data = np.asarray(data, dtype=np.float32)

    if data.ndim > 1:  # type: ignore[union-attr]
        data = data[:, 0]  # type: ignore[index]
    if rate != sample_rate:  # type: ignore[union-comparison]
        data = resample(data, rate, sample_rate)  # type: ignore[arg-type]
    return sample_rate, data.astype(np.float32)


def save_audio(path: str, audio: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> None:
    import soundfile as sf

    sf.write(path, np.asarray(audio, dtype=np.float32), sample_rate)


def _read_with_ffmpeg(path: str) -> tp.Tuple[int, np.ndarray]:
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(path),
        "-f", "f32le", "-ac", "1", "-ar", str(TARGET_SAMPLE_RATE), "-",
    ]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    data = np.frombuffer(raw, dtype=np.float32).copy()
    return TARGET_SAMPLE_RATE, data


def resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """
    Resample audio to a new sample rate

    Parameters
    ----------
    audio : ndarray
        Mono audio samples.
    src_rate : int
        Current sample rate.
    dst_rate : int
        Target sample rate.

    Returns
    -------
    resampled : ndarray
        Audio at the target sample rate.
    """
    if src_rate == dst_rate:
        return audio
    n_out = int(len(audio) * dst_rate / src_rate)
    x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def split_chunks(
    audio: np.ndarray,
    sample_rate: int,
    chunk_seconds: float = 30.0,
    overlap_seconds: float = 2.0,
) -> tp.List[np.ndarray]:
    """
    Split audio into overlapping chunks

    Parameters
    ----------
    audio : ndarray
        Mono audio samples.
    sample_rate : int
        Sample rate of `audio`.
    chunk_seconds : float, default 30.0
        Chunk length in seconds.
    overlap_seconds : float, default 2.0
        Overlap between neighbouring chunks in seconds.

    Returns
    -------
    chunks : list of ndarray
        Overlapping chunks, each at least 2 seconds long.
    """
    chunk_size = int(sample_rate * chunk_seconds)
    step = int(sample_rate * (chunk_seconds - overlap_seconds))
    if step <= 0:
        raise ValueError("chunk_seconds must be greater than overlap_seconds")

    chunks = []
    for start in range(0, len(audio), step):
        chunk = audio[start : start + chunk_size]
        if len(chunk) >= sample_rate * 2:
            chunks.append(chunk)
    return chunks


def reduce_noise(audio: np.ndarray, sample_rate: int, strength: float = 0.5) -> np.ndarray:
    """
    Simple spectral-gating noise reduction

    Parameters
    ----------
    audio : ndarray
        Mono audio samples.
    sample_rate : int
        Sample rate of `audio`.
    strength : float, default 0.5
        How aggressively to attenuate noise. 0 disables the effect.

    Returns
    -------
    denoised : ndarray
        Noise-reduced audio.
    """
    if strength <= 0:
        return audio

    frame = int(sample_rate * 0.02)  # 20 ms frames
    hop = frame // 2
    window = np.hanning(frame)

    n_frames = max(1, (len(audio) - frame) // hop + 1)
    frames = np.stack(
        [audio[i * hop : i * hop + frame] * window for i in range(n_frames)]
    )
    spectrum = np.fft.rfft(frames, axis=1)
    magnitude = np.abs(spectrum)

    noise_floor = np.percentile(magnitude, 20, axis=0)
    gain = np.clip(1.0 - strength * noise_floor / (magnitude + 1e-10), 0.0, 1.0)
    denoised = np.fft.irfft(spectrum * gain, axis=1)

    out = np.zeros(len(audio), dtype=np.float32)
    counts = np.zeros(len(audio), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop
        out[start : start + frame] += denoised[i]
        counts[start : start + frame] += window
    counts[counts == 0] = 1.0
    return (out / counts).astype(np.float32)


def save_temp_wav(audio: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> str:
    fd, path = tempfile.mkstemp(suffix=".wav")
    import os

    os.close(fd)
    save_audio(path, audio, sample_rate)
    return path


def cleanup_temp(path: str) -> None:
    p = Path(path)
    if p.exists():
        p.unlink()