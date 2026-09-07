import argparse
import json
import sys
import typing as tp

from whisper_tools.config import load_config


def main(argv: tp.Optional[tp.List[str]] = None) -> int:
    """
    CLI entry point

    Parameters
    ----------
    argv : list of str | None, default None
        Command line arguments. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    exit_code : int
        0 on success, 1 on error.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.version:
        from whisper_tools import __version__

        print(__version__)
        return 0

    if args.list_devices:
        from whisper_tools.stream import list_devices

        for index, name in list_devices():
            print(f"{index}: {name}")
        return 0

    config = load_config(args.config)

    try:
        if args.stream:
            return _run_stream(args, config)
        if args.files:
            return _run_files(args, config)
        parser.print_help()
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="whisper-tools", description="Transcribe audio with Whisper")
    parser.add_argument("files", nargs="*", help="Audio files to transcribe")
    parser.add_argument("--stream", action="store_true", help="Stream from the microphone")
    parser.add_argument("--list-devices", action="store_true", help="List input devices and exit")
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    parser.add_argument("--config", help="Path to a JSON or YAML config file")

    parser.add_argument("--model", help="Model name for local transcription (default: base)")
    parser.add_argument("--language", help="Language code, or 'auto' for auto-detection (default: ru)")
    parser.add_argument("--device", help="Device for local transcription (default: auto)")
    parser.add_argument("--api-key", help="API key; when set, uses the API backend")
    parser.add_argument("--base-url", help="API base URL (default: OpenAI)")
    parser.add_argument("--api-model", help="API model name (default: whisper-large-v3)")
    parser.add_argument("--noise-reduction", type=float, help="Noise reduction strength (default: 0)")
    parser.add_argument("--device-index", type=int, help="Input device index for streaming")
    parser.add_argument("-o", "--output", help="Write results to a file instead of stdout")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    return parser


def _merge_config(args: argparse.Namespace, config: tp.Dict[str, tp.Any]) -> tp.Dict[str, tp.Any]:
    merged = dict(config)
    for key in (
        "model", "language", "device", "api_key", "base_url",
        "api_model", "noise_reduction", "device_index",
    ):
        value = getattr(args, key, None)
        if value is not None:
            merged[key] = value
    return merged


_DEFAULTS = {
    "model": "base",
    "language": "ru",
    "device": "auto",
    "base_url": "https://api.openai.com/v1",
    "api_model": "whisper-large-v3",
    "noise_reduction": 0.0,
}


def _apply_defaults(cfg: tp.Dict[str, tp.Any]) -> tp.Dict[str, tp.Any]:
    for key, default in _DEFAULTS.items():
        cfg.setdefault(key, default)
    return cfg


def _build_transcriber(cfg: tp.Dict[str, tp.Any]) -> tp.Any:
    """Create a WhisperLocal or WhisperAPI instance from merged config"""
    cfg = _apply_defaults(cfg)
    language = None if cfg["language"] == "auto" else cfg["language"]
    noise_reduction = float(cfg["noise_reduction"])

    if cfg.get("api_key"):
        from whisper_tools.api import WhisperAPI

        return WhisperAPI(
            api_key=cfg["api_key"],
            base_url=cfg["base_url"],
            model=cfg["api_model"],
            language=language,
            noise_reduction=noise_reduction,
        )

    from whisper_tools.local import WhisperLocal

    return WhisperLocal(
        model=cfg["model"],
        language=language,
        device=cfg["device"],
        noise_reduction=noise_reduction,
    )


def _run_files(args: argparse.Namespace, config: tp.Dict[str, tp.Any]) -> int:
    """Transcribe one or more audio files"""
    cfg = _merge_config(args, config)
    transcriber = _build_transcriber(cfg)

    if len(args.files) == 1:
        results = [transcriber.transcribe(args.files[0])]
    else:
        results = transcriber.transcribe_many(args.files)

    lines = []
    for path, result in zip(args.files, results):
        if args.json:
            lines.append(json.dumps(_result_to_dict(result), ensure_ascii=False))
        else:
            lines.append(result.text)

    output = "\n".join(lines) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
    else:
        sys.stdout.write(output)
    return 0


def _run_stream(args: argparse.Namespace, config: tp.Dict[str, tp.Any]) -> int:
    cfg = _merge_config(args, config)
    transcriber = _build_transcriber(cfg)

    from whisper_tools.stream import StreamRecorder

    recorder = StreamRecorder(transcriber, on_text=lambda r: print(r.text, flush=True))
    if cfg.get("device_index") is not None:
        recorder.device = int(cfg["device_index"])

    print("Recording... Press Ctrl+C to stop", file=sys.stderr)
    recorder.start()
    try:
        while True:
            import time

            time.sleep(0.1)
    finally:
        recorder.stop()
    return 0


def _result_to_dict(result: tp.Any) -> tp.Dict[str, tp.Any]:
    return {
        "text": result.text,
        "language": result.language,
        "duration": result.duration,
        "segments": [
            {
                "start": s.start,
                "end": s.end,
                "text": s.text,
                "words": s.words,
            }
            for s in result.segments
        ],
    }


if __name__ == "__main__":
    sys.exit(main())