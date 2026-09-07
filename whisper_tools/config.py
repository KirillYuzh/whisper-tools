import json
import typing as tp
from pathlib import Path

_CONFIG_FILENAMES = ("whisper-tools.json", "whisper-tools.yaml", "whisper-tools.yml")


def load_config(path: tp.Optional[str] = None) -> tp.Dict[str, tp.Any]:
    """
    Load a configuration file.

    When `path` is None, looks for ``whisper-tools.json``,
    ``whisper-tools.yaml`` or ``whisper-tools.yml`` in the current
    directory, in that order.

    Parameters
    ----------
    path : str | None, default None
        Path to a JSON or YAML config file.

    Returns
    -------
    config : dict
        Parsed configuration. Empty dict when no file is found.
    """
    if path is not None:
        return _read_file(Path(path))

    for name in _CONFIG_FILENAMES:
        candidate = Path(name)
        if candidate.exists():
            return _read_file(candidate)
    return {}


def _read_file(path: Path) -> tp.Dict[str, tp.Any]:
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as e:
            raise ImportError(
                "PyYAML is required to read YAML config files. "
                "Install it with `pip install pyyaml`."
            ) from e
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    raise ValueError(f"Unsupported config file format: {path.suffix}")