"""YAML configuration loader with environment variable interpolation."""

import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Load .env once at module import
load_dotenv()

_ENV_VAR_RE = re.compile(r"\$\{([^}]+)\}")


def _resolve_env(match: re.Match) -> str:
    """Replace ${VAR_NAME} or ${VAR_NAME:-default} with its environment value.

    Examples:
        ${LLM_PROVIDER}          → value of LLM_PROVIDER, or empty string
        ${LLM_PROVIDER:-deepseek} → value of LLM_PROVIDER, or "deepseek"
    """
    raw = match.group(1)

    if ":-" in raw:
        var_name, default = raw.split(":-", 1)
        return os.environ.get(var_name.strip(), default.strip())

    return os.environ.get(raw, "")


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file, interpolating ${VAR_NAME} references.

    Args:
        path: Absolute or relative path to the YAML file.

    Returns:
        Parsed and interpolated configuration dict.

    Example:
        >>> config = load_yaml_config("configs/detection.yaml")
        >>> config["encoder_type"]
        'TimesNet'
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, encoding="utf-8") as f:
        raw = f.read()

    # Interpolate environment variables
    resolved = _ENV_VAR_RE.sub(_resolve_env, raw)

    return yaml.safe_load(resolved)


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent.parent
