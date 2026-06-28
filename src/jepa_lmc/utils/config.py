from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"Empty config file: {path}")

    return data