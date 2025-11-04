from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

BASE_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=64)
def get_center_config(center_id: str) -> Dict[str, Any]:
    """Return the JSON configuration for a given center id."""
    if not center_id:
        return {}
    path = BASE_DIR / f"{center_id}.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fp:
        try:
            return json.load(fp)
        except json.JSONDecodeError:
            return {}
