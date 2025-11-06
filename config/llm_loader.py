from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

BASE_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def get_llm_defaults() -> Dict[str, Any]:
    path = BASE_DIR / "llm_defaults.json"
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    return data.copy()
