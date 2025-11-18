from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

BASE_DIR = Path(__file__).resolve().parent
PROMPT_PATH = BASE_DIR / "prompt.json"


@lru_cache(maxsize=1)
def _load_llm_configs() -> Dict[str, Any]:
    if not PROMPT_PATH.exists():
        return {}
    try:
        with PROMPT_PATH.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
            return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


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
            config = json.load(fp)
        except json.JSONDecodeError:
            return {}
    if not isinstance(config, dict):
        return {}

    llm_configs = _load_llm_configs()
    llm_conf = llm_configs.get(center_id) or llm_configs.get("defalt")
    if llm_conf:
        config = dict(config)
        config["llm"] = copy.deepcopy(llm_conf)
    return config
