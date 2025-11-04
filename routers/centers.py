from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException, status

from config.centers.loader import BASE_DIR, get_center_config
from utils.logger import get_logger

logger = get_logger("routers.centers")

router = APIRouter(prefix="/centers", tags=["centers"])

_VALID_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _center_files() -> List[Path]:
    return sorted(path for path in BASE_DIR.glob("*.json") if path.is_file())


@router.get("/list")
def list_centers() -> dict:
    centers = []
    for path in _center_files():
        try:
            with path.open("r", encoding="utf-8") as fp:
                data = json.load(fp)
        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.warning("failed to load center config %s: %s", path.name, exc)
            data = {}
        center_id = data.get("id") or path.stem
        display_name = data.get("displayName") or center_id
        centers.append({"id": center_id, "displayName": display_name})
    return {"centers": centers}


@router.get("/config/{center_id}")
def get_center(center_id: str) -> dict:
    if not _VALID_ID_RE.fullmatch(center_id):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid center id")
    data = get_center_config(center_id)
    if not data:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "center config not found")
    if "id" not in data:
        data = {**data, "id": center_id}
    return data
