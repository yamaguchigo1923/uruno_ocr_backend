from fastapi import APIRouter

from config.settings import get_settings

router = APIRouter(prefix="", tags=["health"])  # root path


@router.get("/healthz", response_model=str)
def healthz() -> str:
    settings = get_settings()
    return f"ok {settings.app_version}"
