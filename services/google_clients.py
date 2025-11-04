from __future__ import annotations

from typing import Optional

from google.oauth2 import service_account
from googleapiclient.discovery import build
from google.cloud import storage

from config.settings import Settings
from utils.logger import get_logger

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

logger = get_logger("services.google_clients")


class GoogleClients:
    def __init__(self, settings: Settings) -> None:
        credentials = self._load_credentials(settings)
        self.credentials = credentials
        self.sheets = build("sheets", "v4", credentials=credentials)
        self.drive = build("drive", "v3", credentials=credentials)
        self.storage: Optional[storage.Client] = None
        if settings.gcs_bucket_name:
            self.storage = storage.Client(credentials=credentials, project=credentials.project_id)

    @staticmethod
    def _load_credentials(settings: Settings):
        path = settings.ensure_service_account_file()
        credentials = service_account.Credentials.from_service_account_file(path, scopes=SCOPES)
        logger.debug("[GOOGLE] credentials loaded project=%s", getattr(credentials, "project_id", "UNKNOWN"))
        return credentials


_cached_clients: GoogleClients | None = None


def get_google_clients(settings: Settings) -> GoogleClients:
    global _cached_clients
    if _cached_clients is None:
        _cached_clients = GoogleClients(settings)
    return _cached_clients
