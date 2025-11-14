from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv(override=False)


class SheetConfig(BaseModel):
    spreadsheet_id: str
    range_prefix: str
    gid: str


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_version: str = Field("2025-10-29a", env="APP_VERSION")

    azure_endpoint: str = Field("", env="AZURE_ENDPOINT")
    azure_key: str = Field("", env="AZURE_KEY")
    # Azure Document Intelligence (Document Analysis) API version and behavior
    # Default aligns with Azure AI Document Intelligence Studio GA as of 2024-11-30
    azure_di_api_version: str | None = Field("2024-11-30", env="AZURE_DI_API_VERSION")
    # Whether to split PDFs per page before analysis (True = current behavior).
    # Set to False to analyze the whole file at once for parity with Studio comparisons.
    di_split_pdf_pages: bool = Field(True, env="DI_SPLIT_PDF_PAGES")

    service_account_file: str = Field("service-account.json", env="SERVICE_ACCOUNT_FILE")
    service_account_json_b64: str | None = Field(None, env="SERVICE_ACCOUNT_JSON_B64")
    service_account: str | None = Field(None, env="SERVICE_ACCOUNT")

    drive_folder_id: str = Field("", env="DRIVE_FOLDER_ID")
    gcs_bucket_name: str = Field("", env="GCS_BUCKET_NAME")

    main_sheet_id: str = Field("1ICVcByL2iEdzR4enIclntlTnHLG3Eze2e8-iH8wuT6Y", env="MAIN_SHEET_ID")
    bid_gid: str = Field("0", env="BID_GID")
    ref_gid: str = Field("885700754", env="REF_GID")

    template_spreadsheet_id: str = Field("1tIO4OvE5SC0NN1tDEIAbZV6xQ_4QiA3Jd7Pi2gLfN1o", env="TEMPLATE_SPREADSHEET_ID")
    template_sheet_id: int = Field(1557733602, env="TEMPLATE_SHEET_ID")
    start_row: int = Field(32, env="START_ROW")

    catalog_range: str = Field("'商品'!A10:H30000", env="CATALOG_RANGE")
    export_range_maker_header: str = Field("B4:E4", env="EXPORT_RANGE_MAKER_HEADER")
    export_range_center_name: str = Field("B27:E27", env="EXPORT_RANGE_CENTER_NAME")
    export_range_month: str = Field("F27", env="EXPORT_RANGE_MONTH")

    batch_chunk_size: int = Field(40, env="BATCH_CHUNK_SIZE")
    poll_max_wait: float = Field(60.0, env="POLL_MAX_WAIT")
    poll_min_ready: float = Field(0.95, env="POLL_MIN_READY")
    ready_col_idx: int = Field(0, env="READY_COL_IDX")

    aoai_endpoint: str | None = Field(None, env="AOAI_ENDPOINT")
    aoai_api_key: str | None = Field(None, env="AOAI_API_KEY")
    aoai_deployment: str | None = Field(None, env="AOAI_DEPLOYMENT")
    aoai_api_version: str | None = Field(None, env="AOAI_API_VERSION")

    # Irregular destination list (依頼先のイレギュラー一覧)
    irregular_dest_spreadsheet_id: str | None = Field(None, env="IRREGULAR_DEST_SPREADSHEET_ID")
    irregular_dest_gid: int | None = Field(None, env="IRREGULAR_DEST_GID")

    @field_validator("azure_endpoint", mode="before")
    def _normalize_endpoint(cls, value: str | None) -> str | None:  # noqa: N805 - pydantic requirement
        if not value:
            return value
        v = value.strip()
        if not v.lower().startswith(("http://", "https://")):
            v = "https://" + v.lstrip("/")
        return v.rstrip("/")

    @property
    def sheets(self) -> Dict[str, SheetConfig]:
        return {
            "入札書": SheetConfig(
                spreadsheet_id=self.main_sheet_id,
                range_prefix="'入札書'!",
                gid=self.bid_gid,
            ),
            "見積書": SheetConfig(
                spreadsheet_id=self.main_sheet_id,
                range_prefix="'見積書'!",
                gid=self.ref_gid,
            ),
        }

    def export_ranges(self) -> Dict[str, str]:
        return {
            "makerHeader": self.export_range_maker_header,
            "centerName": self.export_range_center_name,
            "month": self.export_range_month,
        }

    def ensure_service_account_file(self) -> Path:
        target = Path(self.service_account_file)
        if target.exists():
            return target
        if self.service_account_json_b64:
            target.parent.mkdir(parents=True, exist_ok=True)
            decoded = base64.b64decode(self.service_account_json_b64)
            target.write_bytes(decoded)
            return target
        raise FileNotFoundError(
            f"service account file '{target}' not found and base64 env is not provided"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    # Ensure service account file exists if credentials are provided via env
    try:
        settings.ensure_service_account_file()
    except FileNotFoundError:
        # Allow lazy creation if caller wants to supply file later
        pass
    return settings
