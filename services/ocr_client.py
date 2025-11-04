from __future__ import annotations

import io
import random
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential

try:
    import PyPDF2
except Exception:  # pragma: no cover - optional dependency
    PyPDF2 = None  # type: ignore

from config.settings import Settings
from utils.logger import get_logger

logger = get_logger("services.ocr_client")


@dataclass
class TableBlock:
    rows: List[List[str]]
    row_count: int
    column_count: int
    page_number: Optional[int] = None


@dataclass
class AnalyzedContent:
    tables: List[TableBlock]
    page_count: int


class DocumentAnalyzer:
    def __init__(self, settings: Settings) -> None:
        if not settings.azure_endpoint or not settings.azure_key:
            raise ValueError("Azure endpoint/key must be configured")
        credential = AzureKeyCredential(settings.azure_key)
        # Pin API version for parity with Studio when provided (default 2024-11-30)
        api_version = getattr(settings, "azure_di_api_version", None) or None
        self._client = DocumentIntelligenceClient(
            settings.azure_endpoint,
            credential,
            api_version=api_version,
        )
        self._settings = settings

    def analyze_content(self, content: bytes) -> AnalyzedContent:
        analyze_result = self._analyze_with_backoff(content)
        tables = self._extract_tables(analyze_result)
        pages = len(getattr(analyze_result, "pages", []) or [])
        if not pages:
            page_numbers = []
            for table in getattr(analyze_result, "tables", []) or []:
                try:
                    for region in getattr(table, "bounding_regions", []) or []:
                        page_num = getattr(region, "page_number", getattr(region, "pageNumber", None))
                        if page_num:
                            page_numbers.append(int(page_num))
                except Exception:  # pragma: no cover - defensive
                    continue
            if page_numbers:
                pages = max(page_numbers)
            else:
                pages = 1
        return AnalyzedContent(tables=tables, page_count=pages)

    def analyze_pdf_by_pages(self, pdf_bytes: bytes) -> List[TableBlock]:
        if not PyPDF2:
            return self.analyze_content(pdf_bytes).tables
        splitted = self._split_pdf_pages(pdf_bytes)
        if not splitted:
            return self.analyze_content(pdf_bytes).tables
        tables: List[TableBlock] = []
        for index, page_bytes in enumerate(splitted, start=1):
            logger.debug("[PDF][SPLIT] page=%s", index)
            tables.extend(self.analyze_content(page_bytes).tables)
        return tables

    def analyze_files(self, files: Iterable[tuple[str, bytes]]) -> List[TableBlock]:
        collected: List[TableBlock] = []
        for name, content in files:
            if name.lower().endswith(".pdf"):
                # Switchable behavior: split per-page or analyze whole file
                if getattr(self, "_settings", None) and not getattr(self._settings, "di_split_pdf_pages", True):
                    collected.extend(self.analyze_content(content).tables)
                else:
                    collected.extend(self.analyze_pdf_by_pages(content))
            else:
                collected.extend(self.analyze_content(content).tables)
        return collected

    def split_pdf_pages(self, pdf_bytes: bytes):  # pragma: no cover - thin wrapper
        return self._split_pdf_pages(pdf_bytes)

    def _analyze_with_backoff(self, content: bytes, attempts: int = 3, initial_delay: float = 0.7):
        delay = initial_delay
        for attempt in range(attempts):
            try:
                poller = self._client.begin_analyze_document("prebuilt-layout", content)
                result = poller.result()
                logger.debug("[OCR] ok try=%s", attempt)
                return result
            except Exception as exc:  # pragma: no cover - SDK errors
                if attempt < attempts - 1:
                    logger.debug("[OCR][RETRY] %s; sleep %.2fs", exc.__class__.__name__, delay)
                    time.sleep(delay + random.uniform(0, delay * 0.3))
                    delay = min(delay * 2, 8)
                    continue
                logger.error("[OCR][ERROR] %s", exc)
                raise
        raise RuntimeError("OCR analysis failed")

    def _split_pdf_pages(self, pdf_bytes: bytes):  # pragma: no cover - heavy dependency
        if not PyPDF2:
            return None
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
            outputs = []
            for page in reader.pages:
                writer = PyPDF2.PdfWriter()
                writer.add_page(page)
                buffer = io.BytesIO()
                writer.write(buffer)
                outputs.append(buffer.getvalue())
            return outputs
        except Exception as exc:
            logger.debug("[PDF][SPLIT][WARN] %s", exc)
            return None

    @staticmethod
    def _extract_tables(analyze_result) -> List[TableBlock]:
        tables: List[TableBlock] = []
        for table in getattr(analyze_result, "tables", []) or []:
            try:
                rows = [[""] * table.column_count for _ in range(table.row_count)]
                for cell in table.cells:
                    rows[cell.row_index][cell.column_index] = (cell.content or "").replace("\n", " ")
                page_number: Optional[int] = None
                try:
                    if getattr(table, "bounding_regions", None):
                        region = table.bounding_regions[0]
                        page_number = getattr(region, "page_number", getattr(region, "pageNumber", None))
                        if page_number is not None:
                            page_number = int(page_number)
                except Exception:  # pragma: no cover - defensive
                    page_number = None
                tables.append(
                    TableBlock(
                        rows=rows,
                        row_count=table.row_count,
                        column_count=table.column_count,
                        page_number=page_number,
                    )
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("[OCR][TABLE][WARN] %s", exc)
        return tables
