from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence
from datetime import datetime
from pathlib import Path

try:
    from openai import AzureOpenAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    AzureOpenAI = None  # type: ignore

from config.settings import Settings
from config.llm_loader import get_llm_defaults
from utils.logger import get_logger

COST_LOG_PATH = Path(__file__).resolve().parents[1] / "config" / "llm_cost.txt"

logger = get_logger("services.llm_table_extractor")


class LLMExtractionError(RuntimeError):
    """Raised when the LLM-based table extraction cannot proceed."""


@dataclass
class LLMExtractionResult:
    header: List[str]
    data_rows: List[List[str]]
    raw_response: Optional[Dict[str, Any]] = None

    def to_table(self) -> List[List[str]]:
        return [self.header] + self.data_rows


class LLMTableExtractor:
    """Helper that calls Azure OpenAI (gpt-4o) to normalize OCR tables."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: Optional[AzureOpenAI] = None
        if AzureOpenAI is None:  # pragma: no cover - dependency missing
            logger.info("AzureOpenAI SDK not available; LLM extraction disabled")
            return
        if not settings.aoai_endpoint or not settings.aoai_api_key:
            logger.info("Azure OpenAI credentials not configured; LLM extraction disabled")
            return
        try:
            self._client = AzureOpenAI(
                api_key=settings.aoai_api_key,
                api_version=settings.aoai_api_version or "2024-06-01",
                azure_endpoint=settings.aoai_endpoint,
            )
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to initialize AzureOpenAI client")
            self._client = None

    @property
    def is_available(self) -> bool:
        return self._client is not None

    def extract(
        self,
        tables: Optional[Sequence[Sequence[Sequence[str]]]] = None,
        *,
        images: Optional[Sequence[Dict[str, Any]]] = None,
        center_conf: Dict[str, Any],
        center_id: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        log_fn: Optional[Callable[[str], None]] = None,
    ) -> LLMExtractionResult:
        if not self.is_available:
            raise LLMExtractionError("Azure OpenAI client not configured")

        llm_conf = center_conf.get("llm") if isinstance(center_conf, dict) else None
        if not isinstance(llm_conf, dict):
            raise LLMExtractionError("Center configuration does not define 'llm'")

        # Use file defaults (llm_defaults.json) and override with center config (original behavior)
        defaults = get_llm_defaults()
        effective_conf: Dict[str, Any] = {}
        if isinstance(defaults, dict):
            effective_conf.update(defaults)
        effective_conf.update(llm_conf)

        if not effective_conf.get("enabled", True):
            raise LLMExtractionError("LLM extraction disabled for this center")

        # Support multi-line prompt via array: llm.prompt_lines: ["line1", "line2", ...]
        prompt: Optional[str] = None
        prompt_lines = effective_conf.get("prompt_lines")
        if isinstance(prompt_lines, list) and prompt_lines:
            try:
                prompt = "\n".join(str(x) for x in prompt_lines)
            except Exception:
                prompt = None
        if not prompt:
            prompt = effective_conf.get("prompt")
        schema = effective_conf.get("json_schema")
        header_raw = effective_conf.get("header", [])
        header = list(header_raw) if isinstance(header_raw, list) else []
        if not prompt or not schema:
            raise LLMExtractionError("LLM prompt or schema is not configured")
        if not header:
            header = [
                "番号",
                "商品名・規格名",
                "銘柄・条件",
                "数量",
                "単位",
                "単価",
                "キャンセル可能期限",
                "備考",
                "成分表",
                "見本",
            ]

        # Resolve deployment: effective_conf (defaults -> center overrides) then env
        deployment = (
            effective_conf.get("deployment")
            or effective_conf.get("model")
            or getattr(self._settings, "aoai_deployment", None)
        )
        if not deployment:
            raise LLMExtractionError("Azure OpenAI deployment/model is not specified")

        if log_fn:
            try:
                init_deployment = effective_conf.get("deployment") or effective_conf.get("model") or deployment
                log_fn(
                    f"[LLM][INIT] deployment={init_deployment} client_ready={self.is_available}"
                )
            except Exception:
                pass

        temperature = float(effective_conf.get("temperature", 0.0) or 0.0)
        max_output_tokens = int(effective_conf.get("max_output_tokens", 2048) or 2048)
        schema_name = effective_conf.get("schema_name", "normalized_order_table")

        tables_list: List[List[List[str]]] = []
        if tables is not None:
            tables_list = [[list(cell_row) for cell_row in table] for table in tables]

        images_list: List[Dict[str, Any]] = []
        if images is not None:
            images_list = [dict(item) for item in images]

        if not tables_list and not images_list:
            raise LLMExtractionError("No tables or images provided for LLM extraction")

        payload = self._build_payload(tables_list, center_conf=center_conf, images=images_list)
        payload_json = json.dumps(payload, ensure_ascii=False)
        total_rows = sum(len(t.get("rows", [])) for t in payload.get("tables", []))
        total_cells = sum(
            len(r.get("cells", []))
            for t in payload.get("tables", [])
            for r in t.get("rows", [])
        )
        total_images = len(images_list)

        user_content: List[Dict[str, Any]] = [
            {
                "type": "input_text",
                "text": payload_json,
            }
        ]
        for image_entry in images_list:
            caption_parts: List[str] = []
            page_id = image_entry.get("page")
            file_name = image_entry.get("file_name")
            if page_id is not None:
                caption_parts.append(f"page={page_id}")
            if file_name:
                caption_parts.append(f"file={file_name}")
            if caption_parts:
                user_content.append({"type": "input_text", "text": " / ".join(caption_parts)})
            image_b64 = image_entry.get("image_base64")
            if not image_b64:
                continue
            try:
                data_url = f"data:image/png;base64,{image_b64}"
            except Exception:
                continue
            user_content.append(
                {
                    "type": "input_image",
                    "image_url": data_url,
                }
            )

        metadata_payload: Dict[str, Any] = {}
        if center_id:
            metadata_payload["center_id"] = str(center_id)
        if meta:
            for key, value in meta.items():
                if isinstance(value, (str, int, float, bool)):
                    metadata_payload[key] = str(value)
                else:
                    try:
                        metadata_payload[key] = json.dumps(value, ensure_ascii=False)
                    except Exception:
                        metadata_payload[key] = str(value)

        extra_body: Dict[str, Any] | None = None
        if schema:
            extra_body = {
                "text": {
                    "format": {
                        "name": schema_name,
                        "type": "json_schema",
                        "schema": copy.deepcopy(schema),
                        "strict": True,
                    }
                }
            }

        if log_fn:
            try:
                meta_keys = list(metadata_payload.keys())
            except Exception:
                meta_keys = []
            format_tag = "json_schema" if extra_body else "none"
            try:
                log_fn(
                    f"[LLM] request deployment={deployment} tables={len(payload.get('tables', []))} "
                    f"rows={total_rows} cells={total_cells} images={total_images}"
                )
            except Exception:
                pass
            try:
                log_fn(f"[LLM][META] metadata_keys={meta_keys} format={format_tag}")
            except Exception:
                pass

        try:
            response = self._client.responses.create(  # type: ignore[union-attr]
                model=deployment,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                input=[
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "input_text",
                                "text": prompt,
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": user_content,
                    },
                ],
                metadata=metadata_payload,
                extra_body=extra_body,
            )
        except Exception as exc:
            if log_fn:
                try:
                    log_fn(f"[LLM][ERROR] request_failed type={type(exc).__name__} detail={exc}")
                except Exception:
                    pass
            raise

        text = self._extract_response_text(response)
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "input_tokens", None) if usage else None
        completion_tokens = getattr(usage, "output_tokens", None) if usage else None
        if log_fn and usage:
            try:
                log_fn(
                    f"[LLM] usage prompt_tokens={prompt_tokens} completion_tokens={completion_tokens}"
                )
            except Exception:
                pass
        self._append_cost_log(
            center_id=center_id or str(center_conf.get("id", "")),
            model=deployment,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            meta=meta,
        )

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise LLMExtractionError(f"LLM output is not valid JSON: {exc}") from exc

        if log_fn:
            try:
                preview = text[:200].replace("\n", " ")
                log_fn(f"[LLM][RESPONSE] preview={preview}")
            except Exception:
                pass

        rows = parsed.get("rows") if isinstance(parsed, dict) else None
        if rows is None:
            raise LLMExtractionError("LLM output missing 'rows'")

        table_rows: List[List[str]] = []
        for idx, row_obj in enumerate(rows, start=1):
            if not isinstance(row_obj, dict):
                continue
            table_rows.append(self._normalize_row(row_obj, header, idx))

        raw_response: Optional[Dict[str, Any]] = None
        try:
            raw_response = response.model_dump()  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - optional
            raw_response = None

        return LLMExtractionResult(header=header, data_rows=table_rows, raw_response=raw_response)

    @staticmethod
    def _build_payload(
    tables: Sequence[Sequence[Sequence[str]]],
    *,
    center_conf: Dict[str, Any],
    images: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        payload_tables: List[Dict[str, Any]] = []
        center_id = center_conf.get("id") if isinstance(center_conf, dict) else None
        for table_index, table in enumerate(tables):
            table_entry: Dict[str, Any] = {
                "table_index": table_index,
                "rows": [],
            }
            for row_index, row in enumerate(table):
                cells: List[Dict[str, Any]] = []
                for column_index, cell in enumerate(row):
                    text = "" if cell is None else str(cell)
                    normalized = " ".join(text.split()) if text else ""
                    cells.append(
                        {
                            "column_index": column_index,
                            "text": text,
                            "text_normalized": normalized,
                        }
                    )
                table_entry["rows"].append(
                    {
                        "row_index": row_index,
                        "cells": cells,
                    }
                )
            payload_tables.append(table_entry)

        image_meta: List[Dict[str, Any]] = []
        if images:
            for entry in images:
                image_meta.append(
                    {
                        "file_index": entry.get("file_index"),
                        "file_name": entry.get("file_name"),
                        "page": entry.get("page"),
                        "width": entry.get("width"),
                        "height": entry.get("height"),
                    }
                )

        return {
            "center_id": center_id,
            "tables": payload_tables,
            "image_pages": image_meta,
        }

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        if response is None:  # pragma: no cover - defensive
            return ""
        text = getattr(response, "output_text", None)
        if isinstance(text, str) and text.strip():
            return text
        # Fallback for python SDK structures
        try:
            output = getattr(response, "output", None)
            if output and isinstance(output, list):
                first = output[0]
                content = getattr(first, "content", None)
                if content and isinstance(content, list):
                    piece = content[0]
                    value = getattr(piece, "text", None) or getattr(piece, "value", None)
                    if isinstance(value, str) and value.strip():
                        return value
        except Exception:  # pragma: no cover - defensive
            pass
        raise LLMExtractionError("Unable to read text from LLM response")

    def _append_cost_log(
        self,
        *,
        center_id: Optional[str],
        model: Optional[str],
        prompt_tokens: Optional[int],
        completion_tokens: Optional[int],
        meta: Optional[Dict[str, Any]],
    ) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        center_display = center_id or ""
        model_display = model or ""
        prompt_display = "-" if prompt_tokens is None else str(prompt_tokens)
        completion_display = "-" if completion_tokens is None else str(completion_tokens)
        meta_pairs: List[str] = []
        if meta:
            try:
                for key in sorted(meta.keys()):
                    value = meta[key]
                    meta_pairs.append(f"{key}={value}")
            except Exception:
                meta_pairs.append("meta_parse_error=1")
        line_parts = [
            timestamp,
            f"center_id={center_display}",
            f"model={model_display}",
            f"input_tokens={prompt_display}",
            f"output_tokens={completion_display}",
        ]
        if meta_pairs:
            line_parts.append("meta=" + ",".join(meta_pairs))
        line = " | ".join(line_parts)
        try:
            COST_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with COST_LOG_PATH.open("a", encoding="utf-8") as fp:
                fp.write(line + "\n")
        except Exception:
            logger.exception("Failed to append LLM cost log")

    @staticmethod
    def _normalize_row(row: Dict[str, Any], header: List[str], fallback_index: int) -> List[str]:
        normalized: List[str] = []
        for key in header:
            raw_value = row.get(key, "")
            value = "" if raw_value is None else str(raw_value)
            if key == "番号":
                value = value.strip()
                if not value:
                    value = str(fallback_index)
                normalized.append(value)
                continue

            if key == "成分表":
                flag = (value or "").strip().upper() == "T"
                normalized.append("○" if flag else "")
                continue

            if key == "見本":
                flag = (value or "").strip().upper() == "T"
                normalized.append("3" if flag else "")
                continue

            # default: trim spaces
            normalized.append(value.strip())
        return normalized
