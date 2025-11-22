from __future__ import annotations

from typing import List, Optional, Tuple

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
import json
import threading
import queue
import time

from schemas.order_entry import ExportRequest, ExportResponse, OrderResponse, OrderSelection
from usecases.order_pipeline import (
    OrderPipelineResult,
    export_order_documents,
    process_order,
)
from utils.logger import get_logger

logger = get_logger("routers.order_entry")

router = APIRouter(prefix="/orders", tags=["orders"])


async def _read_upload(upload: UploadFile) -> Tuple[str, bytes]:
    content = await upload.read()
    if not content:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"ファイル '{upload.filename}' が空です")
    return upload.filename or "", content


@router.post("/process", response_model=OrderResponse)
async def order_entry(
    center_id: str = Form(..., description="センターID"),
    sheet_name: str = Form("入札書", description="OCR結果を保存するシート名"),
    reference_file: Optional[UploadFile] = File(None, description="参照Excel/CSV"),
    ocr_files: List[UploadFile] = File(..., description="OCR対象の画像またはPDF（複数可）"),
    auto_export: bool = Form(True, description="Trueの場合は解析時に出力シートも生成します"),
) -> OrderResponse:
    if not ocr_files:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "OCRファイルが指定されていません")

    ref_tuple: Optional[Tuple[str, bytes]] = None
    if reference_file:
        ref_tuple = await _read_upload(reference_file)

    ocr_tuples: List[Tuple[str, bytes]] = []
    for file in ocr_files:
        try:
            ocr_tuples.append(await _read_upload(file))
        except HTTPException:
            continue
    if not ocr_tuples:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "有効なOCRファイルがありません")

    try:
        result: OrderPipelineResult = process_order(
            center_id=center_id,
            sheet_name=sheet_name,
            reference_file=ref_tuple,
            ocr_files=ocr_tuples,
            generate_documents=auto_export,
        )
    except Exception as exc:
        logger.exception("order processing failed")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(exc)) from exc

    selections = [
        OrderSelection(maker=maker, code=code, seibun_flag=seibun, mihon_flag=mihon)
        for maker, code, seibun, mihon in result.selections
    ]

    return OrderResponse(
        ocr_table=result.ocr_table,
        reference_table=result.reference_table,
        selections=selections,
        maker_data=result.maker_data,
        maker_cds=result.maker_cds,
        flags=result.flags,
        ocr_snapshot_url=result.ocr_snapshot_url,
        output_spreadsheet_url=result.output_spreadsheet_url,
        center_name=result.center_name,
        center_month=result.center_month,
        debug_logs=result.debug_logs,
    )


@router.post("/process/stream")
async def order_entry_stream(
    center_id: str = Form(..., description="センターID"),
    sheet_name: str = Form("入札書", description="OCR結果を保存するシート名"),
    reference_file: Optional[UploadFile] = File(None, description="参照Excel/CSV"),
    ocr_files: List[UploadFile] = File(..., description="OCR対象の画像またはPDF（複数可）"),
    auto_export: bool = Form(True, description="Trueの場合は解析時に出力シートも生成します"),
):
    """Streaming version of order processing that yields SSE-style debug events.

    This runs the same synchronous `process_order` under a background thread and
    streams log lines as SSE 'dbg' events as they are produced.
    """
    if not ocr_files:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "OCRファイルが指定されていません")

    ref_tuple: Optional[Tuple[str, bytes]] = None
    if reference_file:
        content = await reference_file.read()
        if not content:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, f"ファイル '{reference_file.filename}' が空です")
        ref_tuple = (reference_file.filename or "", content)

    ocr_tuples: List[Tuple[str, bytes]] = []
    for file in ocr_files:
        content = await file.read()
        if not content:
            continue
        ocr_tuples.append((file.filename or "", content))
    if not ocr_tuples:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "有効なOCRファイルがありません")

    # Thread-safe queue for passing log lines from worker -> streamer
    q: "queue.Queue[str]" = queue.Queue()

    def send_event(event_type: str, data: str) -> bytes:
        payload = json.dumps({"event": event_type, "data": data}, ensure_ascii=False)
        return f"data: {payload}\n\n".encode("utf-8")

    stop_marker = object()

    def worker() -> None:
        try:
            # pass a log_fn that puts messages into the queue
            def log_fn(msg: str) -> None:
                # Filter messages to avoid sending noisy API-level OK lines.
                # Strip timestamp prefix if present and check for allowed keywords.
                try:
                    import re

                    s = re.sub(r'^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]\s*', '', msg)
                except Exception:
                    s = msg

                allowed = [
                    "[MK-SHEET]",
                    "[STEP2]",
                    "[STEP1]",
                    "[CFG]",
                    "[SEL]",
                    "[OCR]",
                    "[TEST]",
                    "[ALIGN]",
                    "[ERROR]",
                    "[RETRY]",
                    "[CENTER]",
                    "[MATCH]",
                    "[POST]",
                ]
                if any(k in s for k in allowed):
                    q.put(msg)
                    try:
                        logger.debug(f"order_entry_stream.worker: queued log (len={len(msg)}) preview={s[:120]}")
                    except Exception:
                        logger.debug("order_entry_stream.worker: queued log (preview unavailable)")

            from usecases.order_pipeline import process_order

            result = process_order(
                center_id=center_id,
                sheet_name=sheet_name,
                reference_file=ref_tuple,
                ocr_files=ocr_tuples,
                generate_documents=auto_export,
                log_fn=log_fn,
            )
            # After processing, push the full structured result so clients
            # that rely on the streaming endpoint can receive the final
            # OrderPipelineResult without issuing an extra synchronous POST.
            try:
                # Filter final debug logs to user-relevant ones only
                try:
                    import re

                    def _strip_ts(m: str) -> str:
                        return re.sub(r'^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]\s*', '', m)

                    allowed = ["[MK-SHEET]", "[STEP2]", "[STEP1]", "[CFG]", "[SEL]", "[OCR]", "[ERROR]", "[RETRY]", "[CENTER]", "[MATCH]", "[POST]", "[ALIGN]"]
                    filtered = [m for m in result.debug_logs if any(k in _strip_ts(m) for k in allowed)]
                except Exception:
                    filtered = result.debug_logs

                final_payload = {
                    "ocr_table": result.ocr_table,
                    "reference_table": result.reference_table,
                    "selections": [list(t) for t in result.selections],
                    "maker_data": result.maker_data,
                    "maker_cds": result.maker_cds,
                    "flags": result.flags,
                    "ocr_snapshot_url": result.ocr_snapshot_url,
                    "output_spreadsheet_url": result.output_spreadsheet_url,
                    "output_folder_id": result.output_folder_id,
                    "extraction_sheet_id": result.extraction_sheet_id,
                    "extraction_sheet_url": result.extraction_sheet_url,
                    "center_name": result.center_name,
                    "center_month": result.center_month,
                    "debug_logs": filtered,
                }
                q.put(json.dumps(final_payload, ensure_ascii=False))
            except Exception:
                # If serialization fails for any reason, fall back to a small summary
                q.put(json.dumps({
                    "ocr_snapshot_url": result.ocr_snapshot_url,
                    "output_spreadsheet_url": result.output_spreadsheet_url,
                    "makers": len(result.maker_data),
                }, ensure_ascii=False))
        except Exception as e:
            q.put(f"[FATAL][STEP] {e}")
        finally:
            q.put(stop_marker)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    def event_stream():
        # Heartbeat timing
        last_hb = time.time()
        while True:
            try:
                item = q.get(timeout=0.5)
            except queue.Empty:
                # periodic heartbeat comment line to keep connection alive
                if time.time() - last_hb >= 5:
                    last_hb = time.time()
                    logger.debug("order_entry_stream.event_stream: sending heartbeat")
                    yield b": hb\n\n"
                continue
            if item is stop_marker:
                # close the stream
                logger.debug("order_entry_stream.event_stream: sending done event")
                yield send_event("done", "finished")
                break
            # If item is JSON-like final payload (dict string) we send as 'result'
            try:
                # If it's already JSON string (final payload), send as 'result'
                obj = json.loads(item)
                logger.debug(f"order_entry_stream.event_stream: sending result (len={len(item)})")
                yield send_event("result", obj)
            except Exception:
                # normal debug line
                try:
                    logger.debug(f"order_entry_stream.event_stream: sending dbg preview={item[:120]}")
                except Exception:
                    logger.debug("order_entry_stream.event_stream: sending dbg (preview unavailable)")
                yield send_event("dbg", item)

    # Add headers to reduce proxy buffering where applicable and disable cache.
    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(
        event_stream(), media_type="text/event-stream; charset=utf-8", headers=headers
    )


@router.post("/export", response_model=ExportResponse)
async def export_documents(payload: ExportRequest) -> ExportResponse:
    try:
        url, debug_logs = export_order_documents(
            center_id=payload.center_id,
            maker_data=payload.maker_data,
            maker_cds=payload.maker_cds,
            flags=payload.flags,
            center_name=payload.center_name,
            center_month=payload.center_month,
            extraction_sheet_id=payload.extraction_sheet_id,
            output_folder_id=payload.output_folder_id,
        )
    except Exception as exc:  # pragma: no cover - Google API errors
        logger.exception("document export failed")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(exc)) from exc
    return ExportResponse(output_spreadsheet_url=url, debug_logs=debug_logs)


@router.post("/export/stream")
async def export_documents_stream(payload: ExportRequest):
    """Streaming version of export that yields SSE-style debug events while
    generating the per-maker spreadsheets.
    """
    q: "queue.Queue[str]" = queue.Queue()

    def send_event(event_type: str, data: str) -> bytes:
        payload = json.dumps({"event": event_type, "data": data}, ensure_ascii=False)
        return f"data: {payload}\n\n".encode("utf-8")

    stop_marker = object()

    def worker() -> None:
        try:
            def log_fn(msg: str) -> None:
                # Filter similar to process stream: keep only user-relevant lines.
                try:
                    import re

                    s = re.sub(r'^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]\s*', '', msg)
                except Exception:
                    s = msg
                allowed = ["[MK-SHEET]", "[STEP2]", "[STEP1]", "[CFG]", "[SEL]", "[OCR]", "[TEST]", "[ERROR]", "[RETRY]", "[CENTER]", "[MATCH]", "[POST]", "[ALIGN]"]
                allowed = ["[MK-SHEET]", "[STEP2]", "[STEP1]", "[CFG]", "[SEL]", "[OCR]", "[TEST]", "[ERROR]", "[RETRY]", "[CENTER]", "[MATCH]", "[POST]", "[ALIGN]"]
                if any(k in s for k in allowed):
                    q.put(msg)
                    try:
                        logger.debug(f"export_documents_stream.worker: queued log preview={s[:120]}")
                    except Exception:
                        logger.debug("export_documents_stream.worker: queued log (preview unavailable)")

            from usecases.order_pipeline import export_order_documents

            url, debug_logs = export_order_documents(
                center_id=payload.center_id,
                maker_data=payload.maker_data,
                maker_cds=payload.maker_cds,
                flags=payload.flags,
                center_name=payload.center_name,
                center_month=payload.center_month,
                extraction_sheet_id=payload.extraction_sheet_id,
                output_folder_id=payload.output_folder_id,
                log_fn=log_fn,
            )
            try:
                try:
                    import re

                    def _strip_ts(m: str) -> str:
                        return re.sub(r'^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]\s*', '', m)

                    allowed = ["[MK-SHEET]", "[STEP2]", "[STEP1]", "[CFG]", "[SEL]", "[OCR]", "[ERROR]", "[RETRY]", "[CENTER]"]
                    filtered = [m for m in debug_logs if any(k in _strip_ts(m) for k in allowed)]
                except Exception:
                    filtered = debug_logs
                final_payload = {
                    "output_spreadsheet_url": url,
                    "debug_logs": filtered,
                }
                q.put(json.dumps(final_payload, ensure_ascii=False))
            except Exception:
                q.put(json.dumps({"output_spreadsheet_url": url}, ensure_ascii=False))
        except Exception as e:
            q.put(f"[FATAL][EXPORT] {e}")
        finally:
            q.put(stop_marker)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    def event_stream():
        last_hb = time.time()
        while True:
            try:
                item = q.get(timeout=0.5)
            except queue.Empty:
                if time.time() - last_hb >= 5:
                    last_hb = time.time()
                    logger.debug("export_documents_stream.event_stream: sending heartbeat")
                    yield b": hb\n\n"
                continue
            if item is stop_marker:
                logger.debug("export_documents_stream.event_stream: sending done")
                yield send_event("done", "finished")
                break
            try:
                obj = json.loads(item)
                logger.debug(f"export_documents_stream.event_stream: sending result (len={len(item)})")
                yield send_event("result", obj)
            except Exception:
                try:
                    logger.debug(f"export_documents_stream.event_stream: sending dbg preview={item[:120]}")
                except Exception:
                    logger.debug("export_documents_stream.event_stream: sending dbg (preview unavailable)")
                yield send_event("dbg", item)

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(event_stream(), media_type="text/event-stream; charset=utf-8", headers=headers)
