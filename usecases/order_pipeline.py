from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from datetime import datetime

from config.centers.loader import get_center_config
from config.settings import get_settings
from services.excel_alignment import current_month_jst, infer_center_metadata, read_reference_table
from services.google_clients import get_google_clients
from services.ocr_client import DocumentAnalyzer, TableBlock
from services.matching_module.engine import build_matching_table
from services.sheet_generator import SheetGenerator
from services.image_preprocessing import crop_image_bytes, _pdf_bytes_to_images
from utils.logger import get_logger

logger = get_logger("usecases.order_pipeline")

Selections = List[Tuple[str, str, str, str]]


@dataclass
class OrderPipelineResult:
    ocr_table: List[List[str]]
    reference_table: List[List[str]]
    selections: Selections
    maker_data: Dict[str, List[List[str]]]
    maker_cds: Dict[str, List[str]]
    flags: List[List[str]]
    ocr_snapshot_url: Optional[str]
    output_spreadsheet_url: Optional[str]
    center_name: str
    center_month: str
    debug_logs: List[str]


def _append_table_rows(
    merged_rows: List[List[str]],
    table: TableBlock,
    header_written: bool,
) -> Tuple[bool, int]:
    if not table.rows:
        return header_written, 0
    if not header_written:
        merged_rows.extend(table.rows)
        return True, table.row_count
    body = table.rows[1:] if table.row_count > 1 else []
    merged_rows.extend(body)
    appended = table.row_count - 1 if table.row_count > 1 else 0
    return header_written, appended


def _collect_ocr_rows(
    analyzer: DocumentAnalyzer,
    ocr_files: Sequence[Tuple[str, bytes]],
    cover_pages: int,
    log: Callable[[str], None],
    center_conf: dict | None = None,
) -> List[List[str]]:
    merged_rows: List[List[str]] = []
    cover_pages_remaining = max(0, cover_pages)
    header_written = False
    skipped_any = False
    log(f"[STEP1] OCR start total_files={len(ocr_files)} (coverPages はページ単位)")
    # Crop configuration (fractions 0.0-1.0). If not provided, default to 0s.
    crop_conf = center_conf.get("crop") if isinstance(center_conf, dict) else {}
    top_pct = float(crop_conf.get("top_pct", 0.0) or 0.0)
    bottom_pct = float(crop_conf.get("bottom_pct", 0.0) or 0.0)
    left_pct = float(crop_conf.get("left_pct", 0.0) or 0.0)
    right_pct = float(crop_conf.get("right_pct", 0.0) or 0.0)
    log(f"[CFG][CROP] top={top_pct} bottom={bottom_pct} left={left_pct} right={right_pct}")
    for file_idx, (filename, content) in enumerate(ocr_files):
        base_name = Path(filename or f"file_{file_idx}").name
        is_pdf = base_name.lower().endswith(".pdf")
        split_pages = analyzer.split_pdf_pages(content) if is_pdf else None
        if split_pages:
            log(
                f"[PDF][SPLIT] file_idx={file_idx} pages={len(split_pages)} coverRemaining={cover_pages_remaining}"
            )
            for page_i, page_bytes in enumerate(split_pages, start=1):
                if cover_pages_remaining > 0:
                    cover_pages_remaining -= 1
                    skipped_any = True
                    log(
                        f"[COVER][SKIP-PAGE] file_idx={file_idx} page={page_i} remaining={cover_pages_remaining}"
                    )
                    continue
                # page_bytes may be a single-page PDF; render to image bytes if possible
                try:
                    imgs = _pdf_bytes_to_images(page_bytes)
                    page_img_bytes = imgs[0] if imgs else page_bytes
                except Exception:
                    page_img_bytes = page_bytes

                # Apply cropping to the page image bytes (no-op if all zero)
                try:
                    log(f"[CROP][PAGE] file_idx={file_idx} page={page_i} applying crop")
                    cropped_bytes, _, _ = crop_image_bytes(
                        page_img_bytes, top_pct=top_pct, bottom_pct=bottom_pct, left_pct=left_pct, right_pct=right_pct, out_format="PNG"
                    )
                    log(f"[CROP][PAGE] file_idx={file_idx} page={page_i} crop applied")
                except Exception as exc:
                    log(f"[CROP][PAGE][WARN] file_idx={file_idx} page={page_i} crop failed: {exc}")
                    cropped_bytes = page_img_bytes

                analyzed = analyzer.analyze_content(cropped_bytes)
                log(
                    f"[OCR][PAGE] file_idx={file_idx} page={page_i} tables={len(analyzed.tables)}"
                )
                for table_idx, table in enumerate(analyzed.tables):
                    page_display = table.page_number if table.page_number is not None else page_i
                    header_written, appended_rows = _append_table_rows(merged_rows, table, header_written)
                    log(
                        f"[OCR][TABLE] file_idx={file_idx} page={page_display} table_idx={table_idx} "
                        f"rows={table.row_count} cols={table.column_count} appended_rows={appended_rows}"
                    )
            continue

        # If PDF but cannot split (e.g., PyPDF2 not installed), render all pages and crop/analyze per page
        if is_pdf:
            try:
                imgs = _pdf_bytes_to_images(content)
            except Exception as exc:
                log(f"[PDF][RENDER][WARN] file_idx={file_idx} render failed: {exc}")
                imgs = []
            if imgs:
                log(
                    f"[PDF][RENDER] file_idx={file_idx} pages={len(imgs)} coverRemaining={cover_pages_remaining}"
                )
                for page_i, page_img_bytes in enumerate(imgs, start=1):
                    if cover_pages_remaining > 0:
                        cover_pages_remaining -= 1
                        skipped_any = True
                        log(
                            f"[COVER][SKIP-PAGE] file_idx={file_idx} page={page_i} remaining={cover_pages_remaining}"
                        )
                        continue
                    try:
                        log(f"[CROP][PAGE] file_idx={file_idx} page={page_i} applying crop")
                        cropped_bytes, _, _ = crop_image_bytes(
                            page_img_bytes, top_pct=top_pct, bottom_pct=bottom_pct, left_pct=left_pct, right_pct=right_pct, out_format="PNG"
                        )
                        log(f"[CROP][PAGE] file_idx={file_idx} page={page_i} crop applied")
                    except Exception as exc:
                        log(f"[CROP][PAGE][WARN] file_idx={file_idx} page={page_i} crop failed: {exc}")
                        cropped_bytes = page_img_bytes

                    analyzed = analyzer.analyze_content(cropped_bytes)
                    log(
                        f"[OCR][PAGE] file_idx={file_idx} page={page_i} tables={len(analyzed.tables)}"
                    )
                    for table_idx, table in enumerate(analyzed.tables):
                        page_display = table.page_number if table.page_number is not None else page_i
                        header_written, appended_rows = _append_table_rows(merged_rows, table, header_written)
                        log(
                            f"[OCR][TABLE] file_idx={file_idx} page={page_display} table_idx={table_idx} "
                            f"rows={table.row_count} cols={table.column_count} appended_rows={appended_rows}"
                        )
                continue

        # For non-PDF or unsplit files, attempt to crop before analysis.
        try:
            cropped_content = content
            # If this is likely an image, crop bytes; otherwise leave as-is.
            if not is_pdf:
                try:
                    log(f"[CROP][FILE] file_idx={file_idx} applying crop to image file")
                    cropped_content, _, _ = crop_image_bytes(
                        content, top_pct=top_pct, bottom_pct=bottom_pct, left_pct=left_pct, right_pct=right_pct, out_format="PNG"
                    )
                    log(f"[CROP][FILE] file_idx={file_idx} crop applied to image file")
                except Exception as exc:
                    log(f"[CROP][FILE][WARN] file_idx={file_idx} crop failed: {exc}")
                    cropped_content = content
        except Exception:
            cropped_content = content

        analyzed = analyzer.analyze_content(cropped_content)
        page_count = analyzed.page_count
        tables = analyzed.tables
        log(
            f"[OCR][FILE] idx={file_idx} path={base_name} pdf={is_pdf} pages={page_count} "
            f"tables={len(tables)} coverRemaining={cover_pages_remaining}"
        )
        if not tables:
            continue
        # If cover_pages_remaining would skip the whole file, be conservative:
        # - If this is the only uploaded file, don't skip it (users often upload a single image)
        # - Otherwise behave as before and consume pages
        if cover_pages_remaining >= page_count:
            if len(ocr_files) == 1:
                # Single-file run: avoid skipping the only file. Log decision and continue processing.
                log(
                    f"[COVER][SKIP-AVOID] single-file run; coverPages={cover_pages} would skip this file but we keep it"
                )
            else:
                cover_pages_remaining -= page_count
                skipped_any = True
                log(
                    f"[COVER][SKIP-FILE] file_idx={file_idx} skip_pages={page_count} remaining={cover_pages_remaining}"
                )
                continue
        skip_pages: set[int] = set()
        if cover_pages_remaining > 0:
            skip_pages = set(range(1, cover_pages_remaining + 1))
            cover_pages_remaining = 0
            log(f"[COVER][PARTIAL] file_idx={file_idx} skip_pages={sorted(skip_pages)}")
        for table_idx, table in enumerate(tables):
            page_no = table.page_number
            if page_no in skip_pages:
                skipped_any = True
                log(f"[COVER][SKIP] file_idx={file_idx} table_idx={table_idx} page={page_no}")
                continue
            header_written, appended_rows = _append_table_rows(merged_rows, table, header_written)
            page_display = page_no if page_no is not None else "None"
            log(
                f"[OCR][TABLE] file_idx={file_idx} table_idx={table_idx} page={page_display} "
                f"rows={table.row_count} cols={table.column_count} appended_rows={appended_rows}"
            )
    if cover_pages and skipped_any and not merged_rows:
        log("[STEP1][WARN] coverPages により全ページスキップ (テーブル無し)")
    return merged_rows


def process_order(
    *,
    center_id: str,
    sheet_name: str,
    reference_file: Optional[Tuple[str, bytes]],
    ocr_files: Sequence[Tuple[str, bytes]],
    generate_documents: bool = True,
    log_fn: Optional[Callable[[str], None]] = None,
) -> OrderPipelineResult:
    settings = get_settings()
    center_conf = get_center_config(center_id) or {}
    analyzer = DocumentAnalyzer(settings)
    google_clients = get_google_clients(settings)
    sheet_generator = SheetGenerator(settings, google_clients)
    debug_logs: List[str] = []

    def _log(message: str) -> None:
        # Always collect locally for return value. If a caller provided a
        # realtime callback (log_fn), call it immediately so the caller can
        # stream messages (eg. SSE). This preserves existing return-style
        # debug_logs while enabling streaming when desired.
        # Prefix all pipeline-emitted messages with a timestamp so that both
        # returned debug_logs and streamed messages include the emission time.
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ts_msg = f"[{ts}] {message}"
        debug_logs.append(ts_msg)
        if log_fn:
            try:
                log_fn(ts_msg)
            except Exception:
                # Avoid breaking processing if the external log_fn fails
                logger.exception("log_fn callback failed")

    # Wire sheet_generator to forward its internal debug messages to our _log
    # function so STEP2 logs appear in the same debug stream (and thus flow
    # to any streaming caller via process_order(log_fn=...)).
    try:
        sheet_generator.log_fn = _log  # type: ignore[attr-defined]
    except Exception:
        logger.exception("failed to set sheet_generator.log_fn")

    def _flush_sheet_logs() -> None:
        if sheet_generator.debug_events:
            debug_logs.extend(sheet_generator.debug_events)
            sheet_generator.debug_events.clear()

    ref_table: List[List[str]] = []
    ref_filename = ""
    if reference_file:
        ref_filename, ref_bytes = reference_file
        _log(f"[STEP1] read ref {ref_filename}")
        ref_table = read_reference_table(ref_bytes, ref_filename)
        header_preview = ref_table[0] if ref_table else []
        rows_count = len(ref_table) - 1 if ref_table else 0
        _log(f"[REF] header={header_preview} rows={rows_count}")
    logger.debug("[STEP] reference rows=%s", len(ref_table))

    if not ocr_files:
        raise ValueError("OCRファイルが指定されていません")

    if "coverPages" in center_conf:
        try:
            cover_pages = int(center_conf.get("coverPages", 0) or 0)
        except Exception:
            cover_pages = 0
        _log(
            f"[CFG] coverPages center(specified)={cover_pages} (先頭 {cover_pages} ファイル/ページをスキップ)"
        )
    else:
        cover_pages = 0
        _log("[CFG] coverPages default(=0)")

    merged_rows = _collect_ocr_rows(analyzer, ocr_files, cover_pages, _log, center_conf=center_conf)
    _log(f"[OCR] rows={len(merged_rows)} cols={(len(merged_rows[0]) if merged_rows else 0)}")

    match_table = build_matching_table(
        merged_rows,
        center_conf=center_conf,
        log_fn=_log,
    )
    processed_rows = match_table.rows
    row_map = match_table.row_map
    _log(f"[POST] processed_rows={len(processed_rows)} row_map={len(row_map)}")

    # --- 新仕様: 最初の列が空の行はスキップし、有効行のみを1..Nで再番号化する ---
    # processed_rows は header を先頭に持つテーブル形式を想定します。
    # オリジナルの行インデックス (ref_table と対応するための index) を保持し、
    # 参照テーブル参照時はそのオリジナル index を使います。
    filtered_orig_indices: List[int] = []
    if processed_rows:
        orig_header = processed_rows[0]
        orig_data = processed_rows[1:]
        # 元のデータ行に対する (orig_index, row) リスト。orig_index は ref_table と
        # 同じインデックス空間（1スタート）を想定します。
        if not row_map or len(row_map) != len(orig_data):
            row_map = list(range(1, len(orig_data) + 1))
        orig_indexed = list(zip(row_map, orig_data))
        filtered_indexed: List[Tuple[int, List[str]]] = []
        for orig_idx, row in orig_indexed:
            try:
                first_col = row[0] if row else ""
            except Exception:
                first_col = ""
            if isinstance(first_col, str) and first_col.strip():
                filtered_indexed.append((orig_idx, row))

        # 新しいヘッダを先頭に付け、行番号を 1 から付与する
        new_header = ["行番号"] + orig_header
        new_rows: List[List[str]] = [new_header]
        for new_num, (orig_idx, row) in enumerate(filtered_indexed, start=1):
            new_rows.append([str(new_num)] + row)
            filtered_orig_indices.append(orig_idx)

        processed_rows = new_rows
    else:
        filtered_orig_indices = []
    # --- ここまで ---

    ocr_snapshot_url = ""
    if processed_rows and sheet_name:
        ocr_snapshot_url = sheet_generator.write_ocr_snapshot(sheet_name, processed_rows)
        _flush_sheet_logs()

    selections: Selections = []
    maker_cds: Dict[str, List[str]] = {}
    flags_list: List[List[str]] = []

    if ref_table and processed_rows:
        ocr_header = processed_rows[0]
        ref_header = ref_table[0]
        _log(f"[SEL] ocr_header={ocr_header}")
        _log(f"[SEL] ref_header={ref_header}")
        has_required = all(
            column in ocr_header if column in {"成分表", "見本"} else column in ref_header
            for column in ["成分表", "見本", "商品CD", "メーカー"]
        )
        if has_required:
            seibun_idx = ocr_header.index("成分表")
            mihon_idx = ocr_header.index("見本")
            cd_idx = ref_header.index("商品CD")
            maker_idx = ref_header.index("メーカー")
            hits = 0
            # processed_rows は先頭に '行番号' を持つ可能性があるため、
            # ref_table 参照は filtered_orig_indices を使ってオリジナルの行番号で行う。
            for new_row_idx, row in enumerate(processed_rows[1:], start=1):
                if new_row_idx - 1 >= len(filtered_orig_indices):
                    # マッピング情報が無い場合は安全のためスキップ
                    continue
                orig_ref_idx = filtered_orig_indices[new_row_idx - 1]
                if orig_ref_idx >= len(ref_table):
                    continue
                seibun_flag = len(row) > seibun_idx and row[seibun_idx] == "○"
                mihon_flag = len(row) > mihon_idx and row[mihon_idx] in {"3", "○"}
                if not (seibun_flag or mihon_flag):
                    continue
                ref_row = ref_table[orig_ref_idx]
                code = (ref_row[cd_idx]).lstrip("0") or "0"
                maker = ref_row[maker_idx]
                maker_key = maker or "メーカー名なし"
                selections.append((maker, code, "○" if seibun_flag else "-", "3" if mihon_flag else "-"))
                maker_cds.setdefault(maker_key, []).append(code)
                flags_list.append([maker_key, code, "○" if seibun_flag else "-", "3" if mihon_flag else "-"])
                hits += 1
            _log(f"[SEL] 成分表/見本 hits={hits} selections={len(selections)}")
        else:
            _log("[SEL][WARN] 必要ヘッダが見つからない")
    _log(f"[SEL] preview_first10={selections[:10]}")
    _log(f"[SEL] selections={len(selections)} makers={len(maker_cds)}")

    maker_data: Dict[str, List[List[str]]] = {}
    if selections:
        ranges_conf = center_conf.get("ranges") if isinstance(center_conf.get("ranges"), dict) else {}
        if "catalog" in ranges_conf:
            _log(f"[CFG] catalog range center(specified)={ranges_conf['catalog']}")
        else:
            _log(f"[CFG] catalog range default={settings.catalog_range}")
        if "templateSpreadsheetId" in center_conf:
            _log(f"[CFG] catalog templateSpreadsheetId center(specified)={center_conf['templateSpreadsheetId']}")
        else:
            _log(f"[CFG] catalog templateSpreadsheetId default={settings.template_spreadsheet_id}")
        catalog_template_id = center_conf.get("templateSpreadsheetId") or settings.template_spreadsheet_id
        catalog_range = (
            center_conf.get("ranges", {}).get("catalog")
            if isinstance(center_conf.get("ranges"), dict)
            else None
        ) or settings.catalog_range
        catalog = sheet_generator.load_product_catalog(catalog_template_id, catalog_range)
        _flush_sheet_logs()
        maker_data = sheet_generator.build_maker_rows(catalog, maker_cds)
        _flush_sheet_logs()
        _log(f"[LOCAL_LOOKUP] makers={len(maker_data)}")

    center_name, center_month = infer_center_metadata(
        ref_table,
        ref_filename,
        log_fn=_log,
    )
    if center_conf.get("displayName"):
        center_name = center_conf["displayName"] or center_name
    center_month = current_month_jst()

    output_url = None
    if maker_data and generate_documents:
        output_url, doc_logs = sheet_generator.generate_documents(
            maker_data,
            maker_cds,
            flags_list,
            center_name=center_name,
            center_month=center_month,
            center_conf=center_conf,
            center_id=center_id,
        )
        # If we have wired sheet_generator.log_fn to stream logs in realtime,
        # those same messages have already been appended to debug_logs via
        # the _log callback, so avoid duplicating them here. Otherwise,
        # include returned doc_logs for backward compatibility.
        if getattr(sheet_generator, "log_fn", None) is None:
            debug_logs.extend(doc_logs)

    _log(f"[STEP1] makers={len(maker_data)}")
    _log("[STEP1] done")

    return OrderPipelineResult(
        ocr_table=processed_rows,
        reference_table=ref_table,
        selections=selections,
        maker_data=maker_data,
        maker_cds=maker_cds,
        flags=flags_list,
        ocr_snapshot_url=ocr_snapshot_url,
        output_spreadsheet_url=output_url,
        center_name=center_name,
        center_month=center_month,
        debug_logs=debug_logs,
    )


def export_order_documents(
    *,
    center_id: str,
    maker_data: Dict[str, List[List[str]]],
    maker_cds: Dict[str, List[str]],
    flags: List[List[str]],
    center_name: str,
    center_month: str,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Tuple[str, List[str]]:
    settings = get_settings()
    center_conf = get_center_config(center_id) or {}
    google_clients = get_google_clients(settings)
    sheet_generator = SheetGenerator(settings, google_clients)
    # If a caller provided a realtime log callback, wire it so
    # generate_documents will call it via sheet_generator._dbg.
    if log_fn:
        try:
            sheet_generator.log_fn = log_fn  # type: ignore[attr-defined]
        except Exception:
            logger.exception("failed to set sheet_generator.log_fn for export")
    url, debug_logs = sheet_generator.generate_documents(
        maker_data,
        maker_cds,
        flags,
        center_name=center_name,
        center_month=center_month,
        center_conf=center_conf,
        center_id=center_id,
    )
    return url, debug_logs
