from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from datetime import datetime

from config.centers.loader import get_center_config
from config.settings import get_settings
from services.excel_alignment import current_month_jst, infer_center_metadata, read_reference_table
from services.google_clients import get_google_clients
from services.llm_table_extractor import LLMExtractionError, LLMTableExtractor
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
    output_folder_id: Optional[str]
    extraction_sheet_id: Optional[str]
    extraction_sheet_url: Optional[str]
    center_name: str
    center_month: str
    debug_logs: List[str]


def _collect_llm_image_inputs(
    ocr_files: Sequence[Tuple[str, bytes]],
    cover_pages: int,
    log: Callable[[str], None],
    center_conf: dict | None = None,
) -> List[Dict[str, Any]]:
    """Prepare per-page image payloads for LLM-based table reconstruction."""

    page_images: List[Dict[str, Any]] = []
    cover_pages_remaining = max(0, cover_pages)
    crop_conf = center_conf.get("crop") if isinstance(center_conf, dict) else {}
    top_pct = float(crop_conf.get("top_pct", 0.0) or 0.0)
    bottom_pct = float(crop_conf.get("bottom_pct", 0.0) or 0.0)
    left_pct = float(crop_conf.get("left_pct", 0.0) or 0.0)
    right_pct = float(crop_conf.get("right_pct", 0.0) or 0.0)
    log(
        f"[LLM][IMG] start files={len(ocr_files)} crop=({top_pct},{bottom_pct},{left_pct},{right_pct})"
    )

    def _encode_png(image_bytes: bytes) -> str:
        return base64.b64encode(image_bytes).decode("ascii")

    for file_idx, (filename, content) in enumerate(ocr_files):
        base_name = Path(filename or f"file_{file_idx}").name
        is_pdf = base_name.lower().endswith(".pdf")
        try:
            page_bytes_list = _pdf_bytes_to_images(content) if is_pdf else None
        except Exception as exc:
            log(f"[LLM][IMG][WARN] pdf_render_failed file_idx={file_idx}: {exc}")
            page_bytes_list = None

        if page_bytes_list:
            log(
                f"[LLM][IMG][PDF] file_idx={file_idx} pages={len(page_bytes_list)} coverRemaining={cover_pages_remaining}"
            )
            for page_i, page_bytes in enumerate(page_bytes_list, start=1):
                if cover_pages_remaining > 0:
                    cover_pages_remaining -= 1
                    log(
                        f"[LLM][IMG][COVER] file_idx={file_idx} page={page_i} skip remaining={cover_pages_remaining}"
                    )
                    continue
                try:
                    cropped_bytes, width, height = crop_image_bytes(
                        page_bytes,
                        top_pct=top_pct,
                        bottom_pct=bottom_pct,
                        left_pct=left_pct,
                        right_pct=right_pct,
                        out_format="PNG",
                    )
                except Exception as exc:
                    log(f"[LLM][IMG][WARN] crop_failed file_idx={file_idx} page={page_i}: {exc}")
                    cropped_bytes = page_bytes
                    width = height = None
                page_images.append(
                    {
                        "file_index": file_idx,
                        "file_name": base_name,
                        "page": page_i,
                        "image_base64": _encode_png(cropped_bytes),
                        "width": width,
                        "height": height,
                    }
                )
                log(
                    f"[LLM][IMG][PAGE] file_idx={file_idx} page={page_i} width={width} height={height}"
                )
            continue

        if is_pdf:
            log(
                f"[LLM][IMG][WARN] pdf_no_pages file_idx={file_idx}; treating as single image"
            )

        if cover_pages_remaining > 0:
            cover_pages_remaining -= 1
            log(
                f"[LLM][IMG][COVER] file_idx={file_idx} skip entire file remaining={cover_pages_remaining}"
            )
            continue

        try:
            cropped_bytes, width, height = crop_image_bytes(
                content,
                top_pct=top_pct,
                bottom_pct=bottom_pct,
                left_pct=left_pct,
                right_pct=right_pct,
                out_format="PNG",
            )
        except Exception as exc:
            log(f"[LLM][IMG][WARN] crop_failed file_idx={file_idx}: {exc}")
            cropped_bytes = content
            width = height = None

        page_images.append(
            {
                "file_index": file_idx,
                "file_name": base_name,
                "page": 1,
                "image_base64": _encode_png(cropped_bytes),
                "width": width,
                "height": height,
            }
        )
        log(
            f"[LLM][IMG][FILE] file_idx={file_idx} width={width} height={height}"
        )

    log(f"[LLM][IMG] total_pages={len(page_images)}")
    return page_images




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

    page_images = _collect_llm_image_inputs(ocr_files, cover_pages, _log, center_conf=center_conf)
    if not page_images:
        raise RuntimeError("LLM処理用のページ画像が生成できませんでした")
    _log(f"[LLM][IMG] collected_pages={len(page_images)}")

    processed_rows: List[List[str]] = []
    row_map: List[int] = []

    llm_conf = center_conf.get("llm") if isinstance(center_conf, dict) else None
    data_rows_accum: List[List[str]] = []
    header_cols: List[str] = []

    if isinstance(llm_conf, dict) and llm_conf.get("enabled", True):
        extractor = LLMTableExtractor(settings)
        if extractor.is_available:
            total_pages = len(page_images)
            _log(f"[LLM][MODE] per-page processing pages={total_pages}")
            for idx, page in enumerate(page_images, start=1):
                try:
                    meta = {
                        "source": "vision",
                        "image_count": 1,
                        "pages": [page.get("page")],
                    }
                    result = extractor.extract(
                        tables=None,
                        images=[page],
                        center_conf=center_conf,
                        center_id=center_conf.get("id") if isinstance(center_conf, dict) else None,
                        log_fn=_log,
                        meta=meta,
                    )
                    if not header_cols:
                        header_cols = list(result.header)
                    # append rows only (skip header beyond the first page)
                    data_rows_accum.extend(result.data_rows)
                    _log(f"[LLM][PAGE] {idx}/{total_pages} rows+={len(result.data_rows)} total={len(data_rows_accum)}")
                except LLMExtractionError as exc:
                    _log(f"[LLM][ERROR][PAGE] {idx}/{total_pages} {exc}")
                except Exception:
                    logger.exception("LLM extraction failed (per-page)")
                    _log(f"[LLM][ERROR][PAGE] {idx}/{total_pages} unexpected failure")
        else:
            _log("[LLM] skipped (Azure OpenAI not configured)")

    if not data_rows_accum or not header_cols:
        raise RuntimeError("LLMによる表抽出に失敗しました。センター設定や画像を確認してください。")

    processed_rows = [header_cols] + data_rows_accum
    row_map = list(range(1, len(data_rows_accum) + 1))
    _log(f"[POST] processed_rows={len(processed_rows)} row_map={len(row_map)}")

    # --- 新仕様: 行全体が空白の行のみスキップし、有効行のみを1..Nで再番号化する ---
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
        skipped_empty = 0
        skipped_indices: List[int] = []
        for orig_idx, row in orig_indexed:
            # 行全体が空白（全セルが空/空白）の場合のみスキップ
            try:
                is_empty_row = True
                for c in row:
                    s = ("" if c is None else str(c)).strip()
                    if s:
                        is_empty_row = False
                        break
            except Exception:
                # 万一評価に失敗した場合は安全側で非空扱い
                is_empty_row = False
            if is_empty_row:
                skipped_empty += 1
                skipped_indices.append(orig_idx)
                continue
            filtered_indexed.append((orig_idx, row))

        # 新しいヘッダを先頭に付け、行番号を 1 から付与する
        new_header = ["行番号"] + orig_header
        new_rows: List[List[str]] = [new_header]
        for new_num, (orig_idx, row) in enumerate(filtered_indexed, start=1):
            new_rows.append([str(new_num)] + row)
            filtered_orig_indices.append(orig_idx)

        processed_rows = new_rows
        if skipped_empty:
            preview = skipped_indices[:10]
            _log(
                f"[POST][FILTER] skipped empty_rows={skipped_empty} indices_sample={preview}"
            )
    else:
        filtered_orig_indices = []
    # --- ここまで ---

    # Emit a compact debug summary from the pipeline side as well so that
    # apps always see match hints even if engine-level logs are filtered.
    # Do this AFTER row filtering/renumbering so 行番号と一致する。
    try:
        if processed_rows:
            header = processed_rows[0]
            data = processed_rows[1:]
            # locate columns
            spec_idx = None
            for i, h in enumerate(header):
                hs = str(h) if h is not None else ""
                if "規格" in hs or "規 格" in hs or ("規" in hs and "格" in hs):
                    spec_idx = i
                    break
            try:
                seibun_idx = header.index("成分表")
            except ValueError:
                seibun_idx = -1
            try:
                mihon_idx = header.index("見本")
            except ValueError:
                mihon_idx = -1

            # collect configured terms for quick hinting (order-preserving de-dup)
            def _terms(block: str) -> list[str]:
                b = center_conf.get(block) if isinstance(center_conf.get(block), dict) else {}
                all_terms: list[str] = []
                for k in ("matching_all", "matching_part", "matching"):
                    v = b.get(k)
                    if isinstance(v, str):
                        all_terms.append(v)
                    elif isinstance(v, list):
                        all_terms.extend([str(x) for x in v if isinstance(x, str)])
                seen: set[str] = set()
                out: list[str] = []
                for t in all_terms:
                    if t not in seen:
                        seen.add(t)
                        out.append(t)
                return out

            nut_terms = _terms("nutrition")
            sam_terms = _terms("sample")

            # gather up to 10 examples (sorted by 行番号の昇順にそのまま走査)
            examples_n: list[str] = []
            examples_s: list[str] = []
            for row in data:
                if spec_idx is None or spec_idx >= len(row):
                    continue
                try:
                    row_no = str(row[0])  # 行番号列
                except Exception:
                    row_no = "?"
                cell = str(row[spec_idx]) if row[spec_idx] is not None else ""
                # nutrition examples
                if seibun_idx >= 0 and seibun_idx < len(row) and row[seibun_idx] == "○" and len(examples_n) < 10:
                    term = next((t for t in nut_terms if t and t in cell), "")
                    preview = (cell[:40] + ("…" if len(cell) > 40 else "")) if cell else ""
                    examples_n.append(f"{row_no}:{term or '-'}:{preview}")
                # sample examples
                if mihon_idx >= 0 and mihon_idx < len(row) and row[mihon_idx] in {"3", "○"} and len(examples_s) < 10:
                    term = next((t for t in sam_terms if t and t in cell), "")
                    preview = (cell[:40] + ("…" if len(cell) > 40 else "")) if cell else ""
                    examples_s.append(f"{row_no}:{term or '-'}:{preview}")
            if examples_n:
                _log(f"[MATCH][SUMMARY][nutrition] examples={examples_n}")
            if examples_s:
                _log(f"[MATCH][SUMMARY][sample] examples={examples_s}")
    except Exception:
        # Do not let debug hinting break the pipeline
        logger.exception("failed to emit match summary (post-renumber)")

    # Step1 新仕様: フォルダ作成 (YYYYMMDD_centerId) し OCR結果と抽出結果シートを格納
    from datetime import datetime as _dt, timezone as _tz, timedelta as _td
    jst_now = _dt.now(_tz(_td(hours=9)))
    # フォルダ名: YYYYMMDDHHMM_centerId 形式（例: 202511141010_hitachinaka）
    folder_title = f"{jst_now.strftime('%Y%m%d%H%M')}_{center_id}"
    try:
        output_folder_id = sheet_generator.create_folder(folder_title, settings.drive_folder_id)
    except Exception:
        output_folder_id = None
        _log("[FOLDER][WARN] failed to create; fallback root")
    ocr_snapshot_url = ""
    ocr_snapshot_sheet_id = None
    if processed_rows:
        try:
            ocr_snapshot_sheet_id, ocr_snapshot_url = sheet_generator.write_rows_to_new_spreadsheet(
                title=f"OCR結果_{int(_dt.now().timestamp())}_{center_id}",
                rows=processed_rows,
                folder_id=output_folder_id,
                sheet_title="OCR結果",
            )
            _flush_sheet_logs()
        except Exception:
            _log("[OCR][WARN] snapshot spreadsheet creation failed")

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

    # 抽出結果シート作成 (人手修正用) flags_list を書き込み
    extraction_sheet_id = None
    extraction_sheet_url = None
    if flags_list:
        try:
            extraction_sheet_id, extraction_sheet_url = sheet_generator.create_extraction_sheet(
                center_id=center_id,
                flags_list=flags_list,
                folder_id=output_folder_id,
            )
            _flush_sheet_logs()
        except Exception:
            _log("[EXTRACT][WARN] create sheet failed")

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
            output_folder_id=output_folder_id,
        )
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
        output_folder_id=output_folder_id,
        extraction_sheet_id=extraction_sheet_id,
        extraction_sheet_url=extraction_sheet_url,
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
    extraction_sheet_id: Optional[str] = None,
    output_folder_id: Optional[str] = None,
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
    # 商品カタログ読み込み（共通）
    catalog_range = (
        center_conf.get("ranges", {}).get("catalog")
        if isinstance(center_conf.get("ranges"), dict)
        else None
    ) or settings.catalog_range
    catalog_template_id = center_conf.get("templateSpreadsheetId") or settings.template_spreadsheet_id
    catalog = sheet_generator.load_product_catalog(catalog_template_id, catalog_range)

    # 抽出結果シートが指定されていれば最新の flags を読み込む
    if extraction_sheet_id:
        try:
            flags = sheet_generator.load_extraction_sheet(extraction_sheet_id)
        except Exception:
            logger.exception("failed to load extraction sheet; fallback to provided flags")

    # flags: [依頼先, メーカー, 商品CD, 成分表, 見本]
    # 依頼先ごとに maker_cds と maker_data を構築
    dest_to_maker_cds: Dict[str, Dict[str, List[str]]] = {}
    dest_to_flags: Dict[str, List[List[str]]] = {}
    for dest, maker, code, s_flag, m_flag in flags:
        dest_key = dest or maker or "メーカー名なし"
        dest_to_flags.setdefault(dest_key, []).append([maker, code, s_flag, m_flag])
        dest_maker_cds = dest_to_maker_cds.setdefault(dest_key, {})
        dest_maker_cds.setdefault(maker, []).append(code)

    # 1つのスプレッドシートファイルを作成し、その中に依頼先ごとのシートを作る
    target_folder = output_folder_id or settings.drive_folder_id
    sheet_generator.debug_events = []
    title = f"依頼書出力_{int(datetime.now().timestamp())}"
    spreadsheet_id, url = sheet_generator.create_output_spreadsheet(
        title=title,
        drive_folder_id=target_folder,
    )

    debug_logs: List[str] = []
    for dest, m_cds in dest_to_maker_cds.items():
        m_data = sheet_generator.build_maker_rows(catalog, m_cds)
        dest_flags = dest_to_flags.get(dest, [])
        # 既存 generate_documents を流用して、同一ファイルにメーカー別シートを追加していく
        _, logs_tmp = sheet_generator.generate_documents(
            m_data,
            m_cds,
            dest_flags,
            center_name=center_name,
            center_month=center_month,
            center_conf=center_conf,
            center_id=center_id,
            output_folder_id=target_folder,
            existing_spreadsheet_id=spreadsheet_id,
            existing_url=url,
        )
        debug_logs.extend(logs_tmp)

    return url, debug_logs
