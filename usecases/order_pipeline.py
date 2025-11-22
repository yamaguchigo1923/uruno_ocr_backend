from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from datetime import datetime

from config.centers.loader import get_center_config
from config.settings import get_settings
from services.excel_alignment import current_month_jst, infer_center_metadata, read_reference_table
from utils.row_alignment import align_rows_by_number
from services.google_clients import get_google_clients
from services.llm_table_extractor import LLMExtractionError, LLMTableExtractor
from services.sheet_generator import SheetGenerator
from services.image_preprocessing import _pdf_bytes_to_images
from utils.logger import get_logger
from services.ocr_client import DocumentAnalyzer

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
    flags_with_number: Optional[List[List[str]]]
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
        # Load irregular destination mappings as early as possible so that
        # we can resolve "依頼先" at the time of reading the reference
        # table. This ensures the mapping is available for downstream
        # consumers (UI, sheet creation) immediately.
        try:
            maker_default_map, maker_code_map = sheet_generator.load_irregular_destinations()
        except Exception:
            maker_default_map, maker_code_map = {}, {}
        # Precompute dest per reference-row index (same indexing as ref_table)
        ref_dest_by_index: Dict[int, str] = {}
        if ref_table:
            ref_header = ref_table[0]
            try:
                cd_idx_pref = ref_header.index("商品CD")
                maker_idx_pref = ref_header.index("メーカー")
            except ValueError:
                cd_idx_pref = -1
                maker_idx_pref = -1
            for i, r in enumerate(ref_table):
                if i == 0:
                    continue
                try:
                    raw_maker = r[maker_idx_pref] if maker_idx_pref >= 0 and maker_idx_pref < len(r) else ""
                    raw_code = r[cd_idx_pref] if cd_idx_pref >= 0 and cd_idx_pref < len(r) else ""
                    maker_val = (raw_maker or "").strip()
                    code_val = (raw_code or "").strip()

                    dest_resolved = ""
                    # Prefer exact (maker, code) matches when code present
                    if maker_val:
                        if code_val:
                            code_norm = code_val.lstrip("0")
                            # try normalized code first, then raw
                            if (maker_val, code_norm) in maker_code_map:
                                dest_resolved = maker_code_map[(maker_val, code_norm)]
                            elif (maker_val, code_val) in maker_code_map:
                                dest_resolved = maker_code_map[(maker_val, code_val)]
                            elif maker_val in maker_default_map:
                                # fallback to maker-only mapping if available
                                dest_resolved = maker_default_map[maker_val]
                            else:
                                dest_resolved = maker_val
                        else:
                            # code empty: use maker-only mapping if exists
                            if maker_val in maker_default_map:
                                dest_resolved = maker_default_map[maker_val]
                            else:
                                dest_resolved = maker_val
                    else:
                        # no maker: no mapping possible
                        dest_resolved = ""
                except Exception:
                    dest_resolved = ""
                ref_dest_by_index[i] = dest_resolved
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

    # --- OCR: use DI-based table extraction identical to label_test ---
    _log("[TEST][DI] start prebuilt-layout")
    try:
        analyzer = DocumentAnalyzer(settings)
    except Exception as exc:
        logger.exception("failed to init DocumentAnalyzer")
        raise

    # Analyze each uploaded file as a whole (like label_test.analyze_with_di_for_test).
    # Maintain a running page offset so multi-file uploads map to increasing page numbers.
    di_page_tables: Dict[int, List[List[List[str]]]] = {}
    max_page_no = 0
    page_offset = 0
    for file_idx, (filename, content) in enumerate(ocr_files, start=1):
        try:
            analyzed = analyzer.analyze_content(content)
        except Exception:
            logger.exception("DI analyze_content failed for file=%s", filename)
            continue
        # analyzed.tables is a list of TableBlock with page_number possibly set
        file_page_count = getattr(analyzed, "page_count", 1) or 1
        _log(f"[TEST][DI][FILE] file_idx={file_idx} name={filename} pages={file_page_count} tables={len(analyzed.tables)}")
        for tb in analyzed.tables:
            # Use table's page_number if present; otherwise assume 1-based index
            pno = (tb.page_number or 1) + page_offset
            max_page_no = max(max_page_no, pno)
            di_page_tables.setdefault(int(pno), []).append(tb.rows)
        page_offset += file_page_count

    di_page_count = max_page_no
    # If cover pages configured, skip those pages (label_test uses page_skip)
    if cover_pages > 0:
        for p in range(1, cover_pages + 1):
            if p in di_page_tables:
                di_page_tables.pop(p, None)
        used_pages = sorted(di_page_tables.keys())
        _log(f"[TEST][DI] pages={di_page_count} tables_by_page={len(di_page_tables)} (skip first {cover_pages} pages, using pages={used_pages})")
    else:
        _log(f"[TEST][DI] pages={di_page_count} tables_by_page={len(di_page_tables)}")

    processed_rows: List[List[str]] = []
    row_map: List[int] = []

    llm_conf = center_conf.get("llm") if isinstance(center_conf, dict) else None
    data_rows_accum: List[List[str]] = []
    header_cols: List[str] = []

    # Use DI-based per-page LLM extraction (same as run_label_test.run_llm_extraction_from_di)
    if isinstance(llm_conf, dict) and llm_conf.get("enabled", True):
        extractor = LLMTableExtractor(settings)
        if extractor.is_available:
            # helper functions copied/adapted from test harness
            def _count_nonempty_columns(table: List[List[str]]) -> int:
                if not table:
                    return 0
                max_cols = max((len(r) for r in table), default=0)
                nonempty = 0
                for c in range(max_cols):
                    has_val = False
                    for r in table:
                        if c < len(r) and str(r[c]).strip():
                            has_val = True
                            break
                    if has_val:
                        nonempty += 1
                return nonempty

            def pick_widest_table(tables: List[List[List[str]]]) -> List[List[str]]:
                if not tables:
                    return []
                best_idx = 0
                best_score = (-1, -1, -1)
                for i, t in enumerate(tables):
                    rows = len(t)
                    raw_cols = max((len(r) for r in t), default=0)
                    nonempty_cols = _count_nonempty_columns(t)
                    score = (nonempty_cols, raw_cols, rows)
                    if score > best_score:
                        best_score = score
                        best_idx = i
                return tables[best_idx]

            def clean_table(table: List[List[str]]) -> List[List[str]]:
                cleaned: List[List[str]] = []
                import re as _re

                for row in table:
                    out_row: List[str] = []
                    for cell in row:
                        s = "" if cell is None else str(cell)
                        s = s.replace(":selected:", "")
                        s = s.replace("\r\n", "\n").replace("\r", "\n")
                        s = _re.sub(r"[\n]+", " ", s)
                        s = _re.sub(r"[\u200b\u200c\u200d\ufeff]", "", s)
                        s = s.replace("\u00A0", " ").replace("\u3000", " ")
                        s = _re.sub(r"[ \t\u00A0\u3000]+", " ", s)
                        s = _re.sub(r"-\s+", "-", s).strip()
                        out_row.append(s)
                    cleaned.append(out_row)
                return cleaned

            def drop_fully_empty_rows(table: List[List[str]]) -> List[List[str]]:
                out: List[List[str]] = []
                for row in table:
                    if any((str(c).strip() if c is not None else "") for c in row):
                        out.append(row)
                return out

            def has_header_keywords(row: List[str]) -> bool:
                keys = ["番号", "商品名", "銘柄", "備考"]
                s = " ".join((str(c) for c in row)).strip()
                return all(k in s for k in keys)

            total_pages = di_page_count
            _log(f"[LLM][MODE] per-page processing pages={total_pages}")

            import json as _json, time as _time

            for page_no in sorted(di_page_tables.keys()):
                page_start = _time.time()
                tables = di_page_tables[page_no]
                chosen = pick_widest_table(tables)
                chosen = clean_table(chosen)
                chosen = drop_fully_empty_rows(chosen)
                if not chosen:
                    continue

                total_rows = len(chosen)
                header_present = total_rows > 0 and has_header_keywords(chosen[0])
                expected_data_rows = max(0, total_rows - (1 if header_present else 0))

                conf_page = _json.loads(_json.dumps(center_conf, ensure_ascii=False))
                llm_conf_page = conf_page.get("llm", {}) or {}
                lines = list(llm_conf_page.get("prompt_lines", []))
                lines.append(
                    f"[DIメタ] このページのテーブルは合計{total_rows}行です。"
                    + ("先頭行はヘッダーです。" if header_present else "先頭行はヘッダーではありません。")
                    + f"データ行は合計{expected_data_rows}行あり、順序を保って全件（欠損は空文字）出力してください。省略禁止。"
                )
                llm_conf_page["prompt_lines"] = lines
                conf_page["llm"] = llm_conf_page

                # LLM 呼び出しにリトライ + バックオフを入れる
                meta = {"source": "di", "image_count": 0, "pages": [page_no]}
                max_retries = 3
                backoff_base = 3.0
                attempt = 0
                timeout_count = 0
                result = None
                while attempt < max_retries:
                    attempt += 1
                    attempt_start = _time.time()
                    try:
                        result = extractor.extract(
                            tables=[chosen],
                            images=None,
                            center_conf=conf_page,
                            center_id=center_conf.get("id") if isinstance(center_conf, dict) else None,
                            log_fn=lambda m: _log(f"[TEST][LLM][p{page_no}] {m}"),
                            meta=meta,
                        )
                        elapsed = _time.time() - attempt_start
                        if elapsed >= 30.0:
                            _log(f"[TEST][LLM][TIMEOUT] page={page_no} attempt={attempt} elapsed={elapsed:.2f}s (>30s)")
                            timeout_count += 1
                            if attempt >= max_retries:
                                _log(f"[TEST][LLM][INFO] page={page_no}: reached max_retries after timeout (count={timeout_count}), proceed with this result")
                                break
                            sleep_sec = backoff_base * attempt
                            _log(f"[TEST][LLM][INFO] page={page_no}: retry after timeout in {sleep_sec:.1f}s (timeout_count={timeout_count}) ...")
                            result = None
                            _time.sleep(sleep_sec)
                            continue
                        break
                    except LLMExtractionError as exc:
                        _log(f"[TEST][LLM][ERROR] page={page_no} attempt={attempt}: {exc}")
                        break
                    except Exception as exc:
                        logger.exception("LLM extraction failed (per-page)")
                        _log(f"[TEST][LLM][WARN] page={page_no} attempt={attempt} failed: {type(exc).__name__}: {exc}")
                        if attempt >= max_retries:
                            _log(f"[TEST][LLM][ERROR] page={page_no}: giving up after {attempt} attempts")
                            break
                        sleep_sec = backoff_base * attempt
                        _log(f"[TEST][LLM][INFO] page={page_no}: retrying in {sleep_sec:.1f}s ...")
                        _time.sleep(sleep_sec)

                if result is None:
                    continue

                if not header_cols:
                    header_cols = list(result.header)
                data_rows_accum.extend(result.data_rows)
                page_elapsed = _time.time() - page_start
                _log(f"[TEST][LLM][PAGE] {page_no} rows+={len(result.data_rows)} total={len(data_rows_accum)} time={page_elapsed:.2f}s")
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

        # --- ここから: excel_col による番号アラインメント ---
        excel_col_raw = center_conf.get("excel_col") if isinstance(center_conf, dict) else None
        try:
            excel_col_idx = int(excel_col_raw) - 1 if excel_col_raw is not None else -1
        except Exception:
            excel_col_idx = -1

        gt_numbers: List[int] = []
        if ref_table and excel_col_idx >= 0:
            # 参照表の指定列から正解番号を取得（1行目はヘッダ想定のためスキップ）
            for row in ref_table[1:]:
                if excel_col_idx < 0 or excel_col_idx >= len(row):
                    gt_numbers.append(-10**9)
                    continue
                cell = (row[excel_col_idx] or "").strip()
                try:
                    n = int(cell) if cell else None
                except ValueError:
                    n = None
                gt_numbers.append(n if n is not None else -10**9)

        ocr_numbers: List[int] = []
        if filtered_indexed:
            # LLM結果テーブルから "番号" 列を探し、その値をOCR番号として利用
            header_for_num = orig_header
            try:
                num_idx = header_for_num.index("番号")
            except ValueError:
                num_idx = -1
            for _orig_idx, row in filtered_indexed:
                if num_idx < 0 or num_idx >= len(row):
                    ocr_numbers.append(-10**9)
                    continue
                cell = (row[num_idx] or "").strip()
                try:
                    n = int(cell) if cell else None
                except ValueError:
                    n = None
                ocr_numbers.append(n if n is not None else -10**9)

        aligned_indices: List[int] = []
        # Always log counts so callers can see why alignment may or may not run
        try:
            _log(f"[ALIGN] gt_len={len(gt_numbers)} ocr_len={len(ocr_numbers)} excel_col_idx={excel_col_idx}")
        except Exception:
            _log("[ALIGN][WARN] failed to log gt/ocr lengths")

        if gt_numbers and ocr_numbers and len(gt_numbers) >= len(filtered_indexed):
            # Log before/after exact-match counts for diagnostics
            try:
                before_exact = 0
                for i in range(min(len(gt_numbers), len(ocr_numbers))):
                    try:
                        if gt_numbers[i] == ocr_numbers[i]:
                            before_exact += 1
                    except Exception:
                        continue
                _log(f"[ALIGN] before_exact_matches={before_exact} gt_len={len(gt_numbers)} ocr_len={len(ocr_numbers)}")
            except Exception:
                _log("[ALIGN][WARN] failed to compute before_exact_matches")

            pairs, skipped_gt, skipped_ocr = align_rows_by_number(gt_numbers, ocr_numbers)
            try:
                after_exact = 0
                for gi, oj in pairs:
                    try:
                        if gt_numbers[gi] == ocr_numbers[oj]:
                            after_exact += 1
                    except Exception:
                        continue
                ops = len(skipped_gt) + len(skipped_ocr)
                _log(f"[ALIGN] after_exact_matches={after_exact} pairs={len(pairs)} skipped_gt={len(skipped_gt)} skipped_ocr={len(skipped_ocr)} ops={ops} skipped_gt_sample={skipped_gt[:10]} skipped_ocr_sample={skipped_ocr[:10]}")
            except Exception:
                _log("[ALIGN][WARN] failed to compute after_exact_matches or ops")
            # pairs: (gi, oj) なので、OCR側インデックス順に正解行インデックスを引き当てる
            gi_by_oj: Dict[int, int] = {}
            for gi, oj in pairs:
                gi_by_oj[oj] = gi

            # Center-level policy: how to treat OCR-side skipped rows
            policy = "skip"
            try:
                if isinstance(center_conf, dict):
                    policy = str(center_conf.get("ocr_skip_policy", "skip") or "skip")
            except Exception:
                policy = "skip"

            if policy == "skip":
                # remove skipped OCR rows entirely
                kept_indexed: List[Tuple[int, List[str]]] = [item for idx, item in enumerate(filtered_indexed) if idx not in skipped_ocr]
                # build aligned_indices for kept rows only
                for local_idx, _ in [(i, None) for i in range(len(filtered_indexed)) if i not in skipped_ocr]:
                    gi = gi_by_oj.get(local_idx, None)
                    aligned_indices.append((gi + 1) if gi is not None else 0)
                # replace filtered_indexed with kept set (preserve order)
                filtered_indexed = kept_indexed
            else:
                # placeholder: insert an empty row tuple at skipped positions so downstream indices keep parity
                placeholder_row = ["" for _ in (filtered_indexed[0][1] if filtered_indexed else [])]
                new_filtered: List[Tuple[int, List[str]]] = []
                for i in range(len(filtered_indexed)):
                    if i in skipped_ocr:
                        new_filtered.append((0, placeholder_row.copy()))
                    else:
                        new_filtered.append(filtered_indexed[i])
                filtered_indexed = new_filtered
                for local_idx in range(len(filtered_indexed)):
                    gi = gi_by_oj.get(local_idx, None)
                    aligned_indices.append((gi + 1) if gi is not None else 0)
        else:
            # 番号情報が十分に取れない場合は、従来通り row_map をそのまま使う
            for orig_idx, _ in filtered_indexed:
                aligned_indices.append(orig_idx)

        # 新しいヘッダを先頭に付け、行番号を 1 から付与する
        new_header = ["行番号"] + orig_header
        new_rows: List[List[str]] = [new_header]
        for new_num, ((_, row), ref_idx) in enumerate(zip(filtered_indexed, aligned_indices), start=1):
            new_rows.append([str(new_num)] + row)
            filtered_orig_indices.append(ref_idx)

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
    # NOTE: match-summary diagnostic removed per request. Avoid emitting
    # potentially confusing example hints here so that downstream logs
    # focus on alignment and selection outcomes.

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
    # flags_list: [依頼先, 成分表, 見本, Excel行...]
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
            try:
                num_idx = ocr_header.index("番号")
            except ValueError:
                num_idx = -1
            cd_idx = ref_header.index("商品CD")
            maker_idx = ref_header.index("メーカー")
            hits = 0
            # prepare a separate flags list for generate_documents which expects
            # rows in the form [メーカー, 商品CD, 成分表フラグ, 見本フラグ]
            flags_for_docs: List[List[str]] = []
            # prepare frontend flags in the shape expected by the UI:
            # [依頼先, メーカー, 商品CD, 成分表, 見本]
            flags_for_frontend: List[List[str]] = []
            flags_for_frontend_with_number: List[List[str]] = []
            # processed_rows は先頭に '行番号' を持つ可能性があるため、
            # ref_table 参照は filtered_orig_indices を使ってオリジナルの行番号で行う。
            for new_row_idx, row in enumerate(processed_rows[1:], start=1):
                if new_row_idx - 1 >= len(filtered_orig_indices):
                    # マッピング情報が無い場合は安全のためスキップ
                    continue
                orig_ref_idx = filtered_orig_indices[new_row_idx - 1]
                # skip invalid mapping (0 means no mapping to data rows)
                if orig_ref_idx <= 0 or orig_ref_idx >= len(ref_table):
                    continue
                seibun_flag = len(row) > seibun_idx and row[seibun_idx] == "○"
                mihon_flag = len(row) > mihon_idx and row[mihon_idx] in {"3", "○"}
                if not (seibun_flag or mihon_flag):
                    continue
                ref_row = ref_table[orig_ref_idx]
                code = (ref_row[cd_idx]).lstrip("0") or "0"
                maker = ref_row[maker_idx]
                maker_key = maker or ""
                # First try a precomputed dest from the reference-table pass
                # (populated right after reading the reference file). If it's
                # not present, fall back to resolving here via irregular maps.
                dest = ""
                try:
                    dest = locals().get("ref_dest_by_index", {}).get(orig_ref_idx, "")
                except Exception:
                    dest = ""
                if not dest:
                    dest = maker_key
                    try:
                        code_norm = code.lstrip("0") or "0"
                        if (maker, code_norm) in maker_code_map:
                            dest = maker_code_map[(maker, code_norm)]
                        elif (maker, code) in maker_code_map:
                            dest = maker_code_map[(maker, code)]
                        elif maker in maker_default_map:
                            dest = maker_default_map[maker]
                    except Exception:
                        dest = maker_key

                selections.append((maker, code, "○" if seibun_flag else "-", "3" if mihon_flag else "-"))
                maker_cds.setdefault(maker_key, []).append(code)
                # 抽出結果1行 (SHEET用): [依頼先, 成分表, 見本, Excel行...]
                flags_list.append([
                    dest,
                    "○" if seibun_flag else "-",
                    "3" if mihon_flag else "-",
                    *ref_row,
                ])
                # For generate_documents we need (maker, code, s_flag, m_flag)
                flags_for_docs.append([
                    maker or "",
                    code,
                    "○" if seibun_flag else "-",
                    "3" if mihon_flag else "-",
                ])
                # Frontend expects flags in the shape:
                # [依頼先, メーカー, 商品CD, 成分表, 見本]
                # append to frontend flags list (without OCR 番号)
                flags_for_frontend.append([
                    dest,
                    maker or "",
                    code,
                    "○" if seibun_flag else "-",
                    "3" if mihon_flag else "-",
                ])
                # Also build a variant including OCR の "番号" for UI convenience
                ocr_no = ""
                try:
                    if num_idx >= 0 and num_idx < len(row):
                        ocr_no = (row[num_idx] or "")
                except Exception:
                    ocr_no = ""
                flags_for_frontend_with_number.append([
                    dest,
                    maker or "",
                    code,
                    ocr_no,
                    "○" if seibun_flag else "-",
                    "3" if mihon_flag else "-",
                ])
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

    # (irregular mappings already loaded earlier when reading the reference
    # table and ref_dest_by_index was computed). If for some reason they
    # weren't loaded there, ensure we have defaults here.
    try:
        maker_default_map = locals().get("maker_default_map", {})
        maker_code_map = locals().get("maker_code_map", {})
    except Exception:
        maker_default_map, maker_code_map = {}, {}

    center_name, center_month = infer_center_metadata(
        ref_table,
        ref_filename,
        log_fn=_log,
    )
    if center_conf.get("displayName"):
        center_name = center_conf["displayName"] or center_name
    center_month = current_month_jst()

    # 抽出結果シート作成 (人手修正用)
    # ユーザー指示: 前処理済みの参照シートをそのまま貼り、
    # 依頼先列の次に OCR の `番号`, `成分表`, `見本` 列を挿入します。
    extraction_sheet_id = None
    extraction_sheet_url = None
    try:
        if ref_table:
            # ref_header: header row of reference table
            ref_header = ref_table[0]
            # Determine excel_col index (備考 column) from center_conf or try to find '備考'
            try:
                excel_col_raw = center_conf.get("excel_col") if isinstance(center_conf, dict) else None
                excel_col_idx = int(excel_col_raw) - 1 if excel_col_raw is not None else -1
            except Exception:
                excel_col_idx = -1
            if excel_col_idx < 0:
                try:
                    excel_col_idx = ref_header.index("備考")
                except Exception:
                    excel_col_idx = -1

            # Find maker index in ref header
            try:
                maker_idx_ref = ref_header.index("メーカー")
            except Exception:
                maker_idx_ref = -1

            # Build extraction header in requested order: [依頼先, 備考(excel_col), 成分表, 見本, メーカー, *rest]
            excel_col_name = ref_header[excel_col_idx] if (0 <= excel_col_idx < len(ref_header)) else "備考"
            extraction_header = [
                "依頼先",
                excel_col_name,
                "成分表",
                "見本",
                "メーカー",
            ]
            # Append remaining ref columns excluding excel_col and maker (and excluding first col if it's maker)
            remaining_cols: List[str] = []
            for idx, col in enumerate(ref_header):
                if idx == maker_idx_ref or idx == excel_col_idx:
                    continue
                # avoid adding duplicate '依頼先' header if present
                if col == "依頼先":
                    continue
                remaining_cols.append(col)
            extraction_header.extend(remaining_cols)

            # Determine OCR header indices for 成分表/見本 in processed_rows
            ocr_header = processed_rows[0] if processed_rows else []
            def idx_of(name: str) -> int:
                try:
                    return ocr_header.index(name)
                except Exception:
                    return -1

            seibun_idx = idx_of("成分表")
            mihon_idx = idx_of("見本")

            extraction_rows: List[List[str]] = [extraction_header]
            # Build a mapping from reference-row-index -> processed_rows index
            refidx_to_procidx: Dict[int, int] = {}
            for proc_idx, ref_idx in enumerate(filtered_orig_indices, start=1):
                try:
                    if isinstance(ref_idx, int) and ref_idx > 0:
                        if ref_idx not in refidx_to_procidx:
                            refidx_to_procidx[ref_idx] = proc_idx
                except Exception:
                    continue

            # For each row in ref_table (excluding header), build an output row.
            for i, ref_r in enumerate(ref_table[1:], start=1):
                # dest (依頼先) should reflect the precomputed resolved destination
                # from `ref_dest_by_index` (computed when reading the reference file).
                # Fall back to the raw first-column value from the reference row.
                try:
                    dest_from_map = locals().get("ref_dest_by_index", {}).get(i, "")
                except Exception:
                    dest_from_map = ""
                raw_dest = (ref_r[0] if len(ref_r) > 0 else "")
                dest = dest_from_map if dest_from_map else raw_dest
                proc_idx = refidx_to_procidx.get(i)
                ocr_row = processed_rows[proc_idx] if proc_idx and proc_idx < len(processed_rows) else None
                seibun_v = ""
                mihon_v = ""
                if ocr_row:
                    try:
                        if seibun_idx >= 0 and seibun_idx < len(ocr_row):
                            seibun_v = str(ocr_row[seibun_idx] or "")
                    except Exception:
                        seibun_v = ""
                    try:
                        if mihon_idx >= 0 and mihon_idx < len(ocr_row):
                            mihon_v = str(ocr_row[mihon_idx] or "")
                    except Exception:
                        mihon_v = ""

                # excel_col value and maker value from reference row
                excel_val = ""
                if 0 <= excel_col_idx < len(ref_r):
                    excel_val = (ref_r[excel_col_idx] or "")
                maker_val = ""
                if 0 <= maker_idx_ref < len(ref_r):
                    maker_val = (ref_r[maker_idx_ref] or "")

                # remaining excel columns (exclude excel_col_idx and maker_idx_ref)
                rest = []
                for idx, cell in enumerate(ref_r):
                    if idx == excel_col_idx or idx == maker_idx_ref:
                        continue
                    # skip the original first column if it's maker and we already included maker
                    rest.append(cell)

                extraction_rows.append([dest, excel_val, seibun_v, mihon_v, maker_val, *rest])

            extraction_sheet_id, extraction_sheet_url = sheet_generator.write_rows_to_new_spreadsheet(
                title=f"抽出結果_{int(datetime.now().timestamp())}_{center_id}",
                rows=extraction_rows,
                folder_id=output_folder_id,
                sheet_title="抽出結果",
            )
            _flush_sheet_logs()
    except Exception:
        _log("[EXTRACT][WARN] create sheet failed")

    output_url = None
    if maker_data and generate_documents:
        flags_for_docs_to_use = locals().get("flags_for_docs", flags_list)
        output_url, doc_logs = sheet_generator.generate_documents(
            maker_data,
            maker_cds,
            flags_for_docs_to_use,
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
        flags=locals().get("flags_for_frontend", flags_list),
        flags_with_number=locals().get("flags_for_frontend_with_number", None),
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

    def _emit(message: str) -> None:
        if log_fn:
            try:
                log_fn(message)
                return
            except Exception:
                logger.exception("failed to emit export log via callback")
        logger.info(message)
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

    def _count_destinations(flag_rows: List[List[str]]) -> int:
        dest_keys = set()
        for dest, maker, *_ in flag_rows:
            dest_keys.add(dest or maker or "メーカー名なし")
        return len(dest_keys)

    # 抽出結果シートが指定されていれば最新の flags を読み込む
    if extraction_sheet_id:
        _emit(f"[STEP2][EXTRACT] center={center_id} sheet_id={extraction_sheet_id}")
        try:
            header, loaded = sheet_generator.load_extraction_sheet(extraction_sheet_id)
            _emit(f"[STEP2][EXTRACT] rows={len(loaded)} header={header}")
            # Normalize loaded rows into flags shape: [依頼先, メーカー, 商品CD, 成分表, 見本]
            normalized: List[List[str]] = []
            # header corresponds to columns starting at A (依頼先, 番号, 成分表, 見本, *rest)
            # 商品CD is expected in the rest part; find its index in header
            try:
                cd_idx = header.index("商品CD") if "商品CD" in header else -1
            except Exception:
                cd_idx = -1
            for row in loaded:
                # row format from loader: [dest, s_flag, m_flag, *excel_part]
                if not row or len(row) < 3:
                    continue
                dest, s_flag, m_flag = row[0], row[1], row[2]
                excel_part = row[3:]
                # skip if no 依頼先
                if not (dest and str(dest).strip()):
                    continue
                # skip if both 成分表 and 見本 empty
                if (not s_flag or s_flag in {"", "-"}) and (not m_flag or m_flag in {"", "-"}):
                    continue
                # determine maker and code from excel_part using header mapping
                maker = ""
                code = ""
                if cd_idx >= 0:
                    # excel_part aligns to header[3:]
                    idx_in_part = cd_idx - 3
                    if 0 <= idx_in_part < len(excel_part):
                        code = (excel_part[idx_in_part] or "").strip()
                # maker: try first column of excel_part if present
                if excel_part:
                    maker = (excel_part[0] or "").strip()
                # fallback: try to heuristically find numeric-like code in excel_part
                if not code:
                    import re

                    for cell in excel_part:
                        if isinstance(cell, str) and re.fullmatch(r"\d+", cell.strip()):
                            code = cell.strip()
                            break
                normalized.append([dest, maker, code, s_flag or "", m_flag or ""])
            flags = normalized
            _emit(f"[STEP2][EXTRACT] normalized_rows={len(flags)}")
        except Exception:
            logger.exception("failed to load extraction sheet; fallback to provided flags")
            _emit("[STEP2][EXTRACT][WARN] load failed; using request payload")
            _emit(f"[STEP2][EXTRACT] rows={len(flags)}")
    else:
        _emit(f"[STEP2][EXTRACT] center={center_id} sheet_id=None (using request payload)")
        _emit(f"[STEP2][EXTRACT] rows={len(flags)}")

    # flags: [依頼先, メーカー, 商品CD, 成分表, 見本]
    # 依頼先ごとに maker_cds と maker_data を構築
    dest_to_maker_cds: Dict[str, Dict[str, List[str]]] = {}
    dest_to_flags: Dict[str, List[List[str]]] = {}
    for dest, maker, code, s_flag, m_flag in flags:
        dest_key = dest or maker or "メーカー名なし"
        dest_to_flags.setdefault(dest_key, []).append([maker, code, s_flag, m_flag])
        dest_maker_cds = dest_to_maker_cds.setdefault(dest_key, {})
        dest_maker_cds.setdefault(maker, []).append(code)

    total_destinations = len(dest_to_maker_cds)

    # Create a dedicated folder for this export and produce a single spreadsheet
    # with one sheet per 依頼先 (template sheet will be copied into this file).
    target_parent = output_folder_id or settings.drive_folder_id
    folder_title = f"依頼書出力_{int(datetime.now().timestamp())}"
    try:
        folder_id = sheet_generator.create_folder(folder_title, parent_folder_id=target_parent)
    except Exception:
        logger.exception("failed to create output folder; falling back to parent folder")
        folder_id = target_parent

    # Create one spreadsheet that will hold per-destination sheets
    ss_title = f"依頼書出力_{int(datetime.now().timestamp())}"
    spreadsheet_id, spreadsheet_url = sheet_generator.create_output_spreadsheet(
        title=ss_title, drive_folder_id=folder_id
    )

    debug_logs: List[str] = []
    _emit(f"[STEP2][CREATE] spreadsheet_id={spreadsheet_id} url={spreadsheet_url}")

    # For each destination, copy the template into the existing spreadsheet
    for idx, (dest, m_cds) in enumerate(dest_to_maker_cds.items(), start=1):
        dest_rows = sum(len(codes) for codes in m_cds.values())
        dest_flags = dest_to_flags.get(dest, [])
        _emit(f"[STEP2][DEST][{idx}/{total_destinations}] dest={dest or '未設定'} rows={dest_rows}")
        m_data = sheet_generator.build_maker_rows(catalog, m_cds)
        try:
            # Pass existing_spreadsheet_id so generate_documents will add a sheet
            url_i, logs_tmp = sheet_generator.generate_documents(
                m_data,
                m_cds,
                dest_flags,
                doc_title=dest,
                center_name=center_name,
                center_month=center_month,
                center_conf=center_conf,
                center_id=center_id,
                output_folder_id=folder_id,
                existing_spreadsheet_id=spreadsheet_id,
                existing_url=spreadsheet_url,
            )
            debug_logs.extend(logs_tmp)
            _emit(f"[STEP2][DEST][DONE] dest={dest} sheet_added_to={spreadsheet_url}")
        except Exception as exc:
            logger.exception("failed to generate document for dest=%s", dest)
            _emit(f"[STEP2][DEST][ERROR] dest={dest} error={exc}")

    # Return the single spreadsheet URL and aggregated debug logs
    return spreadsheet_url, debug_logs
