from __future__ import annotations

import io
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from services.ocr_client import TableBlock
from utils.logger import get_logger

logger = get_logger("services.excel_alignment")

ReferenceTable = List[List[str]]
MergedRows = List[List[str]]
Selections = List[Tuple[str, str, str, str]]


def merge_ocr_tables(table_blocks: Sequence[TableBlock]) -> MergedRows:
    merged: MergedRows = []
    header_written = False
    for block in table_blocks:
        if not block.rows:
            continue
        if not header_written:
            merged.extend(block.rows)
            header_written = True
        else:
            merged.extend(block.rows[1:] if len(block.rows) > 1 else [])
    return merged


def read_reference_table(file_bytes: bytes, filename: str) -> ReferenceTable:
    suffix = Path(filename).suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(io.BytesIO(file_bytes), header=0, dtype=str)
    elif suffix == ".csv":
        df = pd.read_csv(io.BytesIO(file_bytes), header=0, dtype=str)
    else:
        logger.warning("[REF][WARN] unsupported file type: %s", suffix)
        return []
    if df.empty:
        return []
    # すべてのセルを文字列にし、欠損は空文字へ
    df = df.fillna("").astype(str)

    # 行全体が空（全セルが空白 or 空文字）の判定用にトリムしたビューを作成
    # applymap は将来廃止予定のため、列ごとの str.strip を用いる
    trimmed = df.apply(lambda col: col.astype(str).str.strip())
    empty_row = trimmed.apply(lambda r: all(cell == "" for cell in r), axis=1)

    # 2行連続の空行が現れたら、その直前までを有効データとみなす
    cutoff_idx: int | None = None
    if len(empty_row) >= 2:
        for i in range(len(empty_row) - 1):
            if bool(empty_row.iat[i]) and bool(empty_row.iat[i + 1]):
                cutoff_idx = i  # i の直前までがデータ
                break

    if cutoff_idx is not None:
        df = df.iloc[:cutoff_idx]
        trimmed = trimmed.iloc[:cutoff_idx]
        empty_row = trimmed.apply(lambda r: all(cell == "" for cell in r), axis=1)

    # 行全体が空の行は除外（先頭列が空でも他列に値があれば残す）
    if len(empty_row) > 0:
        df = df[~empty_row].reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    # Build initial table (header + rows)
    header = df.columns.tolist()
    rows = df.values.tolist()

    # If there is a 備考-like column, and it contains multiple integers
    # in a single cell, split that row into multiple rows, one per
    # integer. Other columns are duplicated.
    remark_idx = None
    for i, col in enumerate(header):
        if isinstance(col, str) and "備考" in col:
            remark_idx = i
            break

    processed_rows: list[list[str]] = []
    if remark_idx is None:
        processed_rows = rows
    else:
        import re

        for r in rows:
            # ensure r is list and has enough columns
            row = list(r)
            remark_val = ""
            if remark_idx < len(row):
                remark_val = str(row[remark_idx]).strip()
            # find all integer tokens in the remark cell
            nums = re.findall(r"\d+", remark_val)
            if len(nums) <= 1:
                # keep original row (convert NaN->"" already handled)
                processed_rows.append(row)
            else:
                # create one row per integer, placing that integer in remark column
                for n in nums:
                    new_row = row.copy()
                    # ensure list is long enough
                    while len(new_row) <= remark_idx:
                        new_row.append("")
                    new_row[remark_idx] = n
                    processed_rows.append(new_row)

    table: ReferenceTable = [header]
    table.extend(processed_rows)
    return table


def build_selections(
    ocr_rows: MergedRows,
    ref_table: ReferenceTable,
) -> Tuple[Selections, Dict[str, list[str]], List[List[str]]]:
    selections: Selections = []
    maker_cds: Dict[str, list[str]] = {}
    flags_list: List[List[str]] = []
    if not ocr_rows or not ref_table:
        return selections, maker_cds, flags_list

    header = ocr_rows[0]
    ref_header = ref_table[0]
    required_cols = {
        "成分表": header.index("成分表") if "成分表" in header else None,
        "見本": header.index("見本") if "見本" in header else None,
        "商品CD": ref_header.index("商品CD") if "商品CD" in ref_header else None,
        "メーカー": ref_header.index("メーカー") if "メーカー" in ref_header else None,
    }
    if any(v is None for v in required_cols.values()):
        logger.warning("[SEL][WARN] required columns missing: %s", required_cols)
        return selections, maker_cds, flags_list

    seibun_idx = required_cols["成分表"]
    mihon_idx = required_cols["見本"]
    cd_idx = required_cols["商品CD"]
    maker_idx = required_cols["メーカー"]

    hits = 0
    for row_index, row in enumerate(ocr_rows[1:], start=1):
        if row_index >= len(ref_table):
            continue
        seibun_flag = row[seibun_idx] == "○"
        mihon_flag = row[mihon_idx] in {"3", "○"}
        if not (seibun_flag or mihon_flag):
            continue
        ref_row = ref_table[row_index]
        cd = (ref_row[cd_idx]).lstrip("0") or "0"
        maker = ref_row[maker_idx]
        selections.append((maker, cd, "○" if seibun_flag else "-", "3" if mihon_flag else "-"))
        maker_cds.setdefault(maker or "", []).append(cd)
        flags_list.append([maker or "", cd, "○" if seibun_flag else "-", "3" if mihon_flag else "-"])
        hits += 1
    logger.debug("[SEL] hits=%s selections=%s", hits, len(selections))
    return selections, maker_cds, flags_list


def infer_center_metadata(
    ref_table: ReferenceTable,
    original_filename: str,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Tuple[str, str]:
    center_name = ""
    center_month = ""
    if ref_table:
        flat_cells: list[str] = []
        for row in ref_table[:40]:
            for cell in row:
                if isinstance(cell, str) and cell.strip():
                    flat_cells.append(cell.strip())
        month_pattern = re.compile(r"(\d{1,2})\s*月")
        for cell in flat_cells:
            if "センター" in cell:
                center_name = cell
                m = month_pattern.search(cell)
                if m:
                    center_month = m.group(1)
                break
    if not center_name or not center_month:
        try:
            base = Path(original_filename).stem
            patterns = [
                r"^(?P<name>.+?)[（(](?P<mon>\d{1,2})\s*月(?:分)?[)）]$",
                r"^(?P<name>.+?)\s*[-_ ]\s*(?P<mon>\d{1,2})\s*月(?:分)?$",
                r"^(?P<name>.+?)(?:（|\()(?:(?:令和|平成)?\d+年)?(?P<mon>\d{1,2})月(?:分)?[)）]$",
            ]
            matched = False
            for pattern in patterns:
                match = re.match(pattern, base)
                if match:
                    if not center_name:
                        center_name = match.group("name").strip()
                    if not center_month:
                        center_month = match.group("mon")
                    matched = True
                    break
            if not matched and not center_name and "センター" in base:
                center_name = re.sub(r"[（(].*?[）)]", "", base).strip()
            if log_fn:
                log_fn(f"[CENTER][FILENAME] base={base} extracted name={center_name} month={center_month}")
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("[CENTER][FILENAME][WARN] %s", exc)
            if log_fn:
                log_fn(f"[CENTER][FILENAME][WARN] {exc}")
    logger.debug("[CENTER] name=%s month=%s", center_name, center_month)
    if log_fn:
        log_fn(f"[CENTER] name={center_name} month={center_month}")
    return center_name, center_month


def current_month_jst() -> str:
    now = datetime.now(timezone(timedelta(hours=9)))
    return str(now.month)
