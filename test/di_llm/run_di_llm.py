"""
DI + LLM 統合テスト

実行方法 (Windows, 仮想環境有効化済み想定):
    cd /d C:/Users/user/project/uruno_ocr_demo/backend/test/di_llm
    python run_di_llm.py

前提:
- Azure Document Intelligence のエンドポイント/キーを環境変数から読み込みます。
    AZURE_DI_ENDPOINT, AZURE_DI_KEY
- LLM (Azure OpenAI) は既存の .env / 設定から利用します。

処理:
- 入力 PDF: C:/Users/user/project/uruno_ocr_demo/backend/data/module_test_naka_1.pdf
- 出力:      C:/Users/user/project/uruno_ocr_demo/backend/test/di_llm/output/ YYYYmmdd_HHMMSS/
  - di_page_<n>.csv, di_all.csv
  - llm_page_<n>.csv, llm_all.csv
  - run.log (デバッグ出力)

内容:
- Document Intelligence で文書を1回解析し、ページごとのテーブルを復元
- 各ページのテーブルを LLM に渡して正規化（センター: defalt のスキーマ/プロンプトを使用）
- 2段階の結果をCSVに保存

"""

from __future__ import annotations

import os
import sys
import csv
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Tuple
import re

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]  # backend/
PDF_PATH = BASE_DIR / "data" / "C:/Users/user/project/uruno_ocr_demo/backend/data/那珂市学校給食センター（7月分）.pdf"
OUTPUT_BASE = Path(__file__).resolve().parent / "output"
CENTER_DEFALT_JSON = BASE_DIR / "config" / "centers" / "defalt.json"

# LLM
sys.path.insert(0, str(BASE_DIR))
from config.settings import get_settings  # noqa: E402
from services.llm_table_extractor import LLMTableExtractor, LLMExtractionError  # noqa: E402


def log_print(fp, msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        fp.write(line + "\n")
        fp.flush()
    except Exception:
        pass


def load_center_conf() -> Dict[str, Any]:
    with CENTER_DEFALT_JSON.open("r", encoding="utf-8") as f:
        conf = json.load(f)
    return conf


def prompt_override_for_tables(center_conf: Dict[str, Any]) -> Dict[str, Any]:
    conf = json.loads(json.dumps(center_conf, ensure_ascii=False))  # deep copy
    llm = conf.get("llm") if isinstance(conf.get("llm"), dict) else {}
    # テスト用のプロンプト上書きは行わない（センター設定のプロンプト/プロンプト行をそのまま利用）
    conf["llm"] = llm
    return conf


def analyze_with_di(pdf_bytes: bytes, log) -> Tuple[int, Dict[int, List[List[List[str]]]]]:
    # Load .env from backend if present (idempotent)
    load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)
    # Accept either AZURE_DI_* or AZURE_* variable names
    endpoint = os.environ.get("AZURE_DI_ENDPOINT") or os.environ.get("AZURE_ENDPOINT")
    key = os.environ.get("AZURE_DI_KEY") or os.environ.get("AZURE_KEY")
    if not endpoint or not key:
        raise RuntimeError("AZURE_DI_ENDPOINT / AZURE_DI_KEY が設定されていません")

    client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))

    log("[DI] begin analyze prebuilt-layout")
    poller = client.begin_analyze_document(
        model_id="prebuilt-layout",
        body=pdf_bytes,
        content_type="application/octet-stream",
    )
    result = poller.result()
    log("[DI] analyze done")

    # 可能ならページ数を result.pages から取得
    try:
        page_count = len(result.pages) if getattr(result, "pages", None) else 0
    except Exception:
        page_count = 0

    # ページ -> [テーブル(2D配列)]
    page_tables: Dict[int, List[List[List[str]]]] = {}
    tables = getattr(result, "tables", [])
    log(f"[DI] tables={len(tables)}")
    for t_idx, table in enumerate(tables):
        # ページ番号検出（table.bounding_regions[0].page_number または cell 側）
        page_no = None
        try:
            brs = getattr(table, "bounding_regions", None)
            if brs:
                page_no = getattr(brs[0], "page_number", None)
        except Exception:
            page_no = None
        # セルからページ番号を補完
        if page_no is None:
            try:
                cells = getattr(table, "cells", [])
                if cells:
                    first = cells[0]
                    brs = getattr(first, "bounding_regions", None)
                    if brs:
                        page_no = getattr(brs[0], "page_number", None)
            except Exception:
                page_no = None
        if page_no is None:
            page_no = 1  # 既定

        # テーブル形状
        try:
            row_count = getattr(table, "row_count", None)
            column_count = getattr(table, "column_count", None)
        except Exception:
            row_count = None
            column_count = None

        # 2D配列を復元
        cells = getattr(table, "cells", [])
        max_r = row_count if isinstance(row_count, int) and row_count > 0 else 0
        max_c = column_count if isinstance(column_count, int) and column_count > 0 else 0
        coords: List[Tuple[int, int, str]] = []
        for cell in cells:
            try:
                r = int(getattr(cell, "row_index"))
                c = int(getattr(cell, "column_index"))
                text = (getattr(cell, "content", "") or "").strip()
                coords.append((r, c, text))
                max_r = max(max_r, r + 1)
                max_c = max(max_c, c + 1)
            except Exception:
                continue

        grid: List[List[str]] = [["" for _ in range(max_c or 1)] for _ in range(max_r or 1)]
        for r, c, text in coords:
            try:
                grid[r][c] = text
            except Exception:
                pass

        page_tables.setdefault(int(page_no), []).append(grid)
        log(f"[DI][TABLE] t={t_idx} page={page_no} rows={len(grid)} cols={(len(grid[0]) if grid else 0)}")

    return page_count, page_tables


def write_csv(path: Path, rows: List[List[str]]) -> None:
    """Write CSV ensuring no embedded newlines in cells (normalize to spaces)."""
    def _norm_cell(x: Any) -> str:
        s = "" if x is None else str(x)
        # unify line breaks and collapse whitespace
        s = s.replace("\r\n", "\n").replace("\r", "\n")
        s = re.sub(r"[\n]+", " ", s)
        # remove zero-width and normalize spaces (incl. full-width/nbsp)
        s = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", s)
        s = s.replace("\u00A0", " ").replace("\u3000", " ")
        s = re.sub(r"[ \t\u00A0\u3000]+", " ", s).strip()
        # fix common hyphen line-break joins
        s = re.sub(r"-\s+", "-", s)
        return s

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, lineterminator="\n")
        for r in rows:
            w.writerow([_norm_cell(c) for c in r])


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


def pick_widest_table(tables: List[List[List[str]]]) -> Tuple[List[List[str]], Dict[str, int]]:
    """Pick the table with the most non-empty columns. Tie-break by raw cols, then rows."""
    best_idx = -1
    best_score = (-1, -1, -1)  # (nonempty_cols, raw_cols, rows)
    for i, t in enumerate(tables):
        rows = len(t)
        raw_cols = max((len(r) for r in t), default=0)
        nonempty_cols = _count_nonempty_columns(t)
        score = (nonempty_cols, raw_cols, rows)
        if score > best_score:
            best_score = score
            best_idx = i
    if best_idx < 0:
        return (tables[0] if tables else []), {"rows": 0, "raw_cols": 0, "nonempty_cols": 0, "index": 0}
    chosen = tables[best_idx]
    return chosen, {
        "rows": len(chosen),
        "raw_cols": max((len(r) for r in chosen), default=0),
        "nonempty_cols": _count_nonempty_columns(chosen),
        "index": best_idx,
    }


def clean_table(table: List[List[str]]) -> List[List[str]]:
    cleaned: List[List[str]] = []
    for row in table:
        out_row: List[str] = []
        for cell in row:
            s = "" if cell is None else str(cell)
            # strip DI noise tokens
            s = s.replace(":selected:", "")
            # normalize line breaks and whitespace inside cells
            s = s.replace("\r\n", "\n").replace("\r", "\n")
            s = re.sub(r"[\n]+", " ", s)
            s = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", s)
            s = s.replace("\u00A0", " ").replace("\u3000", " ")
            s = re.sub(r"[ \t\u00A0\u3000]+", " ", s)
            s = re.sub(r"-\s+", "-", s).strip()
            out_row.append(s)
        cleaned.append(out_row)
    return cleaned


def drop_fully_empty_rows(table: List[List[str]]) -> List[List[str]]:
    """Remove rows where all cells are empty after stripping."""
    out: List[List[str]] = []
    for row in table:
        if any((str(c).strip() if c is not None else "") for c in row):
            out.append(row)
    return out


def has_header_keywords(row: List[str]) -> bool:
    keys = ["番号", "商品名", "銘柄", "備考"]
    s = " ".join((str(c) for c in row)).strip()
    return all(k in s for k in keys)


def main() -> int:
    run_dir = OUTPUT_BASE / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    with log_path.open("w", encoding="utf-8") as log_fp:
        def LOG(msg: str) -> None:
            log_print(log_fp, msg)

        LOG("DI+LLM test start")
        LOG(f"input={PDF_PATH}")
        if not PDF_PATH.exists():
            LOG("[ERROR] input PDF not found")
            return 2

        pdf_bytes = PDF_PATH.read_bytes()

        # Document Intelligence
        try:
            pages_total, page_tables = analyze_with_di(pdf_bytes, LOG)
        except Exception as exc:
            LOG(f"[ERROR][DI] {type(exc).__name__}: {exc}")
            return 3

        LOG(f"[INFO] pages_total={pages_total} (DI tables aggregated by page={len(page_tables)})")

        # DI 結果をCSV出力（ページ内に複数テーブルがある場合は列数が多いものを採用）
        di_all: List[List[str]] = []
        for page_no in sorted(page_tables.keys()):
            tables = page_tables[page_no]
            chosen, meta = pick_widest_table(tables)
            # debug: list candidates summary
            cand_summaries = []
            for idx, t in enumerate(tables):
                rows = len(t)
                raw_cols = max((len(r) for r in t), default=0)
                nonempty_cols = _count_nonempty_columns(t)
                cand_summaries.append(f"#{idx}:rows={rows},raw_cols={raw_cols},nonempty_cols={nonempty_cols}")
            LOG(f"[DI][PAGE] {page_no}: tables={len(tables)} candidates=[{'; '.join(cand_summaries)}]")
            LOG(f"[DI][PAGE] {page_no}: selected index={meta['index']} rows={meta['rows']} raw_cols={meta['raw_cols']} nonempty_cols={meta['nonempty_cols']}")

            chosen = clean_table(chosen)
            out_rows: List[List[str]] = [[str(c) for c in r] for r in chosen]
            di_page_path = run_dir / f"di_page_{page_no}.csv"
            write_csv(di_page_path, out_rows)
            di_all.extend(out_rows)

        if di_all:
            write_csv(run_dir / "di_all.csv", di_all)

        # LLM による正規化
        center_conf = load_center_conf()
        center_conf = prompt_override_for_tables(center_conf)
        settings = get_settings()
        extractor = LLMTableExtractor(settings)
        if not extractor.is_available:
            LOG("[ERROR] Azure OpenAI not configured")
            return 4

        llm_all_rows: List[List[str]] = []
        header_saved: List[str] = []
        for page_no in sorted(page_tables.keys()):
            tables = page_tables[page_no]
            chosen, meta = pick_widest_table(tables)
            chosen = clean_table(chosen)
            # Drop fully empty rows to avoid confusing the model
            chosen = drop_fully_empty_rows(chosen)
            # Compute expected data row count and add a page-specific prompt hint
            expected_total_rows = len(chosen)
            header_present = expected_total_rows > 0 and has_header_keywords(chosen[0])
            expected_data_rows = max(0, expected_total_rows - (1 if header_present else 0))
            # Create a per-page conf with dynamic instruction
            conf_page = json.loads(json.dumps(center_conf, ensure_ascii=False))
            llm_conf = conf_page.get("llm", {})
            lines = list(llm_conf.get("prompt_lines", []))
            lines.append(
                f"[DIメタ] このページのテーブルは合計{expected_total_rows}行です。"
                + ("先頭行はヘッダーです。" if header_present else "先頭行はヘッダーではありません。")
                + f"データ行は合計{expected_data_rows}行あり、順序を保って全件（欠損は空文字）出力してください。省略禁止。"
            )
            llm_conf["prompt_lines"] = lines
            conf_page["llm"] = llm_conf
            try:
                meta = {"source": "di", "image_count": 0, "pages": [page_no]}
                res = extractor.extract(
                    tables=[chosen],
                    images=None,
                    center_conf=conf_page,
                    center_id=conf_page.get("id"),
                    log_fn=lambda m: LOG(f"[LLM] {m}"),
                    meta=meta,
                )
            except LLMExtractionError as exc:
                LOG(f"[ERROR][LLM][PAGE {page_no}] {exc}")
                continue
            except Exception as exc:
                LOG(f"[ERROR][LLM][PAGE {page_no}] {type(exc).__name__}: {exc}")
                continue

            page_table = res.to_table()
            LOG(f"[LLM][PAGE] {page_no}: rows={len(page_table)-1} cols={len(page_table[0]) if page_table else 0}")

            # ページ別出力
            # normalize cells before writing (guard against any stray newlines)
            write_csv(run_dir / f"llm_page_{page_no}.csv", page_table)

            # 結合（ヘッダーは1回だけ）
            if page_table:
                if not header_saved:
                    header_saved = list(page_table[0])
                    llm_all_rows.append(header_saved)
                llm_all_rows.extend(page_table[1:])

        if llm_all_rows:
            write_csv(run_dir / "llm_all.csv", llm_all_rows)

        LOG("DI+LLM test done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
