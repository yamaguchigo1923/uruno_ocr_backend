from __future__ import annotations

"""成分表/見本ラベル 精度テストスクリプト

実行例:
    cd /d C:/Users/user/project/uruno_ocr_demo/backend
    python -m test.label_test.run_label_test

前提:
- テストPDF: backend/data/pdf_data/<IDNUM>_<MONTH>.pdf  (例: 001_9.pdf)
- 正解ラベルCSV: backend/data/label_data/<IDNUM>_<MONTH>.csv
    - 1行目ヘッダ: 番号,成分表,見本
    - 2行目以降に行データ。成分表/見本は TRUE/FALSE

出力:
- 1テストにつき、CSVの4列目以降に結果列を1列追加
    (ヘッダ = 実行日時、セル値 = "T,F" のような文字列)
- 最終行に精度/処理時間/トークン数などのメトリクスを追記
- ログには実行ID/PDF名/ページ数/テストスタイル/LLMモデル/精度/処理時間/トークン数などを表示
"""

import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]  # backend/

# backend パスを import パスに追加
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.settings import get_settings  # noqa: E402
from config.centers.loader import get_center_config  # noqa: E402
from services.llm_table_extractor import LLMTableExtractor, LLMExtractionError  # noqa: E402
from azure.core.credentials import AzureKeyCredential  # noqa: E402
from azure.ai.documentintelligence import DocumentIntelligenceClient  # noqa: E402
from utils.logger import get_logger  # noqa: E402

try:  # パッケージとして実行される場合
    from .row_alignment import align_rows_by_number  # type: ignore
except ImportError:  # スクリプト直接実行 (python run_label_test.py) 用フォールバック
    from row_alignment import align_rows_by_number  # type: ignore

logger = get_logger("test.label_test")


@dataclass
class LabelTestCase:
    id_num: str
    month: int
    test_style: int  # 1: DIのみ, 2: DI+LLM
    page_skip: int = 0  # PDF先頭からスキップするページ数


def load_test_config() -> List[LabelTestCase]:
    cfg_path = Path(__file__).with_name("label_test_config.json")
    import json

    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    cases: List[LabelTestCase] = []
    for item in data.get("tests", []):
        cases.append(
            LabelTestCase(
                id_num=str(item.get("id_num")),
                month=int(item.get("month")),
                test_style=int(item.get("test_style", 2)),
                page_skip=int(item.get("page_skip", 0)),
            )
        )
    return cases


def main() -> None:
    load_dotenv(BASE_DIR / ".env")
    settings = get_settings()

    cases = load_test_config()
    if not cases:
        print("no tests defined in label_test_config.json")
        return

    for case in cases:
        run_single_case(case, settings)


def run_single_case(case: LabelTestCase, settings) -> None:
    center_id = resolve_center_id_from_num(case.id_num)
    pdf_name = f"{case.id_num}_{case.month}.pdf"
    pdf_path = BASE_DIR / "data" / "pdf_data" / pdf_name

    csv_name = f"{case.id_num}_{case.month}.csv"
    csv_path = BASE_DIR / "data" / "label_data" / csv_name

    print(f"[TEST] id_num={case.id_num} center_id={center_id} month={case.month}")
    print(f"[TEST] pdf={pdf_path}")
    print(f"[TEST] test_style={case.test_style} (1=DI only, 2=DI+LLM)")
    if case.page_skip:
        print(f"[TEST] page_skip={case.page_skip} (skip first {case.page_skip} pages)")

    if not pdf_path.exists():
        print(f"[TEST][ERROR] pdf not found: {pdf_path}")
        return

    if not csv_path.exists():
        print(f"[TEST][ERROR] label csv not found: {csv_path}")
        return

    center_conf = get_center_config(center_id) or {}

    # PDF 読み込み
    pdf_bytes = pdf_path.read_bytes()

    # --- Document Intelligence でテーブル抽出 ---
    print("[TEST][DI] start prebuilt-layout")
    di_page_count, di_page_tables = analyze_with_di_for_test(pdf_bytes)

    # page_skip 分を先頭からスキップ
    if case.page_skip > 0:
        # page_no は 1 始まりとして扱っているので、その分を削除
        for p in range(1, case.page_skip + 1):
            di_page_tables.pop(p, None)
        # 実際に使うページ数を再計算
        used_pages = sorted(di_page_tables.keys())
        print(
            f"[TEST][DI] pages={di_page_count} tables_by_page={len(di_page_tables)} (skip first {case.page_skip} pages, using pages={used_pages})"
        )
    else:
        print(f"[TEST][DI] pages={di_page_count} tables_by_page={len(di_page_tables)}")

    # 進捗ログ: DIページ単位
    start = time.time()
    for i in range(1, di_page_count + 1):
        if i <= case.page_skip:
            print(f"[TEST][PROGRESS] page {i}/{di_page_count} ... (skipped)")
        else:
            print(f"[TEST][PROGRESS] page {i}/{di_page_count} ...")

    # LLM利用モデル名
    llm_conf = center_conf.get("llm") if isinstance(center_conf, dict) else None
    llm_model = "(no LLM)"
    if isinstance(llm_conf, dict) and llm_conf.get("enabled", True):
        deployment = settings.aoai_deployment
        llm_model = deployment or "AOAI (configured)"
    print(f"[TEST] LLM model={llm_model}")

    # --- LLMで表正規化 (DIテーブルを入力) ---
    header, data_rows, used_tokens = run_llm_extraction_from_di(
        di_page_tables, center_conf, settings
    )

    # --- CSV正解との比較 ---
    gt_rows = load_label_csv(csv_path)
    print(f"[TEST] label_csv={csv_path} rows={len(gt_rows)}")

    # 行マッチング開始ログ
    print("[TEST][ALIGN] start number-based row alignment ...")

    # 行マッチング: 正解番号(1列目)と LLM 番号列でシーケンスアラインメント
    gt_numbers: List[int] = []
    import csv

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # ヘッダ
        for row in reader:
            if not row:
                continue
            try:
                n = int(str(row[0]).strip()) if str(row[0]).strip() else None
            except ValueError:
                n = None
            if n is not None:
                gt_numbers.append(n)
            else:
                # 番号が取れない行は、アラインメント上は欠番として扱う
                gt_numbers.append(-10**9)

    # LLM 側の番号リストは build_prediction_map のヘッダ・行から取得
    pred_raw = build_prediction_map(header, data_rows)
    ocr_numbers: List[int] = []
    max_idx_pred = max(pred_raw.keys(), default=-1)
    for i in range(max_idx_pred + 1):
        num_str = pred_raw.get(i, ("", False, False))[0]
        try:
            n = int(num_str) if num_str else None
        except ValueError:
            n = None
        if n is not None:
            ocr_numbers.append(n)
        else:
            ocr_numbers.append(-10**9)

    pairs = align_rows_by_number(gt_numbers, ocr_numbers)
    print(f"[TEST][ALIGN] pairs={len(pairs)} (gt={len(gt_numbers)}, ocr={len(ocr_numbers)})")

    # アラインメント結果に従って、評価用の pred_map を再構築
    aligned_pred_map: Dict[int, Tuple[str, bool, bool]] = {}
    for gi, oj in pairs:
        if gi < 0 or gi >= len(gt_rows):
            continue
        if oj < 0 or oj > max_idx_pred:
            continue
        num_str, s_flag, m_flag = pred_raw.get(oj, ("", False, False))
        aligned_pred_map[gi] = (num_str, s_flag, m_flag)

    # T,F 判定はマッチング後の行について実施
    accuracy = compute_accuracy(gt_rows, aligned_pred_map)

    duration = time.time() - start
    print(f"[TEST][RESULT] accuracy={accuracy * 100:.1f}%")
    print(f"[TEST][RESULT] duration={duration:.2f}s")
    print(f"[TEST][RESULT] tokens={used_tokens}")

    write_result_to_csv(csv_path, gt_rows, aligned_pred_map, accuracy, duration, used_tokens)
    print(f"[TEST] csv write done: {csv_path}")


def resolve_center_id_from_num(id_num: str) -> str:
    """一旦、id_num から center_id へのマップは仮置き。

    後で spreadsheet のサマリーシート等から自動解決するよう拡張可能。
    今は全センター config に id_num="000" を入れているので、
    実際のIDが決まったらここにマップを書くイメージです。
    """

    # centers フォルダ配下の json を走査して id_num が一致するものを探す
    import json

    centers_dir = BASE_DIR / "config" / "centers"
    for path in centers_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(data.get("id_num")) == str(id_num):
            return str(data.get("id") or path.stem)

    # 見つからない場合は defalt
    return "defalt"


def analyze_with_di_for_test(pdf_bytes: bytes) -> Tuple[int, Dict[int, List[List[List[str]]]]]:
    """Document Intelligence prebuilt-layout を使って PDF からページ別テーブル群を取得。"""

    # .env から AZURE_DI_ENDPOINT/AZURE_DI_KEY または AZURE_ENDPOINT/AZURE_KEY を読み込む
    load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)
    endpoint = os.environ.get("AZURE_DI_ENDPOINT") or os.environ.get("AZURE_ENDPOINT")
    key = os.environ.get("AZURE_DI_KEY") or os.environ.get("AZURE_KEY")
    if not endpoint or not key:
        raise RuntimeError("AZURE_DI_ENDPOINT / AZURE_DI_KEY が設定されていません")

    client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))

    poller = client.begin_analyze_document(
        model_id="prebuilt-layout",
        body=pdf_bytes,
        content_type="application/octet-stream",
    )
    result = poller.result()

    try:
        page_count = len(result.pages) if getattr(result, "pages", None) else 0
    except Exception:
        page_count = 0

    page_tables: Dict[int, List[List[List[str]]]] = {}
    tables = getattr(result, "tables", [])
    for table in tables:
        # ページ番号
        page_no = None
        try:
            brs = getattr(table, "bounding_regions", None)
            if brs:
                page_no = getattr(brs[0], "page_number", None)
        except Exception:
            page_no = None
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
            page_no = 1

        try:
            row_count = getattr(table, "row_count", None)
            column_count = getattr(table, "column_count", None)
        except Exception:
            row_count = None
            column_count = None

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

    return page_count, page_tables


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
    """テーブル群から列情報が最もリッチなものを1つ選ぶ。"""

    if not tables:
        return []
    best_idx = 0
    best_score = (-1, -1, -1)  # (nonempty_cols, raw_cols, rows)
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
    for row in table:
        out_row: List[str] = []
        for cell in row:
            s = "" if cell is None else str(cell)
            s = s.replace(":selected:", "")
            s = s.replace("\r\n", "\n").replace("\r", "\n")
            import re as _re

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


def run_llm_extraction_from_di(
    di_page_tables: Dict[int, List[List[List[str]]]],
    center_conf: Dict[str, Any],
    settings,
) -> Tuple[List[str], List[List[str]], int]:
    """DI結果のテーブル群をページごとに直列でLLM正規化する。"""

    extractor = LLMTableExtractor(settings)
    header: List[str] = []
    data_rows: List[List[str]] = []
    total_tokens = 0

    if not extractor.is_available:
        print("[TEST][WARN] Azure OpenAI not configured; skipping LLM")
        return header, data_rows, total_tokens

    center_id = center_conf.get("id") if isinstance(center_conf, dict) else None

    # ページごとのテーブルから「最も情報量の多いテーブル」を1つずつ選び、
    # ページ番号の昇順で直列に LLM に投げる
    import json as _json

    for page_no in sorted(di_page_tables.keys()):
        page_start = time.time()
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
        llm_conf = conf_page.get("llm", {}) or {}
        lines = list(llm_conf.get("prompt_lines", []))
        lines.append(
            f"[DIメタ] このページのテーブルは合計{total_rows}行です。"
            + ("先頭行はヘッダーです。" if header_present else "先頭行はヘッダーではありません。")
            + f"データ行は合計{expected_data_rows}行あり、順序を保って全件（欠損は空文字）出力してください。省略禁止。"
        )
        llm_conf["prompt_lines"] = lines
        conf_page["llm"] = llm_conf

        # LLM 呼び出しにリトライ + バックオフを入れる
        meta = {"source": "di", "image_count": 0, "pages": [page_no]}
        max_retries = 3
        backoff_base = 3.0
        attempt = 0
        timeout_count = 0
        result = None
        while attempt < max_retries:
            attempt += 1
            attempt_start = time.time()
            try:
                result = extractor.extract(
                    tables=[chosen],
                    images=None,
                    center_conf=conf_page,
                    center_id=center_id,
                    log_fn=lambda m: print(f"[LLM][p{page_no}] {m}"),
                    meta=meta,
                )
                elapsed = time.time() - attempt_start
                # 30秒を超えて返ってきた場合はタイムアウト扱い
                if elapsed >= 30.0:
                    print(
                        f"[TEST][LLM][TIMEOUT] page={page_no} attempt={attempt} elapsed={elapsed:.2f}s (>30s)"
                    )
                    timeout_count += 1
                    # タイムアウトが発生した場合も、max_retries の範囲でバックオフして再試行する
                    if attempt >= max_retries:
                        # 上限回数に到達したら、その時点の結果を採用してブレーク
                        print(
                            f"[TEST][LLM][INFO] page={page_no}: reached max_retries after timeout (count={timeout_count}), proceed with this result"
                        )
                        break
                    sleep_sec = backoff_base * attempt
                    print(
                        f"[TEST][LLM][INFO] page={page_no}: retry after timeout in {sleep_sec:.1f}s (timeout_count={timeout_count}) ..."
                    )
                    result = None
                    time.sleep(sleep_sec)
                    continue
                # 正常終了（30秒未満）
                break
            except LLMExtractionError as exc:
                # 明示的な拒否・設定ミスなどは即座にログして中断
                print(f"[TEST][LLM][ERROR] page={page_no} attempt={attempt}: {exc}")
                break
            except Exception as exc:
                # 一時的な失敗とみなしてリトライ（最大 max_retries 回）
                print(
                    f"[TEST][LLM][WARN] page={page_no} attempt={attempt} failed: {type(exc).__name__}: {exc}"
                )
                if attempt >= max_retries:
                    print(f"[TEST][LLM][ERROR] page={page_no}: giving up after {attempt} attempts")
                    break
                sleep_sec = backoff_base * attempt
                print(f"[TEST][LLM][INFO] page={page_no}: retrying in {sleep_sec:.1f}s ...")
                time.sleep(sleep_sec)

        if result is None:
            continue

        if not header:
            header = list(result.header)
        data_rows.extend(result.data_rows)
        page_elapsed = time.time() - page_start
        print(
            f"[TEST][LLM][PAGE] {page_no} rows+={len(result.data_rows)} total={len(data_rows)} "
            f"time={page_elapsed:.2f}s"
        )

    print(f"[TEST][LLM] total_rows={len(data_rows)} cols={len(header) if header else 0}")
    return header, data_rows, total_tokens


def load_label_csv(csv_path: Path) -> List[Tuple[int, bool, bool]]:
    """CSVから (行インデックス, 成分表, 見本) のリストを読み込む。

    評価用のキーは行インデックスのみを使い、
    番号は LLM(DI) 側の出力を結果列に表示する。
    """

    import csv

    rows: List[Tuple[int, bool, bool]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # ヘッダ
        for idx, row in enumerate(reader):
            if not row or len(row) < 3:
                continue

            def _to_bool(v: str) -> bool:
                s = (v or "").strip().upper()
                return s in {"TRUE", "T", "1", "Y", "YES"}

            s_flag = _to_bool(row[1])
            m_flag = _to_bool(row[2])
            rows.append((idx, s_flag, m_flag))
    return rows


def build_prediction_map(
    header: List[str], data_rows: List[List[str]]
) -> Dict[int, Tuple[str, bool, bool]]:
    """LLM結果テーブルから 行インデックス -> (番号, 成分表, 見本) へのマップを作る。

    LLM出力の1行目を index=0 として順番に評価し、
    CSV側の2行目以降と行順で対応付ける。
    """

    pred: Dict[int, Tuple[str, bool, bool]] = {}
    if not header:
        return pred

    def _find_idx(name: str) -> Optional[int]:
        try:
            return header.index(name)
        except ValueError:
            return None

    num_idx = _find_idx("番号")
    s_idx = _find_idx("成分表")
    m_idx = _find_idx("見本")
    if s_idx is None or m_idx is None:
        print("[TEST][WARN] header does not contain 成分表/見本")
        return pred

    for idx, row in enumerate(data_rows):
        if len(row) <= max(x for x in [num_idx, s_idx, m_idx] if x is not None):
            continue
        num_val = str(row[num_idx]).strip() if num_idx is not None and num_idx < len(row) else ""
        s_val = str(row[s_idx]).strip()
        m_val = str(row[m_idx]).strip()
        s_flag = s_val in {"○", "1", "TRUE", "true"}
        m_flag = m_val in {"3", "○", "1", "TRUE", "true"}
        pred[idx] = (num_val, s_flag, m_flag)

    return pred


def _score_number(gt: int, ocr: int) -> int:
    """正解番号とOCR番号の近さに基づくスコア。

    |gt-ocr|==0: +5, 1: +2, 2: +1, それ以外: -3
    """

    d = abs(gt - ocr)
    if d == 0:
        return 5
    if d == 1:
        return 2
    if d == 2:
        return 1
    return -3


def align_rows_by_number(gt_numbers: List[int], ocr_numbers: List[int]) -> List[Tuple[int, int]]:
    """番号のみを用いたシーケンスアラインメントで行マッチングを行う。

    入力:
        gt_numbers: 正解側番号配列 (長さ N)
        ocr_numbers: OCR側番号配列 (長さ M)

    出力:
        (gi, oj) のペアリスト。gi/oj は 0 始まりインデックス。
    """

    n = len(gt_numbers)
    m = len(ocr_numbers)
    gap_penalty = -2

    # dp と trace を (n+1) x (m+1) で確保
    dp: List[List[int]] = [[0] * (m + 1) for _ in range(n + 1)]
    trace: List[List[int]] = [[0] * (m + 1) for _ in range(n + 1)]

    # 初期化
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + gap_penalty
        trace[i][0] = 1  # 正解側スキップ
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + gap_penalty
        trace[0][j] = 2  # OCR側スキップ

    # 遷移
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match_score = dp[i - 1][j - 1] + _score_number(gt_numbers[i - 1], ocr_numbers[j - 1])
            skip_gt = dp[i - 1][j] + gap_penalty
            skip_ocr = dp[i][j - 1] + gap_penalty

            best = match_score
            t = 0
            if skip_gt > best:
                best = skip_gt
                t = 1
            if skip_ocr > best:
                best = skip_ocr
                t = 2

            dp[i][j] = best
            trace[i][j] = t

    # 復元
    pairs: List[Tuple[int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        t = trace[i][j]
        if i > 0 and j > 0 and t == 0:
            # マッチ
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or t == 1):
            # 正解側スキップ (i-1 行目がどの OCR 行にも対応しない)
            i -= 1
        elif j > 0 and (i == 0 or t == 2):
            # OCR側スキップ (j-1 行目がどの 正解 行にも対応しない)
            j -= 1
        else:
            # 保険: どれにも当てはまらない場合は両方デクリメント
            if i > 0:
                i -= 1
            if j > 0:
                j -= 1

    pairs.reverse()
    return pairs


def compute_accuracy(
    gt_rows: List[Tuple[int, bool, bool]],
    pred_map: Dict[int, Tuple[str, bool, bool]],
) -> float:
    if not gt_rows:
        return 0.0
    total = 0
    correct = 0
    for idx, s_gt, m_gt in gt_rows:
        _num_pred, s_pred, m_pred = pred_map.get(idx, ("", False, False))
        if s_gt == s_pred and m_gt == m_pred:
            correct += 1
        total += 1
    return correct / total if total else 0.0


def write_result_to_csv(
    csv_path: Path,
    gt_rows: List[Tuple[int, bool, bool]],
    pred_map: Dict[int, Tuple[str, bool, bool]],
    accuracy: float,
    duration: float,
    used_tokens: int,
) -> None:
    """元CSVに4列目以降として結果列を追加して上書き保存する。"""

    import csv

    # 既存の全データを読み取り
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = list(csv.reader(f))

    if not reader:
        return

    header = reader[0]
    body = reader[1:]

    # 4列目以降の最初の空き列に書くイメージだが、簡易に常に末尾に追加
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    result_header = ts

    # ヘッダ拡張
    header.append(result_header)

    # 本文行: gt_rows と同じ行インデックスに結果を書き込む
    for idx, s_gt, m_gt in gt_rows:
        if idx < 0 or idx >= len(body):
            continue
        num_str, s_pred, m_pred = pred_map.get(idx, ("", False, False))
        val = f"{num_str},{'T' if s_pred else 'F'},{'T' if m_pred else 'F'}"
        # 行の長さを揃える
        while len(body[idx]) < len(header) - 1:
            body[idx].append("")
        body[idx].append(val)

    # メトリクスは「出力列の最後の行」に1セルで入れる
    metric = f"accuracy={accuracy*100:.1f}%, duration={duration:.2f}s, tokens={used_tokens}"
    if body:
        # 最終行の結果セル（今回追加した列）にメトリクスを書き込む
        last_idx = len(body) - 1
        while len(body[last_idx]) < len(header) - 1:
            body[last_idx].append("")
        body[last_idx].append(metric)

    # 書き戻し
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(body)


if __name__ == "__main__":
    main()
