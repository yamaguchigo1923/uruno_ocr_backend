from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Ensure 'services' package (backend/services) is importable even if this
# script is executed from various working directories.
ROOT = Path(__file__).resolve().parents[2]  # backend/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.image_preprocessing import process_file
from services.ocr_client import DocumentAnalyzer
from config.settings import get_settings

# Optional table detection: try to import OpenCV and pytesseract. If not
# available, detect_tables will return an empty list and the CSV field will
# remain empty.
try:
    import cv2
    import numpy as np
except Exception:
    cv2 = None
    np = None

try:
    import pytesseract
except Exception:
    pytesseract = None


def detect_tables(image_path: str) -> List[Dict[str, Any]]:
    """Detect table-like rectangular regions in the image.

    Returns a list of dicts: {"bbox": [x,y,w,h], "text": "..."}
    """
    if cv2 is None:
        print("[run_preprocess] OpenCV not installed — skipping table detection")
        return []

    img = cv2.imread(image_path)
    if img is None:
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Binarize
    thr = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

    # Detect horizontal and vertical lines
    horizontal = thr.copy()
    vertical = thr.copy()
    cols = horizontal.shape[1]
    horizontal_size = max(1, cols // 30)
    horiz_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv2.erode(horizontal, horiz_structure)
    horizontal = cv2.dilate(horizontal, horiz_structure)

    rows = vertical.shape[0]
    vertical_size = max(1, rows // 30)
    vert_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    vertical = cv2.erode(vertical, vert_structure)
    vertical = cv2.dilate(vertical, vert_structure)

    # Combine
    mask = horizontal + vertical

    # Find contours from mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tables: List[Dict[str, Any]] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Filter small regions
        if w < 50 or h < 50:
            continue
        crop = img[y : y + h, x : x + w]
        text = ""
        if pytesseract is not None:
            try:
                text = pytesseract.image_to_string(crop, lang='jpn+eng')
            except Exception:
                text = ""
        tables.append({"bbox": [int(x), int(y), int(w), int(h)], "text": text})

    # Sort by y then x for determinism
    tables.sort(key=lambda t: (t["bbox"][1], t["bbox"][0]))
    return tables


"""
Configuration-style test runner for image/PDF cropping.

Edit the constants below and run this file directly from the `backend`
directory:

  python test\image_preprocessing\run_preprocess.py

No command-line arguments are required; this matches your request to put
settings at the top of the test file.
"""

# === Configuration - edit these values ===
# Input file (image or PDF). Use a path relative to the backend directory
# or an absolute Windows path.
INPUT_PATH = "C:\\Users\\user\\project\\uruno_ocr_demo\\backend\\test\\image_preprocessing\\data\\鉾田学校給食センター（10月分）.pdf"

# Output directory where cropped images and CSV will be written.
OUTPUT_DIR = r"C:\\Users\\user\\project\\uruno_ocr_demo\\backend\\test\\image_preprocessing\\output"

# Cropping percentages (fractions 0.0 - 1.0)
TOP_PCT = 0.1
BOTTOM_PCT = 0.0
LEFT_PCT = 0.0
RIGHT_PCT = 0.0
# =======================================


def run_test() -> int:
    input_path = Path(INPUT_PATH)
    if not input_path.exists():
        print(f"Input not found: {input_path}")
        return 2

    base_outdir = Path(OUTPUT_DIR)
    base_outdir.mkdir(parents=True, exist_ok=True)

    # Create a per-run output directory so each execution's results are grouped.
    run_dir = base_outdir / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        metas = process_file(
            str(input_path),
            str(run_dir),
            top_pct=TOP_PCT,
            bottom_pct=BOTTOM_PCT,
            left_pct=LEFT_PCT,
            right_pct=RIGHT_PCT,
            out_format="PNG",
        )
    except Exception as exc:
        print(f"Processing failed: {exc}")
        return 3

    # Prepare Azure Document Intelligence analyzer if configured.
    settings = get_settings()
    use_azure = bool(settings.azure_endpoint and settings.azure_key)
    analyzer = None
    if use_azure:
        try:
            analyzer = DocumentAnalyzer(settings)
        except Exception as exc:  # pragma: no cover - runtime configuration error
            print(f"[run_preprocess] Azure Document Intelligence not available: {exc}")
            analyzer = None

    csv_path = run_dir / "results.csv"
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as csvf:
        writer = csv.DictWriter(
            csvf,
            fieldnames=[
                "timestamp",
                "input_path",
                "page",
                "output_image_path",
                "orig_width",
                "orig_height",
                "new_width",
                "new_height",
                "top_pct",
                "bottom_pct",
                "left_pct",
                "right_pct",
            ],
        )
        if write_header:
            writer.writeheader()
        for meta in metas:
            writer.writerow(
                {
                    "timestamp": meta.get("timestamp"),
                    "input_path": meta.get("input_path"),
                    "page": meta.get("page"),
                    "output_image_path": meta.get("output_image_path"),
                    "orig_width": meta.get("orig_width"),
                    "orig_height": meta.get("orig_height"),
                    "new_width": meta.get("new_width"),
                    "new_height": meta.get("new_height"),
                    "top_pct": meta.get("top_pct"),
                    "bottom_pct": meta.get("bottom_pct"),
                    "left_pct": meta.get("left_pct"),
                    "right_pct": meta.get("right_pct"),
                }
            )

            # Also write a per-image CSV so it's easy to see which tables/pages
            # came from each output image. We'll run Azure Document Intelligence
            # (preferred) or fall back to local OpenCV/pytesseract detection.
            out_image = Path(str(meta.get("output_image_path")))
            per_csv = run_dir / f"{out_image.stem}.csv"
            per_header = [
                "timestamp",
                "input_path",
                "page",
                "output_image_path",
                "orig_width",
                "orig_height",
                "new_width",
                "new_height",
                "top_pct",
                "bottom_pct",
                "left_pct",
                "right_pct",
                "recognized_tables",
            ]
            write_per = not per_csv.exists()
            with open(per_csv, "a", newline="", encoding="utf-8") as pf:
                pwriter = csv.DictWriter(pf, fieldnames=per_header)
                if write_per:
                    pwriter.writeheader()
                # Run lightweight table detection and OCR (if available)
                    recognized = ""
                    # Prefer Azure Document Intelligence if configured
                    if analyzer is not None:
                        try:
                            b = Path(out_image).read_bytes()
                            analyzed = analyzer.analyze_content(b)
                            # Convert to serializable structure
                            tables_for_json = []
                            for t in analyzed.tables:
                                tables_for_json.append(
                                    {
                                        "page_number": t.page_number,
                                        "row_count": t.row_count,
                                        "column_count": t.column_count,
                                        "rows": t.rows,
                                    }
                                )
                            recognized = json.dumps(tables_for_json, ensure_ascii=False)
                        except Exception as exc:  # pragma: no cover - runtime failures
                            print(f"[run_preprocess] Azure analyze failed for {out_image}: {exc}")
                            recognized = ""
                    else:
                        try:
                            tables = detect_tables(str(out_image))
                            recognized = json.dumps(tables, ensure_ascii=False)
                        except Exception as _exc:
                            print(f"[run_preprocess] table detection failed for {out_image}: {_exc}")
                            recognized = ""

                pwriter.writerow(
                    {
                        "timestamp": meta.get("timestamp"),
                        "input_path": meta.get("input_path"),
                        "page": meta.get("page"),
                        "output_image_path": meta.get("output_image_path"),
                        "orig_width": meta.get("orig_width"),
                        "orig_height": meta.get("orig_height"),
                        "new_width": meta.get("new_width"),
                        "new_height": meta.get("new_height"),
                        "top_pct": meta.get("top_pct"),
                        "bottom_pct": meta.get("bottom_pct"),
                        "left_pct": meta.get("left_pct"),
                        "right_pct": meta.get("right_pct"),
                        "recognized_tables": recognized,
                    }
                )

    print("Processed outputs:")
    for meta in metas:
        print(f" - page={meta.get('page')} -> {meta.get('output_image_path')}")
    print("Results appended to:", str(csv_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(run_test())
