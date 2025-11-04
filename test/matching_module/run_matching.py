"""Test runner for matching modules.

Usage:
  1. Set INPUT_PATH to a CSV / image / PDF containing OCR table data.
  2. Set CENTER_CONFIG_PATH to the center config JSON to load module settings.
  3. Optionally adjust TARGET_OVERRIDES to tweak config without editing JSON.
  4. Run `python run_matching.py` (from this directory or project root).

The script writes `backend/test/matching_module/result.csv` and a timestamped copy.
"""
from __future__ import annotations

import copy
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

CURRENT_DIR = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_DIR
for _ in range(6):
    if (PROJECT_ROOT / "backend").exists():
        break
    PROJECT_ROOT = PROJECT_ROOT.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
for candidate in (BACKEND_DIR, PROJECT_ROOT):
    if candidate.exists():
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)

# NOTE: engine (DEFAULT_CONFIGS, build_matching_table) is imported lazily inside main()

# === Configuration (edit these values) ======================================
# Input file: CSV (preferred for testing) or image/PDF when Azure Document Intelligence is configured.
INPUT_PATH = "C:\\Users\\user\\project\\uruno_ocr_demo\\backend\\data\\module_test_kitaibara.pdf"
# Center config to use for module settings (set to the active center when running real data).
CENTER_CONFIG_PATH = "../../config/centers/kitaibaraki.json"
# Per-target overrides if you need to tweak config without editing the JSON.
TARGET_OVERRIDES: Dict[str, Dict[str, Any]] = {
    # Example override:
    # "nutrition": {"matching": "別の文字列"},
}
# Target evaluation order (each key should exist in the center config with a `module` list).
TARGET_ORDER = ["nutrition", "sample"]

# Fallback configs used when a center JSON lacks the matching block entirely.
# Will be loaded lazily from engine.DEFAULT_CONFIGS inside main()
DEFAULT_CONFIGS: Dict[str, Dict[str, Any]] = {}
# ==========================================================================


def read_csv(path: str) -> List[List[str]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        return [row for row in reader]


def is_csv(path: str) -> bool:
    return path.lower().endswith(".csv")


def is_pdf(path: str) -> bool:
    return path.lower().endswith(".pdf")


def is_image(path: str) -> bool:
    for ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"):
        if path.lower().endswith(ext):
            return True
    return False


def load_rows_from_input(input_path: str, center_conf: Dict[str, Any] | None = None) -> List[List[str]] | None:
    if is_csv(input_path):
        return read_csv(input_path)
    if not (is_pdf(input_path) or is_image(input_path)):
        return None

    try:
        from config.settings import get_settings
        from backend.services.ocr_client import DocumentAnalyzer
        # optional cropping helpers
        from backend.services.image_preprocessing import crop_image_bytes, _pdf_bytes_to_images
    except Exception as exc:  # pragma: no cover - optional dependency
        print("OCR capabilities not available in this environment.")
        print("Provide a CSV input or ensure Azure settings and ocr_client are available.")
        print(f"Import error: {exc}")
        return None

    settings = get_settings()
    try:
        analyzer = DocumentAnalyzer(settings)
    except Exception as exc:
        print(f"Failed to instantiate DocumentAnalyzer: {exc}")
        return None

    with open(input_path, "rb") as f:
        content = f.read()

    # Optional pre-OCR crop: apply when center config provides non-zero percentages.
    crop_conf = (center_conf or {}).get("crop", {}) if isinstance(center_conf, dict) else {}
    top_pct = float(crop_conf.get("top_pct", 0.0) or 0.0)
    bottom_pct = float(crop_conf.get("bottom_pct", 0.0) or 0.0)
    left_pct = float(crop_conf.get("left_pct", 0.0) or 0.0)
    right_pct = float(crop_conf.get("right_pct", 0.0) or 0.0)
    apply_crop = any(v > 0 for v in (top_pct, bottom_pct, left_pct, right_pct))

    try:
        if is_image(input_path) and apply_crop:
            cropped_bytes, _, _ = crop_image_bytes(content, top_pct=top_pct, bottom_pct=bottom_pct, left_pct=left_pct, right_pct=right_pct)
            tables = analyzer.analyze_files([(os.path.basename(input_path), cropped_bytes)])
        elif is_pdf(input_path) and apply_crop:
            # If PDF, try to convert to images and analyze cropped images per page (pipeline parity).
            # Fallback: when no PDF imaging backend is available, analyze the whole PDF without crop.
            try:
                pages = _pdf_bytes_to_images(content)
            except Exception as exc:
                print(f"PDF imaging not available ({exc}); analyzing whole PDF without crop as fallback.")
                tables = analyzer.analyze_files([(os.path.basename(input_path), content)])
            else:
                tables = []
                for idx, img_bytes in enumerate(pages, start=1):
                    try:
                        cropped_bytes, _, _ = crop_image_bytes(img_bytes, top_pct=top_pct, bottom_pct=bottom_pct, left_pct=left_pct, right_pct=right_pct)
                    except Exception:
                        cropped_bytes = img_bytes
                    analyzed = analyzer.analyze_content(cropped_bytes)
                    tables.extend(analyzed.tables)
        else:
            tables = analyzer.analyze_files([(os.path.basename(input_path), content)])
    except Exception as exc:
        print(f"OCR analysis failed: {exc}")
        return None

    combined_rows: List[List[str]] = []
    header: List[str] | None = None
    for table in tables or []:
        if not table.rows:
            continue
        if header is None:
            header = table.rows[0]
            combined_rows.append(header)
            combined_rows.extend(table.rows[1:])
        else:
            combined_rows.extend(table.rows)
    return combined_rows if combined_rows else None


def load_center_config(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"Failed to read center config {path}: {exc}")
        return {}

def write_results(rows: List[Dict[str, Any]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["行番号", "成分表", "見本", "preview"])
        for row in rows:
            writer.writerow([
                row.get("row_index"),
                row.get("成分表", ""),
                row.get("見本", ""),
                row.get("preview", ""),
            ])


def main() -> int:
    base_dir = os.path.dirname(__file__)
    input_path = os.path.join(base_dir, INPUT_PATH) if not os.path.isabs(INPUT_PATH) else INPUT_PATH
    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}\nPlease set INPUT_PATH at the top of this file.")
        return 2

    # Load center config early so we can apply optional crop in OCR path
    center_path = os.path.join(base_dir, CENTER_CONFIG_PATH) if not os.path.isabs(CENTER_CONFIG_PATH) else CENTER_CONFIG_PATH
    center_conf = load_center_config(center_path)

    rows = load_rows_from_input(input_path, center_conf=center_conf)
    if not rows:
        print(f"Unsupported or empty input: {input_path}")
        return 3


    # Apply simple overrides without editing JSON
    merged_conf: Dict[str, Any] = copy.deepcopy(center_conf) if isinstance(center_conf, dict) else {}
    for target, override in (TARGET_OVERRIDES or {}).items():
        try:
            base = merged_conf.get(target, {}) if isinstance(merged_conf.get(target), dict) else {}
            new_conf = copy.deepcopy(base)
            new_conf.update(copy.deepcopy(override))
            merged_conf[target] = new_conf
        except Exception:
            pass

    # Defer engine import until after sys.path adjustments
    from backend.services.matching_module.engine import (  # noqa: E402
        DEFAULT_CONFIGS as ENGINE_DEFAULT_CONFIGS,
        build_matching_table,
    )

    # Use the shared engine to build a unified matching table
    effective_defaults: Dict[str, Any] = copy.deepcopy(ENGINE_DEFAULT_CONFIGS)
    match_table = build_matching_table(
        rows,
        center_conf=merged_conf,
        target_order=TARGET_ORDER,
        defaults=effective_defaults,
        log_fn=lambda m: print(m),
    )

    processed_rows = match_table.rows
    if not processed_rows:
        print("No rows after matching.")
        return 4

    header = processed_rows[0]
    data_rows = processed_rows[1:]
    try:
        seibun_idx = header.index("成分表")
    except ValueError:
        seibun_idx = -1
    try:
        mihon_idx = header.index("見本")
    except ValueError:
        mihon_idx = -1

    out_rows: List[Dict[str, Any]] = []
    for i, row in enumerate(data_rows, start=1):
        seibun = row[seibun_idx] if seibun_idx >= 0 and seibun_idx < len(row) else ""
        mihon = row[mihon_idx] if mihon_idx >= 0 and mihon_idx < len(row) else ""
        preview = " | ".join([c if isinstance(c, str) else str(c) for c in row])
        out_rows.append({
            "row_index": i,
            "成分表": seibun,
            "見本": mihon,
            "preview": preview,
        })

    result_path = os.path.join(base_dir, "result.csv")
    timestamp_path = os.path.join(base_dir, f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    write_results(out_rows, result_path)
    write_results(out_rows, timestamp_path)
    print(f"Wrote {result_path} and {timestamp_path} with {len(out_rows)} rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
