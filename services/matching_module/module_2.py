from __future__ import annotations

from typing import List, Dict, Any, Tuple


def _row_contains_candidate(row: List[Any], candidates: List[str]) -> bool:
    for cell in row:
        if not isinstance(cell, str):
            continue
        for cand in candidates:
            if cand and cand in cell:
                return True
    return False


def _locate_header_and_column(rows: List[List[Any]], candidates: List[str]) -> Tuple[int, int | None]:
    for idx, row in enumerate(rows):
        if _row_contains_candidate(row, candidates):
            for col_idx, cell in enumerate(row):
                if isinstance(cell, str):
                    for cand in candidates:
                        if cand and cand in cell:
                            return idx, col_idx
    return 0, None


def match(rows: List[List[str]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Module 2: Per-row ◯/− style matching.

    config expects:
      - col: list[str] header names/candidates
      - matching_T: list[str] tokens meaning True (e.g., ["◯", "○", "O"])
      - matching_F: list[str] tokens meaning False (e.g., ['-', '－'])

    Returns list of dicts: {row_index, matched(bool or None), cell}
    """
    if not rows:
        return []
    raw_candidates = config.get("col") or []
    candidates = [str(c) for c in raw_candidates if isinstance(c, str)]
    raw_tokens_t = config.get("matching_T")
    raw_tokens_f = config.get("matching_F")
    if isinstance(raw_tokens_t, str):
        tokens_T = [raw_tokens_t]
    else:
        tokens_T = [str(t) for t in (raw_tokens_t or []) if isinstance(t, str)]
    if isinstance(raw_tokens_f, str):
        tokens_F = [raw_tokens_f]
    else:
        tokens_F = [str(t) for t in (raw_tokens_f or []) if isinstance(t, str)]
    header_idx, col_idx = _locate_header_and_column(rows, candidates) if candidates else (0, None)
    data_rows = rows[header_idx + 1 :] if header_idx + 1 < len(rows) else []
    results: List[Dict[str, Any]] = []
    data_index = 0
    for row in data_rows:
        if candidates and _row_contains_candidate(row, candidates):
            continue
        data_index += 1
        cell = ""
        matched = None
        if col_idx is not None and col_idx < len(row):
            cell = row[col_idx]
            if isinstance(cell, str):
                s = cell.strip()
                for t in tokens_T:
                    if t in s:
                        matched = True
                        break
                if matched is None:
                    for f in tokens_F:
                        if f in s:
                            matched = False
                            break
        results.append({"row_index": data_index, "matched": matched, "cell": cell})
    return results
