from __future__ import annotations

from typing import List, Dict, Any, Tuple
from difflib import SequenceMatcher


def _norm(s: Any) -> str:
    if s is None:
        return ""
    t = str(s)
    # unify spaces and remove common punctuation variants that cause header drift
    for ch in ["\u3000", " ", "\t", "\n", "\r", "・", "･", "·", "・", "-", "_", "|", ":", "：", "·"]:
        t = t.replace(ch, "")
    return t.lower()


def _char_similarity(a: Any, b: Any) -> float:
    aa = _norm(a)
    bb = _norm(b)
    if not aa or not bb:
        return 0.0
    return SequenceMatcher(None, aa, bb).ratio()


def _match_exact(text: Any, terms: List[str]) -> bool:
    if not terms:
        return False
    t = _norm(text)
    if not t:
        return False
    for term in terms:
        if _norm(term) == t:
            return True
    return False


def _match_partial(text: Any, terms: List[str]) -> bool:
    if not terms:
        return False
    t = _norm(text)
    if not t:
        return False
    for term in terms:
        if _norm(term) and _norm(term) in t:
            return True
    return False


def _row_contains_candidate(row: List[Any], candidates: List[str]) -> bool:
    for cell in row:
        if not isinstance(cell, str):
            continue
        for cand in candidates:
            if cand and cand in cell:
                return True
    return False


def _is_repeated_header_row(row: List[Any], header_row: List[Any], candidates: List[str]) -> bool:
    """Strict check to detect repeated header rows.

    Uses exact equality against header titles or candidate tokens (after strip)
    and requires at least two matches to avoid false positives like "規格明記".
    """
    header_tokens = {str(x).strip() for x in header_row if isinstance(x, str) and str(x).strip()}
    candidate_tokens = {str(c).strip() for c in candidates if isinstance(c, str) and str(c).strip()}
    count = 0
    for cell in row:
        if not isinstance(cell, str):
            continue
        t = cell.strip()
        if not t:
            continue
        if t in header_tokens or t in candidate_tokens:
            count += 1
    return count >= 2


def _locate_header_and_column(rows: List[List[Any]], candidates: List[str]) -> Tuple[int, int | None, str]:
    """Locate header row and the best matching column.

    Preference order for picking the column on the detected header row:
      1) Exact equality (strip-equal) to any candidate
      2) Substring match fallback
    """
    header_idx = 0
    header_row: List[Any] | None = None
    for idx, row in enumerate(rows):
        if _row_contains_candidate(row, candidates):
            header_idx = idx
            header_row = row
            break
    # If not found by substring candidates, fallback to char-level similarity across first N rows
    if header_row is None:
        best_score = 0.0
        best_row = 0
        best_col = None
        limit = min(len(rows), 10)  # search top 10 rows as header candidates
        for r_idx in range(limit):
            row = rows[r_idx]
            for c_idx, cell in enumerate(row):
                if not isinstance(cell, str) or not cell.strip():
                    continue
                for cand in candidates:
                    score = _char_similarity(cell, cand)
                    if score > best_score:
                        best_score = score
                        best_row = r_idx
                        best_col = c_idx
        header_idx = best_row
        header_row = rows[header_idx]
        return header_idx, best_col, "char_fallback"
    # First try exact equality
    normalized_cands = [c.strip() for c in candidates]
    for col_idx, cell in enumerate(header_row):
        if isinstance(cell, str) and cell.strip() in normalized_cands:
            return header_idx, col_idx, "exact"
    # Fallback to substring
    for col_idx, cell in enumerate(header_row):
        if not isinstance(cell, str):
            continue
        for cand in candidates:
            if cand and cand in cell:
                return header_idx, col_idx, "substring"
    # Final fallback: char-level similarity on the detected header row
    best_score = 0.0
    best_col = None
    for col_idx, cell in enumerate(header_row):
        if not isinstance(cell, str) or not cell.strip():
            continue
        for cand in candidates:
            score = _char_similarity(cell, cand)
            if score > best_score:
                best_score = score
                best_col = col_idx
    return header_idx, best_col, "char_fallback"


def match(rows: List[List[str]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Module 1: Header-based matching using a matching string in the cell.

    config expects:
      - col: list[str] header names/candidates
      - matching: str text to match in the cell meaning 'has nutrition'

    Returns list of dicts: {row_index, matched(bool), cell}
    """
    if not rows:
        return []
    raw_candidates = config.get("col") or []
    candidates = [str(c) for c in raw_candidates if isinstance(c, str)]
    # Read matching terms with new config keys and keep backward compatibility.
    # - matching_part: list of terms for partial match (substring)
    # - matching_all: list of terms for partial match as well (substring)
    #   NOTE: both lists are OR'ed; we do not use exact equality for matching.
    # - matching: legacy key treated as partial match when new keys absent
    match_terms_part: List[str] = []
    match_terms_all: List[str] = []
    raw_matching_part = config.get("matching_part")
    if isinstance(raw_matching_part, str):
        match_terms_part.append(raw_matching_part)
    elif isinstance(raw_matching_part, list):
        match_terms_part.extend([str(m) for m in raw_matching_part if isinstance(m, str)])
    raw_matching_all = config.get("matching_all")
    if isinstance(raw_matching_all, str):
        match_terms_all.append(raw_matching_all)
    elif isinstance(raw_matching_all, list):
        match_terms_all.extend([str(m) for m in raw_matching_all if isinstance(m, str)])
    # Backward compatibility: 'matching' behaves as partial ONLY when no new keys are provided
    raw_matching_legacy = config.get("matching")
    if not match_terms_part and not match_terms_all:
        if isinstance(raw_matching_legacy, str):
            match_terms_part.append(raw_matching_legacy)
        elif isinstance(raw_matching_legacy, list):
            match_terms_part.extend([str(m) for m in raw_matching_legacy if isinstance(m, str)])

    header_idx, col_idx, pick_mode = _locate_header_and_column(rows, candidates) if candidates else (0, None, "none")
    # Prepare meta/debug info to aid callers (first element in result list)
    header_row = rows[header_idx] if 0 <= header_idx < len(rows) else []
    col_header = header_row[col_idx] if isinstance(header_row, list) and col_idx is not None and col_idx < len(header_row) else ""
    meta: Dict[str, Any] = {
        "header_index": header_idx,
        "col_index": col_idx,
        "col_header": col_header if isinstance(col_header, str) else str(col_header),
        "candidates": candidates,
        # For backward-compatible logs, provide merged terms as 'match_terms'
        "match_terms": list(dict.fromkeys(match_terms_all + match_terms_part)),
        "match_terms_all": match_terms_all,
        "match_terms_partial": match_terms_part,
        "match_mode": "partial",
        "col_pick_mode": pick_mode,
    }
    data_rows = rows[header_idx + 1 :] if header_idx + 1 < len(rows) else []
    results: List[Dict[str, Any]] = [{"_meta": meta}]
    data_index = 0
    fallback_row_scan = bool(config.get("fallback_row_scan", False))
    last_result: Dict[str, Any] | None = None
    for offset, row in enumerate(data_rows, start=1):
        src_abs_index = header_idx + offset  # absolute row index in original 'rows' (0-based)
        # Skip only when row is a true repeated header (strict equality on >=2 cells)
        if candidates and _is_repeated_header_row(row, header_row, candidates):
            continue

        # Detect continuation lines: only target column has text, other columns empty
        is_continuation = False
        target_text = ""
        if col_idx is not None and col_idx < len(row):
            target_cell = row[col_idx]
            target_text = target_cell if isinstance(target_cell, str) else str(target_cell)
            if target_text.strip():
                other_has_text = any(
                    (isinstance(c, str) and c.strip()) for j, c in enumerate(row) if j != col_idx
                )
                if not other_has_text:
                    is_continuation = True

        if is_continuation and last_result is not None:
            # Merge with previous row's cell text and recompute matched
            prev_cell = last_result.get("cell", "")
            combined = (f"{prev_cell} {target_text}" if prev_cell else target_text).strip()
            last_result["cell"] = combined
            # Recompute match as partial substring across all configured terms (OR)
            all_terms = match_terms_all + match_terms_part
            matched_now = _match_partial(combined, all_terms)
            last_result["matched"] = matched_now
            # Keep previous src_row_index (continuation belongs to previous visual row)
            # Do not advance data_index; treat as same visual cell
            continue

        data_index += 1
        cell = ""
        matched = False
        if col_idx is not None and col_idx < len(row):
            cell = row[col_idx]
            if isinstance(cell, str):
                # Partial substring match across all configured terms (OR)
                all_terms = match_terms_all + match_terms_part
                matched = _match_partial(cell, all_terms)
        # Optional fallback: scan the whole row (joined) when col is not usable
        if not matched and fallback_row_scan and (match_terms_all or match_terms_part):
            joined = " ".join([c for c in row if isinstance(c, str)])
            all_terms = match_terms_all + match_terms_part
            if _match_partial(joined, all_terms):
                matched = True
        last_result = {"row_index": data_index, "matched": matched, "cell": cell, "src_row_index": src_abs_index}
        results.append(last_result)
    return results
