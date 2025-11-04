from __future__ import annotations

import copy
import importlib
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

DEFAULT_TARGET_ORDER: Sequence[str] = ("nutrition", "sample")
DEFAULT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "nutrition": {
        "module": [1],
        "col": ["規 格", "規格", "銘柄·条件", "銘柄・条件", "銘柄条件"],
        "matching": ["内容表提出", "成分表提出"],
    },
    "sample": {
        "module": [1],
        "col": ["規 格", "規格", "備考"],
        "matching": ["見本提出", "見本 提出", "◯", "○", "O"],
    },
}


@dataclass
class MatchingTable:
    rows: List[List[str]]
    row_map: List[int]
    header_index: int
    target_hits: Dict[str, int]


def _row_contains_candidate(row: Iterable[Any], candidates: Sequence[str]) -> bool:
    for cell in row:
        if not isinstance(cell, str):
            continue
        for cand in candidates:
            if cand and cand in cell:
                return True
    return False


def _is_repeated_header_row(
    row: Sequence[Any], base_header: Sequence[Any], candidates: Sequence[str]
) -> bool:
    """Detect a repeated header row more strictly than substring matching.

    Rules:
    - Exact-equality against header titles or candidate tokens after strip.
    - Require at least two header-like matches in the row to consider it a header.
      This avoids skipping data rows like "規格明記" that merely contain "規格".
    """
    header_tokens = {
        str(x).strip() for x in base_header if isinstance(x, str) and str(x).strip()
    }
    candidate_tokens = {str(c).strip() for c in candidates if str(c).strip()}
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


def _detect_header_index(rows: Sequence[Sequence[Any]], candidates: Sequence[str]) -> int:
    if not rows:
        return 0
    if not candidates:
        return 0
    for idx, row in enumerate(rows):
        if _row_contains_candidate(row, candidates):
            return idx
    return 0


def _normalize_row(row: Sequence[Any]) -> List[str]:
    return [str(cell) if cell is not None else "" for cell in row]


def _resolve_target_config(
    center_conf: Optional[Dict[str, Any]],
    target: str,
    defaults: Optional[Dict[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    if defaults and target in defaults:
        merged.update(copy.deepcopy(defaults[target]))
    if center_conf:
        target_conf = center_conf.get(target)
        if isinstance(target_conf, dict):
            merged.update(copy.deepcopy(target_conf))
    return merged


def _collect_header_candidates(
    center_conf: Optional[Dict[str, Any]],
    target_order: Sequence[str],
    defaults: Optional[Dict[str, Dict[str, Any]]],
) -> List[str]:
    candidates: List[str] = []
    for target in target_order:
        conf = _resolve_target_config(center_conf, target, defaults)
        for col in conf.get("col", []) or []:
            if isinstance(col, str) and col:
                candidates.append(col)
    return candidates


def _run_module(
    module_id: int,
    config: Dict[str, Any],
    rows: List[List[str]],
    log_fn: Optional[Callable[[str], None]],
) -> List[Dict[str, Any]]:
    # Ensure 'backend' package and project root are on sys.path for dynamic imports
    try:
        this_file = Path(__file__).resolve()
        backend_dir = this_file.parents[2]  # .../backend
        project_root = backend_dir.parent
        for p in (backend_dir, project_root):
            p_str = str(p)
            if p_str not in sys.path:
                sys.path.insert(0, p_str)
    except Exception:
        pass

    # Try multiple import paths for robustness across run contexts
    candidates = [
        f"backend.services.matching_module.module_{module_id}",
        f"services.matching_module.module_{module_id}",
        f"matching_module.module_{module_id}",
    ]
    module = None
    last_exc: Optional[Exception] = None
    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
            break
        except Exception as exc:  # pragma: no cover - defensive
            last_exc = exc
            continue
    if module is None:
        if log_fn:
            mod = candidates[0]
            log_fn(f"[MATCH][IMPORT][ERROR] {mod}: {last_exc}")
        return []
    if not hasattr(module, "match"):
        if log_fn:
            log_fn(f"[MATCH][ERROR] module {module_name} has no match()")
        return []
    try:
        result = module.match(rows, config)  # type: ignore[arg-type]
        return result if isinstance(result, list) else []
    except Exception as exc:  # pragma: no cover - defensive
        if log_fn:
            log_fn(f"[MATCH][ERROR] module {module_name} failed: {exc}")
        return []


def build_matching_table(
    rows: List[List[Any]],
    *,
    center_conf: Optional[Dict[str, Any]] = None,
    target_order: Optional[Sequence[str]] = None,
    defaults: Optional[Dict[str, Dict[str, Any]]] = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> MatchingTable:
    if not rows:
        return MatchingTable(rows=[], row_map=[], header_index=0, target_hits={})

    normalized_rows = [_normalize_row(row) for row in rows]
    effective_defaults = defaults or DEFAULT_CONFIGS
    effective_order = tuple(target_order or DEFAULT_TARGET_ORDER)
    header_candidates = _collect_header_candidates(center_conf, effective_order, effective_defaults)
    header_index = _detect_header_index(normalized_rows, header_candidates)
    if log_fn:
        log_fn(
            f"[MATCH] header_index={header_index} candidates={header_candidates[:4]}"
            + ("..." if len(header_candidates) > 4 else "")
        )

    base_header = normalized_rows[header_index] if header_index < len(normalized_rows) else normalized_rows[0]
    header = list(base_header)
    if "成分表" not in header:
        header.append("成分表")
    if "見本" not in header:
        header.append("見本")
    seibun_idx = header.index("成分表")
    mihon_idx = header.index("見本")

    data_rows: List[List[str]] = []
    # IMPORTANT: Keep row alignment consistent with module_1 by using a dynamic
    # current_header_row for repeated-header detection (re-pick), not the fixed base_header.
    current_header_row = base_header
    for row in normalized_rows[header_index + 1 :]:
        # Skip rows that look like repeated header rows (strict equality on >=2 cells)
        # and update the current_header_row baseline just like module_1 does.
        if header_candidates and _is_repeated_header_row(row, current_header_row, header_candidates):
            current_header_row = row
            continue
        normalized = list(row)
        if len(normalized) > len(header):
            extra_cols = len(normalized) - len(header)
            header.extend([""] * extra_cols)
            for existing in data_rows:
                existing.extend([""] * extra_cols)
        while len(normalized) < len(header):
            normalized.append("")
        seibun_idx = header.index("成分表")
        mihon_idx = header.index("見本")
        data_rows.append(normalized)
    row_map = list(range(1, len(data_rows) + 1))

    target_hits: Dict[str, int] = {}
    target_maps: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for target in effective_order:
        conf = _resolve_target_config(center_conf, target, effective_defaults)
        module_ids = conf.get("module") if isinstance(conf.get("module"), list) else None
        if not module_ids:
            if log_fn:
                log_fn(f"[MATCH] skip target={target} (no module)")
            continue
        module_id = module_ids[0]
        module_conf = {k: v for k, v in conf.items() if k != "module"}
        result = _run_module(module_id, module_conf, normalized_rows, log_fn)
        # Handle module-provided meta for richer debugging
        meta: Optional[Dict[str, Any]] = None
        if result and isinstance(result[0], dict) and "_meta" in result[0]:
            try:
                meta = result.pop(0).get("_meta")  # type: ignore[assignment]
            except Exception:
                meta = None
        if log_fn and isinstance(meta, dict):
            cand = meta.get("candidates", [])
            terms = meta.get("match_terms", [])
            mode = meta.get("match_mode", "")
            h_idx = meta.get("header_index")
            c_idx = meta.get("col_index")
            c_hdr = meta.get("col_header", "")
            log_fn(f"[MATCH][{target}] candidates={cand} match_terms={terms} mode={mode}")
            log_fn(f"[MATCH][{target}] picked header_index={h_idx} col_index={c_idx} col_header='{c_hdr}'")
            # If module reported re-picks (mid-table header/column changes), log them for traceability
            repicks = meta.get("repicks") or []
            try:
                if isinstance(repicks, list) and repicks:
                    for r in repicks[:5]:  # limit to 5 entries to avoid log spam
                        try:
                            r_idx = r.get("src_row_index")
                            rc = r.get("col_index")
                            rh = r.get("col_header", "")
                            log_fn(f"[MATCH][{target}][reheader] at src_row_index={r_idx} col_index={rc} col_header='{rh}'")
                        except Exception:
                            continue
            except Exception:
                pass
        if not result:
            target_hits[target] = 0
            target_maps[target] = {}
            continue
        mapping: Dict[int, Dict[str, Any]] = {}
        hits = 0
        for item in result:
            if not isinstance(item, dict):
                continue
            idx = item.get("row_index")
            # Prefer absolute source row index if provided to align header baselines
            src_row_index = item.get("src_row_index")
            if isinstance(src_row_index, int):
                # engine data rows are 1-based after engine header_index
                aligned = src_row_index - header_index
                if isinstance(aligned, int) and aligned >= 1:
                    idx = aligned
            if not isinstance(idx, int) or idx <= 0:
                continue
            mapping[idx] = item
            matched = item.get("matched")
            if isinstance(matched, bool):
                hits += int(matched)
            elif matched:
                hits += 1
        if log_fn:
            matched_rows = [i for i, v in mapping.items() if bool(v.get("matched"))]
            # Include up to 5 examples with matched terms for easier debugging
            examples = []
            cell_examples = []
            for i in matched_rows[:5]:
                entry = mapping.get(i, {})
                term = entry.get("matched_term")
                mode = entry.get("matched_mode")
                cell = entry.get("cell") or ""
                cell_prev = (cell[:40] + ("…" if len(cell) > 40 else "")) if isinstance(cell, str) else ""
                used_hdr = entry.get("col_header_used") or ""
                if term:
                    if mode:
                        examples.append(f"{i}:{term}({mode})")
                    else:
                        examples.append(f"{i}:{term}")
                else:
                    examples.append(str(i))
                if used_hdr:
                    cell_examples.append(f"{i}:{used_hdr}:{cell_prev}")
                else:
                    cell_examples.append(f"{i}:{cell_prev}")
            ex_str = (" examples=" + ",".join(examples)) if examples else ""
            log_fn(f"[MATCH][{target}] hits={hits} matched_rows={matched_rows}{ex_str}")
            if cell_examples:
                log_fn(f"[MATCH][{target}] cells={cell_examples}")

            # Per-hit detailed debug: one line per matched row with full cell and highlighted hit term
            try:
                def _highlight_once(text: str, term: str) -> str:
                    try:
                        idx = text.find(term)
                        if idx == -1:
                            return text
                        return text[:idx] + "«" + term + "»" + text[idx+len(term):]
                    except Exception:
                        return text

                for i in matched_rows:
                    entry = mapping.get(i, {})
                    cell_full = entry.get("cell") or ""
                    term = entry.get("matched_term") or ""
                    mode = entry.get("matched_mode") or ""
                    used_hdr = entry.get("col_header_used") or ""
                    if isinstance(cell_full, str) and isinstance(term, str) and term:
                        highlighted = _highlight_once(cell_full, term)
                    else:
                        highlighted = cell_full if isinstance(cell_full, str) else str(cell_full)
                    # One-line debug per hit row
                    if mode:
                        log_fn(f"[MATCH][{target}][row={i}] term='{term}' mode={mode} col='{used_hdr}' cell='{highlighted}'")
                    else:
                        log_fn(f"[MATCH][{target}][row={i}] term='{term}' col='{used_hdr}' cell='{highlighted}'")
            except Exception:
                pass
        target_hits[target] = hits
        target_maps[target] = mapping

    # Reflect the actual cell used for matching into the preview table's display column (規格など).
    # We choose the first header column that matches any of the header candidates; if not found,
    # we fallback to the first column that contains "規格".
    display_spec_idx: Optional[int] = None
    if header:
        # 1) try header candidates
        cand_set = []
        try:
            cand_set = [str(c) for c in header_candidates]
        except Exception:
            cand_set = []
        found = False
        if cand_set:
            for i, h in enumerate(header):
                hs = str(h) if h is not None else ""
                for c in cand_set:
                    if c and c in hs:
                        display_spec_idx = i
                        found = True
                        break
                if found:
                    break
        # 2) fallback to '規格' literal
        if display_spec_idx is None:
            for i, h in enumerate(header):
                hs = str(h) if h is not None else ""
                if ("規格" in hs) or ("規 格" in hs):
                    display_spec_idx = i
                    break

    # Build a per-row view of the cell actually used by modules (prefer nutrition over sample when both available)
    used_cell_by_row: Dict[int, str] = {}
    if display_spec_idx is not None:
        for i in range(1, len(data_rows) + 1):
            n_entry = target_maps.get("nutrition", {}).get(i)
            s_entry = target_maps.get("sample", {}).get(i)
            val = None
            if isinstance(n_entry, dict):
                val = n_entry.get("cell")
            if (val is None or val == "") and isinstance(s_entry, dict):
                val = s_entry.get("cell")
            if isinstance(val, str) and val != "":
                used_cell_by_row[i] = val

    # Apply used cell preview (if available) and then flags
    for idx, row in enumerate(data_rows, start=1):
        if display_spec_idx is not None and idx in used_cell_by_row and display_spec_idx < len(row):
            row[display_spec_idx] = used_cell_by_row[idx]
        nutrition = target_maps.get("nutrition", {}).get(idx)
        sample = target_maps.get("sample", {}).get(idx)
        nutrition_flag = False
        if nutrition:
            matched = nutrition.get("matched")
            nutrition_flag = bool(matched)
        row[seibun_idx] = "○" if nutrition_flag else "-"
        sample_flag = False
        if sample:
            matched = sample.get("matched")
            sample_flag = bool(matched)
        row[mihon_idx] = "3" if sample_flag else "-"

    processed_rows = [header] + data_rows
    if log_fn:
        log_fn(
            f"[MATCH] rows={len(processed_rows)} nutrition_hits={target_hits.get('nutrition', 0)} "
            f"sample_hits={target_hits.get('sample', 0)}"
        )
    return MatchingTable(rows=processed_rows, row_map=row_map, header_index=header_index, target_hits=target_hits)
