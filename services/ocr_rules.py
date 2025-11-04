from __future__ import annotations

from services.matching_module.engine import MatchingTable, build_matching_table


def apply_default_rules(*args, **kwargs):  # pragma: no cover - legacy alias
    raise RuntimeError("apply_default_rules was removed; use build_matching_table instead")


__all__ = ["MatchingTable", "build_matching_table", "apply_default_rules"]
